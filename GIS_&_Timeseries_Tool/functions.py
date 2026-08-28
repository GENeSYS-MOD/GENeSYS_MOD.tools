import os
import numpy as np
from numpy import sin, cos, deg2rad, rad2deg, pi
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import geopandas as gpd
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from operator import itemgetter
import yaml
from shapely.geometry import Point

from sklearn.neighbors import BallTree

import cartopy.crs as ccrs
from cartopy.crs import PlateCarree as plate
import cartopy.io.shapereader as shpreader

import xarray as xr
import atlite
from atlite.resource import get_windturbineconfig, windturbine_smooth
from atlite.gis import shape_availability, ExclusionContainer
from atlite.gis import get_coords as atlite_get_coords  # local get_coords below shadows the name

import logging
import warnings
import timeit
import gc
from shapely.geometry import box

warnings.simplefilter('ignore')
logging.captureWarnings(False)
logging.basicConfig(level=logging.INFO)


###########################
## preparing coordinates ##
###########################
def _get_coords_custom(cutout, regions, gdf_polygons, offshore_file, bathymetry_file=None):
    """Coordinate assignment for explicit region polygons (a 'region' column).

    Onshore cells are the cutout cells inside a region polygon. Offshore cells
    are cells inside the EEZ (offshore_file) but outside every region polygon,
    each assigned to the nearest onshore region. Cells are pre-filtered to the
    region+EEZ bounding box first so this stays fast at finer resolutions.

    If bathymetry_file is given, each offshore cell also gets a 'depth' column
    (metres below sea level) sampled from that grid, used to classify foundation
    type (fixed vs floating) instead of distance.
    """
    gdf_polygons = gdf_polygons.to_crs("EPSG:4326")[["region", "geometry"]].copy()
    gdf_polygons["_rid"] = range(len(gdf_polygons))

    coords_raw_df = cutout.data[["x", "y"]].to_dataframe().reset_index()[["x", "y"]]
    pts = gpd.GeoDataFrame(
        coords_raw_df,
        geometry=gpd.points_from_xy(coords_raw_df["x"], coords_raw_df["y"]),
        crs="EPSG:4326",
    )

    # restrict cutout cells to the area of interest (regions + offshore) for speed
    minx, miny, maxx, maxy = gdf_polygons.total_bounds
    eez = None
    if offshore_file is not None:
        eez = gpd.read_file(offshore_file).to_crs("EPSG:4326")[["geometry"]]
        ex = eez.total_bounds
        minx, miny = min(minx, ex[0]), min(miny, ex[1])
        maxx, maxy = max(maxx, ex[2]), max(maxy, ex[3])
    pts = pts.cx[minx:maxx, miny:maxy]

    # onshore: cutout cells lying inside a region polygon
    onshore = gpd.sjoin(pts, gdf_polygons[["region", "geometry"]], how="inner", predicate="within")
    onshore = onshore.drop_duplicates(subset=["x", "y"])
    onshore["coords"] = onshore["y"].astype(str) + ", " + onshore["x"].astype(str)
    coords_onshore = onshore[["x", "y", "coords", "region"]].copy()
    if regions:
        coords_onshore = coords_onshore[coords_onshore["region"].isin(regions)]
    coords_onshore = coords_onshore.reset_index(drop=True)

    empty_offshore = gpd.GeoDataFrame(columns=["x", "y", "coords", "region", "distance"])
    if eez is None:
        print("Warning: no offshore_file provided; offshore coordinates skipped")
        return coords_onshore, empty_offshore

    # offshore: cells inside the EEZ but outside every region polygon
    tagged = gpd.sjoin(pts, gdf_polygons[["_rid", "geometry"]], how="left", predicate="within")
    sea = tagged[tagged["_rid"].isna()][["x", "y", "geometry"]].drop_duplicates(subset=["x", "y"])
    sea = gpd.sjoin(sea, eez, how="inner", predicate="within").drop_duplicates(subset=["x", "y"])
    coords_offshore = sea[["x", "y"]].reset_index(drop=True)

    if coords_offshore.empty or coords_onshore.empty:
        print("Warning: No offshore coordinates in the selected cutout")
        return coords_onshore, empty_offshore

    coords_offshore["coords"] = coords_offshore["y"].astype(str) + ", " + coords_offshore["x"].astype(str)

    # assign each offshore cell to the nearest onshore region (distance in nm).
    # haversine expects [lat, lon] == [y, x]; distance = arc * earth_radius_km / km_per_nm
    tree = BallTree(np.deg2rad(coords_onshore[["y", "x"]].values), metric="haversine")
    distances, index = tree.query(np.deg2rad(coords_offshore[["y", "x"]].values))
    coords_offshore["region"] = coords_onshore["region"].values[index.flatten()]
    coords_offshore["distance"] = distances.flatten() * 6371.0 / 1.852

    out_cols = ["x", "y", "coords", "region", "distance"]
    if bathymetry_file is not None:
        # nearest bathymetry sample per offshore cell; depth = metres below sea level.
        # Auto-detect variable + coord names so different products work unchanged:
        # the US file uses z / latitude / longitude, GEBCO uses elevation / lat / lon.
        bds = xr.open_dataset(bathymetry_file)
        zname = next((v for v in ("z", "elevation", "Band1") if v in bds.data_vars),
                     list(bds.data_vars)[0])
        latname = "latitude" if "latitude" in bds.coords else "lat"
        lonname = "longitude" if "longitude" in bds.coords else "lon"
        z = bds[zname].sel(
            **{latname: xr.DataArray(coords_offshore["y"].values, dims="p"),
               lonname: xr.DataArray(coords_offshore["x"].values, dims="p")},
            method="nearest",
        ).values
        bds.close()
        coords_offshore["depth"] = np.round(-z.astype(float), 1)
        out_cols.append("depth")

    return coords_onshore, coords_offshore[out_cols]


def get_coords(
        cutout,
        regions,
        geo_file = None,
        admin=0,
        offshore_file=None,
        bathymetry_file=None
):

    gdf_polygons = gpd.read_file(geo_file)

    # Custom-region mode: geo_file carries explicit region polygons (a 'region'
    # column) rather than country ISO codes. Offshore cells are bounded by the
    # EEZ in offshore_file and (optionally) classified by depth from
    # bathymetry_file. Falls back to the ISO/admin logic when the file is the
    # natural-earth data (has an 'iso_a2' column).
    if ("region" in gdf_polygons.columns) and ("iso_a2" not in gdf_polygons.columns):
        return _get_coords_custom(cutout, regions, gdf_polygons, offshore_file, bathymetry_file)

    coords_raw_df = cutout.data[["x", "y"]].to_dataframe().reset_index()
    coords_raw = coords_raw_df[["x", "y"]].values.tolist()

    geometry = [Point(xy) for xy in coords_raw]
    gdf_points = gpd.GeoDataFrame(geometry=geometry)

    gdf_points = gdf_points.set_crs("EPSG:4326")
    gdf_polygons = gdf_polygons.to_crs("EPSG:4326")

    # Spatial join to assign each point the name of the polygon it resides in
    coords = gpd.sjoin(gdf_points, gdf_polygons, how="left", predicate="within")

    coords = coords.dropna()

    coords["x"] = coords["geometry"].x
    coords["y"] = coords["geometry"].y
    coords["coords"] = coords["y"].astype(str) + ", " + coords["x"].astype(str)

    coords_onshore = coords[coords['iso_a2'].isin(regions)]
    coords_onshore = coords_onshore.rename(columns={'iso_a2': 'region'})
    coords_onshore = coords_onshore.rename(columns={'iso_3166_2': 'subregion'})
    coords_onshore = coords_onshore[["x", "y", "coords", 'region', 'subregion']]

    coords_offshore = coords_raw_df.merge(coords, on=['x', 'y'], how='left', indicator=True)
    coords_offshore = coords_offshore[coords_offshore['_merge'] == 'left_only']

    if coords_offshore.empty:
        print("Warning: No offshore coordinates in the selected cutout")
    else:
        coords_offshore = coords_offshore[["x", "y"]]
        coords_offshore["coords"] = coords_offshore["y"].astype(str) + ", " + coords_offshore["x"].astype(str)

        # haversine formula to get the shortest distance to coastline to assign countries
        tree = BallTree(np.deg2rad(coords[['x', 'y']].values), metric='haversine')

        query_lons = coords_offshore['x']
        query_lats = coords_offshore['y']

        distances, index = tree.query(np.deg2rad(np.c_[query_lons, query_lats]))

        index = [item for sublist in index for item in sublist]
        distance = [item * 6371.0 / 1.852 for sublist in distances for item in sublist]

        nearest = []

        for ind, dist in zip(index, distance):
            nearest.append([coords.iloc[ind]['iso_a2'], coords.iloc[ind]['iso_3166_2'], dist])

        nearest = np.array(nearest)

        coords_offshore['region'] = nearest[:, 0]
        coords_offshore['subregion'] = nearest[:, 1]
        coords_offshore['distance'] = nearest[:, 2]

        coords_offshore = coords_offshore[coords_offshore['region'].isin(regions)]

    if admin == 0:
        coords_onshore = coords_onshore.drop(columns=['subregion'])
        if not coords_offshore.empty:
            coords_offshore = coords_offshore.drop(columns=['subregion'])
    elif admin == 1:
        coords_onshore = coords_onshore.drop(columns=['region']).rename(columns={'subregion': 'region'})
        if not coords_offshore.empty:
            coords_offshore = coords_offshore.drop(columns=['region']).rename(columns={'subregion': 'region'})

    return coords_onshore, coords_offshore


########################################################
## help function to pivot, categorize and write files ##
########################################################
def _site_categories(df_long):
    """Per-coordinate inf/avg/opt label, same rule as pivot_and_categorize.

    df_long has columns coords, capacity_factor, region. Each coordinate is
    classified within its region by the annual-mean capacity factor: <= regional
    30th percentile -> 'inf', >= 70th -> 'opt', else 'avg'. Returns a DataFrame
    [coords, region, category]. Used to colour usable-site maps by site quality.
    """
    m = (df_long.groupby(["region", "coords"], as_index=False)["capacity_factor"]
         .mean())
    # per-region 30th/70th percentile thresholds, broadcast back to each row
    g = m.groupby("region")["capacity_factor"]
    q30 = g.transform(lambda s: s.quantile(0.30))
    q70 = g.transform(lambda s: s.quantile(0.70))
    m["category"] = np.where(m["capacity_factor"] <= q30, "inf",
                             np.where(m["capacity_factor"] >= q70, "opt", "avg"))
    return m[["coords", "region", "category"]]


def pivot_and_categorize(
        df,
        tech,
        write_raw_data=False,
        timeframe=None,
        filename=None,
        output_dir='output/',
        out_tech=None
):
    # out_tech sets the technology label used in OUTPUT filenames, while tech still
    # drives the categorization logic. Lets a 'pv'-type run be written under a
    # different name (e.g. 'pv_rooftop') without changing the quantile split.
    out_tech = out_tech or tech

    if (tech == "wind_onshore") | (tech == "pv"):
        df_pvt = pd.pivot_table(df, values='capacity_factor', index='coords', columns='region', aggfunc=np.mean).copy()
        df_pvt.loc["q30", :] = df_pvt.quantile(.30)
        df_pvt.loc["q70", :] = df_pvt.quantile(.70)

        df_inf = df_pvt[df_pvt.le(df_pvt.loc['q30'], axis=1)].drop(index=['q30', 'q70']).dropna(axis=0, how='all')
        df_avg = df_pvt[(df_pvt.gt(df_pvt.loc['q30'], axis=1)) & (df_pvt.lt(df_pvt.loc['q70'], axis=1))].drop(index=['q30', 'q70']).dropna(axis=0, how='all')
        df_opt = df_pvt[df_pvt.ge(df_pvt.loc['q70'], axis=1)].drop(index=['q30', 'q70']).dropna(axis=0, how='all')

        df_inf = df[df['coords'].isin(df_inf.index)]
        df_avg = df[df['coords'].isin(df_avg.index)]
        df_opt = df[df['coords'].isin(df_opt.index)]

        if write_raw_data == True:
            df_inf.to_csv(output_dir + '/' + timeframe + '_' + filename + '_' + out_tech + '_inf_raw.csv', index=True)
            df_avg.to_csv(output_dir + '/' + timeframe + '_' + filename + '_' + out_tech + '_avg_raw.csv', index=True)
            df_opt.to_csv(output_dir + '/' + timeframe + '_' + filename + '_' + out_tech + '_opt_raw.csv', index=True)

        df_inf = pd.pivot_table(df_inf, values='capacity_factor', index='time', columns='region', aggfunc=np.mean)
        df_avg = pd.pivot_table(df_avg, values='capacity_factor', index='time', columns='region', aggfunc=np.mean)
        df_opt = pd.pivot_table(df_opt, values='capacity_factor', index='time', columns='region', aggfunc=np.mean)

        if write_raw_data == True:
            df.to_csv(output_dir + '/' + timeframe + '_' + out_tech + '_raw.csv', index=True)
            display(df)

        df_inf.to_csv(output_dir + '/' + timeframe + '_' + filename + '_' + out_tech + '_inf.csv', index=True)
        df_avg.to_csv(output_dir + '/' + timeframe + '_' + filename + '_' + out_tech + '_avg.csv', index=True)
        df_opt.to_csv(output_dir + '/' + timeframe + '_' + filename + '_' + out_tech + '_opt.csv', index=True)

        return df_inf, df_avg, df_opt
    
    elif (tech == "horizontal") | (tech == "tilted_horizontal") | (tech == "vertical") | (tech == "dual"):
        
        if write_raw_data == True:
            df.to_csv(output_dir + '/' + timeframe + '_' + filename + '_pv_' + tech + '_raw.csv', index=True)
          
        df_tracking = pd.pivot_table(df, values='capacity_factor', index='time', columns='region',aggfunc=np.mean).copy()
        df_tracking.to_csv(output_dir + '/' + timeframe + '_' + filename + '_pv_' + tech + '.csv', index=True)
        
        return df_tracking 

    elif tech == "wind_offshore":
        if 'depth' in df.columns:
            # classify by water depth (m): fixed-bottom foundations up to ~60 m,
            # floating beyond. shallow=monopile, transitional=jacket (both fixed/
            # concrete), deep=floating. Cells deeper than FLOATING_MAX are dropped.
            FIXED_SHALLOW_MAX = 30
            FIXED_DEEP_MAX = 60
            FLOATING_MAX = 1000
            df_shallow = df[(df['depth'] > 0) & (df['depth'] <= FIXED_SHALLOW_MAX)]
            df_transitional = df[(df['depth'] > FIXED_SHALLOW_MAX) & (df['depth'] <= FIXED_DEEP_MAX)]
            df_deep = df[(df['depth'] > FIXED_DEEP_MAX) & (df['depth'] <= FLOATING_MAX)]
        else:
            df_shallow = df[df['distance'] < 9]
            df_deep = df[(df['distance'] > 27) & (df['distance'] < 120)]
            df_transitional = df[(df['distance'] >= 9) & (df['distance'] <= 27)]

        if write_raw_data == True:
            df_shallow.to_csv(output_dir + '/' + timeframe + '_' + filename + '_wind_offshore_shallow_raw.csv', index=True)
            df_transitional.to_csv(output_dir + '/' + timeframe + '_' + filename + '_wind_offshore_transitional_raw.csv', index=True)
            df_deep.to_csv(output_dir + '/' + timeframe + '_' + filename + '_wind_offshore_deep_raw.csv', index=True)

        df_shallow = pd.pivot_table(df_shallow, values='capacity_factor', index='time', columns='region',aggfunc=np.mean).copy()
        df_transitional = pd.pivot_table(df_transitional, values='capacity_factor', index='time', columns='region',aggfunc=np.mean).copy()
        df_deep = pd.pivot_table(df_deep, values='capacity_factor', index='time', columns='region', aggfunc=np.mean).copy()

        if write_raw_data == True:
            wnd100 = df[df['distance'] < 180]
            wnd100.to_csv(output_dir + '/' + timeframe + '_wind_offshore_raw.csv', index=True)

        df_shallow.to_csv(output_dir + '/' + timeframe + '_' + filename + '_wind_offshore_shallow.csv', index=True)
        df_transitional.to_csv(output_dir + '/' + timeframe + '_' + filename + '_wind_offshore_transitional.csv', index=True)
        df_deep.to_csv(output_dir + '/' + timeframe + '_' + filename + '_wind_offshore_deep.csv', index=True)

        return df_shallow, df_transitional, df_deep

    elif (tech == "horizontal") | (tech == "tilted_horizontal") | (tech == "vertical") | (tech == "dual"):
        
        if write_raw_data == 1:
            df.to_csv(output_dir + '/' + timeframe + '_' + filename + '_pv_' + tech + '_raw.csv', index=True)
          
        df_tracking = pd.pivot_table(df, values='capacity_factor', index='time', columns='region',aggfunc=np.mean).copy()
        df_tracking.to_csv(output_dir + '/' + timeframe + '_' + filename + '_pv_' + tech + '.csv', index=True)
        
        return df_tracking 


######################
## capacity factors ##
######################

## pv capacity factors ##
def pv_capacity_factors(
        cutout,
        coords,
        solar_panel,
        tracking=None,
        bifaciality = 0.,
        pv_slope=36.7,
        pv_azimuth=180,
        altitude_threshold=1.0,
        delete_vars=0,
        timeframe=None,
        filename=None,
        write_raw_data=False,
        output_dir='output/',
        tech_label='pv',
        optimal_tilt=False,
        return_categories=False,
        return_raw_df=False
):
    start = timeit.timeit()

    def pv_angles(sun_alt, sun_azi, pv_slope, pv_azimuth, tracking):
    
        if tracking == None:
            cosincidence = sin(pv_slope) * cos(sun_alt) * cos(
                pv_azimuth - sun_azi
            ) + cos(pv_slope) * sin(sun_alt)

        elif tracking == "horizontal":  # horizontal tracking with horizontal axis
            axis_azimuth = pv_azimuth  # here orientation['azimuth'] refers to the azimuth of the tracker axis.
            rotation = np.arctan(
                (cos(sun_alt) / sin(sun_alt)) * sin(sun_azi - axis_azimuth)
            )
            pv_slope = abs(rotation)
            pv_azimuth = axis_azimuth + np.arcsin(
                sin(rotation / sin(pv_slope))
            )  # the 2nd part yields +/-1 and determines if the panel is facing east or west
            cosincidence = cos(pv_slope) * sin(sun_alt) + sin(
                pv_slope
            ) * cos(sun_alt) * cos(sun_azi - pv_azimuth)

        elif tracking == "tilted_horizontal": # horizontal tracking with tilted axis'
            axis_tilt = pv_slope  # here orientation['slope'] refers to the tilt of the tracker axis.
            axis_azimuth = pv_azimuth

            #Sun's x, y, z coords
            sx = cos(sun_alt) * sin(sun_azi)
            sy = cos(sun_alt) * cos(sun_azi)
            sz = sin(sun_alt)

            #from sun coordinates projected onto surface
            sx_prime = sx * cos(axis_azimuth) - sy * sin(axis_azimuth)
            sz_prime = (
                sx * sin(axis_azimuth) * sin(axis_tilt)
                + sy * sin(axis_tilt) * cos(axis_azimuth)
                + sz * cos(axis_tilt)
            )
            #angle between sun's beam and surface
            rotation = np.arctan2(sx_prime, sz_prime)

            # Clip rotaition between the minimum and maximum angles.
            rotation = np.clip(rotation, -(pi / 2), (pi / 2))

            pv_slope = np.arccos(cos(rotation) * cos(axis_tilt))

            azimuth_difference = np.arcsin(np.clip(sin(rotation) / sin(pv_slope),
                                                   a_min=-1, a_max=1))

            azimuth_difference = np.where(abs(rotation) < (pi / 2),
                                          azimuth_difference,
                                          -azimuth_difference + np.sign(rotation) * pi)

            # handle pv_slope=0 case:
            azimuth_difference = np.where(sin(pv_slope) != 0, azimuth_difference, (pi / 2))

            pv_azimuth = (axis_azimuth + azimuth_difference) % (2*pi)

            cosincidence = cos(pv_slope) * sin(sun_alt) + sin(pv_slope) * cos(sun_alt) * cos(sun_azi - pv_azimuth)

        elif tracking == "vertical":  # vertical tracking, surface azimuth = sun_azi
            cosincidence = sin(pv_slope) * cos(sun_alt) + cos(
                pv_slope
            ) * sin(sun_alt)
        elif tracking == "dual":  # both vertical and horizontal tracking
            cosincidence = np.float64(1.0)
            pv_slope = np.deg2rad(90) - sun_alt
        else:
            assert False, (
                    "Values describing tracking system must be None for no tracking,"
                    + "'horizontal' for 1-axis horizontal tracking,"
                    + "tilted_horizontal' for 1-axis horizontal tracking of tilted panle,"
                    + "vertical' for 1-axis vertical tracking, or 'dual' for 2-axis tracking"
            )

        # fixup incidence angle: if the panel is badly oriented and the sun shines
        # on the back of the panel (incidence angle > 90degree), the irradiation
        # would be negative instead of 0; this is prevented here.
        cosincidence = cosincidence.clip(min=0)

        return cosincidence, pv_slope
    
    sun_alt = cutout.data['solar_altitude']
    sun_azi = cutout.data['solar_azimuth']

    if optimal_tilt and tracking is None:
        # Per-cell optimal fixed tilt/azimuth instead of one value for all sites.
        # Closed-form rule of thumb (Landau), tilt as a function of |latitude| for
        # max annual yield; azimuth faces the equator (180 deg N hemisphere, 0 deg
        # S). This is a per-cell (y, x) array that broadcasts through the existing
        # irradiance math at no extra cost - NOT a per-angle yield search (which
        # would be far heavier). Rough by design; good enough for siting.
        lat = cutout.data.y
        abslat = abs(lat)
        tilt_deg = xr.where(abslat <= 25, abslat * 0.87,
                            abslat * 0.76 + 3.1)   # >50 deg extrapolates; rough
        azi_deg = xr.where(lat >= 0, 180.0, 0.0)
        pv_slope = deg2rad(tilt_deg)
        pv_azimuth = deg2rad(azi_deg)
    else:
        pv_slope = deg2rad(pv_slope)
        pv_azimuth = deg2rad(pv_azimuth)

    cosincidence, pv_slope = pv_angles(sun_alt, sun_azi, pv_slope, pv_azimuth, tracking)

    def irradiance(direct, diffuse, albedo, cosincidence, pv_slope, sun_alt):
        k = cosincidence / sin(sun_alt)
        cos_slope = cos(pv_slope)

        influx = direct + diffuse
        direct_t = k * direct
        diffuse_t = (1.0 + cos_slope) / 2.0 * diffuse + albedo * influx * ((1.0 - cos_slope) / 2.0)
        total_t = direct_t.fillna(0.0) + diffuse_t.fillna(0.0)

        cap_alt = sun_alt < deg2rad(altitude_threshold)
        total_t = total_t.where(~(cap_alt | (direct + diffuse <= 0.01)), 0)
        
        return total_t

    influx_toa = cutout.data['influx_toa']
    influx_direct = cutout.data['influx_direct']
    influx_diffuse = cutout.data['influx_diffuse']
    
    def clip(influx, influx_max):
        return influx.clip(min=0, max=influx_max.transpose(*influx.dims).data)

    direct = clip(influx_direct, influx_toa)
    diffuse = clip(influx_diffuse, influx_toa - influx_direct)
    albedo = cutout.data['albedo']
    
    total_t = irradiance(direct, diffuse, albedo, cosincidence, pv_slope, sun_alt)

    #account for backside of bifacial panel
    '''Source: Durusoy, B., Ozden, T. & Akinoglu, B.G. Solar irradiation on the rear surface of bifacial solar modules: a modeling approach.     Sci Rep 10, 13300 (2020). https://doi.org/10.1038/s41598-020-70235-3'''
    if bifaciality > 0:
        pv_slope_back = deg2rad(180)-pv_slope
        pv_azimuth_back = pv_azimuth + deg2rad(180)

        if tracking == None:
            cosincidence_back, pv_slope_back = pv_angles(sun_alt, sun_azi, pv_slope_back, pv_azimuth_back, tracking)
        else:
            cosincidence_back = 0 #assuming that the sun would never directly hit the back of a tracked panel
            
        irradiance_back = irradiance(direct, diffuse, albedo, cosincidence_back, pv_slope_back, sun_alt)
        total_t = total_t + bifaciality * irradiance_back
        
    with open(f'./solarpanel/{solar_panel}.yaml', "r") as f:
        pc = yaml.safe_load(f)

    def _power_huld(irradiance, t_amb, pc):
        """
        AC power per capacity predicted by Huld model, based on W/m2 irradiance.

        Maximum power point tracking is assumed.

        [1] Huld, T. et al., 2010. Mapping the performance of PV modules,
            effects of module type and data averaging. Solar Energy, 84(2),
            p.324-338. DOI: 10.1016/j.solener.2009.12.002
        """

        # normalized module temperature
        T_ = (pc["c_temp_amb"] * t_amb + pc["c_temp_irrad"] * irradiance) - pc["r_tmod"]

        # normalized irradiance
        G_ = irradiance / pc["r_irradiance"]

        log_G_ = np.log(G_.where(G_ > 0))
        # NB: np.log without base implies base e or ln
        eff = (
                1
                + pc["k_1"] * log_G_
                + pc["k_2"] * (log_G_) ** 2
                + T_ * (pc["k_3"] + pc["k_4"] * log_G_ + pc["k_5"] * log_G_ ** 2)
                + pc["k_6"] * (T_ ** 2)
        )

        eff = eff.fillna(0.0).clip(min=0)

        da = G_ * eff * pc.get("inverter_efficiency", 1.0)
        da.attrs["units"] = "kWh/kWp"
        da = da.rename("capacity_factor")

        return da

    pv_panel = _power_huld(total_t, cutout.data['temperature'], pc)

    pv_df = pv_panel.to_dataframe().reset_index()

    pv_df = pd.merge(pv_df, coords, on=['x', 'y'])
    pv_df = pv_df[['time', 'coords', 'capacity_factor', 'region']]
    pv_df['capacity_factor'] = round(pv_df['capacity_factor'], 4)

    if return_raw_df:
        return pv_df

    if tracking ==None:

        df_inf, df_avg, df_opt = pivot_and_categorize(pv_df, tech='pv', timeframe=timeframe, filename=filename,
                                                      write_raw_data=write_raw_data,output_dir=output_dir,
                                                      out_tech=tech_label)

        if return_categories:
            # per-site inf/avg/opt label, for colouring usable-site maps
            return df_inf, df_avg, df_opt, _site_categories(pv_df)
        if delete_vars == 0:
            return df_inf, df_avg, df_opt
        else:
            del df_inf, df_avg, df_opt, df_pvt

    else:
        df_tracking = pivot_and_categorize(pv_df, tech=tracking, timeframe=timeframe, filename=filename, write_raw_data=write_raw_data,output_dir=output_dir)
    
        if delete_vars == 0:
            return df_tracking
        else:
            del df_tracking, df_pvt

    end = timeit.timeit()
    return print(end - start)



## wind onshore cpacity factors ##
def wind_onshore_capacity_factors(
        cutout,
        coords,
        onshore_turbine='Vestas_V112_3MW',
        delete_vars=0,
        timeframe=None,
        filename=None,
        write_raw_data=False,
        output_dir='output/',
        return_categories=False,
        return_raw_df=False
):

    start = timeit.timeit()

    wnd100 = cutout.data[['wnd100m', 'roughness']].to_dataframe().reset_index()
    wnd100.drop(columns=['lon', 'lat'], inplace=True, errors='ignore')
    wnd100.rename(columns={'wnd100m': 'u100', 'roughness': 'z'}, inplace=True)
    wnd100 = wnd100[(wnd100['x'].isin(coords['x'])) & (wnd100['y'].isin(coords['y']))]

    with open(f'./windturbines/{onshore_turbine}.yaml', "r") as f:
        conf = yaml.safe_load(f)

    turbine = dict(V=np.array(conf["V"]), POW=np.array(conf["POW"]), hub_height=conf["HUB_HEIGHT"],
                   P=np.max(conf["POW"]))
    V, POW, hub_height, P = itemgetter("V", "POW", "hub_height", "P")(turbine)

    wnd100['u'] = wnd100['u100'] * (np.log(hub_height / wnd100['z']) / np.log(100 / wnd100['z']))

    p_curve = pd.DataFrame(data=[V, POW])
    p_curve = p_curve.T
    p_curve.rename(columns={0: 'u', 1: 'P'}, inplace=True)

    wnd100['P'] = np.interp(wnd100['u'], p_curve['u'], p_curve['P'])
    wnd100['capacity_factor'] = round(wnd100['P'] / P, 4)
    wnd100 = wnd100[['time', 'y', 'x', 'capacity_factor']]

    wnd100 = pd.merge(wnd100, coords, on=['x', 'y'])

    wnd100 = wnd100[['time', 'coords', 'capacity_factor', 'region']]
    #wnd100 = wnd100.rename(columns={'iso_a2': 'region'})

    if return_raw_df:
        return wnd100

    df_inf, df_avg, df_opt = pivot_and_categorize(wnd100, tech='wind_onshore', timeframe=timeframe, filename=filename, write_raw_data=write_raw_data,output_dir=output_dir)

    if return_categories:
        return df_inf, df_avg, df_opt, _site_categories(wnd100)
    if delete_vars == 0:
        return df_inf, df_avg, df_opt
    else:
        del df_inf, df_avg, df_opt, df_pvt

    end = timeit.timeit()
    return print(end - start)

## wind offshore cpacity factors ##
def wind_offshore_capacity_factors(
        cutout,
        coords,
        offshore_turbine='Vestas_V112_3MW',
        delete_vars=0,
        timeframe=None,
        filename=None,
        write_raw_data=False,
        output_dir='output/',
        return_raw_df=False
):

    start = timeit.timeit()

    wnd100 = cutout.data[['wnd100m', 'roughness']].to_dataframe().reset_index()
    wnd100.drop(columns=['lon', 'lat'], inplace=True, errors='ignore')
    wnd100.rename(columns={'wnd100m': 'u100', 'roughness': 'z'}, inplace=True)
    wnd100 = wnd100[(wnd100['x'].isin(coords['x'])) & (wnd100['y'].isin(coords['y']))]

    with open(f'./windturbines/{offshore_turbine}.yaml', "r") as f:
        conf = yaml.safe_load(f)

    turbine = dict(V=np.array(conf["V"]), POW=np.array(conf["POW"]), hub_height=conf["HUB_HEIGHT"],
                   P=np.max(conf["POW"]))
    V, POW, hub_height, P = itemgetter("V", "POW", "hub_height", "P")(turbine)

    wnd100['u'] = wnd100['u100'] * (np.log(hub_height / wnd100['z']) / np.log(100 / wnd100['z']))

    p_curve = pd.DataFrame(data=[V, POW])
    p_curve = p_curve.T
    p_curve.rename(columns={0: 'u', 1: 'P'}, inplace=True)

    wnd100['P'] = np.interp(wnd100['u'], p_curve['u'], p_curve['P'])
    wnd100['capacity_factor'] = round(wnd100['P'] / P, 4)
    wnd100 = wnd100[['time', 'y', 'x', 'capacity_factor']]

    wnd100 = pd.merge(wnd100, coords, on=['x', 'y'])

    wnd100['distance'] = round(wnd100['distance'].astype(float), 1)
    keep = ['time', 'coords', 'capacity_factor', 'distance', 'region']
    if 'depth' in wnd100.columns:
        keep.insert(4, 'depth')
    wnd100 = wnd100[keep]

    if return_raw_df:
        return wnd100

    df_shallow, df_transitional, df_deep = pivot_and_categorize(wnd100, tech='wind_offshore', timeframe=timeframe, filename=filename,write_raw_data=write_raw_data,output_dir=output_dir)

    if delete_vars == 0:
        return df_shallow, df_transitional, df_deep
    else:
        del df_shallow, df_transitional, df_deep

    end = timeit.timeit()
    print(end - start)
    return


## temperature-dependent timeseries ##
def temperature_timeseries(
        cutout,
        coords,
        delete_vars=0,
        timeframe=None,
        filename=None,
        write_raw_data=False,
        output_dir='output/',
        return_raw_dfs=False
):

    start = timeit.timeit()

    temp = cutout.data['temperature'].to_dataframe().reset_index()
    
    temp.drop(columns=['lon', 'lat'], inplace=True, errors='ignore')
    
    temp = temp[(temp['x'].isin(coords['x'])) & (temp['y'].isin(coords['y']))]
    
    vorlauftemp = 55+273.15
    temp['heatpump_cop'] = 1/(vorlauftemp/(vorlauftemp-temp['temperature']))
    
    temp['temperature'] = round(temp['temperature']-273.15,2)
    
    temp = pd.merge(temp, coords, on=['x', 'y'])
    
    temp = temp[['time','coords','temperature','heatpump_cop','region']]

    soil_temp = cutout.data['soil temperature'].to_dataframe().reset_index()
    
    soil_temp.drop(columns=['lon', 'lat'], inplace=True, errors='ignore')
    
    soil_temp = soil_temp[(soil_temp['x'].isin(coords['x'])) & (soil_temp['y'].isin(coords['y']))]
    
    vorlauftemp = 55+273.15
    soil_temp['heatpump_cop'] = 1/(vorlauftemp/(vorlauftemp-soil_temp['soil temperature']))
    
    soil_temp['soil temperature'] = round(soil_temp['soil temperature']-273.15,2)
    
    soil_temp = pd.merge(soil_temp, coords, on=['x', 'y'])
    
    soil_temp = soil_temp[['time','coords','soil temperature','heatpump_cop','region']]

    if return_raw_dfs:
        return temp, soil_temp
   
    df_temp = pd.pivot_table(temp, values='temperature', index='time', columns='region', aggfunc=np.mean).copy()
    
    df_heatpump_cop = pd.pivot_table(temp, values='heatpump_cop', index='time', columns='region', aggfunc=np.mean).copy()
    
    #Output raw data, if wanted
    if (write_raw_data==True): 
        temp.to_csv(output_dir+'/'+timeframe+'_temperature_raw.csv', index=True)
    
    df_temp.to_csv(output_dir+'/'+timeframe+'_temperature_'+filename+'.csv', index=True)
    
    temp['heating_demand'] = round((20-temp['temperature']).clip(lower=0),4)
    
    mean_per_region = temp.groupby('region')['heating_demand'].transform('mean')
    
    temp['heating_demand'] = round((temp['heating_demand']/mean_per_region + 0.25) /  1.25,4)
    
    temp['cooling_demand'] = round((temp['temperature']-22).clip(lower=0),4)
    
    df_heating = pd.pivot_table(temp, values='heating_demand', index='time', columns='region', aggfunc=np.mean).copy()
    
    df_cooling = pd.pivot_table(temp, values='cooling_demand', index='time', columns='region', aggfunc=np.mean).copy()
    
    df_heating.to_csv(output_dir+'/'+timeframe+'_heating_'+filename+'.csv', index=True)
    df_cooling.to_csv(output_dir+'/'+timeframe+'_cooling_'+filename+'.csv', index=True)
    
    df_heatpump_cop.to_csv(output_dir+'/'+timeframe+'_heatpump_cop_'+filename+'.csv', index=True)
    
    df_heatpump_ground_cop = pd.pivot_table(soil_temp, values='heatpump_cop', index='time', columns='region', aggfunc=np.mean).copy()
    
    df_heatpump_ground_cop.to_csv(output_dir+'/'+timeframe+'_heatpump_ground_cop_'+filename+'.csv', index=True)


    if delete_vars == 0:
        return df_heatpump_ground_cop, df_cooling, df_heating, df_heatpump_cop
    else:
        del df_heatpump_ground_cop, df_cooling, df_heating, df_heatpump_cop

    end = timeit.timeit()
    print(end - start)
    return

## hydro run-of-river capacity factors ##
def compute_regional_runoff_volume(cutout, geo_file=None, shapes=None):
    """Computes hourly aggregated runoff volume (m^3) for a given region."""
    if "runoff" not in cutout.data.data_vars and "ro" not in cutout.data.data_vars:
        print("Runoff variable missing in cutout. Preparing 'runoff' feature...")
        cutout.prepare(features=["runoff"])
        
    var_name = "runoff" if "runoff" in cutout.data.data_vars else "ro"

    if shapes is None:
        if geo_file is None:
            raise ValueError("Must provide either geo_file or shapes to compute runoff volume.")
        shapes = gpd.read_file(geo_file)
        
    gdf_region = shapes.to_crs("EPSG:4326")
    geom_union = gdf_region.union_all()

    lons, lats = np.meshgrid(cutout.data.x.values, cutout.data.y.values)
    pts = gpd.GeoSeries(gpd.points_from_xy(lons.flatten(), lats.flatten()), crs="EPSG:4326")
    inside_mask = pts.intersects(geom_union).values.reshape(len(cutout.data.y), len(cutout.data.x))

    mask_da = xr.DataArray(
        inside_mask.astype(float),
        coords={"y": cutout.data.y, "x": cutout.data.x},
        dims=["y", "x"]
    )

    if inside_mask.sum() == 0:
        print("Warning: No grid cells matched geometry for runoff calculation.")
        return pd.Series(0.0, index=cutout.data.indexes["time"])

    # Cell area calculation (m^2 per grid cell)
    dx = float(np.abs(cutout.data.x[1] - cutout.data.x[0])) if len(cutout.data.x) > 1 else 0.25
    dy = float(np.abs(cutout.data.y[1] - cutout.data.y[0])) if len(cutout.data.y) > 1 else 0.25
    lat_rad = np.deg2rad(cutout.data.y)
    cell_area_m2 = (dx * 111320.0 * np.cos(lat_rad)) * (dy * 110574.0)
    area_da = cell_area_m2.broadcast_like(mask_da)

    # Runoff volume (m^3) = depth (m) * area (m^2)
    da_runoff = cutout.data[var_name]
    hourly_volume = (da_runoff * mask_da * area_da).sum(dim=["x", "y"]).to_pandas()
    return hourly_volume

def hydro_ror_capacity_factors(
        cutout=None,
        geo_file=None,
        shapes=None,
        raw_volume=None,
        timeframe=None,
        filename=None,
        output_dir='output/',
        smoothing_window=48,
        percentile_sizing=95,
        return_raw_volume=False
):
    start = timeit.timeit()
    
    if raw_volume is None:
        if cutout is None:
            raise ValueError("Must provide either cutout or raw_volume to hydro_ror_capacity_factors.")
        hourly_volume = compute_regional_runoff_volume(cutout, geo_file=geo_file, shapes=shapes)
    else:
        hourly_volume = raw_volume

    if return_raw_volume:
        return hourly_volume

    # 48-hour moving average filter for hydraulic catchment lag
    smooth_series = (
        hourly_volume
        .rolling(window=smoothing_window, min_periods=1, center=True)
        .mean()
        .values
    )

    # 95th percentile sizing normalization
    p95 = np.percentile(smooth_series, percentile_sizing)

    if p95 > 0:
        ror_cf = np.clip(smooth_series / p95, a_min=0.0, a_max=1.0)
    else:
        ror_cf = np.zeros_like(smooth_series)

    df_ror = pd.DataFrame(
        {filename: np.round(ror_cf, 4)},
        index=hourly_volume.index
    )

    csv_out = os.path.join(output_dir, f"{timeframe}_hydro_ror_{filename}.csv")
    df_ror.to_csv(csv_out, index=True)
    
    end = timeit.timeit()
    return df_ror

def create_output_folder(timeframe):
    current_folder = os.getcwd()
    output_dir = os.path.join(current_folder, 'output', timeframe)

    # Check if the directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("You successfully created the output folder.")
    else:
        print("The output folder already exists.")

    return output_dir


def plot_country_map(cutout, zoom=False, size=8, zoom_factor_x=5, zoom_factor_y=5):

    # Retrieve the grid cells
    cells = cutout.grid

    # Load natural earth low resolution data
    df = gpd.read_file(shpreader.natural_earth(resolution='110m', category='cultural', name='admin_0_countries'))

    # Create a GeoSeries from the union of the grid cells
    country_bound = gpd.GeoSeries(cells.union_all())

    # Determine the center of the map
    map_center_x, map_center_y = np.mean(country_bound.centroid.x), np.mean(country_bound.centroid.y)

    # Set up the projection with the calculated center
    projection = ccrs.Orthographic(map_center_x, map_center_y)

    # Create the plot
    fig, ax = plt.subplots(subplot_kw={'projection': projection}, figsize=(size, size))

    # Plot the world data
    df.plot(ax=ax, transform=ccrs.PlateCarree())

    # Plot the country boundaries
    country_bound.plot(ax=ax, edgecolor='orange', facecolor='None', transform=ccrs.PlateCarree())

    # Zoom in if requested
    if zoom:
        min_x, min_y, max_x, max_y = country_bound.total_bounds
        ax.set_extent([min_x-zoom_factor_x, max_x+zoom_factor_x, min_y-zoom_factor_y, max_y+zoom_factor_y], crs=ccrs.PlateCarree())

    # Adjust the layout
    fig.tight_layout()

def get_country_geometry(regions, natural_earth_dataset="admin_0_map_units"):

    # Import country mapping from alpha-2 to full name
    country_mapping = pd.read_csv("geodata/iso_codes_all.csv", usecols=["name", "alpha-2"], index_col=0, encoding="latin").squeeze("columns").to_dict()

    # Rename regions list
    regions_name_en = [country_mapping.get(code, 'Unknown') for code in regions]

    # Use natural earth data for region polygon selection
    shpfilename = shpreader.natural_earth(resolution="10m", category="cultural", name=natural_earth_dataset)
    reader = shpreader.Reader(shpfilename)

    # Create a dictionary with NAME_EN and ADMIN keys for each record
    country_geometries = {r.attributes["NAME_EN"]: r.geometry for r in reader.records()}
    admin_geometries = {r.attributes["ADMIN"]: r.geometry for r in reader.records()}

    geometries = {}

    # Get polygons of the selected regions
    if len(regions[0]) > 2 and regions[0] != "CN-TW":
        print("full_names")
        for region in regions:
            if region in country_geometries:
                geometries[region] = country_geometries[region]
            elif region in admin_geometries:
                geometries[region] = admin_geometries[region]
            else:
                geometries[region] = None

        # Create the GeoSeries using the matched geometries
        country = gpd.GeoSeries(geometries, crs="epsg:4326")
    else:
        for region in regions_name_en:
            if region in country_geometries:
                geometries[region] = country_geometries[region]
            elif region in admin_geometries:
                geometries[region] = admin_geometries[region]
            else:
                geometries[region] = None

        # Create the GeoSeries using the matched geometries
        country = gpd.GeoSeries(geometries, crs="epsg:4326")
    if country.isnull().any():
        raise Exception("Something went wrong: The country geometry could not be created. Please check your region codes and the regions mapping. Otherwise, you can also use full country names by using the argument 'use_full_names=True'")
    return country

# ---------------------------------------------------------------------------
# Persistent ERA5 download cache (atlite monkeypatch)
# ---------------------------------------------------------------------------
# atlite 0.6.1 downloads every ERA5 chunk into a throwaway temp dir and deletes it,
# so any interrupted or failed prepare() re-downloads everything from the CDS. We
# replace atlite's per-chunk download (atlite.datasets.era5.retrieve_data) with a
# version that stores each downloaded chunk in a persistent cache keyed by its
# exact CDS request, and reuses it on the next run. Intermediate files therefore
# live in ERA5_CACHE_DIR (pointed at <folder>/era5_cache by get_cutout) and
# survive crashes, so a re-run only redoes the local merge + write, not downloads.
import cdsapi
import hashlib
import json
from contextlib import nullcontext
from atlite.datasets import era5 as _era5

ERA5_CACHE_DIR = os.path.join("cutouts", "era5_cache")


def _retrieve_data_cached(product, chunks=None, tmpdir=None, lock=None, **updates):
    """Drop-in replacement for atlite.datasets.era5.retrieve_data with an on-disk
    cache. Downloads a CDS chunk only if it is not already cached; otherwise reads
    the cached file. Cached files persist in ERA5_CACHE_DIR across runs."""
    request = {"product_type": ["reanalysis"], "download_format": "unarchived"}
    request.update(updates)
    assert {"year", "month", "variable"}.issubset(request), (
        "Need to specify at least 'variable', 'year' and 'month'"
    )

    data_format = request.get("data_format", "grib")
    suffix = f".{data_format}"
    key = hashlib.md5(
        json.dumps({"product": product, "request": request},
                   sort_keys=True, default=str).encode()
    ).hexdigest()
    os.makedirs(ERA5_CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(ERA5_CACHE_DIR, key + suffix)

    timestr = f"{request['year']}-{request['month']}"
    varstr = ", ".join(np.atleast_1d(request["variable"]))

    if os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
        _era5.logger.info(f"CDS: reusing cached download for {varstr} ({timestr})")
    else:
        client = cdsapi.Client(info_callback=_era5.logger.debug,
                               debug=logging.DEBUG >= logging.root.level)
        result = client.retrieve(product, request)
        with (lock if lock is not None else nullcontext()):
            _era5.logger.info(f"CDS: downloading {varstr} ({timestr})")
            part = cache_file + ".part"
            result.download(part)
            os.replace(part, cache_file)  # only a complete download becomes cache

    # Open from the cache. Pass a non-None tmpdir so atlite does NOT attach a
    # finalizer that would delete our persistent cache file when the dataset closes.
    keepdir = tmpdir if tmpdir is not None else ERA5_CACHE_DIR
    if data_format == "grib":
        return _era5.open_with_grib_conventions(cache_file, chunks=chunks, tmpdir=keepdir)
    return xr.open_dataset(cache_file, chunks=_era5.sanitize_chunks(chunks))


# Apply the patch once, on import.
if getattr(_era5.retrieve_data, "__name__", "") != "_retrieve_data_cached":
    _era5.retrieve_data = _retrieve_data_cached


# ERA5 variables this tool prepares (height, wind, influx, temperature features).
# Used to estimate the total number of downloaded data points.
_CUTOUT_VARS = ('height', 'wnd100m', 'roughness', 'influx_toa', 'influx_direct',
                'influx_diffuse', 'albedo', 'solar_altitude', 'solar_azimuth',
                'temperature', 'soil temperature','runoff')


def _determine_cutout_bounds(regions=None, geo_file=None, cutout_north_west=None,
                             cutout_south_east=None,
                             natural_earth_dataset="admin_0_map_units"):
    """Return the (minx, miny, maxx, maxy) lon/lat bounds the cutout will span.

    Same precedence as get_cutout: explicit NW/SE corners win, else a region
    geojson, else country ISO codes (region/country sources buffered by 1 degree).
    """
    if bool(cutout_north_west) and bool(cutout_south_east):
        nw, se = cutout_north_west, cutout_south_east  # (lat, lon) each
        return (nw[1], se[0], se[1], nw[0])

    shapes = gpd.read_file(geo_file) if geo_file is not None else None
    if shapes is not None and ("region" in shapes.columns) and ("iso_a2" not in shapes.columns):
        return tuple(shapes.to_crs("EPSG:4326").union_all().buffer(1.0).bounds)
    elif regions:
        return tuple(get_country_geometry(regions=regions, natural_earth_dataset=natural_earth_dataset).buffer(1.0).union_all().bounds)
    else:
        raise Exception("get_cutout: provide cutout_north_west/south_east coordinates, a geo_file with region polygons, or a list of country ISO codes in 'regions'")


def _report_cutout_size(bounds, timeframe, dx, dy, n_vars=len(_CUTOUT_VARS)):
    """Print and return the expected cutout grid size for the given bounds.

    Uses atlite's own get_coords so the cell/time counts match what prepare()
    will build exactly (global np.arange grid sliced inclusively by bounds,
    hourly time axis sliced by timeframe).
    """
    minx, miny, maxx, maxy = bounds
    coords = atlite_get_coords(x=slice(minx, maxx), y=slice(miny, maxy),
                               time=timeframe, dx=dx, dy=dy)
    nx, ny, nt = coords.sizes["x"], coords.sizes["y"], coords.sizes["time"]
    n_cells = nx * ny
    cell_hours = n_cells * nt          # data points per variable
    total = cell_hours * n_vars        # across all prepared variables
    raw_bytes = total * 4              # float32, uncompressed (rough upper bound)
    raw_gb = raw_bytes / 1e9
    raw_str = f"{raw_gb:.2f} GB" if raw_gb >= 1 else f"{raw_bytes / 1e6:.1f} MB"

    print("Expected cutout size:")
    print(f"  bounds (lon/lat) : [{minx:.2f}, {maxx:.2f}] x [{miny:.2f}, {maxy:.2f}]")
    print(f"  grid             : {nx} x {ny} = {n_cells:,} cells  (dx={dx}, dy={dy})")
    print(f"  time steps       : {nt:,} hourly  ({timeframe})")
    print(f"  points/variable  : {cell_hours:,}")
    print(f"  total points     : {total:,}  (x{n_vars} variables)")
    print(f"  uncompressed     : ~{raw_str} float32 (on-disk .nc is far smaller, compressed)")
    return {"nx": nx, "ny": ny, "n_cells": n_cells, "n_time": nt,
            "points_per_var": cell_hours, "total_points": total,
            "raw_gb": raw_gb}


def estimate_cutout_size(timeframe, regions=None, geo_file=None,
                         cutout_north_west=None, cutout_south_east=None,
                         dx=0.25, dy=0.25,
                         natural_earth_dataset="admin_0_map_units"):
    """Pre-compute how big a cutout will be BEFORE downloading anything.

    Accepts the same region/coordinate/resolution arguments as get_cutout and
    prints the grid dimensions, number of cells, hourly time steps and total
    data points. Returns a dict with the same numbers.
    """
    bounds = _determine_cutout_bounds(regions=regions, geo_file=geo_file,
                                      cutout_north_west=cutout_north_west,
                                      cutout_south_east=cutout_south_east,
                                      natural_earth_dataset=natural_earth_dataset)
    return _report_cutout_size(bounds, timeframe, dx, dy)


def _cutout_has_data(ds, var="temperature"):
    """True if the cutout actually holds data, not just variable definitions.

    A write that froze/was killed before its data chunks were flushed (e.g. an
    antivirus lock during the HDF5 close) leaves a file with the right variables
    and dimensions but all-NaN fill values. Checking the variable names is not
    enough, so sample one timestep of a variable that is never legitimately NaN
    (2 m temperature) and require at least one finite value.
    """
    if var not in ds.data_vars:
        var = next(iter(ds.data_vars), None)
        if var is None:
            return False
    sample = ds[var].isel(time=0).values
    return bool(np.isfinite(sample).any())


def get_cutout(filename, timeframe, module="era5", regions=None, geo_file=None, cutout_north_west=None, cutout_south_east=None, dx=0.25, dy=0.25, folder="cutouts/", natural_earth_dataset="admin_0_map_units", monthly_requests=None, concurrent_requests=None):
    dir = folder+filename+"_"+timeframe+"_"+str(int(dx*100))+"_"+str(int(dy*100))
    print(dir)

    # Store ERA5 chunk downloads in a persistent cache next to the cutouts, so an
    # interrupted/failed run reuses them instead of re-downloading from the CDS
    # (see the _retrieve_data_cached monkeypatch above).
    global ERA5_CACHE_DIR
    ERA5_CACHE_DIR = os.path.join(folder, "era5_cache")
    os.makedirs(ERA5_CACHE_DIR, exist_ok=True)

    # Reuse an existing cutout if it already holds every variable this tool needs.
    required_vars = {'height', 'wnd100m', 'roughness', 'influx_toa', 'influx_direct',
                     'influx_diffuse', 'albedo', 'solar_altitude', 'solar_azimuth',
                     'temperature', 'soil temperature'}
    if os.path.exists(dir + '.nc'):
        cutout = atlite.Cutout(path=dir)
        if required_vars.issubset(set(cutout.data.data_vars)) and _cutout_has_data(cutout.data):
            print("Cutout already prepared with required variables; skipping prepare().")
            return cutout
        # Incomplete or corrupt (missing vars, or all-NaN from a frozen write).
        # Must delete it: atlite would otherwise see prepared_features set and skip
        # the download, handing back the same NaN data. Close handle first (Windows).
        print("Existing cutout is incomplete/corrupt (all-NaN or missing vars); deleting and rebuilding.")
        cutout.data.close()
        del cutout
        gc.collect()
        os.remove(dir + '.nc')

    # Bounds from explicit corners, a region geojson, or country ISO codes (same
    # precedence for the explicit-coords and bounds paths; get_coords sorts so the
    # resulting grid is identical either way).
    bounds = _determine_cutout_bounds(regions=regions, geo_file=geo_file,
                                      cutout_north_west=cutout_north_west,
                                      cutout_south_east=cutout_south_east,
                                      natural_earth_dataset=natural_earth_dataset)

    # Report expected grid size before downloading anything.
    _report_cutout_size(bounds, timeframe, dx, dy)

    cutout = atlite.Cutout(path=dir,
                           module=module,
                           bounds=bounds,
                           dx=dx,
                           dy=dy,
                           time=timeframe)

    prepare_kwargs = {}
    if monthly_requests is not None:
        prepare_kwargs['monthly_requests'] = monthly_requests
    if concurrent_requests is not None:
        prepare_kwargs['concurrent_requests'] = concurrent_requests

    cutout.prepare(['height', 'wind', 'influx', 'temperature','runoff'], **prepare_kwargs)
    return cutout



def gis_get_country_geometry(regions=None,admin=None,cutout=None):
    if admin == 0:

        shapefile = get_country_geometry(regions=regions)
        country_mapping = pd.read_csv("geodata/iso_codes_all.csv", usecols=["name", "alpha-2"], index_col=0, encoding="latin").squeeze("columns").to_dict()
        regions_name_en = [country_mapping.get(code, 'Unknown') for code in regions]

        shapes = shapefile[shapefile.index.isin(regions_name_en)]
    
    elif admin == 1:
        #admin 1

        admin1_geodata = "geodata/natural_earth_world_admin1.geojson"
        shapefile_admin1_geodata = gpd.read_file(admin1_geodata)   

        country_mapping = pd.read_csv("geodata/iso_codes_all.csv", usecols=["name", "alpha-2"], index_col=0, encoding="latin").squeeze("columns").to_dict()
        regions_name_en = [country_mapping.get(code, 'Unknown') for code in regions] 
        
        # Use natural earth data for region polygon selection
        shpfilename = shpreader.natural_earth(resolution="10m", category="cultural", name="admin_0_map_units")
        reader = shpreader.Reader(shpfilename)
        # Collect records into a list of dictionaries with geometry and attributes
        records = []

        for r in reader.records():
            name_en = r.attributes["NAME_EN"]
            administration = r.attributes["ADMIN"]
            geometry = r.geometry

            # Add region to the list only if it's in the specified regions
            if name_en in regions_name_en:
                records.append({
                    "NAME_EN": name_en,
                    "ADMIN": administration,
                    "geometry": geometry})
        
        # Convert the list of records to a GeoDataFrame
        gdf = gpd.GeoDataFrame(records)

        filtered_shapefile = shapefile_admin1_geodata[shapefile_admin1_geodata.iso_a2.isin(regions)].set_index("iso_3166_2")
        shapes = gpd.overlay(filtered_shapefile,gdf,how='intersection')
        
    bounds = shapes.union_all().buffer(1).bounds
    plt.rc("figure", figsize=[10, 7])
    fig, ax = plt.subplots()
    shapes.plot(ax=ax)
    cutout.grid.plot(ax=ax, edgecolor="grey", color="None")

    return shapes,regions_name_en

from contextlib import contextmanager


@contextmanager
def _quiet_rasterio(verbose=False):
    """Silence rasterio/GDAL WARNING spam during raster masking unless verbose.

    atlite's availability calc emits one 'Value -128 ... changed to -128' GDAL
    warning per region/tile (harmless nodata note), which clogs the log. Raise the
    rasterio logger to ERROR while inside this block; restore it afterwards. Pass
    verbose=True to keep the warnings.
    """
    logger = logging.getLogger("rasterio")
    prev = logger.level
    if not verbose:
        logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(prev)


def calculate_and_plot_available_area(admin=None,cutout=None,shapes=None,regions_name_en=None,excluder=None,verbose=False,color=None):
    gp = shapes.loc[shapes.index].geometry.to_crs(excluder.crs)
    excluder.open_files()
    with _quiet_rasterio(verbose):
        masked, transform = excluder.compute_shape_availability(gp)

        fig, ax = plt.subplots()
        excluder.plot_shape_availability(gp)

        AvailablityMatrix = cutout.availabilitymatrix(shapes, excluder)

    # atlite names the shape dimension after the shapes index name (e.g. 'region'),
    # falling back to 'dim_0' when unnamed. Read it from the result so plotting works
    # for any region set / workflow rather than assuming 'dim_0'.
    shape_dim = AvailablityMatrix.dims[0]
    _cmap = _availability_cmap(color)

    if admin == 1:
        fg = AvailablityMatrix.plot(row=shape_dim, col_wrap=3, cmap=_cmap)
        fg.set_titles("{value}")
        for i, c in enumerate(shapes.index):
            shapes.plot(ax=fg.axs.flatten()[i], edgecolor="k", color="None")
    else:
        for c in AvailablityMatrix[shape_dim].values:
            fig, ax = plt.subplots()
            AvailablityMatrix.sel({shape_dim: c}).plot(cmap=_cmap)
            shapes.loc[[c]].plot(ax=ax, edgecolor="k", color="None")
            cutout.grid.plot(ax=ax, color="None", edgecolor="grey", ls=":")

    return AvailablityMatrix

def calculate_and_plot_available_rooftops(admin=None,cutout=None,shapes=None,regions_name_en=None,cities=None,verbose=False,color=None):
    rooftops = shapes.loc[shapes.index].geometry.to_crs(cities.crs)
    cities.open_files()
    with _quiet_rasterio(verbose):
        masked, transform = cities.compute_shape_availability(rooftops)

        fig, ax = plt.subplots()
        cities.plot_shape_availability(rooftops)

        AvailabilityMatrix_Rooftop = cutout.availabilitymatrix(shapes, cities)

    shape_dim = AvailabilityMatrix_Rooftop.dims[0]
    _cmap = _availability_cmap(color)

    if admin == 1:
        fg = AvailabilityMatrix_Rooftop.plot(row=shape_dim, col_wrap=3, cmap=_cmap)
        fg.set_titles("{value}")
        for i, c in enumerate(shapes.index):
            shapes.plot(ax=fg.axs.flatten()[i], edgecolor="k", color="None")
    else:
        for c in AvailabilityMatrix_Rooftop[shape_dim].values:
            fig, ax = plt.subplots()
            AvailabilityMatrix_Rooftop.sel({shape_dim: c}).plot(cmap=_cmap)
            shapes.loc[[c]].plot(ax=ax, edgecolor="k", color="None")
            cutout.grid.plot(ax=ax, color="None", edgecolor="grey", ls=":")

    return AvailabilityMatrix_Rooftop

def _availability_cmap(color=None):
    """Colormap for availability maps. None -> 'Greens'; a named matplotlib
    colormap is used as-is; a single colour (e.g. hex '#004664') builds a
    white->colour ramp so low availability stays white."""
    if color is None:
        return "Greens"
    if color in plt.colormaps():
        return color
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("avail", ["white", color])


def equal_area_crs(shapes):
    """Region-agnostic equal-area CRS: a Lambert Azimuthal Equal-Area projection
    centred on the data, so areas are correct for any region on Earth without
    hardcoding a national CRS. Accepts a GeoDataFrame/GeoSeries.
    """
    geom = shapes.to_crs("EPSG:4326").union_all()
    c = geom.centroid
    return (f"+proj=laea +lat_0={c.y:.6f} +lon_0={c.x:.6f} "
            "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")


def calculate_capacity_potentials(cutout=None,coords_onshore=None,AvailabilityMatrix=None,AvailabilityMatrix_Rooftop=None,AvailabilityMatrix_Wind=None,pv_cap_per_sqkm=100,pv_percent_land_available=0.03,wind_cap_per_sqkm=27,wind_percent_land_available=0.03,rooftop_cap_per_sqkm=100,rooftop_percent_area_available=0.2,area_crs=None):
    area = cutout.grid.set_index(["y", "x"]).to_crs(area_crs or equal_area_crs(cutout.grid)).area / 1e6
    area.name = "Area [km²]"
    # The availability matrix is indexed by (shape, y, x); collapse to one value per
    # (y, x) cell. Keep only y/x/value so the matrix's own shape-dimension column
    # (named after the shapes index, e.g. 'region') does not collide with the
    # 'region' column from coords_onshore on the merge below.
    availability_df = (AvailabilityMatrix.to_dataframe(name="availability")
                       .groupby(["y", "x"])["availability"].sum().reset_index())
    availability_rooftop_df = (AvailabilityMatrix_Rooftop.to_dataframe(name="availability rooftop")
                               .groupby(["y", "x"])["availability rooftop"].sum().reset_index())
    # Wind can use a separate availability matrix (e.g. extra exclusions for onshore
    # wind only). If none is given, wind reuses the PV/land availability.
    if AvailabilityMatrix_Wind is not None:
        availability_wind_df = (AvailabilityMatrix_Wind.to_dataframe(name="availability wind")
                                .groupby(["y", "x"])["availability wind"].sum().reset_index())
    else:
        availability_wind_df = availability_df.rename(columns={"availability": "availability wind"})
    merged_df = availability_df.merge(coords_onshore, on=['y', 'x'], how='inner')
    merged_df = merged_df.merge(availability_rooftop_df, on=['y', 'x'], how='inner')
    merged_df = merged_df.merge(availability_wind_df, on=['y', 'x'], how='inner')
    merged_df = merged_df.merge(area, on=['y', 'x'], how='inner')

    merged_df["Suitable Area PV [km²]"] = merged_df["Area [km²]"] * merged_df["availability"] * pv_percent_land_available
    merged_df["Suitable Area Wind [km²]"] = merged_df["Area [km²]"] * merged_df["availability wind"] * wind_percent_land_available
    merged_df["Suitable Area Rooftops [km²]"] = merged_df["Area [km²]"] * merged_df["availability rooftop"] * rooftop_percent_area_available
    merged_df["PV Capacity [GW]"] = merged_df["Area [km²]"] * merged_df["availability"] * pv_cap_per_sqkm * pv_percent_land_available / 1000
    merged_df["Wind Capacity [GW]"] = merged_df["Area [km²]"] * merged_df["availability wind"] * wind_cap_per_sqkm * wind_percent_land_available / 1000
    merged_df["Rooftop Capacity [GW]"] = merged_df["Area [km²]"] * merged_df["availability rooftop"] * rooftop_cap_per_sqkm * rooftop_percent_area_available / 1000

    output_df = merged_df.groupby("region")[["Area [km²]","Suitable Area PV [km²]","Suitable Area Wind [km²]","Suitable Area Rooftops [km²]","PV Capacity [GW]","Rooftop Capacity [GW]","Wind Capacity [GW]"]].sum().reset_index()

    return output_df


# ---------------------------------------------------------------------------
# Region-agnostic GIS potential helpers
# ---------------------------------------------------------------------------
# These contain no country-specific logic: the regional specifics (region
# polygons, land-cover raster + its class codes, protected-area files, capacity
# densities) are all passed in as arguments, so the same code works for any region.

def get_region_shapes(geo_file, regions=None, region_col="region"):
    """Load onshore region polygons from a geojson.

    Returns (GeoDataFrame indexed by region name, list of region names). 'regions'
    optionally subsets to a list of names; empty/None keeps all. Multi-row regions
    are dissolved to one geometry each. Use this (region workflow) instead of
    gis_get_country_geometry (country/NUTS workflow) so the GIS potentials use the
    same region polygons as the timeseries.
    """
    gdf = gpd.read_file(geo_file)
    if region_col not in gdf.columns:
        raise ValueError(f"geo_file '{geo_file}' has no '{region_col}' column; "
                         "the GIS potentials need a region-polygon geojson.")
    gdf = gdf.dissolve(by=region_col)
    if regions:
        gdf = gdf.loc[gdf.index.intersection(regions)]
    return gdf, list(gdf.index)


def _raster_crs(raster_file):
    """Native CRS of a raster as a string, or None if the raster declares none."""
    import rasterio
    with rasterio.open(raster_file) as src:
        return src.crs.to_string() if src.crs else None


def _safe_nodata(raster_file):
    """A nodata fill value that fits the raster's dtype.

    atlite's add_raster defaults nodata=255, which overflows signed/8-bit rasters
    (e.g. an int8 land-cover map: valid range -128..127) and makes rasterio.mask
    raise 'Cannot convert fill_value 255 to dtype int8'. Use the raster's own
    declared nodata if it has one, else the dtype minimum as a safe sentinel.
    """
    import rasterio
    import numpy as np
    with rasterio.open(raster_file) as src:
        if src.nodata is not None:
            return src.nodata
        dt = np.dtype(src.dtypes[0])
        if np.issubdtype(dt, np.integer):
            return int(np.iinfo(dt).min)
        return float(np.finfo(dt).min)


def make_land_excluder(land_cover_raster, exclude_codes, crs=None, raster_crs=None,
                       raster_nodata=None,
                       protected_files=None, protected_query=None,
                       protected_layer=None, protected_buffer=0):
    """Build an excluder for utility-scale PV / onshore wind land availability.

    land_cover_raster : categorical land-cover raster (e.g. CORINE).
    exclude_codes     : raster values to mark UNavailable (your strict land list).
    crs               : analysis CRS; defaults to the raster's native CRS (fast,
                         no raster reprojection; national land-cover rasters are
                         usually already equal-area).
    protected_files   : path or list of paths to protected-area vector files.
    protected_query   : pandas query to pre-filter protected areas before excluding
                         (e.g. "IUCN_CAT in ['Ia','Ib','II']" for strict protection).
    protected_buffer  : buffer in CRS units (metres) around protected areas.
    """
    excluder = ExclusionContainer(crs=crs or raster_crs or _raster_crs(land_cover_raster))
    # raster_crs is passed through for rasters that declare no CRS of their own;
    # nodata is forced to a dtype-safe value (atlite's default 255 overflows int8).
    nodata = raster_nodata if raster_nodata is not None else _safe_nodata(land_cover_raster)
    excluder.add_raster(land_cover_raster, codes=list(exclude_codes), crs=raster_crs, nodata=nodata)
    files = protected_files if protected_files is not None else []
    if isinstance(files, (str, os.PathLike, gpd.GeoDataFrame)):
        files = [files]
    for pf in files:
        gdf = _load_vector(pf, layer=protected_layer if not isinstance(pf, gpd.GeoDataFrame) else None,
                           query=protected_query)
        excluder.add_geometry(gdf, buffer=protected_buffer)
    return excluder


# Session cache for exclusion-layer vector reads. Big sources (national
# protected-area databases can hold hundreds of thousands of features and take
# tens of seconds per read) get re-read for every excluder build / scenario /
# offshore exclusion; caching the FILTERED result by (path, layer, query) makes
# repeats free while keeping memory bounded (only the subsets are kept).
_VECTOR_CACHE = {}


def _load_vector(src, layer=None, query=None):
    """Read a vector source with an optional pandas query, cached per session.

    src may be a path or an in-memory GeoDataFrame (returned filtered, uncached).
    Call _VECTOR_CACHE.clear() to free memory or pick up changed files.
    """
    if isinstance(src, gpd.GeoDataFrame):
        return src.query(query) if query else src
    key = (str(src), layer, query)
    if key in _VECTOR_CACHE:
        return _VECTOR_CACHE[key]
    gdf = gpd.read_file(src, layer=layer) if layer else gpd.read_file(src)
    if query:
        gdf = gdf.query(query)
    _VECTOR_CACHE[key] = gdf
    return gdf


def ordinance_effective_bans(ordinance_file, hub_height_m, rotor_diameter_m,
                             setback_threshold_m=1000,
                             setback_feature_types=("Structures", "Property Line"),
                             include_prohibitions=True, query=None):
    """Municipalities whose wind ordinance is a de-facto ban, as exclusion geometries.

    Local-ordinance layers in the reVX format store setbacks in mixed units (absolute metres,
    or multiples of max-tip height / hub height / rotor diameter). This converts
    every setback in `setback_feature_types` to metres using the given turbine
    dimensions and flags a municipality as effectively banned when its setback (from
    dwellings/structures or property lines) is >= setback_threshold_m - large
    setbacks leave no buildable land in practice. Optionally unions in the formal
    'Prohibitions' rows. Returns a GeoDataFrame of the affected polygons, ready for
    add_geometry / add_exclusion_layers.

    hub_height_m / rotor_diameter_m: turbine dims (max-tip = hub + rotor/2).
    Region-agnostic; pass any reVX-format ordinance file.
    """
    g = _load_vector(ordinance_file, query=query)
    max_tip = hub_height_m + rotor_diameter_m / 2.0

    def _to_m(row):
        val = pd.to_numeric(pd.Series([row.get("Value")]), errors="coerce").iloc[0]
        if pd.isna(val):
            # fall back to the explicit minimum-setback-distance column if present
            return pd.to_numeric(pd.Series([row.get("Minimum Setback Distance")]),
                                 errors="coerce").iloc[0]
        vt = str(row.get("Value Type", ""))
        if vt == "meters":
            return val
        if vt == "Max-tip Height Multiplier":
            return val * max_tip
        if vt == "Hub-height Multiplier":
            return val * hub_height_m
        if vt == "Rotor-Diameter Multiplier":
            return val * rotor_diameter_m
        return np.nan

    sb = g[g["Feature Type"].isin(list(setback_feature_types))].copy()
    if len(sb):
        sb["setback_m"] = sb.apply(_to_m, axis=1)
        effective = sb[sb["setback_m"] >= setback_threshold_m]
    else:
        effective = g.iloc[0:0]

    parts = [effective]
    if include_prohibitions:
        parts.append(g[g["Feature Type"] == "Prohibitions"])
    out = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=g.crs)


def add_exclusion_layers(excluder, layers):
    """Append extra vector exclusion layers to an existing ExclusionContainer.

    Region-agnostic. `layers` is a list of dicts, each describing one no-build
    source with independent layer/query/buffer (unlike make_land_excluder which
    applies one query to all files). Each dict:
        {"file": path | GeoDataFrame,        # required
         "layer": <layer name>,              # optional (gpkg / file-gdb)
         "query": <pandas query str>,        # optional pre-filter
         "buffer": <metres in excluder.crs>} # optional

    Used e.g. to build a wind-only excluder = land excluder + additional
    no-build areas (stricter protection classes, municipalities with a wind ban).
    Returns the same excluder (mutated) for chaining.
    """
    for spec in (layers or []):
        gdf = _load_vector(spec["file"], layer=spec.get("layer"), query=spec.get("query"))
        excluder.add_geometry(gdf, buffer=spec.get("buffer", 0))
    return excluder


def make_rooftop_excluder(land_cover_raster, developed_codes, crs=None, raster_crs=None,
                          raster_nodata=None):
    """Build a rooftop-area proxy excluder: keep ONLY the developed/built-up
    land-cover classes (everything else excluded). developed_codes are your
    land-cover dataset's urban/developed class values.
    """
    excluder = ExclusionContainer(crs=crs or raster_crs or _raster_crs(land_cover_raster))
    nodata = raster_nodata if raster_nodata is not None else _safe_nodata(land_cover_raster)
    excluder.add_raster(land_cover_raster, codes=list(developed_codes), invert=True, crs=raster_crs, nodata=nodata)
    return excluder


def calculate_offshore_potentials(cutout, coords_offshore,
                                  offshore_cap_per_sqkm=5,
                                  offshore_percent_available=0.1,
                                  shallow_max=30, transitional_max=60, floating_max=1000,
                                  area_crs=None):
    """Offshore wind potential per region (and depth class), region-agnostic.

    Reuses the EEZ-clipped, depth-classified offshore cells from get_coords
    (coords_offshore), so by construction only actual offshore zones are counted.
    Cell areas are computed in an equal-area projection.

    offshore_cap_per_sqkm / offshore_percent_available may be a scalar (applied to
    all cells) or a dict keyed by depth_class (e.g. a lower density / availability
    for floating deep water). Depth-class labels are read from the data, so nothing
    is hardcoded.
    """
    area = cutout.grid.set_index(["y", "x"]).to_crs(area_crs or equal_area_crs(cutout.grid)).area / 1e6
    area.name = "Area [km²]"
    df = coords_offshore.merge(area.reset_index(), on=["y", "x"], how="inner")

    # classify each offshore cell by water depth, same scheme as the offshore
    # timeseries (shallow/transitional fixed-bottom, deep floating; drop deeper).
    if "depth" in df.columns:
        df = df[(df["depth"] > 0) & (df["depth"] <= floating_max)].copy()
        df["depth_class"] = pd.cut(df["depth"],
                                   bins=[0, shallow_max, transitional_max, floating_max],
                                   labels=["shallow", "transitional", "deep"])
    else:
        df["depth_class"] = "all"

    def _lookup(value, c):
        return value.get(c, 0) if isinstance(value, dict) else value

    classes = df["depth_class"].astype(str)
    df["_cap"] = [ _lookup(offshore_cap_per_sqkm, c) for c in classes ]
    df["_pct"] = [ _lookup(offshore_percent_available, c) for c in classes ]

    df["Suitable Area Offshore [km²]"] = df["Area [km²]"] * df["_pct"]
    df["Offshore Wind Capacity [GW]"] = df["Area [km²]"] * df["_cap"] * df["_pct"] / 1000

    return df.groupby(["region", "depth_class"], observed=True)[
        ["Area [km²]", "Suitable Area Offshore [km²]", "Offshore Wind Capacity [GW]"]
    ].sum().reset_index()


def _split_capacity_by_category(coords_cat, cutout, cap_per_sqkm, percent_available,
                                tech_label, area_crs=None):
    """Split an onshore capacity into inf/avg/opt categories per region.

    coords_cat : usable coords with columns x, y, region, availability, category
                 (the inf/avg/opt site-quality label, from the q30/q70 CF split).
    Returns a wide DataFrame indexed by region with columns
    '{tech_label} {cat} [GW]' for cat in inf/avg/opt plus '{tech_label} Total [GW]'.
    Capacity per cell = cell_area * availability * cap_per_sqkm * percent_available
    / 1000, then summed per region x category. Reuses the categories already
    computed for the usable coords, so no extra capacity-factor run is needed.
    """
    if coords_cat is None or coords_cat.empty or "category" not in coords_cat.columns:
        return None
    area = (cutout.grid.set_index(["y", "x"])
            .to_crs(area_crs or equal_area_crs(cutout.grid)).area / 1e6).rename("area_km2")
    df = coords_cat.merge(area.reset_index(), on=["y", "x"], how="left")
    df["cap_gw"] = (df["area_km2"] * df["availability"]
                    * cap_per_sqkm * percent_available / 1000)
    wide = df.pivot_table(index="region", columns="category", values="cap_gw",
                          aggfunc="sum", observed=True)
    # keep a consistent inf/avg/opt column order where present
    order = [c for c in ["inf", "avg", "opt"] if c in wide.columns]
    wide = wide[order] if order else wide
    wide.columns = [f"{tech_label} {c} [GW]" for c in wide.columns]
    wide[f"{tech_label} Total [GW]"] = wide.sum(axis=1)
    return wide


def calculate_potentials_per_region(cutout, geo_file, coords_onshore,
                                    excluder, cities, wind_excluder=None,
                                    regions=None, region_col="region",
                                    pv_cap_per_sqkm=100, pv_percent_land_available=0.03,
                                    wind_cap_per_sqkm=27, wind_percent_land_available=0.03,
                                    rooftop_cap_per_sqkm=100, rooftop_percent_area_available=0.2,
                                    coords_offshore=None, usable_threshold=0.01,
                                    usable_threshold_rooftop=None, usable_round_to=2,
                                    offshore_cap_per_sqkm=5,
                                    offshore_percent_available=0.1,
                                    offshore_exclude_files=None, offshore_exclude_layer=None,
                                    offshore_exclude_query=None, offshore_exclude_buffer=0,
                                    area_crs=None, output_dir="output/", filename="",
                                    plot=True, plot_cols=3, plot_size=3, verbose=False,
                                    color=None, matrix_cache=None,
                                    add_category=True, pv_solar_panel=None, wind_turbine=None,
                                    pv_slope=36.7, pv_azimuth=180, optimal_tilt=False,
                                    rooftop_pv_slope=25, rooftop_pv_azimuth=180):
    """Memory-safe per-region GIS potentials with a stitched availability map.

    Processes one region at a time so atlite only ever rasterizes the land-cover
    raster over a single region's bounding box, not the whole multi-region extent
    (the part that blows up RAM at 30 m for a continent). Region-agnostic: regions,
    polygons and exclusion layers are all supplied by the caller.

    For each region it computes the land (PV/wind) and rooftop availability
    matrices, derives the capacity potentials, writes a per-region availability
    CSV, and accumulates the result into a single combined (y, x) map. Because the
    region polygons are disjoint, the per-region matrices are summed cell-by-cell
    into one stitched map covering all processed regions.

    Returns (combined_potentials_df, stitched_availability, stitched_rooftop,
             coords_onshore_usable, coords_rooftop_usable, coords_offshore_usable):
    - combined_potentials_df : capacity potentials for every region (one table).
    - stitched_availability  : xr.DataArray (y, x) land availability across regions.
    - stitched_rooftop       : xr.DataArray (y, x) rooftop availability across regions.
    - coords_onshore_usable  : onshore land coords (all regions) with availability
                               >= usable_threshold, for the PV / onshore-wind CF
                               timeseries.
    - coords_rooftop_usable  : coords with rooftop (developed) area, for a separate
                               rooftop-PV CF timeseries.
    - coords_offshore_usable : buildable offshore coords (depth-classed) if
                               coords_offshore was passed, else None.
    """
    shapes_all, names = get_region_shapes(geo_file, regions, region_col=region_col)
    os.makedirs(output_dir, exist_ok=True)

    stitch_land = None      # running (y, x) sum across regions (PV land)
    stitch_wind = None      # running (y, x) sum across regions (wind land)
    stitch_roof = None
    per_region_tables = []
    usable_onshore_parts = []   # usable PV land coords accumulated across regions
    usable_wind_parts = []      # usable wind land coords accumulated across regions
    usable_rooftop_parts = []   # usable rooftop coords accumulated across regions

    for name in names:
        print(f"Processing region: {name}")
        shapes = shapes_all.loc[[name]]

        # one-region availability; atlite crops the raster to this region's bounds.
        # 'land' is the PV/shared land matrix; 'wind' uses a separate excluder when
        # given (e.g. extra exclusions that apply to onshore wind only, such as
        # stricter protection classes or municipal wind bans), else it falls back
        # to the same land matrix.
        # matrix_cache (an externally supplied dict) reuses land/roof matrices across
        # repeated calls with the same excluder/cities - e.g. scenario sweeps where
        # only the wind excluder changes - skipping their recomputation entirely.
        with _quiet_rasterio(verbose):
            if matrix_cache is not None and (name, "land") in matrix_cache:
                land = matrix_cache[(name, "land")]
                roof = matrix_cache[(name, "roof")]
            else:
                land = cutout.availabilitymatrix(shapes, excluder, disable_progressbar=True)
                roof = cutout.availabilitymatrix(shapes, cities, disable_progressbar=True)
                if matrix_cache is not None:
                    matrix_cache[(name, "land")] = land
                    matrix_cache[(name, "roof")] = roof
            wind = (cutout.availabilitymatrix(shapes, wind_excluder, disable_progressbar=True)
                    if wind_excluder is not None else land)

        # drop the singleton shape dimension -> (y, x) map for this region
        shape_dim = land.dims[0]
        land_yx = land.sum(shape_dim)
        wind_yx = wind.sum(shape_dim)
        roof_yx = roof.sum(shape_dim)
        stitch_land = land_yx if stitch_land is None else stitch_land + land_yx
        stitch_wind = wind_yx if stitch_wind is None else stitch_wind + wind_yx
        stitch_roof = roof_yx if stitch_roof is None else stitch_roof + roof_yx

        # capacity potentials for just this region (reuses the existing function).
        # PV uses 'land', wind uses 'wind' (may differ); rooftop uses 'roof'.
        coords_region = coords_onshore[coords_onshore["region"] == name]
        if not coords_region.empty:
            region_df = calculate_capacity_potentials(
                cutout=cutout, coords_onshore=coords_region,
                AvailabilityMatrix=land, AvailabilityMatrix_Rooftop=roof,
                AvailabilityMatrix_Wind=wind,
                pv_cap_per_sqkm=pv_cap_per_sqkm, pv_percent_land_available=pv_percent_land_available,
                wind_cap_per_sqkm=wind_cap_per_sqkm, wind_percent_land_available=wind_percent_land_available,
                rooftop_cap_per_sqkm=rooftop_cap_per_sqkm, rooftop_percent_area_available=rooftop_percent_area_available,
                area_crs=area_crs)
            region_df.to_csv(os.path.join(output_dir, f"{filename}_potentials_{name}.csv"), index=False)
            per_region_tables.append(region_df)

            # usable sites for this region (cells with developable area), for the
            # capacity-factor timeseries on usable locations only. PV land, wind land
            # (own excluder), and rooftop each get their own usable set.
            roof_thresh = (usable_threshold_rooftop if usable_threshold_rooftop
                           is not None else usable_threshold)
            usable_onshore_parts.append(
                usable_onshore_coords(land, coords_region, threshold=usable_threshold,
                                      round_to=usable_round_to))
            usable_wind_parts.append(
                usable_onshore_coords(wind, coords_region, threshold=usable_threshold,
                                      round_to=usable_round_to))
            usable_rooftop_parts.append(
                usable_onshore_coords(roof, coords_region, threshold=roof_thresh,
                                      round_to=usable_round_to))

        # free the per-region full matrices before the next region
        del land, roof, land_yx, roof_yx
        if wind_excluder is not None:
            del wind, wind_yx
        gc.collect()

    combined = (pd.concat(per_region_tables, ignore_index=True)
                if per_region_tables else pd.DataFrame())

    # Append offshore wind potential as extra columns (one per depth class) on the
    # combined per-region table, so onshore + offshore live in one file.
    if coords_offshore is not None and not combined.empty:
        _off = calculate_offshore_potentials(
            cutout, coords_offshore,
            offshore_cap_per_sqkm=offshore_cap_per_sqkm,
            offshore_percent_available=offshore_percent_available,
            area_crs=area_crs)
        if not _off.empty:
            _wide = _off.pivot_table(index="region", columns="depth_class",
                                     values="Offshore Wind Capacity [GW]",
                                     aggfunc="sum", observed=True)
            _wide.columns = [f"Offshore Wind {c} [GW]" for c in _wide.columns]
            _wide["Offshore Wind Total [GW]"] = _wide.sum(axis=1)
            combined = combined.merge(_wide.reset_index(), on="region", how="left")

    # NOTE: the combined CSV is written later (after the inf/avg/opt category split is
    # available), so PV and onshore-wind capacities can be broken out per category.

    if plot and stitch_land is not None:
        # cell areas (km²) on the stitched (y, x) grid, in an equal-area projection,
        # so the headline availability is area-weighted (large cells count more).
        area_ser = (cutout.grid.set_index(["y", "x"])
                    .to_crs(area_crs or equal_area_crs(cutout.grid)).area / 1e6)
        area_da = (area_ser.rename("area").to_xarray()
                   .reindex(y=stitch_land.y, x=stitch_land.x))

        # one stitched chart each: the combined availability map of all regions, with
        # region outlines overlaid. Title shows the area-weighted mean availability
        # over covered cells (sum(availability*area) / sum(area)).
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # colormap: 'color' lets the caller pick the high-availability colour (e.g.
        # a hex like '#004664' -> white->colour ramp), or a named matplotlib colormap.
        _cmap = _availability_cmap(color)
        # size the figure to the data's aspect so a wide, short map does not sit in a
        # square frame (which is what made the default colorbar look oversized).
        xext = float(stitch_land.x.max() - stitch_land.x.min())
        yext = float(stitch_land.y.max() - stitch_land.y.min())
        width = plot_size * plot_cols
        height = max(2.0, width * (yext / xext if xext else 1.0))
        _maps = [("Land (PV)", stitch_land), ("Rooftop area", stitch_roof)]
        if wind_excluder is not None:
            _maps.insert(1, ("Land (wind)", stitch_wind))
        for label, stitched in _maps:
            covered = stitched > 0
            w = area_da.where(covered)
            denom = float(w.sum())
            avg = float((stitched.where(covered) * w).sum() / denom) if denom > 0 else 0.0
            fig, ax = plt.subplots(figsize=(width, height))
            im = stitched.plot(ax=ax, cmap=_cmap, add_colorbar=False)
            shapes_all.boundary.plot(ax=ax, edgecolor="k", linewidth=0.5)
            # colorbar tied to the map axes -> same height as the chart, thin width
            cax = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.1)
            fig.colorbar(im, cax=cax)
            ax.set_title(f"{label} availability: {avg * 100:.1f}%")

    coords_onshore_usable = (pd.concat(usable_onshore_parts, ignore_index=True)
                             if usable_onshore_parts else coords_onshore.iloc[0:0].copy())
    coords_wind_usable = (pd.concat(usable_wind_parts, ignore_index=True)
                          if usable_wind_parts else coords_onshore.iloc[0:0].copy())
    coords_rooftop_usable = (pd.concat(usable_rooftop_parts, ignore_index=True)
                             if usable_rooftop_parts else coords_onshore.iloc[0:0].copy())

    # Attach a site-quality 'category' (inf/avg/opt) per cell so the usable-coords
    # CSVs carry it for later plotting/analysis. PV land + rooftop use the PV
    # capacity factor; wind land uses the onshore-wind capacity factor. Same q30/q70
    # split as the timeseries. CF outputs go to a temp dir so no extra files written.
    if add_category and pv_solar_panel is not None and len(coords_onshore_usable):
        import tempfile
        _tmp = tempfile.mkdtemp()
        *_drop, _on_cat = pv_capacity_factors(
            cutout, coords_onshore_usable, pv_solar_panel,
            pv_slope=pv_slope, pv_azimuth=pv_azimuth, optimal_tilt=optimal_tilt,
            timeframe="catcalc", filename="_catcalc", output_dir=_tmp,
            return_categories=True)
        coords_onshore_usable = coords_onshore_usable.merge(
            _on_cat[["coords", "category"]], on="coords", how="left")
        if len(coords_rooftop_usable):
            *_drop, _rf_cat = pv_capacity_factors(
                cutout, coords_rooftop_usable, pv_solar_panel,
                pv_slope=rooftop_pv_slope, pv_azimuth=rooftop_pv_azimuth,
                timeframe="catcalc", filename="_catcalc", output_dir=_tmp,
                tech_label="pv_rooftop", return_categories=True)
            coords_rooftop_usable = coords_rooftop_usable.merge(
                _rf_cat[["coords", "category"]], on="coords", how="left")
    if add_category and wind_turbine is not None and len(coords_wind_usable):
        import tempfile
        _tmpw = tempfile.mkdtemp()
        *_drop, _wd_cat = wind_onshore_capacity_factors(
            cutout, coords_wind_usable, wind_turbine,
            timeframe="catcalc", filename="_catcalc", output_dir=_tmpw,
            return_categories=True)
        coords_wind_usable = coords_wind_usable.merge(
            _wd_cat[["coords", "category"]], on="coords", how="left")

    # Split PV and onshore-wind potential into inf/avg/opt categories (q30/q70 site
    # quality) and append as columns on the combined per-region table. Rooftop stays
    # a single column (already in `combined`). Reuses the categories computed above.
    if not combined.empty and "category" in coords_onshore_usable.columns:
        _pv_split = _split_capacity_by_category(
            coords_onshore_usable, cutout, pv_cap_per_sqkm, pv_percent_land_available,
            "PV Capacity", area_crs=area_crs)
        if _pv_split is not None:
            combined = combined.merge(_pv_split.reset_index(), on="region", how="left")
    # wind uses its own usable set (own excluder) if present, else the PV land set
    _wind_cat_src = (coords_wind_usable if (wind_excluder is not None
                     and "category" in coords_wind_usable.columns)
                     else coords_onshore_usable)
    if not combined.empty and "category" in _wind_cat_src.columns:
        _wd_split = _split_capacity_by_category(
            _wind_cat_src, cutout, wind_cap_per_sqkm, wind_percent_land_available,
            "Wind Capacity", area_crs=area_crs)
        if _wd_split is not None:
            combined = combined.merge(_wd_split.reset_index(), on="region", how="left")

    combined.to_csv(os.path.join(output_dir, f"{filename}_potentials_combined.csv"), index=False)

    coords_onshore_usable.to_csv(
        os.path.join(output_dir, f"{filename}_coords_onshore_usable.csv"), index=False)
    coords_rooftop_usable.to_csv(
        os.path.join(output_dir, f"{filename}_coords_rooftop_usable.csv"), index=False)
    # Wind usable coords are written only when wind has its own excluder (else they
    # equal the onshore/PV land set).
    if wind_excluder is not None:
        coords_wind_usable.to_csv(
            os.path.join(output_dir, f"{filename}_coords_wind_usable.csv"), index=False)

    coords_offshore_usable = None
    if coords_offshore is not None:
        # offshore 'category' = depth_class (its natural site class)
        coords_offshore_usable = usable_offshore_coords(
            coords_offshore, exclude_files=offshore_exclude_files,
            exclude_layer=offshore_exclude_layer, exclude_query=offshore_exclude_query,
            exclude_buffer=offshore_exclude_buffer)
        if "depth_class" in coords_offshore_usable.columns:
            coords_offshore_usable["category"] = coords_offshore_usable["depth_class"]
        coords_offshore_usable.to_csv(
            os.path.join(output_dir, f"{filename}_coords_offshore_usable.csv"), index=False)

    # Return signature kept at 6 for back-compatibility; the wind usable set (when a
    # separate wind_excluder is used) is written to {filename}_coords_wind_usable.csv.
    return (combined, stitch_land, stitch_roof,
            coords_onshore_usable, coords_rooftop_usable, coords_offshore_usable)


def usable_onshore_coords(availability_matrix, coords_onshore, threshold=0.01,
                          round_to=2):
    """Filter onshore coordinates to the cells that are actually usable.

    An availability matrix gives, per cutout cell, the share (0..1) of that cell
    that survives the exclusion layers. At coarse cutout resolution almost every
    cell has a tiny non-zero share, so a bare `> 0` test keeps nearly all cells.
    To avoid that, cells are kept only where the RAW share >= `threshold` (default
    0.01 = at least 1 % of the cell developable). `round_to` only rounds the value
    stored in the returned 'availability' column for readability; it does NOT
    affect which cells are kept (an earlier version rounded before thresholding,
    which bumped 0.005 up to 0.01 and effectively halved the threshold). Set
    threshold=0 to keep any non-zero cell.

    Rooftop note: the rooftop matrix is the built-up fraction of a cell, which is
    small even in populated cells, so it needs a higher threshold than land to
    drop near-empty cells - pass a larger `threshold` for rooftop.

    Feed the result into pv_capacity_factors / wind_onshore_capacity_factors to get
    a capacity-factor timeseries built ONLY from developable sites. Those functions
    still split the kept sites into opt/avg/inf by their own quantiles. Region-
    agnostic: the matrix's shape dimension is read from the data. Works on any
    availability matrix, e.g. pass a rooftop matrix to get rooftop-usable cells.
    """
    # collapse (shape, y, x) -> one availability value per (y, x) cell
    avail = (availability_matrix.to_dataframe(name="availability")
             .groupby(["y", "x"])["availability"].sum().reset_index())
    merged = coords_onshore.merge(avail, on=["y", "x"], how="inner")
    keep = merged[merged["availability"] >= threshold].copy()   # filter on RAW value
    if round_to is not None:
        keep["availability"] = keep["availability"].round(round_to)   # display only
    return keep.reset_index(drop=True)


def _drop_points_in_geometries(df, exclude_files, exclude_layer=None,
                               exclude_query=None, exclude_buffer=0):
    """Drop rows whose (x, y) point falls inside any exclusion polygon.

    Region-agnostic helper. exclude_files is a path or list of paths to vector
    files (shapefile/GeoJSON/GeoPackage/File-GDB layer). exclude_layer selects a
    layer in a multi-layer source; exclude_query pre-filters features; positive
    exclude_buffer (metres) grows the excluded polygons. Returns the kept rows.
    """
    files = exclude_files
    if isinstance(files, (str, os.PathLike, gpd.GeoDataFrame)):
        files = [files]

    pts = gpd.GeoDataFrame(df.copy(),
                           geometry=gpd.points_from_xy(df["x"], df["y"]),
                           crs="EPSG:4326")
    keep = pts
    for ef in files:
        gdf = _load_vector(ef, layer=exclude_layer if not isinstance(ef, gpd.GeoDataFrame) else None,
                           query=exclude_query)
        gdf = gdf.to_crs("EPSG:4326")
        # Sanitize: invalid polygons give wrong point-in-polygon answers. Some
        # datasets store antimeridian-crossing polygons as invalid bowties that
        # falsely 'contain' most of the globe and silently wipe all offshore
        # cells. Repair invalid geometries and drop antimeridian-wrap artifacts
        # (lon span >= 180 deg).
        gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
        invalid = ~gdf.geometry.is_valid
        if invalid.any():
            gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].make_valid()
        b = gdf.geometry.bounds
        wrap = (b["maxx"] - b["minx"]) >= 180
        if wrap.any():
            gdf = gdf[~wrap]
        if exclude_buffer:
            # buffer in metres via an equal-area projection centred on the data
            aea = equal_area_crs(keep)
            gdf = gdf.to_crs(aea).buffer(exclude_buffer).to_crs("EPSG:4326")
            gdf = gpd.GeoDataFrame(geometry=gdf, crs="EPSG:4326")
        inside = gpd.sjoin(keep, gdf[["geometry"]], how="left", predicate="within")
        keep = keep.loc[inside["index_right"].isna().groupby(level=0).all()]
    return keep.drop(columns="geometry").reset_index(drop=True)


def usable_offshore_coords(coords_offshore, shallow_max=30, transitional_max=60,
                           floating_max=1000, exclude_files=None,
                           exclude_layer=None, exclude_query=None, exclude_buffer=0):
    """Filter offshore coordinates to buildable sites.

    Two filters: (1) water depth within limits - keeps EEZ cells with
    0 < depth <= floating_max and tags depth_class (shallow / transitional / deep)
    matching the offshore timeseries; (2) optional exclusion of marine protected
    areas / other no-build zones via exclude_files (any vector source, e.g. a
    WDPA subset or a national marine-protected-area layer). Feed the
    result into wind_offshore_capacity_factors. Region-agnostic - all exclusion
    inputs are passed in.
    """
    if "depth" not in coords_offshore.columns:
        df = coords_offshore.copy()
    else:
        df = coords_offshore[(coords_offshore["depth"] > 0) &
                             (coords_offshore["depth"] <= floating_max)].copy()
        df["depth_class"] = pd.cut(df["depth"],
                                   bins=[0, shallow_max, transitional_max, floating_max],
                                   labels=["shallow", "transitional", "deep"])
    df = df.reset_index(drop=True)

    if exclude_files is not None and not df.empty:
        before = len(df)
        df = _drop_points_in_geometries(df, exclude_files, exclude_layer,
                                        exclude_query, exclude_buffer)
        print(f"offshore exclusion: dropped {before - len(df)} of {before} cells")
    return df.reset_index(drop=True)


def plot_offshore_map(coords_offshore, shapes=None, offshore_file=None,
                      color_by="depth_class", size=8, point_size=6):
    """Map the usable offshore cells, coloured by depth class (or region).

    coords_offshore : offshore cells from get_coords / usable_offshore_coords
                      (needs x, y and the color_by column).
    shapes          : optional onshore region polygons to draw for context.
    offshore_file   : optional EEZ geojson to outline the offshore zone.
    color_by        : 'depth_class' (default) or 'region'.
    """
    df = coords_offshore.copy()
    if color_by == "depth_class" and "depth_class" not in df.columns and "depth" in df.columns:
        df = usable_offshore_coords(df)

    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]),
                           crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(size, size))
    if offshore_file is not None:
        gpd.read_file(offshore_file).to_crs("EPSG:4326").boundary.plot(
            ax=ax, edgecolor="steelblue", linewidth=0.6)
    if shapes is not None:
        shapes.to_crs("EPSG:4326").boundary.plot(ax=ax, edgecolor="grey", linewidth=0.5)

    if color_by in pts.columns:
        for key, grp in pts.groupby(color_by, observed=True):
            grp.plot(ax=ax, markersize=point_size, label=str(key))
        ax.legend(title=color_by, fontsize=8)
    else:
        pts.plot(ax=ax, markersize=point_size)
    ax.set_title(f"Usable offshore cells by {color_by}")
    return ax

def plot_usable_coords(coords, shapes=None, boundary_file=None, color_by="availability",
                       categories=None, title=None, size=8, point_size=6, cmap="viridis"):
    """Map usable cells (onshore / rooftop / offshore) colour-coded.

    Generic version of plot_offshore_map for any usable-coords frame returned by
    the GIS step (coords_onshore_usable, coords_rooftop_usable, coords_offshore_usable).

    coords        : DataFrame with x, y and the color_by column.
    shapes        : optional region polygons drawn as grey outlines for context.
    boundary_file : optional vector file (e.g. EEZ geojson) outlined in blue.
    color_by      : column to colour by. Continuous (e.g. 'availability') -> colourbar;
                    categorical (e.g. 'category', 'depth_class', 'region') -> legend.
    categories    : optional [coords, category] DataFrame (from a CF function called
                    with return_categories=True). Merged on 'coords' and used as the
                    colour column; pass color_by='category' to colour inf/avg/opt.
    title         : plot title (auto-generated if None).
    """
    import pandas as pd
    df = coords.copy()
    if categories is not None and "coords" in df.columns:
        df = df.merge(categories[["coords", "category"]], on="coords", how="left")
    if color_by not in df.columns:
        # fall back to a sensible default per coord type
        for alt in ("availability", "depth_class", "region"):
            if alt in df.columns:
                color_by = alt
                break
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]),
                           crs="EPSG:4326")

    fig, ax = plt.subplots(figsize=(size, size))
    if boundary_file is not None:
        gpd.read_file(boundary_file).to_crs("EPSG:4326").boundary.plot(
            ax=ax, edgecolor="steelblue", linewidth=0.6)
    if shapes is not None:
        shapes.to_crs("EPSG:4326").boundary.plot(ax=ax, edgecolor="grey", linewidth=0.5)

    if color_by in pts.columns and pd.api.types.is_numeric_dtype(pts[color_by]):
        pts.plot(ax=ax, column=color_by, cmap=cmap, markersize=point_size,
                 legend=True, legend_kwds={"label": color_by, "shrink": 0.5})
    elif color_by in pts.columns:
        # fixed colour + order for the inf/avg/opt site quality classes
        cat_colors = {"inf": "#d73027", "avg": "#fee090", "opt": "#1a9850"}
        keys = (["inf", "avg", "opt"] if color_by == "category"
                else sorted(pts[color_by].dropna().astype(str).unique()))
        for key in keys:
            grp = pts[pts[color_by].astype(str) == key]
            if len(grp):
                grp.plot(ax=ax, markersize=point_size, label=str(key),
                         color=cat_colors.get(key))
        ax.legend(title=color_by, fontsize=8)
    else:
        pts.plot(ax=ax, markersize=point_size)

    ax.set_title(title or f"Usable cells ({len(pts)}) by {color_by}")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    return ax
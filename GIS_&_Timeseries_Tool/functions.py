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

import logging
import warnings
import timeit

warnings.simplefilter('ignore')
logging.captureWarnings(False)
logging.basicConfig(level=logging.INFO)

###########################
## preparing coordinates ##
###########################
def get_coords(
        cutout,
        regions,
        geo_file,
        admin
):

    coords_raw_df = cutout.data[["x", "y"]].to_dataframe().reset_index()
    coords_raw = coords_raw_df[["x", "y"]].values.tolist()

    gdf_polygons = gpd.read_file(geo_file)

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
        coords_offshore = coords_offshore.drop(columns=['subregion'])
    elif admin == 1:
        coords_onshore = coords_onshore.drop(columns=['region']).rename(columns={'subregion': 'region'})
        coords_offshore = coords_offshore.drop(columns=['region']).rename(columns={'subregion': 'region'})

    return coords_onshore, coords_offshore


########################################################
## help function to pivot, categorize and write files ##
########################################################
def pivot_and_categorize(
        df,
        tech,
        write_raw_data=False,
        timeframe=None,
        filename=None,
        output_dir='output/'
):

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
            df_inf.to_csv(output_dir + '/' + timeframe + '_' + filename + '_' + tech + '_inf_raw.csv', index=True)
            df_avg.to_csv(output_dir + '/' + timeframe + '_' + filename + '_' + tech + '_avg_raw.csv', index=True)
            df_opt.to_csv(output_dir + '/' + timeframe + '_' + filename + '_' + tech + '_opt_raw.csv', index=True)

        df_inf = pd.pivot_table(df_inf, values='capacity_factor', index='time', columns='region', aggfunc=np.mean)
        df_avg = pd.pivot_table(df_avg, values='capacity_factor', index='time', columns='region', aggfunc=np.mean)
        df_opt = pd.pivot_table(df_opt, values='capacity_factor', index='time', columns='region', aggfunc=np.mean)

        if write_raw_data == True:
            df.to_csv(output_dir + '/' + timeframe + '_' + tech + '_raw.csv', index=True)
            display(df)

        df_inf.to_csv(output_dir + '/' + timeframe + '_' + filename + '_' + tech + '_inf.csv', index=True)
        df_avg.to_csv(output_dir + '/' + timeframe + '_' + filename + '_' + tech + '_avg.csv', index=True)
        df_opt.to_csv(output_dir + '/' + timeframe + '_' + filename + '_' + tech + '_opt.csv', index=True)

        return df_inf, df_avg, df_opt
    
    elif (tech == "horizontal") | (tech == "tilted_horizontal") | (tech == "vertical") | (tech == "dual"):
        
        if write_raw_data == True:
            df.to_csv(output_dir + '/' + timeframe + '_' + filename + '_pv_' + tech + '_raw.csv', index=True)
          
        df_tracking = pd.pivot_table(df, values='capacity_factor', index='time', columns='region',aggfunc=np.mean).copy()
        df_tracking.to_csv(output_dir + '/' + timeframe + '_' + filename + '_pv_' + tech + '.csv', index=True)
        
        return df_tracking 

    elif tech == "wind_offshore":
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
        output_dir='output/'
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

    if tracking ==None:

        df_inf, df_avg, df_opt = pivot_and_categorize(pv_df, tech='pv', timeframe=timeframe, filename=filename,
                                                      write_raw_data=write_raw_data,output_dir=output_dir)

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
        output_dir='output/'
):

    start = timeit.timeit()

    wnd100 = cutout.data[['wnd100m', 'roughness']].to_dataframe().reset_index()
    wnd100.drop(columns=['lon', 'lat'], inplace=True)
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

    df_inf, df_avg, df_opt = pivot_and_categorize(wnd100, tech='wind_onshore', timeframe=timeframe, filename=filename, write_raw_data=write_raw_data,output_dir=output_dir)

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
        output_dir='output/'
):

    start = timeit.timeit()

    wnd100 = cutout.data[['wnd100m', 'roughness']].to_dataframe().reset_index()
    wnd100.drop(columns=['lon', 'lat'], inplace=True)
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
    wnd100 = wnd100[['time', 'coords', 'capacity_factor', 'distance', 'region']]

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
        output_dir='output/'
):

    start = timeit.timeit()

    temp = cutout.data['temperature'].to_dataframe().reset_index()
    
    temp.drop(columns=['lon', 'lat'], inplace=True)
    
    temp = temp[(temp['x'].isin(coords['x'])) & (temp['y'].isin(coords['y']))]
    
    vorlauftemp = 55+273.15
    temp['heatpump_cop'] = 1/(vorlauftemp/(vorlauftemp-temp['temperature']))
    
    temp['temperature'] = round(temp['temperature']-273.15,2)
    
    temp = pd.merge(temp, coords, on=['x', 'y'])
    
    temp = temp[['time','coords','temperature','heatpump_cop','region']]
   
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
    
    soil_temp = cutout.data['soil temperature'].to_dataframe().reset_index()
    
    soil_temp.drop(columns=['lon', 'lat'], inplace=True)
    
    soil_temp = soil_temp[(soil_temp['x'].isin(coords['x'])) & (soil_temp['y'].isin(coords['y']))]
    
    vorlauftemp = 55+273.15
    soil_temp['heatpump_cop'] = 1/(vorlauftemp/(vorlauftemp-soil_temp['soil temperature']))
    
    soil_temp['soil temperature'] = round(soil_temp['soil temperature']-273.15,2)
    
    soil_temp = pd.merge(soil_temp, coords, on=['x', 'y'])
    
    soil_temp = soil_temp[['time','coords','soil temperature','heatpump_cop','region']]
    
    df_heatpump_ground_cop = pd.pivot_table(soil_temp, values='heatpump_cop', index='time', columns='region', aggfunc=np.mean).copy()
    
    df_heatpump_ground_cop.to_csv(output_dir+'/'+timeframe+'_heatpump_ground_cop_'+filename+'.csv', index=True)


    if delete_vars == 0:
        return df_heatpump_ground_cop, df_cooling, df_heating, df_heatpump_cop
    else:
        del df_heatpump_ground_cop, df_cooling, df_heating, df_heatpump_cop

    end = timeit.timeit()
    print(end - start)
    return


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
    df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

    # Create a GeoSeries from the union of the grid cells
    country_bound = gpd.GeoSeries(cells.unary_union)

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

def get_cutout(filename, timeframe, module="era5", regions=None, cutout_north_west=None, cutout_south_east=None, dx=0.25, dy=0.25, folder="cutouts/", natural_earth_dataset="admin_0_map_units"):
    dir = folder+filename+"_"+timeframe+"_"+str(int(dx*100))+"_"+str(int(dy*100))
    print(dir)
    if regions == None:
        cutout = atlite.Cutout(path=dir,
                               module=module,
                               x=slice(cutout_north_west[1], cutout_south_east[1]), # Longitude
                               y=slice(cutout_north_west[0], cutout_south_east[0]), # Latitude
                               dx=dx,
                               dy=dy,
                               time=timeframe)

        cutout.prepare(['height', 'wind', 'influx', 'temperature'])
        return cutout

    elif (cutout_north_west==None) & (cutout_south_east==None):
        country = get_country_geometry(regions=regions, natural_earth_dataset=natural_earth_dataset)
        buffer = 1.0
        country = country.buffer(buffer)

        cutout = atlite.Cutout(path=dir,
                               module=module,
                               bounds=country.unary_union.bounds,
                               dx=dx,
                               dy=dy,
                               time=timeframe)

        cutout.prepare(['height', 'wind', 'influx', 'temperature'])
        return cutout

    else:
        raise Exception("It seems you specified both regions and cutout coordinates in the function arguments. Please only use one or the other")



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
        
    bounds = shapes.cascaded_union.buffer(1).bounds
    plt.rc("figure", figsize=[10, 7])
    fig, ax = plt.subplots()
    shapes.plot(ax=ax)
    cutout.grid.plot(ax=ax, edgecolor="grey", color="None")

    return shapes,regions_name_en

def calculate_and_plot_available_area(admin=None,cutout=None,shapes=None,regions_name_en=None,excluder=None):
    gp = shapes.loc[shapes.index].geometry.to_crs(excluder.crs)
    excluder.open_files()
    masked, transform = excluder.compute_shape_availability(gp)

    fig, ax = plt.subplots()
    excluder.plot_shape_availability(gp)

    AvailablityMatrix = cutout.availabilitymatrix(shapes, excluder)


    if admin == 1:

        fg = AvailablityMatrix.plot(row="dim_0", col_wrap=3, cmap="Greens")
        fg.set_titles("{value}")
        for i, c in enumerate(shapes.index):
            shapes.plot(ax=fg.axs.flatten()[i], edgecolor="k", color="None")

    elif admin == 0:
        for c in regions_name_en:
            fig, ax = plt.subplots()
            AvailablityMatrix.sel(dim_0=c).plot(cmap="Greens")
            shapes.loc[[c]].plot(ax=ax, edgecolor="k", color="None")
            cutout.grid.plot(ax=ax, color="None", edgecolor="grey", ls=":")

    return AvailablityMatrix 

def calculate_and_plot_available_rooftops(admin=None,cutout=None,shapes=None,regions_name_en=None,cities=None):
    rooftops = shapes.loc[shapes.index].geometry.to_crs(cities.crs)
    cities.open_files()
    masked, transform = cities.compute_shape_availability(rooftops)

    fig, ax = plt.subplots()
    cities.plot_shape_availability(rooftops)

    AvailabilityMatrix_Rooftop = cutout.availabilitymatrix(shapes, cities)

    if admin == 1:

        fg = AvailabilityMatrix_Rooftop.plot(row="dim_0", col_wrap=3, cmap="Greens")
        fg.set_titles("{value}")
        for i, c in enumerate(shapes.index):
            shapes.plot(ax=fg.axs.flatten()[i], edgecolor="k", color="None")

    elif admin == 0:
        for c in regions_name_en:
            fig, ax = plt.subplots()
            AvailabilityMatrix_Rooftop.sel(dim_0=c).plot(cmap="Greens")
            shapes.loc[[c]].plot(ax=ax, edgecolor="k", color="None")
            cutout.grid.plot(ax=ax, color="None", edgecolor="grey", ls=":")

    return AvailabilityMatrix_Rooftop

def calculate_capacity_potentials(cutout=None,coords_onshore=None,AvailabilityMatrix=None,AvailabilityMatrix_Rooftop=None,pv_cap_per_sqkm=100,pv_percent_land_available=0.03,wind_cap_per_sqkm=27,wind_percent_land_available=0.03,rooftop_cap_per_sqkm=100,rooftop_percent_area_available=0.2):
    area = cutout.grid.set_index(["y", "x"]).to_crs(3035).area / 1e6
    area.name = "Area [km²]"
    availability_df = AvailabilityMatrix.to_dataframe(name="availability").reset_index()
    availability_rooftop_df = AvailabilityMatrix_Rooftop.to_dataframe(name="availability rooftop").reset_index()
    merged_df = availability_df.merge(coords_onshore, on=['y', 'x'], how='inner')
    merged_df = merged_df.merge(availability_rooftop_df, on=['y', 'x'], how='inner')
    merged_df = merged_df.merge(area, on=['y', 'x'], how='inner')

    merged_df["Suitable Area PV [km²]"] = merged_df["Area [km²]"] * merged_df["availability"] * pv_percent_land_available
    merged_df["Suitable Area Wind [km²]"] = merged_df["Area [km²]"] * merged_df["availability"] * wind_percent_land_available
    merged_df["Suitable Area Rooftops [km²]"] = merged_df["Area [km²]"] * merged_df["availability rooftop"] * rooftop_percent_area_available
    merged_df["PV Capacity [GW]"] = merged_df["Area [km²]"] * merged_df["availability"] * pv_cap_per_sqkm * pv_percent_land_available / 1000
    merged_df["Wind Capacity [GW]"] = merged_df["Area [km²]"] * merged_df["availability"] * wind_cap_per_sqkm * wind_percent_land_available / 1000
    merged_df["Rooftop Capacity [GW]"] = merged_df["Area [km²]"] * merged_df["availability rooftop"] * rooftop_cap_per_sqkm * rooftop_percent_area_available / 1000

    output_df = merged_df.groupby("region")[["Area [km²]","Suitable Area PV [km²]","Suitable Area Wind [km²]","Suitable Area Rooftops [km²]","PV Capacity [GW]","Rooftop Capacity [GW]","Wind Capacity [GW]"]].sum().reset_index()

    return output_df


































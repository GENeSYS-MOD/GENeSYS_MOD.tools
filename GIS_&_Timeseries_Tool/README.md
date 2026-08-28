# GIS_&_Timeseries_Tool
Tools for generating timeseries used by the Global Energy System Model (GENeSYS-MOD)

## General information
This directory contains a comprehensive Python and Jupyter-based pipeline for extracting, spatializing, and processing climate and weather time series data from the ERA5 Reanalysis dataset (via the Copernicus Climate Data Store/*cdsapi*) and spatial geodatabases for global and regional energy system modeling (e.g., GENeSYS-MOD). 

The pipeline supports automated bounding-box clipping, anti-meridian/longitude-crossing region splitting, site availability/exclusion analysis, and timeseries generation for:
- Solar PV: Fixed utility-scale (with optional latitude-dependent optimal tilt), horizontal 1-axis tracking, rooftop PV, and site quality categorization ($P_{30}$/$P_{70}$ inf/avg/opt profiles).
- Wind Energy: Onshore and offshore wind turbines, marine boundary clipping via EEZs, and depth-stratified offshore classifications (shallow, transitional, deep).
- Thermal Demand & Heat Pumps: Ambient temperature, degree-day heating and cooling demands, and air-source & ground-source heat pump Coefficients of Performance (COP).
- Hydro Run-of-River (RoR): Catchment-aggregated runoff volume integration, hydrological lag smoothing (rolling window filter), and 95th-percentile capacity sizing.
- Mobility & Industrial High-Temperature Heat: Diurnal kernel profiles with calendar/holiday shifting and timezone normalization to UTC.

## Directory Contents & Data Flow
### Folders
*cutouts/*
Stores the generated NetCDF (*.nc*) climate datasets prepared by *atlite* for user-defined spatial bounds, resolutions (*dx*, *dy*), and timeframes.

*geodata/*
Contains spatial inputs such as:
- Continental/national boundary files (e.g., Natural Earth admin-0/admin-1 datasets).
- Exclusive Economic Zone (EEZ) land/marine union shapefiles (e.g., Marine Regions v4).
- High-resolution bathymetry and elevation grids (e.g., NOAA SRTM15+ / GEBCO *.nc* files).
- Dynamically generated regional and sub-regional *.geojson* polygons.

*output/*
Stores all generated time series and potential calculation CSV files organized into subfolders per timeframe (e.g., *output/2018/* or *output/2018-01-01/*).

### File Descriptions
*functions.py*
The primary backend module containing modular, region-agnostic algorithms:
- Coordinate Filtering & Siting (*get_coords*, *_get_coords_custom*): Assigns onshore grid cells to region polygons and attributes offshore cells to nearest coastal regions via spherical BallTrees, with optional seabed depth sampling.
- ERA5 Preparation & Persistent Caching (*get_cutout*, *_retrieve_data_cached*): Fetches required ERA5 atmospheric, radiation, wind, surface, and hydrological parameters (*height*, *wnd100m*, *roughness*, *influx_toa*, *influx_direct*, *influx_diffuse*, *albedo*, *solar_altitude*, *solar_azimuth*, *temperature*, *soil temperature*, *runoff*).
- Renewable & Weather Timeseries Calculators:
    - *pv_capacity_factors()*: Computes total surface irradiance (direct, diffuse, albedo, back-surface bifacial gain) and temperature-dependent power conversion using the Huld PV module model.
    - *wind_onshore_capacity_factors()* / *wind_offshore_capacity_factors()*: Logarithmic wind speed height extrapolation and power-curve interpolation for user-specified turbine configs.
    - *temperature_timeseries()*: Carnot COP formulas for air- and ground-source heat pumps alongside normalized heating and cooling degree-day demands.
    - *compute_regional_runoff_volume()* / *hydro_ror_capacity_factors()*: Latitude-dependent cell area weighting, spatial runoff depth integration, moving-average hydrological lag filtering, and percentile capacity factor scaling.
- GIS & Exclusion Mapping: Builds equal-area raster and vector excluders (land use, urban footprint, protected areas, setback distances) and derives developable gigawatt capacities.

*GENeSYS-MOD_RES_Tool.ipynb*
The master execution notebook structured into two dedicated workflows (Also includes experimental work with a GIS-based renewable energy potentials):

Section A: Standard Regional Workflow (Continuous Longitudes)
- Designed for contiguous regions (e.g., Europe, Africa, the Americas, Asia) that do not cross the $180^\circ$ meridian.
- Pulls regions directly from custom GeoJSONs or dissolves nation lists from *geodata/* boundary shapefiles.
- Builds the regional cutout, extracts onshore/offshore coordinates, and generates full-year or single-day timeseries.

Section B: Anti-Meridian Crossing Workflow (e.g., Oceania / Pacific)
- Resolves the problem where bounding boxes spanning both positive and negative longitudes ($+100^\circ$ to $-120^\circ$) unintentionally span $360^\circ$ across the globe.
- Splits polygons into West ($+100^\circ$ to $+180^\circ$) and East ($-180^\circ$ to $-120^\circ$) sub-regions using spatial *box()* clipping.
- Builds two distinct cutouts (*cutout_west*, *cutout_east*), computes raw coordinate-level data independently, concatenates the resulting dataframes across the unified region tag, and performs centralized pivoting and normalization.

*GENeSYS-MOD_mobility&High_heat_profile_generation_Tool.ipynb*
Generates normalized (annual mean $= 1.0$) 8,760-hour profiles for demand sectors where empirical hourly load profiles are missing:

Passenger Transport / Mobility:
- Constructs profiles using 24-hour diurnal kernels (*standard weekday*, *car-heavy*, *transit-heavy*, *hot-climate*, and *weekend*).
- Integrates Northern/Southern hemisphere seasonality curves, cultural travel surges (e.g., Chunyun / Spring Festival travel adjustments), and circular array index shifting to standardize regional time series to UTC.

High-Temperature Industrial Heat:
- Implements continuous baseload profiles with operational shifts differentiated by Northern vs. Southern hemisphere winter/summer seasons.
- Shifts profiles to align local industrial operation hours to UTC.
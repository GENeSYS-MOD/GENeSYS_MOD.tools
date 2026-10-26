import gdxpds
import yaml
import os
import pandas as pd
import genesysmod_to_iamc.data_wrapper as dw
import logging
from genesysmod_to_iamc._statics import *
from collections import OrderedDict


def loadmap_primary_energy_fuels():
    # currently static
    return {
        'Biomass': 'Primary Energy|Biomass',
        'Hardcoal': 'Primary Energy|Coal',
        'Gas_Natural': 'Primary Energy|Gas',
        'Oil': 'Primary Energy|Oil',
        'Lignite': 'Primary Energy|Coal',
        'Gas_Bio': 'Primary Energy|Biomass',
        'Biofuel': 'Primary Energy|Biomass',
    }


def loadmap_non_bio_renewables():
    # currently static
    return {
        'RES_CSP': 'Solar',
        'RES_Geothermal': 'Geothermal',
        'RES_Hydro_Large': 'Hydro',
        'RES_Hydro_Small': 'Hydro',
        'RES_Ocean': 'Ocean',
        'RES_PV_Rooftop_Commercial': 'Solar',
        'RES_PV_Rooftop_Residential': 'Solar',
        'RES_PV_Utility_Avg': 'Solar',
        'RES_PV_Utility_Inf': 'Solar',
        'RES_PV_Utility_Opt': 'Solar',
        'RES_Wind_Offshore_Transitional': 'Wind',
        'RES_Wind_Offshore_Shallow': 'Wind',
        'RES_Wind_Offshore_Deep': 'Wind',
        'RES_Wind_Onshore_Avg': 'Wind',
        'RES_Wind_Onshore_Inf': 'Wind',
        'RES_Wind_Onshore_Opt': 'Wind',
        'HLR_Solar_Thermal': 'Solar',
        'HLR_Geothermal': 'Geothermal',
        'HLI_Solar_Thermal': 'Solar',
        'HLI_Geothermal': 'Geothermal',
        'HLT_Rooftop_Residential': 'Solar',
        'HLT_Rooftop_Commercial': 'Solar',
        'HLT_Geothermal': 'Geothermal',
        'HHT_Geothermal': 'Geothermal',
    }


def loadmap_from_csv(file):
    filename = file + ".csv"
    return pd.read_csv(DEF_MAPPINGS_PATH / filename, header=None, index_col=0).squeeze("columns").to_dict()

def loaddf_from_csv(file):
    filename = file + ".csv"
    return pd.read_csv(DEF_MAPPINGS_PATH / filename, header=0)

def loadmap_iso2_countries():
    with open(Path(__file__).parent / 'mappings' / 'GENeSYS-MOD 4.0.yaml', 'r') as stream:
        country_codelist = yaml.load(stream, Loader=yaml.FullLoader)

        native_list = country_codelist['native_regions']

        iso2_mapping = {k: v for d in native_list for k, v in d.items()}

    return iso2_mapping


def load_gdx_file(input_file, gams_dir):
    file_name = input_file + ".gdx"
    logging.info('Loading gdx with output results')
    output_gdx = gdxpds.to_dataframes(DEF_INPUT_PATH / file_name, gams_dir=gams_dir)

    return dw.DataWrapper(input_file, output_gdx)

def load_csv_file(input_file):
    """Read the result csv files of the Julia version whose names contain input_file (see DEF_CSV_TABLES)."""
    logging.info('Loading CSVs with output results')
    output_csvs = OrderedDict()

    for filename in os.listdir(DEF_INPUT_PATH):
        if input_file not in filename or not filename.endswith('.csv'):
            continue
        for table, sheet in DEF_CSV_TABLES.items():
            # the underscore keeps other tables out, e.g. output_emissions_* (emission intensities)
            if filename.startswith(table + '_'):
                if sheet in output_csvs:
                    raise ValueError(f'Several {table} files contain "{input_file}" in {DEF_INPUT_PATH}')
                # all columns as text, as in the gdx (e.g. modes are compared with '1'), only values as numbers
                df = pd.read_csv(os.path.join(DEF_INPUT_PATH, filename), dtype=str, keep_default_na=False)
                if table == 'output_technology_costs_detailed':
                    df = _repair_swapped_year_fuel(df)
                output_csvs[sheet] = df.astype({'Value': float})

    missing = [table for table, sheet in DEF_CSV_TABLES.items() if sheet not in output_csvs]
    if missing:
        raise FileNotFoundError(f'No {", ".join(missing)} files containing "{input_file}" in {DEF_INPUT_PATH}')
    return dw.DataWrapper(input_file, output_csvs)


def _repair_swapped_year_fuel(df):
    """The Julia version writes the levelized costs of storages with Fuel and Year swapped (genesysmod_results.jl:
    lc_tot_st is indexed region, technology, year, fuel but written with dim_names Region, Technology, Fuel, Year).
    Swap them back, so that the rows read like all others."""
    swapped = ~df['Year'].str.fullmatch(r'\d{4}') & df['Fuel'].str.fullmatch(r'\d{4}')
    if swapped.any():
        logging.warning(f'{swapped.sum()} rows of output_technology_costs_detailed had Fuel and Year swapped '
                        f'(storage levelized costs of the Julia version); swapped back')
        df.loc[swapped, ['Fuel', 'Year']] = df.loc[swapped, ['Year', 'Fuel']].values
    return df

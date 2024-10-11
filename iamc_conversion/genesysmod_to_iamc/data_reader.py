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

definition_dir = Path(__file__).parent / 'definitions'

def loadmap_iso2_countries():
    with open(definition_dir/ 'region' / 'countries.yaml', 'r') as stream:
        country_codelist = yaml.load(stream, Loader=yaml.FullLoader)

        transformed_codelist = {}

        # Iterate over the list of countries and extract key-value pairs
        for country_entry in country_codelist[0]['Countries']:
            for country_name, country_data in country_entry.items():
                # Remove the redundant 'countries' key inside the nested dictionary
                country_data.pop('countries', None)
                # Add the entry to the transformed dictionary
                transformed_codelist[country_name] = country_data

#    countries = CodeList.from_directory(
#        "region", path=definition_dir/'region', file="countries.yaml"
#    )
    iso2_mapping = dict(
        [(transformed_codelist[c]['iso2'], c) for c in transformed_codelist]
       # add alternative iso2 codes used by the European Commission to the mapping
        + [(transformed_codelist[c]['iso2_alt'], c) for c in transformed_codelist
           if 'iso2_alt' in transformed_codelist[c]]
    )

#    iso2_mapping = dict(
#        [(countries[c]["iso3"], c) for c in countries]
#        + [(countries[c]["iso2"], c) for c in countries]
#        # add alternative iso2 codes used by the European Commission to the mapping
#        + [(countries[c]["iso2_alt"], c) for c in countries if "iso2_alt" in countries[c]]
#    )
#    print(iso2_mapping)
    return iso2_mapping


def load_gdx_file(input_file, gams_dir):
    file_name = input_file + ".gdx"
    logging.info('Loading gdx with output results')
    output_gdx = gdxpds.to_dataframes(DEF_INPUT_PATH / file_name, gams_dir=gams_dir)

    return dw.DataWrapper(input_file, output_gdx)

def load_csv_file(input_file):
    logging.info('Loading CSVs with output results')
    key_list = ['output_energy_balance', 'output_capacity', 'output_emissions', 'output_costs',
                'output_exogenous_costs', 'output_technology_costs_detailed', 'output_trade_capacity']

    output_csvs = OrderedDict()

    for filename in os.listdir(DEF_INPUT_PATH):
        if input_file in filename:
            for keyname in key_list:
                if filename.startswith(keyname) and not filename.startswith('output_energy_balance_annual'):
                    file_path = os.path.join(DEF_INPUT_PATH, filename)
                    df = pd.read_csv(file_path)
                    output_csvs[keyname] = df


    return dw.DataWrapper(input_file, output_csvs)

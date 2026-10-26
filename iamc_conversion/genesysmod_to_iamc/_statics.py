import os
from pathlib import Path

import yaml

# settings that change between releases or projects are read from config.yaml (one folder above this package);
# another settings file can be used by setting the environment variable GENESYSMOD_IAMC_CONFIG
CONFIG_FILE = Path(os.environ.get('GENESYSMOD_IAMC_CONFIG', Path(__file__).parents[1] / 'config.yaml'))
with open(CONFIG_FILE, encoding='utf-8') as _file:
    CONFIG = yaml.safe_load(_file)


def _folder(path):
    """A folder from the settings; relative paths are relative to the settings file."""
    path = Path(os.path.expanduser(path))
    return path if path.is_absolute() else CONFIG_FILE.parent / path


DEF_GAMS_DIR = CONFIG['gams_dir']

DEF_MAPPINGS_PATH = Path(__file__).parent / 'mappings'
DEF_INPUT_PATH = _folder(CONFIG['input_dir'])
DEF_OUTPUT_PATH = _folder(CONFIG['output_dir'])

DEF_MODEL_AND_VERSION = CONFIG['model']
DEF_RELEASE = CONFIG.get('release') or ''
DEF_YEARS = CONFIG['years']
DEF_FILE_TYPE = CONFIG['file_type']
DEF_SCENARIOS = CONFIG['scenarios']
DEF_OUTPUTS = CONFIG['outputs']
DEF_COMBINED_FILE = CONFIG['combined_file']

DEF_REGION_NAME = CONFIG['regions']['total']
DEF_EU27_NAME = CONFIG['regions']['eu27']

# scenario explorer whose upload check is used to validate the outputs (see validation.py)
DEF_RUN_VALIDATION = CONFIG['validation']['run']
DEF_WORKFLOW_REPO_URL = CONFIG['validation']['workflow_repo']
DEF_WORKFLOW_CACHE_DIR = _folder(CONFIG['validation']['cache_dir'])


DEF_PRODUCTION_COLUMNS = ["region",
                          "sector",
                          "technology",
                          "mode",
                          "fuel",
                          "subannual",
                          "type",
                          "unit",
                          "scenario",
                          "year",
                          "value"]
DEF_EMISSION_COLUMNS = ["region",
                        "sector",
                        "emission",
                        "technology",
                        "type",
                        "scenario",
                        "year",
                        "value"]
DEF_CAPACITY_COLUMNS = ["region",
                        "sector",
                        "technology",
                        "type",
                        "scenario",
                        "year",
                        "value"]
DEF_EXOGENOUS_COST_COLUMNS = ["region",
                              "technology",
                              "type",
                              "year",
                              "value"]
DEF_DETAILED_COST_COLUMNS = ["region",
                              "technology",
                             "sector",
                              "type",
                             "unit",
                              "year",
                              "value"]
DEF_TRADE_CAPACITY_COLUMNS = ["region_from",
                              "region_to",
                              "fuel",
                              "type",
                              "year",
                              "value"]

DEF_PRODUCTION_SHEET = 'output_energy_balance'
DEF_CAPACITY_SHEET = 'output_capacity'
DEF_EMISSION_SHEET = 'output_emissions'
DEF_EXOGENOUS_COST_SHEET = 'output_exogenous_costs'
DEF_DETAILED_COST_SHEET = 'output_technology_costs_detailed'
DEF_TRADE_CAPACITY_SHEET = 'output_trade'

# result tables of the Julia version (files <table>_<region>_<pathway>_<scenario>_<run>.csv, written with
# switch_processed_results = 1) and the gdx names under which the converter reads them
DEF_CSV_TABLES = {
    'output_production': DEF_PRODUCTION_SHEET,
    'output_capacity': DEF_CAPACITY_SHEET,
    'output_emission': DEF_EMISSION_SHEET,
    'output_exogenous_costs': DEF_EXOGENOUS_COST_SHEET,
    'output_technology_costs_detailed': DEF_DETAILED_COST_SHEET,
    'output_trade': DEF_TRADE_CAPACITY_SHEET,
}

# EU member states among the model regions (Malta and Cyprus are not modelled), and the regions that add up to
# the total over all model regions (DEF_REGION_NAME)
DEF_EU27 = ['AT', 'BE', 'BG', 'HR', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT',
            'LV', 'LT', 'LU', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE']
DEF_TOTAL_SUBREGIONS = [DEF_EU27_NAME, 'NO', 'NONEU_Balkan', 'TR', 'UK', 'CH']
import logging
import os
from pathlib import Path
from types import SimpleNamespace

# folder of the converter, with the notebook iamc_conversion_outputs.ipynb and run.py; relative folders in the
# settings are relative to it
DEF_BASE_PATH = Path(__file__).parents[1]
DEF_MAPPINGS_PATH = Path(__file__).parent / 'mappings'


class _Settings(SimpleNamespace):
    def __getattr__(self, name):
        raise AttributeError(f'The setting "{name}" is not set: run genesysmod_to_iamc.configure() first (the '
                             f'conversion cell of iamc_conversion_outputs.ipynb and run.py do this)')


# the settings of a conversion, set with configure()
settings = _Settings()


def configure(*, gams_dir, input_dir, file_type, scenarios, model, release, years, transmission, hourly_series,
              load_factors, combine, output_dir, combined_file, total_region, eu27_region, run_validation,
              workflow_repo, leave_out=(), exogenous_costs=False, additional_emissions=False,
              workflow_cache_dir='~/.genesysmod_to_iamc'):
    """Set the settings of the conversion (explained in iamc_conversion_outputs.ipynb and run.py).

    scenarios: the result files (without extension), whose names are also the scenario names in the outputs, or a
    dict {result file: scenario name}. Relative folders are relative to the converter folder (DEF_BASE_PATH)."""
    if file_type not in ('gdx', 'csv'):
        raise ValueError(f"file_type must be 'gdx' or 'csv', not {file_type!r}")
    vars(settings).update(
        gams_dir=gams_dir, input_path=_folder(input_dir), file_type=file_type,
        scenarios=dict(scenarios) if isinstance(scenarios, dict) else {file: file for file in scenarios},
        model=model, release=release or '', years=list(years),
        transmission=transmission, hourly_series=hourly_series, load_factors=load_factors, combine=combine,
        exogenous_costs=exogenous_costs, additional_emissions=additional_emissions,
        output_path=_folder(output_dir), combined_file=combined_file, leave_out=list(leave_out),
        total_region=total_region, eu27_region=eu27_region,
        run_validation=run_validation, workflow_repo=workflow_repo, workflow_cache_path=_folder(workflow_cache_dir))
    logging.info(f'Settings: scenarios {", ".join(settings.scenarios.values())}, release {settings.release or "-"}, '
                 f'results in {settings.input_path}, outputs in {settings.output_path}')


def _folder(path):
    """A folder from the settings; relative paths are relative to the converter folder (DEF_BASE_PATH)."""
    path = Path(os.path.expanduser(path))
    return path if path.is_absolute() else DEF_BASE_PATH / path


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

# EU member states among the model regions (Malta and Cyprus are not modelled), and the other model regions, which
# add up with EU27 to the total over all model regions (total_region in the settings)
DEF_EU27 = ['AT', 'BE', 'BG', 'HR', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR', 'HU', 'IE', 'IT',
            'LV', 'LT', 'LU', 'NL', 'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE']
DEF_NON_EU_REGIONS = ['NO', 'NONEU_Balkan', 'TR', 'UK', 'CH']
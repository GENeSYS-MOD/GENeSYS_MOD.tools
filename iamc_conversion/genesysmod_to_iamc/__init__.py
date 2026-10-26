"""Conversion of GENeSYS-MOD results to the IAMC format of the scenario explorers.

All settings are in config.yaml (in the folder above this package).

    import genesysmod_to_iamc
    genesysmod_to_iamc.run()                           # all scenarios of config.yaml, combined and validated
    genesysmod_to_iamc.generate_data('Go RES')         # one scenario only
    genesysmod_to_iamc.validate('path/to/file.xlsx')   # check an existing IAMC file
"""
from genesysmod_to_iamc._statics import *

import genesysmod_to_iamc.data_reader as dr
import genesysmod_to_iamc.data_transform as dt
import genesysmod_to_iamc.series_generator as sg

import pyam
from genesysmod_to_iamc.validation import validate, update_workflow_repo


import logging
import os
import time


logging.basicConfig(level=logging.INFO)


def run():
    """Convert all scenarios of config.yaml, combine their yearly values in one Excel file and validate it.
    Returns the path of the Excel file."""
    for scenario in DEF_SCENARIOS:
        generate_data(scenario['file'])
    return generate_combined_excel_yearly(files=[scenario['file'] + '_yearly.csv' for scenario in DEF_SCENARIOS])


def generate_data(input_file: str, file_type: str = None, generate_series_data: bool = None,
                  generate_load_factors: bool = None, generate_transmission_data: bool = None,
                  combine_outputs: bool = None):
    """Convert one result file (name without extension, in the input folder). Arguments that are left out are
    taken from config.yaml."""
    file_type = file_type or DEF_FILE_TYPE
    generate_series_data = _option(generate_series_data, 'hourly_series')
    generate_load_factors = _option(generate_load_factors, 'load_factors')
    generate_transmission_data = _option(generate_transmission_data, 'transmission')
    combine_outputs = _option(combine_outputs, 'combine')

    if file_type != "gdx" and file_type != "csv":
        raise Exception("Error: 'file_type' must be one of 'csv' or 'gdx'")

    if not os.path.exists(DEF_OUTPUT_PATH):
        os.makedirs(DEF_OUTPUT_PATH)

    data_wrapper = _generate_yearly_values(input_file, file_type, generate_transmission_data)
    output_name = data_wrapper.input_file + "_yearly.csv"

    data_wrapper.idataframe.to_csv(DEF_OUTPUT_PATH / output_name)

    if generate_series_data:
        # the hourly profiles come from Timeseries_<input_file>.xlsx, which only the series need
        dt.generate_load_demand_series(data_wrapper, input_file)
        _generate_series_data(data_wrapper)

    if generate_load_factors:
        data_wrapper_load = _generate_load_factors(input_file, file_type)
        output_name_load = data_wrapper_load.input_file + "_loadfactors.csv"
        data_wrapper_load.idataframe.to_csv(DEF_OUTPUT_PATH / output_name_load)

    if combine_outputs:
        _combine_data(input_file, generate_series_data, generate_load_factors)


def generate_combined_excel_yearly(run_validation: bool = None, files: list = None):
    """Combine *_yearly.csv outputs (the given file names, or all in the output folder) into one Excel file and
    validate it with the scenario explorer workflow (see validation.py). Returns the path of the Excel file."""
    if files is None:
        files = [filename for filename in os.listdir(DEF_OUTPUT_PATH) if filename.endswith("_yearly.csv")]
    lst = [pyam.IamDataFrame(str(DEF_OUTPUT_PATH / filename)) for filename in files]

    genesys = pyam.concat(lst)
    genesys = pyam.IamDataFrame(genesys.data[pyam.IAMC_IDX + ['year', 'value']])
    name = DEF_COMBINED_FILE.format(date=time.strftime("%d-%m-%Y"), release=DEF_RELEASE)
    excel_file = DEF_OUTPUT_PATH / 'combined_excel' / name
    excel_file.parent.mkdir(parents=True, exist_ok=True)
    genesys.to_excel(excel_file)

    if DEF_RUN_VALIDATION if run_validation is None else run_validation:
        validate(excel_file)
    return excel_file


def _option(value, key):
    """An argument of the functions above, or the setting of the same name in config.yaml if it is left out."""
    return DEF_OUTPUTS.get(key, False) if value is None else value


def _scenario_name(input_file):
    """Scenario name in the outputs: the name config.yaml gives this file (else the file name) and the release."""
    name = next((scenario['name'] for scenario in DEF_SCENARIOS if scenario['file'] == input_file), input_file)
    return f'{name} {DEF_RELEASE}'.strip()


def _load_results(input_file, file_type):
    if file_type == "gdx":
        data_wrapper = dr.load_gdx_file(input_file, DEF_GAMS_DIR)
    elif file_type == "csv":
        data_wrapper = dr.load_csv_file(input_file)
    data_wrapper.scenario = _scenario_name(input_file)
    return data_wrapper


def _generate_yearly_values(input_file, file_type: str = "gdx", generate_transmission_data: bool = False):
    data_wrapper = _load_results(input_file, file_type)

    if generate_transmission_data == True:
        dt.generate_transmission_capacity_values(data_wrapper)
    dt.generate_primary_energy_values(data_wrapper)
    dt.generate_final_energy_values(data_wrapper)
    dt.generate_capacity_values(data_wrapper)
    dt.generate_transport_capacity_values(data_wrapper)
    dt.generate_storage_capacity_values(data_wrapper)
    dt.generate_emissions_values(data_wrapper)
    if _option(None, 'additional_emissions'):
        dt.generate_additional_emissions_values(data_wrapper)
    dt.generate_secondary_energy(data_wrapper)
    if _option(None, 'exogenous_costs'):
        dt.generate_exogenous_costs(data_wrapper)
    dt.generate_detailed_costs(data_wrapper)
    dt.generate_co2_prices(data_wrapper)

    data_wrapper.generate_idataframe(True)

    return data_wrapper


def _generate_load_factors(input_file, file_type: str = "gdx"):
    data_wrapper_series = _load_results(input_file, file_type)

    dt.generate_load_factors(data_wrapper_series, input_file)
    data_wrapper_series.generate_idataframe_renewable_series()
    return data_wrapper_series


def _generate_series_data(data_wrapper):
    logging.info('Executing: generate_final_energy_series')
    if data_wrapper.idataframe is None:
        raise Exception('IamDataFrame of data_wrapper is empty')

    sg.generate_final_energy_series(data_wrapper, 'HeatingLow', 'Final Energy|Residential and Commercial|Electricity')
    sg.generate_final_energy_series(data_wrapper, 'HeatingHigh', 'Final Energy|Industry|Electricity')
    sg.generate_final_energy_series(data_wrapper, 'Transport', 'Final Energy|Transportation|Electricity')
    sg.generate_final_energy_series(data_wrapper, 'Power', 'Final Energy|Electricity')
    sg.generate_final_energy_series(data_wrapper, 'HeatingLow', 'Final Energy|Residential and Commercial')
    sg.generate_final_energy_series(data_wrapper, 'HeatingHigh', 'Final Energy|Industry')

    sg.output_series(data_wrapper)


def _combine_data(input_file, generate_series_data: bool = False, generate_load_factors: bool = False):
    file_combined = input_file + "_combined.csv"
    file_yearly = input_file + '_yearly.csv'
    file_load = input_file + '_loadfactors.csv'
    file_series = input_file + '_series.csv'

    idataframe_base = pyam.IamDataFrame(str(DEF_OUTPUT_PATH / file_yearly))

    if generate_load_factors:
        idataframe_load = pyam.IamDataFrame(str(DEF_OUTPUT_PATH / file_load))
        idataframe_base_with_load = idataframe_base.append(idataframe_load)
    else:
        idataframe_base_with_load = idataframe_base

    if generate_series_data:
        idataframe_series = pyam.IamDataFrame(str(DEF_OUTPUT_PATH / file_series))
        idataframe_all = idataframe_base_with_load.append(idataframe_series)
    else:
        idataframe_all = idataframe_base_with_load

    idataframe_all.to_csv(DEF_OUTPUT_PATH / file_combined)

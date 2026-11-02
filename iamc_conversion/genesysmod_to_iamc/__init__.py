"""Conversion of GENeSYS-MOD results to the IAMC format of the scenario explorers.

The settings are passed with configure(). The notebook iamc_conversion_outputs.ipynb and run.py (for a run from a
terminal) define them as variables and pass them.

    import genesysmod_to_iamc
    genesysmod_to_iamc.configure(...)                  # the settings (explained in the notebook and run.py)
    genesysmod_to_iamc.run()                           # all scenarios of the settings, combined and validated
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
    """Convert all scenarios of the settings, combine their yearly values in one Excel file and validate it.
    Returns the path of the Excel file."""
    for input_file in settings.scenarios:
        generate_data(input_file)
    return generate_combined_excel_yearly(files=[input_file + '_yearly.csv' for input_file in settings.scenarios])


def generate_data(input_file: str, file_type: str = None, generate_series_data: bool = None,
                  generate_load_factors: bool = None, generate_transmission_data: bool = None,
                  combine_outputs: bool = None):
    """Convert one result file (name without extension, in the input folder). Arguments that are left out are
    taken from the settings."""
    file_type = file_type or settings.file_type
    generate_series_data = _option(generate_series_data, 'hourly_series')
    generate_load_factors = _option(generate_load_factors, 'load_factors')
    generate_transmission_data = _option(generate_transmission_data, 'transmission')
    combine_outputs = _option(combine_outputs, 'combine')

    if file_type != "gdx" and file_type != "csv":
        raise Exception("Error: 'file_type' must be one of 'csv' or 'gdx'")

    if not os.path.exists(settings.output_path):
        os.makedirs(settings.output_path)

    data_wrapper = _generate_yearly_values(input_file, file_type, generate_transmission_data)
    output_name = data_wrapper.input_file + "_yearly.csv"

    data_wrapper.idataframe.to_csv(settings.output_path / output_name)

    if generate_series_data:
        # the hourly profiles come from Timeseries_<input_file>.xlsx, which only the series need
        dt.generate_load_demand_series(data_wrapper, input_file)
        _generate_series_data(data_wrapper)

    if generate_load_factors:
        data_wrapper_load = _generate_load_factors(input_file, file_type)
        output_name_load = data_wrapper_load.input_file + "_loadfactors.csv"
        data_wrapper_load.idataframe.to_csv(settings.output_path / output_name_load)

    if combine_outputs:
        _combine_data(input_file, generate_series_data, generate_load_factors)


def generate_combined_excel_yearly(run_validation: bool = None, files: list = None):
    """Combine *_yearly.csv outputs (the given file names, or all in the output folder) into one Excel file and
    validate it with the scenario explorer workflow (see validation.py). Returns the path of the Excel file."""
    if files is None:
        files = [filename for filename in os.listdir(settings.output_path) if filename.endswith("_yearly.csv")]
    lst =[pyam.IamDataFrame(str(settings.output_path / filename)) for filename in files]

    genesys = pyam.concat(lst)
    genesys = pyam.IamDataFrame(genesys.data[pyam.IAMC_IDX + ['year', 'value']])
    if settings.leave_out:
        # variables the explorer does not know yet (leave_out in the settings)
        genesys = genesys.filter(variable=settings.leave_out, keep=False)
    name = settings.combined_file.format(date=time.strftime("%d-%m-%Y"), release=settings.release)
    excel_file = settings.output_path / 'combined_excel' / name
    excel_file.parent.mkdir(parents=True, exist_ok=True)
    genesys.to_excel(excel_file)

    if settings.run_validation if run_validation is None else run_validation:
        validate(excel_file)
    return excel_file


def _option(value, key):
    """An argument of the functions above, or the setting of the same name if it is left out."""
    return getattr(settings, key) if value is None else value


def _scenario_name(input_file):
    """Scenario name in the outputs: the name the settings give this file (else the file name) and the release."""
    name = settings.scenarios.get(input_file, input_file)
    return f'{name} {settings.release}'.strip()


def _load_results(input_file, file_type):
    if file_type == "gdx":
        data_wrapper = dr.load_gdx_file(input_file, settings.gams_dir)
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
    dt.generate_useful_energy_heat(data_wrapper)
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

    idataframe_base = pyam.IamDataFrame(str(settings.output_path / file_yearly))

    if generate_load_factors:
        idataframe_load = pyam.IamDataFrame(str(settings.output_path / file_load))
        idataframe_base_with_load = idataframe_base.append(idataframe_load)
    else:
        idataframe_base_with_load = idataframe_base

    if generate_series_data:
        idataframe_series = pyam.IamDataFrame(str(settings.output_path / file_series))
        idataframe_all = idataframe_base_with_load.append(idataframe_series)
    else:
        idataframe_all = idataframe_base_with_load

    idataframe_all.to_csv(settings.output_path / file_combined)

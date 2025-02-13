from genesysmod_to_iamc._statics import *
from enum import Enum

import genesysmod_to_iamc.data_reader as dr
import genesysmod_to_iamc.data_transform as dt
import genesysmod_to_iamc.series_generator as sg

import nomenclature
import pyam
from genesysmod_to_iamc.workflow import main as workflow


import logging
import os
import time

definition_dir = Path(__file__).parent / 'definitions'
definition = nomenclature.DataStructureDefinition(definition_dir)


logging.basicConfig(level=logging.INFO)

class Pathways(Enum):
#    TF = "TechnoFriendly"
#    DT = "DirectedTransition"
#    GD = "GradualDevelopment"
#    SC = "SocietalCommitment"
    GR = "GoRES"
    RE = "REPowerEU++"
    NE = "NECPEssentials"
    TR = "Trinity"


def generate_data(input_file: str, file_type: str = "gdx", generate_series_data: bool = False, generate_load_factors: bool = False, generate_transmission_data: bool = False,
                  combine_outputs: bool = False):

    if file_type != "gdx" and file_type != "csv":
        raise Exception("Error: 'file_type' must be one of 'csv' or 'gdx'")

    if not os.path.exists(DEF_OUTPUT_PATH):
        os.makedirs(DEF_OUTPUT_PATH)

    data_wrapper = _generate_yearly_values(input_file, file_type, generate_transmission_data)
    output_name = data_wrapper.input_file + "_yearly.csv"

    data_wrapper.idataframe.to_csv(DEF_OUTPUT_PATH / output_name)

    if generate_series_data:
        _generate_series_data(data_wrapper)

    if generate_load_factors:
        data_wrapper_load = _generate_load_factors(input_file, file_type)
        output_name_load = data_wrapper_load.input_file + "_loadfactors.csv"
        data_wrapper_load.idataframe.to_csv(DEF_OUTPUT_PATH / output_name_load)

    if combine_outputs:
        _combine_data(input_file, generate_series_data, generate_load_factors)


def generate_combined_excel_yearly():
    lst = []

    for filename in os.listdir(DEF_OUTPUT_PATH):
        if filename.endswith("_yearly.csv"):
            df = pyam.IamDataFrame(str(DEF_OUTPUT_PATH / filename))
            lst.append(df)

    genesys = pyam.concat(lst)
    genesys = pyam.IamDataFrame(genesys.data[pyam.IAMC_IDX + ['year', 'value']])
    timestr = time.strftime("%d-%m-%Y")
    genesys.to_excel(DEF_OUTPUT_PATH / 'combined_excel' / f'GENeSYS-MOD-pathways_{timestr}.xlsx')


def _generate_yearly_values(input_file, file_type: str = "gdx", generate_transmission_data: bool = False):
    if file_type == "gdx":
        data_wrapper = dr.load_gdx_file(input_file, DEF_GAMS_DIR)
    elif file_type == "csv":
        data_wrapper = dr.load_csv_file(input_file)

    if generate_transmission_data == True:
        dt.generate_transmission_capacity_values(data_wrapper)
    dt.generate_load_demand_series(data_wrapper, input_file)
    dt.generate_primary_energy_values(data_wrapper)
    dt.generate_final_energy_values(data_wrapper)
    dt.generate_capacity_values(data_wrapper)
    dt.generate_transport_capacity_values(data_wrapper)
    dt.generate_storage_capacity_values(data_wrapper)
    dt.generate_emissions_values(data_wrapper)
#    dt.generate_additional_emissions_values(data_wrapper)
    dt.generate_secondary_energy(data_wrapper)
    dt.generate_exogenous_costs(data_wrapper)
    dt.generate_detailed_costs(data_wrapper)
    dt.generate_co2_prices(data_wrapper)

    data_wrapper.generate_idataframe(True)

    df = workflow(pyam.IamDataFrame(data_wrapper.idataframe))
    #df = pyam.IamDataFrame(data_wrapper.idataframe)

    definition.validate(df)
    return data_wrapper


def _generate_load_factors(input_file, file_type: str = "gdx"):
    if file_type == "gdx":
        data_wrapper_series = dr.load_gdx_file(input_file, DEF_GAMS_DIR)
    elif file_type == "csv":
        data_wrapper_series = dr.load_csv_file(input_file)

    dt.generate_load_factors(data_wrapper_series, input_file)
    data_wrapper_series.generate_idataframe_renewable_series()
    #nomenclature.DataStructureDefinition.validate(df=data_wrapper_series.idataframe)
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

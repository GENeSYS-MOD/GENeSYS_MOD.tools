import genesysmod_to_iamc.data_reader as dr
import pandas as pd
import pyam
import logging

from genesysmod_to_iamc._statics import *


def generate_idataframe(data_wrapper, filter_only_yearly_values=False):
    """All converted values of one scenario as an IamDataFrame, with the sums of mappings/aggregations.yaml, and the
    model regions renamed to the region names of the scenario explorer."""
    logging.info('Executing: generate_idataframe')
    idataframe = pyam.IamDataFrame(pd.concat(list(data_wrapper.transformed_data.values())))

    for step in dr.load_aggregations():
        if 'sum' in step:
            idataframe.aggregate(variable=step['sum'], components=step.get('of'), append=True, recursive=False)
        elif 'regions' in step:
            # the total over all model regions, named 'World' by pyam and renamed below
            idataframe.aggregate_region(variable=step['regions'], subregions=DEF_TOTAL_SUBREGIONS, append=True)
        elif 'drop' in step:
            idataframe = pyam.IamDataFrame(idataframe.data[idataframe.data['variable'] != step['drop']])
        else:
            raise ValueError(f'Unknown step in aggregations.yaml: {step}')

    iso2_mapping = dr.loadmap_iso2_countries()

    for r in iso2_mapping:
        idataframe.rename(region={r: iso2_mapping[r]}, inplace=True)

    idataframe = pyam.IamDataFrame(idataframe.data.replace({'region': 'World'}, DEF_REGION_NAME))
    idataframe.rename(region={'NONEU_Balkan': 'Non-EU-Balkans'}, inplace=True)

    if filter_only_yearly_values:
        idataframe.filter(subannual='Year', inplace=True)

    return idataframe


def changeFinalEnergyToGWh(final_energy, variable):
    # switched off: converts one final energy variable from EJ to GWh
    final_energy.loc[(final_energy['variable'] == variable), 'value'] \
        = final_energy.loc[
              (final_energy['variable'] == variable), 'value'] * 1000 * 1000 / 3.6
    final_energy.loc[(final_energy['variable'] == variable), 'unit'] = 'GWh'

    return final_energy


def generate_idataframe_renewable_series(data_wrapper):
    logging.info('Executing: generate_idataframe')
    return pyam.IamDataFrame(pd.concat(list(data_wrapper.transformed_data.values())))

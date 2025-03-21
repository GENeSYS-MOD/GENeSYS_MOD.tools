import genesysmod_to_iamc.data_reader as dr
import genesysmod_to_iamc.data_wrapper as dw
import pandas as pd
import logging
import itertools

from genesysmod_to_iamc._statics import *


def _set_scenarios(frame: pd.DataFrame):
    for entry in DEF_MAP_SCENARIOS:
        frame = frame.replace({'scenario': entry}, DEF_MAP_SCENARIOS[entry])

    return frame


def _set_timeslices(frame: pd.DataFrame, data_wrapper: dw.DataWrapper):
    timeslices = data_wrapper.map_timeslices
    for entry in timeslices:
        frame = frame.replace({'subannual': entry}, timeslices[entry])

    return frame


def _transform_columns(frame: pd.DataFrame, column='fuel'):
    # group by relevant values and take the sum
    _frame = frame.groupby(
        ['model', 'scenario', 'region', column, 'unit', 'subannual', 'year'])['value'].sum().reset_index()

    # agregate to yearly values
    _frame2 = frame.groupby(
        ['model', 'scenario', 'region', column, 'unit', 'year'])['value'].sum().reset_index()
    _frame2['subannual'] = 'Year'
    _frame2 = _frame2.groupby(
        ['model', 'scenario', 'region', column, 'unit', 'subannual', 'year'])['value'].sum().reset_index()

    # reassign column names
    _frame2.columns = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']
    _frame.columns = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']

    _frame = pd.concat([_frame, _frame2])

    return _frame


def generate_primary_energy_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_primary_energy_values')
    map_primary_fuels = dr.loadmap_primary_energy_fuels()
    map_nonbio_renewables = dr.loadmap_non_bio_renewables()
    map_electricity = dr.loadmap_from_csv('primary_energy_electricity')

    prod_values = data_wrapper.production_values.copy()
    use_values = data_wrapper.usage_values.copy()


    traditional_energy = use_values[use_values['fuel'].isin(map_primary_fuels.keys())].copy()
    traditional_energy = traditional_energy[traditional_energy['sector'] != 'Storage']
    traditional_energy['value'] = abs(traditional_energy['value'])
    for entry in map_primary_fuels:
        traditional_energy = traditional_energy.replace({'fuel': entry}, map_primary_fuels[entry])


    non_res_biomass = prod_values[prod_values['technology'].isin(map_nonbio_renewables.keys())].copy()
    for entry in map_nonbio_renewables:
        non_res_biomass = non_res_biomass.replace({'technology': entry},
                                                  'Primary Energy|' + map_nonbio_renewables[
                                                      entry])

    non_res_biomass['fuel'] = non_res_biomass['technology']

    nuclear = use_values[use_values['technology'] == 'P_Nuclear'].copy()

    nuclear['fuel'] = 'Primary Energy|Nuclear'

    electricity = use_values[use_values['technology'].isin(map_electricity.keys())].copy()
    for entry in map_electricity:
        electricity = electricity.replace({'technology': entry}, map_electricity[entry])
    electricity = electricity[electricity['mode'] == '1']

    electricity['fuel'] = electricity['technology']

    frames = [traditional_energy, non_res_biomass, nuclear, electricity]

    primary_energy = pd.concat(frames)


    primary_energy['model'] = DEF_MODEL_AND_VERSION
    primary_energy['unit'] = 'EJ/yr'
    primary_energy['value'] = primary_energy['value'] / 1000
    primary_energy['value'] = abs(primary_energy['value'])

    primary_energy = _set_scenarios(primary_energy)
    primary_energy = _set_timeslices(primary_energy, data_wrapper)
    primary_energy = _transform_columns(primary_energy, 'fuel')

    # add aggregated region EU27
    energy_eu27 = primary_energy[primary_energy['Region'].isin(DEF_EU27)]
    energy_eu27 = energy_eu27.groupby(['Model', 'Scenario', 'Variable', 'Unit', 'Subannual', 'Year'])['Value'].sum().reset_index()
    energy_eu27['Region'] = 'EU27 (excl. Malta & Cyprus)'

    primary_energy = pd.concat([primary_energy, energy_eu27])

    ### adding variables with value 0
    # import and filter for primary energy variables
    df_zero_var = dr.loaddf_from_csv('zero_var')
    df_final_zero = df_zero_var[df_zero_var['Type'] == 'Primary Energy']['Variable'].reset_index()

    # create a df of combinations of all years,regions and variables
    combinations = list(
        itertools.product(primary_energy['Region'].unique(), primary_energy['Year'].unique(), df_final_zero['Variable']))
    df_final_zero = pd.DataFrame(combinations, columns=['Region', 'Year', 'Variable'])

    # drop variables starting with #
    df_final_zero = df_final_zero[~df_final_zero['Variable'].str.startswith('#')]

    # safe general values of other columns and assign them to the new variables, setting value 0
    model, scenario, unit = primary_energy.iloc[1, [0, 1, 4]]
    df_final_zero[['Model', 'Scenario', 'Unit', 'Subannual', 'Value']] = model, scenario, unit, 'Year', 0

    primary_energy = pd.concat([primary_energy, df_final_zero])

    data_wrapper.transformed_data['primary_energy'] = primary_energy
    return primary_energy


def generate_final_energy_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_final_energy_values')
    use_values = data_wrapper.usage_values.copy()
    map_final_sector = dr.loadmap_from_csv('final_energy_sector')
    map_final_fuel = dr.loadmap_from_csv('final_energy_fuel')


    use_values = use_values[use_values['technology'].isin(map_final_sector.keys())]
    use_values = use_values[use_values['fuel'].isin(map_final_fuel.keys())]


    for entry in map_final_sector:
        use_values = use_values.replace({'technology': entry}, map_final_sector[entry])
    for entry in map_final_fuel:
        use_values = use_values.replace({'fuel': entry}, map_final_fuel[entry])

    use_values['techfuel'] = use_values['technology'] + '|' + use_values['fuel']
    use_values['fuel'] = use_values['techfuel']



    use_electricity_values = data_wrapper.usage_values.copy()
    use_electricity_values = use_electricity_values[use_electricity_values['fuel'] == 'Power']
    use_electricity_values = use_electricity_values[use_electricity_values['technology'] == 'Demand']
    use_electricity_values['fuel'] = 'Final Energy|ElectricityDummy'

    use_heat_values = _extract_final_energy_heat(data_wrapper)


    final_energy = pd.concat([use_values, use_electricity_values, use_heat_values])


    final_energy['model'] = DEF_MODEL_AND_VERSION
    final_energy['unit'] = 'EJ/yr'
    final_energy['value'] = final_energy['value'] / 1000
    final_energy['value'] = abs(final_energy['value'])

    final_energy = _set_scenarios(final_energy)
    final_energy = _set_timeslices(final_energy, data_wrapper)

    final_energy = _transform_columns(final_energy, 'fuel')

    # add aggregated region EU27
    energy_eu27 = final_energy[final_energy['Region'].isin(DEF_EU27)]
    energy_eu27 = energy_eu27.groupby(['Model', 'Scenario', 'Variable', 'Unit', 'Subannual', 'Year'])['Value'].sum().reset_index()
    energy_eu27['Region'] = 'EU27 (excl. Malta & Cyprus)'

    final_energy = pd.concat([final_energy, energy_eu27])

    ### adding variables with value 0
    #import and filter for final energy variables
    df_zero_var = dr.loaddf_from_csv('zero_var')
    df_final_zero = df_zero_var[df_zero_var['Type'] == 'Final Energy']['Variable'].reset_index()

    #create a df of combinations of all years,regions and variables
    combinations = list(itertools.product(final_energy['Region'].unique(), final_energy['Year'].unique(), df_final_zero['Variable']))
    df_final_zero = pd.DataFrame(combinations, columns=['Region', 'Year', 'Variable'])

    #drop variables starting with #
    df_final_zero = df_final_zero[~df_final_zero['Variable'].str.startswith('#')]

    #safe general values of other columns and assign them to the new variables, setting value 0
    model, scenario, unit = final_energy.iloc[1,[0,1,4]]
    df_final_zero[['Model', 'Scenario', 'Unit', 'Subannual', 'Value']] = model, scenario, unit, 'Year', 0

    final_energy = pd.concat([final_energy, df_final_zero])

    # removing entries that wrongfully get written, even without existing mapping
    final_energy.drop(final_energy[final_energy['Variable'] == 'Final Energy|Industry|Area_Rooftop_Commercial'].index,inplace=True)

    data_wrapper.transformed_data['final_energy'] = final_energy

    return final_energy


def _extract_final_energy_heat(data_wrapper):
    raw = data_wrapper.usage_values.copy()

    hlr = raw[raw['fuel'] == 'Heat_Low_Residential'].copy()
    hlr = hlr[hlr['technology'] == 'Demand']

    hli = raw[raw['fuel'] == 'Heat_Low_Industrial'].copy()
    hli = hli[hli['technology'] == 'Demand']

    hmi = raw[raw['fuel'] == 'Heat_Medium_Industrial'].copy()
    hmi = hmi[hmi['technology'] == 'Demand']

    hhi = raw[raw['fuel'] == 'Heat_High_Industrial'].copy()
    hhi = hhi[hhi['technology'] == 'Demand']

    heat = pd.concat([hlr, hli, hmi, hhi])

    heat['fuel'] = 'Final Energy|Heat'

    return heat


def generate_capacity_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_capacity_values')
    map_capacity_technology = dr.loadmap_from_csv('capacity_technologies')

    capacity_values = data_wrapper.capacity_values.copy()
    capacity_values = capacity_values[capacity_values['type'] == 'TotalCapacity']
    capacity_values = capacity_values[capacity_values['technology'].isin(map_capacity_technology.keys())]

    for entry in map_capacity_technology:
        capacity_values = capacity_values.replace({'technology': entry}, 'Capacity|' + map_capacity_technology[entry])

    capacity_values['model'] = DEF_MODEL_AND_VERSION
    capacity_values['unit'] = 'GW'
    capacity_values['value'] = abs(capacity_values['value'])
    capacity_values['subannual'] = 'Year'

    capacity_values = _set_scenarios(capacity_values)

    capacity_values = capacity_values.groupby(
        ['model', 'scenario', 'region', 'technology', 'unit', 'subannual', 'year'])['value'].sum().reset_index()
    capacity_values.columns = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']

    # add aggregated region EU27
    capacity_eu27 = capacity_values[capacity_values['Region'].isin(DEF_EU27)]
    capacity_eu27 = capacity_eu27.groupby(['Model', 'Scenario', 'Variable', 'Unit', 'Subannual', 'Year'])['Value'].sum().reset_index()
    capacity_eu27['Region'] = 'EU27 (excl. Malta & Cyprus)'

    capacity_values = pd.concat([capacity_values, capacity_eu27])

    data_wrapper.transformed_data['capacity'] = capacity_values

    return capacity_values


def generate_transmission_capacity_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_transmission_capacity_values')

    trade_values = data_wrapper.trade_capacity_values.copy()
    trade_values = trade_values[trade_values['type'] == 'Power Transmissions Capacity']
    trade_values['model'] = DEF_MODEL_AND_VERSION
    trade_values['unit'] = 'MW'
    trade_values['value'] = abs(trade_values['value'])*1000
    trade_values['subannual'] = 'Year'
    trade_values['variable'] = 'Network|Electricity|Maximum Flow'

    trade_values['scenario'] = data_wrapper.capacity_values['scenario'][0]
    trade_values = _set_scenarios(trade_values)

    trade_values = trade_values.replace({'region_to': 'UK'}, 'GB')
    trade_values = trade_values.replace({'region_from': 'UK'}, 'GB')

    iso2_mapping = dr.loadmap_iso2_countries()

    for r in iso2_mapping:
        trade_values = trade_values.replace({'region_from': r}, iso2_mapping[r])
        trade_values = trade_values.replace({'region_to': r}, iso2_mapping[r])

    trade_values = trade_values.replace({'region_to': 'NONEU_Balkan'}, 'Non-EU-Balkans')
    trade_values = trade_values.replace({'region_from': 'NONEU_Balkan'}, 'Non-EU-Balkans')

    trade_values['region'] = trade_values['region_from'] + ">" + trade_values['region_to']

    trade_values = trade_values.groupby(
        ['model', 'scenario', 'region', 'variable', 'unit', 'subannual', 'year'])['value'].sum().reset_index()
    trade_values.columns = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']


    data_wrapper.transformed_data['transmission'] = trade_values

    return trade_values


def generate_storage_capacity_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_storage_capacity_values')
    map_capacity_storage = dr.loadmap_from_csv('storages')
    map_e2p_ratios = dr.loadmap_from_csv('storages_e2p_ratios')

    capacity_values = data_wrapper.capacity_values.copy()
    capacity_values = capacity_values[capacity_values['type'] == 'TotalCapacity']

    combines = []

    for entry in map_capacity_storage:

        capacity_value = capacity_values[capacity_values['technology'] == entry].copy()
        capacity_value = capacity_value.replace({'technology': entry}, 'Maximum Storage|Electricity|' + map_capacity_storage[entry])

        if map_capacity_storage[entry] == 'Maximum Storage|Electricity|Hydro|Pumped Storage':
            capacity_value['unit'] = 'GWh'
            capacity_value['value'] = abs(capacity_value['value']) * map_e2p_ratios[map_capacity_storage[entry]]
        else:
            capacity_value['unit'] = 'GWh'
            capacity_value['value'] = abs(capacity_value['value']) * map_e2p_ratios[map_capacity_storage[entry]]

        capacity_value['model'] = DEF_MODEL_AND_VERSION
        capacity_value['subannual'] = 'Year'

        combines.append(capacity_value)

    capacity_values = pd.concat(combines)

    capacity_values = _set_scenarios(capacity_values)

    capacity_values = capacity_values.groupby(
        ['model', 'scenario', 'region', 'technology', 'unit', 'subannual', 'year'])['value'].sum().reset_index()
    capacity_values.columns = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']

    # add aggregated region EU27
    capacity_eu27 = capacity_values[capacity_values['Region'].isin(DEF_EU27)]
    capacity_eu27 = capacity_eu27.groupby(['Model', 'Scenario', 'Variable', 'Unit', 'Subannual', 'Year'])['Value'].sum().reset_index()
    capacity_eu27['Region'] = 'EU27 (excl. Malta & Cyprus)'

    capacity_values = pd.concat([capacity_values, capacity_eu27])

    data_wrapper.transformed_data['capacity_storage'] = capacity_values

    return capacity_values


def generate_transport_capacity_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_transport_capacity_values')
    map_capacity_technology = dr.loadmap_from_csv('capacity_transport_technologies')
    map_units_technology = dr.loadmap_from_csv('units_transport_technologies')

    capacity_values = data_wrapper.capacity_values.copy()
    capacity_values = capacity_values[capacity_values['type'] == 'TotalCapacity']
    capacity_values = capacity_values[capacity_values['technology'].isin(map_capacity_technology.keys())]

    capacity_values['unit'] = 'Gvkm'
    for entry in map_capacity_technology:
        capacity_values = capacity_values.replace({'technology': entry}, 'Energy Service|' + map_capacity_technology[entry])

    capacity_values['model'] = DEF_MODEL_AND_VERSION
    for entry in map_units_technology:
        capacity_values.loc[(capacity_values['technology'] == 'Energy Service|' + entry), 'unit'] = map_units_technology[entry]

    capacity_values['value'] = abs(capacity_values['value'])
    capacity_values['subannual'] = 'Year'

    capacity_values = _set_scenarios(capacity_values)

    capacity_values = capacity_values.groupby(
        ['model', 'scenario', 'region', 'technology', 'unit', 'subannual', 'year'])['value'].sum().reset_index()
    capacity_values.columns = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']

    # add aggregated region EU27
    capacity_eu27 = capacity_values[capacity_values['Region'].isin(DEF_EU27)]
    capacity_eu27 = capacity_eu27.groupby(['Model', 'Scenario', 'Variable', 'Unit', 'Subannual', 'Year'])['Value'].sum().reset_index()
    capacity_eu27['Region'] = 'EU27 (excl. Malta & Cyprus)'

    capacity_values = pd.concat([capacity_values, capacity_eu27])

    data_wrapper.transformed_data['capacity_transport'] = capacity_values

    return capacity_values


def generate_emissions_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_emissions_values')
    map_emissions_technologies = dr.loadmap_from_csv('emissions_technologies')

    emission_values = data_wrapper.emission_values.copy()
    emission_values = emission_values[emission_values['technology'].isin(map_emissions_technologies.keys())]

    for entry in map_emissions_technologies:
        emission_values = emission_values.replace({'technology': entry}, map_emissions_technologies[entry])

    emission_values['model'] = DEF_MODEL_AND_VERSION
    emission_values['unit'] = 'Mt CO2/yr'
    emission_values['subannual'] = 'Year'

    emission_values = _set_scenarios(emission_values)

    emission_values = emission_values.groupby(
        ['model', 'scenario', 'region', 'technology', 'unit', 'subannual', 'year'])['value'].sum().reset_index()
    emission_values.columns = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']

    #add aggregated region EU27
    emissions_eu27 = emission_values[emission_values['Region'].isin(DEF_EU27)]
    emissions_eu27 = emissions_eu27.groupby(['Model', 'Scenario', 'Variable', 'Unit', 'Subannual', 'Year'])['Value'].sum().reset_index()
    emissions_eu27['Region'] = 'EU27 (excl. Malta & Cyprus)'

    emission_values = pd.concat([emission_values, emissions_eu27])

    data_wrapper.transformed_data['emissions'] = emission_values

    return emission_values


def generate_additional_emissions_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_additional_emissions_values')
    map_capacity_technology = dr.loadmap_from_csv('capacity_technologies')

    emission_values = data_wrapper.emission_values.copy()
    emission_values = emission_values[emission_values['technology'].isin(map_capacity_technology.keys())]

    for entry in map_capacity_technology:
        emission_values = emission_values.replace({'technology': entry}, 'Emissions|CO2|' + map_capacity_technology[entry])

    emission_values['model'] = DEF_MODEL_AND_VERSION
    emission_values['unit'] = 'Mt CO2/yr'
    emission_values['subannual'] = 'Year'

    emission_values = _set_scenarios(emission_values)

    emission_values = emission_values.groupby(
        ['model', 'scenario', 'region', 'technology', 'unit', 'subannual', 'year'])['value'].sum().reset_index()
    emission_values.columns = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']

    # add aggregated region EU27
    emissions_eu27 = emission_values[emission_values['Region'].isin(DEF_EU27)]
    emissions_eu27 = emissions_eu27.groupby(['Model', 'Scenario', 'Variable', 'Unit', 'Subannual', 'Year'])['Value'].sum().reset_index()
    emissions_eu27['Region'] = 'EU27 (excl. Malta & Cyprus)'

    emission_values = pd.concat([emission_values, emissions_eu27])

    data_wrapper.transformed_data['emissions2'] = emission_values

    return emission_values


def generate_secondary_energy(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_secondary_energy')
    map_secondary_energy_power = dr.loadmap_from_csv('secondary_energy_power')
    prod_values = data_wrapper.production_values.copy()

    power_values = prod_values[prod_values['technology'].isin(map_secondary_energy_power.keys())].copy()
    power_values = power_values[power_values['fuel'] == 'Power']
    for entry in map_secondary_energy_power:
        power_values = power_values.replace({'technology': entry},
                                            'Secondary Energy|' + map_secondary_energy_power[entry])
    power_values['unit'] = 'EJ/yr'

    map_secondary_energy_heat = dr.loadmap_from_csv('secondary_energy_heat')
    heat_values = prod_values[prod_values['technology'].isin(map_secondary_energy_heat.keys())].copy()
    heat_values = heat_values[(heat_values['fuel'] == 'Heat_Low_Residential') |
                              (heat_values['fuel'] == 'Heat_Low_Industrial') |
                              (heat_values['fuel'] == 'Heat_Medium_Industrial') |
                              (heat_values['fuel'] == 'Heat_High_Industrial')]
    for entry in map_secondary_energy_heat:
        heat_values = heat_values.replace({'technology': entry},
                                          'Secondary Energy|' + map_secondary_energy_heat[entry])
    heat_values['unit'] = 'EJ/yr'

    # map_secondary_energy_transport = dr.loadmap_from_csv('secondary_energy_transport')
    # transport_values = prod_values[prod_values['technology'].isin(map_secondary_energy_transport.keys())].copy()
    # transport_values = transport_values[(transport_values['fuel'] == 'Mobility_Freight') |
    #                                     (transport_values['fuel'] == 'Mobility_Passenger')]
    # map_units_technology = dr.loadmap_from_csv('units_transport_technologies')
    # transport_values['unit'] = 'EJ/yr'
    # for entry in map_units_technology:
    #     transport_values = transport_values.replace({'unit': entry}, map_units_technology[entry] + "/yr")
    # for entry in map_secondary_energy_transport:
    #     transport_values = transport_values.replace({'technology': entry},
    #                                                 'Secondary Energy|' + map_secondary_energy_transport[entry])

    map_secondary_energy_other = dr.loadmap_from_csv('secondary_energy_other')
    other_values = prod_values[prod_values['technology'].isin(map_secondary_energy_other.keys())].copy()
    for entry in map_secondary_energy_other:
        other_values = other_values.replace({'technology': entry},
                                            'Secondary Energy|' + map_secondary_energy_other[entry])
    other_values['unit'] = 'EJ/yr'

    secondary_energy = pd.concat([power_values, heat_values, other_values])

    secondary_energy['model'] = DEF_MODEL_AND_VERSION
    secondary_energy['value'] = secondary_energy['value'] / 1000
    secondary_energy['value'] = abs(secondary_energy['value'])

    secondary_energy = _set_scenarios(secondary_energy)
    secondary_energy = _set_timeslices(secondary_energy, data_wrapper)

    secondary_energy = _transform_columns(secondary_energy, 'technology')

    # add aggregated region EU27
    energy_eu27 = secondary_energy[secondary_energy['Region'].isin(DEF_EU27)]
    energy_eu27 = energy_eu27.groupby(['Model', 'Scenario', 'Variable', 'Unit', 'Subannual', 'Year'])['Value'].sum().reset_index()
    energy_eu27['Region'] = 'EU27 (excl. Malta & Cyprus)'

    secondary_energy = pd.concat([secondary_energy, energy_eu27])

    data_wrapper.transformed_data['secondary_energy'] = secondary_energy

    return secondary_energy


def _generate_exogenous_costs_values(data_wrapper: dw.DataWrapper, cost_type: str):
    map_cost_technology = dr.loadmap_from_csv('costs')

    cost_values = data_wrapper.cost_values.copy()
    cost_values = cost_values[cost_values['type'] == cost_type]
    cost_values = cost_values[cost_values['technology'].isin(map_cost_technology.keys())]

    for entry in map_cost_technology:
        cost_values = cost_values.replace({'technology': entry},
                                          cost_type[:-1] + '|' + map_cost_technology[entry])


    cost_values['model'] = DEF_MODEL_AND_VERSION

    convert_eur_to2010_dollar = 1 / 1.08 * 1.17

    if cost_type == "Capital Costs":
        cost_values['unit'] = 'USD_2010/kW'
        cost_values['value'] = abs(cost_values['value']) * convert_eur_to2010_dollar


    elif cost_type == "Fixed Costs":
        cost_values['unit'] = 'USD_2010/kW/yr'
        cost_values['value'] = abs(cost_values['value']) * convert_eur_to2010_dollar


    elif cost_type == "Variable Costs":
        for idx in cost_values.index:
            if 'Variable Cost|Heat' in cost_values.loc[idx,'technology']:
                cost_values.loc[idx,'unit'] = 'USD_2010/MWh'
                cost_values.loc[idx,'value'] *= convert_eur_to2010_dollar*3.6
            else:
                cost_values.loc[idx,'unit'] = 'EUR/MWh' #USD_2010/MWh
                cost_values.loc[idx,'value'] = abs(cost_values.loc[idx,'value'])*3.6

    else:
        cost_values['unit'] = 'n/a'

    cost_values['scenario'] = data_wrapper.capacity_values['scenario'][0]
    cost_values = _set_scenarios(cost_values)

    cost_values['subannual'] = 'Year'

    cost_values = cost_values.groupby(
        ['model', 'scenario', 'region', 'technology', 'unit', 'subannual', 'year'])['value'].mean().reset_index()
    cost_values.columns = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']

    cost_eu27 = cost_values[cost_values['Region'] == 'DE'].copy()
    cost_eu27['Region'] = 'EU27 (excl. Malta & Cyprus)'

    cost_values = pd.concat([cost_values, cost_eu27])

    data_wrapper.transformed_data['Ex ' + str(cost_type)] = cost_values

    return cost_values


# so far only LCOE
def _generate_detailed_costs_values(data_wrapper: dw.DataWrapper, cost_type:str):
    map_cost_technology = dr.loadmap_from_csv('costs')
    map_storage_technology = dr.loadmap_from_csv('storages')

    cost_values = data_wrapper.detailed_cost_values.copy()
    cost_values = cost_values[cost_values['type'] == cost_type]
    cost_values = cost_values[cost_values['technology'].isin(map_cost_technology.keys())]


    for entry in map_cost_technology:
        if cost_type == "Levelized Costs [Capex]":
            cost_values = cost_values.replace({'technology': entry},
                                          "Levelized Cost|Capex" '|' + map_cost_technology[entry])
        elif cost_type == "Levelized Costs [Emissions]":
            cost_values = cost_values.replace({'technology': entry},
                                              "Levelized Cost|Emissions" '|' + map_cost_technology[entry])
        elif cost_type == "Levelized Costs [Generation]":
            cost_values = cost_values.replace({'technology': entry},
                                              "Levelized Cost|Generation" '|' + map_cost_technology[entry])
        elif cost_type == "Levelized Costs [Total w/o Emissions]":
            cost_values = cost_values.replace({'technology': entry},
                                              "Levelized Cost|Total (excl. Emissions)" '|' + map_cost_technology[entry])
        elif cost_type == "Levelized Costs [Total]":
            cost_values = cost_values.replace({'technology': entry},
                                              "Levelized Cost|Total (incl. Emissions)" '|' + map_cost_technology[entry])


    #add LCOE for storages
    if cost_type == "Levelized Costs [Total]":
        storage_values = data_wrapper.detailed_cost_values.copy()
        storage_values = storage_values[storage_values['type'] == cost_type]
        storage_values = storage_values[storage_values['technology'].isin(map_storage_technology.keys())]


        for entry in map_storage_technology:
            if entry == 'D_PHS':
                storage_values = storage_values.replace({'technology': entry},
                                                        'Levelized Cost|Generation|Electricity|' + map_storage_technology[entry])
            else:
                storage_values = storage_values.replace({'technology': entry},
                                                        'Levelized Cost|Electricity|' + map_storage_technology[entry])

        cost_values = pd.concat([cost_values, storage_values], ignore_index=True)


    cost_values['model'] = DEF_MODEL_AND_VERSION
    cost_values['unit'] = 'MEUR_2020/PJ'

    cost_values['scenario'] = data_wrapper.capacity_values['scenario'][0]
    cost_values = _set_scenarios(cost_values)

    cost_values['subannual'] = 'Year'

    cost_values = cost_values.groupby(
        ['model', 'scenario', 'region', 'technology', 'unit', 'subannual', 'year'])['value'].mean().reset_index()
    cost_values.columns = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']

    cost_eu27 = cost_values[cost_values['Region'] == 'DE'].copy()
    cost_eu27['Region'] = 'EU27 (excl. Malta & Cyprus)'

    cost_values = pd.concat([cost_values, cost_eu27])

    data_wrapper.transformed_data[str(cost_type)] = cost_values

    return cost_values


def _generate_exogenous_transport_costs_values(data_wrapper: dw.DataWrapper, cost_type: str):
    map_capacity_technology = dr.loadmap_from_csv('capacity_transport_technologies')
    map_units_technology = dr.loadmap_from_csv('units_transport_technologies')

    cost_values = data_wrapper.cost_values.copy()
    cost_values = cost_values[cost_values['type'] == cost_type]
    cost_values = cost_values[cost_values['technology'].isin(map_capacity_technology.keys())]

    convert_eur_to2010_dollar = 1 / 1.08 * 1.17

    cost_values['unit'] = 'Gvkm'
    for entry in map_capacity_technology:
        cost_values = cost_values.replace({'technology': entry},
                                          cost_type[:-1] + '|' + map_capacity_technology[entry])

    for entry in map_units_technology:
        cost_values.loc[(cost_values['technology'] == cost_type[:-1] + '|' + entry), 'unit'] = map_units_technology[entry][1:-3]

    if cost_type == "Capital Costs":
        cost_values['unit'] = 'USD_2010/' + cost_values['unit']
        cost_values['value'] = abs(cost_values['value']) * convert_eur_to2010_dollar

    elif cost_type == "Fixed Costs":
        cost_values['unit'] = 'USD_2010/' + cost_values['unit'] + '/yr'
        cost_values['value'] = abs(cost_values['value']) * convert_eur_to2010_dollar

    else:
        cost_values['unit'] = 'n/a'

    cost_values['model'] = DEF_MODEL_AND_VERSION

    for entry in map_units_technology:
        cost_values = cost_values.replace({'unit': entry}, map_units_technology[entry])

    cost_values['value'] = abs(cost_values['value'])
    cost_values['subannual'] = 'Year'

    cost_values['scenario'] = data_wrapper.capacity_values['scenario'][0]
    cost_values = _set_scenarios(cost_values)

    cost_values = cost_values.groupby(
        ['model', 'scenario', 'region', 'technology', 'unit', 'subannual', 'year'])['value'].mean().reset_index()
    cost_values.columns = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']

    cost_eu27 = cost_values[cost_values['Region'] == 'DE'].copy()
    cost_eu27['Region'] = 'EU27 (excl. Malta & Cyprus)'

    cost_values = pd.concat([cost_values, cost_eu27])

    data_wrapper.transformed_data['ex_transport ' + str(cost_type)] = cost_values

    return cost_values


def generate_co2_prices(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_co2_prices')

    cost_values = data_wrapper.cost_values.copy()
    cost_values = cost_values[cost_values['type'] == 'Carbon Price']
    cost_values = cost_values[cost_values['technology'] == 'Carbon']

    cost_values['technology'] = 'Price|Carbon'
    cost_values['unit'] = 'EUR_2020/t CO2'
    cost_values['model'] = DEF_MODEL_AND_VERSION
    cost_values['value'] = abs(cost_values['value'])
    cost_values['subannual'] = 'Year'

    cost_values['scenario'] = data_wrapper.capacity_values['scenario'][0]
    cost_values = _set_scenarios(cost_values)

    cost_values = cost_values.groupby(
        ['model', 'scenario', 'region', 'technology', 'unit', 'subannual', 'year'])['value'].mean().reset_index()
    cost_values.columns = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']

    cost_eu27 = cost_values[cost_values['Region'] == 'DE'].copy()
    cost_eu27['Region'] = 'EU27 (excl. Malta & Cyprus)'

    cost_values = pd.concat([cost_values, cost_eu27])

    data_wrapper.transformed_data['carbon_price'] = cost_values

    return cost_values


def generate_exogenous_costs(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_capital_costs_values')
    _generate_exogenous_costs_values(data_wrapper, "Capital Costs")

    logging.info('Executing: generate_fixed_costs_values')
    _generate_exogenous_costs_values(data_wrapper, "Fixed Costs")

    logging.info('Executing: generate_variable_costs_values')
    _generate_exogenous_costs_values(data_wrapper, "Variable Costs")

    logging.info('Executing: generate_transport_capital_costs_values')
    _generate_exogenous_transport_costs_values(data_wrapper, "Capital Costs")

    logging.info('Executing: generate_transport_fixed_costs_values')
    _generate_exogenous_transport_costs_values(data_wrapper, "Fixed Costs")


def generate_detailed_costs(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_levelized_costs_values')
    _generate_detailed_costs_values(data_wrapper, "Levelized Costs [Capex]")
    _generate_detailed_costs_values(data_wrapper, "Levelized Costs [Emissions]")
    _generate_detailed_costs_values(data_wrapper, "Levelized Costs [Generation]")
    _generate_detailed_costs_values(data_wrapper, "Levelized Costs [Total w/o Emissions]")
    _generate_detailed_costs_values(data_wrapper, "Levelized Costs [Total]")



def generate_load_factors(data_wrapper: dw.DataWrapper, input_file):
    logging.info('Executing: generate_load_factors')

    _generate_load_factors(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_PV_AVG", "Solar")
    _generate_load_factors(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_HYDRO_ROR", "Hydro|Run of River")
    _generate_load_factors(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_WIND_OFFSHORE", "Wind|Offshore")
    _generate_load_factors(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_WIND_ONSHORE_AVG", "Wind|Onshore")


def _generate_load_factors(data_wrapper: dw.DataWrapper, excel_file: str, sheet_name: str, nomenclature_name: str):
    #_dataframe = pd.read_csv(DEF_INPUT_PATH / csv_file, sep=';')
    _dataframe = pd.read_excel(DEF_INPUT_PATH / excel_file, sheet_name)
    dataframe = _dataframe.copy()

    timeseries = pd.Timestamp('2015-01-01') + pd.to_timedelta(dataframe.index, unit='h')
    timeseries = timeseries.strftime('%m-%d %H:%M+01:00')
    dataframe.set_index(timeseries, inplace=True)

    dataframe.drop('HOUR', axis=1, inplace=True)
    #dataframe.drop('TIMESLICE', axis=1, inplace=True)

    dataframe = pd.DataFrame(dataframe.stack())
    dataframe.reset_index(inplace=True)

    columns = ['subannual', 'region', 'value']
    dataframe.columns = columns

    dataframe['scenario'] = data_wrapper.capacity_values['scenario'][0]
    dataframe = _set_scenarios(dataframe)

    dataframe['unit'] = '%'
    dataframe['technology'] = 'Load Factor|Electricity|' + nomenclature_name
    dataframe['model'] = DEF_MODEL_AND_VERSION

    frames = []
    for y in DEF_YEARS:
        _df = dataframe.copy()
        _df['year'] = str(y)
        frames.append(_df)

    dataframe = pd.concat(frames)
    dataframe = dataframe.groupby(
        ['model', 'scenario', 'region', 'technology', 'unit', 'subannual', 'year'])['value'].mean().reset_index()
    dataframe.columns = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']

    dataframe = dataframe.replace({'Region': 'UK'}, 'GB')
    dataframe = dataframe.replace({'Region': 'NONEU_Balkan'}, 'Non-EU-Balkans')

    iso2_mapping = dr.loadmap_iso2_countries()

    for r in iso2_mapping:
        dataframe = dataframe.replace({'Region': r}, iso2_mapping[r])

    data_wrapper.transformed_data['loadFactor ' + excel_file] = dataframe


def generate_load_demand_series(data_wrapper: dw.DataWrapper, input_file: str):
    logging.info('Executing: generate_demand_series')
    _generate_demand_series(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_HEAT_LOW", "HeatingLow")
    _generate_demand_series(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_HEAT_HIGH", "HeatingHigh")
    _generate_demand_series(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_LOAD", "Power")
    _generate_demand_series(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_MOBILITY_PSNG", "Transport")


def _generate_demand_series(data_wrapper: dw.DataWrapper, excel_file: str, sheet_name: str, name: str):
    #_dataframe = pd.read_csv(DEF_INPUT_PATH / csv_file, sep=';')
    _dataframe = pd.read_excel(DEF_INPUT_PATH / excel_file, sheet_name=sheet_name)
    dataframe = _dataframe.copy()

    timeseries = pd.Timestamp('2015-01-01') + pd.to_timedelta(dataframe.index, unit='h')
    timeseries = timeseries.strftime('%m-%d %H:%M+01:00')
    dataframe.set_index(timeseries, inplace=True)

    dataframe.drop('HOUR', axis=1, inplace=True)
    #dataframe.drop('TIMESLICE', axis=1, inplace=True)

    dataframe = pd.DataFrame(dataframe.stack())
    dataframe.reset_index(inplace=True)

    columns = ['subannual', 'region', 'value']
    dataframe.columns = columns

    data_wrapper.demand_series[name] = dataframe

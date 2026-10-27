"""Turns the GENeSYS-MOD results of one scenario into IAMC variables.

The variable names come from the files in mappings/ (GENeSYS-MOD name -> IAMC variable, see README.md); this file
only selects the results, converts units and adds EU27. The sums of variables are added afterwards
(pyam_aggregator.py, mappings/aggregations.yaml).
"""
import genesysmod_to_iamc.data_reader as dr
import genesysmod_to_iamc.data_wrapper as dw
import pandas as pd
import logging
import itertools

from genesysmod_to_iamc._statics import *

IAMC_COLUMNS = ['Model', 'Scenario', 'Region', 'Variable', 'Unit', 'Subannual', 'Year', 'Value']


def _set_scenarios(frame: pd.DataFrame, data_wrapper: dw.DataWrapper):
    # every result file holds one scenario; its name in the outputs comes from config.yaml
    return frame.assign(scenario=data_wrapper.scenario)


def _set_timeslices(frame: pd.DataFrame, data_wrapper: dw.DataWrapper):
    return _rename(frame, 'subannual', data_wrapper.map_timeslices)


def _rename(frame: pd.DataFrame, column: str, mapping: dict):
    """Replace the GENeSYS-MOD names in a column by the names of a mapping."""
    return frame.replace({column: mapping})


def _to_iamc(frame: pd.DataFrame, column: str, how: str = 'sum'):
    """One value per IAMC key (sum, or 'mean' for costs and prices); column holds the variable names."""
    frame = frame.groupby(['model', 'scenario', 'region', column, 'unit', 'subannual', 'year'])['value'].agg(how)
    frame = frame.reset_index()
    frame.columns = IAMC_COLUMNS
    return frame


def _transform_columns(frame: pd.DataFrame, column='fuel'):
    """Like _to_iamc, per time slice and for the whole year ('Year')."""
    return pd.concat([_to_iamc(frame, column), _to_iamc(frame.assign(subannual='Year'), column)])


def _add_eu27(frame: pd.DataFrame, copy_of: str = None):
    """Add the region EU27: the sum over the EU member states, or for costs and prices the values of copy_of."""
    if copy_of is None:
        eu27 = frame[frame['Region'].isin(DEF_EU27)]
        eu27 = eu27.groupby(['Model', 'Scenario', 'Variable', 'Unit', 'Subannual', 'Year'])['Value'].sum().reset_index()
    else:
        eu27 = frame[frame['Region'] == copy_of].copy()
    eu27['Region'] = DEF_EU27_NAME
    return pd.concat([frame, eu27])


def _add_zero_variables(frame: pd.DataFrame, kind: str):
    """Add the variables of mappings/zero_var.csv of this kind with the value 0 for every region and year
    (variables starting with # are switched off)."""
    zero_var = dr.loaddf_from_csv('zero_var')
    variables = zero_var[zero_var['Type'] == kind]['Variable']
    zeros = pd.DataFrame(list(itertools.product(frame['Region'].unique(), frame['Year'].unique(), variables)),
                         columns=['Region', 'Year', 'Variable'])
    zeros = zeros[~zeros['Variable'].str.startswith('#')]
    first = frame.iloc[0]
    zeros[['Model', 'Scenario', 'Unit', 'Subannual', 'Value']] = first['Model'], first['Scenario'], first['Unit'], 'Year', 0
    return pd.concat([frame, zeros])


def generate_primary_energy_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_primary_energy_values')
    map_primary_fuels = dr.loadmap_from_csv('primary_energy_fuels')
    map_renewables = dr.loadmap_from_csv('primary_energy_renewables')
    map_electricity = dr.loadmap_from_csv('primary_energy_electricity')

    prod_values = data_wrapper.production_values.copy()
    use_values = data_wrapper.usage_values.copy()

    # fuels used (outside storage), by fuel
    traditional_energy = use_values[use_values['fuel'].isin(map_primary_fuels.keys())].copy()
    traditional_energy = traditional_energy[traditional_energy['sector'] != 'Storage']
    traditional_energy['value'] = abs(traditional_energy['value'])
    traditional_energy = _rename(traditional_energy, 'fuel', map_primary_fuels)

    # non-biomass renewables, by the technology that produces them
    non_res_biomass = prod_values[prod_values['technology'].isin(map_renewables.keys())].copy()
    non_res_biomass = _rename(non_res_biomass, 'technology', map_renewables)
    non_res_biomass['fuel'] = non_res_biomass['technology']

    nuclear = use_values[use_values['technology'] == 'P_Nuclear'].copy()
    nuclear['fuel'] = 'Primary Energy|Nuclear'

    # fuels used for electricity, by the power plant (mode 1)
    electricity = use_values[use_values['technology'].isin(map_electricity.keys())].copy()
    electricity = _rename(electricity, 'technology', map_electricity)
    electricity = electricity[electricity['mode'] == '1']
    electricity['fuel'] = electricity['technology']

    primary_energy = pd.concat([traditional_energy, non_res_biomass, nuclear, electricity])

    primary_energy['model'] = DEF_MODEL_AND_VERSION
    primary_energy['unit'] = 'EJ/yr'
    primary_energy['value'] = abs(primary_energy['value'] / 1000)

    primary_energy = _set_scenarios(primary_energy, data_wrapper)
    primary_energy = _set_timeslices(primary_energy, data_wrapper)
    primary_energy = _transform_columns(primary_energy, 'fuel')
    primary_energy = _add_eu27(primary_energy)
    primary_energy = _add_zero_variables(primary_energy, 'Primary Energy')

    data_wrapper.transformed_data['primary_energy'] = primary_energy
    return primary_energy


def generate_final_energy_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_final_energy_values')
    use_values = data_wrapper.usage_values.copy()
    map_final_sector = dr.loadmap_from_csv('final_energy_sector')
    map_final_fuel = dr.loadmap_from_csv('final_energy_fuel')

    # variable = sector part (final_energy_sector.csv) | fuel part (final_energy_fuel.csv)
    use_values = use_values[use_values['technology'].isin(map_final_sector.keys())]
    use_values = use_values[use_values['fuel'].isin(map_final_fuel.keys())]
    use_values = _rename(use_values, 'technology', map_final_sector)
    use_values = _rename(use_values, 'fuel', map_final_fuel)
    use_values['fuel'] = use_values['technology'] + '|' + use_values['fuel']

    # electricity demand outside the three sectors, removed again after the sums (aggregations.yaml)
    use_electricity_values = data_wrapper.usage_values.copy()
    use_electricity_values = use_electricity_values[use_electricity_values['fuel'] == 'Power']
    use_electricity_values = use_electricity_values[use_electricity_values['technology'] == 'Demand']
    use_electricity_values['fuel'] = 'Final Energy|ElectricityDummy'

    use_heat_values = _extract_final_energy_heat(data_wrapper)

    final_energy = pd.concat([use_values, use_electricity_values, use_heat_values])

    final_energy['model'] = DEF_MODEL_AND_VERSION
    final_energy['unit'] = 'EJ/yr'
    final_energy['value'] = abs(final_energy['value'] / 1000)

    final_energy = _set_scenarios(final_energy, data_wrapper)
    final_energy = _set_timeslices(final_energy, data_wrapper)
    final_energy = _transform_columns(final_energy, 'fuel')
    final_energy = _add_eu27(final_energy)
    final_energy = _add_zero_variables(final_energy, 'Final Energy')

    data_wrapper.transformed_data['final_energy'] = final_energy
    return final_energy


def _extract_final_energy_heat(data_wrapper):
    raw = data_wrapper.usage_values.copy()
    heat_fuels = ['Heat_Buildings', 'Heat_Low_Industrial', 'Heat_MediumLow_Industrial', 'Heat_MediumHigh_Industrial',
                  'Heat_High_Industrial']
    heat = pd.concat([raw[(raw['fuel'] == fuel) & (raw['technology'] == 'Demand')] for fuel in heat_fuels])
    heat['fuel'] = 'Final Energy|Heat'
    return heat


def generate_capacity_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_capacity_values')
    map_capacity_technology = dr.loadmap_from_csv('capacity_technologies')

    capacity_values = data_wrapper.capacity_values.copy()
    capacity_values = capacity_values[capacity_values['type'] == 'TotalCapacity']
    capacity_values = capacity_values[capacity_values['technology'].isin(map_capacity_technology.keys())]
    capacity_values = _rename(capacity_values, 'technology', map_capacity_technology)

    capacity_values['model'] = DEF_MODEL_AND_VERSION
    capacity_values['unit'] = 'GW'
    capacity_values['value'] = abs(capacity_values['value'])
    capacity_values['subannual'] = 'Year'

    capacity_values = _set_scenarios(capacity_values, data_wrapper)
    capacity_values = _add_eu27(_to_iamc(capacity_values, 'technology'))

    data_wrapper.transformed_data['capacity'] = capacity_values
    return capacity_values


def generate_transmission_capacity_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_transmission_capacity_values')

    trade_values = data_wrapper.trade_capacity_values.copy()
    trade_values = trade_values[(trade_values['type'] == 'Transmissions Capacity') & trade_values['fuel'].isin(['Power', 'Gas_Natural', 'H2'])]
    trade_values['model'] = DEF_MODEL_AND_VERSION
    trade_values['value'] = abs(trade_values['value'])*1000
    trade_values['subannual'] = 'Year'

    fuel_to_carrier = {
        'Power': 'Electricity',
        'Gas_Natural': 'Natural Gas',
        'H2': 'Hydrogen'
    }

    trade_values['variable'] = 'Network|' + trade_values['fuel'].map(fuel_to_carrier) + '|Capacity'

    fuel_to_unit = {
        'Power': 'MW',
        'Gas_Natural': 'TJ',
        'H2': 'TJ'
    }

    trade_values['unit'] = trade_values['fuel'].map(fuel_to_unit)

    trade_values = _set_scenarios(trade_values, data_wrapper)

    # regions of a connection are named origin>destination, with the country names of the explorer
    iso2_mapping = dr.loadmap_iso2_countries()
    trade_values = _rename(_rename(trade_values, 'region_from', iso2_mapping), 'region_to', iso2_mapping)
    trade_values['region'] = trade_values['region_from'] + ">" + trade_values['region_to']

    trade_values = _to_iamc(trade_values, 'variable')

    data_wrapper.transformed_data['transmission'] = trade_values
    trade_values.to_csv(DEF_OUTPUT_PATH / "trade_values.csv", index=False)
    return trade_values


def generate_storage_capacity_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_storage_capacity_values')
    map_capacity_storage = dr.loadmap_from_csv('storages')
    map_e2p_ratios = dr.loadmap_from_csv('storages_e2p_ratios')

    capacity_values = data_wrapper.capacity_values.copy()
    capacity_values = capacity_values[capacity_values['type'] == 'TotalCapacity']
    capacity_values = capacity_values[capacity_values['technology'].isin(map_capacity_storage.keys())]
    capacity_values = _rename(capacity_values, 'technology', map_capacity_storage)

    capacity_values['unit'] = 'GW'
    # switched off: storage energy capacity in GWh from the energy-to-power ratios (storages_e2p_ratios.csv)
    # capacity_values['unit'] = 'GWh'
    # capacity_values['value'] = abs(capacity_values['value']) * capacity_values['technology'].map(map_e2p_ratios)
    capacity_values['model'] = DEF_MODEL_AND_VERSION
    capacity_values['subannual'] = 'Year'

    capacity_values = _set_scenarios(capacity_values, data_wrapper)
    capacity_values = _add_eu27(_to_iamc(capacity_values, 'technology'))

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
    capacity_values = _rename(capacity_values, 'technology', map_capacity_technology)

    capacity_values['model'] = DEF_MODEL_AND_VERSION
    # switched off: transport capacity units per technology (units_transport_technologies.csv)
    # for entry in map_units_technology:
    #     capacity_values.loc[(capacity_values['technology'] == 'Capacity|' + entry), 'unit'] = map_units_technology[entry]

    capacity_values['value'] = abs(capacity_values['value'])
    capacity_values['subannual'] = 'Year'

    capacity_values = _set_scenarios(capacity_values, data_wrapper)
    capacity_values = _add_eu27(_to_iamc(capacity_values, 'technology'))

    data_wrapper.transformed_data['capacity_transport'] = capacity_values
    return capacity_values


def generate_emissions_values(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_emissions_values')
    map_emissions_technologies = dr.loadmap_from_csv('emissions_technologies')

    emission_values = data_wrapper.emission_values.copy()
    emission_values = emission_values[emission_values['technology'].isin(map_emissions_technologies.keys())]
    emission_values = _rename(emission_values, 'technology', map_emissions_technologies)

    emission_values['model'] = DEF_MODEL_AND_VERSION
    emission_values['unit'] = 'Mt CO2/yr'
    emission_values['subannual'] = 'Year'

    emission_values = _set_scenarios(emission_values, data_wrapper)
    emission_values = _add_eu27(_to_iamc(emission_values, 'technology'))

    data_wrapper.transformed_data['emissions'] = emission_values
    return emission_values


def generate_additional_emissions_values(data_wrapper: dw.DataWrapper):
    # switched off by default (outputs: additional_emissions in config.yaml): CO2 emissions per technology,
    # named after the capacity variables (Capacity|X -> Emissions|CO2|X)
    logging.info('Executing: generate_additional_emissions_values')
    map_capacity_technology = dr.loadmap_from_csv('capacity_technologies')
    map_emissions = {k: 'Emissions|CO2|' + v.removeprefix('Capacity|') for k, v in map_capacity_technology.items()}

    emission_values = data_wrapper.emission_values.copy()
    emission_values = emission_values[emission_values['technology'].isin(map_emissions.keys())]
    emission_values = _rename(emission_values, 'technology', map_emissions)

    emission_values['model'] = DEF_MODEL_AND_VERSION
    emission_values['unit'] = 'Mt CO2/yr'
    emission_values['subannual'] = 'Year'

    emission_values = _set_scenarios(emission_values, data_wrapper)
    emission_values = _add_eu27(_to_iamc(emission_values, 'technology'))

    data_wrapper.transformed_data['emissions2'] = emission_values
    return emission_values


def generate_secondary_energy(data_wrapper: dw.DataWrapper):
    logging.info('Executing: generate_secondary_energy')
    map_secondary_energy_power = dr.loadmap_from_csv('secondary_energy_power')
    prod_values = data_wrapper.production_values.copy()

    power_values = prod_values[prod_values['technology'].isin(map_secondary_energy_power.keys())].copy()
    power_values = power_values[power_values['fuel'] == 'Power']
    power_values = _rename(power_values, 'technology', map_secondary_energy_power)
    power_values['unit'] = 'EJ/yr'

    map_secondary_energy_heat = dr.loadmap_from_csv('secondary_energy_heat')
    heat_values = prod_values.copy()

    heat_values = heat_values[(heat_values['fuel'] == 'Heat_Buildings') |
                              (heat_values['fuel'] == 'Heat_Low_Industrial') |
                              (heat_values['fuel'] == 'Heat_MediumLow_Industrial') |
                              (heat_values['fuel'] == 'Heat_MediumHigh_Industrial') |
                              (heat_values['fuel'] == 'Heat_High_Industrial')]

    #rename X_Convert_HD technologies, so they can be correctly assigned to their sector
    mask_hd = heat_values["technology"].eq("X_Convert_HD")

    heat_values.loc[mask_hd & heat_values["fuel"].eq("Heat_Low_Industrial"), "technology"] = "X_Convert_HD_HLI"
    heat_values.loc[mask_hd & heat_values["fuel"].eq("Heat_Buildings"), "technology"] = "X_Convert_HD_HB"

    heat_values = heat_values[heat_values["technology"].isin(map_secondary_energy_heat.keys())].copy()
    heat_values = _rename(heat_values, 'technology', map_secondary_energy_heat)
    heat_values['unit'] = 'EJ/yr'

    # switched off: secondary energy of transport technologies
    # map_secondary_energy_transport = dr.loadmap_from_csv('secondary_energy_transport')
    # transport_values = prod_values[prod_values['technology'].isin(map_secondary_energy_transport.keys())].copy()
    # transport_values = transport_values[(transport_values['fuel'] == 'Mobility_Freight') |
    #                                     (transport_values['fuel'] == 'Mobility_Passenger')]
    # map_units_technology = dr.loadmap_from_csv('units_transport_technologies')
    # transport_values['unit'] = 'EJ/yr'
    # for entry in map_units_technology:
    #     transport_values = transport_values.replace({'unit': entry}, map_units_technology[entry] + "/yr")
    # transport_values = _rename(transport_values, 'technology', map_secondary_energy_transport)

    map_secondary_energy_other = dr.loadmap_from_csv('secondary_energy_other')
    other_values = prod_values[prod_values['technology'].isin(map_secondary_energy_other.keys())].copy()
    other_values = _rename(other_values, 'technology', map_secondary_energy_other)
    other_values['unit'] = 'EJ/yr'

    secondary_energy = pd.concat([power_values, heat_values, other_values])

    secondary_energy['model'] = DEF_MODEL_AND_VERSION
    secondary_energy['value'] = abs(secondary_energy['value'] / 1000)

    secondary_energy = _set_scenarios(secondary_energy, data_wrapper)
    secondary_energy = _set_timeslices(secondary_energy, data_wrapper)
    secondary_energy = _transform_columns(secondary_energy, 'technology')
    secondary_energy = _add_eu27(secondary_energy)

    data_wrapper.transformed_data['secondary_energy'] = secondary_energy
    return secondary_energy


def _generate_exogenous_costs_values(data_wrapper: dw.DataWrapper, cost_type: str):
    # variable = cost type without the plural s (e.g. Capital Cost) | technology part (costs.csv)
    map_cost_technology = dr.loadmap_from_csv('costs')
    map_costs = {k: cost_type[:-1] + '|' + v for k, v in map_cost_technology.items()}

    cost_values = data_wrapper.cost_values.copy()
    cost_values = cost_values[cost_values['type'] == cost_type]
    cost_values = cost_values[cost_values['technology'].isin(map_costs.keys())]
    cost_values = _rename(cost_values, 'technology', map_costs)

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

    cost_values = _set_scenarios(cost_values, data_wrapper)
    cost_values['subannual'] = 'Year'
    cost_values = _add_eu27(_to_iamc(cost_values, 'technology', 'mean'), copy_of='DE')

    data_wrapper.transformed_data['Ex ' + str(cost_type)] = cost_values
    return cost_values


def _generate_detailed_costs_values(data_wrapper: dw.DataWrapper, cost_type: str, prefix: str):
    # variable = prefix (cost_types.csv) | technology part (costs.csv); so far only LCOE
    map_cost_technology = dr.loadmap_from_csv('costs')
    map_costs = {k: prefix + '|' + v for k, v in map_cost_technology.items()}
    #map_storage_technology = dr.loadmap_from_csv('storages')

    cost_values = data_wrapper.detailed_cost_values.copy()
    cost_values = cost_values[cost_values['type'] == cost_type]
    cost_values = cost_values[cost_values['technology'].isin(map_costs.keys())]
    cost_values = _rename(cost_values, 'technology', map_costs)

    # switched off: levelized costs of storages
    # if cost_type == "Levelized Costs [Total]":
    #     storage_values = data_wrapper.detailed_cost_values.copy()
    #     storage_values = storage_values[storage_values['type'] == cost_type]
    #     storage_values = storage_values[storage_values['technology'].isin(map_storage_technology.keys())]
    #
    #     for entry in map_storage_technology:
    #         name = map_storage_technology[entry].removeprefix('Capacity|Electricity|')
    #         if entry == 'D_PHS':
    #             storage_values = storage_values.replace({'technology': entry},
    #                                                     'Levelized Cost|Generation|Electricity|' + name)
    #         else:
    #             storage_values = storage_values.replace({'technology': entry},
    #                                                     'Levelized Cost|Electricity|' + name)
    #
    #     cost_values = pd.concat([cost_values, storage_values], ignore_index=True)

    cost_values['model'] = DEF_MODEL_AND_VERSION
    cost_values['unit'] = 'million EUR_2020/PJ'

    cost_values = _set_scenarios(cost_values, data_wrapper)
    cost_values['subannual'] = 'Year'
    cost_values = _add_eu27(_to_iamc(cost_values, 'technology', 'mean'), copy_of='DE')

    data_wrapper.transformed_data[str(cost_type)] = cost_values
    return cost_values


def _generate_exogenous_transport_costs_values(data_wrapper: dw.DataWrapper, cost_type: str):
    # variable = cost type without the plural s | transport technology (capacity_transport_technologies.csv
    # without Capacity|); units per technology from units_transport_technologies.csv
    map_capacity_technology = dr.loadmap_from_csv('capacity_transport_technologies')
    map_units_technology = dr.loadmap_from_csv('units_transport_technologies')
    map_costs = {k: cost_type[:-1] + '|' + v.removeprefix('Capacity|') for k, v in map_capacity_technology.items()}

    cost_values = data_wrapper.cost_values.copy()
    cost_values = cost_values[cost_values['type'] == cost_type]
    cost_values = cost_values[cost_values['technology'].isin(map_costs.keys())]

    convert_eur_to2010_dollar = 1 / 1.08 * 1.17

    cost_values['unit'] = 'Gvkm'
    cost_values = _rename(cost_values, 'technology', map_costs)

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

    cost_values = _set_scenarios(cost_values, data_wrapper)
    cost_values = _add_eu27(_to_iamc(cost_values, 'technology', 'mean'), copy_of='DE')

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

    cost_values = _set_scenarios(cost_values, data_wrapper)
    cost_values = _add_eu27(_to_iamc(cost_values, 'technology', 'mean'), copy_of='DE')

    data_wrapper.transformed_data['carbon_price'] = cost_values
    return cost_values


def generate_exogenous_costs(data_wrapper: dw.DataWrapper):
    # switched off by default (outputs: exogenous_costs in config.yaml)
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
    for cost_type, prefix in dr.loadmap_from_csv('cost_types').items():
        _generate_detailed_costs_values(data_wrapper, cost_type, prefix)


def generate_load_factors(data_wrapper: dw.DataWrapper, input_file):
    logging.info('Executing: generate_load_factors')

    _generate_load_factors(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_PV_AVG", "Solar")
    _generate_load_factors(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_HYDRO_ROR", "Hydro|Run of River")
    _generate_load_factors(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_WIND_OFFSHORE", "Wind|Offshore")
    _generate_load_factors(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_WIND_ONSHORE_AVG", "Wind|Onshore")


def _generate_load_factors(data_wrapper: dw.DataWrapper, excel_file: str, sheet_name: str, nomenclature_name: str):
    dataframe = _read_hourly_series(excel_file, sheet_name)

    dataframe = _set_scenarios(dataframe, data_wrapper)

    dataframe['unit'] = '%'
    dataframe['technology'] = 'Load Factor|Electricity|' + nomenclature_name
    dataframe['model'] = DEF_MODEL_AND_VERSION

    dataframe = pd.concat([dataframe.assign(year=str(y)) for y in DEF_YEARS])
    dataframe = _to_iamc(dataframe, 'technology', 'mean')

    # dataframe = dataframe.replace({'Region': 'UK'}, 'GB')
    # dataframe = dataframe.replace({'Region': 'NONEU_Balkan'}, 'Non-EU-Balkans')
    #
    # iso2_mapping = dr.loadmap_iso2_countries()
    #
    # for r in iso2_mapping:
    #     dataframe = dataframe.replace({'Region': r}, iso2_mapping[r])

    data_wrapper.transformed_data['loadFactor ' + excel_file] = dataframe


def generate_load_demand_series(data_wrapper: dw.DataWrapper, input_file: str):
    logging.info('Executing: generate_demand_series')
    _generate_demand_series(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_HEAT_LOW", "HeatingLow")
    _generate_demand_series(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_HEAT_HIGH", "HeatingHigh")
    _generate_demand_series(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_LOAD", "Power")
    _generate_demand_series(data_wrapper, f"Timeseries_{input_file}.xlsx", "TS_MOBILITY_PSNG", "Transport")


def _generate_demand_series(data_wrapper: dw.DataWrapper, excel_file: str, sheet_name: str, name: str):
    data_wrapper.demand_series[name] = _read_hourly_series(excel_file, sheet_name)


def _read_hourly_series(excel_file: str, sheet_name: str):
    """One sheet of Timeseries_<scenario>.xlsx as rows of hour (as MM-DD HH:MM+01:00), region and value."""
    dataframe = pd.read_excel(DEF_INPUT_PATH / excel_file, sheet_name=sheet_name)

    timeseries = pd.Timestamp('2015-01-01') + pd.to_timedelta(dataframe.index, unit='h')
    timeseries = timeseries.strftime('%m-%d %H:%M+01:00')
    dataframe.set_index(timeseries, inplace=True)

    dataframe.drop('HOUR', axis=1, inplace=True)

    dataframe = pd.DataFrame(dataframe.stack())
    dataframe.reset_index(inplace=True)
    dataframe.columns = ['subannual', 'region', 'value']
    return dataframe

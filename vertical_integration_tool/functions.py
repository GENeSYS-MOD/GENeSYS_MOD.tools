import pandas as pd
import os
import shutil

# ==========================================
# MASTER CALLER FUNCTIONS
# ==========================================

def vertical_integration_standard(output_supermodel: str, input_submodel: str, input_supermodel: str, region: str):
    """Option 1: Standard Vertical Integration"""
    df_export, df_import, exogenous_trade_regions, df_supermodel, df_submodel_sets = read_files(
        output_supermodel, region, input_submodel, input_supermodel, mode='standard'
    )

    base_name, _ = os.path.splitext(input_submodel)
    output_file = f'Output/{base_name}_vi_input.xlsx'
    shutil.copyfile(f'Input/{input_submodel}', output_file)

    df_new_submodel_sets = new_sets_standard(df_submodel_sets, exogenous_trade_regions)
    df_new_export, df_new_import = output_parameters_standard(df_export, df_import)
    df_ExoTradeCapGrowthCosts, df_GrowthRateExoTradeCap = input_parameters_standard(
        df_new_submodel_sets, df_supermodel, region, exogenous_trade_regions
    )
    df_ExoTradeRoute_empty, df_ResidualExoTradeCap_empty, df_CommissionedExoTradeCap_empty = empty_parameters()

    to_excel_standard(
        output_file, df_new_submodel_sets, df_new_export, df_new_import, 
        df_ExoTradeCapGrowthCosts, df_GrowthRateExoTradeCap, 
        df_ExoTradeRoute_empty, df_ResidualExoTradeCap_empty, df_CommissionedExoTradeCap_empty
    )
    print(f"Standard vertical integration complete. Saved to {output_file}")


def vertical_integration_tag(output_supermodel: str, input_submodel: str, input_supermodel: str, region: str):
    """Option 2: Vertical Integration with Tag-System"""
    df_export, df_import, exogenous_trade_regions, df_supermodel, df_submodel = read_files(
        output_supermodel, region, input_submodel, input_supermodel, mode='tag'
    )

    base_name, _ = os.path.splitext(input_submodel)
    output_file = f'Output/{base_name}_vi_input_tag.xlsx'
    shutil.copyfile(f'Input/{input_submodel}', output_file)

    df_new_submodel_sets = new_sets_tag(df_submodel["Sets"], exogenous_trade_regions)
    df_new_export, df_new_import = output_parameters_tag(df_export, df_import)
    df_new_TradeCapGrowthCosts, df_new_GrowthRateTradeCap = input_parameters_tag(
        df_submodel, df_supermodel, region, exogenous_trade_regions
    )
    df_tag = tag_parameter(exogenous_trade_regions)

    to_excel_tag(
        output_file, df_new_submodel_sets, df_new_export, df_new_import, 
        df_new_TradeCapGrowthCosts, df_new_GrowthRateTradeCap, df_tag
    )
    print(f"Tag-based vertical integration complete. Saved to {output_file}")


def trade_output(output_supermodel: str, region: str):
    """Option 3: Retrieve Trade & Emissions Data"""
    df_export, df_import, exogenous_trade_regions, df_AnnualEmissions, df_AnnSectorEmissions = read_files(
        output_supermodel, region, mode='trade'
    )
    
    df_new_export, df_new_import = output_parameters_standard(df_export, df_import)
    df_ExoTradeRegions = exogenous_regions_list(exogenous_trade_regions)
    output_file = f'Output/trade_{region}.xlsx'

    to_excel_trade(output_file, df_new_export, df_new_import, df_ExoTradeRegions, df_AnnualEmissions, df_AnnSectorEmissions)
    print(f"Trade & Emissions data gathered. Saved to {output_file}")


# ==========================================
# DATA READING AND WRANGLING
# ==========================================

def read_files(output_supermodel, region: str, input_submodel: str = None, input_supermodel: str = None, mode: str = 'standard'):
    # Load Super/Submodel Inputs for integration modes
    if mode in ['standard', 'tag']:
        df_supermodel = pd.read_excel(f'Input/{input_supermodel}', sheet_name=['Par_TradeCapacityGrowthCosts', 'Par_GrowthRateTradeCapacity'])
        if mode == 'standard':
            df_submodel = pd.read_excel(f'Input/{input_submodel}', sheet_name='Sets')
        else: # tag
            df_submodel = pd.read_excel(f'Input/{input_submodel}', sheet_name=['Sets', 'Par_TradeCapacityGrowthCosts', 'Par_GrowthRateTradeCapacity'])

    # Helper: Process trade GDX/CSV
    def process_trade_data(df: pd.DataFrame, region: str, year_col: str='y_full') -> tuple[pd.DataFrame, list[str]]:
        df = df[df["REGION_FULL"] == region]
        valid_subset = df[(df['level'] != 0) & (~df['FUEL'].isin(['ETS', 'ETS_Source']))]
        valid_regions = list(valid_subset['rr_full'].unique())
        valid_fuels = list(valid_subset['FUEL'].unique())

        df = df[df['rr_full'].isin(valid_regions)]
        df = df[df['FUEL'].isin(valid_fuels)]

        if year_col == 'y_full':
            df = df[['y_full', 'TIMESLICE_FULL', 'FUEL', 'rr_full', 'level']]
        else:
            df = df[['YEAR_FULL', 'FUEL', 'rr_full', 'level']]
        return df, valid_regions
    
    # Helper: Process emissions GDX
    def process_emission_data(df: pd.DataFrame, region: str):
        col_to_use = "REGION_FULL" if "REGION_FULL" in df.columns else df.columns[0]
        return df[df[col_to_use] == region]

    # Extract output data based on filetype
    if isinstance(output_supermodel, list):
        df_import, i_regions = process_trade_data(pd.read_csv(f'Input/{output_supermodel[0]}'), region, 'y_full')
        df_export, e_regions = process_trade_data(pd.read_csv(f'Input/{output_supermodel[1]}'), region, 'y_full')
    else:
        file_path = f'Input/{output_supermodel}'
        _, file_type = os.path.splitext(output_supermodel) 

        if file_type.lower() == '.gdx':
            import gamspy as gp
            gdx_container = gp.Container(load_from=file_path)
            
            df_import, i_regions = process_trade_data(gdx_container['Import'].records, region, 'y_full')
            df_export, e_regions = process_trade_data(gdx_container['Export'].records, region, 'y_full')

            if mode == 'trade':
                df_AnnualEmissions = process_emission_data(gdx_container['AnnualEmissions'].records, region)
                df_AnnSectorEmissions = process_emission_data(gdx_container['AnnualSectorEmissions'].records, region)

                df_AnnualEmissions = df_AnnualEmissions[['y_full','EMISSION','REGION_FULL','level']]
                df_AnnualEmissions.columns = ['Year','Emission','Region','Value']
                df_AnnualEmissions = df_AnnualEmissions[['Region','Year','Emission','Value']]

                df_AnnSectorEmissions.columns = ['Region','Emission','Sector','Year','Value']
                df_AnnSectorEmissions = df_AnnSectorEmissions[['Region','Year','Emission','Sector','Value']]

        elif file_type.lower() in ['.xlsx', '.xls']:
            xls_file = pd.ExcelFile(file_path)
            df_import, i_regions = process_trade_data(xls_file.parse('Import'), region, 'y_full')
            df_export, e_regions = process_trade_data(xls_file.parse('Export'), region, 'y_full')
            xls_file.close()
        else:
            raise ValueError(f"Unrecognized file type: {file_type}")

    exogenous_trade_regions: list[str] = list(set(e_regions + i_regions))

    if mode in ['standard', 'tag']:
        return df_export, df_import, exogenous_trade_regions, df_supermodel, df_submodel
    else:
        return df_export, df_import, exogenous_trade_regions, df_AnnualEmissions, df_AnnSectorEmissions


# ==========================================
# STANDARD & TRADE HELPERS
# ==========================================

def new_sets_standard(df_sets: pd.DataFrame, exogenous_trade_regions: list[str]) -> pd.DataFrame:
    new_col_data = pd.Series(list(exogenous_trade_regions))
    if len(new_col_data) > len(df_sets):
        df_sets = df_sets.reindex(range(len(new_col_data)))
        
    if "Output Format" in df_sets.columns:
        cutoff_index = df_sets.columns.get_loc("Output Format") + 1
    else:
        cutoff_index = len([c for c in df_sets.columns if not str(c).startswith('Unnamed')])
        
    df_sets = df_sets.iloc[:, :cutoff_index].copy()
    df_sets["Exogenous_region"] = new_col_data
    return df_sets

def output_parameters_standard(df_export: pd.DataFrame, df_import: pd.DataFrame):
    df_export.columns = ['Year', 'Timeslice', 'Fuel', 'Exogenous_region', 'Value']
    df_import.columns = ['Year', 'Timeslice', 'Fuel', 'Exogenous_region', 'Value']
    return df_export, df_import

def input_parameters_standard(df_sets: pd.DataFrame, df_supermodel: dict, region: str, exogenous_trade_regions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_regions = df_sets['Region'].dropna().tolist()
    valid_regions = [r for r in all_regions if r not in ('World', region)]
    df_submodel_regions = pd.DataFrame({'Region': valid_regions})

    def process_param(df, final_columns):
        df_filtered = df[(df['Region'] == region) & (df['Region2'].isin(exogenous_trade_regions))].copy()
        df_filtered = df_filtered.drop(columns=['Region'])
        df_merged = df_filtered.merge(df_submodel_regions, how='cross')
        df_merged = df_merged.rename(columns={'Region2': 'Exogenous_region'})
        final_columns = [col if col != 'Region2' else 'Exogenous_region' for col in final_columns]
        return df_merged[final_columns]

    ExoTradeCapGrowthCosts = process_param(df_supermodel['Par_TradeCapacityGrowthCosts'], ['Region', 'Region2', 'Fuel', 'Value'])
    GrowthRateExoTradeCap = process_param(df_supermodel['Par_GrowthRateTradeCapacity'], ['Region', 'Region2', 'Fuel', 'Year', 'Value'])
    return ExoTradeCapGrowthCosts, GrowthRateExoTradeCap

def empty_parameters() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ExoTradeRoute_empty = pd.DataFrame([], columns=['Region', 'Exogenous_region', 'Fuel', 'Value'])
    ResidualExoTradeCap_empty = pd.DataFrame([], columns=['Region', 'Exogenous_region', 'Fuel', 'Year', 'Value'])
    CommissionedExoTradeCap_empty = pd.DataFrame([], columns=['Region', 'Exogenous_region', 'Fuel', 'Year', 'Value'])
    return ExoTradeRoute_empty, ResidualExoTradeCap_empty, CommissionedExoTradeCap_empty

def exogenous_regions_list(exogenous_trade_regions: list[str]) -> pd.DataFrame:
    return pd.DataFrame({'Region': list(exogenous_trade_regions)})


# ==========================================
# TAG HELPERS
# ==========================================

def new_sets_tag(df_sets: pd.DataFrame, exogenous_trade_regions: list[str]) -> pd.DataFrame:
    if "Output Format" in df_sets.columns:
        cutoff_index = df_sets.columns.get_loc("Output Format") + 1
    else:
        cutoff_index = len([c for c in df_sets.columns if not str(c).startswith('Unnamed')])
        
    df_sets = df_sets.iloc[:, :cutoff_index].copy()
    new_regions = list(exogenous_trade_regions)
    
    combined_regions = df_sets["Region"].dropna().tolist() + new_regions
    combined_regions2 = df_sets["Region2"].dropna().tolist() + new_regions
    
    max_required_length = max(len(combined_regions), len(combined_regions2))
    if max_required_length > len(df_sets):
        df_sets = df_sets.reindex(range(max_required_length))
        
    df_sets["Region"] = pd.Series(combined_regions)
    df_sets["Region2"] = pd.Series(combined_regions2)
    return df_sets

def output_parameters_tag(df_export: pd.DataFrame, df_import: pd.DataFrame):
    df_export.columns = ['Year', 'Timeslice', 'Fuel', 'Region', 'Value']
    df_import.columns = ['Year', 'Timeslice', 'Fuel', 'Region', 'Value']
    return df_export, df_import

def input_parameters_tag(df_submodel: dict, df_supermodel: dict, region: str, exogenous_trade_regions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_regions = df_submodel["Sets"]['Region'].dropna().tolist()
    valid_regions = [r for r in all_regions if r not in ('World', region) and r not in exogenous_trade_regions]
    df_submodel_regions = pd.DataFrame({'Region': valid_regions})

    def process_param(df, final_columns):
        df_filtered = df[(df['Region'] == region) & (df['Region2'].isin(exogenous_trade_regions))].copy()
        df_filtered = df_filtered.drop(columns=['Region'])
        df_merged = df_filtered.merge(df_submodel_regions, how='cross')
        return df_merged[final_columns]

    df_old_TradeCap = df_submodel['Par_TradeCapacityGrowthCosts']
    df_old_GrowthRate = df_submodel['Par_GrowthRateTradeCapacity']

    df_new_TradeCap = process_param(df_supermodel['Par_TradeCapacityGrowthCosts'], ['Region', 'Region2', 'Fuel', 'Value'])
    df_new_GrowthRate = process_param(df_supermodel['Par_GrowthRateTradeCapacity'], ['Region', 'Region2', 'Fuel', 'Year', 'Value'])

    df_final_TradeCap = pd.concat([df_old_TradeCap, df_new_TradeCap], ignore_index=True)
    df_final_GrowthRate = pd.concat([df_old_GrowthRate, df_new_GrowthRate], ignore_index=True)

    return df_final_TradeCap, df_final_GrowthRate

def tag_parameter(exogenous_trade_regions: list[str]) -> pd.DataFrame:
    return pd.DataFrame({'Region': list(exogenous_trade_regions),'Value': 1})


# ==========================================
# EXCEL WRITERS
# ==========================================

def to_excel_standard(output_file: str, df_sets: pd.DataFrame, df_export: pd.DataFrame, df_import: pd.DataFrame, df_ExoTradeCapGrowthCosts: pd.DataFrame, df_GrowthRateExoTradeCap: pd.DataFrame, df_ExoTradeRoute_empty: pd.DataFrame, df_ResidualExoTradeCap_empty: pd.DataFrame, df_CommissionedExoTradeCap_empty: pd.DataFrame):
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_sets.to_excel(writer, sheet_name='Sets', index=False)
        df_export.to_excel(writer, sheet_name='Par_ExogenousDemand', index=False)
        df_import.to_excel(writer, sheet_name='Par_ExogenousProduction', index=False)
        df_ExoTradeCapGrowthCosts.to_excel(writer, sheet_name='Par_ExoTradeCapacityGrowthCosts', index=False)
        df_GrowthRateExoTradeCap.to_excel(writer, sheet_name='Par_GrowthRateExoTradeCapacity', index=False)
        df_ExoTradeRoute_empty.to_excel(writer, sheet_name='Par_ExogenousTradeRoute', index=False)
        df_ResidualExoTradeCap_empty.to_excel(writer, sheet_name='Par_ResidualExoTradeCapacity', index=False)
        df_CommissionedExoTradeCap_empty.to_excel(writer, sheet_name='Par_CommissionedExoTradeCap', index=False)

def to_excel_tag(output_file: str, df_sets: pd.DataFrame, df_export: pd.DataFrame, df_import: pd.DataFrame, df_final_TradeCap: pd.DataFrame, df_final_GrowthRate: pd.DataFrame, df_tag: pd.DataFrame):
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df_sets.to_excel(writer, sheet_name='Sets', index=False)
        df_export.to_excel(writer, sheet_name='Par_ExogenousDemand', index=False)
        df_import.to_excel(writer, sheet_name='Par_ExogenousProduction', index=False)
        df_final_TradeCap.to_excel(writer, sheet_name='Par_TradeCapacityGrowthCosts', index=False)
        df_final_GrowthRate.to_excel(writer, sheet_name='Par_GrowthRateTradeCapacity', index=False)
        df_tag.to_excel(writer, sheet_name='Par_TagExogenousRegion', index=False)

def to_excel_trade(output_file: str, df_export: pd.DataFrame, df_import: pd.DataFrame, df_ExoTradeRegions: pd.DataFrame, df_AnnualEmissions: pd.DataFrame, df_AnnSectorEmissions: pd.DataFrame):
    file_exists = os.path.isfile(output_file)
    write_mode = 'a' if file_exists else 'w'
    writer_kwargs = {'engine': 'openpyxl', 'mode': write_mode}
    if write_mode == 'a':
        writer_kwargs['if_sheet_exists'] = 'replace'

    with pd.ExcelWriter(output_file, **writer_kwargs) as writer:
        df_ExoTradeRegions.to_excel(writer, sheet_name='Exogenous trade regions', index=False)
        df_export.to_excel(writer, sheet_name='Export', index=False)
        df_import.to_excel(writer, sheet_name='Import', index=False)
        df_AnnualEmissions.to_excel(writer, sheet_name='Annual emissions', index=False)
        df_AnnSectorEmissions.to_excel(writer, sheet_name='Annual sector emissions', index=False)
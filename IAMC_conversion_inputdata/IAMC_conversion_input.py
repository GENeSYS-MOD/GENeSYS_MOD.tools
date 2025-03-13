import os
import pandas as pd

# Define paths using the parent directory
data_folder_path = './Input/'
rename_mapping_technologies_path = './Mapping/IAMC_mappings_technolgies.csv'
rename_mapping_fuels_path = './Mapping/IAMC_mappings_fuels.csv'
rename_mapping_regions_path = './Mapping/IAMC_mappings_regions.csv'


# Define the mapping of old technology names to new technology names from csv file
rename_mapping_technologies = pd.read_csv(rename_mapping_technologies_path, index_col=0, header=None).squeeze("columns").to_dict()
rename_mapping_fuels = pd.read_csv(rename_mapping_fuels_path, index_col=0, header=None).squeeze("columns").to_dict()
rename_mapping_regions = pd.read_csv(rename_mapping_regions_path, index_col=0, header=None).squeeze("columns").to_dict()


# Function to rename technologies in CSV files in the data folder
def iamc_conversion(data_folder_path, rename_mapping_technologies, rename_mapping_fuels, rename_mapping_regions):

    # Create regions list
    regions = ["AT","BE","BG","CH","CZ","DE","DK","EE","ES","FI","FR","GR","HR","HU","IE","IT","LT","LU","LV","NL","NO","PL","PT","RO","SE","SI","SK","TR","UK","NONEU_Balkan","World"]

    columns = ["Region", "Variable", "Unit", 2018, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]
    df_final_combined = pd.DataFrame(columns=[["Model", "Scenario", "Region", "Variable", "Unit", 2018, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]])

    # List all items in the data folder
    files = os.listdir(data_folder_path)
    for file in files:
        file_path = os.path.join(data_folder_path, file)

        excel_file = pd.ExcelFile(file_path)

        sheet_names = excel_file.sheet_names

        columns = ["Region", "Variable", "Unit", 2018, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]
        df_final = pd.DataFrame(columns=columns)

        for item in sheet_names:

            # Skip the 'Sets' folder
            if item == 'Sets':
                continue


            full_report = ["Par_AnnualEmissionLimit","Par_AnnualExogenousEmission","Par_AvailabilityFactor","Par_CapitalCost","Par_CapitalCostStorage","Par_CommissionedTradeCapacity","Par_EmissionActivityRatio","Par_EmissionContentPerFuel","Par_FixedCost",
                        "Par_InputActivityRatio","Par_ModelPeriodActivityMaxLimit","Par_ModelPeriodEmissionLimit","Par_OperationalLife","Par_OperationalLifeStorage","Par_OutputActivityRatio","Par_RegionalCCSLimit",
                        "Par_ResidualCapacity","Par_ResidualStorageCapacity","Par_StorageE2PRatio","Par_SpecifiedAnnualDemand","Par_TechnologyDiscountRate","Par_TechnologyFromStorage","Par_TechnologyToStorage","Par_TotalAnnualMaxActivity",
                        "Par_TotalAnnualMaxCapacity","Par_TradeCapacity","Par_TradeCapacityGrowthCosts","Par_VariableCost"]

            specific_reporting = ["Par_ResidualCapacity", "Par_SpecifiedAnnualDemand"]

            if item not in full_report:
                continue

            # # Check if the item is a directory (parameter folder)
            # if os.path.isdir(item_path):
            #     # List all CSV files in the parameter folder
            #     csv_files = [f for f in os.listdir(item_path) if f.endswith('.csv')]
            #
            #     for csv_file in csv_files:
            #         csv_path = os.path.join(item_path, csv_file)

            df = pd.read_excel(file_path, sheet_name = item)


            if "Region" in df.columns:
                df = df[df["Region"].isin(regions)]
                df["Region"] = df["Region"].replace(rename_mapping_regions)

                if "Region2" in df.columns:
                    df = df[df["Region2"].isin(regions)]
                    df["Region2"] = df["Region2"].replace(rename_mapping_regions)

            if item == "Par_AnnualEmissionLimit":

                df_list = []

                for region in regions:
                    region_df = df.copy()

                    region_df['Region'] = region
                    region_df['Variable'] = 'Emissions|Annual Budget'  # This can be adjusted based on different variables
                    region_df['Unit'] = 'Mt CO2/yr'

                    df_list.append(region_df)

                df_temp = pd.concat(df_list)

                #df_temp.drop_duplicates(["Region", "Variable", "Unit", "Year"], keep="first", inplace=True)


                df_temp = df_temp.pivot(index=['Region', 'Variable', 'Unit'], columns='Year',
                                             values='Value')

                # Reset index to get a flat DataFrame
                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final,df_temp])

                print(f'Successfully converted {item}')

            if item == "Par_AnnualExogenousEmission":

                df["Unit"] = "Mt CO2/yr"
                df["Variable"] = "Emissions|Exogenous"

                #df_temp.drop_duplicates(["Region", "Variable", "Unit", "Year"], keep="first", inplace=True)


                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value')

                df_temp = df_temp.reset_index()

                df_temp.rename(columns={2021: 2018}, inplace=True)

                df_final = pd.concat([df_final,df_temp])

                print(f'Successfully converted {item}')

            if item in ["Par_AvailabilityFactor", "Par_CapitalCost", "Par_CapitalCostStorage", "Par_FixedCost", "Par_ModelPeriodActivityMaxLimit", "Par_ResidualCapacity", "Par_ResidualStorageCapacity",
                        "Par_StorageE2PRatio"]:

                if "Technology" in df.columns:
                    df = df[df["Technology"].isin(rename_mapping_technologies.keys())]
                elif "Storage" in df.columns:
                    df = df[df["Storage"].isin(rename_mapping_technologies.keys())]


                if item in ["Par_CapitalCostStorage","Par_ResidualStorageCapacity", "Par_StorageE2PRatio"]:
                    df["Storage"] = df["Storage"].replace(rename_mapping_technologies)
                else:
                    df["Technology"] = df["Technology"].replace(rename_mapping_technologies)

                if item == "Par_AvailabilityFactor":
                    df["Unit"] = "Fraction"
                    df["Variable"] = "Maximum Utilization|" + df["Technology"]

                elif item in ["Par_CapitalCost","Par_FixedCost"]:
                    df_list = []

                    for region in regions:
                        region_df = df.copy()

                        region_df["Unit"] = "EUR_2020/kW"
                        region_df = region_df[region_df["Year"] != 2021]

                        region_df['Region'] = region

                        if item == "Par_FixedCost":
                            region_df["Variable"] = "Fixed Cost|" + region_df["Technology"]
                        elif item == "Par_CapitalCost":
                            region_df["Variable"] = "Capital Cost|" + region_df["Technology"]

                        df_list.append(region_df)

                    df = pd.concat(df_list)

                elif item == "Par_CapitalCostStorage":
                    df["Unit"] = "EUR_2020/GJ"
                    df["Variable"] = "Capital Cost|" + df["Storage"]

                elif item == "Par_ModelPeriodActivityMaxLimit":
                    df["Unit"] = "PJ"
                    df["Variable"] = "Primary Energy|Total Reserves|" + df["Technology"]
                    df["Year"] = 2060

                elif item == "Par_ResidualCapacity":
                    df["Unit"] = "GW"
                    df["Variable"] = "Residual Capacity|" + df["Technology"]

                elif item == "Par_ResidualStorageCapacity":
                    df["Unit"] = "PJ"
                    df["Variable"] = "Residual Storage Capacity|" + df["Storage"]

                elif item == "Par_StorageE2PRatio":
                    df["Year"] = 2060
                    df["Unit"] = "PJ/GW"
                    df["Variable"] = "Energy to Power Ratio|" + df["Storage"]

                    df_list = []

                    for region in regions:
                        region_df = df.copy()

                        region_df['Region'] = region

                        df_list.append(region_df)

                    df = pd.concat(df_list)

                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final,df_temp])

                print(f'Successfully converted {item}')

            if item in ["Par_CommissionedTradeCapacity","Par_TradeCapacity", "Par_TradeCapacityGrowthCosts"]:

                df = df[df["Fuel"].isin(rename_mapping_fuels.keys())]
                df["Fuel"] = df["Fuel"].replace(rename_mapping_fuels)

                df["Region"] = df["Region"] + ">" +df["Region2"]

                if item == "Par_TradeCapacityGrowthCosts":
                    df.loc[df["Fuel"] == "Electricity", "Unit"] = "EUR_2020/kWkm"
                    df.loc[df["Fuel"] != "Electricity", "Unit"] = "EUR_2020/GJkm"
                else:
                    df.loc[df["Fuel"] == "Electricity", "Unit"] = "GW"
                    df.loc[df["Fuel"] != "Electricity", "Unit"] = "PJ"

                if item == "Par_CommissionedTradeCapacity":
                    df["Variable"] = "Network|Commissioned Capacity|" + df["Fuel"]

                elif item == "Par_TradeCapacityGrowthCosts":
                    df["Variable"] = "Network|Capacity Cost|" + df["Fuel"]
                    df["Year"] = 2060

                else:
                    df["Variable"] = "Network|Total Capacity|" + df["Fuel"]
                    df = df[df["Year"] != 2021]


                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final,df_temp])

                print(f'Successfully converted {item}')

            if item == "Par_EmissionActivityRatio":

                df = df[df["Technology"].isin(rename_mapping_technologies.keys())]
                df["Technology"] = df["Technology"].replace(rename_mapping_technologies)

                df = df[(df["Mode_of_operation"] == 1) & (df["Region"] == "World")]

                df["Unit"] = "Fraction"
                df["Variable"] = "Emissions|Activity Ratio|" + df["Technology"]

                df.drop_duplicates(["Region", "Technology", "Mode_of_operation", "Year"], keep="first", inplace=True)

                df_list = []

                for region in regions:
                    region_df = df.copy()

                    region_df['Region'] = region

                    df_list.append(region_df)

                df = pd.concat(df_list)

                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final, df_temp])

                print(f'Successfully converted {item}')

            if item == "Par_EmissionContentPerFuel":

                df = df[df["Fuel"].isin(rename_mapping_fuels.keys())]
                df["Fuel"] = df["Fuel"].replace(rename_mapping_technologies)

                df["Unit"] = "MtCO2/PJ"
                df["Variable"] = "Emissions|Carbon Content|" + df["Fuel"]
                df["Year"] = 2018

                df_list = []

                for region in regions:
                    region_df = df.copy()

                    region_df['Region'] = region

                    df_list.append(region_df)

                df = pd.concat(df_list)

                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final, df_temp])

                print(f'Successfully converted {item}')

            if item == "Par_ModelPeriodEmissionLimit":

                df['Region'] = "Europe"
                df['Variable'] = 'Emissions|Total Budget'
                df['Unit'] = 'Mt CO2'
                df["Year"] = 2060

                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year',
                                             values='Value')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final,df_temp])

            if item == "Par_OperationalLife":

                df = df[df["Technology"].isin(rename_mapping_technologies.keys())]
                df["Technology"] = df["Technology"].replace(rename_mapping_technologies)

                df["Unit"] = "Years"
                df["Variable"] = "Lifetime|" + df["Technology"]
                df["Year"] = 2060

                df_list = []

                for region in regions:
                    region_df = df.copy()

                    region_df['Region'] = region

                    df_list.append(region_df)

                df = pd.concat(df_list)

                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final, df_temp])

                print(f'Successfully converted {item}')

            if item == "Par_RegionalCCSLimit":

                df["Unit"] = "Mt"
                df["Variable"] = "Emissions|Total Carbon Storage Potential"
                df["Year"] = 2060

                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final, df_temp])

                print(f'Successfully converted {item}')

            if item == "Par_SpecifiedAnnualDemand":

                df = df[df["Fuel"].isin(rename_mapping_fuels.keys())]
                df["Fuel"] = df["Fuel"].replace(rename_mapping_fuels)
                df = df[(df["Year"] != 2019) & (df["Year"] != 2021)]

                df["Unit"] = "PJ"
                df.loc[df["Fuel"] == "Transportation|Freight", "Unit"] = "btkm"
                df.loc[df["Fuel"] == "Transportation|Passenger", "Unit"] = "bpkm"

                df["Variable"] = "Demand|" + df["Fuel"]

                df_temp = df.pivot_table(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value', aggfunc='sum')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final, df_temp])

                print(f'Successfully converted {item}')

            if item == "Par_TechnologyDiscountRate":

                df = df[df["Technology"].isin(rename_mapping_technologies.keys())]
                df["Technology"] = df["Technology"].replace(rename_mapping_technologies)

                df["Unit"] = "Fraction"
                df["Variable"] = "Discount Rate|" + df["Technology"]
                df["Year"] = 2018

                df_list = []

                for region in regions:
                    region_df = df.copy()

                    region_df['Region'] = region

                    df_list.append(region_df)

                df = pd.concat(df_list)

                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final, df_temp])

                print(f'Successfully converted {item}')

            if item == "Par_TechnologyFromStorage":

                df = df[df["Technology"].isin(rename_mapping_technologies.keys())]
                df["Technology"] = df["Technology"].replace(rename_mapping_technologies)

                df["Unit"] = "Fraction"
                df["Variable"] = "Discharging Efficiency|" + df["Technology"]

                df_list = []

                for region in regions:
                    region_df = df.copy()

                    region_df['Region'] = region

                    df_list.append(region_df)

                df = pd.concat(df_list)

                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final, df_temp])

                print(f'Successfully converted {item}')

            if item == "Par_TechnologyToStorage":

                df = df[df["Technology"].isin(rename_mapping_technologies.keys())]
                df["Technology"] = df["Technology"].replace(rename_mapping_technologies)

                df["Unit"] = "Fraction"
                df["Variable"] = "Charging Efficiency|" + df["Technology"]

                df_list = []

                for region in regions:
                    region_df = df.copy()

                    region_df['Region'] = region

                    df_list.append(region_df)

                df = pd.concat(df_list)

                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final, df_temp])

                print(f'Successfully converted {item}')

            if item in ["Par_TotalAnnualMaxActivity", "Par_TotalAnnualMaxCapacity"]:

                df = df[df["Technology"].isin(rename_mapping_technologies.keys())]
                df["Technology"] = df["Technology"].replace(rename_mapping_technologies)

                if item == "Par_TotalAnnualMaxCapacity":
                    df["Unit"] = "GW"
                    df["Variable"] = "Maximum Capacity|" + df["Technology"]
                else:
                    df["Unit"] = "PJ"
                    df["Variable"] = "Potential|" + df["Technology"]

                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final, df_temp])

                print(f'Successfully converted {item}')

            if item == "Par_VariableCost":

                df = df[df["Technology"].isin(rename_mapping_technologies.keys())]
                df["Technology"] = df["Technology"].replace(rename_mapping_technologies)

                df["Unit"] = "EUR_2020/GJ"
                df["Variable"] = "Variable Cost|" + df["Technology"]

                df = df[(df["Mode_of_operation"] == 1) & (df["Year"] != 2021)]

                df_list = []

                for region in regions:
                    region_df = df.copy()

                    region_df['Region'] = region

                    df_list.append(region_df)

                df = pd.concat(df_list)

                df.drop_duplicates(["Region", "Technology", "Mode_of_operation", "Year"], keep="first", inplace=True)

                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final, df_temp])

                print(f'Successfully converted {item}')

            if item == "Par_InputActivityRatio":

                df = df[df["Technology"].isin(rename_mapping_technologies.keys())]
                df["Technology"] = df["Technology"].replace(rename_mapping_technologies)

                df["Unit"] = "Fraction"
                df["Variable"] = "Efficiency|" + df["Technology"]

                df = df[(df["Mode_of_operation"] == 1) & (df["Year"] != 2021)]

                df = df.groupby(["Region", "Technology", "Year", "Unit", "Variable"])["Value"].sum().reset_index()

                df["Value"] = 1/df["Value"]

                df_list = []

                for region in regions:
                    region_df = df.copy()

                    region_df['Region'] = region

                    df_list.append(region_df)

                df = pd.concat(df_list)

                df.drop_duplicates(["Region", "Technology", "Unit", "Variable", "Year"], keep="first", inplace=True)

                df_temp = df.pivot(index=['Region', 'Variable', 'Unit'], columns='Year', values='Value')

                df_temp = df_temp.reset_index()

                df_final = pd.concat([df_final, df_temp])

                print(f'Successfully converted {item}')


            scenario = file.partition(".")[0]
            df_final["Model"] = "GENeSYS-MOD 4.0"
            df_final["Scenario"] = scenario + " v0"
            second = df_final.pop("Scenario")
            first = df_final.pop("Model")
            df_final.insert(0, 'Scenario', second)
            df_final.insert(0, 'Model', first)
            #df_final.drop(columns=[2020], axis=1)
            df_final["Region"] = df_final["Region"].replace(rename_mapping_regions)
            df_final.reset_index()
            df_final.to_csv("./Output/"+scenario+".csv")



# Call the function to rename technologies in the CSV files in the data folder
iamc_conversion(data_folder_path, rename_mapping_technologies, rename_mapping_fuels, rename_mapping_regions)

print("CSV file renaming complete.")

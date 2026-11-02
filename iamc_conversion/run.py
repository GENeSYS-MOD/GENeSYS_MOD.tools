"""Convert GENeSYS-MOD results to the IAMC format of the scenario explorers, from a terminal.

Choose the settings below, then run in this folder (with the environment genesysmod-iamc activated):

    python run.py                      convert all scenarios, combine them into the upload file and check it
    python run.py --validate FILE      only check an existing IAMC file (xlsx or csv)

The notebook iamc_conversion_outputs.ipynb runs the same conversion and has the same settings in its settings cells,
but keeps its own values: a change here does not change the notebook. Relative folders are relative to this folder.
"""
import sys

import genesysmod_to_iamc

# ---- the results to convert ----
# GAMS installation, used to read the gdx result files
gams_dir = 'C:/GAMS/46'

# folder with the results
input_dir = 'genesysmod_to_iamc/input'

# type of the results: 'gdx' (GAMS version) or 'csv' (Julia version)
file_type = 'gdx'

# the result file of every scenario, without extension (for csv results: the run name in the file names). The file
# names are also the scenario names in the outputs; to name them differently, write {file: name}, e.g.
# scenarios = {'GoRES_run1': 'Go RES', 'EUTrinity_run1': 'EU Trinity'}
scenarios = ['Go RES', 'REPowerEU++', 'NECP Essentials', 'EU Trinity']

# model years: the outputs cover only these years (other years in the results are left out)
years = [2018, 2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060]

# ---- names in the outputs ----
# model name and release (scenario "Go RES v2.0.1", model "GENeSYS-MOD 4.1")
model = 'GENeSYS-MOD 4.1'
release = 'v2.0.1'

# regions: the sum over all model regions, and the sum over the EU member states in the model
total_region = 'Europe (incl. Turkey)'
eu27_region = 'EU27 (excl. Malta & Cyprus)'

# ---- what to convert ----
transmission = True            # network capacities between regions (Network|...|Capacity)
hourly_series = False          # hourly demand series, in an extra file (large; needs Timeseries_<file>.xlsx)
load_factors = False           # hourly load factors of PV, wind and run-of-river, in an extra file (same)
combine = True                 # per scenario, one csv with the yearly values and the extra files
exogenous_costs = False        # switched off, kept for later use: capital, fixed and variable costs
additional_emissions = False   # switched off, kept for later use: CO2 emissions per technology

# ---- outputs and the check before an upload ----
# folder for the outputs
output_dir = 'genesysmod_to_iamc/out'

# name of the combined (upload) file, in the subfolder combined_excel of the output folder ({date} and {release}
# are filled in)
combined_file = 'GENeSYS-MOD-pathways_{date}.xlsx'

# variables left out of the upload file, e.g. while their definitions wait for a change in the explorer's workflow
# repository (* matches anything); the *_yearly.csv files keep them. Here: useful heat and the solar thermal and
# geothermal heat of buildings, whose iDesignRES definitions are not merged yet
leave_out = ['Useful Energy*',
             'Final Energy|Residential and Commercial|Solar',
             'Final Energy|Residential and Commercial|Geothermal']

# check the upload file with the upload workflow of the scenario explorer (usually
# https://github.com/iiasa/<project>-workflow; a copy of it is kept in ~/.genesysmod_to_iamc)
run_validation = True
workflow_repo = 'https://github.com/iiasa/idesignres-workflow'


if __name__ == '__main__':
    # pass the settings to the converter
    genesysmod_to_iamc.configure(
        gams_dir=gams_dir, input_dir=input_dir, file_type=file_type, scenarios=scenarios, years=years,
        model=model, release=release, total_region=total_region, eu27_region=eu27_region,
        transmission=transmission, hourly_series=hourly_series, load_factors=load_factors, combine=combine,
        exogenous_costs=exogenous_costs, additional_emissions=additional_emissions,
        output_dir=output_dir, combined_file=combined_file, leave_out=leave_out,
        run_validation=run_validation, workflow_repo=workflow_repo)

    if len(sys.argv) == 3 and sys.argv[1] == '--validate':
        sys.exit(0 if genesysmod_to_iamc.validate(sys.argv[2]) else 1)
    elif len(sys.argv) == 1:
        genesysmod_to_iamc.run()
    else:
        sys.exit(__doc__)

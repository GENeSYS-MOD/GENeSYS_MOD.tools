# GENeSYS-MOD to IAMC conversion

Converts GENeSYS-MOD results (gdx files of the GAMS version, or the csv outputs of the Julia version) into the
IAMC format of the IIASA scenario explorers, and checks them with the explorer's own upload workflow before you
upload.

## Setup (once)

    conda env create -f environment.yml
    conda activate genesysmod-iamc

GAMS has to be installed (path in `config.yaml`), because the gdx files are read through it.

## Run

1. Put the results of every scenario into `genesysmod_to_iamc/input/`: `<scenario>.gdx` of the GAMS version, or
   the result csv files of the Julia version (written with `switch_processed_results = 1`; their names contain the
   run name, which goes into `config.yaml` as the scenario's `file`). `Timeseries_<scenario>.xlsx` is only needed
   for the hourly series and load factors.
2. Check `config.yaml`: GAMS path, model version, release, the list of scenarios and the scenario explorer.
3. Run `python run.py` in this folder, or the notebook `iamc_conversion_outputs.ipynb`.

Outputs in `genesysmod_to_iamc/out/`:

- `<scenario>_yearly.csv` for every scenario (and `<scenario>_combined.csv` with the optional hourly files),
- the upload file in `combined_excel/` (name set in `config.yaml`),
- `validation_report.txt` with the result of the check.

## The check before an upload

Every scenario explorer checks uploads with the workflow of its project repository, usually
`https://github.com/iiasa/<project>-workflow`. That repository brings the project's definitions and pulls the IAMC
common-definitions. The converter keeps a copy of it in `~/.genesysmod_to_iamc/`, updates it before every check
and runs the same workflow, so no definitions have to be downloaded by hand. For another project, change
`validation: workflow_repo` in `config.yaml`.

The explorer runs the newest `nomenclature-iamc`, so update the package before an upload:

    pip install -U nomenclature-iamc pyam-iamc

To check a file on its own: `python run.py --validate <file.xlsx>`. The report lists every variable, region or
unit that the explorer would reject. Fix them in the mappings (below) and run the conversion again.

## Where to change what

| To change | Edit |
| --- | --- |
| GAMS path, model version, release, years, scenarios, what to convert, explorer | `config.yaml` |
| GENeSYS-MOD technology or fuel to IAMC variable | `genesysmod_to_iamc/mappings/*.csv` (table below) |
| sums of variables (e.g. Capacity\|Electricity from its parts) and the total over all regions | `genesysmod_to_iamc/mappings/aggregations.yaml` |
| country codes to country names | `genesysmod_to_iamc/mappings/GENeSYS-MOD 4.0.yaml` |
| variables reported as zero when the model has none | `genesysmod_to_iamc/mappings/zero_var.csv` |
| which variables, regions and units are allowed | the explorer's workflow repository (not here) |

The mapping files have two columns and no header: the GENeSYS-MOD name and the full IAMC variable. Several
GENeSYS-MOD names can go to the same variable; their values are added. Technologies and fuels that are in none of
the files are left out, apart from a few that `data_transform.py` selects itself (nuclear primary energy, heat
demand, networks, carbon price).

| File | GENeSYS-MOD name | IAMC variables |
| --- | --- | --- |
| `primary_energy_fuels.csv` | fuel used (outside storage) | `Primary Energy\|Biomass`, `\|Coal`, ... |
| `primary_energy_renewables.csv` | technology producing solar, wind, hydro, ocean or geothermal energy | `Primary Energy\|Solar`, ... |
| `primary_energy_electricity.csv` | power plant (its fuel use) | `Primary Energy\|<fuel>\|Electricity\|...` |
| `final_energy_sector.csv` and `final_energy_fuel.csv` | technology using the fuel, and the fuel | `Final Energy\|<sector>` and `<carrier>`, joined with `\|` |
| `secondary_energy_power.csv`, `_heat.csv`, `_other.csv` | technology producing electricity, heat or fuels | `Secondary Energy\|...` |
| `capacity_technologies.csv`, `storages.csv`, `capacity_transport_technologies.csv` | technology | `Capacity\|...` |
| `emissions_technologies.csv` | technology | `Emissions\|CO2\|...` |
| `cost_types.csv` and `costs.csv` | cost type of the results, and technology | `Levelized Cost\|<type>` and `<technology>`, joined with `\|` |

`aggregations.yaml` lists the sums in the order in which they are added; its first lines explain the entries.

Switched-off parts (exogenous costs, CO2 emissions per technology) can be switched on under `outputs` in
`config.yaml`. Smaller ones are kept for later use: as commented code marked "switched off" in
`data_transform.py`, as commented lines in `aggregations.yaml`, and as rows starting with `#` in the mapping files.
`secondary_energy_transport.csv`, `storages_e2p_ratios.csv` and `units_transport_technologies.csv` are only used
by switched-off code.

## Assumptions in the conversion

- Energy is converted from PJ to EJ/yr; capacities are in GW, transport capacities in Gvkm, emissions in Mt CO2/yr.
- EU27 is the sum over the 25 EU member states in the model (Malta and Cyprus are not modelled). Costs, levelized
  costs and the carbon price of EU27 are those of Germany.
- "Europe (incl. Turkey)" is the sum over EU27, Norway, the non-EU Balkans, Turkey, the UK and Switzerland.
- The variables in `zero_var.csv` are added with the value 0 for every region and year.
- The Julia version writes the levelized costs of storages with the columns Fuel and Year swapped; the converter
  swaps them back when it reads the csv results (with a warning).

## How the code is organised

| File | What it does |
| --- | --- |
| `_statics.py` | reads `config.yaml`; fixed column and sheet names of the GENeSYS-MOD outputs |
| `data_reader.py` | reads the gdx or csv results and the files in `mappings/` |
| `data_wrapper.py` | holds the results of one scenario, split into production, use, capacity, emissions, costs and trade |
| `data_transform.py` | turns them into IAMC variables with the mapping files (primary, final and secondary energy, capacities, emissions, costs, networks), converts units and adds EU27 |
| `pyam_aggregator.py` | adds the sums of `aggregations.yaml` and renames the regions to country names |
| `series_generator.py` | optional hourly demand series |
| `validation.py` | the check with the explorer's workflow |
| `__init__.py` | `run()`, `generate_data()`, `generate_combined_excel_yearly()`, `validate()` |

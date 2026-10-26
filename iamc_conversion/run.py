"""Convert GENeSYS-MOD results to the IAMC format of the scenario explorers (settings in config.yaml).

    python run.py                      convert all scenarios of config.yaml, combine them and validate the result
    python run.py --validate FILE      only check an existing IAMC file (xlsx or csv)
"""
import sys

import genesysmod_to_iamc

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--validate':
        sys.exit(0 if genesysmod_to_iamc.validate(sys.argv[2]) else 1)
    elif len(sys.argv) == 1:
        genesysmod_to_iamc.run()
    else:
        sys.exit(__doc__)

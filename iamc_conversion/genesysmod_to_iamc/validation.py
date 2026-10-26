"""Check IAMC outputs the same way the scenario explorer checks an upload.

Every IIASA scenario explorer validates uploads with the workflow of its project repository
(`workflow.py` with a function `main`, plus that project's definitions and its `nomenclature.yaml`,
which in turn pulls the IAMC common-definitions). This module keeps a local copy of that repository
outside of this repository, updates it before every check and runs its `main` on the data.
The repository is set in `_statics.py` (DEF_WORKFLOW_REPO_URL), so a future project only needs a new url.
"""
import importlib.metadata
import importlib.util
import logging
import time
from pathlib import Path

import git
import pyam

from genesysmod_to_iamc._statics import *


def workflow_repo_dir(url: str = DEF_WORKFLOW_REPO_URL) -> Path:
    """Local folder of the workflow repository, named after the repository."""
    return DEF_WORKFLOW_CACHE_DIR / url.rstrip('/').split('/')[-1].removesuffix('.git')


def update_workflow_repo(url: str = DEF_WORKFLOW_REPO_URL) -> Path:
    """Clone the workflow repository, or pull its newest version if it is already there."""
    local_dir = workflow_repo_dir(url)
    if (local_dir / '.git').is_dir():
        logging.info(f'Updating the workflow repository in {local_dir}')
        repo = git.Repo(local_dir)
        repo.git.reset('--hard')
        repo.remotes.origin.pull()
    else:
        logging.info(f'Cloning {url} to {local_dir}')
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        repo = git.Repo.clone_from(url, local_dir)
    logging.info(f'Workflow repository at commit {_commit(local_dir)}')
    return local_dir


def validate(data, update: bool = True, report_file=None) -> bool:
    """Validate IAMC data (a file path or an IamDataFrame) with the project workflow.

    Writes a report (default: out/validation_report.txt) and returns True if the data would be accepted.
    """
    df = data if isinstance(data, pyam.IamDataFrame) else pyam.IamDataFrame(str(data))
    # the upload files hold yearly values only, without the 'subannual' column
    if 'subannual' in df.extra_cols and set(df.data['subannual'].unique()) == {'Year'}:
        df = pyam.IamDataFrame(df.data.drop(columns='subannual'))

    local_dir = update_workflow_repo() if update else workflow_repo_dir()
    main = _load_workflow(local_dir)

    messages = _LogCollector()
    for name in ('nomenclature', 'pyam'):
        logging.getLogger(name).addHandler(messages)
    try:
        main(df)
        passed, error = True, None
    except Exception as e:
        passed, error = False, f'{type(e).__name__}: {e}'
    finally:
        for name in ('nomenclature', 'pyam'):
            logging.getLogger(name).removeHandler(messages)

    report_file = Path(report_file) if report_file else DEF_OUTPUT_PATH / 'validation_report.txt'
    _write_report(report_file, data, local_dir, passed, error, messages.lines)
    if passed:
        logging.info(f'Validation passed; report: {report_file}')
    else:
        logging.error(f'Validation failed; see {report_file}')
    return passed


def _load_workflow(local_dir: Path):
    spec = importlib.util.spec_from_file_location('project_workflow', local_dir / 'workflow.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def _commit(path: Path) -> str:
    try:
        head = git.Repo(path).head.commit
        return f'{head.hexsha[:7]} ({head.committed_datetime:%Y-%m-%d})'
    except Exception:
        return 'unknown'


def _write_report(report_file: Path, data, local_dir: Path, passed: bool, error, lines):
    versions = ', '.join(f'{p} {importlib.metadata.version(p)}' for p in ('nomenclature-iamc', 'pyam-iamc'))
    text = [
        'Validation of GENeSYS-MOD outputs with the scenario explorer workflow',
        f'date:                {time.strftime("%Y-%m-%d %H:%M")}',
        f'data:                {data if not isinstance(data, pyam.IamDataFrame) else "IamDataFrame"}',
        f'workflow repository: {DEF_WORKFLOW_REPO_URL}, commit {_commit(local_dir)}',
        f'common-definitions:  commit {_commit(local_dir / "common-definitions")}',
        f'packages:            {versions}',
        f'result:              {"PASSED" if passed else "FAILED"}',
        '',
    ]
    if error:
        text += ['error:', error, '']
    if lines:
        text += ['messages from the check:'] + lines
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text('\n'.join(text) + '\n', encoding='utf-8')


class _LogCollector(logging.Handler):
    """Collects the warnings and errors that nomenclature and pyam log during the check."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.lines = []

    def emit(self, record):
        self.lines.append(f'[{record.levelname}] {record.getMessage()}')

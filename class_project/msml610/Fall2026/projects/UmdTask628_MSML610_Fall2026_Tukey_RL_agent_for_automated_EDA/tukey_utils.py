"""
Notebook helpers for `tukey.API.ipynb` and `tukey.example.ipynb`.

The notebooks call these functions instead of holding logic inline, so each
notebook cell stays one task with one visible outcome.

Import as:

import tukey_utils as utils
"""

import importlib.metadata
import logging
from typing import Tuple

import pandas as pd

import helpers.hnotebook as hnotebo

_LOG = logging.getLogger(__name__)


def init_loggers(notebook_log: logging.Logger) -> None:
    """
    Wire the notebook logger into the utils logger.

    :param notebook_log: logger owned by the notebook
    """
    hnotebo.init_loggers(notebook_log, utils_log=_LOG)


# #############################################################################
# Cell 1.1: Environment check
# #############################################################################


# Packages whose versions the notebooks report, in display order.
_REPORTED_PACKAGES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "networkx",
    "causal-learn",
    "lingam",
    "CausalDisco",
    "pgmpy",
    "torch",
)


def cell1_1_get_package_versions(
    *, packages: Tuple[str, ...] = _REPORTED_PACKAGES
) -> pd.DataFrame:
    """
    Report the installed version of each package the project depends on.

    :param packages: distribution names to look up
        - Default: the project's pinned scientific stack
    :return: one row per package with columns `package` and `version`
    """
    rows = []
    for package in packages:
        # Read the version from the installed distribution metadata.
        version = importlib.metadata.version(package)
        rows.append({"package": package, "version": version})
    versions = pd.DataFrame(rows)
    return versions

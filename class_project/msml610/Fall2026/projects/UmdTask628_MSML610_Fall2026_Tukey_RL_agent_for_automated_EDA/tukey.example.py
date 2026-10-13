# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tukey: learning to explore data with verifiable rewards
#
# - This notebook runs the project end to end: synthetic causal worlds with a
#   known answer, a statistical toolbox with a scripted baseline, a policy
#   trained by reinforcement learning to choose tools, and the tests of how far
#   that policy generalizes, including real data
# - The pedagogical arc:
#   - Part 1: the synthetic environment and what its data leak (Milestone 1)
#   - Part 2: the toolbox, the scripted agent and the baseline table
#     (Milestone 2)
#   - Part 3: the trained policy against the scripted baseline (Milestone 3)
#   - Part 4: curriculum, generalization, failure modes and real data
#     (Milestone 4)

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
import helpers.hnotebook as hnotebook
import tukey_utils as utils

# Initialize notebook configuration and logging.
hnotebook.config_notebook()
_LOG = logging.getLogger(__name__)
utils.init_loggers(_LOG)

# Convert `display` into `print()` when running outside IPython.
try:
    from IPython.display import display
except ImportError:
    display = print  # type: ignore

# %% [markdown]
# # Part 1: The synthetic environment

# %% [markdown]
# ## Cell 1.1: Pinned scientific stack
#
# **Goal**
# - Confirm the container runs the pinned versions the results were produced
#   with

# %%
# Report the version of every package the project depends on.
versions = utils.cell1_1_get_package_versions()
display(versions)
# Outcome: one row per package, matching `requirements.txt`.

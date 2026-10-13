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
# # Tukey API
#
# - This notebook tours the Tukey API: a Gym-style environment that hides a
#   random causal graph, statistical tools an agent can call on the training
#   data, a verifiable score, and a policy trained to choose the tools
# - The pedagogical arc:
#   - Part 1: check the environment and the pinned scientific stack
#   - Part 2: sample a hidden graph and score candidate graphs against it
#   - Part 3: call the statistical tools and run the scripted agent
#   - Part 4: run a trained policy and read its tool-call trace

# %%
# %load_ext autoreload
# %autoreload 2

import logging

# %%
import helpers.hintrospection as hintros
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
# ## Library Overview
#
# - **What problem it solves**: learning which statistical test to run next on
#   an unknown dataset, rewarded by how well the discovered graph matches the
#   hidden graph that generated the data
# - **Key abstraction**: an episode = sample a hidden graph and data, call
#   tools on the training rows, emit a graph, receive a verifiable score
# - **Mental model**:
#
# | Object | Description | Comments |
# |--------|-------------|----------|
# | `GraphConfig` | Generator settings | Family, size, noise, scaling |
# | `TukeyEnv.reset()` | Samples a hidden graph and data | The graph stays private |
# | `TukeyEnv.step()` | Runs one tool call or emits a graph | Emitting ends the episode |
# | `score_graph()` | Verifiable score | Structure plus held-out prediction |
#
# - **Key classes**:
#   - `TukeyEnv`: the environment
#   - `TukeyPolicy`: the trained tool-choosing policy

# %% [markdown]
# # Part 1: Environment check

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

# %% [markdown]
# **Implementation** `cell1_1_get_package_versions()`
# - Reads each version from the installed distribution metadata

# %%
hintros.print_obj_info(utils.cell1_1_get_package_versions)

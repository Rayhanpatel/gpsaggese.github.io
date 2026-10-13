# Tukey: learning to explore data with verifiable rewards

- MSML610 Fall 2026 research project, group 31: Rayhan Patel, Ronak Vimal, Sai
  Katupilla; issue gpsaggese/umd_classes#628
- Implements GP's spec
  [`research/ideas/in_progress.RL_for_auto_EDA.md`](https://github.com/gpsaggese/gpsaggese.github.io/blob/master/research/ideas/in_progress.RL_for_auto_EDA.md):
  an agent runs statistical tests on an unknown dataset and emits the causal graph
  that generated it, trained with reinforcement learning on random graphs whose true
  structure is known, so the reward is verifiable
- Named after John Tukey, who named exploratory data analysis
- Builds on #506 and the prototype merged in #527
- Status: setup and integration gates in progress; results land in this README as
  each milestone is verified

## Quick Start

- Build the image (Python 3.12 slim, pinned dependencies):
  ```bash
  > cd class_project/msml610/Fall2026/projects/UmdTask628_MSML610_Fall2026_Tukey_RL_agent_for_automated_EDA
  > ./docker_build.sh
  ```
- Run the unit tests inside the container:
  ```bash
  > ./run_tests.sh
  ```
- Start Jupyter at http://localhost:8888, then open the notebooks in order:
  `tukey.API.ipynb`, then `tukey.example.ipynb`
  ```bash
  > ./docker_jupyter.sh
  ```

## Structure of the Dir

| Path | Content |
|---|---|
| `tukey.API.{ipynb,py}` | Tour of the Tukey API |
| `tukey.example.{ipynb,py}` | End-to-end study, Milestones 1-4 |
| `tukey_utils.py` | Helpers called by the notebooks |
| `tukey_runs.py`, `run_job.sh` | Isolated runs with immutable provenance |
| `check_repo_health.sh`, `run_tests.sh` | Repository and test gates |
| `docs/` | Plan for review and design notes |
| `test/` | Unit tests (`test_tukey_*.py`) and the host-side notebook test |
| Docker files | From `class_project/project_template`, unchanged except the image name |

## Description of Executables

### `run_tests.sh`

- Runs `test/test_tukey_*.py` inside the container, passing explicit file paths
  because `class_project/msml610/conftest.py` ignores directory collection

```bash
> ./run_tests.sh
> ./run_tests.sh -k dependencies
```

### `run_job.sh`

- Runs one job in a container named after the run, from a `git archive`
  snapshot of the committed project dir; the template's `docker_cmd.sh` reuses one
  container name and kills any running container with it, so concurrent jobs go
  through this script
- Writes `runs/<run_id>/provenance.json`, `log.txt`, `pip_freeze.txt` and
  `SHA256SUMS`, then makes the run read-only

```bash
> ./run_job.sh smoke_1 "python -c 'import tukey_utils'"
```

### `check_repo_health.sh`

- Checks the branch before every push: changes only in this dir, commit message
  suffix, no attribution lines, submodule unchanged, file sizes, clean merge with
  upstream, template files unchanged, pre-commit hooks pass

```bash
> ./check_repo_health.sh
```

## Docker

- Base image: the template's Python 3.12 slim, kept because every dependency
  installs from wheels
- Dependencies: pinned in `requirements.txt`; `ipywidgets` removed as
  `docker_scripts.README.md` requires; `torch==2.14.0+cpu` from the PyTorch CPU
  index (plain PyPI pulls CUDA packages on Linux); `lingam==1.12.2` because 1.13.0
  caps `scipy<=1.13.1`
- The Dockerfile is unchanged from the template; resolved versions are in
  `docker_build.version.log`

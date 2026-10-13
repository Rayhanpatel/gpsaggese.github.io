#!/bin/bash
# """
# Run one Tukey job in its own container, from a committed source snapshot.
#
# - `tukey_runs.py prepare` exports the committed project dir into
#   `runs/<run_id>/src/` and writes `provenance.json`
# - The command runs inside a container named `umd_project_tukey.job.<run_id>`.
#   GP's `docker_cmd.sh` uses one fixed name and kills any running container
#   with it, so concurrent jobs must not go through it
# - `tukey_runs.py finalize` records the exit code, hashes the outputs into
#   `SHA256SUMS` and makes the run directory read-only
#
# Usage:
# > ./run_job.sh <run_id> "<command, run from runs/<run_id>/src>"
# > ./run_job.sh --confirmatory <freeze_name> <run_id> "<command>"
# """

# Exit immediately if any command exits with a non-zero status.
set -e

# Import the template's docker helpers.
GIT_ROOT=$(git rev-parse --show-toplevel)
source $GIT_ROOT/class_project/project_template/utils.sh

# Parse the arguments.
CONFIRMATORY_ARGS=""
if [[ "$1" == "--confirmatory" ]]; then
    CONFIRMATORY_ARGS="--confirmatory $2"
    shift 2
fi
RUN_ID=$1
CMD=$2
if [[ -z "$RUN_ID" || -z "$CMD" ]]; then
    echo "Usage: ./run_job.sh [--confirmatory <freeze_name>] <run_id> \"<command>\""
    exit 1
fi

# Work from the project dir.
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REL_DIR=${PROJECT_DIR#"$GIT_ROOT"/}
cd "$PROJECT_DIR"

# Load the image name (REPO_NAME, IMAGE_NAME, FULL_IMAGE_NAME).
get_docker_vars_script ${BASH_SOURCE[0]}
source $DOCKER_NAME

# Snapshot the committed source and write the initial provenance.
export PYTHONPATH=$GIT_ROOT:$GIT_ROOT/helpers_root
python3 tukey_runs.py prepare --run_id "$RUN_ID" $CONFIRMATORY_ARGS --command "$CMD"

# Run the command in a container with a name unique to this job, and remove
# that container (only it) if the script is interrupted.
CONTAINER_NAME="${IMAGE_NAME}.job.${RUN_ID}"
DOCKER_CMD=$(get_docker_cmd)
DOCKER_OPTS=$(get_docker_bash_options $CONTAINER_NAME "" "")
cleanup_container_on_exit
RUN_DIR_IN_CONTAINER=/git_root/$REL_DIR/runs/$RUN_ID
set +e
$DOCKER_CMD run --rm -i $DOCKER_OPTS $FULL_IMAGE_NAME bash -c \
    "cd $RUN_DIR_IN_CONTAINER/src && pip freeze > ../pip_freeze.txt && $CMD" \
    > runs/$RUN_ID/log.txt 2>&1 < /dev/null
RC=$?
set -e

# Close the run: exit code, output hashes, read-only directory.
python3 tukey_runs.py finalize --run_id "$RUN_ID" --exit_code $RC
echo "Run '$RUN_ID' finished with exit code $RC (log: runs/$RUN_ID/log.txt)"
exit $RC

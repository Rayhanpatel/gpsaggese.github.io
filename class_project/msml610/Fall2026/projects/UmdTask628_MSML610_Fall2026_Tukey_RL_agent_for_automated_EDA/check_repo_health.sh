#!/bin/bash
# """
# Check that the branch is fit to push: integration gate G10.
#
# Read-only. Each check prints PASS or FAIL; the script exits non-zero if any
# check fails. Run it before every push and at every gate.
#
# Checks:
# - Changes (staged, unstaged, untracked) exist only under the project dir
# - The branch diff against upstream/master touches only the project dir
# - No tracked stray files (logs, tmp.*, *.html, runs/, test/outcomes/)
# - The branch name is UmdTask628_... (optionally with a _N suffix)
# - Every branch commit subject ends with (gpsaggese/umd_classes#628) and no
#   message carries an AI attribution line
# - The helpers_root submodule pointer equals upstream/master's
# - `git remote -v` prints 3 fields for origin (GP's hgit requires it)
# - No tracked project file is over 500 KB (the pre-commit limit) unless
#   listed in `.large_files_allowlist`
# - upstream/master merges into the branch without conflicts
# - The repo's pre-commit hooks pass on the files the branch changes
#
# Usage:
# > ./check_repo_health.sh
# """

GIT_ROOT=$(git rev-parse --show-toplevel)
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REL_DIR=${PROJECT_DIR#"$GIT_ROOT"/}
ISSUE=628
BRANCH_PREFIX="UmdTask${ISSUE}_MSML610_Fall2026_Tukey_RL_agent_for_automated_EDA"
cd "$GIT_ROOT"
N_FAIL=0

report() {
    # """
    # Print the outcome of one check and count failures.
    #
    # :param 1: check name
    # :param 2: details printed when the check fails (empty means PASS)
    # """
    if [[ -z "$2" ]]; then
        echo "PASS  $1"
    else
        echo "FAIL  $1"
        echo "$2" | sed 's/^/      /'
        N_FAIL=$((N_FAIL + 1))
    fi
}

# Changes only under the project dir.
OUTSIDE=$(git status --porcelain | awk '{print $NF}' | grep -v "^$REL_DIR/" || true)
report "working-tree changes only under the project dir" "$OUTSIDE"

# Branch diff only under the project dir.
MERGE_BASE=$(git merge-base upstream/master HEAD)
OUTSIDE=$(git diff --name-only "$MERGE_BASE" HEAD | grep -v "^$REL_DIR/" || true)
report "branch diff only under the project dir" "$OUTSIDE"

# No tracked stray files.
STRAY=$(git ls-files "$REL_DIR" | grep -E '\.log$|/tmp\.|\.html$|/runs/|/test/outcomes/' \
    | grep -v '/docker_build\.version\.log$' || true)
report "no tracked stray files" "$STRAY"

# Branch name.
BRANCH=$(git rev-parse --abbrev-ref HEAD)
BAD_BRANCH=""
if [[ ! "$BRANCH" =~ ^${BRANCH_PREFIX}(_[0-9]+)?$ ]]; then
    BAD_BRANCH="branch is '$BRANCH'"
fi
report "branch name follows UmdTask${ISSUE}_<tag>" "$BAD_BRANCH"

# Commit message suffix and no attribution lines.
BAD_SUBJECTS=$(git log --format=%s "$MERGE_BASE"..HEAD \
    | grep -v "(gpsaggese/umd_classes#${ISSUE})$" || true)
report "every commit subject ends with (gpsaggese/umd_classes#${ISSUE})" "$BAD_SUBJECTS"
ATTRIBUTION=$(git log --format=%B "$MERGE_BASE"..HEAD \
    | grep -iE 'co-authored-by|generated with' || true)
report "no AI attribution lines in commit messages" "$ATTRIBUTION"

# Submodule pointer unchanged.
OURS=$(git ls-tree HEAD helpers_root | awk '{print $3}')
THEIRS=$(git ls-tree upstream/master helpers_root | awk '{print $3}')
BAD_SUBMODULE=""
if [[ "$OURS" != "$THEIRS" ]]; then
    BAD_SUBMODULE="helpers_root is $OURS, upstream has $THEIRS"
fi
report "helpers_root submodule equals upstream" "$BAD_SUBMODULE"

# Remote format.
N_FIELDS=$(git remote -v | grep origin | grep fetch | awk '{print NF}')
BAD_REMOTE=""
if [[ "$N_FIELDS" != "3" ]]; then
    BAD_REMOTE="origin fetch line has $N_FIELDS fields"
fi
report "git remote -v prints 3 fields for origin" "$BAD_REMOTE"

# Large files.
ALLOWLIST="$REL_DIR/.large_files_allowlist"
LARGE=""
while read -r FILE; do
    [[ -f "$FILE" ]] || continue
    SIZE=$(wc -c < "$FILE" | tr -d ' ')
    if (( SIZE > 512000 )); then
        if [[ ! -f "$ALLOWLIST" ]] || ! grep -qxF "${FILE#$REL_DIR/}" "$ALLOWLIST"; then
            LARGE+="$FILE ($SIZE bytes)"$'\n'
        fi
    fi
done < <(git ls-files "$REL_DIR")
report "no tracked file over 500 KB outside the allowlist" "${LARGE%$'\n'}"

# Clean merge with upstream/master.
BAD_MERGE=""
if ! git merge-tree --write-tree upstream/master HEAD > /dev/null 2>&1; then
    BAD_MERGE="upstream/master does not merge cleanly"
fi
report "upstream/master merges without conflicts" "$BAD_MERGE"

# Docker infrastructure copied from GP's template must stay identical to it
# (docker_name.sh may differ only in IMAGE_NAME). These files are GP's, so they
# are checked against the template rather than by the hooks below.
TEMPLATE_DIR=class_project/project_template
TEMPLATE_FILES=(.dockerignore Dockerfile bashrc docker_bash.sh docker_build.sh
    docker_clean.sh docker_cmd.sh docker_exec.sh docker_jupyter.sh docker_name.sh
    docker_push.sh docker_scripts.README.md etc_sudoers run_jupyter.sh utils.sh
    version.sh)
DRIFT=""
for NAME in "${TEMPLATE_FILES[@]}"; do
    if [[ "$NAME" == "docker_name.sh" ]]; then
        if ! diff -q <(grep -v '^IMAGE_NAME=' "$TEMPLATE_DIR/$NAME") \
            <(grep -v '^IMAGE_NAME=' "$REL_DIR/$NAME") > /dev/null; then
            DRIFT+="$NAME"$'\n'
        fi
    elif ! cmp -s "$TEMPLATE_DIR/$NAME" "$REL_DIR/$NAME"; then
        DRIFT+="$NAME"$'\n'
    fi
done
report "template Docker files unchanged (except IMAGE_NAME)" "${DRIFT%$'\n'}"

# Pre-commit hooks on the files the branch changes, excluding GP's template
# files (deleted files skipped).
CHANGED=()
while read -r FILE; do
    [[ -f "$FILE" ]] || continue
    BASE=${FILE#$REL_DIR/}
    [[ " ${TEMPLATE_FILES[*]} " == *" $BASE "* ]] && continue
    CHANGED+=("$FILE")
done < <(git diff --name-only "$MERGE_BASE" HEAD -- "$REL_DIR")
HOOKS=""
if (( ${#CHANGED[@]} > 0 )); then
    if ! HOOK_OUT=$(uvx pre-commit run --files "${CHANGED[@]}" 2>&1); then
        HOOKS=$(echo "$HOOK_OUT" | grep -E "Failed|hook id" || echo "$HOOK_OUT" | tail -5)
    fi
fi
report "pre-commit hooks pass on the branch's files" "$HOOKS"

echo "---"
if (( N_FAIL > 0 )); then
    echo "Repository health: $N_FAIL check(s) failed"
    exit 1
fi
echo "Repository health: all checks passed"

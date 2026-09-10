#!/usr/bin/env bash
# Copy every .ninja_log under one or more CMake build trees into a named snapshot
# directory for later archiving. Usage:
#   collect_ninja_logs_snapshot.sh <archive_root> <snapshot_name> <build_dir> [<extra_build_dir> ...]
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <archive_root> <snapshot_name> <build_dir> [<extra_build_dir> ...]" >&2
  exit 1
fi

ARCHIVE_ROOT="$1"
SNAPSHOT_NAME="$2"
shift 2

collect_tree() {
  local root="$1"
  local prefix="$2"
  if [[ ! -d "${root}" ]]; then
    echo "Skip missing build directory: ${root}"
    return 0
  fi
  local dest_root="${ARCHIVE_ROOT}/${SNAPSHOT_NAME}"
  if [[ -n "${prefix}" ]]; then
    dest_root="${dest_root}/${prefix}"
  fi
  mapfile -d '' -t logs < <(find "${root}" -name '.ninja_log' -print0 2>/dev/null || true)
  if [[ ${#logs[@]} -eq 0 ]]; then
    echo "No .ninja_log files under ${root}"
    return 0
  fi
  echo "Snapshot ${SNAPSHOT_NAME}: collecting ${#logs[@]} .ninja_log file(s) from ${root}"
  for f in "${logs[@]}"; do
    rel="${f#"${root}"/}"
    dest="${dest_root}/${rel}"
    mkdir -p "$(dirname "${dest}")"
    cp -a "${f}" "${dest}"
  done
}

mkdir -p "${ARCHIVE_ROOT}/${SNAPSHOT_NAME}"

first="$1"
collect_tree "${first}" "$(basename "${first}")"
shift

for extra in "$@"; do
  [[ -z "${extra}" ]] && continue
  collect_tree "${extra}" "$(basename "${extra}")"
done

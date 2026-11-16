#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
build_dir="${repo_root}/build"
venv_dir="${build_dir}/venv"
model_id="${MODEL_ID:?MODEL_ID must be set}"
model_name="${model_id##*/}"
model_dir="${MODELS_DIR:?MODELS_DIR must be set}/${model_name,,}-openvino-ir"

main() {
    python3 -m venv --system-site-packages "${venv_dir}"
    source "${venv_dir}/bin/activate"

    export PYTHONPATH="${build_dir}${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="${build_dir}/openvino_genai${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    python -m pip install "${build_dir}"/wheels/*.whl --force-reinstall

    python -c "import openvino_genai; print(openvino_genai.__version__)"
    bash "${repo_root}/MEGAKERNEL_POC/scripts/benchmark_app.sh"
    python "${repo_root}/MEGAKERNEL_POC/python/e2e_performance_measurement.py" \
        --model-dir "${model_dir}" \
        --frameworks decode_only optimum genai \
        --torch-threads 20
}

main "$@"
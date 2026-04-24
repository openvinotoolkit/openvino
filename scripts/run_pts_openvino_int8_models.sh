#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BASE_DIR="${PTS_OPENVINO_DIR:-$HOME/pts-openvino}"
MODELS_DIR="${BASE_DIR}/models"
REPORT_DIR="${BASE_DIR}/benchmark_reports"
BENCH_APP="${REPO_ROOT}/bin/aarch64/Release/benchmark_app"
OMZ_BUCKET_URL="https://storage.openvinotoolkit.org"
FILETREE_JSON="${BASE_DIR}/filetree.json"
BENCHMARK_DURATION="${BENCHMARK_DURATION:-30}"
BENCHMARK_NITER="${BENCHMARK_NITER:-30}"
PRECISION_DIRS="${PRECISION_DIRS:-FP16-INT8}"

MODELS=(
  face-detection-0206
  age-gender-recognition-retail-0013
  weld-porosity-detection-0001
  vehicle-detection-0202
  face-detection-retail-0005
  handwritten-english-recognition-0001
  road-segmentation-adas-0001
)

mkdir -p "${BASE_DIR}" "${MODELS_DIR}" "${REPORT_DIR}"

if [[ ! -x "${BENCH_APP}" ]]; then
  echo "ERROR: benchmark_app not found at ${BENCH_APP}" >&2
  exit 1
fi

ensure_file() {
  local url="$1"
  local path="$2"
  if [[ -f "${path}" ]]; then
    local mime
    mime=$(file -b --mime-type "${path}" || true)
    if [[ -n "${mime}" && "${mime}" != "text/html" ]]; then
      return 0
    fi
  fi
  curl -L -o "${path}" "${url}"
}

for model in "${MODELS[@]}"; do
  if [[ ! -f "${FILETREE_JSON}" ]]; then
    echo "Downloading OpenVINO storage filetree..."
    curl -L -o "${FILETREE_JSON}" "${OMZ_BUCKET_URL}/filetree.json"
  fi

  model_rel_path=$(
    python3 - <<'PY' "${FILETREE_JSON}" "${model}" "${PRECISION_DIRS}"
import json, re, sys
filetree_path = sys.argv[1]
model = sys.argv[2]
precisions = [p.strip() for p in sys.argv[3].split(',') if p.strip()]
if not precisions:
    precisions = ["INT8"]
ver_re = re.compile(r'^\d+\.\d+$')

def ver_key(v):
    return tuple(int(x) for x in v.split('.'))

with open(filetree_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

matches = []

def walk(node, prefix=''):
    name = node.get('name','')
    path = f"{prefix}/{name}" if prefix else name
    if node.get('type') == 'file':
        for prec in precisions:
            if path.endswith(f"/{model}/{prec}/{model}.xml"):
                parts = path.split('/')
                try:
                    idx = parts.index('open_model_zoo')
                    ver = parts[idx + 1]
                except Exception:
                    continue
                if not ver_re.match(ver):
                    continue
                matches.append((ver, prec, path))
                break
    for child in node.get('children', []) or []:
        walk(child, path)

walk(data)

if not matches:
    sys.exit(2)

prec_idx = {p: i for i, p in enumerate(precisions)}
ver, prec, path = max(
    (ver_key(v), -prec_idx.get(prec, 999), v, prec, p) for v, prec, p in matches
)[2:]
# strip leading production/
if path.startswith('production/'):
    path = path[len('production/'):]
print(path)
PY
  ) || model_rel_path=""

  if [[ -z "${model_rel_path}" ]]; then
    echo "WARN: model ${model} not found in storage filetree" >&2
    continue
  fi

  model_rel_dir="${model_rel_path%/${model}.xml}"
  xml_url="${OMZ_BUCKET_URL}/${model_rel_dir}/${model}.xml"
  bin_url="${OMZ_BUCKET_URL}/${model_rel_dir}/${model}.bin"

  model_dir="${MODELS_DIR}/${model}/FP16-INT8"
  mkdir -p "${model_dir}"
  xml_path="${model_dir}/${model}.xml"
  bin_path="${model_dir}/${model}.bin"

  echo "Downloading ${model} XML..."
  ensure_file "${xml_url}" "${xml_path}"
  echo "Downloading ${model} BIN..."
  ensure_file "${bin_url}" "${bin_path}"

  if [[ -z "${xml_path}" ]]; then
    echo "WARN: model ${model} not found in ${MODELS_DIR}" >&2
    continue
  fi
  out_dir="${REPORT_DIR}/${model}"
  mkdir -p "${out_dir}"
  echo "Running ${model} ..."
  "${BENCH_APP}" -m "${xml_path}" -d CPU -api sync -t "${BENCHMARK_DURATION}" -niter "${BENCHMARK_NITER}" -pc \
    --report_type detailed_counters --report_folder "${out_dir}" --json_stats
  echo "Done ${model}, report: ${out_dir}"
done

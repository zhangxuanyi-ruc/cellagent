#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/root/wanghaoran/zxy/project/cellagent"
CONFIG="${1:-${PROJECT_ROOT}/config/config.yaml}"

cd "${PROJECT_ROOT}"
exec conda run -n scgpt python services/scgpt_feature_service.py --config "${CONFIG}"

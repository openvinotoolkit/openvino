//
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <opencv2/gapi/infer/ov.hpp>  // WorkloadTypeOVPtr{}
#include <opencv2/gapi/infer/onnx.hpp>  // WorkloadTypeONNXPtr{}
#include "parser/parser.hpp"


struct WorkloadTypeInfo {
    cv::gapi::wip::ov::WorkloadTypeOVPtr wl_ov;
    cv::gapi::onnx::WorkloadTypeONNXPtr wl_onnx;
    WorkloadTypeDesc workload_config;
};

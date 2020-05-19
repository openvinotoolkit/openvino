// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../tests_utils.h"
#include "../../common/tests_utils.h"
#include "../../common/utils.h"

#include <string>

// tests_pipelines/tests_pipelines.cpp
TestResult test_create_exenetwork(const std::string &model_name, const std::string &model_path, const std::string &target_device,
                                  const std::array<long, MeasureValueMax> &references);
TestResult test_infer_request_inference(const std::string &model_name, const std::string &model_path, const std::string &target_device,
                                        const std::array<long, MeasureValueMax> &references);
// tests_pipelines/tests_pipelines.cpp

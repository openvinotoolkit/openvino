// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../../common/ie_pipelines/pipelines.h"
#include "../../common/tests_utils.h"
#include "../../common/utils.h"

#include <string>

// tests_pipelines/tests_pipelines.cpp
TestResult common_test_pipeline(const std::vector<std::function<void()>> &test_pipeline, const int &n, const float& threshold_val = -1);
// tests_pipelines/tests_pipelines.cpp

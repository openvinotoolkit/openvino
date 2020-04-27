// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ti_tests.hpp"

ti_test_params ti_test_cases[] = {{std::string("CPU"), 1, InferenceEngine::Precision(InferenceEngine::Precision::FP32)},
                                  {std::string("CPU"), 8, InferenceEngine::Precision(InferenceEngine::Precision::FP32)},
                                  {std::string("CPU"), 1, InferenceEngine::Precision(InferenceEngine::Precision::FP32)},
                                  {std::string("CPU"), 8, InferenceEngine::Precision(InferenceEngine::Precision::FP16)}};


RUN_CASE_P_WITH_SUFFIX(CPU, _smoke, TITest, ti_test_cases);

RUN_CASE_P_WITH_SUFFIX(CPU, _smoke, TITest2, ti_test_cases);

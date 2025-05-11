// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "onnx/quantized_models_tests.hpp"

using namespace ONNXTestsDefinitions;

INSTANTIATE_TEST_SUITE_P(ONNXQuantizedModels, QuantizedModelsTests,
                        ::testing::Values(ov::test::utils::DEVICE_CPU),
                        QuantizedModelsTests::getTestCaseName);

// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//
// NOTE: WILL BE REWORKED (31905)

#include <gtest/gtest.h>

#include <map>

#include "common_test_utils/common_layers_params.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/xml_net_builder/ir_net.hpp"
#include "common_test_utils/xml_net_builder/xml_filler.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ie_core.hpp"
#include "single_layer_tests/eltwise.hpp"

using namespace EltwiseTestNamespace;

std::vector<EltwiseOpType> operations = { EltwiseOpType::ADD, EltwiseOpType::SUBSTRACT, EltwiseOpType::MULTIPLY };
std::vector<ParameterInputIdx> primary_input_idx = { 0, 1 };
std::vector<InputLayerType> secondary_input_types = { InputLayerType::CONSTANT , InputLayerType::PARAMETER };
std::vector<InferenceEngine::Precision> net_precisions = { InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16 };
std::vector<InferenceEngine::SizeVector> flat_shapes = { {1, 200}, {1, 2000}, {1, 20000} };
std::vector<InferenceEngine::SizeVector> non_flat_shapes = { {2, 200}, {10, 200}, {1, 10, 100}, {4, 4, 16} };
std::map<std::string, std::string> additional_config = {};

const auto FlatEltwiseParams =
::testing::Combine(
    ::testing::ValuesIn(operations),
    ::testing::ValuesIn(primary_input_idx),
    ::testing::ValuesIn(secondary_input_types),
    ::testing::ValuesIn(net_precisions),
    ::testing::ValuesIn(flat_shapes),
    ::testing::Values(CommonTestUtils::DEVICE_GPU),
    ::testing::Values(additional_config));

const auto NonFlatEltwiseParams =
::testing::Combine(
    ::testing::ValuesIn(operations),
    ::testing::ValuesIn(primary_input_idx),
    ::testing::ValuesIn(secondary_input_types),
    ::testing::ValuesIn(net_precisions),
    ::testing::ValuesIn(non_flat_shapes),
    ::testing::Values(CommonTestUtils::DEVICE_GPU),
    ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(Eltwise_flat, EltwiseLayerTest, FlatEltwiseParams,
    EltwiseLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(Eltwise_non_flat, EltwiseLayerTest, NonFlatEltwiseParams,
    EltwiseLayerTest::getTestCaseName);

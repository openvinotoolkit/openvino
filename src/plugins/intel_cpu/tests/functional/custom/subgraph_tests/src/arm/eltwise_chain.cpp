// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>

#include "custom/subgraph_tests/src/classes/eltwise_chain.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
using namespace ov::test::utils;
using namespace ov::test::eltwise_chain;

namespace {

std::vector<std::vector<EltwiseTypes>> eltwiseOpsConvertInt8 = {
        { EltwiseTypes::MULTIPLY },
        { EltwiseTypes::ADD },
        { EltwiseTypes::DIVIDE }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain_MergeConvert_int8, EltwiseChainTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(static_shapes_to_test_representation(inputShapesConvert())),
                                 ::testing::Values(InputLayerType::CONSTANT),
                                 ::testing::ValuesIn(inputPrecisionsConvert()),
                                 ::testing::ValuesIn(eltwiseOpsConvertInt8),
                                 ::testing::Values(false),
                                 ::testing::ValuesIn({ov::element::i8, ov::element::u8}),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EltwiseChainTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov

// Copyright (C) 2018-2025 Intel Corporation
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

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain,
                         EltwiseChainTest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapes())),
                                            ::testing::Values(InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(inputPrecisions()),
                                            ::testing::ValuesIn(eltwiseOps()),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EltwiseChainTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_EltwiseChain_MergeConvert,
    EltwiseChainTest,
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesConvert())),
                       ::testing::Values(InputLayerType::CONSTANT),
                       ::testing::ValuesIn(inputPrecisionsConvert()),
                       ::testing::ValuesIn(eltwiseOpsConvert()),
                       ::testing::Values(false),
                       ::testing::Values(ov::element::f32),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    EltwiseChainTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> inputShapesFQ = {
    {{1, 2, 2, 3}, {1, 2, 2, 3}, {1, 2, 2, 3}, {1, 2, 2, 3}},
    {{2, 33, 5, 5}, {2, 33, 5, 5}, {2, 33, 1, 5}, {2, 33, 5, 5}},
    {{2, 33, 5, 17}, {2, 33, 5, 17}, {2, 33, 5, 17}, {2, 33, 5, 17}},
    {{2, 33, 5, 256}, {2, 33, 5, 256}, {2, 33, 5, 256}, {2, 33, 5, 256}},
    {{2, 5, 7, 5}, {2, 5, 1, 5}, {2, 5, 7, 5}, {2, 5, 7, 5}},
    {{2, 17, 7, 5}, {2, 17, 7, 5}, {2, 17, 7, 5}, {2, 17, 7, 5}},
    {{2, 256, 7, 5}, {2, 256, 7, 5}, {2, 256, 1, 5}, {2, 256, 7, 5}},
    {{1, 36, 34, 34}, {1, 36, 34, 34}, {1, 36, 34, 34}, {1, 36, 34, 34}},
    {{1, 12, 1, 1, 6}, {1, 12, 5, 1, 6}, {3, 12, 1, 5, 1}, {3, 12, 5, 1, 1}},
    {{1, 12, 1, 1, 6}, {1, 12, 5, 5, 6}, {3, 12, 1, 5, 1}, {3, 12, 5, 5, 1}},
    {{1, 12, 1, 1, 1}, {1, 12, 5, 1, 7}, {3, 12, 1, 5, 7}, {3, 12, 5, 1, 7}},
    {{1, 7, 1, 1, 12}, {1, 7, 5, 1, 12}, {3, 7, 1, 5, 12}, {3, 7, 5, 1, 12}},
    {{1, 7, 1, 1, 12, 3, 7}, {1, 7, 5, 1, 12, 3, 7}, {3, 7, 1, 5, 12, 3, 7}, {3, 7, 5, 1, 12, 3, 7}},
    {{1, 7, 1, 1, 12, 3, 1}, {1, 7, 5, 1, 12, 3, 7}, {3, 7, 1, 5, 12, 1, 7}, {3, 7, 5, 1, 12, 3, 1}}
};

std::vector<std::vector<ElementType>> inputPrecisionsFQ {
        { ElementType::f32, ElementType::f32, ElementType::f32, ElementType::f32 }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChainWithFQ,
                         EltwiseChainTest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapesFQ)),
                                            ::testing::Values(InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(inputPrecisionsFQ),
                                            ::testing::ValuesIn(eltwiseOps()),
                                            ::testing::Values(true),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EltwiseChainTest::getTestCaseName);

// =============================================== dynamic ==============================================
std::vector<std::vector<InputShape>> inputShapes_dyn = {
    {
        // inp1
        {
            // dynamic
            {-1, -1, -1},
            // target
            {
                {1, 2, 3},
                {5, 2, 7},
                {3, 1, 10},
            }
        },
        // inp2
        {
            // dynamic
            {-1},
            // target
            {
                {3}, {7}, {1},
            }
        },
        // inp3
        {
            // dynamic
            {-1},
            // target
            {
                {3}, {1}, {1}
            }
        },
        // inp4
        {
            // dynamic
            {-1},
            // target
            {
                {3}, {1}, {1}
            }
        }
    },
    {
        // inp1
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 12, 5, 5},
                {5, 16, 1, 5},
                {2, 1, 1, 5},
            }
        },
        // inp2
        {
            // dynamic
            {-1, -1},
            // target
            {
                {5, 5}, {1, 5}, {5, 1},
            }
        },
        // inp3
        {
            // dynamic
            {-1, -1, -1},
            // target
            {
                {12, 5, 5},
                {1, 5, 1},
                {16, 5, 5},
            }
        },
        // inp4
        {
            // dynamic
            {-1},
            // target
            {
                {1}, {1}, {5}
            }
        }
    },
    {
        // inp1
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 2, 2, 3},
                {2, 33, 5, 5},
                {2, 33, 5, 17},
                {2, 33, 5, 256},
                {2, 5, 7, 5},
                {2, 17, 7, 5},
                {2, 256, 7, 5},
                {1, 36, 34, 34},
            }
        },
        // inp2
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 2, 2, 3},
                {2, 33, 5, 5},
                {2, 33, 5, 17},
                {2, 33, 5, 256},
                {2, 5, 1, 5},
                {2, 17, 7, 5},
                {2, 256, 7, 5},
                {1, 36, 34, 34},
            }
        },
        // inp3
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 2, 2, 3},
                {2, 33, 1, 5},
                {2, 33, 5, 17},
                {2, 33, 5, 256},
                {2, 5, 7, 5},
                {2, 17, 7, 5},
                {2, 256, 1, 5},
                {1, 36, 34, 34}
            }
        },
        // inp4
        {
            // dynamic
            {-1, -1, -1, -1},
            // target
            {
                {1, 2, 2, 3},
                {2, 33, 5, 5},
                {2, 33, 5, 17},
                {2, 33, 5, 256},
                {2, 5, 7, 5},
                {2, 17, 7, 5},
                {2, 256, 7, 5},
                {1, 36, 34, 34}
            }
        }
    },
    {
        // inp1
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {1, 12, 1, 1, 6},
                {1, 12, 1, 1, 6},
                {1, 12, 1, 1, 1},
                {1, 7, 1, 1, 12},
            }
        },
        // inp2
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {1, 12, 5, 1, 6},
                {1, 12, 5, 5, 6},
                {1, 12, 5, 1, 7},
                {1, 7, 5, 1, 12},
            }
        },
        // inp3
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {3, 12, 1, 5, 1},
                {3, 12, 1, 5, 1},
                {3, 12, 1, 5, 7},
                {3, 7, 1, 5, 12}
            }
        },
        // inp4
        {
            // dynamic
            {-1, -1, -1, -1, -1},
            // target
            {
                {3, 12, 5, 1, 1},
                {3, 12, 5, 5, 1},
                {3, 12, 5, 1, 7},
                {3, 7, 5, 1, 12}
            }
        }
    },
    {
        // inp1
        {
            // dynamic
            {-1, -1, -1, -1, -1, -1, -1},
            // target
            {
                {1, 7, 1, 1, 12, 3, 7},
                {1, 7, 1, 1, 12, 3, 1},
                {5, 7, 1, 2, 12, 1, 8},
            }
        },
        // inp2
        {
            // dynamic
            {-1, -1, -1, -1, -1, -1, -1},
            // target
            {
                {1, 7, 5, 1, 12, 3, 7},
                {1, 7, 5, 1, 12, 3, 7},
                {1, 7, 5, 1, 12, 3, 8},
            }
        },
        // inp3
        {
            // dynamic
            {-1, -1, -1, -1, -1, -1, -1},
            // target
            {
                {3, 7, 1, 5, 12, 3, 7},
                {3, 7, 1, 5, 12, 1, 7},
                {5, 1, 1, 2, 12, 1, 8},
            }
        },
        // inp4
        {
            // dynamic
            {-1, -1, -1, -1, -1, -1, -1},
            // target
            {
                {3, 7, 5, 1, 12, 3, 7},
                {3, 7, 5, 1, 12, 3, 1},
                {1, 7, 5, 1, 12, 3, 1}
            }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain_dyn,
                         EltwiseChainTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes_dyn),
                                            ::testing::Values(InputLayerType::PARAMETER),
                                            ::testing::ValuesIn(inputPrecisions()),
                                            ::testing::ValuesIn(eltwiseOps()),
                                            ::testing::Values(false),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EltwiseChainTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov

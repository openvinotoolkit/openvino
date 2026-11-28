// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "custom/subgraph_tests/src/classes/eltwise_chain.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/core.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
using namespace ov::test::utils;
using namespace ov::test::eltwise_chain;

namespace {

#if defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_RISCV64)
TEST(EltwiseChain, smoke_SignedUnaryWrapsBeforeConsumer) {
    ov::Core core;
    auto check = [&](auto min_value, const ov::element::Type type) {
        std::array<decltype(min_value), 5> values{min_value, -2, 0, 2, 127};
        const ov::Shape shape{values.size()};
        const ov::Tensor input(type, shape, values.data());
        for (bool negate : {false, true}) {
            auto param = std::make_shared<ov::op::v0::Parameter>(type, shape);
            ov::Output<ov::Node> unary = param;
            if (negate) {
                unary = std::make_shared<ov::op::v0::Negative>(param);
            } else {
                unary = std::make_shared<ov::op::v0::Abs>(param);
            }
            const std::array<float, 5> expected{static_cast<float>(min_value), 2.f, 0.f,
                                                negate ? -2.f : 2.f, negate ? -127.f : 127.f};
            for (bool use_maximum : {false, true}) {
                SCOPED_TRACE(::testing::Message() << type << ", negate=" << negate << ", maximum=" << use_maximum);
                auto value = unary;
                if (use_maximum) {
                    auto zero = ov::op::v0::Constant::create(type, {}, {0});
                    value = std::make_shared<ov::op::v1::Maximum>(unary, zero);
                }
                auto result = std::make_shared<ov::op::v0::Convert>(value, ov::element::f32);
                auto model = std::make_shared<ov::Model>(ov::OutputVector{result}, ov::ParameterVector{param});
                auto request = core.compile_model(model, ov::test::utils::DEVICE_CPU).create_infer_request();
                request.set_input_tensor(input);
                request.infer();
                const auto output = request.get_output_tensor();
                const auto* actual = output.data<const float>();
                for (size_t i = 0; i < expected.size(); ++i) {
                    EXPECT_EQ(actual[i], use_maximum ? std::max(0.f, expected[i]) : expected[i]) << "index " << i;
                }
            }
        }
    };
    check(std::numeric_limits<int8_t>::min(), ov::element::i8);
    check(std::numeric_limits<int32_t>::min(), ov::element::i32);
}
#endif

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChain,
                         EltwiseChainTest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inputShapes())),
                                            ::testing::Values(InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(inputPrecisions()),
                                            ::testing::ValuesIn(eltwiseOps()),
                                            ::testing::Values(PostNode::EMPTY),
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
                       ::testing::Values(PostNode::EMPTY),
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
                                            ::testing::Values(PostNode::QUANTIZE),
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
                                            ::testing::Values(PostNode::EMPTY),
                                            ::testing::Values(ov::element::dynamic),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EltwiseChainTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseChainWithSoftSign_dyn,
                         EltwiseChainTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes_dyn),
                                            ::testing::Values(InputLayerType::PARAMETER),
                                            ::testing::ValuesIn(inputPrecisions()),
                                            ::testing::ValuesIn(eltwiseOps()),
                                            ::testing::Values(PostNode::SOFTSIGN),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         EltwiseChainTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov

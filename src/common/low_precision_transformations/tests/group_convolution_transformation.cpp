// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/group_convolution.hpp"
#include <memory>
#include <sstream>
#include <string>
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/group_convolution.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

class GroupConvolutionTestValues {
public:
    class Actual {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantization;
        std::shared_ptr<ov::op::v0::Constant> weights;
        ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
        ov::builder::subgraph::DequantizationOperations dequantizationOnWeights;
    };

    class Expected {
    public:
        ov::element::Type precisionBeforeDequantization;
        ov::builder::subgraph::DequantizationOperations dequantizationBefore;
        std::shared_ptr<ov::op::v0::Constant> weights;
        ov::builder::subgraph::FakeQuantizeOnWeights fakeQuantizeOnWeights;
        ov::builder::subgraph::DequantizationOperations dequantizationOnWeights;
        ov::element::Type precisionAfterOperation;
        ov::builder::subgraph::DequantizationOperations dequantizationAfter;
        ov::element::Type precisionAfterDequantization;
    };

    TestTransformationParams params;
    size_t group;
    int groupCalculationDimention;
    bool addReshape;
    Actual actual;
    Expected expected;
};

typedef std::tuple<std::pair<ov::PartialShape, ov::PartialShape>,  // inputShape - outputShape
                   GroupConvolutionTestValues>
    ConvolutionTransformationParams;

class GroupConvolutionTransformation : public LayerTransformation,
                                       public testing::WithParamInterface<ConvolutionTransformationParams> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam()).first;
        const auto outputShape = std::get<0>(GetParam()).second;
        const GroupConvolutionTestValues testValues = std::get<1>(GetParam());

        actualFunction =
            ov::builder::subgraph::GroupConvolutionFunction::get(testValues.actual.precisionBeforeDequantization,
                                                                     inputShape,
                                                                     outputShape,
                                                                     testValues.group,
                                                                     testValues.groupCalculationDimention,
                                                                     testValues.actual.dequantization,
                                                                     testValues.actual.weights,
                                                                     testValues.actual.fakeQuantizeOnWeights,
                                                                     testValues.actual.dequantizationOnWeights,
                                                                     ov::element::f32,
                                                                     {},
                                                                     ov::element::f32,
                                                                     testValues.addReshape);

        SimpleLowPrecisionTransformer transform;
        transform.add<ov::pass::low_precision::GroupConvolutionTransformation, ov::opset1::GroupConvolution>(
            testValues.params);
        if (testValues.params.supportAsymmetricQuantization == false) {
            transform.get_pass_config()->set_callback<ov::pass::low_precision::GroupConvolutionTransformation>(
                [](const std::shared_ptr<const ov::Node>& node) -> bool {
                    return ov::pass::low_precision::LayerTransformation::isAsymmetricQuantization(node);
                });
        }
        transform.transform(actualFunction);

        referenceFunction =
            ov::builder::subgraph::GroupConvolutionFunction::get(testValues.expected.precisionBeforeDequantization,
                                                                     inputShape,
                                                                     outputShape,
                                                                     testValues.group,
                                                                     testValues.groupCalculationDimention,
                                                                     testValues.expected.dequantizationBefore,
                                                                     testValues.expected.weights,
                                                                     testValues.expected.fakeQuantizeOnWeights,
                                                                     testValues.expected.dequantizationOnWeights,
                                                                     testValues.expected.precisionAfterOperation,
                                                                     testValues.expected.dequantizationAfter,
                                                                     testValues.expected.precisionAfterDequantization,
                                                                     testValues.addReshape);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConvolutionTransformationParams> obj) {
        const auto inputShape = std::get<0>(obj.param).first;
        const auto outputShape = std::get<0>(obj.param).second;
        const GroupConvolutionTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result << toString(testValues.params) << "_" << inputShape << "_" << outputShape << "_" << testValues.group
               << "_" << testValues.groupCalculationDimention << "_" << testValues.actual.precisionBeforeDequantization
               << "_" << testValues.actual.dequantization << "_"
               << "_add_reshape:" << testValues.addReshape << "_"
               << "_weights_type:" << testValues.actual.weights->get_element_type() << "_"
               << "_weights_shape:" << testValues.actual.weights->get_shape() << "_"
               << "{ " << testValues.actual.weights->cast_vector<float>()[0] << " }_"
               << testValues.actual.fakeQuantizeOnWeights << "_";
        return result.str();
    }
};

TEST_P(GroupConvolutionTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

// clang-format off
namespace testValues1 {

const std::vector<std::pair<ov::PartialShape, ov::PartialShape>> shapesForGroupConv = {
    {{1, 6, 224, 224}, {1, 24, 218, 218}},
    {{-1, -1, -1, -1}, {-1, -1, -1, -1}}
};

const std::vector<GroupConvolutionTestValues> testValuesGroupConv = {
    // group convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
         3ul,
         -1,
         true,
         // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{}, {{128.f}, ov::element::f32, {1, 6, 1, 1}, false}, {}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {{}, {}, {{0.0002f}, ov::element::f32, {}}}  // 0.0002 = 0.02 (on data) * 0.01 (on weights)
        }
    },

    // group convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        3ul,
        0,
        true,
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{}, {{128.f}, ov::element::f32, {1, 6, 1, 1}, false}, {}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {{}, {}, {{0.0002f}, ov::element::f32, {}}}  // 0.0002 = 0.02 (on data) * 0.01 (on weights)
        }
    },

    // group convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
        3ul,
        1,
        true,
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{}, {{128.f}, ov::element::f32, {1, 6, 1, 1}, false}, {}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {{}, {}, {{0.0002f}, ov::element::f32, {}}}  // 0.0002 = 0.02 (on data) * 0.01 (on weights)
        }
    },

    // group convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false),
        3ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {},
            ov::element::f32,
            {}
        }
    },

    // group convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        3ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::f32,
            {{}, {128.f}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::f32,
            {{}, {{128.f}, ov::element::f32, {1, 6, 1, 1}, false}, {}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {{}, {}, {{0.0002f}, ov::element::f32, {}}}  // 0.0002 = 0.02 (on data) * 0.01 (on weights)
        }
    },

    // group convolution, per-channel quantization with different values, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.02f, 0.02f, 0.04f, 0.04f, 0.08f, 0.08f}, ov::element::f32, {1, 6, 1, 1}}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {
                {},
                {},
                {
                    {
                        // 0.0002 = 0.02 (on data) * 0.01 (on weights)
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        // 0.0004 = 0.04 (on data) * 0.01 (on weights)
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        // 0.0008 = 0.08 (on data) * 0.01 (on weights)
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f
                    },
                    ov::element::f32,
                    {1, 24, 1, 1}
                }
            },
        }
    },

    // group convolution, per-channel quantization with the same values, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.02f}, ov::element::f32, {1, 6, 1, 1}}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {},
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {{}, {}, {{0.0002f}, ov::element::f32, {}}},
        }
    },

    // group convolution, without zero point, without convert
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::f32,
            {{}, {}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::f32,
            {{}, {}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{-1.25f}),
            {},
            {},
            ov::element::f32,
            {}
        }
    },

    // group convolution, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::u8,
            {{element::f32}, {}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {{}, {}, {{0.0002f}, ov::element::f32, {}}}
        }
    },

    // per-channel quantization with different values, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.02f, 0.02f, 0.04f, 0.04f, 0.08f, 0.08f}, ov::element::f32, {1, 6, 1, 1}}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{2.f}),
            {},
            {ov::element::f32, {}, {0.01f}}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{2.f}),
            {},
            {},
            ov::element::f32,
            {
                {},
                {},
                {
                    {
                        // 0.0002 = 0.02 (on data) * 0.01 (on weights)
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        // 0.0004 = 0.04 (on data) * 0.01 (on weights)
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        // 0.0008 = 0.08 (on data) * 0.01 (on weights)
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f
                    },
                    ov::element::f32,
                    {1, 24, 1, 1}
                }
            },
        }
    },

    // per-channel quantization with different values, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{255}, ov::element::f32, {}, true, 1, ov::element::u8, true},
                {{0.02f, 0.02f, 0.04f, 0.04f, 0.08f, 0.08f}, ov::element::f32, {1, 6, 1, 1}}
            },
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{2.f}),
            {},
            {
                ov::element::f32,
                {{127}, ov::element::f32, {}, true, 1, ov::element::i8, true},
                {0.01f}
            }
        },
        // ExpectedValues
        {
            ov::element::u8,
            {
                {},
                {std::vector<float>(6ul, 255.f), ov::element::f32, {1, 6, 1, 1}, false, 1, ov::element::u8},
                {}
            },
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{2.f}),
            {},
            {
                {},
                {
                    std::vector<float>(24ul, 127.f),
                    ov::element::f32,
                    {24, 1, 1, 1},
                    false,
                    1,
                    ov::element::i8,
                    false,
                    {{ov::pass::DisableConstantFolding::get_type_info_static(), ov::pass::DisableConstantFolding()}}
                },
                {}
            },
            ov::element::f32,
            {
                {},
                {},
                {
                    {
                        // 0.0002 = 0.02 (on data) * 0.01 (on weights)
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        // 0.0004 = 0.04 (on data) * 0.01 (on weights)
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        // 0.0008 = 0.08 (on data) * 0.01 (on weights)
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f
                    },
                    ov::element::f32,
                    {1, 24, 1, 1}
                }
            },
        }
    },

    // per-channel quantization with different values, without zero point, no reshape - 5D weights
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        false,
        // ActualValues
        {
            ov::element::u8,
            {
                {ov::element::f32},
                {{255}, ov::element::f32, {}, true, 1, ov::element::u8, true},
                {{0.02f, 0.02f, 0.04f, 0.04f, 0.08f, 0.08f}, ov::element::f32, {1, 6, 1, 1}}
            },
            op::v0::Constant::create(ov::element::i8, ov::Shape{3,8,2,7,7}, std::vector<float>{2.f}),
            {},
            {
                ov::element::f32,
                {{127}, ov::element::f32, {}, true, 1, ov::element::i8, true},
                {0.01f}
            }
        },
        // ExpectedValues
        {
            ov::element::u8,
            {
                {},
                {std::vector<float>(6ul, 255.f), ov::element::f32, {1, 6, 1, 1}, false, 1, ov::element::u8},
                {}
            },
            op::v0::Constant::create(ov::element::i8, ov::Shape{3,8,2,7,7}, std::vector<float>{2.f}),
            {},
            {
                {},
                {
                    std::vector<float>(24ul, 127.f),
                    ov::element::f32,
                    {3, 8, 1, 1, 1},
                    false,
                    1,
                    ov::element::i8,
                    false,
                    {{ov::pass::DisableConstantFolding::get_type_info_static(), ov::pass::DisableConstantFolding()}}
                },
                {}
            },
            ov::element::f32,
            {
                {},
                {},
                {
                    {
                        // 0.0002 = 0.02 (on data) * 0.01 (on weights)
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        0.0002f,
                        // 0.0004 = 0.04 (on data) * 0.01 (on weights)
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        0.0004f,
                        // 0.0008 = 0.08 (on data) * 0.01 (on weights)
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f,
                        0.0008f
                    },
                    ov::element::f32,
                    {1, 24, 1, 1}
                }
            },
        }
    },
    // without reshape, per-channel weights quantization
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
         3ul,
         -1,
         false,
         // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({3, 8, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {{}, {}, {{0.0002f}, ov::element::f32, {}}}
        }
    },
    // without reshape, per-channel weights quantization
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true),
         3ul,
         -1,
         false,
         // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {0.1f}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {
                {ov::element::f32},
                {},
                {
                    {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
                     0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
                     0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f},
                    ov::element::f32, {3, 8, 1, 1, 1}
                }
            }
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {{}, {}, {
                {0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f,
                 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f,
                 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f, 0.08f},
                 ov::element::f32, {1, 24, 1, 1}
            }}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         GroupConvolutionTransformation,
                         ::testing::Combine(::testing::ValuesIn(shapesForGroupConv),
                                            ::testing::ValuesIn(testValuesGroupConv)),
                         GroupConvolutionTransformation::getTestCaseName);
}  // namespace testValues1

namespace testValues2 {
const std::vector<std::pair<ov::PartialShape, ov::PartialShape>> shapesForDepthWiseConv = {
    {{1, 6, 224, 224}, {1, 6, 218, 218}},
    {{-1, 6, -1, -1}, {-1, 6, -1, -1}},
};

const std::vector<GroupConvolutionTestValues> testValuesForDepthWiseConv = {
    // depth-wise convolution, per-tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{}, {{128.f}, ov::element::f32, {1, 6, 1, 1}, false}, {}},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {{}, {}, {{0.0002f}, ov::element::f32, {}}}
        }
    },

    // depth-wise convolution, tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        3ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::f32,
            {{}, {128.f}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::f32,
            {{}, {{128.f}, ov::element::f32, {1, 6, 1, 1}, false}, {}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {{}, {}, {{0.0002f}, ov::element::f32, {}}}
        }
    },

    // depth-wise convolution, per-channel quantization with different values, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        6ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.02f, 0.02f, 0.04f, 0.04f, 0.08f, 0.08f}, ov::element::f32, {1, 6, 1, 1}}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {
                {},
                {},
                {
                    {
                       0.0002f,
                       0.0002f,  // 0.0002 = 0.02 (on data) * 0.01 (on weights)
                       0.0004f,
                       0.0004f,  // 0.0004 = 0.04 (on data) * 0.01 (on weights)
                       0.0008f,
                       0.0008f  // 0.0008 = 0.08 (on data) * 0.01 (on weights)
                    },
                    ov::element::f32,
                    {1, 6, 1, 1}
                }
            },
        }
    },

    // depth-wise convolution, per-tensor quantization with the same values, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        6ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {}, {{0.02f}, ov::element::f32, {1, 6, 1, 1}}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {{}, {}, {{0.0002f}, ov::element::f32, {}}},
        }
    },

    // depth-wise convolution, without zero point, without convert
    {
        LayerTransformation::createParamsU8I8(),
        6ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::f32,
            {{}, {}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::f32,
            {{}, {}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{-1.25f}),
            {},
            {},
            ov::element::f32,
            {}
        }
    },

    // depth-wise convolution, without zero point
    {
        LayerTransformation::createParamsU8I8(),
        6ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::u8,
            {{element::f32}, {}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {},
            op::v0::Constant::create(ov::element::i8, ov::Shape{}, std::vector<float>{-125.f}),
            {},
            {},
            ov::element::f32,
            {{}, {}, {{0.0002f}, ov::element::f32, {}}}
        }
    },

    // without dequantization operations
    {
        LayerTransformation::createParamsU8I8(),
        6ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::f32,
            {},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::f32,
            {},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {},
            ov::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         GroupConvolutionTransformation,
                         ::testing::Combine(::testing::ValuesIn(shapesForDepthWiseConv),
                                            ::testing::ValuesIn(testValuesForDepthWiseConv)),
                         GroupConvolutionTransformation::getTestCaseName);
}  // namespace testValues2

namespace testValues3 {
const std::vector<std::pair<ov::PartialShape, ov::PartialShape>> shapesWithDynamicChannel = {
    {PartialShape::dynamic(), PartialShape::dynamic()}
};

const std::vector<GroupConvolutionTestValues> testValuesWithDynamicChannel = {
    // depth-wise convolution, per-tensor quantization, with zero point
    {
        LayerTransformation::createParamsU8I8(),
        3ul,
        -1,
        true,
        // ActualValues
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {}
        },
        // ExpectedValues
        {
            ov::element::u8,
            {{ov::element::f32}, {128.f}, {0.02f}},
            op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{2.f}),
            {255ul, Shape({1, 1, 1, 1}), {0.f}, {254.f}, {-1.27f}, {1.27f}},
            {},
            ov::element::f32,
            {}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         GroupConvolutionTransformation,
                         ::testing::Combine(::testing::ValuesIn(shapesWithDynamicChannel),
                                            ::testing::ValuesIn(testValuesWithDynamicChannel)),
                         GroupConvolutionTransformation::getTestCaseName);
}  // namespace testValues3
// clang-format on

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>
#include <gtest/gtest.h>
#include <utility>

#include "transformations/utils/utils.hpp"
#include "transformations/init_node_info.hpp"
#include "low_precision/multiply.hpp"
#include "low_precision/multiply_to_group_convolution.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ov_lpt_models/multiply.hpp"

namespace {
using namespace testing;
using namespace ov;
using namespace ov::pass;
using namespace ov::builder::subgraph;

class MultiplyBranch {
public:
    ov::builder::subgraph::Constant constant;
    ov::element::Type input_precision;
    ov::builder::subgraph::DequantizationOperations dequantization;
    ov::builder::subgraph::FakeQuantizeOnData fake_quantize;
};

inline std::ostream& operator<<(std::ostream& out, const MultiplyBranch& branch) {
    if (branch.input_precision != element::dynamic) {
        out << "_input=" << branch.input_precision;
    }
    if (!branch.constant.empty()) {
        out << "_constant=" << branch.constant;
    }
    if (!branch.dequantization.empty()) {
        out << "_dequantization=" << branch.dequantization;
    }
    if (!branch.fake_quantize.empty()) {
        out << "_fake_quantize=" << branch.constant;
    }
    return out;
}

class MultiplyValues {
public:
    MultiplyBranch branch1;
    MultiplyBranch branch2;
    ov::builder::subgraph::DequantizationOperations after_dequantization;
};

inline std::ostream& operator<<(std::ostream& out, const MultiplyValues& values) {
    return out << "_branch1=" << values.branch1 << "_branch2=" << values.branch2 << "_after=" << values.after_dequantization;
}

class MultiplyTransformationTestValues {
public:
    // use this value in test case declaration to set precision as input precision
    static const ov::element::Type input_precision;

    // use this value in test case declaration to set precision as model precision
    static const ov::element::Type model_precision;

    TestTransformationParams transformationParams;
    MultiplyValues actual;
    MultiplyValues expected;

    MultiplyTransformationTestValues() = default;

    MultiplyTransformationTestValues(
        TestTransformationParams transformationParams,
        MultiplyValues actual,
        MultiplyValues expected):
        transformationParams(std::move(transformationParams)),
        actual(std::move(actual)),
        expected(std::move(expected)) {}
};

const ov::element::Type MultiplyTransformationTestValues::input_precision = ov::element::dynamic;
const ov::element::Type MultiplyTransformationTestValues::model_precision = ov::element::dynamic;

typedef std::tuple<
    ov::element::Type, // model precision
    std::pair<PartialShape, PartialShape>, // input_shapes
    std::pair<ov::element::Type, ov::element::Type>, // input precisions
    MultiplyTransformationTestValues> MultiplyTransformationParams;

class MultiplyTransformation : public LayerTransformation, public testing::WithParamInterface<MultiplyTransformationParams> {
public:
    void SetUp() override {
        const auto model_precision = std::get<0>(GetParam());
        const auto input_shapes = std::get<1>(GetParam());
        const auto input_precisions = std::get<2>(GetParam());
        MultiplyTransformationTestValues testParams = std::get<3>(GetParam());

        update_input_precisions(input_precisions, testParams);
        update_dequantization_precision(model_precision, testParams);

        // output precision has to be defined by model precision
        if (testParams.expected.after_dequantization.multiply.outPrecision == MultiplyTransformationTestValues::model_precision) {
            testParams.expected.after_dequantization.multiply.outPrecision = model_precision;
        }

        const auto to_multiply_values = [&input_shapes, &input_precisions](const MultiplyValues& values) {
            return ov::builder::subgraph::MultiplyValues(
                ov::builder::subgraph::MultiplyBranch(
                    input_shapes.first, values.branch1.constant, input_precisions.first, values.branch1.dequantization, values.branch1.fake_quantize),
                ov::builder::subgraph::MultiplyBranch(
                    input_shapes.second, values.branch2.constant, input_precisions.second, values.branch2.dequantization, values.branch2.fake_quantize),
                ov::builder::subgraph::DequantizationOperations(values.after_dequantization));
        };

        actualFunction = MultiplyFunction::get(model_precision, to_multiply_values(testParams.actual));

        SimpleLowPrecisionTransformer transform({}, {}, AttributeParameters(), true);
        transform.add<ov::pass::low_precision::MultiplyTransformation, ov::op::v1::Multiply>(testParams.transformationParams);
        transform.cleanup->get_pass_config()->disable<ov::pass::low_precision::MultiplyToGroupConvolutionTransformation>();
        transform.transform(actualFunction);

        referenceFunction = MultiplyFunction::get(model_precision, to_multiply_values(testParams.expected));
    }

    static std::string getTestCaseName(testing::TestParamInfo<MultiplyTransformationParams> obj) {
        const auto model_precision = std::get<0>(obj.param);
        const auto input_shapes = std::get<1>(obj.param);
        const auto input_precisions = std::get<2>(obj.param);
        MultiplyTransformationTestValues testParams = std::get<3>(obj.param);

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(model_precision, input_shapes.first, testParams.transformationParams) <<
            "_SH1=" << input_shapes.first <<
            "_TY1=" << input_precisions.first <<
            "_SH2=" << input_shapes.second <<
            "_TY2=" << input_precisions.second;

        update_input_precisions(input_precisions, testParams);
        update_dequantization_precision(model_precision, testParams);

        result << testParams.actual << testParams.expected;
        return result.str();
    }

private:
    // dequantization output precision has to be defined by input precision
    static void update_dequantization_precision(const ov::element::Type& dequantization_precision,
                                                MultiplyTransformationTestValues& test_values) {
        if (!test_values.actual.after_dequantization.multiply.empty() &&
            test_values.actual.after_dequantization.multiply.outPrecision == MultiplyTransformationTestValues::input_precision) {
            test_values.actual.after_dequantization.multiply.outPrecision = dequantization_precision;
        }

        if (!test_values.expected.after_dequantization.multiply.empty() &&
            test_values.expected.after_dequantization.multiply.outPrecision == MultiplyTransformationTestValues::input_precision) {
            test_values.expected.after_dequantization.multiply.outPrecision = dequantization_precision;
        }
    }

    // low precision has to be defined by tests parameters
    static void update_input_precisions(const std::pair<ov::element::Type, ov::element::Type>& input_precisions,
                                        MultiplyTransformationTestValues& test_values) {
        const auto update_values = [](const std::pair<ov::element::Type, ov::element::Type>& input_precisions, MultiplyValues& values) {
            const auto update_branch = [](const ov::element::Type& input_precision, MultiplyBranch& branch) {
                if (branch.input_precision == MultiplyTransformationTestValues::input_precision) {
                    branch.input_precision = input_precision;
                }

                if (!branch.constant.empty() &&
                    (branch.constant.outPrecision == MultiplyTransformationTestValues::input_precision)) {
                    branch.constant.outPrecision = input_precision;
                }
            };

            update_branch(input_precisions.first, values.branch1);
            update_branch(input_precisions.second, values.branch2);
        };

        update_values(input_precisions, test_values.actual);
        update_values(input_precisions, test_values.expected);
    }
};

TEST_P(MultiplyTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<ov::element::Type> model_precisions = {
    ov::element::f32,
    ov::element::f16
};

const std::vector<std::pair<PartialShape, PartialShape>> input_shapes = {
        {{ 1, 3, 8, 16 }, { 1, 3, 8, 16 }},
        {{ 1, 3, 8, 16 }, { 1, 3, 1, 1 }},
        {{ 1, 3, 1, 1 }, { 1, 3, 8, 16 }},
        {
            { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
            { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() }
        },
        {
            { Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic() },
            { Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic() }
        }
};

namespace multiply_channel_fq {
    const std::vector<std::pair<ov::element::Type, ov::element::Type>> input_precisions = {
        { ov::element::u8, ov::element::f32 },
        { ov::element::u8, ov::element::f16 },
        { ov::element::i8, ov::element::f32 },
        { ov::element::i8, ov::element::f16 }
    };

    const std::vector<MultiplyTransformationTestValues> multiplyTransformationTestValues = {
        {
            LayerTransformation::createParamsU8I8(),
            {
                {
                    {},
                    MultiplyTransformationTestValues::input_precision,
                    {ov::element::f32, { 2.f }, { 10.f }}
                },
                {
                    {{ 0.f, 1.27f, 2.55f }, MultiplyTransformationTestValues::input_precision, ov::Shape{1, 3, 1, 1}}, // Constant as input,
                    {},
                    {},
                    {
                        256,
                        ov::Shape{1, 3, 1, 1},
                        {0.f, 0.f, 0.f},
                        {2.55f, 2.55f, 2.55f},
                        {0.f, 0.f, 0.f},
                        {2.55f, 2.55f, 2.55f},
                        MultiplyTransformationTestValues::input_precision
                    } // FakeQuantize
                },
            },
            {
                {
                    {},
                    MultiplyTransformationTestValues::input_precision,
                    {{}, {{2.f}, ov::element::f32}, {}}
                },
                {
                    {{ 0, 127, 255 }, ov::element::u8, ov::Shape{1, 3, 1, 1}}, // Constant as input,
                    {},
                    {}
                },
                {{}, {}, {{0.1f, 0.1f, 0.1f}}}
            },
        },
    };

    INSTANTIATE_TEST_SUITE_P(
        smoke_LPT,
        MultiplyTransformation,
        ::testing::Combine(
            ::testing::ValuesIn(model_precisions),
            ::testing::ValuesIn(input_shapes),
            ::testing::ValuesIn(input_precisions),
            ::testing::ValuesIn(multiplyTransformationTestValues)),
        MultiplyTransformation::getTestCaseName);
} // namespace multiply_channel_fq

const std::vector<std::pair<ov::element::Type, ov::element::Type>> input_precisions = {
    { ov::element::u8, ov::element::u8 },
    { ov::element::i8, ov::element::i8 },
    { ov::element::u8, ov::element::i8 },
    { ov::element::i8, ov::element::u8 },
    { ov::element::f32, ov::element::f32 },
    { ov::element::f16, ov::element::f16 },
};

namespace multiply_channel {
const std::vector<MultiplyTransformationTestValues> multiplyTransformationTestValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 3.f }, { 7.f }}
            },
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{2.f}, ov::element::f32}, {}}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{3.f}, ov::element::f32}, {}}
            },
            {{}, {}, {{70.f}, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                {{ 7.f, 8.f, 9.f }, MultiplyTransformationTestValues::input_precision, ov::Shape{1, 3, 1, 1}}, // Constant as input,
                {},
                {ov::element::f32, { 3.f }, { 7.f }}
            },
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{2.f}, ov::element::f32}, {}}
            },
            {
                {{ 7.f, 8.f, 9.f }, MultiplyTransformationTestValues::input_precision, ov::Shape{1, 3, 1, 1}}, // Constant as input,
                {},
                {{}, {{3.f}, ov::element::f32}, {}}
            },
            {{}, {}, {{70.f}, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {{ 7.f, 8.f, 9.f }, MultiplyTransformationTestValues::input_precision, ov::Shape{1, 3, 1, 1}}, // Constant as input,
                {},
                {ov::element::f32, { 3.f }, { 7.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 2.f }, { 10.f }}
            }
        },
        {
            {
                {{ 7.f, 8.f, 9.f }, MultiplyTransformationTestValues::input_precision, ov::Shape{1, 3, 1, 1}}, // Constant as input,
                {},
                {{}, {{3.f}, ov::element::f32}, {}}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{2.f}, ov::element::f32}, {}}
            },
            {{}, {}, {{70.f}, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {}, { 10.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {}, { 7.f }}
            }
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {}
            },
            {{}, {}, {{70.f}, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {{ 1.f, 2.f, 3.f }}, {{ 10.f, 11.f, 12.f }}}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {{ 3.f, 4.f, 5.f }}, {{ 7.f, 8.f, 9.f }}}
            }
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{1.f, 2.f, 3.f}, ov::element::f32}, {}}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{3.f, 4.f, 5.f }, ov::element::f32}, {}}
            },
            {{}, {}, {{70.f, 88.f, 108.f}, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { }, { 7.f }}
            }
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{2.f}, ov::element::f32}, {}}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {}
            },
            {{}, {}, {{70.f}, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {}, { 10.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 3.f }, { 7.f }}
            }
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{3.f}, ov::element::f32}, {}}
            },
            {{}, {}, {{70.f}, MultiplyTransformationTestValues::model_precision}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MultiplyTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(model_precisions),
        ::testing::ValuesIn(input_shapes),
        ::testing::ValuesIn(input_precisions),
        ::testing::ValuesIn(multiplyTransformationTestValues)),
    MultiplyTransformation::getTestCaseName);
} // namespace multiply_channel

namespace broadcast_right {
const std::vector<std::pair<PartialShape, PartialShape>> input_shapes = {
    {{ 1, 3, 8, 16 }, { 1, 1, 1, 1 }}
};

const std::vector<MultiplyTransformationTestValues> multiplyTransformationTestValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 3.f }, { 7.f }}
            },
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{ 2.f }, ov::element::f32}, {}}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{ 3.f }, ov::element::f32}, {}}
            },
            {{}, {}, {{ 70.f }, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {}, { 10.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {}, { 7.f }}
            }
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {}
            },
            {{}, {}, {{ 70.f }, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {{ 1.f, 2.f, 3.f }}, {{ 10.f, 11.f, 12.f }}}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 3.f }, { 7.f }}
            }
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{ 1.f, 2.f, 3.f }, ov::element::f32}, {}}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{ 3.f }, ov::element::f32}, {}}
            },
            {{}, {}, {{70.f, 77.f, 84.f}, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {}, { 7.f }}
            }
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{2.f}, ov::element::f32}, {}}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {}
            },
            {{}, {}, {{70.f}, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {}, { 10.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 3.f }, { 7.f }}
            }
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{3.f}, ov::element::f32}, {}}
            },
            {{}, {}, {{70.f}, MultiplyTransformationTestValues::model_precision}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MultiplyTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(model_precisions),
        ::testing::ValuesIn(input_shapes),
        ::testing::ValuesIn(input_precisions),
        ::testing::ValuesIn(multiplyTransformationTestValues)),
    MultiplyTransformation::getTestCaseName);
} // namespace broadcast_right

namespace broadcast_left {
const std::vector<std::pair<PartialShape, PartialShape>> input_shapes = {
    {{ 1, 1, 1, 1 }, { 1, 3, 8, 16 }}
};

const std::vector<MultiplyTransformationTestValues> multiplyTransformationTestValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 3.f }, { 7.f }}
            },
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{ 2.f }, ov::element::f32}, {}}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{ 3.f }, ov::element::f32}, {}}
            },
            {{}, {}, {{ 70.f }, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {}, { 10.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {}, { 7.f }}
            }
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {}
            },
            {{}, {}, {{ 70.f }, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {{ 3.f, 4.f, 5.f }}, {{ 7.f, 8.f, 9.f }}}
            }
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{ 2.f }, ov::element::f32}, {}}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{ 3.f, 4.f, 5.f }, ov::element::f32}, {}}
            },
            {{}, {}, {{70.f, 80.f, 90.f}, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 2.f }, { 10.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {}, { 7.f }}
            }
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{2.f}, ov::element::f32}, {}}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {}
            },
            {{}, {}, {{70.f}, MultiplyTransformationTestValues::model_precision}}
        }
    },

    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, {}, { 10.f }}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {ov::element::f32, { 3.f }, { 7.f }}
            }
        },
        {
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {}
            },
            {
                {},
                MultiplyTransformationTestValues::input_precision,
                {{}, {{3.f}, ov::element::f32}, {}}
            },
            {{}, {}, {{70.f}, MultiplyTransformationTestValues::model_precision}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MultiplyTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(model_precisions),
        ::testing::ValuesIn(input_shapes),
        ::testing::ValuesIn(input_precisions),
        ::testing::ValuesIn(multiplyTransformationTestValues)),
    MultiplyTransformation::getTestCaseName);
} // namespace broadcast_left

} // namespace

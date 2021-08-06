// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <low_precision/relu.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/relu_function.hpp"
#include "simple_low_precision_transformer.hpp"

namespace {

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

class ReluTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    TestTransformationParams params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::PartialShape,
    ReluTransformationTestValues> ReluTransformationParams;

class ReluTransformation : public LayerTransformation, public testing::WithParamInterface<ReluTransformationParams> {
public:
    void SetUp() override {
        const auto inputShape = std::get<0>(GetParam());
        const auto testValues = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::ReluFunction::getOriginal(
            inputShape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::ReluTransformation, ngraph::opset1::Relu>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ReluFunction::getReference(
            inputShape,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ReluTransformationParams> obj) {
        const auto inputShape = std::get<0>(obj.param);
        const auto testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
            toString(testValues.params) << "_" <<
            inputShape << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore;
        return result.str();
    }

protected:
    std::shared_ptr<ngraph::Function> actualFunction;
    std::shared_ptr<ngraph::Function> referenceFunction;
};

TEST_P(ReluTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, false);
    ASSERT_TRUE(res.first) << res.second;
}

namespace testValues1 {
const std::vector<ngraph::PartialShape> shapes = {
    { 1, 3, 16, 16 },
    { Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic() },
};

const std::vector<ReluTransformationTestValues> testValues = {
    // U8: no subtract
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        }
    },
    // U8: no subtract
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}}}
        }
    },
    // U8: no subtract
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, -0.2f, 0.3f}}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, -0.2f, 0.3f}}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // I8: no subtract
    {
        LayerTransformation::createParamsI8I8(),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {0.1f}}
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::i8,
            {{ngraph::element::f32}, {}, {0.1f}}
        }
    },
    // U8: with subtract value
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128 }, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, { 128 }, {0.1f}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // I8: with subtract value
    {
        LayerTransformation::createParamsI8I8().setSupportAsymmetricQuantization(true),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, { 127 }, {0.1f}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, { 127 }, {0.1f}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // I8: with subtract value
    {
        LayerTransformation::createParamsI8I8().setSupportAsymmetricQuantization(false),
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, { 127 }, {0.1f}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, { 127 }, {0.1f}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
    // U8: empty
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {}
        }
    },
    // FP32: empty
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::f32,
            {}
        },
        {
            ngraph::element::f32,
            {},
            ngraph::element::f32,
            {}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ReluTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ReluTransformation::getTestCaseName);
} // namespace testValues1

namespace testValues2 {
const std::vector<ngraph::PartialShape> shapesWithDynamicChannels = {
    { Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic() },
};

const std::vector<ReluTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {{0.1f, 0.2f, 0.3f}}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ReluTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapesWithDynamicChannels),
        ::testing::ValuesIn(testValues)),
    ReluTransformation::getTestCaseName);
}// namespace testValues2

namespace testValues3 {
const std::vector<ngraph::PartialShape> shapesWithDynamicRank = {
    PartialShape::dynamic()
};

const std::vector<ReluTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {}, {0.1f}},
            ngraph::element::f32,
            {{}, {}, {}}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ReluTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapesWithDynamicRank),
        ::testing::ValuesIn(testValues)),
    ReluTransformation::getTestCaseName);
} // namespace testValues3
} // namespace

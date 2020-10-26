// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include "low_precision/interpolate.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/interpolate_function.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph::builder::subgraph;

class interpAttributes {
public:
    ngraph::AxisSet axes;
    std::string mode;
    bool align_corners;
    bool antialias;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;

    interpAttributes(const ngraph::AxisSet& axes,
        const std::string& mode,
        const bool& align_corners,
        const bool& antialias,
        const std::vector<size_t>& pads_begin,
        const std::vector<size_t>& pads_end) :
        axes(axes), mode(mode), align_corners(align_corners),
        antialias(antialias), pads_begin(pads_begin), pads_end(pads_end) {}
};

class InterpolateTransformationTestValues {
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

    ngraph::Shape inputShape;
    ngraph::Shape outputShape;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    //ngraph::op::InterpolateAttrs interpAttrs;
    interpAttributes interpAttrs;
    Actual actual;
    Expected expected;
};

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const std::vector<T>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

class InterpolateTransformation : public LayerTransformation, public testing::WithParamInterface<InterpolateTransformationTestValues> {
public:
    void SetUp() override {
        const InterpolateTransformationTestValues testValues = GetParam();

        ngraph::op::InterpolateAttrs interpAttrs;
        interpAttrs.axes = testValues.interpAttrs.axes;
        interpAttrs.mode = testValues.interpAttrs.mode;
        interpAttrs.align_corners = testValues.interpAttrs.align_corners;
        interpAttrs.antialias = testValues.interpAttrs.antialias;
        interpAttrs.pads_begin = testValues.interpAttrs.pads_begin;
        interpAttrs.pads_end = testValues.interpAttrs.pads_end;

        actualFunction = ngraph::builder::subgraph::InterpolateFunction::getOriginal(
            testValues.inputShape,
            testValues.outputShape,
            interpAttrs,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transformer;
        transformer.add<ngraph::pass::low_precision::InterpolateTransformation, ngraph::opset1::Interpolate>(testValues.params);
        transformer.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::InterpolateFunction::getReference(
            testValues.inputShape,
            testValues.outputShape,
            interpAttrs,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<InterpolateTransformationTestValues> obj) {
        const InterpolateTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.inputShape << "_" <<
            testValues.outputShape << "_" <<
            testValues.interpAttrs.align_corners <<
            testValues.interpAttrs.antialias <<
            testValues.interpAttrs.axes <<
            testValues.interpAttrs.mode <<
            testValues.interpAttrs.pads_begin <<
            testValues.interpAttrs.pads_end <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore;
        return result.str();
    }
};

const std::vector<InterpolateTransformationTestValues> testValues {
    // nearest mode - move dequantization
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        ngraph::Shape{ 1, 4, 32, 32 },
        LayerTransformation::createParamsU8I8(),
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "nearest",
            false,
            false,
            {0},
            {0}),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        }
    },

    // mode is not nearest - not transformed
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        ngraph::Shape{ 1, 4, 32, 32 },
        LayerTransformation::createParamsU8I8(),
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "linear",
            false,
            false,
            {0},
            {0}),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}},
            ngraph::element::u8,
            {{}, {}, {}}
        }
    },

    // AxisSet is not {2,3} - not transformed
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        ngraph::Shape{ 1, 8, 32, 32 },
        LayerTransformation::createParamsU8I8(),
        interpAttributes(
            ngraph::AxisSet{1, 2, 3},
            "nearest",
            false,
            false,
            {0},
            {0}),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}},
            ngraph::element::u8,
            {{}, {}, {}}
        }
    },

    // align_corners set to true - not transformed
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        ngraph::Shape{ 1, 4, 32, 32 },
        LayerTransformation::createParamsU8I8(),
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "nearest",
            true,
            false,
            {0},
            {0}),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}},
            ngraph::element::u8,
            {{}, {}, {}}
        }
    },

    // have pads - not transformed
    {
        ngraph::Shape{ 1, 4, 16, 16 },
        ngraph::Shape{ 1, 4, 32, 32 },
        LayerTransformation::createParamsU8I8(),
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "nearest",
            false,
            false,
            {1},
            {1}),
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}},
            ngraph::element::u8,
            {{}, {}, {}}
        }
    }
};

TEST_P(InterpolateTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_CASE_P(
    LPT,
    InterpolateTransformation,
    ::testing::ValuesIn(testValues),
    InterpolateTransformation::getTestCaseName);

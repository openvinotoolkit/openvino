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
#include <low_precision/transformer.hpp>
#include <low_precision/concat.hpp>
#include <low_precision/concat_multi_channels.hpp>
#include <low_precision/max_pool.hpp>
#include <low_precision/reshape.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/concat_function.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

namespace {

class ConcatTransformationActualValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize3;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationActualValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2;
}

class ConcatTransformationResultValues {
public:
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize1;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize2;
    ngraph::builder::subgraph::FakeQuantizeOnDataWithConstant fakeQuantize3;
    ngraph::element::Type precisionBeforeOp;
    ngraph::element::Type precisionAfterOp;
    ngraph::builder::subgraph::DequantizationOperations dequantizationOperations;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationResultValues& values) {
    return out << "_" << values.fakeQuantize1 << "_" << values.fakeQuantize2 << "_" << values.fakeQuantize3 << "_" << values.dequantizationOperations;
}

class ConcatTransformationTestValues {
public:
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ConcatTransformationActualValues actual;
    ConcatTransformationResultValues result;
};

inline std::ostream& operator<<(std::ostream& out, const ConcatTransformationTestValues& values) {
    return out << "_" << values.actual << "_" << values.result;
}

typedef std::tuple <
    ngraph::element::Type,
    ngraph::Shape,
    ConcatTransformationTestValues
> ConcatTransformationParams;

class ConcatWithReshapeAtTheEndTransformation : public LayerTransformation, public testing::WithParamInterface<ConcatTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        ConcatTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::ConcatFunction::getOriginalWithReshapeAtTheEndTransformation(
            precision,
            shape,
            testValues.actual.fakeQuantize1,
            testValues.actual.fakeQuantize2,
            testValues.actual.fakeQuantize3);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::ConcatMultiChannelsTransformation, ngraph::opset1::Concat>(testValues.params);
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(testValues.params);
        transform.add<ngraph::pass::low_precision::ReshapeTransformation, ngraph::opset1::Reshape>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::ConcatFunction::getReferenceWithReshapeAtTheEndTransformation(
            precision,
            shape,
            testValues.result.fakeQuantize1,
            testValues.result.fakeQuantize2,
            testValues.result.fakeQuantize3,
            testValues.result.precisionBeforeOp,
            testValues.result.precisionAfterOp,
            testValues.result.dequantizationOperations);
    }

    static std::string getTestCaseName(testing::TestParamInfo<ConcatTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const ConcatTransformationTestValues testValues = std::get<2>(obj.param);

        std::ostringstream result;
        result <<
            LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params) << "_" <<
            testValues.actual << "_" <<
            testValues.result << "_";
        return result.str();
    }
};

TEST_P(ConcatWithReshapeAtTheEndTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
};

const std::vector<ConcatTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
        },
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            ngraph::element::u8,
            ngraph::element::u8,
            { ngraph::element::f32, {}, { 0.01f } }
        }
    },
    {
        LayerTransformation::createParamsU8I8().setUpdatePrecisions(false),
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {2.55f} },
        },
        {
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            { 256ul, {}, {0.f}, {2.55f}, {0.f}, {255.f} },
            ngraph::element::f32,
            ngraph::element::f32,
            { {}, {}, { 0.01f } }
        }
    },
    {
        LayerTransformation::createParamsU8I8(),
        {
            {
                256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
                {0.f, 0.f, 0.f}, {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f}, {0.f, 0.f, 0.f}, {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f}
            },
            {
                256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
                {0.f, 0.f, 0.f}, {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f}, {0.f, 0.f, 0.f}, {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f}
            },
            {
                256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}, {1, 3, 1, 1}},
                {0.f, 0.f, 0.f}, {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f}, {0.f, 0.f, 0.f}, {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f}
            },
        },
        {
            {
                256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
                {0.f, 0.f, 0.f}, {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f}, {0.f}, {255.f}
            },
            {
                256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
                {0.f, 0.f, 0.f}, {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f}, {0.f}, {255.f}
            },
            {
                256ul,
                {{1, 3, 1, 1}, {1, 3, 1, 1}, {}, {}},
                {0.f, 0.f, 0.f}, {2.55f / 1.f, 2.55f / 2.f, 2.55f / 3.f}, {0.f}, {255.f}
            },
            ngraph::element::u8,
            ngraph::element::u8,
            { ngraph::element::f32, {}, {{ 0.01f, 0.01f / 2.f, 0.01f / 3.f, 0.01f, 0.01f / 2.f, 0.01f / 3.f, 0.01f, 0.01f / 2.f, 0.01f / 3.f }} }
        }
    }
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 3, 9, 9 },
    { 4, 3, 9, 9 }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    ConcatWithReshapeAtTheEndTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    ConcatWithReshapeAtTheEndTransformation::getTestCaseName);
}  // namespace

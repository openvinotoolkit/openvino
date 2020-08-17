// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/unsqueeze.hpp>
#include <transformations/low_precision/transformer.hpp>
#include <transformations/low_precision/network_helper.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/unsqueeze_function.hpp"

using namespace testing;
using namespace ngraph::pass;

using ngraph::builder::subgraph::UnsqueezeFunction;


inline std::ostream& operator<<(std::ostream& os, const std::vector<float>& values) {
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

class UnsqueezeTransformationTestValues {
public:
    low_precision::LayerTransformation::Params params;
    UnsqueezeFunction::LayerDescription multiplyActual;
    UnsqueezeFunction::LayerDescription multiplyExpected;
    UnsqueezeFunction::LayerDescription subtractActual;
    UnsqueezeFunction::LayerDescription subtractExpected;
    std::vector<float> unsqueezeArgs;
    ngraph::Shape inputShape;
    ngraph::element::Type precision;
};

class UnsqueezeTransformation : public LayerTransformation, public testing::WithParamInterface<UnsqueezeTransformationTestValues> {
public:
    void SetUp() override {
        const UnsqueezeTransformationTestValues testValues = GetParam();
        const ngraph::element::Type precision = testValues.precision;

        actualFunction = ngraph::builder::subgraph::UnsqueezeFunction::getOriginal(
            precision,
            testValues.inputShape,
            testValues.unsqueezeArgs,
            {
                testValues.params.updatePrecisions ? testValues.params.precisionsOnActivations[0] : precision,
                testValues.multiplyActual,
                testValues.subtractActual
            });

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::UnsqueezeTransformation, ngraph::opset1::Unsqueeze>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::UnsqueezeFunction::getReference(
            precision,
            testValues.inputShape,
            testValues.unsqueezeArgs,
            {
                testValues.params.updatePrecisions ? testValues.params.precisionsOnActivations[0] : precision,
                testValues.multiplyExpected,
                testValues.subtractExpected
            });
    }

    static std::string getTestCaseName(testing::TestParamInfo<UnsqueezeTransformationTestValues> obj) {
        const UnsqueezeTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.multiplyActual.shape << "_" <<
            testValues.multiplyExpected.shape << "_" <<
            testValues.subtractActual.shape << "_" <<
            testValues.subtractExpected.shape << "_" <<
            testValues.inputShape << "_" <<
            testValues.precision << "_" <<
            testValues.unsqueezeArgs;

        return result.str();
    }
};

TEST_P(UnsqueezeTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true);

    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<UnsqueezeTransformationTestValues> testValues = {
    /*
    params;
    multiplyActual(@values, @shape);
    multiplyExpected(@values, @shape);
    subtractActual(@values, @shape);
    subtractExpected(@values, @shape);
    unsqueezeArgs;
    inputShape;
    precision;
    */

    {
        LayerTransformation::createParamsU8I8(),
        { { 0.2f }, { 2, 3 } },
        { { 0.2f }, { 1, 2, 3, 1 } },
        { { 128 }, { 2, 3 } },
        { { 128 }, { 1, 2, 3, 1 } },
        { 0.0, 3.0 },
        { 2, 3 },
        ngraph::element::f32
    },

    {
        LayerTransformation::createParamsU8I8(),
        { { 0.5f }, { 1 } },
        { { 0.5f }, { 1 } },
        { { 32 }, { 3, 32, 32, 32 } },
        { { 32 }, { 3, 1, 32, 32, 32 } },
        { 1.0 },
        { 3, 32, 32, 32 },
        ngraph::element::f32
    },
    {
        LayerTransformation::createParamsI8I8(),
        { { 0.1f }, { 1 } },
        { { 0.1f }, { 1 } },
        { { 256 }, { 1 } },
        { { 256 }, { 1 } },
        { 1.0, 2.0, 4.0 },
        { 3, 32, 32, 32 },
        ngraph::element::f32
    },
    {
        LayerTransformation::createParamsI8I8(),
        { { 0.1f }, { 1 } },
        { { 0.1f }, { 1 } },
        { { 256 }, { 1 } },
        { { 256 }, { 1 } },
        { 0.0, 2.0, 4.0 },
        { 3, 32, 32, 32 },
        ngraph::element::f32
    }
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    UnsqueezeTransformation,
    ::testing::ValuesIn(testValues),
    UnsqueezeTransformation::getTestCaseName);

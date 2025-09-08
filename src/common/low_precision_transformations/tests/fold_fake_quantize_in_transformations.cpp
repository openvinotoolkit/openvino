// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "low_precision/fake_quantize.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "layer_transformation.hpp"
#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/fold_fake_quantize.hpp"
#include "simple_low_precision_transformer.hpp"

using namespace testing;
using namespace ov;
using namespace ov::pass;

class FoldFakeQuantizeInTransformationsTestValues {
public:
    class Actual {
    public:
        std::vector<float> constValues;
        ov::element::Type constPrecision;
        ov::builder::subgraph::FakeQuantizeOnData fakeQuantize;
        ov::element::Type fqOutPrecision;
    };

    class Expected {
    public:
        std::vector<float> constValues;
        ov::element::Type constPrecision;
    };

    ov::Shape constShape;
    TestTransformationParams params;
    bool updatePrecision;
    bool roundValues;
    Actual actual;
    Expected expected;
};

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

class FoldFakeQuantizeInTransformations
    : public LayerTransformation,
      public testing::WithParamInterface<FoldFakeQuantizeInTransformationsTestValues> {
public:
    void SetUp() override {
        const FoldFakeQuantizeInTransformationsTestValues testValues = GetParam();

        const auto params = TestTransformationParams(testValues.params).setUpdatePrecisions(testValues.updatePrecision);

        std::shared_ptr<ov::Node> dataSource;
        std::shared_ptr<ov::op::v0::Parameter> parameter;
        bool useParameterAsDataSource = (testValues.actual.constValues.size() == 0);

        if (useParameterAsDataSource) {
            // for data-independent const folding
            parameter =
                std::make_shared<ov::op::v0::Parameter>(testValues.actual.constPrecision, testValues.constShape);
            dataSource = parameter;
        } else {
            dataSource = std::make_shared<ov::op::v0::Constant>(testValues.actual.constPrecision,
                                                                testValues.constShape,
                                                                testValues.actual.constValues);
        }

        std::shared_ptr<Node> fq =
            ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(dataSource,
                                                                   element::f32,
                                                                   testValues.actual.fakeQuantize);
        ov::pass::low_precision::NetworkHelper::setOutDataPrecision(as_type_ptr<ov::op::v0::FakeQuantize>(fq),
                                                                        testValues.actual.fqOutPrecision);
        fq = ov::pass::low_precision::NetworkHelper::fold_fake_quantize(as_type_ptr<ov::op::v0::FakeQuantize>(fq),
                                                                            testValues.roundValues);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(fq)};
        actualFunction = std::make_shared<ov::Model>(
            results,
            parameter ? ov::ParameterVector{parameter} : ov::ParameterVector{},
            "FoldFakeQuantizeFunction");

        referenceFunction =
            ov::builder::subgraph::FoldFakeQuantizeFunction::getReference(testValues.expected.constPrecision,
                                                                              testValues.constShape,
                                                                              testValues.expected.constValues);
    }

    static std::string getTestCaseName(testing::TestParamInfo<FoldFakeQuantizeInTransformationsTestValues> obj) {
        FoldFakeQuantizeInTransformationsTestValues testValues = obj.param;

        std::ostringstream result;
        result << LayerTransformation::getTestCaseNameByParams(testValues.actual.constPrecision,
                                                               testValues.constShape,
                                                               testValues.params)
               << (testValues.updatePrecision ? "" : "_notUpdatePrecision_") << testValues.actual.fakeQuantize
               << testValues.actual.constValues << "_" << testValues.roundValues;
        return result.str();
    }
};

#ifdef OPENVINO_ARCH_ARM64
// Ticket: 122660
TEST_P(FoldFakeQuantizeInTransformations, DISABLED_CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, false);
    ASSERT_TRUE(res.first) << res.second;
}
#else
TEST_P(FoldFakeQuantizeInTransformations, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, false);
    ASSERT_TRUE(res.first) << res.second;
}
#endif

const std::vector<FoldFakeQuantizeInTransformationsTestValues> testValues = {
    {
        Shape{2, 2, 2, 2},
        LayerTransformation::createParamsU8I8(),
        true,
        true,
        {
            {1, 0, 77, 125, 254, 100, 0, 127, 0, 64, 1, 254, 7, 0, 9, 0},
            ov::element::f32,
            {255ul, {}, {0.f}, {254.f}, {-127.f}, {127.f}},
            ov::element::i8,
        },
        {{-126, -127, -50, -2, 127, -27, -127, 0, -127, -63, -126, 127, -120, -127, -118, -127}, ov::element::i8},
    },
    {
        Shape{2, 2, 2, 2},
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {
            {1, -1, 77, 125, 254, 100, 0, 127, -2, 64, 1, 300, 7, -200, 9, -301},
            ov::element::f32,
            {255ul, {}, {0.f}, {254.f}, {-12.7f}, {12.7f}},
            ov::element::f32,
        },
        {{-12.6f,
          -12.7f,
          -5.0f,
          -0.2f,
          12.7f,
          -2.7f,
          -12.7f,
          0.f,
          -12.7f,
          -6.3f,
          -12.6f,
          12.7f,
          -12.0f,
          -12.7f,
          -11.8f,
          -12.7f},
         ov::element::f32},
    },
    {
        Shape{2, 2, 2, 2},
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {{1, -1, 77, 125, 254, 100, 0, 127, -2, 64, 1, 300, 7, -200, 9, -301},
         ov::element::f32,
         {256ul, {}, {0.f}, {255.f}, {-12.8f}, {12.7f}},
         ov::element::f32},
        {{-12.7f,
          -12.8f,
          -5.1f,
          -0.3f,
          12.6f,
          -2.8f,
          -12.8f,
          -0.1f,
          -12.8f,
          -6.4f,
          -12.7f,
          12.7f,
          -12.1f,
          -12.8f,
          -11.9f,
          -12.8f},
         ov::element::f32},
    },
    {
        Shape{2, 2, 2, 2},
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {{1, 0, 77, 125, 254, 100, 0, 127, 0, 64, 1, 255, 7, 0, 9, 0},
         ov::element::u8,
         {256ul, {}, {0.f}, {255.f}, {-128.f}, {127.f}},
         ov::element::i8},
        {{-127, -128, -51, -3, 126, -28, -128, -1, -128, -64, -127, 127, -121, -128, -119, -128}, ov::element::i8},
    },
    {
        Shape{2, 2, 2, 2},
        LayerTransformation::createParamsU8I8(),
        true,
        false,
        {{
             // use empty constValues for data independent const folding
         },
         ov::element::u8,
         {256ul, {}, {0.f}, {255.f}, {127.f}, {127.f}},
         ov::element::i8},
        {{
             127,
             127,
             127,
             127,
             127,
             127,
             127,
             127,
             127,
             127,
             127,
             127,
             127,
             127,
             127,
             127,
         },
         ov::element::i8},
    },
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         FoldFakeQuantizeInTransformations,
                         ::testing::ValuesIn(testValues),
                         FoldFakeQuantizeInTransformations::getTestCaseName);

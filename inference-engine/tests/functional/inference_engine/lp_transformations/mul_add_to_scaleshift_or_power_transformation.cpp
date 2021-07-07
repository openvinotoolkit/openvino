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
#include <legacy/transformations/convert_opset1_to_legacy/convert_mul_or_add_finally.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_mul_add_to_scaleshift_or_power.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/add.hpp"
#include "lpt_ngraph_functions/mul_add_to_scaleshift_or_power_function.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

namespace {

class MulAddToScaleshiftOrPowerParams {
public:
    ngraph::pass::low_precision::LayerTransformation::Params params;
    ngraph::builder::subgraph::DequantizationOperations::Multiply mulValues;
    ngraph::builder::subgraph::Add addValues;
    ngraph::element::Type precisionAfterOperation;
};

typedef std::tuple <
    ngraph::element::Type,
    bool,
    ngraph::Shape,
    MulAddToScaleshiftOrPowerParams
> MulAddToScaleshiftOrPowerTestValues;

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


class MulAddToScaleshiftOrPowerTransformation :
    public LayerTransformation,
    public testing::WithParamInterface<MulAddToScaleshiftOrPowerTestValues> {
public:
    void SetUp() override {
        const auto inputPrecision = std::get<0>(GetParam());
        const auto isDequantization = std::get<1>(GetParam());
        const auto inputShape = std::get<2>(GetParam());
        const auto testValues = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::MulAddToScaleshiftOrPowerFunction::getOriginal(
            inputPrecision,
            inputShape,
            isDequantization,
            testValues.mulValues,
            testValues.addValues);

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::ConvertMulAddToScaleShiftOrPower>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.run_passes(actualFunction);

        referenceFunction = ngraph::builder::subgraph::MulAddToScaleshiftOrPowerFunction::getReference(
            inputPrecision,
            inputShape,
            isDequantization,
            testValues.mulValues,
            testValues.addValues,
            testValues.precisionAfterOperation);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MulAddToScaleshiftOrPowerTestValues> obj) {
        const auto inputPrecision = std::get<0>(obj.param);
        const auto isDequantization = std::get<1>(obj.param);
        const auto inputShape = std::get<2>(obj.param);
        const auto testValues = std::get<3>(obj.param);

        std::ostringstream result;
        result << toString(testValues.params) << "_" << inputPrecision  << "_" << inputShape << "_"
            << testValues.mulValues.values << "_" << testValues.addValues.values << (isDequantization ? "_ScaleShift_" : "_Power_")
            << testValues.precisionAfterOperation;
        return result.str();
    }
};

TEST_P(MulAddToScaleshiftOrPowerTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precision = {
    ngraph::element::i32,
    ngraph::element::f32,
    ngraph::element::u8,
    ngraph::element::i8,
};

const std::vector<bool> isDequantization = { false, true };

const std::vector<ngraph::Shape> inputShape = {
    { 1, 3, 9, 9 },
    { 4, 3, 9, 9 }
};

const std::vector<MulAddToScaleshiftOrPowerParams> testValues = {
    {
        LayerTransformation::createParamsU8I8(),
        { 0.1f },
        { 128.f },
        ngraph::element::f32
    },
    {
        LayerTransformation::createParamsU8I8(),
        { 0.1f },
        { -128.f },
        ngraph::element::f32
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    MulAddToScaleshiftOrPowerTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precision),
        ::testing::ValuesIn(isDequantization),
        ::testing::ValuesIn(inputShape),
        ::testing::ValuesIn(testValues)),
    MulAddToScaleshiftOrPowerTransformation::getTestCaseName);
}  // namespace

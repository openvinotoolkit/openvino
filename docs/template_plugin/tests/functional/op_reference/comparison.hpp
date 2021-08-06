// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <vector>

#include "base_reference_test.hpp"
#include "ngraph_functions/builders.hpp"

namespace reference_tests {
namespace ComparisonOpsRefTestDefinitions {

struct RefComparisonParams {
    ngraph::helpers::ComparisonTypes compType;
    Tensor input1;
    Tensor input2;
    Tensor expected;
};

struct Builder : ParamsBuilder<RefComparisonParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, compType);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input1);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, input2);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceComparisonLayerTest : public testing::TestWithParam<RefComparisonParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.compType, params.input1.shape, params.input2.shape, params.input1.type, params.expected.type);
        inputData = {params.input1.data, params.input2.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RefComparisonParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "comparisonType=" << param.compType << "_";
        result << "inpt_shape1=" << param.input1.shape << "_";
        result << "inpt_shape2=" << param.input2.shape << "_";
        result << "iType=" << param.input1.type << "_";
        result << "oType=" << param.expected.type;
        return result.str();
    }

private:
    static std::shared_ptr<ngraph::Function> CreateFunction(ngraph::helpers::ComparisonTypes comp_op_type, const ngraph::PartialShape& input_shape1,
                                                            const ngraph::PartialShape& input_shape2, const ngraph::element::Type& input_type,
                                                            const ngraph::element::Type& expected_output_type) {
        const auto in = std::make_shared<ngraph::op::Parameter>(input_type, input_shape1);
        const auto in2 = std::make_shared<ngraph::op::Parameter>(input_type, input_shape2);
        const auto comp = ngraph::builder::makeComparison(in, in2, comp_op_type);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector {comp}, ngraph::ParameterVector {in, in2});
    }
};
}  // namespace ComparisonOpsRefTestDefinitions
}  // namespace reference_tests
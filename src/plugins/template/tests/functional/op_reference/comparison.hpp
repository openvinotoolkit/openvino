// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "ngraph_functions/builders.hpp"

using namespace ov;

namespace reference_tests {
namespace ComparisonOpsRefTestDefinitions {

struct RefComparisonParams {
    ngraph::helpers::ComparisonTypes compType;
    reference_tests::Tensor input1;
    reference_tests::Tensor input2;
    reference_tests::Tensor expected;
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
    static std::shared_ptr<ov::Model> CreateFunction(ngraph::helpers::ComparisonTypes comp_op_type, const ov::PartialShape& input_shape1,
                                                            const ov::PartialShape& input_shape2, const ov::element::Type& input_type,
                                                            const ov::element::Type& expected_output_type) {
        const auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape1);
        const auto in2 = std::make_shared<op::v0::Parameter>(input_type, input_shape2);
        const auto comp = ngraph::builder::makeComparison(in, in2, comp_op_type);
        return std::make_shared<ov::Model>(ov::NodeVector {comp}, ov::ParameterVector {in, in2});
    }
};
}  // namespace ComparisonOpsRefTestDefinitions
}  // namespace reference_tests

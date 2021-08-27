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
namespace LogicalOpsRefTestDefinitions {

struct RefLogicalParams {
    ngraph::helpers::LogicalTypes opType;
    std::vector<Tensor> inputs;
    Tensor expected;
};

struct Builder : ParamsBuilder<RefLogicalParams> {
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, opType);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, inputs);
    REFERENCE_TESTS_ADD_SET_PARAM(Builder, expected);
};

class ReferenceLogicalLayerTest : public testing::TestWithParam<RefLogicalParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params.opType, params.inputs);
        for (auto& input : params.inputs) {
            inputData.push_back(input.data);
        }
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<RefLogicalParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "LogicalType=" << param.opType << "_";
        for (size_t i =0; i< param.inputs.size(); i++) {
            const auto input = param.inputs[i];
            result << "inpt_shape" << i << "=" << input.shape << "_";
            result << "inpt_type" << i << "=" << input.type << "_";
        }
        result << "oType=" << param.expected.type;
        return result.str();
    }

private:
    static std::shared_ptr<ngraph::Function> CreateFunction(ngraph::helpers::LogicalTypes op_type, const std::vector<Tensor>& inputs) {
        ngraph::ParameterVector params_vec;
        for (auto& input : inputs) {
            params_vec.push_back(std::make_shared<ngraph::op::Parameter>(input.type, input.shape));
        }

        const auto logical_op = ngraph::builder::makeLogical(params_vec, op_type);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector {logical_op}, ngraph::ParameterVector {params_vec});
    }
};
}  // namespace LogicalOpsRefTestDefinitions
}  // namespace reference_tests

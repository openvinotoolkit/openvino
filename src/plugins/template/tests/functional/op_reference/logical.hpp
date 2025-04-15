// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"

using namespace ov;

namespace reference_tests {
namespace LogicalOpsRefTestDefinitions {

enum LogicalTypes { LOGICAL_AND, LOGICAL_OR, LOGICAL_XOR, LOGICAL_NOT };

struct RefLogicalParams {
    LogicalTypes opType;
    std::vector<reference_tests::Tensor> inputs;
    reference_tests::Tensor expected;
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
        for (size_t i = 0; i < param.inputs.size(); i++) {
            const auto input = param.inputs[i];
            result << "inpt_shape" << i << "=" << input.shape << "_";
            result << "inpt_type" << i << "=" << input.type << "_";
        }
        result << "oType=" << param.expected.type;
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(LogicalTypes op_type,
                                                     const std::vector<reference_tests::Tensor>& inputs) {
        ov::ParameterVector params_vec;
        for (auto& input : inputs) {
            params_vec.push_back(std::make_shared<op::v0::Parameter>(input.type, input.shape));
        }

        std::shared_ptr<ov::Node> logical_op;
        switch (op_type) {
        case LogicalTypes::LOGICAL_AND: {
            logical_op = std::make_shared<ov::op::v1::LogicalAnd>(params_vec[0], params_vec[1]);
            break;
        }
        case LogicalTypes::LOGICAL_OR: {
            logical_op = std::make_shared<ov::op::v1::LogicalOr>(params_vec[0], params_vec[1]);
            break;
        }
        case LogicalTypes::LOGICAL_NOT: {
            logical_op = std::make_shared<ov::op::v1::LogicalNot>(params_vec[0]);
            break;
        }
        case LogicalTypes::LOGICAL_XOR: {
            logical_op = std::make_shared<ov::op::v1::LogicalXor>(params_vec[0], params_vec[1]);
            break;
        }
        default: {
            throw std::runtime_error("Incorrect type of Logical operation");
        }
        }
        return std::make_shared<ov::Model>(ov::NodeVector{logical_op}, ov::ParameterVector{params_vec});
    }
};
}  // namespace LogicalOpsRefTestDefinitions
}  // namespace reference_tests

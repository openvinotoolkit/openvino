// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "common_test_utils/test_enums.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/is_finite.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/not_equal.hpp"

using namespace ov;

namespace reference_tests {
namespace ComparisonOpsRefTestDefinitions {

using ov::test::utils::ComparisonTypes;

struct RefComparisonParams {
    ComparisonTypes compType;
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
        function = CreateFunction(params.compType,
                                  params.input1.shape,
                                  params.input2.shape,
                                  params.input1.type,
                                  params.expected.type);
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
    static std::shared_ptr<ov::Model> CreateFunction(ComparisonTypes comp_op_type,
                                                     const ov::PartialShape& input_shape1,
                                                     const ov::PartialShape& input_shape2,
                                                     const ov::element::Type& input_type,
                                                     const ov::element::Type& expected_output_type) {
        const auto in0 = std::make_shared<op::v0::Parameter>(input_type, input_shape1);
        const auto in1 = std::make_shared<op::v0::Parameter>(input_type, input_shape2);
        std::shared_ptr<ov::Node> comp;
        switch (comp_op_type) {
        case ComparisonTypes::EQUAL: {
            comp = std::make_shared<ov::op::v1::Equal>(in0, in1);
            break;
        }
        case ComparisonTypes::NOT_EQUAL: {
            comp = std::make_shared<ov::op::v1::NotEqual>(in0, in1);
            break;
        }
        case ComparisonTypes::GREATER: {
            comp = std::make_shared<ov::op::v1::Greater>(in0, in1);
            break;
        }
        case ComparisonTypes::GREATER_EQUAL: {
            comp = std::make_shared<ov::op::v1::GreaterEqual>(in0, in1);
            break;
        }
        case ComparisonTypes::IS_FINITE: {
            comp = std::make_shared<ov::op::v10::IsFinite>(in0);
            break;
        }
        case ComparisonTypes::IS_INF: {
            comp = std::make_shared<ov::op::v10::IsInf>(in0);
            break;
        }
        case ComparisonTypes::IS_NAN: {
            comp = std::make_shared<ov::op::v10::IsNaN>(in0);
            break;
        }
        case ComparisonTypes::LESS: {
            comp = std::make_shared<ov::op::v1::Less>(in0, in1);
            break;
        }
        case ComparisonTypes::LESS_EQUAL: {
            comp = std::make_shared<ov::op::v1::LessEqual>(in0, in1);
            break;
        }
        default: {
            throw std::runtime_error("Incorrect type of Comparison operation");
        }
        }
        return std::make_shared<ov::Model>(ov::NodeVector{comp}, ov::ParameterVector{in0, in1});
    }
};
}  // namespace ComparisonOpsRefTestDefinitions
}  // namespace reference_tests

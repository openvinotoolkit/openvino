// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/check_broadcast.hpp"

#include "common_test_utils/common_utils.hpp"
#include "subgraph_converts.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/multiply.hpp"

namespace ov {
namespace test {
namespace snippets {

class CheckBroadcastFunction {
public:
    static std::shared_ptr<ov::Model> get(
        const PartialShape& input_shape1,
        const PartialShape& input_shape2,
        const ov::element::Type input_type,
        const ov::op::AutoBroadcastSpec broadcast) {
        const auto parameter1 = std::make_shared<ov::op::v0::Parameter>(input_type, input_shape1);
        parameter1->set_friendly_name("parameter1");

        const auto parameter2 = std::make_shared<ov::op::v0::Parameter>(input_type, input_shape2);
        parameter2->set_friendly_name("parameter2");

        std::shared_ptr<Node> parent = std::make_shared<ov::op::v1::Multiply>(
            parameter1,
            parameter2,
            broadcast);
        parent->set_friendly_name("multiply");

        const auto result = std::make_shared<ov::op::v0::Result>(parent);
        result->set_friendly_name("result");

        return std::make_shared<ov::Model>(
            ov::ResultVector{ result },
            ov::ParameterVector{ parameter1, parameter2 },
            "CheckBroadcastFunction");
    }
};

std::string CheckBroadcast::getTestCaseName(testing::TestParamInfo<CheckBroadcastParams> obj) {
    const auto& [input_type, test_case_params, target_device] = obj.param;

    std::ostringstream result;
    result << "IS[0]=" << ov::test::utils::partialShape2str({test_case_params.input_shapes.first.first}) << "_";
    result << "TS[0]=";
    for (const auto& shape : test_case_params.input_shapes.first.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "IS[1]=" << ov::test::utils::partialShape2str({test_case_params.input_shapes.second.first}) << "_";
    result << "TS[1]=";
    for (const auto& shape : test_case_params.input_shapes.second.second) {
        result << "(" << ov::test::utils::vec2str(shape) << ")_";
    }
    result << "IT=" << input_type << "_";
    result << "BCT=" << test_case_params.broadcast.m_type << "_";
    result << "BCA=" << test_case_params.broadcast.m_axis << "_";
    result << "#N=" << test_case_params.num_nodes << "_";
    result << "#S=" << test_case_params.num_subgraphs << "_";
    result << "targetDevice=" << target_device;
    return result.str();
}

void CheckBroadcast::SetUp() {
    const auto& [input_type, test_case_params, _targetDevice] = this->GetParam();
    targetDevice = _targetDevice;
    ref_num_nodes = test_case_params.num_nodes;
    ref_num_subgraphs = test_case_params.num_subgraphs;

    init_input_shapes({test_case_params.input_shapes.first, test_case_params.input_shapes.second});

    function = CheckBroadcastFunction::get(
        inputDynamicShapes[0],
        inputDynamicShapes[1],
        input_type,
        test_case_params.broadcast);
    setIgnoreCallbackMode();
}

TEST_P(CheckBroadcast, CompareWithRefImpl) {
    run();
    validateNumSubgraphs();
}

} // namespace snippets
} // namespace test
} // namespace ov

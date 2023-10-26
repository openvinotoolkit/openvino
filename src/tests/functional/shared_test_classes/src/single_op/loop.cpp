// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/loop.hpp"

#include "transformations/control_flow/unroll_tensor_iterator.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/loop.hpp"

namespace ov {
namespace test {
std::string LoopLayerTest::getTestCaseName(const testing::TestParamInfo<LoopParams> &obj) {
    bool execute_first_iteration;
    bool is_body_condition_const;
    bool body_condition; // works only if is_body_condition_const ==
    int64_t trip_count;
    std::vector<InputShape> shapes;
    std::vector<LOOP_IN_TYPE> input_types;
    ov::element::Type model_type;
    std::string targetDevice;
    std::tie(execute_first_iteration, is_body_condition_const, body_condition, trip_count, shapes, input_types, model_type,
                targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "execute_first_iteration" << execute_first_iteration << "_";
    result << "is_body_condition_const=" << is_body_condition_const << "_";
    result << "body_condition=" << body_condition << "_";
    result << "trip_count=" << trip_count << "_";
    result << "types=" << ov::test::utils::vec2str(input_types) << "_";
    result << "modelType=" << model_type.get_type_name() << "_";
    result << "targetDevice=" << targetDevice << "_";
    auto res_str = result.str();
    std::replace(res_str.begin(), res_str.end(), '-', '_');
    return res_str;
}

void LoopLayerTest::SetUp() {
    bool execute_first_iteration;
    bool is_body_condition_const;
    bool body_condition; // works only if is_body_condition_const ==
    int64_t trip_count;
    std::vector<InputShape> shapes;
    std::vector<LOOP_IN_TYPE> input_types;
    ov::element::Type model_type;
    std::tie(execute_first_iteration, is_body_condition_const, body_condition, trip_count, shapes, input_types, model_type,
                targetDevice) = this->GetParam();
    init_input_shapes(shapes);

    // Example:
/*  auto X = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{32, 1, 10});
    auto Y = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{32, 1, 10});
    auto M = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{32, 1, 10});*/
    ov::ParameterVector params;
    for (auto&& shape : inputDynamicShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, shape));
    }

    //Example:
/*  auto Xi = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto Yi = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto M_body = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic());*/

    ov::ParameterVector body_params;
    for (int i = 0; i < inputDynamicShapes.size(); i++) {
        body_params.push_back(std::make_shared<ov::op::v0::Parameter>(model_type, ov::PartialShape::dynamic()));
    }

    std::shared_ptr<ov::Node> body_condition_const;
    if (is_body_condition_const) {
            body_condition_const = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, body_condition);
    }
    auto trip_count_const = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, trip_count);
    auto exec_condition = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{1}, execute_first_iteration);

    // Body
    std::shared_ptr<ov::Node> Zo = body_params[0];
    for (int i = 1; i < body_params.size(); ++i) {
        Zo = std::make_shared<ov::op::v1::Add>(body_params[i], Zo);
    }

    auto body = std::make_shared<ov::Model>(ov::OutputVector{body_condition_const, Zo}, body_params);

    auto loop = std::make_shared<ov::op::v5::Loop>(trip_count_const, exec_condition);
    loop->set_function(body);
    loop->set_special_body_ports(ov::op::v5::Loop::SpecialBodyPorts{-1, 0});

    for (int i = 0; i < body_params.size(); ++i) {
        if (input_types[i] == LOOP_IN_TYPE::INVARIANT) {
            loop->set_invariant_input(body_params[i], params[i]);
        } else if (input_types[i] == LOOP_IN_TYPE::MERGED) {
            // todo: support several merged inputs
            // now supported only one in this sample
            loop->set_merged_input(body_params[i], params[i], Zo);
        }
    }

    // Output 0 is last Zo
    auto out0 = loop->get_iter_value(body_condition_const, -1);
    auto out1 = loop->get_iter_value(Zo, -1);
    // Output 1 is concat of Zos
    // start=0, stride=1, part_size=1, end=-1, axis=1
    auto out2 = loop->get_concatenated_slices(Zo, 0, 1, 1, -1, 1);

    auto result0 = std::make_shared<ov::op::v0::Result>(out0);
    auto result1 = std::make_shared<ov::op::v0::Result>(out1);
    auto result2 = std::make_shared<ov::op::v0::Result>(out2);
    function = std::make_shared<ov::Model>(ov::ResultVector{result0, result1, result2}, params, "loop");
}
}  // namespace test
}  // namespace ov

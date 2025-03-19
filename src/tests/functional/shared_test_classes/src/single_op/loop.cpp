// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/loop.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/less.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/control_flow/unroll_tensor_iterator.hpp"

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

std::string StaticShapeLoopLayerTest::getTestCaseName(const testing::TestParamInfo<StaticShapeLoopParams> &obj) {
    bool unrolling;
    bool static_iter_num;
    bool static_continue_cond;
    int64_t max_iter_num;
    int64_t dynamic_exit;
    int64_t axis;
    int64_t start_value;
    ov::Shape data_shape;
    ov::element::Type model_type;
    std::string target_device;
    auto args_papck = std::tie(static_iter_num, max_iter_num, dynamic_exit, axis);
    std::tie(
        unrolling,
        static_continue_cond,
        args_papck,
        start_value,
        data_shape,
        model_type,
        target_device) = obj.param;

    std::ostringstream result;
    result << "unrolling=" << std::to_string(unrolling) << "_";
    result << "static_iter_num=" << std::to_string(static_iter_num) << "_";
    result << "static_continue_cond=" << std::to_string(static_continue_cond) << "_";
    result << "max_iter_num=" << std::to_string(max_iter_num) << "_";
    result << "dynamic_exit=" << std::to_string(dynamic_exit) << "_";
    result << "axis=" << std::to_string(axis) << "_";
    result << "start_value=" << std::to_string(start_value) << "_";
    result << "max_iter_num=" << std::to_string(max_iter_num) << "_";
    result << "IS=" << ov::test::utils::vec2str(data_shape) << "_";
    result << "modelType=" << model_type.get_type_name() << "_";
    result << "targetDevice=" << target_device << "_";

    auto res_str = result.str();
    std::replace(res_str.begin(), res_str.end(), '-', '_');
    return res_str;
}

void StaticShapeLoopLayerTest::SetUp() {
    bool unrolling;
    bool static_iter_num;
    bool static_continue_cond;
    int64_t max_iter_num;
    int64_t dynamic_exit;
    int64_t axis;
    int64_t start_value;
    ov::Shape data_shape;
    ov::element::Type model_type;
    auto args_papck = std::tie(static_iter_num, max_iter_num, dynamic_exit, axis);
    std::tie(
        unrolling,
        static_continue_cond,
        args_papck,
        start_value,
        data_shape,
        model_type,
        targetDevice) = GetParam();

    const auto ngShape = ov::Shape{data_shape};
    const auto scalarShape = ov::Shape{};

    ov::ParameterVector params{};
    auto cond_input_create = [&params] (ov::element::Type model_type, const ov::Shape &shape, int value = 0, bool is_static = false)
            -> std::shared_ptr<ov::Node> {
        if (is_static)
            return std::make_shared<ov::op::v0::Constant>(model_type, shape, value);

        auto input = std::make_shared<ov::op::v0::Parameter>(model_type, shape);
        params.push_back(input);
        return input;
    };

    auto start = cond_input_create(model_type, ngShape);
    auto count = cond_input_create(ov::element::i64, scalarShape, max_iter_num, static_iter_num);
    auto skip  = cond_input_create(ov::element::boolean, scalarShape, true, static_continue_cond);

    //
    //      count skip  start         count skip      start
    //                  /                             /
    //          ___*___*____           __________*___*____       | idx  | data | out |
    //         |  idx  in   |         | ex_val  idx  in   |      |  0   |  7   |  7  |
    //         |   |  /     |         |   |   /  |  /     |      |  1   |  7   |  8  |
    //         |   add      |         |   less   add      |      |  2   |  8   |  10 |
    //         |   |   true |         |    |     |        |      |  3   |  10  |  13 |
    //         |   |    |   |         |    |     |        |       ~~~~~  * * *  ~~~~~
    //         |  out  cnd  |         |   cnd   out       |
    //         |___*____*___|         |____*_____*________|
    //           Full loop              Dynamic exit loop
    //           n_iter = count         n_iter = ex_val
    //
    auto b_indx = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{});
    auto b_data = std::make_shared<ov::op::v0::Parameter>(model_type, ngShape);
    auto b_indx_cast = std::make_shared<ov::op::v0::Convert>(b_indx, model_type);
    auto b_add  = std::make_shared<ov::op::v1::Add>(b_data, b_indx_cast);

    std::shared_ptr<ov::Node> b_cond;
    if (dynamic_exit == -1) {
        b_cond = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, true);
    } else {
        auto b_exit_value = std::make_shared<ov::op::v0::Constant>(ov::element::i64, scalarShape, dynamic_exit);
        b_cond = std::make_shared<ov::op::v1::Less>(b_indx, b_exit_value);
    }

    auto body = std::make_shared<ov::Model>(
            ov::OutputVector    {b_cond, b_add},    // TODO: check with reverse
            ov::ParameterVector {b_indx, b_data});  // TODO: check with reverse

    auto loop = std::make_shared<ov::op::v5::Loop>(count, skip);
    loop->set_function(body);
    loop->set_special_body_ports({0, 0});
    loop->set_merged_input(b_data, start, b_add);
    if (axis == -1)
        loop->get_iter_value(b_add, -1);
    else
        loop->get_concatenated_slices(b_add, 0, 1, 1, -1, axis);

    function = std::make_shared<ov::Model>(
            ov::OutputVector {loop},
            params);
    if (unrolling) {
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::UnrollTensorIterator>();
        manager.run_passes(function);
    }
}
}  // namespace test
}  // namespace ov

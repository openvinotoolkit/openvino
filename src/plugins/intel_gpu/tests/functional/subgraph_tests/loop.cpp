// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace {
using ov::test::InputShape;

using DynamicShapeLoopParams = typename std::tuple<
        bool,
        std::tuple<
            bool,
            int64_t,
            int64_t,
            int64_t
            >,
        int64_t,
        InputShape,
        ov::element::Type,
        std::string>;

/**
 * Test case with Dynamic SHAPE version of loop operation.
 * Total iteration count is dynamic.
 */
class DynamicShapeLoopTest : public testing::WithParamInterface<DynamicShapeLoopParams>,
                             virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicShapeLoopParams> &obj) {
        const auto& [static_continue_cond, args_pack, start_value, data_shapes, model_type, targetDevice] = obj.param;
        const auto [static_iter_num, max_iter_num, dynamic_exit, axis] = args_pack;

        std::ostringstream result;
        result << "static_iter_num=" << std::to_string(static_iter_num) << "_";
        result << "static_continue_cond=" << std::to_string(static_continue_cond) << "_";
        result << "max_iter_num=" << std::to_string(max_iter_num) << "_";
        result << "dynamic_exit=" << std::to_string(dynamic_exit) << "_";
        result << "axis=" << std::to_string(axis) << "_";
        result << "start_value=" << std::to_string(start_value) << "_";
        result << "max_iter_num=" << std::to_string(max_iter_num) << "_";
        result << "IS=(";
        result << ov::test::utils::partialShape2str({data_shapes.first}) << "_";
        for (size_t i = 0lu; i < data_shapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(data_shapes.second[i]) << "_";
            result << "}_";
        }
        result << ")_";
        result << "netType=" << model_type << "_";
        result << "targetDevice=" << targetDevice << "_";

        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

private:
    bool static_iter_num;       // trip count provided by constant node
    bool static_continue_cond;  // initial_cond provided by constant node
    int64_t max_iter_num;       // -1 means infinity loop (expected dynamic exit condition in body)
    int64_t dynamic_exit;       // -1 means always true
    int64_t axis;               // -1 means no auto concatenation
    int64_t start_value;
    InputShape data_shapes;
    ov::element::Type model_type;

protected:
    void SetUp() override {
        auto args_pack = std::tie(static_iter_num, max_iter_num, dynamic_exit, axis);
        std::tie(
            static_continue_cond,
            args_pack,
            start_value,
            data_shapes,
            model_type,
            targetDevice) = GetParam();

        const auto& inputShape = data_shapes.first;
        const auto scalarShape = ov::Shape{};
        init_input_shapes({data_shapes, data_shapes});

        ov::ParameterVector params{};
        auto cond_input_create = [&params] (ov::element::Type model_type,
                                            const ov::PartialShape &shape,
                                            int value = 0,
                                            bool is_static = false) -> std::shared_ptr<ov::Node> {
            if (is_static)
                return std::make_shared<ov::op::v0::Constant>(model_type, shape.to_shape(), value);

            auto input = std::make_shared<ov::op::v0::Parameter>(model_type, shape);
            params.push_back(input);
            return input;
        };

        auto start_add = cond_input_create(model_type, inputShape, start_value);
        start_add->set_friendly_name("start_add");
        auto start_mul = cond_input_create(model_type, inputShape, 1);
        start_mul->set_friendly_name("start_mul");
        auto count = cond_input_create(ov::element::i64, scalarShape, max_iter_num, static_iter_num);
        count->set_friendly_name("count");
        auto skip  = cond_input_create(ov::element::boolean, scalarShape, true, static_continue_cond);
        skip->set_friendly_name("skip");

        auto b_indx = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{});
        b_indx->set_friendly_name("body_index");
        auto b_data_add = std::make_shared<ov::op::v0::Parameter>(model_type, inputShape);
        b_data_add->set_friendly_name("b_data_add");
        auto b_data_mul = std::make_shared<ov::op::v0::Parameter>(model_type, inputShape);
        b_data_mul->set_friendly_name("b_data_mul");
        auto b_indx_cast = std::make_shared<ov::op::v0::Convert>(b_indx, model_type);
        b_indx_cast->set_friendly_name("body_index_cast");
        auto b_add  = std::make_shared<ov::op::v1::Add>(b_data_add, b_indx_cast);
        b_add->set_friendly_name("body_add");
        auto b_mul  = std::make_shared<ov::op::v1::Multiply>(b_data_mul, b_indx_cast);
        b_mul->set_friendly_name("body_mul");

        std::shared_ptr<ov::Node> b_cond;
        if (dynamic_exit == -1) {
            b_cond = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, true);
            b_cond->set_friendly_name("body_condition");
        } else {
            auto b_exit_value = std::make_shared<ov::op::v0::Constant>(ov::element::i64, scalarShape, dynamic_exit);
            b_exit_value->set_friendly_name("body_exit_value");
            b_cond = std::make_shared<ov::op::v1::Less>(b_indx, b_exit_value);
            b_cond->set_friendly_name("body_condition_with_exit_value");
        }

        auto body = std::make_shared<ov::Model>(
                ov::OutputVector    {b_cond, b_add, b_mul},    // TODO: check with reverse
                ov::ParameterVector {b_indx, b_data_add, b_data_mul});  // TODO: check with reverse
        body->set_friendly_name("body_network");

        auto loop = std::make_shared<ov::op::v5::Loop>(count, skip);
        loop->set_friendly_name("loop");
        loop->set_function(body);
        loop->set_special_body_ports({0, 0});
        loop->set_merged_input(b_data_add, start_add, b_add);
        loop->set_merged_input(b_data_mul, start_mul, b_mul);
        if (axis == -1) {
            loop->get_iter_value(b_add, -1);
            loop->get_iter_value(b_mul, -1);
        } else {
            loop->get_concatenated_slices(b_add, 0, 1, 1, -1, axis);
            loop->get_concatenated_slices(b_mul, 0, 1, 1, -1, axis);
        }

        ov::ResultVector results;
        for (size_t i = 0; i < loop->get_output_size(); i++) {
            auto res = std::make_shared<ov::op::v0::Result>(loop->output(i));
            res->set_friendly_name("loop_output_" + std::to_string(i));
            results.push_back(res);
        }
        function = std::make_shared<ov::Model>(
                results,
                params);
        function->set_friendly_name("outer_body_network");
    }
};


TEST_P(DynamicShapeLoopTest, Inference) {
    run();
}

std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::i32
};

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> dynamic_loop_types_axis_0 {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  10, -1, 0 },  // n_iter 10, no dynamic exit
};

std::vector<InputShape> inputs_0 = {
    InputShape(ov::PartialShape({1, -1, 2}), {{1, 4, 2}, {1, 5, 2}, {1, 10, 2}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapeLoop_axis_0, DynamicShapeLoopTest,
                        testing::Combine(
                        /* static_continue_cond */ testing::Values(true),
                        /* args_pack */ testing::ValuesIn(dynamic_loop_types_axis_0),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_0),
                        /* model_type */ testing::ValuesIn(model_types),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                        DynamicShapeLoopTest::getTestCaseName);

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> dynamic_loop_types_1 {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5, -1,  1 },  // n_iter 5, no dynamic exit
};

std::vector<InputShape> inputs_1 = {
    InputShape(ov::PartialShape({-1, 1, 4, -1}), {{2, 1, 4, 10}, {3, 1, 4, 14}, {6, 1, 4, 16}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapeLoop_axis_1, DynamicShapeLoopTest,
                        testing::Combine(
                        /* static_continue_cond */ testing::Values(true),
                        /* args_pack */ testing::ValuesIn(dynamic_loop_types_1),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_1),
                        /* model_type */ testing::ValuesIn(model_types),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                        DynamicShapeLoopTest::getTestCaseName);

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> dynamic_loop_types_2 {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  10, -1,  2 },  // n_iter 10, no dynamic exit
};

std::vector<InputShape> inputs_2 = {
    InputShape(ov::PartialShape({-1, -1, 1, 6}), {{2, 4, 1, 6}, {10, 40, 1, 6}, {12, 16, 1, 6}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapeLoop_axis_2, DynamicShapeLoopTest,
                        testing::Combine(
                        /* static_continue_cond */ testing::Values(true),
                        /* args_pack */ testing::ValuesIn(dynamic_loop_types_2),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_2),
                        /* model_type */ testing::ValuesIn(model_types),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                        DynamicShapeLoopTest::getTestCaseName);

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> dynamic_loop_types_no_auto_concat {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  10, -1, -1 },  // n_iter 5, no dynamic exit
};

std::vector<InputShape> inputs_no_auto_concat = {
    InputShape(ov::PartialShape({-1, 1, 6}), {{2, 1, 6}, {10, 1, 6}, {12, 1, 6}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapeLoop_no_auto_concat, DynamicShapeLoopTest,
                        testing::Combine(
                        /* static_continue_cond */ testing::Values(true),
                        /* args_pack */ testing::ValuesIn(dynamic_loop_types_no_auto_concat),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_no_auto_concat),
                        /* model_type */ testing::ValuesIn(model_types),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                        DynamicShapeLoopTest::getTestCaseName);

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> dynamic_loop_types_dynamic_exit {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5,  3,  -1 },  // n_iter 3, dynamic exit on 3
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5,  7,   1 },  // n_iter 5, dynamic exit not reached
    std::tuple<bool, int64_t, int64_t, int64_t>{  true , -1,  5,  -1 },  // n_iter 5, inf loop with dynamic exit on 5
};

std::vector<InputShape> inputs_dynamic_exit = {
    InputShape(ov::PartialShape({-1, 1, 2}), {{4, 1, 2}, {10, 1, 2}, {12, 1, 2}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapeLoop_dynamic_exit, DynamicShapeLoopTest,
                        testing::Combine(
                        /* static_continue_cond */ testing::Values(true),
                        /* args_pack */ testing::ValuesIn(dynamic_loop_types_dynamic_exit),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_dynamic_exit),
                        /* model_type */ testing::ValuesIn(model_types),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU)),
                        DynamicShapeLoopTest::getTestCaseName);


using DynamicShapeLoopDynamicInputParams = typename std::tuple<
        bool,
        std::tuple<
            bool,
            int64_t,
            int64_t,
            int64_t
            >,
        int64_t,
        InputShape,
        InputShape,
        ov::element::Type,
        std::string,
        bool>;

class DynamicShapeLoopDynamicInputTest : public testing::WithParamInterface<DynamicShapeLoopDynamicInputParams>,
                                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DynamicShapeLoopDynamicInputParams> &obj) {
        const auto& [static_continue_cond, args_pack, start_value, data_shapes, constant_shapes, model_type, targetDevice, freeze_input] = obj.param;
        const auto [static_iter_num, max_iter_num, dynamic_exit, axis] = args_pack;

        std::ostringstream result;
        result << "static_iter_num=" << std::to_string(static_iter_num) << "_";
        result << "static_continue_cond=" << std::to_string(static_continue_cond) << "_";
        result << "static_input_shape=" << std::to_string(freeze_input) << "_";
        result << "max_iter_num=" << std::to_string(max_iter_num) << "_";
        result << "dynamic_exit=" << std::to_string(dynamic_exit) << "_";
        result << "axis=" << std::to_string(axis) << "_";
        result << "start_value=" << std::to_string(start_value) << "_";
        result << "max_iter_num=" << std::to_string(max_iter_num) << "_";
        result << "IS=(";
        result << ov::test::utils::partialShape2str({data_shapes.first}) << "_";
        for (size_t i = 0lu; i < data_shapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(data_shapes.second[i]) << "_";
            result << "}_";
        }
        result << ")_";
        result << "netType=" << model_type << "_";
        result << "targetDevice=" << targetDevice << "_";

        auto res_str = result.str();
        std::replace(res_str.begin(), res_str.end(), '-', '_');
        return res_str;
    }

private:
    bool static_iter_num;       // trip count provided by constant node
    bool static_continue_cond;  // initial_cond provided by constant node
    bool freeze_input;          // set true to mark input data of broadcast as static shape
    int64_t max_iter_num;       // -1 means infinity loop (expected dynamic exit condition in body)
    int64_t dynamic_exit;       // -1 means always true
    int64_t axis;               // -1 means no auto concatenation
    int64_t start_value;
    InputShape data_shapes;
    InputShape constant_shapes;
    ov::element::Type model_type;

protected:
    void SetUp() override {
        auto args_pack = std::tie(static_iter_num, max_iter_num, dynamic_exit, axis);
        std::tie(
            static_continue_cond,
            args_pack,
            start_value,
            data_shapes,
            constant_shapes,
            model_type,
            targetDevice,
            freeze_input) = GetParam();

        const auto& inputShape = data_shapes.first;
        const auto scalarShape = ov::Shape{};
        init_input_shapes({data_shapes, data_shapes, constant_shapes});

        ov::ParameterVector params{};
        auto cond_input_create = [&params] (ov::element::Type model_type,
                                            const ov::PartialShape &shape,
                                            int value = 0,
                                            bool is_static = false) -> std::shared_ptr<ov::Node> {
            if (is_static)
                return std::make_shared<ov::op::v0::Constant>(model_type, shape.to_shape(), value);

            auto input = std::make_shared<ov::op::v0::Parameter>(model_type, shape);
            params.push_back(input);
            return input;
        };

        // Create function that has smaller shape of init input backedge-to and bigger shape backedge-from
        // It should be updated during iteration
        auto start_add = cond_input_create(model_type, inputShape, start_value);
        start_add->set_friendly_name("start_add");
        auto start_add2 = cond_input_create(model_type, inputShape, 1);
        start_add2->set_friendly_name("start_add2");
        auto count = cond_input_create(ov::element::i64, scalarShape, max_iter_num, static_iter_num);
        count->set_friendly_name("count");
        auto skip  = cond_input_create(ov::element::boolean, scalarShape, true, static_continue_cond);
        skip->set_friendly_name("skip");
        auto init_const = cond_input_create(model_type, constant_shapes.first, 1, freeze_input);
        init_const->set_friendly_name("init_const");

        auto b_indx = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{});
        b_indx->set_friendly_name("body_index");
        auto b_data_add = std::make_shared<ov::op::v0::Parameter>(model_type, inputShape);
        b_data_add->set_friendly_name("b_data_add");
        auto b_data_add2 = std::make_shared<ov::op::v0::Parameter>(model_type, inputShape);
        b_data_add2->set_friendly_name("b_data_add2");
        auto b_data_broadcast = std::make_shared<ov::op::v0::Parameter>(model_type, constant_shapes.first);
        b_data_broadcast->set_friendly_name("b_data_broadcast");
        auto b_indx_cast = std::make_shared<ov::op::v0::Convert>(b_indx, model_type);
        b_indx_cast->set_friendly_name("body_index_cast");
        auto b_add  = std::make_shared<ov::op::v1::Add>(b_data_add, b_indx_cast);
        b_add->set_friendly_name("body_add");
        auto b_add2  = std::make_shared<ov::op::v1::Add>(b_data_add2, b_indx_cast);
        b_add2->set_friendly_name("body_mul");
        auto b_shapeof1 = std::make_shared<ov::op::v3::ShapeOf>(b_data_add2);
        b_shapeof1->set_friendly_name("b_shapeof1");
        auto b_shapeof2 = std::make_shared<ov::op::v3::ShapeOf>(b_data_broadcast);
        b_shapeof2->set_friendly_name("b_shapeof2");
        auto b_max = std::make_shared<ov::op::v1::Maximum>(b_shapeof1, b_shapeof2);
        b_max->set_friendly_name("b_max");
        auto b_broadcast = std::make_shared<ov::op::v3::Broadcast>(b_data_broadcast, b_max);
        b_broadcast->set_friendly_name("b_broadcast");
        auto b_reshape = std::make_shared<ov::op::v1::Reshape>(b_broadcast, b_shapeof1, false);
        b_reshape->set_friendly_name("b_reshape");
        auto b_mul2  = std::make_shared<ov::op::v1::Multiply>(b_reshape, b_add2);
        b_mul2->set_friendly_name("b_mul2");

        std::shared_ptr<ov::Node> b_cond;
        if (dynamic_exit == -1) {
            b_cond = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, true);
            b_cond->set_friendly_name("body_condition");
        } else {
            auto b_exit_value = std::make_shared<ov::op::v0::Constant>(ov::element::i64, scalarShape, dynamic_exit);
            b_exit_value->set_friendly_name("body_exit_value");
            b_cond = std::make_shared<ov::op::v1::Less>(b_indx, b_exit_value);
            b_cond->set_friendly_name("body_condition_with_exit_value");
        }

        auto body = std::make_shared<ov::Model>(
                ov::OutputVector    {b_cond, b_add, b_add2, b_mul2},    // TODO: check with reverse
                ov::ParameterVector {b_indx, b_data_add, b_data_add2, b_data_broadcast});  // TODO: check with reverse
        body->set_friendly_name("body_network");

        auto loop = std::make_shared<ov::op::v5::Loop>(count, skip);
        loop->set_friendly_name("loop");
        loop->set_function(body);
        loop->set_special_body_ports({0, 0});
        loop->set_merged_input(b_data_add, start_add, b_add);
        loop->set_merged_input(b_data_add2, start_add2, b_add2);
        loop->set_merged_input(b_data_broadcast, init_const, b_mul2);
        if (axis == -1) {
            loop->get_iter_value(b_add, -1);
            loop->get_iter_value(b_add2, -1);
            loop->get_iter_value(b_mul2, -1);
        } else {
            loop->get_concatenated_slices(b_add, 0, 1, 1, -1, axis);
            loop->get_concatenated_slices(b_add2, 0, 1, 1, -1, axis);
        }

        ov::ResultVector results;
        for (size_t i = 0; i < loop->get_output_size(); i++) {
            auto res = std::make_shared<ov::op::v0::Result>(loop->output(i));
            res->set_friendly_name("loop_output_" + std::to_string(i));
            results.push_back(res);
        }
        function = std::make_shared<ov::Model>(
                results,
                params);
        function->set_friendly_name("outer_body_network");
    }
};

TEST_P(DynamicShapeLoopDynamicInputTest, Inference) {
    run();
}

static const std::vector<std::tuple<bool, int64_t, int64_t, int64_t>> dynamic_loop_input {
    //  GCC4.8 limitation: have to specify type of each element in list
    //                               static_trip_count |  max | dynamic_exit | axis
    std::tuple<bool, int64_t, int64_t, int64_t>{  true ,  5,  3,  -1 },  // n_iter 3, dynamic exit on 3
    std::tuple<bool, int64_t, int64_t, int64_t>{  true , -1,  5,  -1 },  // n_iter 5, inf loop with dynamic exit on 5
};

std::vector<InputShape> inputs_dynamic_shape = {
    InputShape(ov::PartialShape({-1, 1, -1}), {{4, 1, 2}, {10, 1, 2}, {12, 1, 2}}),
};

std::vector<InputShape> constant_dynamic_shape = {
    InputShape(ov::PartialShape({-1, 1, -1}), {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapeLoop_dynamic, DynamicShapeLoopDynamicInputTest,
                        testing::Combine(
                        /* static_continue_cond */ testing::Values(true),
                        /* args_pack */ testing::ValuesIn(dynamic_loop_input),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_dynamic_shape),
                        /* constant_shape */ testing::ValuesIn(constant_dynamic_shape),
                        /* model_type */ testing::ValuesIn(model_types),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                        /* freeze_input */ testing::Values(false)),
                        DynamicShapeLoopDynamicInputTest::getTestCaseName);

std::vector<InputShape> constant_static_shape = {
    InputShape({1, 1, 1}, {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}),
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicShapeLoop_conflict_dynamic, DynamicShapeLoopDynamicInputTest,
                        testing::Combine(
                        /* static_continue_cond */ testing::Values(true),
                        /* args_pack */ testing::ValuesIn(dynamic_loop_input),
                        /* start_value */ testing::Values<int64_t>(0),
                        /* data_shape */ testing::ValuesIn(inputs_dynamic_shape),
                        /* constant_shape */ testing::ValuesIn(constant_static_shape),
                        /* model_type */ testing::ValuesIn(model_types),
                        /* device */ testing::Values<std::string>(ov::test::utils::DEVICE_GPU),
                        /* freeze_input */ testing::Values(true)),
                        DynamicShapeLoopDynamicInputTest::getTestCaseName);

// Regression test for Loop body containing ScatterUpdate with i32 current_iteration index.
// UnrollTensorIterator replaces the current_iteration Parameter with Constant(i64),
// which the GPU OCL ScatterUpdate kernel does not support as an index type.
// The GPU plugin must convert such i64 constants to i32 after unrolling.
class LoopWithScatterUpdateTest : public testing::WithParamInterface<ov::element::Type>,
                                  virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ov::element::Type>& obj) {
        std::ostringstream result;
        result << "data_type=" << obj.param;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto data_type = GetParam();

        // Outer model inputs
        //   data:    [num_iter, feature_sz] – accumulation buffer
        //   updates: [1, feature_sz]        – single row written each iteration
        const size_t num_iter   = 8;
        const size_t feature_sz = 4;
        const ov::Shape data_shape   {num_iter,  feature_sz};
        const ov::Shape update_shape {1,         feature_sz};

        auto outer_data    = std::make_shared<ov::op::v0::Parameter>(data_type, data_shape);
        auto outer_updates = std::make_shared<ov::op::v0::Parameter>(data_type, update_shape);
        outer_data->set_friendly_name("outer_data");
        outer_updates->set_friendly_name("outer_updates");

        // Loop control inputs
        auto trip_count = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, num_iter);
        auto exec_cond  = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, true);

        // ---- Body ----
        // b_timestep is declared as i32 – the key point of this regression test.
        // UnrollTensorIterator will replace it with Constant(i64), which the GPU
        // OCL ScatterUpdate kernel cannot handle as an index type without the fix.
        auto b_timestep = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{});
        auto b_data     = std::make_shared<ov::op::v0::Parameter>(data_type, data_shape);
        auto b_updates  = std::make_shared<ov::op::v0::Parameter>(data_type, update_shape);
        b_timestep->set_friendly_name("timestep");
        b_data->set_friendly_name("b_data");
        b_updates->set_friendly_name("b_updates");

        // Unsqueeze scalar timestep to shape [1] for use as ScatterUpdate indices
        auto axis_const  = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{0});
        auto timestep_1d = std::make_shared<ov::op::v0::Unsqueeze>(b_timestep, axis_const);
        timestep_1d->set_friendly_name("timestep_1d");

        // ScatterUpdate: b_data[timestep, :] = b_updates[0, :]
        // indices shape [1] requires updates shape [1, feature_sz] — satisfied by b_updates
        auto scatter_axis = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{}, 0);
        auto b_scatter    = std::make_shared<ov::op::v3::ScatterUpdate>(b_data, timestep_1d, b_updates, scatter_axis);
        b_scatter->set_friendly_name("scatter_update");

        auto b_cond = std::make_shared<ov::op::v0::Constant>(ov::element::boolean, ov::Shape{}, true);

        auto body = std::make_shared<ov::Model>(
            ov::OutputVector{b_cond, b_scatter},
            ov::ParameterVector{b_timestep, b_data, b_updates});

        // ---- Loop ----
        auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, exec_cond);
        loop->set_function(body);
        // {0, 0}: b_timestep (param[0]) is current_iteration; body result[0] (b_cond) is the exit condition.
        // body_condition_output_idx must be >= 0, otherwise Loop::validate_and_infer_types() returns
        // early without setting output shapes, leaving the Loop output with a dynamic PartialShape.
        loop->set_special_body_ports({0, 0});
        loop->set_merged_input(b_data, outer_data, b_scatter);
        loop->set_invariant_input(b_updates, outer_updates);
        loop->get_iter_value(b_scatter, -1);

        auto result = std::make_shared<ov::op::v0::Result>(loop->output(0));
        function = std::make_shared<ov::Model>(ov::ResultVector{result},
                                               ov::ParameterVector{outer_data, outer_updates});

        // All shapes are fully static; set targetStaticShapes directly so that
        // compile_model compiles the function as-is without a reshape pass.
        // Calling init_input_shapes() would populate inputDynamicShapes and
        // trigger a reshape of the cloned model, which can cause the Loop op's
        // shape re-inference to produce a dynamic output dimension.
        targetStaticShapes = {{data_shape, update_shape}};
    }
};

TEST_P(LoopWithScatterUpdateTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_LoopWithScatterUpdate, LoopWithScatterUpdateTest,
                         testing::Values(ov::element::f32, ov::element::f16),
                         LoopWithScatterUpdateTest::getTestCaseName);

} // namespace

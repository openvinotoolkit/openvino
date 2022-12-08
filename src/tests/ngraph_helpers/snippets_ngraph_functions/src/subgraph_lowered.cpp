// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_lowered.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/snippets_isa.hpp>
#include "ngraph_functions/builders.hpp"
#include "snippets/pass/loop_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> AddFunctionLoweredBroadcast::initLowered() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    std::shared_ptr<Node> add_input0 = nullptr;
    if (!broadcast_shapes[0].empty() && broadcast_shapes[0].back() != input_shapes[0].rbegin()->get_length()) {
        add_input0 = std::make_shared<ngraph::snippets::op::BroadcastLoad>(data0, broadcast_shapes[0]);
    } else {
        add_input0 = std::make_shared<ngraph::snippets::op::Load>(data0);
    }

    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    std::shared_ptr<Node> add_input1 = nullptr;
    if (!broadcast_shapes[1].empty() && broadcast_shapes[1].back() != input_shapes[1].rbegin()->get_length()) {
        add_input1 = std::make_shared<ngraph::snippets::op::BroadcastLoad>(data1, broadcast_shapes[1]);
    } else {
        add_input1 = std::make_shared<ngraph::snippets::op::Load>(data1);
    }
    auto add = std::make_shared<op::v1::Add>(add_input0, add_input1);
    auto store = std::make_shared<ngraph::snippets::op::Store>(add);
    ParameterVector input_params {data0, data1};
    auto model = std::make_shared<ov::Model>(NodeVector{store}, input_params);

    // Create dummy scheduler to pass graph comparison tests
    // Note that if there is more than one results, they should be reverted
    ResultVector results({model->get_results()[0]});
    const auto& inner_loop_begin = ngraph::snippets::op::insertLoopBegin(input_params);
    std::vector<bool> apply_increments(input_params.size() + results.size(), true);
    insertLoopEnd(results, inner_loop_begin, 1, 1, apply_increments);
    auto outer_WA = std::accumulate(input_shapes.begin(), input_shapes.end(), 0,
                   [](int64_t max_val, const PartialShape& ps) {
                        return std::max(ps[ps.size() - 2].get_length(), max_val);
                    });
    if (outer_WA > 1) {
        const auto& outer_loop_begin = ngraph::snippets::op::insertLoopBegin(input_params);
        insertLoopEnd(results, outer_loop_begin, 1, 1, apply_increments);
    }
    return model;
}
std::shared_ptr<ov::Model> EltwiseThreeInputsLoweredFunction::initLowered() const {
    // todo: implement conversion between std::vector<size_t> and std::vector<Shape>
    auto input_params = ngraph::builder::makeParams(precision,
                                                    {input_shapes[0].get_shape(),
                                                     input_shapes[1].get_shape(),
                                                     input_shapes[2].get_shape()});
    auto load_or_broadcastload = [&](size_t i) -> std::shared_ptr<Node> {
        // user specified that no broadcasting is required
        if (broadcast_shapes[i].empty()) {
            return std::make_shared<ngraph::snippets::op::Load>(input_params[i]);
        // broadcasting is required: could be Load + BroadcastMove or BroiadcastLoad
        } else {
            // The last dim is processed by vector Tile, so BroadcastLoad is required if the last dim being broadcasted
            if (input_shapes[i].rbegin()->get_length() == 1 && broadcast_shapes[i].back() != 1) {
                return std::make_shared<ngraph::snippets::op::BroadcastLoad>(input_params[i], broadcast_shapes[i]);
            // Todo: Cover this logics with functional tests, Review FakeBroadcast Emitter
            // Broadcasting of other dims is handled by BroadcastMove. Strictly speaking, broadcasting is achieved via
            // appropriate pointer arithmetics in this case.
            } else {
                auto load = std::make_shared<ngraph::snippets::op::Load>(input_params[i]);
                return std::make_shared<ngraph::snippets::op::BroadcastMove>(load, broadcast_shapes[i]);
            }
        }
    };
    auto add = std::make_shared<op::v1::Add>(load_or_broadcastload(0), load_or_broadcastload(1));

    const std::vector<float> const_values = CommonTestUtils::generate_float_numbers(1, -10., 10.);
    auto sub_scalar = std::make_shared<ngraph::snippets::op::Scalar>(precision, Shape{1}, const_values[0]);
    std::shared_ptr<Node> sub_load;
    sub_load = std::make_shared<ngraph::snippets::op::Load>(input_params[2]);
    auto sub = std::make_shared<op::v1::Subtract>(sub_load, sub_scalar);
    std::shared_ptr<Node> sub_out;
    if (broadcast_shapes[2].empty())
        sub_out = sub;
    else
        sub_out = std::make_shared<ngraph::snippets::op::BroadcastMove>(sub, broadcast_shapes[2]);
    auto mul = std::make_shared<op::v1::Multiply>(add, sub_out);
    auto store = std::make_shared<ngraph::snippets::op::Store>(mul);
    auto model = std::make_shared<ov::Model>(NodeVector{store}, input_params);

    // Create dummy scheduler to pass graph comparison tests
    // Note that if there is more than one results, they should be reverted
    ResultVector results({model->get_results()[0]});
    const auto& inner_loop_begin = ngraph::snippets::op::insertLoopBegin(input_params);
    std::vector<bool> apply_increments(input_params.size() + results.size(), true);
    const auto& inner_loop_end = insertLoopEnd(results, inner_loop_begin, 1, 1, apply_increments);
    auto outer_WA = std::accumulate(input_shapes.begin(), input_shapes.end(), 0,
                                    [](int64_t max_val, const PartialShape& ps) {
                                        return std::max(ps[ps.size() - 2].get_length(), max_val);
                                    });
    if (outer_WA > 1) {
        const auto& outer_loop_begin = ngraph::snippets::op::insertLoopBegin(input_params);
        insertLoopEnd(results, outer_loop_begin, 1, 1, apply_increments);
    }
    return model;
}

std::shared_ptr<ov::Model> Transpose0213MatMulSinhLoweredFunction::initLowered() const {
    ParameterVector data{std::make_shared<op::v0::Parameter>(precision, input_shapes[0]),
                         std::make_shared<op::v0::Parameter>(precision, input_shapes[1])};
    std::vector<size_t> layout{0, 2, 1, 3};
    // Note: validity of transpose_position values is checked in Transpose0213MatMulSinhFunction constructor
    if (transpose_position <= 1) {
        auto &rt_info = data[transpose_position]->get_rt_info();
        rt_info["Layout"] = layout;
    }
    auto matmul = std::make_shared<ngraph::snippets::op::Brgemm>(data[0], data[1]);
    if (transpose_position == 2) {
        auto &rt_info = matmul->get_rt_info();
        rt_info["Layout"] = layout;
        matmul->validate_and_infer_types();
    }
    return std::make_shared<ov::Model>(NodeVector{matmul}, data);
}

std::shared_ptr<ov::Model> SoftmaxLoweredFunction::initLowered() const {
    auto input_params = ngraph::builder::makeParams(precision, {input_shapes[0].get_shape()});

    const auto data = input_params.front();

    const auto master_shape = input_shapes[0].get_shape();
    const auto shape_rank = master_shape.size();
    const auto dimension = shape_rank - 1;
    const auto work_amount = master_shape[dimension];
    const auto increment = 10;
    const auto inner_dim = shape_rank - 1;
    const auto inner_master_wa = static_cast<int>(master_shape[inner_dim]);
    const int outer_dim = shape_rank > 1 ? shape_rank - 2 : -1;
    const auto has_outer_loop = outer_dim >= 0 && master_shape[outer_dim] > 1;
    const bool is_scalar = work_amount == 1;

    /* ====== ReduceMax decomposition ====== */

    const auto vector_buffer_max = std::make_shared<ngraph::snippets::op::VectorBuffer>();
    const auto loop_max_begin = ngraph::snippets::op::insertLoopBegin(ngraph::OutputVector{data, data});

    // we don't insert Fill here after load_max to verify because in generate() call Fill op is inserted only on vector representation
    const auto load_max = std::make_shared<ngraph::snippets::op::Load>(loop_max_begin->output(0), increment);
    const auto max = std::make_shared<ov::op::v1::Maximum>(load_max, vector_buffer_max);

    std::vector<bool> apply_increments_max(3, false);
    std::vector<int64_t> finalization_offsets_max(3, 0);
    apply_increments_max[0] = data->get_shape()[inner_dim] != 1 && inner_master_wa != 1;
    finalization_offsets_max[0] = data->get_shape()[inner_dim] != 1 ? -inner_master_wa : 0;
    const auto loop_max_end = std::make_shared<ngraph::snippets::op::LoopEnd>(ngraph::OutputVector{loop_max_begin->output(1), loop_max_begin->output(2)},
        work_amount, increment, apply_increments_max, finalization_offsets_max);

    std::shared_ptr<ov::Node> horizon_max = std::make_shared<ngraph::snippets::op::HorizonMax>(max);
    horizon_max->add_control_dependency(loop_max_end);
    const auto prev_horizon_max = horizon_max;
    if (!is_scalar) {
        horizon_max = std::make_shared<ngraph::snippets::op::BroadcastMove>(horizon_max, horizon_max->get_input_partial_shape(0));
    }

    loop_max_begin->add_control_dependency(vector_buffer_max);
    loop_max_end->add_control_dependency(max);

    /* =========================================== */

    /* === Sub + Exp + ReduceSum decomposition === */

    const auto vector_buffer_sum = std::make_shared<ngraph::snippets::op::VectorBuffer>();
    const auto loop_sum_begin = ngraph::snippets::op::insertLoopBegin(ngraph::OutputVector{loop_max_end->output(0)});

    const auto load_sub = std::make_shared<ngraph::snippets::op::Load>(loop_sum_begin->output(0), increment);
    const auto sub = std::make_shared<ov::op::v1::Subtract>(load_sub, horizon_max);
    // we don't insert Fill here after Exp to verify because in generate() call Fill op is inserted only on vector representation
    const auto exp = std::make_shared<ov::op::v0::Exp>(sub);
    const auto sum = std::make_shared<ov::op::v1::Add>(exp, vector_buffer_sum);
    const auto store_exp = std::make_shared<ngraph::snippets::op::Store>(exp, increment);

    std::vector<bool> apply_increments_sum(2, false);
    std::vector<int64_t> finalization_offsets_sum(2, 0);
    apply_increments_sum[0] = load_sub->get_shape()[inner_dim] != 1 && inner_master_wa != 1;
    apply_increments_sum[1] = store_exp->get_shape()[inner_dim] != 1 && inner_master_wa != 1;
    finalization_offsets_sum[0] = has_outer_loop && load_sub->get_shape()[inner_dim] != 1 ? -inner_master_wa : 0;
    finalization_offsets_sum[1] = store_exp->get_shape()[inner_dim] != 1 ? -inner_master_wa : 0;
    const auto loop_sum_end = std::make_shared<ngraph::snippets::op::LoopEnd>(
        ngraph::OutputVector{store_exp, loop_sum_begin->output(1)}, work_amount, increment,
        apply_increments_sum, finalization_offsets_sum);
    loop_sum_end->add_control_dependency(sum);

    const auto horizon_sum = std::make_shared<ngraph::snippets::op::HorizonSum>(sum);
    horizon_sum->add_control_dependency(loop_sum_end);

    const auto buffer_exp = std::make_shared<ngraph::snippets::op::Buffer>(loop_sum_end->output(0));

    loop_sum_begin->add_control_dependency(vector_buffer_sum);
    loop_sum_begin->add_control_dependency(horizon_max);
    loop_sum_begin->add_control_dependency(prev_horizon_max);

    /* =========================================== */

    /* ================== Div ==================== */

    std::shared_ptr<ov::Node> pow = std::make_shared<ngraph::snippets::op::PowerStatic>(horizon_sum, -1);
    const auto prev_pow = pow;
    if (!is_scalar) {
        pow = std::make_shared<ngraph::snippets::op::BroadcastMove>(pow, horizon_sum->get_input_partial_shape(0));
    }

    const auto loop_div_begin = ngraph::snippets::op::insertLoopBegin(ngraph::OutputVector{buffer_exp});

    const auto load_div = std::make_shared<ngraph::snippets::op::Load>(loop_div_begin->output(0), increment);
    const auto mul = std::make_shared<ov::op::v1::Multiply>(load_div, pow);
    const auto store_div = std::make_shared<ngraph::snippets::op::Store>(mul, increment);

    std::vector<bool> apply_increments_div(2, false);
    std::vector<int64_t> finalization_offsets_div(2, 0);
    apply_increments_div[0] = load_div->get_shape()[inner_dim] != 1 && inner_master_wa != 1;
    apply_increments_div[1] = store_div->get_shape()[inner_dim] != 1 && inner_master_wa != 1;
    finalization_offsets_div[0] = has_outer_loop && load_div->get_shape()[inner_dim] != 1 ? -inner_master_wa : 0;
    finalization_offsets_div[1] = has_outer_loop && store_div->get_shape()[inner_dim] != 1 ? -inner_master_wa : 0;
    const auto loop_div_end = std::make_shared<ngraph::snippets::op::LoopEnd>(
        ngraph::OutputVector{store_div, loop_div_begin->output(1)}, work_amount, increment,
        apply_increments_div, finalization_offsets_div);
    loop_div_begin->add_control_dependency(horizon_sum);
    loop_div_begin->add_control_dependency(pow);
    loop_div_begin->add_control_dependency(prev_pow);

    /* =========================================== */

    const auto result = std::make_shared<ov::op::v0::Result>(loop_div_end);
    if (has_outer_loop) {
        const auto need_increment = input_shapes[0].get_shape()[outer_dim] != 1 && input_shapes[0].get_shape()[inner_dim] == 1;
        const auto& outer_loop_begin = ngraph::snippets::op::insertLoopBegin(input_params);
        const auto outer_loop_end = insertLoopEnd(NodeVector{result}, outer_loop_begin, 1, 1, std::vector<bool>{need_increment, need_increment});
        vector_buffer_max->add_control_dependency(outer_loop_begin);
    }

    return std::make_shared<ov::Model>(ResultVector{result}, input_params);
}
std::shared_ptr<ov::Model> AddSoftmaxLoweredFunction::initLowered() const {
    auto input_params = ngraph::builder::makeParams(precision, {input_shapes[0].get_shape(), input_shapes[1].get_shape()});

    auto master_pshape = input_shapes[0];
    ov::PartialShape::broadcast_merge_into(master_pshape, input_shapes[1], op::AutoBroadcastType::NUMPY);
    const auto master_shape = master_pshape.get_shape();
    const auto shape_rank = master_shape.size();
    const auto dimension = shape_rank - 1;
    const auto work_amount = master_shape[dimension];
    const auto increment = 10;
    const auto inner_dim = shape_rank - 1;
    const auto inner_master_wa = static_cast<int>(master_shape[inner_dim]);
    const int outer_dim = shape_rank > 1 ? shape_rank - 2 : -1;
    const auto has_outer_loop = outer_dim >= 0 && master_shape[outer_dim] > 1;
    const bool is_scalar = work_amount == 1;

    /* ================== Add + ReduceMax ==================== */

    const auto vector_buffer_max = std::make_shared<ngraph::snippets::op::VectorBuffer>();
    const auto loop_max_begin = ngraph::snippets::op::insertLoopBegin(input_params);

    std::shared_ptr<ov::Node> load0 = std::make_shared<ngraph::snippets::op::Load>(loop_max_begin->output(0), increment);
    if (!is_scalar && input_shapes[0].get_shape().back() == 1) {
        auto new_shape = input_shapes[0].get_shape();
        new_shape[new_shape.size() - 1] = static_cast<size_t>(inner_master_wa);
        load0 = std::make_shared<ngraph::snippets::op::BroadcastLoad>(loop_max_begin->output(0), new_shape);
    }
    std::shared_ptr<ov::Node> load1 = std::make_shared<ngraph::snippets::op::Load>(loop_max_begin->output(1), increment);
    if (!is_scalar && input_shapes[1].get_shape().back() == 1) {
        auto new_shape = input_shapes[1].get_shape();
        new_shape[new_shape.size() - 1] = static_cast<size_t>(inner_master_wa);
        load1 = std::make_shared<ngraph::snippets::op::BroadcastLoad>(loop_max_begin->output(1), new_shape);
    }
    const auto add = std::make_shared<ov::op::v1::Add>(load0, load1);
    const auto store = std::make_shared<ngraph::snippets::op::Store>(add, increment);

    // we don't insert Fill here after load_max to verify because in generate() call Fill op is inserted only on vector representation
    const auto max = std::make_shared<ov::op::v1::Maximum>(add, vector_buffer_max);

    std::vector<bool> apply_increments_max(3, false);
    std::vector<int64_t> finalization_offsets_max(3, 0);
    apply_increments_max[0] = input_shapes[0].get_shape()[inner_dim] != 1 && inner_master_wa != 1;
    apply_increments_max[1] = input_shapes[1].get_shape()[inner_dim] != 1 && inner_master_wa != 1;
    apply_increments_max[2] = master_shape[inner_dim] != 1 && inner_master_wa != 1;
    finalization_offsets_max[0] = input_shapes[0].get_shape()[inner_dim] != 1 ? -inner_master_wa : 0;
    finalization_offsets_max[1] = input_shapes[1].get_shape()[inner_dim] != 1 ? -inner_master_wa : 0;
    finalization_offsets_max[2] = master_shape[inner_dim] != 1 ? -inner_master_wa : 0;
    const auto loop_max_end = std::make_shared<ngraph::snippets::op::LoopEnd>(ngraph::OutputVector{store, loop_max_begin->output(2)},
        work_amount, increment, apply_increments_max, finalization_offsets_max);

    std::shared_ptr<ov::Node> horizon_max = std::make_shared<ngraph::snippets::op::HorizonMax>(max);
    horizon_max->add_control_dependency(loop_max_end);
    const auto prev_horizon_max = horizon_max;
    if (!is_scalar) {
        horizon_max = std::make_shared<ngraph::snippets::op::BroadcastMove>(horizon_max, horizon_max->get_input_partial_shape(0));
    }

    loop_max_begin->add_control_dependency(vector_buffer_max);
    loop_max_end->add_control_dependency(max);

    /* =========================================== */

    const auto buffer_add = std::make_shared<ngraph::snippets::op::Buffer>(loop_max_end->output(0));

    /* === Sub + Exp + ReduceSum decomposition === */

    const auto vector_buffer_sum = std::make_shared<ngraph::snippets::op::VectorBuffer>();
    const auto loop_sum_begin = ngraph::snippets::op::insertLoopBegin(ngraph::OutputVector{buffer_add->output(0)});

    const auto load_sub = std::make_shared<ngraph::snippets::op::Load>(loop_sum_begin->output(0), increment);
    const auto sub = std::make_shared<ov::op::v1::Subtract>(load_sub, horizon_max);
    // we don't insert Fill here after exp to verify because in generate() call Fill op is inserted only on vector representation
    const auto exp = std::make_shared<ov::op::v0::Exp>(sub);
    const auto sum = std::make_shared<ov::op::v1::Add>(exp, vector_buffer_sum);
    const auto store_exp = std::make_shared<ngraph::snippets::op::Store>(exp, increment);

    std::vector<bool> apply_increments_sum(2, false);
    std::vector<int64_t> finalization_offsets_sum(2, 0);
    apply_increments_sum[0] = load_sub->get_shape()[inner_dim] != 1 && inner_master_wa != 1;
    apply_increments_sum[1] = store_exp->get_shape()[inner_dim] != 1 && inner_master_wa != 1;
    finalization_offsets_sum[0] = has_outer_loop && load_sub->get_shape()[inner_dim] != 1 ? -inner_master_wa : 0;
    finalization_offsets_sum[1] = store_exp->get_shape()[inner_dim] != 1 ? -inner_master_wa : 0;
    const auto loop_sum_end = std::make_shared<ngraph::snippets::op::LoopEnd>(
        ngraph::OutputVector{store_exp, loop_sum_begin->output(1)}, work_amount, increment,
        apply_increments_sum, finalization_offsets_sum);
    loop_sum_end->add_control_dependency(sum);

    const auto horizon_sum = std::make_shared<ngraph::snippets::op::HorizonSum>(sum);
    horizon_sum->add_control_dependency(loop_sum_end);

    const auto buffer_exp = std::make_shared<ngraph::snippets::op::Buffer>(loop_sum_end->output(0));

    loop_sum_begin->add_control_dependency(vector_buffer_sum);
    loop_sum_begin->add_control_dependency(horizon_max);
    loop_sum_begin->add_control_dependency(prev_horizon_max);

    /* =========================================== */

    /* ================== Div ==================== */

    std::shared_ptr<ov::Node> pow = std::make_shared<ngraph::snippets::op::PowerStatic>(horizon_sum, -1);
    const auto prev_pow = pow;
    if (!is_scalar) {
        pow = std::make_shared<ngraph::snippets::op::BroadcastMove>(pow, horizon_sum->get_input_partial_shape(0));
    }

    const auto loop_div_begin = ngraph::snippets::op::insertLoopBegin(ngraph::OutputVector{buffer_exp});

    const auto load_div = std::make_shared<ngraph::snippets::op::Load>(loop_div_begin->output(0), increment);
    const auto mul = std::make_shared<ov::op::v1::Multiply>(load_div, pow);
    const auto store_div = std::make_shared<ngraph::snippets::op::Store>(mul, increment);

    std::vector<bool> apply_increments_div(2, false);
    std::vector<int64_t> finalization_offsets_div(2, 0);
    apply_increments_div[0] = load_div->get_shape()[inner_dim] != 1 && inner_master_wa != 1;
    apply_increments_div[1] = store_div->get_shape()[inner_dim] != 1 && inner_master_wa != 1;
    finalization_offsets_div[0] = has_outer_loop && load_div->get_shape()[inner_dim] != 1 ? -inner_master_wa : 0;
    finalization_offsets_div[1] = has_outer_loop && store_div->get_shape()[inner_dim] != 1 ? -inner_master_wa : 0;
    const auto loop_div_end = std::make_shared<ngraph::snippets::op::LoopEnd>(
        ngraph::OutputVector{store_div, loop_div_begin->output(1)}, work_amount, increment,
        apply_increments_div, finalization_offsets_div);
    loop_div_begin->add_control_dependency(horizon_sum);
    loop_div_begin->add_control_dependency(pow);
    loop_div_begin->add_control_dependency(prev_pow);

    /* =========================================== */

    const auto result = std::make_shared<ov::op::v0::Result>(loop_div_end);
    if (has_outer_loop) {
        const auto need_increment0 = input_shapes[0].get_shape()[outer_dim] != 1 && input_shapes[0].get_shape()[inner_dim] == 1;
        const auto need_increment1 = input_shapes[1].get_shape()[outer_dim] != 1 && input_shapes[1].get_shape()[inner_dim] == 1;
        const auto need_increment2 = master_shape[outer_dim] != 1 && master_shape[inner_dim] == 1;
        const auto outer_loop_begin = ngraph::snippets::op::insertLoopBegin(input_params);
        const auto outer_loop_end = insertLoopEnd(
                NodeVector{result}, outer_loop_begin, 1, 1, std::vector<bool>{need_increment0, need_increment1, need_increment2});
        vector_buffer_max->add_control_dependency(outer_loop_begin);
    }

    return std::make_shared<ov::Model>(ResultVector{result}, input_params);
}
std::shared_ptr<ov::Model> BroadcastAddLoweredFunction::initLowered() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    std::vector<std::shared_ptr<ov::Node>> datas = {data0, data1};
    auto last_dim = std::max(input_shapes[0].get_shape().back(), std::max(input_shapes[1].get_shape().back(), m_target_shape.get_shape().back()));
    std::vector<std::shared_ptr<ov::Node>> loads(datas.size(), nullptr);
    for (auto i = 0; i < datas.size(); i++) {
        if (input_shapes[i].get_shape().back() != last_dim) {
            auto new_shape = input_shapes[i];
            new_shape[new_shape.size() - 1] = last_dim;
            loads[i] = std::make_shared<ngraph::snippets::op::BroadcastLoad>(datas[i], new_shape);
        } else {
            loads[i] = std::make_shared<ngraph::snippets::op::Load>(datas[i]);
        }
    }
    auto add = std::make_shared<op::v1::Add>(loads[0], loads[1]);
    auto store = std::make_shared<ngraph::snippets::op::Store>(add);
    auto model = std::make_shared<Model>(NodeVector{store}, ParameterVector{data0, data1});

    // Create dummy scheduler to pass graph comparison tests
    // Note that if there is more than one results, they should be reverted
    ResultVector results({model->get_results()[0]});
    const auto& inner_loop_begin = ngraph::snippets::op::insertLoopBegin(datas);
    std::vector<bool> apply_increments(datas.size() + results.size(), true);
    insertLoopEnd(results, inner_loop_begin, 1, 1, apply_increments);
    auto outer_WA = std::accumulate(input_shapes.begin(), input_shapes.end(), 0,
                                    [](int64_t max_val, const PartialShape& ps) {
                                        return std::max(ps[ps.size() - 2].get_length(), max_val);
                                    });
    if (outer_WA > 1) {
        const auto& outer_loop_begin = ngraph::snippets::op::insertLoopBegin(datas);
        insertLoopEnd(results, outer_loop_begin, 1, 1, apply_increments);
    }
    return model;
}
}  // namespace snippets
}  // namespace test
}  // namespace ov

// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/remarks.hpp"
#include <snippets/itt.hpp>

#include "snippets/pass/softmax_decomposition.hpp"
#include "snippets/pass/reset_buffer.hpp"
#include "snippets/pass/insert_loops.hpp"
#include "snippets/pass/loop_helpers.hpp"
#include "snippets/snippets_isa.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/validation_util.hpp>


ngraph::snippets::pass::SoftmaxDecomposition::SoftmaxDecomposition(const size_t vector_size, const int32_t buffer_allocation_rank) {
    MATCHER_SCOPE(SoftmaxDecomposition);

    auto m_softmax = ngraph::pattern::wrap_type<ngraph::op::v1::Softmax, ngraph::op::v8::Softmax>();

    auto callback = [=](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::SoftmaxDecomposition")
        auto root = m.get_match_root();
        const auto master_pshape = root->get_input_partial_shape(0);
        const auto rank = master_pshape.rank();
        if (rank.is_dynamic() || master_pshape.is_dynamic())
            return false;

        int64_t axis = 0;
        if (const auto softmax_v8 = ngraph::as_type_ptr<const ov::op::v8::Softmax>(root)) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            axis = ngraph::normalize_axis(root->get_friendly_name(), softmax_v8->get_axis(), rank);
            OPENVINO_SUPPRESS_DEPRECATED_END
        } else if (const auto softmax_v1 = ngraph::as_type_ptr<const ov::op::v1::Softmax>(root)) {
            axis = softmax_v1->get_axis();
        } else {
            return false;
        }

        const auto shape_rank = rank.get_length();
        if (axis != shape_rank - 1)
            return false;

        const auto data = root->get_input_node_shared_ptr(0);

        const auto master_shape = master_pshape.get_shape();
        const auto dimension = shape_rank - 1;
        const auto work_amount = master_shape[dimension];
        const auto increment = vector_size;
        const auto inner_dim = shape_rank - 1;
        const auto inner_master_work_amount = static_cast<size_t>(master_shape[inner_dim]);
        const int outer_dim = shape_rank > 1 ? static_cast<int>(shape_rank - 2) : -1;
        const auto has_outer_loop = outer_dim >= 0 && master_shape[outer_dim] > 1;

        /* ====== ReduceMax decomposition ====== */

        /* We have to have fake edge Data -> Loop[ReduceMax] -> Loop[Sub + Exp + ReduceSum] because ReduceMax is
         * accumulator which finds maximum of elements and save it to vector register. Loop works only with GPR (data) but ReduceMax Loop
         * doesn't save maximum to data. Seems like, LoopEnd shouldn't have outputs:
         *                     Data
         *  VectorBuffer   LoopBegin   \
         *         \         Load    \  |
         *           Maximum         /  |
         *              /   LoopEnd     |
         *       HorizonMax            /
         *             \   LoopBegin[Sub + Exp + ReduceSum]
         * But nGraph doesn't allow to have 0 outputs for Node (at least 1 output).
         * Thus, we propagate data through Loop[ReduceMax] using fake edge because of that Loop[ReduceMax] has two inputs "Data"
         *                    Data
         *  VectorBuffer    LoopBegin
         *         \          Load |  \
         *           Maximum       |  /
         *              /    LoopEnd
         *       HorizonMax     |
         *             \   LoopBegin[Sub + Exp + ReduceSum]
         */
        const auto vector_buffer_max = std::make_shared<ngraph::snippets::op::VectorBuffer>();
        const auto loop_max_begin = ngraph::snippets::op::insertLoopBegin(ngraph::OutputVector{data, data});

        const auto load_max = std::make_shared<ngraph::snippets::op::Load>(loop_max_begin->output(0), increment);
        const auto max = std::make_shared<ov::op::v1::Maximum>(load_max, vector_buffer_max);

        auto apply_increments_max =
                InsertLoops::calculate_inner_apply_increments(master_shape, {data->get_shape(), data->get_shape(), data->get_shape()});
        // Input of softmax is Input and Output of this loop, which isn't used inside (it's just to have one output in Loop at least)
        // So we shouldn't increment pointer after each loop iteration
        apply_increments_max[0] = false;
        apply_increments_max[1] = false;
        // we should always reset data ptr after this loop because in the next Loop this ptr is used
        // Although output isn't a Buffer op, we set finalization offset and ptr increment for output, because ResetBufferState pass
        // normalizes offsets and increments starting from outputs
        const auto finalization_offsets_max =
            std::vector<int64_t>{ 0, 0, ResetBufferState::calculate_required_finalization_offsets(inner_master_work_amount, data->get_shape()[inner_dim]) };
        const auto loop_max_end = std::make_shared<ngraph::snippets::op::LoopEnd>(ngraph::OutputVector{loop_max_begin->output(1), loop_max_begin->output(2)},
            work_amount, increment, apply_increments_max, finalization_offsets_max);

        const auto horizon_max = std::make_shared<ngraph::snippets::op::HorizonMax>(max);

        /* =========================================== */

        /* === Sub + Exp + ReduceSum decomposition === */

        const auto vector_buffer_sum = std::make_shared<ngraph::snippets::op::VectorBuffer>();
        const auto loop_sum_begin = ngraph::snippets::op::insertLoopBegin(ngraph::OutputVector{loop_max_end->output(0)});

        const auto load_sub = std::make_shared<ngraph::snippets::op::Load>(loop_sum_begin->output(0), increment);
        const auto sub = std::make_shared<ov::op::v1::Subtract>(load_sub, horizon_max);
        const auto exp = std::make_shared<ov::op::v0::Exp>(sub);
        const auto sum = std::make_shared<ov::op::v1::Add>(exp, vector_buffer_sum);
        const auto store_exp = std::make_shared<ngraph::snippets::op::Store>(exp, increment);

        auto apply_increments_sum =
                InsertLoops::calculate_inner_apply_increments(master_shape, {load_sub->get_shape(), store_exp->get_shape()});
        std::vector<int64_t> finalization_offsets_sum(2, 0);
        if (has_outer_loop) {
            finalization_offsets_sum =
                InsertLoops::calculate_finalization_offsets(master_shape, {load_sub->get_shape(), store_exp->get_shape()});
        }
        // we should always reset buffer ptr after loop because in the next Loop this buffer ptr is used
        finalization_offsets_sum[1] = ResetBufferState::calculate_required_finalization_offsets(inner_master_work_amount, store_exp->get_shape()[inner_dim]);
        const auto loop_sum_end = std::make_shared<ngraph::snippets::op::LoopEnd>(
            ngraph::OutputVector{store_exp, loop_sum_begin->output(1)}, work_amount, increment,
            apply_increments_sum, finalization_offsets_sum);

        const auto horizon_sum = std::make_shared<ngraph::snippets::op::HorizonSum>(sum);
        const auto buffer_exp = std::make_shared<op::Buffer>(loop_sum_end->output(0), buffer_allocation_rank);

        /* =========================================== */

        /* ================== Div ==================== */

        // Divide is expensive operation, so we decompose it into 1 / x * y, where 1 / x is executed outside loop
        const auto pow = std::make_shared<ngraph::opset1::Power>(horizon_sum,
            ngraph::op::Constant::create(ov::element::f32, ngraph::Shape{}, {-1}));

        const auto loop_div_begin = op::insertLoopBegin(ngraph::OutputVector{buffer_exp});

        const auto load_div = std::make_shared<ngraph::snippets::op::Load>(loop_div_begin->output(0), increment);
        const auto mul = std::make_shared<ov::op::v1::Multiply>(load_div, pow);
        const auto store_div = std::make_shared<ngraph::snippets::op::Store>(mul, increment);

        auto apply_increments_div =
                InsertLoops::calculate_inner_apply_increments(master_shape, {load_div->get_shape(), store_div->get_shape()});
        std::vector<int64_t> finalization_offsets_div(2, 0);
        if (has_outer_loop) {
            finalization_offsets_div =
                InsertLoops::calculate_finalization_offsets(master_shape, {load_div->get_shape(), store_div->get_shape()});
        }
        const auto loop_div_end = std::make_shared<ngraph::snippets::op::LoopEnd>(
            ngraph::OutputVector{store_div, loop_div_begin->output(1)}, work_amount, increment,
            apply_increments_div, finalization_offsets_div);

        /* =========================================== */

        /* ========== Control dependency ============= */

        loop_max_begin->add_control_dependency(vector_buffer_max);
        loop_max_end->add_control_dependency(max);
        horizon_max->add_control_dependency(loop_max_end);
        loop_sum_begin->add_control_dependency(vector_buffer_sum);
        loop_sum_begin->add_control_dependency(horizon_max);
        loop_sum_end->add_control_dependency(sum);
        horizon_sum->add_control_dependency(loop_sum_end);
        loop_div_begin->add_control_dependency(horizon_sum);
        loop_div_begin->add_control_dependency(pow);

        /* =========================================== */

        /* ============= Runtime Info ================ */

        // For tail loop we should fill input of Max by float min and
        // input of Sum by zero to avoid math incorrect calculations
        max->input(0).get_rt_info()["set_fill"] = uint32_t(0xff7fffff);
        sum->input(0).get_rt_info()["set_fill"] = uint32_t(0x00000000);

        // These nodes should be executed outside loops
        ov::NodeVector ops_outside_loop = { vector_buffer_max, horizon_max, vector_buffer_sum, horizon_sum, pow, buffer_exp };
        for (const auto& op : ops_outside_loop) {
            op->get_rt_info()["outside_loop"] = true;
        }

        ngraph::copy_runtime_info(root,
            {vector_buffer_max, loop_max_begin, load_max, max, horizon_max, loop_max_end,
             vector_buffer_sum, loop_sum_begin, load_sub, sub, exp, sum, store_exp, horizon_sum, loop_sum_end, buffer_exp, pow,
             loop_div_begin, load_div, mul, store_div, loop_div_end});

        /* =========================================== */

        ngraph::replace_node(root, loop_div_end);

        /* ============== Outer loop ================= */
        if (has_outer_loop) {
            std::vector<bool> apply_increments =
                    InsertLoops::calculate_outer_apply_increments({root->get_input_shape(0), root->get_output_shape(0)});
            const auto softmax_parameters =
                std::vector<ov::Output<ov::Node>>{loop_max_begin->input(0).get_source_output()};
            const auto output_set = loop_div_end->output(0).get_target_inputs();
            const auto softmax_results = std::vector<ov::Input<ov::Node>>{output_set.begin(), output_set.end()};
            const auto& outer_loop_begin = ngraph::snippets::op::insertLoopBegin(softmax_parameters);
            const auto outer_loop_end = ngraph::snippets::op::insertLoopEndBeforeInputs(
                softmax_results, outer_loop_begin, master_shape[outer_dim], 1, apply_increments);

            vector_buffer_max->add_control_dependency(outer_loop_begin);

            ngraph::copy_runtime_info(root, {outer_loop_begin, outer_loop_end});
        }
        /* =========================================== */

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(m_softmax, matcher_name);
    register_matcher(m, callback);
}

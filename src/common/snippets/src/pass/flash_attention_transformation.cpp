// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/flash_attention_transformation.hpp"

#include "openvino/op/softmax.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/snippets_isa.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace snippets {
namespace pass {

FlashAttentionTransformation::FlashAttentionTransformation() {
    MATCHER_SCOPE(FlashAttentionTransformation);
    auto softmax_v1_m = ov::pass::pattern::wrap_type<ov::op::v1::Softmax>();
    auto softmax_v8_m = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>();
    auto softmax_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{softmax_v1_m, softmax_v8_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::pass::FlashAttentionTransformation")
        // transform pattern brgemm0 + ... + softmax + brgemm1,
        // while N dimension in brgemm0 and K dimension in brgemm1 are the same and will be split on same block size.
        auto softmax = m.get_match_root();
        const auto& pshape = softmax->get_input_partial_shape(0);
        OPENVINO_ASSERT(!pshape.rank().is_dynamic(), "FlashAttentionTransformation doesn't support dynamic ranks");
        if (!pshape.is_static()) {
            return false;
        }
        const auto rank = pshape.size();
        const auto shape = pshape.get_shape();

        const auto &users = softmax->get_users();
        if (users.size() != 1)
            return false;
        auto brgemm1 = as_type_ptr<op::Brgemm>(users[0]);  // brgemm1
        if (!brgemm1)
            return false;
        auto output = brgemm1->output(0);

        size_t axis;
        if (const auto softmax_v8 = ov::as_type_ptr<ov::op::v8::Softmax>(softmax)) {
            axis = ov::util::try_normalize_axis(softmax_v8->get_axis(), rank, *softmax);
        } else if (const auto softmax_v1 = ov::as_type_ptr<ov::op::v1::Softmax>(softmax)) {
            axis = softmax_v1->get_axis();
        } else {
            OPENVINO_THROW("Unexpected node matched");
        }
        if (axis != static_cast<size_t>(-1) && axis != rank - 1) {
            return false;
        }

        // reused in each loop as value in previous loop
        // scratch_old_max and scratch_new_max to remove cycle dependecy, and the memory is shared.
        // MM0(512*64 64*4096) -> softmax(512*4096) buffer512 -> MM1(512*4096 4096*64)
        // M_blk:32, N_blk in MM0 and K_blk in MM1 is 128. 32*64, 64*128 buffer512 is split to 32.
        // require special handling on the first (e.g. initialize with zeros) and on the last (omit Store ops) iterations.
        // Specific iteration handlers can be used to handle that.
        const auto scratch_old_max = std::make_shared<snippets::op::NewMemoryBuffer>(ov::Shape{shape[rank - 2], 1});  // initialized with min
        const auto scratch_old_sum = std::make_shared<snippets::op::NewMemoryBuffer>(ov::Shape{shape[rank - 2], 1});  // initialized with zero.
        // make sure this buffer is inpace with and output memory
        const auto scratch_old_result = std::make_shared<snippets::op::InplaceMemoryBuffer>(OutputVector{brgemm1->output(0)}, brgemm1);

        const auto& softmax_input = softmax->input_value(0);
        const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(softmax_input, axis);  // 512*1
        const auto new_max = std::make_shared<ov::op::v1::Maximum>(scratch_old_max, reduce_max);  // 512*1
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_max);
        const auto subtract = std::make_shared<ov::op::v1::Subtract>(softmax_input, new_max);
        const auto exp = std::make_shared<ov::op::v0::Exp>(subtract);

        const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, axis);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_sum);
        const auto power = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, -1.f);
        const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, power);  // softmax result

        // compensation_scale = (sum_old/sum_new) * exp(max_old-max_new)
        // in first iteration, set sum_old to zero.
        const auto max_diff = std::make_shared<ov::op::v1::Subtract>(scratch_old_max, reduce_max);
        const auto exp_diff = std::make_shared<ov::op::v0::Exp>(max_diff);
        const auto sum_diff = std::make_shared<ov::op::v1::Multiply>(scratch_old_sum, power);
        const auto compensation_scale = std::make_shared<ov::op::v1::Multiply>(exp_diff, sum_diff);

        // save new to old after old usage(compensation calculation) finished.
        const auto scratch_new_max = std::make_shared<snippets::op::InplaceMemoryBuffer>(OutputVector{new_max->output(0)}, scratch_old_max);
        const auto scratch_new_sum = std::make_shared<snippets::op::InplaceMemoryBuffer>(OutputVector{reduce_sum->output(0)}, scratch_old_sum);

        // out = multiply(softmax of K blocked) * matmul->input_value(1)(V)
        const auto brgemm_new = std::make_shared<op::Brgemm>(multiply, brgemm1->input_value(1)); // result saved on a buffer
        // compensation_scale * out_old
        const auto scaled_output = std::make_shared<ov::op::v1::Multiply>(scratch_old_result, compensation_scale);
        // out = softmax result(multiply) * V(brgemm1->input_value(1)) + compensation_scale * out_old
        const auto add = std::make_shared<ov::op::v1::Add>(brgemm_new, scaled_output);

        // remove softmax
        copy_runtime_info(softmax, {reduce_max, subtract, exp, reduce_sum, power, multiply});
        softmax->output(0).replace(softmax->input_value(0));
        // replace brgemm1
        copy_runtime_info(brgemm1, add);
        return ov::replace_node_update_name(brgemm1, add);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(softmax_m, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
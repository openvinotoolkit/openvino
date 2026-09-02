// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/softmax_decomposition.hpp"

#include <cstddef>
#include <memory>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/brgemm.hpp"
#include "snippets/op/powerstatic.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::pass {

SoftmaxDecomposition::SoftmaxDecomposition() {
    MATCHER_SCOPE(SoftmaxDecomposition);
    auto softmax_v1_m = ov::pass::pattern::wrap_type<ov::op::v1::Softmax>();
    auto softmax_v8_m = ov::pass::pattern::wrap_type<ov::op::v8::Softmax>();
    auto softmax_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{softmax_v1_m, softmax_v8_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::SoftmaxDecomposition")
        auto softmax = m.get_match_root();

        const auto& pshape = softmax->get_input_partial_shape(0);
        OPENVINO_ASSERT(!pshape.rank().is_dynamic(), "SoftmaxDecomposition doesn't support dynamic ranks");
        const auto rank = pshape.size();

        const auto axis = ov::snippets::utils::get_softmax_axis(softmax);
        if (!axis) {
            OPENVINO_THROW("Unexpected node matched");
        }
        const auto normalized_axis = static_cast<size_t>(*axis);

        const auto& softmax_input = softmax->input_value(0);
        // veesion: OV_SNIPPETS_SOFTMAX=nomax lowers softmax as exp(x)/sum(exp(x)) rather than
        // the max-shifted form. It is the same function; the shift exists only to keep exp()
        // from overflowing, and it costs a whole extra pass over the score tile -- the max of
        // a row is needed before its first exp, so the ReduceMax loop cannot fuse into the
        // exp loop -- plus a subtract per element. Safe while the logits stay below ~88.
        const char* softmax_mode = std::getenv("OV_SNIPPETS_SOFTMAX");
        std::string mode = softmax_mode == nullptr ? std::string() : std::string(softmax_mode);
        const char* noexp_env = std::getenv("OV_SNIPPETS_NOEXP");
        if (noexp_env != nullptr && noexp_env[0] != '\0') {
            mode = "noexp";  // separate variable: the candidate pins OV_SNIPPETS_SOFTMAX at import
        }
        const bool shift_by_max = mode != "nomax" && mode != "noexp";

        ov::NodeVector decomposed;
        ov::Output<ov::Node> exp_input = softmax_input;
        if (shift_by_max) {
            const auto reduce_max = std::make_shared<ov::snippets::op::ReduceMax>(softmax_input, normalized_axis);
            ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_max);
            const auto subtract = std::make_shared<ov::op::v1::Subtract>(softmax_input, reduce_max);
            exp_input = subtract;
            decomposed.insert(decomposed.end(), {reduce_max, subtract});
        }
        // veesion: OV_SNIPPETS_SOFTMAX=noexp swaps the exponential for a squaring, which is one
        // vmulps over the same tile in the same loop. It computes the wrong function and exists
        // only to price the exp against the rest of the fused softmax.
        std::shared_ptr<ov::Node> exp;
        if (mode == "noexp") {
            exp = std::make_shared<ov::snippets::op::PowerStatic>(exp_input, 2.F);
        } else {
            exp = std::make_shared<ov::op::v0::Exp>(exp_input);
        }

        OPENVINO_ASSERT(normalized_axis < rank, "Softmax has incorrect axis");
        std::vector<size_t> subtensor(rank, 1);
        for (size_t i = normalized_axis; i < rank; ++i) {
            subtensor[i] = utils::get_full_dim_value();
        }

        // veesion: OV_SNIPPETS_SOFTMAX=defer applies the 1/rowsum to the OUTPUT of the matmul
        // that consumes the softmax, rather than to the softmax itself. softmax(s) @ V is
        // identically (exp(s - rowmax) @ V) / rowsum, an exact reassociation, and it changes
        // what a quantiser sees: the deferred operand peaks at exactly 1.0 on every row by
        // construction, so a fixed u8 grid is fully used whatever a row's peakedness. A
        // per-tensor grid on the normalised probabilities cannot manage that -- real attention
        // rows differ in peak by orders of magnitude and the global max starves the flat ones.
        // Both loops are blocked over the same M dimension, so the row sums are still live.
        std::shared_ptr<ov::snippets::op::Brgemm> consumer;
        if (mode == "defer") {
            std::shared_ptr<ov::Node> cursor = softmax;
            for (size_t hop = 0; hop < 16 && !consumer; ++hop) {
                const auto& targets = cursor->get_output_target_inputs(0);
                if (targets.size() != 1) {
                    break;
                }
                const auto& target = *targets.begin();
                auto next = target.get_node()->shared_from_this();
                if (auto brgemm = ov::as_type_ptr<ov::snippets::op::Brgemm>(next)) {
                    if (target.get_index() == 0) {
                        consumer = brgemm;
                    }
                    break;
                }
                cursor = next;
            }
            OPENVINO_ASSERT(consumer, "OV_SNIPPETS_SOFTMAX=defer found no Brgemm consuming the softmax");
        }
        // Deferring only pays when the matmul is int8: the point is to hand the quantiser an
        // operand that peaks at 1.0 by construction. On an f32 matmul it is pure loss -- and it
        // also walks into a FuseLoops soundness gap, where a loop whose only consumer sits in
        // the loop below can be hoisted above the expression that produces its input.
        // Deferring is only expressible when the matmul is int8, and not for the reason one would
        // guess. FuseTransposeBrgemm runs before this pass, so on f32 it has already folded the
        // output transpose into the Brgemm: the product comes out as [B, M, H, K] while the row
        // sums are [B, H, M, 1], and the two do not broadcast. The int8 path keeps an explicit
        // Transpose -- the FakeQuantize on v blocks the fold -- so there the layouts still agree.
        // Rescaling in the folded layout would need the row sums permuted by the same perm, i.e. a
        // Transpose emitted after TransposeDecomposition has already run. It is not worth it: the
        // multiplies saved are 1000*1000*9*12 = 108M/clip, ~0.5 ms of 90, and they sit fused in the
        // exp loop where they are close to free.
        const bool deferred = consumer && consumer->output(0).get_element_type() == ov::element::i32;

        // Iteration 26 tried summing the QUANTISED operand instead of exp, on the theory that the
        // rounded-away tail biases every row downwards. It is much worse (rel 0.50 vs 0.14, and the
        // centred component decorrelates entirely), and the best-fit global scale stays near 1, so
        // the extra f32 consumer on the u8 tensor perturbs the int8 lowering rather than the ratio.
        const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(exp, normalized_axis);
        ov::snippets::op::ReduceBase::compute_and_set_reduce_subtensors(reduce_sum);
        const auto power = std::make_shared<ov::snippets::op::PowerStatic>(reduce_sum, -1.F);
        lowered::PortDescriptorUtils::set_port_descriptor(power->input(0), subtensor);
        lowered::PortDescriptorUtils::set_port_descriptor(power->output(0), subtensor);

        if (deferred) {
            // An int8 Brgemm accumulates in i32 and the conversion back to real arithmetic is a
            // separate node, so anchor the rescale on the first real-typed value downstream
            // rather than on the accumulator itself.
            ov::Output<ov::Node> anchor = consumer->output(0);
            for (size_t hop = 0; hop < 8 && !anchor.get_element_type().is_real(); ++hop) {
                const auto& targets = anchor.get_target_inputs();
                if (targets.size() != 1) {
                    break;
                }
                anchor = targets.begin()->get_node()->output(0);
            }
            OPENVINO_ASSERT(anchor.get_element_type().is_real(),
                            "OV_SNIPPETS_SOFTMAX=defer found no real-typed value after the Brgemm");
            // Nothing downstream of the Brgemm depends on the row sums any more, so a plain
            // topological sort is free to schedule the ReduceSum after the Brgemm -- which forces
            // the exp tile to be re-read in a second pass and inverts the buffer lifetimes. Pin it.
            consumer->add_control_dependency(power);
            const auto sinks = anchor.get_target_inputs();
            const auto rescale = std::make_shared<ov::op::v1::Multiply>(anchor, power);
            for (const auto& sink : sinks) {
                sink.replace_source_output(rescale->output(0));
            }
            decomposed.insert(decomposed.end(), {exp, reduce_sum, power, rescale});
            copy_runtime_info(softmax, decomposed);
            return ov::replace_node_update_name(softmax, exp);
        }

        const auto multiply = std::make_shared<ov::op::v1::Multiply>(exp, power);
        decomposed.insert(decomposed.end(), {exp, reduce_sum, power, multiply});
        copy_runtime_info(softmax, decomposed);
        return ov::replace_node_update_name(softmax, multiply);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(softmax_m, matcher_name);
    register_matcher(m, callback);
}

}  // namespace ov::snippets::pass

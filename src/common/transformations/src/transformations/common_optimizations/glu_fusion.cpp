// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/glu_fusion.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/glu.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::pass {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v4 = ov::op::v4;
namespace v7 = ov::op::v7;
namespace op_util = ov::op::util;

GLUFusion::GLUFusion() {
    auto last_dim_static = [](const ov::Output<ov::Node>& output) {
        auto out_ps = output.get_node()->get_output_partial_shape(0);
        return out_ps.rank().is_static() && out_ps[out_ps.rank().get_length() - 1].is_static() && out_ps.size() <= 5;
    };

    // Detect GLU decomposition pattern
    // GLU(Xw, Xv, beta) = (Xw * (1.0 + exp(-beta * Xw))) * Xv
    auto data_m = pattern::any_input(last_dim_static);

    // VariadicSplit(X, axis, split_lengths) = Xw, Xv
    auto axis_const_m = pattern::wrap_type<v0::Constant>();
    // Accept any split_lengths source, not only a Constant: a half-split via chunk() lowers
    // split_lengths to a runtime Concat under dynamic shapes (FLUX.2-klein). The exact half-split
    // is verified below from the VariadicSplit's static output shapes instead.
    auto split_lengths_m = pattern::any_input();
    auto variadic_split_m = pattern::wrap_type<v1::VariadicSplit>({data_m, axis_const_m, split_lengths_m});
    variadic_split_m->set_output_size(2);

    // Swish(Xw) = Xw * (1.0 + exp(-beta * Xw))
    auto swish_m = pattern::wrap_type<v4::Swish>({variadic_split_m->output(0)});
    auto gelu_m = pattern::wrap_type<v7::Gelu>({variadic_split_m->output(0)});

    // Mul(Xw, Xv) = Swish(Xw) * Xv
    auto glu_m = std::make_shared<pattern::op::Or>(OutputVector{swish_m, gelu_m});
    auto mul_m = pattern::wrap_type<v1::Multiply>({glu_m, variadic_split_m->output(1)});
    auto mul_relaxed_m = pattern::wrap_type<ov::op::TypeRelaxed<v1::Multiply>>({glu_m, variadic_split_m->output(1)});
    auto mul_or_m = std::make_shared<pattern::op::Or>(OutputVector{mul_m, mul_relaxed_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        OPENVINO_ASSERT(pattern_map.count(mul_or_m));
        OPENVINO_ASSERT(pattern_map.count(swish_m) || pattern_map.count(gelu_m));
        OPENVINO_ASSERT(pattern_map.count(variadic_split_m));
        OPENVINO_ASSERT(pattern_map.count(axis_const_m));
        const auto mul_node = pattern_map.at(mul_or_m).get_node_shared_ptr();
        if (!mul_node || transformation_callback(mul_node))
            return false;

        auto isSwiGLU = pattern_map.count(swish_m);
        auto isGeGLU = pattern_map.count(gelu_m);
        size_t split_to_glu_idx = 0;
        ov::op::internal::GLU::GluType glu_type = ov::op::internal::GLU::GluType::Swish;

        if (isSwiGLU) {
            auto swish = ov::as_type_ptr<v4::Swish>(pattern_map.at(swish_m).get_node_shared_ptr());
            glu_type = ov::op::internal::GLU::GluType::Swish;
            split_to_glu_idx = swish->input_value(0).get_index();

            size_t split_in_idx = ov::is_type<v4::Swish>(mul_node->get_input_node_shared_ptr(0)) ? 1 : 0;
            if (mul_node->input_value(split_in_idx).get_index() == split_to_glu_idx)
                return false;
        } else if (isGeGLU) {
            auto gelu = ov::as_type_ptr<v7::Gelu>(pattern_map.at(gelu_m).get_node_shared_ptr());
            glu_type = (gelu->get_approximation_mode() == ov::op::GeluApproximationMode::ERF)
                           ? ov::op::internal::GLU::GluType::Gelu
                           : ov::op::internal::GLU::GluType::Gelu_Tanh;
            split_to_glu_idx = gelu->input_value(0).get_index();

            size_t split_in_idx = ov::is_type<v7::Gelu>(mul_node->get_input_node_shared_ptr(0)) ? 1 : 0;
            if (mul_node->input_value(split_in_idx).get_index() == split_to_glu_idx)
                return false;
        } else {
            OPENVINO_THROW("'glu_type' not initialized");
        }

        auto variadic_split =
            ov::as_type_ptr<v1::VariadicSplit>(pattern_map.at(variadic_split_m).get_node_shared_ptr());
        auto variadic_split_in_ps = variadic_split->get_input_partial_shape(0);
        auto last_dim = variadic_split_in_ps.rank().get_length() - 1;

        auto axis = ov::as_type_ptr<v0::Constant>(pattern_map.at(axis_const_m).get_node_shared_ptr());
        bool valid_axis_const_values =
            op_util::has_constant_value<int64_t>(axis, -1) || op_util::has_constant_value<int64_t>(axis, last_dim);
        if (!valid_axis_const_values)
            return false;
        auto axis_value = axis->cast_vector<int64_t>()[0];

        // Verify the exact half-split along the last dim from the VariadicSplit's static output
        // shapes. This recognizes both a Constant split_lengths and a runtime-computed one
        // (e.g. a Concat produced by chunk() under dynamic shapes, as in FLUX.2-klein).
        const auto& out0_ps = variadic_split->get_output_partial_shape(0);
        const auto& out1_ps = variadic_split->get_output_partial_shape(1);
        if (!out0_ps.rank().is_static() || !out1_ps.rank().is_static())
            return false;
        const auto& out0_last = out0_ps[out0_ps.rank().get_length() - 1];
        const auto& out1_last = out1_ps[out1_ps.rank().get_length() - 1];
        if (!out0_last.is_static() || !out1_last.is_static())
            return false;
        // Allow only case that exactly splits in half along the last dimension
        auto split_length = variadic_split_in_ps[last_dim].get_length() / 2;
        if (out0_last.get_length() != split_length || out1_last.get_length() != split_length)
            return false;
        auto split_lengths_value = out0_last.get_length();

        auto data = pattern_map.at(data_m);
        auto output_type = m.get_match_root()->get_output_element_type(0);

        auto swiglu = std::make_shared<ov::op::internal::GLU>(data,
                                                              axis_value,
                                                              split_lengths_value,
                                                              glu_type,
                                                              split_to_glu_idx,
                                                              output_type);
        swiglu->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), swiglu);
        ov::replace_node(m.get_match_root(), swiglu);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(mul_or_m, "GLUFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::pass

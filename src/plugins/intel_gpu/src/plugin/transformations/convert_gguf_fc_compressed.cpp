// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_gguf_fc_compressed.hpp"

#include <memory>

#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/fully_connected_compressed.hpp"

namespace ov::intel_gpu {

ConvertGGUFFullyConnectedCompressed::ConvertGGUFFullyConnectedCompressed() {
    using namespace ov::pass::pattern;

    auto fc_m = wrap_type<ov::op::internal::FullyConnectedCompressed>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto fc = ov::as_type_ptr<ov::op::internal::FullyConnectedCompressed>(m.get_match_root());
        if (!fc || transformation_callback(fc)) {
            return false;
        }
        // Only handle the GGUF case: weight (input 1) is an opaque GGUF block type. Non-GGUF
        // FullyConnectedCompressed nodes are already produced as the GPU op by
        // ConvertFullyConnectedToFullyConnectedCompressed and must not be touched here.
        if (!fc->get_input_element_type(1).is_gguf_block()) {
            return false;
        }

        // The generic op carries (X, W, bias, weight_scales, weight_zero_points). For GGUF the scale
        // and zero-point inputs are intentionally empty (the scale lives inside each block). The GPU
        // FC op expects an op::Placeholder for absent optional inputs, not an empty Constant — feeding
        // empty dynamic Constants makes several FCs lower to colliding constant primitive ids
        // ("Different primitive with id ... exists already"). Substitute a freshly-created Placeholder
        // for the bias (when empty) and always for scale/zp.
        auto is_empty_optional = [](const ov::Output<ov::Node>& v) {
            auto c = ov::as_type_ptr<ov::op::v0::Constant>(v.get_node_shared_ptr());
            return c && (v.get_element_type() == ov::element::dynamic ||
                         (v.get_partial_shape().is_static() && ov::shape_size(v.get_shape()) == 0));
        };

        const auto& X = fc->input_value(0);
        const auto& W = fc->input_value(1);
        ov::Output<ov::Node> bias = fc->input_value(2);
        if (is_empty_optional(bias)) {
            bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        }
        // GGUF scale/zp live inside each block; mark them present via a dummy scale Constant so the
        // cldnn fully_connected compressed ctor's non-empty-scale assert passes, while the GGUF kernel
        // ignores these inputs entirely (its get_arguments_desc binds only activation + weight).
        ov::Output<ov::Node> w_scale =
            std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{1}, std::vector<float>{1.0f});
        ov::Output<ov::Node> w_zp = std::make_shared<ov::intel_gpu::op::Placeholder>();

        auto gpu_fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(
            X, W, bias, w_scale, w_zp, fc->get_output_element_type(0));

        gpu_fc->set_friendly_name(fc->get_friendly_name());
        ov::copy_runtime_info(fc, gpu_fc);
        ov::replace_node(fc, gpu_fc);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_m, "ConvertGGUFFullyConnectedCompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transpose_fc_decompression_params.hpp"

#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu {

TransposeFCDecompressionParams::TransposeFCDecompressionParams() {
    using namespace ov::pass::pattern;

    // The input count depends on the zero point and dynamic quantization inputs, so the parameters are checked in the callback
    auto fc_predicate = [](const ov::Output<ov::Node>& output) -> bool {
        const auto fc = ov::as_type_ptr<op::FullyConnectedCompressed>(output.get_node_shared_ptr());
        return fc->get_transpose_b() && fc->get_input_partial_shape(1).size() == 2;
    };
    auto fc_m = wrap_type<op::FullyConnectedCompressed>(fc_predicate);

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto fc = ov::as_type_ptr<op::FullyConnectedCompressed>(m.get_match_root());
        OPENVINO_ASSERT(fc, "TransposeFCDecompressionParams: matched node is expected to be FullyConnectedCompressed");
        // Weights are a Constant, so N is static
        const auto n = fc->get_input_shape(1)[0];

        // [N, G] -> [G, N]; false when the parameter keeps its layout
        auto transpose_param = [&](size_t idx, bool transposed, const std::string& name_suffix) {
            if (!transposed || idx >= fc->get_input_size())
                return false;
            const auto param = ov::as_type_ptr<ov::op::v0::Constant>(fc->get_input_node_shared_ptr(idx));
            if (!param)
                return false;
            const auto& shape = param->get_shape();
            if (shape.size() != 2 || shape[0] != n || shape[1] == 1 || param->get_element_type().bitwidth() < 8)
                return false;
            const auto order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {1, 0});
            auto new_param = ov::util::get_constant_from_source(std::make_shared<ov::op::v1::Transpose>(param, order));
            OPENVINO_ASSERT(new_param, "TransposeFCDecompressionParams: failed to fold the transpose of ", param->get_friendly_name());
            new_param->set_friendly_name(fc->get_friendly_name() + name_suffix);
            ov::copy_runtime_info(param, new_param);
            if (param->get_rt_info().count(ov::WeightlessCacheAttribute::get_type_info_static()) != 0u) {
                ov::copy_weightless_cache_attr(param, new_param);
                new_param->get_rt_info()["gpu_weightless_transpose_order"] = std::vector<size_t>{1, 0};
            }
            fc->input(idx).replace_source_output(new_param);
            return true;
        };

        const bool scale_transposed = transpose_param(3, fc->get_transpose_b_scale(), "_scale");
        if (scale_transposed)
            fc->set_transpose_b_scale(false);
        const bool zp_transposed = transpose_param(4, fc->get_transpose_b_zp(), "_zp");
        if (zp_transposed)
            fc->set_transpose_b_zp(false);
        return scale_transposed || zp_transposed;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_m, "TransposeFCDecompressionParams");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu

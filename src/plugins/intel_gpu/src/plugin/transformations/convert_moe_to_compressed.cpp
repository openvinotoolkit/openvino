// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_moe_to_compressed.hpp"

#include <memory>

#include "intel_gpu/op/moe_compressed.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;

ConvertMOEToMOECompressed::ConvertMOEToMOECompressed() {
    auto reshape_ungroup = [](const ov::Output<ov::Node>& output) {
        auto in_ps = output.get_node()->get_input_partial_shape(0);
        auto out_ps = output.get_node()->get_output_partial_shape(0);
        return in_ps.rank().is_static() && out_ps.rank().is_static() && (in_ps.size() == 4 && out_ps.size() == 3);
    };
    // first proj
    auto compressed_weights_m_0 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto zp_m_0 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));

    auto weight_convert_const_m_0 = wrap_type<ov::op::v0::Convert>({compressed_weights_m_0}, type_matches(ov::element::f16));
    auto zp_convert_const_m_0 = wrap_type<ov::op::v0::Convert>({zp_m_0}, type_matches(ov::element::f16));
    auto sub_m_0 = wrap_type<ov::op::v1::Subtract>({weight_convert_const_m_0, zp_convert_const_m_0});

    auto scale_m_0 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::f16));
    auto mul_m_0 = wrap_type<ov::op::v1::Multiply>({sub_m_0, scale_m_0});

    auto reshape_const_m_0 = wrap_type<ov::op::v0::Constant>();
    auto reshape_m_0 = wrap_type<ov::op::v1::Reshape>({mul_m_0, reshape_const_m_0}, reshape_ungroup);

    auto convert_m_0 = wrap_type<ov::op::v0::Convert>({reshape_m_0}, type_matches(ov::element::f32));

    // second proj
    auto compressed_weights_m_1 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto zp_m_1 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));

    auto weight_convert_const_m_1 = wrap_type<ov::op::v0::Convert>({compressed_weights_m_1}, type_matches(ov::element::f16));
    auto zp_convert_const_m_1 = wrap_type<ov::op::v0::Convert>({zp_m_1}, type_matches(ov::element::f16));
    auto sub_m_1 = wrap_type<ov::op::v1::Subtract>({weight_convert_const_m_1, zp_convert_const_m_1});

    auto scale_m_1 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::f16));
    auto mul_m_1 = wrap_type<ov::op::v1::Multiply>({sub_m_1, scale_m_1});

    auto reshape_const_m_1 = wrap_type<ov::op::v0::Constant>();
    auto reshape_m_1 = wrap_type<ov::op::v1::Reshape>({mul_m_1, reshape_const_m_1}, reshape_ungroup);

    auto convert_m_1 = wrap_type<ov::op::v0::Convert>({reshape_m_1}, type_matches(ov::element::f32));

    // third proj
    auto compressed_weights_m_2 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto zp_m_2 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));

    auto weight_convert_const_m_2 = wrap_type<ov::op::v0::Convert>({compressed_weights_m_2}, type_matches(ov::element::f16));
    auto zp_convert_const_m_2 = wrap_type<ov::op::v0::Convert>({zp_m_2}, type_matches(ov::element::f16));
    auto sub_m_2 = wrap_type<ov::op::v1::Subtract>({weight_convert_const_m_2, zp_convert_const_m_2});

    auto scale_m_2 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::f16));
    auto mul_m_2 = wrap_type<ov::op::v1::Multiply>({sub_m_2, scale_m_2});

    auto reshape_const_m_2 = wrap_type<ov::op::v0::Constant>();
    auto reshape_m_2 = wrap_type<ov::op::v1::Reshape>({mul_m_2, reshape_const_m_2}, reshape_ungroup);

    auto convert_m_2 = wrap_type<ov::op::v0::Convert>({reshape_m_2}, type_matches(ov::element::f32));

    auto hidden_states_m = any_input();
    auto routing_weights_m = any_input();
    auto topk_m = any_input();

    auto moe_root = wrap_type<ov::op::internal::MOE>({hidden_states_m, routing_weights_m, topk_m, convert_m_0, convert_m_1, convert_m_2},
        [](const ov::Output<ov::Node>& output) {
            auto moe = ov::as_type_ptr<ov::op::internal::MOE>(output.get_node_shared_ptr());
            return moe->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto moe = ov::as_type_ptr<ov::op::internal::MOE>(pattern_map.at(moe_root).get_node_shared_ptr());
        if (!moe || transformation_callback(moe)) {
            return false;
        }
        OutputVector args(12);
        args[0] = pattern_map.at(hidden_states_m);
        args[1] = pattern_map.at(routing_weights_m);
        args[2] = pattern_map.at(topk_m);
        args[3] = pattern_map.at(compressed_weights_m_0);
        args[4] = pattern_map.at(scale_m_0);
        args[5] = pattern_map.at(zp_m_0);
        args[6] = pattern_map.at(compressed_weights_m_1);
        args[7] = pattern_map.at(scale_m_1);
        args[8] = pattern_map.at(zp_m_1);
        args[9] = pattern_map.at(compressed_weights_m_2);
        args[10] = pattern_map.at(scale_m_2);
        args[11] = pattern_map.at(zp_m_2);
        ov::intel_gpu::op::MOECompressed::Config config;
        auto weight_shape = pattern_map.at(compressed_weights_m_0).get_shape();
        if (weight_shape.size() != 4) {
            return false;
        }
        auto topk_shape = pattern_map.at(topk_m).get_partial_shape();
        config.hidden_size = weight_shape[2] * weight_shape[3];
        config.inter_size = weight_shape[1];
        config.num_expert = weight_shape[0];
        config.group_size = weight_shape[3];
        config.top_k = topk_shape[topk_shape.size() - 1].get_length();
        config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(args, config);

        auto w0 = pattern_map.at(compressed_weights_m_0).get_node_shared_ptr();
        auto s0 = pattern_map.at(scale_m_0).get_node_shared_ptr();
        auto z0 = pattern_map.at(zp_m_0).get_node_shared_ptr();
        auto w1 = pattern_map.at(compressed_weights_m_1).get_node_shared_ptr();
        auto s1 = pattern_map.at(scale_m_1).get_node_shared_ptr();
        auto z1 = pattern_map.at(zp_m_1).get_node_shared_ptr();
        auto w2 = pattern_map.at(compressed_weights_m_2).get_node_shared_ptr();
        auto s2 = pattern_map.at(scale_m_2).get_node_shared_ptr();
        auto z2 = pattern_map.at(zp_m_2).get_node_shared_ptr();
        ov::enable_keep_const_precision(w0);
        ov::enable_keep_const_precision(s0);
        ov::enable_keep_const_precision(z0);
        ov::enable_keep_const_precision(w1);
        ov::enable_keep_const_precision(s1);
        ov::enable_keep_const_precision(z1);
        ov::enable_keep_const_precision(w2);
        ov::enable_keep_const_precision(s2);
        ov::enable_keep_const_precision(z2);

        moe_compressed->set_friendly_name(moe->get_friendly_name());
        ov::copy_runtime_info(moe, moe_compressed);
        ov::replace_node(moe, moe_compressed);

        std::cout << "ConvertMOEToMOECompressed is hit : num_expert = " << config.num_expert << ", top_k = " << config.top_k
                  << ", hidden_size = " << config.hidden_size << ", inter_size = " << config.inter_size << ", group_size = " << config.group_size << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_root, "ConvertMOEToMOECompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu

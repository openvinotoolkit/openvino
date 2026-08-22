// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kv_compression_fusion.hpp"

#include <limits>

#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/any.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

namespace ov::intel_gpu {
namespace {

bool is_quantized_kv_type(const ov::element::Type& type) {
    return type == ov::element::i8 || type == ov::element::u8 || type == ov::element::i4 ||
           type == ov::element::u4;
}

std::vector<uint64_t> compute_kv_group_sizes(const ov::PartialShape& data_shape,
                                             const ov::PartialShape& scale_shape) {
    if (data_shape.rank().is_dynamic()) {
        return {};
    }

    const size_t rank = data_shape.rank().get_length();
    std::vector<uint64_t> group_sizes(rank, 1);
    if (scale_shape.rank().is_static() && static_cast<size_t>(scale_shape.rank().get_length()) == rank) {
        for (size_t i = 0; i < rank; ++i) {
            const bool scale_is_one = scale_shape[i].is_static() && scale_shape[i].get_length() == 1;
            const bool data_is_one = data_shape[i].is_static() && data_shape[i].get_length() == 1;
            if (scale_is_one && !data_is_one) {
                group_sizes[i] = std::numeric_limits<uint64_t>::max();
            }
        }
    } else if (rank > 0) {
        group_sizes[rank - 1] = std::numeric_limits<uint64_t>::max();
    }
    return group_sizes;
}

}  // namespace

SDPAKVCompressionFusion::SDPAKVCompressionFusion() {
    auto query_m = any_input();

    auto key_quantized_m = any_input();
    auto key_convert_m = wrap_type<ov::op::v0::Convert>({key_quantized_m});
    auto key_zero_point_m = any_input();
    auto key_subtract_m = optional<ov::op::v1::Subtract>({key_convert_m, key_zero_point_m});
    auto key_scale_m = any_input();
    auto key_multiply_m = wrap_type<ov::op::v1::Multiply>({key_subtract_m, key_scale_m});
    auto key_dequantized_m = optional<ov::op::v1::Reshape>({key_multiply_m, any_input()});

    auto value_quantized_m = any_input();
    auto value_convert_m = wrap_type<ov::op::v0::Convert>({value_quantized_m});
    auto value_zero_point_m = any_input();
    auto value_subtract_m = optional<ov::op::v1::Subtract>({value_convert_m, value_zero_point_m});
    auto value_scale_m = any_input();
    auto value_multiply_m = wrap_type<ov::op::v1::Multiply>({value_subtract_m, value_scale_m});
    auto value_dequantized_m = optional<ov::op::v1::Reshape>({value_multiply_m, any_input()});

    auto input_attn_mask_m = any_input();
    auto input_scale_m = any_input();
    auto sdpa_without_attn_mask_m = wrap_type<op::SDPA>({query_m, key_dequantized_m, value_dequantized_m});
    auto sdpa_with_attn_mask_m =
        wrap_type<op::SDPA>({query_m, key_dequantized_m, value_dequantized_m, input_attn_mask_m});
    auto sdpa_with_attn_mask_and_scale_m =
        wrap_type<op::SDPA>({query_m, key_dequantized_m, value_dequantized_m, input_attn_mask_m, input_scale_m});
    auto sdpa_m = std::make_shared<Or>(
        OutputVector{sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto sdpa = ov::as_type_ptr<op::SDPA>(m.get_match_root());

        if (!sdpa || transformation_callback(sdpa)) {
            return false;
        }

        if (sdpa->get_kv_compressed() ||
            !is_quantized_kv_type(pattern_map.at(key_quantized_m).get_element_type()) ||
            !is_quantized_kv_type(pattern_map.at(value_quantized_m).get_element_type())) {
            return false;
        }

        const bool asymmetric = pattern_map.count(key_subtract_m) > 0 && pattern_map.count(value_subtract_m) > 0;
        if ((pattern_map.count(key_subtract_m) > 0) != (pattern_map.count(value_subtract_m) > 0)) {
            return false;
        }

        op::SDPA::QuantizationAttribute config;
        config.quantization_type = asymmetric ? ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric
                                              : ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
        config.output_storage_type = ov::op::internal::DynamicQuantize::OutputStorageType::Planar;
        config.quantization_dt = pattern_map.at(key_quantized_m).get_element_type();
        config.scale_dt = pattern_map.at(key_scale_m).get_element_type();
        config.group_sizes = compute_kv_group_sizes(pattern_map.at(key_quantized_m).get_partial_shape(),
                                                    pattern_map.at(key_scale_m).get_partial_shape());
        config.scales_zp_output_order = std::vector<uint64_t>{0, 1, 2, 3};
        if (asymmetric) {
            config.zp_dt = pattern_map.at(key_zero_point_m).get_element_type();
        }

        ov::OutputVector inputs{sdpa->input_value(0),
                                pattern_map.at(key_quantized_m),
                                pattern_map.at(value_quantized_m)};
        for (size_t i = 3; i < sdpa->get_input_size(); ++i) {
            inputs.push_back(sdpa->input_value(i));
        }
        inputs.push_back(pattern_map.at(key_scale_m));
        inputs.push_back(pattern_map.at(value_scale_m));
        if (asymmetric) {
            inputs.push_back(pattern_map.at(key_zero_point_m));
            inputs.push_back(pattern_map.at(value_zero_point_m));
        }

        auto compressed_sdpa = std::make_shared<op::SDPA>(inputs,
                                                           sdpa->get_causal(),
                                                           sdpa->get_input0_transpose_order(),
                                                           sdpa->get_input1_transpose_order(),
                                                           sdpa->get_input2_transpose_order(),
                                                           sdpa->get_output_transpose_order(),
                                                           config,
                                                           sdpa->get_output_type());
        compressed_sdpa->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), compressed_sdpa);
        ov::replace_node(sdpa, compressed_sdpa);
        return true;
    };

    auto m = std::make_shared<Matcher>(sdpa_m, "SDPAKVCompressionFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu

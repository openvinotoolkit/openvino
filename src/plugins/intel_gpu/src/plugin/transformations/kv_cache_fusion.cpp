// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_fusion.hpp"
#include <memory>

#include "intel_gpu/op/kv_cache.hpp"

#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include "openvino/core/graph_util.hpp"

namespace ov::intel_gpu {

KVCacheFusionMatcher::KVCacheFusionMatcher() {
    using namespace ov::pass::pattern;

    auto past = wrap_type<ov::op::v6::ReadValue>();
    auto convert_past = wrap_type<ov::op::v0::Convert>({past});
    auto gather_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past, convert_past});
    auto beam_idx = wrap_type<ov::op::v0::Parameter>();
    auto gather_past = wrap_type<ov::op::v8::Gather>({gather_input, beam_idx, wrap_type<ov::op::v0::Constant>()});
    auto gather_convert = wrap_type<ov::op::v0::Convert>({gather_past});
    auto dst_idx = wrap_type<ov::op::v0::Parameter>();
    auto gather_update = wrap_type<ov::op::v8::Gather>(); 
    auto update_kv = wrap_type<ov::op::v3::ScatterElementsUpdate>({gather_input, dst_idx, gather_update, wrap_type<ov::op::v0::Constant>()});
    auto start = wrap_type<ov::op::v0::Constant>();
    auto past_seq_len = any_input();
    auto stride = wrap_type<ov::op::v0::Constant>();
    auto step = wrap_type<ov::op::v0::Constant>();
    auto slice_axes = wrap_type<ov::op::v0::Constant>();
    auto trim_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{gather_input, gather_past, gather_convert, update_kv});
    auto trim_past = wrap_type<ov::op::v8::Slice>({trim_input, start, past_seq_len, step, slice_axes});
    auto trim_past2 = wrap_type<ov::op::v1::StridedSlice>({trim_input, start, past_seq_len, stride});
    auto concat_past_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{trim_input, trim_past, trim_past2});
    auto concat = wrap_type<ov::op::v0::Concat>({concat_past_input, any_input()});
    auto convert_present = wrap_type<ov::op::v0::Convert>({concat});
    auto present_input = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{concat, convert_present});
    auto present = wrap_type<ov::op::v6::Assign>({present_input});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto concat_node = ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(concat).get_node_shared_ptr());

        auto past_node = ov::as_type_ptr<ov::op::v6::ReadValue>(pattern_map.at(past).get_node_shared_ptr());
        auto present_node = ov::as_type_ptr<ov::op::v6::Assign>(pattern_map.at(present).get_node_shared_ptr());

        if (past_node->get_variable_id() != present_node->get_variable_id())
            return false;

        // TODO: Support conversion internally
        if (ov::is_type<ov::opset8::Gather>(concat_past_input)) {
            if (!concat_node || concat_node->get_output_element_type(0) != past_node->get_output_element_type(0))
                return false;
        }

        auto variable = past_node->get_variable();
        auto concat_axis = concat_node->get_axis();

        std::shared_ptr<ov::Node> variable_initializer = nullptr;
        std::shared_ptr<ov::Node> kv_cache_node = nullptr;
        if (past_node->get_input_size() == 1) {
            variable_initializer = past_node->get_input_node_shared_ptr(0);
        }

        // Replace common ReadValue op with a custom one as common one expects paired Assign operation which is removed by this transform
        auto new_read_value_node = variable_initializer ? std::make_shared<ov::intel_gpu::op::ReadValue>(variable_initializer->output(0), variable)
                                                        : std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        new_read_value_node->set_friendly_name(past_node->get_friendly_name());
        ov::copy_runtime_info(past_node, new_read_value_node);
        ov::replace_node(past_node, new_read_value_node);

        const bool has_beam_idx = pattern_map.count(gather_past) > 0;
        const bool has_update_kv = pattern_map.count(update_kv) > 0;
        const bool has_slice = pattern_map.count(trim_past) > 0;
        const bool has_strided_slice = pattern_map.count(trim_past2) > 0;
        const bool has_trim = has_slice || has_strided_slice;

        const auto adjust_axis_to_positive = [&new_read_value_node](auto axis) ->std::optional<uint64_t> {
            if (axis >= 0) {
                return static_cast<uint64_t>(axis);
            } else {
                const auto input_rank = new_read_value_node->get_output_partial_shape(0).rank();
                if (input_rank.is_static()) {
                    const auto adjusted_axis = input_rank.get_interval().get_min_val() + axis;
                    if (adjusted_axis >= 0) {
                        return static_cast<uint64_t>(adjusted_axis);
                    }
                }
            }
            return std::nullopt;
        };
        std::optional<uint64_t> target_concat_axis = adjust_axis_to_positive(concat_axis);
        OPENVINO_ASSERT(target_concat_axis.has_value(), "concat_axis should be valid, get: ", concat_axis);

        std::shared_ptr<ov::Node> past_seq_len_node;
        if (has_trim) {
            past_seq_len_node = pattern_map.at(past_seq_len).get_node_shared_ptr();
            // StridedSlice uses multi-dim for end tensor, extract only the slice dim
            if (has_strided_slice) {
                const auto strided_slice = ov::as_type_ptr<ov::op::v1::StridedSlice>(concat_node->input_value(0).get_node_shared_ptr());
                if (!strided_slice)
                    return false;
                const auto begin_mask = strided_slice->get_begin_mask();
                const auto end_mask = strided_slice->get_end_mask();
                // begin/end mask should be the same and only last element is 0 (being sliced)
                if (begin_mask != end_mask || begin_mask.empty()) {
                    return false;
                }
                if (static_cast<size_t>(std::count(begin_mask.begin(), begin_mask.end(), 1)) != (begin_mask.size() - 1) || begin_mask.back() != 0) {
                    return false;
                }
                // slice start and stride should be all 1
                const auto slice_start = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(start).get_node_shared_ptr());
                if (const auto start_data = slice_start->cast_vector<int64_t>(); std::any_of(start_data.begin(), start_data.end(), [](const auto val) {
                        return val != 1;
                    })) {
                    return false;
                }
                const auto slice_stride = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(stride).get_node_shared_ptr());
                if (const auto stride_data = slice_stride->cast_vector<int64_t>(); std::any_of(stride_data.begin(), stride_data.end(), [](const auto val) {
                        return val != 1;
                    })) {
                    return false;
                }
                // sliced axis should be the same with concat_axis
                if (begin_mask.size() != *target_concat_axis + 1) {
                    return false;
                }
                const auto slice_axis = ov::op::v0::Constant::create(element::i64, Shape{1}, {concat_axis});
                const auto gather_axis = ov::op::v0::Constant::create(element::i64, Shape{1}, {0});
                past_seq_len_node = std::make_shared<ov::op::v8::Gather>(past_seq_len_node, slice_axis, gather_axis);
            } else {
                // slice start should be 0 and step should be 1
                const auto slice_start = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(start).get_node_shared_ptr());
                if (const auto start_data = slice_start->cast_vector<int64_t>(); start_data.size() != 1 || start_data[0] != 0) {
                    return false;
                }
                const auto slice_step = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(step).get_node_shared_ptr());
                if (const auto step_data = slice_step->cast_vector<int64_t>(); step_data.size() != 1 || step_data[0] != 1) {
                    return false;
                }
                // slice axis should be the same as concat_axis
                const auto slice_axis = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(slice_axes).get_node_shared_ptr());
                if (const auto axis_data = slice_axis->cast_vector<int64_t>();
                    axis_data.size() != 1 || adjust_axis_to_positive(axis_data[0]) != *target_concat_axis) {
                    return false;
                }
            }
        }

        const auto input0 = has_beam_idx ? pattern_map.at(gather_past).get_node_shared_ptr() : new_read_value_node;
        if (has_update_kv) {
            OPENVINO_ASSERT(has_trim);
            kv_cache_node = std::make_shared<op::KVCache>(input0,
                                                          concat_node->input(1).get_source_output(),
                                                          past_seq_len_node,
                                                          pattern_map.at(dst_idx).get_node_shared_ptr(),
                                                          pattern_map.at(gather_update).get_node_shared_ptr(),
                                                          variable,
                                                          concat_axis,
                                                          new_read_value_node->get_output_element_type(0));
        } else if (has_trim) {
            kv_cache_node = std::make_shared<op::KVCache>(input0,
                                                          concat_node->input(1).get_source_output(),
                                                          past_seq_len_node,
                                                          variable,
                                                          concat_axis,
                                                          new_read_value_node->get_output_element_type(0));
        } else {
            kv_cache_node = std::make_shared<op::KVCache>(input0,
                                                          concat_node->input(1).get_source_output(),
                                                          variable,
                                                          concat_axis,
                                                          new_read_value_node->get_output_element_type(0));
        }
        kv_cache_node->set_friendly_name(concat_node->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), kv_cache_node);
        ov::replace_node(concat_node, kv_cache_node);

        if (pattern_map.count(convert_present) > 0) {
            present_node->set_argument(0, kv_cache_node->output(0));
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(present, "KVCacheFusionMatcher");
    this->register_matcher(m, callback);
}

bool KVCacheFusion::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool res = pass::GraphRewrite::run_on_model(m);
    if (res) {
        ov::SinkVector sinks = m->get_sinks();
        for (auto& sink : sinks) {
            if (sink && sink->get_input_node_ptr(0)->get_type_info() == op::KVCache::get_type_info_static()) {
                m->remove_sink(sink);
            }
        }
    }

    return res;
}

KVCacheFusion::KVCacheFusion() {
    add_matcher<ov::intel_gpu::KVCacheFusionMatcher>();
}

}  // namespace ov::intel_gpu

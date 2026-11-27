// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kv_cache_fusion.hpp"
#include <algorithm>
#include <memory>

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/qsa_indexer_in_place.hpp"

#include "intel_gpu/op/read_value.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sink.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/opsets/opset8_decl.hpp"
#include "plugin/transformations/utils.hpp"
#include "openvino/core/graph_util.hpp"

namespace ov::intel_gpu {

KVCacheFusionMatcher::KVCacheFusionMatcher() {
    using namespace ov::pass::pattern;

    // kv_cache has no i32 implementation, so integer states such as cache positions stay unfused.
    auto past = wrap_type<ov::op::v6::ReadValue>([](const ov::Output<ov::Node>& output) {
        return output.get_element_type().is_real();
    });
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

        if (past_node->get_variable_id() != present_node->get_variable_id()) {
            return false;
        }

        // TODO: Support conversion internally
        if (ov::is_type<ov::opset8::Gather>(concat_past_input)) {
            if (!concat_node || concat_node->get_output_element_type(0) != past_node->get_output_element_type(0)) {
                return false;
            }
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
            }
            const auto input_rank = new_read_value_node->get_output_partial_shape(0).rank();
            if (input_rank.is_static()) {
                const auto adjusted_axis = input_rank.get_interval().get_min_val() + axis;
                if (adjusted_axis >= 0) {
                    return static_cast<uint64_t>(adjusted_axis);
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
                if (!strided_slice) {
                    return false;
                }
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

namespace {

// Axes of the QSA indexer block cache: the variable is [B, blocks, D_idx], the indexer reads it as [B, 1, blocks, D_idx].
constexpr int64_t cache_batch_axis = 0;
constexpr int64_t cache_block_axis = 1;
constexpr int64_t cache_dim_axis = 2;
constexpr int64_t indexer_head_axis = 1;
constexpr int64_t indexer_block_axis = 2;
constexpr int64_t indexer_dim_axis = 3;

/// Whether the constant `value` holds the single value `expected`.
bool is_scalar_value(const ov::Output<ov::Node>& value, int64_t expected) {
    const auto constant = ov::as_type_ptr<ov::op::v0::Constant>(value.get_node_shared_ptr());
    if (!constant || ov::shape_size(constant->get_shape()) != 1) {
        return false;
    }
    return constant->cast_vector<int64_t>()[0] == expected;
}

}  // namespace

QSAIndexerBlockCacheFusionMatcher::QSAIndexerBlockCacheFusionMatcher() {
    using namespace ov::pass::pattern;
    using QSAIndexer = ov::op::internal::QSAIndexer;

    auto past = wrap_type<ov::op::v6::ReadValue>();
    auto beam_idx = wrap_type<ov::op::v0::Parameter>();
    auto gather_past = wrap_type<ov::op::v8::Gather>({past, beam_idx, wrap_type<ov::op::v0::Constant>()});
    auto cache = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past, gather_past});
    auto with_head = wrap_type<ov::op::v0::Unsqueeze, ov::op::v1::Reshape>({cache, any_input()}, rank_equals(4) && consumers_count(1));
    auto zero_blocks = any_input();
    auto grown = wrap_type<ov::op::v0::Concat>({with_head, zero_blocks}, consumers_count(1));
    OutputVector indexer_inputs(QSAIndexer::InputIdx::COUNT);
    for (auto& input : indexer_inputs) {
        input = any_input();
    }
    indexer_inputs[QSAIndexer::InputIdx::BLOCK_INDEX_KEYS_CACHE] = grown;
    auto indexer = wrap_type_strict_index<QSAIndexer>(indexer_inputs);
    auto squeezed = wrap_type<ov::op::v0::Squeeze, ov::op::v1::Reshape>({indexer->output(QSAIndexer::OutputIdx::BLOCK_INDEX_KEYS_CACHE), any_input()},
                                                                        rank_equals(3) && consumers_count(1));
    auto slice_start = wrap_type<ov::op::v0::Constant>();
    auto slice_step = wrap_type<ov::op::v0::Constant>();
    auto slice_axis = wrap_type<ov::op::v0::Constant>();
    auto sliced = wrap_type<ov::op::v8::Slice>({squeezed, slice_start, any_input(), slice_step, slice_axis}, consumers_count(1));
    auto present_input = optional<ov::op::v0::Convert>({sliced}, consumers_count(1));
    auto present = wrap_type<ov::op::v6::Assign>({present_input});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto past_node = ov::as_type_ptr<ov::op::v6::ReadValue>(pattern_map.at(past).get_node_shared_ptr());
        const auto present_node = ov::as_type_ptr<ov::op::v6::Assign>(m.get_match_root());
        const auto indexer_node = ov::as_type_ptr<QSAIndexer>(pattern_map.at(indexer).get_node_shared_ptr());
        const auto concat_node = ov::as_type_ptr<ov::op::v0::Concat>(pattern_map.at(grown).get_node_shared_ptr());
        if (!past_node || !present_node || !indexer_node || !concat_node || transformation_callback(indexer_node) ||
            past_node->get_variable_id() != present_node->get_variable_id()) {
            return false;
        }

        // A [B, blocks, D_idx] cache of one index key head, given the head axis 1 and grown along the blocks.
        const ov::Dimension head_dim(static_cast<int64_t>(indexer_node->get_config().indexer_head_dim));
        const auto& cache_shape = past_node->get_output_partial_shape(0);
        const auto& with_head_shape = pattern_map.at(with_head).get_partial_shape();
        const auto cache_type = past_node->get_output_element_type(0);
        if (cache_shape.rank() != 3 || cache_shape[cache_dim_axis] != head_dim || with_head_shape[indexer_head_axis] != 1 ||
            with_head_shape[indexer_dim_axis] != head_dim || ov::util::normalize(concat_node->get_axis(), 4) != indexer_block_axis ||
            cache_type != indexer_node->get_input_element_type(QSAIndexer::InputIdx::BLOCK_INDEX_KEYS_CACHE) ||
            pattern_map.at(zero_blocks).get_element_type() != cache_type) {
            return false;
        }
        // The Assign stores the leading blocks of the updated cache, which has no other reader.
        if (!is_scalar_value(pattern_map.at(slice_start), 0) || !is_scalar_value(pattern_map.at(slice_step), 1) ||
            !is_scalar_value(pattern_map.at(slice_axis), cache_block_axis) ||
            indexer_node->get_output_target_inputs(QSAIndexer::OutputIdx::BLOCK_INDEX_KEYS_CACHE).size() != 1) {
            return false;
        }
        // The past cache is otherwise only read for its shape (the block count FuseQSAIndexer grows it by).
        auto* const with_head_node = pattern_map.at(with_head).get_node();
        const auto gather_node =
            pattern_map.count(gather_past) ? ov::as_type_ptr<ov::op::v8::Gather>(pattern_map.at(gather_past).get_node_shared_ptr()) : nullptr;
        if (gather_node) {
            if (gather_node->get_batch_dims() != 0 || ov::util::normalize(gather_node->get_axis(), 3) != cache_batch_axis ||
                !read_only_by(past_node->output(0), gather_node.get()) || !read_only_by(gather_node->output(0), with_head_node)) {
                return false;
            }
        } else if (!read_only_by(past_node->output(0), with_head_node)) {
            return false;
        }

        // The common ReadValue expects its paired Assign, which is removed.
        const auto variable = past_node->get_variable();
        const auto read_value = past_node->get_input_size() == 1 ? std::make_shared<ov::intel_gpu::op::ReadValue>(past_node->input_value(0), variable)
                                                                 : std::make_shared<ov::intel_gpu::op::ReadValue>(variable);
        read_value->set_friendly_name(past_node->get_friendly_name());
        ov::copy_runtime_info(past_node, read_value);
        ov::replace_node(past_node, read_value);

        const auto head_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {indexer_head_axis});
        const auto new_blocks = std::make_shared<ov::op::v0::Squeeze>(pattern_map.at(zero_blocks), head_axis);
        std::shared_ptr<op::KVCache> kv_cache;
        if (gather_node) {
            kv_cache =
                std::make_shared<op::KVCache>(read_value, new_blocks, gather_node->input_value(1), variable, cache_block_axis, cache_batch_axis, cache_type);
            kv_cache->get_rt_info()[op::KV_CACHE_REQUIRES_IDENTITY_BEAM_IDX] = true;
            // The shape reads of the gathered cache remain: with an identity beam_idx its shape is the one of the past cache.
            gather_node->output(0).replace(read_value->output(0));
        } else {
            kv_cache = std::make_shared<op::KVCache>(read_value, new_blocks, variable, cache_block_axis, cache_type);
        }
        kv_cache->set_friendly_name(concat_node->get_friendly_name());

        auto inputs = indexer_node->input_values();
        inputs[QSAIndexer::InputIdx::BLOCK_INDEX_KEYS_CACHE] = kv_cache->output(0);
        const auto in_place = std::make_shared<op::QSAIndexerInPlace>(inputs, indexer_node->get_config());
        in_place->set_friendly_name(indexer_node->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), ov::NodeVector{head_axis, new_blocks, kv_cache, in_place});
        for (const auto out : {QSAIndexer::OutputIdx::SELECTED_BLOCK_INDICES,
                               QSAIndexer::OutputIdx::SELECTED_BLOCK_COUNT,
                               QSAIndexer::OutputIdx::PAST_INCOMPLETE_RAW_K,
                               QSAIndexer::OutputIdx::PAST_INCOMPLETE_ROPE}) {
            indexer_node->output(out).replace(in_place->output(out));
        }
        // KVCacheFusion removes the Assign of a KVCache.
        present_node->set_argument(0, kv_cache->output(0));
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(present, "QSAIndexerBlockCacheFusionMatcher");
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
    add_matcher<ov::intel_gpu::QSAIndexerBlockCacheFusionMatcher>();
}

}  // namespace ov::intel_gpu

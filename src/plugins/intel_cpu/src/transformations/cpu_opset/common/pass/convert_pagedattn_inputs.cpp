// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_pagedattn_inputs.hpp"

#include <cstdint>
#include <memory>
#include <transformations/utils/gen_pattern.hpp>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/util/log.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov::gen_pattern;

ov::intel_cpu::ConvertPagedAttnInputs::ConvertPagedAttnInputs(const Config& config) : config(config) {
    MATCHER_SCOPE(ConvertPagedAttnInputs);

    auto Q = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto K = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto V = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto key_cache_0 = makePattern<ov::op::v0::Parameter>({});
    auto value_cache_0 = makePattern<ov::op::v0::Parameter>({});
    auto past_lens = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto subsequence_begins = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto block_indices = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto block_indices_begins = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto scale = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto sliding_window = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto alibi_slopes = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto max_context_len = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto rotated_block_indices = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto rotation_deltas = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());
    auto rotation_trig_lut = ov::pass::pattern::any_input(ov::pass::pattern::has_static_rank());

    auto pa_1 = makePattern<op::PagedAttentionExtension>({Q,
                                                          K,
                                                          V,
                                                          key_cache_0,
                                                          value_cache_0,
                                                          past_lens,
                                                          subsequence_begins,
                                                          block_indices,
                                                          block_indices_begins,
                                                          scale,
                                                          sliding_window,
                                                          alibi_slopes,
                                                          max_context_len});

    auto pa_2 = makePattern<op::PagedAttentionExtension>({Q,
                                                          K,
                                                          V,
                                                          key_cache_0,
                                                          value_cache_0,
                                                          past_lens,
                                                          subsequence_begins,
                                                          block_indices,
                                                          block_indices_begins,
                                                          scale,
                                                          sliding_window,
                                                          alibi_slopes,
                                                          max_context_len,
                                                          rotated_block_indices,
                                                          rotation_deltas,
                                                          rotation_trig_lut});
    auto result = pa_1 | pa_2;
    auto key_cache_prec = config.keyCachePrecision;
    auto value_cache_prec = config.valueCachePrecision;
    auto key_cache_gs = config.keyCacheGroupSize;
    auto value_cache_gs = config.valueCacheGroupSize;
    auto infer_prec = config.inferencePrecision;
    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto pa_op = m.get_match_root();
        auto key_cache = ov::as_type_ptr<ov::op::v0::Parameter>(pa_op->get_input_node_shared_ptr(3));
        auto value_cache = ov::as_type_ptr<ov::op::v0::Parameter>(pa_op->get_input_node_shared_ptr(4));
        auto format_cache_precision = [](ov::element::Type cache_precision, ov::element::Type infer_precision) {
            return cache_precision == ov::element::f16 && infer_precision == ov::element::bf16 ? infer_precision
                                                                                               : cache_precision;
        };
        auto init_cache_shape = [&](const size_t head_nums,
                                    const size_t head_size,
                                    const ov::element::Type precision,
                                    const size_t group_size,
                                    const bool bychannel) {
            ov::Dimension::value_type block_size = 32;
            ov::Dimension::value_type _head_nums = head_nums;
            ov::Dimension::value_type _head_size = head_size;
            ov::Dimension::value_type _group_size = group_size;
            _group_size = _group_size ? _group_size : _head_size;
            if (!bychannel) {
                if (_head_size % _group_size != 0) {
                    OPENVINO_THROW("cache head_size ", head_size, "cannot be divided by group_size ", group_size);
                }
            }
            size_t group_num = _head_size / _group_size;
            if (precision == ov::element::u8) {
                if (bychannel) {
                    block_size += 2 * sizeof(float);
                } else {
                    _head_size += sizeof(float) * 2 * group_num;
                }
            } else if (precision == ov::element::u4) {
                _head_size += sizeof(float) * 2 * group_num * 2;
            }
            return ov::PartialShape{-1, _head_nums, block_size, _head_size};
        };
        auto key_cache_precision = format_cache_precision(key_cache_prec, infer_prec);
        auto value_cache_precision = format_cache_precision(value_cache_prec, infer_prec);
        key_cache->set_element_type(key_cache_precision);
        value_cache->set_element_type(value_cache_precision);
        if (!pa_op->get_rt_info().count("num_k_heads") || !pa_op->get_rt_info().count("k_head_size") ||
            !pa_op->get_rt_info().count("num_v_heads") || !pa_op->get_rt_info().count("num_v_heads")) {
            OPENVINO_DEBUG("PagedAttn ",
                           pa_op->get_friendly_name(),
                           " doesn't have rtinfo for num_k_heads/k_head_size/num_v_heads/num_v_heads");
            return false;
        }
        const auto key_cache_shape = init_cache_shape(pa_op->get_rt_info()["num_k_heads"].as<size_t>(),
                                                      pa_op->get_rt_info()["k_head_size"].as<size_t>(),
                                                      key_cache_precision,
                                                      key_cache_gs,
                                                      false);
        const auto value_cache_shape = init_cache_shape(pa_op->get_rt_info()["num_v_heads"].as<size_t>(),
                                                        pa_op->get_rt_info()["v_head_size"].as<size_t>(),
                                                        value_cache_precision,
                                                        value_cache_gs,
                                                        false);

        key_cache->set_partial_shape(key_cache_shape);
        value_cache->set_partial_shape(value_cache_shape);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

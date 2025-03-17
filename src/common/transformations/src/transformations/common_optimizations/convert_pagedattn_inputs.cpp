// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_pagedattn_inputs.hpp"

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

ov::pass::ConvertPagedAttnInputs::ConvertPagedAttnInputs(const KVCacheConfig& config, UpdateShapeFunc func)
    : m_config(config),
      m_update_shape_func(std::move(func)) {
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
                                    const size_t block_size,
                                    const ov::element::Type precision,
                                    const size_t group_size,
                                    const bool bychannel,
                                    const std::vector<size_t>& orders) {
            ov::Dimension::value_type _block_size = block_size;
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
            m_update_shape_func(precision, bychannel, group_num, _head_size, _block_size);

            auto block_shape = ov::PartialShape::dynamic(4);
            block_shape[orders[0]] = -1;
            block_shape[orders[1]] = _head_nums;
            block_shape[orders[2]] = _block_size;
            block_shape[orders[3]] = _head_size;

            return block_shape;
        };
        auto key_cache_precision = format_cache_precision(m_config.keyCachePrecision, m_config.inferencePrecision);
        auto value_cache_precision = format_cache_precision(m_config.valueCachePrecision, m_config.inferencePrecision);
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
                                                      m_config.keyCacheBlockSize,
                                                      key_cache_precision,
                                                      m_config.keyCacheGroupSize,
                                                      m_config.keyCacheQuantBychannel,
                                                      m_config.keyCacheDimOrder);
        const auto value_cache_shape = init_cache_shape(pa_op->get_rt_info()["num_v_heads"].as<size_t>(),
                                                        pa_op->get_rt_info()["v_head_size"].as<size_t>(),
                                                        m_config.valueCacheBlockSize,
                                                        value_cache_precision,
                                                        m_config.valueCacheGroupSize,
                                                        m_config.valueCacheQuantBychannel,
                                                        m_config.valueCacheDimOrder);

        key_cache->set_partial_shape(key_cache_shape);
        value_cache->set_partial_shape(value_cache_shape);

        key_cache->validate_and_infer_types();
        value_cache->validate_and_infer_types();
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

void ov::pass::ConvertPagedAttnInputs::setKVCacheConfig(const KVCacheConfig& config) {
    m_config = config;
}

const ov::pass::ConvertPagedAttnInputs::KVCacheConfig& ov::pass::ConvertPagedAttnInputs::getKVCacheConfig() const {
    return m_config;
}

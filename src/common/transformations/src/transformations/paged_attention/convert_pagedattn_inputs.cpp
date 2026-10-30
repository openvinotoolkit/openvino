// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/convert_pagedattn_inputs.hpp"

#include <cstdint>
#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/log.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass;

namespace v0 = ov::op::v0;

namespace ov::pass {

ConvertPagedAttnInputs::ConvertPagedAttnInputs(const KVCacheConfig& config,
                                               UpdateShapeFunc func,
                                               UpdatePrecisionFunc update_precision_func)
    : m_config(config),
      m_update_shape_func(std::move(func)),
      m_update_precision_func(std::move(update_precision_func)) {
    MATCHER_SCOPE(ConvertPagedAttnInputs);

    auto paged_op = pattern::wrap_type<ov::op::PagedAttentionExtension,
                                       ov::op::internal::PagedCausalConv1D,
                                       ov::op::internal::PagedGatedDeltaNet,
                                       ov::op::internal::PagedSelectiveSSM,
                                       ov::op::internal::PagedSparseAttention,
                                       ov::op::internal::PagedQSAIndexer>();
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto root = m.get_match_root();
        bool status = false;

        if (const auto pa_op = ov::as_type_ptr<ov::op::PagedAttentionExtension>(root)) {
            auto key_cache = ov::as_type_ptr<v0::Parameter>(pa_op->get_input_node_shared_ptr(3));
            auto value_cache = ov::as_type_ptr<v0::Parameter>(pa_op->get_input_node_shared_ptr(4));
            if (!key_cache || !value_cache) {
                return false;
            }
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
                // Update head_size and block_size by precision and quantizing channel mode
                m_update_shape_func(precision, bychannel, group_num, _head_size, _block_size);

                auto block_shape = ov::PartialShape::dynamic(4);
                block_shape[orders[0]] = -1;
                block_shape[orders[1]] = _head_nums;
                block_shape[orders[2]] = _block_size;
                block_shape[orders[3]] = _head_size;

                return block_shape;
            };
            auto key_cache_precision = format_cache_precision(m_config.keyCachePrecision, m_config.inferencePrecision);
            auto value_cache_precision =
                format_cache_precision(m_config.valueCachePrecision, m_config.inferencePrecision);
            key_cache->set_element_type(key_cache_precision);
            value_cache->set_element_type(value_cache_precision);
            enable_keep_const_precision(key_cache);
            enable_keep_const_precision(value_cache);
            if (pa_op->get_rt_info().count("num_k_heads") && pa_op->get_rt_info().count("k_head_size") &&
                pa_op->get_rt_info().count("num_v_heads") && pa_op->get_rt_info().count("v_head_size")) {
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
                status = true;
            } else {
                OPENVINO_DEBUG("PagedAttn ",
                               pa_op->get_friendly_name(),
                               " doesn't have rtinfo for num_k_heads/k_head_size/num_v_heads/v_head_size");
                status = false;
            }

            if (m_update_precision_func) {
                m_update_precision_func(key_cache_precision);
                m_update_precision_func(value_cache_precision);
                key_cache->set_element_type(key_cache_precision);
                value_cache->set_element_type(value_cache_precision);
                enable_keep_const_precision(key_cache);
                enable_keep_const_precision(value_cache);
            }

            // Prevent the ConvertPrecision pass from converting sub-byte cache
            // precisions (e.g. u4→u8). The PA executor handles quantized caches natively.
            if (key_cache_precision.bitwidth() < 8) {
                enable_keep_const_precision(key_cache);
            }
            if (value_cache_precision.bitwidth() < 8) {
                enable_keep_const_precision(value_cache);
            }

            key_cache->validate_and_infer_types();
            value_cache->validate_and_infer_types();
        }

        if (const auto paged_conv = ov::as_type_ptr<ov::op::internal::PagedCausalConv1D>(root)) {
            auto conv_state_table = ov::as_type_ptr<v0::Parameter>(paged_conv->get_input_node_shared_ptr(1));
            if (!conv_state_table) {
                return false;
            }

            auto conv_cache_precision = m_config.inferencePrecision;
            if (m_update_precision_func) {
                m_update_precision_func(conv_cache_precision);
            }

            conv_state_table->set_element_type(conv_cache_precision);
            enable_keep_const_precision(conv_state_table);
            conv_state_table->validate_and_infer_types();
            return true;
        }

        if (const auto paged_gdn = ov::as_type_ptr<ov::op::internal::PagedGatedDeltaNet>(root)) {
            auto gated_delta_state_table = ov::as_type_ptr<v0::Parameter>(paged_gdn->get_input_node_shared_ptr(3));
            if (!gated_delta_state_table) {
                return false;
            }

            auto gated_delta_cache_precision = m_config.inferencePrecision;
            if (m_update_precision_func) {
                m_update_precision_func(gated_delta_cache_precision);
            }

            gated_delta_state_table->set_element_type(gated_delta_cache_precision);
            enable_keep_const_precision(gated_delta_state_table);
            gated_delta_state_table->validate_and_infer_types();
            return true;
        }

        if (const auto paged_ssm = ov::as_type_ptr<ov::op::internal::PagedSelectiveSSM>(root)) {
            auto ssm_state_table = ov::as_type_ptr<v0::Parameter>(paged_ssm->get_input_node_shared_ptr(5));
            if (!ssm_state_table) {
                return false;
            }

            auto ssm_cache_precision = m_config.inferencePrecision;
            if (m_update_precision_func) {
                m_update_precision_func(ssm_cache_precision);
            }

            ssm_state_table->set_element_type(ssm_cache_precision);
            enable_keep_const_precision(ssm_state_table);
            ssm_state_table->validate_and_infer_types();
            return true;
        }

        // The caches of the Qwen Sparse Attention ops keep the layout of their specification and the inference
        // precision (the sparse kernels have no compressed caches). Their head axes are set by PagedQSAFusion, the
        // axis of the tokens of a block is set here.
        auto set_cache = [&](const ov::Output<ov::Node>& input, size_t block_axis, size_t block_extent) {
            const auto cache = ov::as_type_ptr<v0::Parameter>(input.get_node_shared_ptr());
            if (!cache || cache->get_partial_shape().rank().is_dynamic() ||
                block_axis >= cache->get_partial_shape().size()) {
                return false;
            }
            auto cache_precision = m_config.inferencePrecision;
            if (m_update_precision_func) {
                m_update_precision_func(cache_precision);
            }
            auto shape = cache->get_partial_shape();
            shape[block_axis] = static_cast<int64_t>(block_extent);
            cache->set_element_type(cache_precision);
            cache->set_partial_shape(shape);
            enable_keep_const_precision(cache);
            cache->validate_and_infer_types();
            return true;
        };

        if (const auto sparse_attention = ov::as_type_ptr<ov::op::internal::PagedSparseAttention>(root)) {
            // key_cache / value_cache: [num_blocks, Hkv, block_size, S]
            return set_cache(sparse_attention->input_value(3), 2, m_config.keyCacheBlockSize) &&
                   set_cache(sparse_attention->input_value(4), 2, m_config.valueCacheBlockSize);
        }

        if (const auto indexer = ov::as_type_ptr<ov::op::internal::PagedQSAIndexer>(root)) {
            using InputIdx = ov::op::internal::PagedQSAIndexer::InputIdx;
            // The index caches share the page table of the key cache; a page holds block_size / r QSA blocks.
            const size_t block_size = m_config.keyCacheBlockSize;
            const size_t key_block_size = indexer->get_config().key_block_size;
            OPENVINO_ASSERT(key_block_size > 0 && block_size % key_block_size == 0,
                            "The KV cache block size ",
                            block_size,
                            " is not a multiple of the QSA key block size ",
                            key_block_size);
            const size_t blocks_per_page = block_size / key_block_size;
            return set_cache(indexer->input_value(InputIdx::INDEX_KEY_CACHE), 2, block_size) &&
                   set_cache(indexer->input_value(InputIdx::INDEX_BLOCK_ROPE_CACHE), 1, blocks_per_page) &&
                   set_cache(indexer->input_value(InputIdx::BLOCK_INDEX_KEYS_CACHE), 2, blocks_per_page);
        }

        return status;
    };

    auto m = std::make_shared<pattern::Matcher>(paged_op, matcher_name);
    this->register_matcher(m, callback);
}

void ConvertPagedAttnInputs::setKVCacheConfig(const KVCacheConfig& config) {
    m_config = config;
}

const ConvertPagedAttnInputs::KVCacheConfig& ConvertPagedAttnInputs::getKVCacheConfig() const {
    return m_config;
}

}  // namespace ov::pass

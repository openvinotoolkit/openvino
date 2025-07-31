// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>
#include <vector>

#include "openvino/core/dimension.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/attr_types.hpp"

namespace ov::test::utils {
/**
 * @brief Creates a state initializer node for KV cache initialization
 * @param input Input node to derive shape from
 * @param element_type Target element type
 * @param kv_cache_size KV cache partial shape
 * @param qkv_order Order of QKV dimensions
 * @return Shared pointer to the created node
 */
std::shared_ptr<ov::Node> make_state_initializer(ov::Output<ov::Node> input,
                                                 ov::element::Type_t element_type,
                                                 ov::PartialShape kv_cache_size,
                                                 std::vector<int64_t> qkv_order);

/**
 * @brief Creates an attention mask for causal attention
 * @param q Query tensor
 * @param k Key tensor
 * @param element_type Target element type
 * @param qkv_order Order of QKV dimensions
 * @return Shared pointer to the created attention mask node
 */
std::shared_ptr<ov::Node> make_attention_mask(ov::Output<ov::Node> q,
                                              ov::Output<ov::Node> k,
                                              ov::element::Type_t element_type,
                                              std::vector<int64_t> qkv_order);

/**
 * @brief Creates a Grouped Query Attention (GQA)
 * @param kv Key or Value tensor
 * @param num_groups Number of groups for GQA
 * @param target_shape_v Target shape vector
 * @param n_heads Number of heads
 * @return Shared pointer to the created GQA
 */
std::shared_ptr<ov::Node> make_gqa(ov::Output<ov::Node> kv,
                                   size_t num_groups,
                                   std::vector<int32_t> target_shape_v,
                                   int32_t n_heads);

/**
 * @brief Creates a transpose operation for QKV tensors
 * @param qkv Input QKV tensor
 * @param order Transpose order
 * @return Shared pointer to the created transpose node
 */
std::shared_ptr<ov::Node> make_qkv_transpose(ov::Output<ov::Node> qkv, std::vector<int64_t> order);

/**
 * @brief Creates a KV rearrangement operation for beam search
 * @param kv_past Past KV cache
 * @param beam_idx Beam indices
 * @param axis_val Axis value for gather operation
 * @return Shared pointer to the created rearrangement
 */
std::shared_ptr<ov::Node> make_kv_rearrange(ov::Output<ov::Node> kv_past,
                                            ov::Output<ov::Node> beam_idx,
                                            int axis_val = 0);

/**
 * @brief Creates an LLM KV cache pattern model
 * @param batch Batch dimension
 * @param n_heads Number of heads dimension
 * @param n_features Number of features dimension
 * @param element_type Element type
 * @param concat_axis Concatenation axis
 * @param stateful Whether to make the model stateful
 * @param fuse_cache_reorder Whether to fuse cache reorder
 * @param build_state_initializer Whether to build state initializer
 * @param num_groups Number of groups for GQA
 * @return Shared pointer to the created model
 */
std::shared_ptr<ov::Model> make_llm_kv_cache_pattern(ov::Dimension batch = ov::Dimension::dynamic(),
                                                     ov::Dimension n_heads = ov::Dimension::dynamic(),
                                                     ov::Dimension n_features = ov::Dimension::dynamic(),
                                                     ov::element::Type_t element_type = ov::element::f32,
                                                     int64_t concat_axis = 2,
                                                     bool stateful = false,
                                                     bool fuse_cache_reorder = false,
                                                     bool build_state_initializer = false,
                                                     size_t num_groups = 1);

/**
 * @brief Creates an LLM KV cache pattern with Scaled Dot Product Attention (SDPA)
 * @param batch Batch dimension
 * @param n_heads Number of heads dimension
 * @param k_features Key features dimension
 * @param v_features Value features dimension
 * @param element_type Element type
 * @param qkv_order Order of QKV dimensions
 * @param causal Whether to use causal attention
 * @param with_mask Whether to include attention mask
 * @param with_scale Whether to include scaling
 * @param stateful Whether to make the model stateful
 * @param fuse_cache_reorder Whether to fuse cache reorder
 * @param num_groups Number of groups for GQA
 * @return Shared pointer to the created model
 */
std::shared_ptr<ov::Model> make_llm_kv_cache_sdpa_pattern(ov::Dimension batch = ov::Dimension::dynamic(),
                                                          ov::Dimension n_heads = ov::Dimension::dynamic(),
                                                          ov::Dimension k_features = ov::Dimension::dynamic(),
                                                          ov::Dimension v_features = ov::Dimension::dynamic(),
                                                          ov::element::Type_t element_type = ov::element::f32,
                                                          std::vector<int64_t> qkv_order = {0, 1, 2, 3},
                                                          bool causal = false,
                                                          bool with_mask = false,
                                                          bool with_scale = false,
                                                          bool stateful = false,
                                                          bool fuse_cache_reorder = false,
                                                          size_t num_groups = 1);
}  // namespace ov::test::utils

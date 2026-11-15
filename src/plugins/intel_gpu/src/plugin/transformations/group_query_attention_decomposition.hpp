// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/op/sdpa.hpp"
#include "transformations/op_conversions/group_query_attention_decomposition.hpp"

namespace ov::intel_gpu {

class GroupQueryAttentionDecomposition : public ov::pass::GroupQueryAttentionDecomposition {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::GroupQueryAttentionDecomposition");
    GroupQueryAttentionDecomposition(bool sdpa_support_immad) : m_sdpa_support_immad(sdpa_support_immad) {}

protected:
    struct StatelessKVMetadata;

    std::unique_ptr<KVCacheMetadata> create_metadata() override;
    KVCacheOutputs construct_kvcache(const std::shared_ptr<ov::op::internal::GroupQueryAttention>& node,
                                     const ov::Output<ov::Node>& past_key,
                                     const ov::Output<ov::Node>& past_value,
                                     const ov::Output<ov::Node>& key,
                                     const ov::Output<ov::Node>& value,
                                     const ov::Output<ov::Node>& seqlens_1d,
                                     const std::shared_ptr<ov::Node>& kv_seqlen,
                                     const std::shared_ptr<ov::Node>& past_seqlen,
                                     const ov::Output<ov::Node>& current_seqlen_scalar,
                                     KVCacheMetadata& metadata) override;
    std::shared_ptr<ov::Node> make_sdpa(const ov::Output<ov::Node>& query,
                                        const ov::Output<ov::Node>& key,
                                        const ov::Output<ov::Node>& value,
                                        const ov::Output<ov::Node>& mask,
                                        const ov::Output<ov::Node>& scale,
                                        const ov::Output<ov::Node>& sink,
                                        bool is_causal,
                                        int64_t local_window_size,
                                        const KVCacheMetadata& metadata) override;
    std::shared_ptr<ov::Node> make_attention_mask(const ov::Output<ov::Node>& curr_seqlen_scalar,
                                                  const ov::Output<ov::Node>& kv_len_scalar,
                                                  const ov::Output<ov::Node>& kv_len_1d,
                                                  const ov::Output<ov::Node>& past_seqlen,
                                                  const ov::element::Type& compute_type,
                                                  bool causal,
                                                  int64_t local_window_size,
                                                  const ov::Output<ov::Node>& external_bias,
                                                  const ov::Output<ov::Node>& bias_col_offset,
                                                  bool sliding_window_cache,
                                                  float scale,
                                                  bool has_sink,
                                                  const KVCacheMetadata& metadata) override;
    // RMSFusion runs before this pass on GPU, so QK-Norm is emitted as the fused RMS op directly.
    ov::Output<ov::Node> make_qk_norm(const ov::Output<ov::Node>& input, const ov::Output<ov::Node>& weight, float epsilon) override;

    bool prepare_compressed_kv(const std::shared_ptr<ov::op::internal::GroupQueryAttention>& node,
                               const ov::element::Type& key_type,
                               const ov::element::Type& value_type,
                               const ov::PartialShape& key_partial_shape,
                               StatelessKVMetadata& metadata);
                               
    bool m_sdpa_support_immad;
};

}  // namespace ov::intel_gpu

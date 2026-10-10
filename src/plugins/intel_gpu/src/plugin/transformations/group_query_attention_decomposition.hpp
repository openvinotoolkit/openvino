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
    GroupQueryAttentionDecomposition() = default;

protected:
    struct StatelessKVMetadata;

    std::unique_ptr<KVCacheMetadata> create_metadata(const std::shared_ptr<ov::op::internal::GroupQueryAttention>& node) override;
    KVCacheOutputs construct_kvcache(const std::shared_ptr<ov::op::internal::GroupQueryAttention>& node,
                                     const ov::Output<ov::Node>& past_key,
                                     const ov::Output<ov::Node>& past_value,
                                     const ov::Output<ov::Node>& key,
                                     const ov::Output<ov::Node>& value,
                                     const ov::Output<ov::Node>& seqlens_1d,
                                     const ov::Output<ov::Node>& past_seqlen,
                                     const ov::Output<ov::Node>& current_seqlen_scalar,
                                     KVCacheMetadata& metadata) override;
    std::shared_ptr<ov::Node> make_sdpa(const ov::Output<ov::Node>& query,
                                        const ov::Output<ov::Node>& key,
                                        const ov::Output<ov::Node>& value,
                                        const ov::Output<ov::Node>& mask,
                                        const ov::Output<ov::Node>& scale,
                                        const ov::Output<ov::Node>& sink,
                                        bool is_causal,
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
    void prepare_compressed_kv(const std::shared_ptr<ov::op::internal::GroupQueryAttention>& node,
                               const ov::Output<ov::Node>& key,
                               const ov::Output<ov::Node>& value,
                               KVCacheMetadata& metadata);
};

}  // namespace ov::intel_gpu

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/core/type/element_type.hpp"
#include "primitive.hpp"

namespace cldnn {

/// @brief Row-selective dequantizing gather over a raw GGUF block-quantised embedding table.
/// @details Consumes the opaque gguf_* weight tensor [vocab, hidden] directly from HBM (never
/// materialised to dense f16) plus an integer index tensor, and produces an f16 embedding tensor
/// [*indices_dims, hidden]. This mirrors the GGUF FullyConnected decode path (fc_gguf_opt) but for
/// a gather instead of a matmul, so the embedding weight stays in its native GGUF format and the
/// 1.24 GB dense f16 copy (and its .bin) is eliminated.
struct gather_gguf : public primitive_base<gather_gguf> {
    CLDNN_DECLARE_PRIMITIVE(gather_gguf)

    gather_gguf() : primitive_base("", {}) {}

    /// @brief Constructs gather_gguf primitive.
    /// @param id           This primitive id.
    /// @param data         Raw GGUF block-quantised weight tensor [vocab, hidden].
    /// @param indices      Integer token indices (any rank); output is [*indices_dims, hidden].
    /// @param weight_type  GGUF block element type of @p data (gguf_q4_1, gguf_q4_k, ...).
    /// @param vocab_size   Number of embedding rows (data shape[0]).
    /// @param hidden_size  Embedding width (data shape[1]).
    gather_gguf(const primitive_id& id,
                const input_info& data,
                const input_info& indices,
                ov::element::Type weight_type,
                int64_t vocab_size,
                int64_t hidden_size)
        : primitive_base(id, {data, indices}, 1, {optional_data_type{data_types::f16}}),
          weight_type(weight_type),
          vocab_size(vocab_size),
          hidden_size(hidden_size) {}

    ov::element::Type weight_type = ov::element::dynamic;
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, static_cast<size_t>(static_cast<ov::element::Type_t>(weight_type)));
        seed = hash_combine(seed, vocab_size);
        seed = hash_combine(seed, hidden_size);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const gather_gguf>(rhs);
        return weight_type == rhs_casted.weight_type && vocab_size == rhs_casted.vocab_size &&
               hidden_size == rhs_casted.hidden_size;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<gather_gguf>::save(ob);
        ob << static_cast<int64_t>(static_cast<ov::element::Type_t>(weight_type));
        ob << vocab_size;
        ob << hidden_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<gather_gguf>::load(ib);
        int64_t wt = 0;
        ib >> wt;
        weight_type = ov::element::Type(static_cast<ov::element::Type_t>(wt));
        ib >> vocab_size;
        ib >> hidden_size;
    }
};
}  // namespace cldnn

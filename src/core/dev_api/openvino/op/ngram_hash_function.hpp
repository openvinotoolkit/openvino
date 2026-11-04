// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {
/// \note NGramHashFunction op class is under development and subject to change
///
/// \brief Hashes all n-grams (orders 2 .. ngram_size) that end at every token into row indices of a packed,
/// hashed n-gram embedding table, keeping the 64-bit integer arithmetic inside the operation.
///
/// For a token with window `w` (oldest first, `w[ngram_size - 1]` is the token itself) and `x[k] = w[ngram_size-1-k]`
/// (the token k steps back), the n-gram of order `o` is mixed as
///     mix_o = (x[0] * multipliers[0]) ^ ... ^ (x[o-1] * multipliers[o-1])     (wrapping 64-bit arithmetic)
/// and every head `h` of order `o` produces
///     row_ids[..., h] = floor_mod(mix_o, head_vocab_sizes[h]) + head_offsets[h]
/// Heads are grouped by order: first heads_per_order[0] heads of the bigram, then heads_per_order[1] heads of the
/// trigram, and so on.
///
/// Input 0: ngrams [..., ngram_size] of type i32 or i64. Output 0: row_ids [..., H] of type i32,
/// H = sum(heads_per_order).
/// \ingroup ov_ops_cpp_api
class OPENVINO_API NGramHashFunction : public ov::op::Op {
public:
    OPENVINO_OP("NGramHashFunction");

    NGramHashFunction() = default;
    /// \brief Constructs an NGramHashFunction operation.
    ///
    /// \param ngrams           Windows of ngram_size token ids for every token, oldest first.
    /// \param ngram_size       The largest n-gram order (>= 2).
    /// \param heads_per_order  Number of hashed heads of every order 2 .. ngram_size (ngram_size - 1 values).
    /// \param multipliers      Hash multipliers of the token itself, the previous token, ... (ngram_size values).
    /// \param head_vocab_sizes Vocabulary size (modulus) of every head (H values).
    /// \param head_offsets     First row of every head in the packed table (H values).
    NGramHashFunction(const Output<Node>& ngrams,
                      int64_t ngram_size,
                      std::vector<int64_t> heads_per_order,
                      std::vector<int64_t> multipliers,
                      std::vector<int64_t> head_vocab_sizes,
                      std::vector<int64_t> head_offsets);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    int64_t get_ngram_size() const {
        return m_ngram_size;
    }
    const std::vector<int64_t>& get_heads_per_order() const {
        return m_heads_per_order;
    }
    const std::vector<int64_t>& get_multipliers() const {
        return m_multipliers;
    }
    const std::vector<int64_t>& get_head_vocab_sizes() const {
        return m_head_vocab_sizes;
    }
    const std::vector<int64_t>& get_head_offsets() const {
        return m_head_offsets;
    }
    /// \brief Total number of hashed heads H = sum(heads_per_order).
    size_t get_num_heads() const;

private:
    int64_t m_ngram_size{0};
    std::vector<int64_t> m_heads_per_order;
    std::vector<int64_t> m_multipliers;
    std::vector<int64_t> m_head_vocab_sizes;
    std::vector<int64_t> m_head_offsets;
};

}  // namespace internal
}  // namespace op
}  // namespace ov

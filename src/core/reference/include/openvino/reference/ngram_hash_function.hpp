// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace ov {
namespace reference {
/// \brief Computes NGramHashFunction rows for tokens [token_begin, token_end).
///
/// \param ngrams           Windows [num_tokens, ngram_size], oldest first (the token itself last).
/// \param row_ids          Output [num_tokens, num_heads].
/// \param heads_per_order  ngram_size - 1 values, heads of the bigram, trigram, ...
/// \param multipliers      ngram_size values, multipliers[k] applies to the token k steps back.
/// \param head_vocab_sizes num_heads positive moduli.
/// \param head_offsets     num_heads offsets.
///
/// The multiply/xor wrap around modulo 2^64 (two's complement int64) and the modulo is a floor modulo (the sign of
/// the divisor), which matches the int64 Multiply/BitwiseXor/FloorMod graph and torch.remainder.
template <typename T>
void ngram_hash_function(const T* ngrams,
                         int32_t* row_ids,
                         size_t token_begin,
                         size_t token_end,
                         size_t ngram_size,
                         const int64_t* heads_per_order,
                         const int64_t* multipliers,
                         const int64_t* head_vocab_sizes,
                         const int64_t* head_offsets,
                         size_t num_heads) {
    for (size_t t = token_begin; t < token_end; ++t) {
        const T* window = ngrams + t * ngram_size;
        int32_t* out = row_ids + t * num_heads;
        // The token itself (x[0]) is the last element of the window.
        uint64_t mix = static_cast<uint64_t>(static_cast<int64_t>(window[ngram_size - 1])) *
                       static_cast<uint64_t>(multipliers[0]);
        size_t h = 0;
        for (size_t k = 1; k < ngram_size; ++k) {
            mix ^= static_cast<uint64_t>(static_cast<int64_t>(window[ngram_size - 1 - k])) *
                   static_cast<uint64_t>(multipliers[k]);
            const auto value = static_cast<int64_t>(mix);
            for (int64_t e = 0; e < heads_per_order[k - 1]; ++e, ++h) {
                const int64_t vocab = head_vocab_sizes[h];
                int64_t r = value % vocab;
                if (r < 0) {
                    r += vocab;
                }
                out[h] = static_cast<int32_t>(r + head_offsets[h]);
            }
        }
    }
}

/// \brief Computes NGramHashFunction rows for all num_tokens tokens.
template <typename T>
void ngram_hash_function(const T* ngrams,
                         int32_t* row_ids,
                         size_t num_tokens,
                         size_t ngram_size,
                         const int64_t* heads_per_order,
                         const int64_t* multipliers,
                         const int64_t* head_vocab_sizes,
                         const int64_t* head_offsets,
                         size_t num_heads) {
    ngram_hash_function(ngrams,
                        row_ids,
                        0,
                        num_tokens,
                        ngram_size,
                        heads_per_order,
                        multipliers,
                        head_vocab_sizes,
                        head_offsets,
                        num_heads);
}
}  // namespace reference
}  // namespace ov

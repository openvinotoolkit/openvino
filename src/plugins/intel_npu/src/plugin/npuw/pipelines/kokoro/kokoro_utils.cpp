// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kokoro_utils.hpp"

#include <algorithm>

namespace ov {
namespace npuw {
namespace kokoro {

std::size_t find_real_sequence_length(const int64_t* input_ids, std::size_t seq_len) {
    // The Kokoro input format is [BOS=0, tok1, ..., tokN, EOS=0, PAD, ...].
    // Scan from position 1 to find the first zero (EOS).
    for (std::size_t i = 1; i < seq_len; ++i) {
        if (input_ids[i] == 0) {
            return i + 1;  // include the EOS token
        }
    }
    return seq_len;  // no padding detected
}

void fill_text_mask_from_lengths(bool* mask_data, std::size_t seq_len, std::size_t real_len) {
    for (std::size_t i = 0; i < seq_len; ++i) {
        mask_data[i] = (i >= real_len);
    }
}

void zero_padding_durations(int64_t* pred_dur, std::size_t full_len, std::size_t real_len) {
    if (real_len < full_len) {
        std::fill(pred_dur + real_len, pred_dur + full_len, int64_t{0});
    }
}

void zero_padding_durations(int32_t* pred_dur, std::size_t full_len, std::size_t real_len) {
    if (real_len < full_len) {
        std::fill(pred_dur + real_len, pred_dur + full_len, int32_t{0});
    }
}

}  // namespace kokoro
}  // namespace npuw
}  // namespace ov

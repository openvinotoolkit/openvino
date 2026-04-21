// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>

namespace ov {
namespace npuw {
namespace kokoro {

/// Find the real (unpadded) sequence length in a Kokoro input_ids tensor.
///
/// The Kokoro input format is [BOS=0, tok1, ..., tokN, EOS=0, PAD, PAD, ...].
/// This function scans from position 1 to find the first zero (EOS), and returns
/// the real length including the EOS token itself.
///
/// @param input_ids Pointer to int64 input_ids data, shape [1, seq_len].
/// @param seq_len   Total length of the input_ids sequence.
/// @return          Number of valid tokens (including BOS and EOS).
///                  Returns seq_len if no EOS is found after position 0.
std::size_t find_real_sequence_length(const int64_t* input_ids, std::size_t seq_len);

/// Fill a boolean text_mask from the real sequence length.
///
/// text_mask[i] = false for valid positions [0, real_len),
/// text_mask[i] = true  for padding positions [real_len, seq_len).
///
/// This matches the PyTorch semantics where True = masked/padded.
///
/// @param mask_data  Output boolean array of size seq_len.
/// @param seq_len    Total sequence length.
/// @param real_len   Number of valid (non-padding) tokens.
void fill_text_mask_from_lengths(bool* mask_data, std::size_t seq_len, std::size_t real_len);

/// Zero out padding positions in a pred_dur tensor (int64).
///
/// Positions [real_len, full_len) are set to 0 so padded tokens contribute
/// zero frames to the audio output.
///
/// @param pred_dur   Pointer to the pred_dur data.
/// @param full_len   Total size of the pred_dur tensor.
/// @param real_len   Number of valid tokens. Positions beyond this are zeroed.
void zero_padding_durations(int64_t* pred_dur, std::size_t full_len, std::size_t real_len);

/// @overload for int32 pred_dur tensors.
void zero_padding_durations(int32_t* pred_dur, std::size_t full_len, std::size_t real_len);

}  // namespace kokoro
}  // namespace npuw
}  // namespace ov

// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/stft.hpp"

#include <complex>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/reference/fft.hpp"
#include "openvino/reference/multiply.hpp"
#include "openvino/reference/rdft.hpp"
#include "openvino/reference/transpose.hpp"
#include "openvino/reference/utils/fft_common.hpp"

namespace ov {
namespace reference {
using complex_type = std::complex<float>;

using T = float;
void stft(const float* signal,
          const float* window,
          float* rdft_result,
          const Shape& signal_shape,
          const Shape& window_shape,
          const int64_t frame_size,
          const int64_t frame_step,
          const bool frames_first) {
    constexpr size_t signal_axis = 1;
    const auto batch_size = signal_shape[0];
    const auto signal_length = signal_shape[signal_axis];
    const auto num_frames = static_cast<size_t>((signal_length - frame_size) / frame_step) + 1;
    const auto fft_out_shape = Shape{std::floor(frame_size / 2) + 1, 2};
    for (size_t batch = 0; batch < batch_size; ++batch) {
        const auto batch_idx = batch * signal_length;
        for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
            const auto start = batch_idx + frame_idx * frame_step;
            const auto end = start + frame_size;
            std::vector<T> signal_slice(signal + start, signal + end);
            reference::multiply(signal_slice.data(),
                                window,
                                signal_slice.data(),
                                Shape{frame_size},
                                window_shape,
                                op::AutoBroadcastType::NUMPY);
            const auto result_idx = batch * num_frames * fft_out_shape[0] * fft_out_shape[1] +
                                    frame_idx * fft_out_shape[0] * fft_out_shape[1];
            reference::rdft(signal_slice, Shape{frame_size}, {0}, Shape{frame_size, 2}, rdft_result + result_idx);
        }
    }
    if (!frames_first) {
        const auto stft_transp_out_shape = Shape{batch_size, fft_out_shape[0], num_frames, fft_out_shape[1]};
        std::vector<T> signal_t(rdft_result, rdft_result + shape_size(stft_transp_out_shape));
        transpose(reinterpret_cast<const char*>(signal_t.data()),
                  reinterpret_cast<char*>(rdft_result),
                  Shape{batch_size, num_frames, fft_out_shape[0], fft_out_shape[1]},
                  sizeof(T),
                  {0, 2, 1, 3},
                  stft_transp_out_shape);
    }
}
}  // namespace reference
}  // namespace ov

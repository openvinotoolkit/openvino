// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/stft.hpp"

#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/reference/multiply.hpp"
#include "openvino/reference/rdft.hpp"
#include "openvino/reference/transpose.hpp"

namespace ov {
namespace reference {
void stft(const float* signal,
          const float* window,
          float* rdft_result,
          const Shape& signal_shape,
          const Shape& window_shape,
          const int64_t frame_size,
          const int64_t frame_step,
          const bool transpose_frames) {
    const auto is_signal_1D = signal_shape.size() == 1;
    const size_t batch_size = is_signal_1D ? 1 : signal_shape[0];
    const size_t signal_axis = is_signal_1D ? 0 : 1;
    const auto signal_length = signal_shape[signal_axis];
    const auto num_frames = static_cast<size_t>((signal_length - frame_size) / frame_step) + 1;
    const auto frame_size_dim = static_cast<size_t>(frame_size);
    const auto frame_size_dim_shape = Shape{frame_size_dim};
    const auto frame_size_dim_shape_out = Shape{frame_size_dim, 2};
    const auto fft_out_shape = Shape{static_cast<size_t>((frame_size_dim / 2) + 1), 2};

    const auto window_length = window_shape[0] < frame_size_dim ? window_shape[0] : frame_size_dim;
    std::vector<float> pad_window(frame_size, 0);
    std::copy(window, window + window_shape[0], pad_window.begin() + (frame_size_dim - window_length) / 2);

    const auto fft_out_shape_size = shape_size(fft_out_shape);
    for (size_t batch = 0, batch_in_start = 0, batch_frames_out = 0; batch < batch_size; ++batch) {
        for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
            const auto frame_start = batch_in_start + frame_idx * frame_step;
            const auto frame_end = frame_start + frame_size;
            std::vector<float> signal_slice(signal + frame_start, signal + frame_end);
            reference::multiply(signal_slice.data(),
                                pad_window.data(),
                                signal_slice.data(),
                                frame_size_dim_shape,
                                frame_size_dim_shape,
                                op::AutoBroadcastType::NUMPY);
            const auto result_idx = (batch_frames_out + frame_idx) * fft_out_shape_size;
            reference::rdft(signal_slice,
                            frame_size_dim_shape,
                            {0},
                            frame_size_dim_shape_out,
                            rdft_result + result_idx);
        }
        batch_in_start += signal_length;
        batch_frames_out += num_frames;
    }
    if (transpose_frames) {
        const auto stft_transp_out_shape = Shape{batch_size, fft_out_shape[0], num_frames, fft_out_shape[1]};
        std::vector<float> signal_t(rdft_result, rdft_result + shape_size(stft_transp_out_shape));
        transpose(reinterpret_cast<const char*>(signal_t.data()),
                  reinterpret_cast<char*>(rdft_result),
                  Shape{batch_size, num_frames, fft_out_shape[0], fft_out_shape[1]},
                  sizeof(float),
                  {0, 2, 1, 3},
                  stft_transp_out_shape);
    }
}
}  // namespace reference
}  // namespace ov

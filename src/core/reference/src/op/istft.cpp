// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/istft.hpp"

#include <algorithm>
#include <functional>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/reference/irdft.hpp"
#include "openvino/reference/transpose.hpp"

namespace ov {
namespace reference {
void istft(const float* in_data,
           const float* window,
           float* final_result,
           const Shape& data_shape,
           const Shape& window_shape,
           const int64_t frame_size,
           const int64_t frame_step,
           const int64_t length,
           const bool center,
           const bool normalized) {
    const auto is_data_3D = data_shape.size() == 3;
    const size_t frames_axis = 1 + (is_data_3D ? 0 : 1);
    const size_t batch_size = is_data_3D ? 1 : data_shape[0];

    const auto sqrt_frame_size = std::sqrt(frame_size);
    const auto num_frames = data_shape[frames_axis];

    const auto signal_length = (num_frames - 1) * frame_step + frame_size;
    const int64_t final_signal_length = length > 0 ? length : (center ? (signal_length - frame_size) : signal_length);

    std::vector<float> mid_result(batch_size * signal_length, 0);
    float* result = mid_result.data();

    const auto frame_size_dim = static_cast<size_t>(frame_size);
    const auto frame_size_dim_shape = Shape{frame_size_dim};
    const auto frame_size_dim_shape_out = Shape{frame_size_dim, 2};
    const auto fft_out_shape = Shape{static_cast<size_t>((frame_size_dim / 2) + 1), 2};

    const auto window_length = window_shape[0] < frame_size_dim ? window_shape[0] : frame_size_dim;
    std::vector<float> pad_window(frame_size, 0);
    std::copy(window, window + window_shape[0], pad_window.begin() + (frame_size_dim - window_length) / 2);

    const bool transpose_frames = true;
    std::vector<float> data_t(in_data, in_data + shape_size(data_shape));
    if (transpose_frames) {
        const auto stft_transp_out_shape = Shape{batch_size, num_frames, fft_out_shape[0], fft_out_shape[1]};
        transpose(reinterpret_cast<const char*>(in_data),
                  reinterpret_cast<char*>(data_t.data()),
                  Shape{batch_size, fft_out_shape[0], num_frames, fft_out_shape[1]},
                  sizeof(float),
                  {0, 2, 1, 3},
                  stft_transp_out_shape);
    }

    const auto fft_out_shape_size = shape_size(fft_out_shape);
    std::vector<float> window_sum(batch_size * signal_length);

    for (size_t batch = 0, batch_in_start = 0, batch_out_start = 0; batch < batch_size; ++batch) {
        for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
            const auto in_frame_start = batch_in_start + frame_idx * fft_out_shape_size;
            const auto in_frame_end = in_frame_start + fft_out_shape_size;

            const auto out_frame_start = batch_out_start + frame_idx * frame_step;
            const auto out_frame_end = out_frame_start + frame_size;

            std::vector<float> frame_data(data_t.data() + in_frame_start, data_t.data() + in_frame_end);
            std::vector<float> frame_signal(frame_size);

            reference::irdft(frame_data,
                             fft_out_shape,
                             {0},
                             frame_signal.data(),
                             frame_size_dim_shape_out,
                             frame_size_dim_shape,
                             frame_size);

            std::transform(frame_signal.begin(),
                           frame_signal.end(),
                           mid_result.begin() + out_frame_start,
                           mid_result.begin() + out_frame_start,
                           std::plus<float>());

            std::transform(window_sum.begin() + out_frame_start,
                           window_sum.begin() + out_frame_end,
                           pad_window.begin(),
                           window_sum.begin() + out_frame_start,
                           std::plus<float>());
        }

        if (normalized) {
            std::transform(result + batch_out_start,
                           result + batch_out_start + signal_length,
                           result + batch_out_start,
                           [sqrt_frame_size](float a) {
                               return a * sqrt_frame_size;
                           });
        }

        std::transform(result + batch_out_start,
                       result + batch_out_start + signal_length,
                       window_sum.begin(),
                       result + batch_out_start,
                       [](float a, float b) {
                           if (b != 0.f)
                               return a / b;
                           else
                               return 0.f;
                       });

        if (center) {
            const int64_t margin = (frame_size / 2);
            const size_t result_start = batch_out_start + margin;
            const int64_t data_end = signal_length - (frame_size / 2);
            int64_t signal_end = final_signal_length < data_end ? final_signal_length : data_end;
            std::copy(result + result_start,
                      result + result_start + signal_end,
                      final_result + (batch * final_signal_length));
        } else {
            std::copy(result + batch_out_start,
                      result + batch_out_start + final_signal_length,
                      final_result + batch_out_start);
        }

        batch_in_start += (num_frames * fft_out_shape_size);
        batch_out_start += signal_length;
    }
}
}  // namespace reference
}  // namespace ov

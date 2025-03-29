// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/istft.hpp"

#include <algorithm>
#include <functional>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/reference/add.hpp"
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

    const auto sqrt_frame_size = static_cast<float>(std::sqrt(frame_size));
    const auto num_frames = data_shape[frames_axis];

    const auto signal_length = (num_frames - 1) * frame_step + frame_size;
    const int64_t final_signal_length =
        length > 0 ? length : (center ? (signal_length - (frame_size & ~1)) : signal_length);
    std::fill(final_result, final_result + batch_size * final_signal_length, 0.f);

    std::vector<float> mid_result(batch_size * signal_length, 0.f);
    float* result = mid_result.data();

    const auto fft_results_dim = data_shape[data_shape.size() - 3];
    OPENVINO_ASSERT(fft_results_dim == static_cast<size_t>((frame_size / 2) + 1));

    const auto frame_size_dim = static_cast<size_t>(frame_size);
    const auto frame_size_dim_shape = Shape{frame_size_dim};
    const auto frame_size_dim_shape_out = Shape{frame_size_dim, 2};
    const auto fft_out_shape = Shape{fft_results_dim, 2};

    const auto window_length = window_shape[0] < frame_size_dim ? window_shape[0] : frame_size_dim;
    std::vector<float> pad_window(frame_size, 0);
    std::copy(window, window + window_shape[0], pad_window.begin() + (frame_size_dim - window_length) / 2);
    std::vector<float> pow_window(frame_size, 0);
    std::transform(pad_window.begin(), pad_window.end(), pow_window.begin(), [](float win_val) {
        return win_val * win_val;
    });

    std::vector<float> data_t(in_data, in_data + shape_size(data_shape));
    const auto stft_transp_out_shape = Shape{batch_size, num_frames, fft_out_shape[0], fft_out_shape[1]};
    transpose(reinterpret_cast<const char*>(in_data),
              reinterpret_cast<char*>(data_t.data()),
              Shape{batch_size, fft_out_shape[0], num_frames, fft_out_shape[1]},
              sizeof(float),
              {0, 2, 1, 3},
              stft_transp_out_shape);

    // Setting function for the result postprocessing
    const auto norm_window_div = [sqrt_frame_size](float a, float b) {
        if (b != 0.f)
            return (a * sqrt_frame_size) / b;
        else
            return 0.f;
    };
    const auto window_div = [](float a, float b) {
        if (b != 0.f)
            return a / b;
        else
            return 0.f;
    };
    std::function<float(float, float)> postprocess_func;
    if (normalized) {
        postprocess_func = norm_window_div;
    } else {
        postprocess_func = window_div;
    }

    const auto fft_out_shape_size = shape_size(fft_out_shape);
    const auto in_batch_single_step = num_frames * fft_out_shape_size;
    const int64_t margin = center ? (frame_size / 2) : 0;
    const int64_t data_end = signal_length - margin;
    const int64_t copy_end = final_signal_length < data_end ? final_signal_length : data_end;

    std::vector<float> window_sum(batch_size * signal_length);
    std::vector<float> frame_signal(frame_size);

    for (size_t batch = 0, batch_in_start = 0, batch_out_start = 0; batch < batch_size; ++batch) {
        for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
            const auto in_frame_start = batch_in_start + frame_idx * fft_out_shape_size;
            const auto in_frame_end = in_frame_start + fft_out_shape_size;
            const auto out_frame_start = batch_out_start + frame_idx * frame_step;

            std::vector<float> frame_data(data_t.data() + in_frame_start, data_t.data() + in_frame_end);
            reference::irdft(frame_data,
                             fft_out_shape,
                             {0},
                             frame_signal.data(),
                             frame_size_dim_shape_out,
                             frame_size_dim_shape,
                             frame_size);

            // Overlap Add
            float* mid_result_sum = mid_result.data() + out_frame_start;
            float* window_frame_sum = window_sum.data() + out_frame_start;
            for (size_t i = 0; i < frame_signal.size(); ++i) {
                mid_result_sum[i] += frame_signal[i] * pad_window[i];
                window_frame_sum[i] += pow_window[i];
            }
        }

        std::transform(result, result + signal_length, window_sum.begin() + batch_out_start, result, postprocess_func);

        const auto result_start = result + margin;
        std::copy(result_start, result_start + copy_end, final_result);

        batch_in_start += in_batch_single_step;
        batch_out_start += signal_length;
        result += signal_length;
        final_result += final_signal_length;
    }
}
}  // namespace reference
}  // namespace ov

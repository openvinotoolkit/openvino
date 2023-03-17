// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/rdft.hpp"

namespace rfft_v9 {
struct InfoForRFFT9 {
    std::vector<float> input_data;
    std::vector<int64_t> axes_data;
    ngraph::Shape input_data_shape;
    ngraph::Shape axes_data_shape;
    ngraph::Shape fft_output_shape;
    ngraph::Shape output_shape;
};

InfoForRFFT9 get_info_for_rfft9_eval(const std::vector<std::shared_ptr<ngraph::HostTensor>>& inputs) {
    InfoForRFFT9 result;

    result.input_data_shape = inputs[0]->get_shape();
    result.axes_data_shape = inputs[1]->get_shape();
    result.input_data = get_floats(inputs[0], result.input_data_shape);
    result.axes_data = get_integers(inputs[1], result.axes_data_shape);

    auto fft_output_shape = result.input_data_shape;
    auto output_shape = result.input_data_shape;

    int64_t input_rank = static_cast<int64_t>(result.input_data_shape.size());
    auto canonicalized_axes =
        ngraph::runtime::reference::canonicalize_axes(result.axes_data.data(), result.axes_data_shape, input_rank);

    size_t num_of_axes = result.axes_data.size();
    auto signal_size = fft_v7::get_signal_size(inputs, num_of_axes);

    const auto last_axis = canonicalized_axes.back();
    for (size_t i = 0; i < num_of_axes; ++i) {
        int64_t current_axis = canonicalized_axes[i];
        int64_t current_signal_size = signal_size[i];
        if (current_signal_size != -1) {
            fft_output_shape[current_axis] = current_signal_size;
            output_shape[current_axis] = current_signal_size;
        }
    }
    output_shape[last_axis] = fft_output_shape[last_axis] / 2 + 1;
    output_shape.push_back(2);
    fft_output_shape.push_back(2);

    result.fft_output_shape = fft_output_shape;
    result.output_shape = output_shape;

    result.axes_data = canonicalized_axes;

    return result;
}
}  // namespace rfft_v9

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v9::RDFT>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    auto info = rfft_v9::get_info_for_rfft9_eval(inputs);
    outputs[0]->set_shape(info.output_shape);

    std::vector<float> rfft_result(ngraph::shape_size(info.output_shape), 0.0f);
    ngraph::runtime::reference::rdft(info.input_data,
                                     info.input_data_shape,
                                     info.axes_data,
                                     info.fft_output_shape,
                                     rfft_result.data());

    const auto output_type = op->get_input_element_type(0);
    ngraph::runtime::reference::fft_postprocessing(outputs, output_type, rfft_result);
    return true;
}
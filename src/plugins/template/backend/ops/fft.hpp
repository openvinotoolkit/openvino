// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/reference/fft.hpp"

namespace fft_v7 {
struct InfoForFFT7 {
    std::vector<float> input_data;
    std::vector<int64_t> axes_data;
    ngraph::Shape input_data_shape;
    ngraph::Shape axes_data_shape;
    ngraph::Shape output_shape;
};

std::vector<int64_t> get_signal_size(const std::vector<std::shared_ptr<ngraph::HostTensor>>& inputs,
                                     size_t num_of_axes) {
    if (inputs.size() == 3) {
        return get_integers(inputs[2], inputs[2]->get_shape());
    }

    return std::vector<int64_t>(num_of_axes, static_cast<int64_t>(-1));
}

InfoForFFT7 get_info_for_fft7_eval(const std::vector<std::shared_ptr<ngraph::HostTensor>>& inputs) {
    InfoForFFT7 result;

    result.input_data_shape = inputs[0]->get_shape();
    result.axes_data_shape = inputs[1]->get_shape();
    result.input_data = get_floats(inputs[0], result.input_data_shape);
    result.axes_data = get_integers(inputs[1], result.axes_data_shape);

    auto output_shape = result.input_data_shape;

    int64_t input_rank = static_cast<int64_t>(result.input_data_shape.size());
    int64_t complex_data_rank = input_rank - 1;
    auto canonicalized_axes = ngraph::runtime::reference::canonicalize_axes(result.axes_data.data(),
                                                                            result.axes_data_shape,
                                                                            complex_data_rank);

    size_t num_of_axes = result.axes_data.size();
    auto signal_size = get_signal_size(inputs, num_of_axes);

    for (size_t i = 0; i < num_of_axes; ++i) {
        int64_t current_axis = canonicalized_axes[i];
        int64_t current_signal_size = signal_size[i];
        if (current_signal_size != -1) {
            output_shape[current_axis] = current_signal_size;
        }
    }

    result.output_shape = output_shape;

    return result;
}
}  // namespace fft_v7

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v7::DFT>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    auto info = fft_v7::get_info_for_fft7_eval(inputs);
    outputs[0]->set_shape(info.output_shape);

    std::vector<float> fft_result(ngraph::shape_size(info.output_shape), 0.0f);
    ngraph::runtime::reference::fft(info.input_data.data(),
                                    info.input_data_shape,
                                    info.axes_data.data(),
                                    info.axes_data_shape,
                                    fft_result.data(),
                                    info.output_shape,
                                    ngraph::runtime::reference::FFTKind::Forward);

    const auto output_type = op->get_input_element_type(0);
    ngraph::runtime::reference::fft_postprocessing(outputs, output_type, fft_result);
    return true;
}

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v7::IDFT>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    auto info = fft_v7::get_info_for_fft7_eval(inputs);
    outputs[0]->set_shape(info.output_shape);

    std::vector<float> fft_result(ngraph::shape_size(info.output_shape), 0.0f);
    ngraph::runtime::reference::fft(info.input_data.data(),
                                    info.input_data_shape,
                                    info.axes_data.data(),
                                    info.axes_data_shape,
                                    fft_result.data(),
                                    info.output_shape,
                                    ngraph::runtime::reference::FFTKind::Inverse);

    const auto output_type = op->get_input_element_type(0);
    ngraph::runtime::reference::fft_postprocessing(outputs, output_type, fft_result);
    return true;
}
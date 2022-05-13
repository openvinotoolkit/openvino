// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/irdft.hpp"

#include <memory>

#include "irdft_shape_inference.hpp"
#include "itt.hpp"

#include <ngraph/runtime/reference/irdft.hpp>
#include <ngraph/runtime/reference/fft.hpp>

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::op::v9::IRDFT);

ov::op::v9::IRDFT::IRDFT(const Output<Node>& data, const Output<Node>& axes) : FFTBase(data, axes) {
    constructor_validate_and_infer_types();
}

ov::op::v9::IRDFT::IRDFT(const Output<Node>& data, const Output<Node>& axes, const Output<Node>& signal_size)
    : FFTBase(data, axes, signal_size) {
    constructor_validate_and_infer_types();
}

bool ov::op::v9::IRDFT::visit_attributes(AttributeVisitor& visitor) {
    NGRAPH_OP_SCOPE(v9_IRDFT_visit_attributes);
    return true;
}

std::shared_ptr<ov::Node> ov::op::v9::IRDFT::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(v9_IRDFT_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2 || new_args.size() == 3, "Number of inputs must be 2 or 3");

    if (new_args.size() == 2) {
        return std::make_shared<ov::op::v9::IRDFT>(new_args.at(0), new_args.at(1));
    }

    return std::make_shared<ov::op::v9::IRDFT>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void ov::op::v9::IRDFT::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(v9_IRDFT_validate_and_infer_types);

    validate_types();

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape()};
    std::vector<ov::PartialShape> input_shapes;

    const auto& data = get_input_partial_shape(0);
    const auto& axes = get_input_partial_shape(1);
    if (input_values().size() == 2) {
        input_shapes = {data, axes};
    } else {
        const auto& signal_size = get_input_partial_shape(2);
        input_shapes = {data, axes, signal_size};
    }

    ov::op::util::irdft_shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

using namespace ov;
namespace {
namespace irfft_v9 {

std::vector<int64_t> canonicalize_axes(const int64_t* axes_data,
                                       const Shape& axes_data_shape,
                                       int64_t complex_data_rank) {
    size_t num_of_fft_axes = axes_data_shape[0];

    std::vector<int64_t> result(axes_data, axes_data + num_of_fft_axes);
    for (int64_t& axis : result) {
        if (axis < 0) {
            axis += complex_data_rank;
        }
    }
    return result;
}

std::vector<float> get_floats(const std::shared_ptr<HostTensor>& input, const Shape& shape) {
    size_t input_size = shape_size(shape);
    std::vector<float> result(input_size);

    switch (input->get_element_type()) {
    case element::Type_t::bf16: {
        bfloat16* p = input->get_data_ptr<bfloat16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case element::Type_t::f16: {
        float16* p = input->get_data_ptr<float16>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = float(p[i]);
        }
    } break;
    case element::Type_t::f32: {
        float* p = input->get_data_ptr<float>();
        memcpy(result.data(), p, input_size * sizeof(float));
    } break;
    default:
        throw std::runtime_error("Unsupported data type.");
        break;
    }

    return result;
}

std::vector<int64_t> get_integers(const std::shared_ptr<HostTensor>& input, const Shape& shape) {
    size_t input_size = shape_size(shape);
    std::vector<int64_t> result(input_size);

    switch (input->get_element_type()) {
    case element::Type_t::i8: {
        auto p = input->get_data_ptr<int8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::i16: {
        auto p = input->get_data_ptr<int16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::i32: {
        auto p = input->get_data_ptr<int32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::i64: {
        auto p = input->get_data_ptr<int64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u8: {
        auto p = input->get_data_ptr<uint8_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u16: {
        auto p = input->get_data_ptr<uint16_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u32: {
        auto p = input->get_data_ptr<uint32_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    case element::Type_t::u64: {
        auto p = input->get_data_ptr<uint64_t>();
        for (size_t i = 0; i < input_size; ++i) {
            result[i] = int64_t(p[i]);
        }
    } break;
    default:
        throw std::runtime_error("Unsupported data type in op NonMaxSuppression-5");
        break;
    }

    return result;
}

std::vector<int64_t> get_signal_size(const std::vector<std::shared_ptr<HostTensor>>& inputs, size_t num_of_axes) {
    if (inputs.size() == 3) {
        return get_integers(inputs[2], inputs[2]->get_shape());
    }

    return std::vector<int64_t>(num_of_axes, static_cast<int64_t>(-1));
}

struct InfoForIRFFT9 {
    std::vector<float> input_data;
    std::vector<int64_t> axes_data;
    Shape input_data_shape;
    Shape axes_data_shape;
    Shape fft_output_shape;
    Shape output_shape;
    int64_t last_signal_size;
};

InfoForIRFFT9 get_info_for_irfft9_eval(const std::vector<std::shared_ptr<HostTensor>>& inputs) {
    InfoForIRFFT9 result;

    result.input_data_shape = inputs[0]->get_shape();
    result.axes_data_shape = inputs[1]->get_shape();
    result.input_data = get_floats(inputs[0], result.input_data_shape);
    result.axes_data = get_integers(inputs[1], result.axes_data_shape);

    auto fft_output_shape = result.input_data_shape;
    auto output_shape = result.input_data_shape;

    int64_t input_rank = static_cast<int64_t>(result.input_data_shape.size());
    int64_t complex_data_rank = input_rank - 1;
    auto canonicalized_axes =
        canonicalize_axes(result.axes_data.data(), result.axes_data_shape, complex_data_rank);

    size_t num_of_axes = result.axes_data.size();
    auto signal_size = get_signal_size(inputs, num_of_axes);

    const auto last_axis = canonicalized_axes.back();
    for (size_t i = 0; i < num_of_axes; ++i) {
        int64_t current_axis = canonicalized_axes[i];
        int64_t current_signal_size = signal_size[i];
        if (current_signal_size != -1) {
            fft_output_shape[current_axis] = static_cast<size_t>(current_signal_size);
            output_shape[current_axis] = static_cast<size_t>(current_signal_size);
        }
    }
    result.last_signal_size = signal_size.back();
    if (signal_size.back() == -1) {
        output_shape[last_axis] = 2 * (result.input_data_shape[last_axis] - 1);
        fft_output_shape[last_axis] = 2 * (result.input_data_shape[last_axis] - 1);
        result.last_signal_size = 2 * (result.input_data_shape[last_axis] - 1);
    }

    output_shape.pop_back();

    result.fft_output_shape = fft_output_shape;
    result.output_shape = output_shape;
    result.axes_data = canonicalized_axes;

    return result;
}

bool evaluate(const ov::element::Type& type, const HostTensorVector& outputs, const HostTensorVector& inputs) {
    auto info = irfft_v9::get_info_for_irfft9_eval(inputs);
    outputs[0]->set_shape(info.output_shape);

    std::vector<float> irfft_result(shape_size(info.output_shape), 0.0f);
    ngraph::runtime::reference::irdft(info.input_data,
                              info.input_data_shape,
                              info.axes_data,
                              irfft_result.data(),
                              info.fft_output_shape,
                              info.output_shape);

    const auto output_type = type;
    ngraph::runtime::reference::fft_postprocessing(outputs, output_type, irfft_result);
    return true;
}
}  // namespace irfft_v9
}

bool ov::op::v9::IRDFT::evaluate(const ov::HostTensorVector& output_values,
                          const ov::HostTensorVector& input_values) const {
    irfft_v9::evaluate(get_output_element_type(0), output_values, input_values);
    return true;
}

bool ov::op::v9::IRDFT::has_evaluate() const {
    return true;
}

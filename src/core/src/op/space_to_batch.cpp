// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/space_to_batch.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>

#include "itt.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/reference/pad.hpp"
#include "openvino/reference/reshape.hpp"
#include "space_to_batch_shape_inference.hpp"

namespace ov {
namespace op {
namespace v1 {
SpaceToBatch::SpaceToBatch(const Output<Node>& data,
                           const Output<Node>& block_shape,
                           const Output<Node>& pads_begin,
                           const Output<Node>& pads_end)
    : Op({data, block_shape, pads_begin, pads_end}) {
    mark_as_precision_sensitive(input(1));
    mark_as_precision_sensitive(input(2));
    mark_as_precision_sensitive(input(3));
    constructor_validate_and_infer_types();
}

void SpaceToBatch::validate_and_infer_types() {
    OV_OP_SCOPE(v1_SpaceToBatch_validate_and_infer_types);
    const auto& data_type = get_input_element_type(0);
    const auto& block_shape_type = get_input_element_type(1);
    const auto& pads_begin_type = get_input_element_type(2);
    const auto& pads_end_type = get_input_element_type(3);

    NODE_VALIDATION_CHECK(this,
                          block_shape_type.is_integral_number(),
                          "block_shape must be an integral number but got (",
                          block_shape_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          pads_begin_type.is_integral_number(),
                          "pads_begin must be an integral number but got (",
                          pads_begin_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          pads_end_type.is_integral_number(),
                          "pads_end must be an integral number but got (",
                          pads_end_type,
                          ").");
    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, data_type, output_shapes[0]);
}

std::shared_ptr<Node> SpaceToBatch::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_SpaceToBatch_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<SpaceToBatch>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool SpaceToBatch::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_SpaceToBatch_visit_attributes);
    return true;
}

namespace space_to_batch {
namespace {
bool evaluate(TensorVector& outputs, const TensorVector& inputs) {
    const auto& data = inputs[0];
    const auto& out = outputs[0];
    const auto elem_size = data.get_element_type().size();

    auto data_shape = data.get_shape();

    if (!(data.get_shape().size() == 3 || data.get_shape().size() == 4 || data.get_shape().size() == 5)) {
        return false;
    }

    const auto block_values_size = shape_size(inputs[1].get_shape());
    const auto block_values = static_cast<const int64_t*>(inputs[1].data());
    const auto pads_begin = static_cast<const int64_t*>(inputs[2].data());
    const auto pads_end = static_cast<const int64_t*>(inputs[3].data());

    const char* pad_value = nullptr;
    const std::vector<char> pad_zero_value(elem_size, 0);
    if (inputs.size() == 4) {
        pad_value = static_cast<const char*>(inputs[3].data());
    } else {
        pad_value = pad_zero_value.data();
    }
    CoordinateDiff pads_begin_vec(pads_begin, pads_begin + shape_size(inputs[2].get_shape()));
    CoordinateDiff pads_end_vec(pads_end, pads_end + shape_size(inputs[2].get_shape()));

    Shape padded_shape(data_shape.size());
    for (size_t i = 0; i < data_shape.size(); ++i) {
        padded_shape[i] = data_shape[i] + pads_begin_vec[i] + pads_end_vec[i];
    }

    std::vector<char> padded_data(shape_size(padded_shape) * elem_size);
    reference::pad(static_cast<const char*>(data.data()),
                   pad_value,
                   padded_data.data(),
                   elem_size,
                   data_shape,
                   padded_shape,
                   pads_begin_vec,
                   pads_end_vec,
                   op::PadMode::CONSTANT);
    data_shape = std::move(padded_shape);

    Shape dispersed_shape(block_values_size + 1);
    std::vector<size_t> axes_order(block_values_size + 1);
    Shape squeezed_shape(data_shape.begin(), data_shape.end());
    std::vector<size_t> plain_axes_order(block_values_size + 1);
    std::iota(plain_axes_order.begin(), plain_axes_order.end(), 0);

    std::vector<char> flat_data(padded_data.begin(), padded_data.end());
    std::vector<char> dispersed_data(shape_size(data_shape) * elem_size);
    std::vector<char> post_transpose_data(shape_size(data_shape) * elem_size);

    for (int64_t block_idx = block_values_size - 1; block_idx >= 0; --block_idx) {
        int64_t sq_shape_idx = block_values_size - 1;
        int64_t axis_idx = axes_order.size() - 1;
        for (int64_t shape_idx = dispersed_shape.size() - 1; shape_idx >= 0; --shape_idx) {
            if (shape_idx == (block_idx + 1)) {
                dispersed_shape[shape_idx] = block_values[block_idx];
                axes_order[0] = shape_idx;
            } else if (shape_idx == block_idx) {
                dispersed_shape[shape_idx] = squeezed_shape[sq_shape_idx] / block_values[block_idx];
                axes_order[axis_idx] = shape_idx;
                axis_idx--;
                sq_shape_idx--;
            } else {
                dispersed_shape[shape_idx] = squeezed_shape[sq_shape_idx];
                axes_order[axis_idx] = shape_idx;
                axis_idx--;
                sq_shape_idx--;
            }
        }

        reference::reshape(flat_data.data(),
                           dispersed_data.data(),
                           data_shape,
                           plain_axes_order,
                           dispersed_shape,
                           elem_size);
        Shape post_transpose_shape(axes_order.size());
        for (size_t i = 0; i < axes_order.size(); ++i) {
            post_transpose_shape[i] = dispersed_shape[axes_order[i]];
        }

        reference::reshape(dispersed_data.data(),
                           post_transpose_data.data(),
                           dispersed_shape,
                           axes_order,
                           post_transpose_shape,
                           elem_size);
        squeezed_shape[0] *= block_values[block_idx];
        squeezed_shape[block_idx] /= block_values[block_idx];

        reference::reshape(post_transpose_data.data(),
                           flat_data.data(),
                           post_transpose_shape,
                           plain_axes_order,
                           squeezed_shape,
                           elem_size);
        data_shape = squeezed_shape;
    }

    std::memcpy(out.data(out.get_element_type()), flat_data.data(), elem_size * shape_size(out.get_shape()));

    return true;
}
}  // namespace
}  // namespace space_to_batch

bool SpaceToBatch::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_SpaceToBatch_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto input_shapes = ov::util::get_tensors_partial_shapes(inputs);
    const auto output_shape = shape_infer(this, input_shapes, make_tensor_accessor(inputs)).front().to_shape();
    outputs[0].set_shape(output_shape);

    return space_to_batch::evaluate(outputs, inputs);
}

bool SpaceToBatch::has_evaluate() const {
    OV_OP_SCOPE(v1_SpaceToBatch_has_evaluate);
    return !get_input_partial_shape(0).is_dynamic() &&
           (get_input_shape(0).size() == 4 || get_input_shape(0).size() == 5);
}
}  // namespace v1
}  // namespace op
}  // namespace ov

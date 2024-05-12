// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/batch_to_space.hpp"

#include "batch_to_space_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "openvino/op/util/slice_plan.hpp"
#include "openvino/reference/reshape.hpp"
#include "openvino/reference/strided_slice.hpp"

namespace ov {
namespace op {
namespace v1 {

BatchToSpace::BatchToSpace(const Output<Node>& data,
                           const Output<Node>& block_shape,
                           const Output<Node>& crops_begin,
                           const Output<Node>& crops_end)
    : Op({data, block_shape, crops_begin, crops_end}) {
    ov::mark_as_precision_sensitive(input(1));
    ov::mark_as_precision_sensitive(input(2));
    ov::mark_as_precision_sensitive(input(3));
    constructor_validate_and_infer_types();
}

void BatchToSpace::validate_and_infer_types() {
    OV_OP_SCOPE(v1_BatchToSpace_validate_and_infer_types);

    const auto& data_et = get_input_element_type(0);
    const auto& block_shape_et = get_input_element_type(1);
    const auto& crops_begin_et = get_input_element_type(2);
    const auto& crops_end_et = get_input_element_type(3);

    element::Type inputs_integer_et{};
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(inputs_integer_et, crops_begin_et, crops_end_et) &&
                              element::Type::merge(inputs_integer_et, inputs_integer_et, block_shape_et),
                          "block_shape, crops_begin and crops_end inputs must have same element type. Got: ",
                          block_shape_et,
                          ", ",
                          crops_begin_et,
                          " and ",
                          crops_end_et);

    NODE_VALIDATION_CHECK(this,
                          inputs_integer_et.is_integral_number(),
                          "block_shape and crops inputs must have integer element type. Got: ",
                          inputs_integer_et);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, data_et, output_shapes[0]);
}

std::shared_ptr<Node> BatchToSpace::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_BatchToSpace_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<BatchToSpace>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

bool BatchToSpace::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_BatchToSpace_visit_attributes);
    return true;
}

namespace {
bool batch_to_space_evaluate(TensorVector& outputs, const TensorVector& inputs) {
    const auto& in = inputs[0];
    const auto elem_size = in.get_element_type().size();

    auto data_shape = in.get_shape();

    auto const block_values_size = shape_size(inputs[1].get_shape());

    const auto* block_values = inputs[1].data<int64_t>();
    const auto* crops_begin_values = inputs[2].data<int64_t>();
    const auto* crops_end_values = inputs[3].data<int64_t>();

    ov::Shape dispersed_shape(1);
    dispersed_shape.insert(dispersed_shape.end(), data_shape.begin(), data_shape.end());
    std::vector<size_t> axes_order(block_values_size + 1);
    std::vector<size_t> plain_axes_order(block_values_size + 1);
    std::iota(plain_axes_order.begin(), plain_axes_order.end(), 0);
    ov::Shape squeezed_shape(data_shape.begin(), data_shape.end());
    if (squeezed_shape.size() > block_values_size) {
        return false;
    }

    auto* in_first = static_cast<const char*>(in.data());

    // Copy input tensor to not overwrite evaluate's inputs tensors passed as const.
    // The evaluate algorithm should be improved to avoid additional data copy.
    auto flat_in = Tensor(in.get_element_type(), data_shape);
    auto* flat_data = static_cast<char*>(flat_in.data());
    std::memcpy(flat_data, in_first, flat_in.get_byte_size());
    std::vector<char> dispersed_data(shape_size(data_shape) * elem_size);

    ov::Shape post_transpose_shape(axes_order.size());
    std::vector<char> post_transpose_data(shape_size(data_shape) * elem_size);

    for (size_t block_idx = 1; block_idx < block_values_size; ++block_idx) {
        dispersed_shape[0] = block_values[block_idx];
        dispersed_shape[1] /= block_values[block_idx];
        ov::reference::reshape(flat_data,
                               dispersed_data.data(),
                               data_shape,
                               plain_axes_order,
                               dispersed_shape,
                               elem_size);

        for (size_t axis_idx = 0, val = 1; axis_idx <= block_values_size; ++axis_idx) {
            if ((block_idx + 1) == axis_idx) {
                axes_order[axis_idx] = 0;
            } else {
                axes_order[axis_idx] = val;
                ++val;
            }
        }

        for (size_t axis_idx = 0; axis_idx < axes_order.size(); ++axis_idx) {
            post_transpose_shape[axis_idx] = dispersed_shape[axes_order[axis_idx]];
        }

        ov::reference::reshape(dispersed_data.data(),
                               post_transpose_data.data(),
                               dispersed_shape,
                               axes_order,
                               post_transpose_shape,
                               elem_size);
        squeezed_shape[0] = dispersed_shape[1];
        squeezed_shape[block_idx] *= block_values[block_idx];
        dispersed_shape[block_idx + 1] = squeezed_shape[block_idx];
        ov::reference::reshape(post_transpose_data.data(),
                               flat_data,
                               post_transpose_shape,
                               plain_axes_order,
                               squeezed_shape,
                               elem_size);
        data_shape = squeezed_shape;
    }

    std::vector<int64_t> upper_bounds_values(data_shape.size());
    for (size_t i = 0; i < data_shape.size(); ++i) {
        upper_bounds_values[i] = data_shape[i] - crops_end_values[i];
    }

    std::vector<size_t> begin_mask(data_shape.size(), 0);
    std::vector<size_t> end_mask(data_shape.size(), 0);

    std::vector<int64_t> begins(shape_size(inputs[2].get_shape()));
    begins.assign(crops_begin_values, crops_begin_values + shape_size(inputs[2].get_shape()));

    std::vector<int64_t> default_strides(begins.size(), 1);
    const auto slice_plan = ov::op::util::make_slice_plan(data_shape,
                                                          begins,
                                                          upper_bounds_values,
                                                          default_strides,
                                                          begin_mask,
                                                          end_mask,
                                                          AxisSet(),
                                                          AxisSet(),
                                                          AxisSet());
    ov::reference::strided_slice(flat_data, static_cast<char*>(outputs[0].data()), data_shape, slice_plan, elem_size);
    return true;
}
}  // namespace

bool BatchToSpace::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v1_BatchToSpace_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);

    std::vector<ov::PartialShape> input_shapes;
    for (const auto& in : inputs) {
        input_shapes.emplace_back(in.get_shape());
    }

    const auto output_shape = shape_infer(this, input_shapes, ov::make_tensor_accessor(inputs)).front().to_shape();
    outputs[0].set_shape(output_shape);

    return batch_to_space_evaluate(outputs, inputs);
}

bool BatchToSpace::has_evaluate() const {
    OV_OP_SCOPE(v1_BatchToSpace_has_evaluate);
    return !get_input_partial_shape(0).is_dynamic() && get_input_shape(0).size() >= 2 &&
           get_input_shape(0).size() <= shape_size(get_input_shape(1));
}
}  // namespace v1
}  // namespace op
}  // namespace ov

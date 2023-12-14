// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/gather_base.hpp"

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "gather_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/reference/gather.hpp"
#include "validation_util.hpp"

namespace ov {
namespace op {
namespace gather {
namespace {

Shape out_shape_infer(const Shape& data_shape, const Shape& indices_shape, int64_t axis, int64_t batch_dims) {
    Shape out_shape;
    out_shape.reserve(data_shape.size() + indices_shape.size() - 1 - batch_dims);
    auto out_dim_inserter = std::copy_n(data_shape.begin(), axis, std::back_inserter(out_shape));
    out_dim_inserter = std::copy(indices_shape.begin() + batch_dims, indices_shape.end(), out_dim_inserter);
    std::copy(std::next(data_shape.begin(), axis + 1), data_shape.end(), out_dim_inserter);

    return out_shape;
}

bool cf_gather_with_subgraph(OutputVector& output_values,
                             const OutputVector& input_values,
                             const PartialShape& gather_ps) {
    if (gather_ps.is_dynamic() || input_values.size() != 3) {
        return false;
    }

    const auto concat = std::dynamic_pointer_cast<v0::Concat>(input_values[0].get_node_shared_ptr());
    const auto indices = std::dynamic_pointer_cast<v0::Constant>(input_values[1].get_node_shared_ptr());
    const auto axis = std::dynamic_pointer_cast<v0::Constant>(input_values[2].get_node_shared_ptr());

    if (!concat || !indices || !axis) {
        return false;
    }

    // only along axis=0
    if (axis->cast_vector<int64_t>()[0] != 0 || concat->get_axis() != 0) {
        return false;
    }
    // only single indices are accepted
    const auto indices_shape = indices->get_shape();
    if (indices_shape.size() > 1 || (indices_shape.size() == 1 && indices_shape[0] > 1)) {
        return false;
    }
    // concat inputs are 1D and their count is equal to Concat output shape
    if (concat->get_output_partial_shape(0).is_dynamic()) {
        return false;
    }
    const auto concat_inputs = concat->inputs();
    // concat inputs must be single elements
    if (concat_inputs.size() != shape_size(concat->get_shape())) {
        return false;
    }

    const int64_t rank = concat->get_shape()[0];
    const auto raw_index = indices->cast_vector<int64_t>()[0];
    const auto positive_index = ov::util::normalize(raw_index, rank);
    OPENVINO_ASSERT(positive_index >= 0 && positive_index < rank);

    // gather takes exactly one element out of the Concat output
    const auto gathered_concat_input = concat_inputs[positive_index].get_source_output().get_node_shared_ptr();
    // Concat inputs are 1D, resulting tensor shape depends on Gather indices
    auto gathered = gathered_concat_input;
    if (indices_shape.empty()) {
        // gathering a scalar
        const auto axis_const = v0::Constant::create(element::i64, Shape{1}, {0});
        gathered = std::make_shared<v0::Squeeze>(gathered_concat_input, axis_const);
    }

    output_values[0] = gathered;

    return true;
}

bool have_indices_and_axis_bound_set(const util::GatherBase* const gather) {
    return ov::have_node_inputs_bounds_set(gather, 1, 2);
}

}  // namespace
}  // namespace gather

namespace util {

GatherBase::GatherBase(const Output<Node>& data,
                       const Output<Node>& indices,
                       const Output<Node>& axis,
                       const int64_t batch_dims)
    : Op({data, indices, axis}),
      m_batch_dims(batch_dims) {
    constructor_validate_and_infer_types();
}

void GatherBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_GatherBase_validate_and_infer_types);

    const auto& data_type = get_input_element_type(0);
    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));

    set_output_type(0, data_type, output_shapes[0]);
}

int64_t GatherBase::get_axis() const {
    const auto& const_op = ov::util::get_constant_from_source(input_value(2));
    OPENVINO_ASSERT(const_op, "axis value is not set");

    const auto axis = const_op->cast_vector<int64_t>()[0];
    if (axis < 0 && get_input_partial_shape(0).rank().is_static()) {
        return axis + get_input_partial_shape(0).rank().get_length();
    } else {
        return axis;
    }
}

const int64_t& GatherBase::get_batch_dims() const {
    return m_batch_dims;
}

void GatherBase::set_batch_dims(int64_t batch_dims) {
    m_batch_dims = batch_dims;
}

bool GatherBase::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(util_GatherBase_evaluate);

    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 3);

    OPENVINO_ASSERT(inputs[2].get_element_type().is_integral_number(), "axis must be of integral data type.");

    const auto& data = inputs[0];
    const auto& data_shape = data.get_shape();
    const auto indices = get_tensor_data_as<int64_t>(inputs[1]);
    const auto& indices_shape = inputs[1].get_shape();
    const auto element_size = data.get_element_type().size();

    const auto axis = ov::util::normalize(get_tensor_data_as<int64_t>(inputs[2])[0], data_shape.size());
    const auto batch_dims = ov::util::normalize(m_batch_dims, indices_shape.size());

    const auto out_shape = gather::out_shape_infer(data_shape, indices_shape, axis, batch_dims);
    auto& output = outputs[0];
    output.set_shape(out_shape);

    ov::reference::gather(static_cast<const char*>(data.data()),
                          indices.data(),
                          static_cast<char*>(output.data()),
                          data_shape,
                          indices_shape,
                          out_shape,
                          axis,
                          element_size,
                          batch_dims);

    return true;
}

bool GatherBase::has_evaluate() const {
    return true;
}

bool GatherBase::evaluate_lower(TensorVector& output_values) const {
    return gather::have_indices_and_axis_bound_set(this) && default_lower_bound_evaluator(this, output_values);
}

bool GatherBase::evaluate_upper(TensorVector& output_values) const {
    return gather::have_indices_and_axis_bound_set(this) && default_upper_bound_evaluator(this, output_values);
}

bool GatherBase::evaluate_label(TensorLabelVector& output_labels) const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    return gather::have_indices_and_axis_bound_set(this) && ov::util::default_label_evaluator(this, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

bool GatherBase::constant_fold(OutputVector& output_values, const OutputVector& input_values) {
    // try the regular constant folding just for the Gather node
    if (Node::constant_fold(output_values, input_values)) {
        return true;
    } else {
        return gather::cf_gather_with_subgraph(output_values, input_values, get_output_partial_shape(0));
    }
}
}  // namespace util
}  // namespace op
}  // namespace ov

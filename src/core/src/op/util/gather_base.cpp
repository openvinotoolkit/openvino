// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/gather_base.hpp"

#include "bound_evaluate.hpp"
#include "gather_shape_inference.hpp"
#include "itt.hpp"
#include "ngraph/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/reference/gather.hpp"

ov::op::util::GatherBase::GatherBase(const Output<Node>& data,
                                     const Output<Node>& indices,
                                     const Output<Node>& axis,
                                     const int64_t batch_dims)
    : Op({data, indices, axis}),
      m_batch_dims(batch_dims) {
    constructor_validate_and_infer_types();
}

void ov::op::util::GatherBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_GatherBase_validate_and_infer_types);
    const auto& data_type = get_input_element_type(0);

    const auto& data_pshape = get_input_partial_shape(0);
    const auto& indices_pshape = get_input_partial_shape(1);
    const auto& axis_pshape = get_input_partial_shape(2);
    std::vector<PartialShape> input_shapes = {data_pshape, indices_pshape, axis_pshape};
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, data_type, output_shapes[0]);
}

int64_t ov::op::util::GatherBase::get_axis() const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto& const_op = get_constant_from_source(input_value(2));
    OPENVINO_SUPPRESS_DEPRECATED_END
    OPENVINO_ASSERT(const_op, "axis value is not set");

    int64_t axis = const_op->cast_vector<int64_t>()[0];
    if (axis < 0) {
        const auto& data_rank = get_input_partial_shape(0).rank();
        if (data_rank.is_static()) {
            axis += data_rank.get_length();
        }
    }
    return axis;
}

const int64_t& ov::op::util::GatherBase::get_batch_dims() const {
    return m_batch_dims;
}

void ov::op::util::GatherBase::set_batch_dims(int64_t batch_dims) {
    m_batch_dims = batch_dims;
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace gather {
namespace {
template <ov::element::Type_t ET>
bool evaluate(const ngraph::HostTensorPtr& arg0,
              const ngraph::HostTensorPtr& arg1,
              const ngraph::HostTensorPtr& out,
              int64_t axis,
              int64_t batch_dims) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ov::Shape params_shape = arg0->get_shape();
    ov::Shape indices_shape = arg1->get_shape();
    ov::Shape out_shape(params_shape.size() + indices_shape.size() - 1 - batch_dims);
    int64_t i = 0;
    for (; i < axis; i++) {
        out_shape[i] = params_shape[i];
    }
    for (int64_t j = batch_dims; j < static_cast<int64_t>(indices_shape.size()); i++, j++) {
        out_shape[i] = indices_shape[j];
    }
    for (int64_t j = axis + 1; j < static_cast<int64_t>(params_shape.size()); i++, j++) {
        out_shape[i] = params_shape[j];
    }

    out->set_shape(out_shape);

    if (arg1->get_element_type() == ov::element::i64) {
        ov::reference::gather<T, int64_t>(arg0->get_data_ptr<ET>(),
                                          arg1->get_data_ptr<int64_t>(),
                                          out->get_data_ptr<ET>(),
                                          arg0->get_shape(),
                                          arg1->get_shape(),
                                          out->get_shape(),
                                          axis,
                                          batch_dims);
    } else if (arg1->get_element_type() == ov::element::i32) {
        ov::reference::gather<T, int32_t>(arg0->get_data_ptr<ET>(),
                                          arg1->get_data_ptr<int32_t>(),
                                          out->get_data_ptr<ET>(),
                                          arg0->get_shape(),
                                          arg1->get_shape(),
                                          out->get_shape(),
                                          axis,
                                          batch_dims);
    } else {
        OPENVINO_THROW("Unexpected type ", arg1->get_element_type().c_type_string(), " for Gather evaluate method.");
    }

    return true;
}

bool evaluate_gather(const ngraph::HostTensorPtr& arg0,
                     const ngraph::HostTensorPtr& arg1,
                     const ngraph::HostTensorPtr& out,
                     int64_t axis,
                     int64_t batch_dims = 0) {
    bool rc = true;

    using ov::element::Type_t;
    switch (out->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_gather, i32, arg0, arg1, out, axis, batch_dims);
        OPENVINO_TYPE_CASE(evaluate_gather, i64, arg0, arg1, out, axis, batch_dims);
        OPENVINO_TYPE_CASE(evaluate_gather, i8, arg0, arg1, out, axis, batch_dims);
        OPENVINO_TYPE_CASE(evaluate_gather, u8, arg0, arg1, out, axis, batch_dims);
        OPENVINO_TYPE_CASE(evaluate_gather, u32, arg0, arg1, out, axis, batch_dims);
        OPENVINO_TYPE_CASE(evaluate_gather, u64, arg0, arg1, out, axis, batch_dims);
        OPENVINO_TYPE_CASE(evaluate_gather, f32, arg0, arg1, out, axis, batch_dims);
        OPENVINO_TYPE_CASE(evaluate_gather, boolean, arg0, arg1, out, axis, batch_dims);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool cf_gather_with_subgraph(ov::OutputVector& output_values,
                             const ov::OutputVector& input_values,
                             const ov::PartialShape& gather_ps) {
    if (gather_ps.is_dynamic() || input_values.size() != 3) {
        return false;
    }

    const auto concat = std::dynamic_pointer_cast<ov::op::v0::Concat>(input_values[0].get_node_shared_ptr());
    const auto indices = std::dynamic_pointer_cast<ov::op::v0::Constant>(input_values[1].get_node_shared_ptr());
    const auto axis = std::dynamic_pointer_cast<ov::op::v0::Constant>(input_values[2].get_node_shared_ptr());

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
    const int64_t raw_index = indices->cast_vector<int64_t>()[0];
    const int64_t positive_index = raw_index < 0 ? rank + raw_index : raw_index;
    OPENVINO_ASSERT(positive_index >= 0 && positive_index < rank);

    // gather takes exactly one element out of the Concat output
    const auto gathered_concat_input = concat_inputs[positive_index].get_source_output().get_node_shared_ptr();
    // Concat inputs are 1D, resulting tensor shape depends on Gather indices
    auto gathered = gathered_concat_input;
    if (indices_shape.empty()) {
        // gathering a scalar
        const auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        gathered = std::make_shared<ov::op::v0::Squeeze>(gathered_concat_input, axis_const);
    }

    output_values[0] = gathered;

    return true;
}
}  // namespace
}  // namespace gather

bool ov::op::util::GatherBase::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(util_GatherBase_evaluate);
    OPENVINO_SUPPRESS_DEPRECATED_START
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(inputs, 3));
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(outputs, 1));
    OPENVINO_SUPPRESS_DEPRECATED_END
    int64_t axis = 0;
    switch (inputs[2]->get_element_type()) {
    case element::Type_t::i32:
        axis = inputs[2]->get_data_ptr<element::Type_t::i32>()[0];
        break;
    case element::Type_t::i64:
        axis = inputs[2]->get_data_ptr<element::Type_t::i64>()[0];
        break;
    case element::Type_t::i8:
        axis = inputs[2]->get_data_ptr<element::Type_t::i8>()[0];
        break;
    case element::Type_t::i16:
        axis = inputs[2]->get_data_ptr<element::Type_t::i16>()[0];
        break;
    case element::Type_t::u8:
        axis = inputs[2]->get_data_ptr<element::Type_t::u8>()[0];
        break;
    case element::Type_t::u16:
        axis = inputs[2]->get_data_ptr<element::Type_t::u16>()[0];
        break;
    case element::Type_t::u32:
        axis = inputs[2]->get_data_ptr<element::Type_t::u32>()[0];
        break;
    case element::Type_t::u64:
        axis = inputs[2]->get_data_ptr<element::Type_t::u64>()[0];
        break;
    default:
        OPENVINO_THROW("axis must be of integral data type.");
    }

    if (axis < 0) {
        const auto input_rank = inputs[0]->get_shape().size();
        axis += input_rank;
    }

    int64_t batch_dims = m_batch_dims;
    if (batch_dims < 0) {
        const auto indices_rank = inputs[1]->get_shape().size();
        batch_dims += indices_rank;
    }

    return gather::evaluate_gather(inputs[0], inputs[1], outputs[0], axis, batch_dims);
}

bool ov::op::util::GatherBase::evaluate_lower(ov::TensorVector& output_values) const {
    if (!get_input_tensor(1).has_and_set_bound() || !get_input_tensor(2).has_and_set_bound())
        return false;
    return default_lower_bound_evaluator(this, output_values);
}

bool ov::op::util::GatherBase::evaluate_upper(ov::TensorVector& output_values) const {
    if (!get_input_tensor(1).has_and_set_bound() || !get_input_tensor(2).has_and_set_bound())
        return false;
    return default_upper_bound_evaluator(this, output_values);
}

bool ov::op::util::GatherBase::evaluate_label(TensorLabelVector& output_labels) const {
    if (!get_input_tensor(1).has_and_set_bound() || !get_input_tensor(2).has_and_set_bound())
        return false;
    OPENVINO_SUPPRESS_DEPRECATED_START
    return default_label_evaluator(this, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

bool ov::op::util::GatherBase::constant_fold(OutputVector& output_values, const OutputVector& input_values) {
    // try the regular constant folding just for the Gather node
    if (Node::constant_fold(output_values, input_values)) {
        return true;
    } else {
        return gather::cf_gather_with_subgraph(output_values, input_values, get_output_partial_shape(0));
    }
}

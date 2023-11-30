// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "gather_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/util/gather_base.hpp"
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
    const int64_t raw_index = indices->cast_vector<int64_t>()[0];
    const int64_t positive_index = raw_index < 0 ? rank + raw_index : raw_index;
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

struct Evaluate : element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t DATA_ET, class DT = fundamental_type_for<DATA_ET>>
    static result_type visit(const Tensor& data,
                             const Tensor& indices,
                             Tensor& out,
                             const Shape& data_shape,
                             const Shape& indices_shape,
                             const Shape& out_shape,
                             const int64_t axis,
                             const int64_t batch_dims) {
        using namespace ov::element;
        return IfTypeOf<i32, i64>::apply<EvaluateByIndicesType>(indices.get_element_type(),
                                                                data.data<const DT>(),
                                                                indices,
                                                                out.data<DT>(),
                                                                data_shape,
                                                                indices_shape,
                                                                out_shape,
                                                                axis,
                                                                batch_dims);
    }

private:
    struct EvaluateByIndicesType : element::NoAction<bool> {
        using element::NoAction<bool>::visit;

        template <element::Type_t INDICES_ET, class DT, class IT = fundamental_type_for<INDICES_ET>>
        static result_type visit(const DT* const data,
                                 const Tensor& indices,
                                 DT* const output,
                                 const Shape& data_shape,
                                 const Shape& indices_shape,
                                 const Shape& out_shape,
                                 const int64_t axis,
                                 const int64_t batch_dims) {
            reference::gather(data,
                              indices.data<const IT>(),
                              output,
                              data_shape,
                              indices_shape,
                              out_shape,
                              axis,
                              batch_dims);
            return true;
        }
    };
};
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
    const auto& data_pshape = get_input_partial_shape(0);
    const auto& indices_pshape = get_input_partial_shape(1);
    const auto& axis_pshape = get_input_partial_shape(2);
    const std::vector<PartialShape> input_shapes = {data_pshape, indices_pshape, axis_pshape};
    const auto output_shapes = shape_infer(this, input_shapes);

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
    const auto& indices = inputs[1];
    const auto& data_rank = data.get_shape().size();

    const auto axis = ov::util::normalize(get_tensor_data_as<int64_t>(inputs[2])[0], data_rank);
    const auto batch_dims = ov::util::normalize(m_batch_dims, data_rank);

    const auto& data_shape = data.get_shape();
    const auto& indices_shape = indices.get_shape();
    const auto out_shape = gather::out_shape_infer(data_shape, indices_shape, axis, batch_dims);
    auto& output = outputs[0];
    output.set_shape(out_shape);

    using namespace ov::element;
    return IfTypeOf<boolean, f16, f32, i8, i32, i64, u8, u32, u64>::apply<gather::Evaluate>(data.get_element_type(),
                                                                                            data,
                                                                                            indices,
                                                                                            output,
                                                                                            data_shape,
                                                                                            indices_shape,
                                                                                            out_shape,
                                                                                            axis,
                                                                                            batch_dims);
}

bool GatherBase::evaluate_lower(TensorVector& output_values) const {
    return gather::have_indices_and_axis_bound_set(this) && default_lower_bound_evaluator(this, output_values);
}

bool GatherBase::evaluate_upper(TensorVector& output_values) const {
    return gather::have_indices_and_axis_bound_set(this) && default_upper_bound_evaluator(this, output_values);
}

bool GatherBase::evaluate_label(TensorLabelVector& output_labels) const {
    OPENVINO_SUPPRESS_DEPRECATED_START
    return gather::have_indices_and_axis_bound_set(this) && default_label_evaluator(this, output_labels);
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

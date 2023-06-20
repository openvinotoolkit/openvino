// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/scatter_elements_update_base.hpp"

#include <scatter_elements_update_shape_inference.hpp>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/runtime/reference/scatter_elements_update.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace op {

ov::op::util::ScatterElementsUpdateBase::ScatterElementsUpdateBase(const Output<Node>& data,
                                                                   const Output<Node>& indices,
                                                                   const Output<Node>& updates,
                                                                   const Output<Node>& axis)
    : Op({data, indices, updates, axis}) {
    constructor_validate_and_infer_types();
}

void ov::op::util::ScatterElementsUpdateBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_ScatterElementsUpdateBase_validate_and_infer_types);
    OPENVINO_SUPPRESS_DEPRECATED_START
    const element::Type& data_et = get_input_element_type(0);
    const element::Type& indices_et = get_input_element_type(1);
    const element::Type& updates_et = get_input_element_type(2);
    const element::Type& axis_et = get_input_element_type(3);

    NODE_VALIDATION_CHECK(this,
                          indices_et.is_integral(),
                          "Indices element type must be integral_number, but is: ",
                          indices_et);

    NODE_VALIDATION_CHECK(this, axis_et.is_integral(), "Axis element type must be integral_number, but is: ", axis_et);

    element::Type merged_type;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(merged_type, data_et, updates_et),
                          "Data type and updates type are required to be the same. ",
                          "Got: ",
                          data_et,
                          " and: ",
                          updates_et);
    const auto output_shape = shape_infer(this, get_node_input_partial_shapes(*this)).front();
    OPENVINO_SUPPRESS_DEPRECATED_END
    element::Type out_et = get_input_element_type(0);
    std::ignore = element::Type::merge(out_et, get_input_element_type(0), get_input_element_type(2));
    set_output_type(0, out_et, output_shape);
    if (output_shape.is_dynamic()) {
        set_input_is_relevant_to_shape(0);
    }
}

bool op::util::ScatterElementsUpdateBase::has_evaluate() const {
    OV_OP_SCOPE(util_ScatterElementsUpdateBase_has_evaluate);

    switch (get_output_element_type(0)) {
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u32:
    case ngraph::element::u64:
    case ngraph::element::f16:
    case ngraph::element::f32:
        break;
    default:
        return false;
    }
    switch (get_input_element_type(1)) {
    case ngraph::element::i8:
    case ngraph::element::i16:
    case ngraph::element::i32:
    case ngraph::element::i64:
    case ngraph::element::u8:
    case ngraph::element::u16:
    case ngraph::element::u32:
    case ngraph::element::u64:
        break;
    default:
        return false;
    }
    return true;
}

bool op::util::ScatterElementsUpdateBase::evaluate_lower(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(util_ScatterNDUpdate_evaluate_lower);
    return get_input_tensor(1).has_and_set_bound() && ov::default_lower_bound_evaluator(this, output_values);
}

bool op::util::ScatterElementsUpdateBase::evaluate_upper(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(util_ScatterNDUpdate_evaluate_upper);
    return get_input_tensor(1).has_and_set_bound() && ov::default_upper_bound_evaluator(this, output_values);
}

bool op::util::ScatterElementsUpdateBase::evaluate_label(TensorLabelVector& output_labels) const {
    OV_OP_SCOPE(util_ScatterNDUpdate_evaluate_label);

    OPENVINO_SUPPRESS_DEPRECATED_START
    return ov::default_label_evaluator(this, {0, 2}, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

namespace scatter_element_update {
namespace {
template <element::Type_t DT, element::Type_t IT, element::Type_t AT>
bool evaluate(const HostTensorPtr& data,
              const HostTensorPtr& indices,
              const HostTensorPtr& updates,
              const HostTensorPtr& axis,
              const HostTensorPtr& out,
              const int64_t normalized_axis) {
    using DataType = typename element_type_traits<DT>::value_type;
    using IndicesType = typename element_type_traits<IT>::value_type;

    out->set_shape(data->get_shape());

    ngraph::runtime::reference::scatter_elem_update<DataType, IndicesType>(data->get_data_ptr<DT>(),
                                                                           indices->get_data_ptr<IT>(),
                                                                           updates->get_data_ptr<DT>(),
                                                                           normalized_axis,
                                                                           out->get_data_ptr<DT>(),
                                                                           data->get_shape(),
                                                                           indices->get_shape());

    return true;
}

#define TYPE_AXS_CASE(a, ...)                                      \
    case element::Type_t::a: {                                     \
        OV_OP_SCOPE(OV_PP_CAT3(scatter_element_update_axs, _, a)); \
        rc = evaluate<DT, IT, element::Type_t::a>(__VA_ARGS__);    \
    } break;

template <element::Type_t DT, element::Type_t IT>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& arg2,
              const HostTensorPtr& arg3,
              const HostTensorPtr& out,
              const int64_t normalized_axis) {
    auto axis_type = arg3->get_element_type();

    // Dispatch specialization based on axis data type.
    bool rc = true;

    switch (axis_type) {
        TYPE_AXS_CASE(i8, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_AXS_CASE(i16, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_AXS_CASE(i32, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_AXS_CASE(i64, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_AXS_CASE(u8, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_AXS_CASE(u16, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_AXS_CASE(u32, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_AXS_CASE(u64, arg0, arg1, arg2, arg3, out, normalized_axis);
    default:
        rc = false;
        break;
    }
    return rc;
}

#define TYPE_IND_CASE(a, ...)                                      \
    case element::Type_t::a: {                                     \
        OV_OP_SCOPE(OV_PP_CAT3(scatter_element_update_ind, _, a)); \
        rc = evaluate<DT, element::Type_t::a>(__VA_ARGS__);        \
    } break;

template <element::Type_t DT>
bool evaluate(const HostTensorPtr& arg0,
              const HostTensorPtr& arg1,
              const HostTensorPtr& arg2,
              const HostTensorPtr& arg3,
              const HostTensorPtr& out,
              const int64_t normalized_axis) {
    auto indices_type = arg1->get_element_type();

    // Dispatch specialization based on indicies data type.
    bool rc = true;

    switch (indices_type) {
        TYPE_IND_CASE(i8, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_IND_CASE(i16, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_IND_CASE(i32, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_IND_CASE(i64, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_IND_CASE(u8, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_IND_CASE(u16, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_IND_CASE(u32, arg0, arg1, arg2, arg3, out, normalized_axis);
        TYPE_IND_CASE(u64, arg0, arg1, arg2, arg3, out, normalized_axis);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool evaluate_scatter_element_update(const HostTensorPtr& arg0,
                                     const HostTensorPtr& arg1,
                                     const HostTensorPtr& arg2,
                                     const HostTensorPtr& arg3,
                                     const HostTensorPtr& out,
                                     const int64_t normalized_axis) {
    bool rc = true;

    switch (out->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_scatter_element_update, i16, arg0, arg1, arg2, arg3, out, normalized_axis);
        NGRAPH_TYPE_CASE(evaluate_scatter_element_update, i32, arg0, arg1, arg2, arg3, out, normalized_axis);
        NGRAPH_TYPE_CASE(evaluate_scatter_element_update, i64, arg0, arg1, arg2, arg3, out, normalized_axis);
        NGRAPH_TYPE_CASE(evaluate_scatter_element_update, u32, arg0, arg1, arg2, arg3, out, normalized_axis);
        NGRAPH_TYPE_CASE(evaluate_scatter_element_update, u64, arg0, arg1, arg2, arg3, out, normalized_axis);
        NGRAPH_TYPE_CASE(evaluate_scatter_element_update, f16, arg0, arg1, arg2, arg3, out, normalized_axis);
        NGRAPH_TYPE_CASE(evaluate_scatter_element_update, f32, arg0, arg1, arg2, arg3, out, normalized_axis);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace scatter_element_update

bool op::util::ScatterElementsUpdateBase::evaluate_scatter_element_update(const HostTensorVector& outputs,
                                                                          const HostTensorVector& inputs) const {
    NGRAPH_CHECK(inputs[3]->get_element_type().is_integral_number(), "axis element type is not integral data type");

    OPENVINO_SUPPRESS_DEPRECATED_START
    int64_t axis = host_tensor_2_vector<int64_t>(inputs[3])[0];
    OPENVINO_SUPPRESS_DEPRECATED_END
    const auto& input_rank = get_input_partial_shape(0).rank();
    int64_t normalized_axis = axis;

    if (normalized_axis < 0) {
        if (input_rank.is_static()) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            normalized_axis = ngraph::normalize_axis(this, axis, input_rank);
            OPENVINO_SUPPRESS_DEPRECATED_END
        } else {
            OPENVINO_SUPPRESS_DEPRECATED_START
            normalized_axis = ngraph::normalize_axis(this, axis, static_cast<int64_t>(inputs[0]->get_shape().size()));
            OPENVINO_SUPPRESS_DEPRECATED_END
        }
    }

    return scatter_element_update::evaluate_scatter_element_update(inputs[0],
                                                                   inputs[1],
                                                                   inputs[2],
                                                                   inputs[3],
                                                                   outputs[0],
                                                                   normalized_axis);
}

bool op::util::ScatterElementsUpdateBase::evaluate(const HostTensorVector& outputs,
                                                   const HostTensorVector& inputs) const {
    OV_OP_SCOPE(util_ScatterElementsUpdate_evaluate);
    return evaluate_scatter_element_update(outputs, inputs);
}

}  // namespace op
}  // namespace ov

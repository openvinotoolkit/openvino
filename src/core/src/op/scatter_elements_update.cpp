// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_elements_update.hpp"

#include <scatter_elements_update_shape_inference.hpp>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/runtime/reference/scatter_elements_update.hpp"
#include "openvino/core/validation_util.hpp"

using namespace ov;
using namespace std;

op::v3::ScatterElementsUpdate::ScatterElementsUpdate(const Output<Node>& data,
                                                     const Output<Node>& indices,
                                                     const Output<Node>& updates,
                                                     const Output<Node>& axis)
    : ov::op::util::ScatterElementsUpdateBase(data, indices, updates, axis) {
    constructor_validate_and_infer_types();
}

bool op::v3::ScatterElementsUpdate::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_ScatterElementsUpdate_visit_attributes);
    return true;
}

shared_ptr<Node> op::v3::ScatterElementsUpdate::clone_with_new_inputs(const OutputVector& inputs) const {
    OV_OP_SCOPE(v3_ScatterElementsUpdate_clone_with_new_inputs);
    NODE_VALIDATION_CHECK(this,
                          inputs.size() == get_input_size(),
                          "clone_with_new_inputs() required inputs size: ",
                          get_input_size(),
                          "Got: ",
                          inputs.size());

    return make_shared<v3::ScatterElementsUpdate>(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3));
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

bool op::v3::ScatterElementsUpdate::evaluate_scatter_element_update(const HostTensorVector& outputs,
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

bool op::v3::ScatterElementsUpdate::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v3_ScatterElementsUpdate_evaluate);
    return evaluate_scatter_element_update(outputs, inputs);
}

bool op::v3::ScatterElementsUpdate::has_evaluate() const {
    OV_OP_SCOPE(v3_ScatterElementsUpdate_has_evaluate);

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

bool op::v3::ScatterElementsUpdate::evaluate_lower(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate_lower);
    return get_input_tensor(1).has_and_set_bound() && ov::default_lower_bound_evaluator(this, output_values);
}

bool op::v3::ScatterElementsUpdate::evaluate_upper(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate_upper);
    return get_input_tensor(1).has_and_set_bound() && ov::default_upper_bound_evaluator(this, output_values);
}

bool op::v3::ScatterElementsUpdate::evaluate_label(TensorLabelVector& output_labels) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate_label);

    OPENVINO_SUPPRESS_DEPRECATED_START
    return ov::default_label_evaluator(this, {0, 2}, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

namespace ov {
op::v12::ScatterElementsUpdate::ScatterElementsUpdate(const Output<Node>& data,
                                                      const Output<Node>& indices,
                                                      const Output<Node>& updates,
                                                      const Output<Node>& axis,
                                                      const Reduction reduction,
                                                      const bool use_init_val)
    : op::util::ScatterElementsUpdateBase(data, indices, updates, axis),
      m_reduction{reduction},
      m_use_init_val{use_init_val} {
    constructor_validate_and_infer_types();
}

bool op::v12::ScatterElementsUpdate::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v12_ScatterElementsUpdate_visit_attributes);
    visitor.on_attribute("reduction", m_reduction);
    visitor.on_attribute("use_init_val", m_use_init_val);
    return true;
}

shared_ptr<Node> op::v12::ScatterElementsUpdate::clone_with_new_inputs(const OutputVector& inputs) const {
    OV_OP_SCOPE(v12_ScatterElementsUpdate_clone_with_new_inputs);
    NODE_VALIDATION_CHECK(this,
                          inputs.size() == get_input_size(),
                          "clone_with_new_inputs() required inputs size: ",
                          get_input_size(),
                          "Got: ",
                          inputs.size());

    return make_shared<v12::ScatterElementsUpdate>(inputs.at(0),
                                                   inputs.at(1),
                                                   inputs.at(2),
                                                   inputs.at(3),
                                                   m_reduction,
                                                   m_use_init_val);
}

template <>
OPENVINO_API EnumNames<op::v12::ScatterElementsUpdate::Reduction>&
EnumNames<op::v12::ScatterElementsUpdate::Reduction>::get() {
    static auto enum_names = EnumNames<op::v12::ScatterElementsUpdate::Reduction>(
        "op::v12::ScatterElementsUpdate::Reduction",
        {{"none", op::v12::ScatterElementsUpdate::Reduction::NONE},
         {"sum", op::v12::ScatterElementsUpdate::Reduction::SUM},
         {"prod", op::v12::ScatterElementsUpdate::Reduction::PROD},
         {"min", op::v12::ScatterElementsUpdate::Reduction::MIN},
         {"max", op::v12::ScatterElementsUpdate::Reduction::MAX},
         {"mean", op::v12::ScatterElementsUpdate::Reduction::MEAN}});
    return enum_names;
}
namespace op {
std::ostream& operator<<(std::ostream& s, const v12::ScatterElementsUpdate::Reduction& reduction) {
    return s << as_string(reduction);
}
}  // namespace op
}  // namespace ov

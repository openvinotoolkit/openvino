// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_elements_update.hpp"

#include <scatter_elements_update_shape_inference.hpp>

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/scatter_elements_update.hpp"

using namespace std;

namespace ov {
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

void op::v12::ScatterElementsUpdate::validate_and_infer_types() {
    OV_OP_SCOPE(v12_ScatterElementsUpdate_validate_and_infer_types);

    if (m_reduction == Reduction::MEAN) {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(0) != element::boolean,
                              "The 'mean' reduction type is not supported for boolean tensors");
    }

    ScatterElementsUpdateBase::validate_and_infer_types();
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

bool op::v12::ScatterElementsUpdate::has_evaluate() const {
    return ScatterElementsUpdateBase::has_evaluate() ||
           (get_output_element_type(0) == element::boolean && is_supported_index_input_element_type());
}

OPENVINO_SUPPRESS_DEPRECATED_START
namespace scatter_elements_update {
namespace {
template <element::Type_t DT, element::Type_t IT, element::Type_t AT>
bool evaluate(const ngraph::HostTensorPtr& data,
              const ngraph::HostTensorPtr& indices,
              const ngraph::HostTensorPtr& updates,
              const ngraph::HostTensorPtr& axis,
              const ngraph::HostTensorPtr& out,
              const int64_t normalized_axis,
              const op::v12::ScatterElementsUpdate::Reduction reduction_type,
              const bool use_init_value) {
    using DataType = typename element_type_traits<DT>::value_type;
    using IndicesType = typename element_type_traits<IT>::value_type;

    out->set_shape(data->get_shape());

    ov::reference::scatter_elem_update<DataType, IndicesType>(data->get_data_ptr<DT>(),
                                                              indices->get_data_ptr<IT>(),
                                                              updates->get_data_ptr<DT>(),
                                                              normalized_axis,
                                                              out->get_data_ptr<DT>(),
                                                              data->get_shape(),
                                                              indices->get_shape(),
                                                              reduction_type,
                                                              use_init_value);

    return true;
}

#define TYPE_AXS_CASE(a, ...)                                      \
    case element::Type_t::a: {                                     \
        OV_OP_SCOPE(OV_PP_CAT3(scatter_element_update_axs, _, a)); \
        rc = evaluate<DT, IT, element::Type_t::a>(__VA_ARGS__);    \
    } break;

template <element::Type_t DT, element::Type_t IT>
bool evaluate(const ngraph::HostTensorPtr& arg0,
              const ngraph::HostTensorPtr& arg1,
              const ngraph::HostTensorPtr& arg2,
              const ngraph::HostTensorPtr& arg3,
              const ngraph::HostTensorPtr& out,
              const int64_t normalized_axis,
              const op::v12::ScatterElementsUpdate::Reduction reduction_type,
              const bool use_init_value) {
    auto axis_type = arg3->get_element_type();

    // Dispatch specialization based on axis data type.
    bool rc = true;

    switch (axis_type) {
        TYPE_AXS_CASE(i8, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_AXS_CASE(i16, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_AXS_CASE(i32, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_AXS_CASE(i64, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_AXS_CASE(u8, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_AXS_CASE(u16, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_AXS_CASE(u32, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_AXS_CASE(u64, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
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
bool evaluate(const ngraph::HostTensorPtr& arg0,
              const ngraph::HostTensorPtr& arg1,
              const ngraph::HostTensorPtr& arg2,
              const ngraph::HostTensorPtr& arg3,
              const ngraph::HostTensorPtr& out,
              const int64_t normalized_axis,
              const op::v12::ScatterElementsUpdate::Reduction reduction_type,
              const bool use_init_value) {
    auto indices_type = arg1->get_element_type();

    // Dispatch specialization based on indicies data type.
    bool rc = true;

    switch (indices_type) {
        TYPE_IND_CASE(i8, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_IND_CASE(i16, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_IND_CASE(i32, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_IND_CASE(i64, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_IND_CASE(u8, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_IND_CASE(u16, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_IND_CASE(u32, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
        TYPE_IND_CASE(u64, arg0, arg1, arg2, arg3, out, normalized_axis, reduction_type, use_init_value);
    default:
        rc = false;
        break;
    }
    return rc;
}

bool evaluate_scatter_elements_update(
    const ngraph::HostTensorPtr& arg0,
    const ngraph::HostTensorPtr& arg1,
    const ngraph::HostTensorPtr& arg2,
    const ngraph::HostTensorPtr& arg3,
    const ngraph::HostTensorPtr& out,
    const int64_t normalized_axis,
    const op::v12::ScatterElementsUpdate::Reduction reduction_type = op::v12::ScatterElementsUpdate::Reduction::NONE,
    const bool use_init_value = false) {
    bool rc = true;

    switch (out->get_element_type()) {
        OPENVINO_TYPE_CASE(evaluate_scatter_element_update,
                           i16,
                           arg0,
                           arg1,
                           arg2,
                           arg3,
                           out,
                           normalized_axis,
                           reduction_type,
                           use_init_value);
        OPENVINO_TYPE_CASE(evaluate_scatter_element_update,
                           i32,
                           arg0,
                           arg1,
                           arg2,
                           arg3,
                           out,
                           normalized_axis,
                           reduction_type,
                           use_init_value);
        OPENVINO_TYPE_CASE(evaluate_scatter_element_update,
                           i64,
                           arg0,
                           arg1,
                           arg2,
                           arg3,
                           out,
                           normalized_axis,
                           reduction_type,
                           use_init_value);
        OPENVINO_TYPE_CASE(evaluate_scatter_element_update,
                           u32,
                           arg0,
                           arg1,
                           arg2,
                           arg3,
                           out,
                           normalized_axis,
                           reduction_type,
                           use_init_value);
        OPENVINO_TYPE_CASE(evaluate_scatter_element_update,
                           u64,
                           arg0,
                           arg1,
                           arg2,
                           arg3,
                           out,
                           normalized_axis,
                           reduction_type,
                           use_init_value);
        OPENVINO_TYPE_CASE(evaluate_scatter_element_update,
                           f32,
                           arg0,
                           arg1,
                           arg2,
                           arg3,
                           out,
                           normalized_axis,
                           reduction_type,
                           use_init_value);
        OPENVINO_TYPE_CASE(evaluate_scatter_element_update,
                           boolean,
                           arg0,
                           arg1,
                           arg2,
                           arg3,
                           out,
                           normalized_axis,
                           reduction_type,
                           use_init_value);
    default:
        rc = false;
        break;
    }
    return rc;
}
}  // namespace
}  // namespace scatter_elements_update

bool op::v3::ScatterElementsUpdate::evaluate_scatter_elements_update(const HostTensorVector& outputs,
                                                                     const HostTensorVector& inputs) const {
    const auto normalized_axis = get_normalized_axis(inputs);

    return scatter_elements_update::evaluate_scatter_elements_update(inputs[0],
                                                                     inputs[1],
                                                                     inputs[2],
                                                                     inputs[3],
                                                                     outputs[0],
                                                                     normalized_axis);
}

bool op::v12::ScatterElementsUpdate::evaluate_scatter_elements_update(const HostTensorVector& outputs,
                                                                      const HostTensorVector& inputs) const {
    const auto normalized_axis = get_normalized_axis(inputs);

    return scatter_elements_update::evaluate_scatter_elements_update(inputs[0],
                                                                     inputs[1],
                                                                     inputs[2],
                                                                     inputs[3],
                                                                     outputs[0],
                                                                     normalized_axis,
                                                                     m_reduction,
                                                                     m_use_init_val);
}

bool op::v3::ScatterElementsUpdate::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v3_ScatterElementsUpdate_evaluate);
    return evaluate_scatter_elements_update(outputs, inputs);
}

bool op::v12::ScatterElementsUpdate::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v12_ScatterElementsUpdate_evaluate);
    return evaluate_scatter_elements_update(outputs, inputs);
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

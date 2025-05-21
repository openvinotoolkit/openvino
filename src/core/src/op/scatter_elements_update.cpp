// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_elements_update.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/scatter_elements_update.hpp"
#include "scatter_elements_update_shape_inference.hpp"

namespace ov {
op::v3::ScatterElementsUpdate::ScatterElementsUpdate(const Output<Node>& data,
                                                     const Output<Node>& indices,
                                                     const Output<Node>& updates,
                                                     const Output<Node>& axis)
    : ov::op::util::ScatterElementsUpdateBase(data, indices, updates, axis) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v3::ScatterElementsUpdate::clone_with_new_inputs(const OutputVector& inputs) const {
    OV_OP_SCOPE(v3_ScatterElementsUpdate_clone_with_new_inputs);
    NODE_VALIDATION_CHECK(this,
                          inputs.size() == get_input_size(),
                          "clone_with_new_inputs() required inputs size: ",
                          get_input_size(),
                          "Got: ",
                          inputs.size());

    return std::make_shared<v3::ScatterElementsUpdate>(inputs.at(0), inputs.at(1), inputs.at(2), inputs.at(3));
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

std::shared_ptr<Node> op::v12::ScatterElementsUpdate::clone_with_new_inputs(const OutputVector& inputs) const {
    OV_OP_SCOPE(v12_ScatterElementsUpdate_clone_with_new_inputs);
    NODE_VALIDATION_CHECK(this,
                          inputs.size() == get_input_size(),
                          "clone_with_new_inputs() required inputs size: ",
                          get_input_size(),
                          "Got: ",
                          inputs.size());

    return std::make_shared<v12::ScatterElementsUpdate>(inputs.at(0),
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

namespace scatter_elements_update {
struct Evaluate : public element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t DATA_ET, class DT = fundamental_type_for<DATA_ET>>
    static result_type visit(const Tensor& data,
                             const Tensor& indices,
                             const Tensor& updates,
                             Tensor& output,
                             const Shape& data_shape,
                             const Shape& indices_shape,
                             const int64_t axis,
                             const op::v12::ScatterElementsUpdate::Reduction reduction,
                             const bool use_init_value

    ) {
        using namespace ov::element;
        return IF_TYPE_OF(scatter_el_update_idx_type,
                          OV_PP_ET_LIST(i8, i16, i32, i64, u8, u16, u32, u64),
                          EvaluateByIndicesType,
                          indices.get_element_type(),
                          data.data<const DT>(),
                          indices,
                          updates.data<const DT>(),
                          output.data<DT>(),
                          data_shape,
                          indices_shape,
                          axis,
                          reduction,
                          use_init_value);
    }

private:
    struct EvaluateByIndicesType : public element::NoAction<bool> {
        using element::NoAction<bool>::visit;

        template <element::Type_t INDEX_ET, class DT, class IT = fundamental_type_for<INDEX_ET>>
        static result_type visit(const DT* const data,
                                 const Tensor& indices,
                                 const DT* const updates,
                                 DT* const output,
                                 const Shape& data_shape,
                                 const Shape& indices_shape,
                                 const int64_t axis,
                                 const op::v12::ScatterElementsUpdate::Reduction reduction,
                                 const bool use_init_value) {
            reference::scatter_elem_update(data,
                                           indices.data<IT>(),
                                           updates,
                                           axis,
                                           output,
                                           data_shape,
                                           indices_shape,
                                           reduction,
                                           use_init_value);
            return true;
        }
    };
};
namespace {
bool evaluate(const op::util::ScatterElementsUpdateBase* node,
              TensorVector& outputs,
              const TensorVector& inputs,
              const int64_t axis,
              const op::v12::ScatterElementsUpdate::Reduction reduction,
              const bool use_init_value) {
    OPENVINO_ASSERT(inputs.size() == 4);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];
    auto& output = outputs[0];
    const auto& data_shape = data.get_shape();
    const auto& indices_shape = indices.get_shape();
    output.set_shape(data_shape);

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(scatter_evaluate,
                                      node,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(boolean, f32, i16, i32, i64, u32, u64),
                                      scatter_elements_update::Evaluate,
                                      data.get_element_type(),
                                      data,
                                      indices,
                                      updates,
                                      output,
                                      data_shape,
                                      indices_shape,
                                      axis,
                                      reduction,
                                      use_init_value);
}
}  // namespace
}  // namespace scatter_elements_update

bool op::v3::ScatterElementsUpdate::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v3_ScatterElementsUpdate_evaluate);
    constexpr auto reduction = op::v12::ScatterElementsUpdate::Reduction::NONE;
    constexpr auto use_init_value = false;
    return scatter_elements_update::evaluate(this,
                                             outputs,
                                             inputs,
                                             get_normalized_axis(inputs),
                                             reduction,
                                             use_init_value);
}

bool op::v12::ScatterElementsUpdate::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v12_ScatterElementsUpdate_evaluate);
    return scatter_elements_update::evaluate(this,
                                             outputs,
                                             inputs,
                                             get_normalized_axis(inputs),
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

AttributeAdapter<op::v12::ScatterElementsUpdate::Reduction>::~AttributeAdapter() = default;

namespace op {
std::ostream& operator<<(std::ostream& s, const v12::ScatterElementsUpdate::Reduction& reduction) {
    return s << as_string(reduction);
}
}  // namespace op
}  // namespace ov

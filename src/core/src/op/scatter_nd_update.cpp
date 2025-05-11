// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_nd_update.hpp"

#include "bound_evaluate.hpp"
#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/reference/scatter_nd_update.hpp"

namespace ov {
namespace op {
namespace scatter_nd_update {
struct Evaluate : public element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t DATA_ET, class DT = fundamental_type_for<DATA_ET>>
    static result_type visit(const Tensor& data,
                             const Tensor& indices,
                             const Tensor& updates,
                             Tensor& output,
                             const Shape& data_shape,
                             const Shape& indices_shape,
                             const Shape& updates_shape,
                             const v15::ScatterNDUpdate::Reduction reduction) {
        using namespace ov::element;
        return IF_TYPE_OF(sctter_nd_eval_idx_type,
                          OV_PP_ET_LIST(i32, i64),
                          EvaluateByIndicesType,
                          indices.get_element_type(),
                          data.data<const DT>(),
                          indices,
                          updates.data<const DT>(),
                          output.data<DT>(),
                          data_shape,
                          indices_shape,
                          updates_shape,
                          reduction);
    }

private:
    struct EvaluateByIndicesType : public element::NoAction<bool> {
        using element::NoAction<bool>::visit;

        template <element::Type_t INDICES_ET, class DT, class IT = fundamental_type_for<INDICES_ET>>
        static result_type visit(const DT* const data,
                                 const Tensor& indices,
                                 const DT* const updates,
                                 DT* const output,
                                 const Shape& data_shape,
                                 const Shape& indices_shape,
                                 const Shape& updates_shape,
                                 const v15::ScatterNDUpdate::Reduction reduction) {
            reference::scatterNdUpdate(data,
                                       indices.data<IT>(),
                                       updates,
                                       output,
                                       data_shape,
                                       indices_shape,
                                       updates_shape,
                                       reduction);
            return true;
        }
    };
};
namespace {
bool evaluate(const op::util::ScatterNDBase* node,
              TensorVector& outputs,
              const TensorVector& inputs,
              const op::v15::ScatterNDUpdate::Reduction reduction) {
    OPENVINO_ASSERT(inputs.size() == 3);
    OPENVINO_ASSERT(outputs.size() == 1);

    const auto& data = inputs[0];
    const auto& indices = inputs[1];
    const auto& updates = inputs[2];
    auto& output = outputs[0];
    const auto& data_shape = data.get_shape();
    const auto& indices_shape = indices.get_shape();
    const auto& updates_shape = updates.get_shape();
    output.set_shape(data_shape);
    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(scatter_evaluate,
                                      node,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(boolean, f32, i32, i64, u32, u64),
                                      scatter_nd_update::Evaluate,
                                      data.get_element_type(),
                                      data,
                                      indices,
                                      updates,
                                      output,
                                      data_shape,
                                      indices_shape,
                                      updates_shape,
                                      reduction);
}
bool has_evaluate(const op::util::ScatterNDBase* node) {
    switch (node->get_output_element_type(0)) {
    case element::boolean:
    case element::f16:
    case element::f32:
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
        break;
    default:
        return false;
    }
    switch (node->get_input_element_type(1)) {
    case element::i32:
    case element::i64:
        return true;
    default:
        return false;
    }
}
}  // namespace
}  // namespace scatter_nd_update
namespace v3 {
std::shared_ptr<Node> ScatterNDUpdate::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ScatterNDUpdate>(new_args.at(util::ScatterNDBase::INPUTS),
                                             new_args.at(util::ScatterNDBase::INDICES),
                                             new_args.at(util::ScatterNDBase::UPDATES));
}

bool ScatterNDUpdate::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate);
    constexpr auto reduction = op::v15::ScatterNDUpdate::Reduction::NONE;
    return scatter_nd_update::evaluate(this, outputs, inputs, reduction);
}

bool ScatterNDUpdate::has_evaluate() const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_has_evaluate);
    return scatter_nd_update::has_evaluate(this);
}

bool ScatterNDUpdate::evaluate_lower(TensorVector& output_values) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate_lower);
    return get_input_tensor(1).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool ScatterNDUpdate::evaluate_upper(TensorVector& output_values) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate_upper);
    return get_input_tensor(1).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool ScatterNDUpdate::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate_symbol);
    return default_symbol_evaluator(this, {0, 2}, output_symbols);
}
}  // namespace v3

namespace v15 {
ScatterNDUpdate::ScatterNDUpdate(const Output<Node>& inputs,
                                 const Output<Node>& indices,
                                 const Output<Node>& updates,
                                 const ScatterNDUpdate::Reduction reduction)
    : op::util::ScatterNDBase(inputs, indices, updates),
      m_reduction{reduction} {
    constructor_validate_and_infer_types();
}
std::shared_ptr<Node> ScatterNDUpdate::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v15_ScatterNDUpdate_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ScatterNDUpdate>(new_args.at(0), new_args.at(1), new_args.at(2), m_reduction);
}

bool ScatterNDUpdate::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v15_ScatterNDUpdate_visit_attributes);
    visitor.on_attribute("reduction", m_reduction);
    return true;
}

bool ScatterNDUpdate::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v15_ScatterNDUpdate_evaluate);
    return scatter_nd_update::evaluate(this, outputs, inputs, m_reduction);
}

bool ScatterNDUpdate::has_evaluate() const {
    OV_OP_SCOPE(v15_ScatterNDUpdate_has_evaluate);
    return scatter_nd_update::has_evaluate(this);
}

ScatterNDUpdate::Reduction ScatterNDUpdate::get_reduction() const {
    return m_reduction;
}

void ScatterNDUpdate::set_reduction(const ScatterNDUpdate::Reduction reduction) {
    m_reduction = reduction;
}
bool ScatterNDUpdate::evaluate_lower(TensorVector& output_values) const {
    OV_OP_SCOPE(v15_ScatterNDUpdate_evaluate_lower);
    return get_input_tensor(1).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool ScatterNDUpdate::evaluate_upper(TensorVector& output_values) const {
    OV_OP_SCOPE(v15_ScatterNDUpdate_evaluate_upper);
    return get_input_tensor(1).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool ScatterNDUpdate::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    OV_OP_SCOPE(v15_ScatterNDUpdate_evaluate_symbol);
    return default_symbol_evaluator(this, {0, 2}, output_symbols);
}

}  // namespace v15
}  // namespace op
std::ostream& operator<<(std::ostream& s, const op::v15::ScatterNDUpdate::Reduction& reduction) {
    return s << as_string(reduction);
}
template <>
OPENVINO_API EnumNames<op::v15::ScatterNDUpdate::Reduction>& EnumNames<op::v15::ScatterNDUpdate::Reduction>::get() {
    static auto enum_names =
        EnumNames<op::v15::ScatterNDUpdate::Reduction>("op::v15::ScatterNDUpdate::Reduction",
                                                       {{"none", op::v15::ScatterNDUpdate::Reduction::NONE},
                                                        {"sum", op::v15::ScatterNDUpdate::Reduction::SUM},
                                                        {"sub", op::v15::ScatterNDUpdate::Reduction::SUB},
                                                        {"prod", op::v15::ScatterNDUpdate::Reduction::PROD},
                                                        {"min", op::v15::ScatterNDUpdate::Reduction::MIN},
                                                        {"max", op::v15::ScatterNDUpdate::Reduction::MAX}});
    return enum_names;
}

AttributeAdapter<op::v15::ScatterNDUpdate::Reduction>::~AttributeAdapter() = default;
}  // namespace ov

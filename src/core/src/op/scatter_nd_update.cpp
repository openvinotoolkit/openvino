// Copyright (C) 2018-2023 Intel Corporation
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
                             const Shape& updates_shape) {
        using namespace ov::element;
        return IfTypeOf<i32, i64>::apply<EvaluateByIndicesType>(indices.get_element_type(),
                                                                data.data<const DT>(),
                                                                indices,
                                                                updates.data<const DT>(),
                                                                output.data<DT>(),
                                                                data_shape,
                                                                indices_shape,
                                                                updates_shape);
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
                                 const Shape& updates_shape) {
            reference::scatterNdUpdate(data,
                                       indices.data<IT>(),
                                       updates,
                                       output,
                                       data_shape,
                                       indices_shape,
                                       updates_shape);
            return true;
        }
    };
};
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
    return IfTypeOf<boolean, f32, i32, i64, u32, u64>::apply<scatter_nd_update::Evaluate>(data.get_element_type(),
                                                                                          data,
                                                                                          indices,
                                                                                          updates,
                                                                                          output,
                                                                                          data_shape,
                                                                                          indices_shape,
                                                                                          updates_shape);
}

bool ScatterNDUpdate::has_evaluate() const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_has_evaluate);

    switch (get_output_element_type(0)) {
    case element::boolean:
    case element::f32:
    case element::i32:
    case element::i64:
    case element::u32:
    case element::u64:
        break;
    default:
        return false;
    }
    switch (get_input_element_type(1)) {
    case element::i32:
    case element::i64:
        return true;
    default:
        return false;
    }
}

bool ScatterNDUpdate::evaluate_lower(TensorVector& output_values) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate_lower);
    return get_input_tensor(1).has_and_set_bound() && default_lower_bound_evaluator(this, output_values);
}

bool ScatterNDUpdate::evaluate_upper(TensorVector& output_values) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate_upper);
    return get_input_tensor(1).has_and_set_bound() && default_upper_bound_evaluator(this, output_values);
}

bool ScatterNDUpdate::evaluate_label(TensorLabelVector& output_labels) const {
    OV_OP_SCOPE(v3_ScatterNDUpdate_evaluate_label);
    return default_label_evaluator(this, {0, 2}, output_labels);
}
}  // namespace v3
}  // namespace op
}  // namespace ov

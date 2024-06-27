// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/non_zero.hpp"

#include <numeric>

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/reference/non_zero.hpp"

namespace ov {
namespace op {
namespace non_zero {
struct Evaluate : public element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    template <element::Type_t ET, class T = fundamental_type_for<ET>>
    static result_type visit(const Tensor& in, const Shape& in_shape, const size_t in_rank, Tensor& out) {
        const auto in_data = in.data<const T>();
        const size_t non_zero_count = reference::non_zero_get_count(in_data, in_shape);
        const auto out_shape = Shape{in_rank == 0 && non_zero_count > 0 ? 1 : in_rank, non_zero_count};
        out.set_shape(out_shape);

        using namespace ov::element;
        return IF_TYPE_OF(non_zero_out_type,
                          OV_PP_ET_LIST(i32, i64),
                          EvalByOutType,
                          out.get_element_type(),
                          in_data,
                          out,
                          in_shape);
    }

private:
    struct EvalByOutType : public element::NoAction<bool> {
        using element::NoAction<bool>::visit;

        template <element::Type_t ET, class T, class I = fundamental_type_for<ET>>
        static result_type visit(const T* in, Tensor& out, const Shape& in_shape) {
            reference::non_zero(in, out.data<I>(), in_shape);
            return true;
        }
    };
};
}  // namespace non_zero

namespace v3 {
NonZero::NonZero(const Output<Node>& arg) : Op({arg}) {
    constructor_validate_and_infer_types();
}

NonZero::NonZero(const Output<Node>& arg, const std::string& output_type)
    : Op({arg}),
      m_output_type(EnumNames<element::Type_t>::as_enum(output_type)) {
    constructor_validate_and_infer_types();
}

NonZero::NonZero(const Output<Node>& arg, const element::Type& output_type) : Op({arg}), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

bool NonZero::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_NonZero_visit_attributes);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void NonZero::validate_and_infer_types() {
    OV_OP_SCOPE(v3_NonZero_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          m_output_type == element::i64 || m_output_type == element::i32,
                          "Output type must be i32 or i64");
    // For scalar non-zero value case, onnx test case expects output shape {1, 1}
    const auto& input_shape = get_input_partial_shape(0);
    if (input_shape.rank().compatible(0)) {
        set_output_type(0, m_output_type, PartialShape::dynamic(2));
    } else {
        auto output_shape = PartialShape{input_shape.rank(), {0, 1}};
        auto& dim = output_shape[1];
        for (auto&& d : input_shape)
            dim *= d;
        set_output_type(0, m_output_type, output_shape);
    }

    set_input_is_relevant_to_shape(0);

    if (const auto input_constant = ov::util::get_constant_from_source(input_value(0))) {
        // input_value is available to calculate output shape
        const auto inputs = TensorVector{input_constant->get_tensor_view()};
        auto outputs = TensorVector{{m_output_type, {}}};
        if (!evaluate(outputs, inputs))
            return;
        const auto& output = outputs[0];
        set_output_type(0, m_output_type, output.get_shape());

        get_output_tensor(0).set_lower_value(output);
        get_output_tensor(0).set_upper_value(output);
    }
}

std::shared_ptr<Node> NonZero::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_NonZero_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<v3::NonZero>(new_args.at(0), m_output_type);
}

bool NonZero::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v3_NonZero_evaluate);

    const auto& input = inputs[0];
    auto& output = outputs[0];
    using namespace ov::element;
    const auto& input_shape = input.get_shape();
    return IF_TYPE_OF_CONVERT_TENSORS(v3_NonZero_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(boolean, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64),
                                      non_zero::Evaluate,
                                      input.get_element_type(),
                                      input,
                                      input_shape,
                                      input_shape.size(),
                                      output);
}

bool NonZero::has_evaluate() const {
    OV_OP_SCOPE(v3_NonZero_has_evaluate);
    switch (get_input_element_type(0)) {
    case element::boolean:
    case element::bf16:
    case element::f16:
    case element::f32:
    case element::f64:
    case element::i8:
    case element::i16:
    case element::i32:
    case element::i64:
    case element::u8:
    case element::u16:
    case element::u32:
    case element::u64:
        return true;
    default:
        return false;
    }
}
}  // namespace v3
}  // namespace op
}  // namespace ov

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/round.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/reference/round.hpp"

using namespace std;

namespace ov {
namespace op {
namespace round {

class Evaluate : public ov::element::NoAction<bool> {
public:
    using ov::element::NoAction<bool>::visit;

    template <element::Type_t ET, class TMode>
    static result_type visit(const Tensor& arg0, Tensor& out, const size_t count, TMode&& mode) {
        using T = typename element_type_traits<ET>::value_type;
        reference::round(arg0.data<T>(), out.data<T>(), count, std::forward<TMode>(mode));
        return true;
    }
};
}  // namespace round

namespace v5 {
Round::Round(const Output<Node>& arg, RoundMode mode) : util::UnaryElementwiseArithmetic(arg), m_mode(mode) {
    constructor_validate_and_infer_types();
}

bool Round::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v5_Round_visit_attributes);
    visitor.on_attribute("mode", m_mode);
    return true;
}

void Round::validate_and_infer_types() {
    OV_OP_SCOPE(v5_Round_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() == 1, "Only accepts one argument. Got: ", get_input_size());
    set_output_size(1);
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> Round::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v5_Round_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<Round>(new_args.at(0), m_mode);
}

bool Round::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v5_Round_evaluate);
    auto& arg0 = inputs.front();
    auto& out = outputs.front();

    using namespace ov::element;
    return IF_TYPE_OF_CONVERT_TENSORS(v5_Round_evaluate,
                                      this,
                                      outputs,
                                      inputs,
                                      OV_PP_ET_LIST(boolean, i8, i16, i32, i64, u8, u16, u32, u64, f32),
                                      round::Evaluate,
                                      arg0.get_element_type(),
                                      arg0,
                                      out,
                                      shape_size(arg0.get_shape()),
                                      get_mode());
}

bool Round::has_evaluate() const {
    OV_OP_SCOPE(v5_Round_has_evaluate);
    const auto& et = get_input_element_type(0);

    return et.is_static() && (et != element::f64) && (et.is_real() || et.is_integral());
}
}  // namespace v5
}  // namespace op

std::ostream& operator<<(std::ostream& s, const op::v5::Round::RoundMode& type) {
    return s << as_string(type);
}

template <>
OPENVINO_API EnumNames<op::v5::Round::RoundMode>& EnumNames<op::v5::Round::RoundMode>::get() {
    static auto enum_names =
        EnumNames<op::v5::Round::RoundMode>("op::v5::Round::RoundMode",
                                            {{"half_to_even", op::v5::Round::RoundMode::HALF_TO_EVEN},
                                             {"half_away_from_zero", op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO}});
    return enum_names;
}

AttributeAdapter<op::v5::Round::RoundMode>::~AttributeAdapter() = default;
}  // namespace ov

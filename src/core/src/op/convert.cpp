// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert.hpp"

#include "element_visitor.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/select.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace op {
namespace convert {

#define CONVERT_ET_LIST                                                                                              \
    boolean, bf16, f16, f32, f64, i4, i8, i16, i32, i64, u1, u2, u3, u4, u6, u8, u16, u32, u64, nf4, f8e4m3, f8e5m2, \
        f4e2m1, f8e8m0

#define CONVERT_TO_ANY_NO_NF4                                                                                   \
    boolean, bf16, f16, f32, f64, i4, i8, i16, i32, i64, u1, u2, u3, u4, u6, u8, u16, u32, u64, f8e4m3, f8e5m2, \
        f4e2m1, f8e8m0

#define CONVERT_TO_ANY_NO_F4 \
    boolean, bf16, f16, f32, f64, i4, i8, i16, i32, i64, u1, u2, u3, u4, u6, u8, u16, u32, u64, f8e4m3, f8e5m2

struct Evaluate : public element::NoAction<bool> {
    using element::NoAction<bool>::visit;

    // convert from any (except F16, bf16, f32, NF4, f8e8m0) to any except NF4
    template <element::Type_t ET_IN,
              class TI = fundamental_type_for<ET_IN>,
              typename std::enable_if<ET_IN != element::f16 && ET_IN != element::bf16 && ET_IN != element::f32 &&
                                      ET_IN != element::nf4 && ET_IN != element::f4e2m1 &&
                                      ET_IN != element::f8e8m0>::type* = nullptr>
    static result_type visit(const Tensor& arg, Tensor& out, const size_t count) {
        using namespace ov::element;
        return IF_TYPE_OF(Convert_out,
                          CONVERT_TO_ANY_NO_F4,
                          EvalByOutputType,
                          out.get_element_type(),
                          iterator<ET_IN>(reinterpret_cast<const TI*>(arg.data())),
                          out,
                          count);
    }

    // convert from F16 to any
    template <element::Type_t ET_IN,
              class TI = fundamental_type_for<ET_IN>,
              typename std::enable_if<ET_IN == element::f16>::type* = nullptr>
    static result_type visit(const Tensor& arg, Tensor& out, const size_t count) {
        using namespace ov::element;
        return IF_TYPE_OF(Convert_out,
                          CONVERT_ET_LIST,
                          EvalByOutputType,
                          out.get_element_type(),
                          iterator<ET_IN>(reinterpret_cast<const TI*>(arg.data())),
                          out,
                          count);
    }

    // convert from bF16, f32 to any except NF4
    template <element::Type_t ET_IN,
              class TI = fundamental_type_for<ET_IN>,
              typename std::enable_if<ET_IN == element::bf16 || ET_IN == element::f32>::type* = nullptr>
    static result_type visit(const Tensor& arg, Tensor& out, const size_t count) {
        using namespace ov::element;
        return IF_TYPE_OF(Convert_out,
                          CONVERT_TO_ANY_NO_NF4,
                          EvalByOutputType,
                          out.get_element_type(),
                          iterator<ET_IN>(reinterpret_cast<const TI*>(arg.data())),
                          out,
                          count);
    }

    // convert form NF4
    template <element::Type_t ET_IN,
              class TI = fundamental_type_for<ET_IN>,
              typename std::enable_if<ET_IN == element::nf4>::type* = nullptr>
    static result_type visit(const Tensor& arg, Tensor& out, const size_t count) {
        using namespace ov::element;
        return IF_TYPE_OF(Convert_out,
                          OV_PP_ET_LIST(f16, bf16, f32, nf4),
                          EvalByOutputType,
                          out.get_element_type(),
                          iterator<ET_IN>(reinterpret_cast<const TI*>(arg.data())),
                          out,
                          count);
    }

    // convert from F4E2M1
    template <element::Type_t ET_IN,
              class TI = fundamental_type_for<ET_IN>,
              typename std::enable_if<ET_IN == element::f4e2m1>::type* = nullptr>
    static result_type visit(const Tensor& arg, Tensor& out, const size_t count) {
        using namespace ov::element;
        return IF_TYPE_OF(Convert_out,
                          OV_PP_ET_LIST(f16, bf16, f32, f4e2m1),
                          EvalByOutputType,
                          out.get_element_type(),
                          iterator<ET_IN>(reinterpret_cast<const TI*>(arg.data())),
                          out,
                          count);
    }

    // convert from F8E8M0
    template <element::Type_t ET_IN,
              class TI = fundamental_type_for<ET_IN>,
              typename std::enable_if<ET_IN == element::f8e8m0>::type* = nullptr>
    static result_type visit(const Tensor& arg, Tensor& out, const size_t count) {
        using namespace ov::element;
        return IF_TYPE_OF(Convert_out,
                          OV_PP_ET_LIST(f16, bf16, f32, f8e8m0),
                          EvalByOutputType,
                          out.get_element_type(),
                          iterator<ET_IN>(reinterpret_cast<const TI*>(arg.data())),
                          out,
                          count);
    }

private:
    struct EvalByOutputType : public element::NoAction<bool> {
        using element::NoAction<bool>::visit;

        template <element::Type_t ET_OUT, class InputIter, class TO = ov::fundamental_type_for<ET_OUT>>
        static result_type visit(InputIter arg, Tensor& out, const size_t count) {
            reference::convert(arg, element::iterator<ET_OUT>(out.data()), count);
            return true;
        }
    };
};

namespace {
bool evaluate_bound(const Node* const node, TensorVector& output_values, const Tensor& input_bound) {
    OPENVINO_ASSERT(node, output_values.size() == 1);

    if (input_bound) {
        const auto& in_bound_shape = input_bound.get_shape();
        if (is_vector(in_bound_shape) && (in_bound_shape[0] == 0)) {
            return true;
        }

        const auto is_integral_up_to_16_bits = [](const element::Type& et) -> bool {
            return et.is_integral() && et.bitwidth() <= 16;
        };
        const auto& input_et = input_bound.get_element_type();
        const auto& output_et = output_values[0].get_element_type();

        const auto status = node->evaluate(output_values, {input_bound});
        if (!status || is_integral_up_to_16_bits(input_et) || is_integral_up_to_16_bits(output_et)) {
            return status;
        }

        // constants for dynamic values translation
        const auto input_max = ov::util::make_tensor_of_max_value(input_et);
        const auto output_max = ov::util::make_tensor_of_max_value(output_et);
        if (!input_max || !output_max)
            return false;

        // dynamic values translation
        auto outputs = TensorVector{{element::boolean, in_bound_shape}};
        const auto& input_dynamic_mask = outputs[0];

        return v1::Equal().evaluate(outputs, {input_bound, input_max}) &&
               v1::Select().evaluate(output_values, {input_dynamic_mask, output_max, output_values[0]});
    } else {
        return false;
    }
}
}  // namespace
}  // namespace convert
namespace v0 {

Convert::Convert(const Output<Node>& arg, const element::Type& destination_type)
    : Op({arg}),
      m_destination_type(destination_type) {
    constructor_validate_and_infer_types();
}

void Convert::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Convert_validate_and_infer_types);

    set_output_type(0, m_destination_type, get_input_partial_shape(0));
}

bool Convert::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Convert_visit_attributes);
    visitor.on_attribute("destination_type", m_destination_type);
    return true;
}

std::shared_ptr<Node> Convert::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Convert_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Convert>(new_args.at(0), m_destination_type);
}

bool Convert::evaluate(TensorVector& outputs, const TensorVector& inputs) const {
    OV_OP_SCOPE(v0_Convert_evaluate);
    OPENVINO_ASSERT(outputs.size() == 1);
    OPENVINO_ASSERT(inputs.size() == 1);

    if (auto& out = outputs[0]) {
        const auto& in = inputs[0];
        const auto& in_shape = in.get_shape();
        const auto count = shape_size(in_shape);

        out.set_shape(in_shape);

        using namespace ov::element;
        return IF_TYPE_OF(v0_Convert_in_et, CONVERT_ET_LIST, convert::Evaluate, in.get_element_type(), in, out, count);
    } else {
        return false;
    }
}

bool Convert::has_evaluate() const {
    OV_OP_SCOPE(v0_Convert_has_evaluate);

    const auto is_to_nf4_supported = [](const element::Type& from, const element::Type& to) {
        return (from == element::nf4) && (to == element::f16 || to == element::f32 || to == element::nf4);
    };

    const auto can_convert_f16_bf16_f32 = [](const element::Type& et) {
        return et == element::f16 || et == element::bf16 || et == element::f32;
    };

    const auto can_convert_f4e2m1 = [&](const element::Type& et) {
        return can_convert_f16_bf16_f32(et) || et == element::f4e2m1;
    };

    const auto can_convert_f8e8m0 = [&](const element::Type& et) {
        return can_convert_f16_bf16_f32(et) || et == element::f8e8m0;
    };

    const auto is_valid_type = [](const element::Type& et) -> bool {
        switch (et) {
        case element::boolean:
        case element::bf16:
        case element::f16:
        case element::f32:
        case element::f64:
        case element::i4:
        case element::i8:
        case element::i16:
        case element::i32:
        case element::i64:
        case element::u1:
        case element::u4:
        case element::u8:
        case element::u16:
        case element::u32:
        case element::u64:
        case element::f8e4m3:
        case element::f8e5m2:
            return true;
        default:
            return false;
        };
    };

    const auto& input_et = get_input_element_type(0);
    const auto& output_et = get_output_element_type(0);

    return (is_valid_type(input_et) && is_valid_type(output_et)) || is_to_nf4_supported(input_et, output_et) ||
           (can_convert_f4e2m1(input_et) && can_convert_f4e2m1(output_et)) ||
           (can_convert_f8e8m0(input_et) && can_convert_f8e8m0(output_et));
}

bool Convert::evaluate_lower(TensorVector& output_values) const {
    return convert::evaluate_bound(this, output_values, get_input_tensor(0).get_lower_value());
}

bool Convert::evaluate_upper(TensorVector& output_values) const {
    return convert::evaluate_bound(this, output_values, get_input_tensor(0).get_upper_value());
}

bool Convert::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    const auto input_symbols = get_input_tensor(0).get_value_symbol();
    if (input_symbols.empty()) {
        return false;
    } else {
        output_symbols[0] = input_symbols;
        return true;
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov

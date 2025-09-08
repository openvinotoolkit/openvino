// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert_promote_types.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace {

element::Type bitwidth_to_int(size_t bitwidth) {
    switch (bitwidth) {
    case 4:
        return element::i4;
    case 8:
        return element::i8;
    case 16:
        return element::i16;
    case 32:
        return element::i32;
    default:
        return element::i64;
    };
}
bool is_float8(const element::Type& type) {
    // Check for f8 special cases to handle
    switch (type) {
    case element::f8e4m3:
    case element::f8e5m2:
        return true;
    default:
        return false;
    }
}
bool is_float16(const element::Type& type) {
    // Check for f16 special cases to handle
    switch (type) {
    case element::bf16:
    case element::f16:
        return true;
    default:
        return false;
    }
}
bool is_type_supported(const element::Type& type) {
    // Types supported by ConvertPromoteTypes. When adding new type, ensure that existing rules will support it.
    switch (type) {
    case element::dynamic:
    case element::boolean:
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
    case element::bf16:
        return true;
    default:
        return false;
    }
}
element::Type evaluate_common_type(const v14::ConvertPromoteTypes* op) {
    const auto promote_unsafe = op->get_promote_unsafe();
    const auto pytorch_scalar_promotion = op->get_pytorch_scalar_promotion();
    const auto u64_promotion_target = op->get_u64_integer_promotion_target();
    const auto& input_0_type = op->get_input_element_type(0);
    const auto& input_1_type = op->get_input_element_type(1);
    NODE_VALIDATION_CHECK(op, is_type_supported(input_0_type) && is_type_supported(input_1_type));

    // Fast evaluate for trivial cases.
    if (input_0_type.is_dynamic() || input_1_type.is_dynamic()) {
        return element::dynamic;
    } else if (input_0_type == input_1_type)
        return input_0_type;
    else if (input_0_type == element::boolean)
        return input_1_type;
    else if (input_1_type == element::boolean)
        return input_0_type;

    const auto is_input_0_real = input_0_type.is_real();
    const auto is_input_1_real = input_1_type.is_real();

    if (is_input_0_real != is_input_1_real) {
        // Floating and integer mixed, align to floating
        if (promote_unsafe) {
            return is_input_0_real ? input_0_type : input_1_type;
        }
        // Ensure that floating-point bitwidth is at least double than integer for safe conversion.
        const auto real = is_input_0_real ? input_0_type : input_1_type;
        const auto integer = is_input_0_real ? input_1_type : input_0_type;
        NODE_VALIDATION_CHECK(op,
                              (real.bitwidth() >= integer.bitwidth() * 2),
                              "Integer input cannot be safely promoted to floating-point.");
        return real;

    } else if (is_input_0_real == is_input_1_real) {
        // Type formats are the same (both are either floating or integer).
        if (pytorch_scalar_promotion) {
            const auto& input_0_rank = op->get_input_partial_shape(0).rank();
            const auto& input_1_rank = op->get_input_partial_shape(1).rank();
            if (input_0_rank.is_dynamic() || input_1_rank.is_dynamic()) {
                // For pytorch mode, return element::dynamic if ranks affecting output type are dynamic.
                return element::dynamic;
            }
            const auto is_input_0_scalar = input_0_rank.get_length() == 0;
            const auto is_input_1_scalar = input_1_rank.get_length() == 0;
            if (is_input_0_scalar != is_input_1_scalar) {
                // For pytorch mode, when number formats are same, promote to type of non-scalar input.
                const auto& target = is_input_0_scalar ? input_1_type : input_0_type;
                if (!promote_unsafe) {
                    // For safe mode, check wether target type has bitwidth able to hold data from scalar type.
                    const auto& scalar = is_input_0_scalar ? input_0_type : input_1_type;
                    const auto is_pytorch_promote_safe =
                        ((target.is_signed() == scalar.is_signed() && target.bitwidth() >= scalar.bitwidth()) ||
                         (target.is_signed() && !scalar.is_signed() && target.bitwidth() * 2 >= scalar.bitwidth()));
                    NODE_VALIDATION_CHECK(op,
                                          is_pytorch_promote_safe,
                                          "Scalar input cannot be PyTorch-like promoted using safe promotion rules.");
                }
                return target;
            }
        }
        const auto is_input_0_signed = input_0_type.is_signed();
        const auto is_input_1_signed = input_1_type.is_signed();
        const auto input_0_bitwidth = input_0_type.bitwidth();
        const auto input_1_bitwidth = input_1_type.bitwidth();
        if ((is_input_0_signed != is_input_1_signed)) {
            // Signed and unsigned integers are mixed, convert to signed integer with bitwidth able to hold all unsigned
            // data. Exception for u64 + integer - either convert to type from `u64_promotion_target` or fail in safe
            // mode.
            const auto uint_bitwidth = is_input_0_signed ? input_1_bitwidth : input_0_bitwidth;
            const auto int_bitwidth = is_input_0_signed ? input_0_bitwidth : input_1_bitwidth;
            if (uint_bitwidth <= 32 && (promote_unsafe || uint_bitwidth * 2 <= int_bitwidth)) {
                return bitwidth_to_int(std::max({uint_bitwidth * 2, int_bitwidth}));
            }
            NODE_VALIDATION_CHECK(
                op,
                promote_unsafe,
                "Unsigned integer input cannot be safely promoted into any supported signed integer.");
            return u64_promotion_target;
        } else if ((input_0_bitwidth != input_1_bitwidth)) {
            // Both types have same format and sign but mixed bitwidth, promote to one with greater bitwidth.
            return input_0_bitwidth >= input_1_bitwidth ? input_0_type : input_1_type;
        } else if (promote_unsafe && (input_0_bitwidth == input_1_bitwidth)) {
            // Both types have same format, sign and bitwidth. Those are treated like special cases and rules need to be
            // provided manually.
            if (is_float8(input_0_type) && is_float8(input_1_type)) {
                // f8e4m3 and f8e5m2
                return element::f16;
            }
            if (is_float16(input_0_type) && is_float16(input_1_type)) {
                // bf16 and f16
                return element::f32;
            }
        }
    }
    NODE_VALIDATION_CHECK(op, false, "Unsupported input element types for ConvertPromoteTypes with given attributes.");
}

}  // namespace
namespace v14 {

ConvertPromoteTypes::ConvertPromoteTypes(const Output<Node>& input_0,
                                         const Output<Node>& input_1,
                                         const bool promote_unsafe,
                                         const bool pytorch_scalar_promotion,
                                         const element::Type& u64_integer_promotion_target)
    : Op({input_0, input_1}),
      m_promote_unsafe(promote_unsafe),
      m_pytorch_scalar_promotion(pytorch_scalar_promotion),
      m_u64_integer_promotion_target(u64_integer_promotion_target) {
    constructor_validate_and_infer_types();
}

void ConvertPromoteTypes::validate_and_infer_types() {
    OV_OP_SCOPE(v14_ConvertPromoteTypes_validate_and_infer_types);
    const auto common_type = evaluate_common_type(this);
    set_output_type(0, common_type, get_input_partial_shape(0));
    set_output_type(1, common_type, get_input_partial_shape(1));
}

bool ConvertPromoteTypes::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v14_ConvertPromoteTypes_visit_attributes);
    visitor.on_attribute("promote_unsafe", m_promote_unsafe);
    visitor.on_attribute("pytorch_scalar_promotion", m_pytorch_scalar_promotion);
    visitor.on_attribute("u64_integer_promotion_target", m_u64_integer_promotion_target);
    return true;
}

std::shared_ptr<Node> ConvertPromoteTypes::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v14_ConvertPromoteTypes_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ConvertPromoteTypes>(new_args.at(0),
                                                 new_args.at(1),
                                                 m_promote_unsafe,
                                                 m_pytorch_scalar_promotion,
                                                 m_u64_integer_promotion_target);
}

bool ConvertPromoteTypes::get_pytorch_scalar_promotion() const {
    return m_pytorch_scalar_promotion;
}

void ConvertPromoteTypes::set_pytorch_scalar_promotion(bool pytorch_scalar_promotion) {
    m_pytorch_scalar_promotion = pytorch_scalar_promotion;
}

bool ConvertPromoteTypes::get_promote_unsafe() const {
    return m_promote_unsafe;
}

void ConvertPromoteTypes::set_promote_unsafe(bool promote_unsafe) {
    m_promote_unsafe = promote_unsafe;
}

const element::Type& ConvertPromoteTypes::get_u64_integer_promotion_target() const {
    return m_u64_integer_promotion_target;
}

void ConvertPromoteTypes::set_u64_integer_promotion_target(const element::Type& u64_integer_promotion_target) {
    m_u64_integer_promotion_target = u64_integer_promotion_target;
}
}  // namespace v14
}  // namespace op
}  // namespace ov

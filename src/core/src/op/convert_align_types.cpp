// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert_align_types.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace op {
namespace {

std::unordered_map<size_t, element::Type> bit_to_int{
    {4, element::i4},
    {8, element::i8},
    {16, element::i16},
    {32, element::i32},
    {64, element::i64},
};

element::Type infer_types(const v1::ConvertAlignTypes* op) {
    const auto lhs = op->input(0);
    const auto rhs = op->input(1);
    const auto promote_unsafe = op->get_promote_unsafe();
    const auto pytorch_scalar_align = op->get_pytorch_scalar_align();
    const auto u64_promotion_target = op->get_u64_integer_promotion_target();
    const auto supported_types = {element::dynamic,
                                  element::boolean,
                                  element::f16,
                                  element::f32,
                                  element::f64,
                                  element::i4,
                                  element::i8,
                                  element::i16,
                                  element::i32,
                                  element::i64,
                                  element::u1,
                                  element::u4,
                                  element::u8,
                                  element::u16,
                                  element::u32,
                                  element::u64,
                                  element::f8e4m3,
                                  element::f8e5m2,
                                  element::bf16};
    const auto& lhs_type = lhs.get_element_type();
    const auto& rhs_type = rhs.get_element_type();
    NODE_VALIDATION_CHECK(op,
                          std::find(supported_types.begin(), supported_types.end(), lhs_type) != supported_types.end());
    NODE_VALIDATION_CHECK(op,
                          std::find(supported_types.begin(), supported_types.end(), rhs_type) != supported_types.end());
    if (lhs_type.is_dynamic() || rhs_type.is_dynamic()) {
        return element::dynamic;
    }
    if (lhs_type == rhs_type)
        return lhs_type;
    if (lhs_type == element::boolean)
        return rhs_type;
    if (rhs_type == element::boolean)
        return lhs_type;
    const auto& lhs_rank = lhs.get_partial_shape().rank();
    const auto& rhs_rank = rhs.get_partial_shape().rank();
    const bool is_lhs_scalar = lhs_rank.is_static() ? lhs_rank.get_length() == 0 : false;
    const bool is_rhs_scalar = rhs_rank.is_static() ? rhs_rank.get_length() == 0 : false;
    const bool is_lhs_signed = lhs_type.is_signed();
    const bool is_rhs_signed = rhs_type.is_signed();
    const bool is_lhs_real = lhs_type.is_real();
    const bool is_rhs_real = rhs_type.is_real();
    const size_t lhs_bitwidth = lhs_type.bitwidth();
    const size_t rhs_bitwidth = rhs_type.bitwidth();

    if (pytorch_scalar_align && (is_lhs_scalar ^ is_rhs_scalar) && (is_lhs_real == is_rhs_real)) {
        if (promote_unsafe) {
            return is_lhs_scalar ? rhs_type : lhs_type;
        }
        const auto target = is_lhs_scalar ? rhs_type : lhs_type;
        const auto scalar = is_lhs_scalar ? lhs_type : rhs_type;
        if ((target.is_signed() == scalar.is_signed() && target.bitwidth() >= scalar.bitwidth()) ||
            (target.is_signed() && !scalar.is_signed() && target.bitwidth() * 2 >= scalar.bitwidth())) {
            return target;
        }
        NODE_VALIDATION_CHECK(op, false, " Scalar input cannot be PyTorch-like aligned using safe promotion rules.");
    }
    if (is_lhs_real ^ is_rhs_real) {
        if (promote_unsafe) {
            return is_lhs_real ? lhs_type : rhs_type;
        }
        const auto real = is_lhs_real ? lhs_type : rhs_type;
        const auto integer = is_lhs_real ? rhs_type : lhs_type;
        if (real.bitwidth() >= integer.bitwidth() * 2) {
            return real;
        }
        NODE_VALIDATION_CHECK(op, false, "Integer input cannot be safely promoted to floating-point.");
    }
    if ((is_lhs_real == is_rhs_real) && (is_lhs_signed ^ is_rhs_signed)) {
        const auto uint_bitwidth = is_lhs_signed ? rhs_bitwidth : lhs_bitwidth;
        const auto int_bitwidth = is_lhs_signed ? lhs_bitwidth : rhs_bitwidth;
        if (uint_bitwidth <= 32 && (promote_unsafe || uint_bitwidth * 2 <= int_bitwidth)) {
            return bit_to_int.at(std::max({uint_bitwidth * 2, int_bitwidth}));
        } else if (promote_unsafe) {
            return u64_promotion_target;
        } else {
            NODE_VALIDATION_CHECK(
                op,
                false,
                "Unsigned integer input cannot be safely promoted into any supported signed integer.");
        }
    }
    if ((is_lhs_real == is_rhs_real) && (lhs_bitwidth != rhs_bitwidth)) {
        return lhs_bitwidth >= rhs_bitwidth ? lhs_type : rhs_type;
    }
    if (promote_unsafe && (is_lhs_real == is_rhs_real) && (lhs_bitwidth == rhs_bitwidth)) {
        const auto lhs_string = lhs_type.to_string();
        const auto rhs_string = rhs_type.to_string();
        // f8e4m3 and f8e5m2
        const std::set<std::string> float8_types{"f8e4m3", "f8e5m2"};
        if (float8_types.count(lhs_string) && float8_types.count(rhs_string)) {
            return element::f16;
        }
        // bf16 and f16
        const std::set<std::string> float16_types{"f16", "bf16"};
        if (float16_types.count(lhs_string) && float16_types.count(rhs_string)) {
            return element::f32;
        }
    }
    NODE_VALIDATION_CHECK(op, false, "Unsupported input element types for ConvertAlignTypes with given attributes.");
}
}  // namespace
namespace v1 {

ConvertAlignTypes::ConvertAlignTypes(const Output<Node>& lhs,
                                     const Output<Node>& rhs,
                                     const bool promote_unsafe,
                                     const bool pytorch_scalar_align,
                                     const element::Type& u64_integer_promotion_target)
    : Op({lhs, rhs}),
      m_promote_unsafe(promote_unsafe),
      m_pytorch_scalar_align(pytorch_scalar_align),
      m_u64_integer_promotion_target(u64_integer_promotion_target) {
    constructor_validate_and_infer_types();
}

void ConvertAlignTypes::validate_and_infer_types() {
    OV_OP_SCOPE(ConvertAlignTypes_validate_and_infer_types);
    const auto aligned_type = infer_types(this);
    set_output_type(0, aligned_type, get_input_partial_shape(0));
    set_output_type(1, aligned_type, get_input_partial_shape(1));
}

bool ConvertAlignTypes::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(ConvertAlignTypes_visit_attributes);
    visitor.on_attribute("promote_unsafe", m_promote_unsafe);
    visitor.on_attribute("pytorch_scalar_align", m_pytorch_scalar_align);
    visitor.on_attribute("u64_integer_promotion_target", m_u64_integer_promotion_target);
    return true;
}

std::shared_ptr<Node> ConvertAlignTypes::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(ConvertAlignTypes_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ConvertAlignTypes>(new_args.at(0),
                                               new_args.at(1),
                                               m_promote_unsafe,
                                               m_pytorch_scalar_align,
                                               m_u64_integer_promotion_target);
}

}  // namespace v1
}  // namespace op
}  // namespace ov

// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convert_align_types.hpp"

#include "itt.hpp"
#include "validation_util.hpp"

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

element::Type infer_types(const v14::ConvertAlignTypes* op) {
    const auto input_0 = op->input(0);
    const auto input_1 = op->input(1);
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
    const auto& input_0_type = input_0.get_element_type();
    const auto& input_1_type = input_1.get_element_type();
    NODE_VALIDATION_CHECK(
        op,
        std::find(supported_types.begin(), supported_types.end(), input_0_type) != supported_types.end());
    NODE_VALIDATION_CHECK(
        op,
        std::find(supported_types.begin(), supported_types.end(), input_1_type) != supported_types.end());
    if (input_0_type.is_dynamic() || input_1_type.is_dynamic()) {
        return element::dynamic;
    }
    if (input_0_type == input_1_type)
        return input_0_type;
    if (input_0_type == element::boolean)
        return input_1_type;
    if (input_1_type == element::boolean)
        return input_0_type;
    const auto& input_0_rank = input_0.get_partial_shape().rank();
    const auto& input_1_rank = input_1.get_partial_shape().rank();
    const bool is_input_0_scalar = input_0_rank.is_static() ? input_0_rank.get_length() == 0 : false;
    const bool is_input_1_scalar = input_1_rank.is_static() ? input_1_rank.get_length() == 0 : false;
    const bool is_input_0_signed = input_0_type.is_signed();
    const bool is_input_1_signed = input_1_type.is_signed();
    const bool is_input_0_real = input_0_type.is_real();
    const bool is_input_1_real = input_1_type.is_real();
    const size_t input_0_bitwidth = input_0_type.bitwidth();
    const size_t input_1_bitwidth = input_1_type.bitwidth();

    if (pytorch_scalar_align && (is_input_0_scalar != is_input_1_scalar) && (is_input_0_real == is_input_1_real)) {
        if (promote_unsafe) {
            return is_input_0_scalar ? input_1_type : input_0_type;
        }
        const auto target = is_input_0_scalar ? input_1_type : input_0_type;
        const auto scalar = is_input_0_scalar ? input_0_type : input_1_type;
        if ((target.is_signed() == scalar.is_signed() && target.bitwidth() >= scalar.bitwidth()) ||
            (target.is_signed() && !scalar.is_signed() && target.bitwidth() * 2 >= scalar.bitwidth())) {
            return target;
        }
        NODE_VALIDATION_CHECK(op, false, " Scalar input cannot be PyTorch-like aligned using safe promotion rules.");
    }
    if (is_input_0_real != is_input_1_real) {
        if (promote_unsafe) {
            return is_input_0_real ? input_0_type : input_1_type;
        }
        const auto real = is_input_0_real ? input_0_type : input_1_type;
        const auto integer = is_input_0_real ? input_1_type : input_0_type;
        if (real.bitwidth() >= integer.bitwidth() * 2) {
            return real;
        }
        NODE_VALIDATION_CHECK(op, false, "Integer input cannot be safely promoted to floating-point.");
    }
    if ((is_input_0_real == is_input_1_real) && (is_input_0_signed != is_input_1_signed)) {
        const auto uint_bitwidth = is_input_0_signed ? input_1_bitwidth : input_0_bitwidth;
        const auto int_bitwidth = is_input_0_signed ? input_0_bitwidth : input_1_bitwidth;
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
    if ((is_input_0_real == is_input_1_real) && (input_0_bitwidth != input_1_bitwidth)) {
        return input_0_bitwidth >= input_1_bitwidth ? input_0_type : input_1_type;
    }
    if (promote_unsafe && (is_input_0_real == is_input_1_real) && (input_0_bitwidth == input_1_bitwidth)) {
        const auto input_0_string = input_0_type.to_string();
        const auto input_1_string = input_1_type.to_string();
        // f8e4m3 and f8e5m2
        const std::set<std::string> float8_types{"f8e4m3", "f8e5m2"};
        if (float8_types.count(input_0_string) && float8_types.count(input_1_string)) {
            return element::f16;
        }
        // bf16 and f16
        const std::set<std::string> float16_types{"f16", "bf16"};
        if (float16_types.count(input_0_string) && float16_types.count(input_1_string)) {
            return element::f32;
        }
    }
    NODE_VALIDATION_CHECK(op, false, "Unsupported input element types for ConvertAlignTypes with given attributes.");
}
}  // namespace
namespace v14 {

ConvertAlignTypes::ConvertAlignTypes(const Output<Node>& input_0,
                                     const Output<Node>& input_1,
                                     const bool promote_unsafe,
                                     const bool pytorch_scalar_align,
                                     const element::Type& u64_integer_promotion_target)
    : Op({input_0, input_1}),
      m_promote_unsafe(promote_unsafe),
      m_pytorch_scalar_align(pytorch_scalar_align),
      m_u64_integer_promotion_target(u64_integer_promotion_target) {
    constructor_validate_and_infer_types();
}

void ConvertAlignTypes::validate_and_infer_types() {
    OV_OP_SCOPE(v14_ConvertAlignTypes_validate_and_infer_types);
    const auto aligned_type = infer_types(this);
    set_output_type(0, aligned_type, get_input_partial_shape(0));
    set_output_type(1, aligned_type, get_input_partial_shape(1));
}

bool ConvertAlignTypes::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v14_ConvertAlignTypes_visit_attributes);
    visitor.on_attribute("promote_unsafe", m_promote_unsafe);
    visitor.on_attribute("pytorch_scalar_align", m_pytorch_scalar_align);
    visitor.on_attribute("u64_integer_promotion_target", m_u64_integer_promotion_target);
    return true;
}

std::shared_ptr<Node> ConvertAlignTypes::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v14_ConvertAlignTypes_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ConvertAlignTypes>(new_args.at(0),
                                               new_args.at(1),
                                               m_promote_unsafe,
                                               m_pytorch_scalar_align,
                                               m_u64_integer_promotion_target);
}

bool ConvertAlignTypes::get_pytorch_scalar_align() const {
    return m_pytorch_scalar_align;
}

void ConvertAlignTypes::set_pytorch_scalar_align(bool pytorch_scalar_align) {
    m_pytorch_scalar_align = pytorch_scalar_align;
}

bool ConvertAlignTypes::get_promote_unsafe() const {
    return m_promote_unsafe;
}

void ConvertAlignTypes::set_promote_unsafe(bool promote_unsafe) {
    m_promote_unsafe = promote_unsafe;
}

const element::Type& ConvertAlignTypes::get_u64_integer_promotion_target() const {
    return m_u64_integer_promotion_target;
}

void ConvertAlignTypes::set_u64_integer_promotion_target(const element::Type& u64_integer_promotion_target) {
    m_u64_integer_promotion_target = u64_integer_promotion_target;
}
}  // namespace v14
}  // namespace op
}  // namespace ov

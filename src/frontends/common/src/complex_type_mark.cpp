// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov::frontend;
using namespace ov::op;
using namespace std;

ComplexTypeMark::~ComplexTypeMark() = default;

ov::Output<ov::Node> ComplexTypeMark::get_complex_part_by_index(const NodeContext& context,
                                                                const ov::Output<ov::Node>& complex_data,
                                                                int32_t index,
                                                                bool squeezed) {
    // gather the required slice corresponding to Real or Imaginary part
    auto gather_index_shape = squeezed ? Shape{} : Shape{1};
    auto gather_index = context.mark_node(make_shared<v0::Constant>(element::i32, gather_index_shape, index));
    auto gather_axis = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{1}, -1));
    auto complex_part = context.mark_node(make_shared<v8::Gather>(complex_data, gather_index, gather_axis));

    return {complex_part};
}

ov::Output<ov::Node> ComplexTypeMark::create_complex_tensor(const NodeContext& context,
                                                            const ov::Output<ov::Node>& real_part,
                                                            const ov::Output<ov::Node>& imag_part,
                                                            bool needs_broadcast) {
    auto real = real_part;
    auto imag = imag_part;
    if (needs_broadcast) {
        // broadcast both real and imaginary parts to common shape
        auto real_shape = context.mark_node(make_shared<v3::ShapeOf>(real, element::i32));
        auto imag_shape = context.mark_node(make_shared<v3::ShapeOf>(imag, element::i32));
        auto target_shape = context.mark_node(make_shared<v1::Maximum>(real_shape, imag_shape));
        real = context.mark_node(make_shared<v3::Broadcast>(real, target_shape));
        imag = context.mark_node(make_shared<v3::Broadcast>(imag, target_shape));
    }

    // create auxiliary dimensions to concatenate both real and imaginary parts to common shape
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    real = context.mark_node(std::make_shared<v0::Unsqueeze>(real, const_neg_1));
    imag = context.mark_node(std::make_shared<v0::Unsqueeze>(imag, const_neg_1));

    auto complex = context.mark_node(std::make_shared<v0::Concat>(OutputVector{real, imag}, -1));

    return {complex};
}

ov::Output<ov::Node> ComplexTypeMark::add(const NodeContext& context,
                                          const ov::Output<ov::Node>& lhs,
                                          const ov::Output<ov::Node>& rhs,
                                          bool lhs_complex,
                                          bool rhs_complex) {
    if (lhs_complex && rhs_complex) {
        // both operands are of complex type
        auto result = context.mark_node(make_shared<v1::Add>(lhs, rhs))->output(0);
        return {result};
    } else if (lhs_complex) {
        // rhs is of a real type
        auto lhs_real = get_real_part(context, lhs);
        auto lhs_imag = get_imag_part(context, lhs);
        auto result_real = context.mark_node(make_shared<v1::Add>(lhs_real, rhs));
        // Note: complex operand can be broadcasted to real operand
        auto result =
            context.mark_node(create_complex_tensor(context, result_real, lhs_imag, true).get_node_shared_ptr());
        return {result};
    } else if (rhs_complex) {
        // lhs is of a real type
        auto rhs_real = get_real_part(context, rhs);
        auto rhs_imag = get_imag_part(context, rhs);
        auto result_real = context.mark_node(make_shared<v1::Add>(rhs_real, lhs));
        // Note: complex operand can be broadcasted to real operand
        auto result =
            context.mark_node(create_complex_tensor(context, result_real, rhs_imag, true).get_node_shared_ptr());
        return {result};
    }

    // both operands are real
    auto result = context.mark_node(make_shared<v1::Add>(lhs, rhs));

    return {result};
}

ov::Output<ov::Node> ComplexTypeMark::sub(const NodeContext& context,
                                          const ov::Output<ov::Node>& lhs,
                                          const ov::Output<ov::Node>& rhs,
                                          bool lhs_complex,
                                          bool rhs_complex) {
    if (lhs_complex && rhs_complex) {
        // both operands are of complex type
        auto result = context.mark_node(make_shared<v1::Subtract>(lhs, rhs))->output(0);
        return {result};
    } else if (lhs_complex) {
        // rhs is of a real type
        auto lhs_real = get_real_part(context, lhs);
        auto lhs_imag = get_imag_part(context, lhs);
        auto result_real = context.mark_node(make_shared<v1::Subtract>(lhs_real, rhs));
        // Note: complex operand can be broadcasted to real operand
        auto result =
            context.mark_node(create_complex_tensor(context, result_real, lhs_imag, true).get_node_shared_ptr());
        return {result};
    } else if (rhs_complex) {
        // lhs is of a real type
        auto rhs_real = get_real_part(context, rhs);
        auto rhs_imag = get_imag_part(context, rhs);
        auto result_real = context.mark_node(make_shared<v1::Subtract>(lhs, rhs_real));
        rhs_imag = context.mark_node(make_shared<v0::Negative>(rhs_imag));
        // Note: complex operand can be broadcasted to real operand
        auto result =
            context.mark_node(create_complex_tensor(context, result_real, rhs_imag, true).get_node_shared_ptr());
        return {result};
    }

    // both operands are real
    auto result = context.mark_node(make_shared<v1::Subtract>(lhs, rhs));

    return {result};
}

ov::Output<ov::Node> ComplexTypeMark::mul(const NodeContext& context,
                                          const ov::Output<ov::Node>& lhs,
                                          const ov::Output<ov::Node>& rhs,
                                          bool lhs_complex,
                                          bool rhs_complex) {
    if (lhs_complex && rhs_complex) {
        // both operands are of complex type
        // formula for guidance: (a + b*i) * (c + d*i) = (ac-bd) + (ad+bc)*i
        auto lr = get_real_part(context, lhs);
        auto li = get_imag_part(context, lhs);

        auto rr = get_real_part(context, rhs);
        auto ri = get_imag_part(context, rhs);

        auto mul_lr_rr = context.mark_node(make_shared<v1::Multiply>(lr, rr));
        auto mul_li_ri = context.mark_node(make_shared<v1::Multiply>(li, ri));
        auto res_real = context.mark_node(make_shared<v1::Subtract>(mul_lr_rr, mul_li_ri));

        auto mul_lr_ri = context.mark_node(make_shared<v1::Multiply>(lr, ri));
        auto mul_li_rr = context.mark_node(make_shared<v1::Multiply>(li, rr));
        auto res_imag = context.mark_node(make_shared<v1::Add>(mul_lr_ri, mul_li_rr));

        auto result = create_complex_tensor(context, res_real, res_imag);

        return {result};
    } else if (lhs_complex) {
        // rhs is of a real type
        auto unsqueeze_axis = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{1}, -1));
        auto unsqueezed_rhs = context.mark_node(make_shared<v0::Unsqueeze>(rhs, unsqueeze_axis));
        auto result = context.mark_node(make_shared<v1::Multiply>(lhs, unsqueezed_rhs));
        return {result};
    } else if (rhs_complex) {
        // rhs is of a real type
        auto unsqueeze_axis = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{1}, -1));
        auto unsqueezed_lhs = context.mark_node(make_shared<v0::Unsqueeze>(lhs, unsqueeze_axis));
        auto result = context.mark_node(make_shared<v1::Multiply>(unsqueezed_lhs, rhs));
        return {result};
    }

    // both operands are real
    auto result = context.mark_node(make_shared<v1::Multiply>(lhs, rhs));

    return {result};
}

ov::Output<ov::Node> ComplexTypeMark::inv(const NodeContext& context,
                                          const ov::Output<ov::Node>& data,
                                          bool data_complex) {
    if (data_complex) {
        // inverse of complex number:
        // 1 / (a + b*i) = (a/(a^2 + b^2)) + (-b/(a^2 + b^2))*i
        auto two = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{}, 2));
        two = context.mark_node(make_shared<v1::ConvertLike>(two, data));
        auto real = get_real_part(context, data);
        auto imag = get_imag_part(context, data);
        auto real_squared = context.mark_node(make_shared<v1::Power>(real, two));
        auto imag_squared = context.mark_node(make_shared<v1::Power>(imag, two));
        auto denom = context.mark_node(make_shared<v1::Add>(real_squared, imag_squared));

        auto minus_imag = context.mark_node(make_shared<v0::Negative>(imag));

        auto result = create_complex_tensor(context, real, minus_imag);
        result = context.mark_node(make_shared<v1::Divide>(result, denom));
        return {result};
    }

    auto one = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{}, 1));
    one = context.mark_node(make_shared<v1::ConvertLike>(one, data));
    auto result = context.mark_node(make_shared<v1::Divide>(one, data));
    return {result};
}

ov::Output<ov::Node> ComplexTypeMark::div(const NodeContext& context,
                                          const ov::Output<ov::Node>& lhs,
                                          const ov::Output<ov::Node>& rhs,
                                          bool lhs_complex,
                                          bool rhs_complex) {
    if (lhs_complex && rhs_complex) {
        // compute inverse of rhs
        auto inv_rhs = inv(context, rhs, true);
        // multiply lhs by inversed rhs
        auto result = mul(context, lhs, inv_rhs, true, true);
        return {result};
    } else if (lhs_complex) {
        // add auxiliary dimension for rhs
        // it is needed for correct division
        auto unsqueeze_axis = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{1}, -1));
        auto unsqueeze_rhs = context.mark_node(make_shared<v0::Unsqueeze>(rhs, unsqueeze_axis));
        auto result = context.mark_node(make_shared<v1::Divide>(lhs, unsqueeze_rhs));
    } else if (rhs_complex) {
        // a / (b + c*i) = (ab / (b^2 + c^2)) + (-ac / (b^2 + c^2)) * i
        auto rhs_real = get_real_part(context, rhs);
        auto rhs_imag = get_imag_part(context, rhs);
        auto two = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{}, 2));
        two = context.mark_node(make_shared<v1::ConvertLike>(two, rhs));
        auto real_squared = context.mark_node(make_shared<v1::Power>(rhs_real, two));
        auto imag_squared = context.mark_node(make_shared<v1::Power>(rhs_imag, two));
        auto denom = context.mark_node(make_shared<v1::Add>(real_squared, imag_squared));

        auto result_real = context.mark_node(make_shared<v1::Multiply>(lhs, rhs_real));
        auto result_imag = context.mark_node(make_shared<v1::Multiply>(lhs, rhs_imag));
        result_imag = context.mark_node(make_shared<v0::Negative>(result_imag));
        auto result = create_complex_tensor(context, result_real, result_imag);
        result = context.mark_node(make_shared<v1::Divide>(result, denom));
        return {result};
    }

    auto result = context.mark_node(make_shared<v1::Divide>(lhs, rhs));
    return {result};
}

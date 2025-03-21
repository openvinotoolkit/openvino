// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/complex_type_mark.hpp"

#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov::frontend;
using namespace ov::op;
using namespace std;

ComplexTypeMark::ComplexTypeMark(const ov::Output<ov::Node>& input, const ov::element::Type& complex_part_type)
    : ov::op::util::FrameworkNode(ov::OutputVector{input}, 1),
      m_complex_part_type(complex_part_type),
      m_data(input),
      m_real{},
      m_imag{} {
    validate_and_infer_types();
    if (input.get_element_type() != m_complex_part_type && m_complex_part_type.is_static()) {
        m_data = make_shared<v0::Convert>(m_data, m_complex_part_type);
    }

    if (m_complex_part_type.is_dynamic()) {
        m_complex_part_type = m_data.get_element_type();
    }
}

ComplexTypeMark::ComplexTypeMark(const ov::Output<ov::Node>& real,
                                 const ov::Output<ov::Node>& imag,
                                 const ov::element::Type& complex_part_type)
    : ov::op::util::FrameworkNode(ov::OutputVector{real, imag}, 1),
      m_complex_part_type(complex_part_type),
      m_data{},
      m_real{real},
      m_imag{imag} {
    validate_and_infer_types();
    if (m_real.get_element_type() != m_complex_part_type && m_complex_part_type.is_static()) {
        m_real = make_shared<v0::Convert>(m_real, m_complex_part_type);
    }
    if (m_imag.get_element_type() != m_complex_part_type && m_complex_part_type.is_static()) {
        m_imag = make_shared<v0::Convert>(m_imag, m_complex_part_type);
    }

    // need broadcast real and imaginary parts if they are of different shapes
    if (m_real.get_partial_shape().is_dynamic() || m_real.get_partial_shape() != m_imag.get_partial_shape()) {
        auto real_shape = make_shared<v3::ShapeOf>(m_real, element::i32);
        auto imag_shape = make_shared<v3::ShapeOf>(m_imag, element::i32);
        m_real = make_shared<v3::Broadcast>(m_real, imag_shape, BroadcastType::BIDIRECTIONAL);
        m_imag = make_shared<v3::Broadcast>(m_imag, real_shape, BroadcastType::BIDIRECTIONAL);
    }

    if (m_complex_part_type.is_dynamic()) {
        m_complex_part_type = m_real.get_element_type();
    }
}

ov::Output<ov::Node> ComplexTypeMark::get_real(bool squeezed) {
    if (m_real.get_node_shared_ptr() && squeezed) {
        return m_real;
    } else if (m_real.get_node_shared_ptr()) {
        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
        auto unsqueeze_real = make_shared<v0::Unsqueeze>(m_real, minus_one);
        return unsqueeze_real;
    }

    auto gather_index = make_shared<v0::Constant>(element::i32, Shape{}, 0);
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
    m_real = make_shared<v8::Gather>(m_data, gather_index, gather_axis);
    return m_real;
}

ov::Output<ov::Node> ComplexTypeMark::get_imag(bool squeezed) {
    if (m_imag.get_node_shared_ptr() && squeezed) {
        return m_imag;
    } else if (m_imag.get_node_shared_ptr()) {
        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
        auto unsqueeze_imag = make_shared<v0::Unsqueeze>(m_imag, minus_one);
        return unsqueeze_imag;
    }

    auto gather_index = make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto gather_axis = make_shared<v0::Constant>(element::i32, Shape{1}, -1);
    m_imag = make_shared<v8::Gather>(m_data, gather_index, gather_axis);
    return m_imag;
}

ov::Output<ov::Node> ComplexTypeMark::get_data() {
    if (m_data.get_node_shared_ptr()) {
        return m_data;
    }

    // create auxiliary dimensions to concatenate both real and imaginary parts to common shape
    auto const_neg_1 = v0::Constant::create(element::i32, Shape{}, {-1});
    auto unsqueeze_real = make_shared<v0::Unsqueeze>(m_real, const_neg_1);
    auto unsqueeze_imag = make_shared<v0::Unsqueeze>(m_imag, const_neg_1);

    m_data = make_shared<v0::Concat>(OutputVector{unsqueeze_real, unsqueeze_imag}, -1);
    return m_data;
}

ComplexTypeMark::~ComplexTypeMark() = default;

ov::Output<ov::Node> ComplexTypeMark::add(const NodeContext& context,
                                          const ov::Output<ov::Node>& lhs,
                                          const ov::Output<ov::Node>& rhs) {
    auto lhs_complex = as_type_ptr<ComplexTypeMark>(lhs.get_node_shared_ptr());
    auto rhs_complex = as_type_ptr<ComplexTypeMark>(rhs.get_node_shared_ptr());

    if (lhs_complex && rhs_complex) {
        auto lhs_data = lhs_complex->get_data();
        auto rhs_data = rhs_complex->get_data();

        // both operands are of complex type
        auto result = context.mark_node(make_shared<v1::Add>(lhs_data, rhs_data))->output(0);
        auto complex_result = context.mark_node(make_shared<ComplexTypeMark>(result, result.get_element_type()));
        return {complex_result};
    } else if (lhs_complex) {
        // rhs is of a real type
        auto lhs_real = lhs_complex->get_real();
        auto lhs_imag = lhs_complex->get_imag();
        auto result_real = context.mark_node(make_shared<v1::Add>(lhs_real, rhs))->output(0);
        auto complex_result =
            context.mark_node(make_shared<ComplexTypeMark>(result_real, lhs_imag, result_real.get_element_type()));
        return {complex_result};
    } else if (rhs_complex) {
        // lhs is of a real type
        auto rhs_real = rhs_complex->get_real();
        auto rhs_imag = rhs_complex->get_imag();
        auto result_real = context.mark_node(make_shared<v1::Add>(rhs_real, lhs))->output(0);
        auto complex_result =
            context.mark_node(make_shared<ComplexTypeMark>(result_real, rhs_imag, result_real.get_element_type()));
        return {complex_result};
    }

    // both operands are real
    auto result = context.mark_node(make_shared<v1::Add>(lhs, rhs));

    return {result};
}

ov::Output<ov::Node> ComplexTypeMark::sub(const NodeContext& context,
                                          const ov::Output<ov::Node>& lhs,
                                          const ov::Output<ov::Node>& rhs) {
    auto lhs_complex = as_type_ptr<ComplexTypeMark>(lhs.get_node_shared_ptr());
    auto rhs_complex = as_type_ptr<ComplexTypeMark>(rhs.get_node_shared_ptr());

    if (lhs_complex && rhs_complex) {
        auto lhs_data = lhs_complex->get_data();
        auto rhs_data = rhs_complex->get_data();

        // both operands are of complex type
        auto result = context.mark_node(make_shared<v1::Subtract>(lhs_data, rhs_data))->output(0);
        auto complex_result = context.mark_node(make_shared<ComplexTypeMark>(result, result.get_element_type()));
        return {complex_result};
    } else if (lhs_complex) {
        // rhs is of a real type
        auto lhs_real = lhs_complex->get_real();
        auto lhs_imag = lhs_complex->get_imag();
        auto result_real = context.mark_node(make_shared<v1::Subtract>(lhs_real, rhs))->output(0);
        auto complex_result =
            context.mark_node(make_shared<ComplexTypeMark>(result_real, lhs_imag, result_real.get_element_type()));
        return {complex_result};
    } else if (rhs_complex) {
        // lhs is of a real type
        auto rhs_real = rhs_complex->get_real();
        auto rhs_imag = rhs_complex->get_imag();
        auto result_real = context.mark_node(make_shared<v1::Subtract>(lhs, rhs_real))->output(0);
        rhs_imag = context.mark_node(make_shared<v0::Negative>(rhs_imag));

        auto complex_result =
            context.mark_node(make_shared<ComplexTypeMark>(result_real, rhs_imag, result_real.get_element_type()));
        return {complex_result};
    }

    // both operands are real
    auto result = context.mark_node(make_shared<v1::Subtract>(lhs, rhs));

    return {result};
}

ov::Output<ov::Node> ComplexTypeMark::mul(const NodeContext& context,
                                          const ov::Output<ov::Node>& lhs,
                                          const ov::Output<ov::Node>& rhs) {
    auto lhs_complex = as_type_ptr<ComplexTypeMark>(lhs.get_node_shared_ptr());
    auto rhs_complex = as_type_ptr<ComplexTypeMark>(rhs.get_node_shared_ptr());

    if (lhs_complex && rhs_complex) {
        // both operands are of complex type
        // formula for guidance: (a + b*i) * (c + d*i) = (ac-bd) + (ad+bc)*i
        auto lr = lhs_complex->get_real();
        auto li = lhs_complex->get_imag();

        auto rr = rhs_complex->get_real();
        auto ri = rhs_complex->get_imag();

        auto mul_lr_rr = context.mark_node(make_shared<v1::Multiply>(lr, rr));
        auto mul_li_ri = context.mark_node(make_shared<v1::Multiply>(li, ri));
        auto res_real = context.mark_node(make_shared<v1::Subtract>(mul_lr_rr, mul_li_ri))->output(0);

        auto mul_lr_ri = context.mark_node(make_shared<v1::Multiply>(lr, ri));
        auto mul_li_rr = context.mark_node(make_shared<v1::Multiply>(li, rr));
        auto res_imag = context.mark_node(make_shared<v1::Add>(mul_lr_ri, mul_li_rr));

        auto complex_result =
            context.mark_node(make_shared<ComplexTypeMark>(res_real, res_imag, res_real.get_element_type()));
        return {complex_result};
    } else if (lhs_complex) {
        auto lhs_data = lhs_complex->get_data();

        // rhs is of a real type
        auto unsqueeze_axis = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{1}, -1));
        auto unsqueezed_rhs = context.mark_node(make_shared<v0::Unsqueeze>(rhs, unsqueeze_axis));
        auto result = context.mark_node(make_shared<v1::Multiply>(lhs_data, unsqueezed_rhs))->output(0);

        auto complex_result = context.mark_node(make_shared<ComplexTypeMark>(result, result.get_element_type()));
        return {complex_result};
    } else if (rhs_complex) {
        auto rhs_data = rhs_complex->get_data();

        // rhs is of a real type
        auto unsqueeze_axis = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{1}, -1));
        auto unsqueezed_lhs = context.mark_node(make_shared<v0::Unsqueeze>(lhs, unsqueeze_axis));
        auto result = context.mark_node(make_shared<v1::Multiply>(unsqueezed_lhs, rhs_data))->output(0);

        auto complex_result = context.mark_node(make_shared<ComplexTypeMark>(result, result.get_element_type()));
        return {complex_result};
    }

    // both operands are real
    auto result = context.mark_node(make_shared<v1::Multiply>(lhs, rhs));

    return {result};
}

ov::Output<ov::Node> ComplexTypeMark::inv(const NodeContext& context, const ov::Output<ov::Node>& data) {
    auto data_complex = as_type_ptr<ComplexTypeMark>(data.get_node_shared_ptr());

    if (data_complex) {
        auto real = data_complex->get_real();
        auto imag = data_complex->get_imag();

        // inverse of complex number:
        // 1 / (a + b*i) = (a/(a^2 + b^2)) + (-b/(a^2 + b^2))*i
        auto two = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{}, 2));
        two = context.mark_node(make_shared<v1::ConvertLike>(two, real));
        auto real_squared = context.mark_node(make_shared<v1::Power>(real, two));
        auto imag_squared = context.mark_node(make_shared<v1::Power>(imag, two));
        auto denom = context.mark_node(make_shared<v1::Add>(real_squared, imag_squared));

        auto minus_imag = context.mark_node(make_shared<v0::Negative>(imag));

        auto res_real = context.mark_node(make_shared<v1::Divide>(real, denom))->output(0);
        auto res_imag = context.mark_node(make_shared<v1::Divide>(minus_imag, denom));

        auto complex_result =
            context.mark_node(make_shared<ComplexTypeMark>(res_real, res_imag, res_real.get_element_type()));
        return {complex_result};
    }

    auto one = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{}, 1));
    one = context.mark_node(make_shared<v1::ConvertLike>(one, data));
    auto result = context.mark_node(make_shared<v1::Divide>(one, data));
    return {result};
}

ov::Output<ov::Node> ComplexTypeMark::div(const NodeContext& context,
                                          const ov::Output<ov::Node>& lhs,
                                          const ov::Output<ov::Node>& rhs) {
    // compute inverse of rhs
    auto inv_rhs = inv(context, rhs);
    // multiply lhs by inversed rhs
    auto result = mul(context, lhs, inv_rhs);
    return {result};
}

ov::Output<ov::Node> ComplexTypeMark::convert_like(const NodeContext& context,
                                                   const ov::Output<ov::Node>& input,
                                                   const ov::Output<ov::Node>& like) {
    auto like_complex = as_type_ptr<ComplexTypeMark>(like.get_node_shared_ptr());

    ov::Output<ov::Node> like_data = like;
    if (like_complex) {
        like_data =
            (like_complex->get_data().get_node_shared_ptr() ? like_complex->get_data() : like_complex->get_real());
    }

    auto input_complex = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    if (input_complex && input_complex->get_data().get_node_shared_ptr()) {
        auto new_input_data = input_complex->get_data();
        new_input_data = context.mark_node(make_shared<v1::ConvertLike>(new_input_data, like_data));
        return context.mark_node(make_shared<ComplexTypeMark>(new_input_data));
    } else if (input_complex) {
        auto new_real = input_complex->get_real();
        auto new_imag = input_complex->get_imag();
        new_real = context.mark_node(make_shared<v1::ConvertLike>(new_real, like_data));
        new_imag = context.mark_node(make_shared<v1::ConvertLike>(new_imag, like_data));
        return context.mark_node(make_shared<ComplexTypeMark>(new_real, new_imag));
    }

    return context.mark_node(make_shared<v1::ConvertLike>(input, like_data));
}

ov::Output<ov::Node> ComplexTypeMark::abs(const NodeContext& context, const ov::Output<ov::Node>& data) {
    auto data_complex = as_type_ptr<ComplexTypeMark>(data.get_node_shared_ptr());

    if (data_complex) {
        // abs of complex number sqrt(real^2 + imag^2)
        auto data = data_complex->get_data();

        // compute element-wise square for complex representation
        auto const_two = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{}, 2));
        const_two = context.mark_node(make_shared<v1::ConvertLike>(const_two, data));
        auto squared_data = context.mark_node(make_shared<v1::Power>(data, const_two));

        // compute sum of squared real and imaginary parts
        auto const_minus_one = context.mark_node(make_shared<v0::Constant>(element::i32, Shape{}, -1));
        auto complex_abs = context.mark_node(make_shared<v1::ReduceSum>(squared_data, const_minus_one, false));

        return context.mark_node(make_shared<v0::Sqrt>(complex_abs));
    }

    return context.mark_node(make_shared<v0::Abs>(data));
}

ov::Output<ov::Node> ComplexTypeMark::exp(const NodeContext& context, const ov::Output<ov::Node>& data) {
    auto data_complex = as_type_ptr<ComplexTypeMark>(data.get_node_shared_ptr());

    if (data_complex) {
        auto real = data_complex->get_real();
        auto imag = data_complex->get_imag();

        // exp of complex number e^real * cos(imag) + i e^real * sin(imag)
        auto real_exp = context.mark_node(make_shared<v0::Exp>(real));

        auto imag_cos = context.mark_node(make_shared<v0::Cos>(imag));
        auto imag_sin = context.mark_node(make_shared<v0::Sin>(imag));

        auto res_real = context.mark_node(make_shared<v1::Multiply>(real_exp, imag_cos));
        auto res_imag = context.mark_node(make_shared<v1::Multiply>(real_exp, imag_sin));

        return context.mark_node(make_shared<ComplexTypeMark>(res_real, res_imag));
        ;
    }

    return context.mark_node(make_shared<v0::Exp>(data));
}

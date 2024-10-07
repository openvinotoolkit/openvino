// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_rsqrt_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Rsqrt", "RSQRT"}, true);
    auto input = node.get_input(0);
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());

    if (complex_type_mark) {
        element::Type complex_part_type = complex_type_mark->get_complex_part_type();
        input = complex_type_mark->input_value(0);
        // input is complex tensor representation in a form [N1, N2, ..., Nk, 2]
        // where slice [N1, N2, ..., Nk, 0] contains real part of the complex tensor
        // and slice [N1, N2, ..., Nk, 1] contains imaginary part of the complex tensor

        auto gather_index_real = make_shared<v0::Constant>(element::i64, Shape{}, 0);
        auto gather_index_imag = make_shared<v0::Constant>(element::i64, Shape{}, 1);
        auto minus_one = make_shared<v0::Constant>(element::i32, Shape{1}, -1);

        // complex_number: z = a + jb ; real_part=a ; imag_part = b
        auto real_part = make_shared<v8::Gather>(input, gather_index_real, minus_one);
        auto imag_part = make_shared<v8::Gather>(input, gather_index_imag, minus_one);

        auto const_half = create_same_type_const_scalar<float>(real_part, 0.5f);
        auto const_two = create_same_type_const_scalar<float>(real_part, 2.0f);
        auto const_zero = create_same_type_const_scalar<float>(real_part, 0.0f);
        auto const_one = create_same_type_const_scalar<float>(real_part, 1.0f);
        auto const_minus_one = create_same_type_const_scalar<float>(real_part, -1.0f);

        // sum_sq = a^2 + b^2
        auto sum_sq = std::make_shared<v1::Add>(std::make_shared<v1::Power>(real_part, const_two),
                                                std::make_shared<v1::Power>(imag_part, const_two));

        // |z| = sqrt(sum_sq)
        auto norm = std::make_shared<v0::Sqrt>(sum_sq);

        // if real_part >= 0
        auto real_gte_zero = std::make_shared<v1::GreaterEqual>(real_part, const_zero);

        // then
        // new_real = sqrt( (real_part + norm) / 2 )
        auto new_real_when_real_gte_zero = std::make_shared<v0::Sqrt>(
            std::make_shared<v1::Divide>(std::make_shared<v1::Add>(real_part, norm), const_two));

        // new_imag = imag_part / (2 * new_real)
        auto new_imag_when_real_gte_zero =
            std::make_shared<v1::Divide>(imag_part,
                                         std::make_shared<v1::Multiply>(new_real_when_real_gte_zero, const_two));

        // else
        // new_imag = sqrt( (norm -  real_part) / 2 ) * sign
        auto imag_lt_zero = std::make_shared<v1::Less>(imag_part, const_zero);

        auto new_imag_when_real_lt_zero_no_sign = std::make_shared<v0::Sqrt>(
            std::make_shared<v1::Divide>(std::make_shared<v1::Subtract>(norm, real_part), const_two));

        auto new_imag_when_real_lt_zero =
            std::make_shared<v1::Select>(imag_lt_zero,
                                         std::make_shared<v0::Negative>(new_imag_when_real_lt_zero_no_sign),
                                         new_imag_when_real_lt_zero_no_sign);

        // new_real = imag / (2 * new_imag)
        auto new_real_when_real_lt_zero =
            std::make_shared<v1::Divide>(imag_part,
                                         std::make_shared<v1::Multiply>(new_imag_when_real_lt_zero, const_two));
        auto new_real =
            std::make_shared<v1::Select>(real_gte_zero, new_real_when_real_gte_zero, new_real_when_real_lt_zero);

        auto new_imag =
            std::make_shared<v1::Select>(real_gte_zero, new_imag_when_real_gte_zero, new_imag_when_real_lt_zero);
        // endif

        // new_sum_sq = new_real^2 + new_imag^2
        auto new_sum_sq = make_shared<v1::Add>(make_shared<v1::Power>(new_real, const_two),
                                               make_shared<v1::Power>(new_imag, const_two));

        // rsqrt_real = sqrt_real/(sqrt_real^2 + sqrt_imag^2)
        auto rsqrt_real = make_shared<v1::Divide>(new_real, new_sum_sq);

        // rsqrt_imag = - sqrt_imag/(sqrt_real^2 + sqrt_imag^2)
        auto rsqrt_imag = make_shared<v0::Negative>(make_shared<v1::Divide>(new_imag, new_sum_sq));

        // check if z = 0 + 0j
        // then new_z = inf + nan*j
        // else new_z = rsqrt_real + rsqrt_imag*j
        auto new_real_eq_zero = std::make_shared<v1::Equal>(real_part, const_zero);
        auto new_imag_eq_zero = std::make_shared<v1::Equal>(imag_part, const_zero);
        auto real_imag_zero = std::make_shared<v1::LogicalAnd>(new_real_eq_zero, new_imag_eq_zero);

        auto const_inf = create_same_type_const_scalar<float>(real_part, std::numeric_limits<float>::infinity());
        auto rsqrt_real_final = std::make_shared<v1::Select>(real_imag_zero, const_inf, rsqrt_real);

        auto const_nan = create_same_type_const_scalar<float>(real_part, std::numeric_limits<float>::quiet_NaN());
        auto rsqrt_imag_final = std::make_shared<v1::Select>(real_imag_zero, const_nan, rsqrt_imag);

        auto real_unsqueeze = make_shared<v0::Unsqueeze>(rsqrt_real_final, minus_one);
        auto imag_unsqueeze = make_shared<v0::Unsqueeze>(rsqrt_imag_final, minus_one);

        auto concat_result = make_shared<v0::Concat>(OutputVector{real_unsqueeze, imag_unsqueeze}, -1);
        set_node_name(node.get_name(), concat_result);

        auto complex_result = make_shared<ComplexTypeMark>(concat_result, complex_part_type);
        return {complex_result};
    }

    auto exponent = create_same_type_const_scalar<float>(input, -0.5f);
    auto rsqrt = make_shared<v1::Power>(input, exponent);
    set_node_name(node.get_name(), rsqrt);
    return {rsqrt};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov

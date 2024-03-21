// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/power.hpp"
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
        // where slice [N1, N2, ..., Nk, 0] contains real part of the complex
        // tensor and slice [N1, N2, ..., Nk, 1] contains imaginary part of the
        // complex tensor compute sum of squared real and imaginary parts

        auto gather_index_real = make_shared<v0::Constant>(input.get_element_type(), Shape{}, 0);
        auto gather_index_imag = make_shared<v0::Constant>(input.get_element_type(), Shape{}, 1);
        auto minus_one = make_shared<v0::Constant>(input.get_element_type(), Shape{1}, -1);

        // complex_number = a + ib ; real_part=a ; imag_part = b
        auto real_part = make_shared<v8::Gather>(input, gather_index_real, minus_one)->output(0);
        auto imag_part = make_shared<v8::Gather>(input, gather_index_imag, minus_one)->output(0);

        auto const_half = create_same_type_const_scalar<float>(real_part, 0.5f);
        auto const_two = create_same_type_const_scalar<float>(real_part, 2.0f);

        // get square of absolute value
        auto mod_sqr = make_shared<v1::Power>(make_shared<v1::Add>(make_shared<v1::Power>(real_part, const_two),
                                                                   make_shared<v1::Power>(imag_part, const_two)),
                                              const_half);  // a2 + b2

        // sqrt_real = sqrt( a + sqrt( a^2 + b^2) / 2 )
        // sqrt_imag = b/|b| sqrt( -a + sqrt(a^2 + b^2) / 2 )
        auto sqrt_real =
            make_shared<v1::Power>(make_shared<v1::Divide>(make_shared<v1::Add>(mod_sqr, real_part), 2.0), const_half);
        auto sign =
            make_shared<v1::Divide>(imag_part,
                                    make_shared<v1::Power>(make_shared<v1::Power>(imag_part, const_two), const_half));
        auto sqrt_imag = make_shared<v1::Power>(
            make_shared<v1::Divide>(make_shared<v1::Add>(mod_sqr, make_shared<v0::Negative>(real_part)), 2.0),
            const_half);
        // rsqrt_real = sqrt_real/(sqrt_real^2 + sqrt_imag^2)
        // rsqrt_imag = - sqrt_imag/(sqrt_real^2 + sqrt_imag^2)
        auto rsqrt_real = make_shared<v1::Divide>(sqrt_real,
                                                  make_shared<v1::Add>(make_shared<v1::Power>(sqrt_real, const_two),
                                                                       make_shared<v1::Power>(sqrt_imag, const_two)));
        auto rsqrt_imag = make_shared<v0::Negative>(
            make_shared<v1::Divide>(sqrt_imag,
                                    make_shared<v1::Add>(make_shared<v1::Power>(sqrt_real, const_two),
                                                         make_shared<v1::Power>(sqrt_imag, const_two))));

        auto real_unsqueeze = make_shared<v0::Unsqueeze>(rsqrt_real, minus_one);
        auto imag_unsqueeze = make_shared<v0::Unsqueeze>(rsqrt_imag, minus_one);

        auto concat_result = make_shared<v0::Concat>(OutputVector{real_unsqueeze, imag_unsqueeze}, -1);
        set_node_name(node.get_name(), concat_result);

        auto complex_result = make_shared<ComplexTypeMark>(concat_result->output(0), complex_part_type);
        return {complex_result};
    } else {
        auto exponent = create_same_type_const_scalar<float>(input, -0.5f);
        auto rsqrt = make_shared<v1::Power>(input, exponent);
        set_node_name(node.get_name(), rsqrt);
        return {rsqrt};
    }
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov

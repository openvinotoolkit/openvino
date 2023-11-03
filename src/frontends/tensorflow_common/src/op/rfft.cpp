// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/core/any.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/rdft.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_rfft_op(const NodeContext& node) {
    default_op_checks(node, 2, {"RFFT", "RFFT2D", "RFFT3D"});
    auto input = node.get_input(0);
    auto fft_length = node.get_input(1);
    auto tcomplex = node.get_attribute<string>("Tcomplex", "DT_COMPLEX64");
    element::Type complex_part_type = (tcomplex == "DT_COMPLEX64" ? element::f32 : element::f64);

    // compute axes along which to compute inverse RFFT
    auto input_rank = compute_subgraph_scalar_rank(input, element::i32, true);
    auto const_two = make_shared<v0::Constant>(element::i32, Shape{}, 2);
    auto const_one = make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto axes = make_shared<v4::Range>(const_two, input_rank, const_one, element::i32);

    // compute real FFT and align its output type
    auto rfft = make_shared<v9::RDFT>(input, axes, fft_length)->output(0);
    rfft = make_shared<v0::Convert>(rfft, complex_part_type);
    set_node_name(node.get_name(), rfft.get_node_shared_ptr());

    // insert ComplexTypeMark since RFFT generates output of complex type
    auto complex_type_mark = make_shared<ComplexTypeMark>(rfft, complex_part_type);

    return {complex_type_mark};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/irdft.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_irfft_op(const NodeContext& node) {
    default_op_checks(node, 2, {"IRFFT", "IRFFT2D", "IRFFT3D"}, true);
    auto op_type = node.get_op_type();
    auto input = node.get_input(0);
    auto fft_length = node.get_input(1);
    auto treal = node.get_attribute<element::Type>("Treal", element::f32);

    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    TENSORFLOW_OP_VALIDATION(
        node,
        complex_type_mark,
        "[TensorFlow Frontend] internal error: ComplexTypeMark is not created before " + op_type + " operation.");

    // compute a number of inner-most dimensions
    int32_t num_axes = 1;
    if (op_type == "IRFFT2D") {
        num_axes = 2;
    } else if (op_type == "IRFFT3D") {
        num_axes = 3;
    }

    // compute axes along which to compute inverse RFFT
    auto const_num_axes = make_shared<v0::Constant>(element::i32, Shape{}, num_axes);
    auto data = complex_type_mark->get_data();
    auto data_rank = compute_subgraph_scalar_rank(data, element::i32, true);
    auto const_one = make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto data_rank_minus_one = make_shared<v1::Subtract>(data_rank, const_one);
    auto start = make_shared<v1::Subtract>(data_rank_minus_one, const_num_axes);
    auto axes = make_shared<v4::Range>(start, data_rank_minus_one, const_one, element::i32);
    auto irdft = make_shared<v9::IRDFT>(data, axes, fft_length)->output(0);

    // no need to insert ComplexTypeMark because operation generates a floating-point tensor
    irdft = make_shared<v0::Convert>(irdft, treal);
    set_node_name(node.get_name(), irdft.get_node_shared_ptr());

    return {irdft};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov

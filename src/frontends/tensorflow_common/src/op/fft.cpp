// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "helper_ops/complex_type_mark.hpp"
#include "openvino/op/dft.hpp"
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

OutputVector translate_fft_op(const NodeContext& node) {
    default_op_checks(node, 1, {"FFT", "FFT2D", "FFT3D"}, true);
    auto op_type = node.get_op_type();
    auto input = node.get_input(0);

    // check that ComplexTypeMark is set
    auto complex_type_mark = as_type_ptr<ComplexTypeMark>(input.get_node_shared_ptr());
    TENSORFLOW_OP_VALIDATION(
        node,
        complex_type_mark,
        "[TensorFlow Frontend] internal error: ComplexTypeMark is not set to input for " + op_type);
    auto data = complex_type_mark->get_data();
    auto complex_part_type = complex_type_mark->get_complex_part_type();

    // compute a number of inner-most dimensions
    int32_t num_axes = 1;
    if (op_type == "FFT2D") {
        num_axes = 2;
    } else if (op_type == "FFT3D") {
        num_axes = 3;
    }

    // compute axes along which to compute FFT
    auto const_num_axes = make_shared<v0::Constant>(element::i32, Shape{}, num_axes);
    auto const_one = make_shared<v0::Constant>(element::i32, Shape{}, 1);
    auto data_rank = compute_subgraph_scalar_rank(data, element::i32, true);
    // exclude the last dimension since it concatenated real and imaginary parts
    auto data_rank_minus_one = make_shared<v1::Subtract>(data_rank, const_one);
    auto start = make_shared<v1::Subtract>(data_rank_minus_one, const_num_axes);
    auto axes = make_shared<v4::Range>(start, data_rank_minus_one, const_one, element::i32);

    // compute FFT and align its output type
    auto fft = make_shared<v7::DFT>(data, axes);
    set_node_name(node.get_name(), fft);

    // insert ComplexTypeMark since FFT generates output of complex type
    complex_type_mark = make_shared<ComplexTypeMark>(fft, complex_part_type);

    return {complex_type_mark};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {
OutputVector TranslateSoftmaxOp(const NodeContext& node) {
    auto ng_inp = node.get_ng_input(0);
    auto inp_shape = ng_inp.get_shape();
    size_t rank = inp_shape.size();
    int64_t axes = rank - 1;
    if (rank < 1) {
        throw errors::InvalidArgument("TF Softmax logits must be >=1 dimension");
    }

    return {ConstructNgNode<Softmax>(node.get_name(), ng_inp, axes)};
}

}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph

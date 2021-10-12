// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateLogSoftmaxOp(const NodeContext& node) {
    auto ng_inp = node.get_ng_input(0);
    auto inp_shape = ng_inp.get_shape();
    size_t rank = inp_shape.size();
    int64_t axes = rank - 1;

    return {ConstructNgNode<LogSoftmax>(node.get_name(), ng_inp, axes)};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov

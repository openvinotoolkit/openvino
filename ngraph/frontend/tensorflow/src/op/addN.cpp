// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset8.hpp>
#include <numeric>
#include <op_table.hpp>

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tf {
namespace op {

OutputVector TranslateAddNOp(const NodeContext& node) {
    OutputVector ng_arg_vec = node.get_all_ng_inputs();

    auto ng_addn = std::accumulate(std::next(ng_arg_vec.begin()),
                                   ng_arg_vec.end(),
                                   ng_arg_vec.at(0),
                                   [&node](Output<Node> a, Output<Node> b) {
                                       return ConstructNgNode<Add>(node.get_name(), a, b);
                                   });  // accumulation: start with
    // first element. default op is
    // addition
    return {ng_addn};
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ngraph

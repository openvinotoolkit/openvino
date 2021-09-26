// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <numeric>
#include <op_table.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

namespace tensorflow {
namespace ngraph_bridge {

OutputVector TranslateAddNOp(const NodeContext& node) {
    OutputVector ng_arg_vec = node.get_all_ng_inputs();

    auto ng_addn = std::accumulate(std::next(ng_arg_vec.begin()),
                                   ng_arg_vec.end(),
                                   ng_arg_vec.at(0),
                                   [&node](Output<Node> a, Output<Node> b) {
                                       return ConstructNgNode<opset::Add>(node.get_name(), a, b);
                                   });  // accumulation: start with
    // first element. default op is
    // addition
    return {ng_addn};
}
}  // namespace ngraph_bridge
}  // namespace tensorflow
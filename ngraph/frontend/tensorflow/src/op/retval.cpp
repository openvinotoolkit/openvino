// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <default_opset.h>

#include <op_table.hpp>

using namespace std;
using namespace ngraph;
using namespace ngraph::frontend::tensorflow::detail;

namespace tensorflow {
namespace ngraph_bridge {

OutputVector RetvalOp(const NodeContext& node) {
    // Make sure that this _Retval only has one input node.
    if (node.get_ng_input_size() != 1) {
        throw errors::InvalidArgument("_Retval has " + to_string(node.get_ng_input_size()) + " inputs, should have 1");
    }

    // auto ret_val_index = node.get_attribute<int>("index");
    // TODO: Put ret_val_index to RT info that should be later utilized to order outpus by indices

    return {ConstructNgNode<opset::Result>(node.get_name(), node.get_ng_input(0))};
}
}  // namespace ngraph_bridge
}  // namespace tensorflow
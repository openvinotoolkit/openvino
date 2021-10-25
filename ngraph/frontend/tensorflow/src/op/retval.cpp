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

OutputVector RetvalOp(const NodeContext& node) {
    // Make sure that this _Retval only has one input node.
    if (node.get_ng_input_size() != 1) {
        TF_OP_VALIDATION_CHECK(node,
                               false,
                               "_Retval has " + to_string(node.get_ng_input_size()) + " inputs, should have 1");
    }

    // auto ret_val_index = node.get_attribute<int>("index");
    // TODO: Put ret_val_index to RT info that should be later utilized to order outpus by indices

    auto res = make_shared<Result>(node.get_ng_input(0));
    SetNodeNames(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov
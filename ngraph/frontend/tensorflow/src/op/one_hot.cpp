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

OutputVector TranslateOneHotOp(const NodeContext& node) {
    auto ng_features = node.get_ng_input(0);
    auto ng_depth = node.get_ng_input(1);
    auto ng_on = node.get_ng_input(2);
    auto ng_off = node.get_ng_input(3);

    auto one_hot_axis = node.get_attribute<int64_t>("axis");
    auto ng_onehot = make_shared<OneHot>(ng_features, ng_depth, ng_on, ng_off, one_hot_axis);
    ng_onehot->set_friendly_name(ng_onehot->get_friendly_name());
    return ng_onehot->outputs();
}
}  // namespace op
}  // namespace tf
}  // namespace frontend
}  // namespace ov

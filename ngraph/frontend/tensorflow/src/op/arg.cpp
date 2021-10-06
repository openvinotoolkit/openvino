// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset8.hpp>
#include <op_table.hpp>

using namespace std;
using namespace ngraph::opset8;

/*
namespace ngraph {
namespace frontend {
namespace tf {
namespace op {

OutputVector ArgOp(const NodeContext& node) {
    auto ng_et = node.get_attribute<ngraph::element::Type>("T");
    auto overridden_shape = node.get_overridden_shapes().find(node.get_name());
    auto index = node.get_attribute<int>("index");
    auto shape = node.get_indexed_shapes().at(index);
    auto ng_shape = overridden_shape == node.get_overridden_shapes().end() ?
                    shape :
                    overridden_shape->second;
    return {ConstructNgNode<Parameter>(node.get_name(), ng_et, ng_shape)};
}

}
}
 */
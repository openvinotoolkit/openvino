// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_table.hpp>
#include <openvino/opsets/opset8.hpp>

using namespace std;
using namespace ov::opset8;

/*
namespace ov{
namespace frontend {
namespace tf {
namespace op {

OutputVector ArgOp(const NodeContext& node) {
    auto ng_et = node.get_attribute<ov::element::Type>("T");
    auto overridden_shape = node.get_overridden_shapes().find(node.get_name());
    auto index = node.get_attribute<int>("index");
    auto shape = node.get_indexed_shapes().at(index);
    auto ng_shape = overridden_shape == node.get_overridden_shapes().end() ?
                    shape :
                    overridden_shape->second;
    return {make_shared<Parameter>( ng_et, ng_shape)};
}

}
}
 */
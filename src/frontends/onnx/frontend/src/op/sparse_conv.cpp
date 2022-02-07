// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/node.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector sparse_conv(const Node& node) {
    return {std::make_shared<ngraph::opset1::SparseConv>(node.get_ng_inputs().at(0),
                                                         node.get_ng_inputs().at(1),
                                                         node.get_ng_inputs().at(2),
                                                         node.get_ng_inputs().at(3),
                                                         node.get_ng_inputs().at(4))};
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs tanh(const NodeContext& node) {
    const auto x = node.get_ng_input("X");

    return node.default_single_output_mapping({std::make_shared<default_opset::Tanh>(x)}, {"Out"});
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ngraph

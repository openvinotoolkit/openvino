// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "visitors.hpp"

#include "openvino/op/ops.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
ngraph::FactoryRegistry<ov::Node>& ov::test::NodeBuilder::get_ops() {
    static ngraph::FactoryRegistry<Node> registry = [] {
        ngraph::FactoryRegistry<Node> registry;
#define _OPENVINO_OP_REG(NAME, NAMESPACE) registry.register_factory<NAMESPACE::NAME>();
#include "op_version_tbl.hpp"
#undef _OPENVINO_OP_REG
        return registry;
    }();
    return registry;
}
OPENVINO_SUPPRESS_DEPRECATED_END

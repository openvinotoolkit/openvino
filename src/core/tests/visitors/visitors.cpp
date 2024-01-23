// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "visitors.hpp"

#include "openvino/op/ops.hpp"

ov::FactoryRegistry<ov::Node>& ov::test::NodeBuilder::get_ops() {
    static ov::FactoryRegistry<Node> registry = [] {
        ov::FactoryRegistry<Node> registry;
#define _OPENVINO_OP_REG(NAME, NAMESPACE) registry.register_factory<NAMESPACE::NAME>();
#include "op_version_tbl.hpp"
#undef _OPENVINO_OP_REG
        return registry;
    }();
    return registry;
}

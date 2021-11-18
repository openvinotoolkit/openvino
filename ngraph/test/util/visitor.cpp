// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "visitor.hpp"

#include "ngraph/ops.hpp"
#include "ngraph/runtime/host_tensor.hpp"

ngraph::FactoryRegistry<ngraph::Node>& ngraph::test::NodeBuilder::get_ops() {
    static FactoryRegistry<Node> registry = [] {
        FactoryRegistry<Node> registry;
#define NGRAPH_OP(NAME, NAMESPACE, VERSION) registry.register_factory<NAMESPACE::NAME>();
#include "../op_version_tbl.hpp"
#undef NGRAPH_OP
        return registry;
    }();
    return registry;
}

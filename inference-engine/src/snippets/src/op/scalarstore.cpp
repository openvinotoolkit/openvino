// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/scalarstore.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(snippets::op::ScalarStore, "ScalarStore", 0);

snippets::op::ScalarStore::ScalarStore(const Output<Node>& x) : Store(x) {
}

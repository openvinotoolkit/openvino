// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/scalarload.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(snippets::op::ScalarLoad, "ScalarLoad", 0);

snippets::op::ScalarLoad::ScalarLoad(const Output<Node>& x) : Load(x) {
}

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/blockedload.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(snippets::op::BlockedLoad, "BlockedLoad", 0);

snippets::op::BlockedLoad::BlockedLoad(const Output<Node>& x) : Load(x) {
}

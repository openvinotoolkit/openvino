// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/vectorload.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(snippets::op::VectorLoad, "VectorLoad", 0);

snippets::op::VectorLoad::VectorLoad(const Output<Node>& x) : Load(x) {
}

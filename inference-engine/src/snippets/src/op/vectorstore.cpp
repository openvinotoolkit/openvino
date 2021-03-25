// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/vectorstore.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(snippets::op::VectorStore, "VectorStore", 0);

snippets::op::VectorStore::VectorStore(const Output<Node>& x) : Store(x) {
}

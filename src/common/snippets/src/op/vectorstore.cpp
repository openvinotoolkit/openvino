// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/vectorstore.hpp"

using namespace ngraph;

snippets::op::VectorStore::VectorStore(const Output<Node>& x, const size_t count) : Store(x, count) {
}

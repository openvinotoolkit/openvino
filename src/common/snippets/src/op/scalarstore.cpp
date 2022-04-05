// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/scalarstore.hpp"

using namespace ngraph;

snippets::op::ScalarStore::ScalarStore(const Output<Node>& x) : Store(x) {
}

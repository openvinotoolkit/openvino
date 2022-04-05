// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/scalarload.hpp"

using namespace ngraph;

snippets::op::ScalarLoad::ScalarLoad(const Output<Node>& x) : Load(x) {
}

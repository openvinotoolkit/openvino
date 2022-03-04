// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/blockedload.hpp"

using namespace ngraph;

snippets::op::BlockedLoad::BlockedLoad(const Output<Node>& x) : Load(x) {
}

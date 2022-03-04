// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/vectorload.hpp"

using namespace ngraph;

snippets::op::VectorLoad::VectorLoad(const Output<Node>& x) : Load(x) {
}

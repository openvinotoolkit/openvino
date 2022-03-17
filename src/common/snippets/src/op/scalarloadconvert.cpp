// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/scalarloadconvert.hpp"

using namespace ngraph;

snippets::op::ScalarLoadConvert::ScalarLoadConvert(const Output<Node>& x, const ov::element::Type& destination_type) :
    LoadConvert(x, destination_type, 1lu) {
}

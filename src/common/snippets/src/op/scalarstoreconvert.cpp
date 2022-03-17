// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/scalarstoreconvert.hpp"

using namespace ngraph;

snippets::op::ScalarStoreConvert::ScalarStoreConvert(const Output<Node>& x, const ov::element::Type& destination_type) :
    StoreConvert(x, destination_type, 1lu) {
}

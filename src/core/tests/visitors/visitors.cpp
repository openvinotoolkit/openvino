// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "visitors.hpp"

namespace ov {
namespace test {

OpSet& NodeBuilder::opset() {
    static auto opset = OpSet("test_opset");
    return opset;
}
}  // namespace test
}  // namespace ov

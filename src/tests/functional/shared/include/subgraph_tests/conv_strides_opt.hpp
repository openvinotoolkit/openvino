// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "subgraph/conv_strides_opt.hpp"

namespace ov {
namespace test {

TEST_P(ConvStridesOpt, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov

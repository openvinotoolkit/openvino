// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "subgraph/broadcast_eltwise_eliminated.hpp"

namespace ov {
namespace test {

TEST_P(BroadcastEltwiseEliminated, CompareWithRefs){
    run();
};

}  // namespace test
}  // namespace ov

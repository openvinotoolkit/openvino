// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_op/activation.hpp"

namespace ov {
namespace test {

TEST_P(ActivationLayerTest, Inference) {
    run();
}

TEST_P(ActivationParamLayerTest, Inference) {
    run();
}

TEST_P(ActivationLayerTest, QueryModel) {
    query_model();
}

TEST_P(ActivationParamLayerTest, QueryModel) {
    query_model();
}

}  // namespace test
}  // namespace ov

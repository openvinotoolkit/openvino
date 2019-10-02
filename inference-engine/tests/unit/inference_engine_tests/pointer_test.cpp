// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "cpp/ie_cnn_net_reader.h"

using namespace InferenceEngine;

class PointerTests : public ::testing::Test {};

TEST_F(PointerTests, InferenceEnginePtrStoresValues) {
    std::shared_ptr <ICNNNetReader> p(InferenceEngine::CreateCNNNetReader());
    ASSERT_NE(p.get(), nullptr);
}

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using adapoolParams = std::tuple<
        std::vector<InputShape>,            // feature map shape
        std::vector<int>,                   // pooled spatial shape
        std::string,                        // pooling mode
        ov::element::Type,                  // model type
        std::string>;                       // device name

class AdaPoolLayerTest : public testing::WithParamInterface<adapoolParams>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<adapoolParams>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov

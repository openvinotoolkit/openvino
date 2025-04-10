// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        double,                        // epsilon
        ov::element::Type,             // Model type
        std::vector<InputShape>,       // Input shape
        std::string                    // Target device name
> BatchNormLayerTestParams;

class BatchNormLayerTest : public testing::WithParamInterface<BatchNormLayerTestParams>,
                           virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<BatchNormLayerTestParams>& obj);
protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <vector>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        ov::element::Type,          // depth type (any integer type)
        int64_t,                    // depth value
        ov::element::Type,          // On & Off values type (any supported type)
        float,                      // OnValue
        float,                      // OffValue
        int64_t,                    // axis
        ov::element::Type,          // Model type
        std::vector<InputShape>,    // Input shapes
        std::string                 // Target device name
> oneHotLayerTestParamsSet;

class OneHotLayerTest : public testing::WithParamInterface<oneHotLayerTestParamsSet>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<oneHotLayerTestParamsSet>& obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

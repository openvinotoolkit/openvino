// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2018-2025 Intel Corporation
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {

using RNNCellParams = typename std::tuple<
    bool,                              // Use decompose to sub-ops transformation
    size_t,                            // Batch
    size_t,                            // Hidden size
    size_t,                            // Input size
    std::vector<std::string>,          // Activations
    float,                             // Clip
    ov::test::utils::InputLayerType,   // W input type (Constant or Parameter)
    ov::test::utils::InputLayerType,   // R input type (Constant or Parameter)
    ov::test::utils::InputLayerType,   // B input type (Constant or Parameter)
    ov::element::Type,                 // Model type
    ov::test::TargetDevice             // Device name
>;

class RNNCellTest : public testing::WithParamInterface<RNNCellParams >,
                        virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RNNCellParams> &obj);

protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

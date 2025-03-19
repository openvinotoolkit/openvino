// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
typedef std::tuple<
        std::vector<InputShape>,  // Input shapes
        ov::element::Type,        // Model type
        std::vector<int64_t>,     // Axes
        std::vector<int64_t>,     // Signal size
        ov::test::utils::DFTOpType,
        std::string> DFTParams;   // Device name

class DFTLayerTest : public testing::WithParamInterface<DFTParams>,
                     virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DFTParams>& obj);

protected:
    //TO DO, to be removed after 125993
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;
};
} // namespace test
} // namespace ov

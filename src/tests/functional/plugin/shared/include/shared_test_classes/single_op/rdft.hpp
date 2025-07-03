// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/test_enums.hpp"

namespace ov {
namespace test {
using RDFTParams = std::tuple<
    std::vector<size_t>,            // Input shape
    ov::element::Type,              // Model type
    std::vector<int64_t>,           // Axes
    std::vector<int64_t>,           // Signal size
    ov::test::utils::DFTOpType,
    ov::test::TargetDevice          // Device name
>;

class RDFTLayerTest : public testing::WithParamInterface<RDFTParams>,
                      virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RDFTParams>& obj);

protected:
    //TO DO, to be removed after 125993
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void SetUp() override;
};
}  // namespace test
}  // namespace ov

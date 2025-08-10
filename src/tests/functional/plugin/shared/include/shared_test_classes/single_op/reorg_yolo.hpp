// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using ReorgYoloParamsTuple = typename std::tuple<
    std::vector<size_t>,            // Input shape
    size_t,                         // Stride
    ov::element::Type,              // Model type
    ov::test::TargetDevice          // Device name
>;

class ReorgYoloLayerTest : public testing::WithParamInterface<ReorgYoloParamsTuple>,
                            virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ReorgYoloParamsTuple> &obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov

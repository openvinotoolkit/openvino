// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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
using roiPoolingParamsTuple = std::tuple<
        std::vector<InputShape>,                    // Input, coords shapes
        ov::Shape,                                  // Pooled shape {pooled_h, pooled_w}
        float,                                      // Spatial scale
        ov::test::utils::ROIPoolingTypes,           // ROIPooling method
        ov::element::Type,                          // Model type
        ov::test::TargetDevice>;                    // Device name

class ROIPoolingLayerTest : public testing::WithParamInterface<roiPoolingParamsTuple>,
                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<roiPoolingParamsTuple>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using roialignParams = std::tuple<
        std::vector<InputShape>,                 // Feature map shape
        ov::Shape,                               // Proposal coords shape
        int,                                     // Bin's row count
        int,                                     // Bin's column count
        float,                                   // Spatial scale
        int,                                     // Pooling ratio
        std::string,                             // Pooling mode
        ov::element::Type,                       // Model type
        ov::test::TargetDevice>;                 // Device name

class ROIAlignLayerTest : public testing::WithParamInterface<roialignParams>,
                              virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<roialignParams>& obj);
    static void fillCoordTensor(std::vector<float>& coords, int height, int width,
                                float spatialScale, int pooledRatio, int pooledH, int pooledW);
    static void fillIdxTensor(std::vector<int>& idx, int batchSize);

protected:
    void SetUp() override;
};

using roialignV9Params = std::tuple<
        std::vector<InputShape>,                 // Feature map shape
        ov::Shape,                               // Proposal coords shape
        int,                                     // Bin's row count
        int,                                     // Bin's column count
        float,                                   // Spatial scale
        int,                                     // Pooling ratio
        std::string,                             // Pooling mode
        std::string,                             // ROI aligned mode
        ov::element::Type,                       // Model type
        ov::test::TargetDevice>;                 // Device name
class ROIAlignV9LayerTest : public testing::WithParamInterface<roialignV9Params>,
                            virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<roialignV9Params>& obj);

protected:
    void SetUp() override;
};
}  // namespace test
}  // namespace ov

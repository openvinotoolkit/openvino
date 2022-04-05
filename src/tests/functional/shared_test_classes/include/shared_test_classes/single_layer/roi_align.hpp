// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using roialignParams = std::tuple<std::vector<size_t>,  // feature map shape
        std::vector<size_t>,                            // proposal coords shape
        int,                                            // bin's row count
        int,                                            // bin's column count
        float,                                          // spatial scale
        int,                                            // pooling ratio
        std::string,                                    // pooling mode
        InferenceEngine::Precision,                     // net precision
        LayerTestsUtils::TargetDevice>;                 // device name

class ROIAlignLayerTest : public testing::WithParamInterface<roialignParams>,
                              virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<roialignParams>& obj);
    static void fillCoordTensor(std::vector<float>& coords, int height, int width,
                                float spatialScale, int pooledRatio, int pooledH, int pooledW);
    static void fillIdxTensor(std::vector<int>& idx, int batchSize);

protected:
    void SetUp() override;

private:
    int pooledH;
    int pooledW;
    float spatialScale;
    int poolingRatio;
    std::string poolingMode;
};

}  // namespace LayerTestsDefinitions

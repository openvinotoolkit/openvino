// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

using psroiParams = std::tuple<std::vector<size_t>,            // input shape
                               std::vector<size_t>,            // coords shape
                               size_t,                         // output_dim
                               size_t,                         // group_size
                               float,                          // Spatial scale
                               size_t,                         // spatial_bins_x
                               size_t,                         // spatial_bins_y
                               std::string,                    // mode
                               InferenceEngine::Precision,     // Net precision
                               LayerTestsUtils::TargetDevice>; // Device name

class PSROIPoolingLayerTest : public testing::WithParamInterface<psroiParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
        public:
            static std::string getTestCaseName(const testing::TestParamInfo<psroiParams>& obj);
            void GenerateInputs() override;
    static void fillROITensor(float* buffer, int numROIs, int batchSize,
                              int height, int width, int groupSize,
                              float spatialScale, int spatialBinsX, int spatialBinsY, const std::string& mode);

        protected:
            void SetUp() override;

        private:
            size_t groupSize_;
            float spatialScale_;
            size_t spatialBinsX_;
            size_t spatialBinsY_;
            std::string mode_;
    };

}  // namespace LayerTestsDefinitions

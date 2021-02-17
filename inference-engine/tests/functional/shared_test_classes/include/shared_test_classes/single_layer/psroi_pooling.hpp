// Copyright (C) 2020 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

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
            static std::string getTestCaseName(testing::TestParamInfo<psroiParams> obj);
            void Infer() override;

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

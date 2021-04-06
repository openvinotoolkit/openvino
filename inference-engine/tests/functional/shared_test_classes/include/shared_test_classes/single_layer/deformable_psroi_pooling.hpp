// Copyright (C) 2021 Intel Corporation
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

using deformablePSROISpecificParams = std::tuple<
                               std::vector<size_t>,            // data input shape
                               std::vector<size_t>,            // rois input shape
                               std::vector<size_t>,            // trans input shape
                               size_t,                         // output_dim
                               size_t,                         // group_size
                               float,                          // spatial_scale
                               std::vector<size_t>,            // spatial_bins_x_y
                               float,                          // trans_std
                               size_t>;                        // part_size

using deformablePSROILayerTestParams = std::tuple<
                               deformablePSROISpecificParams,
                               InferenceEngine::Precision,     // Net precision
                               LayerTestsUtils::TargetDevice>; // Device name

class DeformablePSROIPoolingLayerTest : public testing::WithParamInterface<deformablePSROILayerTestParams>,
    virtual public LayerTestsUtils::LayerTestsCommon {
        public:
            static std::string getTestCaseName(testing::TestParamInfo<deformablePSROILayerTestParams> obj);
            void GenerateInputs() override;

        protected:
            void SetUp() override;

        private:
            float spatialScale_;
    };

}  // namespace LayerTestsDefinitions

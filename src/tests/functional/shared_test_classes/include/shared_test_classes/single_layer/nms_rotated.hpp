// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "shared_test_classes/base/layer_test_utils.hpp"


namespace LayerTestsDefinitions {

using InputShapeParams = std::tuple<size_t,   // Number of batches
                                    size_t,   // Number of boxes
                                    size_t>;  // Number of classes

using InputPrecisions =
    std::tuple<InferenceEngine::Precision,   // boxes and scores precisions
               InferenceEngine::Precision,   // max_output_boxes_per_class precision
               InferenceEngine::Precision>;  // iou_threshold, score_threshold, soft_nms_sigma precisions

using NmsRotatedParams = std::tuple<InputShapeParams,  // Params using to create 1st and 2nd inputs
                                    InputPrecisions,   // Input precisions
                                    int32_t,           // Max output boxes per class
                                    float,             // IOU threshold
                                    float,             // Score threshold
                                    bool,              // Sort result descending
                                    ov::element::Type, // Output type
                                    bool,              // Clockwise
                                    std::string>;      // Device name

class NmsRotatedLayerTest : public testing::WithParamInterface<NmsRotatedParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NmsRotatedParams>& obj);
    void GenerateInputs() override;
    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) override;

protected:
    void SetUp() override;
    InputShapeParams inShapeParams;
};

}  // namespace LayerTestsDefinitions

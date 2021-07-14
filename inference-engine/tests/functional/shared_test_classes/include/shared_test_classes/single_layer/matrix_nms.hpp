// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

using InputShapeParams = std::tuple<size_t,  // Number of batches
                                    size_t,  // Number of boxes
                                    size_t>; // Number of classes

using InputPrecisions = std::tuple<InferenceEngine::Precision,  // boxes and scores precisions
                                   InferenceEngine::Precision,  // max_output_boxes_per_class precision
                                   InferenceEngine::Precision>; // iou_threshold, score_threshold, soft_nms_sigma precisions

using TopKParams = std::tuple<int,      // Maximum number of boxes to be selected per class
                              int>;     // Maximum number of boxes to be selected per batch element

using ThresholdParams = std::tuple<float,   // minimum score to consider box for the processing
                                   float,   // gaussian_sigma parameter for gaussian decay_function
                                   float>;  // filter out boxes with low confidence score after decaying

using NmsParams = std::tuple<InputShapeParams,                                   // Params using to create 1st and 2nd inputs
                             InputPrecisions,                                    // Input precisions
                             ngraph::op::v8::MatrixNms::SortResultType,          // Order of output elements
                             ngraph::element::Type,                              // Output type
                             TopKParams,                                         // Maximum number of boxes topk params
                             ThresholdParams,                                    // Thresholds: score_threshold, gaussian_sigma, post_threshold
                             int,                                                // Background class id
                             bool,                                               // If boxes are normalized
                             ngraph::op::v8::MatrixNms::DecayFunction,           // Decay function
                             std::string>;                                       // Device name

class MatrixNmsLayerTest : public testing::WithParamInterface<NmsParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<NmsParams> obj);
    void GenerateInputs() override;
    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs)
    override;

protected:
    void SetUp() override;

private:
    size_t numBatches, numBoxes, numClasses;
    size_t maxOutputBoxesPerClass;
    size_t maxOutputBoxesPerBatch;
};

}  // namespace LayerTestsDefinitions

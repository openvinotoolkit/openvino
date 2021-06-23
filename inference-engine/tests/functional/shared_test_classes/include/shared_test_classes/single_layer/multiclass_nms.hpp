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

using NmsParams = std::tuple<InputShapeParams,                                   // Params using to create 1st and 2nd inputs
                             InputPrecisions,                                    // Input precisions
                             ngraph::op::v8::MulticlassNms::SortResultType,      // Order of output elements
                             bool,                                               // If necessary to sort selected boxes across batches
                             ngraph::element::Type,                              // Output type
                             int,                                                // Maximum number of boxes to be selected per class
                             int,                                                // Maximum number of boxes to be selected per batch element
                             int,                                                // Background class id
                             bool,                                               // Boxes normalized or not
                             std::string>;                                       // Device name

class MulticlassNmsLayerTest : public testing::WithParamInterface<NmsParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<NmsParams> obj);
    void GenerateInputs() override;
    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs)
    override;

protected:
    void SetUp() override;

private:
    size_t numOfSelectedBoxes;
};

}  // namespace LayerTestsDefinitions

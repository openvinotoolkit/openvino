// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"

namespace testing {
namespace internal {

template <> inline void
PrintTo(const ::ngraph::op::v5::NonMaxSuppression::BoxEncodingType& value,
    ::std::ostream* os) { }

}
}

namespace LayerTestsDefinitions {

using TargetShapeParams = std::tuple<size_t,   // Number of batches
                                     size_t,   // Number of boxes
                                     size_t>;  // Number of classes

using InputShapeParams = std::tuple<std::vector<ov::Dimension>,       // bounds for input dynamic shape
                                    std::vector<TargetShapeParams>>;  // target input dimensions

using InputPrecisions = std::tuple<InferenceEngine::Precision,  // boxes and scores precisions
                                   InferenceEngine::Precision,  // max_output_boxes_per_class precision
                                   InferenceEngine::Precision>; // iou_threshold, score_threshold, soft_nms_sigma precisions

using ThresholdValues = std::tuple<float,  // IOU threshold
                                   float,  // Score threshold
                                   float>; // Soft NMS sigma

using NmsParams = std::tuple<InputShapeParams,                                   // Params using to create 1st and 2nd inputs
                             InputPrecisions,                                    // Input precisions
                             int32_t,                                            // Max output boxes per class
                             ThresholdValues,                                    // IOU, Score, Soft NMS sigma
                             ngraph::helpers::InputLayerType,                    // max_output_boxes_per_class input type
                             ngraph::op::v5::NonMaxSuppression::BoxEncodingType, // Box encoding
                             bool,                                               // Sort result descending
                             ngraph::element::Type,                              // Output type
                             std::string>;                                       // Device name

class NmsLayerTest : public testing::WithParamInterface<NmsParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NmsParams>& obj);
    void GenerateInputs() override;
    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs)
    override;

protected:
    void SetUp() override;

private:
    void CompareBBoxes(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                       const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs);
    std::vector<TargetShapeParams> targetInDims;
    size_t inferRequestNum = 0;
    int32_t maxOutBoxesPerClass;
};

}  // namespace LayerTestsDefinitions

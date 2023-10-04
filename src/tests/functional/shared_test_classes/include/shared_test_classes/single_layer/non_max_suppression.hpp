// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "ov_models/builders.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace testing {
namespace internal {

template <>
inline void PrintTo(const ::ngraph::op::v5::NonMaxSuppression::BoxEncodingType& value, ::std::ostream* os) {}

}  // namespace internal
}  // namespace testing

namespace LayerTestsDefinitions {

using InputShapeParams = std::tuple<size_t,   // Number of batches
                                    size_t,   // Number of boxes
                                    size_t>;  // Number of classes

using InputPrecisions =
    std::tuple<InferenceEngine::Precision,   // boxes and scores precisions
               InferenceEngine::Precision,   // max_output_boxes_per_class precision
               InferenceEngine::Precision>;  // iou_threshold, score_threshold, soft_nms_sigma precisions

using NmsParams = std::tuple<InputShapeParams,  // Params using to create 1st and 2nd inputs
                             InputPrecisions,   // Input precisions
                             int32_t,           // Max output boxes per class
                             float,             // IOU threshold
                             float,             // Score threshold
                             float,             // Soft NMS sigma
                             ngraph::op::v5::NonMaxSuppression::BoxEncodingType,  // Box encoding
                             bool,                                                // Sort result descending
                             ngraph::element::Type,                               // Output type
                             std::string>;                                        // Device name

class NmsLayerTest : public testing::WithParamInterface<NmsParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NmsParams>& obj);
    void GenerateInputs() override;
    void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
                 const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs) override;

protected:
    void SetUp() override;
    InputShapeParams inShapeParams;

private:
    void CompareBBoxes(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>>& expectedOutputs,
                       const std::vector<InferenceEngine::Blob::Ptr>& actualOutputs);
};

class Nms9LayerTest : public NmsLayerTest {
protected:
    void SetUp() override;
};

}  // namespace LayerTestsDefinitions

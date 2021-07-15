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

using InputfloatVar = std::tuple<float,  // iouThreshold
                                 float,  // scoreThreshold
                                 float>; // nmsEta

using InputboolVar = std::tuple<bool,  // nmsEta
                                bool>; // normalized

using MulticlassNmsParams =
    std::tuple<InputShapeParams, // Params using to create 1st and 2nd inputs
               InputPrecisions,  // Input precisions
               int32_t,          // Max output boxes per class
               InputfloatVar,    // iouThreshold, scoreThreshold, nmsEta
               int32_t,          // background_class
               int32_t,          // keep_top_k
               ngraph::element::Type, // Output type
               ngraph::op::util::NmsBase::SortResultType, // SortResultType
               InputboolVar,       // Sort result across batch, normalized
               std::string>;

class MulticlassNmsLayerTest : public testing::WithParamInterface<MulticlassNmsParams>, virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MulticlassNmsParams> obj);
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

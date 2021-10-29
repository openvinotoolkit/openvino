// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/builders.hpp"

namespace testing {
namespace internal {

template <> inline void
PrintTo(const ::ngraph::op::v5::NonMaxSuppression::BoxEncodingType& value,
    ::std::ostream* os) { }

}
}

namespace ov {
namespace test {
namespace subgraph {

using TargetShapeParams = std::tuple<size_t,   // Number of batches
                                     size_t,   // Number of boxes
                                     size_t>;  // Number of classes

using InputShapeParams = std::tuple<std::vector<ov::Dimension>,       // bounds for input dynamic shape
                                    std::vector<TargetShapeParams>>;  // target input dimensions

using InputPrecisions = std::tuple<ElementType,  // boxes and scores precisions
                                   ElementType,  // max_output_boxes_per_class precision
                                   ElementType>; // iou_threshold, score_threshold, soft_nms_sigma precisions

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

class NmsLayerTest : public testing::WithParamInterface<NmsParams>, virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NmsParams>& obj);
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;
    void compare(const std::vector<ov::runtime::Tensor> &expected, const std::vector<ov::runtime::Tensor> &actual) override;

protected:
    void SetUp() override;

private:
    void CompareBBoxes(const std::vector<ov::runtime::Tensor> &expectedOutputs, const std::vector<ov::runtime::Tensor> &actualOutputs);
    std::vector<TargetShapeParams> targetInDims;
    size_t inferRequestNum = 0;
    int32_t maxOutBoxesPerClass;
};

} // namespace subgraph
} // namespace test
} // namespace ov

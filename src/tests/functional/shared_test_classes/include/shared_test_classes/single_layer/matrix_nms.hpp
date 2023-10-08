// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <tuple>
#include <string>

#include "ov_models/utils/ov_helpers.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

using InputPrecisions = std::tuple<ElementType,   // boxes and scores precisions
                                   ElementType,   // max_output_boxes_per_class
                                                  // precision
                                   ElementType>;  // iou_threshold, score_threshold,

using TopKParams = std::tuple<int,      // Maximum number of boxes to be selected per class
                              int>;     // Maximum number of boxes to be selected per batch element

using ThresholdParams = std::tuple<float,   // minimum score to consider box for the processing
                                   float,   // gaussian_sigma parameter for gaussian decay_function
                                   float>;  // filter out boxes with low confidence score after decaying

using NmsParams = std::tuple<std::vector<InputShape>,                            // Params using to create 1st and 2nd inputs
                             InputPrecisions,                                    // Input precisions
                             ngraph::op::v8::MatrixNms::SortResultType,          // Order of output elements
                             ngraph::element::Type,                              // Output type
                             TopKParams,                                         // Maximum number of boxes topk params
                             ThresholdParams,                                    // Thresholds: score_threshold, gaussian_sigma, post_threshold
                             int,                                                // Background class id
                             bool,                                               // If boxes are normalized
                             ngraph::op::v8::MatrixNms::DecayFunction,           // Decay function
                             bool,                                               // make output shape static
                             std::string>;                                       // Device name

class MatrixNmsLayerTest : public testing::WithParamInterface<NmsParams>,
                           virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NmsParams>& obj);
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;
    void compare(const std::vector<ov::Tensor> &expected, const std::vector<ov::Tensor> &actual) override;

protected:
    void SetUp() override;

private:
    void GetOutputParams(size_t& numBatches, size_t& maxOutputBoxesPerBatch);
    ngraph::op::v8::MatrixNms::Attributes m_attrs;
    bool m_outStaticShape;
};

} // namespace subgraph
} // namespace test
} // namespace ov

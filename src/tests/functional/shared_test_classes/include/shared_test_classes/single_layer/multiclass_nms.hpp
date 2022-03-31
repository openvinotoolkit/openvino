// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "common_test_utils/common_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
namespace subgraph {

using InputPrecisions = std::tuple<ElementType,   // input 'boxes' and 'scores' precisions
                                   ElementType,   // input 'roisnum' precision
                                   ElementType,   // max_output_boxes_per_class
                                                  // precision
                                   ElementType>;  // iou_threshold, score_threshold,
                                                  // soft_nms_sigma precisions

using InputfloatVar = std::tuple<float,   // iouThreshold
                                 float,   // scoreThreshold
                                 float>;  // nmsEta

using InputboolVar = std::tuple<bool,   // nmsEta
                                bool>;  // normalized

using MulticlassNmsParams = std::tuple<std::vector<InputShape>,                    // Params using to create inputs
                                       InputPrecisions,                            // Input precisions
                                       int32_t,                                    // Max output boxes per class
                                       InputfloatVar,                              // iouThreshold, scoreThreshold, nmsEta
                                       int32_t,                                    // background_class
                                       int32_t,                                    // keep_top_k
                                       ngraph::element::Type,                      // Output type
                                       ngraph::op::util::NmsBase::SortResultType,  // SortResultType
                                       InputboolVar,                               // Sort result across batch, normalized
                                       std::string>;

class MulticlassNmsLayerTest : public testing::WithParamInterface<MulticlassNmsParams>,
                               virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MulticlassNmsParams>& obj);
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;
    void compare(const std::vector<ov::Tensor> &expected, const std::vector<ov::Tensor> &actual) override;

protected:
    void SetUp() override;

private:
    void GetOutputParams(size_t& numBatches, size_t& maxOutputBoxesPerBatch);
    ov::op::util::MulticlassNmsBase::Attributes m_attrs;
    bool m_outStaticShape;
};
} // namespace subgraph
} // namespace test
} // namespace ov

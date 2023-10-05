// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "ov_models/utils/ov_helpers.hpp"
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
                                       ngraph::op::util::MulticlassNmsBase::SortResultType,  // SortResultType
                                       InputboolVar,                               // Sort result across batch, normalized
                                       bool,                                       // make output shape static
                                       std::string>;

class MulticlassNmsLayerTest : public testing::WithParamInterface<MulticlassNmsParams>,
                               virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MulticlassNmsParams>& obj);
    void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes) override;
    void compare(const std::vector<ov::Tensor> &expected, const std::vector<ov::Tensor> &actual) override;

protected:
    void SetUp() override;
    virtual std::shared_ptr<op::util::MulticlassNmsBase> CreateNmsOp(const OutputVector& paramOuts) const {
        std::shared_ptr<op::util::MulticlassNmsBase> nms;
        if (paramOuts.size() > 2) {
            nms = std::make_shared<ov::op::v9::MulticlassNms>(paramOuts[0], paramOuts[1], paramOuts[2], m_attrs);
        } else {
            nms = std::make_shared<ov::op::v9::MulticlassNms>(paramOuts[0], paramOuts[1], m_attrs);
        }
        return nms;
    }
    ov::op::util::MulticlassNmsBase::Attributes m_attrs;

private:
    void GetOutputParams(size_t& numBatches, size_t& maxOutputBoxesPerBatch);
    bool m_outStaticShape;
};

class MulticlassNmsLayerTest8 : public MulticlassNmsLayerTest {
protected:
    std::shared_ptr<op::util::MulticlassNmsBase> CreateNmsOp(const OutputVector& paramOuts) const override {
        return std::make_shared<ov::op::v8::MulticlassNms>(paramOuts[0], paramOuts[1], m_attrs);
    }
};
} // namespace subgraph
} // namespace test
} // namespace ov

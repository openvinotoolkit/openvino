// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using InputTypes = std::tuple<ov::element::Type,   // input 'boxes' and 'scores' types
                              ov::element::Type>;  // iou_threshold, score_threshold, soft_nms_sigma precisions

using InputfloatVar = std::tuple<float,   // iouThreshold
                                 float,   // scoreThreshold
                                 float>;  // nmsEta

using InputboolVar = std::tuple<bool,   // nmsEta
                                bool>;  // normalized

using MulticlassNmsParams = std::tuple<std::vector<InputShape>,                    // Params using to create inputs
                                       InputTypes,                                 // Input precisions
                                       int32_t,                                    // Max output boxes per class
                                       InputfloatVar,                              // iouThreshold, scoreThreshold, nmsEta
                                       int32_t,                                    // background_class
                                       int32_t,                                    // keep_top_k
                                       ov::element::Type,                          // Output type
                                       ov::op::util::MulticlassNmsBase::SortResultType,  // SortResultType
                                       InputboolVar,                               // Sort result across batch, normalized
                                       std::string>;

class MulticlassNmsLayerTest : public testing::WithParamInterface<MulticlassNmsParams>,
                               virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MulticlassNmsParams>& obj);

protected:
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

    void SetUp() override;
    bool use_op_v8 = false;
};

class MulticlassNmsLayerTest8 : public MulticlassNmsLayerTest {
protected:
    void SetUp() override;
};
} // namespace test
} // namespace ov

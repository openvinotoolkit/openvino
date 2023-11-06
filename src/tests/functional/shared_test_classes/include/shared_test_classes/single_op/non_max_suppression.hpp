// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <tuple>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
using InputShapeParams = std::tuple<size_t,   // Number of batches
                                    size_t,   // Number of boxes
                                    size_t>;  // Number of classes

using InputTypes =
    std::tuple<ov::element::Type,   // boxes and scores type
               ov::element::Type,   // max_output_boxes_per_class type
               ov::element::Type>;  // iou_threshold, score_threshold, soft_nms_sigma type

using NmsParams = std::tuple<InputShapeParams,  // Params using to create 1st and 2nd inputs
                             InputTypes,   // Input precisions
                             int32_t,           // Max output boxes per class
                             float,             // IOU threshold
                             float,             // Score threshold
                             float,             // Soft NMS sigma
                             ov::op::v5::NonMaxSuppression::BoxEncodingType,  // Box encoding
                             bool,                                                // Sort result descending
                             ov::element::Type,                               // Output type
                             std::string>;                                        // Device name

class NmsLayerTest : public testing::WithParamInterface<NmsParams>, virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NmsParams>& obj);
protected:
    void SetUp() override;
};

class Nms9LayerTest : public NmsLayerTest {
protected:
    void SetUp() override;
};

}  // namespace test
}  // namespace ov

// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using NMSTopKParams = std::tuple<ov::Shape,          // Input shape
                                 ov::element::Type,  // Input Precision
                                 int64_t,            // NMS max output boxes per class
                                 float,              // NMS iou threshold
                                 float,              // NMS score threshold
                                 std::string         // Device name
                                 >;

// NMSTopKTest is intended to cover the case where NMS node gets dynamic output shape because of
// internal dynamism, so the subsequent TopK node will also get dynamic input shape even in
// static model. For this scenario, if runtime sorting dimension size of input shape is less
// than k in TopK node, then k as well as sorting dimension size of output shape should be updated
// to be the valid object number of NMS.
// Note that in real model, plenty of nodes are used to exact location and score information from
// NMS input data based on NMS output indices, then feed to the input of TopK node. While in this
// subgraph test, TopK node is directly connected to NMS node because such usage can already assure
// the internal dynamism behavior be passed down to the TopK node without extra complexity.
class NMSTopKTest : public testing::WithParamInterface<NMSTopKParams>, virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NMSTopKParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

}  // namespace test
}  // namespace ov

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/experimental_detectron_topkrois.hpp"

#include <gtest/gtest.h>

#include "visitors/visitors.hpp"

using namespace std;
using namespace ov;
using ov::test::NodeBuilder;

TEST(attributes, experimental_detectron_topkrois_op) {
    NodeBuilder::opset().insert<ov::op::v6::ExperimentalDetectronTopKROIs>();
    size_t num_rois = 1;
    auto input_rois = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 4});
    auto input_probs = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2});
    auto topkrois = std::make_shared<op::v6::ExperimentalDetectronTopKROIs>(input_rois, input_probs, num_rois);

    NodeBuilder builder(topkrois, {input_rois, input_probs});
    auto g_topkrois = ov::as_type_ptr<ov::op::v6::ExperimentalDetectronTopKROIs>(builder.create());

    EXPECT_EQ(g_topkrois->get_max_rois(), topkrois->get_max_rois());
}

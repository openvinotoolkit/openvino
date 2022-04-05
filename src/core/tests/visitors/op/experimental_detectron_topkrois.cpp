// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset6.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, experimental_detectron_topkrois_op) {
    NodeBuilder::get_ops().register_factory<opset6::ExperimentalDetectronTopKROIs>();
    size_t num_rois = 1;
    auto input_rois = std::make_shared<op::Parameter>(element::f32, Shape{2, 4});
    auto input_probs = std::make_shared<op::Parameter>(element::f32, Shape{2});
    auto topkrois = std::make_shared<op::v6::ExperimentalDetectronTopKROIs>(input_rois, input_probs, num_rois);

    NodeBuilder builder(topkrois);
    auto g_topkrois = ov::as_type_ptr<opset6::ExperimentalDetectronTopKROIs>(builder.create());

    EXPECT_EQ(g_topkrois->get_max_rois(), topkrois->get_max_rois());
}

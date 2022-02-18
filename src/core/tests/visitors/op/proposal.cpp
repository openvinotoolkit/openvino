// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/opsets/opset4.hpp"
#include "ngraph/opsets/opset5.hpp"
#include "util/visitor.hpp"

using namespace std;
using namespace ngraph;
using ngraph::test::NodeBuilder;
using ngraph::test::ValueMap;

TEST(attributes, proposal_op) {
    NodeBuilder::get_ops().register_factory<opset1::Proposal>();
    const auto class_probs = make_shared<op::Parameter>(element::f32, Shape{1024, 2, 128, 128});
    const auto class_logits = make_shared<op::Parameter>(element::f32, Shape{1024, 4, 128, 128});
    const auto image_shape = make_shared<op::Parameter>(element::f32, Shape{4});

    op::ProposalAttrs attrs;
    attrs.base_size = 224;
    attrs.pre_nms_topn = 100;
    attrs.post_nms_topn = 110;
    attrs.nms_thresh = 0.12f;
    attrs.feat_stride = 2;
    attrs.min_size = 10;
    attrs.ratio = vector<float>{1.44f, 0.66f};
    attrs.scale = vector<float>{2.25f, 1.83f};
    attrs.clip_before_nms = true;
    attrs.clip_after_nms = true;
    attrs.normalize = false;
    attrs.box_size_scale = 2.f;
    attrs.box_coordinate_scale = 4.55f;
    attrs.framework = string{"nGraph"};

    auto proposal = make_shared<opset1::Proposal>(class_probs, class_logits, image_shape, attrs);
    NodeBuilder builder(proposal);
    auto g_proposal = ov::as_type_ptr<opset1::Proposal>(builder.create());

    const auto proposal_attrs = proposal->get_attrs();
    const auto g_proposal_attrs = g_proposal->get_attrs();

    EXPECT_EQ(g_proposal_attrs.base_size, proposal_attrs.base_size);
    EXPECT_EQ(g_proposal_attrs.pre_nms_topn, proposal_attrs.pre_nms_topn);
    EXPECT_EQ(g_proposal_attrs.post_nms_topn, proposal_attrs.post_nms_topn);
    EXPECT_EQ(g_proposal_attrs.nms_thresh, proposal_attrs.nms_thresh);
    EXPECT_EQ(g_proposal_attrs.feat_stride, proposal_attrs.feat_stride);
    EXPECT_EQ(g_proposal_attrs.min_size, proposal_attrs.min_size);
    EXPECT_EQ(g_proposal_attrs.ratio, proposal_attrs.ratio);
    EXPECT_EQ(g_proposal_attrs.scale, proposal_attrs.scale);
    EXPECT_EQ(g_proposal_attrs.clip_before_nms, proposal_attrs.clip_before_nms);
    EXPECT_EQ(g_proposal_attrs.clip_after_nms, proposal_attrs.clip_after_nms);
    EXPECT_EQ(g_proposal_attrs.normalize, proposal_attrs.normalize);
    EXPECT_EQ(g_proposal_attrs.box_size_scale, proposal_attrs.box_size_scale);
    EXPECT_EQ(g_proposal_attrs.box_coordinate_scale, proposal_attrs.box_coordinate_scale);
    EXPECT_EQ(g_proposal_attrs.framework, proposal_attrs.framework);
}

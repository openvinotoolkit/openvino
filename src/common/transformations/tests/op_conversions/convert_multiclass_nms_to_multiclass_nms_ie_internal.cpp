// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/opsets/opset9_decl.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/multiclass_nms_ie_internal.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ConvertMulticlassNmsToMulticlassNmsIE) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});

        auto nms = std::make_shared<opset9::MulticlassNms>(boxes, scores, opset9::MulticlassNms::Attributes());

        model = std::make_shared<Model>(OutputVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertMulticlassNmsToMulticlassNmsIE>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<ov::op::internal::MulticlassNmsIEInternal>(boxes,
                                                                               scores,
                                                                               opset9::MulticlassNms::Attributes());

        model_ref = std::make_shared<Model>(OutputVector{nms}, ParameterVector{boxes, scores});
    }
}

// Regression test: dynamic-shape inputs must still be converted, and the resulting
// internal op's output shape must stay dynamic instead of being collapsed to a static
// upper-bound shape (see get_max_shape() usage removed in validate_and_infer_types()).
TEST(TransformationTests, ConvertMulticlassNmsToMulticlassNmsIE_DynamicShape) {
    auto boxes = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, DYN, 4});
    auto scores = std::make_shared<opset1::Parameter>(element::f32, PartialShape{DYN, 1, DYN});

    auto nms = std::make_shared<opset9::MulticlassNms>(boxes, scores, opset9::MulticlassNms::Attributes());
    auto model = std::make_shared<Model>(OutputVector{nms->output(0), nms->output(1), nms->output(2)},
                                         ParameterVector{boxes, scores});

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::pass::ConvertMulticlassNmsToMulticlassNmsIE>();
    OV_ASSERT_NO_THROW(manager.run_passes(model));

    auto internal_nms = ov::as_type_ptr<ov::op::internal::MulticlassNmsIEInternal>(
        model->get_results()[0]->get_input_node_shared_ptr(0));
    ASSERT_NE(internal_nms, nullptr);

    for (size_t i = 0; i < internal_nms->get_output_size(); ++i) {
        const auto& out_pshape = internal_nms->get_output_partial_shape(i);
        ASSERT_TRUE(out_pshape.rank().is_static());
        ASSERT_TRUE(out_pshape[0].is_dynamic()) << "output " << i << " is unexpectedly static: " << out_pshape;
    }
}

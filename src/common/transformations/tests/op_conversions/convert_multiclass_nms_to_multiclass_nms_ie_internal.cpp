// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset9.hpp"
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

        model = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ov::pass::ConvertMulticlassNmsToMulticlassNmsIE>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<ov::op::internal::MulticlassNmsIEInternal>(boxes,
                                                                               scores,
                                                                               opset9::MulticlassNms::Attributes());

        model_ref = std::make_shared<Model>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

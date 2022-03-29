// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <transformations/op_conversions/convert_multiclass_nms_to_multiclass_nms_ie.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph_ops/multiclass_nms_ie_internal.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, ConvertMulticlassNmsToMulticlassNmsIE) {
    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});

        auto nms = std::make_shared<opset9::MulticlassNms>(boxes, scores, opset9::MulticlassNms::Attributes());

        function = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});

        manager.register_pass<ngraph::pass::ConvertMulticlassNmsToMulticlassNmsIE>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }

    {
        auto boxes = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1000, 4});
        auto scores = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 1, 1000});
        auto nms = std::make_shared<op::internal::MulticlassNmsIEInternal>(boxes, scores, opset9::MulticlassNms::Attributes());

        function_ref = std::make_shared<Function>(NodeVector{nms}, ParameterVector{boxes, scores});
    }
}

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/symbolic_transformations/reshape_optimizations.hpp"

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/dimension_tracker.hpp"

using namespace ov;
using namespace ov::op;
using namespace std;

namespace {
    void label_shape(ov::PartialShape& shape) {
        auto table = std::make_shared<ov::TableOfEquivalence>(42);
        auto tracker = ov::DimensionTracker(table);
        tracker.set_up_for_tracking(shape);
    }
}  // namespace

TEST_F(TransformationTestsF, FlattenOptimization) {
    {
        auto shape = PartialShape::dynamic(4);
        label_shape(shape);  // we label shape with consecutive labels: 42, 43, 44, 45

        auto data = make_shared<v0::Parameter>(element::f32, shape);

        auto shape_of = make_shared<v3::ShapeOf>(data);
        auto indices = ov::op::v0::Constant::create(element::i64, {3}, {0, 1});
        auto axis = ov::op::v0::Constant::create(element::i64, {3}, {0, 1});

        auto as_is_dims = make_shared<v1::Gather>(shape_of, indices, axis);

        auto merged_dim = make_shared<v1::Multiply>(
                make_shared<v1::Gather>(shape_of, indices, axis),
                make_shared<v1::Gather>(shape_of, indices, axis));

        auto pattern = make_shared<v0::Concat>(OutputVector{as_is_dims, merged_dim}, 0);

        auto reshape = make_shared<v1::Reshape>(data, pattern, false);

        model = make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
        manager.register_pass<pass::ReshapeOptimizations>();
    }
    {
        auto shape = PartialShape::dynamic(4);
        label_shape(shape);  // we label shape with consecutive labels: 42, 43, 44, 45

        auto data = make_shared<v0::Parameter>(element::f32, shape);
        auto pattern = ov::op::v0::Constant::create(element::i64, {3}, {0, 0, -1});

        auto reshape = make_shared<v1::Reshape>(data, pattern, true);

        model_ref = make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}

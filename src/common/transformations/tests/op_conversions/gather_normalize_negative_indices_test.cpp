// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/validation_util.hpp>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset7.hpp>
#include <openvino/pass/manager.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/op_conversions/gather_normalize_negative_indices.hpp>

#include "common_test_utils/ov_test_utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize) {
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 15, 128});
        auto indices = opset7::Constant::create(element::i32, Shape{}, {-1});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {1});

        auto gather = std::make_shared<opset7::Gather>(data, indices, axis, 0);

        model = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});

        manager.register_pass<ov::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto indices_type = element::i32;

        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 15, 128});
        auto indices = opset7::Constant::create(indices_type, Shape{}, {-1});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {1});

        auto shape_of = std::make_shared<opset7::ShapeOf>(data, indices_type);
        auto input_gather = std::make_shared<opset7::Gather>(shape_of,
                                                             opset7::Constant::create(indices_type, Shape{}, {1}),
                                                             opset7::Constant::create(indices_type, Shape{}, {0}));
        auto add = std::make_shared<opset7::Add>(input_gather, indices);
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto const_add = get_constant_from_source(add);
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (const_add == nullptr)
            OPENVINO_THROW("indices should've been constant folded");
        auto gather = std::make_shared<opset7::Gather>(data, const_add, axis);

        model_ref = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_neg_axis) {
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 15, 128});
        auto indices = opset7::Constant::create(element::i32, Shape{}, {-1});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {-2});

        auto gather = std::make_shared<opset7::Gather>(data, indices, axis, 0);

        model = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});

        manager.register_pass<ov::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto indices_type = element::i32;

        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 15, 128});
        auto indices = opset7::Constant::create(indices_type, Shape{}, {-1});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {-2});

        auto shape_of = std::make_shared<opset7::ShapeOf>(data, indices_type);
        auto input_gather = std::make_shared<opset7::Gather>(shape_of,
                                                             opset7::Constant::create(indices_type, Shape{}, {1}),
                                                             opset7::Constant::create(indices_type, Shape{}, {0}));
        auto add = std::make_shared<opset7::Add>(input_gather, indices);
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto const_add = get_constant_from_source(add);
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (const_add == nullptr)
            OPENVINO_THROW("indices should've been constant folded");
        auto gather = std::make_shared<opset7::Gather>(data, const_add, axis);

        model_ref = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_dif_input_types) {
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 15, 128});
        auto indices = opset7::Constant::create(element::i32, Shape{}, {-1});
        auto axis = opset7::Constant::create(element::i64, Shape{}, {1});

        auto gather = std::make_shared<opset7::Gather>(data, indices, axis, 0);

        model = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});

        manager.register_pass<ov::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto indices_type = element::i32;

        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{1, 15, 128});
        auto indices = opset7::Constant::create(indices_type, Shape{}, {-1});
        auto axis = opset7::Constant::create(element::i64, Shape{}, {1});

        auto shape_of = std::make_shared<opset7::ShapeOf>(data, indices_type);
        auto input_gather = std::make_shared<opset7::Gather>(shape_of,
                                                             opset7::Constant::create(indices_type, Shape{}, {1}),
                                                             opset7::Constant::create(indices_type, Shape{}, {0}));
        auto add = std::make_shared<opset7::Add>(input_gather, indices);
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto const_add = get_constant_from_source(add);
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (const_add == nullptr)
            OPENVINO_THROW("indices should've been constant folded");
        auto gather = std::make_shared<opset7::Gather>(data, const_add, axis);

        model_ref = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_static_axis_dim) {
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, PartialShape{DYN, 15, DYN});
        auto indices = opset7::Constant::create(element::i32, Shape{}, {-1});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {1});

        auto gather = std::make_shared<opset7::Gather>(data, indices, axis, 0);

        model = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});

        manager.register_pass<ov::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto indices_type = element::i32;

        auto data = std::make_shared<opset7::Parameter>(element::f32, PartialShape{DYN, 15, DYN});
        auto indices = opset7::Constant::create(indices_type, Shape{}, {2});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {1});

        auto gather = std::make_shared<opset7::Gather>(data, indices, axis);
        model_ref = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_static_axis_dim_neg_axis) {
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, PartialShape{DYN, 15, DYN});
        auto indices = opset7::Constant::create(element::i32, Shape{}, {-1});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {-2});

        auto gather = std::make_shared<opset7::Gather>(data, indices, axis, 0);

        model = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});

        manager.register_pass<ov::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto indices_type = element::i32;

        auto data = std::make_shared<opset7::Parameter>(element::f32, PartialShape{DYN, 15, DYN});
        auto indices = opset7::Constant::create(indices_type, Shape{}, {2});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {-2});

        auto gather = std::make_shared<opset7::Gather>(data, indices, axis);

        model_ref = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_non_static_axis_dim) {
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, PartialShape{DYN, DYN, DYN});
        auto indices = opset7::Constant::create(element::i32, Shape{}, {-1});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {1});

        auto gather = std::make_shared<opset7::Gather>(data, indices, axis, 0);

        model = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});

        manager.register_pass<ov::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto indices_type = element::i32;

        auto data = std::make_shared<opset7::Parameter>(element::f32, PartialShape{DYN, DYN, DYN});
        auto indices = opset7::Constant::create(indices_type, Shape{}, {-1});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {1});

        auto gather = std::make_shared<opset7::Gather>(data, indices, axis);

        model_ref = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_positive_ind) {
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{2, 3});
        auto indices = opset7::Constant::create(element::i32, Shape{}, {1});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {0});

        auto gather = std::make_shared<opset7::Gather>(data, indices, axis, 0);

        model = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});

        manager.register_pass<ov::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, Shape{2, 3});
        auto indices = opset7::Constant::create(element::i32, Shape{}, {1});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {0});

        auto gather = std::make_shared<opset7::Gather>(data, indices, axis);

        model_ref = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_non_static_rank) {
    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
        auto indices = opset7::Constant::create(element::i32, Shape{}, {-1});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {0});

        auto gather = std::make_shared<opset7::Gather>(data, indices, axis, 0);

        model = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});

        manager.register_pass<ov::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto data = std::make_shared<opset7::Parameter>(element::f32, PartialShape::dynamic());
        auto indices = opset7::Constant::create(element::i32, Shape{}, {-1});
        auto axis = opset7::Constant::create(element::i32, Shape{}, {0});

        auto gather = std::make_shared<opset7::Gather>(data, indices, axis);

        model_ref = std::make_shared<ov::Model>(NodeVector{gather}, ParameterVector{data});
    }
}

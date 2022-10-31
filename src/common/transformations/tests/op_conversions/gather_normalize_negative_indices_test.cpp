// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/validation_util.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/gather_normalize_negative_indices.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize) {
    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 15, 128});
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {1});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, 0);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto indices_type = ngraph::element::i32;

        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 15, 128});
        auto indices = ngraph::opset7::Constant::create(indices_type, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {1});

        auto shape_of = std::make_shared<ngraph::opset7::ShapeOf>(data, indices_type);
        auto input_gather = std::make_shared<ngraph::opset7::Gather>(shape_of,
            ngraph::opset7::Constant::create(indices_type, ngraph::Shape{}, {1}), ngraph::opset7::Constant::create(indices_type, ngraph::Shape{}, {0}));
        auto add = std::make_shared<ngraph::opset7::Add>(input_gather, indices);
        auto const_add = ngraph::get_constant_from_source(add);
        if (const_add == nullptr)
            throw ngraph::ngraph_error("indices should've been constant folded");
        auto gather = std::make_shared<ngraph::opset7::Gather>(data, const_add, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_neg_axis) {
    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 15, 128});
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-2});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, 0);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto indices_type = ngraph::element::i32;

        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 15, 128});
        auto indices = ngraph::opset7::Constant::create(indices_type, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-2});

        auto shape_of = std::make_shared<ngraph::opset7::ShapeOf>(data, indices_type);
        auto input_gather = std::make_shared<ngraph::opset7::Gather>(shape_of,
             ngraph::opset7::Constant::create(indices_type, ngraph::Shape{}, {1}), ngraph::opset7::Constant::create(indices_type, ngraph::Shape{}, {0}));
        auto add = std::make_shared<ngraph::opset7::Add>(input_gather, indices);
        auto const_add = ngraph::get_constant_from_source(add);
        if (const_add == nullptr)
            throw ngraph::ngraph_error("indices should've been constant folded");
        auto gather = std::make_shared<ngraph::opset7::Gather>(data, const_add, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_dif_input_types) {
    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 15, 128});
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, 0);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto indices_type = ngraph::element::i32;

        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 15, 128});
        auto indices = ngraph::opset7::Constant::create(indices_type, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});

        auto shape_of = std::make_shared<ngraph::opset7::ShapeOf>(data, indices_type);
        auto input_gather = std::make_shared<ngraph::opset7::Gather>(shape_of,
            ngraph::opset7::Constant::create(indices_type, ngraph::Shape{}, {1}), ngraph::opset7::Constant::create(indices_type, ngraph::Shape{}, {0}));
        auto add = std::make_shared<ngraph::opset7::Add>(input_gather, indices);
        auto const_add = ngraph::get_constant_from_source(add);
        if (const_add == nullptr)
            throw ngraph::ngraph_error("indices should've been constant folded");
        auto gather = std::make_shared<ngraph::opset7::Gather>(data, const_add, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_static_axis_dim) {
    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::PartialShape{DYN, 15, DYN});
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {1});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, 0);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto indices_type = ngraph::element::i32;

        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::PartialShape{DYN, 15, DYN});
        auto indices = ngraph::opset7::Constant::create(indices_type, ngraph::Shape{}, {2});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {1});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis);
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_static_axis_dim_neg_axis) {
    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::PartialShape{DYN, 15, DYN});
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-2});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, 0);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto indices_type = ngraph::element::i32;

        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::PartialShape{DYN, 15, DYN});
        auto indices = ngraph::opset7::Constant::create(indices_type, ngraph::Shape{}, {2});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-2});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_non_static_axis_dim) {
    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::PartialShape{DYN, DYN, DYN});
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {1});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, 0);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto indices_type = ngraph::element::i32;

        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::PartialShape{DYN, DYN, DYN});
        auto indices = ngraph::opset7::Constant::create(indices_type, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {1});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_positive_ind) {
    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, 0);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, GatherNegativeIndicesNormalize_non_static_rank) {
    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(ngraph::Rank::dynamic()));
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, 0);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::GatherNegativeConstIndicesNormalize>();
    }

    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph_transformations/reshape_1d_ops.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include <ngraph/pass/manager.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace MKLDNNPlugin;

std::shared_ptr<ngraph::Node> get_reshape_before(const ngraph::Output<ngraph::Node>& data) {
    auto shape_of = std::make_shared<ngraph::opset1::ShapeOf>(data);

    auto gather_first = std::make_shared<ngraph::opset1::Gather>(shape_of,
        ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { 0, 1 }),
        ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 0 }));

    auto unsqueezed_dimension = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 });

    auto gather_last = std::make_shared<ngraph::opset1::Gather>(shape_of,
        ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 2 }),
        ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 0 }));

    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::NodeVector{ gather_first, unsqueezed_dimension, gather_last }, 0);
    return std::make_shared<ngraph::opset1::Reshape>(data, concat, true);
}

std::shared_ptr<ngraph::Node> get_reshape_after(const ngraph::Output<ngraph::Node>& data) {
    auto constant = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 3 }, { 0, 0, -1 });
    return std::make_shared<ngraph::opset1::Reshape>(data, constant, true);
}

TEST(TransformationTests, Reshape1DAvgPoolTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16 });
        auto avgPool = std::make_shared<ngraph::opset1::AvgPool>(
            input,
            ngraph::Strides{ 1 },
            ngraph::Shape{ 1 },
            ngraph::Shape{ 0 },
            ngraph::Shape{ 2 },
            true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ avgPool }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<Reshape1DAvgPool>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16 });
        auto reshape_before = ngraph::op::util::reshapeTo(input, { 1, 3, 1, 16 });
        auto avg_pool = std::make_shared<ngraph::opset1::AvgPool>(
            reshape_before,
            ngraph::Strides{ 1, 1 },
            ngraph::Shape{ 0, 1 },
            ngraph::Shape{ 0, 0 },
            ngraph::Shape{ 1, 2 },
            true);
        auto reshape_after = ngraph::op::util::reshapeTo(avg_pool, { 1, 3, 16 });

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, Reshape1DAvgPoolTest2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));
        auto avgPool = std::make_shared<ngraph::opset1::AvgPool>(
            input,
            ngraph::Strides{ 1 },
            ngraph::Shape{ 1 },
            ngraph::Shape{ 0 },
            ngraph::Shape{ 2 },
            true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ avgPool }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<Reshape1DAvgPool>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));
        auto reshape_before = get_reshape_before(input);
        auto avg_pool = std::make_shared<ngraph::opset1::AvgPool>(
            reshape_before,
            ngraph::Strides{ 1, 1 },
            ngraph::Shape{ 0, 1 },
            ngraph::Shape{ 0, 0 },
            ngraph::Shape{ 1, 2 },
            true);
        auto reshape_after = get_reshape_after(avg_pool);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, Reshape1DAvgPoolTest3) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
        auto avgPool = std::make_shared<ngraph::opset1::AvgPool>(
            input,
            ngraph::Strides{ 1, 1 },
            ngraph::Shape{ 1, 1 },
            ngraph::Shape{ 0, 0 },
            ngraph::Shape{ 2, 2 },
            true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ avgPool }, ngraph::ParameterVector{ input });
        f_ref = f;

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<Reshape1DAvgPool>();
        m.run_passes(f);
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, Reshape1DMaxPoolTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16 });
        auto max_pool = std::make_shared<ngraph::opset1::MaxPool>(
            input,
            ngraph::Strides{ 1 },
            ngraph::Shape{ 1 },
            ngraph::Shape{ 0 },
            ngraph::Shape{ 2 });

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ max_pool }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<Reshape1DMaxPool>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16 });
        auto reshape_before = ngraph::op::util::reshapeTo(input, { 1, 3, 1, 16 });
        auto max_pool = std::make_shared<ngraph::opset1::MaxPool>(
            reshape_before,
            ngraph::Strides{ 1, 1 },
            ngraph::Shape{ 0, 1 },
            ngraph::Shape{ 0, 0 },
            ngraph::Shape{ 1, 2 });
        auto reshape_after = ngraph::op::util::reshapeTo(max_pool, { 1, 3, 16 });

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, Reshape1DMaxPoolTest2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));
        auto max_pool = std::make_shared<ngraph::opset1::MaxPool>(
            input,
            ngraph::Strides{ 1 },
            ngraph::Shape{ 1 },
            ngraph::Shape{ 0 },
            ngraph::Shape{ 2 });

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ max_pool }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<Reshape1DMaxPool>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));
        auto reshape_before = get_reshape_before(input);
        auto max_pool = std::make_shared<ngraph::opset1::MaxPool>(
            reshape_before,
            ngraph::Strides{ 1, 1 },
            ngraph::Shape{ 0, 1 },
            ngraph::Shape{ 0, 0 },
            ngraph::Shape{ 1, 2 });
        auto reshape_after = get_reshape_after(max_pool);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, Reshape1DMaxPoolTest3) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto max_pool = std::make_shared<ngraph::opset1::MaxPool>(
            input,
            ngraph::Strides{ 1 },
            ngraph::Shape{ 1 },
            ngraph::Shape{ 0 },
            ngraph::Shape{ 2 });

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ max_pool }, ngraph::ParameterVector{ input });
        f_ref = f;

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<Reshape1DMaxPool>();
        m.run_passes(f);
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, Reshape1DConvolutionTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16 });
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 6, 3, 3 }, { 2.f });
        auto convolution = std::make_shared<ngraph::opset1::Convolution>(
            input,
            weights,
            ngraph::Strides{ 1 },
            ngraph::CoordinateDiff{ 0 },
            ngraph::CoordinateDiff{ 0 },
            ngraph::Strides{ 1 });

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ convolution }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<Reshape1DConvolution>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16 });
        auto reshape_data_before = ngraph::op::util::reshapeTo(input, { 1, 3, 1, 16 });

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 6, 3, 1, 3 }, { 2.f });
        auto convolution = std::make_shared<ngraph::opset1::Convolution>(
            reshape_data_before,
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
        auto reshape_after = ngraph::op::util::reshapeTo(convolution, { 1, 6, 14 });

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, Reshape1DConvolutionTest2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16 });
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 6, 3, 3 }, { 2.f });
        auto convolution = std::make_shared<ngraph::opset1::Convolution>(
            input,
            weights,
            ngraph::Strides{ 1 },
            ngraph::CoordinateDiff{ 0 },
            ngraph::CoordinateDiff{ 0 },
            ngraph::Strides{ 1 });

        auto bias_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 6, 1 }, { 24.f });
        auto bias = std::make_shared<ngraph::opset1::Add>(convolution, bias_const);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ bias }, ngraph::ParameterVector{ input });

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<Reshape1DConvolution>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16 });
        auto reshape_data_before = ngraph::op::util::reshapeTo(input, { 1, 3, 1, 16 });

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 6, 3, 1, 3 }, { 2.f });
        auto convolution = std::make_shared<ngraph::opset1::Convolution>(
            reshape_data_before,
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });

        auto bias_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 6, 1, 1 }, { 24.f });
        auto bias = std::make_shared<ngraph::opset1::Add>(convolution, bias_const);
        auto reshape_after = ngraph::op::util::reshapeTo(bias, { 1, 6, 14 });

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, Reshape1DConvolutionTest3) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::u8, ngraph::Shape{ 1, 3, 16 });
        auto weights = ngraph::opset1::Constant::create(ngraph::element::i8, ngraph::Shape{ 6, 3, 3 }, { 2.f });

        auto relaxed_convolution = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
            ngraph::element::TypeVector{ngraph::element::f32, ngraph::element::f32},
            ngraph::element::TypeVector{ngraph::element::f32},
            ngraph::op::TemporaryReplaceOutputType(input, ngraph::element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(weights, ngraph::element::f32).get(),
            ngraph::Strides{ 1 },
            ngraph::CoordinateDiff{ 0 },
            ngraph::CoordinateDiff{ 0 },
            ngraph::Strides{ 1 });

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ relaxed_convolution }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<Reshape1DConvolution>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::u8, ngraph::Shape{ 1, 3, 16 });
        auto reshape_data_before = ngraph::op::util::reshapeTo(input, { 1, 3, 1, 16 });

        auto weights = ngraph::opset1::Constant::create(ngraph::element::i8, ngraph::Shape{ 6, 3, 1, 3 }, { 2.f });
        auto relaxed_convolution = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Convolution>>(
            ngraph::element::TypeVector{ ngraph::element::f32, ngraph::element::f32 },
            ngraph::element::TypeVector{ ngraph::element::f32 },
            ngraph::op::TemporaryReplaceOutputType(reshape_data_before, ngraph::element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(weights, ngraph::element::f32).get(),
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });

        auto reshape_after = ngraph::op::util::reshapeTo(relaxed_convolution, { 1, 6, 14 });

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, Reshape1DConvolutionTest4) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 6, 3, 3 }, { 2.f });
        auto convolution = std::make_shared<ngraph::opset1::Convolution>(
            input,
            weights,
            ngraph::Strides{ 1 },
            ngraph::CoordinateDiff{ 0 },
            ngraph::CoordinateDiff{ 0 },
            ngraph::Strides{ 1 });

        auto bias_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 6, 1 }, { 24.f });
        auto bias = std::make_shared<ngraph::opset1::Add>(convolution, bias_const);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ bias }, ngraph::ParameterVector{ input });

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<Reshape1DConvolution>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));
        auto reshape_data_before = get_reshape_before(input);

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 6, 3, 1, 3 }, { 2.f });
        auto convolution = std::make_shared<ngraph::opset1::Convolution>(
            reshape_data_before,
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });

        auto bias_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 6, 1, 1 }, { 24.f });
        auto bias = std::make_shared<ngraph::opset1::Add>(convolution, bias_const);
        auto reshape_after = get_reshape_after(bias);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, Reshape1DGroupConvolutionTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16 });
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 1, 3 }, { 2.f });
        auto group_convolution = std::make_shared<ngraph::opset1::GroupConvolution>(
            input,
            weights,
            ngraph::Strides{ 1 },
            ngraph::CoordinateDiff{ 0 },
            ngraph::CoordinateDiff{ 0 },
            ngraph::Strides{ 1 });

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ group_convolution }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<Reshape1DGroupConvolution>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16 });
        auto reshape_data_before = ngraph::op::util::reshapeTo(input, { 1, 3, 1, 16 });

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 1, 1, 3 }, { 2.f });
        auto group_convolution = std::make_shared<ngraph::opset1::GroupConvolution>(
            reshape_data_before,
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
        auto reshape_after = ngraph::op::util::reshapeTo(group_convolution, { 1, 3, 14 });

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, Reshape1DGroupConvolutionTest2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 1, 3 }, { 2.f });
        auto group_convolution = std::make_shared<ngraph::opset1::GroupConvolution>(
            input,
            weights,
            ngraph::Strides{ 1 },
            ngraph::CoordinateDiff{ 0 },
            ngraph::CoordinateDiff{ 0 },
            ngraph::Strides{ 1 });

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ group_convolution }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<Reshape1DGroupConvolution>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));
        auto reshape_data_before = get_reshape_before(input);

        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 1, 1, 1, 3 }, { 2.f });
        auto group_convolution = std::make_shared<ngraph::opset1::GroupConvolution>(
            reshape_data_before,
            weights,
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });
        auto reshape_after = get_reshape_after(group_convolution);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ reshape_after }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

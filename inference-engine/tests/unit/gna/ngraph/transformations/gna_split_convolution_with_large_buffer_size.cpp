// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/split_convolution_with_large_buffer_size.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

namespace testing {
namespace {

// use constexpr uint32_t bufferMaxSize = 65528 from gna_limitations

struct SubGraph
{
    std::shared_ptr<ngraph::opset7::Parameter> input_node;
    ngraph::OutputVector output_nodes;
};

SubGraph createSubGraphSolid(const ngraph::Shape& input_shape, const ngraph::Shape& kernel_shape)
{
    SubGraph subGraph;

    subGraph.input_node = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                      input_shape);
    auto kernel = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                   kernel_shape, {1});

    auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(subGraph.input_node,
                                                                               kernel,
                                                                               ngraph::Strides{1, 1},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::CoordinateDiff{0, 0},
                                                                               ngraph::Strides{1, 1});
    subGraph.output_nodes.push_back(convolution_operation);

    return subGraph;
}

SubGraph createSubGraphSplitted()
{
    SubGraph subGraph;

    subGraph.input_node = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                      ngraph::Shape{1, 64, 4096, 4096});

    auto split_node_c1 = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({1}), std::vector<int64_t>{3});
    auto split_node_c2 = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape({5}), {960, 960, 960, 960, 256});
    auto split_node = std::make_shared<ngraph::opset7::VariadicSplit>(subGraph.input_node,
                                                                      split_node_c1,
                                                                      split_node_c2);
    
    auto kernel = ngraph::opset1::Constant::create(ngraph::element::f32,
                                                   ngraph::Shape{1, 64, 1, 1}, {1});

    for (int i = 0; i < 5; ++i) {
        auto convolution_operation = std::make_shared<ngraph::opset7::Convolution>(split_node->output(i),
                                                                                   kernel,
                                                                                   ngraph::Strides{1, 1},
                                                                                   ngraph::CoordinateDiff{0, 0},
                                                                                   ngraph::CoordinateDiff{0, 0},
                                                                                   ngraph::Strides{1, 1});
        subGraph.output_nodes.push_back(convolution_operation);
    }

    return subGraph;
}

/*
  CreateNodeFunctionT
    arg: input node
    return: new created node
 */
template <typename CreateNodeFunctionT>
void appendSubGraph(SubGraph& subGraph, CreateNodeFunctionT create_node_func)
{
    ngraph::OutputVector new_graph_output;
    for (auto& node: subGraph.output_nodes)
        new_graph_output.push_back(create_node_func(node));
    
    subGraph.output_nodes.swap(new_graph_output);
}

void appendSubGraphAddNode(SubGraph& subGraph)
{
    auto append_func = [] (const ngraph::Output<ngraph::Node>& input_node)
    {
        auto bias = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        return std::make_shared<ngraph::opset7::Add>(input_node, bias);
    };
    appendSubGraph(subGraph, append_func);
}

// ---------------------------------------------------------------------------------------------------------------------

TEST(TransformationTests, SplitConvolutionTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        SubGraph subGraph = createSubGraphSolid(ngraph::Shape{1, 64, 4096, 4096}, ngraph::Shape{1, 64, 1, 1});
        auto result = std::make_shared<ngraph::opset7::Result>(subGraph.output_nodes.front());
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{subGraph.input_node});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolution>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        SubGraph subGraph = createSubGraphSplitted();
        auto concat = std::make_shared<ngraph::opset7::Concat>(subGraph.output_nodes, 3);
        auto result = std::make_shared<ngraph::opset7::Result>(concat);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{subGraph.input_node});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SplitConvolutionTestSmallSize) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        SubGraph subGraph = createSubGraphSolid(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1});
        auto result = std::make_shared<ngraph::opset7::Result>(subGraph.output_nodes.front());
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{subGraph.input_node});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolution>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        SubGraph subGraph = createSubGraphSolid(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1});
        auto result = std::make_shared<ngraph::opset7::Result>(subGraph.output_nodes.front());
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{subGraph.input_node});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SplitConvolutionWithBiasTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        SubGraph subGraph = createSubGraphSolid(ngraph::Shape{1, 64, 4096, 4096}, ngraph::Shape{1, 64, 1, 1});
        appendSubGraphAddNode(subGraph);

        auto result = std::make_shared<ngraph::opset7::Result>(subGraph.output_nodes.front());
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{subGraph.input_node});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolutionWithBias>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        SubGraph subGraph = createSubGraphSplitted();
        appendSubGraphAddNode(subGraph);
        auto concat = std::make_shared<ngraph::opset7::Concat>(subGraph.output_nodes, 3);
        auto result = std::make_shared<ngraph::opset7::Result>(concat);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{subGraph.input_node});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SplitConvolutionWithBiasTestSmallSize) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        SubGraph subGraph = createSubGraphSolid(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1});
        appendSubGraphAddNode(subGraph);
        
        auto result = std::make_shared<ngraph::opset7::Result>(subGraph.output_nodes.front());
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{subGraph.input_node});
        
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolutionWithBias>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        SubGraph subGraph = createSubGraphSolid(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1});
        appendSubGraphAddNode(subGraph);

        auto result = std::make_shared<ngraph::opset7::Result>(subGraph.output_nodes.front());
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{subGraph.input_node});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

// Variant Convolution -> FakeQuantize

TEST(TransformationTests, SplitConvolutionWithFqTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        SubGraph subGraph = createSubGraphSolid(ngraph::Shape{1, 64, 4096, 4096}, ngraph::Shape{1, 64, 1, 1});
        auto append_func = [] (const ngraph::Output<ngraph::Node>& input_node)
        {
            auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
            auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
            auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
            auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
            return std::make_shared<ngraph::opset7::FakeQuantize>(input_node, input_low,
                                                                  input_high, output_low,
                                                                  output_high, 11);
        };
        appendSubGraph(subGraph, append_func);

        auto result = std::make_shared<ngraph::opset7::Result>(subGraph.output_nodes.front());
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{subGraph.input_node});
       
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolutionWithFq>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        SubGraph subGraph = createSubGraphSplitted();
        auto append_func = [] (const ngraph::Output<ngraph::Node>& input_node)
        {
            auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
            auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
            auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
            auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
            return std::make_shared<ngraph::opset7::FakeQuantize>(input_node, input_low,
                                                                  input_high, output_low,
                                                                  output_high, 11);
        };
        appendSubGraph(subGraph, append_func);
        auto concat = std::make_shared<ngraph::opset7::Concat>(subGraph.output_nodes, 3);
        
        auto result = std::make_shared<ngraph::opset7::Result>(concat);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{subGraph.input_node});
    }
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SplitConvolutionWithFqTestSmallSize) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        SubGraph subGraph = createSubGraphSolid(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1});
        auto append_func = [] (const ngraph::Output<ngraph::Node>& input_node)
        {
            auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
            auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
            auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
            auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
            return std::make_shared<ngraph::opset7::FakeQuantize>(input_node, input_low,
                                                                  input_high, output_low,
                                                                  output_high, 11);
        };
        appendSubGraph(subGraph, append_func);

        auto result = std::make_shared<ngraph::opset7::Result>(subGraph.output_nodes.front());
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{subGraph.input_node});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolutionWithFq>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        SubGraph subGraph = createSubGraphSolid(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1});
        auto append_func = [] (const ngraph::Output<ngraph::Node>& input_node)
        {
            auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
            auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
            auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
            auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
            return std::make_shared<ngraph::opset7::FakeQuantize>(input_node, input_low,
                                                                  input_high, output_low,
                                                                  output_high, 11);
        };
        appendSubGraph(subGraph, append_func);

        auto result = std::make_shared<ngraph::opset7::Result>(subGraph.output_nodes.front());
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{subGraph.input_node});
    }
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

// Variant Convolution -> Add -> FakeQuantize

TEST(TransformationTests, SplitConvolutionWithFqAddTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        SubGraph subGraph = createSubGraphSolid(ngraph::Shape{1, 64, 4096, 4096}, ngraph::Shape{1, 64, 1, 1});
        appendSubGraphAddNode(subGraph);
        auto append_func = [] (const ngraph::Output<ngraph::Node>& input_node)
        {
            auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
            auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
            auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
            auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
            return std::make_shared<ngraph::opset7::FakeQuantize>(input_node, input_low,
                                                                  input_high, output_low,
                                                                  output_high, 11);
        };
        appendSubGraph(subGraph, append_func);

        auto result = std::make_shared<ngraph::opset7::Result>(subGraph.output_nodes.front());
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{subGraph.input_node});


        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolutionWithFq>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        SubGraph subGraph = createSubGraphSplitted();
        appendSubGraphAddNode(subGraph);
        auto append_func = [] (const ngraph::Output<ngraph::Node>& input_node)
        {
            auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
            auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
            auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
            auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
            return std::make_shared<ngraph::opset7::FakeQuantize>(input_node, input_low,
                                                                  input_high, output_low,
                                                                  output_high, 11);
        };
        appendSubGraph(subGraph, append_func);
        auto concat = std::make_shared<ngraph::opset7::Concat>(subGraph.output_nodes, 3);
        
        auto result = std::make_shared<ngraph::opset7::Result>(concat);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                            ngraph::ParameterVector{subGraph.input_node});
    }
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, SplitConvolutionWithFqAddTestSmallSize) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        SubGraph subGraph = createSubGraphSolid(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1});
        appendSubGraphAddNode(subGraph);
        auto append_func = [] (const ngraph::Output<ngraph::Node>& input_node)
        {
            auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
            auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
            auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
            auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
            return std::make_shared<ngraph::opset7::FakeQuantize>(input_node, input_low,
                                                                  input_high, output_low,
                                                                  output_high, 11);
        };
        appendSubGraph(subGraph, append_func);

        auto result = std::make_shared<ngraph::opset7::Result>(subGraph.output_nodes.front());
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{subGraph.input_node});
        
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::SplitConvolutionWithFq>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        SubGraph subGraph = createSubGraphSolid(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1});
        appendSubGraphAddNode(subGraph);
        auto append_func = [] (const ngraph::Output<ngraph::Node>& input_node)
        {
            auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
            auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
            auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
            auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
            return std::make_shared<ngraph::opset7::FakeQuantize>(input_node, input_low,
                                                                  input_high, output_low,
                                                                  output_high, 11);
        };
        appendSubGraph(subGraph, append_func);

        auto result = std::make_shared<ngraph::opset7::Result>(subGraph.output_nodes.front());
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{subGraph.input_node});
    }
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

} // namespace
} // namespace testing

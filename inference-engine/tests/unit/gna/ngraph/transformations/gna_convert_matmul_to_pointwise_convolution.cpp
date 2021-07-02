// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <tuple>
#include <memory>

#include "transformations/convert_matmul_to_pointwise_convolution.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include <ngraph/pass/visualize_tree.hpp> // DEBUG

namespace testing {

// TODO: check MatMul input != 2 or output != 2 rank

/*
FIXME: error in ConvertMatmulToPointWiseConvolution
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, data_shape);

        auto constant = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{16, 4}, {1});
        auto matmul_operation = std::make_shared<ngraph::opset7::MatMul>(input_params, constant);

        auto result = std::make_shared<ngraph::opset7::Result>(matmul_operation);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::ConvertMatmulToPointWiseConvolution>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
*/

namespace {

struct Graph
{
    std::shared_ptr<ngraph::opset7::Parameter> input_params;
    std::shared_ptr<ngraph::op::Op> output;
};

// ------------------------------------------------------------------------------------------------------------

// TODO: use std::make_unique when C++14 will be available
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

class CreateGraphDecorator
{
public:
    CreateGraphDecorator(std::unique_ptr<CreateGraphDecorator> prev_builder = nullptr) : prev_builder_(std::move(prev_builder)) {}
    virtual ~CreateGraphDecorator() = default;
    virtual Graph build()
    {
        Graph graph;
        if (prev_builder_)
            graph = prev_builder_->build();
        updateGraph(graph);
        return graph;
    }
protected:
    virtual void updateGraph(Graph&) = 0;
private:
    std::unique_ptr<CreateGraphDecorator> prev_builder_;
};

using CreateGraphDecoratorPtr = std::unique_ptr<CreateGraphDecorator>;

class CreateBaseDecorator : public CreateGraphDecorator
{
public:
    // always the first decorator => no prev_builder
    CreateBaseDecorator() : CreateGraphDecorator(nullptr) {}
protected:
    Graph build() override;
    void updateGraph(Graph&) override {}
};

Graph CreateBaseDecorator::build() {
    Graph graph;
    graph.input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                     ngraph::Shape{16, 8});
    graph.output = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{8, 8}, {1});
    return graph; 
}

class CreateFakeQuantize : public CreateGraphDecorator
{
public:
    CreateFakeQuantize(CreateGraphDecoratorPtr prev_builder = nullptr) : CreateGraphDecorator(std::move(prev_builder)) {}
protected:
    void updateGraph(Graph&) override;
};

void CreateFakeQuantize::updateGraph(Graph& graph)
{
    auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
    auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
    auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
    auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
    auto fq_node = std::make_shared<ngraph::opset7::FakeQuantize>(graph.output, input_low,
                                                                  input_high, output_low,
                                                                  output_high, 11);
    graph.output = fq_node;
}

class CreateMatMul : public CreateGraphDecorator
{
public:
    CreateMatMul(CreateGraphDecoratorPtr prev_builder = nullptr) : CreateGraphDecorator(std::move(prev_builder)) {}
protected:
    void updateGraph(Graph&) override;
};

void CreateMatMul::updateGraph(Graph& graph)
{
    auto matmul_node = std::make_shared<ngraph::opset7::MatMul>(graph.input_params, graph.output);
    graph.output = matmul_node;
}

class CreateAdd : public CreateGraphDecorator
{
public:
    CreateAdd(CreateGraphDecoratorPtr prev_builder = nullptr) : CreateGraphDecorator(std::move(prev_builder)) {}
protected:
    void updateGraph(Graph&) override;
};

void CreateAdd::updateGraph(Graph& graph)
{
    auto bias = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
    auto add_node = std::make_shared<ngraph::opset7::Add>(graph.output, bias);
    graph.output = add_node;
}

template<typename Arg, typename... Args>
auto createBuildDecorator() -> typename std::enable_if<(sizeof...(Args) == 0), CreateGraphDecoratorPtr>::type
{
    CreateGraphDecoratorPtr build_decorator = make_unique<CreateBaseDecorator>();
    return make_unique<Arg>(std::move(build_decorator));
}

template<typename Arg, typename... Args>
auto createBuildDecorator() -> typename std::enable_if<(sizeof...(Args) > 0), CreateGraphDecoratorPtr>::type
{
    CreateGraphDecoratorPtr build_decorator = createBuildDecorator<Args...>();
    return make_unique<Arg>(std::move(build_decorator));
}

template<typename Arg, typename... Args>
Graph createTransformedGraph()
{
    CreateGraphDecoratorPtr build_decorator = createBuildDecorator<Arg, Args...>();
    return build_decorator->build();
}

// ------------------------------------------------------------------------------------------------------------

Graph createReferenceGraph(bool addFakeQuantizeNode, bool insertAddNode)
{
    Graph graph;

    graph.input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                    ngraph::Shape{16, 8});
    auto constant_node = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{8, 8}, {1});

    auto const_reshape_before = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                        ngraph::Shape{4},
                                                                        ngraph::Shape{1, 1, 16, 8});
    auto reshape_before =  std::make_shared<ngraph::opset7::Reshape>(graph.input_params, const_reshape_before, false);
    
    auto const_transpose_before = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                                    ngraph::Shape{4},
                                                                    ngraph::Shape{0, 3, 1, 2});
    auto transpose_before = std::make_shared<ngraph::opset7::Transpose>(reshape_before, const_transpose_before);

    std::shared_ptr<ngraph::op::Op> parent_node = constant_node;
    if (addFakeQuantizeNode)
    {
        auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
        auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
        auto fq_node = std::make_shared<ngraph::opset7::FakeQuantize>(constant_node, input_low,
                                                                        input_high, output_low,
                                                                        output_high, 11);
        parent_node = fq_node;
    }

    auto weights_reshape_const = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                            ngraph::Shape{4}, ngraph::Shape{8, 8, 1, 1});
    auto weights_reshaped =  std::make_shared<ngraph::opset7::Reshape>(parent_node, weights_reshape_const, false);

    auto conv_node = std::make_shared<ngraph::opset7::Convolution>(transpose_before,
                                                                    weights_reshaped,
                                                                    ngraph::Strides{1, 1},
                                                                    ngraph::CoordinateDiff{0, 0},
                                                                    ngraph::CoordinateDiff{0, 0},
                                                                    ngraph::Strides{1, 1},
                                                                    ngraph::op::PadType::VALID);
    
    parent_node = conv_node;
    if (insertAddNode)
    {
        auto bias = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto add_node = std::make_shared<ngraph::opset7::Add>(parent_node, bias);
        parent_node = add_node;
    }

    auto const_transpose_after = ngraph::opset7::Constant::create(ngraph::element::i64,
                                                                    ngraph::Shape{4},
                                                                    ngraph::Shape{0, 2, 3, 1});
    auto transpose_after = std::make_shared<ngraph::opset7::Transpose>(parent_node, const_transpose_after);

    auto const_reshape_after = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                            ngraph::Shape{2},
                                                                            ngraph::Shape{16, 8});
    graph.output = std::make_shared<ngraph::opset7::Reshape>(transpose_after, const_reshape_after, false);

    return graph;
}

} // namespace

// -------------------------------------------------------------------------------------------------------

TEST(TransformationTests, ConvertMatmulToPointWiseConvolutionTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    {
        Graph graph = createTransformedGraph<CreateMatMul>();

        auto result = std::make_shared<ngraph::opset7::Result>(graph.output);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{graph.input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::ConvertMatmulToPointWiseConvolution>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        Graph graph = createReferenceGraph(false /* addFakeQuantizeNode */, false /* insertAddNode */);
        auto result = std::make_shared<ngraph::opset7::Result>(graph.output);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{graph.input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ConvertMatmulToPointWiseConvolutionFqTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    const ngraph::Shape data_shape{16, 8};
    {
        Graph graph = createTransformedGraph<CreateMatMul, CreateFakeQuantize>();

        auto result = std::make_shared<ngraph::opset7::Result>(graph.output);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{graph.input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::ConvertMatmulToPointWiseConvolution>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        Graph graph = createReferenceGraph(true /* addFakeQuantizeNode */, false /* insertAddNode */);
        auto result = std::make_shared<ngraph::opset7::Result>(graph.output);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{graph.input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ConvertMatmulWithBiasToPointWiseConvolutionTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    {
        Graph graph = createTransformedGraph<CreateAdd, CreateMatMul>();

        auto result = std::make_shared<ngraph::opset7::Result>(graph.output);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{graph.input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::ConvertMatmulWithBiasToPointWiseConvolution>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        Graph graph = createReferenceGraph(false /* addFakeQuantizeNode */, true /* insertAddNode */);
        auto result = std::make_shared<ngraph::opset7::Result>(graph.output);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{graph.input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, ConvertMatmulWithBiasToPointWiseConvolutionFqTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    {
        Graph graph = createTransformedGraph<CreateAdd, CreateMatMul, CreateFakeQuantize>();

        auto result = std::make_shared<ngraph::opset7::Result>(graph.output);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{graph.input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::ConvertMatmulWithBiasToPointWiseConvolution>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        Graph graph = createReferenceGraph(true /* addFakeQuantizeNode */, true /* insertAddNode */);
        auto result = std::make_shared<ngraph::opset7::Result>(graph.output);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{graph.input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}


// TODO
TEST(TransformationTests, ConvertMatmulWithFqToPointWiseConvolutionTest) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);
    {
        Graph graph = createTransformedGraph<CreateAdd, CreateMatMul>();

        auto result = std::make_shared<ngraph::opset7::Result>(graph.output);
        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{graph.input_params});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::ConvertMatmulWithFqToPointWiseConvolution>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }
#if 0
    {
        Graph graph = createReferenceGraph(false /* addFakeQuantizeNode */, true /* insertAddNode */);
        auto result = std::make_shared<ngraph::opset7::Result>(graph.output);
        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{graph.input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
#endif
}

} // namespace testing

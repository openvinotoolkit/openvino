// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/split_convolution_with_large_number_of_filters.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

namespace testing {
namespace {

struct Graph {
    std::shared_ptr<ngraph::Function> createFunction();

    std::shared_ptr<ngraph::opset8::Parameter> input_params;
    ngraph::OutputVector output_nodes;
};

std::shared_ptr<ngraph::Function> Graph::createFunction() {
    auto result = std::make_shared<ngraph::opset8::Result>(output_nodes.front());
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                              ngraph::ParameterVector{input_params});
}

// TODO: use std::make_unique when C++14 will be available
template <typename T, typename... Args>
std::unique_ptr<T> createUnique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

class CreateGraphDecorator {
public:
    CreateGraphDecorator(std::unique_ptr<CreateGraphDecorator> prev = nullptr) : prev_(std::move(prev)) {}
    virtual ~CreateGraphDecorator() = default;
    virtual Graph build() {
        Graph graph;
        if (prev_)
            graph = prev_->build();
        updateGraph(graph);
        return graph;
    }
protected:
    virtual void updateGraph(Graph& graph) = 0;
private:
    CreateGraphDecorator(const CreateGraphDecorator&) = delete;
    CreateGraphDecorator& operator=(const CreateGraphDecorator&) = delete;
private:
    std::unique_ptr<CreateGraphDecorator> prev_;
};

using CreateGraphDecoratorPtr = std::unique_ptr<CreateGraphDecorator>;

class CreateAppendableGraphDecorator : public CreateGraphDecorator {
public:
    CreateAppendableGraphDecorator(std::unique_ptr<CreateGraphDecorator> prev = nullptr) :
        CreateGraphDecorator(std::move(prev)) {}
protected:
    void updateGraph(Graph& graph) override {
        ngraph::OutputVector new_graph_output;
        for (auto&& node : graph.output_nodes) {
            new_graph_output.emplace_back(createOutputNode(node));
        }

        if (graph.output_nodes.empty())
            new_graph_output.emplace_back(createOutputNode(graph.input_params));

        graph.output_nodes.swap(new_graph_output);
    }
    virtual ngraph::Output<ngraph::Node> createOutputNode(const ngraph::Output<ngraph::Node>& parent_node) = 0;
};

class CreateBaseDecorator : public CreateGraphDecorator {
public:
    // always the first decorator => no prev_builder
    CreateBaseDecorator(const ngraph::Shape& input_data_shape = ngraph::Shape{1, 64, 1, 1}) :
                        CreateGraphDecorator(nullptr),
                        input_data_shape_(input_data_shape) {}
protected:
    Graph build() override;
    void updateGraph(Graph& graph) override {}
private:
    const ngraph::Shape input_data_shape_;
};

using CreateBaseDecoratorPtr = std::unique_ptr<CreateBaseDecorator>;

Graph CreateBaseDecorator::build() {
    Graph graph;
    graph.input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32,
                                                                     input_data_shape_);
    return graph;
}

class CreateConvolution : public CreateAppendableGraphDecorator {
public:
    CreateConvolution(CreateGraphDecoratorPtr prev, const ngraph::Shape& kernel_shape = ngraph::Shape{33024, 64, 1, 1}) :
        CreateAppendableGraphDecorator(std::move(prev)),
        kernel_shape_(kernel_shape) {}
protected:
    ngraph::Output<ngraph::Node> createOutputNode(const ngraph::Output<ngraph::Node>& parent_node) override;
private:
    const ngraph::Shape kernel_shape_;
};

ngraph::Output<ngraph::Node> CreateConvolution::createOutputNode(const ngraph::Output<ngraph::Node>& parent_node) {
    auto kernel = ngraph::opset8::Constant::create(ngraph::element::f32,
                                                   kernel_shape_, {1});

    return std::make_shared<ngraph::opset8::Convolution>(parent_node,
                                                         kernel,
                                                         ngraph::Strides{1, 1},
                                                         ngraph::CoordinateDiff{0, 0},
                                                         ngraph::CoordinateDiff{0, 0},
                                                         ngraph::Strides{1, 1});
}

// should be used only after CreateBaseDecorator
class CreateSplitConvolution : public CreateGraphDecorator {
public:
    CreateSplitConvolution(CreateGraphDecoratorPtr prev,
                              const ngraph::Shape& kernel_shape = ngraph::Shape{33024, 64, 1, 1},
                              const ngraph::Shape& split_shape = ngraph::Shape{8192, 8192, 8192, 8192, 256}) :
        CreateGraphDecorator(std::move(prev)),
        kernel_shape_(kernel_shape),
        split_shape_(split_shape) {}
protected:
    void updateGraph(Graph& graph) override;
private:
    const ngraph::Shape kernel_shape_;
    const ngraph::Shape split_shape_;
};

void CreateSplitConvolution::updateGraph(Graph& graph) {
    auto split_node_c1 = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
    auto split_node_c2 = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape({split_shape_.size()}), split_shape_);
    auto kernel = ngraph::opset8::Constant::create(ngraph::element::f32, kernel_shape_, {1});
    auto split_node = std::make_shared<ngraph::opset8::VariadicSplit>(kernel, split_node_c1, split_node_c2);

    for (int i = 0; i < split_shape_.size(); ++i) {
        auto convolution_operation = std::make_shared<ngraph::opset8::Convolution>(graph.input_params,
                                                                                   split_node->output(i),
                                                                                   ngraph::Strides{1, 1},
                                                                                   ngraph::CoordinateDiff{0, 0},
                                                                                   ngraph::CoordinateDiff{0, 0},
                                                                                   ngraph::Strides{1, 1});
        graph.output_nodes.push_back(convolution_operation);
    }
}

class CreateAdd : public CreateAppendableGraphDecorator {
public:
    CreateAdd(CreateGraphDecoratorPtr prev) :
        CreateAppendableGraphDecorator(std::move(prev)) {}
protected:
    ngraph::Output<ngraph::Node> createOutputNode(const ngraph::Output<ngraph::Node>& parent_node) override;
};

ngraph::Output<ngraph::Node> CreateAdd::createOutputNode(const ngraph::Output<ngraph::Node>& parent_node) {
    auto bias = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1, 33024, 1, 1}, {1});
    return std::make_shared<ngraph::opset8::Add>(parent_node, bias);
}

// should be used only after CreateBaseDecorator
class CreateSplitAdd : public CreateGraphDecorator {
public:
    CreateSplitAdd(CreateGraphDecoratorPtr prev,
        const ngraph::Shape& bias_shape = ngraph::Shape{1, 33024, 1, 1},
        const ngraph::Shape& split_shape = ngraph::Shape{8192, 8192, 8192, 8192, 256}) :
        CreateGraphDecorator(std::move(prev)),
        bias_shape_(bias_shape),
        split_shape_(split_shape) {}
protected:
    void updateGraph(Graph& graph) override;
private:
    const ngraph::Shape bias_shape_;
    const ngraph::Shape split_shape_;
};

void CreateSplitAdd::updateGraph(Graph& graph) {
    auto split_node_c1 = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
    auto split_node_c2 = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape({split_shape_.size()}), split_shape_);
    auto bias = ngraph::opset8::Constant::create(ngraph::element::f32, bias_shape_, {1});
    auto split_node = std::make_shared<ngraph::opset8::VariadicSplit>(bias, split_node_c1, split_node_c2);

    ngraph::OutputVector new_graph_output;

    for (int i = 0; i < split_shape_.size(); ++i) {
        auto add_operation = std::make_shared<ngraph::opset8::Add>(graph.output_nodes.at(i),
            split_node->output(i));
        new_graph_output.push_back(add_operation);
    }

    graph.output_nodes.swap(new_graph_output);
}

class CreateFakeQuantize : public CreateAppendableGraphDecorator {
public:
    CreateFakeQuantize(CreateGraphDecoratorPtr prev) :
        CreateAppendableGraphDecorator(std::move(prev)) {}
protected:
    ngraph::Output<ngraph::Node> createOutputNode(const ngraph::Output<ngraph::Node>& parent_node) override;
};

ngraph::Output<ngraph::Node> CreateFakeQuantize::createOutputNode(const ngraph::Output<ngraph::Node>& parent_node) {
    auto input_low = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
    auto input_high = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
    auto output_low = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
    auto output_high = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
    return std::make_shared<ngraph::opset8::FakeQuantize>(parent_node, input_low,
                                                          input_high, output_low,
                                                          output_high, 11);
}

class CreateConcat : public CreateGraphDecorator {
public:
    CreateConcat(CreateGraphDecoratorPtr prev) :
        CreateGraphDecorator(std::move(prev)) {}
protected:
    void updateGraph(Graph& graph) override;
};

void CreateConcat::updateGraph(Graph& graph) {
    ngraph::OutputVector new_graph_output;
    new_graph_output.emplace_back(std::make_shared<ngraph::opset8::Concat>(graph.output_nodes, 1));
    graph.output_nodes.swap(new_graph_output);
}

// -------------------------------------------------------------------------------------------------------

template<typename DecorT, typename... DecorTs, typename std::enable_if<(sizeof...(DecorTs) == 0), bool>::type = true>
CreateGraphDecoratorPtr createBuildDecorator() {
    CreateGraphDecoratorPtr build_decorator = createUnique<CreateBaseDecorator>();
    return createUnique<DecorT>(std::move(build_decorator));
}

template<typename DecorT, typename... DecorTs, typename std::enable_if<(sizeof...(DecorTs) > 0), bool>::type = true>
CreateGraphDecoratorPtr createBuildDecorator() {
    CreateGraphDecoratorPtr build_decorator = createBuildDecorator<DecorTs...>();
    return createUnique<DecorT>(std::move(build_decorator));
}

template<typename DecorT, typename... DecorTs>
Graph createGraph() {
    CreateGraphDecoratorPtr build_decorator = createBuildDecorator<DecorT, DecorTs...>();
    return build_decorator->build();
}

CreateGraphDecoratorPtr createBuildDecorator(const ngraph::Shape& input_shape, const ngraph::Shape& kernel_shape) {
    CreateGraphDecoratorPtr base_decorator = createUnique<CreateBaseDecorator>(input_shape);
    return createUnique<CreateConvolution>(std::move(base_decorator), kernel_shape);
}

template<typename DecorT, typename... DecorTs, typename std::enable_if<(sizeof...(DecorTs) == 0), bool>::type = true>
CreateGraphDecoratorPtr createBuildDecorator(const ngraph::Shape& input_shape,
                                             const ngraph::Shape& kernel_shape) {
    CreateGraphDecoratorPtr build_decorator = createBuildDecorator(input_shape, kernel_shape);
    return createUnique<DecorT>(std::move(build_decorator));
}

template<typename DecorT, typename... DecorTs, typename std::enable_if<(sizeof...(DecorTs) > 0), bool>::type = true>
CreateGraphDecoratorPtr createBuildDecorator(const ngraph::Shape& input_shape,
                                             const ngraph::Shape& kernel_shape) {
    CreateGraphDecoratorPtr build_decorator = createBuildDecorator<DecorTs...>(input_shape, kernel_shape);
    return createUnique<DecorT>(std::move(build_decorator));
}

Graph createSolidGraph(const ngraph::Shape& input_shape, const ngraph::Shape& kernel_shape) {
    CreateGraphDecoratorPtr build_decorator = createBuildDecorator(input_shape, kernel_shape);
    return build_decorator->build();
}

template<typename DecorT, typename... DecorTs>
Graph createSolidGraph(const ngraph::Shape& input_shape, const ngraph::Shape& kernel_shape) {
    CreateGraphDecoratorPtr build_decorator = createBuildDecorator<DecorT, DecorTs...>(input_shape, kernel_shape);
    return build_decorator->build();
}

// -------------------------------------------------------------------------------------------------------

class SplitConvolutionFilterFixture: public CommonTestUtils::TestsCommon,
                               public ::testing::WithParamInterface<std::tuple<Graph /* transformed */,
                                                                               Graph /* reference */,
                                                                               ngraph::pass::Manager>> {
public:
    void SetUp() override;
public:
    std::shared_ptr<ngraph::Function> function, reference_function;
    ngraph::pass::Manager pass_manager;
};

void SplitConvolutionFilterFixture::SetUp() {
    // TODO: use auto & [transformed_graph, reference_graph] = this->GetParam() when C++17
    Graph transformed_graph;
    Graph reference_graph;
    std::tie(transformed_graph, reference_graph, pass_manager) = this->GetParam();

    function = transformed_graph.createFunction();
    reference_function = reference_graph.createFunction();
}

void execute_test(std::shared_ptr<ngraph::Function> function,
                  std::shared_ptr<ngraph::Function> reference_function,
                  ngraph::pass::Manager& pass_manager) {
    pass_manager.run_passes(function);
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

template <typename TransformationT>
ngraph::pass::Manager createPassManager() {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<TransformationT>();
    return manager;
}

TEST_P(SplitConvolutionFilterFixture, CompareFunctions) {
    execute_test(function, reference_function, pass_manager);
}

INSTANTIATE_TEST_SUITE_P(SplitConvolutionFilterTestSuite, SplitConvolutionFilterFixture,
                         ::testing::Values(
                             std::make_tuple(createGraph<CreateConvolution>(),
                                                           createGraph<CreateConcat, CreateSplitConvolution>(),
                                                           createPassManager<GNAPluginNS::SplitConvolutionFilter>()),
                                           std::make_tuple(createGraph<CreateAdd, CreateConvolution>(),
                                                           createGraph<CreateConcat, CreateSplitAdd, CreateSplitConvolution>(),
                                                           createPassManager<GNAPluginNS::SplitConvolutionFilterWithBias>()),
                                           std::make_tuple(createGraph<CreateFakeQuantize, CreateConvolution>(),
                                                           createGraph<CreateConcat, CreateFakeQuantize, CreateSplitConvolution>(),
                                                           createPassManager<GNAPluginNS::SplitConvolutionFilterWithFq>()),
                                           std::make_tuple(createGraph<CreateFakeQuantize, CreateAdd, CreateConvolution>(),
                                                           createGraph<CreateConcat, CreateFakeQuantize, CreateSplitAdd, CreateSplitConvolution>(),
                                                           createPassManager<GNAPluginNS::SplitConvolutionFilterWithFq>()),
                                           std::make_tuple(createSolidGraph(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1}),
                                                           createSolidGraph(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1}),
                                                           createPassManager<GNAPluginNS::SplitConvolutionFilter>()),
                                           std::make_tuple(createSolidGraph<CreateAdd>(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1}),
                                                           createSolidGraph<CreateAdd>(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1}),
                                                           createPassManager<GNAPluginNS::SplitConvolutionFilterWithBias>()),
                                           std::make_tuple(createSolidGraph<CreateFakeQuantize>(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1}),
                                                           createSolidGraph<CreateFakeQuantize>(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1}),
                                                           createPassManager<GNAPluginNS::SplitConvolutionFilterWithFq>()),
                                           std::make_tuple(createSolidGraph<CreateAdd, CreateFakeQuantize>(ngraph::Shape{1, 1, 1, 1},
                                                                                                           ngraph::Shape{1, 1, 1, 1}),
                                                           createSolidGraph<CreateAdd, CreateFakeQuantize>(ngraph::Shape{1, 1, 1, 1},
                                                                                                           ngraph::Shape{1, 1, 1, 1}),
                                                           createPassManager<GNAPluginNS::SplitConvolutionFilterWithFq>())));

} // namespace
} // namespace testing

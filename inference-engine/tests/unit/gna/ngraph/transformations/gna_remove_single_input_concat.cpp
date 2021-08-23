// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/remove_single_input_concat.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

namespace testing {
namespace {

// FIXME: move common graph-decorator code to dedicated header

struct Graph {
    std::shared_ptr<ngraph::Function> createFunction();

    std::shared_ptr<ngraph::opset7::Parameter> input_params;
    ngraph::OutputVector output_nodes;
};

std::shared_ptr<ngraph::Function> Graph::createFunction() {
    auto result = std::make_shared<ngraph::opset7::Result>(output_nodes.front());
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

#if 0
// FIXME: should we need this?
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
#endif

class CreateBaseDecorator : public CreateGraphDecorator {
public:
    // always the first decorator => no prev_builder
    CreateBaseDecorator(const ngraph::Shape& input_data_shape = ngraph::Shape{1, 64, 4096, 4096}) :
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
    graph.input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32,
                                                                     input_data_shape_);
    return graph;
}

template <typename DecorT>
class ForEachOutput : CreateGraphDecorator {
public:
    ForEachOutput(CreateGraphDecoratorPtr prev) : CreateGraphDecorator(std::move(prev)) {}
protected:
    void updateGraph(Graph& graph) override
    {
        // TODO
    }
};

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

// -------------------------------------------------------------------------------------------------------

class CreateConcat : public CreateGraphDecorator {
public:
    CreateConcat(CreateGraphDecoratorPtr prev, int64_t axis = 0) :
        CreateGraphDecorator(std::move(prev)), axis_(axis) {}
protected:
    void updateGraph(Graph& graph) override;
private:
    const int64_t axis_;
};

void CreateConcat::updateGraph(Graph& graph) {
    ngraph::OutputVector new_graph_output;
    new_graph_output.emplace_back(std::make_shared<ngraph::opset7::Concat>(graph.output_nodes, axis_));
    graph.output_nodes.swap(new_graph_output);
}

// -------------------------------------------------------------------------------------------------------
#if 0
class RemoveSingleInputConcatFixture: public CommonTestUtils::TestsCommon,
                               public ::testing::WithParamInterface<std::tuple<Graph /* tranformed */,
                                                                               Graph /* reference */,
                                                                               ngraph::pass::Manager>> {
public:
    void SetUp() override;
public:
    std::shared_ptr<ngraph::Function> function, reference_function;
    ngraph::pass::Manager pass_manager;
};

void RemoveSingleInputConcatFixture::SetUp() {
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

TEST_P(RemoveSingleInputConcatFixture, CompareFunctions) {
    execute_test(function, reference_function, pass_manager);
}

INSTANTIATE_TEST_SUITE_P(RemoveSingleInputConcatTestSuite, RemoveSingleInputConcatFixture,
                         ::testing::Values(std::make_tuple(createGraph<CreateConvolution>(),
                                                           createGraph<CreateConcat, CreateSplittedConvolution>(),
                                                           createPassManager<GNAPluginNS::SplitConvolution>()),
                                           std::make_tuple(createGraph<CreateAdd, CreateConvolution>(),
                                                           createGraph<CreateConcat, CreateAdd, CreateSplittedConvolution>(),
                                                           createPassManager<GNAPluginNS::SplitConvolutionWithBias>()),
                                           std::make_tuple(createGraph<CreateFakeQuantize, CreateConvolution>(),
                                                           createGraph<CreateConcat, CreateFakeQuantize, CreateSplittedConvolution>(),
                                                           createPassManager<GNAPluginNS::SplitConvolutionWithFq>()),
                                           std::make_tuple(createGraph<CreateFakeQuantize, CreateAdd, CreateConvolution>(),
                                                           createGraph<CreateConcat, CreateFakeQuantize, CreateAdd, CreateSplittedConvolution>(),
                                                           createPassManager<GNAPluginNS::SplitConvolutionWithFq>()),
                                           std::make_tuple(createSolidGraph(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1}),
                                                           createSolidGraph(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1}),
                                                           createPassManager<GNAPluginNS::SplitConvolution>()),
                                           std::make_tuple(createSolidGraph<CreateAdd>(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1}),
                                                           createSolidGraph<CreateAdd>(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1}),
                                                           createPassManager<GNAPluginNS::SplitConvolutionWithBias>()),
                                           std::make_tuple(createSolidGraph<CreateFakeQuantize>(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1}),
                                                           createSolidGraph<CreateFakeQuantize>(ngraph::Shape{1, 1, 1, 1}, ngraph::Shape{1, 1, 1, 1}),
                                                           createPassManager<GNAPluginNS::SplitConvolutionWithFq>()),
                                           std::make_tuple(createSolidGraph<CreateAdd, CreateFakeQuantize>(ngraph::Shape{1, 1, 1, 1},
                                                                                                           ngraph::Shape{1, 1, 1, 1}),
                                                           createSolidGraph<CreateAdd, CreateFakeQuantize>(ngraph::Shape{1, 1, 1, 1},
                                                                                                           ngraph::Shape{1, 1, 1, 1}),
                                                           createPassManager<GNAPluginNS::SplitConvolutionWithFq>())));

#endif

TEST(TransformationTests, RemoveSingleInputConcatTestOneOutput) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                          ngraph::Shape{1, 3, 64});
       
        auto add_bias = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {2});

        auto add_operation = std::make_shared<ngraph::opset7::Add>(input_params, add_bias);

        auto concat_operation = std::make_shared<ngraph::opset7::Concat>(ngraph::OutputVector{add_operation}, 0);

        auto add_bias_1 = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {3});
        auto add_operation_1 = std::make_shared<ngraph::opset7::Add>(concat_operation, add_bias_1);

        auto result = std::make_shared<ngraph::opset7::Result>(add_operation_1);

        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::RemoveSingleInputConcat>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                          ngraph::Shape{1, 3, 64});
       
        auto add_bias = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {2});

        auto add_operation = std::make_shared<ngraph::opset7::Add>(input_params, add_bias);

        auto add_bias_1 = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {3});
        auto add_operation_1 = std::make_shared<ngraph::opset7::Add>(add_operation, add_bias_1);

        auto result = std::make_shared<ngraph::opset7::Result>(add_operation_1);

        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, RemoveSingleInputConcatTestMultipleOutput) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                          ngraph::Shape{1, 3, 64});
       
        auto add_bias = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {2});

        auto add_operation = std::make_shared<ngraph::opset7::Add>(input_params, add_bias);

        auto concat_operation = std::make_shared<ngraph::opset7::Concat>(ngraph::OutputVector{add_operation}, 0);

        auto add_bias_1 = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {3});
        auto add_operation_1 = std::make_shared<ngraph::opset7::Add>(concat_operation, add_bias_1);

        auto add_bias_2 = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {3});
        auto add_operation_2 = std::make_shared<ngraph::opset7::Add>(concat_operation, add_bias_1);

        auto result_1 = std::make_shared<ngraph::opset7::Result>(add_operation_1);
        auto result_2 = std::make_shared<ngraph::opset7::Result>(add_operation_2);

        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result_1, result_2},
                                                  ngraph::ParameterVector{input_params});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::RemoveSingleInputConcat>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                          ngraph::Shape{1, 3, 64});
       
        auto add_bias = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {2});

        auto add_operation = std::make_shared<ngraph::opset7::Add>(input_params, add_bias);

        auto add_bias_1 = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {3});
        auto add_operation_1 = std::make_shared<ngraph::opset7::Add>(add_operation, add_bias_1);

        auto add_bias_2 = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {3});
        auto add_operation_2 = std::make_shared<ngraph::opset7::Add>(add_operation, add_bias_1);

        auto result_1 = std::make_shared<ngraph::opset7::Result>(add_operation_1);
        auto result_2 = std::make_shared<ngraph::opset7::Result>(add_operation_2);

        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result_1, result_2},
                                                  ngraph::ParameterVector{input_params});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

TEST(TransformationTests, RemoveSingleInputConcatTestMultipleInputOneOutput) {
    std::shared_ptr<ngraph::Function> func(nullptr), reference_func(nullptr);

    {
        auto input_params_1 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                          ngraph::Shape{1, 3, 64});
       
        auto input_params_2 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                          ngraph::Shape{1, 3, 64});

        auto add_bias_1 = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {2});
        auto add_operation_1 = std::make_shared<ngraph::opset7::Add>(input_params_1, add_bias_1);

        auto add_bias_2 = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {2});
        auto add_operation_2 = std::make_shared<ngraph::opset7::Add>(input_params_2, add_bias_2);

        auto concat_operation = std::make_shared<ngraph::opset7::Concat>(ngraph::OutputVector{add_operation_1, add_operation_2}, 0);

        auto add_bias_3 = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {3});
        auto add_operation_3 = std::make_shared<ngraph::opset7::Add>(concat_operation, add_bias_1);

        auto result = std::make_shared<ngraph::opset7::Result>(add_operation_3);

        func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_1, input_params_2});
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<GNAPluginNS::RemoveSingleInputConcat>();
        m.run_passes(func);
        ASSERT_NO_THROW(check_rt_info(func));
    }

    {
        auto input_params_1 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                          ngraph::Shape{1, 3, 64});
       
        auto input_params_2 = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64,
                                                                          ngraph::Shape{1, 3, 64});

        auto add_bias_1 = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {2});
        auto add_operation_1 = std::make_shared<ngraph::opset7::Add>(input_params_1, add_bias_1);

        auto add_bias_2 = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {2});
        auto add_operation_2 = std::make_shared<ngraph::opset7::Add>(input_params_2, add_bias_2);

        auto concat_operation = std::make_shared<ngraph::opset7::Concat>(ngraph::OutputVector{add_operation_1, add_operation_2}, 0);

        auto add_bias_3 = ngraph::opset7::Constant::create(ngraph::element::i64, {1, 1, 1}, {3});
        auto add_operation_3 = std::make_shared<ngraph::opset7::Add>(concat_operation, add_bias_1);

        auto result = std::make_shared<ngraph::opset7::Result>(add_operation_3);

        reference_func = std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params_1, input_params_2});
    }

    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(func, reference_func);
    ASSERT_TRUE(result.valid);
}

} // namespace
} // namespace testing

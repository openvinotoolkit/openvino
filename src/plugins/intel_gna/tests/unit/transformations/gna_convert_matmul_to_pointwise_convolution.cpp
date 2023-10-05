// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <tuple>

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_models/builders.hpp"
#include "transformations/convert_matmul_to_pointwise_convolution.hpp"

namespace testing {

namespace {

struct Graph {
    std::shared_ptr<ngraph::Function> createFunction();

    std::shared_ptr<ngraph::opset7::Parameter> input_params;
    std::shared_ptr<ngraph::op::Op> output;
};

std::shared_ptr<ngraph::Function> Graph::createFunction() {
    auto result = std::make_shared<ngraph::opset7::Result>(output);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{input_params});
}

// ------------------------------------------------------------------------------------------------------------

// TODO: use std::make_unique when C++14 will be available
template <typename T, typename... Args>
std::unique_ptr<T> createUnique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

class CreateGraphDecorator {
public:
    CreateGraphDecorator(std::unique_ptr<CreateGraphDecorator> prev_builder = nullptr)
        : prev_builder_(std::move(prev_builder)) {}
    virtual ~CreateGraphDecorator() = default;
    virtual Graph build() {
        Graph graph;
        if (prev_builder_)
            graph = prev_builder_->build();
        updateGraph(graph);
        return graph;
    }

protected:
    virtual void updateGraph(Graph&) = 0;

private:
    CreateGraphDecorator(const CreateGraphDecorator&) = delete;
    CreateGraphDecorator& operator=(const CreateGraphDecorator&) = delete;

private:
    std::unique_ptr<CreateGraphDecorator> prev_builder_;
};

using CreateGraphDecoratorPtr = std::unique_ptr<CreateGraphDecorator>;

class CreateBaseDecorator : public CreateGraphDecorator {
public:
    // always the first decorator => no prev_builder
    CreateBaseDecorator(const ngraph::Shape& input_data_shape, const ngraph::Shape& input_const_shape)
        : CreateGraphDecorator(nullptr),
          input_data_shape_(input_data_shape),
          input_const_shape_(input_const_shape) {}

protected:
    Graph build() override;
    void updateGraph(Graph&) override {}

private:
    const ngraph::Shape input_data_shape_;
    const ngraph::Shape input_const_shape_;
};

Graph CreateBaseDecorator::build() {
    Graph graph;
    graph.input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, input_data_shape_);
    graph.output = ngraph::opset7::Constant::create(ngraph::element::i64, input_const_shape_, {1});
    return graph;
}

class CreateFakeQuantize : public CreateGraphDecorator {
public:
    CreateFakeQuantize(CreateGraphDecoratorPtr prev_builder = nullptr)
        : CreateGraphDecorator(std::move(prev_builder)) {}

protected:
    void updateGraph(Graph&) override;
};

std::shared_ptr<ngraph::opset7::FakeQuantize> createFakeQuantizeNode(std::shared_ptr<ngraph::op::Op> parent_node) {
    auto input_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
    auto input_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {20});
    auto output_low = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {0});
    auto output_high = ngraph::opset7::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {10});
    return std::make_shared<ngraph::opset7::FakeQuantize>(parent_node,
                                                          input_low,
                                                          input_high,
                                                          output_low,
                                                          output_high,
                                                          11);
}

void CreateFakeQuantize::updateGraph(Graph& graph) {
    graph.output = createFakeQuantizeNode(graph.output);
}

class CreateMatMul : public CreateGraphDecorator {
public:
    CreateMatMul(CreateGraphDecoratorPtr prev_builder = nullptr) : CreateGraphDecorator(std::move(prev_builder)) {}

protected:
    void updateGraph(Graph&) override;
};

void CreateMatMul::updateGraph(Graph& graph) {
    auto matmul_node = std::make_shared<ngraph::opset7::MatMul>(graph.input_params, graph.output);
    graph.output = matmul_node;
}

template <bool ONE_DIMENSIONAL, bool ONE_CHANNEL>
class CreateAdd : public CreateGraphDecorator {
public:
    CreateAdd(CreateGraphDecoratorPtr prev_builder = nullptr) : CreateGraphDecorator(std::move(prev_builder)) {}

protected:
    void updateGraph(Graph&) override;
};

template <bool ONE_DIMENSIONAL, bool ONE_CHANNEL>
void CreateAdd<ONE_DIMENSIONAL, ONE_CHANNEL>::updateGraph(Graph& graph) {
    std::vector<size_t> axes(1, 1);
    if (std::is_same<std::integral_constant<bool, ONE_CHANNEL>, std::integral_constant<bool, false>>::value) {
        auto shape = graph.output->get_output_shape(0);
        if (std::is_same<std::integral_constant<bool, ONE_DIMENSIONAL>, std::integral_constant<bool, false>>::value) {
            axes.resize(shape.size(), 1);
        }
        axes.back() = shape.back();
    }

    auto bias = ngraph::builder::makeConstant<float>(ngraph::element::i64, axes, {}, true);
    auto add_node = std::make_shared<ngraph::opset7::Add>(graph.output, bias);
    graph.output = add_node;
}

template <typename DecorT, typename... DecorTs, typename std::enable_if<(sizeof...(DecorTs) == 0), bool>::type = true>
CreateGraphDecoratorPtr createBuildDecorator(const ngraph::Shape& input_data_shape = ngraph::Shape{16, 8},
                                             const ngraph::Shape& input_const_shape = ngraph::Shape{8, 8}) {
    CreateGraphDecoratorPtr build_decorator = createUnique<CreateBaseDecorator>(input_data_shape, input_const_shape);
    return createUnique<DecorT>(std::move(build_decorator));
}

template <typename DecorT, typename... DecorTs, typename std::enable_if<(sizeof...(DecorTs) > 0), bool>::type = true>
CreateGraphDecoratorPtr createBuildDecorator(const ngraph::Shape& input_data_shape = ngraph::Shape{16, 8},
                                             const ngraph::Shape& input_const_shape = ngraph::Shape{8, 8}) {
    CreateGraphDecoratorPtr build_decorator = createBuildDecorator<DecorTs...>(input_data_shape, input_const_shape);
    return createUnique<DecorT>(std::move(build_decorator));
}

template <typename DecorT, typename... DecorTs>
Graph createTransformedGraph(const ngraph::Shape& input_data_shape = ngraph::Shape{16, 8},
                             const ngraph::Shape& input_const_shape = ngraph::Shape{8, 8}) {
    CreateGraphDecoratorPtr build_decorator =
        createBuildDecorator<DecorT, DecorTs...>(input_data_shape, input_const_shape);
    return build_decorator->build();
}

// ------------------------------------------------------------------------------------------------------------

template <bool ADD_CONST_FAKEQUANTIZE_NODE, bool INSERT_ADD_NODE, bool ONE_CHANNEL, bool ADD_OUT_FAKEQUANTIZE_NODE>
Graph createReferenceGraph() {
    Graph graph;

    graph.input_params = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::i64, ngraph::Shape{16, 8});
    auto constant_node = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{8, 8}, {1});

    auto const_reshape_before = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                           ngraph::Shape{4},
                                                                           ngraph::Shape{1, 1, 16, 8});
    auto reshape_before = std::make_shared<ngraph::opset7::Reshape>(graph.input_params, const_reshape_before, false);

    auto const_transpose_before =
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, ngraph::Shape{0, 3, 1, 2});
    auto transpose_before = std::make_shared<ngraph::opset7::Transpose>(reshape_before, const_transpose_before);

    std::shared_ptr<ngraph::op::Op> parent_node = constant_node;
    if (std::is_same<std::integral_constant<bool, ADD_CONST_FAKEQUANTIZE_NODE>,
                     std::integral_constant<bool, true>>::value) {
        parent_node = createFakeQuantizeNode(constant_node);
    }

    auto weights_reshape_const = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                            ngraph::Shape{4},
                                                                            ngraph::Shape{8, 8, 1, 1});
    auto weights_reshaped = std::make_shared<ngraph::opset7::Reshape>(parent_node, weights_reshape_const, false);

    auto conv_node = std::make_shared<ngraph::opset7::Convolution>(transpose_before,
                                                                   weights_reshaped,
                                                                   ngraph::Strides{1, 1},
                                                                   ngraph::CoordinateDiff{0, 0},
                                                                   ngraph::CoordinateDiff{0, 0},
                                                                   ngraph::Strides{1, 1},
                                                                   ngraph::op::PadType::VALID);

    parent_node = conv_node;
    if (std::is_same<std::integral_constant<bool, INSERT_ADD_NODE>, std::integral_constant<bool, true>>::value) {
        std::vector<size_t> axes(1, 1);
        if (std::is_same<std::integral_constant<bool, ONE_CHANNEL>, std::integral_constant<bool, false>>::value) {
            axes.resize(4, 1);
            axes[1] = 8;
        }

        auto bias = ngraph::builder::makeConstant<float>(ngraph::element::i64, axes, {}, true);
        auto add_node = std::make_shared<ngraph::opset7::Add>(parent_node, bias);
        parent_node = add_node;
    }

    if (std::is_same<std::integral_constant<bool, ADD_OUT_FAKEQUANTIZE_NODE>,
                     std::integral_constant<bool, true>>::value) {
        parent_node = createFakeQuantizeNode(parent_node);
    }

    auto const_transpose_after =
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, ngraph::Shape{0, 2, 3, 1});
    auto transpose_after = std::make_shared<ngraph::opset7::Transpose>(parent_node, const_transpose_after);

    auto const_reshape_after = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
                                                                          ngraph::Shape{2},
                                                                          ngraph::Shape{16, 8});
    graph.output = std::make_shared<ngraph::opset7::Reshape>(transpose_after, const_reshape_after, false);

    return graph;
}

// -------------------------------------------------------------------------------------------------------

class ConvertMatmulToPointWiseConvolutionFixture
    : public ov::test::TestsCommon,
      public ::testing::WithParamInterface<
          std::tuple<Graph /* tranformed */, Graph /* reference */, ngraph::pass::Manager>> {
public:
    void SetUp() override;

public:
    std::shared_ptr<ngraph::Function> function, reference_function;
    ngraph::pass::Manager pass_manager;
};

void ConvertMatmulToPointWiseConvolutionFixture::SetUp() {
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
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

template <typename TransformationT>
ngraph::pass::Manager createPassManager() {
    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<TransformationT>();
    return manager;
}

TEST_P(ConvertMatmulToPointWiseConvolutionFixture, CompareFunctions) {
    execute_test(function, reference_function, pass_manager);
}

namespace {
constexpr bool AddConstFakeQuantizeNode = true;
constexpr bool InsertAddNode = true;
constexpr bool OneDimensional = true;
constexpr bool OneChannel = true;
constexpr bool AddOutFakeQuantizeNode = true;
}  // namespace

INSTANTIATE_TEST_SUITE_P(
    ConvertMatmulToPointWiseConvolutionTestSuite,
    ConvertMatmulToPointWiseConvolutionFixture,
    ::testing::Values(
        std::make_tuple(
            createTransformedGraph<CreateMatMul>(),
            createReferenceGraph<!AddConstFakeQuantizeNode, !InsertAddNode, !OneChannel, !AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateMatMul, CreateFakeQuantize>(),
            createReferenceGraph<AddConstFakeQuantizeNode, !InsertAddNode, !OneChannel, !AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateAdd<OneDimensional, OneChannel>, CreateMatMul>(),
            createReferenceGraph<!AddConstFakeQuantizeNode, InsertAddNode, OneChannel, !AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithBiasToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateAdd<OneDimensional, !OneChannel>, CreateMatMul>(),
            createReferenceGraph<!AddConstFakeQuantizeNode, InsertAddNode, !OneChannel, !AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithBiasToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateAdd<!OneDimensional, !OneChannel>, CreateMatMul>(),
            createReferenceGraph<!AddConstFakeQuantizeNode, InsertAddNode, !OneChannel, !AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithBiasToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateAdd<OneDimensional, OneChannel>, CreateMatMul, CreateFakeQuantize>(),
            createReferenceGraph<AddConstFakeQuantizeNode, InsertAddNode, OneChannel, !AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithBiasToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateAdd<OneDimensional, !OneChannel>, CreateMatMul, CreateFakeQuantize>(),
            createReferenceGraph<AddConstFakeQuantizeNode, InsertAddNode, !OneChannel, !AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithBiasToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateAdd<!OneDimensional, !OneChannel>, CreateMatMul, CreateFakeQuantize>(),
            createReferenceGraph<AddConstFakeQuantizeNode, InsertAddNode, !OneChannel, !AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithBiasToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateFakeQuantize, CreateAdd<OneDimensional, OneChannel>, CreateMatMul>(),
            createReferenceGraph<!AddConstFakeQuantizeNode, InsertAddNode, OneChannel, AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithFqToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateFakeQuantize, CreateAdd<OneDimensional, !OneChannel>, CreateMatMul>(),
            createReferenceGraph<!AddConstFakeQuantizeNode, InsertAddNode, !OneChannel, AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithFqToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateFakeQuantize, CreateAdd<!OneDimensional, !OneChannel>, CreateMatMul>(),
            createReferenceGraph<!AddConstFakeQuantizeNode, InsertAddNode, !OneChannel, AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithFqToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateFakeQuantize,
                                   CreateAdd<OneDimensional, OneChannel>,
                                   CreateMatMul,
                                   CreateFakeQuantize>(),
            createReferenceGraph<AddConstFakeQuantizeNode, InsertAddNode, OneChannel, AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithFqToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateFakeQuantize,
                                   CreateAdd<OneDimensional, !OneChannel>,
                                   CreateMatMul,
                                   CreateFakeQuantize>(),
            createReferenceGraph<AddConstFakeQuantizeNode, InsertAddNode, !OneChannel, AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithFqToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateFakeQuantize,
                                   CreateAdd<!OneDimensional, !OneChannel>,
                                   CreateMatMul,
                                   CreateFakeQuantize>(),
            createReferenceGraph<AddConstFakeQuantizeNode, InsertAddNode, !OneChannel, AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithFqToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateFakeQuantize, CreateMatMul>(),
            createReferenceGraph<!AddConstFakeQuantizeNode, !InsertAddNode, !OneChannel, AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithFqToPointWiseConvolution>()),
        std::make_tuple(
            createTransformedGraph<CreateFakeQuantize, CreateMatMul, CreateFakeQuantize>(),
            createReferenceGraph<AddConstFakeQuantizeNode, !InsertAddNode, !OneChannel, AddOutFakeQuantizeNode>(),
            createPassManager<ov::intel_gna::pass::ConvertMatmulWithFqToPointWiseConvolution>())));

// -------------------------------------------------------------------------------------------------------

class ITransformedGraphFactory {
public:
    virtual ~ITransformedGraphFactory() = default;
    virtual Graph createGraph(const ngraph::Shape& input_data_shape, const ngraph::Shape& input_const_shape) = 0;
};

template <typename DecorT, typename... DecorTs>
class TransformedGraphFactory : public ITransformedGraphFactory {
public:
    TransformedGraphFactory() = default;

    Graph createGraph(const ngraph::Shape& input_data_shape, const ngraph::Shape& input_const_shape) override {
        return createTransformedGraph<DecorT, DecorTs...>(input_data_shape, input_const_shape);
    }

private:
    TransformedGraphFactory(const TransformedGraphFactory&) = delete;
    TransformedGraphFactory& operator=(const TransformedGraphFactory&) = delete;
};

struct FixtureData {
    std::shared_ptr<ITransformedGraphFactory> graph_factory;
    ngraph::pass::Manager pass_manager;

    template <typename TransformationT, typename DecorT, typename... DecorTs>
    static FixtureData create() {
        FixtureData fixture_data;
        fixture_data.graph_factory = std::make_shared<TransformedGraphFactory<DecorT, DecorTs...>>();
        fixture_data.pass_manager = createPassManager<TransformationT>();
        return fixture_data;
    }
};

using FixtureInputShapes = std::tuple<ngraph::Shape /* input data */, ngraph::Shape /* input const */>;

class ConvertMatmulToPointWiseConvolutionInvalidInputFixture
    : public ov::test::TestsCommon,
      public ::testing::WithParamInterface<std::tuple<FixtureData, FixtureInputShapes>> {
public:
    void SetUp() override;

public:
    std::shared_ptr<ngraph::Function> function;
    ngraph::pass::Manager pass_manager;
};

void ConvertMatmulToPointWiseConvolutionInvalidInputFixture::SetUp() {
    // TODO: use auto & [fixture_data, input_shapes] = this->GetParam() when C++17
    FixtureData fixture_data;
    FixtureInputShapes input_shapes;
    std::tie(fixture_data, input_shapes) = this->GetParam();

    ngraph::Shape input_data, input_const;
    std::tie(input_data, input_const) = input_shapes;

    function = fixture_data.graph_factory->createGraph(input_data, input_const).createFunction();
    pass_manager = fixture_data.pass_manager;
}

void execute_test_cloned_function(std::shared_ptr<ngraph::Function> function, ngraph::pass::Manager& pass_manager) {
    std::shared_ptr<ngraph::Function> reference_function = ngraph::clone_function(*function);
    pass_manager.run_passes(function);
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

std::vector<FixtureData> transform_types = {
    FixtureData::create<ov::intel_gna::pass::ConvertMatmulToPointWiseConvolution, CreateMatMul>(),
    FixtureData::create<ov::intel_gna::pass::ConvertMatmulToPointWiseConvolution, CreateMatMul, CreateFakeQuantize>(),
    FixtureData::create<ov::intel_gna::pass::ConvertMatmulWithBiasToPointWiseConvolution,
                        CreateAdd<false, false>,
                        CreateMatMul>(),
    FixtureData::create<ov::intel_gna::pass::ConvertMatmulWithBiasToPointWiseConvolution,
                        CreateAdd<false, false>,
                        CreateMatMul,
                        CreateFakeQuantize>(),
    FixtureData::create<ov::intel_gna::pass::ConvertMatmulWithFqToPointWiseConvolution,
                        CreateFakeQuantize,
                        CreateAdd<false, false>,
                        CreateMatMul>(),
    FixtureData::create<ov::intel_gna::pass::ConvertMatmulWithFqToPointWiseConvolution,
                        CreateFakeQuantize,
                        CreateAdd<false, false>,
                        CreateMatMul,
                        CreateFakeQuantize>(),
    FixtureData::
        create<ov::intel_gna::pass::ConvertMatmulWithFqToPointWiseConvolution, CreateFakeQuantize, CreateMatMul>(),
    FixtureData::create<ov::intel_gna::pass::ConvertMatmulWithFqToPointWiseConvolution,
                        CreateFakeQuantize,
                        CreateMatMul,
                        CreateFakeQuantize>()};

std::vector<FixtureInputShapes> input_shapes = {std::make_tuple(ngraph::Shape{16, 16, 16}, ngraph::Shape{16, 16, 16}),
                                                std::make_tuple(ngraph::Shape{16, 9}, ngraph::Shape{9, 9}),
                                                std::make_tuple(ngraph::Shape{16, 65533}, ngraph::Shape{65533, 2}),
                                                std::make_tuple(ngraph::Shape{16, 769}, ngraph::Shape{769, 2})};

TEST_P(ConvertMatmulToPointWiseConvolutionInvalidInputFixture, CompareFunctions) {
    execute_test_cloned_function(function, pass_manager);
}

INSTANTIATE_TEST_SUITE_P(ConvertMatmulToPointWiseConvolutionInvalidInputTestSuite,
                         ConvertMatmulToPointWiseConvolutionInvalidInputFixture,
                         ::testing::Combine(::testing::ValuesIn(transform_types), ::testing::ValuesIn(input_shapes)));

}  // namespace

}  // namespace testing

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "transformations/remove_single_input_concat.hpp"

namespace testing {
namespace {

using GraphInputs = std::vector<std::shared_ptr<ngraph::opset8::Parameter>>;
using GraphOutputs = ngraph::OutputVector;

struct Graph {
    std::shared_ptr<ngraph::Function> createFunction();

    GraphInputs inputs;
    GraphOutputs outputs;
};

std::shared_ptr<ngraph::Function> Graph::createFunction() {
    ngraph::ResultVector results;
    std::transform(outputs.begin(),
                   outputs.end(),
                   std::back_inserter(results),
                   [](ngraph::Output<ngraph::Node> output) {
                       return std::make_shared<ngraph::opset8::Result>(output);
                   });

    ngraph::ParameterVector params(inputs.begin(), inputs.end());

    return std::make_shared<ngraph::Function>(results, params);
}

// -------------------------------------------------------------------------------------------------------

using Operations = std::vector<std::shared_ptr<ngraph::op::Op>>;

Graph createGraph(int n_inputs, bool has_concat, int n_outputs) {
    GraphInputs inputs;
    Operations outputs;

    for (int i = 0; i < n_inputs; ++i) {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i64, ngraph::Shape{1, 3, 64});
        inputs.push_back(input);
        outputs.push_back(input);
    }

    {
        Operations new_outputs;
        for (auto output : outputs) {
            auto add_bias = ngraph::opset8::Constant::create(ngraph::element::i64, {1, 1, 1}, {2});
            auto add_operation = std::make_shared<ngraph::opset8::Add>(output, add_bias);
            new_outputs.push_back(add_operation);
        }
        outputs.swap(new_outputs);
    }

    if (has_concat) {
        auto concat_operation =
            std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector(outputs.begin(), outputs.end()), 0);
        outputs = {concat_operation};
    }

    {
        Operations new_outputs;
        for (auto output : outputs) {
            for (int i = 0; i < n_outputs; ++i) {
                auto add_bias = ngraph::opset8::Constant::create(ngraph::element::i64, {1, 1, 1}, {3});
                auto add_operation = std::make_shared<ngraph::opset8::Add>(output, add_bias);
                new_outputs.push_back(add_operation);
            }
        }
        outputs.swap(new_outputs);
    }

    Graph graph;
    graph.inputs.swap(inputs);
    graph.outputs.insert(graph.outputs.end(),
                         std::make_move_iterator(outputs.begin()),
                         std::make_move_iterator(outputs.end()));

    return graph;
}

// -------------------------------------------------------------------------------------------------------

class RemoveSingleInputConcatFixture
    : public ov::test::TestsCommon,
      public ::testing::WithParamInterface<std::tuple<Graph /* tranformed */, Graph /* reference */>> {
public:
    void SetUp() override;

public:
    std::shared_ptr<ngraph::Function> function, reference_function;
};

void RemoveSingleInputConcatFixture::SetUp() {
    // TODO: use auto & [transformed_graph, reference_graph] = this->GetParam() when C++17
    Graph transformed_graph;
    Graph reference_graph;
    std::tie(transformed_graph, reference_graph) = this->GetParam();

    function = transformed_graph.createFunction();
    reference_function = reference_graph.createFunction();
}

ngraph::pass::Manager createPassManager() {
    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::RemoveSingleInputConcat>();
    return manager;
}

void execute_test(std::shared_ptr<ngraph::Function> function, std::shared_ptr<ngraph::Function> reference_function) {
    ngraph::pass::Manager pass_manager = createPassManager();
    pass_manager.run_passes(function);
    const FunctionsComparator func_comparator =
        FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

TEST_P(RemoveSingleInputConcatFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(
    RemoveSingleInputConcatTestSuite,
    RemoveSingleInputConcatFixture,
    ::testing::Values(std::make_tuple(createGraph(1 /* n_inputs */, true /* has_concat */, 1 /* n_outputs */),
                                      createGraph(1 /* n_inputs */, false /* has_concat */, 1 /* n_outputs */)),
                      std::make_tuple(createGraph(1 /* n_inputs */, true /* has_concat */, 2 /* n_outputs */),
                                      createGraph(1 /* n_inputs */, false /* has_concat */, 2 /* n_outputs */)),
                      std::make_tuple(createGraph(2 /* n_inputs */, true /* has_concat */, 1 /* n_outputs */),
                                      createGraph(2 /* n_inputs */, true /* has_concat */, 1 /* n_outputs */))));

}  // namespace
}  // namespace testing

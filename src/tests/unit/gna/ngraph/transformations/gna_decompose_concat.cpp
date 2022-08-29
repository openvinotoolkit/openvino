// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <tuple>

#include <ngraph/opsets/opset9.hpp>
#include "transformations/decompose_concat.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

using namespace ngraph;

namespace decomposeConcat {

struct ConcatData {
    size_t num_inputs;
    size_t split_axis = 0;
    size_t leading_shape_product = 1;
    int64_t axis;
    OutputVector concat_parents;
};

std::shared_ptr<Function> create_function(const std::vector<Shape>& input_shapes, const int64_t& concat_axis) {
    ParameterVector params;

    for (const auto& shape : input_shapes) {
        auto param_node = std::make_shared<opset9::Parameter>(element::i32, Shape(shape));
        params.push_back(param_node);
    }

    auto inputs = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(params));
    auto concat = std::make_shared<opset9::Concat>(inputs, concat_axis);
    auto result = std::make_shared<opset9::Result>(concat);

    return std::make_shared<Function>(ResultVector{result}, params);
}

static bool should_decompose(const ParameterVector& params, const int64_t& concat_axis, ConcatData& concat_data) {
    std::vector<Shape> input_shapes;
    concat_data.num_inputs = params.size();

    for (size_t i = 0; i < concat_data.num_inputs; i++) {
        concat_data.concat_parents.push_back(params[i]);
        input_shapes.push_back(params[i]->get_shape());
    }

    concat_data.axis = concat_axis;
    int32_t non_one_axis_count = 0;

    for (int64_t i = 0; i < concat_data.axis; i++) {
        if (input_shapes[0][i] != 1) {
            concat_data.leading_shape_product *= input_shapes[0][i];
            concat_data.split_axis = i;
            non_one_axis_count++;
        }
    }

    // Simple Concats are GNA-compatible already
    if (concat_data.leading_shape_product == 1 ||
        // Difficult cases not yet implemented
        non_one_axis_count > 1) {
        return false;
    }

    return true;
}

static std::shared_ptr<Node> decompose_concat(const ConcatData& concat_data) {
    OutputVector splits;
    OutputVector chunks;

    for (size_t i = 0; i < concat_data.num_inputs; i++) {
        const auto axis_node = opset9::Constant::create(element::i64, Shape{}, {concat_data.split_axis});
        const auto split = std::make_shared<opset9::Split>(concat_data.concat_parents[i], axis_node, concat_data.leading_shape_product);
        splits.push_back(split);
    }

    for (size_t c = 0; c < concat_data.leading_shape_product; c++) {
        OutputVector sub_chunks;
        for (size_t i = 0; i < concat_data.num_inputs; i++) {
            sub_chunks.push_back(splits[i].get_node()->output(c));
        }
        auto new_sub_concat = std::make_shared<opset9::Concat>(sub_chunks, concat_data.axis);
        chunks.push_back(new_sub_concat->output(0));
    }

    auto concat = std::make_shared<opset9::Concat>(chunks, concat_data.split_axis);
    return concat;
}

std::shared_ptr<Function> create_reference_function(const std::vector<Shape>& input_shapes, const int64_t& concat_axis) {
    ParameterVector params;
    ConcatData concat_data = {};
    std::shared_ptr<Node> concat;

    for (const auto& shape : input_shapes) {
        auto param_node = std::make_shared<opset9::Parameter>(element::i32, Shape(shape));
        params.push_back(param_node);
    }

    if (should_decompose(params, concat_axis, concat_data)) {
        concat = decompose_concat(concat_data);
    } else {
        auto inputs = helpers::convert2OutputVector(helpers::castOps2Nodes<op::Parameter>(params));
        concat = std::make_shared<opset9::Concat>(inputs, concat_axis);
    }

    auto result = std::make_shared<opset9::Result>(concat);
    return std::make_shared<Function>(ResultVector{result}, params);
}

} // namespace decomposeConcat

// ---------------------------------------------------------------------------------------------------------------------

using FixtureInputShapes = std::tuple<std::vector<Shape>, int64_t>;

class DecomposeConcatFixture
    : public CommonTestUtils::TestsCommon,
      public ::testing::WithParamInterface<FixtureInputShapes> {
public:
    void SetUp() override;

public:
    std::shared_ptr<Function> function, reference_function;
};

void DecomposeConcatFixture::SetUp() {
    std::vector<Shape> input_shapes;
    int64_t concat_axis;
    std::tie(input_shapes, concat_axis) = this->GetParam();

    function = decomposeConcat::create_function(input_shapes, concat_axis);
    reference_function = decomposeConcat::create_reference_function(input_shapes, concat_axis);
}

void execute_test(std::shared_ptr<Function> function, std::shared_ptr<Function> reference_function) {
    pass::Manager manager, manager_ref;
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::DecomposeConcat>();
    manager.run_passes(function);
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

std::vector<std::vector<Shape>> input_shapes = {{{10, 10, 10, 10}, {10, 10, 10, 10}},
                                                {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}},
                                                {{10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}, {10, 10, 10, 10}}};

std::vector<int64_t> concat_axis = {
    0,
    1,
    2,
};

TEST_P(DecomposeConcatFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(DecomposeConcatTestSuite,
                         DecomposeConcatFixture,
                         ::testing::Combine(
                            ::testing::ValuesIn(input_shapes),
                            ::testing::ValuesIn(concat_axis)));

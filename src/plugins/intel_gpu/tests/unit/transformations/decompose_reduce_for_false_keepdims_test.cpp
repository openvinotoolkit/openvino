// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/pass/manager.hpp>
#include <plugin/transformations/decompose_reduce_for_false_keepdims.hpp>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <tuple>

#include "intel_gpu/primitives/reduce.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "test_utils.h"

using namespace testing;
using namespace ::tests;
using namespace cldnn;

using InputShape = ov::PartialShape;
using KeepDims = bool;
using ReduceAxes = std::vector<int64_t>;
using ReduceType = cldnn::reduce_mode;
using ReshapeShape = std::vector<size_t>;
using NeedDecompose = bool;

class ReduceDecomposeTests
    : public ::testing::Test,
      public testing::WithParamInterface<
          std::tuple<ReduceType, InputShape, ReduceAxes, KeepDims, NeedDecompose, ReshapeShape>> {
public:
    std::shared_ptr<ov::Model> fc;
    bool need_decompose;
    ReshapeShape result_shape;

    void SetUp() override {
        const auto& reduce_type = std::get<0>(GetParam());
        const auto& input_shape = std::get<1>(GetParam());
        const auto& axes = std::get<2>(GetParam());
        const auto& keep_dims = std::get<3>(GetParam());
        need_decompose = std::get<4>(GetParam());
        result_shape = std::get<5>(GetParam());

        fc = get_transformed_function(input_shape, axes, reduce_type, keep_dims);
    }

    static std::shared_ptr<ov::Model> get_transformed_function(const ov::PartialShape& input_shape,
                                                                      const std::vector<int64_t>& axes,
                                                                      const ReduceType& reduce_type,
                                                                      const bool keep_dim) {
        auto param = std::make_shared<ov::opset10::Parameter>(ov::element::f32, input_shape);
        if (reduce_type == reduce_mode::logical_or || reduce_type == reduce_mode::logical_and)
            param = std::make_shared<ov::opset10::Parameter>(ov::element::boolean, input_shape);

        ov::Output<ov::Node> input = param->output(0);

        auto axes_const = ov::opset10::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);

        if (reduce_type == reduce_mode::sum)
            input = std::make_shared<ov::opset10::ReduceSum>(input, axes_const, keep_dim);
        else if (reduce_type == reduce_mode::mean)
            input = std::make_shared<ov::opset10::ReduceMean>(input, axes_const, keep_dim);
        else if (reduce_type == reduce_mode::min)
            input = std::make_shared<ov::opset10::ReduceMin>(input, axes_const, keep_dim);
        else if (reduce_type == reduce_mode::max)
            input = std::make_shared<ov::opset10::ReduceMax>(input, axes_const, keep_dim);
        else if (reduce_type == reduce_mode::prod)
            input = std::make_shared<ov::opset10::ReduceProd>(input, axes_const, keep_dim);
        else if (reduce_type == reduce_mode::logical_or)
            input = std::make_shared<ov::opset10::ReduceLogicalOr>(input, axes_const, keep_dim);
        else if (reduce_type == reduce_mode::logical_and)
            input = std::make_shared<ov::opset10::ReduceLogicalAnd>(input, axes_const, keep_dim);
        else
            throw std::runtime_error("Invalid reduce type for this test-case.");

        return std::make_shared<ov::Model>(ov::NodeVector{input.get_node_shared_ptr()},
                                                  ov::ParameterVector{param});
    }
};

TEST_P(ReduceDecomposeTests, CompareFunctions) {
    ov::pass::Manager m;
    m.set_per_pass_validation(false);
    m.register_pass<ov::pass::InitNodeInfo>();
    m.register_pass<ov::intel_gpu::DecomposeReduceForFalseKeepDims>();
    m.run_passes(fc);

    bool success = false;
    ov::Shape output_shape;
    for (auto& ops : fc->get_ops()) {
        std::string type_name(ops->get_type_name());

        if (type_name.find("Reshape") != std::string::npos) {
            success = true;
        }
        else if (type_name.find("Result") != std::string::npos) {
            output_shape = ops->get_shape();
        }
    }
    ASSERT_TRUE(success == need_decompose);
    ASSERT_TRUE(output_shape == result_shape);
}

INSTANTIATE_TEST_SUITE_P(ReduceDecomposeForFalseKeepdims,
                         ReduceDecomposeTests,
                         testing::Values(std::make_tuple(reduce_mode::prod,
                                                         InputShape{32, 32, 32, 32},
                                                         ReduceAxes{0, 2, 3},
                                                         KeepDims{false},
                                                         true,
                                                         ReshapeShape{32}),
                                         std::make_tuple(reduce_mode::sum,
                                                         InputShape{16, 3, 32, 32},
                                                         ReduceAxes{0, 2, 3},
                                                         KeepDims{false},
                                                         true,
                                                         ReshapeShape{3}),
                                         std::make_tuple(reduce_mode::mean,
                                                         InputShape{16, 3, 32, 32},
                                                         ReduceAxes{0, 2, 3},
                                                         KeepDims{false},
                                                         true,
                                                         ReshapeShape{3}),
                                         std::make_tuple(reduce_mode::min,
                                                         InputShape{16, 3, 32, 32},
                                                         ReduceAxes{0, 2, 3},
                                                         KeepDims{false},
                                                         true,
                                                         ReshapeShape{3}),
                                         std::make_tuple(reduce_mode::max,
                                                         InputShape{16, 3, 32, 32},
                                                         ReduceAxes{0, 2, 3},
                                                         KeepDims{false},
                                                         true,
                                                         ReshapeShape{3}),
                                         std::make_tuple(reduce_mode::max,
                                                         InputShape{8, 3, 64, 64},
                                                         ReduceAxes{0, 2, 3},
                                                         KeepDims{false},
                                                         true,
                                                         ReshapeShape{3})));

INSTANTIATE_TEST_SUITE_P(ReduceDecomposeForFalseKeepdimsNotCase,
                         ReduceDecomposeTests,
                         testing::Values(std::make_tuple(reduce_mode::max,
                                                         InputShape{32, 32, 32, 32},
                                                         ReduceAxes{0, 2},
                                                         KeepDims{false},
                                                         false,
                                                         ReshapeShape{32, 32}),
                                         std::make_tuple(reduce_mode::max,
                                                         InputShape{1, 3, 64, 64},
                                                         ReduceAxes{0, 3},
                                                         KeepDims{false},
                                                         false,
                                                         ReshapeShape{3, 64}),
                                         std::make_tuple(reduce_mode::max,
                                                         InputShape{32, 32, 32, 32},
                                                         ReduceAxes{0},
                                                         KeepDims{false},
                                                         false,
                                                         ReshapeShape{32, 32, 32}),
                                         std::make_tuple(reduce_mode::logical_and,
                                                         InputShape{16, 3, 32, 32},
                                                         ReduceAxes{0, 2, 3},
                                                         KeepDims{false},
                                                         false,
                                                         ReshapeShape{3}),
                                         std::make_tuple(reduce_mode::logical_or,
                                                         InputShape{16, 3, 32, 32},
                                                         ReduceAxes{0, 2, 3},
                                                         KeepDims{false},
                                                         false,
                                                         ReshapeShape{3}),
                                         std::make_tuple(reduce_mode::max,
                                                         InputShape{1, 3, 64, 64},
                                                         ReduceAxes{0},
                                                         KeepDims{false},
                                                         false,
                                                         ReshapeShape{3, 64, 64})));

INSTANTIATE_TEST_SUITE_P(ReduceDecomposeForTrueKeepdims,
                         ReduceDecomposeTests,
                         testing::Values(std::make_tuple(reduce_mode::max,
                                                         InputShape{32, 32, 32, 32},
                                                         ReduceAxes{0, 2, 3},
                                                         KeepDims{true},
                                                         false,
                                                         ReshapeShape{1, 32, 1, 1}),
                                         std::make_tuple(reduce_mode::max,
                                                         InputShape{1, 3, 64, 64},
                                                         ReduceAxes{0, 2, 3},
                                                         KeepDims{true},
                                                         false,
                                                         ReshapeShape{1, 3, 1, 1}),
                                         std::make_tuple(reduce_mode::max,
                                                         InputShape{32, 32, 32, 32},
                                                         ReduceAxes{0, 2},
                                                         KeepDims{true},
                                                         false,
                                                         ReshapeShape{1, 32, 1, 32})));

TEST(DecomposeReduceForFalseKeepDims, Negative) {
    auto f =
        ReduceDecomposeTests::get_transformed_function(ov::PartialShape::dynamic(), {3}, reduce_mode::max, true);
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::DecomposeReduceForFalseKeepDims>();
    OV_ASSERT_NO_THROW(manager.run_passes(f));
}

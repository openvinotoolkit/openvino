// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"

#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <queue>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/op_conversions/convert_reduce_to_pooling.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

using InputShape = ngraph::PartialShape;
using ReduceAxes = std::vector<int64_t>;
using KeepDims = bool;
using ReduceType = std::shared_ptr<ngraph::Node>;

struct ReduceToPoolParams {
    std::vector<int64_t> reshape_begin, reshape_end;
    ngraph::Shape pooling_kernel;

    ReduceToPoolParams() = default;

    ReduceToPoolParams(const std::vector<int64_t> & begin, const ngraph::Shape & pooling_kernel, const std::vector<int64_t> & end)
        : pooling_kernel(pooling_kernel), reshape_begin(begin), reshape_end(end) {}
};

class ConvertReduceToPoolingTests: public CommonTestUtils::TestsCommon,
                                   public testing::WithParamInterface<std::tuple<ReduceType, InputShape, ReduceAxes, KeepDims, ReduceToPoolParams> > {
public:
    std::shared_ptr<ngraph::Function> f, f_ref;

    void SetUp() override {
        const auto& reduce_type      = std::get<0>(GetParam());
        const auto& input_shape      = std::get<1>(GetParam());
        const auto& axes             = std::get<2>(GetParam());
        const auto& keep_dims        = std::get<3>(GetParam());
        const auto& reference_params = std::get<4>(GetParam());

        f = get_initial_function(input_shape, axes, reduce_type, keep_dims);
        f_ref = get_reference_function(input_shape, reduce_type, reference_params);
    }

    static std::shared_ptr<ngraph::Function> get_initial_function(const ngraph::PartialShape & input_shape,
                                                           const std::vector<int64_t> & axes,
                                                           const ReduceType & reduce_type,
                                                           const bool keep_dims) {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto axes_const = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{axes.size()}, axes);

        auto split_axis_const = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto split = std::make_shared<ngraph::opset1::Split>(input,  split_axis_const, 2);
        auto reduce = std::make_shared<ngraph::opset1::ReduceMax>(split->output(0), axes_const, keep_dims);
        // TODO: need to add set_keep_dims method to Reduce ops and line above will be replaced with
        // reduce = reduce_type->copy_with_new_inputs({input, axes_const});
        // reduce->set_keep_dims(keep_dims);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{reduce}, ngraph::ParameterVector{input});
    }

    static std::shared_ptr<ngraph::Function> get_reference_function(const ngraph::PartialShape & input_shape,
                                                             const ReduceType & reduce,
                                                             const ReduceToPoolParams & params) {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);

        ngraph::Output<ngraph::Node> input = param->output(0);

        auto split_axis_const = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        auto split = std::make_shared<ngraph::opset1::Split>(input,  split_axis_const, 2);
        input = split->output(0);

        if (!params.reshape_begin.empty()) {
            input = std::make_shared<ngraph::opset1::Reshape>(input,
                    ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{params.reshape_begin.size()}, params.reshape_begin), false);
        }

        if (!params.pooling_kernel.empty()) {
            if (reduce->get_type_info() == ngraph::opset1::ReduceMax::type_info) {
                input = std::make_shared<ngraph::opset1::MaxPool>(input, ngraph::Strides{1, 1}, ngraph::Shape{0, 0},
                        ngraph::Shape{0, 0}, params.pooling_kernel, ngraph::op::RoundingType::FLOOR /*any*/);
            } else if (reduce->get_type_info() == ngraph::opset1::ReduceMean::type_info ||
                       reduce->get_type_info() == ngraph::opset1::ReduceSum::type_info) {
                input = std::make_shared<ngraph::opset1::AvgPool>(input, ngraph::Strides{1, 1}, ngraph::Shape{0, 0},
                        ngraph::Shape{0, 0}, params.pooling_kernel, false /*any*/, ngraph::op::RoundingType::FLOOR /*any*/);
            } else {
                throw ngraph::ngraph_error("Unsupported Reduce type!");
            }
        }

        // TODO: handle multiply_value for ReduceSum case when set_keep_dims is ready

        if (!params.reshape_end.empty()) {
            input = std::make_shared<ngraph::opset1::Reshape>(input,
                    ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{params.reshape_end.size()}, params.reshape_end), true);
        }

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{input.get_node_shared_ptr()}, ngraph::ParameterVector{param});
    }
};

TEST_P(ConvertReduceToPoolingTests, CompareFunctions) {
    ngraph::pass::InitNodeInfo().run_on_function(f);
    ngraph::pass::ConvertReduceToPooling().run_on_function(f);
    ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

#define MAX std::make_shared<ngraph::opset1::ReduceMax>()

INSTANTIATE_TEST_SUITE_P(ReduceToMaxPooling, ConvertReduceToPoolingTests,
        testing::Values(std::make_tuple(MAX, InputShape{2, 3, 64, 64},  ReduceAxes{3},    KeepDims{true}, ReduceToPoolParams({}, {1, 64}, {})),
                        std::make_tuple(MAX, InputShape{2, 3, 64, 64},  ReduceAxes{3, 2}, KeepDims{true}, ReduceToPoolParams({}, {64, 64}, {}))));

INSTANTIATE_TEST_SUITE_P(ReduceToReshape, ConvertReduceToPoolingTests,
        testing::Values(std::make_tuple(MAX, InputShape{2, 3, 64, 1}, ReduceAxes{3},       KeepDims{false}, ReduceToPoolParams({1, 3, 64}, {}, {})),
                        std::make_tuple(MAX, InputShape{2, 3},        ReduceAxes{-2},      KeepDims{false}, ReduceToPoolParams({3}, {}, {})),
                        std::make_tuple(MAX, InputShape{2, 3, 1},     ReduceAxes{2},       KeepDims{false}, ReduceToPoolParams({1, 3}, {}, {})),
                        std::make_tuple(MAX, InputShape{2, 3, 1, 1},  ReduceAxes{0, 3, 2}, KeepDims{false}, ReduceToPoolParams({3}, {}, {}))));

INSTANTIATE_TEST_SUITE_P(ReduceToReshapePoolReshape, ConvertReduceToPoolingTests,
        testing::Values(std::make_tuple(MAX, InputShape{2, 3, 3},    ReduceAxes{1, 2},    KeepDims{false}, ReduceToPoolParams({1, 1, 9, 1}, {9, 1}, {1})),
                        std::make_tuple(MAX, InputShape{2, 9},       ReduceAxes{-1},      KeepDims{true},  ReduceToPoolParams({1, 1, 9, 1}, {9, 1}, {1, 1})),
                        std::make_tuple(MAX, InputShape{2, 3, 4, 1}, ReduceAxes{1, 3, 2}, KeepDims{false}, ReduceToPoolParams({1, 1, 12, 1}, {12, 1}, {1}))));

TEST(ConvertReduceToPooling, Negative) {
    auto f = ConvertReduceToPoolingTests::get_initial_function(
            ngraph::PartialShape::dynamic(), {3}, MAX, true);
    ASSERT_NO_THROW(ngraph::pass::ConvertReduceToPooling().run_on_function(f));
}

#undef MAX

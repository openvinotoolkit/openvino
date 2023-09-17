// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <legacy/ngraph_ops/convolution_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_convolutions.hpp>
#include <map>
#include <memory>
#include <ngraph/pass/visualize_tree.hpp>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/manager.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"

using namespace testing;
using namespace ov;
using namespace ov::opset1;

using InputShape = PartialShape;
using WeightsShape = PartialShape;

class ConvertConvolutionTest : public ov::test::TestsCommon,
                               public testing::WithParamInterface<std::tuple<InputShape, WeightsShape>> {
public:
    std::shared_ptr<Model> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& weights_shape = std::get<1>(GetParam());

        f = get_initial_model(input_shape, weights_shape);
        f_ref = get_reference_function(input_shape, weights_shape);
    }

private:
    std::shared_ptr<Model> get_initial_model(const PartialShape& input_shape, const PartialShape& weights_shape) {
        assert(weights_shape.is_static());
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(element::f32, input_shape);
        auto weights = Constant::create(element::f32, weights_shape.to_shape(), {1});
        auto conv = std::make_shared<Convolution>(input,
                                                  weights,
                                                  Strides(spatial_dims, 1),
                                                  CoordinateDiff(spatial_dims, 0),
                                                  CoordinateDiff(spatial_dims, 0),
                                                  Strides(spatial_dims, 1));

        return std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }

    std::shared_ptr<Model> get_reference_function(const PartialShape& input_shape, const PartialShape& weights_shape) {
        assert(weights_shape.is_static());
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(element::f32, input_shape);
        auto weights = Constant::create(element::f32, weights_shape.get_shape(), {1});
        auto conv = std::make_shared<ngraph::op::ConvolutionIE>(input,
                                                                weights,
                                                                Strides(spatial_dims, 1),
                                                                Strides(spatial_dims, 1),
                                                                CoordinateDiff(spatial_dims, 0),
                                                                CoordinateDiff(spatial_dims, 0),
                                                                element::f32);

        return std::make_shared<Model>(NodeVector{conv}, ParameterVector{input});
    }
};

TEST_P(ConvertConvolutionTest, CompareFunctions) {
    const auto orig_shape = f->get_output_partial_shape(0);
    pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::ConvertConvolutions>();
    manager.run_passes(f);

    ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
    ASSERT_TRUE(orig_shape.same_scheme(f->get_output_partial_shape(0)))
        << "Shape " << orig_shape << " is not equal to " << f->get_output_partial_shape(0);
}

INSTANTIATE_TEST_SUITE_P(ConvertConvolution,
                         ConvertConvolutionTest,
                         testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN},
                                                         WeightsShape{8, 3, 1, 2, 3}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64, 64}, WeightsShape{8, 3, 1, 2, 3}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64, 64}, WeightsShape{9, 3, 2, 3, 1}),
                                         std::make_tuple(InputShape{3, 3, DYN, 64, 64}, WeightsShape{6, 3, 3, 4, 2}),
                                         std::make_tuple(InputShape{3, 3, 64, DYN, 64}, WeightsShape{5, 3, 3, 4, 3}),
                                         std::make_tuple(InputShape{3, 3, 64, 64, DYN}, WeightsShape{5, 3, 3, 4, 3}),
                                         std::make_tuple(InputShape{1, 3, 64, 64}, WeightsShape{6, 3, 1, 1}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, WeightsShape{7, 3, 1, 1}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64}, WeightsShape{8, 3, 1, 2}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64}, WeightsShape{9, 3, 2, 3}),
                                         std::make_tuple(InputShape{3, 3, DYN, 64}, WeightsShape{6, 3, 3, 4}),
                                         std::make_tuple(InputShape{3, 3, 64, DYN}, WeightsShape{5, 3, 3, 4}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN}, WeightsShape{5, 3, 1}),
                                         std::make_tuple(InputShape{DYN, 3, 10}, WeightsShape{3, 3, 1}),
                                         std::make_tuple(InputShape{2, DYN, 9}, WeightsShape{2, 3, 2}),
                                         std::make_tuple(InputShape{3, 3, DYN}, WeightsShape{1, 3, 3})));

TEST(ConvertConvolutionTest, GroupConvolutionWithReshape) {
    PartialShape input_shape{1, 6, 64, 64};
    PartialShape weights_shape_before{2 * 3, 3, 5, 5};
    PartialShape weights_shape_after{2, 3, 3, 5, 5};
    size_t group = 2;

    std::shared_ptr<Model> f, f_ref;
    {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(element::f32, input_shape);
        auto weights = std::make_shared<Parameter>(element::f32, weights_shape_before);
        auto reshape = std::make_shared<Reshape>(
            weights,
            Constant::create(element::i64,
                             Shape{static_cast<size_t>(weights_shape_after.rank().get_length())},
                             weights_shape_after.to_shape()),
            true);
        auto conv = std::make_shared<GroupConvolution>(input,
                                                       reshape,
                                                       Strides(spatial_dims, 1),
                                                       CoordinateDiff(spatial_dims, 0),
                                                       CoordinateDiff(spatial_dims, 0),
                                                       Strides(spatial_dims, 1));

        f = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input, weights});
    }

    {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(element::f32, input_shape);
        auto weights = std::make_shared<Parameter>(element::f32, weights_shape_before);
        auto conv = std::make_shared<ngraph::op::ConvolutionIE>(input,
                                                                weights,
                                                                Strides(spatial_dims, 1),
                                                                Strides(spatial_dims, 1),
                                                                CoordinateDiff(spatial_dims, 0),
                                                                CoordinateDiff(spatial_dims, 0),
                                                                element::f32,
                                                                group);

        f_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input, weights});
    }

    pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::ConvertConvolutions>();
    manager.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(ConvertConvolutionTest, GroupConvolutionWithReshapeNeg) {
    PartialShape input_shape{1, 6, 64, 64};
    PartialShape weights_shape_before{3, 2, 3, 5, 5};
    PartialShape weights_shape_after{2, 3, 3, 5, 5};
    PartialShape weights_shape_ref{2 * 3, 3, 5, 5};
    size_t group = 2;

    std::shared_ptr<Model> f, f_ref;
    {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(element::f32, input_shape);
        auto weights = std::make_shared<Parameter>(element::f32, weights_shape_before);
        auto reshape = std::make_shared<Reshape>(
            weights,
            Constant::create(element::i64,
                             Shape{static_cast<size_t>(weights_shape_after.rank().get_length())},
                             weights_shape_after.to_shape()),
            true);
        auto conv = std::make_shared<GroupConvolution>(input,
                                                       reshape,
                                                       Strides(spatial_dims, 1),
                                                       CoordinateDiff(spatial_dims, 0),
                                                       CoordinateDiff(spatial_dims, 0),
                                                       Strides(spatial_dims, 1));

        f = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input, weights});
    }

    {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(element::f32, input_shape);
        auto weights_param = std::make_shared<Parameter>(element::f32, weights_shape_before);
        auto reshape = std::make_shared<Reshape>(
            weights_param,
            Constant::create(element::i64,
                             Shape{static_cast<size_t>(weights_shape_after.rank().get_length())},
                             weights_shape_after.to_shape()),
            true);
        auto weights = std::make_shared<Reshape>(
            reshape,
            Constant::create(element::i64,
                             Shape{static_cast<size_t>(weights_shape_ref.rank().get_length())},
                             weights_shape_ref.to_shape()),
            true);
        auto conv = std::make_shared<ngraph::op::ConvolutionIE>(input,
                                                                weights,
                                                                Strides(spatial_dims, 1),
                                                                Strides(spatial_dims, 1),
                                                                CoordinateDiff(spatial_dims, 0),
                                                                CoordinateDiff(spatial_dims, 0),
                                                                element::f32,
                                                                group);

        f_ref = std::make_shared<Model>(NodeVector{conv}, ParameterVector{input, weights_param});
    }

    pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::ConvertConvolutions>();
    manager.run_passes(f);
    // FIXME: 42956
    // ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

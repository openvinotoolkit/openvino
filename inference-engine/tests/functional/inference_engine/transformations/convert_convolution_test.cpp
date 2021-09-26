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
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/op_conversions/convert_convolutions.hpp>
#include <ngraph_ops/convolution_ie.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;
using namespace ngraph::opset1;

using InputShape = PartialShape;
using WeightsShape = PartialShape;

class ConvertConvolutionTest: public CommonTestUtils::TestsCommon,
                              public testing::WithParamInterface<std::tuple<InputShape, WeightsShape> > {
public:
    std::shared_ptr<Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& weights_shape = std::get<1>(GetParam());

        f = get_initial_function(input_shape, weights_shape);
        f_ref = get_reference_function(input_shape, weights_shape);
    }

private:
    std::shared_ptr<Function> get_initial_function(const PartialShape & input_shape,
                                                           const PartialShape & weights_shape) {
        assert(weights_shape.is_static());
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(element::f32, input_shape);
        auto weights = Constant::create(element::f32, weights_shape.to_shape(), {1});
        auto conv = std::make_shared<Convolution>(input, weights, Strides(spatial_dims, 1),
                CoordinateDiff(spatial_dims, 0), CoordinateDiff(spatial_dims, 0), Strides(spatial_dims, 1));

        return std::make_shared<Function>(NodeVector{conv}, ParameterVector{input});
    }

    std::shared_ptr<Function> get_reference_function(const PartialShape & input_shape,
                                                             const PartialShape & weights_shape) {
        assert(weights_shape.is_static());
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(element::f32, input_shape);
        auto weights = Constant::create(element::f32, weights_shape.get_shape(), {1});
        auto conv = std::make_shared<op::ConvolutionIE>(input, weights, Strides(spatial_dims, 1), Strides(spatial_dims, 1),
                CoordinateDiff(spatial_dims, 0), CoordinateDiff(spatial_dims, 0), element::f32);

        return std::make_shared<Function>(NodeVector{conv}, ParameterVector{input});
    }
};

TEST_P(ConvertConvolutionTest, CompareFunctions) {
    const auto & orig_shape = f->get_output_partial_shape(0);
    pass::InitNodeInfo().run_on_function(f);
    pass::ConvertConvolutions().run_on_function(f);
    ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
    ASSERT_TRUE(orig_shape.same_scheme(f->get_output_partial_shape(0))) << "Shape " << orig_shape << " is not equal to " << f->get_output_partial_shape(0);
}

INSTANTIATE_TEST_SUITE_P(ConvertConvolution, ConvertConvolutionTest,
        testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, WeightsShape{8, 3, 1, 2, 3}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64, 64}, WeightsShape{8, 3, 1, 2, 3}),
                        std::make_tuple(InputShape{2, DYN, 64, 64, 64}, WeightsShape{9, 3, 2, 3, 1}),
                        std::make_tuple(InputShape{3, 3, DYN, 64, 64},  WeightsShape{6, 3, 3, 4, 2}),
                        std::make_tuple(InputShape{3, 3, 64, DYN, 64},  WeightsShape{5, 3, 3, 4, 3}),
                        std::make_tuple(InputShape{3, 3, 64, 64, DYN},  WeightsShape{5, 3, 3, 4, 3}),
                        std::make_tuple(InputShape{1, 3, 64, 64},       WeightsShape{6, 3, 1, 1}),
                        std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, WeightsShape{7, 3, 1, 1}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64},     WeightsShape{8, 3, 1, 2}),
                        std::make_tuple(InputShape{2, DYN, 64, 64},     WeightsShape{9, 3, 2, 3}),
                        std::make_tuple(InputShape{3, 3, DYN, 64},      WeightsShape{6, 3, 3, 4}),
                        std::make_tuple(InputShape{3, 3, 64, DYN},      WeightsShape{5, 3, 3, 4}),
                        std::make_tuple(InputShape{DYN, DYN, DYN},      WeightsShape{5, 3, 1}),
                        std::make_tuple(InputShape{DYN, 3, 10},         WeightsShape{3, 3, 1}),
                        std::make_tuple(InputShape{2, DYN, 9},          WeightsShape{2, 3, 2}),
                        std::make_tuple(InputShape{3, 3, DYN},          WeightsShape{1, 3, 3})));

TEST(ConvertConvolutionTest, GroupConvolutionWithReshape) {
    PartialShape input_shape{1, 6, 64, 64};
    PartialShape weights_shape_before{2 * 3, 3, 5, 5};
    PartialShape weights_shape_after{2, 3, 3, 5, 5};
    size_t group = 2;

    std::shared_ptr<Function> f, f_ref;
    {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(element::f32, input_shape);
        auto weights = std::make_shared<Parameter>(element::f32, weights_shape_before);
        auto reshape = std::make_shared<Reshape>(weights, Constant::create(element::i64,
                Shape{static_cast<size_t>(weights_shape_after.rank().get_length())}, weights_shape_after.to_shape()), true);
        auto conv = std::make_shared<GroupConvolution>(input, reshape, Strides(spatial_dims, 1),
                CoordinateDiff(spatial_dims, 0), CoordinateDiff(spatial_dims, 0), Strides(spatial_dims, 1));

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{input, weights});
    }

    {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(element::f32, input_shape);
        auto weights = std::make_shared<Parameter>(element::f32, weights_shape_before);
        auto conv = std::make_shared<op::ConvolutionIE>(input, weights, Strides(spatial_dims, 1), Strides(spatial_dims, 1),
                CoordinateDiff(spatial_dims, 0), CoordinateDiff(spatial_dims, 0), element::f32, group);

        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{input, weights});
    }

    pass::Manager manager;
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertConvolutions>();
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

    std::shared_ptr<Function> f, f_ref;
    {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(element::f32, input_shape);
        auto weights = std::make_shared<Parameter>(element::f32, weights_shape_before);
        auto reshape = std::make_shared<Reshape>(weights, Constant::create(element::i64,
                Shape{static_cast<size_t>(weights_shape_after.rank().get_length())}, weights_shape_after.to_shape()), true);
        auto conv = std::make_shared<GroupConvolution>(input, reshape, Strides(spatial_dims, 1),
                CoordinateDiff(spatial_dims, 0), CoordinateDiff(spatial_dims, 0), Strides(spatial_dims, 1));

        f = std::make_shared<Function>(NodeVector{conv}, ParameterVector{input, weights});
    }

    {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<Parameter>(element::f32, input_shape);
        auto weights_param = std::make_shared<Parameter>(element::f32, weights_shape_before);
        auto reshape = std::make_shared<Reshape>(weights_param, Constant::create(element::i64,
                Shape{static_cast<size_t>(weights_shape_after.rank().get_length())}, weights_shape_after.to_shape()), true);
        auto weights = std::make_shared<Reshape>(reshape, Constant::create(element::i64,
                Shape{static_cast<size_t>(weights_shape_ref.rank().get_length())}, weights_shape_ref.to_shape()), true);
        auto conv = std::make_shared<op::ConvolutionIE>(input, weights, Strides(spatial_dims, 1), Strides(spatial_dims, 1),
                CoordinateDiff(spatial_dims, 0), CoordinateDiff(spatial_dims, 0), element::f32, group);

        f_ref = std::make_shared<Function>(NodeVector{conv}, ParameterVector{input, weights_param});
    }

    pass::Manager manager;
    manager.register_pass<pass::InitNodeInfo>();
    manager.register_pass<pass::ConvertConvolutions>();
    manager.run_passes(f);
    // FIXME: 42956
    // ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
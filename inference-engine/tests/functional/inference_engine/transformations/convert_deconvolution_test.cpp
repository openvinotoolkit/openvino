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
#include <ngraph_ops/deconvolution_ie.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

using InputShape = ngraph::PartialShape;
using WeightsShape = ngraph::Shape;

class ConvertDeconvolutionTest: public CommonTestUtils::TestsCommon,
                                public testing::WithParamInterface<std::tuple<InputShape, WeightsShape> > {
public:
    std::shared_ptr<ngraph::Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& weights_shape = std::get<1>(GetParam());

        f = get_initial_function(input_shape, weights_shape);
        f_ref = get_reference_function(input_shape, weights_shape);
    }

private:
    std::shared_ptr<ngraph::Function> get_initial_function(const ngraph::PartialShape & input_shape,
                                                           const ngraph::Shape & weights_shape) {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, weights_shape, {1});
        auto conv = std::make_shared<ngraph::opset1::ConvolutionBackpropData>(input, weights, ngraph::Strides(spatial_dims, 1),
                ngraph::CoordinateDiff(spatial_dims, 0), ngraph::CoordinateDiff(spatial_dims, 0), ngraph::Strides(spatial_dims, 1));

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{conv}, ngraph::ParameterVector{input});
    }

    std::shared_ptr<ngraph::Function> get_reference_function(const ngraph::PartialShape & input_shape,
                                                             const ngraph::Shape & weights_shape) {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, weights_shape, {1});
        auto conv = std::make_shared<ngraph::op::DeconvolutionIE>(input, weights, ngraph::Strides(spatial_dims, 1), ngraph::Strides(spatial_dims, 1),
                ngraph::CoordinateDiff(spatial_dims, 0), ngraph::CoordinateDiff(spatial_dims, 0), ngraph::element::f32);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{conv}, ngraph::ParameterVector{input});
    }
};

TEST_P(ConvertDeconvolutionTest, CompareFunctions) {
    const auto & orig_shape = f->get_output_partial_shape(0);
    ngraph::pass::InitNodeInfo().run_on_function(f);
    ngraph::pass::ConvertConvolutions().run_on_function(f);
    ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
    ASSERT_TRUE(orig_shape.same_scheme(f->get_output_partial_shape(0))) << "Shape " << orig_shape << " is not equal to " << f->get_output_partial_shape(0);
}

INSTANTIATE_TEST_CASE_P(ConvertDeconvolution, ConvertDeconvolutionTest,
        testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, WeightsShape{3, 8, 1, 2, 3}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64, 64}, WeightsShape{3, 8, 1, 2, 3}),
                        std::make_tuple(InputShape{2, DYN, 64, 64, 64}, WeightsShape{3, 9, 2, 3, 1}),
                        std::make_tuple(InputShape{3, 3, DYN, 64, 64},  WeightsShape{3, 6, 3, 4, 2}),
                        std::make_tuple(InputShape{3, 3, 64, DYN, 64},  WeightsShape{3, 5, 3, 4, 3}),
                        std::make_tuple(InputShape{3, 3, 64, 64, DYN},  WeightsShape{3, 3, 3, 4, 3}),
                        std::make_tuple(InputShape{1, 3, 64, 64},       WeightsShape{3, 6, 1, 1}),
                        std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, WeightsShape{3, 7, 1, 1}),
                        std::make_tuple(InputShape{DYN, 3, 64, 64},     WeightsShape{3, 8, 1, 2}),
                        std::make_tuple(InputShape{2, DYN, 64, 64},     WeightsShape{3, 9, 2, 3}),
                        std::make_tuple(InputShape{3, 3, DYN, 64},      WeightsShape{3, 6, 3, 4}),
                        std::make_tuple(InputShape{3, 3, 64, DYN},      WeightsShape{3, 5, 3, 4}),
                        std::make_tuple(InputShape{DYN, DYN, DYN},      WeightsShape{3, 5, 1}),
                        std::make_tuple(InputShape{DYN, 3, 10},         WeightsShape{3, 3, 1}),
                        std::make_tuple(InputShape{2, DYN, 9},          WeightsShape{3, 2, 2}),
                        std::make_tuple(InputShape{3, 3, DYN},          WeightsShape{3, 1, 3})));

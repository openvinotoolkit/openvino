// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>
#include <legacy/ngraph_ops/convolution_ie.hpp>
#include <legacy/ngraph_ops/deconvolution_ie.hpp>
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

using InputShape = ov::PartialShape;
using WeightsShape = ov::Shape;

class ConvertDeconvolutionTest : public ov::test::TestsCommon,
                                 public testing::WithParamInterface<std::tuple<InputShape, WeightsShape>> {
public:
    std::shared_ptr<ov::Model> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& weights_shape = std::get<1>(GetParam());

        f = get_initial_model(input_shape, weights_shape);
        f_ref = get_reference_function(input_shape, weights_shape);
    }

private:
    std::shared_ptr<ov::Model> get_initial_model(const ov::PartialShape& input_shape, const ov::Shape& weights_shape) {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
        auto weights = ov::opset1::Constant::create(ov::element::f32, weights_shape, {1});
        auto conv = std::make_shared<ov::opset1::ConvolutionBackpropData>(input,
                                                                          weights,
                                                                          ov::Strides(spatial_dims, 1),
                                                                          ov::CoordinateDiff(spatial_dims, 0),
                                                                          ov::CoordinateDiff(spatial_dims, 0),
                                                                          ov::Strides(spatial_dims, 1));

        return std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }

    std::shared_ptr<ov::Model> get_reference_function(const ov::PartialShape& input_shape,
                                                      const ov::Shape& weights_shape) {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, input_shape);
        auto weights = ov::opset1::Constant::create(ov::element::f32, weights_shape, {1});
        auto conv = std::make_shared<ngraph::op::DeconvolutionIE>(input,
                                                                  weights,
                                                                  ov::Strides(spatial_dims, 1),
                                                                  ov::Strides(spatial_dims, 1),
                                                                  ov::CoordinateDiff(spatial_dims, 0),
                                                                  ov::CoordinateDiff(spatial_dims, 0),
                                                                  ov::element::f32);

        return std::make_shared<ov::Model>(ov::NodeVector{conv}, ov::ParameterVector{input});
    }
};

TEST_P(ConvertDeconvolutionTest, CompareFunctions) {
    const auto orig_shape = f->get_output_partial_shape(0);
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::ConvertConvolutions>();
    manager.run_passes(f);
    ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
    ASSERT_TRUE(orig_shape.same_scheme(f->get_output_partial_shape(0)))
        << "Shape " << orig_shape << " is not equal to " << f->get_output_partial_shape(0);
}

INSTANTIATE_TEST_SUITE_P(ConvertDeconvolution,
                         ConvertDeconvolutionTest,
                         testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN},
                                                         WeightsShape{3, 8, 1, 2, 3}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64, 64}, WeightsShape{3, 8, 1, 2, 3}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64, 64}, WeightsShape{3, 9, 2, 3, 1}),
                                         std::make_tuple(InputShape{3, 3, DYN, 64, 64}, WeightsShape{3, 6, 3, 4, 2}),
                                         std::make_tuple(InputShape{3, 3, 64, DYN, 64}, WeightsShape{3, 5, 3, 4, 3}),
                                         std::make_tuple(InputShape{3, 3, 64, 64, DYN}, WeightsShape{3, 3, 3, 4, 3}),
                                         std::make_tuple(InputShape{1, 3, 64, 64}, WeightsShape{3, 6, 1, 1}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, WeightsShape{3, 7, 1, 1}),
                                         std::make_tuple(InputShape{DYN, 3, 64, 64}, WeightsShape{3, 8, 1, 2}),
                                         std::make_tuple(InputShape{2, DYN, 64, 64}, WeightsShape{3, 9, 2, 3}),
                                         std::make_tuple(InputShape{3, 3, DYN, 64}, WeightsShape{3, 6, 3, 4}),
                                         std::make_tuple(InputShape{3, 3, 64, DYN}, WeightsShape{3, 5, 3, 4}),
                                         std::make_tuple(InputShape{DYN, DYN, DYN}, WeightsShape{3, 5, 1}),
                                         std::make_tuple(InputShape{DYN, 3, 10}, WeightsShape{3, 3, 1}),
                                         std::make_tuple(InputShape{2, DYN, 9}, WeightsShape{3, 2, 2}),
                                         std::make_tuple(InputShape{3, 3, DYN}, WeightsShape{3, 1, 3})));

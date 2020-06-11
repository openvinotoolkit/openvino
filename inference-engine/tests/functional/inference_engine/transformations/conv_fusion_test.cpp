// Copyright (C) 2020 Intel Corporation
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
#include <transformations/convert_opset1_to_legacy/conv_bias_fusion.hpp>
#include <ngraph/pass/visualize_tree.hpp>

#include "ngraph_test_utils.hpp"

using namespace testing;

using InputShape = ngraph::PartialShape;
using WeightsShape = ngraph::Shape;
using EltwiseType = ngraph::NodeTypeInfo;
using EltwiseShape = ngraph::Shape;
using IsNegative = bool;

class ConvFusionTests: public CommonTestUtils::TestsCommon,
                       public testing::WithParamInterface<std::tuple<InputShape, WeightsShape, EltwiseType, EltwiseShape, IsNegative> > {
public:
    std::shared_ptr<ngraph::Function> f, f_ref;

    void SetUp() override {
        const auto& input_shape = std::get<0>(GetParam());
        const auto& weights_shape = std::get<1>(GetParam());
        const auto& eltwise_type = std::get<2>(GetParam());
        const auto& eltwise_shape = std::get<3>(GetParam());
        const auto& is_negative = std::get<4>(GetParam());

        f = get_initial_function(input_shape, weights_shape, eltwise_type, eltwise_shape);

        if (is_negative) {
            f_ref = get_initial_function(input_shape, weights_shape, eltwise_type, eltwise_shape);
        } else {
            f_ref = get_reference_function(input_shape, weights_shape, eltwise_type, eltwise_shape);
        }
    }

private:
    std::shared_ptr<ngraph::Function> get_initial_function(const InputShape&   input_shape,
                                                           const WeightsShape& weights_shape,
                                                           const EltwiseType&  eltwise_type,
                                                           const EltwiseShape& eltwise_shape) {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, weights_shape, {1});
        auto conv = std::make_shared<ngraph::op::ConvolutionIE>(input, weights, ngraph::Strides(spatial_dims, 1), ngraph::Strides(spatial_dims, 1),
                ngraph::CoordinateDiff(spatial_dims, 0), ngraph::CoordinateDiff(spatial_dims, 0), ngraph::element::f32);

        auto const_node = ngraph::opset1::Constant::create(ngraph::element::f32, eltwise_shape, {1.1});
        ngraph::Output<ngraph::Node> eltwise;
        if (eltwise_type == ngraph::opset1::Add::type_info) {
            eltwise = std::make_shared<ngraph::opset1::Add>(conv, const_node);
        } else if (eltwise_type == ngraph::opset1::Multiply::type_info) {
            eltwise = std::make_shared<ngraph::opset1::Multiply>(conv, const_node);
        } else {
            throw ngraph::ngraph_error("Unsupported element type");
        }

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{eltwise.get_node_shared_ptr()}, ngraph::ParameterVector{input});
    }

    std::shared_ptr<ngraph::Function> get_reference_function(const InputShape&   input_shape,
                                                             const WeightsShape& weights_shape,
                                                             const EltwiseType&  eltwise_type,
                                                             const EltwiseShape& eltwise_shape) {
        auto spatial_dims = input_shape.rank().get_length() - 2;
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_shape);
        ngraph::Output<ngraph::Node> weights = ngraph::opset1::Constant::create(ngraph::element::f32, weights_shape, {1});
        ngraph::Output<ngraph::Node> conv = std::make_shared<ngraph::op::ConvolutionIE>(input, weights, ngraph::Strides(spatial_dims, 1),
                ngraph::Strides(spatial_dims, 1), ngraph::CoordinateDiff(spatial_dims, 0), ngraph::CoordinateDiff(spatial_dims, 0),
                ngraph::element::f32);

        ngraph::Output<ngraph::Node> const_node;
        const_node = ngraph::opset1::Constant::create(ngraph::element::f32, eltwise_shape, {1.1});
        if (eltwise_type == ngraph::opset1::Add::type_info) {
            if (eltwise_shape.size() != 1) {
                const_node = ngraph::op::util::reshapeTo(const_node, ngraph::Shape{ngraph::shape_size(eltwise_shape)});
            }
            conv = conv.get_node_shared_ptr()->copy_with_new_inputs({input, weights, const_node});
        } else if (eltwise_type == ngraph::opset1::Multiply::type_info) {
            if (eltwise_shape.size() > 1) {
                const_node = ngraph::op::util::reshapeTo(const_node, ngraph::Shape{ngraph::shape_size(eltwise_shape)});
            }
            ngraph::Shape const_shape(weights_shape.size(), 1);
            const_shape[0] = weights_shape[0];
            weights = std::make_shared<ngraph::opset1::Multiply>(weights, ngraph::op::util::reshapeTo(const_node, const_shape));
            conv = conv.get_node_shared_ptr()->copy_with_new_inputs({input, weights});
        } else {
            throw ngraph::ngraph_error("Unsupported element type");
        }

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{conv.get_node_shared_ptr()}, ngraph::ParameterVector{input});
    }
};

TEST_P(ConvFusionTests, CompareFunctions) {
    ngraph::pass::InitNodeInfo().run_on_function(f);
    ngraph::pass::ConvFusion().run_on_function(f);
    f->validate_nodes_and_infer_types();
    // ASSERT_NO_THROW(check_rt_info(f));
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

using add = ngraph::opset1::Add;
using mul = ngraph::opset1::Multiply;

INSTANTIATE_TEST_CASE_P(ConvAddFusion, ConvFusionTests,
        testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, WeightsShape{8, 3, 1, 2, 3}, add::type_info, EltwiseShape{8, 1, 1, 1}, false),
                        std::make_tuple(InputShape{DYN, 3, 64, 64, 64}, WeightsShape{8, 3, 1, 2, 3}, add::type_info, EltwiseShape{8, 1, 1, 1}, false),
                        std::make_tuple(InputShape{2, DYN, 64, 64, 64}, WeightsShape{9, 3, 2, 3, 1}, add::type_info, EltwiseShape{9, 1, 1, 1}, false),
                        std::make_tuple(InputShape{3, 3, DYN, 64, 64},  WeightsShape{6, 3, 3, 4, 2}, add::type_info, EltwiseShape{6, 1, 1, 1}, false),
                        std::make_tuple(InputShape{3, 3, 64, DYN, 64},  WeightsShape{5, 3, 3, 4, 3}, add::type_info, EltwiseShape{5, 1, 1, 1}, false),
                        std::make_tuple(InputShape{3, 3, 64, 64, DYN},  WeightsShape{5, 3, 3, 4, 3}, add::type_info, EltwiseShape{5, 1, 1, 1}, false),
                        std::make_tuple(InputShape{1, 3, 64, 64},       WeightsShape{6, 3, 1, 1},    add::type_info, EltwiseShape{6, 1, 1}, false),
                        std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, WeightsShape{7, 3, 1, 1},    add::type_info, EltwiseShape{7, 1, 1}, false),
                        std::make_tuple(InputShape{DYN, 3, 64, 64},     WeightsShape{8, 3, 1, 2},    add::type_info, EltwiseShape{8, 1, 1}, false),
                        std::make_tuple(InputShape{2, DYN, 64, 64},     WeightsShape{9, 3, 2, 3},    add::type_info, EltwiseShape{9, 1, 1}, false),
                        std::make_tuple(InputShape{3, 3, DYN, 64},      WeightsShape{6, 3, 3, 4},    add::type_info, EltwiseShape{6, 1, 1}, false),
                        std::make_tuple(InputShape{3, 3, 64, DYN},      WeightsShape{5, 3, 3, 4},    add::type_info, EltwiseShape{5, 1, 1}, false),
                        std::make_tuple(InputShape{DYN, DYN, DYN},      WeightsShape{5, 3, 1},       add::type_info, EltwiseShape{5, 1}, false),
                        std::make_tuple(InputShape{DYN, 3, 10},         WeightsShape{3, 3, 1},       add::type_info, EltwiseShape{3, 1}, false),
                        std::make_tuple(InputShape{2, DYN, 9},          WeightsShape{2, 3, 2},       add::type_info, EltwiseShape{2, 1}, false),
                        std::make_tuple(InputShape{3, 3, DYN},          WeightsShape{1, 3, 3},       add::type_info, EltwiseShape{1, 1}, false)));

INSTANTIATE_TEST_CASE_P(DISABLED_ConvAddFusionNegative, ConvFusionTests,
        testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, WeightsShape{8, 3, 1, 2, 3}, add::type_info, EltwiseShape{2, 1}, true),
                        std::make_tuple(InputShape{DYN, 3, 64, 64, 64}, WeightsShape{8, 3, 1, 2, 3}, add::type_info, EltwiseShape{8, 1, 1}, true),
                        std::make_tuple(InputShape{2, DYN, 64, 64, 64}, WeightsShape{9, 3, 2, 3, 1}, add::type_info, EltwiseShape{9, 1, 1, 1, 1}, true)));

INSTANTIATE_TEST_CASE_P(ConvMulFusion, ConvFusionTests,
        testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, WeightsShape{8, 3, 1, 2, 3}, mul::type_info, EltwiseShape{8, 1, 1, 1}, false),
                        std::make_tuple(InputShape{DYN, 3, 64, 64, 64}, WeightsShape{8, 3, 1, 2, 3}, mul::type_info, EltwiseShape{8, 1, 1, 1}, false),
                        std::make_tuple(InputShape{2, DYN, 64, 64, 64}, WeightsShape{9, 3, 2, 3, 1}, mul::type_info, EltwiseShape{9, 1, 1, 1}, false),
                        std::make_tuple(InputShape{3, 3, DYN, 64, 64},  WeightsShape{6, 3, 3, 4, 2}, mul::type_info, EltwiseShape{6, 1, 1, 1}, false),
                        std::make_tuple(InputShape{3, 3, 64, DYN, 64},  WeightsShape{5, 3, 3, 4, 3}, mul::type_info, EltwiseShape{5, 1, 1, 1}, false),
                        std::make_tuple(InputShape{3, 3, 64, 64, DYN},  WeightsShape{5, 3, 3, 4, 3}, mul::type_info, EltwiseShape{5, 1, 1, 1}, false),
                        std::make_tuple(InputShape{1, 3, 64, 64},       WeightsShape{6, 3, 1, 1},    mul::type_info, EltwiseShape{6, 1, 1}, false),
                        std::make_tuple(InputShape{DYN, DYN, DYN, DYN}, WeightsShape{7, 3, 1, 1},    mul::type_info, EltwiseShape{7, 1, 1}, false),
                        std::make_tuple(InputShape{DYN, 3, 64, 64},     WeightsShape{8, 3, 1, 2},    mul::type_info, EltwiseShape{8, 1, 1}, false),
                        std::make_tuple(InputShape{2, DYN, 64, 64},     WeightsShape{9, 3, 2, 3},    mul::type_info, EltwiseShape{9, 1, 1}, false),
                        std::make_tuple(InputShape{3, 3, DYN, 64},      WeightsShape{6, 3, 3, 4},    mul::type_info, EltwiseShape{6, 1, 1}, false),
                        std::make_tuple(InputShape{3, 3, 64, DYN},      WeightsShape{5, 3, 3, 4},    mul::type_info, EltwiseShape{5, 1, 1}, false),
                        std::make_tuple(InputShape{DYN, DYN, DYN},      WeightsShape{5, 3, 1},       mul::type_info, EltwiseShape{5, 1}, false),
                        std::make_tuple(InputShape{DYN, 3, 10},         WeightsShape{3, 3, 1},       mul::type_info, EltwiseShape{3, 1}, false),
                        std::make_tuple(InputShape{2, DYN, 9},          WeightsShape{2, 3, 2},       mul::type_info, EltwiseShape{2, 1}, false),
                        std::make_tuple(InputShape{3, 3, DYN},          WeightsShape{1, 3, 3},       mul::type_info, EltwiseShape{1, 1}, false)));

INSTANTIATE_TEST_CASE_P(DISABLED_ConvMulFusionNegative, ConvFusionTests,
        testing::Values(std::make_tuple(InputShape{DYN, DYN, DYN, DYN, DYN}, WeightsShape{8, 3, 1, 2, 3}, mul::type_info, EltwiseShape{2, 1}, true),
                        std::make_tuple(InputShape{DYN, 3, 64, 64}, WeightsShape{8, 3, 1, 2, 3}, mul::type_info, EltwiseShape{8, 1, 1}, true),
                        std::make_tuple(InputShape{2, DYN, 64, 64}, WeightsShape{9, 3, 2, 3, 1}, mul::type_info, EltwiseShape{9, 1, 1, 1, 1}, true)));

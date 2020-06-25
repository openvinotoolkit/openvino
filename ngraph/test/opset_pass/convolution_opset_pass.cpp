#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "opset0_downgrade.hpp"
#include "util/test_control.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_convolution_upgrade_pass)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 3, 3, 3});
    CoordinateDiff pads_begin{0, 0};
    CoordinateDiff pads_end{0, 0};
    Strides strides{1, 1};
    Strides dilations{1, 1};
    Strides data_dilations_strides{1, 1};
    op::PadType pad_type = op::PadType::EXPLICIT;

    auto convolution_v0 = make_shared<op::v0::Convolution>(
        data, filters, strides, dilations, pads_begin, pads_end, data_dilations_strides, pad_type);
    auto result = make_shared<op::Result>(convolution_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data, filters});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto convolution_s1_result = f->get_results().at(0);
    auto node = convolution_s1_result->get_input_node_shared_ptr(0);
    auto convolution_v1_node = as_type_ptr<op::v1::Convolution>(node);

    ASSERT_TRUE(convolution_v1_node);

    EXPECT_EQ(convolution_v1_node->get_pads_begin(), pads_begin);
    EXPECT_EQ(convolution_v1_node->get_pads_end(), pads_end);
    EXPECT_EQ(convolution_v1_node->get_strides(), strides);
    EXPECT_EQ(convolution_v1_node->get_auto_pad(), pad_type);
    EXPECT_EQ(convolution_v1_node->get_dilations(), dilations);
}

TEST(opset_transform, opset1_convolution_downgrade_pass)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{1, 3, 3, 3});
    CoordinateDiff pads_begin{1, 1};
    CoordinateDiff pads_end{2, 2};
    Strides strides{1, 1};
    Strides dilations{1, 1};
    op::PadType pad_type = op::PadType::EXPLICIT;

    auto convolution_v1 = make_shared<op::v1::Convolution>(
        data, filters, strides, pads_begin, pads_end, dilations, pad_type);
    auto result = make_shared<op::Result>(convolution_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{data, filters});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto conv_s0_result = f->get_results().at(0);
    auto node = conv_s0_result->get_input_node_shared_ptr(0);
    auto conv_v0_node = as_type_ptr<op::v0::Convolution>(node);

    ASSERT_TRUE(conv_v0_node);
    EXPECT_EQ(conv_v0_node->get_window_movement_strides(), strides);
    EXPECT_EQ(conv_v0_node->get_window_dilation_strides(), dilations);
    EXPECT_EQ(conv_v0_node->get_padding_below(), pads_begin);
    EXPECT_EQ(conv_v0_node->get_padding_above(), pads_end);
    EXPECT_EQ(conv_v0_node->get_data_dilation_strides(), (Strides{1, 1}));
    EXPECT_EQ(conv_v0_node->get_pad_type(), pad_type);
}

TEST(opset_transform, opset1_convolution_backprop_data_downgrade_pass)
{
    auto data_batch_shape = op::Constant::create<int64_t>(element::i64, Shape{1}, {100});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{64, 128, 96});
    auto strides = Strides{1};
    auto dilations = Strides{1};
    auto padding_begin = CoordinateDiff{2};
    auto padding_end = CoordinateDiff{3};

    auto conv = make_shared<op::v1::ConvolutionBackpropData>(
        delta, filters, data_batch_shape, strides, padding_begin, padding_end, dilations);
    auto result = make_shared<op::Result>(conv);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{filters, delta});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto conv_s0_result = f->get_results().at(0);
    auto node = conv_s0_result->get_input_node_shared_ptr(0);
    auto conv_v0_node = as_type_ptr<op::v0::ConvolutionBackpropData>(node);

    ASSERT_TRUE(conv_v0_node);
    EXPECT_EQ(conv_v0_node->get_data_batch_shape(), (Shape{64, 3, 100}));
    EXPECT_EQ(conv_v0_node->get_window_movement_strides_forward(), strides);
    EXPECT_EQ(conv_v0_node->get_window_dilation_strides_forward(), dilations);
    EXPECT_EQ(conv_v0_node->get_padding_below_forward(), padding_begin);
    EXPECT_EQ(conv_v0_node->get_padding_above_forward(), padding_end);
    EXPECT_EQ(conv_v0_node->get_data_dilation_strides_forward(), (Strides{1}));
}

TEST(opset_transform, opset1_group_convolution_backprop_data_downgrade_pass)
{
    auto output_shape = op::Constant::create<int64_t>(element::i64, Shape{1}, {100});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{2, 128, 3, 10});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{64, 256, 96});
    size_t groups = 2;
    auto strides = Strides{1};
    auto dilations = Strides{1};

    auto padding_begin = CoordinateDiff{2};
    auto padding_end = CoordinateDiff{3};

    auto group_conv_backprop = make_shared<op::v1::GroupConvolutionBackpropData>(
        delta, filters, output_shape, strides, padding_begin, padding_end, dilations);
    auto result = make_shared<op::Result>(group_conv_backprop);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{filters, delta});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto group_conv_backprop_s0_result = f->get_results().at(0);
    auto node = group_conv_backprop_s0_result->get_input_node_shared_ptr(0);
    auto group_conv_backprop_v0_node = as_type_ptr<op::v0::GroupConvolutionBackpropData>(node);

    ASSERT_TRUE(group_conv_backprop_v0_node);
    EXPECT_EQ(group_conv_backprop_v0_node->get_window_movement_strides(), strides);
    EXPECT_EQ(group_conv_backprop_v0_node->get_window_dilation_strides(), dilations);
    EXPECT_EQ(group_conv_backprop_v0_node->get_padding_below(), padding_begin);
    EXPECT_EQ(group_conv_backprop_v0_node->get_padding_above(), padding_end);
    EXPECT_EQ(group_conv_backprop_v0_node->get_input_shape(1), (Shape{256, 3, 10}));
    EXPECT_EQ(group_conv_backprop_v0_node->get_groups(), groups);
}

TEST(opset_transform, opset1_group_convolution_backprop_data_upgrade_pass)
{
    auto data_batch_shape = op::Constant::create<int64_t>(element::i64, Shape{64, 12, 100}, {0});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{128, 3, 10});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{64, 128, 96});
    auto strides = Strides{1};
    auto dilations = Strides{1};
    auto padding_begin = CoordinateDiff{2};
    auto padding_end = CoordinateDiff{3};
    size_t groups = 4;

    auto group_conv_backprop = make_shared<op::v0::GroupConvolutionBackpropData>(
        data_batch_shape, filters, delta, strides, dilations, padding_begin, padding_end, groups);
    auto result = make_shared<op::Result>(group_conv_backprop);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{filters, delta});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto group_conv_backprop_s1_result = f->get_results().at(0);
    auto node = group_conv_backprop_s1_result->get_input_node_shared_ptr(0);
    auto group_conv_backprop_v1_node = as_type_ptr<op::v1::GroupConvolutionBackpropData>(node);

    ASSERT_TRUE(group_conv_backprop_v1_node);
    EXPECT_EQ(group_conv_backprop_v1_node->get_strides(), strides);
    EXPECT_EQ(group_conv_backprop_v1_node->get_dilations(), dilations);
    EXPECT_EQ(group_conv_backprop_v1_node->get_pads_begin(), padding_begin);
    EXPECT_EQ(group_conv_backprop_v1_node->get_pads_end(), padding_end);
    EXPECT_EQ(node->get_output_shape(0), (data_batch_shape->get_shape()));
    EXPECT_EQ(group_conv_backprop_v1_node->get_auto_pad(), op::PadType::EXPLICIT);
    EXPECT_EQ(group_conv_backprop_v1_node->get_output_padding(), (CoordinateDiff{0}));
}

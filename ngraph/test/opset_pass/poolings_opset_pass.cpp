#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "util/test_control.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_avgpool_upgrade_pass_floor)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    Shape pads_begin{0, 0};
    Shape pads_end{0, 0};
    Strides strides{1, 1};
    Shape kernel_shape{3, 3};
    bool include_pad = true;
    bool ceil_mode = false;
    op::PadType pad_mode = op::PadType::EXPLICIT;

    auto avgpool_v0 = make_shared<op::v0::AvgPool>(
        arg, kernel_shape, strides, pads_begin, pads_end, include_pad, pad_mode, ceil_mode);
    auto result = make_shared<op::Result>(avgpool_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto avgpool_s1_result = f->get_results().at(0);
    auto node = avgpool_s1_result->get_input_node_shared_ptr(0);
    auto avg_pool_v1_node = as_type_ptr<op::v1::AvgPool>(node);
    ASSERT_TRUE(avg_pool_v1_node);

    EXPECT_EQ(avg_pool_v1_node->get_pads_begin(), pads_begin);
    EXPECT_EQ(avg_pool_v1_node->get_pads_end(), pads_end);
    EXPECT_EQ(avg_pool_v1_node->get_strides(), strides);
    EXPECT_EQ(avg_pool_v1_node->get_kernel(), kernel_shape);
    EXPECT_EQ(avg_pool_v1_node->get_rounding_type(), op::RoundingType::FLOOR);
    EXPECT_EQ(avg_pool_v1_node->get_exclude_pad(), !include_pad);
    EXPECT_EQ(avg_pool_v1_node->get_auto_pad(), pad_mode);
}

TEST(opset_transform, opset1_avgpool_upgrade_pass_ceil)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    Shape pads_begin{0, 0};
    Shape pads_end{0, 0};
    Strides strides{1, 1};
    Shape kernel_shape{3, 3};
    bool include_pad = true;
    bool ceil_mode = true;
    op::PadType pad_mode = op::PadType::EXPLICIT;

    auto avgpool_v0 = make_shared<op::v0::AvgPool>(
        arg, kernel_shape, strides, pads_begin, pads_end, include_pad, pad_mode, ceil_mode);
    auto result = make_shared<op::Result>(avgpool_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto avgpool_s1_result = f->get_results().at(0);
    auto node = avgpool_s1_result->get_input_node_shared_ptr(0);
    auto avg_pool_v1_node = as_type_ptr<op::v1::AvgPool>(node);
    ASSERT_TRUE(avg_pool_v1_node);

    EXPECT_EQ(avg_pool_v1_node->get_pads_begin(), pads_begin);
    EXPECT_EQ(avg_pool_v1_node->get_pads_end(), pads_end);
    EXPECT_EQ(avg_pool_v1_node->get_strides(), strides);
    EXPECT_EQ(avg_pool_v1_node->get_kernel(), kernel_shape);
    EXPECT_EQ(avg_pool_v1_node->get_rounding_type(), op::RoundingType::CEIL);
    EXPECT_EQ(avg_pool_v1_node->get_exclude_pad(), !include_pad);
    EXPECT_EQ(avg_pool_v1_node->get_auto_pad(), pad_mode);
}

TEST(opset_transform, opset1_maxpool_upgrade_pass_fllor)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    Shape pads_begin{0, 0};
    Shape pads_end{0, 0};
    Strides strides{1, 1};
    Shape kernel_shape{3, 3};
    bool ceil_mode = false;
    op::PadType pad_mode = op::PadType::EXPLICIT;

    auto maxpool_v0 = make_shared<op::v0::MaxPool>(
        arg, kernel_shape, strides, pads_begin, pads_end, pad_mode, ceil_mode);
    auto result = make_shared<op::Result>(maxpool_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto maxpool_s1_result = f->get_results().at(0);
    auto node = maxpool_s1_result->get_input_node_shared_ptr(0);
    auto max_pool_v1_node = as_type_ptr<op::v1::MaxPool>(node);
    ASSERT_TRUE(max_pool_v1_node);

    EXPECT_EQ(max_pool_v1_node->get_pads_begin(), pads_begin);
    EXPECT_EQ(max_pool_v1_node->get_pads_end(), pads_end);
    EXPECT_EQ(max_pool_v1_node->get_strides(), strides);
    EXPECT_EQ(max_pool_v1_node->get_kernel(), kernel_shape);
    EXPECT_EQ(max_pool_v1_node->get_rounding_type(), op::RoundingType::FLOOR);
    EXPECT_EQ(max_pool_v1_node->get_auto_pad(), pad_mode);
}

TEST(opset_transform, opset1_maxpool_upgrade_pass_ceil)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    Shape pads_begin{0, 0};
    Shape pads_end{0, 0};
    Strides strides{1, 1};
    Shape kernel_shape{3, 3};
    bool ceil_mode = true;
    op::PadType pad_mode = op::PadType::EXPLICIT;

    auto maxpool_v0 = make_shared<op::v0::MaxPool>(
        arg, kernel_shape, strides, pads_begin, pads_end, pad_mode, ceil_mode);
    auto result = make_shared<op::Result>(maxpool_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto maxpool_s1_result = f->get_results().at(0);
    auto node = maxpool_s1_result->get_input_node_shared_ptr(0);
    auto max_pool_v1_node = as_type_ptr<op::v1::MaxPool>(node);
    ASSERT_TRUE(max_pool_v1_node);

    EXPECT_EQ(max_pool_v1_node->get_pads_begin(), pads_begin);
    EXPECT_EQ(max_pool_v1_node->get_pads_end(), pads_end);
    EXPECT_EQ(max_pool_v1_node->get_strides(), strides);
    EXPECT_EQ(max_pool_v1_node->get_kernel(), kernel_shape);
    EXPECT_EQ(max_pool_v1_node->get_rounding_type(), op::RoundingType::CEIL);
    EXPECT_EQ(max_pool_v1_node->get_auto_pad(), pad_mode);
}

TEST(opset_transform, opset1_avgpool_downgrade_pass)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    Shape padding_below{1, 0};
    Shape padding_above{0, 1};
    Strides window_movement_strides{1, 1};
    Shape window_shape{3, 3};
    bool exclude_pad = false;
    auto rounding_type = op::RoundingType::FLOOR;
    op::PadType auto_pad = op::PadType::EXPLICIT;

    auto avgpool_v1 = make_shared<op::v1::AvgPool>(arg,
                                                   window_movement_strides,
                                                   padding_below,
                                                   padding_above,
                                                   window_shape,
                                                   exclude_pad,
                                                   rounding_type,
                                                   auto_pad);
    auto result = make_shared<op::Result>(avgpool_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto avgpool_s0_result = f->get_results().at(0);
    auto node = avgpool_s0_result->get_input_node_shared_ptr(0);
    auto avg_pool_v0_node = as_type_ptr<op::v0::AvgPool>(node);
    ASSERT_TRUE(avg_pool_v0_node);

    EXPECT_EQ(avg_pool_v0_node->get_padding_below(), padding_below);
    EXPECT_EQ(avg_pool_v0_node->get_padding_above(), padding_above);
    EXPECT_EQ(avg_pool_v0_node->get_window_movement_strides(), window_movement_strides);
    EXPECT_EQ(avg_pool_v0_node->get_window_shape(), window_shape);
    EXPECT_EQ(avg_pool_v0_node->get_ceil_mode(), false);
    EXPECT_EQ(avg_pool_v0_node->get_include_padding_in_avg_computation(), !exclude_pad);
    EXPECT_EQ(avg_pool_v0_node->get_pad_type(), auto_pad);
}

TEST(opset_transform, opset1_maxpool_downgrade_pass)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    Shape padding_below{1, 0};
    Shape padding_above{0, 1};
    Strides window_movement_strides{1, 1};
    Shape window_shape{3, 3};
    auto rounding_type = op::RoundingType::FLOOR;
    op::PadType pad_type = op::PadType::EXPLICIT;

    auto maxpool_v1 = make_shared<op::v1::MaxPool>(arg,
                                                   window_movement_strides,
                                                   padding_below,
                                                   padding_above,
                                                   window_shape,
                                                   rounding_type,
                                                   pad_type);
    auto result = make_shared<op::Result>(maxpool_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto maxpool_s0_result = f->get_results().at(0);
    auto node = maxpool_s0_result->get_input_node_shared_ptr(0);
    auto max_pool_v0_node = as_type_ptr<op::v0::MaxPool>(node);
    ASSERT_TRUE(max_pool_v0_node);

    EXPECT_EQ(max_pool_v0_node->get_padding_below(), padding_below);
    EXPECT_EQ(max_pool_v0_node->get_padding_above(), padding_above);
    EXPECT_EQ(max_pool_v0_node->get_window_movement_strides(), window_movement_strides);
    EXPECT_EQ(max_pool_v0_node->get_window_shape(), window_shape);
    EXPECT_EQ(max_pool_v0_node->get_ceil_mode(), false);
    EXPECT_EQ(max_pool_v0_node->get_pad_type(), pad_type);
}

TEST(opset_transform, opset1_avgpool_backprop_downgrade_pass)
{
    auto delta = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    auto forward_arg_shape =
        op::Constant::create(element::i64, Shape{4}, vector<int64_t>{1, 3, 7, 10});
    Shape padding_below{1, 0};
    Shape padding_above{0, 1};
    Strides window_movement_strides{1, 1};
    Shape window_shape{3, 3};
    bool exclude_pad = false;

    auto avgpool_backprop_v1 = make_shared<op::v1::AvgPoolBackprop>(delta,
                                                                    forward_arg_shape,
                                                                    window_movement_strides,
                                                                    padding_below,
                                                                    padding_above,
                                                                    window_shape,
                                                                    exclude_pad);
    auto result = make_shared<op::Result>(avgpool_backprop_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{delta});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto avgpool_backprop_s0_result = f->get_results().at(0);
    auto node = avgpool_backprop_s0_result->get_input_node_shared_ptr(0);
    auto avg_pool_backprop_v0_node = as_type_ptr<op::v0::AvgPoolBackprop>(node);
    ASSERT_TRUE(avg_pool_backprop_v0_node);

    EXPECT_EQ(avg_pool_backprop_v0_node->get_padding_below(), padding_below);
    EXPECT_EQ(avg_pool_backprop_v0_node->get_padding_above(), padding_above);
    EXPECT_EQ(avg_pool_backprop_v0_node->get_window_movement_strides(), window_movement_strides);
    EXPECT_EQ(avg_pool_backprop_v0_node->get_window_shape(), window_shape);
    EXPECT_EQ(avg_pool_backprop_v0_node->get_forward_arg_shape(), Shape({1, 3, 7, 10}));
    EXPECT_EQ(avg_pool_backprop_v0_node->get_include_padding_in_avg_computation(), !exclude_pad);
}

TEST(opset_transform, opset1_maxpool_backprop_downgrade_pass)
{
    auto arg_forward = make_shared<op::Parameter>(element::f32, Shape{1, 3, 7, 10});
    auto delta = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    auto result_forward = make_shared<op::Parameter>(element::f32, Shape{1, 3, 6, 9});
    Shape padding_below{1, 0};
    Shape padding_above{0, 1};
    Strides window_movement_strides{1, 1};
    Shape window_shape{3, 3};

    auto max_pool_backprop_v1 = make_shared<op::v1::MaxPoolBackprop>(arg_forward,
                                                                     delta,
                                                                     result_forward,
                                                                     window_movement_strides,
                                                                     padding_below,
                                                                     padding_above,
                                                                     window_shape);
    auto result = make_shared<op::Result>(max_pool_backprop_v1);
    auto f = make_shared<Function>(ResultVector{result},
                                   ParameterVector{arg_forward, delta, result_forward});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto max_pool_backprop_s0_result = f->get_results().at(0);
    auto node = max_pool_backprop_s0_result->get_input_node_shared_ptr(0);
    auto max_pool_backprop_v0_node = as_type_ptr<op::v0::MaxPoolBackprop>(node);
    ASSERT_TRUE(max_pool_backprop_v0_node);
    EXPECT_EQ(max_pool_backprop_v0_node->get_padding_below(), padding_below);
    EXPECT_EQ(max_pool_backprop_v0_node->get_padding_above(), padding_above);
    EXPECT_EQ(max_pool_backprop_v0_node->get_window_movement_strides(), window_movement_strides);
    EXPECT_EQ(max_pool_backprop_v0_node->get_window_shape(), window_shape);
}

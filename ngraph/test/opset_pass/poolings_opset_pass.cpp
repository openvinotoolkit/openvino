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

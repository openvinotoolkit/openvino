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

TEST(opset_transform, opset0_select_downgrade_pass)
{
    auto cond = make_shared<op::Parameter>(element::boolean, Shape{2});
    auto ptrue = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto pfalse = make_shared<op::Parameter>(element::f32, Shape{4, 2});

    auto v1_node = make_shared<op::v1::Select>(cond, ptrue, pfalse);
    auto result = make_shared<op::Result>(v1_node);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{cond, ptrue, pfalse});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto v0_result = f->get_results().at(0);
    auto node = v0_result->input_value(0).get_node_shared_ptr();
    auto v0_node = as_type_ptr<op::v0::Select>(node);

    ASSERT_TRUE(v0_node);
    EXPECT_EQ(v0_node->get_output_element_type(0), element::f32);
    EXPECT_EQ(v0_node->get_output_shape(0), (Shape{4, 2}));
}

TEST(opset_transform, opset1_select_upgrade_pass)
{
    auto cond = make_shared<op::Parameter>(element::boolean, Shape{4, 2});
    auto ptrue = make_shared<op::Parameter>(element::f32, Shape{4, 2});
    auto pfalse = make_shared<op::Parameter>(element::f32, Shape{4, 2});

    auto v0_node = make_shared<op::v0::Select>(cond, ptrue, pfalse);
    auto result = make_shared<op::Result>(v0_node);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{cond, ptrue, pfalse});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto v1_result = f->get_results().at(0);
    auto node = v1_result->input_value(0).get_node_shared_ptr();
    auto v1_node = as_type_ptr<op::v1::Select>(node);

    ASSERT_TRUE(v1_node);
    EXPECT_EQ(v1_node->get_auto_broadcast(), op::AutoBroadcastSpec());
    EXPECT_EQ(v1_node->get_output_element_type(0), element::f32);
    EXPECT_EQ(v1_node->get_output_shape(0), (Shape{4, 2}));
}

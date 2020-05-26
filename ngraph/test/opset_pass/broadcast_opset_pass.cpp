#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_broadcast_upgrade_pass)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{5, 6});

    auto bcast_v0 = make_shared<op::v0::Broadcast>(arg, Shape{3, 5, 4, 6}, AxisSet{0, 2});
    auto f = make_shared<Function>(NodeVector{bcast_v0}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto bcast_v1 = as_type_ptr<op::v1::Broadcast>(
        f->get_results().at(0)->input_value(0).get_node_shared_ptr());

    ASSERT_TRUE(bcast_v1);
    EXPECT_EQ(bcast_v1->get_broadcast_spec(), op::AutoBroadcastSpec());
    EXPECT_EQ(bcast_v1->get_broadcast_axes(), (std::make_pair<bool, AxisSet>(true, AxisSet{0, 2})));
    ASSERT_TRUE(is_type<op::v0::Constant>(bcast_v1->input_value(1).get_node()));
    ASSERT_TRUE(is_type<op::v0::Constant>(bcast_v1->input_value(2).get_node()));
    EXPECT_EQ(
        as_type_ptr<op::Constant>(bcast_v1->input_value(1).get_node_shared_ptr())->get_shape_val(),
        (Shape{3, 5, 4, 6}));
    EXPECT_EQ(as_type_ptr<op::Constant>(bcast_v1->input_value(2).get_node_shared_ptr())
                  ->get_axis_set_val(),
              (AxisSet{1, 3}));
}

TEST(opset_transform, opset1_broadcast_downgrade_pass)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto target_shape = op::Constant::create<int64_t>(element::i64, Shape{5}, {3, 1, 4, 2, 3});
    auto axes_mapping = op::Constant::create<int64_t>(element::i64, Shape{3}, {1, 3, 4});

    auto bcast_v1 = make_shared<op::v1::Broadcast>(arg, target_shape, axes_mapping);
    auto f = make_shared<Function>(NodeVector{bcast_v1}, ParameterVector{arg});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto bcast_v0 = as_type_ptr<op::v0::Broadcast>(
        f->get_results().at(0)->input_value(0).get_node_shared_ptr());

    ASSERT_TRUE(bcast_v0);
    EXPECT_EQ(bcast_v0->get_broadcast_shape(), (Shape{3, 1, 4, 2, 3}));
    EXPECT_EQ(bcast_v0->get_broadcast_axes(), (AxisSet{0, 2}));
}

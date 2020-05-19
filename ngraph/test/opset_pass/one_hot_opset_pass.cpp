#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/pass/opset1_upgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_one_hot_upgrade_pass)
{
    auto indices = make_shared<op::Parameter>(element::i64, Shape{1, 3, 2, 3});
    const auto depth = 4;
    PartialShape shape{1, 3, 2, depth, 3};
    size_t one_hot_axis = 3;
    auto ont_hot_v0 = make_shared<op::v0::OneHot>(indices, shape, one_hot_axis);

    auto result = make_shared<op::Result>(ont_hot_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{indices});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->get_input_node_shared_ptr(0);
    const auto one_hot_v1 = as_type_ptr<op::v1::OneHot>(pass_replacement_node);
    ASSERT_TRUE(one_hot_v1);
    EXPECT_EQ(one_hot_v1->get_axis(), one_hot_axis);

    auto one_hot_v1_depth =
        as_type_ptr<op::Constant>(one_hot_v1->input_value(1).get_node_shared_ptr());
    EXPECT_EQ(one_hot_v1_depth->get_vector<int64_t>()[0], depth);

    auto one_hot_v1_on_value =
        as_type_ptr<op::Constant>(one_hot_v1->input_value(2).get_node_shared_ptr());
    EXPECT_EQ(one_hot_v1_on_value->get_vector<int64_t>()[0], 1);

    auto one_hot_v1_off_value =
        as_type_ptr<op::Constant>(one_hot_v1->input_value(3).get_node_shared_ptr());
    EXPECT_EQ(one_hot_v1_off_value->get_vector<int64_t>()[0], 0);
}

TEST(opset_transform, opset1_one_hot_downgrade_pass)
{
    auto indices = make_shared<op::Parameter>(element::i64, Shape{1, 3, 2, 3});
    auto depth = op::Constant::create(element::i64, Shape{}, {4});
    auto on_value = op::Constant::create(element::u32, Shape{}, {5});
    auto off_value = op::Constant::create(element::u32, Shape{}, {10});
    int64_t axis = 3;
    auto ont_hot_v1 = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);

    auto result = make_shared<op::Result>(ont_hot_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{indices});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    const auto pass_replacement_node = f->get_result()->input_value(0).get_node_shared_ptr();
    ASSERT_FALSE(is_type<op::v1::OneHot>(pass_replacement_node));

    EXPECT_EQ(pass_replacement_node->get_shape(), (Shape{1, 3, 2, 4, 3}));
}

TEST(opset_transform, opset1_one_hot_downgrade_pass_depth_not_constant)
{
    auto indices = make_shared<op::Parameter>(element::i64, Shape{1, 3, 2, 3});
    auto depth = make_shared<op::Parameter>(element::i64, Shape{});
    auto on_value = op::Constant::create(element::u32, Shape{}, {5});
    auto off_value = op::Constant::create(element::u32, Shape{}, {10});
    int64_t axis = 3;
    auto ont_hot_v1 = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);

    auto result = make_shared<op::Result>(ont_hot_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{indices, depth});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();

    try
    {
        pass_manager.run_passes(f);
        // Should have thrown, so fail if it didn't
        FAIL() << "Not constant depth not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("depth input must be constant"));
    }
    catch (...)
    {
        FAIL() << "OneHot downgrade failed for unexpected reason";
    }
}

TEST(opset_transform, opset1_one_hot_downgrade_pass_output_shape_not_static)
{
    auto indices = make_shared<op::Parameter>(element::i64, PartialShape::dynamic());
    auto depth = op::Constant::create(element::i64, Shape{}, {4});
    auto on_value = op::Constant::create(element::u32, Shape{}, {5});
    auto off_value = op::Constant::create(element::u32, Shape{}, {10});
    int64_t axis = 3;
    auto ont_hot_v1 = make_shared<op::v1::OneHot>(indices, depth, on_value, off_value, axis);

    auto result = make_shared<op::Result>(ont_hot_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{indices});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();

    try
    {
        pass_manager.run_passes(f);
        // Should have thrown, so fail if it didn't
        FAIL() << "Not static output shape not detected";
    }
    catch (const ngraph_error& error)
    {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("output shape must be static"));
    }
    catch (...)
    {
        FAIL() << "OneHot downgrade failed for unexpected reason";
    }
}

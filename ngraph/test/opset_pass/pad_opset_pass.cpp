#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/pass/manager.hpp"
#include "opset0_downgrade.hpp"
#include "opset1_upgrade.hpp"
#include "util/type_prop.hpp"

using namespace std;
using namespace ngraph;

TEST(opset_transform, opset1_pad_upgrade_pass)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto arg_pad_value = make_shared<op::Parameter>(element::f32, Shape{});
    CoordinateDiff padding_below{1, 2};
    CoordinateDiff padding_above{3, 4};
    auto pad_mode = op::PadMode::EDGE;

    auto pad_v0 =
        make_shared<op::v0::Pad>(arg, arg_pad_value, padding_below, padding_above, pad_mode);
    auto result = make_shared<op::Result>(pad_v0);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg, arg_pad_value});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset1Upgrade>();
    pass_manager.run_passes(f);

    auto pad_s1_result = f->get_results().at(0);
    auto node = pad_s1_result->get_input_node_shared_ptr(0);
    auto pad_v1_node = as_type_ptr<op::v1::Pad>(node);
    ASSERT_TRUE(pad_v1_node);
    EXPECT_EQ(pad_v1_node->get_pad_mode(), pad_mode);

    EXPECT_EQ(pad_v1_node->get_pads_begin(), padding_below);
    EXPECT_EQ(pad_v1_node->get_pads_end(), padding_above);
}

TEST(opset_transform, opset1_pad_downgrade_pass)
{
    auto arg = make_shared<op::Parameter>(element::f32, Shape{5, 6});
    auto arg_pad_value = make_shared<op::Parameter>(element::f32, Shape{});
    const auto pads_begin =
        make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{1, 2});
    const auto pads_end = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{3, 4});
    auto pad_mode = op::PadMode::EDGE;

    auto pad_v1 = make_shared<op::v1::Pad>(arg, pads_begin, pads_end, arg_pad_value, pad_mode);
    auto result = make_shared<op::Result>(pad_v1);
    auto f = make_shared<Function>(ResultVector{result}, ParameterVector{arg, arg_pad_value});

    ngraph::pass::Manager pass_manager;
    pass_manager.register_pass<pass::Opset0Downgrade>();
    pass_manager.run_passes(f);

    auto pad_s0_result = f->get_results().at(0);
    auto node = pad_s0_result->get_input_node_shared_ptr(0);
    auto pad_v0_node = as_type_ptr<op::v0::Pad>(node);
    ASSERT_TRUE(pad_v0_node);
    EXPECT_EQ(pad_v0_node->get_pad_mode(), pad_mode);

    EXPECT_EQ(pad_v0_node->get_padding_below(), CoordinateDiff({1, 2}));
    EXPECT_EQ(pad_v0_node->get_padding_above(), CoordinateDiff({3, 4}));
}

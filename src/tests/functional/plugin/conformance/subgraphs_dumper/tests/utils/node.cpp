// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/node.hpp"
#include "openvino/op/ops.hpp"
#include "base_test.hpp"

namespace {

using NodeUtilsTest = SubgraphsDumperBaseTest;

TEST_F(NodeUtilsTest, get_const_ranges) {
    std::vector<float> values = {-1, -2.05, -3.65, 0, 5, 7};
    auto const_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2, 3}), values);
    auto range = ov::util::get_const_ranges<float>(const_node);
    auto range_ref = ov::conformance::InputInfo::Range(-3.65, 7);
    ASSERT_EQ(range, range_ref);
}

TEST_F(NodeUtilsTest, get_input_info_by_node) {
    std::vector<float> values = {-1, -2.05, -3.65, 0, 5, 7};
    auto const_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2, 3}), values);
    const_node->set_friendly_name("const_0");
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({2, 3}));
    param->set_friendly_name("param_0");
    auto add_node = std::make_shared<ov::op::v1::Add>(param, const_node);

    std::map<std::string, ov::conformance::InputInfo> ref_test_info = {
        { "const_0", ov::conformance::InputInfo({2, 3}, -3.65, 7, true) },
        { "param_0", ov::conformance::InputInfo({2, 3}) },
    };
    std::map<std::string, ov::conformance::InputInfo> orig_test_info = ov::util::get_input_info_by_node(add_node);
    ASSERT_EQ(ref_test_info, orig_test_info);
}

TEST_F(NodeUtilsTest, clone_node) {
    std::vector<float> values(512, 1.f);
    auto const_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2, 256}), values);
    const_node->set_friendly_name("const_0");
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({2, 256}));
    param->set_friendly_name("param_0");
    auto add_node_0 = std::make_shared<ov::op::v1::Add>(param, const_node);
    auto erf_node_0 = std::make_shared<ov::op::v0::Erf>(add_node_0);
    auto erf_node_1 = std::make_shared<ov::op::v0::Erf>(const_node);
    auto add_node_1 = std::make_shared<ov::op::v1::Add>(erf_node_0, erf_node_1);

    {
        auto cloned_node = ov::util::clone_node(add_node_1);
        ASSERT_TRUE(ov::op::util::is_parameter(cloned_node->get_input_node_shared_ptr(0)));
        ASSERT_TRUE(ov::op::util::is_parameter(cloned_node->get_input_node_ptr(1)));
    }
    {
        auto cloned_node = ov::util::clone_node(add_node_1, true);
        ASSERT_TRUE(ov::op::util::is_parameter(cloned_node->get_input_node_ptr(0)));
        ASSERT_TRUE(ov::op::util::is_constant(cloned_node->get_input_node_ptr(1)));
    }
    {
        add_node_1 = std::make_shared<ov::op::v1::Add>(const_node, erf_node_1);
        auto cloned_node = ov::util::clone_node(add_node_1, true, true);
        ASSERT_TRUE(ov::op::util::is_constant(cloned_node->get_input_node_ptr(0)));
        ASSERT_TRUE(ov::op::util::is_constant(cloned_node->get_input_node_ptr(1)));
    }
}

TEST_F(NodeUtilsTest, generate_model_by_node) {
    std::vector<float> values = {-1, -2.05, -3.65, 0, 5, 7};
    auto const_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2, 3}), values);
    const_node->set_friendly_name("const_0");
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({2, 3}));
    param->set_friendly_name("param_0");
    auto add_node_0 = std::make_shared<ov::op::v1::Add>(param, const_node);
    auto erf_node_0 = std::make_shared<ov::op::v0::Erf>(add_node_0);
    auto erf_node_1 = std::make_shared<ov::op::v0::Erf>(const_node);
    auto add_node_1 = std::make_shared<ov::op::v1::Add>(erf_node_0, erf_node_1);

    auto model = ov::util::generate_model_by_node(add_node_1);
    auto param_0 = model->inputs().begin() ->get_node_shared_ptr();
    ASSERT_TRUE(ov::op::util::is_parameter(param_0));
    ASSERT_EQ(param_0->get_shape(), ov::Shape({2, 3}));
    ASSERT_EQ(param_0->get_element_type(), ov::element::Type_t::f32);

    auto param_1 = model->inputs().begin()->get_node_shared_ptr();
    ASSERT_TRUE(ov::op::util::is_parameter(param_1));
    ASSERT_EQ(param_1->get_shape(), ov::Shape({2, 3}));
    ASSERT_EQ(param_1->get_element_type(), ov::element::Type_t::f32);

    auto res_0 = model->outputs().rbegin()->get_node_shared_ptr();
    ASSERT_TRUE(ov::op::util::is_output(res_0));
    ASSERT_EQ(res_0->get_shape(), ov::Shape({2, 3}));
    ASSERT_EQ(res_0->get_element_type(), ov::element::Type_t::f32);
}

TEST_F(NodeUtilsTest, get_max_ops_versions) {
    std::unordered_map<std::string, std::pair<std::string, std::string>> max_ops_versions;
    std::string max_opset;
    OV_ASSERT_NO_THROW(std::tie(max_opset, max_ops_versions) = ov::util::get_last_opset_version_map());

    std::vector<float> values = {-1, -2.05, -3.65, 0, 5, 7};
    auto const_node = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2, 3}), values);
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({2, 3}));
    auto add_node_0 = std::make_shared<ov::op::v1::Add>(param, const_node);
    auto erf_node_0 = std::make_shared<ov::op::v0::Erf>(add_node_0);
    auto shapeOf_0 = std::make_shared<ov::op::v3::ShapeOf>(erf_node_0);

    ASSERT_EQ(max_ops_versions[add_node_0->get_type_info().name].second, "opset1");
    ASSERT_EQ(max_ops_versions[erf_node_0->get_type_info().name].second, "opset1");
    ASSERT_EQ(max_ops_versions[shapeOf_0->get_type_info().name].second, "opset3");
}

TEST_F(NodeUtilsTest, get_node_priority_by_version) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{1, 3, 16, 16});

    auto one_opset_node = std::make_shared<ov::op::v0::Convert>(param, ov::element::u16);
    ASSERT_EQ(ov::util::get_node_priority_by_version(one_opset_node), 3);

    auto max_of_several_opset_node = std::make_shared<ov::op::v3::ShapeOf>(param);
    ASSERT_EQ(ov::util::get_node_priority_by_version(max_of_several_opset_node), 3);

    auto min_of_several_opset_node = std::make_shared<ov::op::v0::ShapeOf>(param);
    ASSERT_EQ(ov::util::get_node_priority_by_version(min_of_several_opset_node), 1);
}

}  // namespace

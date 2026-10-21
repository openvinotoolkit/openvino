// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/linear_ir.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include "lir_test_utils.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "snippets/lowered/linear_ir_builder.hpp"
#include "snippets/lowered/pass/serialize_data_flow.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/broadcastmove.hpp"
#include "snippets/op/online_softmax.hpp"
#include "snippets/op/reshape.hpp"
#include "snippets/op/result.hpp"
#include "snippets/op/scalar.hpp"
#include "snippets/op/store.hpp"

namespace ov {
namespace test {
namespace snippets {

using namespace ov::snippets;

TEST(LinearIRReplaceWithNode, PreservesPerOutputDescriptors) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 6});
    const auto op = std::make_shared<ov::snippets::op::OnlineSoftmax>(param);
    const auto result_0 = std::make_shared<ov::snippets::op::Result>(op->output(0));
    const auto result_1 = std::make_shared<ov::snippets::op::Result>(op->output(1));
    const auto model = std::make_shared<ov::Model>(ov::OutputVector{result_0, result_1}, ov::ParameterVector{param});

    auto factory = std::make_shared<ov::snippets::IShapeInferSnippetsFactory>();
    ov::snippets::lowered::LinearIR linear_ir(model, factory);

    const auto& online_sm_expr = linear_ir.get_expr_by_node(op);
    ASSERT_NE(nullptr, online_sm_expr);

    const std::vector<VectorDims> subtensors = {get_default_subtensor(2), VectorDims{1, 3}, VectorDims{1, 6}};
    const std::vector<VectorDims> layouts = {VectorDims{}, VectorDims{0, 1}, VectorDims{1, 0}};
    init_expr_descriptors(online_sm_expr, subtensors, layouts);

    const auto expected_desc_0 = online_sm_expr->get_output_port_descriptor(0)->clone();
    const auto expected_desc_1 = online_sm_expr->get_output_port_descriptor(1)->clone();

    ASSERT_NE(expected_desc_0->get_subtensor(), expected_desc_1->get_subtensor());
    ASSERT_NE(expected_desc_0->get_layout(), expected_desc_1->get_layout());

    const auto new_node = std::make_shared<ov::snippets::op::OnlineSoftmax>(param);
    linear_ir.replace_with_node({online_sm_expr}, new_node);

    const auto& new_expr = linear_ir.get_expr_by_node(new_node);
    ASSERT_NE(nullptr, new_expr);

    const auto& new_desc_0 = new_expr->get_output_port_descriptor(0);
    const auto& new_desc_1 = new_expr->get_output_port_descriptor(1);

    EXPECT_EQ(new_desc_0->get_subtensor(), expected_desc_0->get_subtensor());
    EXPECT_EQ(new_desc_1->get_subtensor(), expected_desc_1->get_subtensor());
    EXPECT_EQ(new_desc_0->get_layout(), expected_desc_0->get_layout());
    EXPECT_EQ(new_desc_1->get_layout(), expected_desc_1->get_layout());
    auto cloned_ir = ov::snippets::lowered::LinearIRBuilder().clone(linear_ir);
    const auto& cloned_param = cloned_ir.front();
    const auto broadcast = std::make_shared<ov::snippets::op::BroadcastMove>(cloned_param->get_node()->output(0), 16);
    const auto broadcast_expr = *cloned_ir.insert_node(broadcast,
                                                        std::vector<ov::snippets::lowered::PortConnectorPtr>{
                                                            cloned_param->get_output_port_connector(0)},
                                                        {},
                                                        false,
                                                        cloned_ir.end());

    broadcast_expr->updateShapes();

    EXPECT_EQ(broadcast_expr->get_output_port_descriptor(0)->get_shape(), (VectorDims{1, 16}));
}

TEST(LinearIRSerializeDataFlow, PreservesAllStoreBranches) {
    const auto param_0 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 6});
    const auto param_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 6});
    const auto store_0 = std::make_shared<ov::snippets::op::Store>(param_0);
    const auto store_1 = std::make_shared<ov::snippets::op::Store>(param_1);
    store_0->set_friendly_name("store_0");
    store_1->set_friendly_name("store_1");
    const auto result = std::make_shared<ov::snippets::op::Result>(store_0, ov::OutputVector{store_1});
    const auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param_0, param_1});

    auto factory = std::make_shared<ov::snippets::IShapeInferSnippetsFactory>();
    ov::snippets::lowered::LinearIR linear_ir(model, factory);

    const auto xml_path = std::filesystem::temp_directory_path() / "ov_snippets_data_flow_two_stores.xml";
    EXPECT_FALSE(ov::snippets::lowered::pass::SerializeDataFlow(xml_path.string()).run(linear_ir));

    std::ifstream xml_file(xml_path);
    ASSERT_TRUE(xml_file.is_open());
    const std::string xml(std::istreambuf_iterator<char>{xml_file}, std::istreambuf_iterator<char>{});
    EXPECT_NE(xml.find("store_0"), std::string::npos);
    EXPECT_NE(xml.find("store_1"), std::string::npos);

    std::filesystem::remove(xml_path);
    auto bin_path = xml_path;
    bin_path.replace_extension(".bin");
    std::filesystem::remove(bin_path);
}

}  // namespace snippets
}  // namespace test
}  // namespace ov

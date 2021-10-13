// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <common_test_utils/file_utils.hpp>

#include "frontend_manager/frontend_manager.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "read_ir.hpp"
#include "transformations/rt_info/attributes.hpp"
#include "util/test_common.hpp"

using namespace ov;

class RTInfoSerializationTest : public ov::test::TestsCommon {
protected:
    std::string test_name = GetTestName() + "_" + GetTimestamp();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
        CommonTestUtils::removeIRFiles(m_out_xml_path, m_out_bin_path);
    }

private:
    ngraph::frontend::FrontEndManager manager;
};

TEST_F(RTInfoSerializationTest, all_attributes_latest) {
    auto init_info = [](RTMap& info) {
        info[VariantWrapper<ngraph::FusedNames>::get_type_info_static()] =
            std::make_shared<VariantWrapper<ngraph::FusedNames>>(ngraph::FusedNames("add"));
        info[ov::PrimitivesPriority::get_type_info_static()] = std::make_shared<ov::PrimitivesPriority>("priority");
        info[ov::OldApiMap::get_type_info_static()] = std::make_shared<ov::OldApiMap>(
            ov::OldApiMapAttr(std::vector<uint64_t>{0, 2, 3, 1}, ngraph::element::Type_t::f32));
    };

    std::shared_ptr<ngraph::Function> function;
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 10, 10});
        auto add = std::make_shared<ov::opset8::Add>(data, data);
        init_info(add->get_rt_info());
        init_info(add->input(0).get_rt_info());
        init_info(add->input(1).get_rt_info());
        init_info(add->output(0).get_rt_info());
        function = std::make_shared<ngraph::Function>(OutputVector{add}, ParameterVector{data});
    }

    pass::Manager m;
    m.register_pass<ov::pass::Serialize>(m_out_xml_path, m_out_bin_path);
    m.run_passes(function);

    auto f = ov::test::readIR(m_out_xml_path, m_out_bin_path);
    ASSERT_NE(nullptr, f);

    auto check_info = [](const RTMap& info) {
        const std::string& key = VariantWrapper<ngraph::FusedNames>::get_type_info_static();
        ASSERT_TRUE(info.count(key));
        auto fused_names_attr = std::dynamic_pointer_cast<VariantWrapper<ngraph::FusedNames>>(info.at(key));
        ASSERT_TRUE(fused_names_attr);
        ASSERT_EQ(fused_names_attr->get().getNames(), "add");

        const std::string& pkey = ov::PrimitivesPriority::get_type_info_static();
        ASSERT_TRUE(info.count(pkey));
        auto primitives_priority_attr = std::dynamic_pointer_cast<ov::PrimitivesPriority>(info.at(pkey));
        ASSERT_TRUE(primitives_priority_attr);
        ASSERT_EQ(primitives_priority_attr->get(), "priority");

        const std::string& old_api_map_key = ov::OldApiMap::get_type_info_static();
        ASSERT_TRUE(info.count(old_api_map_key));
        auto old_api_map_attr = std::dynamic_pointer_cast<ov::OldApiMap>(info.at(old_api_map_key));
        ASSERT_TRUE(old_api_map_attr);
        auto old_api_map_attr_val = old_api_map_attr->get();
        ASSERT_EQ(old_api_map_attr_val.get_order(), std::vector<uint64_t>({0, 2, 3, 1}));
        ASSERT_EQ(old_api_map_attr_val.get_type(), ngraph::element::Type_t::f32);
    };

    auto add = f->get_results()[0]->get_input_node_ptr(0);
    check_info(add->get_rt_info());
    check_info(add->input(0).get_rt_info());
    check_info(add->input(1).get_rt_info());
    check_info(add->output(0).get_rt_info());
}

TEST_F(RTInfoSerializationTest, all_attributes_v10) {
    auto init_info = [](RTMap& info) {
        info[VariantWrapper<ngraph::FusedNames>::get_type_info_static()] =
            std::make_shared<VariantWrapper<ngraph::FusedNames>>(ngraph::FusedNames("add"));
        info[ov::PrimitivesPriority::get_type_info_static()] = std::make_shared<ov::PrimitivesPriority>("priority");
    };

    std::shared_ptr<ngraph::Function> function;
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 10, 10});
        auto add = std::make_shared<ov::opset8::Add>(data, data);
        init_info(add->get_rt_info());
        init_info(add->input(0).get_rt_info());
        init_info(add->input(1).get_rt_info());
        init_info(add->output(0).get_rt_info());
        function = std::make_shared<ngraph::Function>(OutputVector{add}, ParameterVector{data});
    }

    pass::Manager m;
    m.register_pass<ov::pass::Serialize>(m_out_xml_path, m_out_bin_path, ov::pass::Serialize::Version::IR_V10);
    m.run_passes(function);

    auto f = ov::test::readIR(m_out_xml_path, m_out_bin_path);
    ASSERT_NE(nullptr, f);

    auto check_info = [](const RTMap& info) {
        const std::string& key = VariantWrapper<ngraph::FusedNames>::get_type_info_static();
        ASSERT_FALSE(info.count(key));
    };

    auto add = f->get_results()[0]->get_input_node_ptr(0);
    check_info(add->get_rt_info());
    check_info(add->input(0).get_rt_info());
    check_info(add->input(1).get_rt_info());
    check_info(add->output(0).get_rt_info());
}

TEST_F(RTInfoSerializationTest, all_attributes_v11) {
    auto init_info = [](RTMap& info) {
        info[VariantWrapper<ngraph::FusedNames>::get_type_info_static()] =
            std::make_shared<VariantWrapper<ngraph::FusedNames>>(ngraph::FusedNames("add"));
        info[ov::PrimitivesPriority::get_type_info_static()] = std::make_shared<ov::PrimitivesPriority>("priority");
    };

    std::shared_ptr<ngraph::Function> function;
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 10, 10});
        auto add = std::make_shared<ov::opset8::Add>(data, data);
        init_info(add->get_rt_info());
        init_info(add->input(0).get_rt_info());
        init_info(add->input(1).get_rt_info());
        init_info(add->output(0).get_rt_info());
        function = std::make_shared<ngraph::Function>(OutputVector{add}, ParameterVector{data});
    }

    pass::Manager m;
    m.register_pass<ov::pass::Serialize>(m_out_xml_path, m_out_bin_path);
    m.run_passes(function);

    auto f = ov::test::readIR(m_out_xml_path, m_out_bin_path);
    ASSERT_NE(nullptr, f);

    auto check_info = [](const RTMap& info) {
        const std::string& key = VariantWrapper<ngraph::FusedNames>::get_type_info_static();
        ASSERT_TRUE(info.count(key));
        auto fused_names_attr = std::dynamic_pointer_cast<VariantWrapper<ngraph::FusedNames>>(info.at(key));
        ASSERT_TRUE(fused_names_attr);
        ASSERT_EQ(fused_names_attr->get().getNames(), "add");

        const std::string& pkey = ov::PrimitivesPriority::get_type_info_static();
        ASSERT_TRUE(info.count(pkey));
        auto primitives_priority_attr = std::dynamic_pointer_cast<ov::PrimitivesPriority>(info.at(pkey));
        ASSERT_TRUE(primitives_priority_attr);
        ASSERT_EQ(primitives_priority_attr->get(), "priority");
    };

    auto add = f->get_results()[0]->get_input_node_ptr(0);
    check_info(add->get_rt_info());
    check_info(add->input(0).get_rt_info());
    check_info(add->input(1).get_rt_info());
    check_info(add->output(0).get_rt_info());
}

TEST_F(RTInfoSerializationTest, parameter_result_v11) {
    std::shared_ptr<ngraph::Function> function;
    {
        auto param1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ngraph::Shape({1, 3, 24, 24}));
        param1->set_friendly_name("param1");
        param1->output(0).get_tensor().set_names({"data1"});
        auto param2 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ngraph::Shape({1, 3, 24, 24}));
        param2->set_friendly_name("param2");
        param2->output(0).get_tensor().set_names({"data2"});
        auto relu = std::make_shared<ov::opset8::Relu>(param1);
        relu->set_friendly_name("relu_op");
        relu->output(0).get_tensor().set_names({"relu"});
        auto result1 = std::make_shared<ov::opset8::Result>(relu);
        result1->set_friendly_name("result1");
        auto concat = std::make_shared<ov::opset8::Concat>(OutputVector{relu, param2}, 1);
        concat->set_friendly_name("concat_op");
        concat->output(0).get_tensor().set_names({"concat"});
        auto result2 = std::make_shared<ov::opset8::Result>(concat);
        result2->set_friendly_name("result2");
        function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result1, result2},
                                                      ngraph::ParameterVector{param1, param2});
        function->set_friendly_name("SingleRuLU");
    }

    pass::Manager m;
    m.register_pass<ov::pass::Serialize>(m_out_xml_path, m_out_bin_path, ov::pass::Serialize::Version::IR_V11);
    m.run_passes(function);

    auto f = ov::test::readIR(m_out_xml_path, m_out_bin_path);
    ASSERT_NE(nullptr, f);

    ASSERT_EQ(function->get_results().size(), f->get_results().size());
    ASSERT_EQ(function->get_parameters().size(), f->get_parameters().size());
    for (size_t i = 0; i < f->get_parameters().size(); i++) {
        ASSERT_EQ(function->get_parameters()[i]->get_friendly_name(), f->get_parameters()[i]->get_friendly_name());
    }
    for (size_t i = 0; i < f->get_results().size(); i++) {
        ASSERT_EQ(function->get_results()[i]->get_friendly_name(), f->get_results()[i]->get_friendly_name());
    }
}

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/rt_info/attributes.hpp"

class RTInfoSerializationTest : public ov::test::TestsCommon {
protected:
    std::string m_out_xml_path;
    std::string m_out_bin_path;

    void SetUp() override {
        std::string filePrefix = ov::test::utils::generateTestFilePrefix();
        m_out_xml_path = filePrefix + ".xml";
        m_out_bin_path = filePrefix + ".bin";
    }

    void TearDown() override {
        ov::test::utils::removeIRFiles(m_out_xml_path, m_out_bin_path);
    }

    std::shared_ptr<ov::Model> getWithIRFrontend(const std::string& model_path, const std::string& weights_path) {
        ov::frontend::FrontEnd::Ptr FE;
        ov::frontend::InputModel::Ptr inputModel;

        ov::AnyVector params{model_path, weights_path};

        FE = manager.load_by_model(params);
        if (FE)
            inputModel = FE->load(params);

        if (inputModel)
            return FE->convert(inputModel);

        return nullptr;
    }

private:
    ov::frontend::FrontEndManager manager;
};

TEST_F(RTInfoSerializationTest, all_attributes_latest) {
    auto init_info = [](ov::RTMap& info) {
        info[ov::FusedNames::get_type_info_static()] = ov::FusedNames("add");
        info[ov::PrimitivesPriority::get_type_info_static()] = ov::PrimitivesPriority("priority");
        info[ov::OldApiMapOrder::get_type_info_static()] = ov::OldApiMapOrder(std::vector<uint64_t>{0, 2, 3, 1});
        info[ov::OldApiMapElementType::get_type_info_static()] = ov::OldApiMapElementType(ov::element::Type_t::f32);
        info[ov::Decompression::get_type_info_static()] = ov::Decompression{};
    };

    std::shared_ptr<ov::Model> function;
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f32, ov::Shape{1, 3, 10, 10});
        data->set_layout("NCHW");
        auto add = std::make_shared<ov::opset8::Add>(data, data);
        init_info(add->get_rt_info());
        init_info(add->input(0).get_rt_info());
        init_info(add->input(1).get_rt_info());
        init_info(add->output(0).get_rt_info());
        auto result = std::make_shared<ov::opset8::Result>(add);
        result->set_layout("????");
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data});
    }

    ov::pass::Manager m;
    m.register_pass<ov::pass::Serialize>(m_out_xml_path, m_out_bin_path);
    m.run_passes(function);

    auto f = getWithIRFrontend(m_out_xml_path, m_out_bin_path);
    ASSERT_NE(nullptr, f);

    auto check_info = [](const ov::RTMap& info) {
        const std::string& key = ov::FusedNames::get_type_info_static();
        ASSERT_TRUE(info.count(key));
        auto fused_names_attr = info.at(key).as<ov::FusedNames>();
        ASSERT_EQ(fused_names_attr.getNames(), "add");

        const std::string& pkey = ov::PrimitivesPriority::get_type_info_static();
        ASSERT_TRUE(info.count(pkey));
        auto primitives_priority_attr = info.at(pkey).as<ov::PrimitivesPriority>().value;
        ASSERT_EQ(primitives_priority_attr, "priority");

        const std::string& old_api_map_key_order = ov::OldApiMapOrder::get_type_info_static();
        ASSERT_TRUE(info.count(old_api_map_key_order));
        auto old_api_map_attr_val = info.at(old_api_map_key_order).as<ov::OldApiMapOrder>().value;
        ASSERT_EQ(old_api_map_attr_val, std::vector<uint64_t>({0, 2, 3, 1}));

        const std::string& old_api_map_key = ov::OldApiMapElementType::get_type_info_static();
        ASSERT_TRUE(info.count(old_api_map_key));
        auto old_api_map_type_val = info.at(old_api_map_key).as<ov::OldApiMapElementType>().value;
        ASSERT_EQ(old_api_map_type_val, ov::element::Type_t::f32);

        const std::string& dkey = ov::Decompression::get_type_info_static();
        ASSERT_TRUE(info.count(dkey));
        ASSERT_NO_THROW(info.at(dkey).as<ov::Decompression>());
    };

    auto add = f->get_results()[0]->get_input_node_ptr(0);
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), "NCHW");
    EXPECT_EQ(f->get_results()[0]->get_layout(), "????");
    check_info(add->get_rt_info());
    check_info(add->input(0).get_rt_info());
    check_info(add->input(1).get_rt_info());
    check_info(add->output(0).get_rt_info());
}

TEST_F(RTInfoSerializationTest, all_attributes_v10) {
    auto init_info = [](ov::RTMap& info) {
        info[ov::FusedNames::get_type_info_static()] = ov::FusedNames("add");
        info["PrimitivesPriority"] = ov::PrimitivesPriority("priority");
    };

    std::shared_ptr<ov::Model> function;
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::Type_t::f32, ov::Shape{1, 3, 10, 10});
        data->set_layout("NCHW");
        auto add = std::make_shared<ov::opset8::Add>(data, data);
        init_info(add->get_rt_info());
        init_info(add->input(0).get_rt_info());
        init_info(add->input(1).get_rt_info());
        init_info(add->output(0).get_rt_info());
        function = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{data});
    }

    ov::pass::Manager m;
    m.register_pass<ov::pass::Serialize>(m_out_xml_path, m_out_bin_path, ov::pass::Serialize::Version::IR_V10);
    m.run_passes(function);

    auto f = getWithIRFrontend(m_out_xml_path, m_out_bin_path);
    ASSERT_NE(nullptr, f);

    auto check_info = [](const ov::RTMap& info) {
        const std::string& key = ov::FusedNames::get_type_info_static();
        ASSERT_FALSE(info.count(key));
    };

    auto add = f->get_results()[0]->get_input_node_ptr(0);
    check_info(add->get_rt_info());
    check_info(add->input(0).get_rt_info());
    check_info(add->input(1).get_rt_info());
    check_info(add->output(0).get_rt_info());
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), "");
}

// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/file_utils.hpp>
#include <gtest/gtest.h>

#include <file_utils.h>
#include <ie_api.h>
#include <ie_iextension.h>
#include "common_test_utils/ngraph_test_utils.hpp"
#include "ie_core.hpp"
#include "ngraph/ngraph.hpp"
#include "transformations/serialize.hpp"
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/opsets/opset8.hpp>
#include <transformations/rt_info/attributes.hpp>
#include "openvino/frontend/manager.hpp"

using namespace ngraph;

class RTInfoSerializationTest : public CommonTestUtils::TestsCommon {
protected:
    std::string test_name = GetTestName() + "_" + GetTimestamp();
    std::string m_out_xml_path = test_name + ".xml";
    std::string m_out_bin_path = test_name + ".bin";

    void TearDown() override {
        CommonTestUtils::removeIRFiles(m_out_xml_path, m_out_bin_path);
    }

    std::shared_ptr<ngraph::Function> getWithIRFrontend(const std::string& model_path,
                                                        const std::string& weights_path) {
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
    auto init_info = [](RTMap & info) {
        info[ngraph::FusedNames::get_type_info_static()] = ngraph::FusedNames("add");
        info[ov::PrimitivesPriority::get_type_info_static()] = ov::PrimitivesPriority("priority");
        info[ov::OldApiMapOrder::get_type_info_static()] = ov::OldApiMapOrder(std::vector<uint64_t>{0, 2, 3, 1});
        info[ov::OldApiMapElementType::get_type_info_static()] = ov::OldApiMapElementType(ngraph::element::Type_t::f32);
        info[ov::Decompression::get_type_info_static()] = ov::Decompression{};
    };

    std::shared_ptr<ngraph::Function> function;
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 10, 10});
        data->set_layout("NCHW");
        auto add = std::make_shared<ov::opset8::Add>(data, data);
        init_info(add->get_rt_info());
        init_info(add->input(0).get_rt_info());
        init_info(add->input(1).get_rt_info());
        init_info(add->output(0).get_rt_info());
        auto result = std::make_shared<ov::opset8::Result>(add);
        result->set_layout("????");
        function = std::make_shared<ngraph::Function>(ResultVector{result}, ParameterVector{data});
    }

    pass::Manager m;
    m.register_pass<pass::Serialize>(m_out_xml_path, m_out_bin_path);
    m.run_passes(function);

    auto f = getWithIRFrontend(m_out_xml_path, m_out_bin_path);
    ASSERT_NE(nullptr, f);

    auto check_info = [](const RTMap & info) {
        const std::string & key = ngraph::FusedNames::get_type_info_static();
        ASSERT_TRUE(info.count(key));
        auto fused_names_attr = info.at(key).as<ngraph::FusedNames>();
        ASSERT_EQ(fused_names_attr.getNames(), "add");

        const std::string & pkey = ov::PrimitivesPriority::get_type_info_static();
        ASSERT_TRUE(info.count(pkey));
        auto primitives_priority_attr = info.at(pkey).as<ov::PrimitivesPriority>().value;
        ASSERT_EQ(primitives_priority_attr, "priority");

        const std::string & old_api_map_key_order = ov::OldApiMapOrder::get_type_info_static();
        ASSERT_TRUE(info.count(old_api_map_key_order));
        auto old_api_map_attr_val = info.at(old_api_map_key_order).as<ov::OldApiMapOrder>().value;
        ASSERT_EQ(old_api_map_attr_val, std::vector<uint64_t>({0, 2, 3, 1}));

        const std::string & old_api_map_key = ov::OldApiMapElementType::get_type_info_static();
        ASSERT_TRUE(info.count(old_api_map_key));
        auto old_api_map_type_val = info.at(old_api_map_key).as<ov::OldApiMapElementType>().value;
        ASSERT_EQ(old_api_map_type_val, ngraph::element::Type_t::f32);

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
    auto init_info = [](RTMap & info) {
        info[ngraph::FusedNames::get_type_info_static()] = ngraph::FusedNames("add");
        info["PrimitivesPriority"] = ov::PrimitivesPriority("priority");
    };

    std::shared_ptr<ngraph::Function> function;
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 10, 10});
        data->set_layout("NCHW");
        auto add = std::make_shared<ov::opset8::Add>(data, data);
        init_info(add->get_rt_info());
        init_info(add->input(0).get_rt_info());
        init_info(add->input(1).get_rt_info());
        init_info(add->output(0).get_rt_info());
        function = std::make_shared<ngraph::Function>(OutputVector{add}, ParameterVector{data});
    }

    pass::Manager m;
    m.register_pass<pass::Serialize>(m_out_xml_path, m_out_bin_path, pass::Serialize::Version::IR_V10);
    m.run_passes(function);

    auto f = getWithIRFrontend(m_out_xml_path, m_out_bin_path);
    ASSERT_NE(nullptr, f);

    auto check_info = [](const RTMap & info) {
        const std::string & key = ngraph::FusedNames::get_type_info_static();
        ASSERT_FALSE(info.count(key));
    };

    auto add = f->get_results()[0]->get_input_node_ptr(0);
    check_info(add->get_rt_info());
    check_info(add->input(0).get_rt_info());
    check_info(add->input(1).get_rt_info());
    check_info(add->output(0).get_rt_info());
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), "");
}

TEST_F(RTInfoSerializationTest, all_attributes_v11) {
    auto init_info = [](RTMap & info) {
        info[ngraph::FusedNames::get_type_info_static()] = ngraph::FusedNames("add");
        info[ov::PrimitivesPriority::get_type_info_static()] = ov::PrimitivesPriority("priority");
    };

    std::shared_ptr<ngraph::Function> function;
    {
        auto data = std::make_shared<ov::opset8::Parameter>(ngraph::element::Type_t::f32, ngraph::Shape{1, 3, 10, 10});
        data->set_layout("NCHW");
        auto add = std::make_shared<ov::opset8::Add>(data, data);
        init_info(add->get_rt_info());
        init_info(add->input(0).get_rt_info());
        init_info(add->input(1).get_rt_info());
        init_info(add->output(0).get_rt_info());
        auto result = std::make_shared<ov::opset8::Result>(add);
        result->set_layout("????");
        function = std::make_shared<ngraph::Function>(ResultVector{result}, ParameterVector{data});
        auto p = ov::preprocess::PrePostProcessor(function);
        p.input().tensor().set_memory_type("test_memory_type");
        function = p.build();
    }

    pass::Manager m;
    m.register_pass<pass::Serialize>(m_out_xml_path, m_out_bin_path);
    m.run_passes(function);

    auto f = getWithIRFrontend(m_out_xml_path, m_out_bin_path);
    ASSERT_NE(nullptr, f);

    auto check_info = [](const RTMap & info) {
        const std::string & key = ngraph::FusedNames::get_type_info_static();
        ASSERT_TRUE(info.count(key));
        auto fused_names_attr = info.at(key).as<ngraph::FusedNames>();
        ASSERT_EQ(fused_names_attr.getNames(), "add");

        const std::string & pkey = ov::PrimitivesPriority::get_type_info_static();
        ASSERT_TRUE(info.count(pkey));
        auto primitives_priority_attr = info.at(pkey).as<ov::PrimitivesPriority>().value;
        ASSERT_EQ(primitives_priority_attr, "priority");
    };

    auto add = f->get_results()[0]->get_input_node_ptr(0);
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), "NCHW");
    auto var0 = f->input(0).get_rt_info()
        .at(ov::preprocess::TensorInfoMemoryType::get_type_info_static())
        .as<ov::preprocess::TensorInfoMemoryType>().value;
    EXPECT_EQ(var0, "test_memory_type");
    EXPECT_EQ(f->get_results()[0]->get_layout(), "????");
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
    m.register_pass<pass::Serialize>(m_out_xml_path, m_out_bin_path, pass::Serialize::Version::IR_V11);
    m.run_passes(function);

    auto f = getWithIRFrontend(m_out_xml_path, m_out_bin_path);
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

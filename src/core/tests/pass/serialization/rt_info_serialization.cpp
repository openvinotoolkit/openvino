// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/runtime/core.hpp"
#include "transformations/hash.hpp"
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
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape{1, 3, 10, 10});
        data->set_layout("NCHW");
        auto add = std::make_shared<ov::op::v1::Add>(data, data);
        init_info(add->get_rt_info());
        init_info(add->input(0).get_rt_info());
        init_info(add->input(1).get_rt_info());
        init_info(add->output(0).get_rt_info());
        auto result = std::make_shared<ov::op::v0::Result>(add);
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
        OV_ASSERT_NO_THROW(info.at(dkey).as<ov::Decompression>());
    };

    auto add = f->get_results()[0]->get_input_node_ptr(0);
    EXPECT_EQ(f->get_parameters()[0]->get_layout(), "NCHW");
    EXPECT_EQ(f->get_results()[0]->get_layout(), "????");
    check_info(add->get_rt_info());
    check_info(add->input(0).get_rt_info());
    check_info(add->input(1).get_rt_info());
    check_info(add->output(0).get_rt_info());
}

TEST_F(RTInfoSerializationTest, rt_info_precise_test) {
    auto init_info = [](ov::RTMap& info) {
        info[ov::DisableFP16Compression::get_type_info_static()] = ov::DisableFP16Compression{};
    };
    auto check_info = [](const ov::RTMap& info) {
        const std::string& key = ov::DisableFP16Compression::get_type_info_static();
        ASSERT_TRUE(info.count(key));
    };

    std::shared_ptr<ov::Model> function;
    {
        auto data_1 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape{1, 10});
        auto data_2 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape{10, 1});
        auto matmul_1 = std::make_shared<ov::op::v0::MatMul>(data_1, data_2);
        init_info(matmul_1->get_rt_info());
        auto result = std::make_shared<ov::op::v0::Result>(matmul_1);
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{data_1, data_2});
    }
    ov::pass::Manager m;
    m.register_pass<ov::pass::Serialize>(m_out_xml_path, m_out_bin_path);
    m.run_passes(function);
    auto f = getWithIRFrontend(m_out_xml_path, m_out_bin_path);
    ASSERT_NE(nullptr, f);

    auto matmul = f->get_results()[0]->get_input_node_ptr(0);
    check_info(matmul->get_rt_info());
}

TEST_F(RTInfoSerializationTest, all_attributes_v10) {
    auto init_info = [](ov::RTMap& info) {
        info[ov::FusedNames::get_type_info_static()] = ov::FusedNames("add");
        info["PrimitivesPriority"] = ov::PrimitivesPriority("priority");
    };

    std::shared_ptr<ov::Model> function;
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape{1, 3, 10, 10});
        data->set_layout("NCHW");
        auto add = std::make_shared<ov::op::v1::Add>(data, data);
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

TEST_F(RTInfoSerializationTest, tag_names_verification) {
    std::map<std::string, std::string> test_cases = {
        {"0", "bad"},
        {"0a", "bad"},
        {"-a", "bad"},
        {"a 0", "bad"},
        {"a0", "good"},
        {"a.0", "good"},
        {".a0", "bad"},
        {"a_0", "good"},
        {"_0a", "bad"},
        {"aXmL", "good"},
        {"xMLa", "bad"},
        {"XML", "bad"},
    };
    auto init_info = [&test_cases](ov::RTMap& info) {
        for (const auto& item : test_cases) {
            info[item.first] = item.second;
        }
    };

    std::shared_ptr<ov::Model> model;
    {
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape{1, 3, 10, 10});
        model = std::make_shared<ov::Model>(ov::OutputVector{data}, ov::ParameterVector{data});
        init_info(model->get_rt_info());
    }

    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::Serialize>(m_out_xml_path, m_out_bin_path);
    pass_manager.run_passes(model);

    auto ir_model = getWithIRFrontend(m_out_xml_path, m_out_bin_path);
    ASSERT_NE(nullptr, ir_model);

    auto model_rt_info = ir_model->get_rt_info();
    std::for_each(test_cases.begin(),
                  test_cases.end(),
                  [&model_rt_info](const std::pair<std::string, std::string>& item) {
                      ASSERT_TRUE(model_rt_info.count(item.first));
                      ASSERT_EQ(model_rt_info[item.first], item.second);
                  });
}

TEST(OvSerializationTests, SerializeRawMeta) {
    std::string ir_with_rt_info = R"V0G0N(<?xml version="1.0"?>
<net name="Model0" version="11">
	<layers>
		<layer id="0" name="Parameter_1" type="Parameter" version="opset1">
			<data shape="3,20,20" element_type="u8" />
			<output>
				<port id="0" precision="U8">
					<dim>3</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Result_2" type="Result" version="opset1">
			<input>
				<port id="0" precision="U8">
					<dim>3</dim>
					<dim>20</dim>
					<dim>20</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
	</edges>
	<rt_info>
		<custom_rt_info1>
			<item0 value="testvalue1" />
		</custom_rt_info1>
		<custom_rt_info2>
			<item0 value="testvalue2" />
		</custom_rt_info2>
	</rt_info>
</net>
)V0G0N";
    ov::Core core;
    {
        // Don't read meta data. Copy raw pugixml::node from MetaDataWithPugixml to serialized model
        auto model = core.read_model(ir_with_rt_info, ov::Tensor());

        std::stringstream model_ss, weights_ss;
        ov::pass::Serialize(model_ss, weights_ss).run_on_model(model);

        auto serialized_model = model_ss.str();
        EXPECT_EQ(0, serialized_model.compare(ir_with_rt_info));
    }

    {
        // Don't read meta data. Fully serialize AnyMap with meta
        auto model = core.read_model(ir_with_rt_info, ov::Tensor());
        auto custom_rt_info1_value = model->get_rt_info<std::string>("custom_rt_info1", "item0");
        EXPECT_EQ(0, custom_rt_info1_value.compare("testvalue1"));
        auto custom_rt_info2_value = model->get_rt_info<std::string>("custom_rt_info2", "item0");
        EXPECT_EQ(0, custom_rt_info2_value.compare("testvalue2"));

        std::stringstream model_ss, weights_ss;
        ov::pass::Serialize(model_ss, weights_ss).run_on_model(model);

        auto serialized_model = model_ss.str();
        EXPECT_EQ(0, serialized_model.compare(ir_with_rt_info));
    }

    {
        auto model = core.read_model(ir_with_rt_info, ov::Tensor());
        auto custom_rt_info1_value = model->get_rt_info<std::string>("custom_rt_info1", "item0");
        EXPECT_EQ(0, custom_rt_info1_value.compare("testvalue1"));

        std::stringstream model_ss, weights_ss;
        ov::pass::Serialize(model_ss, weights_ss).run_on_model(model);

        auto serialized_model = model_ss.str();
        EXPECT_EQ(0, serialized_model.compare(ir_with_rt_info));
    }
}

namespace ov::test {

TEST(RTInfoSerialization, custom_info) {
    std::string ref_ir_xml = R"V0G0N(<?xml version="1.0"?>
<net name="CustomRTI" version="11">
	<layers>
		<layer id="0" name="node_A" type="Parameter" version="opset1">
			<data shape="10,10" element_type="f32" />
			<rt_info>
				<custom name="node_info_A" value="v_A" version="-1" />
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>10</dim>
					<dim>10</dim>
					<rt_info>
						<custom name="output_info_A" value="o_A" version="-1" />
					</rt_info>
				</port>
			</output>
		</layer>
		<layer id="1" name="node_B" type="Const" version="opset1">
			<data element_type="f32" shape="1" offset="0" size="4" />
			<rt_info>
				<custom name="node_info_B" value="v_B" version="-1" />
			</rt_info>
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<rt_info>
						<custom name="output_info_B" value="o_B" version="-1" />
					</rt_info>
				</port>
			</output>
		</layer>
		<layer id="2" name="node_C" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<rt_info>
				<custom name="node_info_C" value="v_C" version="-1" />
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>10</dim>
					<dim>10</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32">
					<dim>10</dim>
					<dim>10</dim>
					<rt_info>
						<custom name="output_info_C" value="o_C" version="-1" />
						<custom name="output_info_D" value="o_D" version="-1" />
					</rt_info>
				</port>
			</output>
		</layer>
		<layer id="3" name="node_D" type="Result" version="opset1">
			<rt_info>
				<custom name="node_info_D" value="v_D" version="-1" />
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>10</dim>
					<dim>10</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
		<edge from-layer="1" from-port="0" to-layer="2" to-port="1" />
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0" />
	</edges>
	<rt_info />
</net>
)V0G0N";

    const auto data = std::make_shared<op::v0::Parameter>(element::Type_t::f32, Shape{10, 10});
    const auto one = std::make_shared<op::v0::Constant>(element::f32, Shape{1}, std::vector<float>{1.f});
    const auto add = std::make_shared<op::v1::Add>(data, one);
    const auto result = std::make_shared<op::v0::Result>(add);

    const auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    uint64_t bare_model_hash;
    pass::Hash{bare_model_hash}.run_on_model(model);

    model->set_friendly_name("CustomRTI");

    const auto add_info = [](const std::shared_ptr<Node>& node, const std::string& value) {
        node->set_friendly_name("node_" + value);
        node->get_rt_info()["node_info_" + value] = "v_" + value;
        node->output(0).get_rt_info()["output_info_" + value] = "o_" + value;
    };
    add_info(data, "A");
    add_info(one, "B");
    add_info(add, "C");
    add_info(result, "D");
    result->get_rt_info()["__do not serialize"] = "double underscores in front";

    std::stringstream model_ss, weights_ss;
    EXPECT_NO_THROW((ov::pass::Serialize{model_ss, weights_ss}.run_on_model(model)));
    EXPECT_EQ(ref_ir_xml.compare(model_ss.str()), 0);

    uint64_t with_custom_rt_info_hash;
    pass::Hash{with_custom_rt_info_hash}.run_on_model(model);
    EXPECT_EQ(bare_model_hash, with_custom_rt_info_hash)
        << "`ov::pass::Hash' output value must not be affected by custom rt info.";
}

TEST(RTInfoSerialization, AnyMap_info) {
    std::string ref_ir_xml = R"V0G0N(<?xml version="1.0"?>
<net name="CustomRTI" version="11">
	<layers>
		<layer id="0" name="data" type="Parameter" version="opset1">
			<data shape="111" element_type="f64" />
			<output>
				<port id="0" precision="FP64">
					<dim>111</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="abs" type="Abs" version="opset1">
			<rt_info>
				<custom name="AnyMap" version="-1">
					<custom name="a" value="b" version="-1" />
					<custom name="i" value="7" version="-1" />
					<custom name="nested" version="-1">
						<custom name="c" value="d" version="-1" />
					</custom>
					<custom name="x" value="3.14" version="-1" />
				</custom>
			</rt_info>
			<input>
				<port id="0" precision="FP64">
					<dim>111</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP64">
					<dim>111</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="result" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP64">
					<dim>111</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
		<edge from-layer="1" from-port="1" to-layer="2" to-port="0" />
	</edges>
	<rt_info />
</net>
)V0G0N";

    const auto data = std::make_shared<op::v0::Parameter>(element::Type_t::f64, Shape{111});
    const auto abs = std::make_shared<op::v0::Abs>(data);
    const auto result = std::make_shared<op::v0::Result>(abs);

    const auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    uint64_t bare_model_hash;
    pass::Hash{bare_model_hash}.run_on_model(model);

    model->set_friendly_name("CustomRTI");
    data->set_friendly_name("data");
    abs->set_friendly_name("abs");
    result->set_friendly_name("result");

    const auto empty = AnyMap{};
    const auto nested = AnyMap{{"c", "d"}};
    abs->get_rt_info()["AnyMap"] = AnyMap{{"a", "b"}, {"empty", empty}, {"i", 7}, {"x", 3.14}, {"nested", nested}};

    std::stringstream model_ss, weights_ss;
    EXPECT_NO_THROW((ov::pass::Serialize{model_ss, weights_ss}.run_on_model(model)));
    EXPECT_EQ(ref_ir_xml.compare(model_ss.str()), 0);

    uint64_t with_custom_rt_info_hash;
    pass::Hash{with_custom_rt_info_hash}.run_on_model(model);
    EXPECT_EQ(bare_model_hash, with_custom_rt_info_hash)
        << "`ov::pass::Hash' output value must not be affected by custom rt info.";
}

TEST(RTInfoSerialization, nullptr_doesnt_throw) {
    const auto data = std::make_shared<op::v0::Parameter>(element::Type_t::f64, Shape{111});
    const auto abs = std::make_shared<op::v0::Abs>(data);
    const auto result = std::make_shared<op::v0::Result>(abs);

    abs->get_rt_info()["bare"] = ov::Any{nullptr};
    abs->get_rt_info()["void*"] = ov::Any{static_cast<void*>(nullptr)};
    abs->get_rt_info()["shared_ptr"] = ov::Any{std::shared_ptr<void>{}};
    abs->get_rt_info()["RuntimeAttribute"] = ov::Any{std::shared_ptr<ov::RuntimeAttribute>{}};

    const auto model = std::make_shared<Model>(ResultVector{result}, ParameterVector{data});
    std::stringstream model_ss, weights_ss;
    EXPECT_NO_THROW((ov::pass::Serialize{model_ss, weights_ss}.run_on_model(model)));
}
}  // namespace ov::test

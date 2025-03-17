// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/serialize.hpp"
#include "openvino/util/file_util.hpp"
#include "read_ir.hpp"

class DeterministicityCommon {
protected:
    std::string m_out_xml_path_1{};
    std::string m_out_bin_path_1{};
    std::string m_out_xml_path_2{};
    std::string m_out_bin_path_2{};
    std::string filePrefix{};

    void SetupFileNames() {
        filePrefix = ov::test::utils::generateTestFilePrefix();
        m_out_xml_path_1 = filePrefix + "1" + ".xml";
        m_out_bin_path_1 = filePrefix + "1" + ".bin";
        m_out_xml_path_2 = filePrefix + "2" + ".xml";
        m_out_bin_path_2 = filePrefix + "2" + ".bin";
    }

    void RemoveFiles() {
        std::remove(m_out_xml_path_1.c_str());
        std::remove(m_out_xml_path_2.c_str());
        std::remove(m_out_bin_path_1.c_str());
        std::remove(m_out_bin_path_2.c_str());
    }

    bool files_equal(std::ifstream& f1, std::ifstream& f2) {
        if (!f1.good())
            return false;
        if (!f2.good())
            return false;

        while (!f1.eof() && !f2.eof()) {
            if (f1.get() != f2.get()) {
                return false;
            }
        }

        if (f1.eof() != f2.eof()) {
            return false;
        }

        return true;
    }
};

class SerializationDeterministicityTest : public ov::test::TestsCommon, public DeterministicityCommon {
protected:
    void SetUp() override {
        SetupFileNames();
    }

    void TearDown() override {
        RemoveFiles();
    }
};

#ifdef ENABLE_OV_ONNX_FRONTEND

TEST_F(SerializationDeterministicityTest, BasicModel) {
    const std::string model =
        ov::test::utils::getModelFromTestModelZoo(ov::util::path_join({SERIALIZED_ZOO, "ir/add_abc.onnx"}).string());

    auto expected = ov::test::readModel(model, "");
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(expected);
    ov::pass::Serialize(m_out_xml_path_2, m_out_bin_path_2).run_on_model(expected);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}

TEST_F(SerializationDeterministicityTest, ModelWithMultipleLayers) {
    const std::string model =
        ov::test::utils::getModelFromTestModelZoo(ov::util::path_join({SERIALIZED_ZOO, "ir/addmul_abc.onnx"}).string());

    auto expected = ov::test::readModel(model, "");
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(expected);
    ov::pass::Serialize(m_out_xml_path_2, m_out_bin_path_2).run_on_model(expected);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}

#endif

TEST_F(SerializationDeterministicityTest, ModelWithMultipleOutputs) {
    const std::string model = ov::test::utils::getModelFromTestModelZoo(
        ov::util::path_join({SERIALIZED_ZOO, "ir/split_equal_parts_2d.xml"}).string());
    const std::string weights = ov::test::utils::getModelFromTestModelZoo(
        ov::util::path_join({SERIALIZED_ZOO, "ir/split_equal_parts_2d.bin"}).string());

    auto expected = ov::test::readModel(model, weights);
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(expected);
    ov::pass::Serialize(m_out_xml_path_2, m_out_bin_path_2).run_on_model(expected);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}

TEST_F(SerializationDeterministicityTest, ModelWithConstants) {
    const std::string model = ov::test::utils::getModelFromTestModelZoo(
        ov::util::path_join({SERIALIZED_ZOO, "ir/add_abc_initializers.xml"}).string());
    const std::string weights = ov::test::utils::getModelFromTestModelZoo(
        ov::util::path_join({SERIALIZED_ZOO, "ir/add_abc_initializers.bin"}).string());

    auto expected = ov::test::readModel(model, weights);
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(expected);
    ov::pass::Serialize(m_out_xml_path_2, m_out_bin_path_2).run_on_model(expected);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream bin_1(m_out_bin_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    std::ifstream bin_2(m_out_bin_path_2, std::ios::in | std::ios::binary);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
    ASSERT_TRUE(files_equal(bin_1, bin_2));
}

TEST_F(SerializationDeterministicityTest, ModelWithVariable) {
    const auto model = ov::test::utils::getModelFromTestModelZoo(
        ov::util::path_join({SERIALIZED_ZOO, "ir/dynamic_variable.xml"}).string());

    auto expected = ov::test::readModel(model, "");
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1).run_on_model(expected);

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in);
    std::ifstream xml_2(model, std::ios::in);

    ASSERT_TRUE(files_equal(xml_1, xml_2));
}

class SerializationDeterministicityInputOutputTest : public testing::TestWithParam<ov::pass::Serialize::Version>,
                                                     public DeterministicityCommon {
protected:
    std::string input0Name{"input0"};
    std::string input1Name{"input1"};
    std::string output0Name{"output0"};
    std::string output1Name{"output1"};

    std::string xmlFileName{};

    void SetupFileNames() {
        DeterministicityCommon::SetupFileNames();
        xmlFileName = filePrefix + "_TestModel.xml";
    }

    void RemoveFiles() {
        DeterministicityCommon::RemoveFiles();
        std::remove(xmlFileName.c_str());
    }

    void SetUp() override {
        SetupFileNames();
    }

    void TearDown() override {
        RemoveFiles();
    }
};

TEST_P(SerializationDeterministicityInputOutputTest, FromOvModel) {
    auto irVersion = GetParam();

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter0 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 22, 22});
        parameter0->set_friendly_name("input0");
        auto result0 = std::make_shared<ov::opset1::Result>(parameter0);
        result0->set_friendly_name("output0");
        auto parameter1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 22, 22});
        parameter1->set_friendly_name("input1");
        auto result1 = std::make_shared<ov::opset1::Result>(parameter1);
        result1->set_friendly_name("output1");
        modelRef =
            std::make_shared<ov::Model>(ov::NodeVector{result0, result1}, ov::ParameterVector{parameter0, parameter1});
    }

    auto& expected1 = modelRef;
    ov::pass::Serialize(m_out_xml_path_1, m_out_bin_path_1, irVersion).run_on_model(modelRef);
    auto expected2 = ov::test::readModel(m_out_xml_path_1, m_out_bin_path_1);

    ov::pass::Serialize(m_out_xml_path_2, m_out_bin_path_2, irVersion).run_on_model(expected2);

    EXPECT_EQ(input0Name, expected1->input(0).get_node()->get_friendly_name());
    EXPECT_EQ(input1Name, expected1->input(1).get_node()->get_friendly_name());
    EXPECT_EQ(output0Name, expected1->output(0).get_node()->get_friendly_name());
    EXPECT_EQ(output1Name, expected1->output(1).get_node()->get_friendly_name());
    EXPECT_EQ(input0Name, expected2->input(0).get_node()->get_friendly_name());
    EXPECT_EQ(input1Name, expected2->input(1).get_node()->get_friendly_name());
    EXPECT_EQ(output0Name, expected2->output(0).get_node()->get_friendly_name());
    EXPECT_EQ(output1Name, expected2->output(1).get_node()->get_friendly_name());

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    EXPECT_TRUE(files_equal(xml_1, xml_2));
}

TEST_P(SerializationDeterministicityInputOutputTest, FromIrModel) {
    auto irVersion = GetParam();

    std::string irModel_1stPart = R"V0G0N(<?xml version="1.0"?>
<net name="Model0" version=")V0G0N";
    std::string irModel_2ndPart = R"V0G0N(">
	<layers>
		<layer id="0" name="input0" type="Parameter" version="opset1">
			<data shape="1,3,22,22" element_type="f32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="input1" type="Parameter" version="opset1">
			<data shape="1,3,22,22" element_type="f32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="output0" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</input>
		</layer>
		<layer id="3" name="output1" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>3</dim>
					<dim>22</dim>
					<dim>22</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="0" />
		<edge from-layer="1" from-port="0" to-layer="3" to-port="0" />
	</edges>
	<rt_info />
</net>
)V0G0N";
    std::string strVersion = irVersion == ov::pass::Serialize::Version::IR_V11 ? "11" : "10";
    std::string irModel = irModel_1stPart + strVersion + irModel_2ndPart;

    {
        std::ofstream xmlFile;
        xmlFile.open(xmlFileName);
        xmlFile << irModel;
        xmlFile.close();
    }

    auto expected1 = ov::test::readModel(xmlFileName, "");
    ov::pass::Serialize(m_out_xml_path_1, "", irVersion).run_on_model(expected1);
    auto expected2 = ov::test::readModel(m_out_xml_path_1, "");
    ov::pass::Serialize(m_out_xml_path_2, "", irVersion).run_on_model(expected2);

    EXPECT_EQ(input0Name, expected1->input(0).get_node()->get_friendly_name());
    EXPECT_EQ(input1Name, expected1->input(1).get_node()->get_friendly_name());
    EXPECT_EQ(output0Name, expected1->output(0).get_node()->get_friendly_name());
    EXPECT_EQ(output1Name, expected1->output(1).get_node()->get_friendly_name());
    EXPECT_EQ(input0Name, expected2->input(0).get_node()->get_friendly_name());
    EXPECT_EQ(input1Name, expected2->input(1).get_node()->get_friendly_name());
    EXPECT_EQ(output0Name, expected2->output(0).get_node()->get_friendly_name());
    EXPECT_EQ(output1Name, expected2->output(1).get_node()->get_friendly_name());

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    EXPECT_TRUE(files_equal(xml_2, xml_1));
}

TEST_P(SerializationDeterministicityInputOutputTest, FromOvModelBybPath) {
    auto irVersion = GetParam();

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter0 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 22, 22});
        parameter0->set_friendly_name("input0");
        auto result0 = std::make_shared<ov::opset1::Result>(parameter0);
        result0->set_friendly_name("output0");
        auto parameter1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 22, 22});
        parameter1->set_friendly_name("input1");
        auto result1 = std::make_shared<ov::opset1::Result>(parameter1);
        result1->set_friendly_name("output1");
        modelRef =
            std::make_shared<ov::Model>(ov::NodeVector{result0, result1}, ov::ParameterVector{parameter0, parameter1});
    }

    auto& expected1 = modelRef;
    const auto out_xml_path = std::filesystem::path(m_out_xml_path_1);
    const auto out_bin_path = std::filesystem::path(m_out_bin_path_1);
    ov::pass::Serialize(out_xml_path, out_bin_path, irVersion).run_on_model(modelRef);
    auto expected2 = ov::test::readModel(m_out_xml_path_1, m_out_bin_path_1);

    ov::pass::Serialize(m_out_xml_path_2, m_out_bin_path_2, irVersion).run_on_model(expected2);

    EXPECT_EQ(input0Name, expected1->input(0).get_node()->get_friendly_name());
    EXPECT_EQ(input1Name, expected1->input(1).get_node()->get_friendly_name());
    EXPECT_EQ(output0Name, expected1->output(0).get_node()->get_friendly_name());
    EXPECT_EQ(output1Name, expected1->output(1).get_node()->get_friendly_name());
    EXPECT_EQ(input0Name, expected2->input(0).get_node()->get_friendly_name());
    EXPECT_EQ(input1Name, expected2->input(1).get_node()->get_friendly_name());
    EXPECT_EQ(output0Name, expected2->output(0).get_node()->get_friendly_name());
    EXPECT_EQ(output1Name, expected2->output(1).get_node()->get_friendly_name());

    std::ifstream xml_1(m_out_xml_path_1, std::ios::in | std::ios::binary);
    std::ifstream xml_2(m_out_xml_path_2, std::ios::in | std::ios::binary);
    EXPECT_TRUE(files_equal(xml_1, xml_2));
}

INSTANTIATE_TEST_SUITE_P(DeterministicityInputOutput,
                         SerializationDeterministicityInputOutputTest,
                         ::testing::Values(ov::pass::Serialize::Version::IR_V10, ov::pass::Serialize::Version::IR_V11));

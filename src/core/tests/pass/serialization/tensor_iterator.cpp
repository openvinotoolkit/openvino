// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <fstream>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/graph_comparator.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/tensor.hpp"

class SerializationTensorIteratorTest : public ::testing::Test {
protected:
    std::string m_out_xml_path;
    std::string m_out_bin_path;
    std::string tmpXmlFileName;
    std::string tmpBinFileName;

    void SetUp() override {
        std::string filePrefix = ov::test::utils::generateTestFilePrefix();
        m_out_xml_path = filePrefix + ".xml";
        m_out_bin_path = filePrefix + ".bin";
        tmpXmlFileName = filePrefix + "TestModel.xml";
        tmpBinFileName = filePrefix + "TestModel.bin";
    }

    void TearDown() override {
        std::remove(m_out_xml_path.c_str());
        std::remove(m_out_bin_path.c_str());
        std::remove(tmpXmlFileName.c_str());
        std::remove(tmpBinFileName.c_str());
    }

    void createTemporalModelFile(std::string xmlFileContent,
                                 std::vector<unsigned char> binFileContent = std::vector<unsigned char>()) {
        ASSERT_TRUE(xmlFileContent.size() > 0);

        {
            std::ofstream xmlFile;
            xmlFile.open(tmpXmlFileName);
            xmlFile << xmlFileContent;
            xmlFile.close();
        }

        if (binFileContent.size() > 0) {
            std::ofstream binFile;
            binFile.open(tmpBinFileName, std::ios::binary);
            binFile.write((const char*)binFileContent.data(), binFileContent.size());
            binFile.close();
        }
    }

    void serialize_and_compare(const std::string& model_path, ov::Tensor weights) {
        std::stringstream buffer;
        ov::Core core;

        std::ifstream model(model_path);
        ASSERT_TRUE(model);
        buffer << model.rdbuf();

        auto expected = core.read_model(buffer.str(), weights);
        ov::serialize(expected, m_out_xml_path, m_out_bin_path);
        auto result = core.read_model(m_out_xml_path, m_out_bin_path);

        bool success;
        std::string message;
        std::tie(success, message) = compare_functions(result, expected, true, false, false, true, true);
        ASSERT_TRUE(success) << message;
    }
};

TEST_F(SerializationTensorIteratorTest, TiResnet) {
    std::string xmlModel = R"V0G0N(
<net name="Resnet" version="10">
    <layers>
        <layer id="0" name="data1" type="Parameter" version="opset1">
            <data element_type="f32" shape="16,1,512"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>16</dim>
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="data2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,512"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="data3" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,512"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="471/TensorIterator" type="TensorIterator" version="opset1">
            <input>
                <port id="0">
                    <dim>16</dim>
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>16</dim>
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
                <port id="4" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
                <port id="5" precision="FP32">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
            </output>
            <port_map>
                <input axis="0" external_port_id="0" internal_layer_id="0" part_size="1" stride="1"/>
                <input external_port_id="1" internal_layer_id="3"/>
                <input external_port_id="2" internal_layer_id="4"/>
                <output axis="0" external_port_id="3" internal_layer_id="13" part_size="1" stride="1"/>
                <output external_port_id="4" internal_layer_id="9"/>
                <output external_port_id="5" internal_layer_id="10"/>
            </port_map>
            <back_edges>
                <edge from-layer="9" to-layer="3"/>
                <edge from-layer="10" to-layer="4"/>
            </back_edges>
            <body>
                <layers>
                <!--
                ***************************************************************************************
                * nodes are place unordered to check if XmlDeserializer is not depend on layers order *
                ***************************************************************************************
                -->
                    <layer id="1" name="7_const" type="Const" version="opset1">
                        <data element_type="i64" offset="0" shape="2" size="16"/>
                        <output>
                            <port id="1" precision="I64">
                                <dim>2</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="2" name="471/input_squeeze" type="Reshape" version="opset1">
                        <data special_zero="True"/>
                        <input>
                            <port id="0">
                                <dim>1</dim>
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                            <port id="1">
                                <dim>2</dim>
                            </port>
                        </input>
                        <output>
                            <port id="2" precision="FP32">
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="3" name="22" type="Parameter" version="opset1">
                        <data element_type="f32" shape="1,512"/>
                        <output>
                            <port id="0" precision="FP32">
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="4" name="24" type="Parameter" version="opset1">
                        <data element_type="f32" shape="1,512"/>
                        <output>
                            <port id="0" precision="FP32">
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="5" name="471/LSTMCell/Split149_const" type="Const" version="opset1">
                        <data element_type="f32" offset="16" shape="2048,512" size="4194304"/>
                        <output>
                            <port id="1" precision="FP32">
                                <dim>2048</dim>
                                <dim>512</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="6" name="471/LSTMCell/Split150_const" type="Const" version="opset1">
                        <data element_type="f32" offset="4194320" shape="2048,512" size="4194304"/>
                        <output>
                            <port id="1" precision="FP32">
                                <dim>2048</dim>
                                <dim>512</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="7" name="471/inport/2_const" type="Const" version="opset1">
                        <data element_type="f32" offset="8388624" shape="2048" size="8192"/>
                        <output>
                            <port id="1" precision="FP32">
                                <dim>2048</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="8" name="471/LSTMCell" type="LSTMCell" version="opset1">
                        <data hidden_size="512"/>
                        <input>
                            <port id="0">
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                            <port id="1">
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                            <port id="2">
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                            <port id="3">
                                <dim>2048</dim>
                                <dim>512</dim>
                            </port>
                            <port id="4">
                                <dim>2048</dim>
                                <dim>512</dim>
                            </port>
                            <port id="5">
                                <dim>2048</dim>
                            </port>
                        </input>
                        <output>
                            <port id="6" precision="FP32">
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                            <port id="7" precision="FP32">
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="13" name="18/sink_port_0" type="Result" version="opset1">
                        <input>
                            <port id="0">
                                <dim>1</dim>
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                        </input>
                    </layer>
                    <layer id="9" name="471/outport/0/sink_port_0" type="Result" version="opset1">
                        <input>
                            <port id="0">
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                        </input>
                    </layer>
                    <layer id="10" name="471/outport/1/sink_port_0" type="Result" version="opset1">
                        <input>
                            <port id="0">
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                        </input>
                    </layer>
                    <layer id="11" name="15_const" type="Const" version="opset1">
                        <data element_type="i64" offset="8396816" shape="3" size="24"/>
                        <output>
                            <port id="1" precision="I64">
                                <dim>3</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="12" name="471output_unsqueeze" type="Reshape" version="opset1">
                        <data special_zero="True"/>
                        <input>
                            <port id="0">
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                            <port id="1">
                                <dim>3</dim>
                            </port>
                        </input>
                        <output>
                            <port id="2" precision="FP32">
                                <dim>1</dim>
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="0" name="20" type="Parameter" version="opset1">
                        <data element_type="f32" shape="1,1,512"/>
                        <output>
                            <port id="0" precision="FP32">
                                <dim>1</dim>
                                <dim>1</dim>
                                <dim>512</dim>
                            </port>
                        </output>
                    </layer>
                </layers>
                <edges>
                    <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                    <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                    <edge from-layer="2" from-port="2" to-layer="8" to-port="0"/>
                    <edge from-layer="3" from-port="0" to-layer="8" to-port="1"/>
                    <edge from-layer="4" from-port="0" to-layer="8" to-port="2"/>
                    <edge from-layer="5" from-port="1" to-layer="8" to-port="3"/>
                    <edge from-layer="6" from-port="1" to-layer="8" to-port="4"/>
                    <edge from-layer="7" from-port="1" to-layer="8" to-port="5"/>
                    <edge from-layer="8" from-port="6" to-layer="9" to-port="0"/>
                    <edge from-layer="8" from-port="7" to-layer="10" to-port="0"/>
                    <edge from-layer="8" from-port="6" to-layer="12" to-port="0"/>
                    <edge from-layer="11" from-port="1" to-layer="12" to-port="1"/>
                    <edge from-layer="12" from-port="2" to-layer="13" to-port="0"/>
                </edges>
            </body>
        </layer>
        <layer id="4" name="result" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>16</dim>
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
            </input>
        </layer>
        <layer id="5" name="result_2" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
            </input>
        </layer>
        <layer id="6" name="result_3" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>512</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
        <edge from-layer="3" from-port="4" to-layer="5" to-port="0"/>
        <edge from-layer="3" from-port="5" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::vector<unsigned char> buffer(8396840, 0);
    int64_t* int64Buffer = reinterpret_cast<int64_t*>(buffer.data());
    int64Buffer[0] = 1;
    int64Buffer[1] = 512;
    int64Buffer[1049602] = 1;
    int64Buffer[1049603] = 1;
    int64Buffer[1049604] = 512;

    createTemporalModelFile(xmlModel, buffer);

    size_t weights_size = 8396840;

    auto weights = ov::Tensor(ov::element::f32, {weights_size / sizeof(float)});
    ov::test::utils::fill_data(weights.data<float>(), weights.get_size());

    auto* data = static_cast<int64_t*>(weights.data());
    data[0] = 1;
    data[1] = 512;
    data[1049602] = 1;
    data[1049603] = 1;
    data[1049604] = 512;

    serialize_and_compare(tmpXmlFileName, weights);
}

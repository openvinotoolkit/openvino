// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <iostream>

#include "common_test_utils/graph_comparator.hpp"
#include "openvino/core/extension.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/openvino.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset8.hpp"

class IRFrontendTests : public ::testing::Test {
protected:
    ov::Core core;
    ov::frontend::FrontEndManager manager;

    void SetUp() override {
        RemoveTemporalFiles();
    };

    void TearDown() override {
        RemoveTemporalFiles();
    };

    std::string xmlFileName = "IrFrontendTestModel.xml";
    std::string binFileName = "IrFrontendTestModel.bin";

    void createTemporalModelFile(std::string xmlFileContent,
                                 std::vector<unsigned char> binFileContent = std::vector<unsigned char>()) {
        ASSERT_TRUE(xmlFileContent.size() > 0);

        {
            std::ofstream xmlFile;
            xmlFile.open(xmlFileName);
            xmlFile << xmlFileContent;
            xmlFile.close();
        }

        if (binFileContent.size() > 0) {
            std::ofstream binFile;
            binFile.open(binFileName, std::ios::binary);
            binFile.write((const char*)binFileContent.data(), binFileContent.size());
            binFile.close();
        }
    }

    void RemoveTemporalFiles() {
        std::remove(xmlFileName.c_str());
        std::remove(binFileName.c_str());
    }

    std::shared_ptr<ov::Model> getWithIRFrontend(const std::string& model) {
        std::istringstream modelStringStream(model);
        std::istream& modelStream = modelStringStream;

        ov::frontend::FrontEnd::Ptr FE;
        ov::frontend::InputModel::Ptr inputModel;

        ov::AnyVector params{&modelStream};

        FE = manager.load_by_model(params);
        if (FE)
            inputModel = FE->load(params);

        if (inputModel)
            return FE->convert(inputModel);

        return nullptr;
    }
};

TEST_F(IRFrontendTests, ElementaryNetworkReadV11) {
    std::string testNetworkV11 = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;
    ov::RTMap rtInfo;
    uint64_t version;

    ASSERT_NO_THROW(model = getWithIRFrontend(testNetworkV11));
    ASSERT_TRUE(!!model);
    ASSERT_NO_THROW(rtInfo = model->get_rt_info());
    ASSERT_NO_THROW(version = rtInfo["version"].as<int64_t>());
    ASSERT_EQ(11, version);

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 22, 22});
        parameter->set_friendly_name("input");
        auto result = std::make_shared<ov::opset1::Result>(parameter);
        result->set_friendly_name("output");
        modelRef = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{parameter});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::NAMES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, modelRef);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST_F(IRFrontendTests, ElementaryNetworkReadV10) {
    std::string testNetworkV10 = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> modelv10;
    ov::RTMap rtInfoV10;
    uint64_t version;

    ASSERT_NO_THROW(modelv10 = getWithIRFrontend(testNetworkV10));
    ASSERT_TRUE(!!modelv10);
    ASSERT_NO_THROW(rtInfoV10 = modelv10->get_rt_info());
    ASSERT_NO_THROW(version = rtInfoV10["version"].as<int64_t>());
    ASSERT_EQ(10, version);

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 22, 22});
        parameter->set_friendly_name("input");
        auto result = std::make_shared<ov::opset1::Result>(parameter);
        result->set_friendly_name("output");
        modelRef = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{parameter});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::NAMES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(modelv10, modelRef);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST_F(IRFrontendTests, ElementaryNetworkReadV9) {
    std::string testNetworkV9 = R"V0G0N(
<net name="Network" version="9">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> modelv9;
    ASSERT_THROW(modelv9 = core.read_model(testNetworkV9, ov::Tensor()), ov::Exception);
    ASSERT_FALSE(!!modelv9);
}

TEST_F(IRFrontendTests, NetworkWithMissingWeights) {
    std::string testNetworkV11 = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="value1" type="Const" version="opset1">
            <data element_type="i64" shape="4" offset="0" size="32" />
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Transpose0321" type="Transpose" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    ASSERT_THROW((void)core.read_model(testNetworkV11, ov::Tensor()), ov::Exception);
}

TEST_F(IRFrontendTests, NetworkWithWeightsFromDisk) {
    std::string xmlModel = R"V0G0N(
<?xml version="1.0" ?>
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="value1" type="Const" version="opset1">
            <data element_type="i64" shape="4" offset="0" size="32" />
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Transpose0321" type="Transpose" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::vector<unsigned char> buffer(32, 0);
    uint64_t* uint64Buffer = reinterpret_cast<uint64_t*>(buffer.data());
    uint64Buffer[0] = 0;
    uint64Buffer[1] = 3;
    uint64Buffer[2] = 2;
    uint64Buffer[3] = 1;

    createTemporalModelFile(xmlModel, buffer);

    std::shared_ptr<ov::Model> model;

    ASSERT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_TRUE(!!model);

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 22, 22});
        parameter->set_friendly_name("input");
        auto constant =
            std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{4}, std::vector<uint64_t>{0, 3, 2, 1});
        constant->set_friendly_name("value1");
        auto transpose = std::make_shared<ov::opset1::Transpose>(parameter, constant);
        transpose->set_friendly_name("Transpose0321");
        auto result = std::make_shared<ov::opset1::Result>(transpose);
        result->set_friendly_name("output");
        modelRef = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{parameter});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::NAMES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, modelRef);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST_F(IRFrontendTests, NetworkWithoutWeightsFromDisk) {
    std::string xmlModel = R"V0G0N(
<?xml version="1.0" ?>
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    createTemporalModelFile(xmlModel);

    std::shared_ptr<ov::Model> model;

    ASSERT_NO_THROW(model = core.read_model(xmlFileName));
    ASSERT_TRUE(!!model);

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 22, 22});
        parameter->set_friendly_name("input");
        auto result = std::make_shared<ov::opset1::Result>(parameter);
        result->set_friendly_name("output");
        modelRef = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{parameter});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::NAMES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, modelRef);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST_F(IRFrontendTests, NetworkWithWrongShape) {
    std::string xmlModel = R"V0G0N(
<?xml version="1.0" ?>
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="value1" type="Const" version="opset1">
            <data element_type="i64" shape="4" offset="0" size="16" />
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Transpose0321" type="Transpose" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::vector<unsigned char> buffer(32, 0);
    uint64_t* uint64Buffer = reinterpret_cast<uint64_t*>(buffer.data());
    uint64Buffer[0] = 0;
    uint64Buffer[1] = 3;
    uint64Buffer[2] = 2;
    uint64Buffer[3] = 1;

    createTemporalModelFile(xmlModel, buffer);
    std::shared_ptr<ov::Model> model;

    ASSERT_THROW((void)core.read_model(xmlFileName, binFileName), ov::Exception);
}

TEST_F(IRFrontendTests, NetworkWithUnderallocatedWeightsFromDisk) {
    std::string xmlModel = R"V0G0N(
<?xml version="1.0" ?>
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="value1" type="Const" version="opset1">
            <data element_type="i64" shape="4" offset="0" size="32" />
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Transpose0321" type="Transpose" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::vector<unsigned char> buffer(24, 0);
    uint64_t* uint64Buffer = reinterpret_cast<uint64_t*>(buffer.data());
    uint64Buffer[0] = 0;
    uint64Buffer[1] = 3;
    uint64Buffer[2] = 2;

    createTemporalModelFile(xmlModel, buffer);
    std::shared_ptr<ov::Model> model;

    ASSERT_THROW((void)core.read_model(xmlFileName, binFileName), ov::Exception);
}

TEST_F(IRFrontendTests, NetworkWithMissingWeightsFromDisk) {
    std::string xmlModel = R"V0G0N(
<?xml version="1.0" ?>
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="value1" type="Const" version="opset1">
            <data element_type="i64" shape="4" offset="0" size="32" />
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Transpose0321" type="Transpose" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    createTemporalModelFile(xmlModel);
    std::shared_ptr<ov::Model> model;

    ASSERT_THROW((void)core.read_model(xmlFileName, binFileName), ov::Exception);
}

TEST_F(IRFrontendTests, LayerWithoutData) {
    std::string network = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    ASSERT_THROW((void)core.read_model(network, ov::Tensor()), ov::Exception);
}

TEST_F(IRFrontendTests, NetworkWirhWrongDimensions) {
    std::string testNetwork = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>-2</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;

    ASSERT_THROW(model = core.read_model(testNetwork, ov::Tensor()), ov::Exception);
    ASSERT_TRUE(!model);
}

TEST_F(IRFrontendTests, NameIsNotUnique) {
    std::string xmlModel = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="testname" type="Const" version="opset1">
            <data element_type="i64" shape="4" offset="0" size="32" />
            <output>
                <port id="0" precision="I64">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="testname" type="Transpose" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
                <port id="1" precision="I64">
                    <dim>4</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                    <dim>3</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::vector<unsigned char> buffer(32, 0);
    int64_t* int64Buffer = reinterpret_cast<int64_t*>(buffer.data());
    int64Buffer[0] = 0;
    int64Buffer[1] = 3;
    int64Buffer[2] = 2;
    int64Buffer[3] = 1;

    createTemporalModelFile(xmlModel, buffer);
    std::shared_ptr<ov::Model> model;

    ASSERT_THROW((void)core.read_model(xmlFileName, binFileName), ov::Exception);
}

TEST_F(IRFrontendTests, CustomOpsTestWithFrameworkNodeExtension) {
    static std::string customOpsNetwork = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="customOp1" id="1" type="testtype" version="testopset">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::shared_ptr<ov::Model> model;
    auto extension = std::make_shared<ov::OpExtension<ov::op::util::FrameworkNode>>();

    core.add_extension(extension);

    ASSERT_NO_THROW(model = core.read_model(customOpsNetwork, ov::Tensor()));
    ASSERT_TRUE(!!model);
}

TEST_F(IRFrontendTests, CustomOpsTestWithoutExtension) {
    static std::string customOpsNetwork = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="customOp1" id="1" type="testtype" version="testopset">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    std::shared_ptr<ov::Model> model;

    ASSERT_THROW(model = core.read_model(customOpsNetwork, ov::Tensor()), ov::Exception);
    ASSERT_TRUE(!model);
}

TEST_F(IRFrontendTests, DISABLED_PortsMismatch) {
    std::string testNetwork = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,2"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;

    ASSERT_THROW(model = core.read_model(testNetwork, ov::Tensor()), ov::Exception);
    ASSERT_FALSE(!!model);
}

TEST_F(IRFrontendTests, DISABLED_PortDataMismatch) {
    std::string testNetwork = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,2,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;

    ASSERT_THROW(model = core.read_model(testNetwork, ov::Tensor()), ov::Exception);
    ASSERT_FALSE(!!model);
}

TEST_F(IRFrontendTests, EdgeHasWrongPortId) {
    std::string testNetwork = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="10"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;

    ASSERT_THROW(model = core.read_model(testNetwork, ov::Tensor()), ov::Exception);
    ASSERT_FALSE(!!model);
}

TEST_F(IRFrontendTests, EdgeHasWrongLayerId) {
    std::string testNetwork = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;

    ASSERT_THROW(model = core.read_model(testNetwork, ov::Tensor()), ov::Exception);
    ASSERT_FALSE(!!model);
}

TEST_F(IRFrontendTests, NotOpset1) {
    std::string testNetwork = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="shapeof" type="ShapeOf" version="opset3">
            <data output_type="i32" />
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="I32">
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="I32">
                    <dim>4</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;

    ASSERT_NO_THROW(model = getWithIRFrontend(testNetwork));
    ASSERT_TRUE(!!model);

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 22, 22});
        parameter->set_friendly_name("input");
        auto shapeof = std::make_shared<ov::opset3::ShapeOf>(parameter, ov::element::i32);
        shapeof->set_friendly_name("shapeof");
        auto result = std::make_shared<ov::opset1::Result>(shapeof);
        result->set_friendly_name("output");
        modelRef = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{parameter});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::NAMES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, modelRef);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST_F(IRFrontendTests, WrongOpset) {
    std::string testNetwork = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="wrongOpset">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;

    ASSERT_THROW(model = core.read_model(testNetwork, ov::Tensor()), ov::Exception);
    ASSERT_FALSE(!!model);
}

TEST_F(IRFrontendTests, TensorIteratorMergedInput) {
    std::string testNetwork = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer id="0" name="Parameter1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,2,3"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="TensorIterator" type="TensorIterator" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
            <port_map>
                <input external_port_id="0" internal_layer_id="0"/>
                <output external_port_id="1" internal_layer_id="1"/>
            </port_map>
            <back_edges>
                <edge from-layer="1" to-layer="0"/>
            </back_edges>
            <body>
                <layers>
                    <layer id="0" name="internalParameter1" type="Parameter" version="opset1">
                        <data element_type="f32" shape="1,2,3"/>
                        <output>
                            <port id="0" precision="FP32">
                                <dim>1</dim>
                                <dim>2</dim>
                                <dim>3</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="1" name="internalResult1" type="Result" version="opset1">
                        <input>
                            <port id="0">
                                <dim>1</dim>
                                <dim>2</dim>
                                <dim>3</dim>
                            </port>
                        </input>
                    </layer>
                </layers>
                <edges>
                    <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
                </edges>
            </body>
        </layer>
        <layer id="2" name="Result1" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;

    ASSERT_NO_THROW(model = core.read_model(testNetwork, ov::Tensor()));
    ASSERT_TRUE(!!model);

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        parameter->set_friendly_name("Parameter1");
        auto tensor_iterator = std::make_shared<ov::opset8::TensorIterator>();

        std::shared_ptr<ov::Model> body;
        auto internalParameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        internalParameter->set_friendly_name("internalParameter1");
        auto result1 = std::make_shared<ov::opset1::Result>(internalParameter);
        result1->set_friendly_name("internalResult1");
        body = std::make_shared<ov::Model>(ov::NodeVector{result1}, ov::ParameterVector{internalParameter});
        tensor_iterator->set_body(body);
        tensor_iterator->set_friendly_name("TensorIterator");
        tensor_iterator->set_merged_input(internalParameter, parameter, result1);
        auto out0 = tensor_iterator->get_iter_value(result1, -1);

        auto result = std::make_shared<ov::opset1::Result>(tensor_iterator->output(0));
        result->set_friendly_name("Result1");

        modelRef = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{parameter});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::NAMES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, modelRef);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST_F(IRFrontendTests, TensorIteratorSlisedInput) {
    std::string testNetwork = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer id="0" name="Parameter1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,2,3"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="TensorIterator" type="TensorIterator" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </output>
            <port_map>
                <input axis="2" external_port_id="0" internal_layer_id="0" part_size="1" stride="1"/>
                <output axis="2" external_port_id="1" internal_layer_id="1" part_size="1" stride="1"/>
            </port_map>
            <back_edges>
                <edge from-layer="1" to-layer="0"/>
            </back_edges>
            <body>
                <layers>
                    <layer id="0" name="internalParameter1" type="Parameter" version="opset1">
                        <data element_type="f32" shape="1,2,1"/>
                        <output>
                            <port id="0" precision="FP32">
                                <dim>1</dim>
                                <dim>2</dim>
                                <dim>1</dim>
                            </port>
                        </output>
                    </layer>
                    <layer id="1" name="internalResult1" type="Result" version="opset1">
                        <input>
                            <port id="0">
                                <dim>1</dim>
                                <dim>2</dim>
                                <dim>1</dim>
                            </port>
                        </input>
                    </layer>
                </layers>
                <edges>
                    <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
                </edges>
            </body>
        </layer>
        <layer id="2" name="Result1" type="Result" version="opset1">
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>2</dim>
                    <dim>3</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="1" from-port="1" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;

    ASSERT_NO_THROW(model = core.read_model(testNetwork, ov::Tensor()));
    ASSERT_TRUE(!!model);

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        parameter->set_friendly_name("Parameter1");
        auto tensor_iterator = std::make_shared<ov::opset8::TensorIterator>();

        std::shared_ptr<ov::Model> body;
        auto internalParameter = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 2, 3});
        internalParameter->set_friendly_name("internalParameter1");
        auto result1 = std::make_shared<ov::opset1::Result>(internalParameter);
        result1->set_friendly_name("internalResult1");
        body = std::make_shared<ov::Model>(ov::NodeVector{result1}, ov::ParameterVector{internalParameter});
        tensor_iterator->set_body(body);
        tensor_iterator->set_friendly_name("TensorIterator");
        tensor_iterator->set_sliced_input(internalParameter, parameter, 0, 1, 1, -1, 2);
        auto out0 = tensor_iterator->get_concatenated_slices(result1, 0, 1, 1, -1, 2);

        auto result = std::make_shared<ov::opset1::Result>(tensor_iterator->output(0));
        result->set_friendly_name("Result1");

        modelRef = std::make_shared<ov::Model>(ov::NodeVector{result}, ov::ParameterVector{parameter});
    }

    const auto fc = FunctionsComparator::with_default()
                        .enable(FunctionsComparator::ATTRIBUTES)
                        .enable(FunctionsComparator::PRECISIONS)
                        .enable(FunctionsComparator::RUNTIME_KEYS)
                        .enable(FunctionsComparator::NAMES)
                        .enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(model, modelRef);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST_F(IRFrontendTests, PreProcessing) {
    std::string xmlModel = R"V0G0N(
<?xml version="1.0" ?>
<net name="Network" version="10">
    <pre-process mean-precision="FP32" reference-layer-name="input">
        <channel id="0">
            <mean offset="0" size="1936"/>
        </channel>
        <channel id="1">
            <mean offset="1936" size="1936"/>
        </channel>
        <channel id="2">
            <mean offset="3872" size="1936"/>
        </channel>
    </pre-process>
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data shape="1,3,22,22" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
    </edges>
</net>
)V0G0N";

    int dataSizeinFloat = 22 * 22 * 3;
    std::vector<unsigned char> buffer(dataSizeinFloat * sizeof(float), 0);
    float* floatBuffer = reinterpret_cast<float*>(buffer.data());
    for (int i = 0; i < dataSizeinFloat; i++) {
        floatBuffer[i] = 1;
    }

    createTemporalModelFile(xmlModel, buffer);

    std::shared_ptr<ov::Model> model;

    ASSERT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_TRUE(!!model);
}

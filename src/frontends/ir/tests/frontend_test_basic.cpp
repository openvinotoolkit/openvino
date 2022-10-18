// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend_test.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"

class IRFrontendTests : public ::testing::Test, public IRFrontendTestsImpl {
protected:
    void SetUp() override {}

    void TearDown() override {
        RemoveTemporalFiles();
    }
};

TEST_F(IRFrontendTests, elementary_model_reading_v11) {
    std::string testModelV11 = R"V0G0N(
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

    ASSERT_NO_THROW(model = getWithIRFrontend(testModelV11));
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

TEST_F(IRFrontendTests, elementary_model_reading_v10) {
    std::string testModelV10 = R"V0G0N(
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

    ASSERT_NO_THROW(modelv10 = getWithIRFrontend(testModelV10));
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

TEST_F(IRFrontendTests, DISABLED_elementary_model_reading_v9) {
    std::string testModelV9 = R"V0G0N(
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
    ASSERT_THROW(modelv9 = core.read_model(testModelV9, ov::Tensor()), ov::Exception);
    ASSERT_FALSE(!!modelv9);
}

TEST_F(IRFrontendTests, model_with_missing_weights) {
    std::string testModelV11 = R"V0G0N(
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

    ASSERT_THROW(core.read_model(testModelV11, ov::Tensor()), ov::Exception);
}

TEST_F(IRFrontendTests, model_with_weights_reading_from_disk) {
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

TEST_F(IRFrontendTests, model_without_weights_reading_from_disk) {
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

TEST_F(IRFrontendTests, model_with_wrong_shape) {
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

    ASSERT_THROW(core.read_model(xmlFileName, binFileName), ov::Exception);
}

TEST_F(IRFrontendTests, model_with_underallocated_weights_reading_from_disk) {
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

    ASSERT_THROW(core.read_model(xmlFileName, binFileName), ov::Exception);
}

TEST_F(IRFrontendTests, model_with_missing_weights_from_disk) {
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

    ASSERT_THROW(core.read_model(xmlFileName, binFileName), ov::Exception);
}

TEST_F(IRFrontendTests, missing_layer_data) {
    std::string model = R"V0G0N(
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

    ASSERT_THROW(core.read_model(model, ov::Tensor()), ov::Exception);
}

TEST_F(IRFrontendTests, model_with_wrong_dimensions) {
    std::string testModel = R"V0G0N(
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

    ASSERT_THROW(model = core.read_model(testModel, ov::Tensor()), ov::Exception);
    ASSERT_TRUE(!model);
}

TEST_F(IRFrontendTests, name_is_not_unique) {
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

    ASSERT_THROW(core.read_model(xmlFileName, binFileName), ov::Exception);
}

TEST_F(IRFrontendTests, edge_has_wrong_port_id) {
    std::string testModel = R"V0G0N(
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

    ASSERT_THROW(model = core.read_model(testModel, ov::Tensor()), ov::Exception);
    ASSERT_FALSE(!!model);
}

TEST_F(IRFrontendTests, edge_has_wrong_layer_id) {
    std::string testModel = R"V0G0N(
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

    ASSERT_THROW(model = core.read_model(testModel, ov::Tensor()), ov::Exception);
    ASSERT_FALSE(!!model);
}

TEST_F(IRFrontendTests, not_opset1) {
    std::string testModel = R"V0G0N(
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

    ASSERT_NO_THROW(model = getWithIRFrontend(testModel));
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

TEST_F(IRFrontendTests, wrong_opset) {
    std::string testModel = R"V0G0N(
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

    ASSERT_THROW(model = core.read_model(testModel, ov::Tensor()), ov::Exception);
    ASSERT_FALSE(!!model);
}

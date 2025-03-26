// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_assertions.hpp"
#include "frontend_test.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset6.hpp"
#include "utils.hpp"

class IRFrontendTests : public ::testing::Test, public IRFrontendTestsImpl {
protected:
    void SetUp() override {}

    void TearDown() override {
        RemoveTemporalFiles();
    }
};

class IRFrontendMMapTests : public ::testing::TestWithParam<bool>, public IRFrontendTestsImpl {
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
    uint64_t version = 0;

    OV_ASSERT_NO_THROW(model = getWithIRFrontend(testModelV11));
    ASSERT_TRUE(!!model);
    OV_ASSERT_NO_THROW(rtInfo = model->get_rt_info());
    OV_ASSERT_NO_THROW(version = rtInfo["version"].as<int64_t>());
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

TEST_F(IRFrontendTests, elementary_model_reading_v11_undefined_precisoin) {
    std::string testModelV11 = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="undefined" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="UNSPECIFIED">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
            <input>
                <port id="0" precision="UNSPECIFIED">
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
    uint64_t version = 0;

    OV_ASSERT_NO_THROW(model = getWithIRFrontend(testModelV11));
    ASSERT_TRUE(!!model);
    OV_ASSERT_NO_THROW(rtInfo = model->get_rt_info());
    OV_ASSERT_NO_THROW(version = rtInfo["version"].as<int64_t>());
    ASSERT_EQ(11, version);

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::dynamic, ov::Shape{1, 3, 22, 22});
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

TEST_F(IRFrontendTests, elementary_model_reading_v11_dynamic_precisoin) {
    std::string testModelV11 = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="input" type="Parameter" id="0" version="opset1">
            <data element_type="dynamic" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="dynamic">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="1" version="opset1">
            <input>
                <port id="0" precision="dynamic">
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
    uint64_t version = 0;

    OV_ASSERT_NO_THROW(model = getWithIRFrontend(testModelV11));
    ASSERT_TRUE(!!model);
    OV_ASSERT_NO_THROW(rtInfo = model->get_rt_info());
    OV_ASSERT_NO_THROW(version = rtInfo["version"].as<int64_t>());
    ASSERT_EQ(11, version);

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::dynamic, ov::Shape{1, 3, 22, 22});
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
    uint64_t version = 0;

    OV_ASSERT_NO_THROW(modelv10 = getWithIRFrontend(testModelV10));
    ASSERT_TRUE(!!modelv10);
    OV_ASSERT_NO_THROW(rtInfoV10 = modelv10->get_rt_info());
    OV_ASSERT_NO_THROW(version = rtInfoV10["version"].as<int64_t>());
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

TEST_P(IRFrontendMMapTests, model_with_weights_reading_from_disk) {
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

    ov::Core new_core;
    new_core.set_property(ov::enable_mmap(GetParam()));
    OV_ASSERT_NO_THROW(model = new_core.read_model(xmlFileName, binFileName));
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

TEST_P(IRFrontendMMapTests, model_with_lp_weights_reading_from_disk) {
    std::string xmlModel = R"V0G0N(
<?xml version="1.0" ?>
<net name="Model" version="11">
	<layers>
		<layer id="0" name="A" type="Parameter" version="opset1">
			<data shape="" element_type="u4" />
			<output>
				<port id="0" precision="U4" names="" />
			</output>
		</layer>
		<layer id="1" name="my_const" type="Const" version="opset1">
			<data element_type="u4" shape="" offset="0" size="1" />
			<output>
				<port id="0" precision="U4" />
			</output>
		</layer>
		<layer id="2" name="Add_4" type="Add" version="opset1">
			<data auto_broadcast="numpy" />
			<input>
				<port id="0" precision="U4" />
				<port id="1" precision="U4" />
			</input>
			<output>
				<port id="2" precision="U4" />
			</output>
		</layer>
		<layer id="3" name="Result_5" type="Result" version="opset1">
			<input>
				<port id="0" precision="U4" />
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

    std::vector<unsigned char> buffer(1, 0);
    auto buffer_ptr = reinterpret_cast<uint8_t*>(buffer.data());
    buffer_ptr[0] = 0x18;
    auto t = ov::Tensor(ov::element::u4, ov::Shape{}, buffer.data());

    createTemporalModelFile(xmlModel, buffer);

    std::shared_ptr<ov::Model> model;

    ov::Core new_core;
    new_core.set_property(ov::enable_mmap(GetParam()));
    OV_ASSERT_NO_THROW(model = new_core.read_model(xmlFileName, binFileName));
    ASSERT_TRUE(!!model);

    std::shared_ptr<ov::Model> modelRef;
    {
        auto parameter = std::make_shared<ov::opset1::Parameter>(ov::element::u4, ov::Shape{});
        parameter->set_friendly_name("A");
        auto constant = std::make_shared<ov::opset1::Constant>(t);
        constant->set_friendly_name("my_const");
        auto transpose = std::make_shared<ov::opset1::Add>(parameter, constant);
        transpose->set_friendly_name("Add_4");
        auto result = std::make_shared<ov::opset1::Result>(transpose);
        result->set_friendly_name("Result_5");
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

    for (auto&& op : model->get_ops()) {
        if (auto c = ov::as_type<ov::op::v0::Constant>(op.get())) {
            const auto v = c->get_vector<uint8_t>();
            EXPECT_EQ(v[0], 0x08);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(EnableMMapPropery, IRFrontendMMapTests, ::testing::Bool());

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

    OV_ASSERT_NO_THROW(model = core.read_model(xmlFileName));
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

    EXPECT_NO_THROW(core.read_model(xmlFileName, binFileName));
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

    OV_ASSERT_NO_THROW(model = getWithIRFrontend(testModel));
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

TEST_F(IRFrontendTests, extension_proposal_network) {
    // the Proposal with 2 inputs was initially marked as "extension" operation but later was added to opset
    // the test checks that IR reader properly instantiate the "extension" Proposal as "opset6" Proposal
    std::string xmlModel = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer id="0" name="in1" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,12,34,62"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="in2" type="Parameter" version="opset1">
            <data element_type="f32" shape="1,24,34,62"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="in3" type="Const" version="opset1">
            <data element_type="f32" offset="0" shape="3" size="12"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="proposal" type="Proposal" precision="FP32" id="3" version="extension">
            <data feat_stride="16" base_size="16" min_size="16" ratio="2.669000" scale="4.000000,6.000000,9.000000,16.000000,24.000000,32.000000" pre_nms_topn="6000" post_nms_topn="200" nms_thresh="0.600000"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>12</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>24</dim>
                    <dim>34</dim>
                    <dim>62</dim>
                </port>
                <port id="3">
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>1000</dim>
                    <dim>5</dim>
                </port>
                <port id="4" precision="FP32">
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="output" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>200</dim>
                    <dim>5</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="3"/>
        <edge from-layer="3" from-port="4" to-layer="4" to-port="0"/>
    </edges>
    </net>
    )V0G0N";

    std::vector<unsigned char> buffer(12, 0);
    float* floatBuffer = reinterpret_cast<float*>(buffer.data());
    floatBuffer[0] = 0;
    floatBuffer[1] = 0;
    floatBuffer[2] = 0;

    createTemporalModelFile(xmlModel, buffer);
    std::shared_ptr<ov::Model> model;

    OV_ASSERT_NO_THROW(model = core.read_model(xmlFileName, binFileName));
    ASSERT_TRUE(!!model);

    for (auto op : model->get_ordered_ops()) {
        if (op->get_friendly_name() == "proposal" &&
            op->get_type_info() == ov::opset6::Proposal::get_type_info_static()) {
            return;
        }
    }
    FAIL() << "Custom proposal layer is not an opset6 operation.";
}

TEST_F(IRFrontendTests, model_with_tensor_names_with_spaces) {
    std::string testModel = R"V0G0N(
            <net name="graph" version="11">
            <layers>
                <layer id="1" name="input1" type="Parameter" version="opset1">
                    <data shape="1,4,512" element_type="f32"/>
                    <output>
                        <port id="0" precision="FP32" names="input1">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>512</dim>
                        </port>
                    </output>
                </layer>
                <layer id="0" name="input2" type="Parameter" version="opset1">
                    <data shape="1,4,512" element_type="f32"/>
                    <output>
                        <port id="0" precision="FP32" names="model/bert/encoder/layer_0/attention/self/query/Tensordot/MatMul;model/bert/encoder/layer_0/attention/self/query/BiasAdd;model/bert/encoder/layer_0/attention/output/dense/Tensordot/shape;model/bert/encoder/layer_0/attention/self/query/Tensordot;model/bert/encoder/layer_0/attention/self/query/BiasAdd/ReadVariableOp_Gemm__32:0">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>512</dim>
                        </port>
                    </output>
                </layer>
                <layer id="2" name="output 0([1 4 512])" type="Add" version="opset1">
                    <data auto_broadcast="numpy"/>
                    <input>
                        <port id="0" precision="FP32">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>512</dim>
                        </port>
                        <port id="1" precision="FP32">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>512</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2" precision="FP32" names="output 0([1 4 512])">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>512</dim>
                        </port>
                    </output>
                </layer>
                <layer id="3" name="output 0([1 4 512])/sink_port_0" type="Result" version="opset1">
                    <input>
                        <port id="0" precision="FP32">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>512</dim>
                        </port>
                    </input>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
                <edge from-layer="1" from-port="0" to-layer="2" to-port="0"/>
                <edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
            </edges>
        </net>
        )V0G0N";

    std::shared_ptr<ov::Model> model;

    OV_ASSERT_NO_THROW(model = core.read_model(testModel, ov::Tensor()));
    ASSERT_TRUE(!!model);

    auto outputs = model->outputs();
    EXPECT_EQ(outputs.size(), 1);
    auto names = outputs.at(0).get_names();
    EXPECT_EQ(names.size(), 1);
    auto it = names.find("output 0([1 4 512])");
    EXPECT_NE(it, names.end());
}

TEST_F(IRFrontendTests, model_with_tensor_names_add_output) {
    std::string testModel = R"V0G0N(
<net name="graph" version="11">
	<layers>
		<layer id="1" name="input1" type="Parameter" version="opset1">
			<data shape="1,4,512" element_type="f32"/>
			<output>
				<port id="0" precision="FP32" names="input1">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="0" name="input2" type="Parameter" version="opset1">
			<data shape="1,4,512" element_type="f32"/>
			<output>
				<port id="0" precision="FP32" names="input2">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="2" name="Add 221" type="Add" version="opset1">
			<data auto_broadcast="numpy"/>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</input>
			<output>
				<port id="2" precision="FP32" names="output add">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="output 0([1 4 512])" type="ReLU" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</input>
			<output>
				<port id="1" precision="FP32" names="output 0([1 4 512])">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="output 0([1 4 512])/sink_port_0" type="Result" version="opset1">
			<rt_info>
				<attribute name="fused_names" version="0" value="output 0([1 4 512])/sink_port_0"/>
			</rt_info>
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>4</dim>
					<dim>512</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
		<edge from-layer="1" from-port="0" to-layer="2" to-port="0"/>
		<edge from-layer="2" from-port="2" to-layer="3" to-port="0"/>
		<edge from-layer="3" from-port="1" to-layer="4" to-port="0"/>
	</edges>
</net>)V0G0N";

    std::shared_ptr<ov::Model> model;
    std::string tensor_name = "output add";

    OV_ASSERT_NO_THROW(model = core.read_model(testModel, ov::Tensor()));
    ASSERT_TRUE(!!model);

    model->add_output(tensor_name);
    auto outputs = model->outputs();
    EXPECT_EQ(outputs.size(), 2);
    auto names = outputs.at(1).get_names();
    EXPECT_EQ(names.size(), 1);
    auto it = names.find(tensor_name);
    EXPECT_NE(it, names.end());
}

TEST_F(IRFrontendTests, name_with_comma) {
    std::string testModel = R"V0G0N(
<net name="Network" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32" names="input">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32" names="relu\,t, identity_t">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
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
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;
    std::string tensor_name = "relu,t";

    OV_ASSERT_NO_THROW(model = core.read_model(testModel, ov::Tensor()));
    ASSERT_TRUE(!!model);

    model->add_output(tensor_name);
    auto outputs = model->outputs();
    EXPECT_EQ(outputs.size(), 1);
    auto names = outputs.at(0).get_names();
    auto it = names.find(tensor_name);
    EXPECT_NE(it, names.end());
}

TEST_F(IRFrontendTests, model_output_name_with_comma) {
    std::string testModel = R"V0G0N(
<net name="Network" version="11">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data element_type="f32" shape="1,3,22,22"/>
            <output>
                <port id="0" precision="FP32" names="input">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="activation" id="1" type="ReLU" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32" names="relu\,t, identity_t">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1" output_names="relu\,t,custom\, name">
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
    OV_ASSERT_NO_THROW(model = core.read_model(testModel, ov::Tensor()));
    ASSERT_TRUE(!!model);

    {
        const auto output_tensor = model->output("custom, name");
        EXPECT_EQ(output_tensor.get_names().size(), 2);
        EXPECT_EQ(output_tensor.get_node()->get_friendly_name(), "output");
    }
    {
        const auto output_tensor = model->output("relu,t");
        EXPECT_EQ(output_tensor.get_node()->get_friendly_name(), "output");
    }
}

TEST_F(IRFrontendTests, DetectionOutput) {
    std::string testModel = R"V0G0N(
<net name="DetectionOutput" version="11">
	<layers>
		<layer id="2" name="Parameter_186617" type="Parameter" version="opset1">
			<data shape="1,60" element_type="f32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>60</dim>
				</port>
			</output>
		</layer>
		<layer id="1" name="Parameter_186618" type="Parameter" version="opset1">
			<data shape="1,165" element_type="f32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>165</dim>
				</port>
			</output>
		</layer>
		<layer id="0" name="Parameter_186619" type="Parameter" version="opset1">
			<data shape="1,1,60" element_type="f32" />
			<output>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>60</dim>
				</port>
			</output>
		</layer>
		<layer id="3" name="DetectionOutput_186620" type="DetectionOutput" version="opset1">
			<data num_classes="11" background_label_id="0" top_k="75" variance_encoded_in_target="true" keep_top_k="50" code_type="caffe.PriorBoxParameter.CORNER" share_location="true" nms_threshold="0.5" confidence_threshold="0.30000001192092896" clip_after_nms="true" clip_before_nms="true" decrease_label_id="true" normalized="true" input_height="1" input_width="1" objectness_score="0.40000000596046448" />
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>60</dim>
				</port>
				<port id="1" precision="FP32">
					<dim>1</dim>
					<dim>165</dim>
				</port>
				<port id="2" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>60</dim>
				</port>
			</input>
			<output>
				<port id="3" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>50</dim>
					<dim>7</dim>
				</port>
			</output>
		</layer>
		<layer id="4" name="Result_186621" type="Result" version="opset1">
			<input>
				<port id="0" precision="FP32">
					<dim>1</dim>
					<dim>1</dim>
					<dim>50</dim>
					<dim>7</dim>
				</port>
			</input>
		</layer>
	</layers>
	<edges>
		<edge from-layer="0" from-port="0" to-layer="3" to-port="2" />
		<edge from-layer="1" from-port="0" to-layer="3" to-port="1" />
		<edge from-layer="2" from-port="0" to-layer="3" to-port="0" />
		<edge from-layer="3" from-port="3" to-layer="4" to-port="0" />
	</edges>
	<rt_info />
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;

    OV_ASSERT_NO_THROW(model = getWithIRFrontend(testModel));
    ASSERT_TRUE(!!model);
}

TEST_F(IRFrontendTests, load_model_not_exists_at_path) {
    const auto model_name = "not_existing_model";
    auto error_msg = std::string("Could not open the file: ");
    auto model_file_path = FrontEndTestUtils::make_model_path(model_name);
    error_msg += '"' + model_file_path + '"';

    auto fem = ov::frontend::FrontEndManager();
    auto fe = fem.load_by_framework("ir");

    OV_EXPECT_THROW(fe->supported({model_file_path}), ov::Exception, testing::HasSubstr(error_msg));
    OV_EXPECT_THROW(fe->load(model_file_path), ov::Exception, testing::HasSubstr(error_msg));
}

TEST_F(IRFrontendTests, load_model_weights_not_exist_at_path) {
    const auto error_msg = std::string(" cannot be opened");
    const auto name_prefix = ov::test::utils::generateTestFilePrefix();
    const auto model_file_path = name_prefix + "existing_model.xml";
    const auto weights_file_path = name_prefix + "not_existing_weights.bin";

    {
        std::ofstream model_file;
        model_file.open(model_file_path);
        model_file.close();
    }

    auto fem = ov::frontend::FrontEndManager();
    auto fe = fem.load_by_framework("ir");

    OV_EXPECT_THROW(fe->load(model_file_path, weights_file_path),
                    ov::Exception,
                    testing::HasSubstr(weights_file_path + error_msg));

    std::remove(model_file_path.c_str());
}

TEST_F(IRFrontendTests, LongComment) {
    std::string testModel = R"V0G0N(
<?xml version="1.0"?>
<!-- Long comment ............................................................................................................................................................................................................................................................................................................................................................................................................................................................ -->
<net name="IR with long comment" version="11">
    <layers>
        <layer id="0" name="Parameter_1" type="Parameter" version="opset1">
            <data shape="2" element_type="f16" />
            <output>
                <port id="0" precision="FP16">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="Convert_2" type="Convert" version="opset1">
            <data destination_type="f32" />
            <input>
                <port id="0" precision="FP16">
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="1" precision="FP32">
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="Result_3" type="Result" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>2</dim>
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

    std::shared_ptr<ov::Model> model;
    ov::RTMap rtInfo;
    int64_t version = 0;

    OV_ASSERT_NO_THROW(model = getWithIRFrontend(testModel));
    ASSERT_TRUE(!!model);
    OV_ASSERT_NO_THROW(version = model->get_rt_info().at("version").as<int64_t>());
    ASSERT_EQ(11, version);
}

TEST_F(IRFrontendTests, VeryShortValidModel) {
    std::string testModel = R"V0G0N(
<?xml version="1.0"?>
<net name="A" version="11">
<layers>
<layer id="0" name="P" type="Parameter" version="opset1">
<data shape="2" element_type="f16" />
<output>
<port id="0" precision="FP16">
<dim>2</dim>
</port>
</output>
</layer>
<layer id="1" name="R" type="Result" version="opset1">
<input>
<port id="0" precision="FP32">
<dim>2</dim>
</port>
</input>
</layer>
</layers>
<edges>
<edge from-layer="0" from-port="0" to-layer="1" to-port="0" />
</edges>
<rt_info />
</net>
)V0G0N";

    std::shared_ptr<ov::Model> model;
    int64_t version = 0;

    OV_ASSERT_NO_THROW(model = getWithIRFrontend(testModel));
    ASSERT_TRUE(!!model);
    OV_ASSERT_NO_THROW(version = model->get_rt_info().at("version").as<int64_t>());
    ASSERT_EQ(11, version);
}


#include "gtest/gtest.h"

#include "openvino/openvino.hpp"
#include "ie/ie_common.h"
#include "ie/ie_blob.h"
#include "ngraph/opsets/opset8.hpp"
#include "ie/ie_core.hpp"
#include "openvino/op/util/framework_node.hpp"

class IRTEST : public ::testing::Test {
public:
    ov::Core core;
};


TEST_F(IRTEST, ElementaryNetworkReadV11) {
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

    ASSERT_NO_THROW(model = core.read_model(testNetworkV11, ov::Tensor()));
    ASSERT_TRUE(!!model);
    ASSERT_NO_THROW(rtInfo = model->get_rt_info());
    ASSERT_NO_THROW(version = rtInfo["version"].as<int64_t>());
    ASSERT_EQ(11, version);
}

TEST_F(IRTEST, ElementaryNetworkReadV10) {
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

    ASSERT_NO_THROW(modelv10 = core.read_model(testNetworkV10, ov::Tensor()));
    ASSERT_TRUE(!!modelv10);
    ASSERT_NO_THROW(rtInfoV10 = modelv10->get_rt_info());
    ASSERT_NO_THROW(version = rtInfoV10["version"].as<int64_t>());
    ASSERT_EQ(10, version);
}

TEST_F(IRTEST, ElementaryNetworkReadV9) {
std::string testNetworkV9= R"V0G0N(
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


TEST_F(IRTEST, NetworkWithMissingWeights) {
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

TEST_F(IRTEST, NetworkWithWeightsFromDisk) {
    std::string modelName = "test_model_with_weights.xml";
    std::string binName = "test_model_with_weights.bin";

    std::shared_ptr<ov::Model> model;

    ASSERT_NO_THROW(model = core.read_model(modelName, binName));
    ASSERT_TRUE(!!model);
}

TEST_F(IRTEST, NetworkWithoutWeightsFromDisk) {
    std::string modelName = "test_model_without_weights.xml";
    std::shared_ptr<ov::Model> model;

    ASSERT_NO_THROW(model = core.read_model(modelName));
    ASSERT_TRUE(!!model);
}

TEST_F(IRTEST, NetworkWithWrongShape) {
    std::string modelName = "test_model_wrong_shape.xml";
    std::string binName = "test_model_wrong_shape.bin";

    ASSERT_THROW((void)core.read_model(modelName, binName), ov::Exception);
}

TEST_F(IRTEST, NetworkWithUnderallocatedWeightsFromDisk) {
    std::string modelName = "test_model_with_underallocated_weights.xml";
    std::string binName = "test_model_with_underallocated_weights.bin";

    ASSERT_THROW((void)core.read_model(modelName, binName), ov::Exception);
}

TEST_F(IRTEST, NetworkWithMissingWeightsFromDisk) {
    std::string modelName = "test_model_with_missing_weights.xml";

    ASSERT_THROW((void)core.read_model(modelName), ov::Exception);
}



TEST_F(IRTEST, LayerWithoutData) {
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





TEST_F(IRTEST, NetworkWirhWrongDimensions) {
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




TEST_F(IRTEST, NameIsNotUnique) {
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

    InferenceEngine::Core ie;
    InferenceEngine::Blob::Ptr weights;

    InferenceEngine::CNNNetwork model;

    weights = InferenceEngine::make_shared_blob<uint8_t>(InferenceEngine::TensorDesc(InferenceEngine::Precision::U8, {32}, InferenceEngine::Layout::C));
    weights->allocate();

    auto *dataI64 = weights->buffer().as<int64_t *>();
    dataI64[0] = 0;
    dataI64[1] = 3;
    dataI64[2] = 2;
    dataI64[3] = 1;

    ASSERT_THROW(model = ie.ReadNetwork(testNetwork, weights), ov::Exception);

}


class FrameworkNodeExtension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {
        static InferenceEngine::Version ExtensionDescription = {{1, 0}, "1.0", "TestExtension"};

        versionInfo = &ExtensionDescription;
    }

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        static std::map<std::string, ngraph::OpSet> opsets;
        if (opsets.empty()) {
            ngraph::OpSet opset;
            opset.insert<ov::op::util::FrameworkNode>();
            opsets["util"] = opset;
        }

        return opsets;
    }

    void Unload() noexcept override {}
};

TEST_F(IRTEST, CustomOpsTestWithFrameworkNodeExtension)
{
    
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
    auto extension = std::make_shared<FrameworkNodeExtension>();
    core.add_extension(extension);

    ASSERT_NO_THROW(model = core.read_model(customOpsNetwork, ov::Tensor()));
    ASSERT_TRUE(!!model);
}


TEST_F(IRTEST, CustomOpsTestWithoutExtension)
{
    
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

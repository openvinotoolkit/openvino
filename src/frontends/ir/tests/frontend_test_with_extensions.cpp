// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend_test.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset1.hpp"

class IRFrontendExtensionTests : public ::testing::Test, public IRFrontendTestsImpl {
protected:
    void SetUp() override {}

    void TearDown() override {
        RemoveTemporalFiles();
    }
};

TEST_F(IRFrontendExtensionTests, custom_ops_test_with_framework_node_extension) {
    static std::string customOpsModel = R"V0G0N(
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

    OV_ASSERT_NO_THROW(model = core.read_model(customOpsModel, ov::Tensor()));
    ASSERT_TRUE(!!model);
}

TEST_F(IRFrontendExtensionTests, custom_ops_test_without_extension) {
    static std::string customOpsModel = R"V0G0N(
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

    ASSERT_THROW(model = core.read_model(customOpsModel, ov::Tensor()), ov::Exception);
    ASSERT_TRUE(!model);
}

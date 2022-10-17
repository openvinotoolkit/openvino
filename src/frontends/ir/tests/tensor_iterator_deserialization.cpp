// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend_test.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset8.hpp"

class IRFrontendTestsTensorIterator : public ::testing::Test, public IRFrontendTestsImpl {
protected:
    void SetUp() override {}

    void TearDown() override {
        RemoveTemporalFiles();
    }
};

TEST_F(IRFrontendTestsTensorIterator, tensor_iterator_merged_input) {
    std::string testModel = R"V0G0N(
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

    ASSERT_NO_THROW(model = core.read_model(testModel, ov::Tensor()));
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

TEST_F(IRFrontendTestsTensorIterator, tensor_iterator_slised_input) {
    std::string testModel = R"V0G0N(
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

    ASSERT_NO_THROW(model = core.read_model(testModel, ov::Tensor()));
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

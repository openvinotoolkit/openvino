// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <cpp/ie_cnn_network.h>
#include <string>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <memory>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/op/interpolate.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/op/relu.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/graph_util.hpp>

#include <ie_core.hpp>

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/common_utils.hpp"

using namespace testing;
using namespace InferenceEngine;
using namespace CommonTestUtils;

using NGraphReshapeTests = TestsCommon;

TEST_F(NGraphReshapeTests, getBatchSize) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    CNNNetwork cnnNetwork(ngraph);
    ASSERT_EQ(1, cnnNetwork.getBatchSize());
}

TEST_F(NGraphReshapeTests, ReshapedDynamicShapeLayout) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({-1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("A");
        auto relu = std::make_shared<ngraph::op::Relu>(param);

        ngraph::ParameterVector params = {param};

        ngraph = std::make_shared<ngraph::Function>(relu, params);
    }

    CNNNetwork cnnNetwork(ngraph);
    ASSERT_EQ(Layout::SCALAR, cnnNetwork.getInputsInfo()["A"]->getLayout());

    ICNNNetwork::InputShapes new_shape;
    new_shape["A"] = ngraph::Shape{1, 3, 22, 22};
    cnnNetwork.reshape(new_shape);

    ASSERT_EQ(Layout::NCHW, cnnNetwork.getInputsInfo()["A"]->getLayout());
}

TEST_F(NGraphReshapeTests, ReshapeBatchReLU) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));

    {
        ngraph::PartialShape shape({2, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);

        ngraph->replace_parameter(0, param);
        ngraph->validate_nodes_and_infer_types();
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({2, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({2, 3, 22, 22}));
}

TEST_F(NGraphReshapeTests, ReshapeSpatialReLU) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));

    {
        ngraph::PartialShape shape({1, 3, 25, 25});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);

        ngraph->replace_parameter(0, param);
        ngraph->validate_nodes_and_infer_types();
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 25, 25}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 25, 25}));
}

TEST_F(NGraphReshapeTests, CNNReshapeSpatialReLU) {
    std::shared_ptr<const ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("data");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<const ngraph::Function>(results, params);
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));

    CNNNetwork cnnNetwork(ngraph::clone_function(*ngraph));
    std::map<std::string, std::vector<size_t>> shapes;
    shapes["data"] = {1, 3, 25, 25};

    ASSERT_NO_THROW(cnnNetwork.reshape(shapes));

    auto changedFunction = cnnNetwork.getFunction();
    ASSERT_NE(nullptr, changedFunction);
    ASSERT_EQ(changedFunction->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 25, 25}));
    ASSERT_EQ(changedFunction->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 25, 25}));
    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));
}

TEST_F(NGraphReshapeTests, CNNReshapeSpatialReLUWithoutCloneFunction) {
    std::shared_ptr<ngraph::Function> ngraph;
    {
        ngraph::PartialShape shape({1, 3, 22, 22});
        ngraph::element::Type type(ngraph::element::Type_t::f32);
        auto param = std::make_shared<ngraph::op::Parameter>(type, shape);
        param->set_friendly_name("data");
        auto relu = std::make_shared<ngraph::op::Relu>(param);
        auto result = std::make_shared<ngraph::op::Result>(relu);

        ngraph::ParameterVector params = {param};
        ngraph::ResultVector results = {result};

        ngraph = std::make_shared<ngraph::Function>(results, params);
    }

    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));

    CNNNetwork cnnNetwork(ngraph);
    std::map<std::string, std::vector<size_t>> shapes;
    shapes["data"] = {1, 3, 25, 25};

    ASSERT_NO_THROW(cnnNetwork.reshape(shapes));

    auto changedFunction = cnnNetwork.getFunction();
    ASSERT_NE(nullptr, changedFunction);
    ASSERT_EQ(changedFunction->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 25, 25}));
    ASSERT_EQ(changedFunction->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 25, 25}));
    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 25, 25}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 25, 25}));
}

class CustomTestOp: public ngraph::op::Op {
public:
    static constexpr ngraph::NodeTypeInfo type_info{"CustomTestLayer", 0};
    const ngraph::NodeTypeInfo& get_type_info() const override { return type_info;  }

    CustomTestOp() = default;
    CustomTestOp(const ngraph::Output<ngraph::Node>& arg, bool test1, int64_t test2):
        Op({arg}), test1(test1), test2(test2) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        auto input_shape = get_input_partial_shape(0).to_shape();

        ngraph::Shape output_shape(input_shape);
        for (int i = 0; i < input_shape.size(); ++i) {
            output_shape[i] = input_shape[i] * test2 + (test1 ? 0 : 1);
        }

        set_output_type(0, get_input_element_type(0), ngraph::PartialShape(output_shape));
    }

    std::shared_ptr<ngraph::Node> clone_with_new_inputs(const ngraph::OutputVector& new_args) const override {
        if (new_args.size() != 1) {
            throw ngraph::ngraph_error("Incorrect number of new arguments");
        }

        return std::make_shared<CustomTestOp>(new_args.at(0), test1, test2);
    }

    bool visit_attributes(ngraph::AttributeVisitor& visitor) override {
        visitor.on_attribute("test1", test1);
        visitor.on_attribute("test2", test2);
        return true;
    }

private:
    bool test1;
    int64_t test2;
};

constexpr ngraph::NodeTypeInfo CustomTestOp::type_info;

class TestInPlaceExtension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {}

    void Unload() noexcept override {}

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        static std::map<std::string, ngraph::OpSet> opsets;
        if (opsets.empty()) {
            ngraph::OpSet opset;
            opset.insert<CustomTestOp>();
            opsets["test_extension"] = opset;
        }
        return opsets;
    }

private:
};

TEST_F(NGraphReshapeTests, ReshapeNewIRWithNewExtension1) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
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
        <layer name="activation" id="1" type="CustomTestLayer" version="test_extension">
            <data test1="true" test2="2"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
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
    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<TestInPlaceExtension>());
    Blob::Ptr weights;
    SizeVector refBeforeReshape = {1, 3, 22, 22};
    SizeVector refAfterReshape = {4, 6, 44, 44};

    auto network = ie.ReadNetwork(model, weights);
    InferenceEngine::ICNNNetwork::InputShapes newShapes;
    newShapes["in1"] = {2, 3, 22, 22};

    ASSERT_NO_THROW(network.reshape(newShapes));
    auto output = network.getOutputsInfo();
    SizeVector outDims = output["activation"]->getTensorDesc().getDims();
    ASSERT_EQ(outDims, refAfterReshape);
}

TEST_F(NGraphReshapeTests, ReshapeNewIRWithNewExtension2) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
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
        <layer name="activation" id="1" type="CustomTestLayer" version="test_extension">
            <data test1="0" test2="3"/>
            <input>
                <port id="1" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>22</dim>
                    <dim>22</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
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
    InferenceEngine::Core ie;
    ie.AddExtension(std::make_shared<TestInPlaceExtension>());
    Blob::Ptr weights;
    SizeVector refBeforeReshape = {1, 3, 22, 22};
    SizeVector refAfterReshape = {7, 10, 67, 67};

    auto network = ie.ReadNetwork(model, weights);
    InferenceEngine::ICNNNetwork::InputShapes newShapes;
    newShapes["in1"] = {2, 3, 22, 22};

    ASSERT_NO_THROW(network.reshape(newShapes));
    auto output = network.getOutputsInfo();
    SizeVector outDims = output["activation"]->getTensorDesc().getDims();
    ASSERT_EQ(outDims, refAfterReshape);
}

class BadExtension : public InferenceEngine::IExtension {
public:
    BadExtension() {}

    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {};

    void Unload() noexcept override {};

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        static std::map<std::string, ngraph::OpSet> opsets;
        if (opsets.empty()) {
            ngraph::OpSet opset;
            opset.insert<CustomTestOp>();
            opsets["opset1"] = opset;
        }
        return opsets;
    }
};

TEST_F(NGraphReshapeTests, LoadBadNewExtension) {
    InferenceEngine::Core ie;
    ASSERT_THROW(ie.AddExtension(std::make_shared<BadExtension>()), InferenceEngine::Exception);
}

TEST_F(NGraphReshapeTests, TestInterpParameters) {
    auto inp = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4, 5});
    inp->set_friendly_name("test");

    ngraph::op::v0::InterpolateAttrs attrs;
    attrs.pads_begin.push_back(0);
    attrs.pads_end.push_back(0);
    attrs.axes = ngraph::AxisSet{2, 3};
    attrs.align_corners = false;
    attrs.mode = "nearest";
    attrs.antialias = false;

    std::vector<int64_t> shape = {8, 10};
    auto out_shape = std::make_shared<ngraph::op::v0::Constant>(ngraph::element::i64, ngraph::Shape{2}, shape);
    auto interp = std::make_shared<ngraph::op::v0::Interpolate>(inp, out_shape, attrs);

    auto output = std::make_shared<ngraph::op::Result>(interp);
    auto ngraph_function = std::make_shared<ngraph::Function>(ngraph::ResultVector{output},
                           ngraph::ParameterVector{inp});

    CNNNetwork cnn(ngraph_function);
    std::map<std::string, InferenceEngine::SizeVector> inShape;
    inShape["test"] = {1, 3, 4, 5};
    cnn.reshape(inShape);
}

TEST_F(NGraphReshapeTests, ReshapeWithDefaultGenericOps) {
    // the RNNCEll was initially marked as "experimental" operation but later was added to opset
    // the test checks that IR reader properly instantiate the "experimental" RNNCell as "opset6" RNNCell
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,16" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>16</dim>
                </port>
            </output>
        </layer>
        <layer name="in2" type="Parameter" id="1" version="opset1">
            <data shape="1,128" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer name="in3" type="Parameter" id="2" version="opset1">
            <data shape="128,16" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>128</dim>
                    <dim>16</dim>
                </port>
            </output>
        </layer>
        <layer name="in4" type="Parameter" id="3" version="opset1">
            <data shape="128,128" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>128</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer name="in5" type="Parameter" id="4" version="opset1">
            <data shape="128" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer id="5" name="77/RNNCell" type="RNNCell" version="experimental">
            <data hidden_size="128" linear_before_reset="1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>16</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
                <port id="2">
                    <dim>128</dim>
                    <dim>16</dim>
                </port>
                <port id="3">
                    <dim>128</dim>
                    <dim>128</dim>
                </port>
                <port id="4">
                    <dim>128</dim>
                </port>
            </input>
            <output>
                <port id="5" precision="FP32">
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </output>
        </layer>
        <layer name="output" type="Result" id="6" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>128</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="5" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="5" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="5" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="5" to-port="3"/>
        <edge from-layer="4" from-port="0" to-layer="5" to-port="4"/>
        <edge from-layer="5" from-port="5" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    InferenceEngine::Core ie;
    Blob::Ptr weights;

    auto network = ie.ReadNetwork(model, weights);
    InferenceEngine::ICNNNetwork::InputShapes newShapes;
    newShapes["in1"] = {2, 16};
    newShapes["in2"] = {2, 128};

    ASSERT_NO_THROW(network.reshape(newShapes));
}

TEST_F(NGraphReshapeTests, ReshapeEDDetectionOutput) {
    std::string model = R"V0G0N(
<net name="ExperimentalDetectronDetectionOutput" version="10">
    <layers>
        <layer name="in0" type="Parameter" id="0" version="opset1">
            <data shape="1000,4" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="in1" type="Parameter" id="1" version="opset1">
            <data shape="1000,324" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>324</dim>
                </port>
            </output>
        </layer>
       <layer name="in2" type="Parameter" id="2" version="opset1">
            <data shape="1000,81" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>81</dim>
                </port>
            </output>
        </layer>
        <layer name="in3" type="Parameter" id="3" version="opset1">
            <data shape="1,3" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="DO" type="ExperimentalDetectronDetectionOutput" version="experimental">
            <data class_agnostic_box_regression="0" deltas_weights="10.0,10.0,5.0,5.0" max_delta_log_wh="4.135166645050049" max_detections_per_image="100" nms_threshold="0.5" num_classes="81" post_nms_count="2000" score_threshold="0.05000000074505806"/>
            <input>
                <port id="0">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>1000</dim>
                    <dim>324</dim>
                </port>
                <port id="2">
                    <dim>1000</dim>
                    <dim>81</dim>
                </port>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="FP32">
                    <dim>100</dim>
                    <dim>4</dim>
                </port>
                <port id="5" precision="I32">
                    <dim>100</dim>
                </port>
                <port id="6" precision="FP32">
                    <dim>100</dim>
                </port>
            </output>
        </layer>
        <layer name="out_0" type="Result" id="5" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>100</dim>
                    <dim>4</dim>
                </port>
            </input>
        </layer>
        <layer name="out_1" type="Result" id="6" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>100</dim>
                </port>
            </input>
        </layer>
        <layer name="out_2" type="Result" id="7" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>100</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
        <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
        <edge from-layer="4" from-port="5" to-layer="6" to-port="0"/>
        <edge from-layer="4" from-port="6" to-layer="7" to-port="0"/>
    </edges>
</net>
)V0G0N";
    InferenceEngine::Core ie;
    Blob::Ptr weights;
    auto network = ie.ReadNetwork(model, weights);
    InferenceEngine::ICNNNetwork::InputShapes newShapes;
    newShapes["in0"] = {2000, 4};
    newShapes["in1"] = {2000, 324};
    newShapes["in2"] = {2000, 81};

    ASSERT_NO_THROW(network.reshape(newShapes));
}

TEST_F(NGraphReshapeTests, ReshapeEDPriorGridGenerator) {
    std::string model = R"V0G0N(
<net name="PriorGridGenerator" version="10">
    <layers>
        <layer name="in0" type="Parameter" id="0" version="opset1">
            <data shape="3,4" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="in1" type="Parameter" id="1" version="opset1">
            <data shape="1,256,200,336" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
            </output>
        </layer>
       <layer name="in2" type="Parameter" id="2" version="opset1">
            <data shape="1,3,800,1344" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>81</dim>
                </port>
            </output>
        </layer>
        <layer id="3" name="1117" type="ExperimentalDetectronPriorGridGenerator" version="experimental">
            <data flatten="1" h="0" stride_x="4.0" stride_y="4.0" w="0"/>
            <input>
                <port id="0">
                    <dim>3</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>800</dim>
                    <dim>1344</dim>
                </port>
            </input>
            <output>
                <port id="3" precision="FP32">
                    <dim>201600</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="out_0" type="Result" id="4" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>201600</dim>
                    <dim>4</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="3" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="3" to-port="2"/>
        <edge from-layer="3" from-port="3" to-layer="4" to-port="0"/>
    </edges>
</net>
)V0G0N";
    InferenceEngine::Core ie;
    Blob::Ptr weights;
    auto network = ie.ReadNetwork(model, weights);
    InferenceEngine::ICNNNetwork::InputShapes newShapes;
    newShapes["in1"] = {2, 256, 200, 336};
    newShapes["in2"] = {2, 3, 800, 1344};
    ASSERT_NO_THROW(network.reshape(newShapes));
}

TEST_F(NGraphReshapeTests, ReshapeEDGenerateProposalsSingleImage) {
    std::string model = R"V0G0N(
<net name="GenerateProposalsSingleImage" version="10">
    <layers>
        <layer name="in0" type="Parameter" id="0" version="opset1">
            <data shape="3" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="in1" type="Parameter" id="1" version="opset1">
            <data shape="201600,4" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>201600</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
       <layer name="in2" type="Parameter" id="2" version="opset1">
            <data shape="12,200,336" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>12</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
            </output>
        </layer>
        <layer name="in3" type="Parameter" id="3" version="opset1">
            <data shape="3,200,336" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="1133" type="ExperimentalDetectronGenerateProposalsSingleImage" version="experimental">
            <data min_size="0.0" nms_threshold="0.699999988079071" post_nms_count="1000" pre_nms_count="1000"/>
            <input>
                <port id="0">
                    <dim>3</dim>
                </port>
                <port id="1">
                    <dim>201600</dim>
                    <dim>4</dim>
                </port>
                <port id="2">
                    <dim>12</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
                <port id="3">
                    <dim>3</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
            </input>
            <output>
                <port id="4" precision="FP32">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
                <port id="5" precision="FP32">
                    <dim>1000</dim>
                </port>
            </output>
        </layer>
        <layer name="out_0" type="Result" id="5" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
            </input>
        </layer>
        <layer name="out_1" type="Result" id="6" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
        <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
        <edge from-layer="4" from-port="5" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    InferenceEngine::Core ie;
    Blob::Ptr weights;
    auto network = ie.ReadNetwork(model, weights);
    InferenceEngine::ICNNNetwork::InputShapes newShapes;
    newShapes["in2"] = {12, 200, 300};
    newShapes["in3"] = {2, 200, 300};
    ASSERT_NO_THROW(network.reshape(newShapes));
}

TEST_F(NGraphReshapeTests, ReshapeEDGenerateProposalsSingleImage_opset6) {
    std::string model = R"V0G0N(
<net name="GenerateProposalsSingleImage" version="10">
    <layers>
        <layer name="in0" type="Parameter" id="0" version="opset1">
            <data shape="3" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
                </port>
            </output>
        </layer>
        <layer name="in1" type="Parameter" id="1" version="opset1">
            <data shape="201600,4" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>201600</dim>
					<dim>4</dim>
                </port>
            </output>
        </layer>
       <layer name="in2" type="Parameter" id="2" version="opset1">
            <data shape="12,200,336" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>12</dim>
					<dim>200</dim>
					<dim>336</dim>
                </port>
            </output>
        </layer>
        <layer name="in3" type="Parameter" id="3" version="opset1">
            <data shape="3,200,336" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>3</dim>
					<dim>200</dim>
					<dim>336</dim>
                </port>
            </output>
        </layer>
        <layer id="4" name="1133" type="ExperimentalDetectronGenerateProposalsSingleImage" version="opset6">
			<data min_size="0.0" nms_threshold="0.699999988079071" post_nms_count="1000" pre_nms_count="1000"/>
			<input>
				<port id="0">
					<dim>3</dim>
				</port>
				<port id="1">
					<dim>201600</dim>
					<dim>4</dim>
				</port>
				<port id="2">
					<dim>12</dim>
					<dim>200</dim>
					<dim>336</dim>
				</port>
				<port id="3">
					<dim>3</dim>
					<dim>200</dim>
					<dim>336</dim>
				</port>
			</input>
			<output>
				<port id="4" precision="FP32">
					<dim>1000</dim>
					<dim>4</dim>
				</port>
				<port id="5" precision="FP32">
					<dim>1000</dim>
				</port>
			</output>
		</layer>
        <layer name="out_0" type="Result" id="5" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
					<dim>4</dim>
                </port>
            </input>
        </layer>
        <layer name="out_1" type="Result" id="6" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="4" to-port="0"/>
        <edge from-layer="1" from-port="0" to-layer="4" to-port="1"/>
        <edge from-layer="2" from-port="0" to-layer="4" to-port="2"/>
        <edge from-layer="3" from-port="0" to-layer="4" to-port="3"/>
        <edge from-layer="4" from-port="4" to-layer="5" to-port="0"/>
        <edge from-layer="4" from-port="5" to-layer="6" to-port="0"/>
    </edges>
</net>
)V0G0N";
    InferenceEngine::Core ie;
    Blob::Ptr weights;
    auto network = ie.ReadNetwork(model, weights);
    InferenceEngine::ICNNNetwork::InputShapes newShapes;
    newShapes["in2"] = {12, 200, 300};
    newShapes["in3"] = {2, 200, 300};
    ASSERT_NO_THROW(network.reshape(newShapes));
}

TEST_F(NGraphReshapeTests, ReshapeEDROIFeatureExtractor) {
    std::string model = R"V0G0N(
<net name="ExperimentalDetectronROIFeatureExtractor" version="10">
    <layers>
        <layer name="in0" type="Parameter" id="0" version="opset1">
            <data shape="1000,4" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="in1" type="Parameter" id="1" version="opset1">
            <data shape="1,256,200,336" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="1190" type="ExperimentalDetectronROIFeatureExtractor" version="experimental">
            <data aligned="0" output_size="7" pyramid_scales="4" sampling_ratio="2"/>
            <input>
                <port id="0">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer name="out_0" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
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
    Blob::Ptr weights;
    auto network = ie.ReadNetwork(model, weights);
    InferenceEngine::ICNNNetwork::InputShapes newShapes;
    newShapes["in0"] = {1256, 4};
    newShapes["in1"] = {1, 256, 7, 7};
    ASSERT_NO_THROW(network.reshape(newShapes));
}

TEST_F(NGraphReshapeTests, ReshapeEDROIFeatureExtractorOpset6) {
    std::string model = R"V0G0N(
<net name="ExperimentalDetectronROIFeatureExtractor" version="10">
    <layers>
        <layer name="in0" type="Parameter" id="0" version="opset1">
            <data shape="1000,4" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="in1" type="Parameter" id="1" version="opset1">
            <data shape="1,256,200,336" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="1190" type="ExperimentalDetectronROIFeatureExtractor" version="opset6">
            <data aligned="0" output_size="7" pyramid_scales="4" sampling_ratio="2"/>
            <input>
                <port id="0">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>256</dim>
                    <dim>200</dim>
                    <dim>336</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
                </port>
            </output>
        </layer>
        <layer name="out_0" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>256</dim>
                    <dim>7</dim>
                    <dim>7</dim>
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
    Blob::Ptr weights;
    auto network = ie.ReadNetwork(model, weights);
    InferenceEngine::ICNNNetwork::InputShapes newShapes;
    newShapes["in0"] = {1256, 4};
    newShapes["in1"] = {1, 256, 7, 7};
    ASSERT_NO_THROW(network.reshape(newShapes));
}

TEST_F(NGraphReshapeTests, ReshapeEDTopKROIs) {
    std::string model = R"V0G0N(
<net name="ExperimentalDetectronTopKROIs" version="10">
    <layers>
        <layer name="in0" type="Parameter" id="0" version="opset1">
            <data shape="5000,4" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>5000</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="in1" type="Parameter" id="1" version="opset1">
            <data shape="5000" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>5000</dim>
                </port>
            </output>
        </layer>
        <layer id="2" name="1189" type="ExperimentalDetectronTopKROIs" version="experimental">
            <data max_rois="1000"/>
            <input>
                <port id="0">
                    <dim>5000</dim>
                    <dim>4</dim>
                </port>
                <port id="1">
                    <dim>5000</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1000</dim>
                    <dim>4</dim>
                </port>
            </output>
        </layer>
        <layer name="out_0" type="Result" id="3" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1000</dim>
                    <dim>4</dim>
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
    Blob::Ptr weights;
    auto network = ie.ReadNetwork(model, weights);
    InferenceEngine::ICNNNetwork::InputShapes newShapes;
    newShapes["in0"] = {10000, 4};
    newShapes["in1"] = {10000};
    ASSERT_NO_THROW(network.reshape(newShapes));
}

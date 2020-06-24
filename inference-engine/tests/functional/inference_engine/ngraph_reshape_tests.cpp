// Copyright (C) 2018-2020 Intel Corporation
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
#include <ngraph/op/experimental/layers/interpolate.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/parameter.hpp>
#include <ngraph/op/op.hpp>
#include <ngraph/op/relu.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph/opsets/opset.hpp>

#include <ie_util_internal.hpp>
#include <ie_core.hpp>

#include "common_test_utils/test_common.hpp"
#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "generic_ie.hpp"

IE_SUPPRESS_DEPRECATED_START

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
    ASSERT_EQ(ngraph->get_parameters()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));
    ASSERT_EQ(ngraph->get_results()[0]->get_shape(), ngraph::Shape({1, 3, 22, 22}));
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

    std::shared_ptr<ngraph::Node> copy_with_new_args(const ngraph::NodeVector& new_args) const override {
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

    void Release() noexcept override {}

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
    // Convert to CNNNetwork
    auto layer = CommonTestUtils::getLayerByName(network, "activation");
    ASSERT_EQ("CustomTestLayer", layer->type);
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
    // Convert to CNNNetwork
    auto layer = CommonTestUtils::getLayerByName(network, "activation");
    ASSERT_EQ("CustomTestLayer", layer->type);
    ASSERT_EQ("false", layer->params["test1"]);
    ASSERT_EQ("3", layer->params["test2"]);
}

class BadExtension : public InferenceEngine::IExtension {
public:
    BadExtension() {}

    InferenceEngine::StatusCode
    getPrimitiveTypes(char**& types, unsigned int& size, InferenceEngine::ResponseDesc* resp) noexcept override {
        return GENERAL_ERROR;
    };

    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {};

    void Unload() noexcept override {};

    void Release() noexcept override {}

    InferenceEngine::StatusCode
    getFactoryFor(InferenceEngine::ILayerImplFactory*& factory, const InferenceEngine::CNNLayer* cnnLayer,
                  InferenceEngine::ResponseDesc* resp) noexcept override {
        return InferenceEngine::StatusCode::NOT_IMPLEMENTED;
    };

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
    ASSERT_THROW(ie.AddExtension(std::make_shared<BadExtension>()), InferenceEngine::details::InferenceEngineException);
}

TEST_F(NGraphReshapeTests, TestInterpParameters) {
    auto inp = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4, 5});
    inp->set_friendly_name("test");

    ngraph::op::InterpolateAttrs attrs;
    attrs.pads_begin.push_back(0);
    attrs.pads_end.push_back(0);
    attrs.axes = ngraph::AxisSet{2, 3};
    attrs.align_corners = false;
    attrs.mode = "nearest";
    attrs.antialias = false;

    std::vector<int64_t> shape = {8, 10};
    auto out_shape = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{2}, shape);
    auto interp = std::make_shared<ngraph::op::Interpolate>(inp, out_shape, attrs);

    auto output = std::make_shared<ngraph::op::Result>(interp);
    auto ngraph_function = std::make_shared<ngraph::Function>(ngraph::ResultVector{output},
                           ngraph::ParameterVector{inp});

    CNNNetwork cnn(ngraph_function);
    std::map<std::string, InferenceEngine::SizeVector> inShape;
    inShape["test"] = {1, 3, 4, 5};
    cnn.reshape(inShape);
}

TEST_F(NGraphReshapeTests, ReshapeWithDefaultGenericOps) {
    std::string model = R"V0G0N(
<net name="Activation" version="10">
    <layers>
        <layer name="in1" type="Parameter" id="0" version="opset1">
            <data shape="1,256" element_type="f32"/>
            <output>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
            </output>
        </layer>
        <layer id="1" name="77/GRUCell" type="GRUCell" version="experimental">
            <data hidden_size="256" linear_before_reset="1"/>
            <input>
                <port id="0">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
                <port id="1">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
            </input>
            <output>
                <port id="2" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
            </output>
            <blobs>
                <weights offset="0" precision="FP32" size="1572864"/>
                <biases offset="1572864" precision="FP32" size="4096"/>
            </blobs>
        </layer>
        <layer name="output" type="Result" id="2" version="opset1">
            <input>
                <port id="0" precision="FP32">
                    <dim>1</dim>
                    <dim>256</dim>
                </port>
            </input>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="0"/>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="0"/>
    </edges>
</net>
)V0G0N";
    InferenceEngine::Core ie;
    Blob::Ptr weights;
    weights = make_shared_blob<uint8_t>(TensorDesc(Precision::U8, {1576960}, Layout::C));
    weights->allocate();
    fill_data(weights->buffer(), weights->size() / sizeof(float));

    auto network = ie.ReadNetwork(model, weights);
    InferenceEngine::ICNNNetwork::InputShapes newShapes;
    newShapes["in1"] = {2, 256};

    ASSERT_NO_THROW(network.reshape(newShapes));
}

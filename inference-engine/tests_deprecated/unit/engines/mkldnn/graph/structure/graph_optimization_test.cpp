// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock-spec-builders.h>
#include "mkldnn_graph.h"

#include "single_layer_common.hpp"
#include <mkldnn_extension_utils.h>
#include <mkldnn_extension_mngr.h>
#include "tests_common.hpp"
#include <cpp/ie_cnn_net_reader.h>
#include "../test_graph.hpp"


using namespace ::testing;
using namespace std;
using namespace mkldnn;

class MKLDNNGraphOptimizationTests: public TestsCommon {};

TEST_F(MKLDNNGraphOptimizationTests, TestNoFuseConvSumWithOneInput) {
    std::string model = R"V0G0N(
<net name="AlexNet" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" type="Convolution" precision="FP32" id="1">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="3" group="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
            <weights offset="0" size="36"/>
            <biases offset="36" size="12"/>
        </layer>
        <layer name="res2a" type="Eltwise" precision="FP32" id="2">
            <elementwise_data operation="sum"/>
            <input>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="3"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="4"/>
    </edges>
</net>

)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {48}, InferenceEngine::C });
    weights->allocate();
    float * data = weights->buffer();

    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    ASSERT_NO_THROW(graph.CreateGraph(net_reader.getNetwork()));

    bool fused = true;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->getType() == MKLDNNPlugin::Convolution) {
            fused = false;
        }
    }
    ASSERT_FALSE(fused);
}

TEST_F(MKLDNNGraphOptimizationTests, DISABLED_TestNoCrashForFuseConvSumAndInput) {
    std::string model = R"V0G0N(
<net name="AlexNet" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" type="Convolution" precision="FP32" id="1">
            <convolution_data stride-x="1" stride-y="1" pad-x="0" pad-y="0" kernel-x="1" kernel-y="1" output="3" group="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
            <weights offset="0" size="36"/>
            <biases offset="36" size="12"/>
        </layer>
        <layer name="relu1" type="ReLU" precision="FP32" id="2">
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
        <layer name="res2a" type="Eltwise" precision="FP32" id="3">
            <elementwise_data operation="sum"/>
            <input>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
                <port id="4">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </input>
            <output>
                <port id="5">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>5</dim>
                    <dim>5</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="3" to-port="3"/>
        <edge from-layer="2" from-port="2" to-layer="3" to-port="4"/>
    </edges>
</net>

)V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {48}, InferenceEngine::C });
    weights->allocate();
    float * data = weights->buffer();

    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    ASSERT_NO_THROW(graph.CreateGraph(net_reader.getNetwork()));

    bool fused = false;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->isFusedWith(MKLDNNPlugin::Eltwise)) {
            fused = true;
        }
    }
    ASSERT_TRUE(fused);
}

namespace GraphOptimizationUtils {

using fake_ext_factory = std::function<InferenceEngine::ILayerImplFactory*(const InferenceEngine::CNNLayer *)>;

class FakeReLUImpl : public InferenceEngine::ILayerExecImpl {
public:
    FakeReLUImpl(const InferenceEngine::CNNLayer *layer) {
        cnnLayer = const_cast<InferenceEngine::CNNLayer *>(layer);
    }
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf, InferenceEngine::ResponseDesc *resp) noexcept override {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = 0;
        if (cnnLayer->outData.size() != 1 && cnnLayer->insData.size() != 1)
            return InferenceEngine::GENERAL_ERROR;
        InferenceEngine::DataConfig cfg;
        cfg.constant = false;
        cfg.inPlace = 0;
        InferenceEngine::SizeVector order;
        for(size_t i = 0; i < cnnLayer->outData[0]->getTensorDesc().getDims().size(); i++) {
            order.push_back(i);
        }
        cfg.desc = InferenceEngine::TensorDesc(cnnLayer->outData[0]->getTensorDesc().getPrecision(),
                                               cnnLayer->outData[0]->getTensorDesc().getDims(),
                                               {cnnLayer->outData[0]->getTensorDesc().getDims(), order});
        config.outConfs.push_back(cfg);
        config.inConfs.push_back(cfg);
        conf.push_back(config);
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config, InferenceEngine::ResponseDesc *resp) noexcept override {
        if (config.dynBatchSupport)
            return InferenceEngine::NOT_IMPLEMENTED;
        for(auto input : config.inConfs) {
            if (input.constant)
                return InferenceEngine::GENERAL_ERROR;
        }
        for(auto output : config.outConfs) {
            if (output.constant)
                return InferenceEngine::GENERAL_ERROR;
        }
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs, InferenceEngine::ResponseDesc *resp) noexcept override {
        const float *src_data = inputs[0]->buffer();
        float *dst_data = outputs[0]->buffer();
        if (src_data != dst_data)
            return InferenceEngine::GENERAL_ERROR;
        return InferenceEngine::OK;
    }

private:
    InferenceEngine::CNNLayer* cnnLayer;
};

class FakeReLUFactory : public InferenceEngine::ILayerImplFactory {
public:
    FakeReLUFactory(const InferenceEngine::CNNLayer *layer) {
        cnnLayer = const_cast<InferenceEngine::CNNLayer *>(layer);
    }
    // First implementation has more priority than next
    InferenceEngine::StatusCode getImplementations(std::vector<InferenceEngine::ILayerImpl::Ptr>& impls, InferenceEngine::ResponseDesc *resp) noexcept override {
        impls.push_back(InferenceEngine::ILayerImpl::Ptr(new FakeReLUImpl(cnnLayer)));
        return InferenceEngine::OK;
    }

private:
    InferenceEngine::CNNLayer * cnnLayer;
};

class FakeFabric : public InferenceEngine::IExtension {
public:
    FakeFabric() {
        factories["ReLU"] = [](const InferenceEngine::CNNLayer * cnnLayer) -> InferenceEngine::ILayerImplFactory* { return new FakeReLUFactory(cnnLayer); };
    }

    virtual ~FakeFabric() {
        factories.clear();
    }

    void GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept override {}
    void Unload() noexcept override {}
    void Release() noexcept override {
        delete this;
    }
    InferenceEngine::StatusCode getPrimitiveTypes(char**& types, unsigned int& size, InferenceEngine::ResponseDesc* resp) noexcept override {
        types = new char *[factories.size()];
        size_t count = 0;
        for (auto it = factories.begin(); it != factories.end(); it++, count ++) {
            types[count] = new char[it->first.size() + 1];
            std::copy(it->first.begin(), it->first.end(), types[count]);
            types[count][it->first.size() ] = '\0';
        }
        return InferenceEngine::OK;
    };
    InferenceEngine::StatusCode getFactoryFor(InferenceEngine::ILayerImplFactory *&factory,
                                              const InferenceEngine::CNNLayer *cnnLayer,
                                              InferenceEngine::ResponseDesc *resp) noexcept override {
        if (factories.find(cnnLayer->type) == factories.end()) {
            std::string errorMsg = std::string("Factory for ") + cnnLayer->type + " wasn't found!";
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
            return InferenceEngine::NOT_FOUND;
        }
        factory = factories[cnnLayer->type](cnnLayer);
        return InferenceEngine::OK;
    }

    InferenceEngine::StatusCode getShapeInferImpl(InferenceEngine::IShapeInferImpl::Ptr& impl, const char* type,
                                                  InferenceEngine::ResponseDesc* resp) noexcept override {
        return InferenceEngine::NOT_IMPLEMENTED;
    }

private:
    std::map<std::string, fake_ext_factory> factories;
};
}

TEST_F(MKLDNNGraphOptimizationTests, TestNoFuseCustomActivation) {
    std::string model = R"V0G0N(
<net name="AlexNet" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </output>
        </layer>
        <layer name="conv1" type="Convolution" precision="FP32" id="1">
            <convolution_data stride-x="4" stride-y="4" pad-x="0" pad-y="0" kernel-x="11" kernel-y="11" output="96" group="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>227</dim>
                    <dim>227</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
            <weights offset="0" size="139392"/>
            <biases offset="139392" size="384"/>
        </layer>
        <layer name="relu1" type="ReLU" precision="FP32" id="2">
            <input>
                <port id="3">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>96</dim>
                    <dim>55</dim>
                    <dim>55</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="1" from-port="2" to-layer="2" to-port="3"/>
    </edges>
</net>
)V0G0N";

    std::shared_ptr<InferenceEngine::IExtension> extension;
    extension.reset(new GraphOptimizationUtils::FakeFabric());
    MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
    extMgr->AddExtension(extension);

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    InferenceEngine::TBlob<uint8_t> *weights = new InferenceEngine::TBlob<uint8_t>({ InferenceEngine::Precision::U8, {139776}, InferenceEngine::C });
    weights->allocate();
    float * data = weights->buffer();

    fill_data((float *) weights->buffer(), weights->size() / sizeof(float));

    InferenceEngine::TBlob<uint8_t>::Ptr weights_ptr = InferenceEngine::TBlob<uint8_t>::Ptr(weights);

    net_reader.SetWeights(weights_ptr);

    MKLDNNGraphTestClass graph;
    ASSERT_NO_THROW(graph.CreateGraph(net_reader.getNetwork(), extMgr));

    bool fused = true;
    auto& nodes = graph.getNodes();
    for (auto &node : nodes) {
        if (node->getType() == MKLDNNPlugin::Convolution) {
            fused = false;
        }
    }
    ASSERT_FALSE(fused);
}

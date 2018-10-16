// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <ie_iextension.h>
#include <ie_common.h>
#include <ie_layers.h>
#include <tests_common.hpp>
#include <mkldnn_plugin/mkldnn_extension_mngr.h>
#include "graph/test_graph.hpp"

using namespace ::testing;

class ConstLayerImpl : public InferenceEngine::ILayerExecImpl {
public:
    explicit ConstLayerImpl(const InferenceEngine::CNNLayer *layer): cnnLayer(*layer) {}
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf, InferenceEngine::ResponseDesc *resp) noexcept override {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = 0;
        if (cnnLayer.outData.size() != 1 && cnnLayer.insData.size() != 1)
            return InferenceEngine::GENERAL_ERROR;
        InferenceEngine::DataConfig cfg;
        cfg.constant = true;
        cfg.inPlace = 0;
        InferenceEngine::SizeVector order;
        for(size_t i = 0; i < cnnLayer.outData[0]->getTensorDesc().getDims().size(); i++) {
            order.push_back(i);
        }
        cfg.desc = InferenceEngine::TensorDesc(cnnLayer.outData[0]->getTensorDesc().getPrecision(),
                                               cnnLayer.outData[0]->getTensorDesc().getDims(),
                                               {cnnLayer.outData[0]->getTensorDesc().getDims(), order});
        config.outConfs.push_back(cfg);
        config.inConfs.push_back(cfg);
        conf.push_back(config);
        return InferenceEngine::OK;
    }

    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config, InferenceEngine::ResponseDesc *resp) noexcept override {
        value = cnnLayer.GetParamAsInt("const_val", 1);
        if (config.dynBatchSupport)
            return InferenceEngine::NOT_IMPLEMENTED;
        for(auto input : config.inConfs) {
            if (!input.constant)
                return InferenceEngine::GENERAL_ERROR;
        }
        for(auto output : config.outConfs) {
            if (!output.constant)
                return InferenceEngine::GENERAL_ERROR;
        }
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs, InferenceEngine::ResponseDesc *resp) noexcept override {
        float *dst_data = outputs[0]->buffer();

        size_t data_size = outputs[0]->size();
        for (size_t i = 0; i < data_size; i++) {
            dst_data[i] = value;
        }
        return InferenceEngine::OK;
    }

private:
    InferenceEngine::CNNLayer cnnLayer;
    int value = 0;
};

class ConstLayerFactory : public InferenceEngine::ILayerImplFactory {
public:
    ConstLayerFactory(const InferenceEngine::CNNLayer *layer): cnnLayer(*layer) {}
    // set output shapes by input shapes.
    InferenceEngine::StatusCode getShapes(const std::vector<InferenceEngine::TensorDesc>& inShapes, std::vector<InferenceEngine::TensorDesc>& outShapes, InferenceEngine::ResponseDesc *resp) noexcept override {
        outShapes.push_back(inShapes[0]);
        return InferenceEngine::OK;
    }
    // First implementation has more priority than next
    InferenceEngine::StatusCode getImplementations(std::vector<InferenceEngine::ILayerImpl::Ptr>& impls, InferenceEngine::ResponseDesc *resp) noexcept override {
        impls.push_back(InferenceEngine::ILayerImpl::Ptr(new ConstLayerImpl(&cnnLayer)));
        return InferenceEngine::OK;
    }

private:
    InferenceEngine::CNNLayer cnnLayer;
};

using fake_ext_factory = std::function<InferenceEngine::ILayerImplFactory*(const InferenceEngine::CNNLayer *)>;

class FakeConstExtensionFabric : public InferenceEngine::IExtension {
public:
    FakeConstExtensionFabric() {
        factories["ConstLayer"] = [](const InferenceEngine::CNNLayer * cnnLayer) -> InferenceEngine::ILayerImplFactory* { return new ConstLayerFactory(cnnLayer); };
    }

    virtual ~FakeConstExtensionFabric() {
        factories.clear();
    }

    void GetVersion(const InferenceEngine::Version *&versionInfo) const noexcept override {}
    void SetLogCallback(InferenceEngine::IErrorListener &listener) noexcept override {}
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

class MKLDNNConstantPropagationTests: public TestsCommon {
protected:
    virtual void SetUp() {
        TestsCommon::SetUp();
        extension.reset(new FakeConstExtensionFabric());
        extMgr.reset(new MKLDNNPlugin::MKLDNNExtensionManager());
        extMgr->AddExtension(extension);
    }
    MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr;
    std::shared_ptr<InferenceEngine::IExtension> extension;
};

TEST_F(MKLDNNConstantPropagationTests, ConcatAfterConstLayers) {
    std::string model = R"V0G0N(
        <Net Name="CustomConcat_Only" version="2" precision="FP32" batch="1">
            <layers>
                <layer name="in1" type="Input" precision="FP32" id="0">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>10</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="in2" type="Input" precision="FP32" id="1">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="const1" type="ConstLayer" precision="FP32" id="2">
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>10</dim>
                            <dim>5</dim>
                        </port>
                    </input>
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>10</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="const2" type="ConstLayer" precision="FP32" id="3">
                    <data const_val="4"/>
                    <input>
                        <port id="0">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </input>
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="con" id="4" type="Concat" precision="FP32">
                    <concat_data axis="2"/>
                    <input>
                        <port id="1">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>10</dim>
                            <dim>5</dim>
                        </port>
                        <port id="2">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </input>
                    <output>
                        <port id="3">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>15</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="2" to-port="0"/>
                <edge from-layer="1" from-port="0" to-layer="3" to-port="0"/>
                <edge from-layer="2" from-port="1" to-layer="4" to-port="1"/>
                <edge from-layer="3" from-port="1" to-layer="4" to-port="2"/>
            </edges>
        </Net>
        )V0G0N";

    InferenceEngine::CNNNetReader net_reader;
    ASSERT_NO_THROW(net_reader.ReadNetwork(model.data(), model.length()));

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(net_reader.getNetwork(), extMgr);

    InferenceEngine::SizeVector dims_src1 = {1, 2, 10, 5};

    InferenceEngine::Blob::Ptr src1 =
            InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src1);
    src1->allocate();

    InferenceEngine::SizeVector dims_src2 = {1, 2, 5, 5};

    InferenceEngine::Blob::Ptr src2 =
            InferenceEngine::make_shared_blob<float, const InferenceEngine::SizeVector>(InferenceEngine::Precision::FP32, InferenceEngine::NCHW, dims_src2);
    src2->allocate();

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));

    InferenceEngine::OutputsDataMap out;
    out = net_reader.getNetwork().getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs;

    auto it = out.begin();

    std::pair<std::string, InferenceEngine::DataPtr> item = *it;

    InferenceEngine::TensorDesc outputDesc1 = item.second->getTensorDesc();
    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(outputDesc1);
    output->allocate();
    outputBlobs[item.first] = output;

    auto& nodes = graph.getNodes();
    bool existConcat = false;
    for (auto& node : nodes) {
        if (node->getType() != MKLDNNPlugin::Concatenation && node->getType() != MKLDNNPlugin::Generic)
            continue;
        if (node->getName() == "con" && node->getType() == MKLDNNPlugin::Concatenation)
            existConcat = true;
        ASSERT_TRUE(node->isConstant());
    }

    ASSERT_TRUE(existConcat);

    graph.Infer(srcs, outputBlobs);

    // Compare
    float *dst_ptr = output->buffer();

    int len1 = 1, len2 = 1, cycles;
    for (int dim = 2; dim < output->dims().size(); dim++) {
        len1 *= src1->dims()[dim];
        len2 *= src2->dims()[dim];
    }
    cycles = 2;

    int index1 = 0, index2 = 0, index = 0;
    for (int cycle = 0; cycle < cycles; cycle ++) {
        for (int i1 = 0; i1 < len1; i1++) {
            if (1 != dst_ptr[index]) {
                FAIL() << "index: " << index << " src: " << 1 << ", dst: " << dst_ptr[index];
            }
            index1++; index++;
        }
        for (int i2 = 0; i2 < len2; i2++) {
            if (4 != dst_ptr[index]) {
                FAIL() << "index: " << index << " src: " << 4 << ", dst: " << dst_ptr[index];
            }
            index2++; index++;
        }
    }
}

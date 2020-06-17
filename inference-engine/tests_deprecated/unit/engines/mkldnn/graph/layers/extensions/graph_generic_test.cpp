// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_graph.hpp"

#include <ie_iextension.h>
#include <ie_core.hpp>
#include <ie_plugin_config.hpp>
#include "tests_common.hpp"

using namespace ::testing;
using namespace std;
using namespace mkldnn;

class FakeGenericPrimitiveImpl : public InferenceEngine::ILayerExecImpl {
public:
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf, InferenceEngine::ResponseDesc *resp) noexcept override {
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config, InferenceEngine::ResponseDesc *resp) noexcept override {
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs, InferenceEngine::ResponseDesc *resp) noexcept override {
        return InferenceEngine::NOT_IMPLEMENTED;
    }
};

class FakeGenericPrimitiveFactory : public InferenceEngine::ILayerImplFactory {
public:
    // First implementation has more priority than next
    InferenceEngine::StatusCode getImplementations(std::vector<InferenceEngine::ILayerImpl::Ptr>& impls, InferenceEngine::ResponseDesc *resp) noexcept override {
        impls.push_back(InferenceEngine::ILayerImpl::Ptr(new FakeGenericPrimitiveImpl()));
        return InferenceEngine::OK;
    }
};

class DoublePrimitiveImpl : public InferenceEngine::ILayerExecImpl {
public:
    DoublePrimitiveImpl(const InferenceEngine::CNNLayer *layer) {
        cnnLayer = const_cast<InferenceEngine::CNNLayer *>(layer);
    }
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf, InferenceEngine::ResponseDesc *resp) noexcept override {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = true;
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
        for(auto input : config.inConfs) {
            if (input.inPlace < 0)
                return InferenceEngine::GENERAL_ERROR;
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

        size_t data_size = inputs[0]->size();
        for (size_t i = 0; i < data_size; i++) {
            dst_data[i] = src_data[i]*2;
        }
        return InferenceEngine::OK;
    }

private:
    InferenceEngine::CNNLayer* cnnLayer;
};

class ConstPrimitiveImpl : public InferenceEngine::ILayerExecImpl {
public:
    ConstPrimitiveImpl(const InferenceEngine::CNNLayer *layer) {
        cnnLayer = const_cast<InferenceEngine::CNNLayer *>(layer);
    }
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf, InferenceEngine::ResponseDesc *resp) noexcept override {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = 0;
        if (cnnLayer->outData.size() != 1 && cnnLayer->insData.size() != 1)
            return InferenceEngine::GENERAL_ERROR;
        InferenceEngine::DataConfig cfg;
        cfg.constant = true;
        // Cannot be in-place because memory will change a memory.
        cfg.inPlace = -1;
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
            if (input.inPlace >= 0)
                return InferenceEngine::GENERAL_ERROR;
            if (!input.constant)
                return InferenceEngine::GENERAL_ERROR;
        }
        for(auto output : config.outConfs) {
            if (output.inPlace >= 0)
                return InferenceEngine::GENERAL_ERROR;
            if (!output.constant)
                return InferenceEngine::GENERAL_ERROR;
        }
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs, InferenceEngine::ResponseDesc *resp) noexcept override {
        float *dst_data = outputs[0]->buffer();

        size_t data_size = outputs[0]->size();
        for (size_t i = 0; i < data_size; i++) {
            dst_data[i] = 2;
        }
        return InferenceEngine::OK;
    }

private:
    InferenceEngine::CNNLayer *cnnLayer;
};

class ConstPrimitiveFactory : public InferenceEngine::ILayerImplFactory {
public:
    ConstPrimitiveFactory(const InferenceEngine::CNNLayer *layer) {
        cnnLayer = const_cast<InferenceEngine::CNNLayer *>(layer);
    }
    // First implementation has more priority than next
    InferenceEngine::StatusCode getImplementations(std::vector<InferenceEngine::ILayerImpl::Ptr>& impls, InferenceEngine::ResponseDesc *resp) noexcept override {
        impls.push_back(InferenceEngine::ILayerImpl::Ptr(new ConstPrimitiveImpl(cnnLayer)));
        return InferenceEngine::OK;
    }

private:
    InferenceEngine::CNNLayer * cnnLayer;
};

class DoublePrimitiveFactory : public InferenceEngine::ILayerImplFactory {
public:
    DoublePrimitiveFactory(const InferenceEngine::CNNLayer *layer) {
        cnnLayer = const_cast<InferenceEngine::CNNLayer *>(layer);
    }
    // First implementation has more priority than next
    InferenceEngine::StatusCode getImplementations(std::vector<InferenceEngine::ILayerImpl::Ptr>& impls, InferenceEngine::ResponseDesc *resp) noexcept override {
        impls.push_back(InferenceEngine::ILayerImpl::Ptr(new DoublePrimitiveImpl(cnnLayer)));
        return InferenceEngine::OK;
    }

private:
    InferenceEngine::CNNLayer * cnnLayer;
};

class TwoDifferentOutputsImpl : public InferenceEngine::ILayerExecImpl {
public:
    TwoDifferentOutputsImpl(const InferenceEngine::CNNLayer *layer) {
        cnnLayer = const_cast<InferenceEngine::CNNLayer *>(layer);
    }
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf, InferenceEngine::ResponseDesc *resp) noexcept override {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = 0;
        if (cnnLayer->outData.size() != 2 && cnnLayer->insData.size() != 1)
            return InferenceEngine::GENERAL_ERROR;
        InferenceEngine::DataConfig cfg;
        cfg.constant = false;
        cfg.inPlace = -1;
        InferenceEngine::SizeVector order;
        for(size_t i = 0; i < cnnLayer->outData[0]->getTensorDesc().getDims().size(); i++) {
            order.push_back(i);
        }
        cfg.desc = InferenceEngine::TensorDesc(cnnLayer->outData[0]->getTensorDesc().getPrecision(),
                                               cnnLayer->outData[0]->getTensorDesc().getDims(),
                                               {cnnLayer->outData[0]->getTensorDesc().getDims(), order});
        config.outConfs.push_back(cfg);
        cfg.desc = InferenceEngine::TensorDesc(cnnLayer->outData[1]->getTensorDesc().getPrecision(),
                                               cnnLayer->outData[1]->getTensorDesc().getDims(),
                                               {cnnLayer->outData[1]->getTensorDesc().getDims(), order});
        config.outConfs.push_back(cfg);
        cfg.desc = InferenceEngine::TensorDesc(cnnLayer->insData[0].lock()->getTensorDesc().getPrecision(),
                              cnnLayer->insData[0].lock()->getTensorDesc().getDims(),
                              {cnnLayer->insData[0].lock()->getTensorDesc().getDims(), order});
        config.inConfs.push_back(cfg);
        conf.push_back(config);
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config, InferenceEngine::ResponseDesc *resp) noexcept override {
        if (config.dynBatchSupport)
            return InferenceEngine::NOT_IMPLEMENTED;
        for(auto input : config.inConfs) {
            if (input.inPlace >= 0)
                return InferenceEngine::GENERAL_ERROR;
            if (input.constant)
                return InferenceEngine::GENERAL_ERROR;
        }
        for(auto output : config.outConfs) {
            if (output.inPlace >= 0)
                return InferenceEngine::GENERAL_ERROR;
            if (output.constant)
                return InferenceEngine::GENERAL_ERROR;
        }
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs, std::vector<InferenceEngine::Blob::Ptr>& outputs, InferenceEngine::ResponseDesc *resp) noexcept override {
        const float *src_data = inputs[0]->buffer();
        float *dst_data0 = outputs[0]->buffer();
        float *dst_data1 = outputs[1]->buffer();

        size_t out_data_size0 = outputs[0]->size();
        size_t out_data_size1 = outputs[1]->size();
        for (size_t i = 0; i < out_data_size0; i++) {
            dst_data0[i] = (*(src_data++))*2;
        }

        for (size_t i = 0; i < out_data_size1; i++) {
            dst_data1[i] = (*(src_data++))*3;
        }
        return InferenceEngine::OK;
    }

private:
    InferenceEngine::CNNLayer* cnnLayer;
};

class TwoDifferentOutputsFactory : public InferenceEngine::ILayerImplFactory {
public:
    TwoDifferentOutputsFactory(const InferenceEngine::CNNLayer *layer) {
        cnnLayer = const_cast<InferenceEngine::CNNLayer *>(layer);
    }
    // First implementation has more priority than next
    InferenceEngine::StatusCode getImplementations(std::vector<InferenceEngine::ILayerImpl::Ptr>& impls, InferenceEngine::ResponseDesc *resp) noexcept override {
        impls.push_back(InferenceEngine::ILayerImpl::Ptr(new TwoDifferentOutputsImpl(cnnLayer)));
        return InferenceEngine::OK;
    }

private:
    InferenceEngine::CNNLayer * cnnLayer;
};

class CustomConcatImpl : public InferenceEngine::ILayerExecImpl {
public:
    CustomConcatImpl(const InferenceEngine::CNNLayer *layer) {
        cnnLayer = const_cast<InferenceEngine::CNNLayer *>(layer);
    }
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf, InferenceEngine::ResponseDesc *resp) noexcept override {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = 0;
        if (cnnLayer->outData.size() != 1 && cnnLayer->insData.size() != 2)
            return InferenceEngine::GENERAL_ERROR;
        InferenceEngine::DataConfig cfg;
        cfg.constant = false;
        cfg.inPlace = -1;
        InferenceEngine::SizeVector order;
        for(size_t i = 0; i < cnnLayer->outData[0]->getTensorDesc().getDims().size(); i++) {
            order.push_back(i);
        }
        cfg.desc = InferenceEngine::TensorDesc(cnnLayer->outData[0]->getTensorDesc().getPrecision(),
                                               cnnLayer->outData[0]->getTensorDesc().getDims(),
                                               {cnnLayer->outData[0]->getTensorDesc().getDims(), order});
        config.outConfs.push_back(cfg);
        cfg.inPlace = 0;
        InferenceEngine::SizeVector dims = cnnLayer->insData[0].lock()->getTensorDesc().getDims();
        cfg.desc = InferenceEngine::TensorDesc(cnnLayer->insData[0].lock()->getTensorDesc().getPrecision(),
                                               dims, {dims, order});
        size_t dataSize = std::accumulate(std::begin(dims), std::end(dims), (size_t) 1, std::multiplies<size_t>());
        config.inConfs.push_back(cfg);
        cfg.desc = InferenceEngine::TensorDesc(cnnLayer->insData[1].lock()->getTensorDesc().getPrecision(),
                                               cnnLayer->insData[1].lock()->getTensorDesc().getDims(),
                                               {cnnLayer->insData[1].lock()->getTensorDesc().getDims(), order,
                                                dataSize});
        config.inConfs.push_back(cfg);
        conf.push_back(config);
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config, InferenceEngine::ResponseDesc *resp) noexcept override {
        if (config.dynBatchSupport)
            return InferenceEngine::NOT_IMPLEMENTED;
        for(auto input : config.inConfs) {
            if (input.inPlace < 0)
                return InferenceEngine::GENERAL_ERROR;
            if (input.constant)
                return InferenceEngine::GENERAL_ERROR;
        }
        for(auto output : config.outConfs) {
            if (output.inPlace >= 0)
                return InferenceEngine::GENERAL_ERROR;
            if (output.constant)
                return InferenceEngine::GENERAL_ERROR;
        }
        return InferenceEngine::OK;
    }

    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs,
                                        std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                        InferenceEngine::ResponseDesc *resp) noexcept override {
        return InferenceEngine::OK;
    }
private:
    InferenceEngine::CNNLayer * cnnLayer;
};

class CustomConcatFactory : public InferenceEngine::ILayerImplFactory {
public:
    CustomConcatFactory(const InferenceEngine::CNNLayer *layer) {
        cnnLayer = const_cast<InferenceEngine::CNNLayer *>(layer);
    }
    // First implementation has more priority than next
    InferenceEngine::StatusCode getImplementations(std::vector<InferenceEngine::ILayerImpl::Ptr>& impls, InferenceEngine::ResponseDesc *resp) noexcept override {
        impls.push_back(InferenceEngine::ILayerImpl::Ptr(new CustomConcatImpl(cnnLayer)));
        return InferenceEngine::OK;
    }

private:
    InferenceEngine::CNNLayer * cnnLayer;
};

class CustomSplitImpl : public InferenceEngine::ILayerExecImpl {
public:
    CustomSplitImpl(const InferenceEngine::CNNLayer *layer) {
        cnnLayer = const_cast<InferenceEngine::CNNLayer *>(layer);
    }
    InferenceEngine::StatusCode getSupportedConfigurations(std::vector<InferenceEngine::LayerConfig>& conf, InferenceEngine::ResponseDesc *resp) noexcept override {
        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = 0;
        if (cnnLayer->outData.size() != 2 && cnnLayer->insData.size() != 1)
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
        size_t dataSize = std::accumulate(std::begin(cnnLayer->outData[0]->getTensorDesc().getDims()),
                                          std::end(cnnLayer->outData[0]->getTensorDesc().getDims()),
                                          (size_t) 1, std::multiplies<size_t>());
        cfg.desc = InferenceEngine::TensorDesc(cnnLayer->outData[1]->getTensorDesc().getPrecision(),
                                               cnnLayer->outData[1]->getTensorDesc().getDims(),
                                               {cnnLayer->outData[1]->getTensorDesc().getDims(), order, dataSize});
        config.outConfs.push_back(cfg);
        cfg.inPlace = -1;
        cfg.desc = InferenceEngine::TensorDesc(cnnLayer->insData[0].lock()->getTensorDesc().getPrecision(),
                                               cnnLayer->insData[0].lock()->getTensorDesc().getDims(),
                                               {cnnLayer->insData[0].lock()->getTensorDesc().getDims(), order});
        config.inConfs.push_back(cfg);
        conf.push_back(config);
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode init(InferenceEngine::LayerConfig& config, InferenceEngine::ResponseDesc *resp) noexcept override {
        if (config.dynBatchSupport)
            return InferenceEngine::NOT_IMPLEMENTED;
        for(auto input : config.inConfs) {
            if (!input.inPlace)
                return InferenceEngine::GENERAL_ERROR;
            if (input.constant)
                return InferenceEngine::GENERAL_ERROR;
        }
        for(auto output : config.outConfs) {
            if (output.constant)
                return InferenceEngine::GENERAL_ERROR;
        }
        return InferenceEngine::OK;
    }
    InferenceEngine::StatusCode execute(std::vector<InferenceEngine::Blob::Ptr>& inputs,
                                        std::vector<InferenceEngine::Blob::Ptr>& outputs,
                                        InferenceEngine::ResponseDesc *resp) noexcept override {
        return InferenceEngine::OK;
    }
private:
    InferenceEngine::CNNLayer * cnnLayer;
};

class CustomSplitFactory : public InferenceEngine::ILayerImplFactory {
public:
    CustomSplitFactory(const InferenceEngine::CNNLayer *layer) {
        cnnLayer = const_cast<InferenceEngine::CNNLayer *>(layer);
    }
    // First implementation has more priority than next
    InferenceEngine::StatusCode getImplementations(std::vector<InferenceEngine::ILayerImpl::Ptr>& impls, InferenceEngine::ResponseDesc *resp) noexcept override {
        impls.push_back(InferenceEngine::ILayerImpl::Ptr(new CustomSplitImpl(cnnLayer)));
        return InferenceEngine::OK;
    }

private:
    InferenceEngine::CNNLayer * cnnLayer;
};
using fake_ext_factory = std::function<InferenceEngine::ILayerImplFactory*(const InferenceEngine::CNNLayer *)>;

class FakeExtensionFabric : public InferenceEngine::IExtension {
public:
    FakeExtensionFabric() {
        factories["CustomNewConvolution"] = [](const InferenceEngine::CNNLayer * cnnLayer) -> InferenceEngine::ILayerImplFactory* { return new FakeGenericPrimitiveFactory(); };
        factories["NewDoubleLayer"] = [](const InferenceEngine::CNNLayer * cnnLayer) -> InferenceEngine::ILayerImplFactory* { return new DoublePrimitiveFactory(cnnLayer); };
        factories["NewTwoDifferentOutputs"] = [](const InferenceEngine::CNNLayer * cnnLayer) -> InferenceEngine::ILayerImplFactory* { return new TwoDifferentOutputsFactory(cnnLayer); };
        factories["ConstPrim"] = [](const InferenceEngine::CNNLayer * cnnLayer) -> InferenceEngine::ILayerImplFactory* { return new ConstPrimitiveFactory(cnnLayer); };
        factories["CustomInPlaceConcat"] = [](const InferenceEngine::CNNLayer * cnnLayer) -> InferenceEngine::ILayerImplFactory* { return new CustomConcatFactory(cnnLayer); };
        factories["CustomInPlaceSplit"] = [](const InferenceEngine::CNNLayer * cnnLayer) -> InferenceEngine::ILayerImplFactory* { return new CustomSplitFactory(cnnLayer); };
    }

    virtual ~FakeExtensionFabric() {
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

class MKLDNNGraphGenericTests: public TestsCommon {
protected:
    virtual void SetUp() {
        TestsCommon::SetUp();
        extension.reset(new FakeExtensionFabric());
    }
    std::shared_ptr<InferenceEngine::IExtension> extension;
};

TEST_F(MKLDNNGraphGenericTests, canGetPrimitiveDescriptorsList) {
    MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
    extMgr->AddExtension(extension);
    std::shared_ptr<MKLDNNPlugin::MKLDNNNode> node;
    InferenceEngine::DataPtr dataPtr;
    dataPtr.reset(new InferenceEngine::Data("test", { InferenceEngine::Precision::FP32, {5, 4, 3, 1}, InferenceEngine::Layout::NCHW }));
    InferenceEngine::CNNLayerPtr layerPtr;
    layerPtr.reset(new InferenceEngine::CNNLayer({"name", "CustomNewConvolution", InferenceEngine::Precision::FP32}));
    layerPtr->outData.push_back(dataPtr);

    mkldnn::engine eng(mkldnn::engine(mkldnn::engine::kind::cpu, 0));
    MKLDNNPlugin::MKLDNNWeightsSharing::Ptr cache;
    node.reset(MKLDNNPlugin::MKLDNNNode::CreateNode(layerPtr, eng, extMgr, cache));
    ASSERT_EQ(MKLDNNPlugin::Type::Generic, node->getType());

    ASSERT_NO_THROW(node->getSupportedDescriptors());
}

template <typename data_t>
void ref_double(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst) {
    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    for (int i=0; i < src.size(); i++)
        dst_data[i] = src_data[i]*2;
}

template <typename data_t>
void ref_double_batch1(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst) {
    const data_t *src_data = src.readOnly();
    data_t *dst_data = dst.data();

    for (int i= 0; i < src.size() / 2; i++)
        dst_data[i] = src_data[i]*2;

    for (int i= src.size() / 2; i < src.size(); i++)
        dst_data[i] = 0;
}

template <typename data_t>
void ref_twoDifferent(const InferenceEngine::TBlob<data_t> &src, InferenceEngine::TBlob<data_t> &dst1, InferenceEngine::TBlob<data_t> &dst2) {
    const data_t *src_data = src.readOnly();
    data_t *dst_data1 = dst1.data();
    data_t *dst_data2 = dst2.data();

    for (int i=0; i < dst1.size(); i++)
        dst_data1[i] = (*(src_data++))*2;

    for (int i=0; i < dst2.size(); i++)
        dst_data2[i] = (*(src_data++))*6;
}

TEST_F(MKLDNNGraphGenericTests, DontCreateGPUGenericPrimitive) {
    std::string model = R"V0G0N(
        <Net Name="DoubleLayer_Only" version="2" precision="FP32" batch="1">
            <layers>
                <layer name="in1" type="Input" precision="FP32" id="0">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="gpulayer" id="1" type="CustomGPUConvolution" precision="FP32">
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
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
            </edges>
        </Net>
        )V0G0N";
    MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
    extMgr->AddExtension(extension);

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

    MKLDNNGraphTestClass graph;
    ASSERT_THROW(graph.CreateGraph(network, extMgr), InferenceEngine::details::InferenceEngineException);
}

TEST_F(MKLDNNGraphGenericTests, ExecuteConstGenericPrimitive) {
    std::string model = R"V0G0N(
        <Net Name="DoubleLayer_Only" version="2" precision="FP32" batch="1">
            <layers>
                <layer name="in1" type="Input" precision="FP32" id="0">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="const_layer" id="1" type="ConstPrim" precision="FP32">
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
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
            </edges>
        </Net>
        )V0G0N";
    MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
    extMgr->AddExtension(extension);

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(network, extMgr);

    InferenceEngine::SizeVector dims_src = {1, 3, 5, 5};

    InferenceEngine::Blob::Ptr src =
            InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
    src->allocate();
    fill_data(src->buffer(), src->size());

    InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

    if (srcPtr == nullptr)
        FAIL() << "Cannot cast blob to TBlob<float>.";

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

    InferenceEngine::OutputsDataMap out;
    out = network.getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs;

    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);
    graph.Infer(srcs, outputBlobs);

    InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
    dst_ref.allocate();

    float * dst_data = dst_ref.buffer();
    for (size_t i = 0; i < dst_ref.size(); i++) {
        dst_data[i] = 2;
    }

    compare(*output, dst_ref);
}

TEST_F(MKLDNNGraphGenericTests, ExecuteGenericPrimitive) {
    std::string model = R"V0G0N(
        <Net Name="DoubleLayer_Only" version="2" precision="FP32" batch="1">
            <layers>
                <layer name="in1" type="Input" precision="FP32" id="0">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="double_layer" id="1" type="NewDoubleLayer" precision="FP32">
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
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
            </edges>
        </Net>
        )V0G0N";
    MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
    extMgr->AddExtension(extension);

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(network, extMgr);

    InferenceEngine::SizeVector dims_src = {1, 3, 5, 5};

    InferenceEngine::Blob::Ptr src =
           InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
    src->allocate();
    fill_data(src->buffer(), src->size());

    InferenceEngine::TBlob<float>* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

    if (srcPtr == nullptr)
        FAIL() << "Cannot cast blob to TBlob<float>.";

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

    InferenceEngine::OutputsDataMap out;
    out = network.getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs;

    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
    dst_ref.allocate();

    ref_double(*srcPtr, dst_ref);

    compare(*output, dst_ref);
}

TEST_F(MKLDNNGraphGenericTests, ExecuteGenericPrimitiveWithTwoOutputs) {
    std::string model = R"V0G0N(
        <Net Name="DoubleLayer_Only" version="2" precision="FP32" batch="1">
            <layers>
                <layer name="in1" type="Input" precision="FP32" id="0">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="two_diff_layer" id="1" type="NewTwoDifferentOutputs" precision="FP32">
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
                            <dim>1</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                        <port id="3">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="power" id="3" type="Power" precision="FP32">
                    <power_data power="1" scale="2" shift="0"/>
                    <input>
                        <port id="1">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="con" id="2" type="Concat" precision="FP32">
                    <concat_data axis="1"/>
                    <input>
                        <port id="4">
                            <dim>1</dim>
                            <dim>1</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                        <port id="5">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </input>
                    <output>
                        <port id="6">
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
                <edge from-layer="1" from-port="2" to-layer="2" to-port="4"/>
                <edge from-layer="1" from-port="3" to-layer="3" to-port="1"/>
                <edge from-layer="3" from-port="2" to-layer="2" to-port="5"/>
            </edges>
        </Net>
        )V0G0N";
    MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
    extMgr->AddExtension(extension);

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(network, extMgr);

    InferenceEngine::SizeVector dims_src = {1, 3, 5, 5};

    InferenceEngine::Blob::Ptr src =
            InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
    src->allocate();

    float * data_src = src->buffer();
    for (size_t i = 0; i < src->size(); i++)
        data_src[i] = 1;
    auto * srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

    if (srcPtr == nullptr)
        FAIL() << "Cannot cast blob to TBlob<float>.";

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

    InferenceEngine::OutputsDataMap out;
    out = network.getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs;

    auto it = out.begin();

    std::pair<std::string, InferenceEngine::DataPtr> item = *it;
    InferenceEngine::DataPtr data1 = item.second;

    InferenceEngine::TensorDesc outputDesc1 = item.second->getTensorDesc();
    InferenceEngine::TBlob<float>::Ptr output1;
    output1 = InferenceEngine::make_shared_blob<float>(outputDesc1);
    output1->allocate();
    outputBlobs[item.first] = output1;

    graph.Infer(srcs, outputBlobs);

    float * data = outputBlobs.begin()->second->buffer();
    for (size_t i = 0; i < 25; i++) {
        ASSERT_EQ(*data, 2);
        data++;
    }
    for (size_t i = 0; i < 50; i++) {
        ASSERT_EQ(*data, 6);
        data++;
    }
}

TEST_F(MKLDNNGraphGenericTests, ExecuteGenericInPlaceConcat) {
    std::string model = R"V0G0N(
        <Net Name="CustomConcat_Only" version="2" precision="FP32" batch="1">
            <layers>
                <layer name="in1" type="Input" precision="FP32" id="0">
                    <output>
                        <port id="0">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
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
                <layer name="con" id="2" type="CustomInPlaceConcat" precision="FP32">
                    <input>
                        <port id="1">
                            <dim>1</dim>
                            <dim>3</dim>
                            <dim>5</dim>
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
                            <dim>5</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
                <edge from-layer="1" from-port="0" to-layer="2" to-port="2"/>
            </edges>
        </Net>
        )V0G0N";
    MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
    extMgr->AddExtension(extension);

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(network, extMgr);

    InferenceEngine::SizeVector dims_src1 = {1, 3, 5, 5};

    InferenceEngine::Blob::Ptr src1 =
            InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src1, InferenceEngine::NCHW});
    src1->allocate();

    float * data_src1 = src1->buffer();
    for (size_t i = 0; i < src1->size(); i++)
        data_src1[i] = 1;

    InferenceEngine::SizeVector dims_src2 = {1, 2, 5, 5};

    InferenceEngine::Blob::Ptr src2 =
            InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src2, InferenceEngine::NCHW});
    src2->allocate();

    float * data_src2 = src2->buffer();
    for (size_t i = 0; i < src2->size(); i++)
        data_src2[i] = 2;

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src1));
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in2", src2));

    InferenceEngine::OutputsDataMap out;
    out = network.getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs;

    auto it = out.begin();

    std::pair<std::string, InferenceEngine::DataPtr> item = *it;

    InferenceEngine::TensorDesc outputDesc1 = item.second->getTensorDesc();
    InferenceEngine::TBlob<float>::Ptr output1;
    output1 = InferenceEngine::make_shared_blob<float>(outputDesc1);
    output1->allocate();
    outputBlobs[item.first] = output1;

    graph.Infer(srcs, outputBlobs);

    float * data = outputBlobs.begin()->second->buffer();
    for (size_t i = 0; i < 75; i++) {
        ASSERT_EQ(*data, 1);
        data++;
    }
    for (size_t i = 0; i < 50; i++) {
        ASSERT_EQ(*data, 2);
        data++;
    }
}

TEST_F(MKLDNNGraphGenericTests, ExecuteGenericInPlaceSplit) {
    std::string model = R"V0G0N(
        <net name="ConcatOnly" version="2" precision="FP32" batch="1">
            <layers>
                <layer name="in1" type="Input" precision="FP32" id="1">
                    <output>
                        <port id="1">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>4</dim>
                            <dim>4</dim>
                        </port>
                    </output>
                </layer>
                <layer name="split" id="2" type="CustomInPlaceSplit" precision="FP32">
                    <input>
                        <port id="1">
                            <dim>1</dim>
                            <dim>4</dim>
                            <dim>4</dim>
                            <dim>4</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>4</dim>
                            <dim>4</dim>
                        </port>
                        <port id="3">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>4</dim>
                            <dim>4</dim>
                        </port>
                    </output>
                </layer>
                <layer name="power1" id="3" type="Power" precision="FP32">
                    <power_data power="1" scale="1" shift="3"/>
                    <input>
                        <port id="1">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>4</dim>
                            <dim>4</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>4</dim>
                            <dim>4</dim>
                        </port>
                    </output>
                </layer>
                <layer name="power2" id="4" type="Power" precision="FP32">
                    <power_data power="1" scale="1" shift="2"/>
                    <input>
                        <port id="1">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>4</dim>
                            <dim>4</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2">
                            <dim>1</dim>
                            <dim>2</dim>
                            <dim>4</dim>
                            <dim>4</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="1" from-port="1" to-layer="2" to-port="1"/>
                <edge from-layer="2" from-port="2" to-layer="3" to-port="1"/>
                <edge from-layer="2" from-port="3" to-layer="4" to-port="1"/>
            </edges>
        </net>
        )V0G0N";
    MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
    extMgr->AddExtension(extension);

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(network, extMgr);

    InferenceEngine::SizeVector dims_src = {1, 4, 4, 4};

    InferenceEngine::Blob::Ptr src =
            InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
    src->allocate();

    float * data_src = src->buffer();
    for (size_t i = 0; i < src->size(); i++) {
        if (i < src->size() / 2)
            data_src[i] = 1;
        else
            data_src[i] = 2;
    }

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

    InferenceEngine::OutputsDataMap out;
    out = network.getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs;
    auto it = out.begin();

    std::pair<std::string, InferenceEngine::DataPtr> item = *it;

    InferenceEngine::TBlob<float>::Ptr output1;
    output1 = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output1->allocate();
    outputBlobs[item.first] = output1;

    item = *(++it);
    InferenceEngine::TBlob<float>::Ptr output2;
    output2 = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output2->allocate();
    outputBlobs[item.first] = output2;

    graph.Infer(srcs, outputBlobs);

    float * data = output1->buffer();
    for (size_t i = 0; i < output1->size(); i++) {
        ASSERT_EQ(*data, 4);
        data++;
    }
    data = output2->buffer();
    for (size_t i = 0; i < output2->size(); i++) {
        ASSERT_EQ(*data, 4);
        data++;
    }
}

TEST_F(MKLDNNGraphGenericTests, ExecuteGenericPrimitiveWithDynamicBatch) {
    std::string model = R"V0G0N(
        <Net Name="DoubleLayer_Only" version="2" precision="FP32" batch="2">
            <layers>
                <layer name="in1" type="Input" precision="FP32" id="0">
                    <output>
                        <port id="0">
                            <dim>2</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
                <layer name="double_layer" id="1" type="NewDoubleLayer" precision="FP32">
                    <input>
                        <port id="1">
                            <dim>2</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </input>
                    <output>
                        <port id="2">
                            <dim>2</dim>
                            <dim>3</dim>
                            <dim>5</dim>
                            <dim>5</dim>
                        </port>
                    </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
            </edges>
        </Net>
        )V0G0N";
    MKLDNNPlugin::MKLDNNExtensionManager::Ptr extMgr(new MKLDNNPlugin::MKLDNNExtensionManager());
    extMgr->AddExtension(extension);

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(network, extMgr);

    InferenceEngine::SizeVector dims_src = {2, 3, 5, 5};

    InferenceEngine::Blob::Ptr src =
            InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
    src->allocate();
    fill_data(src->buffer(), src->size());

    auto* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

    if (srcPtr == nullptr)
        FAIL() << "Cannot cast blob to TBlob<float>.";

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("in1", src));

    InferenceEngine::OutputsDataMap out;
    out = network.getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs;

    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    float *dstData = output->data();

    for (size_t i = 0; i < output->size(); i++) {
        dstData[i] = 0;
    }

    graph.Infer(srcs, outputBlobs);

    InferenceEngine::TBlob<float> dst_ref(item.second->getTensorDesc());
    dst_ref.allocate();

    ref_double(*srcPtr, dst_ref);

    compare(*output, dst_ref);

    graph.setProperty({{InferenceEngine::PluginConfigParams::KEY_DYN_BATCH_LIMIT, "1"}});

    for (size_t i = 0; i < output->size(); i++) {
        dstData[i] = 0;
    }

    graph.Infer(srcs, outputBlobs);

    InferenceEngine::TBlob<float> dst_ref2(item.second->getTensorDesc());
    dst_ref2.allocate();

    ref_double_batch1(*srcPtr, dst_ref2);

    compare(*output, dst_ref2);
}

TEST_F(MKLDNNGraphGenericTests, ExecuteNotInLineGRN) {
    std::string model = R"V0G0N(
<net name="default" version="2" batch="1">
    <layers>
        <layer name="data" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="norm_8x_grn" type="GRN" precision="FP32" id="1">
            <data bias="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="norm_4x_grn" type="GRN" precision="FP32" id="2">
            <data bias="1"/>
            <input>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="merge_4x_8x_concat" type="Concat" precision="FP32" id="3">
            <concat_data axis="1"/>
            <input>
                <port id="5">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
                <port id="6">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="7">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="1" to-port="1"/>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="3"/>
        <edge from-layer="1" from-port="2" to-layer="3" to-port="5"/>
        <edge from-layer="2" from-port="4" to-layer="3" to-port="6"/>
    </edges>
</net>)V0G0N";
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(network);

    InferenceEngine::SizeVector dims_src = {1, 3, 2, 2};

    InferenceEngine::Blob::Ptr src =
            InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
    src->allocate();
    fill_data(src->buffer(), src->size());

    auto* srcPtr = dynamic_cast<InferenceEngine::TBlob<float>*>(src.get());

    if (srcPtr == nullptr)
        FAIL() << "Cannot cast blob to TBlob<float>.";

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data", src));

    InferenceEngine::OutputsDataMap out;
    out = network.getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs;

    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    std::vector<float> refDst = {0.000f, 0.503f, 0.659f, 0.117f, -0.474f, -0.573f, -0.202f, 0.545f, 0.619f, 0.246f,
                                 0.000f, 0.000f, 0.000f, 0.503f, 0.659f, 0.117f, -0.474f, -0.573f, -0.202f, 0.545f,
                                 0.619f, 0.246f, 0.000f, 0.000f};

    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc(), refDst.data());

    compare(*output, *dstOut);
}

TEST_F(MKLDNNGraphGenericTests, ExecuteInLineGRN) {
    std::string model = R"V0G0N(
<net name="default" version="2" batch="1">
    <layers>
        <layer name="data1" type="Input" precision="FP32" id="0">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="data2" type="Input" precision="FP32" id="1">
            <output>
                <port id="0">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="norm_8x_grn" type="GRN" precision="FP32" id="2">
            <data bias="1"/>
            <input>
                <port id="1">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="2">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="norm_4x_grn" type="GRN" precision="FP32" id="3">
            <data bias="1"/>
            <input>
                <port id="3">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="4">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
        <layer name="merge_4x_8x_concat" type="Concat" precision="FP32" id="4">
            <concat_data axis="1"/>
            <input>
                <port id="5">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
                <port id="6">
                    <dim>1</dim>
                    <dim>3</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </input>
            <output>
                <port id="7">
                    <dim>1</dim>
                    <dim>6</dim>
                    <dim>2</dim>
                    <dim>2</dim>
                </port>
            </output>
        </layer>
    </layers>
    <edges>
        <edge from-layer="0" from-port="0" to-layer="2" to-port="1"/>
        <edge from-layer="1" from-port="0" to-layer="3" to-port="3"/>
        <edge from-layer="2" from-port="2" to-layer="4" to-port="5"/>
        <edge from-layer="3" from-port="4" to-layer="4" to-port="6"/>
    </edges>
</net>)V0G0N";

    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network;
    ASSERT_NO_THROW(network = core.ReadNetwork(model, InferenceEngine::Blob::CPtr()));

    MKLDNNGraphTestClass graph;
    graph.CreateGraph(network);

    InferenceEngine::SizeVector dims_src = {1, 3, 2, 2};

    InferenceEngine::Blob::Ptr src1 =
            InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
    src1->allocate();
    fill_data(src1->buffer(), src1->size());

    InferenceEngine::Blob::Ptr src2 =
            InferenceEngine::make_shared_blob<float>({InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NCHW});
    src2->allocate();
    fill_data(src2->buffer(), src2->size());

    InferenceEngine::BlobMap srcs;
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data1", src1));
    srcs.insert(std::pair<std::string, InferenceEngine::Blob::Ptr>("data2", src2));

    InferenceEngine::OutputsDataMap out;
    out = network.getOutputsInfo();
    InferenceEngine::BlobMap outputBlobs;

    std::pair<std::string, InferenceEngine::DataPtr> item = *out.begin();

    InferenceEngine::TBlob<float>::Ptr output;
    output = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc());
    output->allocate();
    outputBlobs[item.first] = output;

    graph.Infer(srcs, outputBlobs);

    std::vector<float> refDst = {0.000f, 0.503f, 0.659f, 0.117f, -0.474f, -0.573f, -0.202f, 0.545f, 0.619f, 0.246f,
                                 0.000f, 0.000f, 0.000f, 0.503f, 0.659f, 0.117f, -0.474f, -0.573f, -0.202f, 0.545f,
                                 0.619f, 0.246f, 0.000f, 0.000f};

    InferenceEngine::TBlob<float>::Ptr dstOut = InferenceEngine::make_shared_blob<float>(item.second->getTensorDesc(), refDst.data());

    compare(*output, *dstOut);
}

// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_builders.hpp>
#include <ie_cnn_layer_builder.h>
#include <cnn_network_impl.hpp>

#include <memory>
#include <vector>
#include <unordered_set>
#include <string>

using namespace InferenceEngine;

class BaseConverter {
public:
    explicit BaseConverter(const std::string& type): type(type) {}

    virtual CNNLayer::Ptr createLayer(const std::shared_ptr<const ILayer>& layer, Precision precision) = 0;
    virtual bool canCreate(const std::string& nodeType) const = 0;

protected:
    std::string type;
};

template <class CLT>
class LayerConverter: public BaseConverter {
public:
    explicit LayerConverter(const std::string& type): BaseConverter(type) {}

    CNNLayer::Ptr createLayer(const std::shared_ptr<const ILayer>& layer, Precision precision) override {
        LayerParams params = {layer->getName(), layer->getType(), precision};
        auto res = std::make_shared<CLT>(params);

        auto * weightLayerPtr = dynamic_cast<WeightableLayer *>(res.get());

        for (const auto& port : layer->getInputPorts()) {
            if (port.getParameters().find("type") == port.getParameters().end() ||
                    port.getData()->getData()->cbuffer() == nullptr)
                continue;
            res->blobs[port.getParameters().at("type")] = port.getData()->getData();
            if (weightLayerPtr == nullptr)
                continue;
            if (port.getParameters().at("type").as<std::string>() == "weights") {
                weightLayerPtr->_weights = port.getData()->getData();
            } else if (port.getParameters().at("type").as<std::string>() == "biases") {
                weightLayerPtr->_biases = port.getData()->getData();
            }
        }

        // For constant layers
        for (auto& it : layer->getParameters()) {
            if (it.second.is<Blob::CPtr>()) {
                res->blobs[it.first] = std::const_pointer_cast<Blob>(it.second.as<Blob::CPtr>());
            } else if (it.second.is<Blob::Ptr>()) {
                res->blobs[it.first] = it.second.as<Blob::Ptr>();
            }
        }

        res->params = InferenceEngine::Builder::convertParameters2Strings(layer->getParameters());
        return res;
    }

    bool canCreate(const std::string& nodeType) const override {
        details::CaselessEq<std::string> comparator;
        return comparator(nodeType, type);
    }
};

class ActivationConverter: public BaseConverter {
public:
    ActivationConverter(): BaseConverter("Activation") {}

    CNNLayer::Ptr createLayer(const std::shared_ptr<const ILayer>& layer, Precision precision) override {
        LayerParams params = {layer->getName(), layer->getType(), precision};
        static details::caseless_map<std::string, std::shared_ptr<BaseConverter>> activationCreators = {
                {"relu", std::make_shared<LayerConverter<InferenceEngine::ReLULayer>>("ReLU")},
                {"prelu", std::make_shared<LayerConverter<InferenceEngine::PReLULayer>>("PReLU")},
                {"clamp", std::make_shared<LayerConverter<InferenceEngine::ClampLayer>>("Clamp")},
                {"elu", std::make_shared<LayerConverter<InferenceEngine::CNNLayer>>("ELU")},
                {"sigmoid", std::make_shared<LayerConverter<InferenceEngine::CNNLayer>>("Sigmoid")},
                {"tanh", std::make_shared<LayerConverter<InferenceEngine::CNNLayer>>("TanH")},
                {"log", std::make_shared<LayerConverter<InferenceEngine::CNNLayer>>("Log")},
                {"divbyn", std::make_shared<LayerConverter<InferenceEngine::CNNLayer>>("DivByN")},
                {"exp", std::make_shared<LayerConverter<InferenceEngine::CNNLayer>>("exp")},
        };

        auto typeIt = layer->getParameters().find("type");
        if (typeIt == layer->getParameters().end())
            THROW_IE_EXCEPTION << "Unsupported Activation layer. Type is unknown.";

        auto activationBuilder = activationCreators.find(typeIt->second);
        if (activationBuilder == activationCreators.end()) {
            THROW_IE_EXCEPTION << "Unsupported Activation layer type: " << typeIt->second.as<std::string>();
        }

        auto activation = activationBuilder->second->createLayer(layer, precision);

        activation->type = activationBuilder->first;
        activation->params.erase("type");
        activation->validateLayer();
        return activation;
    }

    bool canCreate(const std::string& nodeType) const override {
        details::CaselessEq<std::string> comparator;
        return comparator(nodeType, type);
    }
};

class RNNSequenceConverter: public BaseConverter {
public:
    RNNSequenceConverter(): BaseConverter("RNN") {}

    CNNLayer::Ptr createLayer(const std::shared_ptr<const ILayer>& layer, Precision precision) override {
        auto rnnLayer = LayerConverter<InferenceEngine::RNNSequenceLayer>("RNN").createLayer(layer, precision);
        rnnLayer->type = "RNN";
        std::string type = layer->getType();
        size_t pos = type.find("Sequence");
        if (pos != std::string::npos)
            type.erase(pos);
        rnnLayer->params["cell_type"] = type;
        return rnnLayer;
    }

    bool canCreate(const std::string& nodeType) const override {
        static const details::caseless_set<std::string> supportedRnnTypes {
            "LSTMSequence", "GRUSequence", "RNNSequence"
        };
        return supportedRnnTypes.find(nodeType) != supportedRnnTypes.end();
    }
};

const std::shared_ptr<ICNNNetwork> Builder::convertToICNNNetwork(const INetwork::CPtr& network) {
    auto createCNNLayer = [](const std::shared_ptr<const ILayer>& layer, Precision precision) {
        static std::vector<std::shared_ptr<BaseConverter>> convertors = {
                std::make_shared<LayerConverter<InferenceEngine::PowerLayer>>("Power"),
                std::make_shared<LayerConverter<InferenceEngine::ConvolutionLayer>>("Convolution"),
                std::make_shared<LayerConverter<InferenceEngine::DeformableConvolutionLayer>>("DeformableConvolution"),
                std::make_shared<LayerConverter<InferenceEngine::DeconvolutionLayer>>("Deconvolution"),
                std::make_shared<LayerConverter<InferenceEngine::PoolingLayer>>("Pooling"),
                std::make_shared<LayerConverter<InferenceEngine::FullyConnectedLayer>>("InnerProduct"),
                std::make_shared<LayerConverter<InferenceEngine::FullyConnectedLayer>>("FullyConnected"),
                std::make_shared<LayerConverter<InferenceEngine::NormLayer>>("LRN"),
                std::make_shared<LayerConverter<InferenceEngine::NormLayer>>("Norm"),
                std::make_shared<LayerConverter<InferenceEngine::SoftMaxLayer>>("Softmax"),
                std::make_shared<LayerConverter<InferenceEngine::SoftMaxLayer>>("LogSoftmax"),
                std::make_shared<LayerConverter<InferenceEngine::GRNLayer>>("GRN"),
                std::make_shared<LayerConverter<InferenceEngine::MVNLayer>>("MVN"),
                std::make_shared<LayerConverter<InferenceEngine::ReLULayer>>("ReLU"),
                std::make_shared<LayerConverter<InferenceEngine::ClampLayer>>("Clamp"),
                std::make_shared<LayerConverter<InferenceEngine::SplitLayer>>("Split"),
                std::make_shared<LayerConverter<InferenceEngine::SplitLayer>>("Slice"),
                std::make_shared<LayerConverter<InferenceEngine::ConcatLayer>>("Concat"),
                std::make_shared<LayerConverter<InferenceEngine::EltwiseLayer>>("Eltwise"),
                std::make_shared<LayerConverter<InferenceEngine::ScaleShiftLayer>>("ScaleShift"),
                std::make_shared<LayerConverter<InferenceEngine::PReLULayer>>("PReLU"),
                std::make_shared<LayerConverter<InferenceEngine::CropLayer>>("Crop"),
                std::make_shared<LayerConverter<InferenceEngine::ReshapeLayer>>("Reshape"),
                std::make_shared<LayerConverter<InferenceEngine::ReshapeLayer>>("Flatten"),
                std::make_shared<LayerConverter<InferenceEngine::TileLayer>>("Tile"),
                std::make_shared<LayerConverter<InferenceEngine::PadLayer>>("Pad"),
                std::make_shared<ActivationConverter>(),
                std::make_shared<RNNSequenceConverter>(),
                std::make_shared<LayerConverter<InferenceEngine::BatchNormalizationLayer>>("BatchNormalization"),
        };
        for (auto &convertor : convertors) {
            if (!convertor->canCreate(layer->getType()))
                continue;
            return convertor->createLayer(layer, precision);
        }
        static LayerConverter<CNNLayer> genericCreator("");
        return genericCreator.createLayer(layer, precision);
    };

    auto keep_input_info = [](std::unique_ptr<details::CNNNetworkImpl>& network, DataPtr &in_data,
            PreProcessInfo preProc) {
        InputInfo::Ptr info(new InputInfo());
        info->getPreProcess() = preProc;
        info->setInputData(in_data);
        Precision prc = info->getPrecision();

        // Convert precision into native format (keep element size)
        prc = prc == Precision::Q78 ? Precision::I16 :
              prc == Precision::FP16 ? Precision::FP32 :
              static_cast<Precision::ePrecision>(prc);

        info->setPrecision(prc);
        network->setInputInfo(info);
    };

    std::unique_ptr<details::CNNNetworkImpl> cnnNetworkImpl(new details::CNNNetworkImpl());

    Precision detectedPrecision = Precision::UNSPECIFIED;
    for (const auto& layer : *network) {
        for (const auto& port : layer->getInputPorts()) {
            Precision prc = port.getData()->getData()->getTensorDesc().getPrecision();
            if (prc != Precision::UNSPECIFIED) {
                detectedPrecision = prc;
                break;
            }
        }
        for (const auto& port : layer->getOutputPorts()) {
            Precision prc = port.getData()->getData()->getTensorDesc().getPrecision();
            if (prc != Precision::UNSPECIFIED) {
                detectedPrecision = prc;
                break;
            }
        }
        if (detectedPrecision != Precision::UNSPECIFIED)
            break;
    }
    if (detectedPrecision == Precision::UNSPECIFIED)
        detectedPrecision = Precision::FP32;

    details::CaselessEq<std::string> eq;
    cnnNetworkImpl->setName(network->getName());
    cnnNetworkImpl->setPrecision(Precision::UNSPECIFIED);
    for (const auto& layer : *network) {
        bool isInternalLayer = eq(layer->getType(), "Const");
        for (const auto& connection : network->getLayerConnections(layer->getId())) {
            if (!isInternalLayer)
                break;
            if (connection.from().layerId() != layer->getId())
                continue;
            const auto& port = network->getLayer(connection.to().layerId())->getInputPorts()[connection.to().portId()];
            isInternalLayer = isInternalLayer &&
                    port.getParameters().find("type") != port.getParameters().end();
        }
        isInternalLayer = isInternalLayer || eq(layer->getType(), "Output");

        if (isInternalLayer)
            continue;

        CNNLayerPtr cnnLayer = createCNNLayer(layer, detectedPrecision);
        if (cnnLayer == nullptr)
            THROW_IE_EXCEPTION << "Could not create CNN layer '" << layer->getName() << "'";
        if (cnnNetworkImpl->getPrecision() == Precision::UNSPECIFIED) {
            cnnNetworkImpl->setPrecision(cnnLayer->precision);
        } else if (cnnNetworkImpl->getPrecision() == Precision::MIXED &&
                   cnnNetworkImpl->getPrecision() != cnnLayer->precision) {
            cnnNetworkImpl->setPrecision(Precision::MIXED);
        }

        auto connections = network->getLayerConnections(layer->getId());
        std::unordered_set<idx_t> inputNum, outputNum;
        for (const auto& connection : connections) {
            if (connection.from().layerId() != layer->getId()) {
                const auto& port = layer->getInputPorts()[connection.to().portId()];
                if (port.getParameters().find("type") == port.getParameters().end())
                    inputNum.insert(connection.to().portId());
            } else {
                outputNum.insert(connection.from().portId());
            }
        }
        cnnLayer->insData.resize(inputNum.size());
        cnnLayer->outData.resize(outputNum.size());
        cnnNetworkImpl->addLayer(cnnLayer);
    }

    for (const auto& layer : *network) {
        auto connections = network->getLayerConnections(layer->getId());
        CNNLayerPtr cnnLayer;
        StatusCode sts = cnnNetworkImpl->getLayerByName(layer->getName().c_str(), cnnLayer, nullptr);

        if (sts != OK && (eq(layer->getType(), "Output") || eq(layer->getType(), "Const")))
            continue;
        else if (sts != OK)
            THROW_IE_EXCEPTION << "Cannot find CNNLayer by name " << layer->getName();

        for (const auto& connection : connections) {
            if (connection.from().layerId() != layer->getId())
                continue;

            const auto& outLayer = network->getLayer(connection.to().layerId());

            CNNLayerPtr cnnOutLayer;
            sts = cnnNetworkImpl->getLayerByName(outLayer->getName().c_str(), cnnOutLayer, nullptr);
            if (sts != OK && !eq(outLayer->getType(), "Output") && !eq(layer->getType(), "Const"))
                THROW_IE_EXCEPTION << "Cannot find CNNLayer by name " << outLayer->getName();

            std::string dataName = layer->getName();
            if (cnnLayer->outData.size() > 1) {
                dataName += "." + std::to_string(connection.from().portId());
            }
            DataPtr& data = cnnNetworkImpl->getData(dataName);
            if (!data) {
                TensorDesc dataDesc(detectedPrecision, layer->getOutputPorts()[connection.from().portId()].shape(),
                                    TensorDesc::getLayoutByDims(layer->getOutputPorts()[connection.from().portId()].shape()));
                data = std::make_shared<Data>(dataName, dataDesc);
                data->getCreatorLayer() = cnnLayer;
            }
            cnnLayer->outData[connection.from().portId()] = data;

            idx_t realPortId(0);
            const auto inputPorts = outLayer->getInputPorts();
            for (size_t i = 0; i < connection.to().portId() && i < inputPorts.size(); i++) {
                if (inputPorts[i].getParameters().find("type") == inputPorts[i].getParameters().end())
                    realPortId++;
            }
            if (cnnOutLayer) {
                data->getInputTo()[outLayer->getName()] = cnnOutLayer;
                cnnOutLayer->insData[realPortId] = data;
            } else {
                cnnNetworkImpl->addOutput(data->getName());
            }
        }

        cnnLayer->validateLayer();
        if (eq(cnnLayer->type, "Input")) {
            PreProcessInfo preProc;
            if (layer->getParameters().find("preProcess") != layer->getParameters().end())
                preProc = layer->getParameters().at("preProcess");
            keep_input_info(cnnNetworkImpl, *cnnLayer->outData.begin(), preProc);
        }
    }

    // Set default output precision to FP32 (for back-compatibility)
    OutputsDataMap outputsInfo;
    cnnNetworkImpl->getOutputsInfo(outputsInfo);
    for (auto outputInfo : outputsInfo) {
        if (outputInfo.second->getPrecision() != Precision::FP32 &&
            outputInfo.second->getPrecision() != Precision::I32) {
            outputInfo.second->setPrecision(Precision::FP32);
        }
    }

    return std::shared_ptr<ICNNNetwork>(cnnNetworkImpl.release());
}

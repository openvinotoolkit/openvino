// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <set>
#include <unordered_set>
#include "v2_format_parser.h"
#include "v2_layer_parsers.h"
#include "xml_parse_utils.h"
#include "ie_blob_proxy.hpp"
#include "range_iterator.hpp"
#include <fstream>
#include "ie_icnn_network_stats.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace XMLParseUtils;
using namespace std;

void LayerParseParameters::addOutputPort(const LayerPortData &port) {
    outputPorts.insert(std::upper_bound(outputPorts.begin(), outputPorts.end(), port,
                                        [=](const LayerParseParameters::LayerPortData &lhs,
                                                const LayerParseParameters::LayerPortData &rhs) {
                                            return lhs.portId < rhs.portId;
                                        }), port);
}


void LayerParseParameters::addInputPort(const LayerPortData &port) {
    inputPorts.insert(std::upper_bound(inputPorts.begin(), inputPorts.end(), port,
                                       [=](const LayerParseParameters::LayerPortData &lhs,
                                               const LayerParseParameters::LayerPortData &rhs) {
                                           return lhs.portId < rhs.portId;
                                       }), port);
}

inline void ParseSegment(LayerParseParameters& prms, const pugi::xml_node &blob) {
    int size = GetIntAttr(blob, "size", 0);
    int start = GetIntAttr(blob, "offset", 0);
    if (!size)
        return;
    if (size < 0 || start < 0)
        THROW_IE_EXCEPTION << "Layer " << prms.prms.name << " has incorrect blob: " << blob.name();

    WeightSegment& segment = prms.blobs[blob.name()];
    segment.start = static_cast<size_t>(start);
    segment.size = static_cast<size_t>(size);
    const std::string &preStr = GetStrAttr(blob, "precision", "");
    if (!preStr.empty())
        segment.precision = Precision::FromStr(preStr);
    else
        segment.precision = prms.prms.precision;
}

int BaseCreator::version_ = 3;

void V2FormatParser::ParsePort(LayerParseParameters::LayerPortData& port, pugi::xml_node &node) const {
    port.portId = GetIntAttr(node, "id");
    ParseDims(port.dims, node);
    const std::string &preStr = GetStrAttr(node, "precision", "");
    if (!preStr.empty()) port.precision = Precision::FromStr(preStr);
}

void V2FormatParser::ParseGenericParams(pugi::xml_node& node, LayerParseParameters& layerParsePrms) const {
    layerParsePrms.layerId = GetIntAttr(node, "id");

    InferenceEngine::LayerParams& prms = layerParsePrms.prms;
    prms.type = XMLParseUtils::GetStrAttr(node, "type");
    prms.precision = _defPrecision;

    prms.name = GetStrAttr(node, "name");
    const std::string& preStr = GetStrAttr(node, "precision", "");
    if (!preStr.empty())
        prms.precision = Precision::FromStr(preStr);

    if (prms.precision == Precision::MIXED) {
        THROW_IE_EXCEPTION << "Layer precision must not be MIXED, at layer name: " << prms.name << ", offset: "
                           << node.offset_debug();
    }

    auto outNode = node.child("output");
    if (!outNode.empty()) {
        FOREACH_CHILD(_cn, outNode, "port") {
            LayerParseParameters::LayerPortData port;
            port.precision = prms.precision;
            ParsePort(port, _cn);
            layerParsePrms.addOutputPort(port);
        }
    }
    auto inpNode = node.child("input");
    if (!inpNode.empty()) {
        FOREACH_CHILD(_cn, inpNode, "port") {
            LayerParseParameters::LayerPortData port;
            port.precision = prms.precision;
            ParsePort(port, _cn);
            layerParsePrms.addInputPort(port);
        }
    }
    auto blob = node.child("biases");
    if (!blob.empty()) {
        ParseSegment(layerParsePrms, blob);
    }
    blob = node.child("weights");
    if (!blob.empty()) {
        ParseSegment(layerParsePrms, blob);
    }
    auto blobs = node.child("blobs");
    if (!blobs.empty()) {
    for (blob = blobs.first_child(); !blob.empty(); blob = blob.next_sibling()) {
        ParseSegment(layerParsePrms, blob);
        }
    }
}

InferenceEngine::CNNLayer::Ptr V2FormatParser::CreateLayer(pugi::xml_node& node,
                                                       LayerParseParameters& layerParsePrms) const {
    for (auto &creator : getCreators()) {
        if (!creator->shouldCreate(layerParsePrms.prms.type))
            continue;
        return creator->CreateLayer(node, layerParsePrms);
    }
    static V2LayerCreator<GenericLayer> genericCreator("");
    return genericCreator.CreateLayer(node, layerParsePrms);
}

void V2FormatParser::SetLayerInput(CNNNetworkImpl& network, const std::string& dataId,
                                   CNNLayerPtr& targetLayer, int inputPort) {
    DataPtr& dataPtr = _portsToData[dataId];
    if (!dataPtr) THROW_IE_EXCEPTION << "in Layer " << targetLayer->name
                                     << ": trying to connect an edge to non existing output port: " << dataId;

    dataPtr->getInputTo()[targetLayer->name] = targetLayer;
    const LayerParseParameters& parseInfo = layersParseInfo[targetLayer->name];
    if (targetLayer->insData.empty()) {
        targetLayer->insData.resize(parseInfo.inputPorts.size());
    }
    for (unsigned i = 0; i < parseInfo.inputPorts.size(); i++) {
        if (parseInfo.inputPorts[i].portId != inputPort) continue;
        if (parseInfo.inputPorts[i].precision != dataPtr->getPrecision()) {
            if (dataPtr->getPrecision() == Precision::UNSPECIFIED) {
                dataPtr->setPrecision(parseInfo.inputPorts[i].precision);
            } else {
                // TODO: Make a correct exception

                /*THROW_IE_EXCEPTION << "in Layer " << targetLayer->name
                                   << ": trying to connect an edge to mismatch precision of output port: "
                                   << dataPtr->getName();*/
            }
        }
        if (!equal(parseInfo.inputPorts[i].dims, dataPtr->getDims()))
            THROW_IE_EXCEPTION << "in Layer " << targetLayer->name
                               << ": trying to connect an edge to mismatch dimensions of output port: "
                               << dataPtr->getName()
                               << " dims input: " << dumpVec(parseInfo.inputPorts[i].dims)
                               << " dims output: " << dumpVec(dataPtr->getDims());
        targetLayer->insData[i] = dataPtr;
        return;
    }
    THROW_IE_EXCEPTION << "input port " << inputPort << " does not exist in layer " << targetLayer->name;
}

V2FormatParser::V2FormatParser(int version): _version(version) {
    BaseCreator::version_ = version;
}

CNNNetworkImplPtr V2FormatParser::Parse(pugi::xml_node& root) {
    _network.reset(new CNNNetworkImpl());
    _network->setName(GetStrAttr(root, "name", ""));
    _defPrecision = Precision::FromStr(GetStrAttr(root, "precision", "UNSPECIFIED"));
    _network->setPrecision(_defPrecision);
    // parse the input Data
    DataPtr inputData;
    if (_version == 1) {
        inputData = ParseInputData(root);
        _portsToData[inputData->getName()] = inputData;  // hack as this input does not have ports ids
        InputInfo::Ptr info(new InputInfo());
        info->setInputData(inputData);
        _network->setInputInfo(info);
    }

    // parse the graph layers
    auto allLayersNode = root.child("layers");
    std::vector< CNNLayer::Ptr> inputLayers;
    int nodeCnt = 0;
    std::map<int, CNNLayer::Ptr> layerById;
    bool identifyNetworkPrecision = _defPrecision == Precision::UNSPECIFIED;
    for (auto node = allLayersNode.child("layer"); !node.empty(); node = node.next_sibling("layer")) {
        LayerParseParameters lprms;
        ParseGenericParams(node, lprms);

        CNNLayer::Ptr layer = CreateLayer(node, lprms);
        if (!layer) THROW_IE_EXCEPTION << "Don't know how to create Layer type: " << lprms.prms.type;

        layersParseInfo[layer->name] = lprms;
        _network->addLayer(layer);
        layerById[lprms.layerId] = layer;

        if (equal(layer->type, "input")) {
            inputLayers.push_back(layer);
        }

        if (identifyNetworkPrecision) {
            if (!_network->getPrecision()) {
                _network->setPrecision(lprms.prms.precision);
            }
            if (_network->getPrecision() != lprms.prms.precision) {
                _network->setPrecision(Precision::MIXED);
                identifyNetworkPrecision = false;
            }
        }

        for (const auto& outPort : lprms.outputPorts) {
            const std::string outId = details::stringFormat("%d.%d", lprms.layerId, outPort.portId);
            const std::string outName = lprms.outputPorts.size() == 1 ? lprms.prms.name
                : details::stringFormat("%s.%d", lprms.prms.name.c_str(), outPort.portId);
            DataPtr& ptr = _network->getData(outName.c_str());
            if (!ptr) {
                ptr.reset(new Data(outName, outPort.dims, outPort.precision, TensorDesc::getLayoutByDims(outPort.dims)));
                ptr->setDims(outPort.dims);
            }
            _portsToData[outId] = ptr;

            if (ptr->getCreatorLayer().lock())
                THROW_IE_EXCEPTION << "two layers set to the same output [" << outName << "], conflict at offset "
                                   << node.offset_debug();

            ptr->getCreatorLayer() = layer;
            layer->outData.push_back(ptr);
        }
        nodeCnt++;
    }

    // connect the edges
    pugi::xml_node edges = root.child("edges");

    FOREACH_CHILD(_ec, edges, "edge") {
        int fromLayer = GetIntAttr(_ec, "from-layer");
        int fromPort = GetIntAttr(_ec, "from-port");
        int toLayer = GetIntAttr(_ec, "to-layer");
        int toPort = GetIntAttr(_ec, "to-port");

        auto dataId = details::stringFormat("%d.%d", fromLayer, fromPort);
        auto targetLayer = layerById[toLayer];
        if (!targetLayer)
            THROW_IE_EXCEPTION << "Layer ID " << toLayer << " was not found while connecting edge at offset "
                               << _ec.offset_debug();

        SetLayerInput(*_network, dataId, targetLayer, toPort);
    }

    if (_version == 1) {
        // a hacK: set input to the first layer that is not connected...
        bool inputWasSet = false;
        for (auto& kvp : layerById) {
            CNNLayer::Ptr& layer = kvp.second;
            const LayerParseParameters& parseInfo = layersParseInfo[layer->name];
            size_t inSize = layer->insData.size();
            if (inSize != 0) continue;
            if (parseInfo.inputPorts.size() == 0)
                THROW_IE_EXCEPTION << "Layer " << layer->name << " does not have any input";
            SetLayerInput(*_network, inputData->getName(), layer, parseInfo.inputPorts[0].portId);
            inputWasSet = true;

            // Modification of default input precision which should be used for input blob
            // Q78 needs an I16 otherwise pixels will overflow
            // FP16 requires to pass FP32 as input from user
            Precision inputPrecision;
            inputPrecision = layer->precision == Precision::Q78 ? Precision::I16 :
                layer->precision == Precision::FP16 ? Precision::FP32 : static_cast<Precision::ePrecision>(layer->precision);

            auto inputLayer = std::make_shared<GenericLayer>(LayerParams({inputData->getName(), "input",  inputPrecision}));
            inputLayer->outData.push_back(inputData);
            _network->addLayer(inputLayer);
            inputData->creatorLayer = inputLayer;

            InputsDataMap inputs;
            _network->getInputsInfo(inputs);
            if (inputs.size() != 1) {
                THROW_IE_EXCEPTION << "IR v1 must have one input layer";
            }
            inputs.begin()->second->setInputPrecision(inputPrecision);

            // And we need to leave original input precision unmodified for proper handling in plugin
            inputData->setPrecision(layer->precision);
            break;
        }
        if (!inputWasSet) THROW_IE_EXCEPTION << "network does not have any input layer";
    } else {  // version 2: inputs are marked as input layers
        auto prepareInputLayer = [](vector<CNNLayer::Ptr> layers,
                                    function<void(const InputInfo::Ptr&)> register_input) {
            for (auto inLayer : layers) {
                if (inLayer->outData.size() != 1) {
                    THROW_IE_EXCEPTION << "Input layer must have 1 output.\n"
                        "See documentation for details, "
                        "'Notice On Using Model Optimizer tool' in UseOfTheInferenceEngine.html.\n"
                        "You need to modify prototxt and generate new IR.";
                }
                InputInfo::Ptr info(new InputInfo());
                info->setInputData(*inLayer->outData.begin());
                Precision inputPrecision = info->getInputPrecision();
                if (inputPrecision == Precision::Q78)
                    info->setInputPrecision(Precision::I16);
                if (inputPrecision == Precision::FP16)
                    info->setInputPrecision(Precision::FP32);

                register_input(info);
            }
        };

        prepareInputLayer(inputLayers, [&](const InputInfo::Ptr& info) {
            _network->setInputInfo(info);
        });
    }

    auto statNode = root.child("statistics");
    ParseStatisticSection(statNode);

    if (!_network->allLayers().size())
        THROW_IE_EXCEPTION << "Incorrect model! Network doesn't contain layers.";

    // check all input ports are occupied
    for (const auto& kvp : _network->allLayers()) {
        const CNNLayer::Ptr& layer = kvp.second;
        if (_version) {
            const LayerParseParameters& parseInfo = layersParseInfo[layer->name];
            size_t inSize = layer->insData.size();
            if (inSize != parseInfo.inputPorts.size())
                THROW_IE_EXCEPTION << "Layer " << layer->name << " does not have any edge connected to it";

            for (unsigned i = 0; i < inSize; i++) {
                if (!layer->insData[i].lock()) {
                    THROW_IE_EXCEPTION << "Layer " << layer->name.c_str() << " input port "
                        << parseInfo.inputPorts[i].portId << " is not connected to any data";
                }
            }
        }
        layer->validateLayer();
    }

    if (_version) {
        // parse mean image
        ParsePreProcess(root);
        _network->resolveOutput();

        // Set default output precision to FP32 (for back-compatibility)
        OutputsDataMap outputsInfo;
        _network->getOutputsInfo(outputsInfo);
        for (auto outputInfo : outputsInfo) {
            outputInfo.second->setPrecision(Precision::FP32);
        }

        if (_version == 1) {
            int batchSize = GetIntAttr(root, "batch", 1);
            _network->setBatchSize(batchSize);
        }
    }

    return _network;
}

template<typename BlobType>
inline Blob::Ptr GetTypedBlobFromSegment(const TBlob<uint8_t>::Ptr& weights, const WeightSegment& segment) {
    if (segment.getEnd() > weights->size())
        THROW_IE_EXCEPTION << "metadata is incorrect - segment exceeds given buffer limits. Please validate input data";

    size_t noOfElement = segment.size / sizeof(BlobType);
    // RanC: TODO: IR does not provide me with weight slayout.
    // So far I knew it since I know what layer it is. In generic layers I don't
    // so until the IR will have the layout and sizes I will pass it as vector and the plugin will have to
    // validate and undertand what he should get...
    SizeVector w_dims({noOfElement});

    typename TBlobProxy<BlobType>::Ptr binBlob(new TBlobProxy<BlobType>(segment.precision, Layout::C, weights, segment.start, w_dims));

    /* this validation is not reduntant I have no prior knowledge of the weights anymore...
    if (pbpWeights->byteSize() != lprms.weights.size)
        THROW_IE_EXCEPTION << "bytes size weights for " << pWL->name << " mismatch, expecting "
        << pbpWeights->byteSize() << " bytes which are " << pbpWeights->size() << " elements";
        */
    return binBlob;
}

Blob::Ptr V2FormatParser::GetBlobFromSegment(const TBlob<uint8_t>::Ptr& weights, const WeightSegment& segment) const {
    if (segment.precision == Precision::FP32) {
        return GetTypedBlobFromSegment<float>(weights, segment);
    } else if (segment.precision == Precision::I16 || segment.precision == Precision::Q78 || segment.precision == Precision::FP16) {
        return GetTypedBlobFromSegment<short>(weights, segment);
    } else if (segment.precision == Precision::U8) {
        return GetTypedBlobFromSegment<uint8_t>(weights, segment);
    } else if (segment.precision == Precision::I8) {
        return GetTypedBlobFromSegment<int8_t>(weights, segment);
    } else {
        THROW_IE_EXCEPTION << "precision " << segment.precision << " is not supported...";
    }
}

void V2FormatParser::CopyBlobsByName(void* layerParsePrms, std::string name) {
    auto internalParams = layersParseInfo.find(name);

    LayerParseParameters* params = static_cast<LayerParseParameters *>(layerParsePrms);
    params->blobs = internalParams->second.blobs;
}

void V2FormatParser::SetWeights(const TBlob<uint8_t>::Ptr& weights) {
    for (auto& kvp : _network->allLayers()) {
        auto fit = layersParseInfo.find(kvp.second->name);
        // todo: may check that earlier - while parsing...
        if (fit == layersParseInfo.end())
            THROW_IE_EXCEPTION << "Internal Error: ParseInfo for " << kvp.second->name << " are missing...";
        auto& lprms = fit->second;

        WeightableLayer* pWL = dynamic_cast<WeightableLayer*>(kvp.second.get());
        if (pWL != nullptr) {
            if (lprms.blobs.find("weights") != lprms.blobs.end()) {
                pWL->_weights = GetBlobFromSegment(weights, lprms.blobs["weights"]);
                pWL->blobs["weights"] = pWL->_weights;
            }
            if (lprms.blobs.find("biases") != lprms.blobs.end()) {
                pWL->_biases  = GetBlobFromSegment(weights, lprms.blobs["biases"]);
                pWL->blobs["biases"] = pWL->_biases;
            }
        } else {
            auto pGL = dynamic_cast<GenericLayer *>(kvp.second.get());
            if (pGL == nullptr) continue;
            for (auto s : lprms.blobs) {
                pGL->blobs[s.first] = GetBlobFromSegment(weights, s.second);
            }
        }
    }
    for (auto &kvp : _preProcessSegments) {
        const std::string &inputName = kvp.first;
        auto &segments = kvp.second;
        auto inputInfo = _network->getInput(inputName);
        if (!inputInfo) THROW_IE_EXCEPTION << "Internal error: missing input name " << inputName;

        auto dims = inputInfo->getDims();
        auto width = dims[0];
        auto height = dims[1];

        PreProcessInfo &pp = inputInfo->getPreProcess();

        for (size_t c = 0; c < segments.size(); c++) {
            if (segments[c].size == 0)
                continue;
            Blob::Ptr blob = GetBlobFromSegment(weights, segments[c]);
            blob->Reshape({ width, height }, Layout::HW);  // to fit input image sizes (summing it is an image)
            pp.setMeanImageForChannel(blob, c);
        }
    }
}

void V2FormatParser::ParseDims(SizeVector& dims, const pugi::xml_node &parentNode) const {
    for (auto node = parentNode.child("dim"); !node.empty(); node = node.next_sibling("dim")) {
        unsigned int dim = 0;
        const pugi::char_t* dimVal = node.child_value();
        if (!sscanf(dimVal, "%u", &dim) || dim == 0) {
            THROW_IE_EXCEPTION << "dimension (" << dimVal << ") in node must be a positive integer: at offset "
                << node.offset_debug();
        }
        dims.push_back(dim);
    }

    if (dims.empty()) {
        THROW_IE_EXCEPTION << "input must have dimensions";
    }

    if (_version == 1)
        dims.insert(dims.begin(), 1);  // for batch, in version 1, in version 2 it is already there.
}

DataPtr V2FormatParser::ParseInputData(pugi::xml_node& root) const {
    auto inputNode = root.child("input");
    if (inputNode.empty()) {
        THROW_IE_EXCEPTION << "No input node in network, missing <input>";
    }

    auto inputName = GetStrAttr(inputNode, "name", "input");
    SizeVector inputDims;

    ParseDims(inputDims, inputNode);

    DataPtr& inputData = _network->getData(inputName);
    inputData.reset(new Data(inputName, inputDims, _network->getPrecision(), TensorDesc::getLayoutByDims(inputDims)));
    inputData->setDims(inputDims);
    return inputData;
}

void V2FormatParser::ParsePreProcess(pugi::xml_node& root) {
    /*
    <pre-process mean-precision="FP32">
        <channel id = ”0”>
            <mean value = ”104” / >  // in case of constant
        // or
            <mean offset = "121930449" size = "51529" / >  // in case of array – ref to the .bin file
            <scale value = "1.2">
        </channel>
    </pre-process>
    */

    auto ppNode = root.child("pre-process");
    if (ppNode.empty()) {
        return;
    }
    // find out to what input this belongs to
    std::string inputName;
    InputInfo::Ptr preProcessInput;

    inputName = GetStrAttr(ppNode, "reference-layer-name", "");
    inputName = trim(inputName);
    if (inputName.empty()) {
        // fallback (old format), look for the picture in the inputs
        InputsDataMap inputs;
        _network->getInputsInfo(inputs);

        if (inputs.empty()) THROW_IE_EXCEPTION << "network has no input";

        for (auto i : inputs) {
            if (i.second->getDims().size() == 4) {
                preProcessInput = i.second;
                break;
            }
        }
        if (!preProcessInput) {
            preProcessInput = inputs.begin()->second;
        }

        inputName = preProcessInput->name();
    } else {
        preProcessInput = _network->getInput(inputName);
        if (!preProcessInput)
            THROW_IE_EXCEPTION << "pre-process name ref '" << inputName << "' refers to un-existing input";
    }

    // dims vector without batch size
    SizeVector inputDims = preProcessInput->getDims();

    if (inputDims.size() < 2)
        THROW_IE_EXCEPTION << "network did not define input dimensions properly";
    size_t noOfChannels = inputDims[inputDims.size() - 2];
    size_t width = inputDims[0];
    size_t height = inputDims[1];

    PreProcessInfo &pp = preProcessInput->getPreProcess();
    std::vector<WeightSegment> &segments = _preProcessSegments[inputName];

    pp.init(noOfChannels);

    segments.resize(noOfChannels);

    auto meanSegmentPrecision = GetPrecisionAttr(ppNode, "mean-precision", Precision::UNSPECIFIED);

    ResponseDesc resp;
    InferenceEngine::PreProcessChannel::Ptr preProcessChannel;

    int lastChanNo = -1;
    std::unordered_set<int> idsForMeanValue;
    std::unordered_set<int> idsForMeanImage;

    FOREACH_CHILD(chan, ppNode, "channel") {
        int chanNo = GetIntAttr(chan, "id", lastChanNo + 1);
        if (chanNo >= static_cast<int>(noOfChannels) || chanNo < 0) {
            THROW_IE_EXCEPTION << "Pre-process channel id invalid: " << chanNo;
        }
        lastChanNo = chanNo;
        preProcessChannel = pp[chanNo];
        WeightSegment& preProcessSegment = segments[chanNo];

        auto meanNode = chan.child("mean");
        if (!meanNode.empty()) {
            if (!meanNode.attribute("value") && (!meanNode.attribute("size"))) {
                THROW_IE_EXCEPTION << "mean should have at least one of the following attribute: value, size";
            }
            if (meanNode.attribute("value")) {
                preProcessChannel->meanValue = GetFloatAttr(meanNode, "value");
                idsForMeanValue.insert(chanNo);
            }
            if (meanNode.attribute("size")) {
                idsForMeanImage.insert(chanNo);
                preProcessSegment.size = static_cast<size_t>(GetIntAttr(meanNode, "size"));
                preProcessSegment.start = static_cast<size_t>(GetIntAttr(meanNode, "offset"));
                preProcessSegment.precision = meanSegmentPrecision;
                if (width*height*meanSegmentPrecision.size() != preProcessSegment.size) {
                    THROW_IE_EXCEPTION << "mean blob size mismatch expected input, got: "
                                       << preProcessSegment.size << " extpecting " << width
                                       << " x " << height << " x " << meanSegmentPrecision.size();
                }
                if (!meanSegmentPrecision || meanSegmentPrecision == Precision::MIXED)
                    THROW_IE_EXCEPTION << "mean blob defined without specifying precision.";
            }
        }
        auto scaleNode = chan.child("scale");
        if (!scaleNode.empty() && scaleNode.attribute("value")) {
            preProcessChannel->stdScale = GetFloatAttr(scaleNode, "value");
        }
    }

    if (idsForMeanImage.size() == noOfChannels) {
        pp.setVariant(MEAN_IMAGE);
    } else if (idsForMeanValue.size() == noOfChannels) {
        pp.setVariant(MEAN_VALUE);
    } else if ((idsForMeanImage.size() == 0) && (idsForMeanValue.size() == 0)) {
        pp.setVariant(NONE);
    } else {
        std::string validMeanValuesIds = "";
        std::string validMeanImageIds = "";
        for (auto id : idsForMeanValue) { validMeanValuesIds += std::to_string(id) + " "; }
        for (auto id : idsForMeanImage) { validMeanImageIds += std::to_string(id) + " "; }
        THROW_IE_EXCEPTION << "mean is not provided for all channels\n"
                "Provided mean values for : " << validMeanValuesIds << "\n"
                                   "Provided mean image for: " << validMeanImageIds;
    }
}

const std::vector<std::shared_ptr<BaseCreator> >& V2FormatParser::getCreators() const {
    // there should be unique_ptr but it cant be used with initializer lists
    static std::vector<std::shared_ptr<BaseCreator> > creators = {
        std::make_shared<V2LayerCreator<PowerLayer>>("Power"),
        std::make_shared<V2LayerCreator<ConvolutionLayer>>("Convolution"),
        std::make_shared<V2LayerCreator<DeconvolutionLayer>>("Deconvolution"),
        std::make_shared<V2LayerCreator<PoolingLayer>>("Pooling"),
        std::make_shared<V2LayerCreator<FullyConnectedLayer>>("InnerProduct"),
        std::make_shared<V2LayerCreator<FullyConnectedLayer>>("FullyConnected"),
        std::make_shared<V2LayerCreator<NormLayer>>("LRN"),
        std::make_shared<V2LayerCreator<NormLayer>>("Norm"),
        std::make_shared<V2LayerCreator<SoftMaxLayer>>("Softmax"),
        std::make_shared<V2LayerCreator<GRNLayer>>("GRN"),
        std::make_shared<V2LayerCreator<MVNLayer>>("MVN"),
        std::make_shared<V2LayerCreator<RNNLayer>>("RNN"),
        std::make_shared<V2LayerCreator<LSTMCell>>("LSTMCell"),
        std::make_shared<V2LayerCreator<ReLULayer>>("ReLU"),
        std::make_shared<V2LayerCreator<ClampLayer>>("Clamp"),
        std::make_shared<V2LayerCreator<SplitLayer>>("Split"),
        std::make_shared<V2LayerCreator<SplitLayer>>("Slice"),
        std::make_shared<V2LayerCreator<ConcatLayer>>("Concat"),
        std::make_shared<V2LayerCreator<EltwiseLayer>>("Eltwise"),
        std::make_shared<V2LayerCreator<ScaleShiftLayer>>("ScaleShift"),
        std::make_shared<V2LayerCreator<PReLULayer>>("PReLU"),
        std::make_shared<V2LayerCreator<CropLayer>>("Crop"),
        std::make_shared<V2LayerCreator<ReshapeLayer>>("Reshape"),
        std::make_shared<V2LayerCreator<ReshapeLayer>>("Flatten"),
        std::make_shared<V2LayerCreator<TileLayer>>("Tile"),
        std::make_shared<ActivationLayerCreator>("Activation"),
        std::make_shared<V2LayerCreator<BatchNormalizationLayer>>("BatchNormalization"),
        std::make_shared<TILayerCreator>("TensorIterator"),
    };
    return creators;
}

void V2FormatParser::ParseStatisticSection(const pugi::xml_node& statNode) {
    auto splitParseCommas = [&](const string& s) ->vector<float> {
        vector<float> res;
        stringstream ss(s);

        float val;

        while (ss >> val) {
            res.push_back(val);

            if (ss.peek() == ',')
                ss.ignore();
        }

        return res;
    };

    map<string, NetworkNodeStatsPtr> newNetNodesStats;

    for (auto layer : statNode.children("layer")) {
        NetworkNodeStatsPtr nodeStats = NetworkNodeStatsPtr(new NetworkNodeStats());

        string name = layer.child("name").text().get();

        newNetNodesStats[name] = nodeStats;

        nodeStats->_minOutputs = splitParseCommas(layer.child("min").text().get());
        nodeStats->_maxOutputs = splitParseCommas(layer.child("max").text().get());
    }

    ICNNNetworkStats *pstats = nullptr;
    StatusCode s = _network->getStats(&pstats, nullptr);
    if (s == StatusCode::OK && pstats) {
        pstats->setNodesStats(newNetNodesStats);
    }
}

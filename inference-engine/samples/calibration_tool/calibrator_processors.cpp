// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "calibrator_processors.h"
#include <string>       // std::string
#include <iostream>     // std::cout
#include <sstream>      // std::stringstream
#include <iomanip>
#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <list>
#include "details/ie_cnn_network_tools.h"
#include "details/caseless.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

using InferenceEngine::details::InferenceEngineException;

CNNLayerPtr Int8Calibrator::addScaleShiftBeforeLayer(std::string name, CNNLayer::Ptr beforeLayer, size_t port, std::vector<float> scale) {
    if (beforeLayer->insData.size() < port) {
        THROW_IE_EXCEPTION << "cannot find appropraite port for addScaleShiftBeforeLayer";
    }

    DataPtr pData = beforeLayer->insData[port].lock();
    LayerParams params;
    params.name = name;
    params.precision = Precision::FP32;
    params.type = "ScaleShift";
    CNNLayerPtr lptr = std::make_shared<ScaleShiftLayer>(params);
    ScaleShiftLayer *pScaleShift = dynamic_cast<ScaleShiftLayer *>(lptr.get());

    SizeVector wdims({ pData->dims[2] });

    if (scale.size() == 1) {
        scale.resize(wdims[0]);
        for (int i = 1; i < wdims[0]; i++) {
            scale[i] = scale[0];
        }
    }

    if (scale.size() != pData->dims[2]) {
        THROW_IE_EXCEPTION << "Failed to add scaleshift before " << beforeLayer->name << " due to scales and layer output dims incossitency";
    }

    Blob::Ptr weights = nullptr;
    weights = make_shared_blob<float>(Precision::FP32, Layout::C, wdims);
    weights->allocate();
    float *buffer = weights->buffer().as<float *>();
    if (buffer == nullptr) {
        THROW_IE_EXCEPTION << "Could not allocate weights buffer";
    }
    for (size_t i = 0, idx = 0; i < pData->dims[2]; i++) {
        buffer[i] = scale[i];
    }
    pScaleShift->_weights = weights;


    SizeVector bdims({ pData->dims[2] });
    Blob::Ptr biases = nullptr;
    biases = make_shared_blob<float>(Precision::FP32, Layout::C, bdims);
    biases->allocate();
    buffer = biases->buffer().as<float *>();
    for (size_t i = 0, idx = 0; i < pData->dims[2]; i++) {
        buffer[i] = 0.f;
    }
    pScaleShift->_biases = biases;

    Data *edge2 = new Data(*pData.get());
    DataPtr newEdge(edge2);
    lptr->insData.push_back(pData);
    lptr->outData.push_back(newEdge);
    newEdge->name = /*"EdgeAfter_" +*/ params.name;
    newEdge->creatorLayer = lptr;
    newEdge->inputTo.clear();
    newEdge->inputTo[beforeLayer->name] = beforeLayer;

    pData->inputTo.erase(beforeLayer->name);
    pData->inputTo[params.name] = lptr;

    for (size_t i = 0; i < beforeLayer->insData.size(); i++) {
        DataPtr d = beforeLayer->insData[i].lock();
        if (d == pData) {
            beforeLayer->insData[i] = newEdge;
            break;
        }
    }
    return lptr;
}


float Int8Calibrator::compare_NRMSD(InferenceEngine::Blob::Ptr res, InferenceEngine::Blob::Ptr ref) {
    float *res_ptr = res->buffer().as<float *>();
    size_t res_size = res->size();

    float *ref_ptr = ref->buffer().as<float *>();
    size_t ref_size = ref->size();

    float sum = 0;

    float mmin = ref_ptr[0], mmax = ref_ptr[0];

    for (size_t i = 0; i < ref_size; i++) {
        float sqr = (ref_ptr[i] - res_ptr[i]);
        sqr *= sqr;
        sum += sqr;

        mmin = std::min(mmin, ref_ptr[i]);
        mmax = std::max(mmax, ref_ptr[i]);
    }
    sum /= ref_size;

    sum = pow(sum, 0.5);

    sum /= mmax - mmin;

    return sum;
}


InferenceEngine::NetworkStatsMap Int8Calibrator::getStatistic(float threshold) {
    InferenceEngine::NetworkStatsMap netNodesStats;
    // go over all outputs and get aggregated statistics
    for (auto l : _statData.registeredLayers()) {
        NetworkNodeStatsPtr nodeStats;
        size_t channels = _statData.getNumberChannels(l);
        if (netNodesStats.find(l) == netNodesStats.end()) {
            nodeStats = NetworkNodeStatsPtr(new NetworkNodeStats(channels));

            netNodesStats[l] = nodeStats;
        } else {
            nodeStats = netNodesStats[l];
        }
        for (size_t c = 0; c < channels; c++) {
            _statData.getDataMinMax(l, c, nodeStats->_minOutputs[c], nodeStats->_maxOutputs[c], threshold);
        }
    }
    return netNodesStats;
}


void Int8Calibrator::collectFP32Statistic() {
    _collectByLayer = false;
    _collectStatistic = true;

    networkReaderC = InferenceEngine::CNNNetReader();
    networkReaderC.ReadNetwork(_modelFileNameI8C);
    if (!networkReaderC.isParseSuccess()) THROW_IE_EXCEPTION << "cannot load a failed Model";
    if (_cBatch == 0) {
        // Zero means "take batch value from the IR"
        _cBatch = networkReaderC.getNetwork().getBatchSize();
    } else {
        // Not zero means "use the specified value"
        networkReaderC.getNetwork().setBatchSize(_cBatch);
    }

    /** Extract model name and load weights **/
    std::string binFileName = fileNameNoExt(_modelFileNameI8C) + ".bin";
    networkReaderC.ReadWeights(binFileName.c_str());

    auto network = networkReaderC.getNetwork();


    std::vector<CNNLayerPtr> layersAfterInputs;

    std::string hackPrefix = "scaleshifted_input:";

    for (auto &&layer : network) {
        if (layer->insData.size() > 0) {
            std::string inName = layer->input()->getName();
            for (auto &&input : network.getInputsInfo()) {
                if (inName == input.first) {
                    layersAfterInputs.push_back(layer);
                    _inputsFromLayers[hackPrefix + layer->name] = inName;
                }
            }
        }
    }

    for (auto &&layer : layersAfterInputs) {
        std::string firstInputName = hackPrefix + layer->name;
        auto scaleShiftLayer = addScaleShiftBeforeLayer(firstInputName, layer, 0, { 1.f });
        ((ICNNNetwork&)network).addLayer(scaleShiftLayer);
    }


    // 1. add all layers as output one
    for (auto &&layer : network) {
        std::string layerType = network.getLayerByName(layer->name.c_str())->type;
        if (/*layerType != "Split" &&*/layerType != "Input") {
            network.addOutput(layer->name);
        }
        _statData.registerLayer(layer->name);
    }

    ExecutableNetwork executable_network = _pluginI8C.LoadNetwork(network, { { CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES) } });
    _inferRequestI8C = executable_network.CreateInferRequest();
}

void Int8Calibrator::validateInt8Config(const InferenceEngine::NetworkStatsMap &stat,
                                        const std::map<std::string, bool> &layersToInt8) {
    _collectByLayer = false;
    _collectStatistic = false;
    networkReaderC = InferenceEngine::CNNNetReader();
    networkReaderC.ReadNetwork(_modelFileNameI8C);
    if (!networkReaderC.isParseSuccess()) THROW_IE_EXCEPTION << "cannot load a failed Model";
    if (_cBatch == 0) {
        // Zero means "take batch value from the IR"
        _cBatch = networkReaderC.getNetwork().getBatchSize();
    } else {
        // Not zero means "use the specified value"
        networkReaderC.getNetwork().setBatchSize(_cBatch);
    }

    /** Extract model name and load weights **/
    std::string binFileName = fileNameNoExt(_modelFileNameI8C) + ".bin";
    networkReaderC.ReadWeights(binFileName.c_str());

    // Initialize statistic
    ICNNNetworkStats *pstats = nullptr;
    StatusCode s = ((ICNNNetwork&)networkReaderC.getNetwork()).getStats(&pstats, nullptr);
    if (s == StatusCode::OK && pstats) {
        pstats->setNodesStats(stat);
    }

    auto network = networkReaderC.getNetwork();
    for (auto l : layersToInt8) {
        network.getLayerByName(l.first.c_str())->
            params["quantization_level"] = (l.second == false) ? "FP32" : "I8";
    }

    ExecutableNetwork executable_network = _pluginI8C.LoadNetwork(network, { { CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES) } });
    _inferRequestI8C = executable_network.CreateInferRequest();
}

CNNNetwork Int8Calibrator::createICNNNetworkForLayer(CNNLayer::Ptr layerToClone) {
    CNNLayer::Ptr layerRelU = layerToClone->outData[0]->inputTo.begin()->second;

    InferenceEngine::CNNNetReader reader1;
    std::string inpuitName = layerToClone->insData[0].lock()->name;
    std::string model = "<net name=\"L\" version=\"2\" batch=\"1\"><layers> "        \
        "<layer name=\"" +
        inpuitName +
        "\" type=\"Input\" precision=\"FP32\" id=\"0\"> "\
        "<output>"\
        "<port id=\"0\">"\
        "<dim>1</dim>"\
        "<dim>3</dim>"\
        "<dim>224</dim>"\
        "<dim>224</dim>"\
        "</port>"\
        "</output>"\
        "</layer>" \
        "<layer name=\"" +
        layerToClone->name +
        "\" type=\"Convolution\" precision=\"FP32\" id=\"1\">" \
        "<convolution_data stride-x=\"2\" stride-y=\"2\" pad-x=\"3\" pad-y=\"3\" kernel-x=\"7\" kernel-y=\"7\" output=\"64\" group=\"1\" />"\
        "<input>"\
        "<port id=\"1\">"\
        "<dim>1</dim>"\
        "<dim>3</dim>"\
        "<dim>224</dim>"\
        "<dim>224</dim>"\
        "</port>"\
        "</input>"\
        "<output>"\
        "<port id=\"2\">"\
        "<dim>1</dim>"\
        "<dim>64</dim>"\
        "<dim>112</dim>"\
        "<dim>112</dim>"\
        "</port>"\
        "</output>"\
        "</layer>"\
        "<layer name=\"" +
        layerRelU->name +
        "\" type=\"ReLU\" precision=\"FP32\" id=\"2\">"\
        "<input>"
        "<port id=\"3\">"\
        "<dim>1</dim>"\
        "<dim>64</dim>"\
        "<dim>112</dim>"\
        "<dim>112</dim>"\
        "</port>"\
        "</input>"\
        "<output>"\
        "<port id=\"4\">"\
        "<dim>1</dim>"\
        "<dim>64</dim>"\
        "<dim>112</dim>"\
        "<dim>112</dim>"\
        "</port>"\
        "</output>"\
        "</layer>"\
        "<layer name=\"" +
        layerToClone->name +
        "_\" type=\"ScaleShift\" precision=\"FP32\" id=\"3\">"\
        "<input>"
        "<port id=\"5\">"\
        "<dim>1</dim>"\
        "<dim>64</dim>"\
        "<dim>112</dim>"\
        "<dim>112</dim>"\
        "</port>"\
        "</input>"\
        "<output>"\
        "<port id=\"6\">"\
        "<dim>1</dim>"\
        "<dim>64</dim>"\
        "<dim>112</dim>"\
        "<dim>112</dim>"\
        "</port>"\
        "</output>"\
        "</layer>"\
        "</layers> <edges>"\
        "<edge from-layer=\"0\" from-port=\"0\" to-layer=\"1\" to-port=\"1\"/> "\
        "<edge from-layer=\"1\" from-port=\"2\" to-layer=\"2\" to-port=\"3\"/> "\
        "<edge from-layer=\"2\" from-port=\"4\" to-layer=\"3\" to-port=\"5\"/> "\
        "</edges></net>";

    reader1.ReadNetwork(model.c_str(), model.length());
    ICNNNetwork &n = reader1.getNetwork();

    InferenceEngine::InputsDataMap inputs;
    n.getInputsInfo(inputs);
    CNNLayerPtr inputLayer = inputs.begin()->second->getInputData()->creatorLayer.lock();

    CNNLayerPtr convLayer;
    n.getLayerByName(layerToClone->name.c_str(), convLayer, nullptr);
    ConvolutionLayer *pConvS = dynamic_cast<ConvolutionLayer *>(layerToClone.get());
    ConvolutionLayer *pConvT = dynamic_cast<ConvolutionLayer *>(convLayer.get());
    pConvT->_kernel_x = pConvS->_kernel_x;
    pConvT->_kernel_y = pConvS->_kernel_y;
    pConvT->_stride_x = pConvS->_stride_x;
    pConvT->_stride_y = pConvS->_stride_y;
    pConvT->_out_depth = pConvS->_out_depth;
    pConvT->_padding_x = pConvS->_padding_x;
    pConvT->_padding_y = pConvS->_padding_y;
    pConvT->_dilation_x = pConvS->_dilation_x;
    pConvT->_dilation_y = pConvS->_dilation_y;
    pConvT->_group = pConvS->_group;
    pConvT->_weights = pConvS->_weights;
    pConvT->_biases = pConvS->_biases;
    pConvT->blobs = pConvS->blobs;

    std::shared_ptr<Data> cur = layerToClone->insData[0].lock();
    if (cur == nullptr) {
        THROW_IE_EXCEPTION << "[Samples] shared ptr layerToClone->insData[0].lock() return nullptr";
    }
    DataPtr inputEdge = std::make_shared<Data>(*cur.get());

    inputEdge->getInputTo().clear();
    inputEdge->name = inpuitName;
    inputEdge->creatorLayer = inputLayer;
    inputEdge->inputTo[layerToClone->name] = convLayer;
    inputEdge->getInputTo().clear();
    inputEdge->inputTo[layerToClone->name] = convLayer;

    inputs.begin()->second->setInputData(inputEdge);

    convLayer->insData.clear();
    convLayer->insData.push_back(inputEdge);

    inputLayer->outData.clear();
    inputLayer->outData.push_back(inputEdge);

    DataPtr convEdge = std::make_shared<Data>(*layerToClone->outData[0].get());
    convEdge->getInputTo().clear();
    convEdge->creatorLayer = convLayer;
    convEdge->name = convLayer->name;
    convLayer->outData.clear();
    convLayer->outData.push_back(convEdge);

    CNNLayerPtr reluLayer;
    n.getLayerByName(layerRelU->name.c_str(), reluLayer, nullptr);
    DataPtr reluEdge = std::make_shared<Data>(*layerRelU->outData[0].get());
    reluEdge->getInputTo().clear();
    reluEdge->creatorLayer = reluLayer;
    reluEdge->name = reluLayer->name;
    reluLayer->insData.clear();
    reluLayer->insData.push_back(convEdge);
    reluLayer->outData.clear();
    reluLayer->outData.push_back(reluEdge);

    convEdge->inputTo[reluLayer->name] = reluLayer;

    CNNLayerPtr ssLayer;
    std::string ssLayerName = convLayer->name + "_";
    n.getLayerByName(ssLayerName.c_str(), ssLayer, nullptr);
    DataPtr ssEdge = std::make_shared<Data>(*layerRelU->outData[0].get());
    ssEdge->getInputTo().clear();
    ssEdge->creatorLayer = ssLayer;
    ssEdge->name = ssLayer->name;
    ssLayer->insData.clear();
    ssLayer->insData.push_back(reluEdge);
    ssLayer->outData.clear();
    ssLayer->outData.push_back(ssEdge);

    reluEdge->inputTo[ssLayer->name] = ssLayer;

    n.addOutput(ssLayer->name);

    // filling weights and biases
    size_t channels = ssEdge->getTensorDesc().getDims()[1];
    Blob::Ptr weights = nullptr;
    SizeVector wdims;
    wdims.push_back(channels);
    weights = make_shared_blob<float, const SizeVector>(Precision::FP32, Layout::C, wdims);
    weights->allocate();
    float *dataw = weights->buffer().as<float *>();
    for (size_t i = 0; i < channels; i++) {
        dataw[i] = 1.0f;
    }
    ssLayer->blobs["weights"] = weights;

    Blob::Ptr biases = nullptr;
    SizeVector bdims;
    bdims.push_back(channels);
    biases = make_shared_blob<float, const SizeVector>(Precision::FP32, Layout::C, bdims);
    biases->allocate();
    float *datab = biases->buffer().as<float *>();
    for (size_t i = 0; i < channels; i++) {
        datab[i] = 0.0f;
    }
    ssLayer->blobs["biases"] = biases;

    auto wss = dynamic_cast<WeightableLayer*>(ssLayer.get());
    wss->_weights = weights;
    wss->_biases = biases;

    return reader1.getNetwork();
}

void Int8Calibrator::collectByLayerStatistic(const InferenceEngine::NetworkStatsMap &stat) {
    _collectByLayer = true;
    _collectStatistic = false;
    networkReaderC = InferenceEngine::CNNNetReader();
    networkReaderC.ReadNetwork(_modelFileNameI8C);
    if (!networkReaderC.isParseSuccess()) THROW_IE_EXCEPTION << "cannot load a failed Model";
    if (_cBatch != 0) {
        networkReaderC.getNetwork().setBatchSize(_cBatch);
    }

    /** Extract model name and load weights **/
    std::string binFileName = fileNameNoExt(_modelFileNameI8C) + ".bin";
    networkReaderC.ReadWeights(binFileName.c_str());

    auto network = networkReaderC.getNetwork();
    // 1. add all layers as output one
    for (auto &&layer : network) {
        std::string layerType = network.getLayerByName(layer->name.c_str())->type;
        if (/*layerType != "Split" &&*/layerType != "Input") {
            network.addOutput(layer->name);
        }

        if (layerType == "Convolution") {
            _layersAccuracyDrop[layer->name] = 0.f;
        }
    }

    ExecutableNetwork executable_network = _pluginI8C.LoadNetwork(network, { { CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES) } });
    _inferRequestI8C = executable_network.CreateInferRequest();

    // 2. go over all layers which affect accuracy and create network basing on it
    for (auto l : _layersAccuracyDrop) {
        CNNLayerPtr layerToClone = network.getLayerByName(l.first.c_str());
        CNNLayerPtr layerRelU = nullptr;
        // verification if there is Conv-RELU patern
        // currently it is only supported

        // if only one output from conv and if it is an output to relu
        bool quattization = false;
        if (layerToClone->outData.size() == 1 && layerToClone->outData[0]->inputTo.size() == 1) {
            layerRelU = layerToClone->outData[0]->inputTo.begin()->second;
            if (layerRelU->type == "ReLU") {
                quattization = true;
            }
        }

        if (quattization) {
            CNNNetwork n = createICNNNetworkForLayer(layerToClone);
            if (_cBatch != 0) {
                n.setBatchSize(_cBatch);
            }

            // Initialize statistic
            ICNNNetworkStats *pstats = nullptr;
            ICNNNetwork &in = n;
            StatusCode s = in.getStats(&pstats, nullptr);
            if (s == StatusCode::OK && pstats) {
                pstats->setNodesStats(stat);
            }

            InferenceEngine::InputsDataMap inputs = n.getInputsInfo();
            DataPtr q = inputs.begin()->second->getInputData();

            ExecutableNetwork enetwork = _pluginI8C.LoadNetwork(n, { { CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES) } });
            _singleLayerNetworks.push_back(enetwork);
            InferenceEngine::InferRequest request = enetwork.CreateInferRequest();
            std::string inpuitName = layerToClone->insData[0].lock()->name;
            request.SetBlob(inpuitName, _inferRequestI8C.GetBlob(inpuitName));
            _singleLayerRequests[layerToClone->name] = { request, layerRelU->name, layerToClone->name };
        }
    }
}


void Int8Calibrator::collectCalibrationStatistic() {
    if (_collectByLayer) {
        std::map<std::string, SingleLayerData>::iterator it = _singleLayerRequests.begin();
        while (it != _singleLayerRequests.end()) {
            it->second._request.Infer();
            Blob::Ptr expected = _inferRequestI8C.GetBlob(it->second._outputName);
            std::string i8Out = it->second._outputI8Name + "_";
            Blob::Ptr result = it->second._request.GetBlob(i8Out.c_str());
            float diff = compare_NRMSD(result, expected);
            it->second._int8Accuracy.push_back(diff);
            it++;
        }
    }
    if (_collectStatistic) {
        for (auto l : _statData.registeredLayers()) {
            auto outBlob = _inferRequestI8C.GetBlob(l);

            std::string outName = l;
            if (_inputsFromLayers.find(l) != _inputsFromLayers.end()) {
                outName = _inputsFromLayers[l];
            }

            size_t N, C, statCount;
            if (outBlob->dims().size() == 4 && outBlob->layout() == Layout::NCHW) {
                N = outBlob->dims()[3];
                C = outBlob->dims()[2];
                statCount = C;
            } else if (outBlob->dims().size() == 2 && outBlob->layout() == Layout::NC) {
                N = outBlob->dims()[1];
                C = outBlob->dims()[0];
                statCount = 1;
            } else {
                continue;
            }

            // Counting min/max outputs per channel
            for (size_t n = 0; n < N; n++) {
                if (outBlob->dims().size() == 4) {
                    size_t _HW = outBlob->dims()[0] * outBlob->dims()[1];
                    for (size_t c = 0; c < C; c++) {
                        if (outBlob->getTensorDesc().getPrecision() == Precision::FP32) {
                            float *ptr = &outBlob->buffer().as<float *>()[(n * C + c) * _HW];
                            _statData.addTensorStatistics(outName, c, ptr, _HW);
                        } else if (outBlob->getTensorDesc().getPrecision() == Precision::U8) {
                            uint8_t *ptr = &outBlob->buffer().as<uint8_t *>()[(n * C + c) * _HW];
                            _statData.addTensorStatistics(outName, c, ptr, _HW);
                        } else {
                            throw std::logic_error(std::string("Unsupported precision: ") + outBlob->getTensorDesc().getPrecision().name());
                        }
                    }
                } else if (outBlob->dims().size() == 2) {
                    if (outBlob->getTensorDesc().getPrecision() == Precision::FP32) {
                        float *ptr = &outBlob->buffer().as<float *>()[n * C];
                        _statData.addTensorStatistics(outName, 0, ptr, C);
                    } else if (outBlob->getTensorDesc().getPrecision() == Precision::U8) {
                        uint8_t *ptr = &outBlob->buffer().as<uint8_t *>()[n * C];
                        _statData.addTensorStatistics(outName, 0, ptr, C);
                    } else {
                        throw std::logic_error(std::string("Unsupported precision: ") + outBlob->getTensorDesc().getPrecision().name());
                    }
                }
            }
        }
    }
}

void Int8Calibrator::calculateLayersAccuracyDrop() {
    _layersAccuracyDrop.clear();

    std::map<std::string, SingleLayerData>::iterator it = _singleLayerRequests.begin();
    while (it != _singleLayerRequests.end()) {
        // calculate average metric per layer over all images and sort in desc order
        float mo = 0.f;
        for (auto d : it->second._int8Accuracy) {
            mo += d;
        }
        mo = mo / it->second._int8Accuracy.size();
        _layersAccuracyDrop[it->first] = mo;
        it++;
    }

    // correction of accuracy drop to have sorted values for cases when accuracy drop is equal
    // correction is added according to topological order
    // this will prioritize returning of layers to FP32 starting from layers closer to the end of network
    std::vector<CNNLayerPtr> ordered = InferenceEngine::details::CNNNetSortTopologically(networkReaderC.getNetwork());
    float c = 0.00001f;
    for (auto l : ordered) {
        auto it = _layersAccuracyDrop.find(l->name);
        if (it != _layersAccuracyDrop.end()) {
            it->second += c;
        }
        c += 0.00001f;
    }
    _singleLayerRequests.clear();
}

std::map<std::string, float> Int8Calibrator::layersAccuracyDrop() {
    return _layersAccuracyDrop;
}



//--------------------------------------------------------------------------------------------------

ClassificationCalibrator::ClassificationCalibrator(int nPictures, const std::string &flags_m,
                                                   const std::string &flags_d, const std::string &flags_i,
                                                   int flags_b, InferenceEngine::InferencePlugin plugin,
                                                   CsvDumper &dumper, const std::string &flags_l,
                                                     PreprocessingOptions preprocessingOptions, bool zeroBackground) :
    ClassificationProcessor(flags_m, flags_d, flags_i, flags_b,
                            plugin, dumper, flags_l,
                            preprocessingOptions, zeroBackground) {
    _modelFileNameI8C = modelFileName;
    _pluginI8C = plugin;
    _nPictures = nPictures;
    _cBatch = flags_b;
}

shared_ptr<Processor::InferenceMetrics> ClassificationCalibrator::Process() {
    inferRequest = _inferRequestI8C;
    int top1Result = 0, total = 0;

    ClassificationSetGenerator generator;

    auto validationMap = generator.getValidationMap(imagesPath);
    ImageDecoder decoder;

    // ----------------------------Do inference-------------------------------------------------------------
    std::vector<int> expected(batch);
    std::vector<std::string> files(batch);
    int captured = 0;

    if (!_nPictures) {
        _nPictures = validationMap.size();
    }


    ConsoleProgress progress(_nPictures);

    CalibrationMetrics im;

    std::string firstInputName = this->inputInfo.begin()->first;
    std::string firstOutputName = this->outInfo.begin()->first;
    auto firstInputBlob = inferRequest.GetBlob(firstInputName);
    auto firstOutputBlob = inferRequest.GetBlob(firstOutputName);

    size_t ipics = 0;
    auto iter = validationMap.begin();
    while (iter != validationMap.end() && ipics < _nPictures) {
        int b = 0;
        int filesWatched = 0;
        for (; b < batch && iter != validationMap.end() && ipics + b < _nPictures ; b++, iter++, filesWatched++) {
            expected[b] = iter->first;
            try {
                decoder.insertIntoBlob(iter->second, b, *firstInputBlob, preprocessingOptions);
                files[b] = iter->second;
            } catch (const InferenceEngineException &iex) {
                slog::warn << "Can't read file " << iter->second << slog::endl;
                // Could be some non-image file in directory
                b--;
                continue;
            }
        }
        ipics += batch;

        Infer(progress, filesWatched, im);
        collectCalibrationStatistic();

        std::vector<unsigned> results;
        auto firstOutputData = firstOutputBlob->buffer().as<PrecisionTrait<Precision::FP32>::value_type *>();
        InferenceEngine::TopResults(1, *firstOutputBlob, results);

        for (int i = 0; i < b; i++) {
            int expc = expected[i];
            if (zeroBackground) expc++;
            bool top1Scored = (results[i] == expc);
            if (top1Scored) top1Result++;
            total++;
        }
    }
    progress.finish();

    calculateLayersAccuracyDrop();

    im.AccuracyResult = static_cast<float>(top1Result) / static_cast<float>(total);

    return std::shared_ptr<Processor::InferenceMetrics>(new CalibrationMetrics(im));
}

//--------------------------------------------------------------------------------------------------
SSDObjectDetectionCalibrator::SSDObjectDetectionCalibrator(int nPictures, const std::string &flags_m,
                                                           const std::string &flags_d, const std::string &flags_i,
                                                           const std::string &subdir, int flags_b,
                                                             double threshold,
                                                             InferencePlugin plugin, CsvDumper &dumper,
                                                             const std::string &flags_a, const std::string &classes_list_file) :
    SSDObjectDetectionProcessor(flags_m, flags_d, flags_i, subdir, flags_b,
                                  threshold,
                                  plugin, dumper,
                                  flags_a, classes_list_file) {
    _modelFileNameI8C = modelFileName;
    _pluginI8C = plugin;
    _nPictures = nPictures;
}

shared_ptr<Processor::InferenceMetrics> SSDObjectDetectionCalibrator::Process() {
    inferRequest = _inferRequestI8C;

    // Parsing PASCAL VOC2012 format
    VOCAnnotationParser vocAnnParser;
    VOCAnnotationCollector annCollector(annotationsPath);

    if (annCollector.annotations().size() == 0) {
        ObjectDetectionInferenceMetrics emptyIM(this->threshold);

        return std::shared_ptr<InferenceMetrics>(new ObjectDetectionInferenceMetrics(emptyIM));
    }

    // Getting desired results from annotations
    std::map<std::string, ImageDescription> desiredForFiles;

    for (auto &ann : annCollector.annotations()) {
        std::list<DetectedObject> dobList;
        for (auto &obj : ann.objects) {
            DetectedObject dob(classes[obj.name], obj.bndbox.xmin, obj.bndbox.ymin, obj.bndbox.xmax, obj.bndbox.ymax, 1.0, obj.difficult != 0);
            dobList.push_back(dob);
        }
        ImageDescription id(dobList);
        desiredForFiles.insert(std::pair<std::string, ImageDescription>(ann.folder + "/" + (!subdir.empty() ? subdir + "/" : "") + ann.filename, id));
    }


    ImageDecoder decoder;

    const int maxProposalCount = outputDims[1];
    const int objectSize = outputDims[0];

    for (auto &item : outInfo) {
        DataPtr outputData = item.second;
        if (!outputData) {
            throw std::logic_error("output data pointer is not valid");
        }
    }
    // -----------------------------------------------------------------------------------------------------

    // ----------------------------Do inference-------------------------------------------------------------

    std::vector<VOCAnnotation> expected(batch);

    if (!_nPictures) {
        _nPictures = annCollector.annotations().size();
    }

    ConsoleProgress progress(_nPictures);

    ObjectDetectionInferenceMetrics im(threshold);

    vector<VOCAnnotation>::const_iterator iter = annCollector.annotations().begin();

    std::map<std::string, ImageDescription> scaledDesiredForFiles;

    std::string firstInputName = this->inputInfo.begin()->first;
    auto firstInputBlob = inferRequest.GetBlob(firstInputName);
    size_t ipics = 0;

    while (iter != annCollector.annotations().end() && ipics < _nPictures) {
        std::vector<std::string> files;
        int b = 0;

        int filesWatched = 0;
        for (; b < batch && iter != annCollector.annotations().end(); b++, iter++, filesWatched++) {
            expected[b] = *iter;
            string filename = iter->folder + "/" + (!subdir.empty() ? subdir + "/" : "") + iter->filename;
            try {
                Size orig_size = decoder.insertIntoBlob(std::string(imagesPath) + "/" + filename, b, *firstInputBlob, preprocessingOptions);
                float scale_x, scale_y;

                scale_x = 1.0 / iter->size.width;  // orig_size.width;
                scale_y = 1.0 / iter->size.height;  // orig_size.height;

                if (scaleProposalToInputSize) {
                    scale_x *= firstInputBlob->dims()[0];
                    scale_y *= firstInputBlob->dims()[1];
                }

                // Scaling the desired result (taken from the annotation) to the network size
                scaledDesiredForFiles.insert(std::pair<std::string, ImageDescription>(filename, desiredForFiles.at(filename).scale(scale_x, scale_y)));

                files.push_back(filename);
            } catch (const InferenceEngineException &iex) {
                slog::warn << "Can't read file " << this->imagesPath + "/" + filename << slog::endl;
                // Could be some non-image file in directory
                b--;
                continue;
            }
            ipics++;
        }

        if (files.size() == batch) {
            InferenceEngine::StatusCode sts;
            InferenceEngine::ResponseDesc dsc;

            // Infer model
            Infer(progress, filesWatched, im);
            collectCalibrationStatistic();

            // Processing the inference result
            std::map<std::string, std::list<DetectedObject>> detectedObjects = processResult(files);

            // Calculating similarity
            //
            for (int b = 0; b < files.size(); b++) {
                ImageDescription result(detectedObjects[files[b]]);
                im.apc.consumeImage(result, scaledDesiredForFiles.at(files[b]));
            }
        }
    }
    progress.finish();

    calculateLayersAccuracyDrop();

    CalibrationMetrics imCalibration;
    const ObjectDetectionInferenceMetrics &odim = dynamic_cast<const ObjectDetectionInferenceMetrics&>(im);
    if (im.nRuns > 0) {
        std::map<int, double> appc = odim.apc.calculateAveragePrecisionPerClass();

        double mAP = 0;
        for (auto i : appc) {
            mAP += i.second;
        }
        imCalibration.AccuracyResult = mAP / appc.size();
    }
    return std::shared_ptr<Processor::InferenceMetrics>(new CalibrationMetrics(imCalibration));
}



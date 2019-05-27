// Copyright (C) 2018-2019 Intel Corporation
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
#include <limits>
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
    if (pScaleShift == nullptr) {
        THROW_IE_EXCEPTION << "Layer " << lptr->name << " is not instance of ScaleShiftLayer class";
    }

    SizeVector wdims({ pData->dims[2] });

    if (scale.size() == 1) {
        scale.resize(wdims[0]);
        for (size_t i = 1; i < wdims[0]; i++) {
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
    for (size_t i = 0; i < pData->dims[2]; i++) {
        buffer[i] = scale[i];
    }
    pScaleShift->_weights = weights;


    SizeVector bdims({ pData->dims[2] });
    Blob::Ptr biases = nullptr;
    biases = make_shared_blob<float>(Precision::FP32, Layout::C, bdims);
    biases->allocate();
    buffer = biases->buffer().as<float *>();
    for (size_t i = 0; i < pData->dims[2]; i++) {
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
    auto *res_ptr = res->buffer().as<float *>();

    auto *ref_ptr = ref->buffer().as<float *>();

    size_t ref_size = ref->size();
    if (ref_size == 0) {
        throw std::logic_error("ref_size can't be equal to zero");
    }

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

    sum = pow(sum, 0.5f);

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
    /** Extract model name and load weights **/
    std::string binFileName = fileNameNoExt(_modelFileNameI8C) + ".bin";
    networkReaderC.ReadWeights(binFileName.c_str());
    if (_cBatch == 0) {
        // Zero means "take batch value from the IR"
        _cBatch = networkReaderC.getNetwork().getBatchSize();
    } else {
        // Not zero means "use the specified value"
        auto input_shapes = networkReaderC.getNetwork().getInputShapes();
        std::string input_name;
        SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        input_shape[0] = _cBatch;
        input_shapes[input_name] = input_shape;
        networkReaderC.getNetwork().reshape(input_shapes);
    }

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
        if (layerType != "Const") {
            if (/*layerType != "Split" &&*/layerType != "Input") {
                network.addOutput(layer->name);
            }
            _statData.registerLayer(layer->name);
        }
    }

    ExecutableNetwork executable_network = _pluginI8C.LoadNetwork(network, { { CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES) } });
    _inferRequestI8C = executable_network.CreateInferRequest();
}

void Int8Calibrator::validateInt8Config(const InferenceEngine::NetworkStatsMap &stat,
                                        const std::map<std::string, bool> &layersToInt8,
                                        bool convertFullyConnected) {
    _collectByLayer = false;
    _collectStatistic = false;
    networkReaderC = InferenceEngine::CNNNetReader();
    networkReaderC.ReadNetwork(_modelFileNameI8C);
    if (!networkReaderC.isParseSuccess()) THROW_IE_EXCEPTION << "cannot load a failed Model";
    /** Extract model name and load weights **/
    std::string binFileName = fileNameNoExt(_modelFileNameI8C) + ".bin";
    networkReaderC.ReadWeights(binFileName.c_str());
    if (_cBatch == 0) {
        // Zero means "take batch value from the IR"
        _cBatch = networkReaderC.getNetwork().getBatchSize();
    } else {
        // Not zero means "use the specified value"
        auto input_shapes = networkReaderC.getNetwork().getInputShapes();
        std::string input_name;
        SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        input_shape[0] = _cBatch;
        input_shapes[input_name] = input_shape;
        networkReaderC.getNetwork().reshape(input_shapes);
    }

    // Initialize statistic
    ICNNNetworkStats *pstats = nullptr;
    StatusCode s = ((ICNNNetwork&)networkReaderC.getNetwork()).getStats(&pstats, nullptr);
    if (s == StatusCode::OK && pstats) {
        pstats->setNodesStats(stat);
    }

    auto network = networkReaderC.getNetwork();

    for (auto l : network) {
        if (l->type == "FullyConnected") {
            l->params["quantization_level"] = (convertFullyConnected == false) ? "FP32" : "I8";
        }
    }

    for (auto l : layersToInt8) {
        network.getLayerByName(l.first.c_str())->
            params["quantization_level"] = (l.second == false) ? "FP32" : "I8";
    }

    ExecutableNetwork executable_network = _pluginI8C.LoadNetwork(network, { { CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(YES) } });
    _inferRequestI8C = executable_network.CreateInferRequest();
}

CNNNetwork Int8Calibrator::createICNNNetworkForLayer(CNNLayer::Ptr layerToClone, bool hasReLU) {
    CNNLayer::Ptr layerRelU = layerToClone->outData[0]->inputTo.begin()->second;

    InferenceEngine::CNNNetReader reader1;
    DataPtr inputData = layerToClone->insData[0].lock();
    std::string inputName = inputData->name;

    size_t inputBatch = inputData->getTensorDesc().getDims()[0];
    size_t inputChannels = inputData->getTensorDesc().getDims()[1];
    size_t inputHeight = inputData->getTensorDesc().getDims()[2];
    size_t inputWidth = inputData->getTensorDesc().getDims()[3];

    DataPtr outputData = layerToClone->outData[0];
    size_t outputBatch = outputData->getTensorDesc().getDims()[0];
    size_t outputChannels = outputData->getTensorDesc().getDims()[1];
    size_t outputHeight = outputData->getTensorDesc().getDims()[2];
    size_t outputWidth = outputData->getTensorDesc().getDims()[3];

    ConvolutionLayer *pConvS = dynamic_cast<ConvolutionLayer *>(layerToClone.get());
    if (pConvS == nullptr) {
        THROW_IE_EXCEPTION << "Layer " << layerToClone->name << " is not instance of ConvolutionLayer class";
    }

    std::string model = "<net name=\"L\" version=\"2\" batch=\"1\"><layers> "\
        "<layer name=\"" +
        inputName +
        "\" type=\"Input\" precision=\"FP32\" id=\"0\"> "\
        "<output>"\
        "<port id=\"0\">"\
        "<dim>" + std::to_string(inputBatch) + "</dim>"\
        "<dim>" + std::to_string(inputChannels) + "</dim>"\
        "<dim>" + std::to_string(inputHeight) + "</dim>"\
        "<dim>" + std::to_string(inputWidth) + "</dim>"\
        "</port>"\
        "</output>"\
        "</layer>"\
        "<layer name=\"" +
        layerToClone->name +
        "\" type=\"Convolution\" precision=\"FP32\" id=\"1\">"\
        "<convolution_data stride-x=\"" + std::to_string(pConvS->_stride_x) +
        "\" stride-y=\"" + std::to_string(pConvS->_stride_y) +
        "\" pad-x=\"" + std::to_string(pConvS->_padding_x) +
        "\" pad-y=\"" + std::to_string(pConvS->_padding_y) +
        "\" kernel-x=\"" + std::to_string(pConvS->_kernel_x) +
        "\" kernel-y=\"" + std::to_string(pConvS->_kernel_y) +
        "\" dilation-x=\"" + std::to_string(pConvS->_dilation_x) +
        "\" dilation-y=\"" + std::to_string(pConvS->_dilation_y) +
        "\" output=\"" + std::to_string(pConvS->_out_depth) +
        "\" group=\"" + std::to_string(pConvS->_group) + "\" />"\
        "<input>"\
        "<port id=\"1\">"\
        "<dim>" + std::to_string(inputBatch) + "</dim>"\
        "<dim>" + std::to_string(inputChannels) + "</dim>"\
        "<dim>" + std::to_string(inputHeight) + "</dim>"\
        "<dim>" + std::to_string(inputWidth) + "</dim>"\
        "</port>"\
        "</input>"\
        "<output>"\
        "<port id=\"2\">"\
        "<dim>" + std::to_string(outputBatch) + "</dim>"\
        "<dim>" + std::to_string(outputChannels) + "</dim>"\
        "<dim>" + std::to_string(outputHeight) + "</dim>"\
        "<dim>" + std::to_string(outputWidth) + "</dim>"\
        "</port>"\
        "</output>"\
        "</layer>";
    if (hasReLU) {
        model += "<layer name=\"" +
            layerRelU->name +
            "\" type=\"ReLU\" precision=\"FP32\" id=\"2\">"\
            "<input>"
            "<port id=\"3\">"\
            "<dim>" + std::to_string(outputBatch) + "</dim>"\
            "<dim>" + std::to_string(outputChannels) + "</dim>"\
            "<dim>" + std::to_string(outputHeight) + "</dim>"\
            "<dim>" + std::to_string(outputWidth) + "</dim>"\
            "</port>"\
            "</input>"\
            "<output>"\
            "<port id=\"4\">"\
            "<dim>" + std::to_string(outputBatch) + "</dim>"\
            "<dim>" + std::to_string(outputChannels) + "</dim>"\
            "<dim>" + std::to_string(outputHeight) + "</dim>"\
            "<dim>" + std::to_string(outputWidth) + "</dim>"\
            "</port>"\
            "</output>"\
            "</layer>";
    }
    model += "</layers> <edges>"\
        "<edge from-layer=\"0\" from-port=\"0\" to-layer=\"1\" to-port=\"1\"/> ";
    if (hasReLU) {
        model += "<edge from-layer=\"1\" from-port=\"2\" to-layer=\"2\" to-port=\"3\"/> ";
    }
    model += "</edges></net>";

    reader1.ReadNetwork(model.c_str(), model.length());
    ICNNNetwork &n = reader1.getNetwork();

    InferenceEngine::InputsDataMap inputs;
    n.getInputsInfo(inputs);
    CNNLayerPtr inputLayer = inputs.begin()->second->getInputData()->creatorLayer.lock();

    CNNLayerPtr convLayer;
    n.getLayerByName(layerToClone->name.c_str(), convLayer, nullptr);
    ConvolutionLayer *pConvT = dynamic_cast<ConvolutionLayer *>(convLayer.get());
    if (pConvT == nullptr) {
        THROW_IE_EXCEPTION << "Layer " << convLayer->name << " is not instance of ConvolutionLayer class";
    }

    pConvT->_weights = pConvS->_weights;
    pConvT->_biases = pConvS->_biases;
    pConvT->blobs = pConvS->blobs;

    return reader1.getNetwork();
}

void Int8Calibrator::collectByLayerStatistic(const InferenceEngine::NetworkStatsMap &stat) {
    _collectByLayer = true;
    _collectStatistic = false;
    networkReaderC = InferenceEngine::CNNNetReader();
    networkReaderC.ReadNetwork(_modelFileNameI8C);
    if (!networkReaderC.isParseSuccess()) THROW_IE_EXCEPTION << "cannot load a failed Model";
    /** Extract model name and load weights **/
    std::string binFileName = fileNameNoExt(_modelFileNameI8C) + ".bin";
    networkReaderC.ReadWeights(binFileName.c_str());
    if (_cBatch != 0) {
        auto input_shapes = networkReaderC.getNetwork().getInputShapes();
        std::string input_name;
        SizeVector input_shape;
        std::tie(input_name, input_shape) = *input_shapes.begin();
        input_shape[0] = _cBatch;
        input_shapes[input_name] = input_shape;
        networkReaderC.getNetwork().reshape(input_shapes);
    }

    auto network = networkReaderC.getNetwork();
    // 1. add all layers as output one
    for (auto &&layer : network) {
        std::string layerType = network.getLayerByName(layer->name.c_str())->type;
        if (/*layerType != "Split" &&*/layerType != "Input" && layerType != "Const") {
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
        if (layerToClone->outData.size() == 1
            && layerToClone->outData[0]->inputTo.size() == 1
            && CaselessEq<std::string>()(layerToClone->outData[0]->inputTo.begin()->second->name, "relu")) {
            layerRelU = layerToClone->outData[0]->inputTo.begin()->second;
        }

        CNNNetwork n = createICNNNetworkForLayer(layerToClone, layerRelU ? true : false);
        if (_cBatch != 0) {
            auto input_shapes = n.getInputShapes();
            std::string input_name;
            SizeVector input_shape;
            std::tie(input_name, input_shape) = *input_shapes.begin();
            input_shape[0] = _cBatch;
            input_shapes[input_name] = input_shape;
            n.reshape(input_shapes);
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
        std::string inputName = layerToClone->insData[0].lock()->name;
        request.SetBlob(inputName, _inferRequestI8C.GetBlob(inputName));
        _singleLayerRequests[layerToClone->name] = { request, layerRelU ? layerRelU->name : layerToClone->name, layerToClone->name };
    }
}


void Int8Calibrator::collectCalibrationStatistic(size_t pics) {
    if (_collectByLayer) {
        std::map<std::string, SingleLayerData>::iterator it = _singleLayerRequests.begin();
        while (it != _singleLayerRequests.end()) {
            it->second._request.Infer();
            Blob::Ptr expected = _inferRequestI8C.GetBlob(it->second._outputName);
            Blob::Ptr result = it->second._request.GetBlob(it->second._outputName);
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

            size_t N, C;
            if (outBlob->dims().size() == 4 && outBlob->layout() == Layout::NCHW) {
                // TODO(amalyshe) cahnge to using of tensor desc
                N = pics;
                C = outBlob->dims()[2];
            } else if (outBlob->dims().size() == 2 && outBlob->layout() == Layout::NC) {
                N = pics;
                C = outBlob->dims()[0];
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

shared_ptr<Processor::InferenceMetrics> ClassificationCalibrator::Process(bool stream_output) {
    inferRequest = _inferRequestI8C;
    int top1Result = 0, total = 0;

    ClassificationSetGenerator generator;

    try {
        generator.readLabels(labelFileName);
    } catch (InferenceEngine::details::InferenceEngineException& ex) {
        slog::warn << "Can't read labels file " << labelFileName << slog::endl;
        slog::warn << "Error: " << ex.what() << slog::endl;
    }
    auto validationMap = generator.getValidationMap(imagesPath);

    if (validationMap.empty()) {
        THROW_IE_EXCEPTION << "The validation dataset in " << imagesPath << "is empty. Check the dataset file or folder and the labels file";
    }

    ImageDecoder decoder;

    // ----------------------------Do inference-------------------------------------------------------------
    std::vector<int> expected(batch);
    std::vector<std::string> files(batch);

    if (!_nPictures) {
        _nPictures = validationMap.size();
    }


    ConsoleProgress progress(_nPictures, stream_output);

    CalibrationMetrics im;

    std::string firstInputName = this->inputInfo.begin()->first;
    std::string firstOutputName = this->outInfo.begin()->first;
    auto firstInputBlob = inferRequest.GetBlob(firstInputName);
    auto firstOutputBlob = inferRequest.GetBlob(firstOutputName);

    size_t ipics = 0;
    auto iter = validationMap.begin();
    while (iter != validationMap.end() && ipics < _nPictures) {
        size_t b = 0;
        int filesWatched = 0;
        for (; b < batch && iter != validationMap.end() && ipics + b < _nPictures ; b++, iter++, filesWatched++) {
            expected[b] = iter->first;
            try {
                decoder.insertIntoBlob(iter->second, b, *firstInputBlob, preprocessingOptions);
                files[b] = iter->second;
            } catch (const InferenceEngineException &iex) {
                slog::warn << "Can't read file " << iter->second << slog::endl;
                slog::warn << "Error: " << iex.what() << slog::endl;
                // Could be some non-image file in directory
                b--;
                continue;
            }
        }
        ipics += batch;

        Infer(progress, filesWatched, im);
        collectCalibrationStatistic(b);

        std::vector<unsigned> results;
        InferenceEngine::TopResults(1, *firstOutputBlob, results);
        for (size_t i = 0; i < b; i++) {
            int expc = expected[i];
            if (zeroBackground) expc++;
            bool top1Scored = (static_cast<int>(results[i]) == expc);
            if (top1Scored) top1Result++;
            total++;
        }
    }
    progress.finish();

    calculateLayersAccuracyDrop();

    if (total == 0) {
        throw std::logic_error("total can't be equal to zero");
    }

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
    _cBatch = flags_b;
}

shared_ptr<Processor::InferenceMetrics> SSDObjectDetectionCalibrator::Process(bool stream_output) {
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
            DetectedObject dob(classes[obj.name], static_cast<float>(obj.bndbox.xmin), static_cast<float>(obj.bndbox.ymin),
                               static_cast<float>(obj.bndbox.xmax), static_cast<float>(obj.bndbox.ymax), 1.0f, obj.difficult != 0);
            dobList.push_back(dob);
        }
        ImageDescription id(dobList);
        desiredForFiles.insert(std::pair<std::string, ImageDescription>(ann.folder + "/" + (!subdir.empty() ? subdir + "/" : "") + ann.filename, id));
    }

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

    ConsoleProgress progress(_nPictures, stream_output);

    ObjectDetectionInferenceMetrics im(threshold);

    vector<VOCAnnotation>::const_iterator iter = annCollector.annotations().begin();

    std::map<std::string, ImageDescription> scaledDesiredForFiles;

    std::string firstInputName = this->inputInfo.begin()->first;
    auto firstInputBlob = inferRequest.GetBlob(firstInputName);
    size_t ipics = 0;

    while (iter != annCollector.annotations().end() && ipics < _nPictures) {
        std::vector<std::string> files;
        size_t b = 0;

        int filesWatched = 0;
        for (; b < batch && iter != annCollector.annotations().end(); b++, iter++, filesWatched++) {
            expected[b] = *iter;
            string filename = iter->folder + "/" + (!subdir.empty() ? subdir + "/" : "") + iter->filename;
            try {
                float scale_x, scale_y;

                scale_x = 1.0f / iter->size.width;  // orig_size.width;
                scale_y = 1.0f / iter->size.height;  // orig_size.height;

                if (scaleProposalToInputSize) {
                    scale_x *= firstInputBlob->dims()[0];
                    scale_y *= firstInputBlob->dims()[1];
                }

                // Scaling the desired result (taken from the annotation) to the network size
                scaledDesiredForFiles.insert(std::pair<std::string, ImageDescription>(filename, desiredForFiles.at(filename).scale(scale_x, scale_y)));

                files.push_back(filename);
            } catch (const InferenceEngineException &iex) {
                slog::warn << "Can't read file " << this->imagesPath + "/" + filename << slog::endl;
                slog::warn << "Error: " << iex.what() << slog::endl;
                // Could be some non-image file in directory
                b--;
                continue;
            }
            ipics++;
        }

        // Infer model
        Infer(progress, filesWatched, im);
        collectCalibrationStatistic(b);

        // Processing the inference result
        std::map<std::string, std::list<DetectedObject>> detectedObjects = processResult(files);

        // Calculating similarity
        //
        for (size_t j = 0; j < files.size(); j++) {
            ImageDescription result(detectedObjects[files[j]]);
            im.apc.consumeImage(result, scaledDesiredForFiles.at(files[j]));
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
        imCalibration.AccuracyResult = static_cast<float>(mAP / appc.size());
    }
    return std::shared_ptr<Processor::InferenceMetrics>(new CalibrationMetrics(imCalibration));
}



// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <map>
#include <vector>
#include <string>

#include "details/ie_cnn_network_tools.h"
#include "details/caseless.hpp"
#include "network_serializer.h"
#include "exec_graph_info.hpp"
#include "xml_parse_utils.h"

using namespace InferenceEngine;
using namespace details;

template<typename T> std::string arrayToIRProperty(const T& property) {
    std::string sProperty;
    for (size_t i = 0; i < property.size(); i++) {
        sProperty = sProperty + std::to_string(property[i]) +
            std::string((i != property.size() - 1) ? "," : "");
    }
    return sProperty;
}

template<typename T> std::string arrayRevertToIRProperty(const T& property) {
    std::string sProperty;
    for (size_t i = 0; i < property.size(); i++) {
        sProperty = sProperty + std::to_string(property[property.size() - i - 1]) +
            std::string((i != property.size() - 1) ? "," : "");
    }
    return sProperty;
}


void NetworkSerializer::serialize(
    const std::string &xmlPath,
    const std::string &binPath,
    const InferenceEngine::ICNNNetwork& network) {
    const std::vector<CNNLayerPtr> ordered = CNNNetSortTopologically(network);

    // A flag for serializing executable graph information (not complete IR)
    bool execGraphInfoSerialization = false;
    // If first layer has perfCounter parameter set then it's executable graph info serialization.
    // All other layers must also have this parameter set.
    if (ordered[0]->params.find(ExecGraphInfoSerialization::PERF_COUNTER) != ordered[0]->params.end()) {
        execGraphInfoSerialization = true;
        for (const auto &layer : ordered) {
            if (layer->params.find(ExecGraphInfoSerialization::PERF_COUNTER) == layer->params.end()) {
                THROW_IE_EXCEPTION << "Each node must have " << ExecGraphInfoSerialization::PERF_COUNTER
                                   << " parameter set in case of executable graph info serialization";
            }
        }
    }

    bool dumpWeights = !execGraphInfoSerialization & !binPath.empty();
    std::ofstream ofsBin;
    if (dumpWeights) {
        ofsBin.open(binPath, std::ofstream::out | std::ofstream::binary);
        if (!ofsBin) {
            THROW_IE_EXCEPTION << "File '" << binPath << "' is not opened as out file stream";
        }
    }

    pugi::xml_document doc;
    pugi::xml_node netXml = doc.append_child("net");
    netXml.append_attribute("name").set_value(network.getName().c_str());

    // no need to print this information for executable graph information serialization because it is not IR.
    if (!execGraphInfoSerialization) {
        netXml.append_attribute("version").set_value("3");
        netXml.append_attribute("batch").set_value(network.getBatchSize());
    }

    pugi::xml_node layers = netXml.append_child("layers");

    std::map<CNNLayer::Ptr, size_t> matching;
    for (size_t i = 0; i < ordered.size(); i++) {
        matching[ordered[i]] = i;
    }

    const std::string dataName = "data";
    size_t dataOffset = 0;
    for (size_t i = 0; i < ordered.size(); ++i) {
        const CNNLayerPtr node = ordered[i];

        pugi::xml_node layer = layers.append_child("layer");
        const Precision precision = node->precision;
        layer.append_attribute("name").set_value(node->name.c_str());
        layer.append_attribute("type").set_value(node->type.c_str());
        layer.append_attribute("precision").set_value(precision.name());
        layer.append_attribute("id").set_value(i);

        if (!execGraphInfoSerialization) {
            updateStdLayerParams(node);
        }

        const auto &params = node->params;
        if (!params.empty()) {
            pugi::xml_node data = layer.append_child(dataName.c_str());

            for (const auto &it : params) {
                data.append_attribute(it.first.c_str()).set_value(it.second.c_str());
            }
        }

        if (!node->insData.empty()) {
            pugi::xml_node input = layer.append_child("input");

            for (size_t iport = 0; iport < node->insData.size(); iport++) {
                const DataPtr d = node->insData[iport].lock();
                pugi::xml_node port = input.append_child("port");

                port.append_attribute("id").set_value(iport);

                for (auto dim : d->getDims()) {
                    port.append_child("dim").text().set(dim);
                }
            }
        }
        if (!node->outData.empty()) {
            pugi::xml_node input = layer.append_child("output");
            for (size_t oport = 0; oport < node->outData.size(); oport++) {
                pugi::xml_node port = input.append_child("port");

                port.append_attribute("id").set_value(node->insData.size() + oport);

                for (const auto dim : node->outData[oport]->getDims()) {
                    port.append_child("dim").text().set(dim);
                }
            }
        }
        if (dumpWeights && !node->blobs.empty()) {
            auto blobsNode = layer.append_child("blobs");
            for (const auto &dataIt : node->blobs) {
                const char *dataPtr = dataIt.second->buffer().as<char*>();

                size_t dataSize = dataIt.second->byteSize();
                pugi::xml_node data = blobsNode.append_child(dataIt.first.c_str());
                data.append_attribute("offset").set_value(dataOffset);
                data.append_attribute("size").set_value(dataSize);

                dataOffset += dataSize;
                ofsBin.write(dataPtr, dataSize);
                if (!ofsBin.good()) {
                    THROW_IE_EXCEPTION << "Error during '" << binPath << "' writing";
                }
            }
        }
    }

    if (dumpWeights) {
        ofsBin.close();
        if (!ofsBin.good()) {
            THROW_IE_EXCEPTION << "Error during '" << binPath << "' closing";
        }
    }

    pugi::xml_node edges = netXml.append_child("edges");

    for (const auto &ord : ordered) {
        const CNNLayer::Ptr node = ord;

        if (!node->outData.empty()) {
            auto itFrom = matching.find(node);
            if (itFrom == matching.end()) {
                THROW_IE_EXCEPTION << "Internal error, cannot find " << node->name << " in matching container during serialization of IR";
            }
            for (size_t oport = 0; oport < node->outData.size(); oport++) {
                const DataPtr outData = node->outData[oport];
                for (const auto &inputTo : outData->inputTo) {
                    auto itTo = matching.find(inputTo.second);
                    if (itTo == matching.end()) {
                        THROW_IE_EXCEPTION << "Broken edge form layer " << node->name << " to layer "  << inputTo.first<< "during serialization of IR";
                    }

                    int foundPort = -1;
                    for (int iport = 0; iport < inputTo.second->insData.size(); iport++) {
                        if (inputTo.second->insData[iport].lock() == outData) {
                            foundPort = iport;
                        }
                    }
                    if (foundPort == -1) {
                        THROW_IE_EXCEPTION << "Broken edge from layer to parent, cannot find parent " << outData->name << " for layer " << inputTo.second->name
                            << "\ninitial layer for edge output " << node->name;
                    }
                    pugi::xml_node edge = edges.append_child("edge");

                    edge.append_attribute("from-layer").set_value(itFrom->second);
                    edge.append_attribute("from-port").set_value(oport + node->insData.size());

                    edge.append_attribute("to-layer").set_value(itTo->second);
                    edge.append_attribute("to-port").set_value(foundPort);
                }
            }
        }
    }

    // no need to print this info in case of executable graph info serialization
    if (!execGraphInfoSerialization) {
        updatePreProcInfo(network, netXml);
        updateStatisticsInfo(network, netXml);
    }

    if (!doc.save_file(xmlPath.c_str())) {
        THROW_IE_EXCEPTION << "file '" << xmlPath << "' was not serialized";
    }
}

void NetworkSerializer::updateStdLayerParams(const CNNLayer::Ptr &layer) {
    auto layerPtr = layer.get();
    auto &params = layer->params;

    if (CaselessEq<std::string>()(layer->type, "power")) {
        auto *lr = dynamic_cast<PowerLayer *>(layerPtr);

        params["scale"] = std::to_string(lr->scale);
        params["shift"] = std::to_string(lr->offset);
        params["power"] = std::to_string(lr->power);
    } else if (CaselessEq<std::string>()(layer->type, "convolution") ||
               CaselessEq<std::string>()(layer->type, "deconvolution")) {
        auto *lr = dynamic_cast<ConvolutionLayer *>(layerPtr);

        params["kernel"] = arrayRevertToIRProperty(lr->_kernel);
        params["pads_begin"] = arrayRevertToIRProperty(lr->_padding);
        params["pads_end"] = arrayRevertToIRProperty(lr->_pads_end);
        params["strides"] = arrayRevertToIRProperty(lr->_stride);
        params["dilations"] = arrayRevertToIRProperty(lr->_dilation);
        params["output"] = std::to_string(lr->_out_depth);
        params["group"] = std::to_string(lr->_group);
    } else if (CaselessEq<std::string>()(layer->type, "relu")) {
        auto *lr = dynamic_cast<ReLULayer *>(layerPtr);
        if (lr->negative_slope != 0.0f) {
            params["negative_slope"] = std::to_string(lr->negative_slope);
        }
    } else if (CaselessEq<std::string>()(layer->type, "norm") ||
               CaselessEq<std::string>()(layer->type, "lrn")) {
        auto *lr = dynamic_cast<NormLayer *>(layerPtr);

        params["alpha"] = std::to_string(lr->_alpha);
        params["beta"] = std::to_string(lr->_beta);
        params["local-size"] = std::to_string(lr->_size);
        params["region"] = lr->_isAcrossMaps ? "across" : "same";
    } else if (CaselessEq<std::string>()(layer->type, "pooling")) {
        auto *lr = dynamic_cast<PoolingLayer *>(layerPtr);

        params["kernel"] = arrayRevertToIRProperty(lr->_kernel);
        params["pads_begin"] = arrayRevertToIRProperty(lr->_padding);
        params["pads_end"] = arrayRevertToIRProperty(lr->_pads_end);
        params["strides"] = arrayRevertToIRProperty(lr->_stride);

        switch (lr->_type) {
            case PoolingLayer::MAX:
                params["pool-method"] = "max";
                break;
            case PoolingLayer::AVG:
                params["pool-method"] = "avg";
                break;

            default:
                THROW_IE_EXCEPTION << "Found unsupported pooling method: " << lr->_type;
        }
    } else if (CaselessEq<std::string>()(layer->type, "split")) {
        auto *lr = dynamic_cast<SplitLayer *>(layerPtr);
        params["axis"] = std::to_string(lr->_axis);
    } else if (CaselessEq<std::string>()(layer->type, "concat")) {
        auto *lr = dynamic_cast<ConcatLayer *>(layerPtr);
        params["axis"] = std::to_string(lr->_axis);
    } else if (CaselessEq<std::string>()(layer->type, "FullyConnected") ||
               CaselessEq<std::string>()(layer->type, "InnerProduct")) {
        auto *lr = dynamic_cast<FullyConnectedLayer *>(layerPtr);
        params["out-size"] = std::to_string(lr->_out_num);
    } else if (CaselessEq<std::string>()(layer->type, "softmax")) {
        auto *lr = dynamic_cast<SoftMaxLayer *>(layerPtr);
        params["axis"] = std::to_string(lr->axis);
    } else if (CaselessEq<std::string>()(layer->type, "reshape")) {
        // need to add here support of flatten layer if it is created from API
        auto *lr = dynamic_cast<ReshapeLayer *>(layerPtr);
        params["dim"] = arrayToIRProperty(lr->shape);
    } else if (CaselessEq<std::string>()(layer->type, "Eltwise")) {
        auto *lr = dynamic_cast<EltwiseLayer *>(layerPtr);

        std::string op;

        switch (lr->_operation) {
            case EltwiseLayer::Sum:
                op = "sum";
                break;
            case EltwiseLayer::Prod:
                op = "prod";
                break;
            case EltwiseLayer::Max:
                op = "max";
                break;
            default:
                break;
        }

        params["operation"] = op;
    } else if (CaselessEq<std::string>()(layer->type, "scaleshift")) {
        auto *lr = dynamic_cast<ScaleShiftLayer *>(layerPtr);
        params["broadcast"] = std::to_string(lr->_broadcast);
    } else if (CaselessEq<std::string>()(layer->type, "crop")) {
        auto *lr = dynamic_cast<CropLayer *>(layerPtr);
        params["axis"] = arrayToIRProperty(lr->axis);
        params["offset"] = arrayToIRProperty(lr->offset);
        params["dim"] = arrayToIRProperty(lr->dim);
    } else if (CaselessEq<std::string>()(layer->type, "tile")) {
        auto *lr = dynamic_cast<TileLayer *>(layerPtr);
        params["axis"] = std::to_string(lr->axis);
        params["tiles"] = std::to_string(lr->tiles);
    } else if (CaselessEq<std::string>()(layer->type, "prelu")) {
        auto *lr = dynamic_cast<PReLULayer *>(layerPtr);
        params["channel_shared"] = std::to_string(lr->_channel_shared);
    } else if (CaselessEq<std::string>()(layer->type, "clamp")) {
        auto *lr = dynamic_cast<ClampLayer *>(layerPtr);
        params["min"] = std::to_string(lr->min_value);
        params["max"] = std::to_string(lr->max_value);
    } else if (CaselessEq<std::string>()(layer->type, "BatchNormalization")) {
        auto *lr = dynamic_cast<BatchNormalizationLayer *>(layerPtr);
        params["epsilon"] = std::to_string(lr->epsilon);
    } else if (CaselessEq<std::string>()(layer->type, "grn")) {
        auto *lr = dynamic_cast<GRNLayer *>(layerPtr);
        params["bias"] = std::to_string(lr->bias);
    } else if (CaselessEq<std::string>()(layer->type, "mvn")) {
        auto *lr = dynamic_cast<MVNLayer *>(layerPtr);
        params["across_channels"] = std::to_string(lr->across_channels);
        params["normalize_variance"] = std::to_string(lr->normalize);
    } else if (CaselessEq<std::string>()(layer->type, "rnn") ||
               CaselessEq<std::string>()(layer->type, "TensorIterator") ||
               CaselessEq<std::string>()(layer->type, "LSTMCell")) {
        THROW_IE_EXCEPTION << "Not covered layers for writing to IR";
    }

    if (layer->params.find("quantization_level") != layer->params.end()) {
        params["quantization_level"] = layer->params["quantization_level"];
    }

    // update of weightable layers
    auto *pwlayer = dynamic_cast<WeightableLayer *>(layerPtr);
    if (pwlayer) {
        if (pwlayer->_weights) {
            pwlayer->blobs["weights"] = pwlayer->_weights;
        }
        if (pwlayer->_biases) {
            pwlayer->blobs["biases"] = pwlayer->_biases;
        }
    }
}

void NetworkSerializer::updatePreProcInfo(const InferenceEngine::ICNNNetwork& network, pugi::xml_node &netXml) {
    InputsDataMap inputInfo;
    network.getInputsInfo(inputInfo);

    // Assume that you preprocess only one input
    for (auto ii : inputInfo) {
        const PreProcessInfo &pp = ii.second->getPreProcess();
        size_t nInChannels = pp.getNumberOfChannels();
        if (nInChannels) {
            pugi::xml_node preproc = netXml.append_child("pre-process");

            preproc.append_attribute("reference-layer-name").set_value(ii.first.c_str());
            preproc.append_attribute("mean-precision").set_value(Precision(Precision::FP32).name());

            for (size_t ch = 0; ch < nInChannels; ch++) {
                const PreProcessChannel::Ptr &preProcessChannel = pp[ch];
                auto channel = preproc.append_child("channel");
                channel.append_attribute("id").set_value(ch);

                auto mean = channel.append_child("mean");

                if (!preProcessChannel->meanData) {
                    mean.append_attribute("value").set_value(preProcessChannel->meanValue);
                } else {
                    THROW_IE_EXCEPTION << "Mean data is not supported yet for serialization of the model";
                }
            }
        }
    }
}

void NetworkSerializer::updateStatisticsInfo(const InferenceEngine::ICNNNetwork& network, pugi::xml_node &netXml) {
    // If statistics exists, add it to the file
    ICNNNetworkStats *netNodesStats = nullptr;
    auto stats = netXml.append_child("statistics");
    network.getStats(&netNodesStats, nullptr);
    const NetworkStatsMap statsmap = netNodesStats->getNodesStats();

    auto joinCommas = [&](const std::vector<float> &v) -> std::string {
        std::string res;

        for (size_t i = 0; i < v.size(); ++i) {
            res += std::to_string(v[i]);
            if (i < v.size() - 1) {
                res += ", ";
            }
        }

        return res;
    };

    for (const auto &itStats : statsmap) {
        auto layer = stats.append_child("layer");

        layer.append_child("name").text().set(itStats.first.c_str());

        layer.append_child("min").text().set(joinCommas(itStats.second->_minOutputs).c_str());
        layer.append_child("max").text().set(joinCommas(itStats.second->_maxOutputs).c_str());
    }
}
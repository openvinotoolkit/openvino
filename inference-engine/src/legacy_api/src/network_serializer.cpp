// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "network_serializer.h"

#include <fstream>
#include <map>
#include <queue>
#include <deque>
#include <string>
#include <vector>
#include <unordered_set>
#include <sstream>

#include "details/caseless.hpp"
#include "details/ie_cnn_network_tools.h"
#include "exec_graph_info.hpp"
#include "xml_parse_utils.h"
#include "ie_ngraph_utils.hpp"
#include <ngraph/variant.hpp>

namespace InferenceEngine {
namespace Serialization {

namespace {
template <typename T>
std::string arrayToIRProperty(const T& property) {
    std::string sProperty;
    for (size_t i = 0; i < property.size(); i++) {
        sProperty = sProperty + std::to_string(property[i]) + std::string((i != property.size() - 1) ? "," : "");
    }
    return sProperty;
}

template <typename T>
std::string arrayRevertToIRProperty(const T& property) {
    std::string sProperty;
    for (size_t i = 0; i < property.size(); i++) {
        sProperty = sProperty + std::to_string(property[property.size() - i - 1]) +
                    std::string((i != property.size() - 1) ? "," : "");
    }
    return sProperty;
}

std::size_t updatePreProcInfo(const InferenceEngine::ICNNNetwork& network, pugi::xml_node& netXml,
                              const std::size_t weightsDataOffset) {
    InputsDataMap inputInfo;
    network.getInputsInfo(inputInfo);

    // Assume that you preprocess only one input
    auto dataOffset = weightsDataOffset;
    for (auto ii : inputInfo) {
        const PreProcessInfo& pp = ii.second->getPreProcess();
        size_t nInChannels = pp.getNumberOfChannels();
        if (nInChannels) {
            pugi::xml_node preproc = netXml.append_child("pre-process");

            preproc.append_attribute("reference-layer-name").set_value(ii.first.c_str());
            preproc.append_attribute("mean-precision").set_value(Precision(Precision::FP32).name());

            for (size_t ch = 0; ch < nInChannels; ch++) {
                const PreProcessChannel::Ptr& preProcessChannel = pp[ch];
                auto channel = preproc.append_child("channel");
                channel.append_attribute("id").set_value(ch);

                auto mean = channel.append_child("mean");

                if (!preProcessChannel->meanData) {
                    mean.append_attribute("value").set_value(preProcessChannel->meanValue);
                } else {
                    auto size = preProcessChannel->meanData->byteSize();
                    mean.append_attribute("size").set_value(size);
                    mean.append_attribute("offset").set_value(dataOffset);
                    dataOffset += size;
                }

                if (1.f != preProcessChannel->stdScale) {
                    channel.append_child("scale").append_attribute("value").set_value(
                        CNNLayer::ie_serialize_float(preProcessChannel->stdScale).c_str());
                }
            }
        }
    }
    return dataOffset;
}

void UpdateStatisticsInfo(const InferenceEngine::ICNNNetwork& network, pugi::xml_node& netXml) {
    // If statistics exists, add it to the file
    ICNNNetworkStats* netNodesStats = nullptr;
    auto stats = netXml.append_child("statistics");
    auto resultCode = network.getStats(&netNodesStats, nullptr);
    if (resultCode != StatusCode::OK) {
        THROW_IE_EXCEPTION << InferenceEngine::details::as_status << resultCode
                           << "Can't get statistics info for serialization of the model";
    }
    const NetworkStatsMap statsmap = netNodesStats->getNodesStats();

    auto joinCommas = [&](const std::vector<float>& v) -> std::string {
        std::string res;

        for (size_t i = 0; i < v.size(); ++i) {
            res += std::to_string(v[i]);
            if (i < v.size() - 1) {
                res += ", ";
            }
        }

        return res;
    };

    for (const auto& itStats : statsmap) {
        auto layer = stats.append_child("layer");

        layer.append_child("name").text().set(itStats.first.c_str());

        layer.append_child("min").text().set(joinCommas(itStats.second->_minOutputs).c_str());
        layer.append_child("max").text().set(joinCommas(itStats.second->_maxOutputs).c_str());
    }
}

void UpdateStdLayerParams(const CNNLayer::Ptr& layer) {
    auto layerPtr = layer.get();
    auto& params = layer->params;
    using ::InferenceEngine::details::CaselessEq;
    if (CaselessEq<std::string>()(layer->type, "power")) {
        auto* lr = dynamic_cast<PowerLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of PowerLayer class";
        }
        params["scale"] = CNNLayer::ie_serialize_float(lr->scale);
        params["shift"] = CNNLayer::ie_serialize_float(lr->offset);
        params["power"] = CNNLayer::ie_serialize_float(lr->power);
    } else if (CaselessEq<std::string>()(layer->type, "convolution") ||
               CaselessEq<std::string>()(layer->type, "deconvolution")) {
        auto* lr = dynamic_cast<ConvolutionLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of ConvolutionLayer class";
        }
        params["kernel"] = arrayRevertToIRProperty(lr->_kernel);
        params["pads_begin"] = arrayRevertToIRProperty(lr->_padding);
        params["pads_end"] = arrayRevertToIRProperty(lr->_pads_end);
        params["strides"] = arrayRevertToIRProperty(lr->_stride);
        params["dilations"] = arrayRevertToIRProperty(lr->_dilation);
        params["output"] = std::to_string(lr->_out_depth);
        params["group"] = std::to_string(lr->_group);
    } else if (CaselessEq<std::string>()(layer->type, "deformable_convolution")) {
        auto* lr = dynamic_cast<DeformableConvolutionLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of DeformableConvolutionLayer class";
        }
        params["kernel"] = arrayRevertToIRProperty(lr->_kernel);
        params["pads_begin"] = arrayRevertToIRProperty(lr->_padding);
        params["pads_end"] = arrayRevertToIRProperty(lr->_pads_end);
        params["strides"] = arrayRevertToIRProperty(lr->_stride);
        params["dilations"] = arrayRevertToIRProperty(lr->_dilation);
        params["output"] = std::to_string(lr->_out_depth);
        params["group"] = std::to_string(lr->_group);
        params["deformable_group"] = std::to_string(lr->_deformable_group);
    } else if (CaselessEq<std::string>()(layer->type, "relu")) {
        auto* lr = dynamic_cast<ReLULayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of ReLULayer class";
        }
        if (lr->negative_slope != 0.0f) {
            params["negative_slope"] = CNNLayer::ie_serialize_float(lr->negative_slope);
        }
    } else if (CaselessEq<std::string>()(layer->type, "norm") || CaselessEq<std::string>()(layer->type, "lrn")) {
        auto* lr = dynamic_cast<NormLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of NormLayer class";
        }
        params["alpha"] = CNNLayer::ie_serialize_float(lr->_alpha);
        params["beta"] = CNNLayer::ie_serialize_float(lr->_beta);
        params["local-size"] = std::to_string(lr->_size);
        params["region"] = lr->_isAcrossMaps ? "across" : "same";
    } else if (CaselessEq<std::string>()(layer->type, "pooling")) {
        auto* lr = dynamic_cast<PoolingLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of PoolingLayer class";
        }
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
        auto* lr = dynamic_cast<SplitLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of SplitLayer class";
        }
        params["axis"] = std::to_string(lr->_axis);
    } else if (CaselessEq<std::string>()(layer->type, "concat")) {
        auto* lr = dynamic_cast<ConcatLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of ConcatLayer class";
        }
        params["axis"] = std::to_string(lr->_axis);
    } else if (CaselessEq<std::string>()(layer->type, "FullyConnected") ||
               CaselessEq<std::string>()(layer->type, "InnerProduct")) {
        auto* lr = dynamic_cast<FullyConnectedLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of FullyConnectedLayer class";
        }
        params["out-size"] = std::to_string(lr->_out_num);
    } else if (CaselessEq<std::string>()(layer->type, "softmax")) {
        auto* lr = dynamic_cast<SoftMaxLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of SoftMaxLayer class";
        }
        params["axis"] = std::to_string(lr->axis);
    } else if (CaselessEq<std::string>()(layer->type, "reshape")) {
        // need to add here support of flatten layer if it is created from API
        auto* lr = dynamic_cast<ReshapeLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of ReshapeLayer class";
        }
        params["dim"] = arrayToIRProperty(lr->shape);
    } else if (CaselessEq<std::string>()(layer->type, "Eltwise")) {
        auto* lr = dynamic_cast<EltwiseLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of EltwiseLayer class";
        }

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
        case EltwiseLayer::Sub:
            op = "sub";
            break;
        default:
            break;
        }

        params["operation"] = op;
    } else if (CaselessEq<std::string>()(layer->type, "scaleshift")) {
        auto* lr = dynamic_cast<ScaleShiftLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of ScaleShiftLayer class";
        }
        params["broadcast"] = std::to_string(lr->_broadcast);
    } else if (CaselessEq<std::string>()(layer->type, "crop")) {
        auto* lr = dynamic_cast<CropLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of CropLayer class";
        }
        params["axis"] = arrayToIRProperty(lr->axis);
        params["offset"] = arrayToIRProperty(lr->offset);
        params["dim"] = arrayToIRProperty(lr->dim);
    } else if (CaselessEq<std::string>()(layer->type, "tile")) {
        auto* lr = dynamic_cast<TileLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of TileLayer class";
        }
        params["axis"] = std::to_string(lr->axis);
        params["tiles"] = std::to_string(lr->tiles);
    } else if (CaselessEq<std::string>()(layer->type, "prelu")) {
        auto* lr = dynamic_cast<PReLULayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of PReLULayer class";
        }
        params["channel_shared"] = std::to_string(lr->_channel_shared);
    } else if (CaselessEq<std::string>()(layer->type, "clamp")) {
        auto* lr = dynamic_cast<ClampLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of ClampLayer class";
        }
        params["min"] = CNNLayer::ie_serialize_float(lr->min_value);
        params["max"] = CNNLayer::ie_serialize_float(lr->max_value);
    } else if (CaselessEq<std::string>()(layer->type, "BatchNormalization")) {
        auto* lr = dynamic_cast<BatchNormalizationLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of BatchNormalizationLayer class";
        }
        params["epsilon"] = CNNLayer::ie_serialize_float(lr->epsilon);
    } else if (CaselessEq<std::string>()(layer->type, "grn")) {
        auto* lr = dynamic_cast<GRNLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of GRNLayer class";
        }
        params["bias"] = CNNLayer::ie_serialize_float(lr->bias);
    } else if (CaselessEq<std::string>()(layer->type, "mvn")) {
        auto* lr = dynamic_cast<MVNLayer*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of MVNLayer class";
        }
        params["across_channels"] = std::to_string(lr->across_channels);
        params["normalize_variance"] = std::to_string(lr->normalize);
    } else if (CaselessEq<std::string>()(layer->type, "LSTMCell")) {
        auto* lr = dynamic_cast<RNNCellBase*>(layerPtr);
        if (lr == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << layerPtr->name << " is not instance of LSTMCell class";
        }
        params["hidden_size"] = std::to_string(lr->hidden_size);
    } else if (CaselessEq<std::string>()(layer->type, "rnn") ||
               CaselessEq<std::string>()(layer->type, "TensorIterator")) {
        THROW_IE_EXCEPTION << "Not covered layers for writing to IR";
    }

    if (layer->params.find("quantization_level") != layer->params.end()) {
        params["quantization_level"] = layer->params["quantization_level"];
    }

    // update of weightable layers
    auto* pwlayer = dynamic_cast<WeightableLayer*>(layerPtr);
    if (pwlayer) {
        if (pwlayer->_weights) {
            pwlayer->blobs["weights"] = pwlayer->_weights;
        }
        if (pwlayer->_biases) {
            pwlayer->blobs["biases"] = pwlayer->_biases;
        }
    }
}
}  //  namespace

std::vector<CNNLayerPtr> TopologicalSort(const ICNNNetwork& network) {
    std::vector<CNNLayerPtr> ordered;
    std::unordered_set<std::string> used;

    OutputsDataMap outputs;
    network.getOutputsInfo(outputs);

    InputsDataMap inputs;
    network.getInputsInfo(inputs);

    auto get_consumers = [](const CNNLayerPtr& node) -> std::vector<CNNLayerPtr> {
        std::vector<CNNLayerPtr> consumers;
        for (const auto & output : node->outData) {
            for (const auto &consumer : output->getInputTo()) {
                consumers.push_back(consumer.second);
            }
        }
        return consumers;
    };
    auto bfs = [&used, &ordered, &get_consumers](const CNNLayerPtr& start_node, bool traverse_via_outputs = false) {
        if (!start_node) return;
        std::deque<CNNLayerPtr> q;
        q.push_front(start_node);
        while (!q.empty()) {
            auto node = q.front();
            q.pop_front();
            if (used.insert(node->name).second) {
                ordered.push_back(node);
            }

            // Traverse via inputs
            for (const auto & input : node->insData) {
                auto locked_input = input.lock();
                if (!locked_input) {
                    THROW_IE_EXCEPTION << "insData for " << node->name << " is not valid.";
                }
                if (auto next_node = locked_input->getCreatorLayer().lock()) {
                    if (!used.count(next_node->name)) {
                        // Check that all consumers were used
                        bool all_consumers_used(true);
                        for (const auto & consumer : get_consumers(next_node)) {
                            if (!used.count(consumer->name)) all_consumers_used = false;
                        }
                        if (all_consumers_used) {
                            q.push_front(next_node);
                        }
                    }
                }
            }

            // Traverse via outputs
            if (traverse_via_outputs) {
                for (const auto &consumer : get_consumers(node)) {
                    if (!used.count(consumer->name)) {
                        q.push_front(consumer);
                    }
                }
            }
        }
    };

    // First we run bfs starting from outputs that provides deterministic graph traverse
    for (const auto & output : outputs) {
        if (!used.count(output.first)) {
            bfs(output.second->getCreatorLayer().lock());
        }
    }

    // For cases when graph has no outputs we start bfs from inputs to ensure topological sort
    for (const auto & input : inputs) {
        const auto data_ptr = input.second->getInputData();
        for (const auto & consumer : data_ptr->getInputTo())
        if (!used.count(consumer.first)) {
            bfs(consumer.second, true);
        }
    }

    std::reverse(ordered.begin(), ordered.end());
    return ordered;
}

namespace {

void FillXmlDocWithExecutionNGraph(const InferenceEngine::ICNNNetwork& network,
                                   pugi::xml_document& doc) {
    std::shared_ptr<const ngraph::Function> function = network.getFunction();
    if (function == nullptr) {
        THROW_IE_EXCEPTION << network.getName() << " does not represent ngraph::Function";
    }

    std::vector<std::shared_ptr<ngraph::Node>> ordered = function->get_ordered_ops();
    pugi::xml_node netXml = doc.append_child("net");
    netXml.append_attribute("name").set_value(network.getName().c_str());

    pugi::xml_node layers = netXml.append_child("layers");
    std::unordered_map<std::shared_ptr<ngraph::Node>, size_t> matching;

    for (size_t i = 0; i < ordered.size(); ++i) {
        matching[ordered[i]] = i;
        const std::shared_ptr<ngraph::Node> node = ordered[i];
        auto params = node->get_rt_info();

        auto layerTypeVariant = params.find(ExecGraphInfoSerialization::LAYER_TYPE);
        if (layerTypeVariant == params.end()) {
            THROW_IE_EXCEPTION << node->get_friendly_name() << " does not define "
                               << ExecGraphInfoSerialization::LAYER_TYPE << " attribute.";
        }
        using VariantString = ngraph::VariantImpl<std::string>;
        auto layerTypeValueStr = std::dynamic_pointer_cast<VariantString>(layerTypeVariant->second);
        IE_ASSERT(layerTypeValueStr != nullptr);
        params.erase(layerTypeVariant);

        pugi::xml_node layer = layers.append_child("layer");
        layer.append_attribute("name").set_value(node->get_friendly_name().c_str());
        layer.append_attribute("type").set_value(layerTypeValueStr->get().c_str());
        layer.append_attribute("id").set_value(i);

        if (!params.empty()) {
            pugi::xml_node data = layer.append_child("data");

            for (const auto& it : params) {
                if (auto strValue = std::dynamic_pointer_cast<VariantString>(it.second))
                    data.append_attribute(it.first.c_str()).set_value(strValue->get().c_str());
            }
        }

        if (node->get_input_size() > 0) {
            pugi::xml_node input = layer.append_child("input");

            for (size_t iport = 0; iport < node->get_input_size(); iport++) {
                const ngraph::Shape & dims = node->get_input_shape(iport);
                pugi::xml_node port = input.append_child("port");

                port.append_attribute("id").set_value(iport);
                for (auto dim : dims) {
                    port.append_child("dim").text().set(dim);
                }
            }
        }
        if (node->get_output_size() > 0 &&
            // ngraph::op::Result still have single output while we should not print it
            !std::dynamic_pointer_cast<ngraph::op::Result>(node)) {
            pugi::xml_node output = layer.append_child("output");

            for (size_t oport = 0; oport < node->get_output_size(); oport++) {
                pugi::xml_node port = output.append_child("port");
                Precision outputPrecision = details::convertPrecision(node->get_output_element_type(oport));

                port.append_attribute("id").set_value(node->get_input_size() + oport);
                port.append_attribute("precision").set_value(outputPrecision.name());

                for (const auto dim : node->get_output_shape(oport)) {
                    port.append_child("dim").text().set(dim);
                }
            }
        }
    }

    pugi::xml_node edges = netXml.append_child("edges");

    for (const auto& ord : ordered) {
        const std::shared_ptr<ngraph::Node> parentNode = ord;

        if (parentNode->get_output_size() > 0) {
            auto itFrom = matching.find(parentNode);
            if (itFrom == matching.end()) {
                THROW_IE_EXCEPTION << "Internal error, cannot find " << parentNode->get_friendly_name()
                                   << " in matching container during serialization of IR";
            }
            for (size_t oport = 0; oport < parentNode->get_output_size(); oport++) {
                ngraph::Output<ngraph::Node> parentPort = parentNode->output(oport);
                for (const auto& childPort : parentPort.get_target_inputs()) {
                    ngraph::Node * childNode = childPort.get_node();
                    for (int iport = 0; iport < childNode->get_input_size(); iport++) {
                        if (childNode->input_value(iport).get_node() == parentPort.get_node()) {
                            auto itTo = matching.find(childNode->shared_from_this());
                            if (itTo == matching.end()) {
                                THROW_IE_EXCEPTION << "Broken edge form layer "
                                                   << parentNode->get_friendly_name() << " to layer "
                                                   << childNode->get_friendly_name()
                                                   << "during serialization of IR";
                            }
                            pugi::xml_node edge = edges.append_child("edge");
                            edge.append_attribute("from-layer").set_value(itFrom->second);
                            edge.append_attribute("from-port").set_value(oport + parentNode->get_input_size());

                            edge.append_attribute("to-layer").set_value(itTo->second);
                            edge.append_attribute("to-port").set_value(iport);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

std::size_t FillXmlDoc(const InferenceEngine::ICNNNetwork& network, pugi::xml_document& doc,
                       const bool execGraphInfoSerialization, const bool dumpWeights) {
    const std::vector<CNNLayerPtr> ordered = TopologicalSort(network);
    pugi::xml_node netXml = doc.append_child("net");
    netXml.append_attribute("name").set_value(network.getName().c_str());

    // no need to print this information for executable graph information serialization because it is not IR.
    if (!execGraphInfoSerialization) {
        netXml.append_attribute("version").set_value("6");
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
            UpdateStdLayerParams(node);
        }

        const auto& params = node->params;
        if (!params.empty()) {
            pugi::xml_node data = layer.append_child(dataName.c_str());

            for (const auto& it : params) {
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
            pugi::xml_node output = layer.append_child("output");
            for (size_t oport = 0; oport < node->outData.size(); oport++) {
                pugi::xml_node port = output.append_child("port");

                port.append_attribute("id").set_value(node->insData.size() + oport);
                port.append_attribute("precision").set_value(node->outData[oport]->getPrecision().name());

                for (const auto dim : node->outData[oport]->getDims()) {
                    port.append_child("dim").text().set(dim);
                }
            }
        }
        if (dumpWeights && !node->blobs.empty()) {
            auto blobsNode = layer.append_child("blobs");
            for (const auto& dataIt : node->blobs) {
                if (!dataIt.second) continue;
                size_t dataSize = dataIt.second->byteSize();
                pugi::xml_node data = blobsNode.append_child(dataIt.first.c_str());
                data.append_attribute("offset").set_value(dataOffset);
                data.append_attribute("size").set_value(dataSize);
                data.append_attribute("precision").set_value(dataIt.second->getTensorDesc().getPrecision().name());

                dataOffset += dataSize;
            }
        }
    }

    pugi::xml_node edges = netXml.append_child("edges");

    for (const auto& ord : ordered) {
        const CNNLayer::Ptr node = ord;

        if (!node->outData.empty()) {
            auto itFrom = matching.find(node);
            if (itFrom == matching.end()) {
                THROW_IE_EXCEPTION << "Internal error, cannot find " << node->name
                                   << " in matching container during serialization of IR";
            }
            for (size_t oport = 0; oport < node->outData.size(); oport++) {
                const DataPtr outData = node->outData[oport];
                for (const auto& inputTo : outData->getInputTo()) {
                    for (int iport = 0; iport < inputTo.second->insData.size(); iport++) {
                        if (inputTo.second->insData[iport].lock() == outData) {
                            auto itTo = matching.find(inputTo.second);
                            if (itTo == matching.end()) {
                                THROW_IE_EXCEPTION << "Broken edge form layer " << node->name << " to layer "
                                                   << inputTo.first << "during serialization of IR";
                            }
                            pugi::xml_node edge = edges.append_child("edge");
                            edge.append_attribute("from-layer").set_value(itFrom->second);
                            edge.append_attribute("from-port").set_value(oport + node->insData.size());

                            edge.append_attribute("to-layer").set_value(itTo->second);
                            edge.append_attribute("to-port").set_value(iport);
                        }
                    }
                }
            }
        }
    }

    // no need to print this info in case of executable graph info serialization
    if (!execGraphInfoSerialization) {
        dataOffset = updatePreProcInfo(network, netXml, dataOffset);
        UpdateStatisticsInfo(network, netXml);
    }

    return dataOffset;
}

void SerializeBlobs(std::ostream& stream, const InferenceEngine::ICNNNetwork& network) {
    const std::vector<CNNLayerPtr> ordered = TopologicalSort(network);
    for (auto&& node : ordered) {
        if (!node->blobs.empty()) {
            for (const auto& dataIt : node->blobs) {
                if (!dataIt.second) continue;
                const char* dataPtr = dataIt.second->buffer().as<char*>();
                size_t dataSize = dataIt.second->byteSize();
                stream.write(dataPtr, dataSize);
                if (!stream.good()) {
                    THROW_IE_EXCEPTION << "Error during writing blob weights";
                }
            }
        }
    }

    InputsDataMap inputInfo;
    network.getInputsInfo(inputInfo);

    for (auto ii : inputInfo) {
        const PreProcessInfo& pp = ii.second->getPreProcess();
        size_t nInChannels = pp.getNumberOfChannels();
        if (nInChannels) {
            for (size_t ch = 0; ch < nInChannels; ch++) {
                const PreProcessChannel::Ptr& preProcessChannel = pp[ch];
                if (preProcessChannel->meanData) {
                    const char* dataPtr = preProcessChannel->meanData->buffer().as<char*>();
                    size_t dataSize = preProcessChannel->meanData->byteSize();
                    stream.write(dataPtr, dataSize);
                    if (!stream.good()) {
                        THROW_IE_EXCEPTION << "Error during writing mean data";
                    }
                }
            }
        }
    }
}

void Serialize(const std::string& xmlPath, const std::string& binPath,
               const InferenceEngine::ICNNNetwork& network) {
    // A flag for serializing executable graph information (not complete IR)
    bool execGraphInfoSerialization = false;
    pugi::xml_document doc;

    if (auto function = network.getFunction()) {
        execGraphInfoSerialization = true;

        // go over all operations and check whether performance stat is set
        for (const auto & op : function->get_ops()) {
            auto & rtInfo = op->get_rt_info();
            if (rtInfo.find(ExecGraphInfoSerialization::PERF_COUNTER) == rtInfo.end()) {
                execGraphInfoSerialization = false;
                break;
            }
        }

        if (execGraphInfoSerialization) {
            FillXmlDocWithExecutionNGraph(network, doc);

            if (!doc.save_file(xmlPath.c_str())) {
                THROW_IE_EXCEPTION << "file '" << xmlPath << "' was not serialized";
            }

            return;
        }
    }

    const std::vector<CNNLayerPtr> ordered = TopologicalSort(network);
    // If first layer has perfCounter parameter set then it's executable graph info serialization.
    // All other layers must also have this parameter set.
    if (ordered[0]->params.find(ExecGraphInfoSerialization::PERF_COUNTER) != ordered[0]->params.end()) {
        execGraphInfoSerialization = true;
        for (const auto& layer : ordered) {
            if (layer->params.find(ExecGraphInfoSerialization::PERF_COUNTER) == layer->params.end()) {
                THROW_IE_EXCEPTION << "Each node must have " << ExecGraphInfoSerialization::PERF_COUNTER
                                   << " parameter set in case of executable graph info serialization";
            }
        }
    }

    bool dumpWeights = !execGraphInfoSerialization & !binPath.empty();
    FillXmlDoc(network, doc, execGraphInfoSerialization, dumpWeights);

    if (!doc.save_file(xmlPath.c_str())) {
        THROW_IE_EXCEPTION << "file '" << xmlPath << "' was not serialized";
    }

    if (dumpWeights) {
        std::ofstream ofsBin;
        ofsBin.open(binPath, std::ofstream::out | std::ofstream::binary);
        if (!ofsBin.is_open()) {
            THROW_IE_EXCEPTION << "File '" << binPath << "' is not opened as out file stream";
        }
        try {
            SerializeBlobs(ofsBin, network);
        } catch (std::exception& e) {
            ofsBin.close();
            throw e;
        }
        ofsBin.close();
        if (!ofsBin.good()) {
            THROW_IE_EXCEPTION << "Error during '" << binPath << "' closing";
        }
    }
}
}  //  namespace Serialization
}  //  namespace InferenceEngine

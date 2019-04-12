// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_util_internal.hpp"
#include "graph_tools.hpp"
#include "details/caseless.hpp"
#include "ie_utils.hpp"
#include "ie_icnn_network_stats.hpp"
#include "details/ie_cnn_network_tools.h"

#include <ie_layers.h>

#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <deque>
#include <string>
#include <cassert>
#include <memory>
#include <utility>
#include <iomanip>

using std::string;

namespace InferenceEngine {

using namespace details;

namespace {
template<typename Visitor>
void groupSubgraphsHelper(const InferenceEngine::CNNLayerPtr& layer,
                          Visitor&& visitor) {
    for (auto&& out : layer->outData) {
        for (auto&& out_link : out->getInputTo()) {
            auto& nextLayer = out_link.second;
            if (nullptr != nextLayer &&
                visitor(layer, nextLayer)) {
                groupSubgraphsHelper(nextLayer, std::forward<Visitor>(visitor));
            }
        }
    }
}
}  // namespace

std::vector<std::vector<CNNLayerPtr> >
groupSubgraphs(ICNNNetwork& network,
               std::function<bool(const CNNLayerPtr&,
                                  const CNNLayerPtr&)> splitter) {
    // TODO splitter std::function is heavy and can be replaced with
    // llvm::function_ref-like lightweight callable when we add one
    std::unordered_set<InferenceEngine::CNNLayerPtr> visitedObjects;
    std::deque<InferenceEngine::CNNLayerPtr> layersToCheck;
    InputsDataMap inputs;
    network.getInputsInfo(inputs);
    for (auto&& input : inputs) {
        auto data = input.second->getInputData();
        for (auto&& to : data->getInputTo()) {
            auto nextLayer = to.second;
            assert(nullptr != nextLayer);
            layersToCheck.push_front(nextLayer);
        }
    }

    std::vector<std::vector<InferenceEngine::CNNLayerPtr>> ret;

    while (!layersToCheck.empty()) {
        auto layer = layersToCheck.back();
        layersToCheck.pop_back();
        if (visitedObjects.find(layer) == visitedObjects.end()) {
            visitedObjects.insert(layer);
            std::vector<InferenceEngine::CNNLayerPtr> subgraph;
            subgraph.push_back(layer);
            groupSubgraphsHelper(layer,
                                 [&](const InferenceEngine::CNNLayerPtr& layer1,
                                     const InferenceEngine::CNNLayerPtr& layer2) {
                if (visitedObjects.find(layer2) == visitedObjects.end()) {
                    if (splitter(layer1, layer2)) {
                        // Layer belongs to different subgraph
                        // Do not add it to visited objects list here,
                        // because we need to visit it during next while iteration
                        layersToCheck.push_front(layer2);
                        return false;
                    } else {
                        // Layer belongs to same subgraph
                        // add it to list
                        subgraph.push_back(layer2);
                        visitedObjects.insert(layer2);
                        return true;
                    }
                }
                return false;
            });
            ret.emplace_back(std::move(subgraph));
        }
    }

    return ret;
}


DataPtr cloneData(const InferenceEngine::Data& source) {
    auto cloned = std::make_shared<InferenceEngine::Data>(source);
    cloned->getCreatorLayer().reset();
    cloned->getInputTo().clear();
    return cloned;
}

namespace {
template<typename T>
CNNLayerPtr layerCloneImpl(const CNNLayer* source) {
    auto layer = dynamic_cast<const T*>(source);
    if (nullptr != layer) {
        auto newLayer = std::make_shared<T>(*layer);
        newLayer->_fusedWith = nullptr;
        newLayer->outData.clear();
        newLayer->insData.clear();
        return std::static_pointer_cast<CNNLayer>(newLayer);
    }
    return nullptr;
}

}  // namespace

CNNLayerPtr clonelayer(const CNNLayer& source) {
    using fptr = CNNLayerPtr (*)(const CNNLayer*);
    // Most derived layers must go first in this list
    static const fptr cloners[] = {
        &layerCloneImpl<BatchNormalizationLayer>,
        &layerCloneImpl<PowerLayer             >,
        &layerCloneImpl<ScaleShiftLayer        >,
        &layerCloneImpl<PReLULayer             >,
        &layerCloneImpl<TileLayer              >,
        &layerCloneImpl<ReshapeLayer           >,
        &layerCloneImpl<CropLayer              >,
        &layerCloneImpl<EltwiseLayer           >,
        &layerCloneImpl<GemmLayer              >,
        &layerCloneImpl<PadLayer               >,
        &layerCloneImpl<GatherLayer            >,
        &layerCloneImpl<StridedSliceLayer      >,
        &layerCloneImpl<ShuffleChannelsLayer   >,
        &layerCloneImpl<DepthToSpaceLayer      >,
        &layerCloneImpl<SpaceToDepthLayer      >,
        &layerCloneImpl<ReverseSequenceLayer   >,
        &layerCloneImpl<SqueezeLayer           >,
        &layerCloneImpl<UnsqueezeLayer         >,
        &layerCloneImpl<RangeLayer             >,
        &layerCloneImpl<FillLayer              >,
        &layerCloneImpl<ExpandLayer            >,
        &layerCloneImpl<ClampLayer             >,
        &layerCloneImpl<ReLULayer              >,
        &layerCloneImpl<SoftMaxLayer           >,
        &layerCloneImpl<GRNLayer               >,
        &layerCloneImpl<MVNLayer               >,
        &layerCloneImpl<NormLayer              >,
        &layerCloneImpl<SplitLayer             >,
        &layerCloneImpl<ConcatLayer            >,
        &layerCloneImpl<FullyConnectedLayer    >,
        &layerCloneImpl<PoolingLayer           >,
        &layerCloneImpl<DeconvolutionLayer     >,
        &layerCloneImpl<ConvolutionLayer       >,
        &layerCloneImpl<TensorIterator         >,
        &layerCloneImpl<RNNSequenceLayer       >,
        &layerCloneImpl<RNNCellBase            >,
        &layerCloneImpl<QuantizeLayer          >,
        &layerCloneImpl<BinaryConvolutionLayer >,
        &layerCloneImpl<WeightableLayer        >,
        &layerCloneImpl<CNNLayer               >
    };
    for (auto cloner : cloners) {
        auto cloned = cloner(&source);
        if (nullptr != cloned) {
            return cloned;
        }
    }
    assert(!"All layers derived from CNNLayer so we must never get here");
    return nullptr;  // Silence "control may reach end of non-void function" warning
}

details::CNNNetworkImplPtr cloneNet(const ICNNNetwork &network) {
    std::vector<CNNLayerPtr> layers;
    details::CNNNetworkIterator i(const_cast<ICNNNetwork *>(&network));
    while (i != details::CNNNetworkIterator()) {
        layers.push_back(*i);
        i++;
    }

    InferenceEngine::ICNNNetworkStats* pstatsSrc = nullptr;
    if (StatusCode::OK != network.getStats(&pstatsSrc, nullptr)) {
        pstatsSrc = nullptr;
    }
    // copy of the network
    details::CNNNetworkImplPtr net = cloneNet(layers, pstatsSrc);
    // going over output layers and duplicatig them:
    OutputsDataMap outputs;
    network.getOutputsInfo(outputs);
    for (auto o : outputs) {
        net->addOutput(o.first);
    }
    net->setPrecision(network.getPrecision());
    net->setName(network.getName());
    net->setTargetDevice(network.getTargetDevice());

    InputsDataMap externalInputsData;
    network.getInputsInfo(externalInputsData);

    InputsDataMap clonedInputs;
    net->getInputsInfo(clonedInputs);
    for (auto &&it : externalInputsData) {
        auto inp = clonedInputs.find(it.first);
        if (inp != clonedInputs.end() && nullptr != inp->second) {
            inp->second->setInputPrecision(it.second->getInputPrecision());
            inp->second->getPreProcess() = it.second->getPreProcess();
        }
    }

    return net;
}


details::CNNNetworkImplPtr cloneNet(const std::vector<CNNLayerPtr>& layers,
                                    const ICNNNetworkStats* networkStats,
                                    std::function<CNNLayerPtr(const CNNLayer&)> layerCloner) {
    // TODO layerCloner std::function is heavy and can be replaced with
    // llvm::function_ref-like lightweight callable when we add one
    auto net = std::make_shared<InferenceEngine::details::CNNNetworkImpl>();

    // Src to cloned data map
    std::unordered_map<InferenceEngine::DataPtr, InferenceEngine::DataPtr> dataMap;
    // Cloned to src data map
    std::unordered_map<InferenceEngine::DataPtr, InferenceEngine::DataPtr> clonedDataMap;
    std::vector<InferenceEngine::DataPtr> clonedDatas;

    auto createDataImpl = [&](const InferenceEngine::DataPtr& data) {
        assert(nullptr != data);
        if (!contains(dataMap, data)) {
            auto clonedData = cloneData(*data);
            dataMap[data] = clonedData;
            clonedDataMap[clonedData] = data;
            clonedDatas.push_back(clonedData);
            net->getData(clonedData->getName()) = clonedData;
            return clonedData;
        }
        return dataMap[data];
    };

    auto cloneLayerImpl = [&](const CNNLayer &srcLayer) {
        CNNLayerPtr clonedLayer = layerCloner(srcLayer);
        clonedLayer->_fusedWith = nullptr;
        // We will need to reconstruct all connections in new graph
        clonedLayer->outData.clear();
        clonedLayer->insData.clear();
        net->addLayer(clonedLayer);
        return clonedLayer;
    };

    for (auto&& srcLayer : layers) {
        CNNLayerPtr clonedLayer = cloneLayerImpl(*srcLayer);
        for (auto&& src : srcLayer->insData) {
            auto data = src.lock();
            auto clonedData = createDataImpl(data);

            string inputName;
            // Find input name
            for (auto&& inp : data->getInputTo()) {
                if (srcLayer == inp.second) {
                    inputName = inp.first;
                    break;
                }
            }
            assert(!inputName.empty());
            clonedData->getInputTo().insert({ inputName, clonedLayer });
            clonedLayer->insData.push_back(clonedData);
        }

        for (auto&& data : srcLayer->outData) {
            auto clonedData = createDataImpl(data);
            clonedData->getCreatorLayer() = clonedLayer;
            clonedLayer->outData.push_back(clonedData);
            for (auto&& inp : data->getInputTo()) {
                auto layer = inp.second;
                // TODO(amalyshe) is it the best place to check priorbox and remove
                // such edge from outputs?
                if (std::find(layers.begin(), layers.end(), layer) == layers.end() &&
                    !(CaselessEq<string>()(layer->type, "priorbox") ||
                      CaselessEq<string>()(layer->type, "PriorBoxClustered"))) {
                    net->addOutput(data->getName());
                    break;
                }
            }
        }
    }

    for (auto&& data : clonedDatas) {
        auto layer = data->getCreatorLayer().lock();
        // create an artificial input layer because logic in some algorithms rely
        // on existence of these layers in the network
        if (nullptr == layer) {
            assert(contains(clonedDataMap, data));
            auto originalData = clonedDataMap[data];
            assert(nullptr != originalData);

            if (auto originalLayer = originalData->creatorLayer.lock()) {
                if (CaselessEq<string>()(originalLayer->type, "input") ||
                    CaselessEq<string>()(originalLayer->type, "const")) {
                    layer = cloneLayerImpl(*originalLayer);
                    layer->outData.push_back(data);
                    data->getCreatorLayer() = layer;
                }
            }

            if (nullptr == layer) {
                LayerParams params;
                params.name = data->getName();
                params.precision = data->getPrecision();
                params.type = "Input";
                layer = std::make_shared<CNNLayer>(params);
                // this place should be transactional
                layer->outData.push_back(data);
                data->getCreatorLayer() = layer;
                net->addLayer(layer);
            }
        }
        if (CaselessEq<string>()(layer->type, "input")) {
            auto input = std::make_shared<InferenceEngine::InputInfo>();
            input->setInputData(data);
            net->setInputInfo(input);
        }
    }

    net->resolveOutput();

    // cloning of statistics
    InferenceEngine::ICNNNetworkStats* pstatsTarget = nullptr;
    if (networkStats != nullptr && !networkStats->isEmpty()) {
        StatusCode st = net->getStats(&pstatsTarget, nullptr);
        if (st == StatusCode::OK && pstatsTarget) {
            pstatsTarget->setNodesStats(networkStats->getNodesStats());
        }
    }

    return net;
}

namespace traverse {

void forward(const CNNLayerPtr& layer, std::deque<InferenceEngine::CNNLayerPtr>& layers) {
    for (const auto& out : layer->outData) {
        for (const auto& out_link : out->getInputTo()) {
            const auto& nextLayer = out_link.second;
            if (nullptr != nextLayer) {
                layers.emplace_back(nextLayer);
            }
        }
    }
}

void backward(const CNNLayerPtr& layer, std::deque<InferenceEngine::CNNLayerPtr>& layers) {
    for (const auto& data : layer->insData) {
        const auto data_ptr = data.lock();
        const auto creatorLayer = data_ptr->creatorLayer.lock();
        if (nullptr != creatorLayer &&
            creatorLayer->type != "Input" &&
            creatorLayer->type != "input" ) {
            layers.emplace_back(creatorLayer);
        }
    }
}

void traverse(InferenceEngine::ICNNNetwork& network,
              std::function<void(InferenceEngine::CNNLayerPtr& layer)> apply,
              std::function<void(const InferenceEngine::CNNLayerPtr& layer, std::deque<InferenceEngine::CNNLayerPtr>& layers)> expand) {
    std::vector<InferenceEngine::CNNLayerPtr> layers;

    InferenceEngine::InputsDataMap inputs;
    network.getInputsInfo(inputs);
    for (const auto& input : inputs) {
        const auto data = input.second->getInputData();
        for (const auto& to : data->getInputTo()) {
            const auto nextLayer = to.second;
            assert(nullptr != nextLayer);
            layers.emplace_back(nextLayer);
        }
    }

    traverse(layers, apply, expand);
}

}  // namespace traverse


struct NodePrinter {
    enum FILL_COLOR { DATA, SUPPORTED_LAYER, UNSOPPORTED_LAYER };

    std::unordered_set<InferenceEngine::Data*> printed_data;
    std::unordered_set<InferenceEngine::CNNLayer*> printed_layers;
    std::ostream &out;

    printer_callback layer_cb;

    explicit NodePrinter(std::ostream &os, printer_callback cb)
        : out(os), layer_cb(std::move(cb)) {}

    bool isPrinted(const CNNLayerPtr &layer) {
        return static_cast<bool>(printed_layers.count(layer.get()));
    }

    bool isPrinted(const DataPtr &datum) {
        return static_cast<bool>(printed_data.count(datum.get()));
    }

    string colorToStr(FILL_COLOR color) {
        switch (color) {
            case DATA :
                return "#FCF6E3";
            case SUPPORTED_LAYER:
                return "#D9EAD3";
            case UNSOPPORTED_LAYER:
                return "#F4CCCC";
            default:
                return "#FFFFFF";
        }
    }

    string formatSize_(const std::vector<unsigned int>& spatialDims) {
        string result;
        if (spatialDims.empty()) return result;
        result = std::to_string(spatialDims[0]);
        for (auto dim : spatialDims) {
            result += "x" + std::to_string(dim);
        }
        return result;
    }

    string cleanNodeName_(string node_name) const {
        // remove dot and dash symbols from node name. It is incorrectly displayed in xdot
        node_name.erase(remove(node_name.begin(), node_name.end(), '.'), node_name.end());
        std::replace(node_name.begin(), node_name.end(), '-', '_');
        std::replace(node_name.begin(), node_name.end(), ':', '_');
        return node_name;
    }

    void printLayerNode(const CNNLayerPtr &layer) {
        auto node_name = "layer_" + cleanNodeName_(layer->name);
        printed_layers.insert(layer.get());

        ordered_properties printed_properties;

        ordered_properties node_properties = {
            {"shape", "box"},
            {"style", "filled"},
            {"fillcolor", colorToStr(SUPPORTED_LAYER)}
        };

        auto type = layer->type;
        printed_properties.emplace_back("type", type);

        if (type == "Convolution") {
            auto* conv = dynamic_cast<ConvolutionLayer*>(layer.get());

            unsigned int
                depth = conv->_out_depth,
                group = conv->_group;

            printed_properties.emplace_back("kernel size", formatSize_({&(conv->_kernel[0]), &(conv->_kernel[conv->_kernel.size() - 1])}));
            printed_properties.emplace_back("output depth", std::to_string(depth));
            printed_properties.emplace_back("group", std::to_string(group));
            printed_properties.emplace_back("padding begin", formatSize_({&(conv->_padding[0]), &(conv->_padding[conv->_padding.size() - 1])}));
            printed_properties.emplace_back("padding end", formatSize_({&(conv->_pads_end[0]), &(conv->_pads_end[conv->_pads_end.size() - 1])}));
            printed_properties.emplace_back("strides", formatSize_({&(conv->_stride[0]), &(conv->_stride[conv->_stride.size() - 1])}));
            printed_properties.emplace_back("dilations", formatSize_({&(conv->_dilation[0]), &(conv->_dilation[conv->_dilation.size() - 1])}));
        } else if (type == "Pooling") {
            auto* pool = dynamic_cast<PoolingLayer*>(layer.get());

            printed_properties.emplace_back("window size", formatSize_({&(pool->_kernel[0]), &(pool->_kernel[pool->_kernel.size() - 1])}));
            printed_properties.emplace_back("padding begin", formatSize_({&(pool->_padding[0]), &(pool->_padding[pool->_padding.size() - 1])}));
            printed_properties.emplace_back("padding end", formatSize_({&(pool->_pads_end[0]), &(pool->_pads_end[pool->_pads_end.size() - 1])}));
            printed_properties.emplace_back("strides", formatSize_({&(pool->_stride[0]), &(pool->_stride[pool->_stride.size() - 1])}));
        } else if (type == "ReLU") {
            auto* relu = dynamic_cast<ReLULayer*>(layer.get());

            float negative_slope = relu->negative_slope;

            if (negative_slope != 0.0f)
                printed_properties.emplace_back("negative_slope", std::to_string(negative_slope));
        } else if (type == "Eltwise") {
            auto* eltwise = dynamic_cast<EltwiseLayer*>(layer.get());

            std::string operation;

            if (eltwise->_operation == EltwiseLayer::Sum)
                operation = "Sum";
            else if (eltwise->_operation == EltwiseLayer::Prod)
                operation = "Prod";
            else if (eltwise->_operation == EltwiseLayer::Max)
                operation = "Max";
            else if (eltwise->_operation == EltwiseLayer::Sub)
                operation = "Sub";
            else if (eltwise->_operation == EltwiseLayer::Min)
                operation = "Min";
            else if (eltwise->_operation == EltwiseLayer::Div)
                operation = "Div";
            else if (eltwise->_operation == EltwiseLayer::Squared_diff)
                operation = "Squared_diff";
            else if (eltwise->_operation == EltwiseLayer::Equal)
                operation = "Equal";
            else if (eltwise->_operation == EltwiseLayer::Not_equal)
                operation = "Not_equal";
            else if (eltwise->_operation == EltwiseLayer::Less)
                operation = "Less";
            else if (eltwise->_operation == EltwiseLayer::Less_equal)
                operation = "Less_equal";
            else if (eltwise->_operation == EltwiseLayer::Greater)
                operation = "Greater";
            else if (eltwise->_operation == EltwiseLayer::Greater_equal)
                operation = "Greater_equal";
            else if (eltwise->_operation == EltwiseLayer::Logical_AND)
                operation = "Logical_AND";
            else if (eltwise->_operation == EltwiseLayer::Logical_OR)
                operation = "Logical_OR";
            else if (eltwise->_operation == EltwiseLayer::Logical_XOR)
                operation = "Logical_XOR";

            printed_properties.emplace_back("operation", operation);
        }

        if (layer_cb != nullptr) {
            layer_cb(layer, printed_properties, node_properties);
        }

        printNode(node_name, layer->name, node_properties, printed_properties);
    }

    void printDataNode(const std::shared_ptr<Data> &data) {
        auto node_name = "data_" + cleanNodeName_(data->getName());
        printed_data.insert(data.get());

        ordered_properties printed_properties;
        ordered_properties node_properties = {
            {"shape", "ellipse"},
            {"style", "filled"},
            {"fillcolor", colorToStr(DATA)}
        };

        std::stringstream dims_ss;
        size_t idx = data->getTensorDesc().getDims().size();
        dims_ss << '[';
        for (auto &dim : data->getTensorDesc().getDims()) {
            dims_ss << dim << ((--idx) != 0u ? ", " : "");
        }
        dims_ss << ']';

        printed_properties.emplace_back("dims", dims_ss.str());
        printed_properties.emplace_back("precision", data->getPrecision().name());

        printNode(node_name, data->name, node_properties, printed_properties);
    }

    void printNode(string const &node_name, const string &node_title,
                   ordered_properties const &node_properties,
                   ordered_properties const &printed_properties) {
        // normalization of names, removing all prohibited symbols like "/"
        string nodeNameN = node_name;
        std::replace(nodeNameN.begin(), nodeNameN.end(), '/', '_');
        string dataNameN = node_title;
        std::replace(dataNameN.begin(), dataNameN.end(), '/', '_');

        out << '\t' << nodeNameN << " [";
        for (auto &node_property : node_properties) {
            out << node_property.first << "=\"" << node_property.second << "\", ";
        }

        out << "label=\"" << node_title;
        for (auto &printed_property : printed_properties) {
            out << "\\n" << printed_property.first << ": " << printed_property.second;
        }
        out << "\"];\n";
    }

    void printEdge(const CNNLayerPtr &from_, const DataPtr &to_, bool reverse) {
        auto from_name = "layer_" + cleanNodeName_(from_->name);
        auto to_name = "data_" + cleanNodeName_(to_->getName());
        std::replace(from_name.begin(), from_name.end(), '/', '_');
        std::replace(to_name.begin(), to_name.end(), '/', '_');
        if (reverse)
            std::swap(from_name, to_name);
        out << '\t' << from_name << " -> " << to_name << ";\n";
    }
};

void saveGraphToDot(InferenceEngine::ICNNNetwork &network, std::ostream &out, printer_callback layer_cb) {
    NodePrinter printer(out, std::move(layer_cb));

    CNNLayerSet inputs;
    for (auto& data : getRootDataObjects(network)) {
        assert(nullptr != data);
        for (auto& l :  data->getInputTo()) {
            inputs.insert(l.second);
        }
    }

    out << "strict digraph Network {\n";
    // Traverse graph and print nodes
    for (const auto &layer : details::CNNNetSortTopologically(network)) {
        printer.printLayerNode(layer);

        // Print output Data Object
        for (auto &dataptr : layer->outData) {
            if (!printer.isPrinted(dataptr)) {
                printer.printDataNode(dataptr);
            }
            printer.printEdge(layer, dataptr, false);
        }

        // Print input Data objects
        for (auto &datum : layer->insData) {
            auto dataptr = datum.lock();
            if (!printer.isPrinted(dataptr)) {
                printer.printDataNode(dataptr);
            }
            // in order not to keep additional set with
            // printed edges, strict keyword for digraph is used
            // to remove duplicate edges
            printer.printEdge(layer, dataptr, true);
        }
    }
    out << "}" << std::endl;
}

std::unordered_set<DataPtr> getRootDataObjects(ICNNNetwork &network) {
    std::unordered_set<DataPtr> ret;
    details::CNNNetworkIterator i(&network);
    while (i != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *i;

        // TODO: Data without creatorLayer
        if (CaselessEq<string>()(layer->type, "input") ||
            CaselessEq<string>()(layer->type, "const")) {
            ret.insert(layer->outData.begin(), layer->outData.end());
        }
        i++;
    }
    return ret;
}

}  // namespace InferenceEngine

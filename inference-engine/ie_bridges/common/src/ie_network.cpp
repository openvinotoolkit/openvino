#include <cpp/ie_cnn_net_reader.h>
#include "../../../src/inference_engine/ie_ir_reader.hpp"
#include "details/ie_cnn_network_tools.h"


#include "ie_network.h"
#include "ie_core.h"
#include "helpers.h"

InferenceEngineBridge::IENetwork::IENetwork(const InferenceEngine::CNNNetwork& cnn_network) : actual(cnn_network) {
    name = actual.getName();
    batch_size = actual.getBatchSize();
    precision = actual.getPrecision().name();
}

InferenceEngineBridge::IENetwork::IENetwork(const std::string &model, const std::string &weights, const bool &ngraph_compatibility = false) {
    if (ngraph_compatibility){
        InferenceEngine::IRReader ir_reader;
        auto ngraph_function = ir_reader.read(model, weights);
        actual = InferenceEngine::CNNNetwork(InferenceEngine::convertFunctionToICNNNetwork(ngraph_function));
    } else {
        InferenceEngine::CNNNetReader net_reader;
        net_reader.ReadNetwork(model);
        net_reader.ReadWeights(weights);
        actual = net_reader.getNetwork();
    }
    name = actual.getName();
    batch_size = actual.getBatchSize();
    precision = actual.getPrecision().name();
}


void InferenceEngineBridge::IENetwork::load_from_buffer(const char *xml, std::size_t xml_size, uint8_t *bin, const std::size_t &bin_size) {
    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(xml, xml_size);
    InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, {bin_size}, InferenceEngine::Layout::C);
    auto weights_blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc, bin, bin_size);
    net_reader.SetWeights(weights_blob);
    name = net_reader.getName();
    actual = net_reader.getNetwork();
    batch_size = actual.getBatchSize();
    precision = actual.getPrecision().name();
}

void InferenceEngineBridge::IENetwork::serialize(const std::string &path_to_xml, const std::string &path_to_bin) {
    actual.serialize(path_to_xml, path_to_bin);
}

void InferenceEngineBridge::IENetwork::setBatch(const size_t &size) {
    actual.setBatchSize(size);
}

void InferenceEngineBridge::IENetwork::reshape(const std::map<std::string, std::vector<size_t>> &input_shapes) {
    actual.reshape(input_shapes);
}
void InferenceEngineBridge::IENetwork::addOutput(const std::string &out_layer, size_t port_id) {
    actual.addOutput(out_layer, port_id);
}


const std::map<std::string, std::map<std::string, std::vector<float>>> InferenceEngineBridge::IENetwork::getStats() {
    InferenceEngine::ICNNNetworkStats *pstats = nullptr;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(((InferenceEngine::ICNNNetwork &) actual).getStats(&pstats, &response));
    auto statsMap = pstats->getNodesStats();
    std::map<std::string, std::map<std::string, std::vector<float>>> map;
    for (const auto &it : statsMap) {
        std::map<std::string, std::vector<float>> stats;
        stats.emplace("min", it.second->_minOutputs);
        stats.emplace("max", it.second->_maxOutputs);
        map.emplace(it.first, stats);
    }
    return map;
}

void InferenceEngineBridge::IENetwork::setStats(const std::map<std::string, std::map<std::string,
        std::vector<float>>> &stats) {
    InferenceEngine::ICNNNetworkStats *pstats = nullptr;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(((InferenceEngine::ICNNNetwork &) actual).getStats(&pstats, &response));
    std::map<std::string, InferenceEngine::NetworkNodeStatsPtr> newNetNodesStats;
    for (const auto &it : stats) {
        InferenceEngine::NetworkNodeStatsPtr nodeStats = InferenceEngine::NetworkNodeStatsPtr(
                new InferenceEngine::NetworkNodeStats());
        newNetNodesStats.emplace(it.first, nodeStats);
        nodeStats->_minOutputs = it.second.at("min");
        nodeStats->_maxOutputs = it.second.at("max");
    }
    pstats->setNodesStats(newNetNodesStats);
}

const std::vector<std::pair<std::string, InferenceEngineBridge::IENetLayer>> InferenceEngineBridge::IENetwork::getLayers() {
    std::vector<std::pair<std::string, InferenceEngineBridge::IENetLayer>> result;
    std::vector<InferenceEngine::CNNLayerPtr> sorted_layers = InferenceEngine::details::CNNNetSortTopologically(actual);
    for (const auto &layer : sorted_layers) {
        InferenceEngineBridge::IENetLayer layer_info;

        layer_info.layer_ptr = layer;
        layer_info.network_ptr = actual;
        layer_info.name = layer->name;
        layer_info.type = layer->type;
        layer_info.precision = layer->precision.name();
        layer_info.params = layer->params;
        layer_info.affinity = layer->affinity;
        std::vector<std::string> parents;
        for (const auto &i : layer->insData) {
            auto data = i.lock();
            if (data) {
                parents.emplace_back(data->getName());
            }
        }
        layer_info.parents = parents;
        std::vector<std::string> children;
        for (const auto &data : layer->outData) {
            auto inputTo = data->getInputTo();
            for (auto layer_iter : inputTo) {
                InferenceEngine::CNNLayerPtr layer_in_data = layer_iter.second;
                if (!layer_in_data) {
                    THROW_IE_EXCEPTION << "Layer which takes data " << data->getName() << " is nullptr";
                }
                children.emplace_back(layer_in_data->name);
            }
        }
        layer_info.children = children;
        const InferenceEngine::TensorDesc &inputTensorDesc = layer->outData[0]->getTensorDesc();
        for (const auto &it : layout_map) {
            if (it.second == inputTensorDesc.getLayout()) {
                layer_info.layout = it.first;
            }
        }
        auto dims = inputTensorDesc.getDims();
        std::string string_dims = "";
        for (const auto &it : dims) {
            string_dims += std::to_string(it) + " ";
        }
        string_dims = string_dims.substr(0, string_dims.size() - 1);
        layer_info.shape = string_dims;
        result.emplace_back(std::make_pair(layer->name, layer_info));
    }
    return result;
}
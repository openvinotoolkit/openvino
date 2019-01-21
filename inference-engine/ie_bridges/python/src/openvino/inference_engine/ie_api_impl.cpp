// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ie_api_impl.hpp"
#include "hetero/hetero_plugin_config.hpp"
#include "ie_iinfer_request.hpp"
#include "details/ie_cnn_network_tools.h"

std::map<std::string, InferenceEngine::Precision> precision_map = {{"FP32", InferenceEngine::Precision::FP32},
                                                                   {"FP16", InferenceEngine::Precision::FP16},
                                                                   {"Q78",  InferenceEngine::Precision::Q78},
                                                                   {"I32",  InferenceEngine::Precision::I32},
                                                                   {"I16",  InferenceEngine::Precision::I16},
                                                                   {"I8",   InferenceEngine::Precision::I8},
                                                                   {"U16",  InferenceEngine::Precision::U16},
                                                                   {"U8",   InferenceEngine::Precision::U8}};

std::map<std::string, InferenceEngine::Layout> layout_map = {{"ANY",     InferenceEngine::Layout::ANY},
                                                             {"NCHW",    InferenceEngine::Layout::NCHW},
                                                             {"NHWC",    InferenceEngine::Layout::NHWC},
                                                             {"OIHW",    InferenceEngine::Layout::OIHW},
                                                             {"C",       InferenceEngine::Layout::C},
                                                             {"CHW",     InferenceEngine::Layout::CHW},
                                                             {"HW",      InferenceEngine::Layout::HW},
                                                             {"NC",      InferenceEngine::Layout::NC},
                                                             {"CN",      InferenceEngine::Layout::CN},
                                                             {"BLOCKED", InferenceEngine::Layout::BLOCKED}};
#define stringify(name) # name
#define IE_CHECK_CALL(expr) {                       \
    auto ret = (expr);                              \
    if (ret != InferenceEngine::StatusCode::OK) {   \
        THROW_IE_EXCEPTION << response.msg;         \
    }                                               \
}                                                   \


InferenceEnginePython::IENetwork::IENetwork(const std::string &model, const std::string &weights) {
    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(model);
    net_reader.ReadWeights(weights);
    name = net_reader.getName();
    actual = net_reader.getNetwork();
    batch_size = actual.getBatchSize();
}

void InferenceEnginePython::IENetwork::serialize(const std::string &path_to_xml, const std::string &path_to_bin) {
    actual.serialize(path_to_xml, path_to_bin);
}

const std::vector<std::pair<std::string, InferenceEnginePython::IENetLayer>>
InferenceEnginePython::IENetwork::getLayers() {
    std::vector<std::pair<std::string, InferenceEnginePython::IENetLayer>> result;
    std::vector<InferenceEngine::CNNLayerPtr> sorted_layers = InferenceEngine::details::CNNNetSortTopologically(actual);
    for (const auto &layer : sorted_layers) {
        InferenceEnginePython::IENetLayer layer_info;

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
                    THROW_IE_EXCEPTION << "Layer which takes data " << data->name << " is nullptr";
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

const std::map<std::string, InferenceEnginePython::InputInfo> InferenceEnginePython::IENetwork::getInputs() {
    std::map<std::string, InferenceEnginePython::InputInfo> inputs;
    const InferenceEngine::InputsDataMap &inputsInfo = actual.getInputsInfo();
    for (auto &in : inputsInfo) {
        InferenceEnginePython::InputInfo info;
        info.actual = *in.second;
        const InferenceEngine::TensorDesc &inputTensorDesc = in.second->getTensorDesc();
        info.dims = inputTensorDesc.getDims();
        for (auto it : precision_map)
            if (it.second == in.second->getPrecision())
                info.precision = it.first;
        for (auto it : layout_map)
            if (it.second == in.second->getLayout())
                info.layout = it.first;
        inputs[in.first] = info;
    }
    return inputs;
}

const std::map<std::string, InferenceEnginePython::OutputInfo> InferenceEnginePython::IENetwork::getOutputs() {
    std::map<std::string, InferenceEnginePython::OutputInfo> outputs;
    const InferenceEngine::OutputsDataMap &outputsInfo = actual.getOutputsInfo();
    for (auto &out : outputsInfo) {
        InferenceEnginePython::OutputInfo info;
        info.actual = out.second;
        const InferenceEngine::TensorDesc &inputTensorDesc = out.second->getTensorDesc();
        info.dims = inputTensorDesc.getDims();
        for (auto it : precision_map)
            if (it.second == out.second->getPrecision())
                info.precision = it.first;
        for (auto it : layout_map)
            if (it.second == out.second->getLayout())
                info.layout = it.first;
        outputs[out.first] = info;
    }
    return outputs;
}

void
InferenceEnginePython::IENetwork::addOutputs(const std::vector<std::string> &out_layers, const std::string &precision) {
    for (auto &&l : out_layers) {
        InferenceEngine::OutputsDataMap outputsDataMap = actual.getOutputsInfo();
        if (outputsDataMap.find(l) != outputsDataMap.end()) {
            continue;
        }
        InferenceEngine::CNNLayerPtr cnnLayer = actual.getLayerByName(l.c_str());
        std::vector<InferenceEngine::DataPtr> outData = cnnLayer->outData;
        if (outData.size() != 1) {
            std::cout << "Layer " << l << " has " << outData.size() << " output blobs and can not be set as output."
                      << std::endl;
            continue;
        }
        actual.addOutput(l);
        InferenceEngine::OutputsDataMap outputsDataMapUpd = actual.getOutputsInfo();
        outputsDataMapUpd[l]->setPrecision(precision_map[precision]);
    }
}

void InferenceEnginePython::IENetwork::setBatch(const size_t size) {
    actual.setBatchSize(size);
}

void InferenceEnginePython::IENetwork::reshape(const std::map<std::string, std::vector<size_t>> &input_shapes) {
    actual.reshape(input_shapes);
}

const std::map<std::string, std::map<std::string, std::vector<float>>> InferenceEnginePython::IENetwork::getStats() {
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

void
InferenceEnginePython::IENetwork::setStats(
        const std::map<std::string, std::map<std::string, std::vector<float>>> &stats) {
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

void InferenceEnginePython::InputInfo::setPrecision(std::string precision) {
    actual.setPrecision(precision_map[precision]);
}

void InferenceEnginePython::InputInfo::setLayout(std::string layout) {
    actual.setLayout(layout_map[layout]);
}

void InferenceEnginePython::OutputInfo::setPrecision(std::string precision) {
    actual->setPrecision(precision_map[precision]);
}

InferenceEnginePython::IEPlugin::IEPlugin(const std::string &device, const std::vector<std::string> &plugin_dirs) {
    InferenceEngine::PluginDispatcher dispatcher{plugin_dirs};
    actual = dispatcher.getPluginByDevice(device);
    const InferenceEngine::Version *pluginVersion;
    actual->GetVersion(pluginVersion);
    version = std::to_string(pluginVersion->apiVersion.major) + ".";
    version += std::to_string(pluginVersion->apiVersion.minor) + ".";
    version += pluginVersion->buildNumber;
    device_name = device;
}

void InferenceEnginePython::IEPlugin::setInitialAffinity(const InferenceEnginePython::IENetwork &net) {
    InferenceEngine::HeteroPluginPtr hetero_plugin(actual);
    InferenceEngine::ResponseDesc response;
    auto &network = net.actual;
    IE_CHECK_CALL(hetero_plugin->SetAffinity(network, {}, &response));
}

std::set<std::string> InferenceEnginePython::IEPlugin::queryNetwork(const InferenceEnginePython::IENetwork &net) {
    const InferenceEngine::CNNNetwork &network = net.actual;
    InferenceEngine::QueryNetworkResult queryRes;
    actual->QueryNetwork(network, queryRes);
    return queryRes.supportedLayers;
}


void InferenceEnginePython::IENetLayer::setAffinity(const std::string &target_affinity) {
    layer_ptr->affinity = target_affinity;
}

void InferenceEnginePython::IENetLayer::setParams(const std::map<std::string, std::string> &params_map) {
    layer_ptr->params = params_map;
}

std::map<std::string, InferenceEngine::Blob::Ptr> InferenceEnginePython::IENetLayer::getWeights() {
    auto w_layer = std::dynamic_pointer_cast<InferenceEngine::WeightableLayer>(layer_ptr);
    // IF current layer is weightable gather weights and biases from casted WeightableLayer and all other blobs
    // considered as custom and gathered from blobs field pf CNNLayer.
    std::map<std::string, InferenceEngine::Blob::Ptr> weights;
    if (w_layer != nullptr) {
        if (w_layer->_weights != nullptr) {
            weights["weights"] = w_layer->_weights;
        }
        if (w_layer->_biases != nullptr) {
            weights["biases"] = w_layer->_biases;
        }
        for (auto it : w_layer->blobs) {
            if (it.first == "weights" || it.first == "biases") {
                continue;
            }
            weights[it.first] = it.second;
        }
    } else {
        // Otherwise all layer's blobs are considered as custom and gathered from CNNLayer
        std::map<std::string, InferenceEngine::Blob::Ptr> map_placeholder;
        weights = map_placeholder;  // If layer has no blobs it should not be missed from weights map
        for (auto it : layer_ptr->blobs) {
            weights[it.first] = it.second;
        }
    }
    return weights;
}

void InferenceEnginePython::IENetLayer::setPrecision(std::string precision) {
    layer_ptr->precision = precision_map[precision];
}

void InferenceEnginePython::IEPlugin::addCpuExtension(const std::string &extension_path) {
    InferenceEngine::ResponseDesc response;
    auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(extension_path);
    auto extension = std::dynamic_pointer_cast<InferenceEngine::IExtension>(extension_ptr);
    IE_CHECK_CALL(actual->AddExtension(extension, &response))
}

std::unique_ptr<InferenceEnginePython::IEExecNetwork>
InferenceEnginePython::IEPlugin::load(const InferenceEnginePython::IENetwork &net,
                                      int num_requests,
                                      const std::map<std::string, std::string> &config) {
    InferenceEngine::ResponseDesc response;
    auto exec_network = InferenceEnginePython::make_unique<InferenceEnginePython::IEExecNetwork>(net.name,
                                                                                                 num_requests);

    IE_CHECK_CALL(actual->LoadNetwork(exec_network->actual, net.actual, config, &response))

    for (size_t i = 0; i < num_requests; ++i) {
        InferRequestWrap &infer_request = exec_network->infer_requests[i];
        IE_CHECK_CALL(exec_network->actual->CreateInferRequest(infer_request.request_ptr, &response))
    }

    return exec_network;
}

void InferenceEnginePython::IEPlugin::setConfig(const std::map<std::string, std::string> &config) {
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(actual->SetConfig(config, &response))
}

InferenceEnginePython::IEExecNetwork::IEExecNetwork(const std::string &name, size_t num_requests) :
        infer_requests(num_requests), name(name) {
}

void InferenceEnginePython::IEExecNetwork::infer() {
    InferenceEngine::ResponseDesc response;
    InferRequestWrap &request = infer_requests[0];
    request.request_ptr->Infer(&response);
}


void InferenceEnginePython::InferRequestWrap::getBlobPtr(const std::string &blob_name, InferenceEngine::Blob::Ptr &blob_ptr)
{
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(request_ptr->GetBlob(blob_name.c_str(), blob_ptr, &response));
}


void InferenceEnginePython::InferRequestWrap::setBatch(int size) {
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(request_ptr->SetBatch(size, &response));
}

void InferenceEnginePython::InferRequestWrap::infer() {
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(request_ptr->Infer(&response));
}

void InferenceEnginePython::InferRequestWrap::infer_async() {
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(request_ptr->StartAsync(&response));
}

int InferenceEnginePython::InferRequestWrap::wait(int64_t timeout) {
    InferenceEngine::ResponseDesc responseDesc;
    InferenceEngine::StatusCode code = request_ptr->Wait(timeout, &responseDesc);
    return static_cast<int >(code);
}

std::map<std::string, InferenceEnginePython::ProfileInfo>
InferenceEnginePython::InferRequestWrap::getPerformanceCounts() {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perf_counts;
    InferenceEngine::ResponseDesc response;
    request_ptr->GetPerformanceCounts(perf_counts, &response);
    std::map<std::string, InferenceEnginePython::ProfileInfo> perf_map;

    for (auto it : perf_counts) {
        InferenceEnginePython::ProfileInfo profile_info;
        switch (it.second.status) {
            case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
                profile_info.status = "EXECUTED";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
                profile_info.status = "NOT_RUN";
                break;
            case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
                profile_info.status = "OPTIMIZED_OUT";
                break;
            default:
                profile_info.status = "UNKNOWN";
        }
        profile_info.exec_type = it.second.exec_type;
        profile_info.layer_type = it.second.layer_type;
        profile_info.cpu_time = it.second.cpu_uSec;
        profile_info.real_time = it.second.realTime_uSec;
        perf_map[it.first] = profile_info;
    }
    return perf_map;
}

std::string InferenceEnginePython::get_version() {
    auto version = InferenceEngine::GetInferenceEngineVersion();
    std::string version_str = std::to_string(version->apiVersion.major) + ".";
    version_str += std::to_string(version->apiVersion.minor) + ".";
    version_str += version->buildNumber;
    return version_str;
}

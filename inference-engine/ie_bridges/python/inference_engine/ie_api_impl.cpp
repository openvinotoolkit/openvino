// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_api_impl.hpp"
#include "hetero/hetero_plugin_config.hpp"
#include "ie_iinfer_request.hpp"
#define stringify( name ) # name
#define IE_CHECK_CALL(expr) {                       \
    auto ret = (expr);                              \
    if (ret != InferenceEngine::StatusCode::OK) {   \
        THROW_IE_EXCEPTION << response.msg;         \
    }                                               \
}                                                   \

InferenceEnginePython::IENetwork InferenceEnginePython::IENetReader::read(std::string const &model,
                                                                     std::string const &weights)
{
    InferenceEngine::CNNNetReader net_reader;
    net_reader.ReadNetwork(model);
    net_reader.ReadWeights(weights);

    const std::string &net_name = net_reader.getName();
    std::map<std::string, std::vector<size_t>> inputs;
    const InferenceEngine::InputsDataMap &inputsInfo = net_reader.getNetwork().getInputsInfo();
    for (auto &item : inputsInfo)
    {
        const InferenceEngine::TensorDesc &inputTensorDesc = item.second->getTensorDesc();
        InferenceEngine::SizeVector dims = inputTensorDesc.getDims();
        inputs[item.first] = dims;
    }

    // TODO: store output shapes for each output
    std::vector<std::string> outputs;
    const InferenceEngine::OutputsDataMap &outputsInfo = net_reader.getNetwork().getOutputsInfo();
    for (auto &item : outputsInfo)
    {
        outputs.push_back(item.first);
    }
    InferenceEngine::CNNNetwork network = net_reader.getNetwork();
    std::size_t batch_size = network.getBatchSize();
    return {network, net_name, batch_size, inputs, outputs};
}

std::map<std::string, InferenceEnginePython::IENetLayer> InferenceEnginePython::IENetwork::getLayers()
{
    std::map<std::string, InferenceEnginePython::IENetLayer> result;
    std::unordered_set<std::string> visisted;
    const InferenceEngine::InputsDataMap &networkInputs = actual.getInputsInfo();

    using CNNLayerPtrCref = const InferenceEngine::CNNLayerPtr &;
    std::function<void(CNNLayerPtrCref)> DFS = [&](CNNLayerPtrCref layer) {
        InferenceEnginePython::IENetLayer layer_info;
        /* Assumes no cycles in graph */
        for (auto &od : layer->outData)
        {
            for (auto nl : od->getInputTo())
            {
                auto i = visisted.find(nl.second->name);
                if (i != visisted.end())
                {
                    continue;
                }
                DFS(nl.second);
            }
        }
        visisted.emplace(layer->name);
        layer_info.layer_ptr = layer;
        layer_info.name = layer->name;
        layer_info.type = layer->type;
        std::string precision = layer->precision.name();
        layer_info.precision = precision;
        layer_info.params = layer->params;
        layer_info.affinity = layer->affinity;
        result[layer->name] = layer_info;
    };

    std::set<InferenceEngine::CNNLayerPtr> inputs;
    for (auto input : networkInputs) {
        for (auto l : input.second->getInputData()->inputTo) {
            inputs.insert(l.second);
        }
    }

    for (auto &layer : inputs)
    {
        DFS(layer);
    }

    return result;

}

void InferenceEnginePython::IENetwork::addOutputs(const std::vector<std::string> & out_layers, const std::string &precision)
{
    std::map <std::string,InferenceEngine::Precision> precision_map = {{"fp32", InferenceEngine::Precision::FP32},
                                                                       {"fp16", InferenceEngine::Precision::FP16},
                                                                       {"q78", InferenceEngine::Precision::Q78},
                                                                       {"i32",  InferenceEngine::Precision::I32},
                                                                       {"i16",  InferenceEngine::Precision::I16},
                                                                       {"i8",  InferenceEngine::Precision::I8},
                                                                       {"u16",  InferenceEngine::Precision::U16},
                                                                       {"u8",  InferenceEngine::Precision::U8}};
    for (auto && l : out_layers)
    {
        InferenceEngine::OutputsDataMap outputsDataMap = actual.getOutputsInfo();
        if (outputsDataMap.find(l) != outputsDataMap.end())
        {
            continue;
        }
        InferenceEngine::CNNLayerPtr cnnLayer = actual.getLayerByName(l.c_str());
        std::vector<InferenceEngine::DataPtr> outData = cnnLayer->outData;
        if (outData.size() != 1) {
            std::cout << "Layer " << l << " has " << outData.size() << " output blobs and can not be set as output." << std::endl;
            continue;
        }
        actual.addOutput(l);
        InferenceEngine::OutputsDataMap outputsDataMapUpd = actual.getOutputsInfo();
        outputsDataMapUpd[l]->setPrecision(precision_map[precision]);
        outputs.push_back(l);
    }
}

void InferenceEnginePython::IENetwork::setBatch(const size_t size)
{
    actual.setBatchSize(size);
    const InferenceEngine::InputsDataMap &inputsInfo = actual.getInputsInfo();
    for (auto &item : inputsInfo)
    {
        const InferenceEngine::TensorDesc &inputTensorDesc = item.second->getTensorDesc();
        InferenceEngine::SizeVector dims = inputTensorDesc.getDims();
        inputs[item.first] = dims;
    }
}
void InferenceEnginePython::IENetwork::reshape(const std::map<std::string, std::vector<size_t>> & input_shapes){
    actual.reshape(input_shapes);
    const InferenceEngine::InputsDataMap &inputsInfo = actual.getInputsInfo();
    for (auto &item : inputsInfo)
    {
        const InferenceEngine::TensorDesc &inputTensorDesc = item.second->getTensorDesc();
        InferenceEngine::SizeVector dims = inputTensorDesc.getDims();
        inputs[item.first] = dims;
    }

}
InferenceEnginePython::IEPlugin::IEPlugin(const std::string &device, const std::vector<std::string> &plugin_dirs)
{

    InferenceEngine::PluginDispatcher dispatcher{plugin_dirs};
    actual = dispatcher.getPluginByDevice(device);
    const InferenceEngine::Version *pluginVersion;
    actual->GetVersion(pluginVersion);
    version = std::to_string(pluginVersion->apiVersion.major) + ".";
    version += std::to_string(pluginVersion->apiVersion.minor) + ".";
    version += pluginVersion->buildNumber;
    device_name = device;
}

void InferenceEnginePython::IEPlugin::setInitialAffinity(InferenceEnginePython::IENetwork &net)
{
    InferenceEngine::HeteroPluginPtr hetero_plugin(actual);
    InferenceEngine::ResponseDesc response;
    auto &network = net.actual;
    IE_CHECK_CALL(hetero_plugin->SetAffinity(network, {}, &response));
}
std::set<std::string> InferenceEnginePython::IEPlugin::queryNetwork(InferenceEnginePython::IENetwork &net)
{
    InferenceEngine::CNNNetwork &network = net.actual;
    InferenceEngine::QueryNetworkResult queryRes;
    actual->QueryNetwork(network, queryRes);
    return queryRes.supportedLayers;
}


void InferenceEnginePython::IENetLayer::setAffinity(const std::string & target_affinity){
    layer_ptr->affinity = target_affinity;
}

void InferenceEnginePython::IENetLayer::setParams(const std::map<std::string, std::string> & params_map){
    layer_ptr->params = params_map;
}

std::map<std::string, InferenceEngine::Blob::Ptr> InferenceEnginePython::IENetLayer::getWeights(){
    auto w_layer = std::dynamic_pointer_cast<InferenceEngine::WeightableLayer>(layer_ptr);
    // IF current layer is weightable gather weights and biases from casted WeightableLayer and all other blobs
    // considered as custom and gathered from blobs field pf CNNLayer.
    std::map<std::string, InferenceEngine::Blob::Ptr> weights;
    if (w_layer != nullptr){
        if (w_layer->_weights != nullptr){
            weights["weights"] = w_layer->_weights;
        }
        if (w_layer->_biases != nullptr){
            weights["biases"] = w_layer->_biases;
        }
        for (auto it : w_layer->blobs){
            if (it.first == "weights" || it.first == "biases"){
                continue;
            }
            weights[it.first] = it.second;
        }
    }
    // Otherwise all layer's blobs are considered as custom and gathered from CNNLayer
    else {
        std::map<std::string, InferenceEngine::Blob::Ptr> map_placeholder;
        weights = map_placeholder; // If layer has no blobs it should not be missed from weights map
        for (auto it : layer_ptr->blobs){
            weights[it.first] = it.second;
        }
    }
    return weights;
}

void InferenceEnginePython::IEPlugin::addCpuExtension(const std::string &extension_path)
{
    InferenceEngine::ResponseDesc response;
    auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(extension_path);
    auto extension = std::dynamic_pointer_cast<InferenceEngine::IExtension>(extension_ptr);
    IE_CHECK_CALL(actual->AddExtension(extension, &response))
}

std::unique_ptr<InferenceEnginePython::IEExecNetwork>
InferenceEnginePython::IEPlugin::load(InferenceEnginePython::IENetwork &net,
                                      int num_requests,
                                      const std::map<std::string, std::string> &config)
{
    InferenceEngine::ResponseDesc response;
    auto exec_network = InferenceEnginePython::make_unique<InferenceEnginePython::IEExecNetwork>(net.name, num_requests);

    IE_CHECK_CALL(actual->LoadNetwork(exec_network->actual, net.actual, config, &response))
    const InferenceEngine::InputsDataMap &inputs_info = net.actual.getInputsInfo();
    const InferenceEngine::OutputsDataMap &outputs_info = net.actual.getOutputsInfo();


    for (size_t i = 0; i < num_requests; ++i) {
        InferRequestWrap &infer_request = exec_network->infer_requests[i];
        IE_CHECK_CALL(exec_network->actual->CreateInferRequest(infer_request.request_ptr, &response))

        for (const auto& input : inputs_info) {
            infer_request.inputs[input.first] = nullptr;
            infer_request.request_ptr->GetBlob(input.first.c_str(), infer_request.inputs[input.first], &response);
        }
        for (const auto& output : outputs_info) {
            infer_request.request_ptr->GetBlob(output.first.c_str(), infer_request.outputs[output.first], &response);
        }
    }

    return exec_network;
}

void InferenceEnginePython::IEPlugin::setConfig(const std::map<std::string, std::string> & config) {
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(actual->SetConfig(config, &response))
}

InferenceEnginePython::IEExecNetwork::IEExecNetwork(const std::string &name, size_t num_requests) :
    infer_requests(num_requests), name(name)
{
}

void InferenceEnginePython::IEExecNetwork::infer()
{
    InferenceEngine::ResponseDesc response;
    InferRequestWrap &request = infer_requests[0];
    request.request_ptr->Infer(&response);
}


InferenceEngine::Blob::Ptr &InferenceEnginePython::InferRequestWrap::getInputBlob(const std::string &blob_name)
{
    return inputs.at(blob_name);
}

InferenceEngine::Blob::Ptr &InferenceEnginePython::InferRequestWrap::getOutputBlob(const std::string &blob_name)
{
    return outputs.at(blob_name);
}

std::vector<std::string> InferenceEnginePython::InferRequestWrap::getInputsList() {
    std::vector<std::string> inputs_list;
    inputs_list.reserve(inputs.size());
    std::transform(inputs.begin(), inputs.end(), std::back_inserter(inputs_list), [] (InferenceEngine::BlobMap::value_type it) -> std::string {
        return it.first;
    });
    return inputs_list;
}

std::vector<std::string> InferenceEnginePython::InferRequestWrap::getOutputsList() {
    std::vector<std::string> outputs_list;
    outputs_list.reserve(inputs.size());
    std::transform(outputs.begin(), outputs.end(), std::back_inserter(outputs_list), [] (InferenceEngine::BlobMap::value_type it) -> std::string {
        return it.first;
    });
    return outputs_list;
}

void InferenceEnginePython::InferRequestWrap::infer() {
    InferenceEngine::ResponseDesc responseDesc;
    request_ptr->Infer(&responseDesc);
}

void InferenceEnginePython::InferRequestWrap::infer_async() {
    InferenceEngine::ResponseDesc responseDesc;
    request_ptr->StartAsync(&responseDesc);
}

int InferenceEnginePython::InferRequestWrap::wait(int64_t timeout) {
    InferenceEngine::ResponseDesc responseDesc;
    InferenceEngine::StatusCode code = request_ptr->Wait(timeout, &responseDesc);
    return static_cast<int >(code);
}

std::map<std::string, InferenceEnginePython::ProfileInfo> InferenceEnginePython::InferRequestWrap::getPerformanceCounts(){
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perf_counts;
    InferenceEngine::ResponseDesc response;
    request_ptr->GetPerformanceCounts(perf_counts, &response);
    std::map<std::string, InferenceEnginePython::ProfileInfo> perf_map;

    for (auto it : perf_counts){
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

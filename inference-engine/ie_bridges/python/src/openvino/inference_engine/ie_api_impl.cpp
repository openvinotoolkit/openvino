// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
                                                             {"NCDHW",   InferenceEngine::Layout::NCDHW},
                                                             {"BLOCKED", InferenceEngine::Layout::BLOCKED}};
#define stringify(name) # name
#define IE_CHECK_CALL(expr) {                       \
    auto ret = (expr);                              \
    if (ret != InferenceEngine::StatusCode::OK) {   \
        THROW_IE_EXCEPTION << response.msg;         \
    }                                               \
}                                                   \

uint32_t getOptimalNumberOfRequests(const InferenceEngine::IExecutableNetwork::Ptr actual) {
    try {
        InferenceEngine::ResponseDesc response;
        InferenceEngine::Parameter parameter_value;
        IE_CHECK_CALL(actual->GetMetric(METRIC_KEY(SUPPORTED_METRICS), parameter_value, &response));
        auto supported_metrics = parameter_value.as<std::vector<std::string>>();
        std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
        if (std::find(supported_metrics.begin(), supported_metrics.end(), key) != supported_metrics.end()) {
            IE_CHECK_CALL(actual->GetMetric(key, parameter_value, &response));
            if (parameter_value.is<unsigned int>())
                return parameter_value.as<unsigned int>();
            else
                THROW_IE_EXCEPTION << "Unsupported format for " << key << "!"
                                  << " Please specify number of infer requests directly!";
        } else {
            THROW_IE_EXCEPTION << "Can't load network: " << key << " is not supported!"
                               << " Please specify number of infer requests directly!";
        }
    } catch (const std::exception& ex) {
        THROW_IE_EXCEPTION << "Can't load network: " << ex.what()
                           << " Please specify number of infer requests directly!";
    }
}

PyObject* parse_parameter(const InferenceEngine::Parameter & param){
    // Check for std::string
    if (param.is<std::string>()){
        return PyUnicode_FromString(param.as<std::string>().c_str());
    }
        // Check for int
    else if (param.is<int>()) {
        auto val = param.as<int>();
        return PyLong_FromLong((long)val);
    }
        // Check for unsinged int
    else if (param.is<unsigned int>()) {
        auto val = param.as<unsigned int>();
        return PyLong_FromLong((unsigned long)val);
    }
        // Check for float
    else if (param.is<float>()) {
        auto val = param.as<float>();
        return PyFloat_FromDouble((double)val);
    }
        // Check for bool
    else if (param.is<bool>()) {
        auto val = param.as<bool>();
        return val ? Py_True : Py_False;
    }
        // Check for std::vector<std::string>
    else if (param.is<std::vector<std::string>>()) {
        auto val = param.as<std::vector<std::string>>();
        PyObject *list = PyList_New(0);
        for (const auto & it : val){
            PyObject *str_val = PyUnicode_FromString(it.c_str());
            PyList_Append(list, str_val);
        }
        return list;
    }
        // Check for std::vector<int>
    else if (param.is<std::vector<int>>()){
        auto val = param.as<std::vector<int>>();
        PyObject *list = PyList_New(0);
        for (const auto & it : val){
            PyList_Append(list, PyLong_FromLong(it));
        }
        return list;
    }
        // Check for std::vector<unsigned int>
    else if (param.is<std::vector<unsigned int>>()){
        auto val = param.as<std::vector<unsigned int>>();
        PyObject *list = PyList_New(0);
        for (const auto & it : val){
            PyList_Append(list, PyLong_FromLong(it));
        }
        return list;
    }
        // Check for std::vector<float>
    else if (param.is<std::vector<float>>()){
        auto val = param.as<std::vector<float>>();
        PyObject *list = PyList_New(0);
        for (const auto & it : val){
            PyList_Append(list, PyFloat_FromDouble((double)it));
        }
        return list;
    }
        // Check for std::tuple<unsigned int, unsigned int>
    else if (param.is<std::tuple<unsigned int, unsigned int >>()) {
        auto val = param.as<std::tuple<unsigned int, unsigned int >>();
        PyObject *tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long)std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long)std::get<1>(val)));
        return tuple;
    }
        // Check for std::tuple<unsigned int, unsigned int, unsigned int>
    else if (param.is<std::tuple<unsigned int, unsigned int, unsigned int >>()) {
        auto val = param.as<std::tuple<unsigned int, unsigned int, unsigned int >>();
        PyObject *tuple = PyTuple_New(3);
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long)std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long)std::get<1>(val)));
        PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLong((unsigned long)std::get<2>(val)));
        return tuple;
    }
        // Check for std::map<std::string, std::string>
    else if (param.is<std::map<std::string, std::string>>()) {
        auto val = param.as<std::map<std::string, std::string>>();
        PyObject *dict = PyDict_New();
        for (const auto &it : val){
            PyDict_SetItemString(dict, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
        }
        return dict;
    }
        // Check for std::map<std::string, int>
    else if (param.is<std::map<std::string, int>>()) {
        auto val = param.as<std::map<std::string, int>>();
        PyObject *dict = PyDict_New();
        for (const auto &it : val){
            PyDict_SetItemString(dict, it.first.c_str(), PyLong_FromLong((long)it.second));
        }
        return dict;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Failed to convert parameter to Python representation!");
        return (PyObject *) NULL;
    }
}
InferenceEnginePython::IENetwork::IENetwork(const std::string &model, const std::string &weights, bool ngraph_compatibility = false) {
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

InferenceEnginePython::IENetwork::IENetwork(const InferenceEngine::CNNNetwork& cnn_network)
    : actual(cnn_network) {
    name = actual.getName();
    batch_size = actual.getBatchSize();
    precision = actual.getPrecision().name();
}

void InferenceEnginePython::IENetwork::load_from_buffer(const char *xml, size_t xml_size, uint8_t *bin, size_t bin_size) {
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

const std::map<std::string, InferenceEnginePython::InputInfo> InferenceEnginePython::IENetwork::getInputs() {
    std::map<std::string, InferenceEnginePython::InputInfo> inputs;
    const InferenceEngine::InputsDataMap &inputsInfo = actual.getInputsInfo();
    for (auto &in : inputsInfo) {
        InferenceEnginePython::InputInfo info;
        info.actual = in.second;
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
InferenceEnginePython::IENetwork::addOutput(const std::string &out_layer, size_t port_id) {
    actual.addOutput(out_layer, port_id);
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

void InferenceEnginePython::IENetwork::setStats(const std::map<std::string, std::map<std::string,
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

void InferenceEnginePython::InputInfo::setPrecision(std::string precision) {
    actual->setPrecision(precision_map[precision]);
}

void InferenceEnginePython::InputInfo::setLayout(std::string layout) {
    actual->setLayout(layout_map[layout]);
}

void InferenceEnginePython::OutputInfo::setPrecision(std::string precision) {
    actual->setPrecision(precision_map[precision]);
}

InferenceEnginePython::IEPlugin::IEPlugin(const std::string &device, const std::vector<std::string> &plugin_dirs) {
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::PluginDispatcher dispatcher{plugin_dirs};
    actual = dispatcher.getPluginByDevice(device);
    IE_SUPPRESS_DEPRECATED_END
    auto pluginVersion = actual.GetVersion();
    version = std::to_string(pluginVersion->apiVersion.major) + ".";
    version += std::to_string(pluginVersion->apiVersion.minor) + ".";
    version += pluginVersion->buildNumber;
    device_name = device;
}

void InferenceEnginePython::IEPlugin::setInitialAffinity(const InferenceEnginePython::IENetwork &net) {
    InferenceEngine::InferenceEnginePluginPtr hetero_plugin(actual);
    InferenceEngine::QueryNetworkResult queryRes;
    auto &network = net.actual;

    hetero_plugin->QueryNetwork(network, {}, queryRes);

    if (queryRes.rc != InferenceEngine::StatusCode::OK) {
        THROW_IE_EXCEPTION << queryRes.resp.msg;
    }

    for (auto && layer : queryRes.supportedLayersMap) {
        network.getLayerByName(layer.first.c_str())->affinity = layer.second;
    }
}

std::set<std::string> InferenceEnginePython::IEPlugin::queryNetwork(const InferenceEnginePython::IENetwork &net) {
    const InferenceEngine::CNNNetwork &network = net.actual;
    InferenceEngine::QueryNetworkResult queryRes;
    actual.QueryNetwork(network, {}, queryRes);

    std::set<std::string> supportedLayers;
    for (auto && layer : queryRes.supportedLayersMap) {
        supportedLayers.insert(layer.first);
    }

    return supportedLayers;
}


void InferenceEnginePython::IENetLayer::setAffinity(const std::string &target_affinity) {
    layer_ptr->affinity = target_affinity;
}

void InferenceEnginePython::IENetLayer::setParams(const std::map<std::string, std::string> &params_map) {
    layer_ptr->params = params_map;
}

std::vector<InferenceEngine::DataPtr> InferenceEnginePython::IENetLayer::getOutData() {
    return layer_ptr->outData;
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
    auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(extension_path);
    auto extension = std::dynamic_pointer_cast<InferenceEngine::IExtension>(extension_ptr);
    actual.AddExtension(extension);
}

std::unique_ptr<InferenceEnginePython::IEExecNetwork>
InferenceEnginePython::IEPlugin::load(const InferenceEnginePython::IENetwork &net,
                                      int num_requests,
                                      const std::map<std::string, std::string> &config) {
    InferenceEngine::ResponseDesc response;
    auto exec_network = InferenceEnginePython::make_unique<InferenceEnginePython::IEExecNetwork>(net.name,
                                                                                                 num_requests);
    exec_network->actual = actual.LoadNetwork(net.actual, config);

    if (0 == num_requests) {
        num_requests = getOptimalNumberOfRequests(exec_network->actual);
        exec_network->infer_requests.resize(num_requests);
    }

    for (size_t i = 0; i < num_requests; ++i) {
        InferRequestWrap &infer_request = exec_network->infer_requests[i];
        IE_CHECK_CALL(exec_network->actual->CreateInferRequest(infer_request.request_ptr, &response))
    }

    return exec_network;
}

void InferenceEnginePython::IEPlugin::setConfig(const std::map<std::string, std::string> &config) {
    actual.SetConfig(config);
}

InferenceEnginePython::IEExecNetwork::IEExecNetwork(const std::string &name, size_t num_requests) :
        infer_requests(num_requests), name(name) {
}

void InferenceEnginePython::IEExecNetwork::infer() {
    InferRequestWrap &request = infer_requests[0];
    request.infer();
}

InferenceEnginePython::IENetwork InferenceEnginePython::IEExecNetwork::GetExecGraphInfo() {
    InferenceEngine::ResponseDesc response;
    InferenceEngine::ICNNNetwork::Ptr graph;
    IE_CHECK_CALL(actual->GetExecGraphInfo(graph, &response));
    return IENetwork(InferenceEngine::CNNNetwork(graph));
}

PyObject* InferenceEnginePython::IEExecNetwork::getMetric(const std::string &metric_name) {
    InferenceEngine::Parameter parameter;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(actual->GetMetric(metric_name, parameter, &response));
    return parse_parameter(parameter);
}

PyObject* InferenceEnginePython::IEExecNetwork::getConfig(const std::string &metric_name) {
    InferenceEngine::Parameter parameter;
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(actual->GetMetric(metric_name, parameter, &response));
    return parse_parameter(parameter);
}

void InferenceEnginePython::InferRequestWrap::getBlobPtr(const std::string &blob_name,
                                                         InferenceEngine::Blob::Ptr &blob_ptr) {
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(request_ptr->GetBlob(blob_name.c_str(), blob_ptr, &response));
}


void InferenceEnginePython::InferRequestWrap::setBatch(int size) {
    InferenceEngine::ResponseDesc response;
    IE_CHECK_CALL(request_ptr->SetBatch(size, &response));
}

void latency_callback(InferenceEngine::IInferRequest::Ptr request, InferenceEngine::StatusCode code) {
    if (code != InferenceEngine::StatusCode::OK) {
        THROW_IE_EXCEPTION << "Async Infer Request failed with status code " << code;
    }
    InferenceEnginePython::InferRequestWrap *requestWrap;
    InferenceEngine::ResponseDesc dsc;
    request->GetUserData(reinterpret_cast<void **>(&requestWrap), &dsc);
    auto end_time = Time::now();
    auto execTime = std::chrono::duration_cast<ns>(end_time - requestWrap->start_time);
    requestWrap->exec_time = static_cast<double>(execTime.count()) * 0.000001;
    if (requestWrap->user_callback) {
        requestWrap->user_callback(requestWrap->user_data, code);
    }
}

void InferenceEnginePython::InferRequestWrap::setCyCallback(cy_callback callback, void *data) {
    user_callback = callback;
    user_data = data;
}

void InferenceEnginePython::InferRequestWrap::infer() {
    InferenceEngine::ResponseDesc response;
    start_time = Time::now();
    IE_CHECK_CALL(request_ptr->Infer(&response));
    auto end_time = Time::now();
    auto execTime = std::chrono::duration_cast<ns>(end_time - start_time);
    exec_time = static_cast<double>(execTime.count()) * 0.000001;
}


void InferenceEnginePython::InferRequestWrap::infer_async() {
    InferenceEngine::ResponseDesc response;
    start_time = Time::now();
    IE_CHECK_CALL(request_ptr->SetUserData(this, &response));
    request_ptr->SetCompletionCallback(latency_callback);
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
        profile_info.execution_index = it.second.execution_index;
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


InferenceEnginePython::IECore::IECore(const std::string & xmlConfigFile) {
    actual = InferenceEngine::Core(xmlConfigFile);
}

std::map<std::string, InferenceEngine::Version> InferenceEnginePython::IECore::getVersions(const std::string &deviceName) {
    return actual.GetVersions(deviceName);
}

std::unique_ptr<InferenceEnginePython::IEExecNetwork> InferenceEnginePython::IECore::loadNetwork(IENetwork network,
        const std::string & deviceName, const std::map<std::string, std::string> & config, int num_requests){

    InferenceEngine::ResponseDesc response;
    auto exec_network = InferenceEnginePython::make_unique<InferenceEnginePython::IEExecNetwork>(network.name,
                                                                                                 num_requests);
    exec_network->actual = actual.LoadNetwork(network.actual, deviceName, config);

    if (0 == num_requests) {
        num_requests = getOptimalNumberOfRequests(exec_network->actual);
        exec_network->infer_requests.resize(num_requests);
    }

    for (size_t i = 0; i < num_requests; ++i) {
        InferRequestWrap &infer_request = exec_network->infer_requests[i];
        IE_CHECK_CALL(exec_network->actual->CreateInferRequest(infer_request.request_ptr, &response))
    }

    return exec_network;
}

std::map<std::string, std::string> InferenceEnginePython::IECore::queryNetwork(InferenceEnginePython::IENetwork network,
                                                                  const std::string &deviceName,
                                                                  const std::map<std::string, std::string> &config) {
    auto res = actual.QueryNetwork(network.actual, deviceName, config);
    return res.supportedLayersMap;
}

void InferenceEnginePython::IECore::setConfig(const std::map<std::string, std::string> &config,
                                              const std::string &deviceName) {
    actual.SetConfig(config, deviceName);
}

void InferenceEnginePython::IECore::registerPlugin(const std::string & pluginName, const std::string &deviceName) {
    actual.RegisterPlugin(pluginName, deviceName);
}

void InferenceEnginePython::IECore::unregisterPlugin(const std::string & deviceName){
    actual.UnregisterPlugin(deviceName);
}

void InferenceEnginePython::IECore::registerPlugins(const std::string & xmlConfigFile){
    actual.RegisterPlugins(xmlConfigFile);
}

void InferenceEnginePython::IECore::addExtension(const std::string & ext_lib_path, const std::string &deviceName) {
    auto extension_ptr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(ext_lib_path);
    auto extension = std::dynamic_pointer_cast<InferenceEngine::IExtension>(extension_ptr);
    actual.AddExtension(extension, deviceName);
}

std::vector<std::string> InferenceEnginePython::IECore::getAvailableDevices() {
    return actual.GetAvailableDevices();
}

PyObject* InferenceEnginePython::IECore::getMetric(const std::string &deviceName, const std::string &name) {
    InferenceEngine::Parameter param = actual.GetMetric(deviceName, name);
    return parse_parameter(param);
}

PyObject* InferenceEnginePython::IECore::getConfig(const std::string &deviceName, const std::string &name) {
    InferenceEngine::Parameter param = actual.GetConfig(deviceName, name);
    return parse_parameter(param);
}

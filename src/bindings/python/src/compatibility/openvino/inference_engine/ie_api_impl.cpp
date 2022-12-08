// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_api_impl.hpp"

#include "ie_plugin_config.hpp"
#include "openvino/op/util/framework_node.hpp"

const std::string EXPORTED_NETWORK_NAME = "undefined";
std::map<std::string, InferenceEngine::Precision> precision_map = {{"FP32", InferenceEngine::Precision::FP32},
                                                                   {"FP64", InferenceEngine::Precision::FP64},
                                                                   {"FP16", InferenceEngine::Precision::FP16},
                                                                   {"I8", InferenceEngine::Precision::I8},
                                                                   {"I16", InferenceEngine::Precision::I16},
                                                                   {"I32", InferenceEngine::Precision::I32},
                                                                   {"I64", InferenceEngine::Precision::I64},
                                                                   {"U8", InferenceEngine::Precision::U8},
                                                                   {"U16", InferenceEngine::Precision::U16},
                                                                   {"U32", InferenceEngine::Precision::U32},
                                                                   {"U64", InferenceEngine::Precision::U64}};

std::map<std::string, InferenceEngine::Layout> layout_map = {{"ANY", InferenceEngine::Layout::ANY},
                                                             {"NCHW", InferenceEngine::Layout::NCHW},
                                                             {"NHWC", InferenceEngine::Layout::NHWC},
                                                             {"OIHW", InferenceEngine::Layout::OIHW},
                                                             {"C", InferenceEngine::Layout::C},
                                                             {"CHW", InferenceEngine::Layout::CHW},
                                                             {"HW", InferenceEngine::Layout::HW},
                                                             {"NC", InferenceEngine::Layout::NC},
                                                             {"CN", InferenceEngine::Layout::CN},
                                                             {"NCDHW", InferenceEngine::Layout::NCDHW},
                                                             {"BLOCKED", InferenceEngine::Layout::BLOCKED}};
#define stringify(name) #name
#define IE_CHECK_CALL(expr)                           \
    {                                                 \
        auto ret = (expr);                            \
        if (ret != InferenceEngine::StatusCode::OK) { \
            IE_THROW() << response.msg;               \
        }                                             \
    }

static uint32_t getOptimalNumberOfRequests(const InferenceEngine::ExecutableNetwork& actual) {
    try {
        auto parameter_value = actual.GetMetric(METRIC_KEY(SUPPORTED_METRICS));
        auto supported_metrics = parameter_value.as<std::vector<std::string>>();
        const std::string key = METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS);
        if (std::find(supported_metrics.begin(), supported_metrics.end(), key) != supported_metrics.end()) {
            parameter_value = actual.GetMetric(key);
            if (parameter_value.is<unsigned int>())
                return parameter_value.as<unsigned int>();
            else
                IE_THROW() << "Unsupported format for " << key << "!"
                           << " Please specify number of infer requests directly!";
        } else {
            IE_THROW() << "Can't load network: " << key << " is not supported!"
                       << " Please specify number of infer requests directly!";
        }
    } catch (const std::exception& ex) {
        IE_THROW() << "Can't load network: " << ex.what() << " Please specify number of infer requests directly!";
    }
}

static PyObject* parse_parameter(const InferenceEngine::Parameter& param) {
    // Check for std::string
    if (param.is<std::string>()) {
        return PyUnicode_FromString(param.as<std::string>().c_str());
    }
    // Check for int
    else if (param.is<int>()) {
        auto val = param.as<int>();
        return PyLong_FromLong((long)val);
    }
    // Check for unsigned int
    else if (param.is<unsigned int>()) {
        auto val = param.as<unsigned int>();
        return PyLong_FromLong((unsigned long)val);
    }
    // Check for uint64_t
    else if (param.is<uint64_t>()) {
        auto val = param.as<uint64_t>();
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
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyObject* str_val = PyUnicode_InternFromString(it.c_str());
            PyList_Append(list, str_val);
        }
        return list;
    }
    // Check for std::vector<int>
    else if (param.is<std::vector<int>>()) {
        auto val = param.as<std::vector<int>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyList_Append(list, PyLong_FromLong(it));
        }
        return list;
    }
    // Check for std::vector<unsigned int>
    else if (param.is<std::vector<unsigned int>>()) {
        auto val = param.as<std::vector<unsigned int>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyList_Append(list, PyLong_FromLong(it));
        }
        return list;
    }
    // Check for std::vector<float>
    else if (param.is<std::vector<float>>()) {
        auto val = param.as<std::vector<float>>();
        PyObject* list = PyList_New(0);
        for (const auto& it : val) {
            PyList_Append(list, PyFloat_FromDouble((double)it));
        }
        return list;
    }
    // Check for std::tuple<unsigned int, unsigned int>
    else if (param.is<std::tuple<unsigned int, unsigned int>>()) {
        auto val = param.as<std::tuple<unsigned int, unsigned int>>();
        PyObject* tuple = PyTuple_New(2);
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long)std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long)std::get<1>(val)));
        return tuple;
    }
    // Check for std::tuple<unsigned int, unsigned int, unsigned int>
    else if (param.is<std::tuple<unsigned int, unsigned int, unsigned int>>()) {
        auto val = param.as<std::tuple<unsigned int, unsigned int, unsigned int>>();
        PyObject* tuple = PyTuple_New(3);
        PyTuple_SetItem(tuple, 0, PyLong_FromUnsignedLong((unsigned long)std::get<0>(val)));
        PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLong((unsigned long)std::get<1>(val)));
        PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLong((unsigned long)std::get<2>(val)));
        return tuple;
    }
    // Check for std::map<std::string, std::string>
    else if (param.is<std::map<std::string, std::string>>()) {
        auto val = param.as<std::map<std::string, std::string>>();
        PyObject* dict = PyDict_New();
        for (const auto& it : val) {
            PyDict_SetItemString(dict, it.first.c_str(), PyUnicode_FromString(it.second.c_str()));
        }
        return dict;
    }
    // Check for std::map<std::string, int>
    else if (param.is<std::map<std::string, int>>()) {
        auto val = param.as<std::map<std::string, int>>();
        PyObject* dict = PyDict_New();
        for (const auto& it : val) {
            PyDict_SetItemString(dict, it.first.c_str(), PyLong_FromLong((long)it.second));
        }
        return dict;
    } else if (param.is<std::map<InferenceEngine::Precision, float>>()) {
        auto val = param.as<std::map<InferenceEngine::Precision, float>>();
        PyObject* dict = PyDict_New();
        for (const auto& it : val) {
            std::stringstream s;
            s << it.first;
            PyDict_SetItemString(dict, s.str().c_str(), PyFloat_FromDouble((double)it.second));
        }
        return dict;
    } else if (param.is<InferenceEngine::Metrics::DeviceType>()) {
        auto val = param.as<InferenceEngine::Metrics::DeviceType>();
        using namespace InferenceEngine;
        std::stringstream s;
        s << val;
        return PyUnicode_FromString(s.str().c_str());
    } else {
        PyErr_SetString(PyExc_TypeError, "Failed to convert parameter to Python representation!");
        return (PyObject*)NULL;
    }
}

/* FrameworkNodeExtension is a temporary extension that is needed to enable FrameworkNode usage
 * in IRReader for all unknown opsets and operations. To have a connection between Extension and
 * IRReader we register extensions with specific version equal to "framework_node_ext" which
 * triggers FrameworkNode usage
 */
class FrameworkNodeExtension : public InferenceEngine::IExtension {
public:
    void GetVersion(const InferenceEngine::Version*& versionInfo) const noexcept override {
        static InferenceEngine::Version ExtensionDescription = {{1, 0}, "1.0", "framework_node_ext"};

        versionInfo = &ExtensionDescription;
    }

    std::map<std::string, ngraph::OpSet> getOpSets() override {
        std::map<std::string, ngraph::OpSet> opsets;
        ngraph::OpSet opset;
        opset.insert<ov::op::util::FrameworkNode>();
        opsets["util"] = opset;
        return opsets;
    }

    void Unload() noexcept override {}
};

InferenceEnginePython::IENetwork InferenceEnginePython::read_network(std::string path_to_xml, std::string path_to_bin) {
    InferenceEngine::Core core;
    core.AddExtension(std::make_shared<FrameworkNodeExtension>());
    auto net = core.ReadNetwork(path_to_xml, path_to_bin);
    return InferenceEnginePython::IENetwork(std::make_shared<InferenceEngine::CNNNetwork>(net));
}

InferenceEnginePython::IENetwork::IENetwork(const std::shared_ptr<InferenceEngine::CNNNetwork>& cnn_network)
    : actual(cnn_network) {
    if (actual == nullptr)
        IE_THROW() << "IENetwork was not initialized.";
    name = actual->getName();
    batch_size = actual->getBatchSize();
}

InferenceEnginePython::IENetwork::IENetwork(PyObject* network) {
    auto* capsule_ptr = PyCapsule_GetPointer(network, "ngraph_function");
    auto* function_sp = static_cast<std::shared_ptr<ngraph::Function>*>(capsule_ptr);
    if (function_sp == nullptr)
        IE_THROW() << "Cannot create CNNNetwork from capsule! Capsule doesn't "
                      "contain nGraph function!";

    InferenceEngine::CNNNetwork cnnNetwork(*function_sp);
    actual = std::make_shared<InferenceEngine::CNNNetwork>(cnnNetwork);
    name = actual->getName();
    batch_size = actual->getBatchSize();
}

void InferenceEnginePython::IENetwork::serialize(const std::string& path_to_xml, const std::string& path_to_bin) {
    actual->serialize(path_to_xml, path_to_bin);
}

PyObject* InferenceEnginePython::IENetwork::getFunction() {
    const char* py_capsule_name = "ngraph_function";
    auto ngraph_func_ptr = actual->getFunction();
    // create a shared pointer on the heap before putting it in the capsule
    // this secures the lifetime of the object transferred by the capsule
    auto* sp_copy = new std::shared_ptr<const ngraph::Function>(ngraph_func_ptr);

    // a destructor callback that will delete the heap allocated shared_ptr
    // when the capsule is destructed
    auto sp_deleter = [](PyObject* capsule) {
        auto* capsule_ptr = PyCapsule_GetPointer(capsule, "ngraph_function");
        auto* function_sp = static_cast<std::shared_ptr<ngraph::Function>*>(capsule_ptr);
        if (function_sp) {
            delete function_sp;
        }
    };
    if (ngraph_func_ptr) {
        // return PyCapsule_New(&ngraph_func_ptr, py_capsule_name, NULL);
        return PyCapsule_New(sp_copy, py_capsule_name, sp_deleter);
    } else {
        return nullptr;
    }
}

const std::map<std::string, InferenceEngine::InputInfo::Ptr> InferenceEnginePython::IENetwork::getInputsInfo() {
    std::map<std::string, InferenceEngine::InputInfo::Ptr> inputs;
    const InferenceEngine::InputsDataMap& inputsInfo = actual->getInputsInfo();
    for (auto& in : inputsInfo) {
        inputs[in.first] = in.second;
    }
    return inputs;
}

const std::map<std::string, InferenceEngine::DataPtr> InferenceEnginePython::IENetwork::getOutputs() {
    std::map<std::string, InferenceEngine::DataPtr> outputs;
    const InferenceEngine::OutputsDataMap& outputsInfo = actual->getOutputsInfo();
    for (auto& out : outputsInfo) {
        outputs[out.first] = out.second;
    }
    return outputs;
}

std::string InferenceEnginePython::IENetwork::getOVNameForTensor(const std::string& orig_name) {
    return actual->getOVNameForTensor(orig_name);
}

void InferenceEnginePython::IENetwork::addOutput(const std::string& out_layer, size_t port_id) {
    actual->addOutput(out_layer, port_id);
}

void InferenceEnginePython::IENetwork::setBatch(const size_t size) {
    actual->setBatchSize(size);
}

size_t InferenceEnginePython::IENetwork::getBatch() {
    return actual->getBatchSize();
}

void InferenceEnginePython::IENetwork::reshape(const std::map<std::string, std::vector<size_t>>& input_shapes) {
    actual->reshape(input_shapes);
}

InferenceEnginePython::IEExecNetwork::IEExecNetwork(const std::string& name, size_t num_requests)
    : infer_requests(num_requests),
      name(name) {
    request_queue_ptr = std::make_shared<IdleInferRequestQueue>();
}

void InferenceEnginePython::IEExecNetwork::infer() {
    InferRequestWrap& request = infer_requests[0];
    request.infer();
}

InferenceEnginePython::IENetwork InferenceEnginePython::IEExecNetwork::GetExecGraphInfo() {
    return IENetwork(std::make_shared<InferenceEngine::CNNNetwork>(actual->GetExecGraphInfo()));
}

PyObject* InferenceEnginePython::IEExecNetwork::getMetric(const std::string& metric_name) {
    return parse_parameter(actual->GetMetric(metric_name));
}

PyObject* InferenceEnginePython::IEExecNetwork::getConfig(const std::string& name) {
    return parse_parameter(actual->GetConfig(name));
}

void InferenceEnginePython::IEExecNetwork::setConfig(const std::map<std::string, std::string>& config) {
    std::map<std::string, InferenceEngine::Parameter> newConfig;
    for (const auto& item : config) {
        newConfig[item.first] = InferenceEngine::Parameter(item.second);
    }
    actual->SetConfig(newConfig);
}

void InferenceEnginePython::IEExecNetwork::exportNetwork(const std::string& model_file) {
    actual->Export(model_file);
}

std::map<std::string, InferenceEngine::InputInfo::CPtr> InferenceEnginePython::IEExecNetwork::getInputsInfo() {
    InferenceEngine::ConstInputsDataMap inputsDataMap = actual->GetInputsInfo();
    std::map<std::string, InferenceEngine::InputInfo::CPtr> pyInputs;
    for (const auto& item : inputsDataMap) {
        pyInputs[item.first] = item.second;
    }
    return pyInputs;
}

std::map<std::string, InferenceEngine::CDataPtr> InferenceEnginePython::IEExecNetwork::getOutputs() {
    InferenceEngine::ConstOutputsDataMap outputsDataMap = actual->GetOutputsInfo();
    std::map<std::string, InferenceEngine::CDataPtr> pyOutputs;
    for (const auto& item : outputsDataMap) {
        pyOutputs[item.first] = item.second;
    }
    return pyOutputs;
}

std::shared_ptr<InferenceEngine::ExecutableNetwork> InferenceEnginePython::IEExecNetwork::getPluginLink() {
    return actual;
}

void InferenceEnginePython::InferRequestWrap::setBlob(const std::string& blob_name,
                                                      const InferenceEngine::Blob::Ptr& blob_ptr) {
    request_ptr.SetBlob(blob_name.c_str(), blob_ptr);
}

void InferenceEnginePython::InferRequestWrap::setBlob(const std::string& blob_name,
                                                      const InferenceEngine::Blob::Ptr& blob_ptr,
                                                      const InferenceEngine::PreProcessInfo& info) {
    request_ptr.SetBlob(blob_name.c_str(), blob_ptr, info);
}

const InferenceEngine::PreProcessInfo& InferenceEnginePython::InferRequestWrap::getPreProcess(
    const std::string& blob_name) {
    return request_ptr.GetPreProcess(blob_name.c_str());
}

InferenceEngine::Blob::Ptr InferenceEnginePython::InferRequestWrap::getBlobPtr(const std::string& blob_name) {
    return request_ptr.GetBlob(blob_name.c_str());
}

void InferenceEnginePython::InferRequestWrap::setBatch(int size) {
    request_ptr.SetBatch(size);
}

std::vector<InferenceEnginePython::CVariableState> InferenceEnginePython::InferRequestWrap::queryState() {
    auto queryStateVec = request_ptr.QueryState();
    std::vector<InferenceEnginePython::CVariableState> memoryStates;
    for (const auto& state : queryStateVec) {
        InferenceEnginePython::CVariableState st;
        st.variableState = state;
        memoryStates.push_back(st);
    }
    return memoryStates;
}

void InferenceEnginePython::InferRequestWrap::setCyCallback(cy_callback callback, void* data) {
    user_callback = callback;
    user_data = data;
}

void InferenceEnginePython::InferRequestWrap::infer() {
    start_time = Time::now();
    request_ptr.Infer();
    auto end_time = Time::now();
    auto execTime = std::chrono::duration_cast<ns>(end_time - start_time);
    exec_time = static_cast<double>(execTime.count()) * 0.000001;
}

void InferenceEnginePython::InferRequestWrap::infer_async() {
    request_queue_ptr->setRequestBusy(index);
    start_time = Time::now();
    request_ptr.StartAsync();
}

int InferenceEnginePython::InferRequestWrap::wait(int64_t timeout) {
    InferenceEngine::StatusCode code = request_ptr.Wait(timeout);
    if (code != InferenceEngine::RESULT_NOT_READY) {
        request_queue_ptr->setRequestIdle(index);
    }
    return static_cast<int>(code);
}

std::map<std::string, InferenceEnginePython::ProfileInfo>
InferenceEnginePython::InferRequestWrap::getPerformanceCounts() {
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perf_counts = request_ptr.GetPerformanceCounts();
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
    return version->buildNumber;
}

InferenceEnginePython::IECore::IECore(const std::string& xmlConfigFile) {
    actual = InferenceEngine::Core(xmlConfigFile);
}

std::map<std::string, InferenceEngine::Version> InferenceEnginePython::IECore::getVersions(
    const std::string& deviceName) {
    return actual.GetVersions(deviceName);
}

int InferenceEnginePython::IEExecNetwork::wait(int num_requests, int64_t timeout) {
    return request_queue_ptr->wait(num_requests, timeout);
}

int InferenceEnginePython::IEExecNetwork::getIdleRequestId() {
    return request_queue_ptr->getIdleRequestId();
}

int InferenceEnginePython::IdleInferRequestQueue::wait(int num_requests, int64_t timeout) {
    std::unique_lock<std::mutex> lock(mutex);
    if (timeout > 0) {
        if (!cv.wait_for(lock, std::chrono::milliseconds(timeout), [this, num_requests]() {
                return static_cast<int>(idle_ids.size()) >= num_requests;
            }))
            return static_cast<int>(InferenceEngine::StatusCode::RESULT_NOT_READY);
    } else
        cv.wait(lock, [this, num_requests]() {
            return static_cast<int>(idle_ids.size()) >= num_requests;
        });
    return static_cast<int>(InferenceEngine::StatusCode::OK);
}

void InferenceEnginePython::IdleInferRequestQueue::setRequestIdle(int index) {
    std::unique_lock<std::mutex> lock(mutex);
    idle_ids.emplace_back(index);
    cv.notify_all();
}

void InferenceEnginePython::IdleInferRequestQueue::setRequestBusy(int index) {
    std::lock_guard<std::mutex> lock(mutex);
    idle_ids.remove(index);
}

int InferenceEnginePython::IdleInferRequestQueue::getIdleRequestId() {
    std::lock_guard<std::mutex> lock(mutex);
    return idle_ids.size() ? idle_ids.front() : -1;
}

void InferenceEnginePython::IEExecNetwork::createInferRequests(int num_requests) {
    if (0 == num_requests) {
        num_requests = getOptimalNumberOfRequests(*actual);
    }
    infer_requests.resize(num_requests);

    for (int i = 0; i < num_requests; ++i) {
        InferRequestWrap& infer_request = infer_requests[i];
        infer_request.index = i;
        request_queue_ptr->setRequestIdle(i);
        infer_request.request_queue_ptr = request_queue_ptr;
        infer_request.request_ptr = actual->CreateInferRequest();

        infer_request.request_ptr
            .SetCompletionCallback<std::function<void(InferenceEngine::InferRequest r, InferenceEngine::StatusCode)>>(
                [&](InferenceEngine::InferRequest request, InferenceEngine::StatusCode code) {
                    if (code != InferenceEngine::StatusCode::OK) {
                        IE_EXCEPTION_SWITCH(code,
                                            ExceptionType,
                                            InferenceEngine::details::ThrowNow<ExceptionType>{} <<=
                                            std::stringstream{}
                                            << IE_LOCATION
                                            << InferenceEngine::details::ExceptionTraits<ExceptionType>::string());
                    }

                    auto end_time = Time::now();
                    auto execTime = std::chrono::duration_cast<ns>(end_time - infer_request.start_time);
                    infer_request.exec_time = static_cast<double>(execTime.count()) * 0.000001;
                    if (infer_request.user_callback) {
                        infer_request.user_callback(infer_request.user_data, code);
                    }
                    infer_request.request_queue_ptr->setRequestIdle(infer_request.index);
                });
    }
}

InferenceEnginePython::IENetwork InferenceEnginePython::IECore::readNetwork(const std::string& modelPath,
                                                                            const std::string& binPath) {
    InferenceEngine::CNNNetwork net = actual.ReadNetwork(modelPath, binPath);
    return IENetwork(std::make_shared<InferenceEngine::CNNNetwork>(net));
}

InferenceEnginePython::IENetwork InferenceEnginePython::IECore::readNetwork(const std::string& model,
                                                                            const uint8_t* bin,
                                                                            size_t bin_size) {
    InferenceEngine::MemoryBlob::Ptr weights_blob;
    if (bin_size != 0) {
        InferenceEngine::TensorDesc tensorDesc(InferenceEngine::Precision::U8, {bin_size}, InferenceEngine::Layout::C);
        weights_blob = InferenceEngine::make_shared_blob<uint8_t>(tensorDesc);
        weights_blob->allocate();
        memcpy(weights_blob->rwmap().as<uint8_t*>(), bin, bin_size);
    }
    InferenceEngine::CNNNetwork net = actual.ReadNetwork(model, weights_blob);
    return IENetwork(std::make_shared<InferenceEngine::CNNNetwork>(net));
}

std::unique_ptr<InferenceEnginePython::IEExecNetwork> InferenceEnginePython::IECore::loadNetwork(
    IENetwork network,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config,
    int num_requests) {
    auto exec_network =
        InferenceEnginePython::make_unique<InferenceEnginePython::IEExecNetwork>(network.name, num_requests);
    exec_network->actual =
        std::make_shared<InferenceEngine::ExecutableNetwork>(actual.LoadNetwork(*network.actual, deviceName, config));
    exec_network->createInferRequests(num_requests);

    return exec_network;
}

std::unique_ptr<InferenceEnginePython::IEExecNetwork> InferenceEnginePython::IECore::loadNetwork(
    IENetwork network,
    const std::map<std::string, std::string>& config,
    int num_requests) {
    auto exec_network =
        InferenceEnginePython::make_unique<InferenceEnginePython::IEExecNetwork>(network.name, num_requests);
    exec_network->actual =
        std::make_shared<InferenceEngine::ExecutableNetwork>(actual.LoadNetwork(*network.actual, config));
    exec_network->createInferRequests(num_requests);

    return exec_network;
}

std::unique_ptr<InferenceEnginePython::IEExecNetwork> InferenceEnginePython::IECore::loadNetworkFromFile(
    const std::string& modelPath,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config,
    int num_requests) {
    auto exec_network =
        InferenceEnginePython::make_unique<InferenceEnginePython::IEExecNetwork>(modelPath, num_requests);
    exec_network->actual =
        std::make_shared<InferenceEngine::ExecutableNetwork>(actual.LoadNetwork(modelPath, deviceName, config));
    exec_network->createInferRequests(num_requests);

    return exec_network;
}

std::unique_ptr<InferenceEnginePython::IEExecNetwork> InferenceEnginePython::IECore::loadNetworkFromFile(
    const std::string& modelPath,
    const std::map<std::string, std::string>& config,
    int num_requests) {
    auto exec_network =
        InferenceEnginePython::make_unique<InferenceEnginePython::IEExecNetwork>(modelPath, num_requests);
    exec_network->actual = std::make_shared<InferenceEngine::ExecutableNetwork>(actual.LoadNetwork(modelPath, config));
    exec_network->createInferRequests(num_requests);

    return exec_network;
}

std::unique_ptr<InferenceEnginePython::IEExecNetwork> InferenceEnginePython::IECore::importNetwork(
    const std::string& modelFIle,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config,
    int num_requests) {
    auto exec_network =
        InferenceEnginePython::make_unique<InferenceEnginePython::IEExecNetwork>(EXPORTED_NETWORK_NAME, num_requests);
    exec_network->actual =
        std::make_shared<InferenceEngine::ExecutableNetwork>(actual.ImportNetwork(modelFIle, deviceName, config));
    exec_network->createInferRequests(num_requests);

    return exec_network;
}

std::map<std::string, std::string> InferenceEnginePython::IECore::queryNetwork(
    InferenceEnginePython::IENetwork network,
    const std::string& deviceName,
    const std::map<std::string, std::string>& config) {
    auto res = actual.QueryNetwork(*network.actual, deviceName, config);
    return res.supportedLayersMap;
}

void InferenceEnginePython::IECore::setConfig(const std::map<std::string, std::string>& config,
                                              const std::string& deviceName) {
    actual.SetConfig(config, deviceName);
}

void InferenceEnginePython::IECore::registerPlugin(const std::string& pluginName, const std::string& deviceName) {
    actual.RegisterPlugin(pluginName, deviceName);
}

void InferenceEnginePython::IECore::unregisterPlugin(const std::string& deviceName) {
    actual.UnregisterPlugin(deviceName);
}

void InferenceEnginePython::IECore::registerPlugins(const std::string& xmlConfigFile) {
    actual.RegisterPlugins(xmlConfigFile);
}

void InferenceEnginePython::IECore::addExtension(const std::string& ext_lib_path, const std::string& deviceName) {
    auto extension_ptr = std::make_shared<InferenceEngine::Extension>(ext_lib_path);
    auto extension = std::dynamic_pointer_cast<InferenceEngine::IExtension>(extension_ptr);
    actual.AddExtension(extension, deviceName);
}

std::vector<std::string> InferenceEnginePython::IECore::getAvailableDevices() {
    return actual.GetAvailableDevices();
}

PyObject* InferenceEnginePython::IECore::getMetric(const std::string& deviceName, const std::string& name) {
    InferenceEngine::Parameter param = actual.GetMetric(deviceName, name);
    return parse_parameter(param);
}

PyObject* InferenceEnginePython::IECore::getConfig(const std::string& deviceName, const std::string& name) {
    InferenceEngine::Parameter param = actual.GetConfig(deviceName, name);
    return parse_parameter(param);
}

void InferenceEnginePython::CVariableState::reset() {
    variableState.Reset();
}

std::string InferenceEnginePython::CVariableState::getName() {
    return variableState.GetName();
}

InferenceEngine::Blob::Ptr InferenceEnginePython::CVariableState::getState() {
    InferenceEngine::Blob::CPtr c_blob = variableState.GetState();
    return std::const_pointer_cast<InferenceEngine::Blob>(c_blob);
}

void InferenceEnginePython::CVariableState::setState(InferenceEngine::Blob::Ptr state) {
    variableState.SetState(state);
}

const size_t InferenceEnginePython::product(const InferenceEngine::SizeVector& dims) {
    return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>{});
}

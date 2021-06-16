// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_extension.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <ie_core.hpp>
#include <iostream>
#include <iterator>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "Python.h"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

namespace InferenceEnginePython {

struct ProfileInfo {
    std::string status;
    std::string exec_type;
    std::string layer_type;
    int64_t real_time;
    int64_t cpu_time;
    unsigned execution_index;
};

struct CVariableState {
    InferenceEngine::VariableState variableState;
    void reset();
    std::string getName();
    InferenceEngine::Blob::Ptr getState();
    void setState(InferenceEngine::Blob::Ptr state);
};

struct IENetwork {
    std::shared_ptr<InferenceEngine::CNNNetwork> actual;
    std::string name;
    std::size_t batch_size;
    PyObject* getFunction();

    void setBatch(const size_t size);

    size_t getBatch();

    void addOutput(const std::string& out_layer, size_t port_id);

    const std::map<std::string, InferenceEngine::InputInfo::Ptr> getInputsInfo();

    const std::map<std::string, InferenceEngine::DataPtr> getInputs();

    const std::map<std::string, InferenceEngine::DataPtr> getOutputs();

    void reshape(const std::map<std::string, std::vector<size_t>>& input_shapes);

    void serialize(const std::string& path_to_xml, const std::string& path_to_bin);

    void load_from_buffer(const char* xml, size_t xml_size, uint8_t* bin, size_t bin_size);

    IENetwork(const std::string& model, const std::string& weights);

    IENetwork(const std::shared_ptr<InferenceEngine::CNNNetwork>& cnn_network);

    IENetwork(PyObject* network);

    IENetwork() = default;

    void convertToOldRepresentation();

    std::string getOVNameForTensor(const std::string& orig_name);
};

struct IdleInferRequestQueue {
    std::list<size_t> idle_ids;
    std::mutex mutex;
    std::condition_variable cv;

    void setRequestIdle(int index);
    void setRequestBusy(int index);

    int wait(int num_requests, int64_t timeout);

    int getIdleRequestId();

    using Ptr = std::shared_ptr<IdleInferRequestQueue>;
};

struct InferRequestWrap {
    int index;
    using cy_callback = void (*)(void*, int);

    InferenceEngine::InferRequest request_ptr;
    Time::time_point start_time;
    double exec_time;
    cy_callback user_callback;
    void* user_data;
    IdleInferRequestQueue::Ptr request_queue_ptr;

    void infer();

    void infer_async();

    int wait(int64_t timeout);

    void setCyCallback(cy_callback callback, void* data);

    InferenceEngine::Blob::Ptr getBlobPtr(const std::string& blob_name);

    void setBlob(const std::string& blob_name, const InferenceEngine::Blob::Ptr& blob_ptr);

    void setBlob(const std::string& name, const InferenceEngine::Blob::Ptr& data, const InferenceEngine::PreProcessInfo& info);

    void setBatch(int size);

    const InferenceEngine::PreProcessInfo& getPreProcess(const std::string& blob_name);

    std::map<std::string, InferenceEnginePython::ProfileInfo> getPerformanceCounts();

    std::vector<InferenceEnginePython::CVariableState> queryState();
};

struct IEExecNetwork {
    std::shared_ptr<InferenceEngine::ExecutableNetwork> actual;
    std::vector<InferRequestWrap> infer_requests;
    std::string name;
    IdleInferRequestQueue::Ptr request_queue_ptr;

    IEExecNetwork(const std::string& name, size_t num_requests);

    IENetwork GetExecGraphInfo();

    void infer();
    void exportNetwork(const std::string& model_file);

    std::map<std::string, InferenceEngine::InputInfo::CPtr> getInputsInfo();
    std::map<std::string, InferenceEngine::DataPtr> getInputs();
    std::map<std::string, InferenceEngine::CDataPtr> getOutputs();

    PyObject* getMetric(const std::string& metric_name);
    PyObject* getConfig(const std::string& name);

    int wait(int num_requests, int64_t timeout);
    int getIdleRequestId();

    void createInferRequests(int num_requests);

    // binds plugin to InputInfo and Data, so that they can be destroyed before plugin (ussue 28996)
    std::shared_ptr<InferenceEngine::ExecutableNetwork> getPluginLink();
};

struct IECore {
    InferenceEngine::Core actual;
    explicit IECore(const std::string& xmlConfigFile = std::string());
    std::map<std::string, InferenceEngine::Version> getVersions(const std::string& deviceName);
    InferenceEnginePython::IENetwork readNetwork(const std::string& modelPath, const std::string& binPath);
    InferenceEnginePython::IENetwork readNetwork(const std::string& model, const uint8_t* bin, size_t bin_size);
    std::unique_ptr<InferenceEnginePython::IEExecNetwork> loadNetwork(IENetwork network, const std::string& deviceName,
                                                                      const std::map<std::string, std::string>& config, int num_requests);
    std::unique_ptr<InferenceEnginePython::IEExecNetwork> loadNetworkFromFile(const std::string& modelPath, const std::string& deviceName,
                                                                              const std::map<std::string, std::string>& config, int num_requests);
    std::unique_ptr<InferenceEnginePython::IEExecNetwork> importNetwork(const std::string& modelFIle, const std::string& deviceName,
                                                                        const std::map<std::string, std::string>& config, int num_requests);
    std::map<std::string, std::string> queryNetwork(IENetwork network, const std::string& deviceName, const std::map<std::string, std::string>& config);
    void setConfig(const std::map<std::string, std::string>& config, const std::string& deviceName = std::string());
    void registerPlugin(const std::string& pluginName, const std::string& deviceName);
    void unregisterPlugin(const std::string& deviceName);
    void registerPlugins(const std::string& xmlConfigFile);
    void addExtension(const std::string& ext_lib_path, const std::string& deviceName);
    std::vector<std::string> getAvailableDevices();
    PyObject* getMetric(const std::string& deviceName, const std::string& name);
    PyObject* getConfig(const std::string& deviceName, const std::string& name);
};

template <class T>
T* get_buffer(InferenceEngine::Blob& blob) {
    return blob.buffer().as<T*>();
}

template <class T, class... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

std::string get_version();

InferenceEnginePython::IENetwork read_network(std::string path_to_xml, std::string path_to_bin);

};  // namespace InferenceEnginePython

// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "Python.h"

#include <iterator>
#include <string>
#include <utility>
#include <map>
#include <vector>
#include <set>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <chrono>
#include <queue>
#include <condition_variable>
#include <mutex>

#include <ie_extension.h>
#include "inference_engine.hpp"

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

struct IENetwork {
    std::shared_ptr<InferenceEngine::CNNNetwork> actual;
    std::string name;
    std::size_t batch_size;
    std::string precision;
    PyObject* getFunction();

    void setBatch(const size_t size);

    size_t getBatch();

    void addOutput(const std::string &out_layer, size_t port_id);

    const std::vector <InferenceEngine::CNNLayerPtr> getLayers();

    const std::map<std::string, InferenceEngine::DataPtr> getInputs();

    const std::map<std::string, InferenceEngine::DataPtr> getOutputs();

    void reshape(const std::map<std::string, std::vector<size_t>> &input_shapes);

    void serialize(const std::string &path_to_xml, const std::string &path_to_bin);

    void setStats(const std::map<std::string, std::map<std::string, std::vector<float>>> &stats);

    const std::map<std::string, std::map<std::string, std::vector<float>>> getStats();

    void load_from_buffer(const char* xml, size_t xml_size, uint8_t* bin, size_t bin_size);

    IENetwork(const std::string &model, const std::string &weights);

    IENetwork(const std::shared_ptr<InferenceEngine::CNNNetwork>  &cnn_network);

    IENetwork(PyObject* network);

    IENetwork() = default;
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

    InferenceEngine::IInferRequest::Ptr request_ptr;
    Time::time_point start_time;
    double exec_time;
    cy_callback user_callback;
    void *user_data;
    IdleInferRequestQueue::Ptr  request_queue_ptr;

    void infer();

    void infer_async();

    int  wait(int64_t timeout);

    void setCyCallback(cy_callback callback, void *data);

    void getBlobPtr(const std::string &blob_name, InferenceEngine::Blob::Ptr &blob_ptr);

    void setBatch(int size);

    std::map<std::string, InferenceEnginePython::ProfileInfo> getPerformanceCounts();
};


struct IEExecNetwork {
    InferenceEngine::IExecutableNetwork::Ptr actual;
    std::vector<InferRequestWrap> infer_requests;
    std::string name;
    IdleInferRequestQueue::Ptr  request_queue_ptr;

    IEExecNetwork(const std::string &name, size_t num_requests);

    IENetwork GetExecGraphInfo();

    void infer();
    void exportNetwork(const std::string & model_file);

    std::map<std::string, InferenceEngine::DataPtr> getInputs();
    std::map<std::string, InferenceEngine::CDataPtr> getOutputs();

    PyObject* getMetric(const std::string & metric_name);
    PyObject* getConfig(const std::string & name);

    int wait(int num_requests, int64_t timeout);
    int getIdleRequestId();

    void createInferRequests(int num_requests);
};


struct IEPlugin {
    std::unique_ptr<InferenceEnginePython::IEExecNetwork> load(const InferenceEnginePython::IENetwork &net,
                                                               int num_requests,
                                                               const std::map<std::string, std::string> &config);

    std::string device_name;
    std::string version;

    void setConfig(const std::map<std::string, std::string> &);

    void addCpuExtension(const std::string &extension_path);

    void setInitialAffinity(const InferenceEnginePython::IENetwork &net);

    IEPlugin(const std::string &device, const std::vector<std::string> &plugin_dirs);

    IEPlugin() = default;

    std::set<std::string> queryNetwork(const InferenceEnginePython::IENetwork &net);

    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::InferencePlugin actual;
    IE_SUPPRESS_DEPRECATED_END
};

struct IECore {
    InferenceEngine::Core actual;
    explicit IECore(const std::string & xmlConfigFile = std::string());
    std::map<std::string, InferenceEngine::Version> getVersions(const std::string & deviceName);
    InferenceEnginePython::IENetwork readNetwork(const std::string& modelPath, const std::string& binPath);
    InferenceEnginePython::IENetwork readNetwork(const std::string& model, uint8_t *bin, size_t bin_size);
    std::unique_ptr<InferenceEnginePython::IEExecNetwork> loadNetwork(IENetwork network, const std::string & deviceName,
            const std::map<std::string, std::string> & config, int num_requests);
    std::unique_ptr<InferenceEnginePython::IEExecNetwork> importNetwork(const std::string & modelFIle, const std::string & deviceName,
                                                                      const std::map<std::string, std::string> & config, int num_requests);
    std::map<std::string, std::string> queryNetwork(IENetwork network, const std::string & deviceName,
                                       const std::map<std::string, std::string> & config);
    void setConfig(const std::map<std::string, std::string> &config, const std::string & deviceName = std::string());
    void registerPlugin(const std::string & pluginName, const std::string & deviceName);
    void unregisterPlugin(const std::string & deviceName);
    void registerPlugins(const std::string & xmlConfigFile);
    void addExtension(const std::string & ext_lib_path, const std::string & deviceName);
    std::vector<std::string> getAvailableDevices();
    PyObject* getMetric(const std::string & deviceName, const std::string & name);
    PyObject* getConfig(const std::string & deviceName, const std::string & name);
};

template<class T>
T *get_buffer(InferenceEngine::Blob &blob) {
    return blob.buffer().as<T *>();
}

template<class T, class... Args>
std::unique_ptr<T> make_unique(Args &&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

std::string get_version();
};  // namespace InferenceEnginePython

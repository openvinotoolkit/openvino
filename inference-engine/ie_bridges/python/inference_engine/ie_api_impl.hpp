// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef INFERENCE_ENGINE_DRIVER_IE_API_IMPL_HPP
#define INFERENCE_ENGINE_DRIVER_IE_API_IMPL_HPP

#include <string>
#include <inference_engine.hpp>
#include <iterator>
#include <iostream>
#include <algorithm>
#include <sstream>
#include "ie_extension.h"

namespace InferenceEnginePython {
//struct BlobInfo {
//    int layout;
//    std::vector<std::size_t> dims;
//    std::string name;
//    std::vector<std::string> inputTo;
//};
struct IENetLayer {
    InferenceEngine::CNNLayerPtr layer_ptr;
    std::string name;
    std::string type;
    std::string precision;
    std::string affinity;
    std::map<std::string, std::string> params;
//    std::map<std::string, InferenceEnginePython::BlobInfo> blob_info;
//    std::map<std::string, InferenceEngine::Blob::Ptr> weights;
    void setAffinity(const std::string & target_affinity);
    void setParams(const std::map<std::string, std::string> & params_map);
    std::map<std::string, InferenceEngine::Blob::Ptr> getWeights();
};
struct ProfileInfo {
    std::string status;
    std::string exec_type;
    std::string layer_type;
    long long real_time;
    long long cpu_time;
    unsigned execution_index;
};
struct IENetwork {
    InferenceEngine::CNNNetwork actual;
    std::string name;
    std::size_t batch_size;
    std::map<std::string, std::vector<size_t>> inputs;
    std::vector<std::string> outputs;
    void setPrecision() {
        InferenceEngine::CNNNetwork one;
        InferenceEngine::CNNNetwork second(std::move(one));
    }
    void setBatch(const size_t size);
    void addOutputs(const std::vector<std::string> &out_layers, const std::string &precision);
    std::map<std::string, InferenceEnginePython::IENetLayer> getLayers();
    void reshape(const std::map<std::string, std::vector<size_t>> & input_shapes);
};

struct IENetReader {
    static IENetwork read(std::string const &model, std::string const &weights);
    std::vector<std::pair<std::string, std::string>> getLayers();
};

struct InferRequestWrap {
    InferenceEngine::IInferRequest::Ptr request_ptr;
    InferenceEngine::BlobMap inputs;
    InferenceEngine::BlobMap outputs;

    void infer();
    void infer_async();
    int  wait(int64_t timeout);
    InferenceEngine::Blob::Ptr &getInputBlob(const std::string &blob_name);
    InferenceEngine::Blob::Ptr &getOutputBlob(const std::string &blob_name);
    std::vector<std::string> getInputsList();
    std::vector<std::string> getOutputsList();
    std::map<std::string, InferenceEnginePython::ProfileInfo> getPerformanceCounts();
};


struct IEExecNetwork {
    InferenceEngine::IExecutableNetwork::Ptr actual;
    std::vector<InferRequestWrap> infer_requests;
    IEExecNetwork(const std::string &name, size_t num_requests);

    std::string name;
    int next_req_index = 0;
    bool async;
    void infer();
};


struct IEPlugin {
    std::unique_ptr<InferenceEnginePython::IEExecNetwork> load(InferenceEnginePython::IENetwork &net,
                                                                   int num_requests,
                                                                   const std::map<std::string,std::string> &config);
    std::string device_name;
    std::string version;
    void setConfig(const std::map<std::string, std::string> &);
    void addCpuExtension(const std::string &extension_path);
    void setInitialAffinity(InferenceEnginePython::IENetwork &net);
    IEPlugin(const std::string &device, const std::vector<std::string> &plugin_dirs);
    IEPlugin() = default;
    std::set<std::string> queryNetwork(InferenceEnginePython::IENetwork &net);
    InferenceEngine::InferenceEnginePluginPtr actual;

};

template<class T>
T* get_buffer(InferenceEngine::Blob& blob) {
    return blob.buffer().as<T *>();
}

template<class T, class... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

std::string get_version();
}; // InferenceEnginePython

#endif //INFERENCE_ENGINE_DRIVER_IE_API_IMPL_HPP

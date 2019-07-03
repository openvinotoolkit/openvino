// Copyright (C) 2018-2019 Intel Corporation
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

#pragma once

#include <ie_extension.h>
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
#include "inference_engine.hpp"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

namespace InferenceEnginePython {
struct IENetLayer {
    InferenceEngine::CNNLayerPtr layer_ptr;
    InferenceEngine::CNNNetwork network_ptr;
    std::string name;
    std::string type;
    std::string precision;
    std::string shape;
    std::string layout;
    std::vector<std::string> children;
    std::vector<std::string> parents;
    std::string affinity;
    std::map<std::string, std::string> params;

    void setAffinity(const std::string &target_affinity);

    void setParams(const std::map<std::string, std::string> &params_map);

    std::map<std::string, InferenceEngine::Blob::Ptr> getWeights();

    void setPrecision(std::string precision);
};

struct InputInfo {
    InferenceEngine::InputInfo actual;
    std::vector<size_t> dims;
    std::string precision;
    std::string layout;

    void setPrecision(std::string precision);

    void setLayout(std::string layout);
};

struct OutputInfo {
    InferenceEngine::DataPtr actual;
    std::vector<size_t> dims;
    std::string precision;
    std::string layout;

    void setPrecision(std::string precision);
};

struct ProfileInfo {
    std::string status;
    std::string exec_type;
    std::string layer_type;
    int64_t real_time;
    int64_t cpu_time;
    unsigned execution_index;
};

struct IENetwork {
    InferenceEngine::CNNNetwork actual;
    std::string name;
    std::size_t batch_size;

    void setBatch(const size_t size);

    void addOutputs(const std::vector<std::string> &out_layers, const std::string &precision);

    const std::vector<std::pair<std::string, InferenceEnginePython::IENetLayer>> getLayers();

    const std::map<std::string, InferenceEnginePython::InputInfo> getInputs();

    const std::map<std::string, InferenceEnginePython::OutputInfo> getOutputs();

    void reshape(const std::map<std::string, std::vector<size_t>> &input_shapes);

    void serialize(const std::string &path_to_xml, const std::string &path_to_bin);

    void setStats(const std::map<std::string, std::map<std::string, std::vector<float>>> &stats);

    const std::map<std::string, std::map<std::string, std::vector<float>>> getStats();

    IENetwork(const std::string &model, const std::string &weights);

    IENetwork() = default;
};

struct InferRequestWrap {
    InferenceEngine::IInferRequest::Ptr request_ptr;
    Time::time_point start_time;
    double exec_time;
    void infer();

    void infer_async();

    int  wait(int64_t timeout);

    void getBlobPtr(const std::string &blob_name, InferenceEngine::Blob::Ptr &blob_ptr);

    void setBatch(int size);

    std::map<std::string, InferenceEnginePython::ProfileInfo> getPerformanceCounts();
};


struct IEExecNetwork {
    InferenceEngine::IExecutableNetwork::Ptr actual;
    std::vector<InferRequestWrap> infer_requests;
    std::string name;

    IEExecNetwork(const std::string &name, size_t num_requests);

    void infer();
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

    InferenceEngine::InferenceEnginePluginPtr actual;
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

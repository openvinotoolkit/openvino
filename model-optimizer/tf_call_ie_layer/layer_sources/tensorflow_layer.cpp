/*
// Copyright (c) 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/
/**
* \brief Implementation of custom TF subgraph call
*/
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "tensorflow_layer.h"
#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>

using namespace IECustomExtension;
using namespace InferenceEngine;
using namespace tensorflow;
using namespace std;

vector<string> splitString(const string& s, char delimiter, bool allow_empty = false)
{
    vector<string> result;
    size_t start = 0;
    size_t cur = 0;
    while(cur < s.size())
    {
        start = cur;
        while(cur < s.size() && s[cur] != delimiter) ++cur;
        if (!(cur == start && !allow_empty))
            result.push_back(s.substr(start, cur - start));
        cur++;
    }
    return result;
}

Status loadGraphDefFromString(const string& protobuf, Session* session)
{
    Status status;

    // save the GraphDef to the file so we can use ReadTextProto to initialize GraphDef
    auto default_env = Env::Default();
    string graph_def_file_name = tmpnam(nullptr);
    status = WriteStringToFile(default_env, graph_def_file_name, protobuf);
    if (!status.ok())
        return status;

    // create GraphDef with subgraph
    GraphDef graph_def;
    status = ReadTextProto(Env::Default(), graph_def_file_name, &graph_def);
    if (!status.ok())
        return status;
    remove(graph_def_file_name.c_str());

    // Add the graph to the session
    status = session->Create(graph_def);
    if (!status.ok())
      return status;
    return status;
}

tensorflow::TensorShape SizeVectorToTensorShape(const SizeVector& size_vector)
{
    TensorShape shape;
    for (auto & dim: size_vector)
        shape.AddDim(dim);
    return shape;
}

StatusCode TensorflowImplementation::getSupportedConfigurations(std::vector<LayerConfig>& conf, ResponseDesc *resp) noexcept {
    if (!errorMsg.empty()) {
        if (resp)
            errorMsg.copy(resp->msg, sizeof(resp->msg) - 1);
        return GENERAL_ERROR;
    }
    LayerConfig config;

    config.dynBatchSupport = false;
    for (size_t i = 0; i < _layer.insData.size(); i++) {
        DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        std::vector<size_t> order;
        for (size_t j = 0; j < _layer.insData[i].lock()->getTensorDesc().getDims().size(); j++)
            order.push_back(j);
        dataConfig.desc = TensorDesc(InferenceEngine::Precision::FP32, _layer.insData[i].lock()->getTensorDesc().getDims(),
                                     {_layer.insData[i].lock()->getTensorDesc().getDims(), order});
        config.inConfs.push_back(dataConfig);
    }

    for (size_t i = 0; i < _layer.outData.size(); i++) {
        DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        std::vector<size_t> order;
        for (size_t j = 0; j < _layer.outData[i]->getTensorDesc().getDims().size(); j++)
            order.push_back(j);
        dataConfig.desc = TensorDesc(InferenceEngine::Precision::FP32, _layer.outData[i]->getTensorDesc().getDims(),
                                     {_layer.outData[i]->getTensorDesc().getDims(), order});
        config.outConfs.push_back(dataConfig);
    }
    conf.push_back(config);
    return OK;
}

StatusCode TensorflowImplementation::init(LayerConfig& config, ResponseDesc *resp) noexcept {
    for (auto& input : config.inConfs)
    {
        for (auto& offset : input.desc.getBlockingDesc().getOffsetPaddingToData())
            if (offset)
                return GENERAL_ERROR;
        if (input.desc.getBlockingDesc().getOffsetPadding())
            return GENERAL_ERROR;
        for (size_t i = 0; i < input.desc.getBlockingDesc().getOrder().size(); i++)
            if (input.desc.getBlockingDesc().getOrder()[i] != i)
                return GENERAL_ERROR;
    }
    for (auto& output : config.outConfs)
    {
        for (auto& offset : output.desc.getBlockingDesc().getOffsetPaddingToData())
            if (offset)
                return GENERAL_ERROR;
        if (output.desc.getBlockingDesc().getOffsetPadding())
            return GENERAL_ERROR;
        for (size_t i = 0; i < output.desc.getBlockingDesc().getOrder().size(); i++)
            if (output.desc.getBlockingDesc().getOrder()[i] != i)
                return GENERAL_ERROR;
    }
    return OK;
}

StatusCode TensorflowImplementation::execute(std::vector<Blob::Ptr>& inputs, std::vector<Blob::Ptr>& outputs,
                   ResponseDesc *resp) noexcept {
    try {
        // Initialize a tensorflow session
        Session *session;
        Status status = NewSession(SessionOptions(), &session);
        if (!status.ok()) {
            cerr << status.ToString() << endl;
            return GENERAL_ERROR;
        }

        // initialize input nodes with values provided by IE inputs
        vector<pair<string, tensorflow::Tensor>> tf_placeholders;
        vector<string> input_names = splitString(input_nodes_names, ' ', false);
        vector<string> output_tensors = splitString(output_tensors_names, ' ', false);
        vector<string> real_input_dims_str = splitString(real_input_dims, ';', false);

        for (size_t i = 0; i < input_names.size(); ++i) {
            string &input_name = input_names[i];
            size_t total_size = inputs[i]->byteSize();
            SizeVector in_tensor_size;
            vector<string> dims_str = splitString(real_input_dims_str[i], ' ', false);
            vector<size_t> dims_int = vector<size_t>(dims_str.size());
            for (size_t j = 0; j < dims_int.size(); ++j)
                dims_int[j] = stoi(dims_str[j]);
            tensorflow::Tensor t = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT,
                                                      TensorShape(SizeVectorToTensorShape(dims_int)));

            // copy data from an IE blob to a TF tensor as FP32
            memcpy(t.flat<float>().data(), inputs[i]->buffer().as<float *>(), total_size);

            tf_placeholders.push_back(make_pair(input_name, t));
        }

        status = loadGraphDefFromString(protobuf, session);
        if (!status.ok()) {
            cerr << status.ToString() << endl;
            return GENERAL_ERROR;
        }

        // output tensors of the sub-graph
        vector<tensorflow::Tensor> tf_outputs;

        status = session->Run(tf_placeholders, output_tensors, {}, &tf_outputs);
        if (!status.ok()) {
            cerr << "session->Run() error: " << status.ToString() << endl;
            return GENERAL_ERROR;
        }
        for (size_t out_id = 0; out_id < outputs.size(); ++out_id) {
            size_t total_size = outputs[out_id]->byteSize();

            DataType output_data_type = tf_outputs[out_id % tf_outputs.size()].dtype();
            if (output_data_type == DT_FLOAT)
                memcpy(outputs[out_id]->buffer(), tf_outputs[out_id % tf_outputs.size()].flat<float>().data(),
                       total_size);
            else
                memcpy(outputs[out_id]->buffer(), tf_outputs[out_id % tf_outputs.size()].flat<int64>().data(),
                       total_size);
        }
    } catch (...) {
        return GENERAL_ERROR;
    }
    return OK;
}

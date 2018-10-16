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

void TensorflowLayer::Execute() noexcept {
    /* Stores layer params */
    auto genLayer    = reinterpret_cast<GenericLayer*>(_layer.get());
    auto data_params = genLayer->params;
    string protobuf = genLayer->GetParamAsString("protobuf");
    const string output_tensors_names = genLayer->GetParamAsString("output_tensors_names");
    const string input_node_names = genLayer->GetParamAsString("input_nodes_names");
    const string real_input_dims = genLayer->GetParamAsString("real_input_dims");

    // Initialize a tensorflow session
    Session* session;
    Status status = NewSession(SessionOptions(), &session);
    if (!status.ok()) {
        cerr << status.ToString() << "\n";
        return;
    }

    // initialize input nodes with values provided by IE inputs
    vector<pair<string, tensorflow::Tensor>> tf_placeholders;
    vector<string> input_names = splitString(input_node_names, ' ', false);
    vector<string> output_tensors = splitString(output_tensors_names, ' ', false);
    vector<string> real_input_dims_str = splitString(real_input_dims, ';', false);

    for (size_t i = 0; i < input_names.size(); ++i)
    {
        string& input_name = input_names[i];
        size_t total_size = accumulate(inputs[i].dims.begin(), inputs[i].dims.end(), 1, multiplies<size_t>());
        SizeVector in_tensor_size;
        vector<string> dims_str = splitString(real_input_dims_str[i], ' ', false);
        vector<size_t> dims_int = vector<size_t>(dims_str.size());
        for(size_t j = 0; j < dims_int.size(); ++j)
            dims_int[j] = stoi(dims_str[j]);
        tensorflow::Tensor t = tensorflow::Tensor(tensorflow::DataType::DT_FLOAT, TensorShape(SizeVectorToTensorShape(dims_int)));

        // copy data from an IE blob to a TF tensor as FP32
        memcpy(t.flat<float>().data(), inputs[i].data, total_size * sizeof(float));

        tf_placeholders.push_back(make_pair(input_name, t));
    }

    status = loadGraphDefFromString(protobuf, session);
    if (!status.ok()) {
        cerr << status.ToString() << "\n";
        return;
    }

    // output tensors of the sub-graph
    vector<tensorflow::Tensor> tf_outputs;

    status = session->Run(tf_placeholders, output_tensors, {}, &tf_outputs);
    if (!status.ok()) {
        cerr << "session->Run() error: " << status.ToString() << "\n";
        return;
    }
    for (size_t out_id = 0; out_id < outputs.size(); ++out_id)
    {
        size_t total_size = 1;
        for (size_t i = 0; i < outputs[out_id].dims.size(); ++i)
            total_size *= outputs[out_id].dims[i];

        DataType output_data_type = tf_outputs[out_id % tf_outputs.size()].dtype();
        if (output_data_type == DT_FLOAT)
            memcpy(outputs[out_id].data, tf_outputs[out_id % tf_outputs.size()].flat<float>().data(), total_size * sizeof(float));
        else
            memcpy(outputs[out_id].data, tf_outputs[out_id % tf_outputs.size()].flat<int64>().data(), total_size * sizeof(int64));
    }
}

vector<InferenceEngine::MKLDNNPlugin::MKLDNNGenericFormats> TensorflowLayer::GetSupportedFormats() noexcept {
    return
         {
            {
                {InferenceEngine::MKLDNNPlugin::MemoryFormat::nchw}, // input
                {InferenceEngine::MKLDNNPlugin::MemoryFormat::nchw} // output
            }
         };
}

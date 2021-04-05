//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************


#include <algorithm>
#include <numeric>
#include <chrono>
#include <memory>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <fstream>

#include "framework.pb.h"

#include "../include/paddlepaddle_frontend/model.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset6.hpp>

#include "utility.hpp"
#include "decoder.hpp"
#include "node_context.hpp"
#include "op_table.hpp"

#include <functional>


namespace ngraph {
namespace frontend {
namespace pdpd {

std::shared_ptr<ngraph::Node>
make_ng_node(std::map<std::string, google::protobuf::RepeatedPtrField<std::string>> &inputs,
             std::map<std::string, std::shared_ptr<ngraph::Node>> &nodes,
             const paddle::framework::proto::OpDesc &op,
             const paddle::framework::proto::BlockDesc &block,
             const std::map<std::string, CreatorFunction>& CREATORS_MAP) {
    std::cout << "Making node: " << op.type() << std::endl;

    MY_ASSERT(CREATORS_MAP.find(op.type()) != CREATORS_MAP.end(), "No creator found");
    std::map<std::string, std::vector<std::shared_ptr<ngraph::Node>>> inputs_preproc;
    for (const auto &item : inputs) {
        inputs_preproc[item.first] = std::vector<std::shared_ptr<ngraph::Node>>();
        for (const auto &input_name : item.second) {
            // TODO: refactor to not search every time
            inputs_preproc[item.first].push_back(nodes[input_name]);
        }
    }

    // TODO: Temporary repacking data to fit new creator API based on OutputVector instead of direct
    // TODO: nodes manipulation.

    NamedInputs named_inputs;
    for(const auto& input: inputs_preproc)
    {
        for(auto node: input.second)
            named_inputs[input.first].push_back(node);
    }

    OutputVector outputs = CREATORS_MAP.at(op.type())(NodeContext(op, named_inputs));
    MY_ASSERT(outputs.size() == 1);
    return outputs[0].get_node_shared_ptr();
}

std::shared_ptr<ngraph::opset6::Constant> read_tensor(const paddle::framework::proto::VarDesc& var,
                std::shared_ptr<ngraph::frontend::InputModelPDPD> model)
{
    std::cout << "Reading tensor " << var.name() << std::endl;
    MY_ASSERT(var.type().type() == paddle::framework::proto::VarType::LOD_TENSOR);
    auto tensor = var.type().lod_tensor().tensor();
    auto tensor_length = std::accumulate(
        tensor.dims().cbegin(), tensor.dims().cend(), 1, std::multiplies<int64_t>());
    std::vector<float> tensor_data(tensor_length, 0);

    std::ifstream is;
    std::ifstream* stream_ptr;
    if (model->weights_composed) {
        stream_ptr = &model->weights_stream;
    } else {
        is = std::ifstream(model->path + "/" + var.name(), std::ios::in | std::ifstream::binary);
        if (!is || !is.is_open())
        {
            std::cout << "File not opened" << std::endl;
        }
        stream_ptr = &is;
    }
    std::vector<char> leading_zeros(16, 0);
    stream_ptr->read(&leading_zeros[0], 16);
    uint32_t dims_len = 0;
    stream_ptr->read(reinterpret_cast<char*>(&dims_len), 4);
    std::vector<char> dims_struct(dims_len, 0);
    stream_ptr->read(&dims_struct[0], dims_len);
    stream_ptr->read(reinterpret_cast<char*>(&tensor_data[0]), tensor_length * 4);

    auto shape = std::vector<size_t>(tensor.dims().cbegin(), tensor.dims().cend());
    return ngraph::opset6::Constant::create(
        ngraph::element::f32, ngraph::Shape(shape), tensor_data);
}

bool endsWith(const std::string &str, const std::string &suffix) {
    if (str.length() >= suffix.length()) {
        return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
    }
    return false;
}

std::shared_ptr<ngraph::Function>
    convert_model(std::shared_ptr<ngraph::frontend::InputModelPDPD> model)
{
    std::cout << "Convert Model Start" << std::endl;    
    
    paddle::framework::proto::ProgramDesc fw_model;
    std::ifstream pb_stream(model->model_file, std::ios::binary);
    std::cout << "Model Parsed: " << fw_model.ParseFromIstream(&pb_stream) << std::endl;

    std::map<std::string, std::shared_ptr<ngraph::Node>> nodes_dict;
    ngraph::ParameterVector parameter_nodes;
    ngraph::ResultVector result_nodes;

    std::cout << "Blocks number: " << fw_model.blocks().size() << std::endl;
    const auto& global_block = fw_model.blocks()[0];
    // We need to read variables in sorted by name order. This is the order variables written in composed file.
    std::map<std::string, paddle::framework::proto::VarDesc> sorted_vars;
    for (auto& var : global_block.vars())
    {
        sorted_vars[var.name()] = var;
    }
    for (const auto& name_var : sorted_vars)
    {
        if (endsWith(name_var.first, "feed") || endsWith(name_var.first, "fetch"))
            continue;
        if (!name_var.second.persistable())
            continue;
        nodes_dict[name_var.first] = read_tensor(name_var.second, model);
    }
    std::cout << "Reading consts finished" << std::endl;

    std::map<std::string, CreatorFunction> CREATORS_MAP = get_supported_ops();

    for (const auto& block : fw_model.blocks()) {
        std::map<std::string, paddle::framework::proto::VarType> vars_dict;
        for (const auto &var : block.vars()) {
            vars_dict[var.name()] = var.type();
        }
        for (int i = 0; i < block.ops_size(); i++) {
            std::cerr << "Observing index i = " << i << "\n";
            const auto &op = block.ops()[i];
            std::cerr << "Observing " << op.type() << "\n";
            std::map<std::string, google::protobuf::RepeatedPtrField<std::string>> outputs_dict;
            for (const auto &output : op.outputs()) {
                outputs_dict[output.parameter()] = output.arguments();
                std::cerr << output.parameter() << "\n";
            }
            std::map<std::string, google::protobuf::RepeatedPtrField<std::string>> inputs_dict;
            for (const auto &input : op.inputs()) {
                inputs_dict[input.parameter()] = input.arguments();
            }
            if (op.type() == "feed") {
                auto layer_name = outputs_dict["Out"][0];
                std::cout << "Creating parameter: " << layer_name << std::endl;
                auto var = vars_dict[layer_name];
                MY_ASSERT(var.type() == paddle::framework::proto::VarType::LOD_TENSOR);
                auto tensor_desc = var.lod_tensor().tensor();
                auto dtype = tensor_desc.data_type();
                std::vector<size_t> shape;
                // set all -1 dims to 1
                // TODO: remove when input shape can be specified
                for (auto dim : tensor_desc.dims()) {
                    if (dim >= 0) {
                        shape.push_back(dim);
                    } else {
                        shape.push_back(1);
                    }
                }
                auto param = std::make_shared<ngraph::opset6::Parameter>(TYPE_MAP[dtype],
                                                                         ngraph::Shape(shape));
                param->set_friendly_name(layer_name);
                nodes_dict[layer_name] = param;
                parameter_nodes.push_back(param);
                std::cout << "Parameter created" << std::endl;
            } else if (op.type() == "fetch") {
                auto input_node = inputs_dict["X"][0];
                MY_ASSERT(nodes_dict.find(input_node) != nodes_dict.end());
                auto result = std::make_shared<ngraph::opset6::Result>(nodes_dict[input_node]);
                result->set_friendly_name(input_node);
                result_nodes.push_back(result);
            } else {
                auto node = make_ng_node(inputs_dict, nodes_dict, op, block, CREATORS_MAP);
                std::string layer_name;
                for (const auto& outs : outputs_dict) {
                    for (const auto& fieldName : outs.second) {
                        layer_name += fieldName;
                    }
                }
                node->set_friendly_name(layer_name);

                std::cerr << "Named with " << node->get_friendly_name() << "\n";
                for (const auto &item : outputs_dict) {
                    MY_ASSERT(item.second.size() <= 1);
                    if (item.second.size() == 1) {
                        nodes_dict[item.second[0]] = node;
                    }
                }
            }
        }
    }
    return std::make_shared<ngraph::Function>(result_nodes, parameter_nodes);
}

}

std::shared_ptr<ngraph::Function> ngraph::frontend::FrontEndPDPD::convert(InputModel::Ptr model) const {
    std::cerr << "[ INFO ] PFrontEndPDPD::convert invoked\n";
    auto pdpd_model = std::dynamic_pointer_cast<ngraph::frontend::InputModelPDPD>(model);
    std::string ext = ".pdmodel";
    if (pdpd_model->path.length() >= ext.length() &&
        (0 ==
         pdpd_model->path.compare(pdpd_model->path.length() - ext.length(), ext.length(), ext)))
    {
        pdpd_model->weights_composed = true;
        pdpd_model->model_file = pdpd_model->path;
        auto weights_file =
            pdpd_model->path.replace(pdpd_model->path.size() - ext.size(), ext.size(), ".pdiparams");
        pdpd_model->weights_stream = std::ifstream(weights_file, std::ios::binary);
        if (!pdpd_model->weights_stream || !pdpd_model->weights_stream.is_open())
        {
            std::cout << "File not opened" << std::endl;
        }
    }
    else
    {
        pdpd_model->weights_composed = false;
        pdpd_model->model_file = pdpd_model->path + "/__model__";
    }
    auto f = pdpd::convert_model(pdpd_model);
    std::cerr << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << "\n";
    return f;
}

}
}

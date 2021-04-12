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
make_ng_node(std::map<std::string, std::shared_ptr<ngraph::Node>> &nodes,
             std::shared_ptr<OpPlacePDPD> place,
             const std::map<std::string, CreatorFunction>& CREATORS_MAP) {
    auto op = (paddle::framework::proto::OpDesc*)place->op;
    std::cout << "Making node: " << op->type() << std::endl;

    MY_ASSERT(CREATORS_MAP.find(op->type()) != CREATORS_MAP.end(), "No creator found");
    std::map<std::string, std::vector<std::shared_ptr<ngraph::Node>>> inputs_preproc;
    for (const auto &item : place->inputs) {
        inputs_preproc[item.first] = std::vector<std::shared_ptr<ngraph::Node>>();
        for (auto& var_place : item.second) {
            // TODO: refactor to not search every time
            auto var = (paddle::framework::proto::VarDesc*)var_place.lock()->var;
            inputs_preproc[item.first].push_back(nodes[var->name()]);
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

    OutputVector outputs = CREATORS_MAP.at(op->type())(NodeContext(*op, named_inputs));
    MY_ASSERT(outputs.size() == 1);
    return outputs[0].get_node_shared_ptr();
}

bool endsWith(const std::string &str, const std::string &suffix) {
    if (str.length() >= suffix.length()) {
        return (0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix));
    }
    return false;
}

} // namespace pdpd

std::shared_ptr<opset6::Constant> FrontEndPDPD::read_tensor(std::shared_ptr<VarPlacePDPD> place,
                std::shared_ptr<InputModelPDPD> model) const
{
    auto _var = (paddle::framework::proto::VarDesc*)place->var;
    std::cout << "Reading tensor " << _var->name() << std::endl;
    MY_ASSERT(_var->type().type() == paddle::framework::proto::VarType::LOD_TENSOR);
    auto tensor = _var->type().lod_tensor().tensor();
    auto tensor_length = std::accumulate(
        tensor.dims().cbegin(), tensor.dims().cend(), 1, std::multiplies<int64_t>());
    // TODO: implement for other types
    auto tensor_data = model->getWeight(_var->name(), tensor_length);    

    auto shape = std::vector<size_t>(tensor.dims().cbegin(), tensor.dims().cend());
    return opset6::Constant::create(element::f32, Shape(shape), tensor_data);
}

std::shared_ptr<Function>
    FrontEndPDPD::convert_model(std::shared_ptr<InputModelPDPD> model) const
{
    std::cout << "Convert Model Start" << std::endl;    
    
    std::map<std::string, std::shared_ptr<Node>> nodes_dict;
    ParameterVector parameter_nodes;
    ResultVector result_nodes;
    
    const auto& global_var_places = model->getVarPlaces(0);
    for (const auto& name_var : global_var_places)
    {
        auto var = (paddle::framework::proto::VarDesc*)name_var.second->var;
        if (pdpd::endsWith(name_var.first, "feed") || pdpd::endsWith(name_var.first, "fetch"))
            continue;
        if (!var->persistable())
            continue;
        nodes_dict[name_var.first] = read_tensor(name_var.second, model);
    }
    std::cout << "Reading consts finished" << std::endl;

    std::map<std::string, pdpd::CreatorFunction> CREATORS_MAP = pdpd::get_supported_ops();

    for (int i = 0; i < model->getBlockNumber(); i++) {
        const auto& op_places = model->getOpPlaces(i);
        const auto& var_places = model->getVarPlaces(i);
        for (int j = 0; j < op_places.size(); j++) {
            std::cerr << "Observing index i = " << j << "\n";
            const auto &op_place = op_places[j];
            auto op = (paddle::framework::proto::OpDesc*)op_place->op;
            std::cerr << "Observing " << op->type() << "\n";
            if (op->type() == "feed") {
                auto out_var = op_place->outputs.at("Out")[0];
                auto var = (paddle::framework::proto::VarDesc*)out_var.lock()->var;
                MY_ASSERT(var->type().type() == paddle::framework::proto::VarType::LOD_TENSOR);
                auto tensor_desc = var->type().lod_tensor().tensor();
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
                param->set_friendly_name(var->name());
                nodes_dict[var->name()] = param;
                parameter_nodes.push_back(param);
                std::cout << "Parameter created" << std::endl;
            } else if (op->type() == "fetch") {
                // TODO: resolve names for multiple outputs from one node
                auto in_var = op_place->inputs.at("X")[0];
                auto var = (paddle::framework::proto::VarDesc*)in_var.lock()->var;
                auto input_var_name = var->name();
                auto result = std::make_shared<ngraph::opset6::Result>(nodes_dict.at(input_var_name));
                result->set_friendly_name(input_var_name + "/Result");
                result_nodes.push_back(result);
            } else {
                auto node = pdpd::make_ng_node(nodes_dict, op_place, CREATORS_MAP);
                // set layer name by the name of first output var
                auto& first_output_var_place = op_place->outputs.begin()->second[0];
                auto var = (paddle::framework::proto::VarDesc*)first_output_var_place.lock()->var;
                node->set_friendly_name(var->name());

                std::cerr << "Named with " << node->get_friendly_name() << "\n";
                for (const auto &item : op_place->outputs) {
                    MY_ASSERT(item.second.size() <= 1);
                    if (item.second.size() == 1) {
                        auto var = (paddle::framework::proto::VarDesc*)item.second[0].lock()->var;
                        nodes_dict[var->name()] = node;
                    }
                }
            }
        }
    }
    return std::make_shared<ngraph::Function>(result_nodes, parameter_nodes);
}

std::shared_ptr<ngraph::Function> ngraph::frontend::FrontEndPDPD::convert(InputModel::Ptr model) const {
    std::cerr << "[ INFO ] PFrontEndPDPD::convert invoked\n";
    auto pdpd_model = std::dynamic_pointer_cast<ngraph::frontend::InputModelPDPD>(model);    
    auto f = convert_model(pdpd_model);
    std::cerr << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << "\n";
    return f;
}

} // namespace frontend
} // namespace ngraph

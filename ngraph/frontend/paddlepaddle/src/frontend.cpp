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
#include <chrono>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "framework.pb.h"

#include <paddlepaddle_frontend/frontend.hpp>
#include <paddlepaddle_frontend/model.hpp>
#include <paddlepaddle_frontend/place.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset7.hpp>

#include <paddlepaddle_frontend/exceptions.hpp>
#include "decoder.hpp"
#include "node_context.hpp"
#include "op_table.hpp"

#include <functional>

using namespace ngraph::opset7;

using namespace ngraph;
using namespace ngraph::frontend;
namespace ngraph {
    namespace frontend {
        namespace pdpd {
            NamedOutputs make_ng_node(std::map<pdpd::TensorName, Output<Node>> &nodes,
                                      const std::shared_ptr<OpPlacePDPD> &op_place,
                                      const std::map<std::string, CreatorFunction> &CREATORS_MAP) {
                const auto &op = op_place->getDesc();
                std::cout << "Making node: " << op->type() << std::endl;

                PDPD_CHECK(ngraph::frontend::ErrorCode::NGRAPH_NODE_CREATION_FAILED,
                           CREATORS_MAP.find(op->type()) != CREATORS_MAP.end(), "No creator found for ", op->type(),
                           " node.");
                pdpd::NamedInputs named_inputs;
                const auto &input_ports = op_place->getInputPorts();
                for (const auto &name_to_ports : input_ports) {
                    for (const auto &port : name_to_ports.second) {
                        const auto &var_desc = port->getSourceTensorPDPD()->getDesc();
                        if (nodes.count(var_desc->name()))
                            named_inputs[name_to_ports.first].push_back(nodes.at(var_desc->name()));
                        else
                            // return empty map when not all inputs exist. It usually means that
                            // these nodes are not used because model inputs were overwritten
                            return NamedOutputs();
                    }
                }

                return CREATORS_MAP.at(op->type())(NodeContext(DecoderPDPDProto(op_place), named_inputs));
            }
        } // namespace pdpd
    }
}

std::shared_ptr<Function> FrontEndPDPD::convert_model(const std::shared_ptr<InputModelPDPD>& model)
{
    std::cout << "Convert Model Start" << std::endl;

    std::map<pdpd::TensorName, Output<Node>> nodes_dict(model->getTensorValues());
    ParameterVector parameter_nodes;
    ResultVector result_nodes;

    std::map<std::string, pdpd::CreatorFunction> CREATORS_MAP = pdpd::get_supported_ops();
    for (const auto& _inp_place : model->getInputs())
    {
        const auto& inp_place = std::dynamic_pointer_cast<TensorPlacePDPD>(_inp_place);
        const auto& var = inp_place->getDesc();
        const auto& shape = inp_place->getPartialShape();
        const auto& type = inp_place->getElementType();
        auto param = std::make_shared<Parameter>(type, shape);
        param->set_friendly_name(var->name());
        nodes_dict[var->name()] = param;
        parameter_nodes.push_back(param);
    }

    const auto& op_places = model->getOpPlaces();
    for (const auto& op_place : op_places)
    {
        const auto& op_type = op_place->getDesc()->type();
        std::cerr << "Observing " << op_type << "\n";
        if (op_type == "feed" || op_type == "fetch")
        {
            // inputs and outputs are stored in the model already
            continue;
        }
        else
        {
            const auto& named_outputs = pdpd::make_ng_node(nodes_dict, op_place, CREATORS_MAP);

            // set layer name by the name of first output var
            if (!named_outputs.empty())
            {
                const auto& first_output_var = op_place->getOutputPorts()
                                                   .begin()
                                                   ->second.at(0)
                                                   ->getTargetTensorPDPD()
                                                   ->getDesc();
                auto node = named_outputs.begin()->second[0].get_node_shared_ptr();
                node->set_friendly_name(first_output_var->name());
                std::cerr << "Named with " << node->get_friendly_name() << "\n";
            }

            const auto& out_ports = op_place->getOutputPorts();
            for (const auto& name_to_outputs : named_outputs)
            {
                const auto& ports = out_ports.at(name_to_outputs.first);

                PDPD_CHECK(ngraph::frontend::ErrorCode::NGRAPH_NODE_CREATION_FAILED, ports.size() == name_to_outputs.second.size(),
                            "The number of output tensors must be equal to "
                            "the number of outputs of the ngraph node.");
                for (size_t idx = 0; idx < ports.size(); ++idx)
                {
                    const auto& var = ports[idx]->getTargetTensorPDPD()->getDesc();
                    name_to_outputs.second[idx].get_tensor().set_names({var->name()});
                    // if nodes_dict already has node mapped to this tensor name it usually
                    // means that it was overwritten using setTensorValue
                    if (!nodes_dict.count(var->name()))
                        nodes_dict[var->name()] = name_to_outputs.second[idx];
                }
            }
        }
    }

    for (const auto& _outp_place : model->getOutputs())
    {
        const auto& outp_place = std::dynamic_pointer_cast<TensorPlacePDPD>(_outp_place);
        auto var = outp_place->getDesc();
        auto input_var_name = var->name();
        auto result = std::make_shared<Result>(nodes_dict.at(input_var_name));
        result->set_friendly_name(input_var_name + "/Result");
        result_nodes.push_back(result);
    }

    return std::make_shared<Function>(result_nodes, parameter_nodes);
}

InputModel::Ptr FrontEndPDPD::loadFromFile(const std::string& path) const
{
    return loadFromFiles({path});
}

InputModel::Ptr FrontEndPDPD::loadFromFiles(const std::vector<std::string>& paths) const
{
    if (paths.size() == 1)
    {
        // The case when folder with __model__ and weight files is provided or .pdmodel file
        return std::make_shared<InputModelPDPD>(paths[0]);
    }
    else if (paths.size() == 2)
    {
        // The case when .pdmodel and .pdparams files are provided
        std::ifstream model_stream(paths[0], std::ios::in | std::ifstream::binary);
        PDPD_CHECK(ngraph::frontend::ErrorCode::INITIALIZATION_ERROR, model_stream && model_stream.is_open(), "Cannot open model file.");
        std::ifstream weights_stream(paths[1], std::ios::in | std::ifstream::binary);
        PDPD_CHECK(ngraph::frontend::ErrorCode::INITIALIZATION_ERROR, weights_stream && weights_stream.is_open(), "Cannot open weights file.");
        return loadFromStreams({&model_stream, &weights_stream});
    }
    PDPD_CHECK(ngraph::frontend::ErrorCode::INITIALIZATION_ERROR, false, "Model can be loaded either from 1 or 2 files");
}

InputModel::Ptr FrontEndPDPD::loadFromStream(std::istream& model_stream) const
{
    return loadFromStreams({&model_stream});
}

InputModel::Ptr FrontEndPDPD::loadFromStreams(const std::vector<std::istream*>& streams) const
{
    return std::make_shared<InputModelPDPD>(streams);
}

std::shared_ptr<Function> FrontEndPDPD::convert(InputModel::Ptr model) const
{
    std::cerr << "[ INFO ] PFrontEndPDPD::convert invoked\n";
    auto pdpd_model = std::dynamic_pointer_cast<InputModelPDPD>(model);
    auto f = convert_model(pdpd_model);
    std::cerr << "[ INFO ] Resulting nGraph function contains " << f->get_ops().size() << "\n";
    return f;
}

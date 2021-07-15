// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "framework.pb.h"

#include <paddlepaddle_frontend/exceptions.hpp>
#include <paddlepaddle_frontend/frontend.hpp>
#include <paddlepaddle_frontend/model.hpp>
#include <paddlepaddle_frontend/place.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/variant.hpp>

#include "decoder.hpp"
#include "node_context.hpp"
#include "op_table.hpp"
#include "pdpd_utils.hpp"

#include "frontend_manager/frontend_manager.hpp"

using namespace ngraph::opset7;
using namespace ngraph;
using namespace ngraph::frontend;

namespace ngraph
{
    namespace frontend
    {
        namespace pdpd
        {
            NamedOutputs make_ng_node(std::map<pdpd::TensorName, Output<Node>>& nodes,
                                      const std::shared_ptr<OpPlacePDPD>& op_place,
                                      const std::map<std::string, CreatorFunction>& CREATORS_MAP)
            {
                const auto& op = op_place->getDesc();
                // std::cout << "Making node: " << op->type() << std::endl;

                FRONT_END_OP_CONVERSION_CHECK(CREATORS_MAP.find(op->type()) != CREATORS_MAP.end(),
                                              "No creator found for ",
                                              op->type(),
                                              " node.");
                pdpd::NamedInputs named_inputs;
                const auto& input_ports = op_place->getInputPorts();
                for (const auto& name_to_ports : input_ports)
                {
                    for (const auto& port : name_to_ports.second)
                    {
                        const auto& var_desc = port->getSourceTensorPDPD()->getDesc();
                        if (nodes.count(var_desc->name()))
                            named_inputs[name_to_ports.first].push_back(nodes.at(var_desc->name()));
                        else
                            // return empty map when not all inputs exist. It usually means that
                            // these nodes are not used because model inputs were overwritten
                            return NamedOutputs();
                    }
                }

                try
                {
                    return CREATORS_MAP.at(op->type())(
                        NodeContext(DecoderPDPDProto(op_place), named_inputs));
                }
                catch (...)
                {
                    // TODO: define exception types
                    // In case of partial conversion we need to create generic ngraph op here
                    return NamedOutputs();
                }
            }

            std::istream* variant_to_stream_ptr(const std::shared_ptr<Variant>& variant,
                                                std::ifstream& ext_stream)
            {
                if (is_type<VariantWrapper<std::shared_ptr<std::istream>>>(variant))
                {
                    auto m_stream =
                        as_type_ptr<VariantWrapper<std::shared_ptr<std::istream>>>(variant)->get();
                    return m_stream.get();
                }
                else if (is_type<VariantWrapper<std::string>>(variant))
                {
                    const auto& model_path =
                        as_type_ptr<VariantWrapper<std::string>>(variant)->get();
                    ext_stream.open(model_path, std::ios::in | std::ifstream::binary);
                }
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
                else if (is_type<VariantWrapper<std::wstring>>(variant))
                {
                    const auto& model_path =
                        as_type_ptr<VariantWrapper<std::wstring>>(variant)->get();
                    ext_stream.open(model_path, std::ios::in | std::ifstream::binary);
                }
#endif
                FRONT_END_INITIALIZATION_CHECK(ext_stream && ext_stream.is_open(),
                                               "Cannot open model file.");
                return &ext_stream;
            }

        } // namespace pdpd

        std::shared_ptr<Function>
            FrontEndPDPD::convert_model(const std::shared_ptr<InputModelPDPD>& model)
        {
            // std::cout << "Convert Model Start" << std::endl;

            std::map<pdpd::TensorName, Output<Node>> nodes_dict(model->getTensorValues());
            ParameterVector parameter_nodes;
            ResultVector result_nodes;

            std::map<std::string, pdpd::CreatorFunction> CREATORS_MAP = pdpd::get_supported_ops();
            for (const auto& _inp_place : model->get_inputs())
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
                if (op_type == "feed" || op_type == "fetch")
                {
                    // inputs and outputs are stored in the model already
                    continue;
                }
                else
                {
                    const auto& named_outputs =
                        pdpd::make_ng_node(nodes_dict, op_place, CREATORS_MAP);

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
                    }

                    const auto& out_ports = op_place->getOutputPorts();
                    for (const auto& name_to_outputs : named_outputs)
                    {
                        const auto& ports = out_ports.at(name_to_outputs.first);
                        FRONT_END_OP_CONVERSION_CHECK(
                            ports.size() == name_to_outputs.second.size(),
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

            for (const auto& _outp_place : model->get_outputs())
            {
                const auto& outp_place = std::dynamic_pointer_cast<TensorPlacePDPD>(_outp_place);
                auto var = outp_place->getDesc();
                auto input_var_name = var->name();
                auto result = std::make_shared<Result>(nodes_dict.at(input_var_name));
                result->set_friendly_name(input_var_name + "/Result");
                result_nodes.push_back(result);
            }

            return std::make_shared<ngraph::Function>(result_nodes, parameter_nodes);
        }

        bool FrontEndPDPD::supported_impl(
            const std::vector<std::shared_ptr<Variant>>& variants) const
        {
            // FrontEndPDPD can only load model specified by one path, one file or two files.
            if (variants.empty() || variants.size() > 2)
                return false;

            // Validating first path, it must contain a model
            if (is_type<VariantWrapper<std::string>>(variants[0]))
            {
                std::string suffix = ".pdmodel";
                std::string model_path =
                    as_type_ptr<VariantWrapper<std::string>>(variants[0])->get();
                if (!pdpd::endsWith(model_path, suffix))
                {
                    model_path += pdpd::get_path_sep<char>() + "__model__";
                }
                std::ifstream model_str(model_path, std::ios::in | std::ifstream::binary);
                // It is possible to validate here that protobuf can read model from the stream,
                // but it will complicate the check, while it should be as quick as possible
                return model_str && model_str.is_open();
            }
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
            else if (is_type<VariantWrapper<std::wstring>>(variants[0]))
            {
                std::wstring suffix = L".pdmodel";
                std::wstring model_path =
                    as_type_ptr<VariantWrapper<std::wstring>>(variants[0])->get();
                if (!pdpd::endsWith(model_path, suffix))
                {
                    model_path += pdpd::get_path_sep<wchar_t>() + L"__model__";
                }
                std::ifstream model_str(model_path, std::ios::in | std::ifstream::binary);
                // It is possible to validate here that protobuf can read model from the stream,
                // but it will complicate the check, while it should be as quick as possible
                return model_str && model_str.is_open();
            }
#endif
            else if (is_type<VariantWrapper<std::shared_ptr<std::istream>>>(variants[0]))
            {
                // Validating first stream, it must contain a model
                std::shared_ptr<std::istream> p_model_stream =
                    as_type_ptr<VariantWrapper<std::shared_ptr<std::istream>>>(variants[0])->get();
                paddle::framework::proto::ProgramDesc fw;
                return fw.ParseFromIstream(p_model_stream.get());
            }
            return false;
        }

        InputModel::Ptr
            FrontEndPDPD::load_impl(const std::vector<std::shared_ptr<Variant>>& variants) const
        {
            if (variants.size() == 1)
            {
                // The case when folder with __model__ and weight files is provided or .pdmodel file
                if (is_type<VariantWrapper<std::string>>(variants[0]))
                {
                    std::string m_path =
                        as_type_ptr<VariantWrapper<std::string>>(variants[0])->get();
                    return std::make_shared<InputModelPDPD>(m_path);
                }
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
                else if (is_type<VariantWrapper<std::wstring>>(variants[0]))
                {
                    std::wstring m_path =
                        as_type_ptr<VariantWrapper<std::wstring>>(variants[0])->get();
                    return std::make_shared<InputModelPDPD>(m_path);
                }
#endif
                // The case with only model stream provided and no weights. This means model has
                // no learnable weights
                else if (is_type<VariantWrapper<std::shared_ptr<std::istream>>>(variants[0]))
                {
                    std::shared_ptr<std::istream> p_model_stream =
                        as_type_ptr<VariantWrapper<std::shared_ptr<std::istream>>>(variants[0])
                            ->get();
                    return std::make_shared<InputModelPDPD>(
                        std::vector<std::istream*>{p_model_stream.get()});
                }
            }
            else if (variants.size() == 2)
            {
                // The case when .pdmodel and .pdparams files are provided
                std::ifstream model_stream;
                std::ifstream weights_stream;
                std::istream* p_model_stream =
                    pdpd::variant_to_stream_ptr(variants[0], model_stream);
                std::istream* p_weights_stream =
                    pdpd::variant_to_stream_ptr(variants[1], weights_stream);
                if (p_model_stream && p_weights_stream)
                {
                    return std::make_shared<InputModelPDPD>(
                        std::vector<std::istream*>{p_model_stream, p_weights_stream});
                }
            }
            PDPD_THROW("Model can be loaded either from 1 or 2 files/streams");
        }

        std::shared_ptr<ngraph::Function> FrontEndPDPD::convert(InputModel::Ptr model) const
        {
            auto pdpd_model = std::dynamic_pointer_cast<InputModelPDPD>(model);
            auto f = convert_model(pdpd_model);
            return f;
        }

    } // namespace frontend
} // namespace ngraph

extern "C" PDPD_API FrontEndVersion GetAPIVersion()
{
    return OV_FRONTEND_API_VERSION;
}

extern "C" PDPD_API void* GetFrontEndData()
{
    FrontEndPluginInfo* res = new FrontEndPluginInfo();
    res->m_name = "pdpd";
    res->m_creator = []() { return std::make_shared<FrontEndPDPD>(); };
    return res;
}
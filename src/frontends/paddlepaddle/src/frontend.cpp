// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paddlepaddle_frontend/frontend.hpp"

#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "decoder.hpp"
#include "framework.pb.h"
#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/opsets/opset7.hpp"
#include "paddlepaddle_frontend/exceptions.hpp"
#include "paddlepaddle_frontend/model.hpp"
#include "paddlepaddle_frontend/place.hpp"
#include "pdpd_fw_node.hpp"
#include "pdpd_utils.hpp"

using namespace ov::opset7;
using namespace ov;
using namespace ov::frontend;

namespace ov {
namespace frontend {
namespace pdpd {
namespace {
NamedOutputs make_ng_node(const std::map<pdpd::TensorName, Output<Node>>& nodes,
                          const std::shared_ptr<OpPlacePDPD>& op_place,
                          const std::map<std::string, CreatorFunction>& CREATORS_MAP) {
    const auto& op_desc = op_place->get_desc();

    auto creator_it = CREATORS_MAP.find(op_desc.type());
    FRONT_END_OP_CONVERSION_CHECK(creator_it != CREATORS_MAP.end(), "No creator found for ", op_desc.type(), " node.");
    NamedInputs named_inputs;
    for (const auto& input_port : op_desc.inputs()) {
        for (const auto& in_tensor_name : input_port.arguments()) {
            auto node_it = nodes.find(in_tensor_name);
            // general check, because in case of error partial conversion should fail
            FRONT_END_GENERAL_CHECK(node_it != nodes.end(),
                                    "Input ",
                                    in_tensor_name,
                                    " for node with type ",
                                    op_desc.type(),
                                    " wasn't found. It may happen if model was cut incorrectly.");
            named_inputs[input_port.parameter()].push_back(node_it->second);
        }
    }
    NamedOutputs outputs;
    // In case the conversion function throws exception
    try {
        outputs = creator_it->second(NodeContext(DecoderPDPDProto(op_place), named_inputs));
    } catch (std::exception& ex) {
        FRONT_END_OP_CONVERSION_CHECK(false, "Fail to convert " + op_desc.type() + " Exception " + ex.what());
    }

    return outputs;
}

NamedOutputs make_framework_node(const std::map<pdpd::TensorName, Output<Node>>& nodes,
                                 const std::shared_ptr<OpPlacePDPD>& op_place) {
    const auto& op_desc = op_place->get_desc();

    OutputVector inputs_vector;
    std::vector<std::string> inputs_names;
    NamedOutputs named_outputs;
    for (const auto& input_port : op_desc.inputs()) {
        for (const auto& in_tensor_name : input_port.arguments()) {
            auto it = nodes.find(in_tensor_name);
            // general check, because in case of error partial conversion should fail
            FRONT_END_GENERAL_CHECK(it != nodes.end(),
                                    "Input ",
                                    in_tensor_name,
                                    " for node with type ",
                                    op_desc.type(),
                                    " wasn't found. It may happen if model was cut incorrectly.");
            inputs_vector.push_back(it->second);
            inputs_names.push_back(in_tensor_name);
        }
    }

    auto node =
        std::make_shared<ov::frontend::PDPDFrameworkNode>(DecoderPDPDProto(op_place), inputs_vector, inputs_names);

    return node->return_named_outputs();
}

bool normalize_framework_node(const std::shared_ptr<PDPDFrameworkNode>& node,
                              const std::map<std::string, CreatorFunction>& CREATORS_MAP) {
    auto type = node->get_op_type();
    auto creator_it = CREATORS_MAP.find(type);
    FRONT_END_OP_CONVERSION_CHECK(creator_it != CREATORS_MAP.end(), "No creator found for ", type, " node.");

    auto new_node_outputs = creator_it->second(NodeContext(node->get_decoder(), node->get_named_inputs()));
    auto new_node = new_node_outputs.begin()->second[0].get_node_shared_ptr();
    new_node->set_friendly_name(node->get_friendly_name());
    auto node_outputs = node->return_named_outputs();

    auto new_ports = new_node_outputs.begin();
    auto old_ports = node_outputs.begin();
    for (; new_ports != new_node_outputs.end() && old_ports != node_outputs.end(); ++new_ports, ++old_ports) {
        FRONT_END_OP_CONVERSION_CHECK(new_ports->first == old_ports->first,
                                      "Node outputs inconsistent after normalization: ",
                                      node->get_friendly_name());
        auto new_output = new_ports->second.begin();
        auto old_output = old_ports->second.begin();
        for (; new_output != new_ports->second.end() && old_output != old_ports->second.end();
             ++old_output, ++new_output) {
            old_output->replace(*new_output);
        }
    }
    return true;
}

std::istream* variant_to_stream_ptr(const ov::Any& variant, std::ifstream& ext_stream) {
    if (variant.is<std::istream*>()) {
        return variant.as<std::istream*>();
    } else if (variant.is<std::string>()) {
        const auto& model_path = variant.as<std::string>();
        ext_stream.open(model_path, std::ios::in | std::ifstream::binary);
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variant.is<std::wstring>()) {
        const auto& model_path = variant.as<std::wstring>();
        ext_stream.open(model_path, std::ios::in | std::ifstream::binary);
    }
#endif
    FRONT_END_INITIALIZATION_CHECK(ext_stream && ext_stream.is_open(), "Cannot open model file.");
    return &ext_stream;
}
}  // namespace
}  // namespace pdpd

std::shared_ptr<ngraph::Function> FrontEndPDPD::convert_each_node(
    const std::shared_ptr<InputModelPDPD>& model,
    std::function<std::map<std::string, OutputVector>(const std::map<std::string, Output<Node>>&,
                                                      const std::shared_ptr<OpPlacePDPD>&)> func) {
    auto nodes_dict(model->get_tensor_values());
    ParameterVector parameter_nodes;
    ResultVector result_nodes;

    for (const auto& _inp_place : model->get_inputs()) {
        const auto& inp_place = std::dynamic_pointer_cast<TensorPlacePDPD>(_inp_place);
        const auto& var = inp_place->get_desc();
        const auto& shape = inp_place->get_partial_shape();
        const auto& type = inp_place->get_element_type();
        auto param = std::make_shared<Parameter>(type, shape);
        param->set_friendly_name(var.name());
        param->output(0).get_tensor().add_names({var.name()});
        nodes_dict[var.name()] = param;
        parameter_nodes.push_back(param);
    }

    const auto& op_places = model->get_op_places();
    for (const auto& op_place : op_places) {
        const auto& op_desc = op_place->get_desc();
        if (op_desc.type() == "feed" || op_desc.type() == "fetch") {
            // inputs and outputs are stored in the model already
            continue;
        } else {
            pdpd::NamedOutputs named_outputs = func(nodes_dict, op_place);

            if (!named_outputs.empty()) {
                if (op_desc.outputs().begin()->arguments().size() > 0) {
                    const auto& tensor_name = op_desc.outputs().begin()->arguments()[0];
                    auto node = named_outputs.begin()->second[0].get_node_shared_ptr();
                    node->set_friendly_name(tensor_name);
                }

                const auto& out_ports = op_desc.outputs();
                for (const auto& port : out_ports) {
                    // TODO: figure a way to safely handle unused outputs
                    if (named_outputs.count(port.parameter())) {
                        const auto& ng_outputs = named_outputs.at(port.parameter());
                        FRONT_END_OP_CONVERSION_CHECK(ng_outputs.size() == port.arguments_size(),
                                                      "The number of output tensors must be equal to "
                                                      "the number of outputs of the OV node.");
                        for (size_t idx = 0; idx < ng_outputs.size(); ++idx) {
                            const auto& var_name = port.arguments()[idx];
                            ng_outputs[idx].get_tensor().set_names({var_name});
                            // if nodes_dict already has node mapped to this tensor name it
                            // usually means that it was overwritten using setTensorValue
                            if (!nodes_dict.count(var_name))
                                nodes_dict[var_name] = ng_outputs[idx];
                        }
                    }
                }
            }
        }
    }

    for (const auto& _outp_place : model->get_outputs()) {
        const auto& outp_place = std::dynamic_pointer_cast<TensorPlacePDPD>(_outp_place);
        auto var = outp_place->get_desc();
        auto input_var_name = var.name();
        auto result = std::make_shared<Result>(nodes_dict.at(input_var_name));
        result->set_friendly_name(input_var_name + "/Result");
        result_nodes.push_back(result);
    }

    return std::make_shared<ov::Model>(result_nodes, parameter_nodes);
}

bool FrontEndPDPD::supported_impl(const std::vector<ov::Any>& variants) const {
    // FrontEndPDPD can only load model specified by one path, one file or two files.
    if (variants.empty() || variants.size() > 2)
        return false;

    // Validating first path, it must contain a model
    if (variants[0].is<std::string>()) {
        std::string suffix = ".pdmodel";
        std::string model_path = variants[0].as<std::string>();
        if (!pdpd::endsWith(model_path, suffix)) {
            model_path += pdpd::get_path_sep<char>() + "__model__";
        }
        std::ifstream model_str(model_path, std::ios::in | std::ifstream::binary);
        // It is possible to validate here that protobuf can read model from the stream,
        // but it will complicate the check, while it should be as quick as possible
        return model_str && model_str.is_open();
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variants[0].is<std::wstring>()) {
        std::wstring suffix = L".pdmodel";
        std::wstring model_path = variants[0].as<std::wstring>();
        if (!pdpd::endsWith(model_path, suffix)) {
            model_path += pdpd::get_path_sep<wchar_t>() + L"__model__";
        }
        std::ifstream model_str(model_path, std::ios::in | std::ifstream::binary);
        // It is possible to validate here that protobuf can read model from the stream,
        // but it will complicate the check, while it should be as quick as possible
        return model_str && model_str.is_open();
    }
#endif
    else if (variants[0].is<std::istream*>()) {
        // Validating first stream, it must contain a model
        auto p_model_stream = variants[0].as<std::istream*>();
        paddle::framework::proto::ProgramDesc fw;
        return fw.ParseFromIstream(p_model_stream);
    }
    return false;
}

InputModel::Ptr FrontEndPDPD::load_impl(const std::vector<ov::Any>& variants) const {
    if (variants.size() == 1) {
        // The case when folder with __model__ and weight files is provided or .pdmodel file
        if (variants[0].is<std::string>()) {
            std::string m_path = variants[0].as<std::string>();
            return std::make_shared<InputModelPDPD>(m_path, m_telemetry);
        }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        else if (variants[0].is<std::wstring>()) {
            std::wstring m_path = variants[0].as<std::wstring>();
            return std::make_shared<InputModelPDPD>(m_path, m_telemetry);
        }
#endif
        // The case with only model stream provided and no weights. This means model has
        // no learnable weights
        else if (variants[0].is<std::istream*>()) {
            auto p_model_stream = variants[0].as<std::istream*>();
            return std::make_shared<InputModelPDPD>(std::vector<std::istream*>{p_model_stream}, m_telemetry);
        }
    } else if (variants.size() == 2) {
        // The case when .pdmodel and .pdparams files are provided
        std::ifstream model_stream;
        std::ifstream weights_stream;
        std::istream* p_model_stream = pdpd::variant_to_stream_ptr(variants[0], model_stream);
        std::istream* p_weights_stream = pdpd::variant_to_stream_ptr(variants[1], weights_stream);
        if (p_model_stream && p_weights_stream) {
            return std::make_shared<InputModelPDPD>(std::vector<std::istream*>{p_model_stream, p_weights_stream},
                                                    m_telemetry);
        }
    }
    PDPD_THROW("Model can be loaded either from 1 or 2 files/streams");
}

std::shared_ptr<ov::Model> FrontEndPDPD::convert(InputModel::Ptr model) const {
    auto pdpd_model = std::dynamic_pointer_cast<InputModelPDPD>(model);
    std::map<std::string, pdpd::CreatorFunction> CREATORS_MAP = pdpd::get_supported_ops();
    auto f = convert_each_node(
        pdpd_model,
        [&](const std::map<std::string, Output<Node>>& nodes_dict, const std::shared_ptr<OpPlacePDPD>& op_place) {
            return pdpd::make_ng_node(nodes_dict, op_place, CREATORS_MAP);
        });
    return f;
}

void FrontEndPDPD::convert(std::shared_ptr<ov::Model> partiallyConverted) const {
    for (const auto& node : partiallyConverted->get_ordered_ops()) {
        if (ov::is_type<PDPDFrameworkNode>(node)) {
            pdpd::normalize_framework_node(std::dynamic_pointer_cast<PDPDFrameworkNode>(node),
                                           pdpd::get_supported_ops());
        }
    }
    for (auto result : partiallyConverted->get_results()) {
        result->validate_and_infer_types();
    }
}

std::shared_ptr<ov::Model> FrontEndPDPD::convert_partially(InputModel::Ptr model) const {
    auto pdpd_model = std::dynamic_pointer_cast<InputModelPDPD>(model);
    std::map<std::string, pdpd::CreatorFunction> CREATORS_MAP = pdpd::get_supported_ops();
    auto f = convert_each_node(
        pdpd_model,
        [&](const std::map<std::string, Output<Node>>& nodes_dict, const std::shared_ptr<OpPlacePDPD>& op_place) {
            pdpd::NamedOutputs named_outputs;
            try {
                named_outputs = pdpd::make_ng_node(nodes_dict, op_place, CREATORS_MAP);
            } catch (const OpConversionFailure&) {
                named_outputs = pdpd::make_framework_node(nodes_dict, op_place);
            }
            return named_outputs;
        });
    return f;
}

std::shared_ptr<ov::Model> FrontEndPDPD::decode(InputModel::Ptr model) const {
    auto pdpd_model = std::dynamic_pointer_cast<InputModelPDPD>(model);
    std::map<std::string, pdpd::CreatorFunction> CREATORS_MAP = pdpd::get_supported_ops();
    auto f = convert_each_node(pdpd_model, pdpd::make_framework_node);
    return f;
}

std::string FrontEndPDPD::get_name() const {
    return "paddle";
}

void FrontEndPDPD::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_telemetry = telemetry;
    }
}

}  // namespace frontend
}  // namespace ov

PDPD_C_API FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

PDPD_C_API void* GetFrontEndData() {
    FrontEndPluginInfo* res = new FrontEndPluginInfo();
    res->m_name = "paddle";
    res->m_creator = []() {
        return std::make_shared<FrontEndPDPD>();
    };
    return res;
}

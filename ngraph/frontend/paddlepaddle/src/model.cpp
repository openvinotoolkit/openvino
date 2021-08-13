// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <ngraph/opsets/opset7.hpp>
#include <paddlepaddle_frontend/exceptions.hpp>
#include <paddlepaddle_frontend/model.hpp>
#include <paddlepaddle_frontend/place.hpp>
#include <queue>

#include "decoder.hpp"
#include "framework.pb.h"
#include "node_context.hpp"
#include "pdpd_utils.hpp"

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
#    include <codecvt>
#    include <locale>
#endif

namespace ngraph {
namespace frontend {
using namespace paddle::framework::proto;

class InputModelPDPD::InputModelPDPDImpl {
public:
    template <typename T>
    InputModelPDPDImpl(const std::basic_string<T>& path, const InputModel& input_model);
    InputModelPDPDImpl(const std::vector<std::istream*>& streams, const InputModel& input_model);
    std::vector<Place::Ptr> getInputs() const;
    std::vector<Place::Ptr> getOutputs() const;
    Place::Ptr getPlaceByTensorName(const std::string& tensorName) const;
    void overrideAllOutputs(const std::vector<Place::Ptr>& outputs);
    void overrideAllInputs(const std::vector<Place::Ptr>& inputs);
    void extractSubgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs);
    void setDefaultShape(Place::Ptr place, const ngraph::Shape&);
    void setPartialShape(Place::Ptr place, const ngraph::PartialShape&);
    ngraph::PartialShape getPartialShape(Place::Ptr place) const;
    void setElementType(Place::Ptr place, const ngraph::element::Type&);
    void setTensorValue(Place::Ptr place, const void* value);

    std::vector<std::shared_ptr<OpPlacePDPD>> get_op_places() const;
    std::map<std::string, std::shared_ptr<TensorPlacePDPD>> get_var_places() const {
        return m_var_places;
    }
    std::map<pdpd::TensorName, Output<Node>> get_tensor_values() const {
        return m_tensor_values;
    };

private:
    void loadPlaces();
    template <typename T>
    void loadConsts(const std::basic_string<T>& folder_with_weights, std::istream* weight_stream);
    std::vector<std::shared_ptr<OpPlacePDPD>> determine_cut_nodes() const;

    std::vector<std::shared_ptr<OpPlacePDPD>> m_op_places;
    std::map<std::string, std::shared_ptr<TensorPlacePDPD>> m_var_places;
    std::shared_ptr<ProgramDesc> m_fw_ptr;
    const InputModel& m_input_model;
    std::vector<Place::Ptr> m_inputs;
    std::vector<Place::Ptr> m_outputs;
    std::map<pdpd::TensorName, Output<Node>> m_tensor_values;

    // shows if some nodes might be deleted from graph
    bool m_graph_changed = false;
};

void InputModelPDPD::InputModelPDPDImpl::loadPlaces() {
    const int cnt_of_blocks = m_fw_ptr->blocks_size();
    const auto& blocks = m_fw_ptr->blocks();

    for (int block_idx = 0; block_idx < cnt_of_blocks; block_idx++) {
        const auto& block = blocks[block_idx];

        for (const auto& var : block.vars()) {
            m_var_places[var.name()] = std::make_shared<TensorPlacePDPD>(m_input_model, var);
        }

        for (const auto& op : block.ops()) {
            auto op_place = std::make_shared<OpPlacePDPD>(m_input_model, op);
            m_op_places.push_back(op_place);

            for (const auto& output : op.outputs()) {
                for (const auto& var_name : output.arguments()) {
                    auto out_port = std::make_shared<OutPortPlacePDPD>(m_input_model);

                    // connect out_port and tensor
                    const auto& tensor = m_var_places.at(var_name);
                    tensor->add_producing_port(out_port);
                    out_port->set_target_tensor(tensor);

                    // connect out_port and op
                    op_place->add_out_port(out_port, output.parameter());
                    out_port->set_op(op_place);
                }
            }

            for (const auto& input : op.inputs()) {
                for (const auto& var_name : input.arguments()) {
                    auto in_port = std::make_shared<InPortPlacePDPD>(m_input_model);

                    // connect in_port and tensor
                    const auto& tensor = m_var_places.at(var_name);
                    tensor->add_consuming_port(in_port);
                    in_port->set_source_tensor(tensor);

                    // connect in_port and op
                    op_place->add_in_port(in_port, input.parameter());
                    in_port->set_op(op_place);
                }
            }

            // Determine outputs and inputs
            if (op.type() == "feed") {
                const auto& place = op_place->get_output_port_pdpd("Out", 0);
                const auto& var_place = std::dynamic_pointer_cast<TensorPlacePDPD>(place->get_target_tensor_pdpd());
                const auto& tensor_desc = var_place->get_desc().type().lod_tensor().tensor();
                const auto& dims = tensor_desc.dims();

                var_place->set_element_type(TYPE_MAP[tensor_desc.data_type()]);
                var_place->set_partial_shape(PartialShape(std::vector<Dimension>(dims.begin(), dims.end())));
                m_inputs.push_back(var_place);
            } else if (op.type() == "fetch") {
                auto place = op_place->get_input_port_pdpd("X", 0);
                m_outputs.push_back(place->get_source_tensor_pdpd());
            }
        }
    }
}

namespace pdpd {
bool read_tensor(std::istream& is, char* data, size_t len) {
    std::vector<char> header(16);
    is.read(&header[0], 16);
    uint32_t dims_len = 0;
    is.read(reinterpret_cast<char*>(&dims_len), 4);
    std::vector<char> dims_struct(dims_len);
    is.read(&dims_struct[0], dims_len);
    is.read(data, len);
    if (is.gcount() != len)
        return false;
    return true;
}

template <typename T>
std::basic_string<T> get_const_path(const std::basic_string<T>& folder_with_weights, const std::string& name) {
    return folder_with_weights + pdpd::get_path_sep<T>() + name;
}

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_const_path(const std::basic_string<wchar_t>& folder, const std::string& name) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    std::wstring _name = converter.from_bytes(name);
    return folder + pdpd::get_path_sep<wchar_t>() + _name;
}
#endif

template <typename T>
std::basic_string<T> get_model_path(const std::basic_string<T>& path, std::ifstream* weights_stream) {
    std::string model_file{path};
    std::string ext = ".pdmodel";
    if (pdpd::endsWith(model_file, ext)) {
        std::string params_ext = ".pdiparams";
        std::string weights_file{path};
        weights_file.replace(weights_file.size() - ext.size(), ext.size(), params_ext);
        weights_stream->open(weights_file, std::ios::binary);
        // Don't throw error if file isn't opened
        // It may mean that model don't have constants
    } else {
        model_file += pdpd::get_path_sep<T>() + "__model__";
    }
    return model_file;
}

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
template <>
std::basic_string<wchar_t> get_model_path(const std::basic_string<wchar_t>& path, std::ifstream* weights_stream) {
    std::wstring model_file{path};
    std::wstring ext = L".pdmodel";
    if (pdpd::endsWith(model_file, ext)) {
        std::wstring params_ext = L".pdiparams";
        std::wstring weights_file{path};
        weights_file.replace(weights_file.size() - ext.size(), ext.size(), params_ext);
        weights_stream->open(weights_file, std::ios::binary);
        // Don't throw error if file isn't opened
        // It may mean that model don't have constants
    } else {
        model_file += pdpd::get_path_sep<wchar_t>() + L"__model__";
    }
    return model_file;
}
#endif
}  // namespace pdpd

std::vector<std::shared_ptr<OpPlacePDPD>> InputModelPDPD::InputModelPDPDImpl::get_op_places() const {
    if (m_graph_changed) {
        return determine_cut_nodes();
    }
    return m_op_places;
}

std::vector<std::shared_ptr<OpPlacePDPD>> InputModelPDPD::InputModelPDPDImpl::determine_cut_nodes() const {
    std::queue<OpPlacePDPD*> q;
    std::unordered_set<OpPlacePDPD*> visited;
    std::vector<std::shared_ptr<OpPlacePDPD>> new_op_places;
    new_op_places.reserve(m_op_places.size());
    // Marking nodes from outputs to inputs/constants
    for (const auto& output : getOutputs()) {
        if (!output->is_input()) {
            auto pdpd_output_op = std::dynamic_pointer_cast<OpPlacePDPD>(output->get_producing_operation());
            PDPD_ASSERT(pdpd_output_op != nullptr, "Output doesn't have producing operation");
            if (!visited.count(pdpd_output_op.get())) {
                visited.insert(pdpd_output_op.get());
                q.push(pdpd_output_op.get());
                new_op_places.push_back(pdpd_output_op);
            }
        }
    }
    while (!q.empty()) {
        auto p_op = q.front();
        q.pop();
        for (const auto& map_pair : p_op->get_input_ports()) {
            for (const auto& port : map_pair.second) {
                auto tensor = port->get_source_tensor();
                if (tensor && !tensor->is_input() && !m_tensor_values.count(tensor->get_names()[0])) {
                    std::shared_ptr<OpPlacePDPD> pdpd_op =
                        std::dynamic_pointer_cast<OpPlacePDPD>(tensor->get_producing_operation());
                    if (pdpd_op && !visited.count(pdpd_op.get())) {
                        visited.insert(pdpd_op.get());
                        q.push(pdpd_op.get());
                        new_op_places.push_back(pdpd_op);
                    }
                }
            }
        }
    }
    std::reverse(new_op_places.begin(), new_op_places.end());
    return new_op_places;
}

template <typename T>
void InputModelPDPD::InputModelPDPDImpl::loadConsts(const std::basic_string<T>& folder_with_weights,
                                                    std::istream* weight_stream) {
    for (const auto& item : m_var_places) {
        const auto& var_desc = item.second->get_desc();
        const auto& name = item.first;
        if (pdpd::endsWith(name, std::string{"feed"}) || pdpd::endsWith(name, std::string{"fetch"}))
            continue;
        if (!var_desc.persistable())
            continue;

        FRONT_END_GENERAL_CHECK(var_desc.type().type() == paddle::framework::proto::VarType::LOD_TENSOR);
        const auto& tensor = var_desc.type().lod_tensor().tensor();
        Shape shape(tensor.dims().cbegin(), tensor.dims().cend());
        const auto& type = TYPE_MAP[tensor.data_type()];
        const auto& data_length = shape_size(shape) * type.size();
        std::vector<uint8_t> tensor_data(data_length);

        bool read_succeed = false;
        if (weight_stream) {
            read_succeed = pdpd::read_tensor(*weight_stream, reinterpret_cast<char*>(&tensor_data[0]), data_length);
        } else if (!folder_with_weights.empty()) {
            std::ifstream is(pdpd::get_const_path(folder_with_weights, name), std::ios::in | std::ifstream::binary);
            FRONT_END_GENERAL_CHECK(is && is.is_open(), "Cannot open file for constant value.");
            read_succeed = pdpd::read_tensor(is, reinterpret_cast<char*>(&tensor_data[0]), data_length);
        } else {
            FRONT_END_GENERAL_CHECK(false, "Either folder with weights or stream must be provided.");
        }
        FRONT_END_GENERAL_CHECK(read_succeed,
                                "File containing constant with name ",
                                name,
                                " wasn't successfully read.");

        auto const_node = opset7::Constant::create(type, shape, &tensor_data[0]);
        const_node->set_friendly_name(name);
        m_tensor_values[name] = const_node;
    }
}

template <typename T>
InputModelPDPD::InputModelPDPDImpl::InputModelPDPDImpl(const std::basic_string<T>& path, const InputModel& input_model)
    : m_fw_ptr{std::make_shared<ProgramDesc>()},
      m_input_model(input_model) {
    std::string empty_str = "";
    std::ifstream weights_stream;
    std::ifstream pb_stream(pdpd::get_model_path<T>(path, &weights_stream), std::ios::in | std::ifstream::binary);

    FRONT_END_GENERAL_CHECK(pb_stream && pb_stream.is_open(), "Model file doesn't exist");
    FRONT_END_GENERAL_CHECK(m_fw_ptr->ParseFromIstream(&pb_stream), "Model can't be parsed");

    loadPlaces();
    if (weights_stream && weights_stream.is_open()) {
        loadConsts(std::basic_string<T>{}, &weights_stream);
    } else {
        loadConsts(path, nullptr);
    }
}

InputModelPDPD::InputModelPDPDImpl::InputModelPDPDImpl(const std::vector<std::istream*>& streams,
                                                       const InputModel& input_model)
    : m_fw_ptr{std::make_shared<ProgramDesc>()},
      m_input_model(input_model) {
    if (streams.size() != 1) {
        FRONT_END_GENERAL_CHECK(streams.size() == 2,
                                "Two streams are needed to load a model: model and weights streams");
    }
    FRONT_END_GENERAL_CHECK(m_fw_ptr->ParseFromIstream(streams[0]), "Model can't be parsed");

    loadPlaces();
    if (streams.size() > 1)
        loadConsts(std::string(), streams[1]);
}

std::vector<Place::Ptr> InputModelPDPD::InputModelPDPDImpl::getInputs() const {
    return m_inputs;
}

std::vector<Place::Ptr> InputModelPDPD::InputModelPDPDImpl::getOutputs() const {
    return m_outputs;
}

Place::Ptr InputModelPDPD::InputModelPDPDImpl::getPlaceByTensorName(const std::string& tensorName) const {
    if (m_var_places.count(tensorName))
        return m_var_places.at(tensorName);
    return nullptr;
}

namespace pdpd {
std::shared_ptr<TensorPlacePDPD> castToTensorPlace(const Place::Ptr& place) {
    if (auto var_place = std::dynamic_pointer_cast<TensorPlacePDPD>(place)) {
        return var_place;
    } else if (auto in_port_place = std::dynamic_pointer_cast<InPortPlacePDPD>(place)) {
        return in_port_place->get_source_tensor_pdpd();
    } else if (auto out_port_place = std::dynamic_pointer_cast<OutPortPlacePDPD>(place)) {
        return out_port_place->get_target_tensor_pdpd();
    }
    FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlacePDPD.");
}

}  // namespace pdpd

void InputModelPDPD::InputModelPDPDImpl::overrideAllInputs(const std::vector<Place::Ptr>& inputs) {
    m_graph_changed = true;
    m_inputs.clear();
    for (const auto& inp : inputs) {
        m_inputs.push_back(pdpd::castToTensorPlace(inp));
    }
}

void InputModelPDPD::InputModelPDPDImpl::overrideAllOutputs(const std::vector<Place::Ptr>& outputs) {
    m_graph_changed = true;
    m_outputs.clear();
    for (const auto& outp : outputs) {
        m_outputs.push_back(pdpd::castToTensorPlace(outp));
    }
}

void InputModelPDPD::InputModelPDPDImpl::extractSubgraph(const std::vector<Place::Ptr>& inputs,
                                                         const std::vector<Place::Ptr>& outputs) {
    m_graph_changed = true;
    overrideAllInputs(inputs);
    overrideAllOutputs(outputs);
}

void InputModelPDPD::InputModelPDPDImpl::setDefaultShape(Place::Ptr place, const ngraph::Shape& shape) {
    FRONT_END_NOT_IMPLEMENTED("setDefaultShape");
}

void InputModelPDPD::InputModelPDPDImpl::setPartialShape(Place::Ptr place, const ngraph::PartialShape& p_shape) {
    pdpd::castToTensorPlace(place)->set_partial_shape(p_shape);
}

ngraph::PartialShape InputModelPDPD::InputModelPDPDImpl::getPartialShape(Place::Ptr place) const {
    return pdpd::castToTensorPlace(place)->get_partial_shape();
}

void InputModelPDPD::InputModelPDPDImpl::setElementType(Place::Ptr place, const ngraph::element::Type& type) {
    pdpd::castToTensorPlace(place)->set_element_type(type);
}

void InputModelPDPD::InputModelPDPDImpl::setTensorValue(Place::Ptr place, const void* value) {
    m_graph_changed = true;
    auto tensor_place = pdpd::castToTensorPlace(place);
    auto p_shape = tensor_place->get_partial_shape();
    auto type = tensor_place->get_element_type();
    auto constant = opset7::Constant::create(type, p_shape.to_shape(), value);
    auto name = tensor_place->get_names()[0];
    constant->set_friendly_name(name);
    m_tensor_values[name] = constant;
}

InputModelPDPD::InputModelPDPD(const std::string& path) : _impl{std::make_shared<InputModelPDPDImpl>(path, *this)} {}

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
InputModelPDPD::InputModelPDPD(const std::wstring& path) : _impl{std::make_shared<InputModelPDPDImpl>(path, *this)} {}
#endif

InputModelPDPD::InputModelPDPD(const std::vector<std::istream*>& streams)
    : _impl{std::make_shared<InputModelPDPDImpl>(streams, *this)} {}

std::vector<std::shared_ptr<OpPlacePDPD>> InputModelPDPD::get_op_places() const {
    return _impl->get_op_places();
}

std::map<std::string, std::shared_ptr<TensorPlacePDPD>> InputModelPDPD::get_var_places() const {
    return _impl->get_var_places();
}

std::map<pdpd::TensorName, Output<Node>> InputModelPDPD::get_tensor_values() const {
    return _impl->get_tensor_values();
}

std::vector<Place::Ptr> InputModelPDPD::get_inputs() const {
    return _impl->getInputs();
}

std::vector<Place::Ptr> InputModelPDPD::get_outputs() const {
    return _impl->getOutputs();
}

Place::Ptr InputModelPDPD::get_place_by_tensor_name(const std::string& tensorName) const {
    return _impl->getPlaceByTensorName(tensorName);
}

void InputModelPDPD::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    _impl->overrideAllOutputs(outputs);
}

void InputModelPDPD::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    _impl->overrideAllInputs(inputs);
}

void InputModelPDPD::extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    _impl->extractSubgraph(inputs, outputs);
}

void InputModelPDPD::set_partial_shape(Place::Ptr place, const ngraph::PartialShape& p_shape) {
    _impl->setPartialShape(place, p_shape);
}

ngraph::PartialShape InputModelPDPD::get_partial_shape(Place::Ptr place) const {
    return _impl->getPartialShape(place);
}

void InputModelPDPD::set_element_type(Place::Ptr place, const ngraph::element::Type& type) {
    _impl->setElementType(place, type);
}

void InputModelPDPD::set_tensor_value(Place::Ptr place, const void* value) {
    _impl->setTensorValue(place, value);
}

}  // namespace frontend
}  // namespace ngraph

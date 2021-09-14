// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fstream>
#include <numeric>
#include <queue>
#include <frontend_manager/frontend_exceptions.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <tensorflow_frontend/model.hpp>
#include <tensorflow_frontend/place.hpp>
#include <tensorflow_frontend/utility.hpp>

//#include "graph.pb.h"
//#include "tensor.pb.h"

#include <ngraph/pass/manager.hpp>

#include "ngraph/op/util/logical_reduction.hpp"
#include "ngraph/pass/constant_folding.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/pass_config.hpp"
#include "ngraph/slice_plan.hpp"
//#include <ngraph/pass/transpose_sinking.h>
#include <frontend_manager/frontend_exceptions.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "default_opset.h"
#include "graph.hpp"
#include "ngraph_builder.h"
#include "ngraph_conversions.h"

using namespace google;

using namespace ngraph::frontend;

using ::tensorflow::GraphDef;
using ::tensorflow::ngraph_bridge::GraphIteratorProto;
using ::tensorflow::ngraph_bridge::NodeProtoWrapper;

/*
InputModelTensorflow::InputModelTensorflow(const std::string& _path) : path(_path) {
    std::ifstream pb_stream(path, std::ios::binary);
    graph_def = std::make_shared<GraphDef>();

    FRONT_END_GENERAL_CHECK(pb_stream && pb_stream.is_open(), "Model file doesn't exist");
    FRONT_END_GENERAL_CHECK(graph_def->ParseFromIstream(&pb_stream), "Model can't be parsed");

    std::cout << "[ INFO ] Loaded model contains " << graph_def->node_size() << " nodes." << std::endl;
    graph_impl = std::make_shared<::tensorflow::ngraph_bridge::GraphIteratorProto>(graph_def.get());

    initial_traverse_graph();
}

InputModelTensorflow::InputModelTensorflow(std::shared_ptr<::tensorflow::GraphDef> _graph_def,
                                           std::vector<ngraph::PartialShape> _input_shapes)
    : input_shapes(_input_shapes) {
    graph_impl = std::make_shared<::tensorflow::ngraph_bridge::GraphIteratorProto>(_graph_def.get());

    initial_traverse_graph();
}

InputModelTensorflow::InputModelTensorflow(const std::vector<std::shared_ptr<::tensorflow::NodeDef>>& _nodes_def,
                                           std::vector<ngraph::PartialShape> _input_shapes)
    : input_shapes(_input_shapes) {
    graph_impl = std::make_shared<::tensorflow::ngraph_bridge::GraphIteratorProto>(_nodes_def);

    initial_traverse_graph();
}

std::vector<Place::Ptr> InputModelTensorflow::get_inputs() const {
    return m_inputs;
}

std::vector<Place::Ptr> InputModelTensorflow::get_outputs() const {
    return m_outputs;
}

void InputModelTensorflow::set_partial_shape(Place::Ptr place, const ngraph::PartialShape& pshape) {
    auto place_tf = std::dynamic_pointer_cast<PlaceTF>(place);
    partialShapes[place_tf->get_names()[0]] = pshape;
}

ngraph::PartialShape InputModelTensorflow::get_partial_shape(Place::Ptr place) const {
    auto place_tf = std::dynamic_pointer_cast<PlaceTF>(place);
    ngraph::PartialShape result_shape;
    // TODO: replace by node cache without going through all nodes each time
    for (; !graph_impl->is_end(); graph_impl->next()) {
        auto node = graph_impl->get();
        if (node->name() == place_tf->get_names()[0]) {
            node->getAttrValue2("shape", &result_shape);
            break;
        }
    }
    // WARNING! Redesign GraphIterator -- it is not really good thing, detach an iterator from graph itself
    graph_impl->reset();
    return result_shape;
}

std::vector<std::shared_ptr<ngraph::frontend::OpPlaceTF>> InputModelTensorflow::get_op_places() const {
    // TODO: call that ONLY if model modified
    return determine_cut_nodes();
}

void InputModelTensorflow::initial_traverse_graph() {
    std::set<std::string> all_names;
    std::set<std::string> names_with_consumers;

    m_inputs.clear();
    for (; !graph_impl->is_end(); graph_impl->next()) {
        auto op = graph_impl->get();
        all_names.insert(op->name());
        m_ops_topology_sorted.push_back(std::make_shared<OpPlaceTF>(*this, op));
        m_ops[op->name()] = m_ops_topology_sorted.back();
        if (graph_impl->get()->op() == "Placeholder") {
            m_inputs.push_back(m_ops_topology_sorted.back());
        }
        for (size_t i = 0; i < op->num_inputs(); ++i) {
            std::string input_name;
            size_t port_idx;
            try {
                op->input_node(i, &input_name, &port_idx);
                auto port_place = std::make_shared<OutPortPlaceTF>(*this);
                port_place->set_op(m_ops_topology_sorted.back());
                m_ops[input_name]->add_out_port(std::make_shared<OutPortPlaceTF>(*this), port_idx);
                names_with_consumers.insert(input_name);
            } catch (const std::exception& e) {
                std::cerr << "[ ERROR ] Exception happened when preparing input " << i << " for op '" << op->name()
                          << "', expected input name: '" << input_name << "', expected input port index: " << port_idx
                          << '\n';
                throw;
            }
        }
    }
    std::set<std::string> names_without_consumers;
    std::set_difference(all_names.begin(),
                        all_names.end(),
                        names_with_consumers.begin(),
                        names_with_consumers.end(),
                        std::inserter(names_without_consumers, names_without_consumers.begin()));
    graph_impl->reset();

    m_outputs.clear();
    for (auto& out_name : names_without_consumers) {
        m_outputs.push_back(m_ops[out_name]);
    }
}

std::vector<std::shared_ptr<ngraph::frontend::OpPlaceTF>> InputModelTensorflow::determine_cut_nodes() const {
    std::queue<tensorflow::detail::TFNodeDecoder*> q;
    std::unordered_set<std::string> visited;
    std::vector<std::shared_ptr<ngraph::frontend::OpPlaceTF>> new_ops;
    for (const auto& output_op : m_outputs) {
        auto op_name = output_op->get_names()[0];
        if (!visited.count(op_name)) {
            visited.insert(op_name);
            auto out_op_place = std::dynamic_pointer_cast<ngraph::frontend::OpPlaceTF>(output_op);
            if (out_op_place) {
                // TODO: throw if nullptr
                new_ops.push_back(out_op_place);
                q.push(out_op_place->get_desc().get());
            }
        }
    }
    while (!q.empty()) {
        auto op = q.front();
        q.pop();
        for (size_t i = 0; i < op->num_inputs(); ++i) {
            std::string input_name;
            size_t port_idx;
            try {
                op->input_node(i, &input_name, &port_idx);
            } catch (const std::exception& e) {
                std::cerr << "[ ERROR ] Exception happened when preparing input " << i << " for op '" << op->name()
                          << "', expected input name: '" << input_name << "', expected input port index: " << port_idx
                          << '\n';
                throw;
            }
            auto op_it = m_ops.find(input_name);
            // if (tensor && !tensor->is_input() && !m_tensor_values.count(tensor->get_names()[0]))
            if (op_it != m_ops.end() && !visited.count(input_name)) {
                visited.insert(input_name);
                new_ops.push_back(op_it->second);
                // TODO: check that op is frozen
                if (!op_it->second->is_input()) {
                    q.push(op_it->second->get_desc().get());
                }
            }
        }
    }
    std::reverse(new_ops.begin(), new_ops.end());
    return new_ops;
}
*/

// ------------------------------------------------------

namespace ngraph {
namespace frontend {

class InputModelTF::InputModelTFImpl {
public:
    template <typename T>
    InputModelTFImpl(const std::basic_string<T>& path, const InputModel& input_model);
    InputModelTFImpl(const std::vector<std::istream*>& streams, const InputModel& input_model);
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

    std::vector<std::shared_ptr<OpPlaceTF>> get_op_places() const;
    std::map<std::string, std::shared_ptr<TensorPlaceTF>> get_var_places() const {
        return m_var_places;
    }
    std::map<std::string, Output<Node>> get_tensor_values() const {
        return m_tensor_values;
    };

private:
    void loadPlaces();
    template <typename T>
    void loadConsts(const std::basic_string<T>& folder_with_weights, std::istream* weight_stream);
    std::vector<std::shared_ptr<OpPlaceTF>> determine_cut_nodes() const;

    std::vector<std::shared_ptr<OpPlaceTF>> m_op_places;
    std::map<std::string, std::shared_ptr<OpPlaceTF>> m_op_places_map;
    mutable std::map<std::string, std::shared_ptr<TensorPlaceTF>> m_var_places;
    std::shared_ptr<::tensorflow::GraphDef> m_graph_def;
    std::shared_ptr<::tensorflow::ngraph_bridge::GraphIteratorProto> m_graph_impl;
    const InputModel& m_input_model;
    std::vector<Place::Ptr> m_inputs;
    std::vector<Place::Ptr> m_outputs;
    std::map<std::string, Output<Node>> m_tensor_values;

    // shows if some nodes might be deleted from graph
    bool m_graph_changed = false;
};

void InputModelTF::InputModelTFImpl::loadPlaces() {
    std::set<std::string> all_names;
    std::set<std::string> names_with_consumers;

    m_inputs.clear();
    for (; !m_graph_impl->is_end(); m_graph_impl->next()) {
        auto op = m_graph_impl->get();
        all_names.insert(op->name());
        m_op_places.push_back(std::make_shared<OpPlaceTF>(m_input_model, op));
        m_op_places_map[op->name()] = m_op_places.back();
        if (m_graph_impl->get()->op() == "Placeholder") {
            //m_inputs.push_back(m_op_places.back());
            ngraph::PartialShape pshape;
            op->getAttrValue2("shape", &pshape);
            ngraph::element::Type type;
            op->getAttrValue2("dtype", &type);
            TensorPlaceTF a(m_input_model, pshape, type, {op->name()});
            std::vector<std::string> names = {op->name()};
            auto m_var_place = std::make_shared<TensorPlaceTF>(m_input_model, pshape, type, names);
            m_var_places[op->name()] = m_var_place;
            m_inputs.push_back(m_var_place);

        }
        for (size_t i = 0; i < op->num_inputs(); ++i) {
            std::string input_name;
            size_t port_idx;
            try {
                op->input_node(i, &input_name, &port_idx);
                names_with_consumers.insert(input_name);
            } catch (const std::exception& e) {
                std::cerr << "[ ERROR ] Exception happened when preparing input " << i << " for op '" << op->name()
                          << "', expected input name: '" << input_name << "', expected input port index: " << port_idx
                          << '\n';
                throw;
            }
        }
    }
    std::set<std::string> names_without_consumers;
    std::set_difference(all_names.begin(),
                        all_names.end(),
                        names_with_consumers.begin(),
                        names_with_consumers.end(),
                        std::inserter(names_without_consumers, names_without_consumers.begin()));
    m_graph_impl->reset();

    m_outputs.clear();
    for (auto& out_name : names_without_consumers) {
        std::vector<std::string> names = {out_name};
        m_outputs.push_back(std::make_shared<TensorPlaceTF>(m_input_model,
                                                            ngraph::PartialShape({}),
                                                            ngraph::element::undefined,
                                                            names));
    }
}

/*
namespace tf {
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
*/

std::vector<std::shared_ptr<OpPlaceTF>> InputModelTF::InputModelTFImpl::get_op_places() const {
    if (m_graph_changed) {
        return determine_cut_nodes();
    }
    return m_op_places;
}

static std::string extract_operation_name(const std::string& port_name) {
    constexpr char delimeter[] = ":";
    auto pos = port_name.find(delimeter);
    if (pos == std::string::npos) {
        return port_name;
    }

    FRONT_END_GENERAL_CHECK((0 < pos) && (pos + 1 < port_name.length()), "Incorrect port name specified: " + port_name);

    auto left_part = port_name.substr(0, pos);
    auto right_part = port_name.substr(pos + 1, port_name.length() - pos);

    if (left_part.find_first_not_of("0123456789") == std::string::npos) {
        return right_part;
    } else if (right_part.find_first_not_of("0123456789") == std::string::npos) {
        return left_part;
    } else {
        FRONT_END_GENERAL_CHECK(false, "Incorrect port name specified: " + port_name);
    }
}

std::vector<std::shared_ptr<OpPlaceTF>> InputModelTF::InputModelTFImpl::determine_cut_nodes() const {
    std::queue<tensorflow::detail::TFNodeDecoder*> q;
    std::unordered_set<std::string> visited;
    std::vector<std::shared_ptr<ngraph::frontend::OpPlaceTF>> new_ops;
    for (const auto& output_op : m_outputs) {
        auto op_name = output_op->get_names()[0];
        auto output_name = op_name;
        auto operation_name = extract_operation_name(output_name);
        if (!visited.count(operation_name)) {
            visited.insert(operation_name);
            auto out_op_place = m_op_places_map.at(operation_name);
            if (out_op_place) {
                // TODO: throw if nullptr
                new_ops.push_back(out_op_place);
                q.push(out_op_place->get_desc().get());
            }
        }
    }
    while (!q.empty()) {
        auto op = q.front();
        q.pop();
        auto current_op_node_name = op->name();
        for (size_t i = 0; i < op->num_inputs(); ++i) {
            std::string input_name;
            size_t port_idx;
            try {
                op->input_node(i, &input_name, &port_idx);
            } catch (const std::exception& e) {
                std::cerr << "[ ERROR ] Exception happened when preparing input " << i << " for op '" << op->name()
                          << "', expected input name: '" << input_name << "', expected input port index: " << port_idx
                          << '\n';
                throw;
            }

            // check if the current node is pruned by its input port
            bool is_input = false;
            std::string input_port_name = std::to_string(i) + ":" + current_op_node_name;
            if (m_var_places.count(input_port_name)) { // m_var_places -> m_inputs
                // override_all_inputs ({"ReluOp:0", "0:ReluOp"}) <-> override_all_inputs ({"ReluOp:0"})
                // ReluOp(0, 1, 2)
                is_input = true;
            }

            // check if the producer node is pruned by its output port
            std::string output_port_name = input_name + ":" + std::to_string(port_idx);
            if (m_var_places.count(output_port_name)) {
                is_input = true;
            }

            // check if the current node is an input
            auto op_it = m_op_places_map.find(input_name);
            if (m_var_places.count(input_name)) {
                is_input = true;
            }
            is_input = is_input || op_it->second->is_input();

            if (op_it != m_op_places_map.end() && !is_input && !op_it->second->is_input() &&
                !m_tensor_values.count(op_it->second->get_names()[0]) && !visited.count(input_name)) {
                visited.insert(input_name);
                new_ops.push_back(op_it->second);
                // TODO: check that op is frozen
                if (!op_it->second->is_input()) {
                    q.push(op_it->second->get_desc().get());
                }
            }
        }
    }
    std::reverse(new_ops.begin(), new_ops.end());
    return new_ops;
}

template <typename T>
InputModelTF::InputModelTFImpl::InputModelTFImpl(const std::basic_string<T>& path, const InputModel& input_model)
    : m_graph_def{std::make_shared<GraphDef>()},
      m_input_model(input_model) {
    std::ifstream pb_stream(path, std::ios::in | std::ifstream::binary);

    FRONT_END_GENERAL_CHECK(pb_stream && pb_stream.is_open(), "Model file does not exist");
    FRONT_END_GENERAL_CHECK(m_graph_def->ParseFromIstream(&pb_stream), "Model cannot be parsed");

    // TODO: move GraphIterator to constructor arguments
    // TODO: rename GraphIterator
    m_graph_impl = std::make_shared<::tensorflow::ngraph_bridge::GraphIteratorProto>(m_graph_def.get());
    // TODO: move NodeDecoder () to constructor arguments


    loadPlaces();
}

InputModelTF::InputModelTFImpl::InputModelTFImpl(const std::vector<std::istream*>& streams,
                                                       const InputModel& input_model)
    : m_graph_def{std::make_shared<GraphDef>()},
      m_input_model(input_model) {
    FRONT_END_GENERAL_CHECK(streams.size() == 1,
                            "One stream is needed to load a model in .pb format");
    FRONT_END_GENERAL_CHECK(m_graph_def->ParseFromIstream(streams[0]), "Model can't be parsed");

    loadPlaces();
}

std::vector<Place::Ptr> InputModelTF::InputModelTFImpl::getInputs() const {
    return m_inputs;
}

std::vector<Place::Ptr> InputModelTF::InputModelTFImpl::getOutputs() const {
    return m_outputs;
}

Place::Ptr InputModelTF::InputModelTFImpl::getPlaceByTensorName(const std::string& tensorName) const {
    if (m_var_places.count(tensorName))
        return m_var_places.at(tensorName);

    // check that operation node exists for which this place is specified
    auto op_name = extract_operation_name(tensorName);
    if (m_op_places_map.count(op_name)) {
        std::vector<std::string> names = {tensorName};
        auto m_var_place = std::make_shared<TensorPlaceTF>(m_input_model, ngraph::PartialShape(),
            ngraph::element::f32, names);
        m_var_places[tensorName] = m_var_place;
        return m_var_place;    
    }

    return nullptr;
}

namespace tf {
std::shared_ptr<TensorPlaceTF> castToTensorPlace(const Place::Ptr& place) {
    if (auto var_place = std::dynamic_pointer_cast<TensorPlaceTF>(place)) {
        return var_place;
    } else if (auto in_port_place = std::dynamic_pointer_cast<InPortPlaceTF>(place)) {
        return in_port_place->get_source_tensor_tf();
    } else if (auto out_port_place = std::dynamic_pointer_cast<OutPortPlaceTF>(place)) {
        return out_port_place->get_target_tensor_tf();
    }
    FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlaceTF.");
}

}  // namespace tf

void InputModelTF::InputModelTFImpl::overrideAllInputs(const std::vector<Place::Ptr>& inputs) {
    m_graph_changed = true;
    m_inputs.clear();
    m_var_places.clear();
    for (const auto& inp : inputs) {
        auto tensor_place = tf::castToTensorPlace(inp);
        for (const auto& name : inp->get_names()) {
            m_var_places[name] = tensor_place;
        }
        m_inputs.push_back(tensor_place);
    }
}

void InputModelTF::InputModelTFImpl::overrideAllOutputs(const std::vector<Place::Ptr>& outputs) {
    m_graph_changed = true;
    m_outputs.clear();
    for (const auto& outp : outputs) {
        m_outputs.push_back(tf::castToTensorPlace(outp));
    }
}

void InputModelTF::InputModelTFImpl::extractSubgraph(const std::vector<Place::Ptr>& inputs,
                                                         const std::vector<Place::Ptr>& outputs) {
    m_graph_changed = true;
    overrideAllInputs(inputs);
    overrideAllOutputs(outputs);
}

void InputModelTF::InputModelTFImpl::setDefaultShape(Place::Ptr place, const ngraph::Shape& shape) {
    FRONT_END_NOT_IMPLEMENTED("setDefaultShape");
}

void InputModelTF::InputModelTFImpl::setPartialShape(Place::Ptr place, const ngraph::PartialShape& p_shape) {
    tf::castToTensorPlace(place)->set_partial_shape(p_shape);
}

ngraph::PartialShape InputModelTF::InputModelTFImpl::getPartialShape(Place::Ptr place) const {
    return tf::castToTensorPlace(place)->get_partial_shape();
}

void InputModelTF::InputModelTFImpl::setElementType(Place::Ptr place, const ngraph::element::Type& type) {
    tf::castToTensorPlace(place)->set_element_type(type);
}

void InputModelTF::InputModelTFImpl::setTensorValue(Place::Ptr place, const void* value) {
    m_graph_changed = true;
    auto tensor_place = tf::castToTensorPlace(place);
    auto p_shape = tensor_place->get_partial_shape();
    auto type = tensor_place->get_element_type();
    auto constant = opset7::Constant::create(type, p_shape.to_shape(), value);
    auto name = tensor_place->get_names()[0];
    constant->set_friendly_name(name);
    m_tensor_values[name] = constant;
}

InputModelTF::InputModelTF(const std::string& path) : _impl{std::make_shared<InputModelTFImpl>(path, *this)} {}

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
InputModelTF::InputModelTF(const std::wstring& path) : _impl{std::make_shared<InputModelTFImpl>(path, *this)} {}
#endif

InputModelTF::InputModelTF(const std::vector<std::istream*>& streams)
    : _impl{std::make_shared<InputModelTFImpl>(streams, *this)} {}

std::vector<std::shared_ptr<OpPlaceTF>> InputModelTF::get_op_places() const {
    return _impl->get_op_places();
}

std::map<std::string, std::shared_ptr<TensorPlaceTF>> InputModelTF::get_var_places() const {
    return _impl->get_var_places();
}

std::map<std::string, Output<Node>> InputModelTF::get_tensor_values() const {
    return _impl->get_tensor_values();
}

std::vector<Place::Ptr> InputModelTF::get_inputs() const {
    return _impl->getInputs();
}

std::vector<Place::Ptr> InputModelTF::get_outputs() const {
    return _impl->getOutputs();
}

Place::Ptr InputModelTF::get_place_by_tensor_name(const std::string& tensorName) const {
    return _impl->getPlaceByTensorName(tensorName);
}

void InputModelTF::override_all_outputs(const std::vector<Place::Ptr>& outputs) {
    _impl->overrideAllOutputs(outputs);
}

void InputModelTF::override_all_inputs(const std::vector<Place::Ptr>& inputs) {
    _impl->overrideAllInputs(inputs);
}

void InputModelTF::extract_subgraph(const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs) {
    _impl->extractSubgraph(inputs, outputs);
}

void InputModelTF::set_partial_shape(Place::Ptr place, const ngraph::PartialShape& p_shape) {
    _impl->setPartialShape(place, p_shape);
}

ngraph::PartialShape InputModelTF::get_partial_shape(Place::Ptr place) const {
    return _impl->getPartialShape(place);
}

void InputModelTF::set_element_type(Place::Ptr place, const ngraph::element::Type& type) {
    _impl->setElementType(place, type);
}

void InputModelTF::set_tensor_value(Place::Ptr place, const void* value) {
    _impl->setTensorValue(place, value);
}

}  // namespace frontend
}  // namespace ngraph

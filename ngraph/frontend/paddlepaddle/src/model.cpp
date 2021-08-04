// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <paddlepaddle_frontend/exceptions.hpp>
#include <paddlepaddle_frontend/model.hpp>
#include <paddlepaddle_frontend/place.hpp>

#include <fstream>
#include <queue>

#include <ngraph/opsets/opset7.hpp>

#include "decoder.hpp"
#include "framework.pb.h"
#include "node_context.hpp"
#include "pdpd_utils.hpp"

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
#include <codecvt>
#include <locale>
#endif

namespace ngraph
{
    namespace frontend
    {
        using namespace paddle::framework::proto;

        class InputModelPDPD::InputModelPDPDImpl
        {
        public:
            template <typename T>
            InputModelPDPDImpl(const std::basic_string<T>& path, const InputModelPDPD& input_model);
            InputModelPDPDImpl(const std::vector<std::istream*>& streams,
                               const InputModelPDPD& input_model);
            std::vector<Place::Ptr> getInputs() const;
            std::vector<Place::Ptr> getOutputs() const;
            Place::Ptr getPlaceByTensorName(const std::string& tensorName) const;
            void overrideAllOutputs(const std::vector<Place::Ptr>& outputs);
            void overrideAllInputs(const std::vector<Place::Ptr>& inputs);
            void extractSubgraph(const std::vector<Place::Ptr>& inputs,
                                 const std::vector<Place::Ptr>& outputs);
            void setDefaultShape(Place::Ptr place, const ngraph::Shape&);
            void setPartialShape(Place::Ptr place, const ngraph::PartialShape&);
            ngraph::PartialShape getPartialShape(Place::Ptr place) const;
            void setElementType(Place::Ptr place, const ngraph::element::Type&);
            void setTensorValue(Place::Ptr place, const void* value);

            std::vector<std::shared_ptr<OpPlacePDPD>> getOpPlaces() const;
            std::vector<std::shared_ptr<TensorPlacePDPD>> getVarPlaces() const
            {
                return m_var_places;
            }
            std::map<pdpd::TensorName, Output<Node>> getTensorValues() const
            {
                return m_tensor_values;
            };
            std::map<std::string, std::shared_ptr<TensorPlacePDPD>> getVarNames() const {
                return m_var_names;
            }
            Place::Ptr addOutput(Place::Ptr place);
            void removeOutput(Place::Ptr place);
            void cutAndAddNewOutput(Place::Ptr place, const std::string& new_name_optional);
            void cutAndAddNewInput(Place::Ptr place, const std::string& new_name_optional);

            void freeNameForTensor(const std::string& name);
            void addNameForTensor(Place::Ptr tensor, const std::string& new_name);
            void setNameForTensor(Place::Ptr tensor, const std::string& new_name);

        private:
            void loadPlaces();
            template <typename T>
            void loadConsts(const std::basic_string<T>& folder_with_weights,
                            std::istream* weight_stream);
            std::vector<std::shared_ptr<OpPlacePDPD>> determine_cut_nodes() const;

            void traverse_up(const std::vector<Place::Ptr>& start_nodes, std::vector<std::shared_ptr<OpPlacePDPD>>* ordered_ops,
                             std::vector<std::shared_ptr<TensorPlacePDPD>>* ordered_tensors) const;
            void traverse_down(const std::vector<Place::Ptr>& start_nodes, std::vector<std::shared_ptr<OpPlacePDPD>>* ordered_ops,
                               std::vector<std::shared_ptr<TensorPlacePDPD>>* ordered_tensors) const;
            void clean_up_inputs_outputs();

            std::vector<std::shared_ptr<OpPlacePDPD>> m_op_places;
            std::vector<std::shared_ptr<TensorPlacePDPD>> m_var_places;
            std::shared_ptr<ProgramDesc> m_fw_ptr;
            const InputModelPDPD& m_input_model;
            std::vector<Place::Ptr> m_inputs;
            std::vector<Place::Ptr> m_outputs;
            std::map<pdpd::TensorName, Output<Node>> m_tensor_values;
            std::map<std::string, std::shared_ptr<TensorPlacePDPD>> m_var_names;

            // shows if some nodes might be deleted from graph
            bool m_graph_changed = false;
        };

        void InputModelPDPD::InputModelPDPDImpl::loadPlaces()
        {
            const int cnt_of_blocks = m_fw_ptr->blocks_size();
            const auto& blocks = m_fw_ptr->blocks();

            for (int block_idx = 0; block_idx < cnt_of_blocks; block_idx++)
            {
                const auto& block = blocks[block_idx];

                for (const auto& var : block.vars())
                {
                    auto tensor = std::make_shared<TensorPlacePDPD>(m_input_model, var);
                    m_var_places.push_back(tensor);
                    m_var_names[var.name()] = tensor;
                }

                for (const auto& op : block.ops())
                {
                    auto op_place = std::make_shared<OpPlacePDPD>(m_input_model, op);
                    m_op_places.push_back(op_place);

                    for (const auto& output : op.outputs())
                    {
                        for (const auto& var_name : output.arguments())
                        {
                            auto out_port = std::make_shared<OutPortPlacePDPD>(m_input_model);

                            // connect out_port and tensor
                            const auto& tensor = m_var_names.at(var_name);
                            tensor->add_producing_port(out_port);
                            out_port->set_target_tensor(tensor);

                            // connect out_port and op
                            op_place->add_out_port(out_port, output.parameter());
                            out_port->set_op(op_place);
                        }
                    }

                    for (const auto& input : op.inputs())
                    {
                        for (const auto& var_name : input.arguments())
                        {
                            auto in_port = std::make_shared<InPortPlacePDPD>(m_input_model);

                            // connect in_port and tensor
                            const auto& tensor = m_var_names.at(var_name);
                            tensor->add_consuming_port(in_port);
                            in_port->set_source_tensor(tensor);

                            // connect in_port and op
                            op_place->add_in_port(in_port, input.parameter());
                            in_port->set_op(op_place);
                        }
                    }

                    // Determine outputs and inputs
                    if (op.type() == "feed")
                    {
                        const auto& place = op_place->get_output_port_pdpd("Out", 0);
                        const auto& var_place = std::dynamic_pointer_cast<TensorPlacePDPD>(
                            place->get_target_tensor_pdpd());
                        const auto& tensor_desc =
                            var_place->get_desc().type().lod_tensor().tensor();
                        const auto& dims = tensor_desc.dims();

                        var_place->set_element_type(TYPE_MAP[tensor_desc.data_type()]);
                        var_place->set_partial_shape(
                            PartialShape(std::vector<Dimension>(dims.begin(), dims.end())));
                        m_inputs.push_back(var_place);
                    }
                    else if (op.type() == "fetch")
                    {
                        auto place = op_place->get_input_port_pdpd("X", 0);
                        m_outputs.push_back(place->get_source_tensor_pdpd());
                    }
                }
            }
        }

        namespace pdpd
        {
            bool read_tensor(std::istream& is, char* data, size_t len)
            {
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
            std::basic_string<T> get_const_path(const std::basic_string<T>& folder_with_weights,
                                                const std::string& name)
            {
                return folder_with_weights + pdpd::get_path_sep<T>() + name;
            }

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
            template <>
            std::basic_string<wchar_t> get_const_path(const std::basic_string<wchar_t>& folder,
                                                      const std::string& name)
            {
                std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
                std::wstring _name = converter.from_bytes(name);
                return folder + pdpd::get_path_sep<wchar_t>() + _name;
            }
#endif

            template <typename T>
            std::basic_string<T> get_model_path(const std::basic_string<T>& path,
                                                std::ifstream* weights_stream)
            {
                std::string model_file{path};
                std::string ext = ".pdmodel";
                if (pdpd::endsWith(model_file, ext))
                {
                    std::string params_ext = ".pdiparams";
                    std::string weights_file{path};
                    weights_file.replace(weights_file.size() - ext.size(), ext.size(), params_ext);
                    weights_stream->open(weights_file, std::ios::binary);
                    // Don't throw error if file isn't opened
                    // It may mean that model don't have constants
                }
                else
                {
                    model_file += pdpd::get_path_sep<T>() + "__model__";
                }
                return model_file;
            }

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
            template <>
            std::basic_string<wchar_t> get_model_path(const std::basic_string<wchar_t>& path,
                                                      std::ifstream* weights_stream)
            {
                std::wstring model_file{path};
                std::wstring ext = L".pdmodel";
                if (pdpd::endsWith(model_file, ext))
                {
                    std::wstring params_ext = L".pdiparams";
                    std::wstring weights_file{path};
                    weights_file.replace(weights_file.size() - ext.size(), ext.size(), params_ext);
                    weights_stream->open(weights_file, std::ios::binary);
                    // Don't throw error if file isn't opened
                    // It may mean that model don't have constants
                }
                else
                {
                    model_file += pdpd::get_path_sep<wchar_t>() + L"__model__";
                }
                return model_file;
            }
#endif
        } // namespace pdpd

        std::vector<std::shared_ptr<OpPlacePDPD>>
            InputModelPDPD::InputModelPDPDImpl::getOpPlaces() const
        {
            if (m_graph_changed)
            {
                return determine_cut_nodes();
            }
            return m_op_places;
        }

        std::vector<std::shared_ptr<OpPlacePDPD>>
            InputModelPDPD::InputModelPDPDImpl::determine_cut_nodes() const
        {
            std::vector<std::shared_ptr<OpPlacePDPD>> new_op_places;
            new_op_places.reserve(m_op_places.size());
            traverse_up(m_outputs, &new_op_places, nullptr);
            std::reverse(new_op_places.begin(), new_op_places.end());
            return new_op_places;
        }

        template <typename T>
        void InputModelPDPD::InputModelPDPDImpl::loadConsts(
            const std::basic_string<T>& folder_with_weights, std::istream* weight_stream)
        {
            for (const auto& item : m_var_places)
            {
                const auto& var_desc = item->get_desc();
                const auto& name = var_desc.name();
                if (pdpd::endsWith(name, std::string{"feed"}) ||
                    pdpd::endsWith(name, std::string{"fetch"}))
                    continue;
                if (!var_desc.persistable())
                    continue;

                FRONT_END_GENERAL_CHECK(var_desc.type().type() ==
                                        paddle::framework::proto::VarType::LOD_TENSOR);
                const auto& tensor = var_desc.type().lod_tensor().tensor();
                Shape shape(tensor.dims().cbegin(), tensor.dims().cend());
                const auto& type = TYPE_MAP[tensor.data_type()];
                const auto& data_length = shape_size(shape) * type.size();
                std::vector<uint8_t> tensor_data(data_length);

                bool read_succeed = false;
                if (weight_stream)
                {
                    read_succeed = pdpd::read_tensor(
                        *weight_stream, reinterpret_cast<char*>(&tensor_data[0]), data_length);
                }
                else if (!folder_with_weights.empty())
                {
                    std::ifstream is(pdpd::get_const_path(folder_with_weights, name),
                                     std::ios::in | std::ifstream::binary);
                    FRONT_END_GENERAL_CHECK(is && is.is_open(),
                                            "Cannot open file for constant value.");
                    read_succeed = pdpd::read_tensor(
                        is, reinterpret_cast<char*>(&tensor_data[0]), data_length);
                }
                else
                {
                    FRONT_END_GENERAL_CHECK(
                        false, "Either folder with weights or stream must be provided.");
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
        InputModelPDPD::InputModelPDPDImpl::InputModelPDPDImpl(const std::basic_string<T>& path,
                                                               const InputModelPDPD& input_model)
            : m_fw_ptr{std::make_shared<ProgramDesc>()}
            , m_input_model(input_model)
        {
            std::string empty_str = "";
            std::ifstream weights_stream;
            std::ifstream pb_stream(pdpd::get_model_path<T>(path, &weights_stream),
                                    std::ios::in | std::ifstream::binary);

            FRONT_END_GENERAL_CHECK(pb_stream && pb_stream.is_open(), "Model file doesn't exist");
            FRONT_END_GENERAL_CHECK(m_fw_ptr->ParseFromIstream(&pb_stream),
                                    "Model can't be parsed");

            loadPlaces();
            if (weights_stream && weights_stream.is_open())
            {
                loadConsts(std::basic_string<T>{}, &weights_stream);
            }
            else
            {
                loadConsts(path, nullptr);
            }
        }

        InputModelPDPD::InputModelPDPDImpl::InputModelPDPDImpl(
            const std::vector<std::istream*>& streams, const InputModelPDPD& input_model)
            : m_fw_ptr{std::make_shared<ProgramDesc>()}
            , m_input_model(input_model)
        {
            if (streams.size() != 1)
            {
                FRONT_END_GENERAL_CHECK(
                    streams.size() == 2,
                    "Two streams are needed to load a model: model and weights streams");
            }
            FRONT_END_GENERAL_CHECK(m_fw_ptr->ParseFromIstream(streams[0]),
                                    "Model can't be parsed");

            loadPlaces();
            if (streams.size() > 1)
                loadConsts(std::string(), streams[1]);
        }

        std::vector<Place::Ptr> InputModelPDPD::InputModelPDPDImpl::getInputs() const
        {
            return m_inputs;
        }

        std::vector<Place::Ptr> InputModelPDPD::InputModelPDPDImpl::getOutputs() const
        {
            return m_outputs;
        }

        Place::Ptr InputModelPDPD::InputModelPDPDImpl::getPlaceByTensorName(
            const std::string& tensorName) const
        {
            if (m_var_names.count(tensorName))
                return m_var_names.at(tensorName);
            return nullptr;
        }

        namespace pdpd
        {
            std::shared_ptr<TensorPlacePDPD> castToTensorPlace(const Place::Ptr& place)
            {
                if (auto var_place = std::dynamic_pointer_cast<TensorPlacePDPD>(place))
                {
                    return var_place;
                }
                else if (auto in_port_place = std::dynamic_pointer_cast<InPortPlacePDPD>(place))
                {
                    return in_port_place->get_source_tensor_pdpd();
                }
                else if (auto out_port_place = std::dynamic_pointer_cast<OutPortPlacePDPD>(place))
                {
                    return out_port_place->get_target_tensor_pdpd();
                }
                FRONT_END_GENERAL_CHECK(false, "Cannot cast this Place to TensorPlacePDPD.");
            }
        } // namespace pdpd

        void InputModelPDPD::InputModelPDPDImpl::overrideAllInputs(
            const std::vector<Place::Ptr>& inputs)
        {
            m_graph_changed = true;
            m_inputs.clear();
            for (const auto& inp : inputs)
            {
                m_inputs.push_back(pdpd::castToTensorPlace(inp));
            }
        }

        void InputModelPDPD::InputModelPDPDImpl::overrideAllOutputs(
            const std::vector<Place::Ptr>& outputs)
        {
            m_graph_changed = true;
            m_outputs.clear();
            for (const auto& outp : outputs)
            {
                m_outputs.push_back(pdpd::castToTensorPlace(outp));
            }
        }

        void InputModelPDPD::InputModelPDPDImpl::extractSubgraph(
            const std::vector<Place::Ptr>& inputs, const std::vector<Place::Ptr>& outputs)
        {
            m_graph_changed = true;
            overrideAllInputs(inputs);
            overrideAllOutputs(outputs);
        }

        void InputModelPDPD::InputModelPDPDImpl::setDefaultShape(Place::Ptr place,
                                                                 const ngraph::Shape& shape)
        {
            FRONT_END_NOT_IMPLEMENTED("setDefaultShape");
        }

        void
            InputModelPDPD::InputModelPDPDImpl::setPartialShape(Place::Ptr place,
                                                                const ngraph::PartialShape& p_shape)
        {
            pdpd::castToTensorPlace(place)->set_partial_shape(p_shape);
        }

        ngraph::PartialShape
            InputModelPDPD::InputModelPDPDImpl::getPartialShape(Place::Ptr place) const
        {
            return pdpd::castToTensorPlace(place)->get_partial_shape();
        }

        void InputModelPDPD::InputModelPDPDImpl::setElementType(Place::Ptr place,
                                                                const ngraph::element::Type& type)
        {
            pdpd::castToTensorPlace(place)->set_element_type(type);
        }

        void InputModelPDPD::InputModelPDPDImpl::setTensorValue(Place::Ptr place, const void* value)
        {
            m_graph_changed = true;
            auto tensor_place = pdpd::castToTensorPlace(place);
            auto p_shape = tensor_place->get_partial_shape();
            auto type = tensor_place->get_element_type();
            auto constant = opset7::Constant::create(type, p_shape.to_shape(), value);
            auto name = tensor_place->get_names()[0];
            constant->set_friendly_name(name);
            m_tensor_values[name] = constant;
        }

        Place::Ptr InputModelPDPD::InputModelPDPDImpl::addOutput(Place::Ptr place) {
            auto tensor_place = pdpd::castToTensorPlace(place);
            if (!tensor_place->is_output()) {
                m_outputs.push_back(place);
                return tensor_place;
            }
            FRONT_END_THROW("Place is already output.");
        }

        void InputModelPDPD::InputModelPDPDImpl::removeOutput(Place::Ptr place) {
            auto tensor_place = pdpd::castToTensorPlace(place);
            m_outputs.erase(std::remove(m_outputs.begin(), m_outputs.end(), tensor_place), m_outputs.end());
            m_graph_changed = true;
        }

        void
        InputModelPDPD::InputModelPDPDImpl::cutAndAddNewOutput(Place::Ptr place, const std::string &new_name_optional) {
            auto tensor_place = pdpd::castToTensorPlace(place);
            if (tensor_place) {
                std::vector<std::shared_ptr<TensorPlacePDPD>> new_tensors;

                // Find outputs that are connected to the cut place. These target outputs should be cut off.
                traverse_down({tensor_place}, nullptr, &new_tensors);
                std::vector<std::shared_ptr<TensorPlacePDPD>> target_out_tensor;
                for (const auto& tensor : new_tensors) {
                    if (tensor->is_output()) {
                        target_out_tensor.push_back(tensor);
                    }
                }
                m_outputs.push_back(tensor_place);

                // If some target outputs still connected with the model inputs, the selected cut place is incorrect.
                new_tensors.clear();
                traverse_down(m_inputs, nullptr, &new_tensors);
                std::vector<Place::Ptr> new_outputs;
                for (const auto& tensor : new_tensors) {
                    if (tensor->is_output()) {
                        FRONT_END_GENERAL_CHECK(std::find(target_out_tensor.begin(), target_out_tensor.end(), tensor) !=
                                                target_out_tensor.end(),
                                                "Incorrect Place for cutting.");
                        new_outputs.push_back(tensor);
                    }
                }

                std::swap(new_outputs, m_outputs);
                m_graph_changed = true;
                if (!new_name_optional.empty()) {
                    setNameForTensor(place, new_name_optional);
                }
            }
        }

        void
        InputModelPDPD::InputModelPDPDImpl::cutAndAddNewInput(Place::Ptr place, const std::string &new_name_optional) {
            auto tensor_place = pdpd::castToTensorPlace(place);
            if (tensor_place) {
                m_inputs.push_back(tensor_place);
                std::vector<std::shared_ptr<TensorPlacePDPD>> new_tensors;
                traverse_up(m_outputs, nullptr, &new_tensors);

                std::vector<Place::Ptr> new_inputs;
                for (const auto& in : m_inputs) {
                    if (std::find(new_tensors.begin(), new_tensors.end(), in) != new_tensors.end())
                        new_inputs.push_back(in);
                }
                std::swap(m_inputs, new_inputs);
                m_graph_changed = true;
                if (!new_name_optional.empty()) {
                    setNameForTensor(place, new_name_optional);
                }
            }
        }

        void InputModelPDPD::InputModelPDPDImpl::freeNameForTensor(const std::string &name) {
            if (m_var_names.count(name) != 0) {
                m_var_names[name]->remove_name(name);
                m_var_names.erase(name);
            }
        }

        void InputModelPDPD::InputModelPDPDImpl::addNameForTensor(Place::Ptr tensor, const std::string &new_name) {
            auto it = std::find(m_var_places.begin(), m_var_places.end(), tensor);
            FRONT_END_GENERAL_CHECK(it != m_var_places.end(), "Model doesn't own the provided tensor.");
            FRONT_END_GENERAL_CHECK(m_var_names.count(new_name) == 0, "The provided name is already used in the model");
            (*it)->add_name(new_name);
            m_var_names[new_name] = *it;
        }

        void InputModelPDPD::InputModelPDPDImpl::setNameForTensor(Place::Ptr tensor, const std::string &new_name) {
            auto it = std::find(m_var_places.begin(), m_var_places.end(), tensor);
            FRONT_END_GENERAL_CHECK(it != m_var_places.end(), "Model doesn't own the provided tensor.");
            for (const auto& name : (*it)->get_names()) {
                freeNameForTensor(name);
            }

            (*it)->set_name(new_name);
            m_var_names[new_name] = *it;
        }

        void InputModelPDPD::InputModelPDPDImpl::traverse_down(const std::vector<Place::Ptr> &start_nodes,
                                                               std::vector<std::shared_ptr<OpPlacePDPD>>* ordered_ops,
                                                               std::vector<std::shared_ptr<TensorPlacePDPD>>* ordered_tensors) const {
            std::queue<OpPlacePDPD*> q;
            std::unordered_set<OpPlacePDPD*> visited;

            auto check_and_update = [&](const std::shared_ptr<OpPlacePDPD>& op) -> bool {
                if (op && !visited.count(op.get())) {
                    visited.insert(op.get());
                    q.push(op.get());
                    if (ordered_ops)
                        ordered_ops->push_back(op);
                    return true;
                }
                return false;
            };

            for (const auto& node : start_nodes)
            {
                if(!check_and_update(std::dynamic_pointer_cast<OpPlacePDPD>(node)) && !node->is_output())
                {
                    if(ordered_tensors)
                        ordered_tensors->push_back(pdpd::castToTensorPlace(node));
                    for (const auto& op : node->get_consuming_operations()) {
                        auto pdpd_output_op = std::dynamic_pointer_cast<OpPlacePDPD>(op);
                        PDPD_ASSERT(pdpd_output_op != nullptr, "Invalid consuming operation");
                        check_and_update(pdpd_output_op);
                    }
                }
            }
            while (!q.empty())
            {
                auto p_op = q.front();
                q.pop();
                for (const auto& map_pair : p_op->get_output_ports())
                {
                    for (const auto& port : map_pair.second)
                    {
                        auto tensor = std::dynamic_pointer_cast<TensorPlacePDPD>(port->get_target_tensor());
                        if (tensor && !tensor->is_output() &&
                            !m_tensor_values.count(tensor->get_names()[0]))
                        {
                            if (ordered_tensors)
                                ordered_tensors->push_back(tensor);
                            for (const auto& op : tensor->get_consuming_operations()) {
                                check_and_update(std::dynamic_pointer_cast<OpPlacePDPD>(op));
                            }
                        }
                    }
                }
            }
        }

        void InputModelPDPD::InputModelPDPDImpl::traverse_up(const std::vector<Place::Ptr> &start_nodes,
                                                             std::vector<std::shared_ptr<OpPlacePDPD>>* ordered_ops,
                                                             std::vector<std::shared_ptr<TensorPlacePDPD>>* ordered_tensors) const {
            std::queue<OpPlacePDPD*> q;
            std::unordered_set<OpPlacePDPD*> visited;

            auto check_and_update = [&](const std::shared_ptr<OpPlacePDPD>& op) -> bool {
                if (op && !visited.count(op.get())) {
                    visited.insert(op.get());
                    q.push(op.get());
                    if (ordered_ops)
                        ordered_ops->push_back(op);
                    return true;
                }
                return false;
            };

            for (const auto& node : start_nodes)
            {
                if(!check_and_update(std::dynamic_pointer_cast<OpPlacePDPD>(node)) && !node->is_input())
                {
                    if (ordered_tensors)
                        ordered_tensors->push_back(pdpd::castToTensorPlace(node));
                    auto pdpd_output_op = std::dynamic_pointer_cast<OpPlacePDPD>(node->get_producing_operation());
                    FRONT_END_GENERAL_CHECK(pdpd_output_op != nullptr, "Output doesn't have producing operation");
                    check_and_update(pdpd_output_op);
                }
            }
            while (!q.empty())
            {
                auto p_op = q.front();
                q.pop();
                for (const auto& map_pair : p_op->get_input_ports())
                {
                    for (const auto& port : map_pair.second)
                    {
                        auto tensor = std::dynamic_pointer_cast<TensorPlacePDPD>(port->get_source_tensor());
                        if (tensor && !tensor->is_input() &&
                            !m_tensor_values.count(tensor->get_names()[0]))
                        {
                            if (ordered_tensors)
                                ordered_tensors->push_back(tensor);
                            check_and_update(std::dynamic_pointer_cast<OpPlacePDPD>(tensor->get_producing_operation()));
                        }
                    }
                }
            }
        }

        InputModelPDPD::InputModelPDPD(const std::string& path)
            : _impl{std::make_shared<InputModelPDPDImpl>(path, *this)}
        {
        }

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        InputModelPDPD::InputModelPDPD(const std::wstring& path)
            : _impl{std::make_shared<InputModelPDPDImpl>(path, *this)}
        {
        }
#endif

        InputModelPDPD::InputModelPDPD(const std::vector<std::istream*>& streams)
            : _impl{std::make_shared<InputModelPDPDImpl>(streams, *this)}
        {
        }

        std::vector<std::shared_ptr<OpPlacePDPD>> InputModelPDPD::getOpPlaces() const
        {
            return _impl->getOpPlaces();
        }

        std::vector<std::shared_ptr<TensorPlacePDPD>> InputModelPDPD::getVarPlaces() const
        {
            return _impl->getVarPlaces();
        }

        std::map<pdpd::TensorName, Output<Node>> InputModelPDPD::getTensorValues() const
        {
            return _impl->getTensorValues();
        }

        std::vector<Place::Ptr> InputModelPDPD::get_inputs() const { return _impl->getInputs(); }

        std::vector<Place::Ptr> InputModelPDPD::get_outputs() const { return _impl->getOutputs(); }

        Place::Ptr InputModelPDPD::get_place_by_tensor_name(const std::string& tensorName) const
        {
            return _impl->getPlaceByTensorName(tensorName);
        }

        void InputModelPDPD::override_all_outputs(const std::vector<Place::Ptr>& outputs)
        {
            _impl->overrideAllOutputs(outputs);
        }

        void InputModelPDPD::override_all_inputs(const std::vector<Place::Ptr>& inputs)
        {
            _impl->overrideAllInputs(inputs);
        }

        void InputModelPDPD::extract_subgraph(const std::vector<Place::Ptr>& inputs,
                                              const std::vector<Place::Ptr>& outputs)
        {
            _impl->extractSubgraph(inputs, outputs);
        }

        void InputModelPDPD::set_partial_shape(Place::Ptr place,
                                               const ngraph::PartialShape& p_shape)
        {
            _impl->setPartialShape(place, p_shape);
        }

        ngraph::PartialShape InputModelPDPD::get_partial_shape(Place::Ptr place) const
        {
            return _impl->getPartialShape(place);
        }

        void InputModelPDPD::set_element_type(Place::Ptr place, const ngraph::element::Type& type)
        {
            _impl->setElementType(place, type);
        }

        void InputModelPDPD::set_tensor_value(Place::Ptr place, const void* value)
        {
            _impl->setTensorValue(place, value);
        }

        void InputModelPDPD::set_name_for_tensor(Place::Ptr tensor, const std::string &new_name) {
            _impl->setNameForTensor(tensor, new_name);
        }

        void InputModelPDPD::add_name_for_tensor(Place::Ptr tensor, const std::string &new_name) {
            _impl->addNameForTensor(tensor, new_name);
        }

        void InputModelPDPD::free_name_for_tensor(const std::string &name) {
            _impl->freeNameForTensor(name);
        }

        void InputModelPDPD::cut_and_add_new_input(Place::Ptr place, const std::string &new_name_optional) {
            _impl->cutAndAddNewInput(place, new_name_optional);
        }

        void InputModelPDPD::cut_and_add_new_output(Place::Ptr place, const std::string &new_name_optional) {
            _impl->cutAndAddNewOutput(place, new_name_optional);
        }

        void InputModelPDPD::remove_output(Place::Ptr place) {
            _impl->removeOutput(place);
        }

        Place::Ptr InputModelPDPD::add_output(Place::Ptr place) {
            return _impl->addOutput(place);
        }

        std::map<std::string, std::shared_ptr<TensorPlacePDPD>> InputModelPDPD::getVarNames() const {
            return _impl->getVarNames();
        }

    } // namespace frontend
} // namespace ngraph

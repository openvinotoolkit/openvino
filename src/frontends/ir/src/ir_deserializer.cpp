// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ir_deserializer.hpp"

#include <pugixml.hpp>
#include <regex>

#include "openvino/core/except.hpp"
#include "openvino/core/meta_data.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/assign_base.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/runtime/string_aligned_buffer.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "rt_info_deserializer.hpp"
#include "transformations/rt_info/attributes.hpp"
#include "utils.hpp"

using namespace ov::util;

ov::XmlDeserializer::IoMap ov::XmlDeserializer::updated_io_map(const pugi::xml_node& node,
                                                               const pugi::xml_node& body_node) {
    if (body_node.empty()) {
        OPENVINO_THROW("Missing body part.");
    }
    // Fill map: parameter/result id to parameter/result number in Function

    auto extend_io_map = io_map;

    FOREACH_CHILD (layer, body_node.child("layers"), "layer") {
        auto type = pugixml::get_str_attr(layer, "type");

        if (type == "Parameter") {
            auto id = static_cast<size_t>(pugixml::get_uint64_attr(layer, "id"));
            extend_io_map.inputs.insert({id, -1});  // try add as unconnected
        } else if (type == "Result") {
            auto id = static_cast<size_t>(pugixml::get_uint64_attr(layer, "id"));
            extend_io_map.outputs.insert({id, -1});  // try add as unconnected
        }
    }
    return extend_io_map;
}

std::vector<std::shared_ptr<ov::op::util::SubGraphOp::InputDescription>> ov::XmlDeserializer::parse_input_description(
    const pugi::xml_node& node,
    const std::string& body_name,
    const std::string& port_map_name) {
    std::vector<std::shared_ptr<ov::op::util::SubGraphOp::InputDescription>> inputs;
    auto body_node = node.child(body_name.c_str());

    const auto up_io_map = updated_io_map(node, body_node);

    // Parse PortMap: external_port_id for inputs does not always appear in consecutive order
    std::map<uint64_t, pugi::xml_node> input_map;
    FOREACH_CHILD (input, node.child(port_map_name.c_str()), "input") {
        int64_t ext_port_id = pugixml::get_int64_attr(input, "external_port_id");
        input_map.emplace(ext_port_id, input);
    }

    for (const auto& input : input_map) {
        auto& xml_input = input.second;
        auto axis_attr = xml_input.attribute("axis");
        int64_t ti_input_index = pugixml::get_int64_attr(xml_input, "external_port_id");
        size_t body_parameter_index = static_cast<size_t>(pugixml::get_uint64_attr(xml_input, "internal_layer_id"));

        // if axis is set, then slicing is enabled. Create ov::TensorIterator::SlicedInput.
        if (!axis_attr.empty()) {
            size_t axis = static_cast<size_t>(pugixml::get_uint64_attr(xml_input, "axis"));
            int64_t start = pugixml::get_int64_attr(xml_input, "start", 0);
            int64_t stride = pugixml::get_int64_attr(xml_input, "stride", 1);
            int64_t end = pugixml::get_int64_attr(xml_input, "end", -1);
            int64_t part_size = pugixml::get_int64_attr(xml_input, "part_size", 1);

            const auto input_index = up_io_map.inputs.at(body_parameter_index);

            inputs.push_back(std::make_shared<ov::op::util::SubGraphOp::SliceInputDescription>(ti_input_index,
                                                                                               input_index,
                                                                                               start,
                                                                                               stride,
                                                                                               part_size,
                                                                                               end,
                                                                                               axis));
        } else {
            // otherwise find corresponding back edge and create ov::TensorIterator::MergedInput
            bool is_back_edge_exist = false;
            FOREACH_CHILD (xml_edge, node.child("back_edges"), "edge") {
                size_t to_layer = static_cast<size_t>(pugixml::get_uint64_attr(xml_edge, "to-layer"));

                if (to_layer == body_parameter_index) {
                    size_t from_layer = static_cast<size_t>(pugixml::get_uint64_attr(xml_edge, "from-layer"));

                    const auto input_index = up_io_map.inputs.at(body_parameter_index);
                    const auto output_index = up_io_map.outputs.at(from_layer);

                    inputs.push_back(std::make_shared<ov::op::util::SubGraphOp::MergedInputDescription>(ti_input_index,
                                                                                                        input_index,
                                                                                                        output_index));

                    is_back_edge_exist = true;
                    break;
                }
            }

            // ti_input_index = -1 means that Parameter of the body is not connected to inputs of
            // TensorIterator and is used only for internal needs.
            if (!is_back_edge_exist && ti_input_index >= 0) {
                const auto input_index = up_io_map.inputs.at(body_parameter_index);

                inputs.push_back(
                    std::make_shared<ov::op::util::SubGraphOp::InvariantInputDescription>(ti_input_index, input_index));
            }
        }
    }
    return inputs;
}

std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>>
ov::XmlDeserializer::parse_output_description(const pugi::xml_node& node,
                                              const std::string& body_name,
                                              const std::string& port_map_name) {
    std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>> outputs;
    auto body_node = node.child(body_name.c_str());
    const auto up_io_map = updated_io_map(node, body_node);

    // Parse PortMap: outputs
    std::map<int64_t, pugi::xml_node> output_map;
    FOREACH_CHILD (output, node.child(port_map_name.c_str()), "output") {
        int64_t ext_port_id = pugixml::get_int64_attr(output, "external_port_id");
        output_map.emplace(ext_port_id, output);
    }

    uint64_t output_number = 0;
    for (const auto& output : output_map) {
        auto& xml_output = output.second;
        auto axis_attr = xml_output.attribute("axis");
        size_t body_result_index = static_cast<size_t>(pugixml::get_uint64_attr(xml_output, "internal_layer_id"));

        // if external_port_id < 0 it means that this body result isn't connected to the Loop output
        // and is used only for internal needs. For TensorIterator external_port_id is always > 0.
        if (pugixml::get_int64_attr(xml_output, "external_port_id") >= 0) {
            // if axis is set, then concatenation is enabled. Create
            // ov::TensorIterator::ConcatOutput.
            if (!axis_attr.empty()) {
                int64_t axis = pugixml::get_int64_attr(xml_output, "axis");
                int64_t start = pugixml::get_int64_attr(xml_output, "start", 0);
                int64_t stride = pugixml::get_int64_attr(xml_output, "stride", 1);
                int64_t end = pugixml::get_int64_attr(xml_output, "end", -1);
                int64_t part_size = pugixml::get_int64_attr(xml_output, "part_size", 1);

                const auto output_index = up_io_map.outputs.at(body_result_index);

                outputs.push_back(
                    std::make_shared<ov::op::util::MultiSubGraphOp::ConcatOutputDescription>(output_index,
                                                                                             output_number,
                                                                                             start,
                                                                                             stride,
                                                                                             part_size,
                                                                                             end,
                                                                                             axis));
            } else {
                // otherwise create ov::TensorIterator::BodyOutput. -1 means last iteration.
                const auto output_index = up_io_map.outputs.at(body_result_index);

                outputs.push_back(std::make_shared<ov::op::util::MultiSubGraphOp::BodyOutputDescription>(output_index,
                                                                                                         output_number,
                                                                                                         -1));
            }
            output_number++;
        }
    }
    return outputs;
}

ov::op::v5::Loop::SpecialBodyPorts ov::XmlDeserializer::parse_purpose_attribute(const pugi::xml_node& node) {
    ov::op::v5::Loop::SpecialBodyPorts result = {-1, -1};
    auto body_node = node.child("body");
    const auto up_io_map = updated_io_map(node, body_node);

    OPENVINO_ASSERT(!up_io_map.inputs.empty() || !up_io_map.outputs.empty(),
                    "No parameters or results found in body Model.");

    // Parse PortMap: external_port_id for inputs/outputs does not always appear in consecutive
    // order
    std::map<uint64_t, pugi::xml_node> input_map;
    FOREACH_CHILD (input, node.child("port_map"), "input") {
        int64_t ext_port_id = pugixml::get_int64_attr(input, "external_port_id");
        input_map.emplace(ext_port_id, input);
    }
    std::map<int64_t, pugi::xml_node> output_map;
    FOREACH_CHILD (output, node.child("port_map"), "output") {
        int64_t ext_port_id = pugixml::get_int64_attr(output, "external_port_id");
        output_map.emplace(ext_port_id, output);
    }

    for (const auto& input : input_map) {
        auto& xml_input = input.second;
        auto purpose = pugixml::get_str_attr(xml_input, "purpose", "");
        size_t body_parameter_index = static_cast<size_t>(pugixml::get_uint64_attr(xml_input, "internal_layer_id"));
        if (purpose == "current_iteration") {
            result.current_iteration_input_idx = up_io_map.inputs.at(body_parameter_index);
        }
    }

    for (const auto& output : output_map) {
        auto& xml_output = output.second;
        auto purpose = pugixml::get_str_attr(xml_output, "purpose", "");
        size_t body_parameter_index = static_cast<size_t>(pugixml::get_uint64_attr(xml_output, "internal_layer_id"));
        if (purpose == "execution_condition") {
            result.body_condition_output_idx = up_io_map.outputs.at(body_parameter_index);
        }
    }

    return result;
}

void ov::XmlDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) {
    static const std::unordered_set<std::string> skip_names = {"input_descriptions",
                                                               "output_descriptions",
                                                               "special_body_ports",
                                                               "then_inputs",
                                                               "else_inputs",
                                                               "then_outputs",
                                                               "else_outputs"};
    std::string val;

    // for TensorIterator look for 'port_map' as 'data' does not exist
    if (m_node.child("port_map") || m_node.child("then_port_map") || m_node.child("else_port_map")) {
        std::string body_name = "body";
        std::string port_map_name = "port_map";
        if (name == "then_inputs" || name == "then_outputs") {
            body_name = "then_body";
            port_map_name = "then_port_map";
        } else if (name == "else_inputs" || name == "else_outputs") {
            body_name = "else_body";
            port_map_name = "else_port_map";
        }
        if (auto a = ov::as_type<
                ov::AttributeAdapter<std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::InputDescription>>>>(
                &adapter)) {
            a->set(parse_input_description(m_node, body_name, port_map_name));
        } else if (auto a = ov::as_type<ov::AttributeAdapter<
                       std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>>>>(&adapter)) {
            a->set(parse_output_description(m_node, body_name, port_map_name));
        } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::op::v5::Loop::SpecialBodyPorts>>(&adapter)) {
            a->set(parse_purpose_attribute(m_node));
        }
    }

    if (skip_names.count(name) && !getStrAttribute(m_node.child("data"), name, val))
        return;
    if (auto a = ov::as_type<ov::AttributeAdapter<ov::element::Type>>(&adapter)) {
        static_cast<ov::element::Type&>(*a) = ov::element::Type(val);
    } else if (auto a = ov::as_type<ov::AttributeAdapter<PartialShape>>(&adapter)) {
        PartialShape shape;
        if (!get_partial_shape_from_attribute(m_node.child("data"), name, shape))
            return;
        a->set(shape);
    } else if (auto a = ov::as_type<ov::AttributeAdapter<Dimension>>(&adapter)) {
        Dimension dim;
        if (!get_dimension_from_attribute(m_node.child("data"), name, dim))
            return;
        a->set(dim);
    } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::Shape>>(&adapter)) {
        std::vector<size_t> shape;
        if (!getParameters<size_t>(m_node.child("data"), name, shape))
            return;
        static_cast<ov::Shape&>(*a) = ov::Shape(shape);
    } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::Strides>>(&adapter)) {
        std::vector<size_t> shape;
        if (!getParameters<size_t>(m_node.child("data"), name, shape))
            return;
        static_cast<ov::Strides&>(*a) = ov::Strides(shape);
#if defined(__APPLE__) || defined(__EMSCRIPTEN__)
    } else if (auto a = ov::as_type<ov::AttributeAdapter<std::vector<size_t>>>(&adapter)) {
        std::vector<size_t> result;
        if (!getParameters<size_t>(m_node.child("data"), name, result))
            return;
        static_cast<std::vector<size_t>&>(*a) = result;
#else
    } else if (auto a = ov::as_type<ov::AttributeAdapter<std::vector<size_t>>>(&adapter)) {
        std::vector<size_t> result;
        if (!getParameters<size_t>(m_node.child("data"), name, result))
            return;
        a->set(result);
#endif
    } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::AxisSet>>(&adapter)) {
        std::vector<size_t> axes;
        if (!getParameters<size_t>(m_node.child("data"), name, axes))
            return;
        static_cast<ov::AxisSet&>(*a) = ov::AxisSet(axes);
    } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::op::TopKSortType>>(&adapter)) {
        if (!getStrAttribute(m_node.child("data"), name, val))
            return;
        static_cast<ov::op::TopKSortType&>(*a) = ov::as_enum<ov::op::TopKSortType>(val);
    } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::op::TopKMode>>(&adapter)) {
        if (!getStrAttribute(m_node.child("data"), name, val))
            return;
        static_cast<ov::op::TopKMode&>(*a) = ov::as_enum<ov::op::TopKMode>(val);
    } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::CoordinateDiff>>(&adapter)) {
        std::vector<size_t> shape;
        if (!getParameters<size_t>(m_node.child("data"), name, shape))
            return;
        std::vector<std::ptrdiff_t> coord_diff(shape.begin(), shape.end());
        static_cast<ov::CoordinateDiff&>(*a) = ov::CoordinateDiff(coord_diff);
    } else if (auto a = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::op::util::Variable>>>(&adapter)) {
        std::string variable_id;
        if (!getStrAttribute(m_node.child("data"), name, variable_id))
            return;
        if (!m_variables.count(variable_id)) {
            m_variables[variable_id] = std::make_shared<ov::op::util::Variable>(
                ov::op::util::VariableInfo{ov::PartialShape::dynamic(), ov::element::dynamic, variable_id});
        }
        a->set(m_variables[variable_id]);
    } else if (auto a = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>>(&adapter)) {
        std::string value;
        pugi::xml_node dn = m_node.child("data");
        auto type = pugixml::get_str_attr(m_node, "type");

        if (dn.empty())
            OPENVINO_THROW("No attrtibutes defined for ", type, " op!");

        if (getStrAttribute(dn, name, value)) {
            auto buffer = std::make_shared<ov::AlignedBuffer>(value.size());
            auto data = static_cast<char*>(buffer->get_ptr());
            value.copy(data, value.size());
            a->set(buffer);
        } else if (name == "value" && type == "Const") {
            std::vector<int64_t> shape;
            std::string el_type_str;

            size_t offset = static_cast<size_t>(pugixml::get_uint64_attr(dn, "offset"));
            size_t size = static_cast<size_t>(pugixml::get_uint64_attr(dn, "size"));
            if (!getStrAttribute(dn, "element_type", el_type_str))
                return;
            if (!getParameters<int64_t>(dn, "shape", shape))
                return;

            ov::element::Type el_type = ov::element::Type(el_type_str);

            if (!m_weights)
                OPENVINO_THROW("Empty weights data in bin file or bin file cannot be found!");
            if (m_weights->size() < offset + size)
                OPENVINO_THROW("Incorrect weights in bin file!");
            char* data = m_weights->get_ptr<char>() + offset;

            if (el_type == element::string) {
                auto buffer =
                    ov::AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>::unpack_string_tensor(data, size);
                a->set(buffer);
            } else {
                if (size < ((ov::shape_size(shape) * el_type.bitwidth() + 7) >> 3))
                    OPENVINO_THROW("Attribute and shape size are inconsistent for ", type, " op!");

                auto buffer =
                    std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(data, size, m_weights);
                a->set(buffer);
            }
        }
    } else if (auto a = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>>(&adapter)) {
        pugi::xml_node dn = m_node.child("data");
        const auto& type = pugixml::get_str_attr(m_node, "type");
        if (name == "value" && type == "Const") {
            std::vector<int64_t> shape;
            std::string el_type_str;

            size_t offset = static_cast<size_t>(pugixml::get_uint64_attr(dn, "offset"));
            size_t size = static_cast<size_t>(pugixml::get_uint64_attr(dn, "size"));
            if (!getStrAttribute(dn, "element_type", el_type_str))
                return;
            if (!getParameters<int64_t>(dn, "shape", shape))
                return;

            if (!m_weights)
                OPENVINO_THROW("Empty weights data in bin file or bin file cannot be found!");
            if (m_weights->size() < offset + size)
                OPENVINO_THROW("Incorrect weights in bin file!");
            char* data = m_weights->get_ptr<char>() + offset;
            auto buffer =
                ov::AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>::unpack_string_tensor(data, size);
            a->set(buffer);
        }
    } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::op::util::FrameworkNodeAttrs>>(&adapter)) {
        const auto& type = pugixml::get_str_attr(m_node, "type");
        const auto& version = pugixml::get_str_attr(m_node, "version");

        ov::op::util::FrameworkNodeAttrs node_attrs;
        node_attrs.set_opset_name(version);
        node_attrs.set_type_name(type);

        pugi::xml_node dn = m_node.child("data");

        if (!dn.empty()) {
            for (const auto& data_attr : dn.attributes()) {
                node_attrs[data_attr.name()] = data_attr.as_string();
            }
        }

        a->set(node_attrs);
    } else if (const auto& a = ov::as_type<ov::AttributeAdapter<ov::element::TypeVector>>(&adapter)) {
        ov::element::TypeVector types;
        if (!getParameters<ov::element::Type>(m_node.child("data"), name, types))
            return;
        a->set(types);
    } else {
        OPENVINO_THROW("Error IR reading. Attribute adapter can not be found for ", name, " parameter");
    }
}

void ov::XmlDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<std::shared_ptr<ov::Model>>& adapter) {
    std::shared_ptr<ov::Model> model;
    io_map = {};

    if (!name.compare("body") || !name.compare("then_body") || !name.compare("else_body")) {
        auto body_node = m_node.child(name.c_str());
        if (body_node.empty()) {
            OPENVINO_THROW("TensorIterator has no body.");
        }
        model = parse_function(m_node.child(name.c_str()), m_weights);
    } else if (!name.compare("net")) {
        model = parse_function(m_node, m_weights);
    } else {
        OPENVINO_THROW("Error: not recognized adapter name: ", name, ".");
    }
    adapter.set(model);
}

std::shared_ptr<ov::Model> ov::XmlDeserializer::parse_function(const pugi::xml_node& root,
                                                               const std::shared_ptr<ov::AlignedBuffer>& weights) {
    // OV_ITT_SCOPE_CHAIN(FIRST_INFERENCE, taskChain, itt::domains::V10Reader_RT, "V10Parser", "Parse");

    struct FunctionNodes {
        ov::ParameterVector parameters;
        ov::ResultVector results;
        ov::NodeVector all;
        ov::SinkVector sinks;
    };

    struct Edge {
        size_t fromLayerId, fromPortId, toPortId;
    };
    struct NodeParams {
        pugi::xml_node xml;
        GenericLayerParams params;
    };

    std::map<size_t /*layer-id*/, NodeParams> params;

    std::vector<size_t /*layer-id*/> outputs;

    std::vector<size_t> order;
    std::set<size_t> dfs_used_nodes;
    std::map<size_t /*to-layer-id*/, std::vector<Edge>> edges;
    // Read all layers and store their parameters in params map
    FOREACH_CHILD (node, root.child("layers"), "layer") {
        auto node_param = parse_generic_params(node);
        params[node_param.layerId] = {node, node_param};
        if (node_param.type == "Result" || node_param.type == "Assign") {
            outputs.push_back(node_param.layerId);
        }
        if (node_param.type == "Parameter") {
            // Save Parameters order according to order in XML.
            // To do so, handle nodes manually and ignore during DFS
            dfs_used_nodes.insert(node_param.layerId);
            order.push_back(node_param.layerId);
            edges[node_param.layerId] = {};
        }
    }

    // Read all edges and store them for further usage
    FOREACH_CHILD (_ec, root.child("edges"), "edge") {
        size_t fromLayer = static_cast<size_t>(pugixml::get_uint64_attr(_ec, "from-layer"));
        size_t fromPort = static_cast<size_t>(pugixml::get_uint64_attr(_ec, "from-port"));
        size_t toLayer = static_cast<size_t>(pugixml::get_uint64_attr(_ec, "to-layer"));
        size_t toPort = static_cast<size_t>(pugixml::get_uint64_attr(_ec, "to-port"));
        edges[toLayer].push_back({fromLayer, fromPort, toPort});
    }

    // Run DFS starting from outputs to get nodes topological order
    std::function<void(size_t)> dfs = [&edges, &order, &dfs_used_nodes, &dfs](const size_t id) {
        if (dfs_used_nodes.count(id))
            return;
        dfs_used_nodes.insert(id);
        for (auto& edge : edges[id]) {
            dfs(edge.fromLayerId);
        }
        order.push_back(id);
    };
    std::for_each(outputs.begin(), outputs.end(), dfs);

    FunctionNodes func_nodes;
    std::map<size_t, std::shared_ptr<ov::Node>> id_to_node;
    std::map<std::string, std::shared_ptr<ov::Node>> variable_id_to_read_value;

    //  Following topological order create OpenVINO operations
    for (auto& layer_id : order) {
        auto& p = params[layer_id];
        const auto& edgeIt = edges.find(layer_id);
        if (edgeIt == edges.end())
            continue;
        ov::OutputVector inputs(edgeIt->second.size());
        for (auto& e : edgeIt->second) {
            auto input_node = id_to_node[e.fromLayerId];
            if (!input_node) {
                OPENVINO_THROW("Attempt to access node ", e.fromLayerId, " that not in graph.");
            }
            auto& p_output = params[e.fromLayerId].params;
            size_t const realInputPortId = p.params.get_real_input_port_id(e.toPortId);
            if (realInputPortId >= inputs.size())
                OPENVINO_THROW(p.params.type,
                               " layer ",
                               p.params.name,
                               " with id: ",
                               p.params.layerId,
                               " is inconsistent!");
            inputs[realInputPortId] = input_node->output(p_output.get_real_output_port_id(e.fromPortId));
        }

        auto node = create_node(inputs, p.xml, weights, p.params);
        id_to_node[layer_id] = node;

        if (const auto& parameter_node = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node)) {
            io_map.inputs.insert({layer_id, func_nodes.parameters.size()});
            func_nodes.parameters.emplace_back(parameter_node);
        }

        if (const auto& result_node = std::dynamic_pointer_cast<ov::op::v0::Result>(node)) {
            io_map.outputs.insert({layer_id, func_nodes.results.size()});
            func_nodes.results.emplace_back(result_node);
        }

        if (const auto& sink = std::dynamic_pointer_cast<ov::op::Sink>(node)) {
            auto subgraph_op = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(node);
            if (subgraph_op) {
                for (const auto& body_model : subgraph_op->get_functions()) {
                    if (body_model->get_sinks().size()) {
                        func_nodes.sinks.emplace_back(sink);
                        break;
                    }
                }
            } else {
                func_nodes.sinks.emplace_back(sink);
            }
        }

        if (const auto& read_value = std::dynamic_pointer_cast<ov::op::util::ReadValueBase>(node)) {
            variable_id_to_read_value[read_value->get_variable_id()] = read_value;
        }

        func_nodes.all.emplace_back(node);
    }

    auto function = std::make_shared<ov::Model>(func_nodes.results,
                                                func_nodes.sinks,
                                                func_nodes.parameters,
                                                pugixml::get_str_attr(root, "name", ""));
    for (const auto& sink : func_nodes.sinks) {
        if (const auto& assign = std::dynamic_pointer_cast<ov::op::util::AssignBase>(sink)) {
            assign->add_control_dependency(variable_id_to_read_value.at(assign->get_variable_id()));
        }
    }

    // Read meta data from legacy representation
    if (root.child("rt_info").empty()) {
        // Legacy representation
        // meta_data - MO meta
        // quantization_parameters - NNCF quantization section
        std::unordered_set<std::string> meta_names = {"meta_data", "quantization_parameters"};
        read_legacy_meta_data(function, meta_names, root);
    } else {
        read_meta_data(function, root.child("rt_info"));
    }

    return function;
}

class MetaDataParser : public ov::MetaDataWithPugixml {
public:
    MetaDataParser(const std::string& name, const pugi::xml_node& meta, bool accessible_by_pugixml_node = true)
        : m_name(name),
          m_accessible_by_pugixml_node(accessible_by_pugixml_node) {
        m_meta.append_copy(meta);
        if (accessible_by_pugixml_node) {
            m_meta_node = m_meta.child(m_name.c_str());
        } else {
            m_meta_node = pugi::xml_node();
        }
    }

    operator const ov::AnyMap&() const override {
        parse();
        return m_parsed_map;
    }

    operator ov::AnyMap&() override {
        parse();
        return m_parsed_map;
    }

    const pugi::xml_node& get_pugi_node() const override {
        if (!m_meta_node.empty() && !m_accessible_by_pugixml_node) {
            // Meta cannot be accessed by pugixml node. Return empty node
            m_meta_node = pugi::xml_node();
        }
        return m_meta_node;
    };

private:
    bool has_attr(const pugi::xml_node& node, const std::string& name = "value") const {
        auto attr = node.attribute(name.c_str());
        return !attr.empty();
    }

    ov::Any parse_value(const pugi::xml_node& node) const {
        if (has_attr(node)) {
            return pugixml::get_str_attr(node, "value");
        } else if (std::string(node.name()) == "unset" && has_attr(node, "unset_cli_parameters")) {
            return pugixml::get_str_attr(node, "unset_cli_parameters");
        } else {
            return parse_node(node);
        }
    }

    ov::AnyMap parse_node(const pugi::xml_node& node) const {
        ov::AnyMap result;
        // Old version may produce nodes like <name value="..."/>, but it may brake xml-naming convention
        // Now it should look like <info name="..." value="..."/>.
        // Also we keep an option to read an old XMLs where it doesn't have name attribute
        const auto name_attr = node.attribute("name");
        const std::string node_name = name_attr.empty() ? node.name() : name_attr.value();
        for (const auto& data : node.children()) {
            const auto name_attr = data.attribute("name");
            const std::string data_name = name_attr.empty() ? data.name() : name_attr.value();
            // WA for legacy POT config
            if (data_name == "config" && node_name == "quantization_parameters") {
                // Read legacy pot config
                std::stringstream stream;
                data.print(stream);
                std::string str_config = stream.str();
                str_config = std::regex_replace(str_config, std::regex("<config>"), "");
                str_config = std::regex_replace(str_config, std::regex("</config>"), "");
                str_config = std::regex_replace(str_config, std::regex("\n"), "");
                str_config = std::regex_replace(str_config, std::regex("( +)"), " ");
                result[data_name] = str_config;
            } else {
                result[data_name] = parse_value(data);
            }
        }
        return result;
    }

    void parse() const {
        std::call_once(m_oc, [this]() {
            m_accessible_by_pugixml_node = false;
            const pugi::xml_node& node = m_meta.child(m_name.c_str());
            m_parsed_map = parse_node(node);
        });
    }

    pugi::xml_document m_meta;
    const std::string m_name;
    mutable ov::AnyMap m_parsed_map;
    mutable std::once_flag m_oc;
    mutable std::atomic_bool m_accessible_by_pugixml_node;
    mutable pugi::xml_node m_meta_node;
};

void ov::XmlDeserializer::read_meta_data(const std::shared_ptr<ov::Model>& model, const pugi::xml_node& meta_section) {
    if (meta_section.empty())
        return;
    auto& rt_info = model->get_rt_info();
    for (const auto& data : meta_section.children()) {
        if (data.empty())
            continue;
        // Old version may produce nodes like <name value="..."/>, but it may brake xml-naming convention
        // Now it should look like <info name="..." value="..."/>.
        // Also we keep an option to read an old XMLs where it doesn't have name attribute
        const auto name_attr = data.attribute("name");
        const auto node_name = name_attr.empty() ? data.name() : name_attr.value();
        if (!data.attribute("value").empty()) {
            rt_info[node_name] = pugixml::get_str_attr(data, "value");
        } else {
            // Use meta data for set of parameters
            std::shared_ptr<ov::Meta> meta = std::make_shared<MetaDataParser>(data.name(), data);
            rt_info[node_name] = meta;
        }
    }
}

void ov::XmlDeserializer::read_legacy_meta_data(const std::shared_ptr<ov::Model>& model,
                                                const std::unordered_set<std::string>& names,
                                                const pugi::xml_node& root_section) {
    const auto& read_meta =
        [](const std::shared_ptr<ov::Model>& model, const std::string& name, const pugi::xml_node& meta_section) {
            auto& rt_info = model->get_rt_info();
            if (name == "meta_data") {
                for (const auto& data : meta_section.children()) {
                    const std::string& section_name = data.name();
                    // Rename cli_parameters to conversion_parameters
                    if (section_name == "cli_parameters") {
                        std::shared_ptr<ov::Meta> meta = std::make_shared<MetaDataParser>("cli_parameters", data);
                        rt_info["conversion_parameters"] = meta;
                    } else if (!data.attribute("value").empty()) {
                        rt_info[data.name()] = pugixml::get_str_attr(data, "value");
                    } else {
                        OPENVINO_THROW("Unsupported legacy argument: ", data.name());
                    }
                }
            } else if (name == "quantization_parameters") {
                // Rename quantization_parameters to optimization
                // Legacy implementation. Have to be parsed inside MetaDataParser. Do not allow to serialize it as raw
                // pugi::xml_node.
                std::shared_ptr<ov::Meta> meta =
                    std::make_shared<MetaDataParser>("quantization_parameters", meta_section, false);
                rt_info["optimization"] = meta;
            }
        };
    for (const auto& it : names)
        read_meta(model, it, root_section.child(it.c_str()));
}

ov::GenericLayerParams ov::XmlDeserializer::parse_generic_params(const pugi::xml_node& node) {
    const auto parsePort = [](const pugi::xml_node& parentNode,
                              const GenericLayerParams& params,
                              bool input) -> GenericLayerParams::LayerPortData {
        GenericLayerParams::LayerPortData port;

        port.portId = static_cast<size_t>(pugixml::get_uint64_attr(parentNode, "id"));

        FOREACH_CHILD (node, parentNode, "dim") {
            int64_t dim = 0;
            const pugi::char_t* dimVal = node.child_value();
            std::stringstream ss(dimVal);
            if (!(ss >> dim) || dim < -1) {
                OPENVINO_THROW("dimension (",
                               dimVal,
                               ") in node ",
                               node.name(),
                               " must be greater or equal to -1: at offset ",
                               node.offset_debug());
            }
            port.dims.emplace_back(dim);
        }

        ov::element::Type type(ov::element::Type_t::undefined);
        // Input port hasn't precision
        if (!input) {
            const std::string& preStr = pugixml::get_str_attr(parentNode, "precision");
            type = ov::element::Type(preStr);
        }
        port.precision = type;
        std::vector<std::string> names;
        if (getParameters<std::string>(parentNode, "names", names)) {
            for (size_t i = 0; i < names.size(); i++) {
                std::string name = names[i];
                // Restore original name if it contains delimiter
                // getParameters(...) returns the vector of names which were split by delimiter ','
                // but some names can contain ',' as a part of name, in this case we use '\' to
                // escape delimiter the cycle below is needed in order to find names which contained
                // delimiter and restore the original name
                while (i < names.size() && names[i].at(names[i].length() - 1) == '\\') {
                    name.replace(names[i].length() - 1, 1, ",");
                    name += names[++i];
                }
                port.names.emplace(name);
            }
        }
        return port;
    };
    GenericLayerParams params;

    params.layerId = static_cast<size_t>(pugixml::get_uint64_attr(node, "id"));
    params.version = pugixml::get_str_attr(node, "version");

    params.type = pugixml::get_str_attr(node, "type");

    params.name = pugixml::get_str_attr(node, "name");

    auto outNode = node.child("output");
    if (!outNode.empty()) {
        FOREACH_CHILD (_cn, outNode, "port") {
            params.outputPorts.emplace_back(parsePort(_cn, params, false));
        }
    }
    auto inpNode = node.child("input");
    if (!inpNode.empty()) {
        FOREACH_CHILD (_cn, inpNode, "port") {
            params.inputPorts.emplace_back(parsePort(_cn, params, true));
        }
    }
    return params;
}

// Symmetric function to translate type name.
// See translate_type_name in src/core/src/pass/serialize.cpp.
static const std::string& translate_type_name(const std::string& name) {
    static const std::unordered_map<std::string, std::string> translate_type_name_translator = {{"Const", "Constant"},
                                                                                                {"PReLU", "PRelu"},
                                                                                                {"ReLU", "Relu"},
                                                                                                {"SoftMax", "Softmax"}};
    auto found = translate_type_name_translator.find(name);
    if (found != end(translate_type_name_translator)) {
        return found->second;
    }
    return name;
}

std::shared_ptr<ov::Node> ov::XmlDeserializer::create_node(const std::vector<ov::Output<ov::Node>>& inputs,
                                                           const pugi::xml_node& node,
                                                           const std::shared_ptr<ov::AlignedBuffer>& weights,
                                                           const GenericLayerParams& params) {
    // Check that inputs are correctly defined
    for (size_t i = 0; i < inputs.size(); i++) {
        if (!inputs[i].get_node())
            OPENVINO_THROW(params.type,
                           " layer ",
                           params.name,
                           " with id: ",
                           params.layerId,
                           " has incorrect input with index ",
                           i,
                           "!");
        if (ov::element::Type_t::undefined == inputs[i].get_element_type())
            OPENVINO_THROW(params.type,
                           " layer ",
                           params.name,
                           " with id: ",
                           params.layerId,
                           " has undefined element type for input with index ",
                           i,
                           "!");
    }

    const std::string& type_name = translate_type_name(params.type);

    std::shared_ptr<ov::Node> ovNode;
    ov::DiscreteTypeInfo type(type_name.c_str(), params.version.c_str());
    auto extensionIt = m_extensions.find(type);

    if (extensionIt != m_extensions.end()) {
        XmlDeserializer visitor(node, weights, m_opsets, m_extensions, m_variables, m_version);
        ovNode = (*extensionIt->second).create(inputs, visitor).at(0).get_node_shared_ptr();
    }

    // Find registered opset
    auto opsetIt = m_opsets.find(params.version);

    // Try to create operation from loaded opsets
    static const std::unordered_set<std::string> experimental_ops_added_to_opset = {
        "ExperimentalDetectronDetectionOutput",
        "ExperimentalDetectronGenerateProposalsSingleImage",
        "ExperimentalDetectronPriorGridGenerator",
        "ExperimentalDetectronROIFeatureExtractor",
        "ExperimentalDetectronTopKROIs",
        "GRUCell",
        "RNNCell",
        "Proposal"};

    if (experimental_ops_added_to_opset.count(type_name) &&
        (params.version == "experimental" || params.version == "extension")) {
        opsetIt = m_opsets.find("opset6");
    }

    if (!ovNode && opsetIt != m_opsets.end()) {
        if (params.version == "opset1") {
            // MVN, ROIPooling and ReorgYolo were missing in opset1
            if (type_name == "MVN" || type_name == "ROIPooling" || type_name == "ReorgYolo") {
                opsetIt = m_opsets.find("opset2");
                if (opsetIt == m_opsets.end()) {
                    OPENVINO_THROW("Cannot create ",
                                   params.type,
                                   " layer ",
                                   params.name,
                                   " id:",
                                   params.layerId,
                                   " from unsupported opset: ",
                                   params.version);
                }
            }
        }

        auto const& opset = opsetIt->second;

        ovNode = std::shared_ptr<ov::Node>(opset.create_insensitive(type_name));
        if (!ovNode) {
            OPENVINO_THROW("Opset ", params.version, " doesn't contain the operation with type: ", type_name);
        }
        // Share Weights form constant blob
        if (auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(ovNode)) {
            constant->alloc_buffer_on_visit_attributes(false);
        }
        ovNode->set_arguments(inputs);
        XmlDeserializer visitor(node, weights, m_opsets, m_extensions, m_variables, m_version);

        if (ovNode->visit_attributes(visitor)) {
            ovNode->constructor_validate_and_infer_types();
        }

        // To be sure that all default values will be initialized:
        ovNode = ovNode->clone_with_new_inputs(ovNode->input_values());
    }
    if (!ovNode && m_extensions.count(ov::op::util::FrameworkNode::get_type_info_static())) {
        ovNode = std::make_shared<ov::op::util::FrameworkNode>(inputs);
        XmlDeserializer visitor(node, weights, m_opsets, m_extensions, m_variables, m_version);
        ovNode->visit_attributes(visitor);

        size_t index{0};
        for (const auto& output_params : params.outputPorts) {
            ovNode->set_output_type(index, output_params.precision, ov::PartialShape(output_params.dims));
            ++index;
        }
    }

    if (!ovNode) {
        OPENVINO_THROW("Cannot create ",
                       params.type,
                       " layer ",
                       params.name,
                       " id:",
                       params.layerId,
                       " from unsupported opset: ",
                       params.version);
    }

    // Save run time info
    auto& rtInfo = ovNode->get_rt_info();
    pugi::xml_node dn = node.child("data");
    if (dn) {
        const auto pr_data = dn.attribute("PrimitivesPriority");
        if (pr_data) {
            rtInfo.emplace(ov::PrimitivesPriority::get_type_info_static(), ov::PrimitivesPriority{pr_data.value()});
        }
        const auto aw_data = dn.attribute("alt_width");
        if (aw_data) {
            rtInfo["alt_width"] = aw_data.value();
        }
        const auto size = dn.attribute("size");
        const auto offset = dn.attribute("offset");
        if (size && offset) {
            rtInfo[ov::WeightlessCacheAttribute::get_type_info_static()] =
                ov::WeightlessCacheAttribute(static_cast<size_t>(pugixml::get_uint64_attr(dn, "size")),
                                             static_cast<size_t>(pugixml::get_uint64_attr(dn, "offset")));
        }
    }

    ovNode->set_friendly_name(params.name);
    for (size_t i = 0; i < params.outputPorts.size() && i < ovNode->get_output_size(); ++i) {
        if (!params.outputPorts[i].names.empty())
            ovNode->get_output_tensor(i).set_names(params.outputPorts[i].names);
    }

    ov::pass::Attributes attrs_factory;
    auto set_runtime_info = [&attrs_factory](RTMap& rt_info, const pugi::xml_node& rt_attrs) {
        if (!rt_attrs)
            return;
        for (const auto& item : rt_attrs) {
            std::string attribute_name, attribute_version;
            // For view:
            // <attribute name="old_api_map_order" version="0" value="0,3,1,2"/>
            if (!getStrAttribute(item, "name", attribute_name) || !getStrAttribute(item, "version", attribute_version))
                continue;

            const auto& type_info = ov::DiscreteTypeInfo(attribute_name.c_str(), attribute_version.c_str());
            auto attr = attrs_factory.create_by_type_info(type_info);
            if (!attr.empty()) {
                if (attr.is<ov::RuntimeAttribute>()) {
                    RTInfoDeserializer attribute_visitor(item);
                    if (attr.as<ov::RuntimeAttribute>().visit_attributes(attribute_visitor)) {
                        auto res = rt_info.emplace(type_info, attr);
                        if (!res.second) {
                            OPENVINO_THROW("multiple rt_info attributes are detected: ", attribute_name);
                        }
                    } else {
                        OPENVINO_THROW("VisitAttributes is not supported for: ", item.name(), " attribute");
                    }
                } else {
                    OPENVINO_THROW("Attribute: ", item.name(), " is not recognized as runtime attribute");
                }
            } else {
                // As runtime attributes are optional, so we skip attribute if it is unknown to avoid exception
                // when loading new IR with new attribute in old OV version.
            }
        }
    };

    // read runtime info only for IR v11+
    if (m_version > 10) {
        // set node runtime info attributes
        set_runtime_info(ovNode->get_rt_info(), node.child("rt_info"));

        // set output ports runtime info attributes
        auto out_node = node.child("output");
        if (!out_node.empty()) {
            size_t index{0};
            FOREACH_CHILD (rt_node, out_node, "port") {
                set_runtime_info(ovNode->output(index).get_rt_info(), rt_node.child("rt_info"));
                ++index;
            }
        }

        // set input ports runtime info attributes
        auto in_node = node.child("input");
        if (!in_node.empty()) {
            size_t index{0};
            FOREACH_CHILD (rt_node, in_node, "port") {
                set_runtime_info(ovNode->input(index).get_rt_info(), rt_node.child("rt_info"));
                ++index;
            }
        }
    }

    return ovNode;
}

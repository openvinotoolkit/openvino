// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/xml_deserializer.hpp"

#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/xml_parse_utils.hpp"

namespace intel_npu {

NPUXmlDeserializer::NPUXmlDeserializer(
    const pugi::xml_node& node,
    const std::shared_ptr<ov::AlignedBuffer>& weights,
    const std::unordered_map<std::string, ov::OpSet>& opsets,
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>>& variables,
    size_t version)
    : ov::util::XmlDeserializer(node,
                                weights,
                                opsets,
                                std::unordered_map<ov::DiscreteTypeInfo, ov::BaseOpExtension::Ptr>(),
                                variables,
                                version) {}

std::vector<std::shared_ptr<ov::op::util::SubGraphOp::InputDescription>> NPUXmlDeserializer::parse_input_description(
    const pugi::xml_node& node,
    const std::string& body_name,
    const std::string& port_map_name) {
    std::vector<std::shared_ptr<ov::op::util::SubGraphOp::InputDescription>> inputs;
    auto body_node = node.child(body_name.c_str());

    const auto up_io_map = updated_io_map(node, body_node);

    // Parse PortMap: external_port_id for inputs does not always appear in consecutive order
    std::map<uint64_t, pugi::xml_node> input_map;
    FOREACH_CHILD (input, node.child(port_map_name.c_str()), "input") {
        int64_t ext_port_id = ov::util::pugixml::get_int64_attr(input, "external_port_id");
        input_map.emplace(ext_port_id, input);
    }

    for (const auto& input : input_map) {
        auto& xml_input = input.second;
        auto axis_attr = xml_input.attribute("axis");
        int64_t ti_input_index = ov::util::pugixml::get_int64_attr(xml_input, "external_port_id");
        size_t body_parameter_index =
            static_cast<size_t>(ov::util::pugixml::get_uint64_attr(xml_input, "internal_layer_id"));

        // if axis is set, then slicing is enabled. Create ov::TensorIterator::SlicedInput.
        if (!axis_attr.empty()) {
            size_t axis = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(xml_input, "axis"));
            int64_t start = ov::util::pugixml::get_int64_attr(xml_input, "start", 0);
            int64_t stride = ov::util::pugixml::get_int64_attr(xml_input, "stride", 1);
            int64_t end = ov::util::pugixml::get_int64_attr(xml_input, "end", -1);
            int64_t part_size = ov::util::pugixml::get_int64_attr(xml_input, "part_size", 1);

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
                size_t to_layer = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(xml_edge, "to-layer"));

                if (to_layer == body_parameter_index) {
                    size_t from_layer = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(xml_edge, "from-layer"));

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
NPUXmlDeserializer::parse_output_description(const pugi::xml_node& node,
                                             const std::string& body_name,
                                             const std::string& port_map_name) {
    std::vector<std::shared_ptr<ov::op::util::MultiSubGraphOp::OutputDescription>> outputs;
    auto body_node = node.child(body_name.c_str());
    const auto up_io_map = updated_io_map(node, body_node);

    // Parse PortMap: outputs
    std::map<int64_t, pugi::xml_node> output_map;
    FOREACH_CHILD (output, node.child(port_map_name.c_str()), "output") {
        int64_t ext_port_id = ov::util::pugixml::get_int64_attr(output, "external_port_id");
        output_map.emplace(ext_port_id, output);
    }

    uint64_t output_number = 0;
    for (const auto& output : output_map) {
        auto& xml_output = output.second;
        auto axis_attr = xml_output.attribute("axis");
        size_t body_result_index =
            static_cast<size_t>(ov::util::pugixml::get_uint64_attr(xml_output, "internal_layer_id"));

        // if external_port_id < 0 it means that this body result isn't connected to the Loop output
        // and is used only for internal needs. For TensorIterator external_port_id is always > 0.
        if (ov::util::pugixml::get_int64_attr(xml_output, "external_port_id") >= 0) {
            // if axis is set, then concatenation is enabled. Create
            // ov::TensorIterator::ConcatOutput.
            if (!axis_attr.empty()) {
                int64_t axis = ov::util::pugixml::get_int64_attr(xml_output, "axis");
                int64_t start = ov::util::pugixml::get_int64_attr(xml_output, "start", 0);
                int64_t stride = ov::util::pugixml::get_int64_attr(xml_output, "stride", 1);
                int64_t end = ov::util::pugixml::get_int64_attr(xml_output, "end", -1);
                int64_t part_size = ov::util::pugixml::get_int64_attr(xml_output, "part_size", 1);

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

ov::op::v5::Loop::SpecialBodyPorts NPUXmlDeserializer::parse_purpose_attribute(const pugi::xml_node& node) {
    ov::op::v5::Loop::SpecialBodyPorts result = {-1, -1};
    auto body_node = node.child("body");
    const auto up_io_map = updated_io_map(node, body_node);

    OPENVINO_ASSERT(!up_io_map.inputs.empty() || !up_io_map.outputs.empty(),
                    "No parameters or results found in body Model.");

    // Parse PortMap: external_port_id for inputs/outputs does not always appear in consecutive
    // order
    std::map<uint64_t, pugi::xml_node> input_map;
    FOREACH_CHILD (input, node.child("port_map"), "input") {
        int64_t ext_port_id = ov::util::pugixml::get_int64_attr(input, "external_port_id");
        input_map.emplace(ext_port_id, input);
    }
    std::map<int64_t, pugi::xml_node> output_map;
    FOREACH_CHILD (output, node.child("port_map"), "output") {
        int64_t ext_port_id = ov::util::pugixml::get_int64_attr(output, "external_port_id");
        output_map.emplace(ext_port_id, output);
    }

    for (const auto& input : input_map) {
        auto& xml_input = input.second;
        auto purpose = ov::util::pugixml::get_str_attr(xml_input, "purpose", "");
        size_t body_parameter_index =
            static_cast<size_t>(ov::util::pugixml::get_uint64_attr(xml_input, "internal_layer_id"));
        if (purpose == "current_iteration") {
            result.current_iteration_input_idx = up_io_map.inputs.at(body_parameter_index);
        }
    }

    for (const auto& output : output_map) {
        auto& xml_output = output.second;
        auto purpose = ov::util::pugixml::get_str_attr(xml_output, "purpose", "");
        size_t body_parameter_index =
            static_cast<size_t>(ov::util::pugixml::get_uint64_attr(xml_output, "internal_layer_id"));
        if (purpose == "execution_condition") {
            result.body_condition_output_idx = up_io_map.outputs.at(body_parameter_index);
        }
    }

    return result;
}

NPUXmlDeserializer::IoMap NPUXmlDeserializer::updated_io_map(const pugi::xml_node& node,
                                                             const pugi::xml_node& body_node) {
    if (body_node.empty()) {
        OPENVINO_THROW("Missing body part.");
    }
    // Fill map: parameter/result id to parameter/result number in Function

    auto extend_io_map = io_map;

    FOREACH_CHILD (layer, body_node.child("layers"), "layer") {
        auto type = ov::util::pugixml::get_str_attr(layer, "type");

        if (type == "Parameter") {
            auto id = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(layer, "id"));
            extend_io_map.inputs.insert({id, -1});  // try add as unconnected
        } else if (type == "Result") {
            auto id = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(layer, "id"));
            extend_io_map.outputs.insert({id, -1});  // try add as unconnected
        }
    }
    return extend_io_map;
}

void NPUXmlDeserializer::on_adapter(const std::string& name, ov::ValueAccessor<void>& adapter) {
    static const std::unordered_set<std::string> skip_names = {"input_descriptions",
                                                               "output_descriptions",
                                                               "special_body_ports",
                                                               "then_inputs",
                                                               "else_inputs",
                                                               "then_outputs",
                                                               "else_outputs"};
    std::string val;
    const pugi::xml_node& m_node = get_node();
    const std::shared_ptr<ov::AlignedBuffer>& m_weights = get_weights();

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
    } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::PartialShape>>(&adapter)) {
        ov::PartialShape shape;
        if (!get_partial_shape_from_attribute(m_node.child("data"), name, shape))
            return;
        a->set(shape);
    } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::Dimension>>(&adapter)) {
        ov::Dimension dim;
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
                ov::op::util::VariableInfo{ov::ov::PartialShape::dynamic(), ov::element::dynamic, variable_id});
        }
        a->set(m_variables[variable_id]);
    } else if (auto a = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::AlignedBuffer>>>(&adapter)) {
        std::string value;
        pugi::xml_node dn = m_node.child("data");
        auto type = ov::util::pugixml::get_str_attr(m_node, "type");

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
            if (dn.attribute("offset") && dn.attribute("size")) {
                size_t offset = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(dn, "offset"));
                size_t size = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(dn, "size"));
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
                        ov::AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>::unpack_string_tensor(data,
                                                                                                             size);
                    a->set(buffer);
                } else {
                    if (size < ((ov::shape_size(shape) * el_type.bitwidth() + 7) >> 3))
                        OPENVINO_THROW("Attribute and shape size are inconsistent for ", type, " op!");

                    auto buffer =
                        std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(data, size, m_weights);
                    a->set(buffer);
                }
            } else {
                uintptr_t ptr = static_cast<uintptr_t>(dn.attribute("ptr").as_ullong());
                int type = dn.attribute("type").as_int();
                size_t size = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(dn, "size"));

                if (!getStrAttribute(dn, "element_type", el_type_str))
                    return;
                if (!getParameters<int64_t>(dn, "shape", shape))
                    return;
                ov::element::Type el_type = ov::element::Type(el_type_str);

                std::shared_ptr<ov::AlignedBuffer> buffer;
                if (type == 0) {
                    buffer = std::make_shared<ov::SharedStringAlignedBuffer>((char*)ptr, size);
                    // buffer = *(reinterpret_cast<std::shared_ptr<ov::StringAlignedBuffer>*>(ptr));

                    if (!buffer)
                        OPENVINO_THROW("Incorrect weights in map!");
                    // std::cout << "using StringAlignedBuffer" << std::endl;
                } else if (type == 1) {
                    // buffer = *(reinterpret_cast<std::shared_ptr<ov::SharedStringAlignedBuffer>*>(ptr));
                    buffer = std::make_shared<ov::SharedStringAlignedBuffer>((char*)ptr, size);
                    if (!buffer)
                        OPENVINO_THROW("Incorrect weights in map!");
                    // std::cout << "using SharedStringAlignedBuffer" << std::endl;
                } else if (type == 2) {
                    // buffer = *(reinterpret_cast<std::shared_ptr<ov::AlignedBuffer>*>(ptr));
                    std::shared_ptr<ov::AlignedBuffer> placeholder;
                    buffer = std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>((char*)ptr,
                                                                                                    size,
                                                                                                    placeholder);
                    if (!buffer)
                        OPENVINO_THROW("Incorrect weights in map!");
                    // std::cout << "using AlignedBuffer" << std::endl;
                } else {
                    OPENVINO_THROW("Unknown weights type in map!");
                }
                // char* data = m_weights->get_ptr<char>() + offset;

                if (el_type == element::string) {
                    // auto buffer =
                    //     ov::AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>::unpack_string_tensor(data,
                    //     size);
                    // std::cout << "Warning: For StringAlignedBuffer!" << std::endl;
                    a->set(buffer);
                } else {
                    if (buffer->size() < ((ov::shape_size(shape) * el_type.bitwidth() + 7) >> 3))
                        OPENVINO_THROW("Attribute and shape size are inconsistent for ", type, " op!");

                    // auto buffer =
                    //     std::make_shared<ov::SharedBuffer<std::shared_ptr<ov::AlignedBuffer>>>(data, size,
                    //     m_weights);
                    // std::cout << "Warning: for AlignedBuffer" << std::endl;
                    a->set(buffer);
                }
            }
        }
    } else if (auto a = ov::as_type<ov::AttributeAdapter<std::shared_ptr<ov::StringAlignedBuffer>>>(&adapter)) {
        pugi::xml_node dn = m_node.child("data");
        const auto& type = ov::util::pugixml::get_str_attr(m_node, "type");
        if (name == "value" && type == "Const") {
            std::vector<int64_t> shape;
            std::string el_type_str;
            if (dn.attribute("offset") && dn.attribute("size")) {
                size_t offset = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(dn, "offset"));
                size_t size = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(dn, "size"));
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
            } else {
                uintptr_t ptr = static_cast<uintptr_t>(dn.attribute("ptr").as_ullong());
                int type = dn.attribute("type").as_int();
                size_t size = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(dn, "size"));
                // std::cout << "key : " << key << ", size: " << size << ", type: " << type << std::endl;
                if (!getStrAttribute(dn, "element_type", el_type_str))
                    return;
                if (!getParameters<int64_t>(dn, "shape", shape))
                    return;

                std::shared_ptr<ov::StringAlignedBuffer> buffer;
                if (type == 0) {
                    // buffer = *(reinterpret_cast<std::shared_ptr<ov::StringAlignedBuffer>*>(ptr));
                    buffer = std::make_shared<ov::SharedStringAlignedBuffer>((char*)ptr, size);
                    if (!buffer)
                        OPENVINO_THROW("Incorrect weights in map!");
                    // std::cout << "using StringAlignedBuffer" << std::endl;
                } else if (type == 1) {
                    // buffer = *(reinterpret_cast<std::shared_ptr<ov::SharedStringAlignedBuffer>*>(ptr));
                    buffer = std::make_shared<ov::SharedStringAlignedBuffer>((char*)ptr, size);
                    if (!buffer)
                        OPENVINO_THROW("Incorrect weights in map!");
                    // std::cout << "using SharedStringAlignedBuffer" << std::endl;
                } else {
                    OPENVINO_THROW("Unknown weights type in map!");
                }

                // std::cout << "Warning: For StringAlignedBuffer!" << std::endl;
                a->set(buffer);
            }
        }
    } else if (auto a = ov::as_type<ov::AttributeAdapter<ov::op::util::FrameworkNodeAttrs>>(&adapter)) {
        const auto& type = ov::util::pugixml::get_str_attr(m_node, "type");
        const auto& version = ov::util::pugixml::get_str_attr(m_node, "version");

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

std::shared_ptr<ov::Model> deserialize_ir_model(std::string_view serialized_graph, const ov::Tensor& weights) {
    ov::util::StringViewStreamBuf mb(serialized_graph);
    std::istream modelStream(&mb);
    pugi::xml_document m_xml_doc;
    pugi::xml_parse_result res = m_xml_doc.load(modelStream);
    OPENVINO_ASSERT(res.status == pugi::status_ok, res.description(), " at offset ", res.offset);
    pugi::xml_node root = m_xml_doc.document_element();

    std::shared_ptr<ov::AlignedBuffer> weights_buffer =
        std::make_shared<ov::SharedBuffer<ov::Tensor>>(reinterpret_cast<char*>(const_cast<void*>(weights.data())),
                                                       weights.get_byte_size(),
                                                       weights);

    std::unordered_map<std::string, ov::OpSet> opsets;
    for (const auto& it : ov::get_available_opsets()) {
        opsets[it.first] = it.second();
    }
    std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> variables;
    size_t version = static_cast<size_t>(ov::util::pugixml::get_uint64_attr(root, "version", 0));

    NPUXmlDeserializer visitor(root, weights_buffer, opsets, variables, version);
    std::shared_ptr<ov::Model> model;
    visitor.on_attribute("net", model);
    model->get_rt_info()["version"] = int64_t(version);

    return model;
}

}  // namespace intel_npu

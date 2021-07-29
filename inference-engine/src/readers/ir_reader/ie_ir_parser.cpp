// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_ir_parser.hpp"
#include "ie_ir_itt.hpp"

#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/op/util/sub_graph_base.hpp>
#include <ngraph/op/util/variable.hpp>
#include <ngraph/ops.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/variant.hpp>
#include <ngraph_ops/framework_node.hpp>
#include <set>
#include <sstream>
#include <string>
#include <typeinfo>
#include <unordered_set>
#include <vector>

#include <cpp/ie_cnn_network.h>
#include <ie_ngraph_utils.hpp>
#include "blob_factory.hpp"
#include "caseless.hpp"
#include "precision_utils.h"

using namespace XMLParseUtils;
namespace InferenceEngine {

IRParser::IRParser(size_t version) : IRParser(version, {}) {}

IRParser::IRParser(size_t version, const std::vector<InferenceEngine::IExtensionPtr>& exts) {
    switch (version) {
    case 10:
        parser = std::make_shared<V10Parser>(exts);
        break;
    default:
        IE_THROW() << "Unsupported IR version: " << version;
    }
}

CNNNetwork IRParser::parse(
    const pugi::xml_node& root, const Blob::CPtr& weights) {
    return parser->parse(root, weights);
}

namespace {

bool getStrAttribute(const pugi::xml_node& node, const std::string& name, std::string& value) {
    if (!node) return false;

    auto attr = node.attribute(name.c_str());
    if (attr.empty()) return false;
    value = std::string(attr.value());
    return true;
}

template <class T>
bool getParameters(const pugi::xml_node& node, const std::string& name, std::vector<T>& value) {
    std::string param;
    if (!getStrAttribute(node, name, param)) return false;
    std::stringstream ss(param);
    std::string field;
    while (getline(ss, field, ',')) {
        if (field.empty())
            IE_THROW() << "Cannot get vector of parameters! \"" << param
                               << "\" is incorrect";
        std::stringstream fs(field);
        T val;
        fs >> val;
        value.emplace_back(val);
    }
    return true;
}

template <class T>
T stringToType(const std::string& valStr) {
    T ret{0};
    std::istringstream ss(valStr);
    if (!ss.eof()) {
        ss >> ret;
    }
    return ret;
}

class XmlDeserializer : public ngraph::AttributeVisitor {
public:
    /// TODO: move whole class to src file
    explicit XmlDeserializer(
        const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const std::unordered_map<std::string, ngraph::OpSet>& opsets,
        std::unordered_map<std::string, std::shared_ptr<ngraph::Variable>>& variables)
        : node(node), weights(weights), opsets(opsets), variables(variables) {}

    void on_adapter(const std::string& name, ngraph::ValueAccessor<std::string>& value) override {
        std::string val;
        if (!getStrAttribute(node.child("data"), name, val)) return;
        value.set(val);
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<bool>& value) override {
        std::string val;
        if (!getStrAttribute(node.child("data"), name, val)) return;
        std::transform(val.begin(), val.end(), val.begin(), [](char ch) {
            return std::tolower(static_cast<unsigned char>(ch));
        });
        std::set<std::string> true_names{"true", "1"};
        std::set<std::string> false_names{"false", "0"};

        bool is_true = true_names.find(val) != true_names.end();
        bool is_false = false_names.find(val) != false_names.end();

        if (!is_true && !is_false) return;
        value.set(is_true);
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) override;

    void on_adapter(const std::string& name, ngraph::ValueAccessor<double>& adapter) override {
        std::string val;
        if (!getStrAttribute(node.child("data"), name, val)) return;
        adapter.set(stringToType<double>(val));
    }
    void on_adapter(const std::string& name, ngraph::ValueAccessor<int64_t>& adapter) override {
        std::string val;
        if (!getStrAttribute(node.child("data"), name, val)) return;
        adapter.set(stringToType<int64_t>(val));
    }

    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>& adapter) override;

    void on_adapter(
        const std::string& name, ngraph::ValueAccessor<std::vector<int32_t>>& adapter) override {
        std::vector<int32_t> value;
        if (!getParameters<int32_t>(node.child("data"), name, value)) return;
        adapter.set(value);
    }

    void on_adapter(
        const std::string& name, ngraph::ValueAccessor<std::vector<int64_t>>& adapter) override {
        std::vector<int64_t> value;
        if (!getParameters<int64_t>(node.child("data"), name, value)) return;
        adapter.set(value);
    }

    void on_adapter(
        const std::string& name, ngraph::ValueAccessor<std::vector<float>>& adapter) override {
        std::vector<float> value;
        if (!getParameters<float>(node.child("data"), name, value)) return;
        adapter.set(value);
    }

    void on_adapter(
        const std::string& name,
        ngraph::ValueAccessor<std::vector<std::string>>& adapter) override {
        std::vector<std::string> value;
        if (!getParameters<std::string>(node.child("data"), name, value)) return;
        adapter.set(value);
    }

    void use_framework_node(bool flag) { m_use_framework_node = flag; }

private:
    struct IoMap {
        using NodeIdToIoIndex =
            std::unordered_map<size_t /*xml node id*/, uint64_t /*body io index*/>;
        NodeIdToIoIndex inputs;
        NodeIdToIoIndex outputs;
    };

    /// \brief Traverses port_map in order to create vector of InputDescription shared_ptrs.
    /// Shall be used only for ops which have port_map attribute.
    /// \param node xml op representation
    std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>>
    parseInputDescription(const pugi::xml_node& node);
    /// \brief Traverses port_map in order to create vector of OutputDescription shared_ptrs.
    /// Shall be used only for ops which have port_map attribute.
    /// \param node xml op representation
    std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>>
    parseOutputDescription(const pugi::xml_node& node);

    // TODO consider to call only once per layer/TI-Loop node
    IoMap updated_io_map(const pugi::xml_node& node);

    /// \brief Traverses xml node representation in order to create nGraph function for it.
    /// \param node xml node representation
    /// \param weights weights attached to current node
    /// \return shared pointer to function representing input node
    std::shared_ptr<ngraph::Function> parse_function(
        const pugi::xml_node& root, const Blob::CPtr& weights);
    /// \brief Traverses xml node representation in order to get the purpose attribute of
    /// inputs/outputs in the body of Loop op. \param node xml node representation \return struct
    /// with value of purpuse attribute
    ngraph::op::v5::Loop::SpecialBodyPorts parsePurposeAttribute(const pugi::xml_node& node);

    V10Parser::GenericLayerParams parseGenericParams(const pugi::xml_node& node);

    std::shared_ptr<ngraph::Node> createNode(
        const ngraph::OutputVector& inputs,
        const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const V10Parser::GenericLayerParams& params);

    // -- DATA --
    const pugi::xml_node node;
    const Blob::CPtr& weights;
    const std::unordered_map<std::string, ngraph::OpSet>& opsets;
    std::unordered_map<std::string, std::shared_ptr<ngraph::Variable>>& variables;

    ///
    /// store information about parameters/results order during function creation
    /// it will be used during Inputs/Outputs Description creation in SubGraph processing
    ///
    IoMap io_map;

    bool m_use_framework_node{false};
};

XmlDeserializer::IoMap XmlDeserializer::updated_io_map(const pugi::xml_node& node) {
    auto body_node = node.child("body");

    if (body_node.empty()) {
        IE_THROW() << "Missing body part.";
    }
    // Fill map: parameter/result id to parameter/result number in Function

    auto extend_io_map = io_map;

    FOREACH_CHILD(layer, body_node.child("layers"), "layer") {
        auto type = XMLParseUtils::GetStrAttr(layer, "type");

        if (type == "Parameter") {
            auto id = XMLParseUtils::GetUIntAttr(layer, "id");
            extend_io_map.inputs.insert({id, -1});  // try add as unconnected
        } else if (type == "Result") {
            auto id = XMLParseUtils::GetUIntAttr(layer, "id");
            extend_io_map.outputs.insert({id, -1});  // try add as unconnected
        }
    }
    return extend_io_map;
}

std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>>
XmlDeserializer::parseInputDescription(const pugi::xml_node& node) {
    std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>> inputs;
    const auto up_io_map = updated_io_map(node);

    // Parse PortMap: external_port_id for inputs does not always appear in consecutive order
    std::map<uint64_t, pugi::xml_node> input_map;
    FOREACH_CHILD(input, node.child("port_map"), "input") {
        int64_t ext_port_id = GetInt64Attr(input, "external_port_id");
        input_map.emplace(ext_port_id, input);
    }

    for (const auto& input : input_map) {
        auto& xml_input = input.second;
        auto axis_attr = xml_input.attribute("axis");
        int64_t ti_input_index = XMLParseUtils::GetInt64Attr(xml_input, "external_port_id");
        size_t body_parameter_index = XMLParseUtils::GetUIntAttr(xml_input, "internal_layer_id");

        // if axis is set, then slicing is enabled. Create ngraph::TensorIterator::SlicedInput.
        if (!axis_attr.empty()) {
            size_t axis = XMLParseUtils::GetUIntAttr(xml_input, "axis");
            int64_t start = XMLParseUtils::GetInt64Attr(xml_input, "start", 0);
            int64_t stride = XMLParseUtils::GetInt64Attr(xml_input, "stride", 1);
            int64_t end = XMLParseUtils::GetInt64Attr(xml_input, "end", -1);
            int64_t part_size = XMLParseUtils::GetInt64Attr(xml_input, "part_size", 1);

            const auto input_index = up_io_map.inputs.at(body_parameter_index);

            inputs.push_back(std::make_shared<ngraph::op::util::SubGraphOp::SliceInputDescription>(
                ti_input_index, input_index, start, stride, part_size, end, axis));
        } else {
            // otherwise find corresponding back edge and create ngraph::TensorIterator::MergedInput
            bool is_back_edge_exist = false;
            FOREACH_CHILD(xml_edge, node.child("back_edges"), "edge") {
                size_t to_layer = XMLParseUtils::GetUIntAttr(xml_edge, "to-layer");

                if (to_layer == body_parameter_index) {
                    size_t from_layer = XMLParseUtils::GetUIntAttr(xml_edge, "from-layer");

                    const auto input_index = up_io_map.inputs.at(body_parameter_index);
                    const auto output_index = up_io_map.outputs.at(from_layer);

                    inputs.push_back(
                        std::make_shared<ngraph::op::util::SubGraphOp::MergedInputDescription>(
                            ti_input_index, input_index, output_index));

                    is_back_edge_exist = true;
                    break;
                }
            }

            // ti_input_index = -1 means that Parameter of the body is not connected to inputs of
            // TensorIterator and is used only for internal needs.
            if (!is_back_edge_exist && ti_input_index >= 0) {
                const auto input_index = up_io_map.inputs.at(body_parameter_index);

                inputs.push_back(
                    std::make_shared<ngraph::op::util::SubGraphOp::InvariantInputDescription>(
                        ti_input_index, input_index));
            }
        }
    }
    return inputs;
}

std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>>
XmlDeserializer::parseOutputDescription(const pugi::xml_node& node) {
    std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>> outputs;
    const auto up_io_map = updated_io_map(node);

    // Parse PortMap: outputs
    std::map<int64_t, pugi::xml_node> output_map;
    FOREACH_CHILD(output, node.child("port_map"), "output") {
        int64_t ext_port_id = GetInt64Attr(output, "external_port_id");
        output_map.emplace(ext_port_id, output);
    }

    uint64_t output_number = 0;
    for (const auto& output : output_map) {
        auto& xml_output = output.second;
        auto axis_attr = xml_output.attribute("axis");
        size_t body_result_index = XMLParseUtils::GetUIntAttr(xml_output, "internal_layer_id");

        // if external_port_id < 0 it means that this body result isn't connected to the Loop output
        // and is used only for internal needs. For TensorIterator external_port_id is always > 0.
        if (XMLParseUtils::GetInt64Attr(xml_output, "external_port_id") >= 0) {
            // if axis is set, then concatenation is enabled. Create
            // ngraph::TensorIterator::ConcatOutput.
            if (!axis_attr.empty()) {
                int64_t axis = XMLParseUtils::GetInt64Attr(xml_output, "axis");
                int64_t start = XMLParseUtils::GetInt64Attr(xml_output, "start", 0);
                int64_t stride = XMLParseUtils::GetInt64Attr(xml_output, "stride", 1);
                int64_t end = XMLParseUtils::GetInt64Attr(xml_output, "end", -1);
                int64_t part_size = XMLParseUtils::GetInt64Attr(xml_output, "part_size", 1);

                const auto output_index = up_io_map.outputs.at(body_result_index);

                outputs.push_back(
                    std::make_shared<ngraph::op::util::SubGraphOp::ConcatOutputDescription>(
                        output_index, output_number, start, stride, part_size, end, axis));
            } else {
                // otherwise create ngraph::TensorIterator::BodyOutput. -1 means last iteration.
                const auto output_index = up_io_map.outputs.at(body_result_index);

                outputs.push_back(
                    std::make_shared<ngraph::op::util::SubGraphOp::BodyOutputDescription>(
                        output_index, output_number, -1));
            }
            output_number++;
        }
    }
    return outputs;
}

ngraph::op::v5::Loop::SpecialBodyPorts XmlDeserializer::parsePurposeAttribute(
    const pugi::xml_node& node) {
    ngraph::op::v5::Loop::SpecialBodyPorts result = {-1, -1};
    const auto up_io_map = updated_io_map(node);

    NGRAPH_CHECK(
        !up_io_map.inputs.empty() || !up_io_map.outputs.empty(),
        "No parameters or results found in body Function.");

    // Parse PortMap: external_port_id for inputs/outputs does not always appear in consecutive
    // order
    std::map<uint64_t, pugi::xml_node> input_map;
    FOREACH_CHILD(input, node.child("port_map"), "input") {
        int64_t ext_port_id = GetInt64Attr(input, "external_port_id");
        input_map.emplace(ext_port_id, input);
    }
    std::map<int64_t, pugi::xml_node> output_map;
    FOREACH_CHILD(output, node.child("port_map"), "output") {
        int64_t ext_port_id = GetInt64Attr(output, "external_port_id");
        output_map.emplace(ext_port_id, output);
    }

    for (const auto& input : input_map) {
        auto& xml_input = input.second;
        auto purpose = XMLParseUtils::GetStrAttr(xml_input, "purpose", "");
        size_t body_parameter_index = XMLParseUtils::GetUIntAttr(xml_input, "internal_layer_id");
        if (purpose == "current_iteration") {
            result.current_iteration_input_idx = up_io_map.inputs.at(body_parameter_index);
        }
    }

    for (const auto& output : output_map) {
        auto& xml_output = output.second;
        auto purpose = XMLParseUtils::GetStrAttr(xml_output, "purpose", "");
        size_t body_parameter_index = XMLParseUtils::GetUIntAttr(xml_output, "internal_layer_id");
        if (purpose == "execution_condition") {
            result.body_condition_output_idx = up_io_map.outputs.at(body_parameter_index);
        }
    }

    return result;
}

void XmlDeserializer::on_adapter(const std::string& name, ngraph::ValueAccessor<void>& adapter) {
    static const std::unordered_set<std::string> skip_names = {
        "input_descriptions", "output_descriptions", "special_body_ports"};
    std::string val;

    // for TensorIterator look for 'port_map' as 'data' does not exist
    if (node.child("port_map")) {
        if (auto a = ngraph::as_type<ngraph::AttributeAdapter<
                std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::InputDescription>>>>(
                &adapter)) {
            a->set(parseInputDescription(node));
        } else if (
            auto a = ngraph::as_type<ngraph::AttributeAdapter<
                std::vector<std::shared_ptr<ngraph::op::util::SubGraphOp::OutputDescription>>>>(
                &adapter)) {
            a->set(parseOutputDescription(node));
        } else if (
            auto a =
                ngraph::as_type<ngraph::AttributeAdapter<ngraph::op::v5::Loop::SpecialBodyPorts>>(
                    &adapter)) {
            a->set(parsePurposeAttribute(node));
        }
    }

    if (skip_names.count(name) && !getStrAttribute(node.child("data"), name, val)) return;
    if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::element::Type>>(&adapter)) {
        static_cast<ngraph::element::Type&>(*a) = details::convertPrecision(val);
    } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::PartialShape>>(&adapter)) {
        std::vector<int64_t> shape;
        std::vector<ngraph::Dimension> dims;
        if (!getParameters<int64_t>(node.child("data"), name, shape)) return;
        for (const auto& dim : shape) dims.emplace_back(dim);
        static_cast<ngraph::PartialShape&>(*a) = ngraph::PartialShape(dims);
    } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::Shape>>(&adapter)) {
        std::vector<size_t> shape;
        if (!getParameters<size_t>(node.child("data"), name, shape)) return;
        static_cast<ngraph::Shape&>(*a) = ngraph::Shape(shape);
    } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::Strides>>(&adapter)) {
        std::vector<size_t> shape;
        if (!getParameters<size_t>(node.child("data"), name, shape)) return;
        static_cast<ngraph::Strides&>(*a) = ngraph::Strides(shape);
#ifdef __APPLE__
    } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<std::vector<size_t>>>(&adapter)) {
        std::vector<size_t> result;
        if (!getParameters<size_t>(node.child("data"), name, result)) return;
        static_cast<std::vector<size_t>&>(*a) = result;
#else
    } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<std::vector<size_t>>>(&adapter)) {
        std::vector<size_t> result;
        if (!getParameters<size_t>(node.child("data"), name, result)) return;
        a->set(result);
#endif
    } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::AxisSet>>(&adapter)) {
        std::vector<size_t> axes;
        if (!getParameters<size_t>(node.child("data"), name, axes)) return;
        static_cast<ngraph::AxisSet&>(*a) = ngraph::AxisSet(axes);
    } else if (
        auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::op::TopKSortType>>(&adapter)) {
        if (!getStrAttribute(node.child("data"), name, val)) return;
        static_cast<ngraph::op::TopKSortType&>(*a) = ngraph::as_enum<ngraph::op::TopKSortType>(val);
    } else if (auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::op::TopKMode>>(&adapter)) {
        if (!getStrAttribute(node.child("data"), name, val)) return;
        static_cast<ngraph::op::TopKMode&>(*a) = ngraph::as_enum<ngraph::op::TopKMode>(val);
    } else if (
        auto a = ngraph::as_type<ngraph::AttributeAdapter<ngraph::CoordinateDiff>>(&adapter)) {
        std::vector<size_t> shape;
        if (!getParameters<size_t>(node.child("data"), name, shape)) return;
        std::vector<std::ptrdiff_t> coord_diff(shape.begin(), shape.end());
        static_cast<ngraph::CoordinateDiff&>(*a) = ngraph::CoordinateDiff(coord_diff);
    } else if (
        auto a = ngraph::as_type<ngraph::AttributeAdapter<std::shared_ptr<ngraph::Variable>>>(
            &adapter)) {
        std::string variable_id;
        if (!getStrAttribute(node.child("data"), name, variable_id)) return;
        if (!variables.count(variable_id)) {
            variables[variable_id] = std::make_shared<ngraph::Variable>(ngraph::VariableInfo{
                ngraph::PartialShape::dynamic(), ngraph::element::dynamic, variable_id});
        }
        a->set(variables[variable_id]);
    } else if (
        auto a = ngraph::as_type<
            ngraph::AttributeAdapter<std::shared_ptr<ngraph::runtime::AlignedBuffer>>>(&adapter)) {
        std::string value;
        pugi::xml_node dn = node.child("data");
        auto type = XMLParseUtils::GetStrAttr(node, "type");

        if (dn.empty()) IE_THROW() << "No attrtibutes defined for " << type << " op!";

        if (getStrAttribute(dn, name, value)) {
            auto buffer = std::make_shared<ngraph::runtime::AlignedBuffer>(value.size());
            auto data = static_cast<char*>(buffer->get_ptr());
            value.copy(data, value.size());
            a->set(buffer);
        } else if (name == "value" && type == "Const") {
            std::vector<int64_t> shape;
            std::string el_type_str;

            size_t offset = XMLParseUtils::GetUInt64Attr(dn, "offset");
            size_t size = XMLParseUtils::GetUInt64Attr(dn, "size");
            if (!getStrAttribute(dn, "element_type", el_type_str)) return;
            if (!getParameters<int64_t>(dn, "shape", shape)) return;

            ngraph::element::Type el_type = details::convertPrecision(el_type_str);

            size_t length = weights->byteSize();
            if (!length)
                IE_THROW() << "Empty weights data in bin file or bin file cannot be found!";
            if (length < offset + size) IE_THROW() << "Incorrect weights in bin file!";
            if (size < std::ceil(ngraph::shape_size(shape) * el_type.bitwidth() / 8.f))
                IE_THROW() << "Attribute and shape size are inconsistent for " << type
                                   << " op!";

            char* data = weights->cbuffer().as<char*>() + offset;

            using SharedBuffer = ngraph::runtime::SharedBuffer<const Blob::CPtr>;
            auto buffer = std::make_shared<SharedBuffer>(data, size, weights);
            a->set(buffer);
        }
    } else if (auto a = ngraph::as_type<
                        ngraph::AttributeAdapter<ngraph::op::FrameworkNodeAttrs>>(&adapter)) {
        const auto & type = XMLParseUtils::GetStrAttr(node, "type");
        const auto & version = XMLParseUtils::GetStrAttr(node, "version");

        ngraph::op::FrameworkNodeAttrs node_attrs;
        node_attrs.set_opset_name(version);
        node_attrs.set_type_name(type);

        pugi::xml_node dn = node.child("data");

        if (!dn.empty()) {
            for (const auto & data_attr : dn.attributes()) {
                node_attrs[data_attr.name()] = data_attr.as_string();
            }
        }

        a->set(node_attrs);
    } else {
        IE_THROW() << "Error IR reading. Attribute adapter can not be found for " << name
                           << " parameter";
    }
}

void XmlDeserializer::on_adapter(
    const std::string& name, ngraph::ValueAccessor<std::shared_ptr<ngraph::Function>>& adapter) {
    std::shared_ptr<ngraph::Function> ngraph_function;
    if (!name.compare("body")) {
        auto body_node = node.child(name.c_str());
        if (body_node.empty()) {
            IE_THROW() << "TensorIterator has no body.";
        }
        ngraph_function = parse_function(node.child(name.c_str()), weights);
    } else if (!name.compare("net")) {
        ngraph_function = parse_function(node, weights);
    } else {
        IE_THROW() << "Error: not recognized adapter name: " << name << ".";
    }
    adapter.set(ngraph_function);
}

std::shared_ptr<ngraph::Function> XmlDeserializer::parse_function(
    const pugi::xml_node& root, const Blob::CPtr& weights) {
    OV_ITT_SCOPE_CHAIN(FIRST_INFERENCE, taskChain, itt::domains::V10Reader_RT, "V10Parser", "Parse");

    struct FunctionNodes {
        ngraph::ParameterVector parameters;
        ngraph::ResultVector results;
        ngraph::NodeVector all;
        ngraph::SinkVector sinks;
    };

    struct edge {
        size_t fromLayerId, fromPortId, toPortId;
    };
    struct node_params {
        pugi::xml_node xml;
        V10Parser::GenericLayerParams params;
    };

    std::map<size_t/*layer-id*/, node_params> params;

    std::vector<size_t/*layer-id*/> outputs;
    std::unordered_set<std::string> opName;

    // Read all layers and store their parameters in params map
    FOREACH_CHILD(node, root.child("layers"), "layer") {
        auto node_param = parseGenericParams(node);
        if (opName.find(node_param.name) != opName.end() && node_param.type != "Result")
            IE_THROW() << "Invalid IR! " << node_param.name << " name is not unique!";
        opName.insert(node_param.name);
        params[node_param.layerId] = {node, node_param};
        if (node_param.type == "Result" || node_param.type == "Assign") {
            outputs.push_back(node_param.layerId);
        }
    }

    std::map<size_t/*to-layer-id*/, std::vector<edge>> edges;
    std::map<size_t, std::shared_ptr<ngraph::Node>> id_to_node;

    // Read all edges and store them for further usage
    FOREACH_CHILD(_ec, root.child("edges"), "edge") {
        size_t fromLayer = GetUIntAttr(_ec, "from-layer");
        size_t fromPort = GetUIntAttr(_ec, "from-port");
        size_t toLayer = GetUIntAttr(_ec, "to-layer");
        size_t toPort = GetUIntAttr(_ec, "to-port");
        edges[toLayer].push_back({fromLayer, fromPort, toPort});
    }

    // Run DFS starting from outputs to get nodes topological order
    std::set<size_t> used;
    std::vector<size_t> order;
    std::function<void(size_t)> dfs = [&edges, &order, &used, &dfs](const size_t id) {
        if (used.count(id)) return;
        used.insert(id);
        for (auto& edge : edges[id]) {
            dfs(edge.fromLayerId);
        }
        order.push_back(id);
    };
    std::for_each(outputs.begin(), outputs.end(), dfs);

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "ConstructNgraphNodes");

    FunctionNodes func_nodes;

    std::map<std::string, std::shared_ptr<ngraph::Node>> variable_id_to_read_value;

    //  Following topological order create nGraph operations
    for (auto& layer_id : order) {
        auto& p = params[layer_id];
        ngraph::OutputVector inputs(edges[layer_id].size());
        for (auto& e : edges[layer_id]) {
            auto input_node = id_to_node[e.fromLayerId];
            if (!input_node) {
                IE_THROW() << "Attempt to access node " << e.fromLayerId
                                   << " that not in graph.";
            }
            auto& p_output = params[e.fromLayerId].params;
            size_t const realInputPortId = p.params.getRealInputPortId(e.toPortId);
            if (realInputPortId >= inputs.size())
                IE_THROW() << p.params.type << " layer " << p.params.name
                                   << " with id: " << p.params.layerId << " is inconsistent!";
            inputs[realInputPortId] =
                input_node->output(p_output.getRealOutputPortId(e.fromPortId));
        }

        auto node = createNode(inputs, p.xml, weights, p.params);
        id_to_node[layer_id] = node;

        // Check that output shape after nGraph node validation the same as in IR
        // because IR always right!
        // Temporary disabled!
        //        for (size_t i = 0; i < p.params.outputPorts.size(); ++i) {
        //            if (p.params.outputPorts[i].dims != node->output(i).get_shape()) {
        //                IE_THROW() << "Shape after nGraph infer " <<
        //                details::dumpVec(node->output(i).get_shape())
        //                                   << " differ from IR shapes: " <<
        //                                   details::dumpVec(p.params.outputPorts[i].dims);
        //            }
        //        }

        if (const auto& parameter_node = std::dynamic_pointer_cast<ngraph::op::Parameter>(node)) {
            io_map.inputs.insert({layer_id, func_nodes.parameters.size()});
            func_nodes.parameters.emplace_back(parameter_node);
        }

        if (const auto& result_node = std::dynamic_pointer_cast<ngraph::op::Result>(node)) {
            io_map.outputs.insert({layer_id, func_nodes.results.size()});
            func_nodes.results.emplace_back(result_node);
        }

        if (const auto& sink = std::dynamic_pointer_cast<ngraph::op::Sink>(node)) {
            func_nodes.sinks.emplace_back(sink);
        }

        if (const auto& read_value = std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(node)) {
            variable_id_to_read_value[read_value->get_variable_id()] = read_value;
        }

        func_nodes.all.emplace_back(node);
    }

    OV_ITT_SCOPE_NEXT(FIRST_INFERENCE, taskChain, "ConstructNgraphFunction");

    auto function = std::make_shared<ngraph::Function>(
        func_nodes.results, func_nodes.sinks, func_nodes.parameters, GetStrAttr(root, "name", ""));
    for (const auto& sink : func_nodes.sinks) {
        if (const auto& assign = std::dynamic_pointer_cast<ngraph::op::AssignBase>(sink)) {
            assign->add_control_dependency(variable_id_to_read_value.at(assign->get_variable_id()));
        }
    }

    return function;
}

V10Parser::V10Parser::GenericLayerParams XmlDeserializer::parseGenericParams(
    const pugi::xml_node& node) {
    const auto parsePort = [this](
                               const pugi::xml_node& parentNode,
                               const V10Parser::GenericLayerParams& params,
                               bool input) -> V10Parser::GenericLayerParams::LayerPortData {
        V10Parser::GenericLayerParams::LayerPortData port;

        port.portId = GetIntAttr(parentNode, "id");

        FOREACH_CHILD(node, parentNode, "dim") {
            int64_t dim = 0;
            const pugi::char_t* dimVal = node.child_value();
            std::stringstream ss(dimVal);
            if (!(ss >> dim) || dim < 0) {
                IE_THROW() << "dimension (" << dimVal << ") in node " << node.name()
                                   << " must be a non-negative integer: at offset "
                                   << node.offset_debug();
            }
            port.dims.push_back(dim);
        }

        ngraph::element::Type type(ngraph::element::Type_t::undefined);
        // Input port hasn't precision
        if (!input) {
            const std::string& preStr = GetStrAttr(parentNode, "precision");
            type = InferenceEngine::details::convertPrecision(preStr);
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
    V10Parser::GenericLayerParams params;

    params.layerId = GetIntAttr(node, "id");
    params.version = GetStrAttr(node, "version");

    params.type = XMLParseUtils::GetStrAttr(node, "type");

    params.name = GetStrAttr(node, "name");

    auto outNode = node.child("output");
    if (!outNode.empty()) {
        FOREACH_CHILD(_cn, outNode, "port") {
            params.outputPorts.emplace_back(parsePort(_cn, params, false));
        }
    }
    auto inpNode = node.child("input");
    if (!inpNode.empty()) {
        FOREACH_CHILD(_cn, inpNode, "port") {
            params.inputPorts.emplace_back(parsePort(_cn, params, true));
        }
    }
    return params;
}

std::shared_ptr<ngraph::Node> XmlDeserializer::createNode(
    const std::vector<ngraph::Output<ngraph::Node>>& inputs,
    const pugi::xml_node& node,
    const Blob::CPtr& weights,
    const V10Parser::GenericLayerParams& params) {
    // Check that inputs are correctly defined
    for (size_t i = 0; i < inputs.size(); i++) {
        if (!inputs[i].get_node())
            IE_THROW() << params.type << " layer " << params.name
                               << " with id: " << params.layerId
                               << " has incorrect input with index " << i << "!";
        if (ngraph::element::Type_t::undefined == inputs[i].get_element_type())
            IE_THROW() << params.type << " layer " << params.name
                               << " with id: " << params.layerId
                               << " has undefined element type for input with index " << i << "!";
    }

    std::shared_ptr<ngraph::Node> ngraphNode;

    // Find registered opset
    auto opsetIt = opsets.find(params.version);

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

    if (experimental_ops_added_to_opset.count(params.type) &&
        (params.version == "experimental" || params.version == "extension")) {
        opsetIt = opsets.find("opset6");
    }

    if (!ngraphNode && opsetIt != opsets.end()) {
        auto const& type = params.type == "Const" ? "Constant" : params.type;

        if (params.version == "opset1") {
            // MVN, ROIPooling and ReorgYolo were missing in opset1
            if (type == "MVN" || type == "ROIPooling" || type == "ReorgYolo") {
                opsetIt = opsets.find("opset2");
                if (opsetIt == opsets.end()) {
                    IE_THROW() << "Cannot create " << params.type << " layer "
                                       << params.name << " id:" << params.layerId
                                       << " from unsupported opset: " << params.version;
                }
            }
        }

        auto const& opset = opsetIt->second;

        ngraphNode = std::shared_ptr<ngraph::Node>(opset.create_insensitive(type));
        if (!ngraphNode) {
            IE_THROW() << "Opset " << params.version
                               << " doesn't contain the operation with type: " << type;
        }
        // Share Weights form constant blob
        if (auto constant = std::dynamic_pointer_cast<ngraph::opset6::Constant>(ngraphNode)) {
            constant->alloc_buffer_on_visit_attributes(false);
        }
        ngraphNode->set_arguments(inputs);
        XmlDeserializer visitor(node, weights, opsets, variables);

        if (ngraphNode->visit_attributes(visitor)) {
            ngraphNode->constructor_validate_and_infer_types();
        }

        // To be sure that all default values will be initialized:
        ngraphNode = ngraphNode->clone_with_new_inputs(ngraphNode->input_values());
    }

    if (!ngraphNode && m_use_framework_node) {
        ngraphNode = std::make_shared<ngraph::op::FrameworkNode>(inputs);
        XmlDeserializer visitor(node, weights, opsets, variables);
        ngraphNode->visit_attributes(visitor);

        size_t index{0};
        for (const auto & output_params : params.outputPorts) {
            ngraphNode->set_output_type(index, output_params.precision, ngraph::Shape(output_params.dims));
            ++index;
        }
    }

    if (!ngraphNode) {
        IE_THROW() << "Cannot create " << params.type << " layer " << params.name
                           << " id:" << params.layerId
                           << " from unsupported opset: " << params.version;
    }

    // Save run time info
    auto& rtInfo = ngraphNode->get_rt_info();
    pugi::xml_node dn = node.child("data");
    if (dn) {
        const auto pr_data = dn.attribute("PrimitivesPriority");
        if (pr_data) {
            rtInfo["PrimitivesPriority"] =
                std::make_shared<::ngraph::VariantWrapper<std::string>>(pr_data.value());
        }
        const auto aw_data = dn.attribute("alt_width");
        if (aw_data) {
            rtInfo["alt_width"] =
                std::make_shared<::ngraph::VariantWrapper<std::string>>(aw_data.value());
        }
    }

    ngraphNode->set_friendly_name(params.name);
    for (size_t i = 0; i < params.outputPorts.size() && i < ngraphNode->get_output_size(); ++i) {
        if (!params.outputPorts[i].names.empty())
            ngraphNode->get_output_tensor(i).set_names(params.outputPorts[i].names);
    }

    return ngraphNode;
}

}  // namespace

V10Parser::V10Parser(const std::vector<IExtensionPtr>& exts) : _exts(exts) {
    // Load default opsets
    opsets["opset1"] = ngraph::get_opset1();
    opsets["opset2"] = ngraph::get_opset2();
    opsets["opset3"] = ngraph::get_opset3();
    opsets["opset4"] = ngraph::get_opset4();
    opsets["opset5"] = ngraph::get_opset5();
    opsets["opset6"] = ngraph::get_opset6();
    opsets["opset7"] = ngraph::get_opset7();
    opsets["opset8"] = ngraph::get_opset8();

    // Load custom opsets
    for (const auto& ext : exts) {
        for (const auto& it : ext->getOpSets()) {
            if (opsets.find(it.first) != opsets.end())
                IE_THROW() << "Cannot add opset with name: " << it.first
                                   << ". Opset with the same name already exists.";
            opsets[it.first] = it.second;
        }
    }
}

CNNNetwork V10Parser::parse(
    const pugi::xml_node& root, const Blob::CPtr& weights) {
    std::shared_ptr<ngraph::Function> function;
    XmlDeserializer visitor(root, weights, opsets, variables);
    bool use_framework_node{false};
    for (const auto & ext : _exts) {
        const InferenceEngine::Version * version = nullptr;
        ext->GetVersion(version);
        if (version && version->description && strcmp(version->description, "framework_node_ext") == 0) {
            use_framework_node = true;
            break;
        }
    }
    visitor.use_framework_node(use_framework_node);
    visitor.on_attribute("net", function);

    OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::V10Reader_RT, "ConstructCNNNetwork");

    CNNNetwork net(function, _exts);
    parsePreProcess(net, root, weights);

    return net;
}

void V10Parser::parsePreProcess(
    CNNNetwork& network, const pugi::xml_node& root, const Blob::CPtr& weights) {
    /*
        <pre-process mean-precision="FP32">
        <channel id = ”0”>
        <mean offset = "121930449" size = "51529" / >  // in case of array – ref to the .bin file
        </channel>
        </pre-process>
    */

    auto ppNode = root.child("pre-process");
    if (ppNode.empty()) {
        return;
    }
    // find out to what input this belongs to
    std::string inputName;
    InputInfo::Ptr preProcessInput;

    inputName = GetStrAttr(ppNode, "reference-layer-name", "");
    inputName = ngraph::trim(inputName);
    if (inputName.empty()) {
        // fallback (old format), look for the picture in the inputs
        InputsDataMap inputs = network.getInputsInfo();

        if (inputs.empty()) IE_THROW() << "network has no input";

        for (auto i : inputs) {
            if (i.second->getTensorDesc().getDims().size() == 4) {
                preProcessInput = i.second;
                break;
            }
        }
        if (!preProcessInput) {
            preProcessInput = inputs.begin()->second;
        }

        inputName = preProcessInput->name();
    } else {
        preProcessInput = network.getInputsInfo()[inputName];
        if (!preProcessInput)
            IE_THROW() << "pre-process name ref '" << inputName
                               << "' refers to un-existing input";
    }

    // dims vector without batch size
    SizeVector inputDims = preProcessInput->getTensorDesc().getDims();
    size_t noOfChannels = 0, width = 0, height = 0;

    if (inputDims.size() < 2) {
        IE_THROW() << "network did not define input dimensions properly";
    } else if (inputDims.size() == 2) {  // NC
        noOfChannels = inputDims[1];
        width = inputDims[1];
        height = inputDims[0];
    } else if (inputDims.size() == 3) {
        width = inputDims[2];
        height = inputDims[1];
        noOfChannels = inputDims[0];
    } else if (inputDims.size() == 4) {
        width = inputDims[3];
        height = inputDims[2];
        noOfChannels = inputDims[1];
    } else if (inputDims.size() == 5) {
        width = inputDims[4];
        height = inputDims[3];
        noOfChannels = inputDims[2];
    }

    PreProcessInfo& pp = preProcessInput->getPreProcess();
    pp.init(noOfChannels);

    auto meanSegmentPrecision = GetPrecisionAttr(ppNode, "mean-precision", Precision::UNSPECIFIED);
    if (!meanSegmentPrecision || meanSegmentPrecision == Precision::MIXED)
        IE_THROW() << "mean blob defined without specifying precision.";

    int lastChanNo = -1;
    std::unordered_set<int> idsForMeanImage;

    FOREACH_CHILD(chan, ppNode, "channel") {
        int chanNo = GetIntAttr(chan, "id", lastChanNo + 1);
        if (chanNo >= static_cast<int>(noOfChannels) || chanNo < 0) {
            IE_THROW() << "Pre-process channel id invalid: " << chanNo;
        }
        lastChanNo = chanNo;

        auto meanNode = chan.child("mean");
        if (!meanNode.empty()) {
            if (!meanNode.attribute("size")) {
                IE_THROW() << "mean should have the attribute: size";
            }
            if (meanNode.attribute("size")) {
                idsForMeanImage.insert(chanNo);
                size_t size = static_cast<size_t>(GetIntAttr(meanNode, "size"));
                size_t offset = static_cast<size_t>(GetIntAttr(meanNode, "offset"));
                if (width * height * meanSegmentPrecision.size() != size) {
                    IE_THROW() << "mean blob size mismatch expected input, got: " << size
                                       << " extpecting " << width << " x " << height << " x "
                                       << meanSegmentPrecision.size();
                }
                auto meanData = make_blob_with_precision(
                    TensorDesc(meanSegmentPrecision, {height, width}, Layout::HW));
                meanData->allocate();
                auto lockedMem = meanData->buffer();
                char* data = lockedMem.as<char*>();
                uint8_t* src_data = weights->cbuffer().as<uint8_t*>() + offset;
                memcpy(data, src_data, size);

                pp.setMeanImageForChannel(meanData, chanNo);
            }
        }
    }

    if (idsForMeanImage.size() == noOfChannels) {
        pp.setVariant(MEAN_IMAGE);
    } else if (idsForMeanImage.size() == 0) {
        pp.setVariant(NONE);
    } else {
        std::string validMeanImageIds = "";
        for (auto id : idsForMeanImage) {
            validMeanImageIds += std::to_string(id) + " ";
        }
        IE_THROW() << "mean is not provided for all channels\n"
                              "Provided mean image for: "
                           << validMeanImageIds;
    }
}

size_t V10Parser::GenericLayerParams::getRealInputPortId(size_t id) const {
    size_t real_id = 0;
    for (auto& it : inputPorts) {
        if (it.portId == id) {
            return real_id;
        }
        ++real_id;
    }
    IE_THROW() << "Can not find input port with id " << id << " in layer " << name;
}

size_t V10Parser::GenericLayerParams::getRealOutputPortId(size_t id) const {
    size_t real_id = 0;
    for (auto& it : outputPorts) {
        if (it.portId == id) {
            return real_id;
        }
        ++real_id;
    }
    IE_THROW() << "Can not find output port with id " << id << " in layer " << name;
}
}  // namespace InferenceEngine

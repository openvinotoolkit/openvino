// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_ir_parser.hpp"

#include <ie_memcpy.h>

#include <typeinfo>
#include <unordered_set>
#include <algorithm>
#include <deque>
#include <map>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <ngraph/op/strided_slice.hpp>
#include <ngraph/op/not_equal.hpp>
#include <ngraph/ops.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/variant.hpp>

#include <cpp/ie_cnn_network.h>
#include "ie_blob_stream.hpp"
#include "details/caseless.hpp"
#include "ie_ngraph_utils.hpp"
#include "generic_ie.hpp"
#include "precision_utils.h"
#include "blob_factory.hpp"

using namespace InferenceEngine;
using namespace XMLParseUtils;

IRParser::IRParser(size_t version): IRParser(version, {}) {}
IRParser::IRParser(size_t version, const std::vector<InferenceEngine::IExtensionPtr>& exts) {
    switch (version) {
    case 10:
        parser = std::make_shared<V10Parser>(exts);
        break;
    default:
        THROW_IE_EXCEPTION << "Unsupported IR version: " << version;
    }
}

std::shared_ptr<ICNNNetwork> IRParser::parse(const pugi::xml_node& root, std::istream& binStream) {
    return parser->parse(root, binStream);
}

/**
 * Hold original blob in order to avoid situations when original blob is allocated on stack
 */
class WeightsHolderBlob : public TBlob<uint8_t> {
    Blob::CPtr originBlob;

public:
    explicit WeightsHolderBlob(const Blob::CPtr& weights) :
        TBlob<uint8_t>(weights->getTensorDesc(),
                       weights->cbuffer().as<uint8_t*>()),
        originBlob(weights) { }
};

V10Parser::V10Parser(const std::vector<IExtensionPtr>& exts) {
    // Load default opsets
    opsets["opset1"] = ngraph::get_opset1();
    opsets["opset2"] = ngraph::get_opset2();
    opsets["opset3"] = ngraph::get_opset3();
    opsets["opset4"] = ngraph::get_opset4();

    // Load custom opsets
    for (const auto& ext : exts) {
        std::map<std::string, ngraph::OpSet> extOpsets = ext->getOpSets();
        for (const auto& it : extOpsets) {
            if (opsets.find(it.first) != opsets.end())
                THROW_IE_EXCEPTION << "Cannot add opset with name: " << it.first << ". Opset with the same name already exists.";
            opsets[it.first] = it.second;
        }
    }
}

std::shared_ptr<ICNNNetwork> V10Parser::parse(const pugi::xml_node& root, std::istream& binStream) {
    using node_params = struct {
        pugi::xml_node xml;
        GenericLayerParams params;
    };
    std::map<size_t, node_params> params;

    std::vector<size_t> outputs;
    std::unordered_set<std::string> opName;

    // Read all layers and store their parameters in params map
    FOREACH_CHILD(node, root.child("layers"), "layer") {
        auto node_param = parseGenericParams(node);
        if (opName.find(node_param.name) != opName.end())
            THROW_IE_EXCEPTION << "Invalid IR! " << node_param.name << " name is not unique!";
        opName.insert(node_param.name);
        params[node_param.layerId] = {node, node_param};
        if (node_param.type == "Result" || node_param.type == "Assign") {
            outputs.push_back(node_param.layerId);
        }
    }

    using edge = struct { size_t fromLayerId, fromPortId, toPortId; };
    std::map<size_t, std::vector<edge>> edges;
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

    ngraph::ParameterVector parameter_nodes;
    ngraph::ResultVector result_nodes;
    ngraph::NodeVector allNodes;
    std::vector<std::shared_ptr<ngraph::op::Assign>> assign_nodes;
    std::map<std::string, std::shared_ptr<ngraph::Node>> variable_id_to_read_value;

    //  Following topological order create nGraph operations
    for (auto& layer_id : order) {
        auto& p = params[layer_id];
        ngraph::OutputVector inputs(edges[layer_id].size());
        for (auto& e : edges[layer_id]) {
            auto input_node = id_to_node[e.fromLayerId];
            if (!input_node) {
                THROW_IE_EXCEPTION << "Attempt to access node " << e.fromLayerId << " that not in graph.";
            }
            auto& p_output = params[e.fromLayerId].params;
            if (p.params.getRealInputPortId(e.toPortId) >= inputs.size())
                THROW_IE_EXCEPTION << p.params.type << " layer " << p.params.name << " with id: " << p.params.layerId
                    << " is inconsistent!";
            inputs[p.params.getRealInputPortId(e.toPortId)] =
                input_node->output(p_output.getRealOutputPortId(e.fromPortId));
        }

        auto node = createNode(inputs, p.xml, binStream, p.params);
        id_to_node[layer_id] = node;

        // Check that output shape after nGraph node validation the same as in IR
        // because IR always right!
        // Temporary disabled!
        //        for (size_t i = 0; i < p.params.outputPorts.size(); ++i) {
        //            if (p.params.outputPorts[i].dims != node->output(i).get_shape()) {
        //                THROW_IE_EXCEPTION << "Shape after nGraph infer " <<
        //                details::dumpVec(node->output(i).get_shape())
        //                                   << " differ from IR shapes: " <<
        //                                   details::dumpVec(p.params.outputPorts[i].dims);
        //            }
        //        }

        if (auto parameter_node = std::dynamic_pointer_cast<ngraph::op::Parameter>(node)) {
            parameter_nodes.emplace_back(parameter_node);
        }

        if (auto result_node = std::dynamic_pointer_cast<ngraph::op::Result>(node)) {
            result_nodes.emplace_back(result_node);
        }

        if (auto assign_node = std::dynamic_pointer_cast<ngraph::op::Assign>(node)) {
            assign_nodes.emplace_back(assign_node);
        }

        if (auto read_value_node = std::dynamic_pointer_cast<ngraph::op::ReadValue>(node)) {
            variable_id_to_read_value[read_value_node->get_variable_id()] = read_value_node;
        }
        allNodes.emplace_back(node);
    }

    ::ngraph::op::GenericIE::DisableReshape noReshape(allNodes);
    auto function = std::make_shared<ngraph::Function>(result_nodes, parameter_nodes, GetStrAttr(root, "name", ""));
    if (!result_nodes.empty()) {
        for (const auto& assign : assign_nodes) {
            assign->add_control_dependency(variable_id_to_read_value.at(assign->get_variable_id()));
            // often Assign node is a leaf of the graph, we add control_dependency for one of the results
            // to make Assign node visible for traversals get_ops(), get_ordered_ops()
            result_nodes[0]->add_control_dependency(assign);
        }
    }
    return CNNNetwork(function);
}

V10Parser::GenericLayerParams V10Parser::parseGenericParams(const pugi::xml_node& node) {
    const auto parsePort = [](const pugi::xml_node& parentNode,
                              const GenericLayerParams& params,
                              bool input) -> GenericLayerParams::LayerPortData {
        GenericLayerParams::LayerPortData port;

        port.portId = GetIntAttr(parentNode, "id");

        for (auto node = parentNode.child("dim"); !node.empty(); node = node.next_sibling("dim")) {
            size_t dim = 0;
            const pugi::char_t* dimVal = node.child_value();
            std::stringstream ss(dimVal);
            if (!(ss >> dim) || dim == 0) {
                THROW_IE_EXCEPTION << "dimension (" << dimVal << ") in node " << node.name()
                                   << " must be a positive integer: at offset " << node.offset_debug();
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
        return port;
    };
    GenericLayerParams params;

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

bool V10Parser::LayerBaseCreator::shouldCreate(const std::string& nodeType) const {
    InferenceEngine::details::CaselessEq<std::string> comparator;
    return comparator(nodeType, type);
}

std::shared_ptr<ngraph::Node> V10Parser::createNode(const std::vector<ngraph::Output<ngraph::Node>>& inputs,
                                                    const pugi::xml_node& node, std::istream& binStream,
                                                    const GenericLayerParams& params) {
    static std::vector<std::shared_ptr<LayerBaseCreator>> creators = {
        std::make_shared<LayerCreator<ngraph::op::Abs>>("Abs"),
        std::make_shared<LayerCreator<ngraph::op::Acos>>("Acos"),
        std::make_shared<LayerCreator<ngraph::op::v1::Add>>("Add"),
        std::make_shared<LayerCreator<ngraph::op::Asin>>("Asin"),
        std::make_shared<LayerCreator<ngraph::op::Atan>>("Atan"),
        std::make_shared<LayerCreator<ngraph::op::v1::AvgPool>>("AvgPool"),
        std::make_shared<LayerCreator<ngraph::op::BatchNormInference>>("BatchNormInference"),
        std::make_shared<LayerCreator<ngraph::op::Ceiling>>("Ceiling"),
        std::make_shared<LayerCreator<ngraph::op::Clamp>>("Clamp"),
        std::make_shared<LayerCreator<ngraph::op::Concat>>("Concat"),
        std::make_shared<LayerCreator<ngraph::op::Constant>>("Const"),
        std::make_shared<LayerCreator<ngraph::op::Convert>>("Convert"),
        std::make_shared<LayerCreator<ngraph::op::CTCGreedyDecoder>>("CTCGreedyDecoder"),
        std::make_shared<LayerCreator<ngraph::op::ReverseSequence>>("ReverseSequence"),
        std::make_shared<LayerCreator<ngraph::op::Cos>>("Cos"),
        std::make_shared<LayerCreator<ngraph::op::Cosh>>("Cosh"),
        std::make_shared<LayerCreator<ngraph::op::DetectionOutput>>("DetectionOutput"),
        std::make_shared<LayerCreator<ngraph::op::v1::DeformableConvolution>>("DeformableConvolution"),
        std::make_shared<LayerCreator<ngraph::op::v1::DeformablePSROIPooling>>("DeformablePSROIPooling"),
        std::make_shared<LayerCreator<ngraph::op::v1::Divide>>("Divide"),
        std::make_shared<LayerCreator<ngraph::op::SpaceToDepth>>("SpaceToDepth"),
        std::make_shared<LayerCreator<ngraph::op::DepthToSpace>>("DepthToSpace"),
        std::make_shared<LayerCreator<ngraph::op::v1::Subtract>>("Subtract"),
        std::make_shared<LayerCreator<ngraph::op::MatMul>>("MatMul"),
        std::make_shared<LayerCreator<ngraph::op::v1::Broadcast>>("Broadcast"),
        std::make_shared<LayerCreator<ngraph::op::v1::Reshape>>("Reshape"),
        std::make_shared<LayerCreator<ngraph::op::v1::StridedSlice>>("StridedSlice"),
        std::make_shared<LayerCreator<ngraph::op::Elu>>("ELU"),
        std::make_shared<LayerCreator<ngraph::op::Exp>>("Exp"),
        std::make_shared<LayerCreator<ngraph::op::Erf>>("Erf"),
        std::make_shared<LayerCreator<ngraph::op::FakeQuantize>>("FakeQuantize"),
        std::make_shared<LayerCreator<ngraph::op::Floor>>("Floor"),
        std::make_shared<LayerCreator<ngraph::op::v1::Gather>>("Gather"),
        std::make_shared<LayerCreator<ngraph::op::v1::GatherTree>>("GatherTree"),
        std::make_shared<LayerCreator<ngraph::op::v1::Greater>>("Greater"),
        std::make_shared<LayerCreator<ngraph::op::v1::GreaterEqual>>("GreaterEqual"),
        std::make_shared<LayerCreator<ngraph::op::v1::Convolution>>("Convolution"),
        std::make_shared<LayerCreator<ngraph::op::v1::GroupConvolution>>("GroupConvolution"),
        std::make_shared<LayerCreator<ngraph::op::v1::ConvolutionBackpropData>>("ConvolutionBackpropData"),
        std::make_shared<LayerCreator<ngraph::op::v1::GroupConvolutionBackpropData>>("GroupConvolutionBackpropData"),
        std::make_shared<LayerCreator<ngraph::op::v1::BinaryConvolution>>("BinaryConvolution"),
        std::make_shared<LayerCreator<ngraph::op::GRN>>("GRN"),
        std::make_shared<LayerCreator<ngraph::op::HardSigmoid>>("HardSigmoid"),
        std::make_shared<LayerCreator<ngraph::op::Interpolate>>("Interpolate"),
        std::make_shared<LayerCreator<ngraph::op::Log>>("Log"),
        std::make_shared<LayerCreator<ngraph::op::SquaredDifference>>("SquaredDifference"),
        std::make_shared<LayerCreator<ngraph::op::v1::Less>>("Less"),
        std::make_shared<LayerCreator<ngraph::op::v1::LessEqual>>("LessEqual"),
        std::make_shared<LayerCreator<ngraph::op::v1::Equal>>("Equal"),
        std::make_shared<LayerCreator<ngraph::op::v1::NotEqual>>("NotEqual"),
        std::make_shared<LayerCreator<ngraph::op::v1::FloorMod>>("FloorMod"),
        std::make_shared<LayerCreator<ngraph::op::v1::Select>>("Select"),
        std::make_shared<LayerCreator<ngraph::op::LRN>>("LRN"),
        std::make_shared<LayerCreator<ngraph::op::MVN>>("MVN"),
        std::make_shared<LayerCreator<ngraph::op::LSTMCell>>("LSTMCell"),
        std::make_shared<LayerCreator<ngraph::op::v1::MaxPool>>("MaxPool"),
        std::make_shared<LayerCreator<ngraph::op::v1::Maximum>>("Maximum"),
        std::make_shared<LayerCreator<ngraph::op::v1::Minimum>>("Minimum"),
        std::make_shared<LayerCreator<ngraph::op::v1::Multiply>>("Multiply"),
        std::make_shared<LayerCreator<ngraph::op::Negative>>("Negative"),
        std::make_shared<LayerCreator<ngraph::op::v1::NonMaxSuppression>>("NonMaxSuppression"),
        std::make_shared<LayerCreator<ngraph::op::NormalizeL2>>("NormalizeL2"),
        std::make_shared<LayerCreator<ngraph::op::v1::OneHot>>("OneHot"),
        std::make_shared<LayerCreator<ngraph::op::PRelu>>("PReLU"),
        std::make_shared<LayerCreator<ngraph::op::Relu>>("ReLU"),
        std::make_shared<LayerCreator<ngraph::op::v1::Pad>>("Pad"),
        std::make_shared<LayerCreator<ngraph::op::v1::Power>>("Power"),
        std::make_shared<LayerCreator<ngraph::op::Range>>("Range"),
        std::make_shared<LayerCreator<ngraph::op::PriorBox>>("PriorBox"),
        std::make_shared<LayerCreator<ngraph::op::PriorBoxClustered>>("PriorBoxClustered"),
        std::make_shared<LayerCreator<ngraph::op::Proposal>>("Proposal"),
        std::make_shared<LayerCreator<ngraph::op::v1::ReduceMax>>("ReduceMax"),
        std::make_shared<LayerCreator<ngraph::op::v1::ReduceMin>>("ReduceMin"),
        std::make_shared<LayerCreator<ngraph::op::v1::ReduceMean>>("ReduceMean"),
        std::make_shared<LayerCreator<ngraph::op::v1::ReduceProd>>("ReduceProd"),
        std::make_shared<LayerCreator<ngraph::op::v1::ReduceSum>>("ReduceSum"),
        std::make_shared<LayerCreator<ngraph::op::ReorgYolo>>("ReorgYolo"),
        std::make_shared<LayerCreator<ngraph::op::RegionYolo>>("RegionYolo"),
        std::make_shared<LayerCreator<ngraph::op::Result>>("Result"),
        std::make_shared<LayerCreator<ngraph::op::ROIPooling>>("ROIPooling"),
        std::make_shared<LayerCreator<ngraph::op::PSROIPooling>>("PSROIPooling"),
        std::make_shared<LayerCreator<ngraph::op::ShapeOf>>("ShapeOf"),
        std::make_shared<LayerCreator<ngraph::op::v0::Selu>>("Selu"),
        std::make_shared<LayerCreator<ngraph::op::Sigmoid>>("Sigmoid"),
        std::make_shared<LayerCreator<ngraph::op::Sin>>("Sin"),
        std::make_shared<LayerCreator<ngraph::op::Sign>>("Sign"),
        std::make_shared<LayerCreator<ngraph::op::Sinh>>("Sinh"),
        std::make_shared<LayerCreator<ngraph::op::v1::Softmax>>("Softmax"),
        std::make_shared<LayerCreator<ngraph::op::v1::Split>>("Split"),
        std::make_shared<LayerCreator<ngraph::op::VariadicSplit>>("VariadicSplit"),
        std::make_shared<LayerCreator<ngraph::op::Sqrt>>("Sqrt"),
        std::make_shared<LayerCreator<ngraph::op::Squeeze>>("Squeeze"),
        std::make_shared<LayerCreator<ngraph::op::Tan>>("Tan"),
        std::make_shared<LayerCreator<ngraph::op::Tanh>>("TanH"),
        std::make_shared<LayerCreator<ngraph::op::Tile>>("Tile"),
        std::make_shared<LayerCreator<ngraph::op::v1::TopK>>("TopK"),
        std::make_shared<LayerCreator<ngraph::op::TensorIterator>>("TensorIterator"),
        std::make_shared<LayerCreator<ngraph::op::Transpose>>("Transpose"),
        std::make_shared<LayerCreator<ngraph::op::Unsqueeze>>("Unsqueeze"),
        std::make_shared<LayerCreator<ngraph::op::v1::LogicalAnd>>("LogicalAnd"),
        std::make_shared<LayerCreator<ngraph::op::v1::LogicalOr>>("LogicalOr"),
        std::make_shared<LayerCreator<ngraph::op::v1::LogicalXor>>("LogicalXor"),
        std::make_shared<LayerCreator<ngraph::op::v1::LogicalNot>>("LogicalNot"),
        std::make_shared<LayerCreator<ngraph::op::v1::ReduceLogicalAnd>>("ReduceLogicalAnd"),
        std::make_shared<LayerCreator<ngraph::op::v1::ReduceLogicalOr>>("ReduceLogicalOr"),
    };

    // Check that operation in default opsets
    auto isDefaultOpSet = [](const std::string& version) -> bool {
        for (size_t i = 1; i <= 3; i++) {
            std::string opset_name = "opset" + std::to_string(i);
            if (version == opset_name)
                return true;
        }
        return false;
    };

    for (size_t i = 0; i < inputs.size(); i++) {
        if (!inputs[i].get_node())
            THROW_IE_EXCEPTION << params.type << " layer " << params.name << " with id: " << params.layerId
                << " has incorrect input with index " << i << "!";
        if (inputs[i].get_element_type().get_type_enum() == ngraph::element::Type_t::undefined)
            THROW_IE_EXCEPTION << params.type << " layer " << params.name << " with id: " << params.layerId
                << " has undefined element type for input with index " << i << "!";
    }

    std::shared_ptr<ngraph::Node> ngraphNode;
    if (isDefaultOpSet(params.version)) {
        // Try to create operation from creators
        for (const auto& creator : creators) {
            if (creator->shouldCreate(params.type)) {
                bool useCreator = false;
                // Check that opset is registered
                useCreator |= opsets.find(params.version) == opsets.end();
                if (!useCreator) {
                    // Check that creator can create operation with the version from opset
                    const auto opset = opsets.at(params.version);
                    // Opset should contains the same version of operation or doesn't contain operation with current type
                    useCreator |= opset.contains_type(creator->getNodeType()) || !opset.contains_type(params.type);
                }
                if (useCreator)
                    ngraphNode = creator->createLayer(inputs, node, binStream, params);
                break;
            }
        }
    }

    // Try to create operation from loaded opsets
    if (!ngraphNode && opsets.count(params.version)) {
        auto opset = opsets.at(params.version);

        if (!opset.contains_type(params.type)) {
            THROW_IE_EXCEPTION << "Opset " << params.version << " doesn't contain the operation with type: " << params.type;
        }

        ngraphNode = std::shared_ptr<ngraph::Node>(opset.create(params.type));
        ngraphNode->set_arguments(inputs);
        XmlDeserializer visitor(node);
        if (ngraphNode->visit_attributes(visitor))
            ngraphNode->constructor_validate_and_infer_types();
    }

    // Create GenericIE operation for backward compatibility
    if (!ngraphNode && (params.version == "experimental" || params.version == "extension")) {
        // Try to create Generic node for backward compatibility
        std::map<std::string, Parameter> parameters;
        pugi::xml_node dn = node.child("data");
        if (dn) {
            for (const auto& attr : dn.attributes()) {
                parameters[attr.name()] = std::string(attr.value());
            }
        }

        auto blobs = node.child("blobs");
        if (!blobs.empty()) {
            binStream.seekg(0, std::ios::end);
            std::streampos length = binStream.tellg();

            for (pugi::xml_node blob = blobs.first_child(); !blob.empty(); blob = blob.next_sibling()) {
                size_t size = GetUInt64Attr(blob, "size", 0);
                uint64_t offset = GetUInt64Attr(blob, "offset", 0);
                Precision precision(Precision::U8);
                const std::string& preStr = GetStrAttr(blob, "precision", "");
                if (!preStr.empty())
                    precision = Precision::FromStr(preStr);
                if (!size) continue;
                if (!length)
                    THROW_IE_EXCEPTION << "Cannot read network! The model requires weights data! "
                        << "Bin file cannot be found! Please specify the path to bin file.";
                if (length < offset + size)
                    THROW_IE_EXCEPTION << "Cannot create " << params.type << " layer with name: " << params.name
                                       << ". Layer has incorrect weights!";
                Blob::Ptr wBlob = make_blob_with_precision(TensorDesc(precision, {size / precision.size()}, Layout::C));
                wBlob->allocate();
                char* data = wBlob->buffer().as<char*>();
                binStream.seekg(offset, std::ios::beg);
                binStream.read(data, size);

                Blob::CPtr cBlob = wBlob;
                parameters[blob.name()] = wBlob;
            }
        }
        std::vector<ngraph::op::GenericIE::PortIE> outputs;
        for (const auto& port : params.outputPorts) {
            ngraph::op::GenericIE::PortIE iePort;
            iePort.dims = port.dims;
            iePort.precision = InferenceEngine::details::convertPrecision(port.precision);
            outputs.emplace_back(iePort);
        }

        ngraphNode = std::make_shared<ngraph::op::GenericIE>(inputs, parameters, params.type, outputs);
    }

    if (!ngraphNode) {
        THROW_IE_EXCEPTION << "Cannot create " << params.type << " layer " << params.name << " id:" << params.layerId;
    }

    // Save run time info
    auto& rtInfo = ngraphNode->get_rt_info();
    pugi::xml_node dn = node.child("data");
    if (dn) {
        const auto pr_data = dn.attribute("PrimitivesPriority");
        if (pr_data) {
            rtInfo["PrimitivesPriority"] = std::make_shared<::ngraph::VariantWrapper<std::string> >(pr_data.value());
        }
    }

    ngraphNode->set_friendly_name(params.name);

    return ngraphNode;
}

namespace InferenceEngine {


// DetectionOutput layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::DetectionOutput>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    ngraph::op::DetectionOutputAttrs attr;

    attr.num_classes = GetIntAttr(dn, "num_classes");
    attr.background_label_id = GetIntAttr(dn, "background_label_id", 0);
    attr.top_k = GetIntAttr(dn, "top_k", -1);
    attr.variance_encoded_in_target = GetIntAttr(dn, "variance_encoded_in_target", 0) != 0;
    attr.keep_top_k = getParameters<int>(dn, "keep_top_k", {});
    attr.code_type = GetStrAttr(dn, "code_type", "caffe.PriorBoxParameter.CORNER");
    attr.share_location = GetIntAttr(dn, "share_location", 1) != 0;
    attr.clip_after_nms = GetIntAttr(dn, "clip_after_nms", 0) != 0;
    attr.clip_before_nms = GetIntAttr(dn, "clip_before_nms", 0) != 0;
    attr.decrease_label_id = GetIntAttr(dn, "decrease_label_id", 0) != 0;
    attr.normalized = GetIntAttr(dn, "normalized", 0) != 0;
    attr.input_height = GetUIntAttr(dn, "input_height", 1);
    attr.input_width = GetUIntAttr(dn, "input_width", 1);
    attr.objectness_score = GetFloatAttr(dn, "objectness_score", 0);
    attr.nms_threshold = GetFloatAttr(dn, "nms_threshold");
    attr.confidence_threshold = GetFloatAttr(dn, "confidence_threshold", 0);

    if (inputs.size() != 3 && inputs.size() != 5) {
        THROW_IE_EXCEPTION << "DetectionOutput has incorrect number of input ports!";
    }

    if (inputs.size() == 3) {
        return std::make_shared<ngraph::op::DetectionOutput>(inputs[0],
                                                             inputs[1],
                                                             inputs[2],
                                                             attr);
    } else {
        return std::make_shared<ngraph::op::DetectionOutput>(inputs[0],
                                                             inputs[1],
                                                             inputs[2],
                                                             inputs[3],
                                                             inputs[4],
                                                             attr);
    }
}

// TensorIterator layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::TensorIterator>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    auto tensor_iterator = std::make_shared<ngraph::op::TensorIterator>();
    tensor_iterator->set_friendly_name(GetStrAttr(node, "name"));
    auto body_node = node.child("body");

    if (body_node.empty()) {
        THROW_IE_EXCEPTION << "TensorIterator has no body.";
    }

    // Fill map: result/parameter id to name
    std::map<uint64_t, std::string> layer_idx_to_name;
    FOREACH_CHILD(_layer, body_node.child("layers"), "layer") {
        auto type = GetStrAttr(_layer, "type");

        if (type == "Result" || type == "Parameter") {
            auto id = GetUIntAttr(_layer, "id");
            auto name = GetStrAttr(_layer, "name");
            layer_idx_to_name[id] = name;
        }
    }

    // Create ngraph::Function, convert it to ngraph::BodyLambda and set it as TensorIterator body
    IRParser parser(10);
    auto ngraph_function = parser.parse(node.child("body"), binStream)->getFunction();
    auto parameter_nodes = ngraph_function->get_parameters();
    auto result_nodes = ngraph_function->get_results();
    // Disabled reshape for generic operations in the TI body
    ::ngraph::op::GenericIE::DisableReshape noReshape(ngraph_function);
    auto body = std::make_shared<ngraph::op::TensorIterator::BodyLambda>(result_nodes, parameter_nodes);
    tensor_iterator->set_body(body);

    // Parse PortMap: inputs
    std::map<uint64_t, pugi::xml_node> input_map;
    FOREACH_CHILD(_input, node.child("port_map"), "input") {
        int64_t ext_port_id = GetUIntAttr(_input, "external_port_id");
        input_map[ext_port_id] = _input;
    }

    bool is_sliced_input_exists = false;
    for (const auto& input : input_map) {
        auto &_input = input.second;
        auto axis_attr = _input.attribute("axis");
        size_t ti_input_index = GetUIntAttr(_input, "external_port_id");
        size_t body_parameter_index = GetUIntAttr(_input, "internal_layer_id");

        auto body_param = std::find_if(parameter_nodes.begin(), parameter_nodes.end(),
                                       [&](const std::shared_ptr<ngraph::op::Parameter>& param) {
                                           return param->get_friendly_name() == layer_idx_to_name[body_parameter_index];
                                       });

        if (body_param == parameter_nodes.end()) {
            THROW_IE_EXCEPTION << "PortMap input parsing error. Body parameter with id = " << body_parameter_index
                               << " not found.";
        }
        if (ti_input_index >= inputs.size())
            THROW_IE_EXCEPTION << "TensorIterator " << layerParsePrms.name << " has incorrect number of inputs!";

        // if axis is set, then slicing is enabled. Create ngraph::TensorIterator::SlicedInput.
        if (!axis_attr.empty()) {
            size_t axis = GetUIntAttr(_input, "axis");
            int64_t start = GetInt64Attr(_input, "start", 0);
            int64_t stride = GetInt64Attr(_input, "stride", 1);
            int64_t end = GetInt64Attr(_input, "end", -1);
            int64_t part_size = GetInt64Attr(_input, "part_size", 1);
            tensor_iterator->set_sliced_input(*body_param, inputs[ti_input_index], start, stride, part_size, end, axis);
            is_sliced_input_exists = true;
        } else {
            // otherwise find corresponding back edge and create ngraph::TensorIterator::MergedInput
            bool is_back_edge_exist = false;
            FOREACH_CHILD(_edge, node.child("back_edges"), "edge") {
                size_t to_layer = GetUIntAttr(_edge, "to-layer");

                if (to_layer == body_parameter_index) {
                    size_t from_layer = GetUIntAttr(_edge, "from-layer");

                    auto body_result = std::find_if(
                        result_nodes.begin(), result_nodes.end(), [&](std::shared_ptr<ngraph::op::Result>& result) {
                            return result->get_friendly_name() == layer_idx_to_name[from_layer];
                        });

                    if (body_result == result_nodes.end()) {
                        THROW_IE_EXCEPTION << "PortMap input parsing error. Body result with id = " << from_layer
                                           << " not found.";
                    }

                    tensor_iterator->set_merged_input(*body_param, inputs[ti_input_index], *body_result);
                    is_back_edge_exist = true;
                    break;
                }
            }

            if (!is_back_edge_exist) {
                tensor_iterator->set_invariant_input(*body_param, inputs[ti_input_index]);
            }
        }
    }

    // Parse PortMap: outputs
    std::map<uint32_t, pugi::xml_node> output_map;
    FOREACH_CHILD(_output, node.child("port_map"), "output") {
        uint32_t ext_port_id = GetUIntAttr(_output, "external_port_id");
        output_map[ext_port_id] = _output;
    }

    for (const auto& output : output_map) {
        auto& _output = output.second;
        auto axis_attr = _output.attribute("axis");
        size_t body_result_index = GetUIntAttr(_output, "internal_layer_id");

        auto body_result =
            std::find_if(result_nodes.begin(), result_nodes.end(), [&](std::shared_ptr<ngraph::op::Result>& result) {
                return result->get_friendly_name() == layer_idx_to_name[body_result_index];
            });

        if (body_result == result_nodes.end()) {
            THROW_IE_EXCEPTION << "PortMap output parsing error. Body result with id = " << body_result_index
                               << " not found.";
        }

        // if axis is set, then concatenation is enabled. Create ngraph::TensorIterator::ConcatOutput.
        if (!axis_attr.empty()) {
            uint32_t axis = GetUIntAttr(_output, "axis");
            int64_t start = GetInt64Attr(_output, "start", 0);
            int64_t stride = GetInt64Attr(_output, "stride", 1);
            int64_t end = GetInt64Attr(_output, "end", -1);
            int64_t part_size = GetInt64Attr(_output, "part_size", 1);
            tensor_iterator->get_concatenated_slices(*body_result, start, stride, part_size, end, axis);

            if (!is_sliced_input_exists) {
                tensor_iterator->set_num_iterations((std::abs(end - start)) / part_size);
            }
        } else {
            // otherwise create ngraph::TensorIterator::BodyOutput. -1 means last iteration.
            tensor_iterator->get_iter_value(*body_result, -1);
        }
    }

    tensor_iterator->validate_and_infer_types();
    return tensor_iterator;
}

// PriorBoxClustered layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::PriorBoxClustered>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    ngraph::op::PriorBoxClusteredAttrs attr;
    attr.widths = getParameters<float>(dn, "width");
    attr.heights = getParameters<float>(dn, "height");
    attr.variances = getParameters<float>(dn, "variance");
    attr.offset = GetFloatAttr(dn, "offset");
    float step = GetFloatAttr(dn, "step", 0);
    attr.step_heights = GetFloatAttr(dn, "step_h", step);
    attr.step_widths = GetFloatAttr(dn, "step_w", step);
    if (step != 0) {
        attr.step_heights = step;
        attr.step_widths = step;
    }
    attr.clip = (GetIntAttr(dn, "clip") != 0);

    return std::make_shared<ngraph::op::PriorBoxClustered>(inputs[0], inputs[1], attr);
}

// Proposal layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Proposal>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 3);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    ngraph::op::ProposalAttrs attr;
    attr.base_size = GetUIntAttr(dn, "base_size");
    attr.pre_nms_topn = GetUIntAttr(dn, "pre_nms_topn");
    attr.post_nms_topn = GetUIntAttr(dn, "post_nms_topn");
    attr.nms_thresh = GetFloatAttr(dn, "nms_thresh");
    attr.feat_stride = GetUIntAttr(dn, "feat_stride");
    attr.min_size = GetUIntAttr(dn, "min_size");
    attr.ratio = getParameters<float>(dn, "ratio");
    attr.scale = getParameters<float>(dn, "scale");
    attr.clip_after_nms = (GetIntAttr(dn, "clip_after_nms", 0) != 0);
    attr.clip_before_nms = (GetIntAttr(dn, "clip_before_nms", 1) != 0);
    attr.normalize = (GetIntAttr(dn, "normalize", 0) != 0);
    attr.box_size_scale = GetFloatAttr(dn, "box_size_scale", 1.0f);
    attr.box_coordinate_scale = GetFloatAttr(dn, "box_coordinate_scale", 1.0f);
    attr.framework = GetStrAttr(dn, "framework", "");

    return std::make_shared<ngraph::op::Proposal>(inputs[0], inputs[1], inputs[2], attr);
}

// PriorBox layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::PriorBox>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    ngraph::op::PriorBoxAttrs attr;
    attr.min_size = getParameters<float>(dn, "min_size", {});
    attr.max_size = getParameters<float>(dn, "max_size", {});
    attr.density = getParameters<float>(dn, "density", {});
    attr.fixed_size = getParameters<float>(dn, "fixed_size", {});
    attr.fixed_ratio = getParameters<float>(dn, "fixed_ratio", {});
    attr.aspect_ratio = getParameters<float>(dn, "aspect_ratio", {});
    attr.variance = getParameters<float>(dn, "variance", {});
    attr.step = GetFloatAttr(dn, "step", 0);
    attr.offset = GetFloatAttr(dn, "offset");
    attr.clip = (GetIntAttr(dn, "clip") != 0);
    attr.flip = (GetIntAttr(dn, "flip") != 0);
    attr.scale_all_sizes = (GetIntAttr(dn, "scale_all_sizes", 1) != 0);

    return std::make_shared<ngraph::op::PriorBox>(inputs[0], inputs[1], attr);
}

// ShapeOf layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::ShapeOf>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::ShapeOf>(inputs[0]);
}

// FakeQuantize layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::FakeQuantize>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 5);
    pugi::xml_node dn = node.child("data");
    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::FakeQuantize>(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                                                      GetUIntAttr(dn, "levels"));
}

// ReverseSequence layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::ReverseSequence>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
                                                                                                std::istream& binStream,
                                                                                                const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");
    return std::make_shared<ngraph::op::ReverseSequence>(inputs[0], inputs[1], GetIntAttr(dn, "batch_axis", 0), GetIntAttr(dn, "seq_axis", 1));
}

// Covnert layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Convert>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");
    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::Convert>(inputs[0],
                                                 details::convertPrecision(GetStrAttr(dn, "destination_type")));
}

// LSTMCell layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::LSTMCell>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 6);
    pugi::xml_node dn = node.child("data");
    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    std::vector<std::string> activations = getParameters<std::string>(dn, "activations", {"sigmoid", "tanh", "tanh"});
    std::vector<float> activations_alpha = getParameters<float>(dn, "activations_alpha", {});
    std::vector<float> activations_beta = getParameters<float>(dn, "activations_beta", {});
    float clip = GetFloatAttr(dn, "clip", 0.f);
    return std::make_shared<ngraph::op::LSTMCell>(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                                                  GetUInt64Attr(dn, "hidden_size"), ngraph::op::LSTMWeightsFormat::IFCO,
                                                  activations, activations_alpha, activations_beta, clip);
}

// BatchNormInference layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::BatchNormInference>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 5);
    pugi::xml_node dn = node.child("data");
    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    float eps = GetFloatAttr(dn, "eps");
    return std::make_shared<ngraph::op::BatchNormInference>(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], eps);
}

// CTCGreedyDecoder layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::CTCGreedyDecoder>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");
    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::CTCGreedyDecoder>(inputs[0], inputs[1],
                                                          GetBoolAttr(dn, "ctc_merge_repeated", true));
}

// TopK layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::TopK>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");
    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    size_t axis = GetUInt64Attr(dn, "axis");
    std::string str_mode = GetStrAttr(dn, "mode");
    std::string str_sort = GetStrAttr(dn, "sort");

    ngraph::op::v1::TopK::Mode mode;
    ngraph::op::v1::TopK::SortType sort;
    if (str_mode == "max") {
        mode = ngraph::op::v1::TopK::Mode::MAX;
    } else if (str_mode == "min") {
        mode = ngraph::op::v1::TopK::Mode::MIN;
    } else {
        THROW_IE_EXCEPTION << "Unsupported mode: " << str_mode;
    }

    if (str_sort == "none") {
        sort = ngraph::op::v1::TopK::SortType::NONE;
    } else if (str_sort == "value") {
        sort = ngraph::op::v1::TopK::SortType::SORT_VALUES;
    } else if (str_sort == "index") {
        sort = ngraph::op::v1::TopK::SortType::SORT_INDICES;
    } else {
        THROW_IE_EXCEPTION << "Unsupported sort type: " << str_sort;
    }

    return std::make_shared<ngraph::op::v1::TopK>(inputs[0], inputs[1], axis, mode, sort);
}

// Pad layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Pad>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    std::string pad_mode_str = GetStrAttr(dn, "pad_mode");
    ngraph::op::PadMode pad_mode;

    if (pad_mode_str == "constant") {
        pad_mode = ngraph::op::PadMode::CONSTANT;
    } else if (pad_mode_str == "edge") {
        pad_mode = ngraph::op::PadMode::EDGE;
    } else if (pad_mode_str == "reflect") {
        pad_mode = ngraph::op::PadMode::REFLECT;
    } else if (pad_mode_str == "symmetric") {
        pad_mode = ngraph::op::PadMode::SYMMETRIC;
    } else {
        THROW_IE_EXCEPTION << "Pad mode: " << pad_mode_str << " is not supported";
    }

    if (pad_mode == ngraph::op::PadMode::CONSTANT) {
        if (inputs.size() == 3) {
            return std::make_shared<ngraph::op::v1::Pad>(inputs[0], inputs[1], inputs[2], pad_mode);
        }
        checkParameters(inputs, layerParsePrms, 4);
        return std::make_shared<ngraph::op::v1::Pad>(inputs[0], inputs[1], inputs[2], inputs[3], pad_mode);
    }

    checkParameters(inputs, layerParsePrms, 3);
    return std::make_shared<ngraph::op::v1::Pad>(inputs[0], inputs[1], inputs[2], pad_mode);
}

// SquaredDifference layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::SquaredDifference>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::SquaredDifference>(inputs[0], inputs[1]);
}

// Greater layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Greater>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::Greater>(inputs[0], inputs[1]);
}

// GreaterEqual layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::GreaterEqual>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::GreaterEqual>(inputs[0], inputs[1]);
}

// Less layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Less>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::Less>(inputs[0], inputs[1]);
}

// LessEqual layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::LessEqual>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::LessEqual>(inputs[0], inputs[1]);
}

// Equal layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Equal>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::Equal>(inputs[0], inputs[1]);
}

// NotEqual layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::NotEqual>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::NotEqual>(inputs[0], inputs[1]);
}

// FloorMod layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::FloorMod>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::FloorMod>(inputs[0], inputs[1]);
}

// Select layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Select>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 3);
    return std::make_shared<ngraph::op::v1::Select>(inputs[0], inputs[1], inputs[2]);
}

// MVN layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::MVN>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    double eps = GetFloatAttr(dn, "eps");
    bool across = GetUIntAttr(dn, "across_channels", 0) == 1;
    bool normalize_variance = GetUIntAttr(dn, "normalize_variance", 0) == 1;
    return std::make_shared<ngraph::op::MVN>(inputs[0], across, normalize_variance, eps);
}

// Log layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Log>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Log>(inputs[0]);
}

// LRN layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::LRN>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");
    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::LRN>(inputs[0],
                                             inputs[1],
                                             GetFloatAttr(dn, "alpha"),
                                             GetFloatAttr(dn, "beta"),
                                             GetFloatAttr(dn, "bias"),
                                             GetUInt64Attr(dn, "size"));
}

// Clamp layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Clamp>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    double maxVal = GetFloatAttr(dn, "max");
    double minVal = GetFloatAttr(dn, "min");
    return std::make_shared<ngraph::op::Clamp>(inputs[0], minVal, maxVal);
}

// VariadicSplit layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::VariadicSplit>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 3);
    return std::make_shared<ngraph::op::VariadicSplit>(inputs[0], inputs[1], inputs[2]);
}

// Split layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Split>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    int num_splits = GetIntAttr(dn, "num_splits");
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::Split>(inputs[0], inputs[1], num_splits);
}

// Sigmoid layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Sigmoid>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Sigmoid>(inputs[0]);
}

// ELU layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Elu>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::Elu>(inputs[0], GetFloatAttr(dn, "alpha"));
}

// SpaceToDepth layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::SpaceToDepth>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::SpaceToDepth>(inputs[0], GetStrAttr(dn, "mode"), GetIntAttr(dn, "block_size", 1));
}

// DepthToSpace layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::DepthToSpace>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::DepthToSpace>(inputs[0], GetStrAttr(dn, "mode"), GetIntAttr(dn, "block_size", 1));
}

// SeLU layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v0::Selu>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 3);
    return std::make_shared<ngraph::op::v0::Selu>(inputs[0], inputs[1], inputs[2]);
}

// PReLU layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::PRelu>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::PRelu>(inputs[0], inputs[1]);
}

// Exp layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Exp>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Exp>(inputs[0]);
}

// ReLU layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Relu>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Relu>(inputs[0]);
}

// Negative layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Negative>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Negative>(inputs[0]);
}

// Range layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Range>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 3);
    return std::make_shared<ngraph::op::Range>(inputs[0], inputs[1], inputs[2]);
}

// Tanh layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Tanh>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Tanh>(inputs[0]);
}

// Result layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Result>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Result>(inputs[0]);
}

// Tile layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Tile>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::Tile>(inputs[0], inputs[1]);
}

// StridedSlice layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::StridedSlice>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {

    pugi::xml_node dn = node.child("data");

    std::vector<int64_t> begin_mask = getParameters<int64_t>(dn, "begin_mask");
    std::vector<int64_t> end_mask = getParameters<int64_t>(dn, "end_mask");
    std::vector<int64_t> new_axis = getParameters<int64_t>(dn, "new_axis_mask");
    std::vector<int64_t> shrink_axis = getParameters<int64_t>(dn, "shrink_axis_mask");
    std::vector<int64_t> ellipsis_mask = getParameters<int64_t>(dn, "ellipsis_mask");

    if (inputs.size() == 3) {
        return std::make_shared<ngraph::op::v1::StridedSlice>(inputs[0], inputs[1], inputs[2], begin_mask,
                                                              end_mask, new_axis, shrink_axis, ellipsis_mask);
    } else if (inputs.size() == 4) {
        return std::make_shared<ngraph::op::v1::StridedSlice>(inputs[0], inputs[1], inputs[2], inputs[3], begin_mask,
                                                              end_mask, new_axis, shrink_axis, ellipsis_mask);
    } else {
        THROW_IE_EXCEPTION << "Incorrect number of inputs " << inputs.size() << " for " << getType() << " layer with name: " << layerParsePrms.name;
    }
}

// Reshape layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Reshape>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);

    pugi::xml_node dn = node.child("data");
    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::v1::Reshape>(inputs[0], inputs[1], GetBoolAttr(dn, "special_zero"));
}

// Squeeze layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Squeeze>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::Squeeze>(inputs[0], inputs[1]);
}

// Unsqueeze layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Unsqueeze>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::Unsqueeze>(inputs[0], inputs[1]);
}

// Interpolate layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Interpolate>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);

    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    ngraph::op::InterpolateAttrs attrs;
    for (auto& axis : getParameters<int64_t>(dn, "axes")) {
        attrs.axes.insert(axis);
    }

    std::set<std::string> available_modes {"linear", "nearest", "cubic", "area"};
    attrs.mode = GetStrAttr(dn, "mode");
    if (!available_modes.count(attrs.mode)) {
        THROW_IE_EXCEPTION << "Interpolate mode: " << attrs.mode << " is unsupported!";
    }
    attrs.align_corners = GetIntAttr(dn, "align_corners", 1);
    attrs.antialias = GetIntAttr(dn, "antialias", 0);
    for (auto& pad : getParameters<int64_t>(dn, "pads_begin")) {
        attrs.pads_begin.push_back(pad);
    }
    for (auto& pad : getParameters<int64_t>(dn, "pads_end")) {
        attrs.pads_end.push_back(pad);
    }

    return std::make_shared<ngraph::op::Interpolate>(inputs[0], inputs[1], attrs);
}

// Abs layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Abs>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Abs>(inputs[0]);
}

// Add layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Add>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::Add>(inputs[0], inputs[1]);
}

// Minimum layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Minimum>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::Minimum>(inputs[0], inputs[1]);
}

// Maximum layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Maximum>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::Maximum>(inputs[0], inputs[1]);
}

// Divide layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Divide>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::Divide>(inputs[0], inputs[1]);
}

// Subtract layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Subtract>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::Subtract>(inputs[0], inputs[1]);
}

// Multiply layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Multiply>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::Multiply>(inputs[0], inputs[1]);
}

// Broadcast layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Broadcast>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    if (inputs.size() == 2) {
        return std::make_shared<ngraph::op::v1::Broadcast>(inputs[0], inputs[1]);
    } else if (layerParsePrms.inputPorts.size() == 3) {
        return std::make_shared<ngraph::op::v1::Broadcast>(inputs[0], inputs[1], inputs[2]);
    }
    THROW_IE_EXCEPTION << "Invalid number of inputs: " << layerParsePrms.inputPorts.size();
}

// Constant layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Constant>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 0);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    size_t offset = GetUInt64Attr(dn, "offset");
    size_t size = GetUInt64Attr(dn, "size");

    binStream.seekg(0, std::ios::end);
    std::streampos length = binStream.tellg();
    if (!length)
        THROW_IE_EXCEPTION << "Cannot read network! The model requires weights data! "
            << "Bin file cannot be found! Please specify the path to bin file.";
    if (length < offset + size)
        THROW_IE_EXCEPTION << "Cannot create " << getType() << " layer with name: " << layerParsePrms.name
                           << ". Layer has incorrect weights!";

    auto port = layerParsePrms.outputPorts[0];
    ngraph::Shape shape(port.dims);
    ngraph::element::Type el_type(port.precision);
    if (size < std::ceil(ngraph::shape_size(shape) * el_type.bitwidth() / 8.f))
        THROW_IE_EXCEPTION << "Cannot create Constant op " << layerParsePrms.name << " size attribute and shape size are inconsistent!";

    auto constant = std::make_shared<ngraph::op::Constant>(port.precision, shape);
    char* data = const_cast<char*>(reinterpret_cast<const char*>(constant->get_data_ptr()));
    binStream.seekg(offset, std::ios::beg);
    binStream.read(data, size);
    return constant;
}

// Power layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Power>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::Power>(inputs[0], inputs[1]);
}

// MatMul layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::MatMul>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    auto transpose_a = GetBoolAttr(dn, "transpose_a", false);
    auto transpose_b = GetBoolAttr(dn, "transpose_b", false);

    return std::make_shared<ngraph::op::MatMul>(inputs[0], inputs[1], transpose_a, transpose_b);
}

// Softmax layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Softmax>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::v1::Softmax>(inputs[0], GetUIntAttr(dn, "axis"));
}

// Sqrt layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Sqrt>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Sqrt>(inputs[0]);
}

// RegionYolo layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::RegionYolo>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto axis = GetIntAttr(dn, "axis");
    auto classes = GetUIntAttr(dn, "classes");
    auto coords = GetUIntAttr(dn, "coords");
    auto do_softmax = GetIntAttr(dn, "do_softmax");
    auto end_axis = GetIntAttr(dn, "end_axis");
    auto num = GetUIntAttr(dn, "num");
    auto mask = getParameters<int64_t>(dn, "mask", {});
    auto anchors = getParameters<float>(dn, "anchors", {});

    return std::make_shared<ngraph::op::RegionYolo>(inputs[0], coords, classes, num, do_softmax,
                                                    mask, axis, end_axis, anchors);
}

// ReorgYolo layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::ReorgYolo>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto stride = GetUIntAttr(dn, "stride");
    return std::make_shared<ngraph::op::ReorgYolo>(inputs[0], ngraph::Strides {stride});
}

// ReduceMin layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::ReduceMin>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::v1::ReduceMin>(inputs[0], inputs[1], GetBoolAttr(dn, "keep_dims", false));
}

// ReduceMax layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::ReduceMax>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::v1::ReduceMax>(inputs[0], inputs[1], GetBoolAttr(dn, "keep_dims", false));
}

// ReduceMean layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::ReduceMean>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::v1::ReduceMean>(inputs[0], inputs[1], GetBoolAttr(dn, "keep_dims", false));
}

// ReduceProd layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::ReduceProd>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::v1::ReduceProd>(inputs[0], inputs[1], GetBoolAttr(dn, "keep_dims", false));
}

// ReduceSum layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::ReduceSum>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::v1::ReduceSum>(inputs[0], inputs[1], GetBoolAttr(dn, "keep_dims", false));
}

// Transpose layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Transpose>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::Transpose>(inputs[0], inputs[1]);
}

// BinaryConvolution layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::BinaryConvolution>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    size_t group = GetUIntAttr(dn, "group", 1);
    if (group != 1) THROW_IE_EXCEPTION << "Cannot create grouped BinaryConvolution layer " << layerParsePrms.name;

    ngraph::op::PadType pad_type = ngraph::op::PadType::EXPLICIT;
    std::string auto_pad = GetStrAttr(dn, "auto_pad", "");
    if (auto_pad == "same_lower") {
        pad_type = ngraph::op::PadType::SAME_LOWER;
    } else if (auto_pad == "same_upper") {
        pad_type = ngraph::op::PadType::SAME_UPPER;
    } else if (auto_pad == "valid") {
        pad_type = ngraph::op::PadType::VALID;
    }

    auto strides = ngraph::Strides(getParameters<size_t>(dn, "strides"));
    auto dilations = ngraph::Strides(getParameters<size_t>(dn, "dilations"));
    auto pads_begin = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_begin"));
    auto pads_end = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_end"));
    auto mode = GetStrAttr(dn, "mode");
    auto pad_value = GetFloatAttr(dn, "pad_value");

    return std::make_shared<ngraph::op::v1::BinaryConvolution>(inputs[0], inputs[1], strides, pads_begin, pads_end,
                                                               dilations, mode, pad_value, pad_type);
}

// Convolution layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Convolution>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    ngraph::op::PadType pad_type = ngraph::op::PadType::EXPLICIT;
    std::string auto_pad = GetStrAttr(dn, "auto_pad", "");
    if (auto_pad == "same_lower") {
        pad_type = ngraph::op::PadType::SAME_LOWER;
    } else if (auto_pad == "same_upper") {
        pad_type = ngraph::op::PadType::SAME_UPPER;
    } else if (auto_pad == "valid") {
        pad_type = ngraph::op::PadType::VALID;
    }

    auto strides = ngraph::Strides(getParameters<size_t>(dn, "strides"));
    auto dilations = ngraph::Strides(getParameters<size_t>(dn, "dilations"));
    auto pads_begin = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_begin", {}));
    auto pads_end = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_end", {}));

    return std::make_shared<ngraph::op::v1::Convolution>(inputs[0], inputs[1], strides, pads_begin, pads_end,
                                                         dilations, pad_type);
}

// GroupConvolution layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::GroupConvolution>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    ngraph::op::PadType pad_type = ngraph::op::PadType::EXPLICIT;
    std::string auto_pad = GetStrAttr(dn, "auto_pad", "");
    if (auto_pad == "same_lower") {
        pad_type = ngraph::op::PadType::SAME_LOWER;
    } else if (auto_pad == "same_upper") {
        pad_type = ngraph::op::PadType::SAME_UPPER;
    } else if (auto_pad == "valid") {
        pad_type = ngraph::op::PadType::VALID;
    }

    auto strides = ngraph::Strides(getParameters<size_t>(dn, "strides"));
    auto dilations = ngraph::Strides(getParameters<size_t>(dn, "dilations"));
    auto pads_begin = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_begin", {}));
    auto pads_end = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_end", {}));

    return std::make_shared<ngraph::op::v1::GroupConvolution>(inputs[0], inputs[1], strides, pads_begin, pads_end,
                                                              dilations, pad_type);
}

// DeformableConvolution layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::DeformableConvolution>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 3);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    size_t group = GetUIntAttr(dn, "group");
    size_t deformable_group = GetUIntAttr(dn, "deformable_group");

    ngraph::op::PadType pad_type = ngraph::op::PadType::EXPLICIT;
    std::string auto_pad = GetStrAttr(dn, "auto_pad", "");
    if (auto_pad == "same_lower") {
        pad_type = ngraph::op::PadType::SAME_LOWER;
    } else if (auto_pad == "same_upper") {
        pad_type = ngraph::op::PadType::SAME_UPPER;
    } else if (auto_pad == "valid") {
        pad_type = ngraph::op::PadType::VALID;
    }

    auto strides = ngraph::Strides(getParameters<size_t>(dn, "strides"));
    auto dilations = ngraph::Strides(getParameters<size_t>(dn, "dilations"));
    auto pads_begin = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_begin"));
    auto pads_end = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_end"));

    return std::make_shared<ngraph::op::v1::DeformableConvolution>(inputs[0], inputs[1], inputs[2], strides, pads_begin,
                pads_end, dilations, pad_type, group, deformable_group);
}

// ConvolutionBackpropData layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::ConvolutionBackpropData>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    ngraph::op::PadType pad_type = ngraph::op::PadType::EXPLICIT;
    std::string auto_pad = GetStrAttr(dn, "auto_pad", "");
    if (auto_pad == "same_lower") {
        pad_type = ngraph::op::PadType::SAME_LOWER;
    } else if (auto_pad == "same_upper") {
        pad_type = ngraph::op::PadType::SAME_UPPER;
    } else if (auto_pad == "valid") {
        pad_type = ngraph::op::PadType::VALID;
    }

    auto strides = ngraph::Strides(getParameters<size_t>(dn, "strides"));
    auto dilations = ngraph::Strides(getParameters<size_t>(dn, "dilations"));
    auto pads_begin = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_begin", {}));
    auto pads_end = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_end", {}));
    auto output_padding = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "output_padding", {}));
    if (inputs.size() != 3 && inputs.size() != 2) {
        THROW_IE_EXCEPTION << layerParsePrms.type << " layer " << layerParsePrms.name << " has incorrect number of input ports!";
    }

    if (inputs.size() == 3) {
        return std::make_shared<ngraph::op::v1::ConvolutionBackpropData>(inputs[0], inputs[1], inputs[2], strides, pads_begin, pads_end,
                                                                         dilations, pad_type, output_padding);
    } else {
        return std::make_shared<ngraph::op::v1::ConvolutionBackpropData>(inputs[0], inputs[1], strides, pads_begin, pads_end,
                                                                         dilations, pad_type, output_padding);
    }
}

// GroupConvolutionBackpropData layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::GroupConvolutionBackpropData>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    ngraph::op::PadType pad_type = ngraph::op::PadType::EXPLICIT;
    std::string auto_pad = GetStrAttr(dn, "auto_pad", "");
    if (auto_pad == "same_lower") {
        pad_type = ngraph::op::PadType::SAME_LOWER;
    } else if (auto_pad == "same_upper") {
        pad_type = ngraph::op::PadType::SAME_UPPER;
    } else if (auto_pad == "valid") {
        pad_type = ngraph::op::PadType::VALID;
    }

    auto strides = ngraph::Strides(getParameters<size_t>(dn, "strides"));
    auto dilations = ngraph::Strides(getParameters<size_t>(dn, "dilations"));
    auto pads_begin = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_begin", {}));
    auto pads_end = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_end", {}));
    auto output_padding = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "output_padding", {}));

    if (inputs.size() != 3 && inputs.size() != 2) {
        THROW_IE_EXCEPTION << layerParsePrms.type << " layer " << layerParsePrms.name << " has incorrect number of input ports!";
    }

    if (inputs.size() == 3) {
        return std::make_shared<ngraph::op::v1::GroupConvolutionBackpropData>(inputs[0], inputs[1], inputs[2], strides, pads_begin, pads_end,
                                                                              dilations, pad_type, output_padding);
    } else {
        return std::make_shared<ngraph::op::v1::GroupConvolutionBackpropData>(inputs[0], inputs[1], strides, pads_begin, pads_end,
                                                                              dilations, pad_type, output_padding);
    }
}

// AvgPool layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::AvgPool>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto exclude_pad = GetStrAttr(dn, "exclude-pad") == "true";
    auto strides = ngraph::Strides(getParameters<size_t>(dn, "strides"));
    auto kernel = ngraph::Shape(getParameters<size_t>(dn, "kernel"));
    auto pads_begin = ngraph::Shape(getParameters<std::size_t>(dn, "pads_begin"));
    auto pads_end = ngraph::Shape(getParameters<std::size_t>(dn, "pads_end"));
    auto pad_type = ngraph::op::PadType::EXPLICIT;

    auto pad_type_str = GetStrAttr(dn, "auto_pad", "");
    if (pad_type_str == "same_lower") {
        pad_type = ngraph::op::PadType::SAME_LOWER;
    } else if (pad_type_str == "same_upper") {
        pad_type = ngraph::op::PadType::SAME_UPPER;
    } else if (pad_type_str == "valid") {
        pad_type = ngraph::op::PadType::VALID;
    }

    ngraph::op::RoundingType rounding_type;
    auto str_rounding_type = GetStrAttr(dn, "rounding_type", "floor");
    if (str_rounding_type == "floor") {
        rounding_type = ngraph::op::RoundingType::FLOOR;
    } else if (str_rounding_type == "ceil") {
        rounding_type = ngraph::op::RoundingType::CEIL;
    } else {
        THROW_IE_EXCEPTION << "Unsuppored rounding type: " << str_rounding_type;
    }

    return std::make_shared<ngraph::op::v1::AvgPool>(inputs[0], strides, pads_begin, pads_end, kernel, exclude_pad,
                                                     rounding_type, pad_type);
}

// MaxPool layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::MaxPool>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto strides = ngraph::Strides(getParameters<size_t>(dn, "strides"));
    auto kernel = ngraph::Shape(getParameters<size_t>(dn, "kernel"));
    auto pads_begin = ngraph::Shape(getParameters<std::size_t>(dn, "pads_begin"));
    auto pads_end = ngraph::Shape(getParameters<std::size_t>(dn, "pads_end"));
    auto pad_type = ngraph::op::PadType::EXPLICIT;

    auto pad_type_str = GetStrAttr(dn, "auto_pad", "");
    if (pad_type_str == "same_lower") {
        pad_type = ngraph::op::PadType::SAME_LOWER;
    } else if (pad_type_str == "same_upper") {
        pad_type = ngraph::op::PadType::SAME_UPPER;
    } else if (pad_type_str == "valid") {
        pad_type = ngraph::op::PadType::VALID;
    }

    ngraph::op::RoundingType rounding_type;
    auto str_rounding_type = GetStrAttr(dn, "rounding_type", "floor");
    if (str_rounding_type == "floor") {
        rounding_type = ngraph::op::RoundingType::FLOOR;
    } else if (str_rounding_type == "ceil") {
        rounding_type = ngraph::op::RoundingType::CEIL;
    } else {
        THROW_IE_EXCEPTION << "Unsuppored rounding type: " << str_rounding_type;
    }

    return std::make_shared<ngraph::op::v1::MaxPool>(inputs[0], strides, pads_begin, pads_end, kernel, rounding_type,
                                                     pad_type);
}

// ROIPooling layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::ROIPooling>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto pooled_h = GetUIntAttr(dn, "pooled_h");
    auto pooled_w = GetUIntAttr(dn, "pooled_w");
    auto spatial_scale = GetFloatAttr(dn, "spatial_scale");
    auto method = GetStrAttr(dn, "method", "max");
    return std::make_shared<ngraph::op::ROIPooling>(inputs[0], inputs[1],
                                                    ngraph::Shape {pooled_h, pooled_w}, spatial_scale, method);
}

// PSROIPooling layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::PSROIPooling>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto output_dim = GetIntAttr(dn, "output_dim");
    auto group_size = GetIntAttr(dn, "group_size", 1);
    auto spatial_bins_x = GetIntAttr(dn, "spatial_bins_x", 1);
    auto spatial_bins_y = GetIntAttr(dn, "spatial_bins_y", 1);
    auto spatial_scale = GetFloatAttr(dn, "spatial_scale");
    auto mode = GetStrAttr(dn, "mode", "average");

    return std::make_shared<ngraph::op::PSROIPooling>(inputs[0], inputs[1],
                                                      output_dim, group_size, spatial_scale, spatial_bins_x,
                                                      spatial_bins_y, mode);
}

// DeformablePSROIPooling layer

template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::DeformablePSROIPooling>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto output_dim = GetIntAttr(dn, "output_dim");
    auto group_size = GetIntAttr(dn, "group_size", 1);
    auto spatial_bins_x = GetIntAttr(dn, "spatial_bins_x", 1);
    auto spatial_bins_y = GetIntAttr(dn, "spatial_bins_y", 1);
    auto spatial_scale = GetFloatAttr(dn, "spatial_scale");
    auto mode = GetStrAttr(dn, "mode", "bilinear_deformable");
    auto trans_std = GetFloatAttr(dn, "trans_std", 1.0);
    auto part_size = GetIntAttr(dn, "part_size", 1);

    if (inputs.size() == 3) {
        return std::make_shared<ngraph::op::v1::DeformablePSROIPooling>(inputs[0],
                                                                        inputs[1],
                                                                        inputs[2], output_dim,
                                                                        spatial_scale, group_size, mode, spatial_bins_x,
                                                                        spatial_bins_y, trans_std, part_size);
    } else if (inputs.size() == 2) {
        return std::make_shared<ngraph::op::v1::DeformablePSROIPooling>(inputs[0],
                                                                        inputs[1], output_dim,
                                                                        spatial_scale, group_size, mode, spatial_bins_x,
                                                                        spatial_bins_y, trans_std, part_size);
    } else {
        THROW_IE_EXCEPTION << "Wrong number of inputs for " << getType() << " layer with name: " << layerParsePrms.name;
    }
}

// Concat layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Concat>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, -1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::Concat>(inputs, GetUIntAttr(dn, "axis"));
}

// Gather layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Gather>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 3);
    return std::make_shared<ngraph::op::v1::Gather>(inputs[0], inputs[1], inputs[2]);
}

// GatherTree layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::GatherTree>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 4);
    return std::make_shared<ngraph::op::v1::GatherTree>(inputs[0], inputs[1], inputs[2], inputs[3]);
}

// OneHot layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::OneHot>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 4);

    pugi::xml_node dn = node.child("data");
    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::v1::OneHot>(inputs[0], inputs[1], inputs[2], inputs[3], GetInt64Attr(dn, "axis"));
}

// NormalizeL2 layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::NormalizeL2>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    float eps = GetFloatAttr(dn, "eps");
    std::string eps_mode = GetStrAttr(dn, "eps_mode");
    ngraph::op::EpsMode em;
    if (eps_mode == "add") {
        em = ngraph::op::EpsMode::ADD;
    } else if (eps_mode == "max") {
        em = ngraph::op::EpsMode::MAX;
    } else {
        THROW_IE_EXCEPTION << "NormalizeL2 unsupported eps_mode: " << eps_mode;
    }

    return std::make_shared<ngraph::op::NormalizeL2>(inputs[0], inputs[1], eps, em);
}

// Erf layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Erf>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Erf>(inputs[0]);
}

// Sin layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Sin>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Sin>(inputs[0]);
}

// Sign layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Sign>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Sign>(inputs[0]);
}

// Sinh layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Sinh>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Sinh>(inputs[0]);
}

// Asin layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Asin>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Asin>(inputs[0]);
}

// Cos layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Cos>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Cos>(inputs[0]);
}

// Cosh layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Cosh>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Cosh>(inputs[0]);
}

// Acos layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Acos>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Acos>(inputs[0]);
}

// Tan layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Tan>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Tan>(inputs[0]);
}

// Atan layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Atan>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Atan>(inputs[0]);
}

// Floor layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Floor>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Floor>(inputs[0]);
}

// Ceiling layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Ceiling>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Ceiling>(inputs[0]);
}

// HardSigmoid layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::HardSigmoid>::createLayer(
    const ngraph::OutputVector & inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 3);
    return std::make_shared<ngraph::op::HardSigmoid>(inputs[0], inputs[1], inputs[2]);
}

// GRN layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::GRN>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::GRN>(inputs[0], GetFloatAttr(dn, "bias"));
}

// LogicalAnd layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::LogicalAnd>::createLayer(
    const ngraph::OutputVector & inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::LogicalAnd>(inputs[0], inputs[1]);
}

// LogicalOr layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::LogicalOr>::createLayer(
    const ngraph::OutputVector & inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::LogicalOr>(inputs[0], inputs[1]);
}

// LogicalXor layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::LogicalXor>::createLayer(
    const ngraph::OutputVector & inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::LogicalXor>(inputs[0], inputs[1]);
}

// LogicalNot layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::LogicalNot>::createLayer(
    const ngraph::OutputVector & inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::v1::LogicalNot>(inputs[0]);
}

// ReduceLogicalAnd layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::ReduceLogicalAnd>::createLayer(
    const ngraph::OutputVector & inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::v1::ReduceLogicalAnd>(inputs[0], inputs[1], GetBoolAttr(dn, "keep_dims"));
}

// ReduceLogicalOr layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::ReduceLogicalOr>::createLayer(
    const ngraph::OutputVector & inputs, const pugi::xml_node& node, std::istream& binStream,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::v1::ReduceLogicalOr>(inputs[0], inputs[1], GetBoolAttr(dn, "keep_dims"));
}

// NonMaxSuppression layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::NonMaxSuppression>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, std::istream& binStream,
        const GenericLayerParams& layerParsePrms) {
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto box_encoding_string = GetStrAttr(dn, "box_encoding");
    ngraph::op::v1::NonMaxSuppression::BoxEncodingType box_enc_type;
    if (box_encoding_string == "corner") {
        box_enc_type = ngraph::op::v1::NonMaxSuppression::BoxEncodingType::CORNER;
    } else if (box_encoding_string == "center") {
        box_enc_type = ngraph::op::v1::NonMaxSuppression::BoxEncodingType::CENTER;
    } else {
        THROW_IE_EXCEPTION << "Unsupported box encoding type " << box_encoding_string << " for " << getType() <<
        " layer with name: " << layerParsePrms.name;
    }

    auto sort_flag = GetBoolAttr(dn, "sort_result_descending");

    std::vector<ngraph::Output<ngraph::Node>> new_inputs{inputs.begin(), inputs.end()};
    if (new_inputs.size() == 2)
        new_inputs.push_back(ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0}));
    for (size_t ind = new_inputs.size(); ind < 5; ++ind)
        new_inputs.push_back(ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{}, {.0f}));
    return std::make_shared<ngraph::op::v1::NonMaxSuppression>(new_inputs[0], new_inputs[1], new_inputs[2], new_inputs[3], new_inputs[4],
            box_enc_type, sort_flag);
}

}  // namespace InferenceEngine

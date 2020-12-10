// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_ir_parser.hpp"
#include "ie_ir_itt.hpp"

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
#include <ngraph/ops.hpp>
#include <ngraph/opsets/opset.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/variant.hpp>

#include <cpp/ie_cnn_network.h>
#include "ie_blob_stream.hpp"
#include "caseless.hpp"
#include <ie_ngraph_utils.hpp>
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

std::shared_ptr<ICNNNetwork> IRParser::parse(const pugi::xml_node& root, const Blob::CPtr& weights) {
    return parser->parse(root, weights);
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

V10Parser::V10Parser(const std::vector<IExtensionPtr>& exts) : _exts(exts) {
    // Load default opsets
    opsets["opset1"] = ngraph::get_opset1();
    opsets["opset2"] = ngraph::get_opset2();
    opsets["opset3"] = ngraph::get_opset3();
    opsets["opset4"] = ngraph::get_opset4();
    opsets["opset5"] = ngraph::get_opset5();

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

std::shared_ptr<ICNNNetwork> V10Parser::parse(const pugi::xml_node& root, const Blob::CPtr& weights) {
    OV_ITT_TASK_CHAIN(taskChain, itt::domains::V10Reader_RT, "V10Parser", "Parse");

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

    OV_ITT_TASK_NEXT(taskChain, "ConstructNgraphNodes");

    ngraph::ParameterVector parameter_nodes;
    ngraph::ResultVector result_nodes;
    ngraph::NodeVector allNodes;
    ngraph::SinkVector assign_nodes;
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

        auto node = createNode(inputs, p.xml, weights, p.params);
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

    OV_ITT_TASK_NEXT(taskChain, "ConstructNgraphFunction");

    ::ngraph::op::GenericIE::DisableReshape noReshape(allNodes);
    auto function = std::make_shared<ngraph::Function>(result_nodes, assign_nodes, parameter_nodes, GetStrAttr(root, "name", ""));
    for (const auto& assign : assign_nodes) {
        assign->add_control_dependency(
            variable_id_to_read_value.at(std::dynamic_pointer_cast<ngraph::op::Assign>(assign)->get_variable_id()));
    }

    OV_ITT_TASK_NEXT(taskChain, "ConstructCNNNetwork");

    CNNNetwork net(function, _exts);

    parsePreProcess(net, root, weights);

    return net;
}

void V10Parser::parsePreProcess(CNNNetwork& network, const pugi::xml_node& root, const Blob::CPtr& weights) {
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

        if (inputs.empty()) THROW_IE_EXCEPTION << "network has no input";

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
            THROW_IE_EXCEPTION << "pre-process name ref '" << inputName << "' refers to un-existing input";
    }

    // dims vector without batch size
    SizeVector inputDims = preProcessInput->getTensorDesc().getDims();
    size_t noOfChannels = 0, width = 0, height = 0;

    if (inputDims.size() < 2) {
        THROW_IE_EXCEPTION << "network did not define input dimensions properly";
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
        THROW_IE_EXCEPTION << "mean blob defined without specifying precision.";

    ResponseDesc resp;
    InferenceEngine::PreProcessChannel::Ptr preProcessChannel;

    int lastChanNo = -1;
    std::unordered_set<int> idsForMeanImage;

    FOREACH_CHILD(chan, ppNode, "channel") {
        int chanNo = GetIntAttr(chan, "id", lastChanNo + 1);
        if (chanNo >= static_cast<int>(noOfChannels) || chanNo < 0) {
            THROW_IE_EXCEPTION << "Pre-process channel id invalid: " << chanNo;
        }
        lastChanNo = chanNo;
        preProcessChannel = pp[chanNo];

        auto meanNode = chan.child("mean");
        if (!meanNode.empty()) {
            if (!meanNode.attribute("size")) {
                THROW_IE_EXCEPTION << "mean should have the attribute: size";
            }
            if (meanNode.attribute("size")) {
                idsForMeanImage.insert(chanNo);
                size_t size = static_cast<size_t>(GetIntAttr(meanNode, "size"));
                size_t offset = static_cast<size_t>(GetIntAttr(meanNode, "offset"));
                if (width * height * meanSegmentPrecision.size() != size) {
                    THROW_IE_EXCEPTION << "mean blob size mismatch expected input, got: " << size
                                       << " extpecting " << width << " x " << height << " x "
                                       << meanSegmentPrecision.size();
                }
                preProcessChannel->meanData = make_blob_with_precision(TensorDesc(meanSegmentPrecision, {height, width}, Layout::HW));
                preProcessChannel->meanData->allocate();
                auto lockedMem = preProcessChannel->meanData->buffer();
                char* data = lockedMem.as<char *>();
                uint8_t* src_data = weights->cbuffer().as<uint8_t*>() + offset;
                memcpy(data, src_data, size);
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
        THROW_IE_EXCEPTION << "mean is not provided for all channels\n"
                              "Provided mean image for: "
                           << validMeanImageIds;
    }
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
                                                    const pugi::xml_node& node, const Blob::CPtr& weights,
                                                    const GenericLayerParams& params) {
    static std::vector<std::shared_ptr<LayerBaseCreator>> creators = {
        std::make_shared<LayerCreator<ngraph::op::v1::AvgPool>>("AvgPool"),
        std::make_shared<LayerCreator<ngraph::op::CTCGreedyDecoder>>("CTCGreedyDecoder"),
        std::make_shared<LayerCreator<ngraph::op::v1::DeformableConvolution>>("DeformableConvolution"),
        std::make_shared<LayerCreator<ngraph::op::v1::DeformablePSROIPooling>>("DeformablePSROIPooling"),
        std::make_shared<LayerCreator<ngraph::op::v1::Broadcast>>("Broadcast"),
        std::make_shared<LayerCreator<ngraph::op::v1::StridedSlice>>("StridedSlice"),
        std::make_shared<LayerCreator<ngraph::op::v1::GreaterEqual>>("GreaterEqual"),
        std::make_shared<LayerCreator<ngraph::op::v1::GroupConvolution>>("GroupConvolution"),
        std::make_shared<LayerCreator<ngraph::op::v1::ConvolutionBackpropData>>("ConvolutionBackpropData"),
        std::make_shared<LayerCreator<ngraph::op::v1::GroupConvolutionBackpropData>>("GroupConvolutionBackpropData"),
        std::make_shared<LayerCreator<ngraph::op::v1::BinaryConvolution>>("BinaryConvolution"),
        std::make_shared<LayerCreator<ngraph::op::SquaredDifference>>("SquaredDifference"),
        std::make_shared<LayerCreator<ngraph::op::v1::LessEqual>>("LessEqual"),
        std::make_shared<LayerCreator<ngraph::op::v1::Equal>>("Equal"),
        std::make_shared<LayerCreator<ngraph::op::v0::LSTMCell>>("LSTMCell"),
        std::make_shared<LayerCreator<ngraph::op::v1::MaxPool>>("MaxPool"),
        std::make_shared<LayerCreator<ngraph::op::v1::NonMaxSuppression>>("NonMaxSuppression"),
        std::make_shared<LayerCreator<ngraph::op::ReorgYolo>>("ReorgYolo"),
        std::make_shared<LayerCreator<ngraph::op::RegionYolo>>("RegionYolo"),
        std::make_shared<LayerCreator<ngraph::op::Result>>("Result"),
        std::make_shared<LayerCreator<ngraph::op::PSROIPooling>>("PSROIPooling"),
        std::make_shared<LayerCreator<ngraph::op::VariadicSplit>>("VariadicSplit"),
        std::make_shared<LayerCreator<ngraph::op::TensorIterator>>("TensorIterator"),
        std::make_shared<LayerCreator<ngraph::opset5::Loop>>("Loop"),
        std::make_shared<LayerCreator<ngraph::op::v1::LogicalAnd>>("LogicalAnd"),
        std::make_shared<LayerCreator<ngraph::op::v1::LogicalOr>>("LogicalOr"),
        std::make_shared<LayerCreator<ngraph::op::v1::LogicalXor>>("LogicalXor"),
        std::make_shared<LayerCreator<ngraph::op::v1::LogicalNot>>("LogicalNot"),
    };

    // Check that operation in default opsets
    auto isDefaultOpSet = [](const std::string& version) -> bool {
        for (size_t i = 1; i <= 5; i++) {
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
        if (ngraph::element::Type_t::undefined == inputs[i].get_element_type())
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
                    ngraphNode = creator->createLayer(inputs, node, weights, params);
                break;
            }
        }
    }

    // Try to create operation from loaded opsets
    if (!ngraphNode && opsets.count(params.version)) {
        auto opset = opsets.at(params.version);
        std::string type = params.type;

        if (type == "Const") {
            type = "Constant";
        }

        if (params.version == "opset1") {
            // MVN and ROIPooling were missing in opset1
            if (type == "MVN" || type == "ROIPooling") {
                opset = opsets.at("opset2");
            }
        }

        if (!opset.contains_type_insensitive(type)) {
            THROW_IE_EXCEPTION << "Opset " << params.version << " doesn't contain the operation with type: " << type;
        }

        ngraphNode = std::shared_ptr<ngraph::Node>(opset.create_insensitive(type));
        ngraphNode->set_friendly_name(params.name);
        ngraphNode->set_arguments(inputs);
        XmlDeserializer visitor(node, weights);
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
            size_t length = weights->byteSize();

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
                if (static_cast<uint64_t>(length) < offset + size)
                    THROW_IE_EXCEPTION << "Cannot create " << params.type << " layer with name: " << params.name
                                       << ". Layer has incorrect weights!";
                uint8_t* data = weights->cbuffer().as<uint8_t*>() + offset;
                Blob::Ptr wBlob = make_shared_blob<uint8_t>({Precision::U8, { size / precision.size() }, C }, data);

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
        THROW_IE_EXCEPTION << "Cannot create " << params.type << " layer " << params.name << " id:" << params.layerId
            << " from unsupported opset: " << params.version;
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


// SubGraph layer
std::shared_ptr<ngraph::Node>
V10Parser::LayerBaseCreator::fillSubGraphLayer(const ngraph::OutputVector &inputs, const pugi::xml_node &node,
                                               const Blob::CPtr& weights,
                                               const V10Parser::GenericLayerParams &layerParsePrms,
                                               std::shared_ptr<ngraph::op::util::SubGraphOp> subgraph_op) {
    subgraph_op->set_friendly_name(GetStrAttr(node, "name"));
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

    // Create ngraph::Function and set it as body of TensorIterator layer
    IRParser parser(10);
    auto ngraph_function = parser.parse(node.child("body"), weights)->getFunction();
    auto parameter_nodes = ngraph_function->get_parameters();
    auto result_nodes = ngraph_function->get_results();
    // Disabled reshape for generic operations in the TI body
    ::ngraph::op::GenericIE::DisableReshape noReshape(ngraph_function);
    auto body = std::make_shared<ngraph::Function>(result_nodes, parameter_nodes);
    subgraph_op->set_function(body);

    // Parse PortMap: inputs
    std::map<uint64_t, pugi::xml_node> input_map;
    FOREACH_CHILD(_input, node.child("port_map"), "input") {
        int64_t ext_port_id = GetInt64Attr(_input, "external_port_id");
        input_map[ext_port_id] = _input;
    }

    bool is_sliced_input_exists = false;
    for (const auto& input : input_map) {
        auto &_input = input.second;
        auto axis_attr = _input.attribute("axis");
        auto purpose = GetStrAttr(_input, "purpose", "");
        int64_t ti_input_index = GetInt64Attr(_input, "external_port_id");
        size_t body_parameter_index = GetUIntAttr(_input, "internal_layer_id");

        auto body_param = std::find_if(parameter_nodes.begin(), parameter_nodes.end(),
                                       [&](const std::shared_ptr<ngraph::op::Parameter>& param) {
                                           return param->get_friendly_name() == layer_idx_to_name[body_parameter_index];
                                       });

        if (body_param == parameter_nodes.end()) {
            THROW_IE_EXCEPTION << "PortMap input parsing error. Body parameter with id = " << body_parameter_index
                               << " not found.";
        }

        if (ti_input_index >=  static_cast<int64_t>(inputs.size()))
            THROW_IE_EXCEPTION << "TensorIterator " << layerParsePrms.name << " has incorrect number of inputs!";

        // if axis is set, then slicing is enabled. Create ngraph::TensorIterator::SlicedInput.
        if (!axis_attr.empty()) {
            size_t axis = GetUIntAttr(_input, "axis");
            int64_t start = GetInt64Attr(_input, "start", 0);
            int64_t stride = GetInt64Attr(_input, "stride", 1);
            int64_t end = GetInt64Attr(_input, "end", -1);
            int64_t part_size = GetInt64Attr(_input, "part_size", 1);
            subgraph_op->set_sliced_input(*body_param, inputs.at(ti_input_index), start, stride, part_size, end, axis);
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

                    subgraph_op->set_merged_input(*body_param, inputs.at(ti_input_index), *body_result);
                    is_back_edge_exist = true;
                    break;
                }
            }

            // ti_input_index = -1 means that Parameter of the body is not connected to inputs of TensorIterator
            // and is used only for internal needs.
            if (!is_back_edge_exist && ti_input_index >= 0) {
                subgraph_op->set_invariant_input(*body_param, inputs.at(ti_input_index));
            }

            if (purpose == "current_iteration") {
                auto loop = std::dynamic_pointer_cast<ngraph::opset5::Loop>(subgraph_op);
                if (!loop)
                    THROW_IE_EXCEPTION << "PortMap output parsing error. Purpose attribute is available only for Loop operation.";
                loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{ngraph_function->get_parameter_index(*body_param),
                                                                                    -1});
            }
        }
    }

    // Parse PortMap: outputs
    std::map<int64_t, pugi::xml_node> output_map;
    FOREACH_CHILD(_output, node.child("port_map"), "output") {
        int64_t ext_port_id = GetInt64Attr(_output, "external_port_id");
        output_map[ext_port_id] = _output;
    }

    int i = 0;
    for (const auto& output : output_map) {
        auto& _output = output.second;
        auto axis_attr = _output.attribute("axis");
        auto purpose = GetStrAttr(_output, "purpose", "");
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
            int64_t axis = GetInt64Attr(_output, "axis");
            int64_t start = GetInt64Attr(_output, "start", 0);
            int64_t stride = GetInt64Attr(_output, "stride", 1);
            int64_t end = GetInt64Attr(_output, "end", -1);
            int64_t part_size = GetInt64Attr(_output, "part_size", 1);
            subgraph_op->get_concatenated_slices(*body_result, start, stride, part_size, end, axis);

            if (!is_sliced_input_exists) {
                if (auto ti = std::dynamic_pointer_cast<ngraph::op::TensorIterator>(subgraph_op))
                    // for Loop op we just skip this call
                    if (ti)
                        ti->set_num_iterations((std::abs(end - start)) / part_size);
            }
        } else if (purpose == "execution_condition") {
            auto loop = std::dynamic_pointer_cast<ngraph::opset5::Loop>(subgraph_op);
            if (!loop)
                THROW_IE_EXCEPTION << "PortMap output parsing error. Purpose attribute is available only for Loop operation.";
            loop->set_special_body_ports(ngraph::opset5::Loop::SpecialBodyPorts{loop->get_special_body_ports().current_iteration_input_idx,
                                                                                ngraph_function->get_result_index(*body_result)});
            // if external_port_id < 0,
            // it means that this body result isn't connected to the Loop output and is used only for internal needs.
            if (output.first >= 0) {
                subgraph_op->get_iter_value(*body_result, -1);
            }
        } else {
            // otherwise create ngraph::TensorIterator::BodyOutput. -1 means last iteration.
            subgraph_op->get_iter_value(*body_result, -1);
        }
    }

    subgraph_op->validate_and_infer_types();
    return subgraph_op;
}


// TensorIterator layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::TensorIterator>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    auto ti = std::make_shared<ngraph::op::TensorIterator>();
    return fillSubGraphLayer(inputs, node, weights, layerParsePrms, ti);
    }

// Loop layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::opset5::Loop>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    auto loop = std::make_shared<ngraph::opset5::Loop>(inputs[0], inputs[1]);
    return fillSubGraphLayer(inputs, node, weights, layerParsePrms, loop);
}

// LSTMCell layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v0::LSTMCell>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 6);
    pugi::xml_node dn = node.child("data");
    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    std::vector<std::string> activations = getParameters<std::string>(dn, "activations", {"sigmoid", "tanh", "tanh"});
    std::vector<float> activations_alpha = getParameters<float>(dn, "activations_alpha", {});
    std::vector<float> activations_beta = getParameters<float>(dn, "activations_beta", {});
    float clip = GetFloatAttr(dn, "clip", 0.f);
    return std::make_shared<ngraph::op::v0::LSTMCell>(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                                                  GetUInt64Attr(dn, "hidden_size"), ngraph::op::LSTMWeightsFormat::IFCO,
                                                  activations, activations_alpha, activations_beta, clip);
}

// CTCGreedyDecoder layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::CTCGreedyDecoder>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    pugi::xml_node dn = node.child("data");
    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::CTCGreedyDecoder>(inputs[0], inputs[1],
                                                          GetBoolAttr(dn, "ctc_merge_repeated", true));
}

// SquaredDifference layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::SquaredDifference>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::SquaredDifference>(inputs[0], inputs[1]);
}

// GreaterEqual layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::GreaterEqual>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::GreaterEqual>(inputs[0], inputs[1]);
}

// LessEqual layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::LessEqual>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::LessEqual>(inputs[0], inputs[1]);
}

// Equal layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Equal>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::Equal>(inputs[0], inputs[1]);
}

// VariadicSplit layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::VariadicSplit>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 3);
    return std::make_shared<ngraph::op::VariadicSplit>(inputs[0], inputs[1], inputs[2]);
}

// DepthToSpace layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::DepthToSpace>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::DepthToSpace>(inputs[0], GetStrAttr(dn, "mode"), GetIntAttr(dn, "block_size", 1));
}

// Result layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Result>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::Result>(inputs[0]);
}

// StridedSlice layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::StridedSlice>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
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

// Broadcast layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::Broadcast>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
    const GenericLayerParams& layerParsePrms) {
    if (inputs.size() == 2) {
        return std::make_shared<ngraph::op::v1::Broadcast>(inputs[0], inputs[1]);
    } else if (layerParsePrms.inputPorts.size() == 3) {
        return std::make_shared<ngraph::op::v1::Broadcast>(inputs[0], inputs[1], inputs[2]);
    }
    THROW_IE_EXCEPTION << "Invalid number of inputs: " << layerParsePrms.inputPorts.size();
}

// RegionYolo layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::RegionYolo>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
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
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto stride = GetUIntAttr(dn, "stride");
    return std::make_shared<ngraph::op::ReorgYolo>(inputs[0], ngraph::Strides {stride});
}

// BinaryConvolution layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::BinaryConvolution>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
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

// GroupConvolution layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::GroupConvolution>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
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
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
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
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
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
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
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
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
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
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
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

// PSROIPooling layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::PSROIPooling>::createLayer(
    const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
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
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
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

// LogicalAnd layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::LogicalAnd>::createLayer(
    const ngraph::OutputVector & inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::LogicalAnd>(inputs[0], inputs[1]);
}

// LogicalOr layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::LogicalOr>::createLayer(
    const ngraph::OutputVector & inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::LogicalOr>(inputs[0], inputs[1]);
}

// LogicalXor layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::LogicalXor>::createLayer(
    const ngraph::OutputVector & inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 2);
    return std::make_shared<ngraph::op::v1::LogicalXor>(inputs[0], inputs[1]);
}

// LogicalNot layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::LogicalNot>::createLayer(
    const ngraph::OutputVector & inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
    const GenericLayerParams& layerParsePrms) {
    checkParameters(inputs, layerParsePrms, 1);
    return std::make_shared<ngraph::op::v1::LogicalNot>(inputs[0]);
}

// NonMaxSuppression layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::v1::NonMaxSuppression>::createLayer(
        const ngraph::OutputVector& inputs, const pugi::xml_node& node, const Blob::CPtr& weights,
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

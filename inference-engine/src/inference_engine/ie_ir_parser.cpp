// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_ngraph_utils.hpp"
#include "ie_ir_parser.hpp"
#include <sstream>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <set>

#include <ngraph/axis_vector.hpp>
#include <ngraph/coordinate_diff.hpp>
#include <ngraph/descriptor/input.hpp>
#include <ngraph/op/add.hpp>
#include <ngraph/op/avg_pool.hpp>
#include <ngraph/op/concat.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/convolution.hpp>
#include <ngraph/op/divide.hpp>
#include <ngraph/op/dot.hpp>
#include <ngraph/op/exp.hpp>
#include <ngraph/op/experimental/dyn_broadcast.hpp>
#include <ngraph/op/experimental/dyn_reshape.hpp>
#include <ngraph/op/experimental/dyn_slice.hpp>
#include <ngraph/op/experimental/layers/detection_output.hpp>
#include <ngraph/op/experimental/layers/interpolate.hpp>
#include <ngraph/op/experimental/layers/prior_box.hpp>
#include <ngraph/op/experimental/layers/prior_box_clustered.hpp>
#include <ngraph/op/experimental/layers/proposal.hpp>
#include <ngraph/op/experimental/shape_of.hpp>
#include <ngraph/op/experimental/tile.hpp>
#include <ngraph/op/experimental/transpose.hpp>
#include <ngraph/op/fused/clamp.hpp>
#include <ngraph/op/fused/elu.hpp>
#include <ngraph/op/fused/group_conv.hpp>
#include <ngraph/op/fused/leaky_relu.hpp>
#include <ngraph/op/fused/mvn.hpp>
#include <ngraph/op/fused/prelu.hpp>
#include <ngraph/op/fused/split.hpp>
#include <ngraph/op/fused/squeeze.hpp>
#include <ngraph/op/fused/unsqueeze.hpp>
#include <ngraph/op/lrn.hpp>
#include <ngraph/op/max_pool.hpp>
#include <ngraph/op/maximum.hpp>
#include <ngraph/op/multiply.hpp>
#include <ngraph/op/power.hpp>
#include <ngraph/op/relu.hpp>
#include <ngraph/op/result.hpp>
#include <ngraph/op/sigmoid.hpp>
#include <ngraph/op/softmax.hpp>
#include <ngraph/op/tanh.hpp>
#include <ngraph/op/topk.hpp>
#include <ngraph/partial_shape.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/strides.hpp>

#include "ngraph_ops/dummy.hpp"

using namespace InferenceEngine;
using namespace XMLParseUtils;

IRParser::IRParser(size_t version) {
    switch (version) {
    case 10:
        parser = std::make_shared<V10Parser>();
        break;
    default:
        THROW_IE_EXCEPTION << "Unsupported IR version: " << version;
    }
}

std::shared_ptr<ngraph::Function> IRParser::parse(const pugi::xml_node &root, const Blob::CPtr& weights) {
    return parser->parse(root, weights);
}

std::shared_ptr<ngraph::Function> V10Parser::parse(const pugi::xml_node& root, const Blob::CPtr& weights) {
    auto dumpVec = [](const std::vector<size_t>& vec) -> std::string {
        if (vec.empty()) return "[]";
        std::stringstream oss;
        oss << "[" << vec[0];
        for (size_t i = 1; i < vec.size(); i++) oss << "," << vec[i];
        oss << "]";
        return oss.str();
    };

    auto name = GetStrAttr(root, "name", "");

    std::vector<std::shared_ptr<ngraph::Node>> nodes;
    std::vector<GenericLayerParams> nodeParams;
    std::map<size_t, size_t> id2node;
    auto allLayersNode = root.child("layers");
    for (auto node = allLayersNode.child("layer"); !node.empty(); node = node.next_sibling("layer")) {
        auto params = parseGenericParams(node);
        // FIXME: WA for Deconvolution
        if (params.type == "Deconvolution") {
            auto port = params.inputPorts[0];
            params.inputPorts[0] = params.inputPorts[1];
            params.inputPorts[1] = port;
        }
        nodeParams.emplace_back(params);
        id2node[params.layerId] = nodes.size();
        nodes.emplace_back(createNode(node, weights, params));
    }

    pugi::xml_node edges = root.child("edges");
    FOREACH_CHILD(_ec, edges, "edge") {
        size_t fromLayer = GetUIntAttr(_ec, "from-layer");
        size_t fromPort = GetUIntAttr(_ec, "from-port");
        size_t toLayer = GetUIntAttr(_ec, "to-layer");
        size_t toPort = GetUIntAttr(_ec, "to-port");
        size_t fromPortIdx(0), toPortIdx(0);
        bool portFound(false);
        auto outPorts = nodeParams[id2node[fromLayer]].outputPorts;
        for (size_t idx = 0; idx < outPorts.size() && !portFound; idx++) {
            if (outPorts[idx].portId == fromPort) {
                portFound = true;
                fromPortIdx = idx;
            }
        }
        if (!portFound) {
            THROW_IE_EXCEPTION << "Output port " << fromPort << " was not found for layer "
                << nodeParams[id2node[fromLayer]].name << " id: " << nodeParams[id2node[fromLayer]].layerId;
        }
        portFound = false;

        auto inPorts = nodeParams[id2node[toLayer]].inputPorts;
        for (size_t idx = 0; idx < inPorts.size() && !portFound; idx++) {
            if (inPorts[idx].portId == toPort) {
                portFound = true;
                toPortIdx = idx;
            }
        }
        if (!portFound) {
            THROW_IE_EXCEPTION << "Input port " << toPort << " was not found for layer "
                << nodeParams[id2node[toLayer]].name << " id: " << nodeParams[id2node[toLayer]].layerId;
        }

        if (inPorts[toPortIdx].dims != outPorts[fromPortIdx].dims)
            THROW_IE_EXCEPTION << "Cannot create connection from "
                << nodeParams[id2node[fromLayer]].type << " layer "
                << nodeParams[id2node[fromLayer]].name << " port " << fromPortIdx << " to "
                << nodeParams[id2node[toLayer]].type << " layer "
                << nodeParams[id2node[toLayer]].name << " port " << toPortIdx << ". "
                               << "Mismatch dimensions! dims input: " << dumpVec(inPorts[toPortIdx].dims)
                               << " dims output: " << dumpVec(outPorts[fromPortIdx].dims);

        // if (inPorts[toPortIdx].precision != outPorts[fromPortIdx].precision)
        //     THROW_IE_EXCEPTION << "Cannot create connection from "
        //         << nodeParams[id2node[fromLayer]].type << " layer "
        //         << nodeParams[id2node[fromLayer]].name << " port " << fromPortIdx << " to "
        //         << nodeParams[id2node[toLayer]].type << " layer "
        //         << nodeParams[id2node[toLayer]].name << " port " << toPortIdx << ". "
        //                        << "Ports have different precisions!";
        connectNodes(nodes[id2node[fromLayer]], fromPortIdx, nodes[id2node[toLayer]], toPortIdx);
    }

    ngraph::ParameterVector params;
    ngraph::ResultVector results;

    for (auto& node : nodes) {
        auto parameter = std::dynamic_pointer_cast<ngraph::op::Parameter>(node);
        if (parameter) {
            params.emplace_back(parameter);
        }
        auto result = std::dynamic_pointer_cast<ngraph::op::Result>(node);
        if (result) {
            results.emplace_back(result);
        }
    }

    return std::make_shared<ngraph::Function>(results, params, name);
}

V10Parser::GenericLayerParams V10Parser::parseGenericParams(const pugi::xml_node& node) {
    const auto parsePort = [](const pugi::xml_node& parentNode, const GenericLayerParams& params) -> GenericLayerParams::LayerPortData {
        GenericLayerParams::LayerPortData port;

        port.portId = GetIntAttr(parentNode, "id");

        for (auto node = parentNode.child("dim"); !node.empty(); node = node.next_sibling("dim")) {
            size_t dim = 0;
            const pugi::char_t* dimVal = node.child_value();
            std::stringstream ss(dimVal);
            if (!(ss >> dim) || dim == 0) {
                THROW_IE_EXCEPTION << "dimension (" << dimVal << ") in node " << node.name() << " must be a positive integer: at offset "
                    << node.offset_debug();
            }
            port.dims.push_back(dim);
        }

        const std::string &preStr = GetStrAttr(parentNode, "precision", "");
        if (!preStr.empty())
            port.precision = Precision::FromStr(preStr);
        else
            port.precision = params.precision;
        return port;
    };
    GenericLayerParams params;

    params.layerId = GetIntAttr(node, "id");

    params.type = XMLParseUtils::GetStrAttr(node, "type");

    params.name = GetStrAttr(node, "name");
    const std::string& preStr = GetStrAttr(node, "precision", "");
    if (!preStr.empty())
        params.precision = Precision::FromStr(preStr);

    if (params.precision == Precision::MIXED) {
        THROW_IE_EXCEPTION << "Layer precision must not be MIXED, at layer name: " << params.name << ", offset: "
                           << node.offset_debug();
    }

    auto outNode = node.child("output");
    if (!outNode.empty()) {
        FOREACH_CHILD(_cn, outNode, "port") {
            params.outputPorts.emplace_back(parsePort(_cn, params));
        }
    }
    auto inpNode = node.child("input");
    if (!inpNode.empty()) {
        FOREACH_CHILD(_cn, inpNode, "port") {
            params.inputPorts.emplace_back(parsePort(_cn, params));
        }
    }
    return params;
}

bool V10Parser::LayerBaseCreator::shouldCreate(const std::string& nodeType) const {
    InferenceEngine::details::CaselessEq<std::string> comparator;
    return comparator(nodeType, type);
}

std::shared_ptr<ngraph::op::Parameter> V10Parser::LayerBaseCreator::createParameter(const GenericLayerParams::LayerPortData& port) {
    ngraph::PartialShape shape(port.dims);
    return std::make_shared<ngraph::op::Parameter>(details::ngraph::convertPrecision(port.precision), shape);
}

template <class T>
std::shared_ptr<ngraph::op::Constant> V10Parser::LayerBaseCreator::createConstant(const GenericLayerParams::LayerPortData& port, std::vector<T> value) {
    ngraph::Shape shape(port.dims);
    return std::make_shared<ngraph::op::Constant>(details::ngraph::convertPrecision(port.precision), shape, value);
}

std::shared_ptr<ngraph::Node> V10Parser::LayerBaseCreator::createOptionalParameter(const GenericLayerParams::LayerPortData& port) {
    return std::make_shared<ngraph::op::Dummy>();
}

void V10Parser::connectNodes(std::shared_ptr<ngraph::Node>& parent, size_t outPort, std::shared_ptr<ngraph::Node>& child, size_t inPort) {
    if (child->get_input_size() <= inPort || parent->get_output_size() <= outPort)
        THROW_IE_EXCEPTION << "Cannot connect " << parent->get_friendly_name() << " port " << outPort
            << " to " << child->get_friendly_name() << " port " << inPort << ". Incorrect parameters!";
    auto& input = child->get_inputs().at(inPort);
    input.replace_output(parent, outPort);
}

std::shared_ptr<ngraph::Node> V10Parser::createNode(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& params) {
    static std::vector<std::shared_ptr<LayerBaseCreator>> creators = {
        std::make_shared<LayerCreator<ngraph::op::Add>>("Add"),
        std::make_shared<LayerCreator<ngraph::op::AvgPool>>("AvgPool"),
        std::make_shared<LayerCreator<ngraph::op::Clamp>>("Clamp"),
        std::make_shared<LayerCreator<ngraph::op::Concat>>("Concat"),
        std::make_shared<LayerCreator<ngraph::op::Constant>>("Const"),
        std::make_shared<LayerCreator<ngraph::op::ConvolutionBackpropData>>("Deconvolution"),
        std::make_shared<LayerCreator<ngraph::op::DetectionOutput>>("DetectionOutput"),
        std::make_shared<LayerCreator<ngraph::op::Divide>>("Divide"),
        std::make_shared<LayerCreator<ngraph::op::Dot>>("MatMul"),
        std::make_shared<LayerCreator<ngraph::op::DynBroadcast>>("Broadcast"),
        std::make_shared<LayerCreator<ngraph::op::DynReshape>>("Reshape"),
        std::make_shared<LayerCreator<ngraph::op::DynSlice>>("StridedSlice"),
        std::make_shared<LayerCreator<ngraph::op::Elu>>("ELU"),
        std::make_shared<LayerCreator<ngraph::op::Exp>>("Exp"),
        std::make_shared<LayerCreator<ngraph::op::GroupConvolution>>("Convolution"),
        std::make_shared<LayerCreator<ngraph::op::Interpolate>>("Interpolate"),
        std::make_shared<LayerCreator<ngraph::op::LRN>>("LRN"),
        std::make_shared<LayerCreator<ngraph::op::LeakyRelu>>("LeakyReLU"),
        std::make_shared<LayerCreator<ngraph::op::MVN>>("MVN"),
        std::make_shared<LayerCreator<ngraph::op::MaxPool>>("MaxPool"),
        std::make_shared<LayerCreator<ngraph::op::Maximum>>("Maximum"),
        std::make_shared<LayerCreator<ngraph::op::Multiply>>("Multiply"),
        std::make_shared<LayerCreator<ngraph::op::PRelu>>("PReLU"),
        std::make_shared<LayerCreator<ngraph::op::Parameter>>("Parameter"),
        std::make_shared<LayerCreator<ngraph::op::Power>>("Pow"),
        std::make_shared<LayerCreator<ngraph::op::PriorBox>>("PriorBox"),
        std::make_shared<LayerCreator<ngraph::op::PriorBoxClustered>>("PriorBoxClustered"),
        std::make_shared<LayerCreator<ngraph::op::Proposal>>("Proposal"),
        std::make_shared<LayerCreator<ngraph::op::Relu>>("ReLU"),
        std::make_shared<LayerCreator<ngraph::op::Result>>("Result"),
        std::make_shared<LayerCreator<ngraph::op::ShapeOf>>("ShapeOf"),
        std::make_shared<LayerCreator<ngraph::op::Sigmoid>>("Sigmoid"),
        std::make_shared<LayerCreator<ngraph::op::Softmax>>("Softmax"),
        std::make_shared<LayerCreator<ngraph::op::Split>>("Split"),
        std::make_shared<LayerCreator<ngraph::op::Squeeze>>("Squeeze"),
        std::make_shared<LayerCreator<ngraph::op::Tanh>>("TanH"),
        std::make_shared<LayerCreator<ngraph::op::Tile>>("Tile"),
        std::make_shared<LayerCreator<ngraph::op::TopK>>("TopK"),
        std::make_shared<LayerCreator<ngraph::op::Transpose>>("Transpose"),
        std::make_shared<LayerCreator<ngraph::op::Unsqueeze>>("Unsqueeze"),
    };
    std::shared_ptr<ngraph::Node> ngraphNode;
    for (const auto& creator : creators) {
        if (creator->shouldCreate(params.type)) {
            ngraphNode = creator->createLayer(node, weights, params);
            break;
        }
    }

    if (!ngraphNode) {
        THROW_IE_EXCEPTION << "Cannot create " << params.type << " layer " << params.name << " id:" << params.layerId;
    }

    ngraphNode->set_friendly_name(params.name);

    return ngraphNode;
}

namespace InferenceEngine {

// Parameter layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Parameter>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 0, 1);
    return createParameter(layerParsePrms.outputPorts[0]);
}

// DetectionOutput layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::DetectionOutput>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    ngraph::op::DetectionOutputAttrs attr;

    attr.num_classes = GetIntAttr(dn, "num_classes");
    attr.background_label_id = GetIntAttr(dn, "background_label_id", 0);
    attr.top_k = GetIntAttr(dn, "top_k");
    attr.variance_encoded_in_target = GetIntAttr(dn, "variance_encoded_in_target", 0) != 0;
    attr.keep_top_k = getParameters<int>(dn, "keep_top_k");
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
    attr.confidence_threshold = GetFloatAttr(dn, "confidence_threshold");


    if (layerParsePrms.inputPorts.size() > 3) {
        THROW_IE_EXCEPTION << "DetectionOutput has more than 3 inputs!";
    }

    return std::make_shared<ngraph::op::DetectionOutput>(createParameter(layerParsePrms.inputPorts[0]),
            createParameter(layerParsePrms.inputPorts[1]),
            createParameter(layerParsePrms.inputPorts[2]),
            createOptionalParameter(GenericLayerParams::LayerPortData()),
            createOptionalParameter(GenericLayerParams::LayerPortData()),
            attr);
}

// PriorBoxClustered layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::PriorBoxClustered>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    ngraph::op::PriorBoxClusteredAttrs attr;
    attr.widths = getParameters<float>(dn, "width");
    attr.heights = getParameters<float>(dn, "height");
    attr.variances = getParameters<float>(dn, "variance");
    attr.offset = GetFloatAttr(dn, "offset");
    float step = GetFloatAttr(dn, "step", 1);
    attr.step_heights = GetFloatAttr(dn, "step_h", step);
    attr.step_widths = GetFloatAttr(dn, "step_w", step);
    attr.clip = (GetIntAttr(dn, "clip") != 0);
    attr.num_priors = GetUIntAttr(dn, "num_priors", attr.widths.size());

    auto inputShapePort1 = layerParsePrms.inputPorts[0];
    inputShapePort1.precision = Precision::I64;

    auto inputShapePort2 = layerParsePrms.inputPorts[1];
    inputShapePort2.precision = Precision::I64;

    return std::make_shared<ngraph::op::PriorBoxClustered>(createParameter(inputShapePort1),
            createParameter(inputShapePort2), attr);
}

// Proposal layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Proposal>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 3, 1);
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
    attr.normalize = (GetIntAttr(dn, "normalize", 1) != 0);
    attr.box_size_scale = GetFloatAttr(dn, "box_size_scale", 1.0f);
    attr.box_coordinate_scale = GetFloatAttr(dn, "box_coordinate_scale", 1.0f);
    attr.framework = GetStrAttr(dn, "framework", "");


    auto inputShapePort3 = layerParsePrms.inputPorts[2];
    inputShapePort3.precision = Precision::I64;

    return std::make_shared<ngraph::op::Proposal>(createParameter(layerParsePrms.inputPorts[0]),
            createParameter(layerParsePrms.inputPorts[1]), createParameter(inputShapePort3), attr);
}

// PriorBox layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::PriorBox>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    ngraph::op::PriorBoxAttrs attr;
    attr.min_size = getParameters<float>(dn, "min_size");
    attr.max_size = getParameters<float>(dn, "max_size");
    attr.aspect_ratio = getParameters<float>(dn, "aspect_ratio");
    attr.variance = getParameters<float>(dn, "variance");
    attr.step = GetFloatAttr(dn, "step", 0);
    attr.offset = GetFloatAttr(dn, "offset");
    attr.clip = (GetIntAttr(dn, "clip") != 0);
    attr.flip = (GetIntAttr(dn, "flip") != 0);
    attr.scale_all_sizes = (GetIntAttr(dn, "scale_all_sizes", 1) != 0);

    auto inputShapePort1 = layerParsePrms.inputPorts[0];
    inputShapePort1.precision = Precision::I64;

    auto inputShapePort2 = layerParsePrms.inputPorts[1];
    inputShapePort2.precision = Precision::I64;

    return std::make_shared<ngraph::op::PriorBox>(createParameter(inputShapePort1),
            createParameter(inputShapePort2), attr);
}

// ShapeOf layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::ShapeOf>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    return std::make_shared<ngraph::op::ShapeOf>(createParameter(layerParsePrms.inputPorts[0]));
}

// TopK layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::TopK>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    size_t axis = GetUInt64Attr(dn, "axis");
    size_t maxMode = GetStrAttr(dn, "mode", "max") == "max";
    return nullptr;
    // return std::make_shared<ngraph::op::LRN>(createParameter(layerParsePrms.inputPorts[0]), alpha, beta, bias, size);
}

// MVN layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::MVN>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    double eps = GetFloatAttr(dn, "eps");
    bool across = GetUIntAttr(dn, "across_channels", 0) == 1;
    bool normalize_variance = GetUIntAttr(dn, "normalize_variance", 0) == 1;
    return std::make_shared<ngraph::op::MVN>(createParameter(layerParsePrms.inputPorts[0]), across, normalize_variance, eps);
}

// LRN layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::LRN>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    double alpha = GetFloatAttr(dn, "alpha");
    double beta = GetFloatAttr(dn, "beta");
    double bias = GetFloatAttr(dn, "bias");
    size_t size = GetUInt64Attr(dn, "local-size");
    return std::make_shared<ngraph::op::LRN>(createParameter(layerParsePrms.inputPorts[0]), alpha, beta, bias, size);
}

// Clamp layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Clamp>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    double maxVal = GetFloatAttr(dn, "max");
    double minVal = GetFloatAttr(dn, "min");
    return std::make_shared<ngraph::op::Clamp>(createParameter(layerParsePrms.inputPorts[0]), minVal, maxVal);
}

// Split layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Split>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    int axis = GetIntAttr(dn, "axis");
    std::vector<size_t> splits;
    for (const auto& outPort : layerParsePrms.outputPorts) {
        splits.emplace_back(outPort.dims[axis]);
    }
    checkParameters(layerParsePrms, 1, splits.size());
    return std::make_shared<ngraph::op::Split>(createParameter(layerParsePrms.inputPorts[0]), axis, splits);
}

// Sigmoid layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Sigmoid>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    return std::make_shared<ngraph::op::Sigmoid>(createParameter(layerParsePrms.inputPorts[0]));
}

// ELU layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Elu>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    float alpha = GetFloatAttr(dn, "alpha");
    std::vector<float> alphaVec;
    alphaVec.emplace_back(alpha);
    ngraph::Shape shape;
    auto alphaInput = std::make_shared<ngraph::op::Constant>(details::ngraph::convertPrecision(layerParsePrms.inputPorts[0].precision),
            shape, alphaVec.data());
    return std::make_shared<ngraph::op::Elu>(createParameter(layerParsePrms.inputPorts[0]), alphaInput);
}

// PReLU layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::PRelu>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::PRelu>(createParameter(layerParsePrms.inputPorts[0]),
            createParameter(layerParsePrms.inputPorts[1]));
}

// Leaky ReLU layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::LeakyRelu>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    float alpha = GetFloatAttr(dn, "negative_slope");
    std::vector<float> alphaVec;
    alphaVec.emplace_back(alpha);
    ngraph::Shape shape;
    auto alphaInput = std::make_shared<ngraph::op::Constant>(details::ngraph::convertPrecision(layerParsePrms.inputPorts[0].precision),
            shape, alphaVec.data());
    return std::make_shared<ngraph::op::LeakyRelu>(createParameter(layerParsePrms.inputPorts[0]), alphaInput);
}

// Exp layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Exp>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    return std::make_shared<ngraph::op::Exp>(createParameter(layerParsePrms.inputPorts[0]));
}

// ReLU layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Relu>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    return std::make_shared<ngraph::op::Relu>(createParameter(layerParsePrms.inputPorts[0]));
}

// Tanh layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Tanh>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    return std::make_shared<ngraph::op::Tanh>(createParameter(layerParsePrms.inputPorts[0]));
}

// Result layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Result>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 0);
    return std::make_shared<ngraph::op::Result>(createParameter(layerParsePrms.inputPorts[0]));
}

// Tile layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Tile>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    auto inputShapePort = layerParsePrms.inputPorts[1];
    inputShapePort.precision = Precision::I64;
    return std::make_shared<ngraph::op::Tile>(createParameter(layerParsePrms.inputPorts[0]),
            createParameter(inputShapePort));
}

// StridedSlice layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::DynSlice>::createLayer(const pugi::xml_node& node,
                                                                                        const Blob::CPtr& weights,
                                                                                        const GenericLayerParams& layerParsePrms) {
    // TODO: how to check params if the last input is optional
    checkParameters(layerParsePrms, 4, 1);
    auto inputShapePort1 = layerParsePrms.inputPorts[1];
    inputShapePort1.precision = Precision::I64;

    auto inputShapePort2 = layerParsePrms.inputPorts[2];
    inputShapePort2.precision = Precision::I64;

    auto inputShapePort3 = layerParsePrms.inputPorts[3];
    inputShapePort3.precision = Precision::I64;

    pugi::xml_node dn = node.child("data");

    ngraph::AxisSet lower_bounds_mask;
    ngraph::AxisSet upper_bounds_mask;
    ngraph::AxisSet new_axis;
    ngraph::AxisSet shrink_axis;
    ngraph::AxisSet ellipsis_mask;

    size_t axis = 0;
    for (auto & i : getParameters<int64_t >(dn, "begin_mask")) {
        if (i == 0) lower_bounds_mask.insert(axis);
        ++axis;
    }
    axis = 0;
    for (auto & i : getParameters<int64_t >(dn, "end_mask")) {
        if (i == 0) upper_bounds_mask.insert(axis);
        ++axis;
    }

    axis = 0;
    for (auto & i : getParameters<int64_t >(dn, "new_axis_mask")) {
        if (i != 0) new_axis.insert(axis);
        ++axis;
    }
    axis = 0;
    for (auto & i : getParameters<int64_t >(dn, "shrink_axis_mask")) {
        if (i != 0) shrink_axis.insert(axis);
        ++axis;
    }
    axis = 0;
    for (auto & i : getParameters<int64_t >(dn, "ellipsis_mask")) {
        if (i != 0) ellipsis_mask.insert(axis);
        ++axis;
    }

    return std::make_shared<ngraph::op::DynSlice>(createParameter(layerParsePrms.inputPorts[0]),
                                                  createParameter(inputShapePort1),
                                                  createParameter(inputShapePort2),
                                                  createParameter(inputShapePort3),
                                                  lower_bounds_mask,
                                                  upper_bounds_mask,
                                                  new_axis,
                                                  shrink_axis,
                                                  ellipsis_mask);
}

// Reshape layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::DynReshape>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    auto inputShapePort = layerParsePrms.inputPorts[1];
    inputShapePort.precision = Precision::I64;
    return std::make_shared<ngraph::op::DynReshape>(createParameter(layerParsePrms.inputPorts[0]),
            createParameter(inputShapePort), true);
}

// Squeeze layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Squeeze>::createLayer(const pugi::xml_node& node,
                                                                                          const Blob::CPtr& weights,
                                                                                          const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    auto inputShapePort = layerParsePrms.inputPorts[1];
    inputShapePort.precision = Precision::I64;
    int64_t axis = -1;
    for (size_t i = 0; i < layerParsePrms.inputPorts[0].dims.size(); ++i) {
        if (layerParsePrms.inputPorts[0].dims[i] == 1) axis = i;
    }
    if (axis == -1) {
        THROW_IE_EXCEPTION << "Can\'t find any dim that is equal to 1. Probably something wrong with Squeeze operation";
    }
    return std::make_shared<ngraph::op::Squeeze>(createParameter(layerParsePrms.inputPorts[0]),
                                                 createConstant<int64_t>(inputShapePort, {axis}));
}

// Unsqueeze layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Unsqueeze>::createLayer(const pugi::xml_node& node,
                                                                                          const Blob::CPtr& weights,
                                                                                          const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    auto inputShapePort = layerParsePrms.inputPorts[1];
    inputShapePort.precision = Precision::I64;
    return std::make_shared<ngraph::op::Unsqueeze>(createParameter(layerParsePrms.inputPorts[0]),
                                                   createConstant<int64_t>(inputShapePort, {0}));
}

// Interpolate layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Interpolate>::createLayer(const pugi::xml_node& node,
                                                                                           const Blob::CPtr& weights,
                                                                                           const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    auto inputShapePort = layerParsePrms.inputPorts[1];
    inputShapePort.precision = Precision::I64;

    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    ngraph::op::InterpolateAttrs attrs;
    for (auto & axis : getParameters<int64_t >(dn, "axes")) {
        attrs.axes.insert(axis);
    }

    std::set<std::string> available_modes {"linear", "nearest", "cubic", "area"};
    attrs.mode = GetStrAttr(dn, "mode");
    if (!available_modes.count(attrs.mode)) {
        THROW_IE_EXCEPTION << "Interpolate mode: " << attrs.mode << " is unsupported!";
    }
    attrs.align_corners = GetIntAttr(dn, "align_corners", 1);
    attrs.antialias = GetIntAttr(dn, "antialias", 0);
    for (auto & pad : getParameters<int64_t >(dn, "pads_begin")) {
        attrs.pads_begin.push_back(pad);
    }
    for (auto & pad : getParameters<int64_t >(dn, "pads_end")) {
        attrs.pads_end.push_back(pad);
    }

    return std::make_shared<ngraph::op::Interpolate>(createParameter(layerParsePrms.inputPorts[0]),
                                                     createParameter(inputShapePort),
                                                     attrs);
}

// Add layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Add>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Add>(createParameter(layerParsePrms.inputPorts[0]),
            createParameter(layerParsePrms.inputPorts[1]));
}

// Maximum layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Maximum>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Maximum>(createParameter(layerParsePrms.inputPorts[0]),
            createParameter(layerParsePrms.inputPorts[1]));
}

// Divide layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Divide>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Divide>(createParameter(layerParsePrms.inputPorts[0]),
            createParameter(layerParsePrms.inputPorts[1]));
}

// Multiply layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Multiply>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Multiply>(createParameter(layerParsePrms.inputPorts[0]),
            createParameter(layerParsePrms.inputPorts[1]));
}

// Broadcast layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::DynBroadcast>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 3, 1);
    auto firstInputShapePort = layerParsePrms.inputPorts[1];
    firstInputShapePort.precision = Precision::I64;
    auto secondInputShapePort = layerParsePrms.inputPorts[2];
    secondInputShapePort.precision = Precision::I64;
    return std::make_shared<ngraph::op::DynBroadcast>(createParameter(layerParsePrms.inputPorts[0]),
            createParameter(firstInputShapePort), createParameter(secondInputShapePort));
}

// Constant layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Constant>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 0, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    size_t offset = GetUIntAttr(dn, "offset");
    size_t size = GetUIntAttr(dn, "size");
    if (!weights || weights->cbuffer() == nullptr || weights->byteSize() < offset + size)
        THROW_IE_EXCEPTION << "Cannot create " << getType() << " layer with name: " << layerParsePrms.name
            << ". Layer has incorrect weights!";
    const void* data = weights->cbuffer().as<uint8_t *>() + offset;
    auto port = layerParsePrms.outputPorts[0];
    ngraph::Shape shape(port.dims);
    return std::make_shared<ngraph::op::Constant>(details::ngraph::convertPrecision(port.precision), shape, data);
}

// Power layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Power>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Power>(createParameter(layerParsePrms.inputPorts[0]),
            createParameter(layerParsePrms.inputPorts[1]));
}

// MatMul layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Dot>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Dot>(createParameter(layerParsePrms.inputPorts[0]),
            createParameter(layerParsePrms.inputPorts[1]));
}

// Softmax layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Softmax>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto axis = ngraph::AxisSet(getParameters<size_t>(dn, "axis"));
    return std::make_shared<ngraph::op::Softmax>(createParameter(layerParsePrms.inputPorts[0]), axis);
}

// Transpose layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Transpose>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    auto firstInputShapePort = layerParsePrms.inputPorts[1];
    firstInputShapePort.precision = Precision::I64;
    return std::make_shared<ngraph::op::Transpose>(createParameter(layerParsePrms.inputPorts[0]),
            createParameter(firstInputShapePort));
}

// Convolution layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::GroupConvolution>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    size_t group = GetUIntAttr(dn, "group");

    ngraph::op::PadType pad_type = ngraph::op::PadType::EXPLICIT;
    std::string auto_pad = GetStrAttr(dn, "auto_pad", "");
    if (auto_pad == "same_lower") {
        pad_type = ngraph::op::PadType::SAME_LOWER;
    } else if (auto_pad == "same_upper") {
        pad_type = ngraph::op::PadType::SAME_UPPER;
    }

    auto strides = ngraph::Strides(getParameters<size_t>(dn, "strides"));
    auto dilations = ngraph::Strides(getParameters<size_t>(dn, "dilations"));
    auto pads_begin = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_begin"));
    auto pads_end = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_end"));

    if (group != 1) {
        return std::make_shared<ngraph::op::GroupConvolution>(createParameter(layerParsePrms.inputPorts[0]),
                                                              createParameter(layerParsePrms.inputPorts[1]),
                                                              strides,
                                                              dilations,
                                                              pads_begin,
                                                              pads_end,
                                                              ngraph::Strides{},
                                                              group,
                                                              pad_type);
    } else {
        return std::make_shared<ngraph::op::Convolution>(createParameter(layerParsePrms.inputPorts[0]),
                                                         createParameter(layerParsePrms.inputPorts[1]),
                                                         strides,
                                                         dilations,
                                                         pads_begin,
                                                         pads_end,
                                                         ngraph::Strides{},
                                                         pad_type);
    }
}

// Deconvolution layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::ConvolutionBackpropData>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    size_t group = GetUIntAttr(dn, "group", 1);
    if (group != 1)
        THROW_IE_EXCEPTION << "Cannot create grouped deconvolution layer " << layerParsePrms.name;

    ngraph::op::PadType pad_type = ngraph::op::PadType::EXPLICIT;
    std::string auto_pad = GetStrAttr(dn, "auto_pad", "");
    if (auto_pad == "same_lower") {
        pad_type = ngraph::op::PadType::SAME_LOWER;
    } else if (auto_pad == "same_upper") {
        pad_type = ngraph::op::PadType::SAME_UPPER;
    }

    ngraph::Shape shape(layerParsePrms.outputPorts[0].dims);
    if (shape.size() != 4 && shape.size() != 5) {
        THROW_IE_EXCEPTION << "Cannot create deconvolution layer " << layerParsePrms.name << " Supports only 2D and 3D deconvolution";
    }

    auto strides = ngraph::Strides(getParameters<size_t>(dn, "strides"));
    auto dilations = ngraph::Strides(getParameters<size_t>(dn, "dilations", {}));
    auto pads_begin = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_begin"));
    auto pads_end = ngraph::CoordinateDiff(getParameters<std::ptrdiff_t>(dn, "pads_end"));
    if (dilations.empty()) {
        for (size_t i = 0; i < pads_end.size(); i++) {
            dilations.emplace_back(1);
        }
    }
    return std::make_shared<ngraph::op::ConvolutionBackpropData>(shape,
                                                          createParameter(layerParsePrms.inputPorts[0]),
                                                          createParameter(layerParsePrms.inputPorts[1]),
                                                          strides,
                                                          dilations,
                                                          pads_begin,
                                                          pads_end,
                                                          (strides.size() == 2 ? ngraph::Strides{1, 1} : ngraph::Strides{1, 1, 1}));
}

// AvgPool layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::AvgPool>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto includePad = GetStrAttr(dn, "exclude-pad") == "false";
    auto strides = ngraph::Strides(getParameters<size_t>(dn, "strides"));
    auto kernel = ngraph::Shape(getParameters<size_t>(dn, "kernel"));
    auto pads_begin = ngraph::Shape(getParameters<std::size_t>(dn, "pads_begin"));
    auto pads_end = ngraph::Shape(getParameters<std::size_t>(dn, "pads_end"));
    auto pad_type = ngraph::op::PadType::EXPLICIT;
    auto ceil_mode = GetStrAttr(dn, "rounding_type", "floor") == "ceil";
    return std::make_shared<ngraph::op::AvgPool>(createParameter(layerParsePrms.inputPorts[0]), kernel,
            strides, pads_begin, pads_end, includePad, pad_type, ceil_mode);
}

// MaxPool layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::MaxPool>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto strides = ngraph::Strides(getParameters<size_t>(dn, "strides"));
    auto kernel = ngraph::Shape(getParameters<size_t>(dn, "kernel"));
    auto pads_begin = ngraph::Shape(getParameters<std::size_t>(dn, "pads_begin"));
    auto pads_end = ngraph::Shape(getParameters<std::size_t>(dn, "pads_end"));
    auto pad_type = ngraph::op::PadType::EXPLICIT;
    auto ceil_mode = GetStrAttr(dn, "rounding_type", "floor") == "ceil";
    return std::make_shared<ngraph::op::MaxPool>(createParameter(layerParsePrms.inputPorts[0]), kernel,
            strides, pads_begin, pads_end, pad_type, ceil_mode);
}

// Concat layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Concat>::createLayer(const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, -1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    size_t axis = GetUIntAttr(dn, "axis");

    ngraph::NodeVector inNodes;
    for (const auto inPort : layerParsePrms.inputPorts) {
        inNodes.emplace_back(createParameter(inPort));
    }
    return std::make_shared<ngraph::op::Concat>(inNodes, axis);
}

}  // namespace InferenceEngine

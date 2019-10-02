// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_ngraph_utils.hpp"
#include "ie_ir_parser.hpp"
#include "ie_format_parser.h"
#include "details/ie_cnn_network_tools.h"
#include "cnn_network_impl.hpp"
#include "debug.h"

#include <algorithm>
#include <sstream>
#include <memory>
#include <vector>
#include <string>
#include <deque>
#include <map>
#include <set>

#include <ngraph/axis_vector.hpp>
#include <ngraph/coordinate_diff.hpp>
#include <ngraph/descriptor/input.hpp>
#include <ngraph/op/abs.hpp>
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
#include <ngraph/op/experimental/layers/roi_pooling.hpp>
#include <ngraph/op/experimental/layers/psroi_pooling.hpp>
#include <ngraph/op/experimental/layers/region_yolo.hpp>
#include <ngraph/op/experimental/layers/reorg_yolo.hpp>
#include <ngraph/op/experimental/shape_of.hpp>
#include <ngraph/op/experimental/tile.hpp>
#include <ngraph/op/experimental/transpose.hpp>
#include <ngraph/op/fused/clamp.hpp>
#include <ngraph/op/fused/elu.hpp>
#include <ngraph/op/fused/group_conv.hpp>
#include <ngraph/op/fused/mvn.hpp>
#include <ngraph/op/fused/normalize_l2.hpp>
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
#include <ngraph/autodiff/adjoints.hpp>

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
    using node_params = struct  {
        pugi::xml_node xml;
        GenericLayerParams params;
    };
    std::map<size_t, node_params> params;

    std::vector<size_t> outputs;

    // Read all layers and store their parameters in params map
    FOREACH_CHILD(node, root.child("layers"), "layer") {
        auto node_param = parseGenericParams(node);
        params[node_param.layerId] = {node, node_param};
        if (node_param.type == "Result") {
            outputs.push_back(node_param.layerId);
        }
    }

    using edge = struct {
        size_t fromLayerId, fromPortId, toPortId;
    };
    std::map<size_t, std::vector<edge> > edges;
    std::map<size_t, std::shared_ptr<ngraph::Node> > id_to_node;

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
        for (auto & edge : edges[id]) {
            dfs(edge.fromLayerId);
        }
        order.push_back(id);
    };
    std::for_each(outputs.begin(), outputs.end(), dfs);

    ngraph::ParameterVector parameter_nodes;
    ngraph::ResultVector result_nodes;

    //  Following topological order create nGraph operations
    for (auto & layer_id : order) {
        auto & p = params[layer_id];
        ngraph::OutputVector inputs(edges[layer_id].size());
        for (auto & e  : edges[layer_id]) {
            auto input_node = id_to_node[e.fromLayerId];
            if (!input_node) {
                THROW_IE_EXCEPTION << "Attempt to access node " << e.fromLayerId << " that not in graph.";
            }
            auto & p_output = params[e.fromLayerId].params;
            inputs[p.params.getRealInputPortId(e.toPortId)] = input_node->output(p_output.getRealOutputPortId(e.fromPortId));
        }

        auto node = createNode(inputs, p.xml, weights, p.params);
        id_to_node[layer_id] = node;

        // Check that output shape after nGraph node validation the same as in IR
        // because IR always right!
        // Temporary disabled!
//        for (size_t i = 0; i < p.params.outputPorts.size(); ++i) {
//            if (p.params.outputPorts[i].dims != node->output(i).get_shape()) {
//                THROW_IE_EXCEPTION << "Shape after nGraph infer " << details::dumpVec(node->output(i).get_shape())
//                                   << " differ from IR shapes: " << details::dumpVec(p.params.outputPorts[i].dims);
//            }
//        }

        if (auto parameter_node = std::dynamic_pointer_cast<ngraph::op::Parameter> (node)) {
            parameter_nodes.emplace_back(parameter_node);
        }

        if (auto result_node = std::dynamic_pointer_cast<ngraph::op::Result> (node)) {
            result_nodes.emplace_back(result_node);
        }
    }

    return std::make_shared<ngraph::Function>(result_nodes, parameter_nodes, GetStrAttr(root, "name", ""));
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

std::shared_ptr<ngraph::Node> V10Parser::LayerBaseCreator::createOptionalParameter(const GenericLayerParams::LayerPortData& port) {
    return std::make_shared<ngraph::op::Dummy>();
}

std::shared_ptr<ngraph::Node> V10Parser::createNode(const std::vector<ngraph::Output<ngraph::Node> > & inputs,
                                                    const pugi::xml_node& node,
                                                    const Blob::CPtr& weights,
                                                    const GenericLayerParams& params) {
    static std::vector<std::shared_ptr<LayerBaseCreator>> creators = {
        std::make_shared<LayerCreator<ngraph::op::Abs>>("Abs"),
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
        std::make_shared<LayerCreator<ngraph::op::MVN>>("MVN"),
        std::make_shared<LayerCreator<ngraph::op::MaxPool>>("MaxPool"),
        std::make_shared<LayerCreator<ngraph::op::Maximum>>("Maximum"),
        std::make_shared<LayerCreator<ngraph::op::Multiply>>("Multiply"),
        std::make_shared<LayerCreator<ngraph::op::NormalizeL2>>("NormalizeL2"),
        std::make_shared<LayerCreator<ngraph::op::PRelu>>("PReLU"),
        std::make_shared<LayerCreator<ngraph::op::Parameter>>("Parameter"),
        std::make_shared<LayerCreator<ngraph::op::Power>>("Pow"),
        std::make_shared<LayerCreator<ngraph::op::PriorBox>>("PriorBox"),
        std::make_shared<LayerCreator<ngraph::op::PriorBoxClustered>>("PriorBoxClustered"),
        std::make_shared<LayerCreator<ngraph::op::Proposal>>("Proposal"),
        std::make_shared<LayerCreator<ngraph::op::ReorgYolo>>("ReorgYolo"),
        std::make_shared<LayerCreator<ngraph::op::RegionYolo>>("RegionYolo"),
        std::make_shared<LayerCreator<ngraph::op::Relu>>("ReLU"),
        std::make_shared<LayerCreator<ngraph::op::Result>>("Result"),
        std::make_shared<LayerCreator<ngraph::op::ROIPooling>>("ROIPooling"),
        std::make_shared<LayerCreator<ngraph::op::PSROIPooling>>("PSROIPooling"),
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
            ngraphNode = creator->createLayer(inputs, node, weights, params);
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
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Parameter>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 0, 1);
    ngraph::PartialShape shape(layerParsePrms.outputPorts[0].dims);
    return std::make_shared<ngraph::op::Parameter>(details::ngraph::convertPrecision(layerParsePrms.outputPorts[0].precision), shape);
}

// DetectionOutput layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::DetectionOutput>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
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
    // TODO: add DO constructor with Output<Node> args instead of nodes
    return std::make_shared<ngraph::op::DetectionOutput>(inputs[0].get_node_shared_ptr(), inputs[1].get_node_shared_ptr(), inputs[2].get_node_shared_ptr(),
            createOptionalParameter(GenericLayerParams::LayerPortData()),
            createOptionalParameter(GenericLayerParams::LayerPortData()),
            attr);
}

// PriorBoxClustered layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::PriorBoxClustered>::createLayer(const ngraph::OutputVector & inputs,
        const pugi::xml_node& node,
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
    float step = GetFloatAttr(dn, "step", 0);
    attr.step_heights = GetFloatAttr(dn, "step_h", step);
    attr.step_widths = GetFloatAttr(dn, "step_w", step);
    attr.clip = (GetIntAttr(dn, "clip") != 0);
    attr.num_priors = GetUIntAttr(dn, "num_priors", attr.widths.size());

    auto inputShapePort1 = layerParsePrms.inputPorts[0];
    inputShapePort1.precision = Precision::I64;

    auto inputShapePort2 = layerParsePrms.inputPorts[1];
    inputShapePort2.precision = Precision::I64;
    // TODO: add PriorBoxClustered constructor with Output<Node> args instead of nodes
    return std::make_shared<ngraph::op::PriorBoxClustered>(inputs[0].get_node_shared_ptr(), inputs[1].get_node_shared_ptr(), attr);
}

// Proposal layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Proposal>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
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
    // TODO: add Proposal constructor with Output<Node> args instead of nodes
    return std::make_shared<ngraph::op::Proposal>(inputs[0].get_node_shared_ptr(), inputs[1].get_node_shared_ptr(), inputs[2].get_node_shared_ptr(), attr);
}

// PriorBox layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::PriorBox>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
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
    // TODO: add PriorBox constructor with Output<Node> args instead of nodes
    return std::make_shared<ngraph::op::PriorBox>(inputs[0].get_node_shared_ptr(), inputs[1].get_node_shared_ptr(), attr);
}

// ShapeOf layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::ShapeOf>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    return std::make_shared<ngraph::op::ShapeOf>(inputs[0]);
}

// TopK layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::TopK>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    size_t axis = GetUInt64Attr(dn, "axis");
    size_t maxMode = GetStrAttr(dn, "mode", "max") == "max";
    THROW_IE_EXCEPTION << "Operation TopK is not supported yet";
}

// MVN layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::MVN>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    double eps = GetFloatAttr(dn, "eps");
    bool across = GetUIntAttr(dn, "across_channels", 0) == 1;
    bool normalize_variance = GetUIntAttr(dn, "normalize_variance", 0) == 1;
    return std::make_shared<ngraph::op::MVN>(inputs[0], across, normalize_variance, eps);
}

// LRN layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::LRN>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    pugi::xml_node dn = node.child("data");
    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    double alpha = GetFloatAttr(dn, "alpha");
    double beta = GetFloatAttr(dn, "beta");
    double bias = GetFloatAttr(dn, "bias");
    size_t size = GetUInt64Attr(dn, "local-size");

    if (layerParsePrms.inputPorts.size() == 1)
        return std::make_shared<ngraph::op::LRN>(inputs[0], alpha, beta, bias, size);
    else if (layerParsePrms.inputPorts.size() == 2)
        return std::make_shared<ngraph::op::LRN>(inputs[0], inputs[1], alpha, beta, bias, size);
    else
        THROW_IE_EXCEPTION << layerParsePrms.type << " layer " << layerParsePrms.name << " with id: "
        << layerParsePrms.layerId << " has incorrect number of input ports!";
}

// Clamp layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Clamp>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    double maxVal = GetFloatAttr(dn, "max");
    double minVal = GetFloatAttr(dn, "min");
    return std::make_shared<ngraph::op::Clamp>(inputs[0], minVal, maxVal);
}

// Split layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Split>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
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
    return std::make_shared<ngraph::op::Split>(inputs[0], axis, splits);
}

// Sigmoid layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Sigmoid>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    return std::make_shared<ngraph::op::Sigmoid>(inputs[0]);
}

// ELU layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Elu>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::Elu>(inputs[0], GetFloatAttr(dn, "alpha"));
}

// PReLU layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::PRelu>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::PRelu>(inputs[0], inputs[1]);
}

// Exp layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Exp>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    return std::make_shared<ngraph::op::Exp>(inputs[0]);
}

// ReLU layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Relu>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    return std::make_shared<ngraph::op::Relu>(inputs[0]);
}

// Tanh layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Tanh>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    return std::make_shared<ngraph::op::Tanh>(inputs[0]);
}

// Result layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Result>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 0);
    return std::make_shared<ngraph::op::Result>(inputs[0]);
}

// Tile layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Tile>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Tile>(inputs[0], inputs[1]);
}

// StridedSlice layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::DynSlice>::createLayer(const ngraph::OutputVector & inputs,
                                                                                         const pugi::xml_node& node,
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

    return std::make_shared<ngraph::op::DynSlice>(inputs[0],
                                                  inputs[1],
                                                  inputs[2],
                                                  inputs[3],
                                                  lower_bounds_mask,
                                                  upper_bounds_mask,
                                                  new_axis,
                                                  shrink_axis,
                                                  ellipsis_mask);
}

// Reshape layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::DynReshape>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    auto inputShapePort = layerParsePrms.inputPorts[1];
    inputShapePort.precision = Precision::I64;
    return std::make_shared<ngraph::op::DynReshape>(inputs[0], inputs[1], true);
}

// Squeeze layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Squeeze>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
                                                                                        const Blob::CPtr& weights,
                                                                                        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Squeeze>(inputs[0], inputs[1]);
}

// Unsqueeze layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Unsqueeze>::createLayer(const ngraph::OutputVector & inputs,
                                                                                          const pugi::xml_node& node,
                                                                                          const Blob::CPtr& weights,
                                                                                          const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Unsqueeze>(inputs[0], inputs[1]);
}

// Interpolate layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Interpolate>::createLayer(const ngraph::OutputVector & inputs,
                                                                                            const pugi::xml_node& node,
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
    // TODO: add Interpolate constructor with Output<Node> args instead of nodes
    return std::make_shared<ngraph::op::Interpolate>(inputs[0].get_node_shared_ptr(), inputs[1].get_node_shared_ptr(), attrs);
}

// Abs layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Abs>::createLayer(const ngraph::OutputVector & inputs,
                                                                                    const pugi::xml_node& node,
                                                                                    const Blob::CPtr& weights,
                                                                                    const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    return std::make_shared<ngraph::op::Abs>(inputs[0]);
}

// Add layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Add>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Add>(inputs[0], inputs[1]);
}

// Maximum layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Maximum>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Maximum>(inputs[0], inputs[1]);
}

// Divide layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Divide>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Divide>(inputs[0], inputs[1]);
}

// Multiply layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Multiply>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Multiply>(inputs[0], inputs[1]);
}

// Broadcast layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::DynBroadcast>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    if (inputs.size() == 2) {
//        NOT SUPPORTED BY NGRAPH
//        return std::make_shared<ngraph::op::DynBroadcast>(createParameter(layerParsePrms.inputPorts[0]),
//                                                          createParameter(firstInputShapePort));
        THROW_IE_EXCEPTION << "nGraph do not supports 2 inputs configuration";
    } else if (layerParsePrms.inputPorts.size() == 3) {
        auto secondInputShapePort = layerParsePrms.inputPorts[2];
        secondInputShapePort.precision = Precision::I64;
        return std::make_shared<ngraph::op::DynBroadcast>(inputs[0], inputs[1], inputs[2]);
    }
    THROW_IE_EXCEPTION << "Invalid number of inputs: " << layerParsePrms.inputPorts.size();
}

// Constant layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Constant>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
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
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Power>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Power>(inputs[0], inputs[1]);
}

// MatMul layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Dot>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    return std::make_shared<ngraph::op::Dot>(inputs[0], inputs[1]);
}

// Softmax layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Softmax>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto axis = ngraph::AxisSet(getParameters<size_t>(dn, "axis"));
    return std::make_shared<ngraph::op::Softmax>(inputs[0], axis);
}


// RegionYolo layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::RegionYolo>::createLayer(const ngraph::OutputVector & inputs,
                                                                                           const pugi::xml_node& node,
                                                                                           const Blob::CPtr& weights,
                                                                                           const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
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

    return std::make_shared<ngraph::op::RegionYolo>(inputs[0].get_node_shared_ptr(), coords, classes, num, do_softmax, mask, axis, end_axis);
}

// ReorgYolo layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::ReorgYolo>::createLayer(const ngraph::OutputVector & inputs,
                                                                                          const pugi::xml_node& node,
                                                                                          const Blob::CPtr& weights,
                                                                                          const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto stride = GetUIntAttr(dn, "stride");
    return std::make_shared<ngraph::op::ReorgYolo>(inputs[0].get_node_shared_ptr(), ngraph::Strides{stride});
}


// Transpose layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Transpose>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    auto firstInputShapePort = layerParsePrms.inputPorts[1];
    firstInputShapePort.precision = Precision::I64;
    return std::make_shared<ngraph::op::Transpose>(inputs[0], inputs[1]);
}

// Convolution layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::GroupConvolution>::createLayer(const ngraph::OutputVector & inputs,
        const pugi::xml_node& node,
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
        return std::make_shared<ngraph::op::GroupConvolution>(inputs[0],
                                                              inputs[1],
                                                              strides,
                                                              dilations,
                                                              pads_begin,
                                                              pads_end,
                                                              ngraph::Strides{},
                                                              group,
                                                              pad_type);
    } else {
        return std::make_shared<ngraph::op::Convolution>(inputs[0],
                                                         inputs[1],
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
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::ConvolutionBackpropData>::createLayer(const ngraph::OutputVector & inputs,
        const pugi::xml_node& node,
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
                                                          inputs[1],
                                                          inputs[0],
                                                          strides,
                                                          dilations,
                                                          pads_begin,
                                                          pads_end,
                                                          (strides.size() == 2 ? ngraph::Strides{1, 1} : ngraph::Strides{1, 1, 1}));
}

// AvgPool layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::AvgPool>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
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
    return std::make_shared<ngraph::op::AvgPool>(inputs[0], kernel,
            strides, pads_begin, pads_end, includePad, pad_type, ceil_mode);
}

// MaxPool layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::MaxPool>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
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
    return std::make_shared<ngraph::op::MaxPool>(inputs[0], kernel,
            strides, pads_begin, pads_end, pad_type, ceil_mode);
}

// ROIPooling layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::ROIPooling>::createLayer(const ngraph::OutputVector & inputs,
                                                                                           const pugi::xml_node& node,
                                                                                           const Blob::CPtr& weights,
                                                                                           const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto pooled_h = GetUIntAttr(dn, "pooled_h");
    auto pooled_w = GetUIntAttr(dn, "pooled_w");
    auto spatial_scale = GetFloatAttr(dn, "spatial_scale");
    auto method = GetStrAttr(dn, "method", "max");
    // TODO: add ROIPooling constructor with Output<Node> args instead of nodes
    return std::make_shared<ngraph::op::ROIPooling>(inputs[0].get_node_shared_ptr(),
                                                    inputs[1].get_node_shared_ptr(),
                                                    ngraph::Shape{pooled_h, pooled_w}, spatial_scale, method);
}

// PSROIPooling layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::PSROIPooling>::createLayer(const ngraph::OutputVector & inputs,
                                                                                           const pugi::xml_node& node,
                                                                                           const Blob::CPtr& weights,
                                                                                           const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    auto output_dim = GetIntAttr(dn, "output_dim");
    auto group_size = GetIntAttr(dn, "group_size", 1);
    auto spatial_bins_x = GetIntAttr(dn, "spatial_bins_x", 1);
    auto spatial_bins_y = GetIntAttr(dn, "spatial_bins_y", 1);
    auto spatial_scale = GetFloatAttr(dn, "spatial_scale");
    auto mode = GetStrAttr(dn, "mode", "average");

    return std::make_shared<ngraph::op::PSROIPooling>(inputs[0].get_node_shared_ptr(),
                                                      inputs[1].get_node_shared_ptr(),
                                                      output_dim, group_size, spatial_scale, spatial_bins_x, spatial_bins_y, mode);
}

// Concat layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::Concat>::createLayer(const ngraph::OutputVector & inputs, const pugi::xml_node& node,
        const Blob::CPtr& weights,
        const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, -1, 1);
    pugi::xml_node dn = node.child("data");

    if (dn.empty())
        THROW_IE_EXCEPTION << "Cannot read parameter for " << getType() << " layer with name: " << layerParsePrms.name;

    return std::make_shared<ngraph::op::Concat>(inputs, GetUIntAttr(dn, "axis"));
}

// NormalizeL2 layer
template <>
std::shared_ptr<ngraph::Node> V10Parser::LayerCreator<ngraph::op::NormalizeL2>::createLayer(const ngraph::OutputVector & inputs,
                                                                                            const pugi::xml_node& node,
                                                                                            const Blob::CPtr& weights,
                                                                                            const GenericLayerParams& layerParsePrms) {
    checkParameters(layerParsePrms, 2, 1);
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

}  // namespace InferenceEngine

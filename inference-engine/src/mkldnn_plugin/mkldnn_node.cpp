// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_node.h"
#include "mkldnn_extension_mngr.h"
#include "mkldnn_itt.h"

#include "caseless.hpp"
#include <vector>
#include <string>
#include <limits>
#include <cstdint>
#include <unordered_map>

#include <nodes/mkldnn_concat_node.h>
#include <nodes/mkldnn_conv_node.h>
#include <nodes/mkldnn_deconv_node.h>
#include <nodes/mkldnn_eltwise_node.h>
#include <nodes/mkldnn_matmul_node.h>
#include <nodes/mkldnn_fullyconnected_node.h>
#include <nodes/mkldnn_generic_node.h>
#include <nodes/mkldnn_input_node.h>
#include <nodes/mkldnn_lrn_node.h>
#include <nodes/mkldnn_pooling_node.h>
#include <nodes/mkldnn_reorder_node.h>
#include <nodes/mkldnn_reshape_node.h>
#include <nodes/mkldnn_softmax_node.h>
#include <nodes/mkldnn_tile_node.h>
#include <nodes/mkldnn_split_node.h>
#include <nodes/mkldnn_pad_node.h>
#include <nodes/mkldnn_transpose_node.h>
#include <nodes/mkldnn_memory_node.hpp>
#include <nodes/mkldnn_mvn_node.h>
#include <nodes/mkldnn_normalize_node.h>
#include <nodes/mkldnn_reduce_node.h>
#include <nodes/mkldnn_tensoriterator_node.h>
#include <nodes/mkldnn_scatter_update_node.h>
#include <nodes/mkldnn_interpolate_node.h>
#include <nodes/mkldnn_depth_to_space_node.h>
#include <nodes/mkldnn_space_to_depth_node.h>
#include <nodes/mkldnn_strided_slice_node.h>
#include <nodes/mkldnn_shuffle_channels_node.h>
#include <nodes/mkldnn_reference_node.h>
#include <nodes/mkldnn_fake_quantize_node.h>
#include <mkldnn_types.h>
#include <dnnl_types.h>
#include "mkldnn_extension_utils.h"

#include "nodes/common/cpu_memcpy.h"
#include "mkldnn_debug.h"
#include "utils/rt_info/memory_formats_attribute.hpp"

#include <ie_ngraph_utils.hpp>
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"
#include "nodes/common/cpu_convert.h"
#include "cpu_memory_desc_utils.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace openvino;

using namespace InferenceEngine::details;
namespace MKLDNNPlugin {
static const InferenceEngine::details::caseless_unordered_map<std::string, Type> type_to_name_tbl = {
        { "Constant", Input },
        { "Parameter", Input },
        { "Result", Output },
        { "Convolution", Convolution },
        { "GroupConvolution", Convolution },
        { "MatMul", MatMul },
        { "FullyConnected", FullyConnected },
        { "MaxPool", Pooling },
        { "AvgPool", Pooling },
        { "Add", Eltwise },
        { "Subtract", Eltwise },
        { "Multiply", Eltwise },
        { "Divide", Eltwise },
        { "SquaredDifference", Eltwise },
        { "Maximum", Eltwise },
        { "Minimum", Eltwise },
        { "Mod", Eltwise },
        { "FloorMod", Eltwise },
        { "Power", Eltwise },
        { "PowerStatic", Eltwise },
        { "Equal", Eltwise },
        { "NotEqual", Eltwise },
        { "Greater", Eltwise },
        { "GreaterEqual", Eltwise },
        { "Less", Eltwise },
        { "LessEqual", Eltwise },
        { "LogicalAnd", Eltwise },
        { "LogicalOr", Eltwise },
        { "LogicalXor", Eltwise },
        { "LogicalNot", Eltwise },
        { "Relu", Eltwise },
        { "LeakyRelu", Eltwise },
        { "Gelu", Eltwise },
        { "Elu", Eltwise },
        { "Tanh", Eltwise },
        { "Sigmoid", Eltwise },
        { "Abs", Eltwise },
        { "Sqrt", Eltwise },
        { "Clamp", Eltwise },
        { "Exp", Eltwise },
        { "SwishCPU", Eltwise },
        { "HSwish", Eltwise },
        { "Mish", Eltwise },
        { "HSigmoid", Eltwise },
        { "Round", Eltwise },
        { "PRelu", Eltwise },
        { "Erf", Eltwise },
        { "SoftPlus", Eltwise },
        { "Reshape", Reshape },
        { "Squeeze", Reshape },
        { "Unsqueeze", Reshape },
        { "Softmax", Softmax },
        { "Reorder", Reorder },
        { "BatchToSpace", BatchToSpace },
        { "SpaceToBatch", SpaceToBatch },
        { "DepthToSpace", DepthToSpace },
        { "SpaceToDepth", SpaceToDepth },
        { "Roll", Roll },
        { "LRN", Lrn },
        { "Split", Split },
        { "VariadicSplit", Split },
        { "Concat", Concatenation },
        { "ConvolutionBackpropData", Deconvolution },
        { "GroupConvolutionBackpropData", Deconvolution },
        { "StridedSlice", StridedSlice },
        { "Tile", Tile },
        { "ROIAlign", ROIAlign },
        { "ROIPooling", ROIPooling },
        { "PSROIPooling", PSROIPooling },
        { "DeformablePSROIPooling", PSROIPooling },
        { "Pad", Pad },
        { "Transpose", Transpose },
        { "LSTMCell", RNNCell },
        { "GRUCell", RNNCell },
        { "RNNCell", RNNCell },
        { "LSTMSequence", RNNSeq },
        { "GRUSequence", RNNSeq },
        { "RNNSequence", RNNSeq },
        { "FakeQuantize", FakeQuantize },
        { "BinaryConvolution", BinaryConvolution },
        { "DeformableConvolution", DeformableConvolution },
        { "TensorIterator", TensorIterator },
        { "Loop", TensorIterator },
        { "ReadValue", MemoryInput},  // for construction from name ctor, arbitrary name is used
        { "Assign", MemoryOutput },  // for construction from layer ctor
        { "Convert", Convert },
        { "MVN", MVN},
        { "NormalizeL2", NormalizeL2},
        { "ScatterUpdate", ScatterUpdate},
        { "ScatterElementsUpdate", ScatterElementsUpdate},
        { "ScatterNDUpdate", ScatterNDUpdate},
        { "Interpolate", Interpolate},
        { "ReduceL1", Reduce},
        { "ReduceL2", Reduce},
        { "ReduceLogicalAnd", Reduce},
        { "ReduceLogicalOr", Reduce},
        { "ReduceMax", Reduce},
        { "ReduceMean", Reduce},
        { "ReduceMin", Reduce},
        { "ReduceProd", Reduce},
        { "ReduceSum", Reduce},
        { "ReduceLogSum", Reduce},
        { "ReduceLogSumExp", Reduce},
        { "ReduceSumSquare", Reduce},
        { "Broadcast", Broadcast},
        { "EmbeddingSegmentsSum", EmbeddingSegmentsSum},
        { "EmbeddingBagPackedSum", EmbeddingBagPackedSum},
        { "EmbeddingBagOffsetsSum", EmbeddingBagOffsetsSum},
        { "Gather", Gather},
        { "GatherElements", GatherElements},
        { "GatherND", GatherND},
        { "OneHot", OneHot},
        { "RegionYolo", RegionYolo},
        { "Select", Select},
        { "ShuffleChannels", ShuffleChannels},
        { "DFT", DFT},
        { "IDFT", DFT},
        { "Abs", Math},
        { "Acos", Math},
        { "Acosh", Math},
        { "Asin", Math},
        { "Asinh", Math},
        { "Atan", Math},
        { "Atanh", Math},
        { "Ceil", Math},
        { "Ceiling", Math},
        { "Cos", Math},
        { "Cosh", Math},
        { "Floor", Math},
        { "HardSigmoid", Math},
        { "Log", Math},
        { "Neg", Math},
        { "Reciprocal", Math},
        { "Selu", Math},
        { "Sign", Math},
        { "Sin", Math},
        { "Sinh", Math},
        { "SoftPlus", Math},
        { "Softsign", Math},
        { "Tan", Math},
        { "CTCLoss", CTCLoss},
        { "Bucketize", Bucketize},
        { "CTCGreedyDecoder", CTCGreedyDecoder},
        { "CTCGreedyDecoderSeqLen", CTCGreedyDecoderSeqLen},
        { "CumSum", CumSum},
        { "DetectionOutput", DetectionOutput},
        { "ExperimentalDetectronDetectionOutput", ExperimentalDetectronDetectionOutput},
        { "LogSoftmax", LogSoftmax},
        { "TopK", TopK},
        { "GatherTree", GatherTree},
        { "GRN", GRN},
        { "Range", Range},
        { "Proposal", Proposal},
        { "ReorgYolo", ReorgYolo},
        { "ReverseSequence", ReverseSequence},
        { "ExperimentalDetectronTopKROIs", ExperimentalDetectronTopKROIs},
        { "ExperimentalDetectronROIFeatureExtractor", ExperimentalDetectronROIFeatureExtractor},
        { "ExperimentalDetectronPriorGridGenerator", ExperimentalDetectronPriorGridGenerator},
        { "ExperimentalDetectronGenerateProposalsSingleImage", ExperimentalDetectronGenerateProposalsSingleImage},
        { "ExtractImagePatches", ExtractImagePatches},
        { "NonMaxSuppressionIEInternal", NonMaxSuppression}
};

Type TypeFromName(const std::string type) {
    auto itType = type_to_name_tbl.find(type);
    if (type_to_name_tbl.end() != itType) {
        return itType->second;
    } else {
        return Unknown;
    }
}

}  //  namespace MKLDNNPlugin

MKLDNNNode::NodesFactory & MKLDNNNode::factory() {
    static NodesFactory factoryInstance;
    return factoryInstance;
}

MKLDNNNode::MKLDNNNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &w_cache)
        : selectedPrimitiveDescriptorIndex(-1), permanent(false), temporary(false), constant(ConstantType::Unknown),
          weightCache(w_cache), engine(eng), name(op->get_friendly_name()), typeStr(op->get_type_name()),
          type(TypeFromName(op->get_type_name())), profiling(op->get_friendly_name()) {
    algorithm = Algorithm::Undefined;
    fusingPort = -1;
    const std::string errorPrefix = "Ngraph operation " + std::string(op->get_type_name()) + " with name " + op->get_friendly_name();

    for (size_t i = 0; i < op->get_input_size(); i++) {
        const auto &shape = op->get_input_partial_shape(i);

        bool isScalar = false;
        if (shape.rank().is_static()) {
            isScalar = shape.rank().get_length() == 0;
        }
        inputShapes.emplace_back(isScalar ? ngraph::PartialShape{1} : shape);
        originalInputPrecisions.emplace_back(details::convertPrecision(op->get_input_element_type(i)));
    }

    if (typeStr != "Result" && typeStr != "Assign") {
        if (op->get_output_size() == 0) {
            IE_THROW() << "Node with type '" << typeStr << "' and name '" << name << "' does not have any outputs.";
        }
        for (size_t i = 0; i < op->get_output_size(); i++) {
            const auto &shape = op->get_output_partial_shape(i);

            bool isScalar = false;
            if (shape.rank().is_static()) {
                isScalar = shape.rank().get_length() == 0;
            }
            outputShapes.emplace_back(isScalar ? ngraph::PartialShape{1} : shape);
            originalOutputPrecisions.emplace_back(details::convertPrecision(op->get_output_element_type(i)));
        }
    }

    const auto& rtInfo = op->get_rt_info();
    if (rtInfo.count("originalLayersNames")) {
        originalLayers = getRTInfoValue(rtInfo, "originalLayersNames");
    }

    if (originalLayers.empty()) {
        addOriginalLayer(name);
    }

    auto primitivesPriority = getPrimitivesPriorityValue(op);
    if (!primitivesPriority.empty()) {
        std::istringstream stream(primitivesPriority);
        std::string str;
        while (getline(stream, str, ',')) {
            if (str.substr(0, 4) != "cpu:")
                continue;
            implPriorities.push_back(parse_impl_name(str));
            if (implPriorities[implPriorities.size() - 1] == impl_desc_type::unknown &&
                str != "cpu:unknown")
                IE_THROW() << "Unsupported CPU implementation " << str << " for node " << getName();
        }
    }

    if (op != nullptr) {
        std::string inputMemoryFormats = ngraph::getMLKDNNInputMemoryFormats(op);
        if (!inputMemoryFormats.empty()) {
            std::istringstream stream(inputMemoryFormats);
            std::string str;
            while (getline(stream, str, ',')) {
                if (str.substr(0, 4) != "cpu:")
                    continue;
                inputMemoryFormatsFilter.push_back(mkldnn::utils::str2fmt(str.substr(4, str.size()).c_str()));
            }
        }

        std::string outputMemoryFormats = ngraph::getMLKDNNOutputMemoryFormats(op);
        if (!outputMemoryFormats.empty()) {
            std::istringstream stream(outputMemoryFormats);
            std::string str;
            while (getline(stream, str, ',')) {
                if (str.substr(0, 4) != "cpu:")
                    continue;
                outputMemoryFormatsFilter.push_back(mkldnn::utils::str2fmt(str.substr(4, str.size()).c_str()));
            }
        }
    }
}

MKLDNNNode::MKLDNNNode(const std::string& type, const std::string& name, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &w_cache)
        : selectedPrimitiveDescriptorIndex(-1), permanent(false), temporary(false), constant(ConstantType::Unknown),
          weightCache(w_cache), engine(eng), name(name), typeStr(type),
          type(TypeFromName(type)), profiling(name) {
    // TODO [NM]: What about filling inDims and outDims?
}

void MKLDNNNode::addEdge(const MKLDNNEdgeWeakPtr& edge) {
    auto edgePtr = edge.lock();
    if (!edgePtr)
        return;
    auto parentPtr = edgePtr->getParent();
    auto childPtr = edgePtr->getChild();
    if (!parentPtr || !childPtr)
        return;

    parentPtr->childEdges.push_back(edge);
    childPtr->parentEdges.push_back(edge);
}

void MKLDNNNode::removeEdge(const MKLDNNEdgeWeakPtr& edge) {
    auto edgePtr = edge.lock();
    if (!edgePtr)
        return;
    auto parentPtr = edgePtr->getParent();
    auto childPtr = edgePtr->getChild();
    if (!parentPtr || !childPtr)
        return;
    for (auto it = childPtr->parentEdges.begin(); it != childPtr->parentEdges.end(); it++) {
        auto parentEdge = (*it).lock();
        if (parentEdge && parentEdge->getChild() == childPtr && parentEdge->getParent() == parentPtr) {
            childPtr->parentEdges.erase(it);
            break;
        }
    }
    for (auto it = parentPtr->childEdges.begin(); it != parentPtr->childEdges.end(); it++) {
        auto childEdge = (*it).lock();
        if (childEdge && childEdge->getChild() == childPtr && childEdge->getParent() == parentPtr) {
            parentPtr->childEdges.erase(it);
            break;
        }
    }
}

void MKLDNNNode::remove() {
    auto parent_edges = parentEdges;
    for (const auto &parentEdge : parent_edges) {
        removeEdge(parentEdge);
    }
    auto child_edges = childEdges;
    for (const auto &childEdge : child_edges) {
        removeEdge(childEdge);
    }
}

bool MKLDNNNode::isEdgesEmpty(const std::vector<MKLDNNEdgeWeakPtr>& edges) const {
    for (auto &edge : edges) {
        if (edge.lock())
            return false;
    }
    return true;
}

void MKLDNNNode::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getPrimitivesPriority(), false);
}

void MKLDNNNode::selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority, bool ignoreConstInputs) {
    for (auto& type : priority) {
        int selectedPrimitive = -1;
        int equalsFormatCount = -1;
        for (size_t i = 0; i < getSupportedPrimitiveDescriptors().size(); i++) {
            impl_desc_type supportedType = getSupportedPrimitiveDescriptors()[i].getImplementationType();
            if (type == supportedType) {
                int equalsLocalFormatCount = 0;
                if (getSupportedPrimitiveDescriptors()[i].getConfig().inConfs.size() > getParentEdges().size())
                    continue;
                for (size_t j = 0; j < getSupportedPrimitiveDescriptors()[i].getConfig().inConfs.size(); j++) {
                    auto parentEdge = getParentEdgeAt(j);
                    auto parentPtr = parentEdge->getParent();

                    // We don't take into account constant edges since reorders on them will be executed on load network stage
                    if (ignoreConstInputs && j > 0 && parentPtr->isConstant()) {
                        equalsLocalFormatCount++;
                        continue;
                    }

                    auto parent_spd = parentPtr->getSelectedPrimitiveDescriptor();

                    if (parent_spd != nullptr && !parent_spd->getConfig().outConfs.empty()) {
                        int inNum = parentEdge->getInputNum();
                        if (inNum < 0 || inNum >= parent_spd->getConfig().outConfs.size()) {
                            inNum = 0;
                        }
                        auto& curDesc = getSupportedPrimitiveDescriptors()[i].getConfig().inConfs[j].desc;
                        auto& parentDesc = parent_spd->getConfig().outConfs[inNum].desc;

                        if (curDesc->isCompatible(*parentDesc)) {
                            equalsLocalFormatCount++;
                        }
                    }
                }
                if (equalsLocalFormatCount > equalsFormatCount) {
                    equalsFormatCount = equalsLocalFormatCount;
                    selectedPrimitive = static_cast<int>(i);
                }
            }
        }
        if (selectedPrimitive >= 0) {
            selectPrimitiveDescriptorByIndex(selectedPrimitive);
            return;
        }
    }

    if (getSupportedPrimitiveDescriptors().empty())
        IE_THROW() << "Supported primitive descriptors list is empty for node: " << getName();
    // fallback. If there are no primitives from priority list just select a first
    selectPrimitiveDescriptorByIndex(0);
}

bool MKLDNNNode::canBeInPlace() const {
    if (getParentEdges().size() != 1 || getParentEdgeAt(0)->getParent()->getChildEdges().size() != 1 ||
            (getParentEdgeAt(0)->getParent()->isConstant() && !getParentEdgeAt(0)->getChild()->isConstant()))
        return false;

    // TODO: we need to extend this logic to properly handle all possible inplace conflicts
    if (getParentEdges().size() == 1 && getParentEdgeAt(0)->getParent()->getType() == Reshape) {
        auto reshapeNode = getParentEdgeAt(0)->getParent();
        if (reshapeNode->getParentEdgeAt(0)->getParent()->getChildEdges().size() != 1)
            return false;
    }

    auto inShape = getParentEdgeAt(0)->getShape();
    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
        if (getChildEdgeAt(cIdx)->getShape() != inShape) {
            return false;
        }
    }
    return true;
}

void MKLDNNNode::resolveNotAllocatedEdges() {
    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (!selected_pd)
        IE_THROW() << "Cannot find selected primitive descriptor for node: " << getName();
    for (size_t i = 0; i < getParentEdges().size() && i < selected_pd->getConfig().inConfs.size(); i++) {
        auto parentEdge = getParentEdgeAt(i);

        if (parentEdge->getStatus() != MKLDNNEdge::Status::NotAllocated || selected_pd->getConfig().inConfs[i].inPlace < 0)
            continue;

        auto * memPtr = reinterpret_cast<char*>(parentEdge->getMemory().GetData());
        parentEdge->getMemoryPtr().reset(new MKLDNNMemory(getEngine()));
        parentEdge->getMemoryPtr()->Create(*selected_pd->getConfig().inConfs[i].desc, memPtr);

        parentEdge->changeStatus(MKLDNNEdge::Status::Allocated);
    }
    for (size_t i = 0; i < getChildEdges().size() && i < selected_pd->getConfig().outConfs.size(); i++) {
        auto childEdge = getChildEdgeAt(i);

        if (childEdge->getStatus() != MKLDNNEdge::Status::NotAllocated || selected_pd->getConfig().outConfs[i].inPlace < 0)
            continue;

        auto * memPtr = reinterpret_cast<char*>(childEdge->getMemory().GetData());
        childEdge->getMemoryPtr().reset(new MKLDNNMemory(getEngine()));
        childEdge->getMemoryPtr()->Create(*selected_pd->getConfig().outConfs[i].desc, memPtr);

        childEdge->changeStatus(MKLDNNEdge::Status::Allocated);
    }
}

std::string MKLDNNNode::getPrimitiveDescriptorType() {
    auto selectedPrimitiveDesc = getSelectedPrimitiveDescriptor();

    impl_desc_type type = impl_desc_type::undef;
    if (selectedPrimitiveDesc) {
        type = selectedPrimitiveDesc->getImplementationType();
    }

    std::string str_type;

    auto add_type = [&](std::string t) {
        if (!str_type.empty() && t.c_str()[0] != '_')
            str_type += "_";
        str_type += t;
    };

#define SEARCH_TYPE(_type)                                          \
    if ((type & impl_desc_type::_type) == impl_desc_type::_type)    \
        add_type(#_type)

    SEARCH_TYPE(undef);
    SEARCH_TYPE(reorder);
    SEARCH_TYPE(jit);
    SEARCH_TYPE(gemm);
    SEARCH_TYPE(ref);

    SEARCH_TYPE(avx512);
    SEARCH_TYPE(avx2);
    SEARCH_TYPE(avx);
    SEARCH_TYPE(sse42);
    SEARCH_TYPE(blas);
    SEARCH_TYPE(any);
    SEARCH_TYPE(uni);

    SEARCH_TYPE(winograd);
    SEARCH_TYPE(_dw);
    SEARCH_TYPE(_1x1);

    if (type == impl_desc_type::unknown)
        str_type = "unknown";
    else if (str_type.empty())
        str_type = "undef";

    // adding layer precision to the performance counters as one of the token
    // currently we treat a layer executing in int8 mode if its input is I8 or U8. if input is U8, we still
    // add I8 since I8 is special placeholder. The real calc precision might be quite complex and in most cases
    // it is mixed precision.
    if (selectedPrimitiveDesc) {
        if (!selectedPrimitiveDesc->getConfig().inConfs.empty()) {
            if (selectedPrimitiveDesc->getConfig().inConfs[0].desc->getPrecision() != InferenceEngine::Precision::U8) {
                str_type += "_" + std::string(selectedPrimitiveDesc->getConfig().inConfs[0].desc->getPrecision().name());
            } else {
                str_type += "_I8";
            }
        } else {
            if (selectedPrimitiveDesc->getConfig().outConfs[0].desc->getPrecision() != InferenceEngine::Precision::U8) {
                str_type += "_" + std::string(selectedPrimitiveDesc->getConfig().outConfs[0].desc->getPrecision().name());
            } else {
                str_type += "_I8";
            }
        }
    }

    return str_type;
}

const MKLDNNEdgePtr MKLDNNNode::getParentEdgeAt(size_t idx) const {
    if (idx >= parentEdges.size())
        IE_THROW() << "Node " << getName() << " contains less parent edges than " << idx;
    auto parentEdgePtr = parentEdges[idx].lock();
    if (!parentEdgePtr)
        IE_THROW() << "Node " << getName() << " contains empty parent edge for index " << idx;
    return parentEdgePtr;
}

const MKLDNNEdgePtr MKLDNNNode::getChildEdgeAt(size_t idx) const {
    if (idx >= childEdges.size())
        IE_THROW() << "Node " << getName() << " contains less child edges than " << idx;
    auto childEdgePtr = childEdges[idx].lock();
    if (!childEdgePtr)
        IE_THROW() << "Node " << getName() << " contains empty child edge for index " << idx;
    return childEdgePtr;
}

const std::vector<MKLDNNEdgePtr> MKLDNNNode::getParentEdgesAtPort(size_t idx) const {
    if (idx >= inputShapes.size())
        IE_THROW() << "Node " << getName() << " contains less input ports than " << idx;

    std::vector<MKLDNNEdgePtr> res;
    for (auto &edge_w : parentEdges) {
        auto edge = edge_w.lock();
        if (!edge)
            IE_THROW() << "Node " << getName() << " contains dead weak ptr";
        if (edge->getOutputNum() == idx) res.push_back(edge);
    }
    return res;
}

const std::vector<MKLDNNEdgePtr> MKLDNNNode::getChildEdgesAtPort(size_t idx) const {
    if (idx >= outputShapes.size())
        IE_THROW() << "Node " << getName() << " contains less output ports than " << idx;

    std::vector<MKLDNNEdgePtr> res;
    for (auto &edge_w : childEdges) {
        auto edge = edge_w.lock();
        if (!edge)
            IE_THROW() << "Node " << getName() << " contains dead weak ptr";
        if (edge->getInputNum() == idx) res.push_back(edge);
    }
    return res;
}


std::vector<memory::format_tag> MKLDNNNode::getAvailableFormatsForDims(const Shape &dims) const {
    if (dims.getRank() == 0)
        return {memory::format_tag::x};
    else if (dims.getRank() == 1)
        return {memory::format_tag::x};
    else if (dims.getRank() == 2)
        return {memory::format_tag::nc};
    else if (dims.getRank() == 3)
        return {memory::format_tag::tnc, memory::format_tag::ntc};
    else if (dims.getRank() == 4)
        return {memory::format_tag::nchw, memory::format_tag::nChw8c, memory::format_tag::nChw16c};
    else if (dims.getRank() == 5)
        return {memory::format_tag::ncdhw, memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c};
    return {memory::format_tag::any};
}

void MKLDNNNode::execute(mkldnn::stream strm) {
    if (prim) {
        (*prim).execute(strm, primArgs);
    }
}

void MKLDNNNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    for (auto& desc : descs) {
        auto itpd = desc.createPrimitiveDescriptorIterator(engine);

        while (static_cast<bool>(itpd)) {
            NodeConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace = -1;
                portConfig.constant = false;
                portConfig.desc = MemoryDescUtils::applyUndefinedOffset(*getSrcMemDesc(itpd, i));
                config.inConfs.push_back(portConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace = canBeInPlace() ? 0 : -1;
                portConfig.constant = false;
                portConfig.desc = MemoryDescUtils::applyUndefinedOffset(*getDstMemDesc(itpd, i));
                config.outConfs.push_back(portConfig);
            }
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}

void MKLDNNNode::filterSupportedPrimitiveDescriptors() {
    // Compare by partial layout descriptor (without particular strides values)
    auto areCompatible = [](const MemoryDesc& desc, mkldnn::memory::format_tag fmt) -> bool {
        MKLDNNMemoryDesc fmt_tdesc = MKLDNNMemoryDesc{
            MKLDNNDims(desc.getShape().getStaticDims()),
            MKLDNNExtensionUtils::IEPrecisionToDataType(desc.getPrecision()),
            fmt};

        return desc.isCompatible(fmt_tdesc);
// TODO [DS]: code clean up
//        auto tmp_partial_tdesc = PartialBlkDesc::extractFrom(fmt_tdesc);
//        auto actual_partial_tdesc = PartialBlkDesc::extractFrom(tdesc);
//        return tmp_partial_tdesc == actual_partial_tdesc;
    };

    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty()) {
        auto itpd = supportedPrimitiveDescriptors.begin();
        while (itpd != supportedPrimitiveDescriptors.end()) {
            const auto &config = itpd->getConfig();
            if (inputMemoryFormatsFilter.size() > config.inConfs.size() || outputMemoryFormatsFilter.size() > config.outConfs.size())
                IE_THROW() << "Incorrect number of input or output memory formats";

            bool isSuitableDesc = true;
            for (int i = 0; i < inputMemoryFormatsFilter.size(); i++) {
                const bool matched = areCompatible(*config.inConfs[i].desc, inputMemoryFormatsFilter[i]);
                isSuitableDesc &= matched;
            }
            for (int i = 0; i < outputMemoryFormatsFilter.size(); i++) {
                const bool matched = areCompatible(*config.outConfs[i].desc, outputMemoryFormatsFilter[i]);
                isSuitableDesc &= matched;
            }
            if (!isSuitableDesc) {
                itpd = supportedPrimitiveDescriptors.erase(itpd);
            } else {
                itpd++;
            }
        }
    }
}

void MKLDNNNode::initDescriptor(const NodeConfig& config) {
    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }
    std::vector<const MemoryDesc*> inDescs;
    for (const auto& inConf : config.inConfs)
        inDescs.push_back(inConf.desc.get());
    std::vector<const MemoryDesc*> outDescs;
    for (const auto& outConf : config.outConfs)
        outDescs.push_back(outConf.desc.get());
    createDescriptor(inDescs, outDescs);

    std::shared_ptr<mkldnn::primitive_attr> attr = initPrimitiveAttr();

    NodeConfig rightConfig = selectedPD->getConfig();
    size_t selected_count = 0;
    for (size_t j = 0; j < descs.size(); j++) {
        const auto &desc = descs[j];
        primitive_desc_iterator itpd;
        if (attr == nullptr) {
            itpd = desc.createPrimitiveDescriptorIterator(engine);
        } else {
            itpd = desc.createPrimitiveDescriptorIterator(engine, *(attr.get()));
        }
        while (static_cast<bool>(itpd)) {
            NodeConfig cfg;
            cfg.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                PortConfig dataConfig;
                dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                dataConfig.constant = false;
                dataConfig.desc = getSrcMemDesc(itpd, i);
                cfg.inConfs.push_back(dataConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = getDstMemDesc(itpd, i);
                cfg.outConfs.push_back(dataConfig);
            }
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
            if (selected_count == selectedPrimitiveDescriptorIndex) {
                if (impl_type != selectedPD->getImplementationType()) {
                    IE_THROW() << "Cannot get the original layer configuration!";
                }
                rightConfig = cfg;
            }
            if (j == descs.size() - 1) {
                if (impl_type == selectedPD->getImplementationType()) {
                    rightConfig = config;
                }
            }
            selected_count++;
            if (!itpd.next_impl())
                break;
        }
    }

    if (descs.empty()) {
        const auto& selectedConfig = selectedPD->getConfig();
        if (selectedConfig.inConfs.size() != config.inConfs.size() || selectedConfig.outConfs.size() != config.outConfs.size())
            return;

        for (size_t i = 0; i < selectedConfig.inConfs.size(); i++) {
            if (!selectedConfig.inConfs[i].desc->isCompatible(*config.inConfs[i].desc))
                IE_THROW() << "Incorrect descriptor for node: " << getName();
        }

        for (size_t i = 0; i < selectedConfig.outConfs.size(); i++) {
            if (!selectedConfig.outConfs[i].desc->isCompatible(*config.outConfs[i].desc))
                IE_THROW() << "Incorrect descriptor for node: " << getName();
        }
        rightConfig = config;
    }

    selectedPD->setConfig(rightConfig);
}

void MKLDNNNode::prepareMemory(const NodeDesc *selected_pd, mkldnn::primitive_desc_iterator& itpd) {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto &dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            IE_THROW() << "Destination memory didn't allocate for node " << getName()
                               << " to node " << getChildEdgeAt(i)->getChild()->getName() << ".";
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto &srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            IE_THROW() << "Destination memory didn't allocate for node " << getName()
                               << " from node " << getParentEdgeAt(i)->getParent()->getName() << ".";
    }
    std::vector<MKLDNNMemoryDesc> intDescs;
    for (auto &it : internalBlobDesc)
        intDescs.push_back(it(itpd, 0));

    internalBlobMemory.clear();
    for (size_t i = 0; i < internalBlobs.size(); i++) {
        const auto &internalBlob = internalBlobs[i];

        auto create = [&] () {
            // TODO [DS]: internal blobs should be removed or rewritten using Memory object
            auto newDesc = MemoryDescUtils::convertToMKLDNNMemoryDesc(internalBlob->getTensorDesc());

            MKLDNNMemory memory{ engine };
            memory.Create(newDesc, internalBlob->buffer());

            MKLDNNMemoryPtr _ptr = MKLDNNMemoryPtr(new MKLDNNMemory(engine));
            _ptr->Create(intDescs[i]);
            _ptr->SetData(memory);

            return _ptr;
        };

        MKLDNNMemoryPtr ptr;
        if (weightCache != nullptr) {
            const uint64_t data_hash = weightCache->GetHashFunc().hash(
                    internalBlob->buffer(), internalBlob->byteSize());

            const std::string string_hash = name + "_" + std::to_string(i)
                                            + "_" + std::to_string(internalBlob->byteSize())
                                            + "_" + std::to_string(data_hash);

            ptr = *weightCache->findOrCreate(string_hash, create);
        } else {
            ptr = create();
        }

        internalBlobMemory.push_back(ptr);
    }
}

bool MKLDNNNode::isInplace() const {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();

    for (auto &in : config.inConfs) if (in.inPlace >= 0) return true;
    for (auto &out : config.outConfs) if (out.inPlace >= 0) return true;
    return false;
}

bool MKLDNNNode::isConstant() {
    if (constant == ConstantType::Unknown) {
        std::vector<MKLDNNNodePtr> checkNodes;
        for (size_t i = 0; i < getChildEdges().size(); i++) {
            checkNodes.push_back(getChildEdgeAt(i)->getChild());
        }
        while (constant != ConstantType::NoConst && !checkNodes.empty()) {
            constant = checkNodes.front()->checkConstant(LOOK_DOWN, checkNodes);
            checkNodes.erase(checkNodes.begin());
        }
        if (constant != ConstantType::Const) {
            constant = ConstantType::Unknown;
            checkNodes.clear();
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                checkNodes.push_back(getParentEdgeAt(i)->getParent());
            }
            while (constant != ConstantType::NoConst && !checkNodes.empty()) {
                constant = checkNodes.front()->checkConstant(LOOK_UP, checkNodes);
                checkNodes.erase(checkNodes.begin());
            }
        }
        if (constant == ConstantType::Unknown)
            constant = ConstantType::NoConst;
    }
    return constant == ConstantType::Const;
}

MKLDNNNode::ConstantType MKLDNNNode::checkConstant(LOOK look, std::vector<MKLDNNNodePtr>& checkNodes) {
    if (constant == ConstantType::Unknown) {
        if (look == LOOK_DOWN) {
            for (size_t i = 0; i < getChildEdges().size(); i++) {
                if (std::find(checkNodes.begin(), checkNodes.end(), getChildEdgeAt(i)->getChild()) == checkNodes.end())
                    checkNodes.push_back(getChildEdgeAt(i)->getChild());
            }
        } else {
            for (size_t i = 0; i < getParentEdges().size(); i++) {
                if (std::find(checkNodes.begin(), checkNodes.end(), getParentEdgeAt(i)->getParent()) == checkNodes.end())
                    checkNodes.push_back(getParentEdgeAt(i)->getParent());
            }
        }
    }
    return constant;
}

void MKLDNNNode::addOriginalLayer(const std::string& layerName) {
    if (layerName.empty()) return;
    if (originalLayers.empty()) {
        originalLayers = layerName;
    } else {
        originalLayers += "," + layerName;
    }
}

void MKLDNNNode::cleanup() {
    internalBlobs.clear();

    for (auto it : fusedWith) {
        it->cleanup();
    }

    for (auto it : mergedWith) {
        it->cleanup();
    }
}

const std::vector<impl_desc_type>& MKLDNNNode::getPrimitivesPriority() {
    std::vector<impl_desc_type> priorities = {
            impl_desc_type::unknown,
            impl_desc_type::jit_uni_dw,
            impl_desc_type::jit_uni_1x1,
            impl_desc_type::jit_uni,
            impl_desc_type::jit_avx512_dw,
            impl_desc_type::jit_avx512_1x1,
            impl_desc_type::jit_avx512,
            impl_desc_type::jit_avx2_dw,
            impl_desc_type::jit_avx2_1x1,
            impl_desc_type::jit_avx2,
            impl_desc_type::jit_avx_dw,
            impl_desc_type::jit_avx_1x1,
            impl_desc_type::jit_avx,
            impl_desc_type::jit_sse42_dw,
            impl_desc_type::jit_sse42_1x1,
            impl_desc_type::jit_sse42,
            impl_desc_type::gemm_any,
            impl_desc_type::gemm_blas,
            impl_desc_type::gemm_avx512,
            impl_desc_type::gemm_avx2,
            impl_desc_type::gemm_avx,
            impl_desc_type::gemm_sse42,
            impl_desc_type::jit_gemm,
            impl_desc_type::ref_any,
            impl_desc_type::ref,
    };
    for (const auto& impl : priorities) {
        if (std::find(implPriorities.begin(), implPriorities.end(), impl) == implPriorities.end())
            implPriorities.push_back(impl);
    }
    return implPriorities;
}

std::unique_ptr<MemoryDesc> MKLDNNNode::getDefinedInputDesc(const NodeConfig &config, size_t idx) const {
    int num = getParentEdgeAt(idx)->getInputNum();
    auto *selectedPD = getParentEdgeAt(idx)->getParent()->getSelectedPrimitiveDescriptor();
    if (!selectedPD)
        IE_THROW() << "Cannot get selected primitive descriptor for node: " << getParentEdgeAt(idx)->getParent()->getName();

    if (config.inConfs[idx].desc->isDefined()) {
        return config.inConfs[idx].desc->clone();
    }

    if (config.inConfs[idx].inPlace >= 0) {
        return getDefinedOutputDesc(config, static_cast<size_t>(config.inConfs[idx].inPlace));
    }

    if (num >= 0) {
        auto parentConf = selectedPD->getConfig().outConfs[num];
        parentConf.desc->setPrecision(config.inConfs[idx].desc->getPrecision());
        if (!parentConf.desc->isDefined() && parentConf.inPlace >= 0)
            getParentEdgeAt(idx)->getParent()->initOptimalPrimitiveDescriptor();
        parentConf = getParentEdgeAt(idx)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num];
        if (parentConf.desc->isDefined() && parentConf.desc->isCompatible(*config.inConfs[idx].desc)) {
            return parentConf.desc->clone();
        }
    }

    return MemoryDescUtils::resetOffset(config.inConfs[idx].desc.get());
}

std::unique_ptr<MemoryDesc> MKLDNNNode::getDefinedOutputDesc(const NodeConfig &config, size_t idx) const {
    int num = getChildEdgeAt(idx)->getOutputNum();
    auto *selectedPD = getChildEdgeAt(idx)->getChild()->getSelectedPrimitiveDescriptor();
    if (!selectedPD)
        IE_THROW() << "Cannot get selected primitive descriptor for node: " << getChildEdgeAt(idx)->getChild()->getName();

    if (config.outConfs[idx].desc->isDefined()) {
        return config.outConfs[idx].desc->clone();
    }

    if (config.outConfs[idx].inPlace >= 0) {
        return getDefinedInputDesc(config, static_cast<size_t>(config.outConfs[idx].inPlace));
    }

    if (num >= 0) {
        auto childConf = selectedPD->getConfig().inConfs[num];
        childConf.desc->setPrecision(config.outConfs[idx].desc->getPrecision());
        if (!childConf.desc->isDefined() && childConf.inPlace >= 0)
            getChildEdgeAt(idx)->getChild()->initOptimalPrimitiveDescriptor();
        childConf = getChildEdgeAt(idx)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num];
        if (childConf.desc->isDefined() && childConf.desc->isCompatible(*config.outConfs[idx].desc)) {
            return childConf.desc->clone();
        }
    }

    return MemoryDescUtils::resetOffset(config.outConfs[idx].desc.get());
}

void MKLDNNNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (!isConfigDefined(config)) {
        for (size_t i = 0; i < config.inConfs.size(); i++) {
            config.inConfs[i].desc = getDefinedInputDesc(config, i);
        }

        for (size_t i = 0; i < config.outConfs.size(); i++) {
            config.outConfs[i].desc = getDefinedOutputDesc(config, i);
        }

        initDescriptor(config);
    } else if (getType() != RNNSeq && getType() != RNNCell) {
        initDescriptor(config);
    }
}

bool MKLDNNNode::isConfigDefined(const NodeConfig &config) const {
    for (const auto& configs : {config.inConfs, config.outConfs}) {
        for (const auto &dc : configs) {
            if (!dc.desc->isDefined())
                return false;
        }
    }
    return true;
}

std::unique_ptr<MKLDNNMemoryDesc> MKLDNNNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    return MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(primitive_desc_it.src_desc(idx));
    // TODO [DS]: uncomment or remove
//    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.src_desc(idx));
//    if (desc.getLayout() == InferenceEngine::Layout::ANY)
//        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
//                                                            getParentEdgeAt(idx)->getShape().getDims(),
//                                                            desc.getLayout()));
//    else
//        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
//                                                            getParentEdgeAt(idx)->getShape().getDims(),
//                                                            desc.getBlockingDesc()));
}

std::unique_ptr<MKLDNNMemoryDesc> MKLDNNNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    return MKLDNNPlugin::make_unique<MKLDNNMemoryDesc>(primitive_desc_it.dst_desc(idx));
    // TODO [DS]: uncomment or remove
//    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.dst_desc(idx));
//    if (desc.getLayout() == InferenceEngine::Layout::ANY)
//        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
//                                                            getChildEdgeAt(idx)->getShape().getDims(),
//                                                            desc.getLayout()));
//    else
//        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
//                                                            getChildEdgeAt(idx)->getShape().getDims(),
//                                                            desc.getBlockingDesc()));
}

int MKLDNNNode::batchToProcess() {
    return dynBatchLim == 0 ? getMaxBatch() : std::min<int>(getMaxBatch(), dynBatchLim);
}

int MKLDNNNode::getMaxBatch() {
    // FIXME: batch != 0 dims number
    if (!inputShapes.empty()) {
        if (inputShapes[0].getRank())
            return static_cast<int>(inputShapes[0].getStaticDims()[0]);
        else
            return 1;
    }
    if (!outputShapes.empty()) {
        if (outputShapes[0].getRank())
            return static_cast<int>(outputShapes[0].getStaticDims()[0]);
        else
            return 1;
    }
    return 0;
}

void MKLDNNNode::setDynamicBatchLim(int lim) {
    dynBatchLim = lim;

    auto setDynamicBatch = [this](int argType, int newBatch) {
        auto param = primArgs.find(argType);
        if (param != primArgs.end()) {
            auto oldMem = param->second;
            mkldnn::memory::desc newMemDesc(oldMem.get_desc());
            newMemDesc.data.dims[0] = newBatch;
            newMemDesc.data.padded_dims[0] = newBatch;
            mkldnn::memory newMem(newMemDesc, oldMem.get_engine(), oldMem.get_data_handle());
            primArgs.at(argType) = newMem;
        }
    };

    if (!primArgs.empty()) {
        int newBatch = batchToProcess();
        setDynamicBatch(DNNL_ARG_SRC, newBatch);
        setDynamicBatch(DNNL_ARG_DST, newBatch);
        setDynamicBatch(DNNL_ARG_DIFF_SRC, newBatch);
        setDynamicBatch(DNNL_ARG_DIFF_DST, newBatch);
    }
}

bool MKLDNNNode::isFusedWith(Type fusedNodeType) const {
    for (auto fusedNode : fusedWith) {
        if (fusedNode->type == fusedNodeType)
            return true;
    }

    return false;
}

Layout MKLDNNNode::getWeightsLayoutByDims(SizeVector dims, bool isGrouped) {
    switch (dims.size()) {
        case 0:
            return Layout::SCALAR;
        case 1:
            return Layout::C;
        case 2:
            return Layout::NC;
        case 3:
            return Layout::CHW;
        case 4:
            return Layout::OIHW;
        case 5:
            return isGrouped ? Layout::GOIHW : Layout::OIDHW;
        case 6:
            return isGrouped ? Layout::GOIDHW : Layout::BLOCKED;
        default:
            return Layout::BLOCKED;
    }
}

void MKLDNNNode::appendPostOps(mkldnn::post_ops& ops) {
    IE_THROW() << "Fusing of " << this->getType() << " operation is not implemented";
}

std::vector<InferenceEngine::Precision> MKLDNNNode::getInputPrecisions() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == MKLDNNEdge::Status::Validated) {
            inputPrecisions.emplace_back(MKLDNNExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }
    return inputPrecisions;
}

std::vector<InferenceEngine::Precision> MKLDNNNode::getOutputPrecisions() const {
    std::vector<InferenceEngine::Precision> outputPrecisions;
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto childEdge = getChildEdgeAt(i);
        if (childEdge && childEdge->getStatus() == MKLDNNEdge::Status::Validated) {
            outputPrecisions.emplace_back(MKLDNNExtensionUtils::DataTypeToIEPrecision((childEdge->getMemoryPtr()->GetDataType())));
        }
    }
    return outputPrecisions;
}

InferenceEngine::Precision MKLDNNNode::getRuntimePrecision() const {
    // Base implementation consider precision only on data path and
    // assumes it is placed on 0-th port (which is true for almost all layers)
    InferenceEngine::Precision runtimePrecision = Precision::UNSPECIFIED;
    auto inputPrecisions = getInputPrecisions();
    if (!inputPrecisions.empty()) {
        runtimePrecision = inputPrecisions[0];
    } else {
        auto outputPrecisions = getOutputPrecisions();
        if (!outputPrecisions.empty()) {
            runtimePrecision = outputPrecisions[0];
        }
    }

    return runtimePrecision;
}

MKLDNNNode* MKLDNNNode::NodesFactory::create(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng,
                                             const MKLDNNExtensionManager::Ptr& extMgr, MKLDNNWeightsSharing::Ptr &w_cache) {
    MKLDNNNode *newNode = nullptr;
    std::string errorMessage;
    try {
        std::unique_ptr<MKLDNNNode> ol(createNodeIfRegistered(MKLDNNPlugin, Generic, op, eng, w_cache));
        if (ol != nullptr && ol->created(extMgr))
            newNode = ol.release();
    } catch (const InferenceEngine::Exception& ex) {
        IE_SUPPRESS_DEPRECATED_START
        if (ex.getStatus() != NOT_IMPLEMENTED) {
            throw;
        } else {
            errorMessage += getExceptionDescWithoutStatus(ex);
        }
        IE_SUPPRESS_DEPRECATED_END
    }

    if (newNode == nullptr) {
        try {
            std::unique_ptr<MKLDNNNode> ol(createNodeIfRegistered(MKLDNNPlugin, TypeFromName(op->get_type_name()), op, eng, w_cache));
            if (ol != nullptr && ol->created(extMgr))
                newNode = ol.release();
        } catch (const InferenceEngine::Exception& ex) {
            IE_SUPPRESS_DEPRECATED_START
            if (ex.getStatus() != NOT_IMPLEMENTED) {
                throw;
            } else {
                errorMessage += getExceptionDescWithoutStatus(ex);
            }
            IE_SUPPRESS_DEPRECATED_END
        }
    }

    if (newNode == nullptr) {
        try {
            std::unique_ptr<MKLDNNNode> ol(new MKLDNNReferenceNode(op, eng, w_cache, errorMessage));
            if (ol != nullptr && ol->created(extMgr))
                newNode = ol.release();
        } catch (const InferenceEngine::Exception& ex) {
            IE_SUPPRESS_DEPRECATED_START
            if (ex.getStatus() != NOT_IMPLEMENTED) {
                throw;
            } else {
                errorMessage += getExceptionDescWithoutStatus(ex);
            }
            IE_SUPPRESS_DEPRECATED_END
        }
    }

    //  WA-start : TI node requires all attributes to construct internal subgpath
    //             including extManager, socket and mkldnn::eng.
    MKLDNNTensorIteratorNode *ti = dynamic_cast<MKLDNNTensorIteratorNode*>(newNode);
    if (ti != nullptr)
        ti->setExtManager(extMgr);
    //  WA-end

    if (!newNode) {
        std::string errorDetails;
        if (!errorMessage.empty()) {
            errorDetails = "\nDetails: \n" + errorMessage;
        }
        IE_THROW() << "Unsupported operation of type: " << op->get_type_name() << " name: " << op->get_friendly_name() << errorDetails;
    }

    return newNode;
}

bool MKLDNNNode::canBePerformedAsScaleShift(const MKLDNNNode *parentNode) const {
    size_t fusingPort = 0;
    for (size_t i = (parentNode == nullptr ? 1 : 0); i < getParentEdges().size(); i++) {
        MKLDNNNode *node = getParentEdgeAt(i)->getParent().get();
        if (node == nullptr) {
            IE_THROW() << "Cannot get parent node for " << getName() << " on " << i << " port";
        }
        if (node == parentNode) {
            fusingPort = i;
            continue;
        }
        if (node->getType() != Input || !node->isConstant()) {
            return false;
        }
    }

    const auto isBroadcastableToDataInput = [&]() {
        const auto dataShape = getParentEdgeAt(fusingPort)->getShape().getStaticDims();
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            if (i == fusingPort)
                continue;
            auto weightShape = getParentEdgeAt(i)->getShape().getStaticDims();
            if (!isPerTensorOrPerChannelBroadcastable(dataShape, weightShape))
                return false;
        }
        return true;
    };

    const auto isConvertablePowerStatic = [&]() {
        if (getAlgorithm() == EltwisePowerStatic) {
            const auto eltwise = dynamic_cast<const MKLDNNEltwiseNode *>(this);
            if (!eltwise) {
                IE_THROW() << "Cannot cast " << getName() << " to MKLDNNEltwiseNode";
            }
            return eltwise->getAlpha() == 1.0f;
        }
        return false;
    };

    return (one_of(getAlgorithm(), EltwiseAdd, EltwiseMultiply, EltwiseSubtract, EltwiseDivide, EltwisePrelu, EltwiseMulAdd) && isBroadcastableToDataInput())
            || isConvertablePowerStatic();
}

bool MKLDNNNode::canFuseSimpleOperation(const MKLDNNNodePtr& node) const {
    if (node->getType() == FakeQuantize) {
        return node->getAlgorithm() != FQBinarization;
    } else if (node->getType() == Eltwise) {
        return one_of(node->getAlgorithm(), EltwiseRelu, EltwiseGelu, EltwiseElu, EltwiseSigmoid, EltwiseClamp, EltwiseTanh,
                                            EltwiseSwish, EltwiseHswish, EltwiseMish, EltwiseHsigmoid, EltwiseRoundHalfToEven,
                                            EltwiseRoundHalfAwayFromZero, EltwiseAbs, EltwiseSqrt, EltwiseSoftRelu) ||
                      node->canBePerformedAsScaleShift(this);
    }
    return false;
}

void MKLDNNNode::fillScalesAndShifts(const MKLDNNNode *parentNode, std::vector<float> &scales, std::vector<float> &shifts, int align) {
    scales.clear();
    shifts.clear();
    const auto fillValuesFrom = [&](const MKLDNNNodePtr& constInput, std::vector<float>& buffer) {
        auto *constInputNode = dynamic_cast<MKLDNNInputNode *>(constInput.get());
        auto constBlob = constInputNode->getMemoryPtr();
        auto const elementsCount = constBlob->GetElementsCount();
        buffer.resize(elementsCount);
        cpu_convert(constBlob->GetPtr(),
                    &buffer[0],
                    MKLDNNExtensionUtils::DataTypeToIEPrecision(constBlob->GetDataType()),
                    Precision::FP32,
                    elementsCount);
    };

    const size_t constPort = getParentEdgesAtPort(0)[0]->getParent().get() == parentNode ? 1 : 0;

    if (one_of(getAlgorithm(), EltwiseMultiply, EltwiseDivide, EltwisePrelu)) {
        fillValuesFrom(getParentEdgesAtPort(constPort)[0]->getParent(), scales);
    } else if (one_of(getAlgorithm(), EltwiseAdd, EltwiseSubtract)) {
        fillValuesFrom(getParentEdgesAtPort(constPort)[0]->getParent(), shifts);
    } else if (one_of(getAlgorithm(), EltwiseMulAdd)) {
        fillValuesFrom(getParentEdgesAtPort(1)[0]->getParent(), scales);
        fillValuesFrom(getParentEdgesAtPort(2)[0]->getParent(), shifts);
    } else if (one_of(getAlgorithm(), EltwisePowerStatic)) {
        const auto power = dynamic_cast<const MKLDNNEltwiseNode *>(this);
        if (!power) {
            IE_THROW() << "Cannot cast " << getName() << " to MKLDNNEltwiseNode";
        }
        scales.push_back(power->getBeta());
        shifts.push_back(power->getGamma());
    } else {
        IE_THROW() << "Can't fill scale and shifts for node: " << getName() << " with type: " << NameFromType(getType());
    }

    const size_t bufferSize = static_cast<size_t>(outputShapes[0].getStaticDims()[outputShapes[0].getRank() > 1 ? 1 : 0]);
    if (align == -1) {
        align = bufferSize;
    }
    const size_t bufferSizeAligned = rnd_up(bufferSize, static_cast<size_t>(align));

    size_t initSize = scales.size();
    if (initSize > 0) {
        scales.resize(bufferSizeAligned, 0);
        if (initSize == 1) {
            std::fill(scales.begin() + 1, scales.begin() + bufferSize, scales[0]);
        }
    }

    initSize = shifts.size();
    if (initSize > 0) {
        shifts.resize(bufferSizeAligned, 0);
        if (initSize == 1) {
            std::fill(shifts.begin() + 1, shifts.begin() + bufferSize, shifts[0]);
        }
    }

    switch (getAlgorithm()) {
        case EltwiseAdd: {
            scales.resize(bufferSizeAligned, 1.0f);
            break;
        }
        case EltwiseSubtract: {
            scales.resize(bufferSizeAligned, 1.0f);
            std::transform(shifts.begin(), shifts.end(), shifts.begin(), [](float shift){ return -1.0f * shift; });
            break;
        }
        case EltwiseMultiply: {
            shifts.resize(bufferSizeAligned, 0.0f);
            break;
        }
        case EltwiseDivide: {
            shifts.resize(bufferSizeAligned, 0.0f);
            std::transform(scales.begin(), scales.end(), scales.begin(), [](float scale){ return 1.0f / scale; });
            break;
        }
        default: break;
    }
}

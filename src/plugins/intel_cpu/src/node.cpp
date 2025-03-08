// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node.h"

#include <dnnl_debug.h>
#include <dnnl_types.h>

#include <common/primitive_desc.hpp>
#include <common/primitive_desc_iface.hpp>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <openvino/opsets/opset1.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "edge.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/conv.h"
#include "nodes/eltwise.h"
#include "nodes/input.h"
#include "nodes/reference.h"
#include "nodes/reorder.h"
#include "openvino/core/type/element_type.hpp"
#include "partitioned_mem_blk.h"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"
#include "utils/ngraph_utils.hpp"
#include "utils/rt_info/memory_formats_attribute.hpp"

using namespace dnnl;
using namespace openvino;
using namespace ov::intel_cpu::node;

namespace ov::intel_cpu {

Node::NodesFactory& Node::factory() {
    static NodesFactory factoryInstance;
    return factoryInstance;
}

Node::Node(const std::shared_ptr<ov::Node>& op, GraphContext::CPtr ctx, const ShapeInferFactory& shapeInferFactory)
    : context(std::move(ctx)),

      fusingPort(-1),
      engine(context->getEngine()),
      name(op->get_friendly_name()),
      typeStr(op->get_type_name()),
      type(TypeFromName(op->get_type_name())),
      profiling(op->get_friendly_name()) {
    for (size_t i = 0; i < op->get_input_size(); i++) {
        const auto& shape = op->get_input_partial_shape(i);
        if (shape.rank().is_dynamic()) {
            OPENVINO_THROW("Unexpected: CPU plug-in doesn't support ",
                           getTypeStr(),
                           " operation with dynamic rank. Operation name: ",
                           getName());
        }

        bool isScalar = shape.rank().get_length() == 0;
        inputShapes.emplace_back(isScalar ? ov::PartialShape{1} : shape);
        originalInputPrecisions.emplace_back(op->get_input_element_type(i));
    }

    parentEdges.reserve(inputShapes.size());

    if (typeStr != "Result" && typeStr != "Assign") {
        if (op->get_output_size() == 0) {
            OPENVINO_THROW("Node with type '", typeStr, "' and name '", name, "' does not have any outputs.");
        }
        for (size_t i = 0; i < op->get_output_size(); i++) {
            const auto& shape = op->get_output_partial_shape(i);
            if (shape.rank().is_dynamic()) {
                OPENVINO_THROW("Unexpected: CPU plug-in doesn't support ",
                               getTypeStr(),
                               " operation with dynamic rank. Operation name: ",
                               getName());
            }

            bool isScalar = shape.rank().get_length() == 0;
            outputShapes.emplace_back(isScalar ? ov::PartialShape{1} : shape);
            originalOutputPrecisions.emplace_back(op->get_output_element_type(i));
        }

        childEdges.reserve(outputShapes.size());
    }

    isDynamic = std::any_of(inputShapes.begin(),
                            inputShapes.end(),
                            [](const Shape& shape) {
                                return shape.isDynamic();
                            }) ||
                std::any_of(outputShapes.begin(), outputShapes.end(), [](const Shape& shape) {
                    return shape.isDynamic();
                });

    if (isDynamic) {
        shapeInference = shapeInferFactory.makeShapeInfer();
    }

    const auto& rtInfo = op->get_rt_info();
    if (rtInfo.count("originalLayersNames")) {
        originalLayers = getRTInfoValue(rtInfo, "originalLayersNames");
    }

    if (rtInfo.count("parallelDomain")) {
        parallelDomain = getRTInfoValue(rtInfo, "parallelDomain");
    }

    if (originalLayers.empty()) {
        addOriginalLayer(name);
    }

    primitivesPriority = getImplPriorityValue(op);
    if (!primitivesPriority.empty()) {
        std::istringstream stream(primitivesPriority);
        std::string str;
        while (getline(stream, str, ',')) {
            if (str.substr(0, 4) != "cpu:") {
                continue;
            }
            customImplPriorities.push_back(parse_impl_name(str));
            if (customImplPriorities.back() == impl_desc_type::unknown && str != "cpu:unknown") {
                OPENVINO_THROW("Unsupported CPU implementation ", str, " for node ", getName());
            }
        }
        const auto& defaultImplPriorities = getDefaultImplPriority();
        customImplPriorities.insert(customImplPriorities.end(),
                                    defaultImplPriorities.begin(),
                                    defaultImplPriorities.end());
    }

    std::string inputMemoryFormats = getInputMemoryFormats(op);
    if (!inputMemoryFormats.empty()) {
        std::istringstream stream(inputMemoryFormats);
        std::string str;
        while (getline(stream, str, ',')) {
            if (str.substr(0, 4) != "cpu:") {
                continue;
            }
            inputMemoryFormatsFilter.push_back(dnnl::utils::str2fmt(str.substr(4, str.size()).c_str()));
        }
    }

    std::string outputMemoryFormats = getOutputMemoryFormats(op);
    if (!outputMemoryFormats.empty()) {
        std::istringstream stream(outputMemoryFormats);
        std::string str;
        while (getline(stream, str, ',')) {
            if (str.substr(0, 4) != "cpu:") {
                continue;
            }
            outputMemoryFormatsFilter.push_back(dnnl::utils::str2fmt(str.substr(4, str.size()).c_str()));
        }
    }

    const auto it = rtInfo.find("enforceBF16evenForGraphTail");
    if (it != rtInfo.end()) {
        enforceBF16evenForGraphTail = it->second.as<bool>();
    }
    if (ov::fp16_compression_is_disabled(op)) {
        keepOriginalPrecision = true;
    }
}

Node::Node(const std::string& type,
           std::vector<Shape> inShapes,
           std::vector<Shape> outShapes,
           std::vector<ov::element::Type> inputPrecisions,
           std::vector<ov::element::Type> outputPrecisions,
           const std::string& name,
           const GraphContext::CPtr& ctx)
    : inputShapes(std::move(inShapes)),
      outputShapes(std::move(outShapes)),

      context(ctx),
      originalInputPrecisions(std::move(inputPrecisions)),
      originalOutputPrecisions(std::move(outputPrecisions)),
      fusingPort(-1),
      engine(ctx->getEngine()),
      name(name),
      typeStr(type),
      type(TypeFromName(type)),
      profiling(name) {
    parentEdges.reserve(inputShapes.size());
    childEdges.reserve(outputShapes.size());
}

void Node::addEdge(const EdgePtr& edge) {
    auto parent = edge->getParent();
    auto child = edge->getChild();
    assert(parent && child);

    parent->addChildEdge(edge);
    child->addParentEdge(edge);
}

void Node::remove() {
    auto drop = [](const std::vector<EdgeWeakPtr>& edges) {
        for (auto& edge : edges) {
            auto edgePtr = edge.lock();
            if (!edgePtr) {
                continue;
            }
            edgePtr->getParent()->removeChildEdge(edgePtr);
            edgePtr->getChild()->removeParentEdge(edgePtr);
        }
    };

    drop(parentEdges);
    drop(childEdges);
}

bool Node::isEdgesEmpty(const std::vector<EdgeWeakPtr>& edges) const {
    for (auto& edge : edges) {
        if (edge.lock()) {
            return false;
        }
    }
    return true;
}

void Node::createPrimitive() {
    if (inputShapesDefined() && isExecutable()) {
        if (needPrepareParams()) {
            prepareParams();
        }
        updateLastInputDims();
    }
}

void Node::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getImplPriority(), false);
}

void Node::selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority, bool ignoreConstInputs) {
    for (auto& type : priority) {
        int selectedPrimitive = -1;
        int equalsFormatCount = -1;
        for (size_t i = 0; i < getSupportedPrimitiveDescriptors().size(); i++) {
            const auto& supportedPrimitiveDesc = getSupportedPrimitiveDescriptors()[i];
            const impl_desc_type supportedType = supportedPrimitiveDesc.getImplementationType();

            if (supportedType != type) {
                continue;
            }

            int equalsLocalFormatCount = 0;
            const size_t descInConfSize = supportedPrimitiveDesc.getConfig().inConfs.size();

            if (descInConfSize > getParentEdges().size()) {
                OPENVINO_THROW(getName(),
                               " Desc ",
                               i,
                               " with type: ",
                               supportedType,
                               " has more input ports than node: ",
                               descInConfSize,
                               " vs ",
                               getParentEdges().size());
                continue;
            }

            for (size_t j = 0; j < descInConfSize; j++) {
                auto parentEdge = getParentEdgeAt(j);
                auto parentPtr = parentEdge->getParent();

                // We don't take into account constant edges since reorders on them will be executed on load network
                // stage
                if (ignoreConstInputs && j > 0 && parentPtr->isConstant()) {
                    equalsLocalFormatCount++;
                    continue;
                }

                auto parent_spd = parentPtr->getSelectedPrimitiveDescriptor();

                if (parent_spd != nullptr && !parent_spd->getConfig().outConfs.empty()) {
                    int inNum = parentEdge->getInputNum();
                    if (inNum < 0 || inNum >= static_cast<int>(parent_spd->getConfig().outConfs.size())) {
                        inNum = 0;
                    }
                    auto curDesc = supportedPrimitiveDesc.getConfig().inConfs[j].getMemDesc();
                    auto parentDesc = parent_spd->getConfig().outConfs[inNum].getMemDesc();

                    const bool isCompatible = curDesc->isCompatible(*parentDesc);
                    if (isCompatible) {
                        equalsLocalFormatCount++;
                    }

                    DEBUG_LOG(getName(),
                              " pd[",
                              i,
                              "].inConfs[",
                              j,
                              "]"
                              " is ",
                              (isCompatible ? "compatible" : "not compatible"),
                              " with parent ",
                              parentPtr->getName(),
                              " outConfs[",
                              inNum,
                              "], equalsLocalFormatCount add to ",
                              equalsLocalFormatCount);
                }

                if (equalsLocalFormatCount > equalsFormatCount) {
                    equalsFormatCount = equalsLocalFormatCount;
                    selectedPrimitive = static_cast<int>(i);
                    DEBUG_LOG(getName(), " Select primitive desc: ", i, " ", supportedPrimitiveDesc);
                }
            }
        }

        if (selectedPrimitive >= 0) {
            selectPrimitiveDescriptorByIndex(selectedPrimitive);
            return;
        }
    }

    OPENVINO_ASSERT(!getSupportedPrimitiveDescriptors().empty(),
                    "Supported primitive descriptors list is empty for node: ",
                    getName(),
                    " type: ",
                    NameFromType(getType()));

    // fallback. If there are no primitives from priority list just select a first
    selectPrimitiveDescriptorByIndex(0);
}

bool Node::isOneDimShape(const ov::PartialShape& pshape) {
    int value_1_num = 0;
    auto sz = static_cast<int>(pshape.size());
    for (const auto& s : pshape) {
        if (s.is_static() && s.get_length() == 1) {
            value_1_num++;
        }
    }
    return value_1_num >= sz - 1;
}

bool Node::isReorderRequired(const ov::intel_cpu::MemoryDescPtr& desc1, const ov::intel_cpu::MemoryDescPtr& desc2) {
    bool samePrec = desc1->getPrecision() == desc2->getPrecision();
    bool isOneDimShape1 = isOneDimShape(desc1->getShape().toPartialShape());
    bool isOneDimShape2 = isOneDimShape(desc2->getShape().toPartialShape());
    return !(isOneDimShape1 && isOneDimShape2 && samePrec);
}

void Node::selectPreferPrimitiveDescriptorWithShape(const std::vector<impl_desc_type>& priority,
                                                    bool ignoreConstInputs) {
    // Filter out dynamic shape.
    if (isDynamic) {
        return selectPreferPrimitiveDescriptor(priority, ignoreConstInputs);
    }

    auto estimateReorderOverhead = [&](const ov::intel_cpu::NodeDesc& supportedPrimitiveDesc, size_t i) {
        int estimate = 0;
        auto inputNodesNum = supportedPrimitiveDesc.getConfig().inConfs.size();
        for (size_t j = 0; j < inputNodesNum; j++) {
            auto parentEdge = getParentEdgeAt(j);
            auto parentPtr = parentEdge->getParent();

            // We don't take into account constant edges since reorders on them will be executed on load network
            // stage
            if (ignoreConstInputs && j > 0 && parentPtr->isConstant()) {
                continue;
            }

            auto parent_spd = parentPtr->getSelectedPrimitiveDescriptor();
            if (parent_spd != nullptr && !parent_spd->getConfig().outConfs.empty()) {
                int inNum = parentEdge->getInputNum();
                if (inNum < 0 || inNum >= static_cast<int>(parent_spd->getConfig().outConfs.size())) {
                    inNum = 0;
                }
                auto curDesc = supportedPrimitiveDesc.getConfig().inConfs[j].getMemDesc();
                auto parentDesc = parent_spd->getConfig().outConfs[inNum].getMemDesc();

                const bool isCompatible = curDesc->isCompatible(*parentDesc);
                if (!isCompatible) {
                    if (!isReorderRequired(parentDesc, curDesc)) {
                        estimate += 1;
                    } else {
                        estimate += ov::shape_size<ov::intel_cpu::VectorDims>(curDesc->getShape().getMinDims());
                    }
                }

                DEBUG_LOG(getName(),
                          " pd[",
                          i,
                          "].inConfs[",
                          j,
                          "]"
                          " is ",
                          (isCompatible ? "compatible" : "not compatible"),
                          " shape is ",
                          (isOneDimShape(curDesc->getShape().toPartialShape()) ? "one dim shape" : "not one dim shape"),
                          " with parent ",
                          parentPtr->getName(),
                          " outConfs[",
                          inNum,
                          "], estimate add to ",
                          estimate);
            }
        }
        return estimate;
    };

    auto selectSPDwithType = [&](const impl_desc_type type) {
        int selectedPrimitive = -1;
        int bestEstimate = std::numeric_limits<int>::max();
        for (size_t i = 0; i < getSupportedPrimitiveDescriptors().size(); i++) {
            const auto& supportedPrimitiveDesc = getSupportedPrimitiveDescriptors()[i];
            const impl_desc_type supportedType = supportedPrimitiveDesc.getImplementationType();
            if (supportedType != type) {
                continue;
            }

            const size_t descInConfSize = supportedPrimitiveDesc.getConfig().inConfs.size();

            if (descInConfSize > getParentEdges().size()) {
                OPENVINO_THROW(getName(),
                               " Desc ",
                               i,
                               " with type: ",
                               supportedType,
                               " has more input ports than node: ",
                               descInConfSize,
                               " vs ",
                               getParentEdges().size());
                continue;
            }

            auto estimate = estimateReorderOverhead(supportedPrimitiveDesc, i);

            if (estimate < bestEstimate) {
                bestEstimate = estimate;
                selectedPrimitive = static_cast<int>(i);
                DEBUG_LOG(getName(), " Select primitive desc: ", i, " ", supportedPrimitiveDesc);
            }
        }
        return selectedPrimitive;
    };

    // loop kernel priority
    for (auto& type : priority) {
        int selectedPrimitive = selectSPDwithType(type);
        if (selectedPrimitive >= 0) {
            selectPrimitiveDescriptorByIndex(selectedPrimitive);
            return;
        }
    }

    OPENVINO_ASSERT(!getSupportedPrimitiveDescriptors().empty(),
                    "Supported primitive descriptors list is empty for node: ",
                    getName(),
                    " type: ",
                    NameFromType(getType()));

    // fallback. If there are no primitives from priority list just select a first
    selectPrimitiveDescriptorByIndex(0);
}

bool Node::canBeInPlace() const {
    // TODO [DS]: enable inPlace for dynamic shapes
    if (isDynamicNode()) {
        return false;
    }

    if (getParentEdges().size() != 1 || getParentEdgeAt(0)->getParent()->getChildEdges().size() != 1 ||
        (getParentEdgeAt(0)->getParent()->isConstant() && !getParentEdgeAt(0)->getChild()->isConstant())) {
        return false;
    }

    // TODO: we need to extend this logic to properly handle all possible inplace conflicts
    if (getParentEdges().size() == 1 && getParentEdgeAt(0)->getParent()->getType() == Type::Reshape) {
        auto reshapeNode = getParentEdgeAt(0)->getParent();
        if (reshapeNode->getParentEdgeAt(0)->getParent()->getChildEdges().size() != 1) {
            return false;
        }
    }

    auto inShape = getInputShapeAtPort(0);
    for (size_t cIdx = 0; cIdx < outputShapes.size(); cIdx++) {
        if (getOutputShapeAtPort(cIdx) != inShape) {
            return false;
        }
    }
    return true;
}

void Node::resolveInPlaceEdges(Edge::LOOK look) {
    const NodeDesc* selected_pd = getSelectedPrimitiveDescriptor();
    if (!selected_pd) {
        OPENVINO_THROW("Cannot find selected primitive descriptor for node: ", getName());
    }
    if (look & Edge::LOOK_DOWN) {
        for (size_t i = 0; i < getParentEdges().size() && i < selected_pd->getConfig().inConfs.size(); i++) {
            auto inplaceOutIndx = selected_pd->getConfig().inConfs[i].inPlace();

            if (inplaceOutIndx < 0) {
                continue;
            }

            auto parentEdge = getParentEdgeAt(i);
            OPENVINO_ASSERT(parentEdge->getStatus() == Edge::Status::NotAllocated,
                            " Unexpected inplace resolve call to an allocated edge: ",
                            *parentEdge);

            // search for already allocated edge
            const auto& childEdges = getChildEdgesAtPort(inplaceOutIndx);
            auto itr = std::find_if(childEdges.begin(), childEdges.end(), [](const EdgePtr& edge) {
                return edge->getStatus() == Edge::Status::Allocated;
            });
            OPENVINO_ASSERT(itr != childEdges.end(),
                            " Could not find an allocated edge to resolve in-place for node: ",
                            getName());

            auto baseMemBlock = (*itr)->getMemory().getMemoryBlock();
            auto memBlock = std::make_shared<PartitionedMemoryBlock>(baseMemBlock);
            auto newMem =
                std::make_shared<Memory>(getEngine(), selected_pd->getConfig().inConfs[i].getMemDesc(), memBlock);
            parentEdge->reuse(newMem);
        }
    }
    if (look & Edge::LOOK_UP) {
        for (size_t i = 0; i < getChildEdges().size() && i < selected_pd->getConfig().outConfs.size(); i++) {
            auto inplaceInpIndx = selected_pd->getConfig().outConfs[i].inPlace();

            if (inplaceInpIndx < 0) {
                continue;
            }

            auto baseMemBlock = getParentEdgeAt(inplaceInpIndx)->getMemory().getMemoryBlock();
            auto memBlock = std::make_shared<PartitionedMemoryBlock>(baseMemBlock);
            const auto& childEdges = getChildEdgesAtPort(i);

            for (auto& childEdge : childEdges) {
                OPENVINO_ASSERT(childEdge->getStatus() == Edge::Status::NotAllocated,
                                " Unexpected inplace resolve call to an allocated edge: ",
                                *childEdge);
                auto newMem =
                    std::make_shared<Memory>(getEngine(), selected_pd->getConfig().outConfs[i].getMemDesc(), memBlock);
                childEdge->reuse(newMem);
            }
        }
    }
}

MemoryDescPtr Node::getBaseMemDescAtInputPort(size_t portNum) const {
    if (auto primDesc = getSelectedPrimitiveDescriptor()) {
        const auto& inConfs = primDesc->getConfig().inConfs;
        OPENVINO_ASSERT(portNum < inConfs.size(),
                        "Can't get input memory desc at port: ",
                        portNum,
                        ", incorrect port number");
        return inConfs[portNum].getMemDesc();
    }
    OPENVINO_THROW("Can't get input memory desc, primitive descriptor is not selected");
}

MemoryDescPtr Node::getBaseMemDescAtOutputPort(size_t portNum) const {
    if (auto primDesc = getSelectedPrimitiveDescriptor()) {
        const auto& outConfs = primDesc->getConfig().outConfs;
        OPENVINO_ASSERT(portNum < outConfs.size(),
                        "Can't get output memory desc at port: ",
                        portNum,
                        ", incorrect port number");
        return outConfs[portNum].getMemDesc();
    }
    OPENVINO_THROW("Can't get output memory desc, primitive descriptor is not selected");
}

MemoryDescPtr Node::getParentOutputMemDesc(const EdgePtr& edge) {
    const auto parentPtr = edge->getParent();
    const auto parentSpd = parentPtr->getSelectedPrimitiveDescriptor();
    OPENVINO_ASSERT(parentSpd, "Parent selected primitive descriptor is missed");

    const auto& parentOutConfs = parentSpd->getConfig().outConfs;
    OPENVINO_ASSERT(!parentOutConfs.empty(), "Parent output configuration is empty");

    const int inNum = edge->getInputNum();

    return parentSpd->getConfig().outConfs[inNum].getMemDesc();
}

std::string Node::getPrimitiveDescriptorType() const {
    auto selectedPrimitiveDesc = getSelectedPrimitiveDescriptor();

    impl_desc_type type = impl_desc_type::undef;
    if (selectedPrimitiveDesc) {
        type = selectedPrimitiveDesc->getImplementationType();
    }

    std::string str_type;

    auto add_type = [&](const std::string& t) {
        if (!str_type.empty() && t.c_str()[0] != '_') {
            str_type += "_";
        }
        str_type += t;
    };

#define SEARCH_TYPE(_type)                                       \
    if ((type & impl_desc_type::_type) == impl_desc_type::_type) \
    add_type(#_type)

    SEARCH_TYPE(undef);
    SEARCH_TYPE(reorder);
    SEARCH_TYPE(jit);
    SEARCH_TYPE(gemm);
    SEARCH_TYPE(brgconv);
    SEARCH_TYPE(brgemm);
    SEARCH_TYPE(ref);

    SEARCH_TYPE(avx512);
    SEARCH_TYPE(amx);
    SEARCH_TYPE(avx2);
    SEARCH_TYPE(avx);
    SEARCH_TYPE(sse42);
    SEARCH_TYPE(blas);
    SEARCH_TYPE(mlas);
    SEARCH_TYPE(any);
    SEARCH_TYPE(uni);

    SEARCH_TYPE(winograd);
    SEARCH_TYPE(sparse);
    SEARCH_TYPE(acl);
    SEARCH_TYPE(shl);
    SEARCH_TYPE(kleidiai);
    SEARCH_TYPE(_dw);
    SEARCH_TYPE(_1x1);

#undef SEARCH_TYPE

    if (type == impl_desc_type::unknown) {
        str_type = "unknown";
    } else if (str_type.empty()) {
        str_type = "undef";
    }

    // adding layer precision to the performance counters as one of the token
    // currently we treat a layer executing in int8 mode if its input is I8 or U8. if input is U8, we still
    // add I8 since I8 is special placeholder. The real calc precision might be quite complex and in most cases
    // it is mixed precision.
    if (selectedPrimitiveDesc) {
        if (!selectedPrimitiveDesc->getConfig().inConfs.empty()) {
            if (selectedPrimitiveDesc->getConfig().inConfs[0].getMemDesc()->getPrecision() != ov::element::u8) {
                str_type +=
                    "_" +
                    static_cast<std::string>(
                        selectedPrimitiveDesc->getConfig().inConfs[0].getMemDesc()->getPrecision().get_type_name());
            } else {
                str_type += "_I8";
            }
        } else {
            if (selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision() != ov::element::u8) {
                str_type +=
                    "_" +
                    static_cast<std::string>(
                        selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision().get_type_name());
            } else {
                str_type += "_I8";
            }
        }
    }

    return str_type;
}

EdgePtr Node::getParentEdgeAt(size_t idx) const {
    if (idx >= parentEdges.size()) {
        OPENVINO_THROW("Node ", getName(), " contains less parent edges than ", idx);
    }
    auto parentEdgePtr = parentEdges[idx].lock();
    if (!parentEdgePtr) {
        OPENVINO_THROW("Node ", getName(), " contains empty parent edge for index ", idx);
    }
    return parentEdgePtr;
}

EdgePtr Node::getChildEdgeAt(size_t idx) const {
    if (idx >= childEdges.size()) {
        OPENVINO_THROW("Node ", getName(), " contains less child edges than ", idx);
    }
    auto childEdgePtr = childEdges[idx].lock();
    if (!childEdgePtr) {
        OPENVINO_THROW("Node ", getName(), " contains empty child edge for index ", idx);
    }
    return childEdgePtr;
}

std::vector<EdgePtr> Node::getChildEdgesAtPort(int inputNum) const {
    if (inputNum < 0) {
        OPENVINO_THROW("Node ", getName(), ". negative input number is not supported ", inputNum);
    }

    if (static_cast<size_t>(inputNum) >= outputShapes.size()) {
        OPENVINO_THROW("Node ", getName(), " contains less output ports than ", inputNum);
    }

    std::vector<EdgePtr> res;
    for (auto& edge_w : childEdges) {
        auto edge = edge_w.lock();
        if (!edge) {
            OPENVINO_THROW("Node ", getName(), " contains dead weak ptr");
        }
        if (edge->getInputNum() == inputNum) {
            res.emplace_back(std::move(edge));
        }
    }
    return res;
}

std::vector<memory::format_tag> Node::getAvailableFormatsForDims(const Shape& dims) const {
    switch (dims.getRank()) {
    case 0:
    case 1:
        return {memory::format_tag::x};
    case 2:
        return {memory::format_tag::nc};
    case 3:
        return {memory::format_tag::tnc,
                memory::format_tag::ntc,
                memory::format_tag::ncw,
                memory::format_tag::nCw8c,
                memory::format_tag::nCw16c};
    case 4:
        return {memory::format_tag::nchw, memory::format_tag::nChw8c, memory::format_tag::nChw16c};
    case 5:
        return {memory::format_tag::ncdhw, memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c};
    default:
        return {memory::format_tag::any};
    }
}

static void fetchRawMemory(const MemoryPtr& mem) {
    // TODO: conceptually fetchRawMemory is a very bad solution
    if (mem->getDesc().getPrecision() == element::string) {
        return;
    }
    auto block = mem->getMemoryBlock();
    if (mem->isDefined()) {
        block->resize(mem->getSize());
    }
}

void Node::updateShapes() {
    OPENVINO_ASSERT(isDynamicNode(),
                    "Node::updateShapes() is called to a static shape node of type: ",
                    getTypeStr(),
                    " with name: ",
                    getName());
    try {
        if (needShapeInfer()) {
            auto result = shapeInfer();
            if (ShapeInferStatus::success == result.status) {
                redefineOutputMemory(result.dims);
            }
        } else {
            // guard check for internal dynamic nodes to avoid possible overestimation of the required memory size
            if (shapeInference && FULL_PORT_MASK == shapeInference->get_port_mask()) {
                return;
            }

            for (auto&& edge : getChildEdges()) {
                auto edge_ptr = edge.lock();
                CPU_NODE_ASSERT(edge_ptr, " has null edge");
                if (edge_ptr->inPlace(Edge::LOOK_UP)) {
                    continue;
                }

                auto mem = edge_ptr->getMemoryPtr();
                CPU_NODE_ASSERT(mem, " has null output memory");

                if (mem->getShape().hasZeroDims()) {
                    continue;
                }
                fetchRawMemory(mem);
            }
        }
    } catch (const std::exception& exp) {
        THROW_CPU_NODE_ERR(exp.what());
    }
}

void Node::updateDynamicParams() {
    OPENVINO_ASSERT(isDynamicNode(),
                    "Node::updateDynamicParams() is called to a static shape node of type: ",
                    getTypeStr(),
                    " with name: ",
                    getName());
    try {
        if (isExecutable()) {
            if (needPrepareParams()) {
                OPENVINO_ASSERT(inputShapesDefined(), "Input shapes are not defined.");
                DEBUG_LOG(" prepareParams() on #",
                          getExecIndex(),
                          " ",
                          getTypeStr(),
                          " ",
                          algToString(getAlgorithm()),
                          " ",
                          getName(),
                          " ",
                          getOriginalLayers());
                prepareParams();
            }
        }
    } catch (const std::exception& e) {
        THROW_CPU_NODE_ERR(e.what());
    }
}

void Node::execute(const dnnl::stream& strm, int numaId) {
    if (isDynamicNode()) {
        return executeDynamic(strm, numaId);
    }
    return executeStatic(strm, numaId);
}

void Node::executeStatic(const dnnl::stream& strm, int numaId) {
    toNumaNode(numaId);
    execute(strm);
}

void Node::executeDynamic(const dnnl::stream& strm, int numaId) {
    if (isExecutable()) {
        toNumaNode(numaId);
        executeDynamicImpl(strm);
    }

    updateLastInputDims();
}

bool Node::outputShapeDataDependency() const {
    auto port_mask = shapeInference->get_port_mask();
    if (EMPTY_PORT_MASK != port_mask) {
        for (size_t i = 0; i < getParentEdges().size(); ++i) {
            if ((port_mask & (1 << i)) && !getParentEdgeAt(i)->getParent()->isConstant()) {
                return true;
            }
        }
    }
    return false;
}

void Node::redefineOutputMemory(const std::vector<VectorDims>& newOutputShapes) {
    if (newOutputShapes.size() != outputShapes.size()) {
        OPENVINO_THROW("Number shapes mismatch with real outputs number for node with name: ", getName());
    }
    for (size_t i = 0lu; i < outputShapes.size(); i++) {
        redefineOutputMemory(i, newOutputShapes[i]);
    }
}

void Node::redefineOutputMemory(const size_t port, const VectorDims& new_output_shape) {
    const auto edges = getChildEdgesAtPort(port);

    static const VectorDims single_element_shape = {1};

    // avoid 0D shape incompatible
    const auto& new_shape = new_output_shape.empty() ? single_element_shape : new_output_shape;

    const auto& curr_desc = edges[0]->getMemory().getDesc();
    if (curr_desc.getShape().isStatic() && curr_desc.getShape().getStaticDims() == new_shape) {
        for (auto&& edge : edges) {
            fetchRawMemory(edge->getMemoryPtr());
        }
        return;
    }

    const bool has_zero_dims = std::count(std::begin(new_shape), std::end(new_shape), 0lu) > 0;
    const auto mem_desc = getBaseMemDescAtOutputPort(port)->cloneWithNewDims(new_shape, has_zero_dims);
    for (size_t j = 0lu; j < edges.size(); j++) {  // NOLINT(modernize-loop-convert)
        edges[j]->getMemoryPtr()->redefineDesc(mem_desc);
    }
}

void Node::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    auto addSupportedPrimitiveDescriptor = [&](const dnnl::primitive_desc& prim_desc) {
        std::vector<PortConfig> inConfs, outConfs;
        const int inPlaceOutPort = canBeInPlace() ? 0 : -1;

        for (size_t i = 0; i < descInputNumbers(); i++) {
            auto desc = getSrcMemDesc(prim_desc, i);

            inConfs.emplace_back(desc, BlockedMemoryDesc::EMPTY_MASK);
        }

        for (size_t i = 0; i < descOutputNumbers(); i++) {
            auto desc = getDstMemDesc(prim_desc, i);

            outConfs.emplace_back(desc, BlockedMemoryDesc::EMPTY_MASK, inPlaceOutPort);
        }

        const NodeConfig config(inConfs, outConfs);
        const impl_desc_type impl_type = parse_impl_name(prim_desc.impl_info_str());

        supportedPrimitiveDescriptors.emplace_back(config, impl_type);
    };

    /* When custom implementation priorities are NOT defined it is enough to
     * just use the first implementation from the priority list.
     * When custom implementation priorities are defined, all the implementations should be considered,
     * since custom implementations can be not available at all, so a fallback to the default ones must happen
     * To achive the fallback, it is necessary to create a supported primitive descriptor for each implementation
     * since oneDNN primitive is mutating while iterating */
#ifdef CPU_DEBUG_CAPS
    {
        if (!customImplPriorities.empty()) {
            DEBUG_LOG("#",
                      getName(),
                      " customImplPriorities [",
                      0,
                      "/",
                      customImplPriorities.size(),
                      "]: ",
                      impl_type_to_string(customImplPriorities[0]));
        }
    }
#endif
    for (auto& desc : descs) {
        auto first_desc = dnnl::primitive_desc(DnnlExtensionUtils::clone_primitive_desc(desc.get()));
        const bool first_match = customImplPriorities.empty();
        DEBUG_LOG("#",
                  getName(),
                  ", itpd.impl_info_str(): ",
                  desc.impl_info_str(),
                  ", parsed imp_type: ",
                  impl_type_to_string(parse_impl_name(desc.impl_info_str())),
                  ", first_match: ",
                  first_match ? "true" : "false");
        DnnlExtensionUtils::for_each_implementation(
            desc,
            first_match,
            [&](impl_desc_type implType) {
                return contains(getImplPriority(), implType);
            },
            [&](dnnl::primitive_desc& desc) {
                addSupportedPrimitiveDescriptor(desc);
            });

        // fallback. if none of the primitive types is present in the priority list just add first implementation
        // @todo this fallback is not necessary if primitive priority list is filled correctly
        if (supportedPrimitiveDescriptors.empty()) {
            addSupportedPrimitiveDescriptor(first_desc);
        }
    }
}

void Node::filterSupportedPrimitiveDescriptors() {
    if (inputMemoryFormatsFilter.empty() && outputMemoryFormatsFilter.empty()) {
        return;
    }

    // Compare by format tag
    auto areCompatible = [](const MemoryDesc& desc, dnnl::memory::format_tag fmt) -> bool {
        auto data_type = DnnlExtensionUtils::ElementTypeToDataType(desc.getPrecision());
        auto fmt_tdesc = DnnlBlockedMemoryDesc(desc.getShape(), data_type, fmt);
        return desc.isCompatible(fmt_tdesc);
    };

    auto isNotSuitableDesc = [&](const NodeDesc& desc) {
        const auto& config = desc.getConfig();
        if (inputMemoryFormatsFilter.size() > config.inConfs.size() ||
            outputMemoryFormatsFilter.size() > config.outConfs.size()) {
            OPENVINO_THROW("Incorrect number of input or output memory formats");
        }

        for (size_t i = 0; i < inputMemoryFormatsFilter.size(); i++) {
            if (!areCompatible(*config.inConfs[i].getMemDesc(), inputMemoryFormatsFilter[i])) {
                DEBUG_LOG(getName(),
                          " input memory format filter: ",
                          inputMemoryFormatsFilter[i],
                          " not matched. Erase desc from supported primitive descriptors: ",
                          desc);
                return true;
            }
        }

        for (size_t i = 0; i < outputMemoryFormatsFilter.size(); i++) {
            if (!areCompatible(*config.outConfs[i].getMemDesc(), outputMemoryFormatsFilter[i])) {
                DEBUG_LOG(getName(),
                          " Output memory format filter: ",
                          outputMemoryFormatsFilter[i],
                          " not matched. Erase desc from supported primitive descriptors: ",
                          desc);
                return true;
            }
        }

        return false;
    };

    supportedPrimitiveDescriptors.erase(
        std::remove_if(supportedPrimitiveDescriptors.begin(), supportedPrimitiveDescriptors.end(), isNotSuitableDesc),
        supportedPrimitiveDescriptors.end());

    OPENVINO_ASSERT(!supportedPrimitiveDescriptors.empty(),
                    getName(),
                    " type: ",
                    NameFromType(getType()),
                    " No supported primitive descriptors matched the provided input / output memory format filters.");
}

void Node::initDescriptor(const NodeConfig& config) {
    auto* selectedPD = getSelectedPrimitiveDescriptor();

    if (!selectedPD) {
        return;
    }

    if (descs.empty()) {
        const auto& selectedConfig = selectedPD->getConfig();
        if (selectedConfig.inConfs.size() != config.inConfs.size() ||
            selectedConfig.outConfs.size() != config.outConfs.size()) {
            return;
        }

        for (size_t i = 0; i < selectedConfig.inConfs.size(); i++) {
            if (!selectedConfig.inConfs[i].getPortDesc()->isCompatible(*config.inConfs[i].getPortDesc())) {
                OPENVINO_THROW("Incorrect descriptor for node: ", getName(), " on ", i, " intput port");
            }
        }

        for (size_t i = 0; i < selectedConfig.outConfs.size(); i++) {
            if (!selectedConfig.outConfs[i].getPortDesc()->isCompatible(*config.outConfs[i].getPortDesc())) {
                OPENVINO_THROW("Incorrect descriptor for node: ", getName(), " on ", i, " output port");
            }
        }
        selectedPD->setConfig(config);

        return;
    }

    auto updateNodeConfig = [&](const NodeConfig& cfg) {
        auto updatedConfig = cfg;

        for (size_t i = 0; i < descInputNumbers(); i++) {
            PortConfig& dataConfig = updatedConfig.inConfs[i];
            dataConfig.inPlace(canBeInPlace() ? 0 : -1);     // update inPlace
            dataConfig.setMemDesc(dataConfig.getMemDesc());  // reset desc with default compatibility mask
        }

        for (size_t i = 0; i < descOutputNumbers(); i++) {
            PortConfig& dataConfig = updatedConfig.outConfs[i];
            dataConfig.inPlace(-1);                          // update inPlace
            dataConfig.setMemDesc(dataConfig.getMemDesc());  // reset desc with default compatibility mask
        }

        return updatedConfig;
    };

    descs.clear();

    std::vector<MemoryDescPtr> inDescs;
    inDescs.reserve(config.inConfs.size());
    for (const auto& inConf : config.inConfs) {
        inDescs.emplace_back(inConf.getMemDesc());
    }
    std::vector<MemoryDescPtr> outDescs;
    outDescs.reserve(config.outConfs.size());
    for (const auto& outConf : config.outConfs) {
        outDescs.emplace_back(outConf.getMemDesc());
    }
    createDescriptor(inDescs, outDescs);

    for (auto& desc : descs) {
        if (DnnlExtensionUtils::find_implementation(desc, selectedPD->getImplementationType())) {
            selectedPD->setConfig(config);
            return;
        }
    }

    const auto& currentConfig = selectedPD->getConfig();
    const auto& updatedConfig = updateNodeConfig(currentConfig);

    selectedPD->setConfig(updatedConfig);
}

void Node::prepareMemory(const DnnlMemoryDescPtr& intDesc, size_t indx) {
    size_t minSize = indx + 1;
    if (internalBlobMemory.size() < minSize) {
        internalBlobMemory.resize(minSize);
    }

    if (minSize > internalBlobs.size()) {
        OPENVINO_THROW("Can't prepare memory for internal blob, requested index: ",
                       indx,
                       " is out of bounds of the internalBlobs vector of size ",
                       internalBlobs.size());
    }

    const auto& internalBlob = internalBlobs[indx];

    auto create = [&]() {
        auto newDesc = internalBlob->getDescPtr();
        Memory memory{engine, newDesc, internalBlob->getData()};

        MemoryPtr _ptr = std::make_shared<Memory>(engine, intDesc);
        node::Reorder::reorderData(memory, *_ptr, context->getParamsCache());
        return _ptr;
    };

    MemoryPtr ptr;
    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr && memory::format_kind::blocked == intDesc->getDnnlDesc().get_format_kind()) {
        const auto string_hash = name + "_" + std::to_string(indx) + "_" +
                                 DnnlExtensionUtils::computeWeightsStringHash(internalBlob, intDesc);
        ptr = *weightCache->findOrCreate(string_hash, create);
    } else {
        ptr = create();
    }

    internalBlobMemory[indx] = ptr;
}

void Node::prepareMemory(const std::vector<DnnlMemoryDescPtr>& intDescs) {
    if (internalBlobs.size() != intDescs.size()) {
        OPENVINO_THROW("Can't prepare memory for internal blob, internal blob and internal descs number do not match ",
                       internalBlobs.size(),
                       " vs ",
                       intDescs.size());
    }

    internalBlobMemory.clear();
    for (size_t i = 0; i < internalBlobs.size(); i++) {
        prepareMemory(intDescs[i], i);
    }
}

void Node::prepareMemory(dnnl::primitive_desc_iterator& itpd) {
    std::vector<DnnlMemoryDescPtr> intDescs;
    intDescs.reserve(internalBlobDesc.size());
    for (auto& it : internalBlobDesc) {
        intDescs.push_back(it(itpd, 0));
    }

    Node::prepareMemory(intDescs);
}

MemoryPtr Node::prepareWeightMemory(DnnlMemoryDescPtr dstWeightDesc, DnnlMemoryDescPtr srcWeightDesc) {
    if (!getParentEdgeAt(1)->getParent()->isConstant()) {
        OPENVINO_THROW("Weight input is not const for node ", getName(), ".");
    }
    auto edgeMem = getSrcMemoryAtPort(1);
    if (!edgeMem) {
        OPENVINO_THROW("Cannot get const weights edgeMem for node ", getName(), ".");
    }

    if (!srcWeightDesc) {
        auto constDnnlMemOutDesc = edgeMem->getDescWithType<DnnlMemoryDesc>();
        auto weightSrcDesc = constDnnlMemOutDesc->getDnnlDesc();
        weightSrcDesc = weightSrcDesc.reshape(dstWeightDesc->getDnnlDesc().get_dims());
        srcWeightDesc = DnnlExtensionUtils::makeDescriptor(weightSrcDesc);
    }

    auto create = [&]() {
        Memory srcMemory{getEngine(), srcWeightDesc, edgeMem->getData()};
        MemoryPtr _ptr = std::make_shared<Memory>(getEngine(), dstWeightDesc);
        node::Reorder::reorderData(srcMemory, *_ptr, context->getParamsCache());

        return _ptr;
    };

    MemoryPtr ptr;
    const auto& format = dstWeightDesc->serializeFormat();

    OPENVINO_ASSERT(privateWeightCache, "privateWeightCache is nullptr");

    auto itr = privateWeightCache->find(format);
    if (privateWeightCache->end() != itr) {
        return itr->second;
    }

    auto weightCache = context->getWeightsCache();
    if (weightCache != nullptr) {
        const auto string_hash = DnnlExtensionUtils::computeWeightsStringHash(edgeMem, dstWeightDesc);
        ptr = *weightCache->findOrCreate(string_hash, create);
    } else {
        ptr = create();
    }

    (*privateWeightCache)[format] = ptr;

    return ptr;
}

void Node::toNumaNode(int numaNodeID) {
    if (numaNodeID < 0) {
        return;
    }

    return toNumaNodeImpl(numaNodeID);
}

void Node::toNumaNodeImpl(int numaNodeID) {
    if (curNumaNode == numaNodeID) {
        return;
    }

    // create scratch pad from specified numa node
    if (scratchpadMem) {
        scratchpadMem = context->getScratchPad()->createScratchPadMem(scratchpadMem->getDescPtr());
        primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMem->getPrimitive();
    }

    // mbind constant prim args to numa nodes
    if (primArgs.count(DNNL_ARG_WEIGHTS)) {
        mbind_move(primArgs[DNNL_ARG_WEIGHTS], numaNodeID);
    }
    if (primArgs.count(DNNL_ARG_BIAS)) {
        mbind_move(primArgs[DNNL_ARG_BIAS], numaNodeID);
    }

    curNumaNode = numaNodeID;
}

bool Node::isInPlace() const {
    if (inplace == InPlaceType::Unknown) {
        auto selected_pd = getSelectedPrimitiveDescriptor();
        if (selected_pd == nullptr) {
            OPENVINO_THROW("Preferable primitive descriptor is not set.");
        }

        inplace = InPlaceType::NoInPlace;
        auto config = selected_pd->getConfig();
        for (auto& in : config.inConfs) {
            if (in.inPlace() >= 0) {
                inplace = InPlaceType::InPlace;
                break;
            }
        }
        for (auto& out : config.outConfs) {
            if (out.inPlace() >= 0) {
                inplace = InPlaceType::InPlace;
                break;
            }
        }
    }

    return inplace == InPlaceType::InPlace;
}

Node::ConstantType Node::getConstantType() const {
    return constant;
}

bool Node::isConstant() {
    return getConstantType() == ConstantType::Const;
}

void Node::updateConstantType() {
    if (constant == ConstantType::StrictNoConst) {
        return;
    }

    bool isConst = true;
    for (const auto& parentEdge : getParentEdges()) {
        isConst &= parentEdge.lock()->getParent()->isConstant();
    }

    const auto prevConstantType = constant;
    constant = isConst ? ConstantType::Const : ConstantType::NoConst;
    if (constant == prevConstantType) {
        return;  // state has not changed, no reason to continue
    }

    for (const auto& childEdge : getChildEdges()) {
        const auto childNode = childEdge.lock()->getChild();
        childNode->updateConstantType();
    }
}

void Node::addOriginalLayer(const std::string& layerName) {
    if (layerName.empty()) {
        return;
    }
    if (originalLayers.empty()) {
        originalLayers = layerName;
    } else {
        originalLayers += "," + layerName;
    }
}

void Node::cleanup() {
    internalBlobs.clear();

    for (const auto& it : fusedWith) {
        it->cleanup();
    }

    for (const auto& it : mergedWith) {
        it->cleanup();
    }
}

const std::vector<impl_desc_type>& Node::getDefaultImplPriority() {
    static const std::vector<impl_desc_type> priorities {
        impl_desc_type::unknown,
            // Undef impl type is used to express use-cases there real type is unkown during compilation
            // Undef has higher priority than defined types in order to force primitive selection logic to make decision
            // based on other properties
            impl_desc_type::undef, impl_desc_type::brgconv_avx512_amx_1x1, impl_desc_type::brgconv_avx512_amx,
            impl_desc_type::jit_avx512_amx_dw, impl_desc_type::jit_avx512_amx_1x1, impl_desc_type::jit_avx512_amx,
            // Brgconv kernels disabled in order to prevent perf degradations on non AMX HW
            // impl_desc_type::brgconv_avx512_1x1,
            // impl_desc_type::brgconv_avx512,
            impl_desc_type::jit_uni_dw, impl_desc_type::jit_uni_1x1, impl_desc_type::jit_uni,
            impl_desc_type::jit_avx512_dw, impl_desc_type::jit_avx512_1x1, impl_desc_type::jit_avx512,
            impl_desc_type::jit_avx2_dw, impl_desc_type::jit_avx2_1x1, impl_desc_type::jit_avx2,
            impl_desc_type::jit_avx_dw, impl_desc_type::jit_avx_1x1, impl_desc_type::jit_avx,
            impl_desc_type::jit_sse42_dw, impl_desc_type::jit_sse42_1x1, impl_desc_type::jit_sse42,
#if defined(OPENVINO_ARCH_ARM64)
            impl_desc_type::jit_asimd,
#elif defined(OPENVINO_ARCH_RISCV64)
            impl_desc_type::jit_gv,
#endif
            impl_desc_type::gemm_any, impl_desc_type::gemm_blas, impl_desc_type::gemm_avx512, impl_desc_type::gemm_avx2,
            impl_desc_type::gemm_avx, impl_desc_type::gemm_sse42, impl_desc_type::gemm_acl, impl_desc_type::acl,
            impl_desc_type::gemm_kleidiai, impl_desc_type::kleidiai, impl_desc_type::jit_gemm, impl_desc_type::ref_any,
            impl_desc_type::ref,
    };

    return priorities;
}

const std::vector<impl_desc_type>& Node::getImplPriority() {
    if (!customImplPriorities.empty()) {
        return customImplPriorities;
    }

    return getDefaultImplPriority();
}

PortDescBasePtr Node::getConsistentInputDesc(const NodeConfig& config, size_t idx) const {
    const auto& inConf = config.inConfs[idx];

    if (inConf.inPlace() >= 0) {  // node have inplace input
        auto inplaceIndx = static_cast<size_t>(inConf.inPlace());
        PortDescBasePtr outPortDesc;
        const auto& outConf = config.outConfs[inplaceIndx];
        if (outConf.inPlace() ==
            static_cast<int>(idx)) {              // the input desc port is the same port used for inplace output
            outPortDesc = outConf.getPortDesc();  // just use desc from this output port
        } else {
            outPortDesc = getConsistentOutputDesc(config, inplaceIndx);  // get consistent desc otherwise
        }
        if (inConf.getPortDesc()->isCompatible(*outPortDesc)) {  // use the desc if compatible
            return outPortDesc;
        }
    }

    auto* parentSelectedPD = getParentEdgeAt(idx)->getParent()->getSelectedPrimitiveDescriptor();
    if (!parentSelectedPD) {
        OPENVINO_THROW("Cannot get selected primitive descriptor for node: ",
                       getParentEdgeAt(idx)->getParent()->getName());
    }

    int num = getParentEdgeAt(idx)->getInputNum();
    if (num >= 0) {
        auto parentConf = parentSelectedPD->getConfig().outConfs[num];
        const auto desc = parentConf.getMemDesc()->cloneWithNewPrecision(inConf.getMemDesc()->getPrecision());
        parentConf.setMemDesc(desc);

        if (!parentConf.getMemDesc()->isDefined() && parentConf.inPlace() >= 0) {
            getParentEdgeAt(idx)->getParent()->initOptimalPrimitiveDescriptor();
        }

        // config might be changed
        parentConf = parentSelectedPD->getConfig().outConfs[num];
        if (parentConf.getMemDesc()->isDefined() && inConf.getPortDesc()->isCompatible(*parentConf.getPortDesc())) {
            return parentConf.getPortDesc();
        }
    }

    return inConf.getPortDesc();
}

PortDescBasePtr Node::getConsistentOutputDesc(const NodeConfig& config, size_t idx) const {
    const auto& outConf = config.outConfs[idx];

    if (outConf.inPlace() >= 0) {  // node have inplace output
        auto inplaceIndx = static_cast<size_t>(outConf.inPlace());
        PortDescBasePtr inpPortDesc;
        const auto& inpConf = config.inConfs[inplaceIndx];
        if (inpConf.inPlace() ==
            static_cast<int>(idx)) {              // the input desc port is the same port used for inplace output
            inpPortDesc = inpConf.getPortDesc();  // just use desc from this output port
        } else {
            inpPortDesc = getConsistentInputDesc(config, inplaceIndx);  // get consistent desc otherwise
        }
        if (outConf.getPortDesc()->isCompatible(*inpPortDesc)) {  // use the desc if compatible
            return inpPortDesc;
        }
    }

    auto* childSelectedPD = getChildEdgeAt(idx)->getChild()->getSelectedPrimitiveDescriptor();
    if (!childSelectedPD) {
        OPENVINO_THROW("Cannot get selected primitive descriptor for node: ",
                       getChildEdgeAt(idx)->getChild()->getName());
    }

    int num = getChildEdgeAt(idx)->getOutputNum();
    if (num >= 0) {
        auto childConf = childSelectedPD->getConfig().inConfs[num];
        const auto desc = childConf.getMemDesc()->cloneWithNewPrecision(outConf.getMemDesc()->getPrecision());
        childConf.setMemDesc(desc);

        if (!childConf.getMemDesc()->isDefined() && childConf.inPlace() >= 0) {
            getChildEdgeAt(idx)->getChild()->initOptimalPrimitiveDescriptor();
        }

        // config might be changed
        childConf = childSelectedPD->getConfig().inConfs[num];
        if (childConf.getMemDesc()->isDefined() && outConf.getPortDesc()->isCompatible(*childConf.getPortDesc())) {
            return childConf.getPortDesc();
        }
    }

    return outConf.getPortDesc();
}

void Node::initOptimalPrimitiveDescriptor() {
    if (one_of(getType(), Type::RNNCell, Type::RNNSeq)) {  // can be skipped for RNN node
        return;
    }

    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr) {
        OPENVINO_THROW("Preferable primitive descriptor is not set for ", getName());
    }

    auto config = selected_pd->getConfig();
    for (size_t i = 0; i < config.inConfs.size(); i++) {
        if (!isDynamicNode() || config.inConfs[i].getMemDesc()->isDefined()) {
            auto inpPortDesc = getConsistentInputDesc(config, i);
            DEBUG_LOG(getName(), ": input PortDesc before: ", *inpPortDesc->getMemDesc());
            config.inConfs[i].setMemDesc(inpPortDesc->getMemDesc());
            DEBUG_LOG(getName(), ": input PortDesc after: ", *config.inConfs[i].getMemDesc());
        }
    }

    for (size_t i = 0; i < config.outConfs.size(); i++) {
        auto outMemDesc = config.outConfs[i].getMemDesc();
        if (!isDynamicNode() || outMemDesc->isDefined()) {
            auto outPortDesc = getConsistentOutputDesc(config, i);
            DEBUG_LOG(getName(), ": output PortDesc before: ", *outPortDesc->getMemDesc());
            config.outConfs[i].setMemDesc(outPortDesc->getMemDesc());
        } else {
            // it is assumed that the nodes will define dense tensors on output edges
            // if it is not the case the implementation must redefine this behaviour
            if (outMemDesc->getType() & Blocked) {
                config.outConfs[i].setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(outMemDesc),
                                              BlockedMemoryDesc::FULL_MASK);
            }
        }
    }

    initDescriptor(config);
}

bool Node::isConfigDefined(const NodeConfig& config) const {
    for (const auto& configs : {config.inConfs, config.outConfs}) {
        for (const auto& dc : configs) {
            if (!dc.getMemDesc()->isDefined()) {
                return false;
            }
        }
    }
    return true;
}

MemoryDescPtr Node::getSrcMemDesc(const dnnl::primitive_desc& prim_desc, size_t idx) const {
    if (getInputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(prim_desc.src_desc(idx), getInputShapeAtPort(idx));
    }
    return DnnlExtensionUtils::makeDescriptor(prim_desc.src_desc(idx));
}

MemoryDescPtr Node::getDstMemDesc(const dnnl::primitive_desc& prim_desc, size_t idx) const {
    if (getOutputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(prim_desc.dst_desc(idx), getOutputShapeAtPort(idx));
    }
    return DnnlExtensionUtils::makeDescriptor(prim_desc.dst_desc(idx));
}

void Node::appendPostOpArgs(const dnnl::primitive_attr& attr,
                            std::unordered_map<int, dnnl::memory>& primArgs,
                            const std::unordered_map<int, MemoryPtr>& postOpsArgs) {
    for (auto& entry : postOpsArgs) {
        primArgs[entry.first] = entry.second->getPrimitive();
    }
}

bool Node::isFusedWith(Type fusedNodeType) const {
    for (const auto& fusedNode : fusedWith) {
        if (fusedNode->type == fusedNodeType) {
            return true;
        }
    }

    return false;
}

dnnl::memory::format_tag Node::getWeightsFormatTagByDims(const VectorDims& dims) const {
    switch (dims.size()) {
    case 1:
        return dnnl::memory::format_tag::a;
    case 2:
        return dnnl::memory::format_tag::ab;
    case 3:
        return dnnl::memory::format_tag::abc;
    case 4:
        return dnnl::memory::format_tag::abcd;
    case 5:
        return dnnl::memory::format_tag::abcde;
    case 6:
        return dnnl::memory::format_tag::abcdef;
    default:
        OPENVINO_THROW("getWeightsFormatTagByDims doesn't support dims.size() = ", dims.size());
    }
}

void Node::appendPostOps(dnnl::post_ops& ops,
                         const VectorDims& postOpDims,
                         std::unordered_map<int, MemoryPtr>& postOpsMem,
                         const int channelAxis) {
    OPENVINO_THROW("Fusing of ", NameFromType(this->getType()), " operation is not implemented");
}

void Node::appendPostOps(dnnl::post_ops& ops,
                         const VectorDims& postOpDims,
                         std::vector<const void*>& postOpsMem,
                         const int channelAxis) {
    OPENVINO_THROW("Fusing of ", NameFromType(this->getType()), " operation is not implemented");
}

std::vector<ov::element::Type> Node::getInputPrecisions() const {
    std::vector<ov::element::Type> inputPrecisions;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated) {
            inputPrecisions.emplace_back(parentEdge->getMemoryPtr()->getDesc().getPrecision());
        }
    }
    return inputPrecisions;
}

std::vector<ov::element::Type> Node::getOutputPrecisions() const {
    std::vector<ov::element::Type> outputPrecisions;
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto childEdge = getChildEdgeAt(i);
        if (childEdge && childEdge->getStatus() == Edge::Status::Validated) {
            outputPrecisions.emplace_back(childEdge->getMemoryPtr()->getDesc().getPrecision());
        }
    }
    return outputPrecisions;
}

ov::element::Type Node::getRuntimePrecision() const {
    // Base implementation consider precision only on data path and
    // assumes it is placed on 0-th port (which is true for almost all layers)
    ov::element::Type runtimePrecision = ov::element::dynamic;
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

Node* Node::NodesFactory::create(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context) {
    Node* newNode = nullptr;
    std::string errorMessage;
    if (newNode == nullptr) {
        try {
            std::unique_ptr<Node> ol(createNodeIfRegistered(intel_cpu, TypeFromName(op->get_type_name()), op, context));
            if (ol != nullptr && ol->created()) {
                newNode = ol.release();
            }
        } catch (const ov::Exception& ex) {
            if (dynamic_cast<const ov::NotImplemented*>(&ex) != nullptr) {
                errorMessage += ex.what();
            } else {
                throw;
            }
        }
    }

    if (newNode == nullptr) {
        try {
            std::unique_ptr<Node> ol(new Reference(op, context, errorMessage));
            if (ol != nullptr && ol->created()) {
                newNode = ol.release();
            }
        } catch (const ov::Exception& ex) {
            if (dynamic_cast<const ov::NotImplemented*>(&ex) != nullptr) {
                const std::string currErrorMess = ex.what();
                if (!currErrorMess.empty()) {
                    errorMessage += errorMessage.empty() ? currErrorMess : "\n" + currErrorMess;
                }
            } else {
                throw;
            }
        }
    }

    if (!newNode) {
        std::string errorDetails;
        if (!errorMessage.empty()) {
            errorDetails = "\nDetails:\n" + errorMessage;
        }
        OPENVINO_THROW("Unsupported operation of type: ",
                       op->get_type_name(),
                       " name: ",
                       op->get_friendly_name(),
                       errorDetails);
    }

    return newNode;
}

bool Node::canBePerformedAsScaleShift(const Node* parentNode) const {
#if defined(OPENVINO_ARCH_X86_64)
    OPENVINO_ASSERT(parentNode);

    size_t fusingPort = 0;
    const auto channelAxis = parentNode->getFusingAxis();

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        Node* node = getParentEdgeAt(i)->getParent().get();
        if (node == nullptr) {
            OPENVINO_THROW("Cannot get parent node for ", getName(), " on ", i, " port");
        }
        if (node == parentNode) {
            fusingPort = i;
            continue;
        }
        if (node->getType() != Type::Input || !node->isConstant()) {
            return false;
        }
    }

    const auto isBroadcastableToDataInput = [&]() {
        auto& dataShape = getInputShapeAtPort(fusingPort).getDims();
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            if (i == fusingPort) {
                continue;
            }
            auto& weightShape = getInputShapeAtPort(i).getDims();
            if (getParentEdgeAt(i)->getParent()->getChildEdges().size() != 1 ||
                !isPerTensorOrPerChannelBroadcastable(dataShape, weightShape, channelAxis, true)) {
                return false;
            }
        }
        return true;
    };

    const auto isConvertablePowerStatic = [&]() {
        if (getAlgorithm() == Algorithm::EltwisePowerStatic) {
            const auto eltwise = dynamic_cast<const Eltwise*>(this);
            if (!eltwise) {
                OPENVINO_THROW("Cannot cast ", getName(), " to Eltwise");
            }
            return eltwise->getAlpha() == 1.0f;
        }
        return false;
    };

    return (one_of(getAlgorithm(),
                   Algorithm::EltwiseAdd,
                   Algorithm::EltwiseMultiply,
                   Algorithm::EltwiseSubtract,
                   Algorithm::EltwiseDivide,
                   Algorithm::EltwisePrelu,
                   Algorithm::EltwiseMulAdd) &&
            isBroadcastableToDataInput()) ||
           isConvertablePowerStatic();
#else
    // TODO: provide correct list of operations for other backends
    return false;
#endif
}

// @todo shifts for Subtract and scales for Divide are replaced with
// Add (with opposite sign) and Multiply (with inverse value) for legacy dephwise post ops
// This can be avoided after dephwise post ops are gone
std::pair<std::vector<float>, std::vector<float>> Node::getScalesAndShifts(const Node* parentNode) const {
    std::vector<float> scales, shifts;

    const auto fillValuesFrom = [&](const NodePtr& constInput, std::vector<float>& buffer) {
        auto* constInputNode = dynamic_cast<node::Input*>(constInput.get());
        if (!constInputNode) {
            OPENVINO_THROW("Cannot cast ", constInput->getName(), " to Input");
        }
        auto constBlob = constInputNode->getMemoryPtr();
        const auto elementsCount = constBlob->getDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
        buffer.resize(elementsCount);
        cpu_convert(constBlob->getData(),
                    &buffer[0],
                    DnnlExtensionUtils::DataTypeToElementType(constBlob->getDataType()),
                    ov::element::f32,
                    elementsCount);
    };

    const auto constPort = getParentEdgeAt(0)->getParent().get() == parentNode ? 1 : 0;

    if (one_of(getAlgorithm(), Algorithm::EltwiseMultiply, Algorithm::EltwiseDivide, Algorithm::EltwisePrelu)) {
        fillValuesFrom(getParentEdgeAt(constPort)->getParent(), scales);
    } else if (one_of(getAlgorithm(), Algorithm::EltwiseAdd, Algorithm::EltwiseSubtract)) {
        fillValuesFrom(getParentEdgeAt(constPort)->getParent(), shifts);
    } else if (one_of(getAlgorithm(), Algorithm::EltwiseMulAdd)) {
        fillValuesFrom(getParentEdgeAt(1)->getParent(), scales);
        fillValuesFrom(getParentEdgeAt(2)->getParent(), shifts);
    } else if (one_of(getAlgorithm(), Algorithm::EltwisePowerStatic)) {
        const auto power = dynamic_cast<const Eltwise*>(this);
        if (!power) {
            OPENVINO_THROW("Cannot cast ", getName(), " to Eltwise");
        }
        scales.push_back(power->getBeta());
        shifts.push_back(power->getGamma());
    } else {
        OPENVINO_THROW("Can't fill scale and shifts for node: ", getName(), " with type: ", NameFromType(getType()));
    }

    switch (getAlgorithm()) {
    case Algorithm::EltwiseAdd: {
        scales.resize(shifts.size(), 1.0f);
        break;
    }
    case Algorithm::EltwiseSubtract: {
        scales.resize(shifts.size(), 1.0f);
        std::transform(shifts.begin(), shifts.end(), shifts.begin(), [](float shift) {
            return -1.0f * shift;
        });
        break;
    }
    case Algorithm::EltwiseMultiply: {
        shifts.resize(scales.size(), 0.0f);
        break;
    }
    case Algorithm::EltwiseDivide: {
        shifts.resize(scales.size(), 0.0f);
        std::transform(scales.begin(), scales.end(), scales.begin(), [](float scale) {
            return 1.0f / scale;
        });
        break;
    }
    default:
        break;
    }

    return {scales, shifts};
}

bool Node::isInputTensorAtPortEmpty(size_t port) const {
    if (inputShapes.size() <= port) {
        OPENVINO_THROW("Incorrect input port number for node ", getName());
    }

    if (inputShapes[port].hasZeroDims()) {
        return true;
    }
    auto edge = getParentEdgeAt(port);
    if (one_of(edge->getStatus(), Edge::Status::Allocated, Edge::Status::Validated)) {
        auto&& mem = edge->getMemory();
        if (mem.isDefined() && !mem.getDesc().empty()) {
            return mem.getShape().hasZeroDims();
        }
    }
    return false;
}

bool Node::isOutputTensorAtPortEmpty(size_t port) const {
    if (outputShapes.size() <= port) {
        OPENVINO_THROW("Incorrect output port number for node ", getName());
    }
    if (outputShapes[port].isStatic()) {
        return outputShapes[port].hasZeroDims();
    }
    auto&& mem = getChildEdgeAt(port)->getMemory();
    if (mem.isDefined() && !mem.getDesc().empty()) {
        return mem.getShape().hasZeroDims();
    }
    return false;
}

bool Node::hasEmptyInputTensors() const {
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        if (isInputTensorAtPortEmpty(i)) {
            return true;
        }
    }
    return false;
}

bool Node::hasEmptyOutputTensors() const {
    for (size_t i = 0; i < outputShapes.size(); i++) {
        if (isOutputTensorAtPortEmpty(i)) {
            return true;
        }
    }
    return false;
}

bool Node::inputShapesDefined() const {
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        if (!getParentEdgeAt(i)->getMemory().getDesc().isDefined()) {
            return false;
        }
    }
    return true;
}

bool Node::outputShapesDefined() const {
    for (size_t i = 0; i < outputShapes.size(); i++) {
        if (!getChildEdgeAt(i)->getMemory().getDesc().isDefined()) {
            return false;
        }
    }
    return true;
}

bool Node::shapesDefined() const {
    return inputShapesDefined() && outputShapesDefined();
}

bool Node::needPrepareParams() const {
    return inputShapesModified();
}

bool Node::inputShapesModified() const {
    if (lastInputDims.size() != getParentEdges().size()) {
        if (lastInputDims.empty()) {
            return true;
        }
        OPENVINO_THROW("Input dims and parent edges number mismatch!");
    }

    for (size_t i = 0; i < lastInputDims.size(); i++) {
        if (lastInputDims[i] != getParentEdgeAt(i)->getMemory().getStaticDims()) {
            return true;
        }
    }
    return false;
}

bool Node::needShapeInfer() const {
    return inputShapesModified();
}

std::vector<VectorDims> Node::shapeInferGeneric(const std::vector<Shape>& shapes) const {
    try {
        std::vector<std::reference_wrapper<const VectorDims>> input_shapes;
        auto input_value_port_mask = shapeInference->get_port_mask();

        input_shapes.reserve(shapes.size());
        for (size_t i = 0; i < shapes.size(); i++) {  // NOLINT(modernize-loop-convert)
            input_shapes.emplace_back(std::ref(shapes[i].getStaticDims()));
        }

        std::unordered_map<size_t, MemoryPtr> input_values;
        if (input_value_port_mask) {
            for (size_t port = 0; port < inputShapes.size(); ++port) {
                if (input_value_port_mask & (1 << port)) {
                    input_values[port] = getSrcMemoryAtPort(port);
                }
            }
        }

        auto result = shapeInference->infer(input_shapes, input_values);
        if (ShapeInferStatus::success != result.status) {
            OPENVINO_THROW("Unexpected: Shape inference unexpectedly skipped");
        }

        return std::move(result.dims);
    } catch (const std::exception& exp) {
        OPENVINO_THROW("Shape inference of ", getTypeStr(), " node with name ", getName(), " failed: ", exp.what());
    }
}

IShapeInfer::Result Node::shapeInfer() const {
    std::vector<std::reference_wrapper<const VectorDims>> input_shapes;
    auto input_value_port_mask = shapeInference->get_port_mask();

    input_shapes.reserve(inputShapes.size());
    for (size_t port = 0; port < inputShapes.size(); ++port) {
        input_shapes.emplace_back(std::ref(getParentEdgeAt(port)->getMemory().getStaticDims()));
    }

    std::unordered_map<size_t, MemoryPtr> input_values;
    if (input_value_port_mask) {
        for (size_t port = 0; port < inputShapes.size(); ++port) {
            if (input_value_port_mask & (1 << port)) {
                input_values[port] = getSrcMemoryAtPort(port);
            }
        }
    }

    return shapeInference->infer(input_shapes, input_values);
}

void Node::updateLastInputDims() {
    if (lastInputDims.size() != getParentEdges().size()) {
        if (!lastInputDims.empty()) {
            OPENVINO_THROW("Input dims and parent edges number mismatch!");
        }
        lastInputDims.resize(getParentEdges().size());
    }

    for (size_t i = 0; i < lastInputDims.size(); i++) {
        lastInputDims[i] = getParentEdgeAt(i)->getMemory().getDesc().getShape().getDims();
    }
}

bool Node::canFuseSimpleOperation(const NodePtr& node) const {
    if (node->getType() == Type::FakeQuantize) {
        bool ret = node->getAlgorithm() != Algorithm::FQBinarization;
        for (size_t i = 1; i < node->getParentEdges().size(); i++) {
            ret &= node->getParentEdgeAt(i)->getParent()->getChildEdges().size() == 1;
        }
        return ret;
    }
    if (node->getType() == Type::Eltwise) {
        return DnnlExtensionUtils::isUnarySupportedAsPostOp(node->getAlgorithm()) ||
               node->canBePerformedAsScaleShift(this);
    }
    return false;
}

void Node::addFusedNode(const NodePtr& fusingNode) {
    fusedWith.push_back(fusingNode);
}

void Node::addSupportedPrimDesc(const std::vector<PortConfigurator>& inPortConfigs,
                                const std::vector<PortConfigurator>& outPortConfigs,
                                impl_desc_type implType) {
    auto fill_port = [](const PortConfigurator& portConfigurator,
                        const Shape& shape,
                        ov::element::Type prc,
                        std::vector<PortConfig>& port) -> bool {
        // In order to simplify particular node initialization logic we just don't add config in case target shape is
        // not supported by blockedDescCreator. This should be suitable for major of scenarios since almost all nodes
        // add `ncsp` blockedDescCreator which supports any shape rank.
        if (shape.getRank() < portConfigurator.blockedDescCreator->getMinimalRank()) {
            return false;
        }

        PortConfig portConfig;
        portConfig.inPlace(portConfigurator.inPlace);
        portConfig.constant(portConfigurator.constant);
        portConfig.setMemDesc(portConfigurator.blockedDescCreator->createSharedDesc(prc, shape));

        port.push_back(std::move(portConfig));

        return true;
    };

    NodeConfig config;
    for (size_t i = 0; i < inPortConfigs.size(); i++) {
        auto shape = inPortConfigs[i].shape.getRank() == 0 ? getInputShapeAtPort(i) : inPortConfigs[i].shape;
        auto prc =
            (inPortConfigs[i].prc == ov::element::dynamic) ? getOriginalInputPrecisionAtPort(i) : inPortConfigs[i].prc;
        if (!fill_port(inPortConfigs[i], shape, prc, config.inConfs)) {
            return;
        }
    }

    for (size_t i = 0; i < outPortConfigs.size(); i++) {
        auto dims = outPortConfigs[i].shape.getRank() == 0 ? getOutputShapeAtPort(i) : outPortConfigs[i].shape;
        auto prc = (outPortConfigs[i].prc == ov::element::dynamic) ? getOriginalOutputPrecisionAtPort(i)
                                                                   : outPortConfigs[i].prc;
        if (!fill_port(outPortConfigs[i], dims, prc, config.outConfs)) {
            return;
        }
    }

    supportedPrimitiveDescriptors.emplace_back(config, implType);
}

void Node::fuseDQScales(const float* scaleData, const size_t scaleSize) {
    if (DQScales.empty()) {
        DQScales.resize(scaleSize, 1.0);
    }
    OPENVINO_ASSERT(scaleSize == 1 || DQScales.size() == 1 || DQScales.size() == scaleSize,
                    "set invalid scales size , DQScales vector size: ",
                    DQScales.size(),
                    ", scale data size: ",
                    scaleSize,
                    "Node: ##",
                    getName());
    if (scaleSize > DQScales.size()) {
        DQScales.resize(scaleSize, DQScales[0]);
    }
    if (1 == scaleSize) {
        std::transform(DQScales.begin(), DQScales.end(), DQScales.begin(), [=](float val) {
            return (scaleData[0] * val);
        });
    } else {
        for (size_t i = 0; i < DQScales.size(); i++) {
            DQScales[i] *= scaleData[i];
        }
    }
    if (std::all_of(DQScales.begin(), DQScales.end(), [OV_CAPTURE_CPY_AND_THIS](float val) {
            return (val == DQScales[0]);
        })) {
        DQScales.resize(1);
    }
}

int Node::inPlaceInputPort(int portIdx) const {
    if (inputShapes.empty()) {
        // special case - a dead end node
        return -1;
    }

    const NodeDesc* selected_pd = getSelectedPrimitiveDescriptor();
    if (!selected_pd) {
        OPENVINO_THROW("Cannot find selected primitive descriptor for node: ", getName());
    }

    const auto& conf = selected_pd->getConfig();

    OPENVINO_ASSERT(portIdx >= 0 && portIdx < static_cast<int>(conf.inConfs.size()),
                    "Wrong portIndx: ",
                    portIdx,
                    " acceptable interval: [0, ",
                    conf.inConfs.size(),
                    ")");

    return conf.inConfs[portIdx].inPlace();
}

int Node::inPlaceOutPort(int portIdx) const {
    if (outputShapes.empty()) {
        // special case - a dead end node
        return -1;
    }

    const NodeDesc* selected_pd = getSelectedPrimitiveDescriptor();
    if (!selected_pd) {
        OPENVINO_THROW("Cannot find selected primitive descriptor for node: ", getName());
    }

    const auto& conf = selected_pd->getConfig();

    OPENVINO_ASSERT(portIdx >= 0 && portIdx < static_cast<int>(conf.outConfs.size()),
                    "Wrong portIndx: ",
                    portIdx,
                    " acceptable interval: [0, ",
                    conf.outConfs.size(),
                    ")");

    return conf.outConfs[portIdx].inPlace();
}

void Node::resolveInPlaceDirection() {
    enum InplaceDirectionType : uint8_t { UP, DOWN, CYCLIC, NONE };
    enum PortType : uint8_t { INPUT, OUTPUT };

    auto inPlaceDirection = [](const Node* node, PortType portType, int portNum) -> InplaceDirectionType {
        if (PortType::INPUT == portType) {
            auto inPlaceInpPort = node->inPlaceInputPort(portNum);
            if (inPlaceInpPort >= 0) {
                auto inPlaceOutPort = node->inPlaceOutPort(inPlaceInpPort);
                if (inPlaceOutPort == inPlaceInpPort) {
                    return InplaceDirectionType::CYCLIC;
                }
                if (inPlaceOutPort < 0) {
                    return InplaceDirectionType::DOWN;
                }
                OPENVINO_THROW("Non trivial inPlace memory dependency has been detected");
            }
            // the requested port has a negative inPlace tag, let's check whether it is referenced from the output
            auto& config = node->getSelectedPrimitiveDescriptor()->getConfig();
            for (auto& portConf : config.outConfs) {
                if (portConf.inPlace() == portNum) {
                    return InplaceDirectionType::UP;
                }
            }
        } else if (PortType::OUTPUT == portType) {
            auto inPlaceOutPort = node->inPlaceOutPort(portNum);
            if (inPlaceOutPort >= 0) {
                auto inPlaceInpPort = node->inPlaceInputPort(inPlaceOutPort);
                if (inPlaceOutPort == inPlaceInpPort) {
                    return InplaceDirectionType::CYCLIC;
                }
                if (inPlaceInpPort < 0) {
                    return InplaceDirectionType::UP;
                }
                OPENVINO_THROW("Non trivial inPlace memory dependency has been detected");
            }
            // the requested port has a negative inPlace tag, let's check whether it is referenced from the input
            auto& config = node->getSelectedPrimitiveDescriptor()->getConfig();
            for (auto& portConf : config.inConfs) {
                if (portConf.inPlace() == portNum) {
                    return InplaceDirectionType::DOWN;
                }
            }
        }
        return InplaceDirectionType::NONE;
    };

    auto& inpEdges = getParentEdges();
    for (auto& wEdge : inpEdges) {
        if (auto pEdge = wEdge.lock()) {
            auto inpPort = pEdge->getOutputNum();
            auto inPlaceInpPort = inPlaceInputPort(inpPort);
            if (inPlaceInpPort < 0 ||
                inPlaceDirection(this, PortType::INPUT, inpPort) != InplaceDirectionType::CYCLIC) {
                continue;
            }
            // inPlace memory cyclic dependency detected, need to resolve
            // let's check the parent node first
            auto pParent = pEdge->getParent().get();
            auto parentInPlaceDirection = inPlaceDirection(pParent, PortType::OUTPUT, pEdge->getInputNum());
            if (parentInPlaceDirection == InplaceDirectionType::UP) {
                auto config = getSelectedPrimitiveDescriptor()->getConfig();
                config.inConfs[inpPort].inPlace(-1);
                initDescriptor(config);
            } else if (parentInPlaceDirection == InplaceDirectionType::DOWN) {
                // search if siblings already have downstream direction
                auto downstreamPeers = [&] {
                    for (auto& peerEdge : pParent->getChildEdgesAtPort(pEdge->getInputNum())) {
                        auto peerNode = peerEdge->getChild().get();
                        if (peerNode == this) {
                            continue;
                        }
                        if (inPlaceDirection(peerNode, PortType::INPUT, peerEdge->getOutputNum()) ==
                            InplaceDirectionType::DOWN) {
                            return true;
                        }
                    }
                    return false;
                }();
                if (downstreamPeers) {
                    // when there is an downstream peer we have to resolve upstream inplace for the node
                    // to avoid inplace conflict
                    auto config = getSelectedPrimitiveDescriptor()->getConfig();
                    config.inConfs[inpPort].inPlace(-1);
                    initDescriptor(config);
                } else {
                    auto config = getSelectedPrimitiveDescriptor()->getConfig();
                    config.outConfs[inPlaceInpPort].inPlace(-1);
                    initDescriptor(config);
                }
            } else {
                // the parent node does not use inPlace memory, let's check children
                std::function<InplaceDirectionType(const Node* node, int portIdx)> searchNonCyclicDirection;
                searchNonCyclicDirection = [&](const Node* node, int portIdx) -> InplaceDirectionType {
                    auto childEdges = node->getChildEdgesAtPort(portIdx);
                    for (auto& edge : childEdges) {
                        auto pChild = edge->getChild().get();
                        auto result = inPlaceDirection(pChild, PortType::INPUT, edge->getOutputNum());
                        if (InplaceDirectionType::UP == result || InplaceDirectionType::DOWN == result) {
                            return result;
                        }
                        if (InplaceDirectionType::CYCLIC == result) {
                            return searchNonCyclicDirection(pChild, pChild->inPlaceInputPort(edge->getOutputNum()));
                        }
                    }
                    return InplaceDirectionType::NONE;
                };
                auto result = searchNonCyclicDirection(this, inPlaceInpPort);
                if (InplaceDirectionType::UP == result) {
                    auto config = getSelectedPrimitiveDescriptor()->getConfig();
                    config.inConfs[inpPort].inPlace(-1);
                    initDescriptor(config);
                } else if (InplaceDirectionType::DOWN == result) {
                    auto config = getSelectedPrimitiveDescriptor()->getConfig();
                    config.outConfs[inPlaceInpPort].inPlace(-1);
                    initDescriptor(config);
                } else if (InplaceDirectionType::NONE == result) {
                    // resolve cyclic inplace to downstream instead of upstream for the node
                    // when there is only one output referencing to the edges of it,
                    // thus benefits zero-copy of outputs.
                    size_t numConflicts = 0;

                    // the parent node does not use inPlace memory, but it is an Input.
                    if (Type::Input == pParent->getType() || Type::MemoryInput == pParent->getType()) {
                        auto config = getSelectedPrimitiveDescriptor()->getConfig();
                        config.inConfs[inpPort].inPlace(-1);
                        initDescriptor(config);
                        continue;
                    }

                    // search descendants
                    if (numConflicts <= 1) {
                        // note: there are only non-inplace or cyclic-inplace descendants at the moment.
                        std::function<void(const Node* node, int portIdx)> searchReferencingOutput;
                        searchReferencingOutput = [&](const Node* node, int portIdx) -> void {
                            if (numConflicts > 1) {
                                return;  // early stop
                            }
                            auto childEdges = node->getChildEdgesAtPort(portIdx);
                            for (auto& edge : childEdges) {
                                auto pChild = edge->getChild().get();
                                if (Type::Output == pChild->getType()) {
                                    numConflicts++;
                                } else {
                                    auto result = inPlaceDirection(pChild, PortType::INPUT, edge->getOutputNum());
                                    if (InplaceDirectionType::CYCLIC == result) {
                                        return searchReferencingOutput(pChild,
                                                                       pChild->inPlaceInputPort(edge->getOutputNum()));
                                    }
                                }
                            }
                        };
                        searchReferencingOutput(this, inPlaceInpPort);
                    }

                    // search siblings
                    if (numConflicts <= 1) {
                        // note: the parent node does not use inPlace memory at the moment, let's check the siblings
                        for (auto& peerEdge : pParent->getChildEdgesAtPort(pEdge->getInputNum())) {
                            auto peerNode = peerEdge->getChild().get();
                            if (peerNode == this) {
                                continue;
                            }
                            if (Type::Output == peerNode->getType()) {
                                numConflicts++;
                            } else {
                                auto result = inPlaceDirection(peerNode, PortType::INPUT, peerEdge->getOutputNum());
                                if (one_of(result, InplaceDirectionType::DOWN, InplaceDirectionType::CYCLIC)) {
                                    numConflicts++;
                                }
                            }
                        }
                    }

                    if (numConflicts == 1) {  // downstream to make the only output edge be referenced.
                        auto config = getSelectedPrimitiveDescriptor()->getConfig();
                        config.outConfs[inPlaceInpPort].inPlace(-1);
                        initDescriptor(config);
                    } else {  // the default direction of upstream
                        auto config = getSelectedPrimitiveDescriptor()->getConfig();
                        config.inConfs[inpPort].inPlace(-1);
                        initDescriptor(config);
                    }
                } else {
                    OPENVINO_THROW("A node without an inPlace memory cyclic dependency has not been found");
                }
            }
        }
    }
}

#ifndef CPU_DEBUG_CAPS
std::ostream& operator<<(std::ostream& out, const Node& node) {
    return out << "Node " << node.getName() << " of type " << node.getTypeStr() << "\n";
}

std::ostream& operator<<(std::ostream& out, const Node* node) {
    return operator<<(out, (*node));
}
#endif

}  // namespace ov::intel_cpu

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node.h"
#include "edge.h"
#include "extension_mngr.h"
#include "itt.h"

#include "caseless.hpp"
#include <vector>
#include <string>
#include <limits>
#include <cstdint>
#include <unordered_map>

#include "nodes/concat.h"
#include "nodes/conv.h"
#include "nodes/deconv.h"
#include "nodes/eltwise.h"
#include "nodes/matmul.h"
#include "nodes/fullyconnected.h"
#include "nodes/generic.h"
#include "nodes/if.h"
#include "nodes/input.h"
#include "nodes/lrn.h"
#include "nodes/pooling.h"
#include "nodes/reorder.h"
#include "nodes/reshape.h"
#include "nodes/softmax.h"
#include "nodes/tile.h"
#include "nodes/split.h"
#include "nodes/pad.h"
#include "nodes/transpose.h"
#include "nodes/memory.hpp"
#include "nodes/mvn.h"
#include "nodes/normalize.h"
#include "nodes/reduce.h"
#include "nodes/tensoriterator.h"
#include "nodes/scatter_update.h"
#include "nodes/interpolate.h"
#include "nodes/depth_to_space.h"
#include "nodes/space_to_depth.h"
#include "nodes/strided_slice.h"
#include "nodes/shuffle_channels.h"
#include "nodes/reference.h"
#include "nodes/fake_quantize.h"
#include "dnnl_extension_utils.h"

#include "nodes/common/cpu_memcpy.h"
#include "utils/rt_info/memory_formats_attribute.hpp"
#include <ngraph/opsets/opset1.hpp>

#include <dnnl_types.h>
#include <dnnl_debug.h>
#include <ie_ngraph_utils.hpp>
#include "utils/general_utils.h"
#include "utils/cpu_utils.hpp"
#include "utils/verbose.h"
#include "nodes/common/cpu_convert.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <common/primitive_desc.hpp>
#include <common/primitive_desc_iface.hpp>

using namespace dnnl;
using namespace openvino;
using namespace ov::intel_cpu::node;

using namespace InferenceEngine::details;

namespace ov {
namespace intel_cpu {

Node::NodesFactory & Node::factory() {
    static NodesFactory factoryInstance;
    return factoryInstance;
}

Node::Node(const std::shared_ptr<ngraph::Node>& op,
           const GraphContext::CPtr ctx,
           const ShapeInferFactory& shapeInferFactory)
    : selectedPrimitiveDescriptorIndex(-1),
      permanent(false),
      temporary(false),
      constant(ConstantType::Unknown),
      context(ctx),
      engine(ctx->getEngine()),
      name(op->get_friendly_name()),
      typeStr(op->get_type_name()),
      type(TypeFromName(op->get_type_name())),
      profiling(op->get_friendly_name()) {
    algorithm = Algorithm::Default;
    fusingPort = -1;
    const std::string errorPrefix = "Ngraph operation " + std::string(op->get_type_name()) + " with name " + op->get_friendly_name();

    for (size_t i = 0; i < op->get_input_size(); i++) {
        const auto &shape = op->get_input_partial_shape(i);
        if (shape.rank().is_dynamic()) {
            IE_THROW(Unexpected) << "CPU plug-in doesn't support " << getTypeStr() << " operation with dynamic rank. Operation name: " << getName();
        }

        bool isScalar = shape.rank().get_length() == 0;
        inputShapes.emplace_back(isScalar ? ngraph::PartialShape{1} : shape);
        originalInputPrecisions.emplace_back(details::convertPrecision(op->get_input_element_type(i)));
    }

    if (typeStr != "Result" && typeStr != "Assign") {
        if (op->get_output_size() == 0) {
            IE_THROW() << "Node with type '" << typeStr << "' and name '" << name << "' does not have any outputs.";
        }
        for (size_t i = 0; i < op->get_output_size(); i++) {
            const auto &shape = op->get_output_partial_shape(i);
            if (shape.rank().is_dynamic()) {
                IE_THROW(Unexpected) << "CPU plug-in doesn't support " << getTypeStr() << " operation with dynamic rank. Operation name: " << getName();
            }

            bool isScalar = shape.rank().get_length() == 0;
            outputShapes.emplace_back(isScalar ? ngraph::PartialShape{1} : shape);
            originalOutputPrecisions.emplace_back(details::convertPrecision(op->get_output_element_type(i)));
        }
    }

    isDynamic = std::any_of(inputShapes.begin(), inputShapes.end(), [](const Shape& shape){ return shape.isDynamic(); }) ||
                std::any_of(outputShapes.begin(), outputShapes.end(), [](const Shape& shape){ return shape.isDynamic(); });

    if (isDynamic) {
        shapeInference = shapeInferFactory.makeShapeInfer();
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

    std::string inputMemoryFormats = getInputMemoryFormats(op);
    if (!inputMemoryFormats.empty()) {
        std::istringstream stream(inputMemoryFormats);
        std::string str;
        while (getline(stream, str, ',')) {
            if (str.substr(0, 4) != "cpu:")
                continue;
            inputMemoryFormatsFilter.push_back(dnnl::utils::str2fmt(str.substr(4, str.size()).c_str()));
        }
    }

    std::string outputMemoryFormats = getOutputMemoryFormats(op);
    if (!outputMemoryFormats.empty()) {
        std::istringstream stream(outputMemoryFormats);
        std::string str;
        while (getline(stream, str, ',')) {
            if (str.substr(0, 4) != "cpu:")
                continue;
            outputMemoryFormatsFilter.push_back(dnnl::utils::str2fmt(str.substr(4, str.size()).c_str()));
        }
    }

    const auto it = rtInfo.find("enforceBF16evenForGraphTail");
    if (it != rtInfo.end()) {
        enforceBF16evenForGraphTail = it->second.as<bool>();
    }
}

Node::Node(const std::string& type, const std::string& name, const GraphContext::CPtr ctx)
    : selectedPrimitiveDescriptorIndex(-1),
      permanent(false),
      temporary(false),
      constant(ConstantType::Unknown),
      context(ctx),
      fusingPort(-1),
      engine(ctx->getEngine()),
      name(name),
      typeStr(type),
      type(TypeFromName(type)),
      profiling(name) {
    // TODO [NM]: What about filling inDims and outDims?
}

void Node::addEdge(const EdgeWeakPtr& edge) {
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

void Node::removeEdge(const EdgeWeakPtr& edge) {
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

void Node::remove() {
    auto parent_edges = parentEdges;
    for (const auto &parentEdge : parent_edges) {
        removeEdge(parentEdge);
    }
    auto child_edges = childEdges;
    for (const auto &childEdge : child_edges) {
        removeEdge(childEdge);
    }
}

bool Node::isEdgesEmpty(const std::vector<EdgeWeakPtr>& edges) const {
    for (auto &edge : edges) {
        if (edge.lock())
            return false;
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
    selectPreferPrimitiveDescriptor(getPrimitivesPriority(), false);
}

void Node::selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority, bool ignoreConstInputs) {
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
                        auto curDesc = getSupportedPrimitiveDescriptors()[i].getConfig().inConfs[j].getMemDesc();
                        auto parentDesc = parent_spd->getConfig().outConfs[inNum].getMemDesc();

                        if (curDesc->isCompatible(*parentDesc)) {
                            equalsLocalFormatCount++;
                            DEBUG_LOG(getName(), " pd[", i, "].inConfs[", j, "]"
                                      " is compatible with parent ", parentPtr->getName(),
                                      " outConfs[", inNum, "], equalsLocalFormatCount add to ", equalsLocalFormatCount);
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

bool Node::canBeInPlace() const {
    // TODO [DS]: enable inPlace for dynamic shapes
    if (isDynamicNode()) {
        return false;
    }

    if (getParentEdges().size() != 1 || getParentEdgeAt(0)->getParent()->getChildEdges().size() != 1 ||
            (getParentEdgeAt(0)->getParent()->isConstant() && !getParentEdgeAt(0)->getChild()->isConstant()))
        return false;

    // TODO: we need to extend this logic to properly handle all possible inplace conflicts
    if (getParentEdges().size() == 1 && getParentEdgeAt(0)->getParent()->getType() == Type::Reshape) {
        auto reshapeNode = getParentEdgeAt(0)->getParent();
        if (reshapeNode->getParentEdgeAt(0)->getParent()->getChildEdges().size() != 1)
            return false;
    }

    auto inShape = getInputShapeAtPort(0);
    for (size_t cIdx = 0; cIdx < outputShapes.size(); cIdx++) {
        if (getOutputShapeAtPort(cIdx) != inShape) {
            return false;
        }
    }
    return true;
}

void Node::resolveInPlaceEdges() {
    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (!selected_pd)
        IE_THROW() << "Cannot find selected primitive descriptor for node: " << getName();
    for (size_t i = 0; i < getParentEdges().size() && i < selected_pd->getConfig().inConfs.size(); i++) {
        auto parentEdge = getParentEdgeAt(i);

        if (parentEdge->getStatus() != Edge::Status::NotAllocated || selected_pd->getConfig().inConfs[i].inPlace() < 0)
            continue;

        auto memMgr = parentEdge->getMemory().getDnnlMemoryMngr();
        parentEdge->getMemoryPtr().reset(new Memory(getEngine()));
        parentEdge->getMemoryPtr()->Create(selected_pd->getConfig().inConfs[i].getMemDesc(), memMgr);

        parentEdge->changeStatus(Edge::Status::Allocated);
    }
    for (size_t i = 0; i < getChildEdges().size() && i < selected_pd->getConfig().outConfs.size(); i++) {
        auto childEdge = getChildEdgeAt(i);

        if (childEdge->getStatus() != Edge::Status::NotAllocated || selected_pd->getConfig().outConfs[i].inPlace() < 0)
            continue;

        auto memMgr = childEdge->getMemory().getDnnlMemoryMngr();
        childEdge->getMemoryPtr().reset(new Memory(getEngine()));
        childEdge->getMemoryPtr()->Create(selected_pd->getConfig().outConfs[i].getMemDesc(), memMgr);

        childEdge->changeStatus(Edge::Status::Allocated);
    }
}

MemoryDescPtr Node::getBaseMemDescAtInputPort(size_t portNum) const {
    if (auto primDesc = getSelectedPrimitiveDescriptor()) {
        const auto& inConfs = primDesc->getConfig().inConfs;
        if (inConfs.size() < portNum) {
            IE_THROW() << "Can't get input memory desc at port: " << portNum << ", incorrect port number";
        }
        return inConfs[portNum].getMemDesc();
    }
    IE_THROW() << "Can't get input memory desc, primitive descriptor is not selected";
}

MemoryDescPtr Node::getBaseMemDescAtOutputPort(size_t portNum) const {
    if (auto primDesc = getSelectedPrimitiveDescriptor()) {
        const auto& outConfs = primDesc->getConfig().outConfs;
        if (outConfs.size() < portNum) {
            IE_THROW() << "Can't get output memory desc at port: " << portNum << ", incorrect port number";
        }
        return outConfs[portNum].getMemDesc();
    }
    IE_THROW() << "Can't get output memory desc, primitive descriptor is not selected";
}

std::string Node::getPrimitiveDescriptorType() {
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
    SEARCH_TYPE(brgconv);
    SEARCH_TYPE(brgemm);
    SEARCH_TYPE(ref);

    SEARCH_TYPE(avx512);
    SEARCH_TYPE(amx);
    SEARCH_TYPE(avx2);
    SEARCH_TYPE(avx);
    SEARCH_TYPE(sse42);
    SEARCH_TYPE(blas);
    SEARCH_TYPE(any);
    SEARCH_TYPE(uni);

    SEARCH_TYPE(winograd);
    SEARCH_TYPE(sparse);
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
            if (selectedPrimitiveDesc->getConfig().inConfs[0].getMemDesc()->getPrecision() != InferenceEngine::Precision::U8) {
                str_type += "_" + std::string(selectedPrimitiveDesc->getConfig().inConfs[0].getMemDesc()->getPrecision().name());
            } else {
                str_type += "_I8";
            }
        } else {
            if (selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision() != InferenceEngine::Precision::U8) {
                str_type += "_" + std::string(selectedPrimitiveDesc->getConfig().outConfs[0].getMemDesc()->getPrecision().name());
            } else {
                str_type += "_I8";
            }
        }
    }

    return str_type;
}

const EdgePtr Node::getParentEdgeAt(size_t idx) const {
    if (idx >= parentEdges.size())
        IE_THROW() << "Node " << getName() << " contains less parent edges than " << idx;
    auto parentEdgePtr = parentEdges[idx].lock();
    if (!parentEdgePtr)
        IE_THROW() << "Node " << getName() << " contains empty parent edge for index " << idx;
    return parentEdgePtr;
}

const EdgePtr Node::getChildEdgeAt(size_t idx) const {
    if (idx >= childEdges.size())
        IE_THROW() << "Node " << getName() << " contains less child edges than " << idx;
    auto childEdgePtr = childEdges[idx].lock();
    if (!childEdgePtr)
        IE_THROW() << "Node " << getName() << " contains empty child edge for index " << idx;
    return childEdgePtr;
}

const std::vector<EdgePtr> Node::getParentEdgesAtPort(size_t idx) const {
    if (idx >= inputShapes.size())
        IE_THROW() << "Node " << getName() << " contains less input ports than " << idx;

    std::vector<EdgePtr> res;
    for (auto &edge_w : parentEdges) {
        auto edge = edge_w.lock();
        if (!edge)
            IE_THROW() << "Node " << getName() << " contains dead weak ptr";
        if (edge->getOutputNum() == idx) res.push_back(edge);
    }
    return res;
}

const std::vector<EdgePtr> Node::getChildEdgesAtPort(size_t idx) const {
    if (idx >= outputShapes.size())
        IE_THROW() << "Node " << getName() << " contains less output ports than " << idx;

    std::vector<EdgePtr> res;
    for (auto &edge_w : childEdges) {
        auto edge = edge_w.lock();
        if (!edge)
            IE_THROW() << "Node " << getName() << " contains dead weak ptr";
        if (edge->getInputNum() == idx) res.push_back(edge);
    }
    return res;
}


std::vector<memory::format_tag> Node::getAvailableFormatsForDims(const Shape &dims) const {
    if (dims.getRank() == 0)
        return {memory::format_tag::x};
    else if (dims.getRank() == 1)
        return {memory::format_tag::x};
    else if (dims.getRank() == 2)
        return {memory::format_tag::nc};
    else if (dims.getRank() == 3)
        return {memory::format_tag::tnc, memory::format_tag::ntc,
                memory::format_tag::ncw, memory::format_tag::nCw8c, memory::format_tag::nCw16c };
    else if (dims.getRank() == 4)
        return {memory::format_tag::nchw, memory::format_tag::nChw8c, memory::format_tag::nChw16c};
    else if (dims.getRank() == 5)
        return {memory::format_tag::ncdhw, memory::format_tag::nCdhw8c, memory::format_tag::nCdhw16c};
    return {memory::format_tag::any};
}

void Node::execute(dnnl::stream strm) {
    if (prim) {
        prim.execute(strm, primArgs);
    }
}

void Node::updateShapes() {
    IE_ASSERT(isDynamicNode()) << "Node::updateShapes() is called to a static shape node of type: " << getTypeStr() << " with name: " << getName();
    if (needShapeInfer()) {
        auto result = shapeInfer();
        if (ShapeInferStatus::success == result.status) {
            redefineOutputMemory(result.dims);
        }
    }
}

void Node::updateDynamicParams() {
    IE_ASSERT(isDynamicNode()) << "Node::updateDynamicParams() is called to a static shape node of type: " << getTypeStr() << " with name: " << getName();
    if (isExecutable()) {
        if (needPrepareParams()) {
            IE_ASSERT(inputShapesDefined()) << "Can't prepare params for " << getTypeStr() << " node with name: " << getName() <<
                " since the input shapes are not defined.";
            DEBUG_LOG(" prepareParams() on #", getExecIndex(), " ", getTypeStr(), " ", algToString(getAlgorithm()),
                      " ", getName(), " ", getOriginalLayers());
            prepareParams();
#ifdef CPU_DEBUG_CAPS
            if (prim) {
                auto pd_c = prim.get_primitive_desc();
                auto* pd = reinterpret_cast<const dnnl_primitive_desc*>(pd_c);
                DEBUG_LOG("verbose##", getName(), "##", pd->info(), "\n");
            }
#endif
        }
    }
}
void Node::executeDynamic(dnnl::stream strm) {
    if (isExecutable()) {
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

void Node::redefineOutputMemory(const std::vector<VectorDims> &newOutputShapes) {
    if (newOutputShapes.size() != outputShapes.size()) {
        IE_THROW() << "Number shapes mismatch with real outputs number for node with name: " << getName();
    }
    for (size_t i = 0; i < outputShapes.size(); i++) {
        const auto edges = getChildEdgesAtPort(i);

        // avoid 0D shape incompatible
        auto newOutputShape = newOutputShapes[i];
        if (newOutputShape.empty()) {
            newOutputShape.push_back(1);
        }

        const auto &currDesc = edges[0]->getMemory().getDesc();
        if (currDesc.getShape().isStatic() && currDesc.getShape().getStaticDims() == newOutputShape)
            continue;

        const bool hasZeroDims = std::count(std::begin(newOutputShape), std::end(newOutputShape), 0) > 0;
        const auto memDesc = getBaseMemDescAtOutputPort(i)->cloneWithNewDims(newOutputShape, hasZeroDims);
        for (size_t j = 0; j < edges.size(); j++) {
            edges[j]->getMemoryPtr()->redefineDesc(memDesc);
        }
    }
}

void Node::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto attr = initPrimitiveAttr();

    for (auto& desc : descs) {
        primitive_desc_iterator itpd;
        if (attr) {
            itpd = desc.createPrimitiveDescriptorIterator(engine, *attr);
        } else {
            itpd = desc.createPrimitiveDescriptorIterator(engine);
        }

        while (static_cast<bool>(itpd)) {
            NodeConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace(-1);
                portConfig.constant(false);
                auto desc = getSrcMemDesc(itpd, i);
                if (desc->getType() & MemoryDescType::Blocked) {
                    portConfig.setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(desc), BLOCKED_DESC_EMPTY_MASK);
                } else {
                    portConfig.setMemDesc(std::move(desc));
                }
                config.inConfs.push_back(portConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig portConfig;
                portConfig.inPlace(canBeInPlace() ? 0 : -1);
                portConfig.constant(false);
                auto desc = getDstMemDesc(itpd, i);
                if (desc->getType() & MemoryDescType::Blocked) {
                    portConfig.setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(desc), BLOCKED_DESC_EMPTY_MASK);
                } else {
                    portConfig.setMemDesc(std::move(desc));
                }
                config.outConfs.push_back(portConfig);
            }
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            if (!itpd.next_impl())
                break;
        }
    }
}

void Node::filterSupportedPrimitiveDescriptors() {
    // Compare by format tag
    auto areCompatible = [](const MemoryDesc& desc, dnnl::memory::format_tag fmt) -> bool {
        auto fmt_tdesc = DnnlBlockedMemoryDesc(desc.getShape(),
                                               DnnlExtensionUtils::IEPrecisionToDataType(desc.getPrecision()),
                                               fmt);
        return desc.isCompatible(fmt_tdesc);
    };

    if (!inputMemoryFormatsFilter.empty() || !outputMemoryFormatsFilter.empty()) {
        auto itpd = supportedPrimitiveDescriptors.begin();
        while (itpd != supportedPrimitiveDescriptors.end()) {
            const auto &config = itpd->getConfig();
            if (inputMemoryFormatsFilter.size() > config.inConfs.size() || outputMemoryFormatsFilter.size() > config.outConfs.size())
                IE_THROW() << "Incorrect number of input or output memory formats";

            bool isSuitableDesc = true;
            for (int i = 0; i < inputMemoryFormatsFilter.size(); i++) {
                const bool matched = areCompatible(*config.inConfs[i].getMemDesc(), inputMemoryFormatsFilter[i]);
                isSuitableDesc &= matched;
            }
            for (int i = 0; i < outputMemoryFormatsFilter.size(); i++) {
                const bool matched = areCompatible(*config.outConfs[i].getMemDesc(), outputMemoryFormatsFilter[i]);
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

void Node::initDescriptor(const NodeConfig& config) {
    if (!getSelectedPrimitiveDescriptor()) {
        return;
    }
    std::vector<MemoryDescPtr> inDescs;
    for (const auto& inConf : config.inConfs)
        inDescs.emplace_back(inConf.getMemDesc());
    std::vector<MemoryDescPtr> outDescs;
    for (const auto& outConf : config.outConfs)
        outDescs.emplace_back(outConf.getMemDesc());
    createDescriptor(inDescs, outDescs);

    AttrPtr attr = initPrimitiveAttr();

    auto* selectedPD = getSelectedPrimitiveDescriptor();
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
                dataConfig.inPlace(canBeInPlace() ? 0 : -1);
                dataConfig.constant(false);
                dataConfig.setMemDesc(getSrcMemDesc(itpd, i));
                cfg.inConfs.push_back(dataConfig);
            }

            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                PortConfig dataConfig;
                dataConfig.inPlace(-1);
                dataConfig.constant(false);
                dataConfig.setMemDesc(getDstMemDesc(itpd, i));
                cfg.outConfs.push_back(dataConfig);
            }
            impl_desc_type impl_type = parse_impl_name(itpd.impl_info_str());
            if (selected_count == selectedPrimitiveDescriptorIndex) {
                if (impl_type != selectedPD->getImplementationType()) {
                    IE_THROW() << getName() << " expected selectedPD impl_type: " << impl_type_to_string(selectedPD->getImplementationType())
                                << " but got: " << impl_type_to_string(impl_type);
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
            if (!selectedConfig.inConfs[i].getPortDesc()->isCompatible(*config.inConfs[i].getPortDesc()))
                IE_THROW() << "Incorrect descriptor for node: " << getName() << " on " << i << " intput port";
        }

        for (size_t i = 0; i < selectedConfig.outConfs.size(); i++) {
            if (!selectedConfig.outConfs[i].getPortDesc()->isCompatible(*config.outConfs[i].getPortDesc()))
                IE_THROW() << "Incorrect descriptor for node: " << getName() << " on " << i << " output port";
        }
        rightConfig = config;
    }

    selectedPD->setConfig(rightConfig);
}

void Node::prepareMemory(const std::vector<DnnlMemoryDescPtr>& intDescs) {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto &dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->isAllocated())
            IE_THROW() << "Destination memory didn't allocate for node " << getName()
                               << " to node " << getChildEdgeAt(i)->getChild()->getName() << ".";
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto &srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->isAllocated())
            IE_THROW() << "Destination memory didn't allocate for node " << getName()
                               << " from node " << getParentEdgeAt(i)->getParent()->getName() << ".";
    }

    if (internalBlobs.size() != intDescs.size()) {
        IE_THROW() << "Can't prepare memory for internal blob, internal blob and internal descs number do not match "
                   << internalBlobs.size() << " vs " << intDescs.size();
    }

    internalBlobMemory.clear();
    for (size_t i = 0; i < internalBlobs.size(); i++) {
        const auto &internalBlob = internalBlobs[i];

        auto create = [&] () {
            // TODO [DS]: internal blobs should be removed or rewritten using Memory object
            auto newDesc = MemoryDescUtils::convertToDnnlBlockedMemoryDesc(internalBlob->getTensorDesc());

            Memory memory{ engine };
            memory.Create(newDesc, internalBlob->buffer());

            MemoryPtr _ptr = MemoryPtr(new Memory(engine));
            _ptr->Create(*intDescs[i]);
            _ptr->SetData(memory);

            return _ptr;
        };

        MemoryPtr ptr;
        auto weightCache = context->getWeightsCache();
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

void Node::prepareMemory(dnnl::primitive_desc_iterator& itpd) {
    std::vector<DnnlMemoryDescPtr> intDescs;
    for (auto &it : internalBlobDesc)
        intDescs.push_back(it(itpd, 0));

    Node::prepareMemory(intDescs);
}

bool Node::isInPlace() {
    if (inplace == InPlaceType::Unknown) {
        auto selected_pd = getSelectedPrimitiveDescriptor();
        if (selected_pd == nullptr)
            IE_THROW() << "Preferable primitive descriptor is not set.";

        inplace = InPlaceType::NoInPlace;
        auto config = selected_pd->getConfig();
        for (auto &in : config.inConfs) {
            if (in.inPlace() >= 0) {
                inplace = InPlaceType::InPlace;
                break;
            }
        }
        for (auto &out : config.outConfs) {
            if (out.inPlace() >= 0) {
                inplace = InPlaceType::InPlace;
                break;
            }
        }
    }

    return inplace == InPlaceType::InPlace;
}

bool Node::isConstant() {
    if (constant == ConstantType::Unknown) {
        std::vector<NodePtr> checkNodes;
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

Node::ConstantType Node::checkConstant(LOOK look, std::vector<NodePtr>& checkNodes) {
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

void Node::addOriginalLayer(const std::string& layerName) {
    if (layerName.empty()) return;
    if (originalLayers.empty()) {
        originalLayers = layerName;
    } else {
        originalLayers += "," + layerName;
    }
}

void Node::cleanup() {
    internalBlobs.clear();

    for (auto it : fusedWith) {
        it->cleanup();
    }

    for (auto it : mergedWith) {
        it->cleanup();
    }
}

const std::vector<impl_desc_type>& Node::getPrimitivesPriority() {
    std::vector<impl_desc_type> priorities = {
            impl_desc_type::unknown,
            impl_desc_type::brgconv_avx512_amx_1x1,
            impl_desc_type::brgconv_avx512_amx,
            impl_desc_type::jit_avx512_amx_dw,
            impl_desc_type::jit_avx512_amx_1x1,
            impl_desc_type::jit_avx512_amx,
            // Brgconv kernels disabled in order to prevent perf degradations on non AMX HW
            // impl_desc_type::brgconv_avx512_1x1,
            // impl_desc_type::brgconv_avx512,
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

PortDescBasePtr Node::getConsistentInputDesc(const NodeConfig &config, size_t idx) const {
    int num = getParentEdgeAt(idx)->getInputNum();
    auto *selectedPD = getParentEdgeAt(idx)->getParent()->getSelectedPrimitiveDescriptor();
    if (!selectedPD)
        IE_THROW() << "Cannot get selected primitive descriptor for node: " << getParentEdgeAt(idx)->getParent()->getName();

    if (config.inConfs[idx].inPlace() >= 0) {
        auto inplaceIndx = static_cast<size_t>(config.inConfs[idx].inPlace());
        PortDescBasePtr outPortDesc;
        const auto& outConf = config.outConfs[inplaceIndx];
        if (outConf.inPlace() == idx) {
            outPortDesc = outConf.getPortDesc();
        } else {
            outPortDesc = getConsistentOutputDesc(config, inplaceIndx);
        }
        if (config.inConfs[idx].getPortDesc()->isCompatible(*outPortDesc)) {
            return outPortDesc;
        }
    }

    if (num >= 0) {
        auto parentConf = selectedPD->getConfig().outConfs[num];
        parentConf.setMemDesc(parentConf.getMemDesc()->cloneWithNewPrecision(config.inConfs[idx].getMemDesc()->getPrecision()));
        if (!parentConf.getMemDesc()->isDefined() && parentConf.inPlace() >= 0)
            getParentEdgeAt(idx)->getParent()->initOptimalPrimitiveDescriptor();
        parentConf = getParentEdgeAt(idx)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num];
        if (parentConf.getMemDesc()->isDefined() && config.inConfs[idx].getPortDesc()->isCompatible(*parentConf.getPortDesc())) {
            return parentConf.getPortDesc();
        }
    }

    return config.inConfs[idx].getPortDesc();
}

PortDescBasePtr Node::getConsistentOutputDesc(const NodeConfig &config, size_t idx) const {
    int num = getChildEdgeAt(idx)->getOutputNum();
    auto *selectedPD = getChildEdgeAt(idx)->getChild()->getSelectedPrimitiveDescriptor();
    if (!selectedPD)
        IE_THROW() << "Cannot get selected primitive descriptor for node: " << getChildEdgeAt(idx)->getChild()->getName();

    if (config.outConfs[idx].inPlace() >= 0) {
        auto inplaceIndx = static_cast<size_t>(config.outConfs[idx].inPlace());
        PortDescBasePtr inpPortDesc;
        const auto& inpConf = config.inConfs[inplaceIndx];
        if (inpConf.inPlace() == idx) {
            inpPortDesc = inpConf.getPortDesc();
        } else {
            inpPortDesc = getConsistentInputDesc(config, inplaceIndx);
        }
        if (config.outConfs[idx].getPortDesc()->isCompatible(*inpPortDesc)) {
            return inpPortDesc;
        }
    }

    if (num >= 0) {
        auto childConf = selectedPD->getConfig().inConfs[num];
        childConf.setMemDesc(childConf.getMemDesc()->cloneWithNewPrecision(config.outConfs[idx].getMemDesc()->getPrecision()));
        if (!childConf.getMemDesc()->isDefined() && childConf.inPlace() >= 0)
            getChildEdgeAt(idx)->getChild()->initOptimalPrimitiveDescriptor();
        childConf = getChildEdgeAt(idx)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num];
        if (childConf.getMemDesc()->isDefined() && config.outConfs[idx].getPortDesc()->isCompatible(*childConf.getPortDesc())) {
            return childConf.getPortDesc();
        }
    }

    return config.outConfs[idx].getPortDesc();
}

void Node::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (isDynamicNode()) {
        // it is assumed that the nodes will define dense tensors on output edges
        // if it is not the case the implementation must redefine this behaviour
        for (size_t i = 0; i < config.outConfs.size(); i++) {
            auto outMemDesc = config.outConfs[i].getMemDesc();
            if (outMemDesc && (outMemDesc->getType() & Blocked)) {
                config.outConfs[i].setMemDesc(std::dynamic_pointer_cast<BlockedMemoryDesc>(outMemDesc), BLOCKED_DESC_FULL_MASK);
            }
        }
    } else {
        for (size_t i = 0; i < config.inConfs.size(); i++) {
            auto inpPortDesc = getConsistentInputDesc(config, i);
            config.inConfs[i].setMemDesc(inpPortDesc->getMemDesc());
        }

        for (size_t i = 0; i < config.outConfs.size(); i++) {
            auto outPortDesc = getConsistentOutputDesc(config, i);
            config.outConfs[i].setMemDesc(outPortDesc->getMemDesc());
        }
    }
    if (getType() != Type::RNNSeq && getType() != Type::RNNCell) {
        initDescriptor(config);
    }
}

bool Node::isConfigDefined(const NodeConfig &config) const {
    for (const auto& configs : {config.inConfs, config.outConfs}) {
        for (const auto &dc : configs) {
            if (!dc.getMemDesc()->isDefined())
                return false;
        }
    }
    return true;
}

MemoryDescPtr Node::getSrcMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    if (getInputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(primitive_desc_it.src_desc(idx), getInputShapeAtPort(idx));
    }
    return DnnlExtensionUtils::makeDescriptor(primitive_desc_it.src_desc(idx));
}

MemoryDescPtr Node::getDstMemDesc(dnnl::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    if (getOutputShapeAtPort(idx).isDynamic()) {
        return DnnlExtensionUtils::makeUndefinedDesc(primitive_desc_it.dst_desc(idx), getOutputShapeAtPort(idx));
    }
    return DnnlExtensionUtils::makeDescriptor(primitive_desc_it.dst_desc(idx));
}

int Node::batchToProcess() const {
    return dynBatchLim == 0 ? getMaxBatch() : std::min<int>(getMaxBatch(), dynBatchLim);
}

// TODO [DS]: how we should process this for dynamic shape?
size_t Node::getMaxBatch() const {
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

void Node::setDynamicBatchLim(int lim) {
    dynBatchLim = lim;

    auto setDynamicBatch = [this](int argType, int newBatch) {
        auto param = primArgs.find(argType);
        if (param != primArgs.end()) {
            auto oldMem = param->second;
            dnnl::memory::desc newMemDesc(oldMem.get_desc());
            newMemDesc.data.dims[0] = newBatch;
            newMemDesc.data.padded_dims[0] = newBatch;
            dnnl::memory newMem(newMemDesc, oldMem.get_engine(), oldMem.get_data_handle());
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

void Node::appendPostOpArgs(const dnnl::primitive_attr& attr,
                                  std::unordered_map<int, dnnl::memory>& primArgs,
                                  const std::unordered_map<int, MemoryPtr>& postOpsArgs) {
    for (auto & entry : postOpsArgs) {
        primArgs[entry.first] = entry.second->GetPrimitive();
    }
}

bool Node::isFusedWith(Type fusedNodeType) const {
    for (auto fusedNode : fusedWith) {
        if (fusedNode->type == fusedNodeType)
            return true;
    }

    return false;
}

InferenceEngine::Layout Node::getWeightsLayoutByDims(SizeVector dims, bool isGrouped) {
    switch (dims.size()) {
        case 0:
            return InferenceEngine::Layout::SCALAR;
        case 1:
            return InferenceEngine::Layout::C;
        case 2:
            return InferenceEngine::Layout::NC;
        case 3:
            return InferenceEngine::Layout::CHW;
        case 4:
            return InferenceEngine::Layout::OIHW;
        case 5:
            return isGrouped ? InferenceEngine::Layout::GOIHW : InferenceEngine::Layout::OIDHW;
        case 6:
            return isGrouped ? InferenceEngine::Layout::GOIDHW : InferenceEngine::Layout::BLOCKED;
        default:
            return InferenceEngine::Layout::BLOCKED;
    }
}

void Node::appendPostOps(dnnl::post_ops& ops, const VectorDims &postOpDims, std::unordered_map<int, MemoryPtr>& postOpsMem, const int channelAxis) {
    IE_THROW() << "Fusing of " << NameFromType(this->getType()) << " operation is not implemented";
}

void Node::appendPostOps(dnnl::post_ops& ops, const VectorDims &postOpDims, std::vector<const void*>& postOpsMem, const int channelAxis) {
    IE_THROW() << "Fusing of " << NameFromType(this->getType()) << " operation is not implemented";
}

std::vector<InferenceEngine::Precision> Node::getInputPrecisions() const {
    std::vector<InferenceEngine::Precision> inputPrecisions;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto parentEdge = getParentEdgeAt(i);
        if (parentEdge && parentEdge->getStatus() == Edge::Status::Validated) {
            inputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToIEPrecision((parentEdge->getMemoryPtr()->GetDataType())));
        }
    }
    return inputPrecisions;
}

std::vector<InferenceEngine::Precision> Node::getOutputPrecisions() const {
    std::vector<InferenceEngine::Precision> outputPrecisions;
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto childEdge = getChildEdgeAt(i);
        if (childEdge && childEdge->getStatus() == Edge::Status::Validated) {
            outputPrecisions.emplace_back(DnnlExtensionUtils::DataTypeToIEPrecision((childEdge->getMemoryPtr()->GetDataType())));
        }
    }
    return outputPrecisions;
}

InferenceEngine::Precision Node::getRuntimePrecision() const {
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

Node* Node::NodesFactory::create(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context) {
    // getExceptionDescWithoutStatus removes redundant information from the exception message. For instance, the NotImplemented
    // exception is generated in the form: full_path_to_src_file:line_number [ NOT_IMPLEMENTED ] reason.
    // An example for gather node:
    // /path-to-openVino-root/src/plugins/intel_cpu/nodes/gather.cpp:42 [ NOT_IMPLEMENTED ] Only opset7 Gather operation is supported
    // The most important part of the message is the reason, so the lambda trims everything up to "]"
    // Note that the op type and its friendly name will also be provided if we fail to create the node.
    auto getExceptionDescWithoutStatus = [](const InferenceEngine::Exception& ex) {
        std::string desc = ex.what();
        size_t pos = desc.find("]");
        if (pos != std::string::npos) {
            if (desc.size() == pos + 1) {
                desc.erase(0, pos + 1);
            } else {
                desc.erase(0, pos + 2);
            }
        }
        return desc;
    };
    Node *newNode = nullptr;
    std::string errorMessage;
    {
        std::unique_ptr<Node> ol(createNodeIfRegistered(intel_cpu, Type::Generic, op, context));
        if (ol != nullptr && ol->created(context->getExtensionManager()))
            newNode = ol.release();
    }

    if (newNode == nullptr) {
        try {
            std::unique_ptr<Node> ol(createNodeIfRegistered(intel_cpu, TypeFromName(op->get_type_name()), op, context));
            if (ol != nullptr && ol->created(context->getExtensionManager()))
                newNode = ol.release();
        } catch (const InferenceEngine::Exception& ex) {
            if (dynamic_cast<const InferenceEngine::NotImplemented*>(&ex) != nullptr) {
                errorMessage += getExceptionDescWithoutStatus(ex);
            } else {
                throw;
            }
        }
    }

    if (newNode == nullptr) {
        try {
            std::unique_ptr<Node> ol(new Reference(op, context, errorMessage));
            if (ol != nullptr && ol->created(context->getExtensionManager()))
                newNode = ol.release();
        } catch (const InferenceEngine::Exception& ex) {
            if (dynamic_cast<const InferenceEngine::NotImplemented*>(&ex) != nullptr) {
                const auto currErrorMess = getExceptionDescWithoutStatus(ex);
                if (!currErrorMess.empty())
                    errorMessage += errorMessage.empty() ? currErrorMess : "\n" + currErrorMess;
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
        IE_THROW() << "Unsupported operation of type: " << op->get_type_name() << " name: " << op->get_friendly_name() << errorDetails;
    }

    return newNode;
}

bool Node::canBePerformedAsScaleShift(const Node *parentNode) const {
    IE_ASSERT(parentNode);

    size_t fusingPort = 0;
    const auto channelAxis = parentNode->getFusingAxis();

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        Node *node = getParentEdgesAtPort(i)[0]->getParent().get();
        if (node == nullptr) {
            IE_THROW() << "Cannot get parent node for " << getName() << " on " << i << " port";
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
            if (i == fusingPort)
                continue;
            auto& weightShape = getInputShapeAtPort(i).getDims();
            if (getParentEdgesAtPort(i)[0]->getParent()->getChildEdges().size() != 1 ||
                !isPerTensorOrPerChannelBroadcastable(dataShape, weightShape, channelAxis, true))
                return false;
        }
        return true;
    };

    const auto isConvertablePowerStatic = [&]() {
        if (getAlgorithm() == Algorithm::EltwisePowerStatic) {
            const auto eltwise = dynamic_cast<const Eltwise *>(this);
            if (!eltwise) {
                IE_THROW() << "Cannot cast " << getName() << " to Eltwise";
            }
            return eltwise->getAlpha() == 1.0f;
        }
        return false;
    };

    return (one_of(getAlgorithm(), Algorithm::EltwiseAdd,
                                   Algorithm::EltwiseMultiply,
                                   Algorithm::EltwiseSubtract,
                                   Algorithm::EltwiseDivide,
                                   Algorithm::EltwisePrelu,
                                   Algorithm::EltwiseMulAdd) && isBroadcastableToDataInput())
            || isConvertablePowerStatic();
}

// @todo shifts for Subtract and scales for Divide are replaced with
// Add (with opposite sign) and Multiply (with inverse value) for legacy dephwise post ops
// This can be avoided after dephwise post ops are gone
std::pair<std::vector<float>, std::vector<float>> Node::getScalesAndShifts(const Node *parentNode) const {
    std::vector<float> scales, shifts;

    const auto fillValuesFrom = [&](const NodePtr& constInput, std::vector<float>& buffer) {
        auto *constInputNode = dynamic_cast<node::Input *>(constInput.get());
        if (!constInputNode) {
            IE_THROW() << "Cannot cast " << constInput->getName() << " to Input";
        }
        auto constBlob = constInputNode->getMemoryPtr();
        const auto elementsCount = constBlob->GetDescWithType<BlockedMemoryDesc>()->getPaddedElementsCount();
        buffer.resize(elementsCount);
        cpu_convert(constBlob->GetPtr(),
                    &buffer[0],
                    DnnlExtensionUtils::DataTypeToIEPrecision(constBlob->GetDataType()),
                    Precision::FP32,
                    elementsCount);
    };

    const auto constPort = getParentEdgesAtPort(0)[0]->getParent().get() == parentNode ? 1 : 0;

    if (one_of(getAlgorithm(), Algorithm::EltwiseMultiply, Algorithm::EltwiseDivide, Algorithm::EltwisePrelu)) {
        fillValuesFrom(getParentEdgesAtPort(constPort)[0]->getParent(), scales);
    } else if (one_of(getAlgorithm(), Algorithm::EltwiseAdd, Algorithm::EltwiseSubtract)) {
        fillValuesFrom(getParentEdgesAtPort(constPort)[0]->getParent(), shifts);
    } else if (one_of(getAlgorithm(), Algorithm::EltwiseMulAdd)) {
        fillValuesFrom(getParentEdgesAtPort(1)[0]->getParent(), scales);
        fillValuesFrom(getParentEdgesAtPort(2)[0]->getParent(), shifts);
    } else if (one_of(getAlgorithm(), Algorithm::EltwisePowerStatic)) {
        const auto power = dynamic_cast<const Eltwise *>(this);
        if (!power) {
            IE_THROW() << "Cannot cast " << getName() << " to Eltwise";
        }
        scales.push_back(power->getBeta());
        shifts.push_back(power->getGamma());
    } else {
        IE_THROW() << "Can't fill scale and shifts for node: " << getName() << " with type: " << NameFromType(getType());
    }

    switch (getAlgorithm()) {
        case Algorithm::EltwiseAdd: {
            scales.resize(shifts.size(), 1.0f);
            break;
        }
        case Algorithm::EltwiseSubtract: {
            scales.resize(shifts.size(), 1.0f);
            std::transform(shifts.begin(), shifts.end(), shifts.begin(), [](float shift){ return -1.0f * shift; });
            break;
        }
        case Algorithm::EltwiseMultiply: {
            shifts.resize(scales.size(), 0.0f);
            break;
        }
        case Algorithm::EltwiseDivide: {
            shifts.resize(scales.size(), 0.0f);
            std::transform(scales.begin(), scales.end(), scales.begin(), [](float scale){ return 1.0f / scale; });
            break;
        }
        default: break;
    }

    return {scales, shifts};
}

bool Node::isInputTensorAtPortEmpty(size_t port) const {
    if (inputShapes.size() <= port) {
        IE_THROW() << "Incorrect input port number for node " << getName();
    }
    return getParentEdgesAtPort(port)[0]->getMemory().GetShape().hasZeroDims();
}

bool Node::isOutputTensorAtPortEmpty(size_t port) const {
    if (outputShapes.size() <= port) {
        IE_THROW() << "Incorrect output port number for node " << getName();
    }
    return getChildEdgesAtPort(port)[0]->getMemory().GetShape().hasZeroDims();
}

bool Node::hasEmptyInputTensors() const {
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        if (isInputTensorAtPortEmpty(i))
            return true;
    }
    return false;
}

bool Node::hasEmptyOutputTensors() const {
    for (size_t i = 0; i < outputShapes.size(); i++) {
        if (isOutputTensorAtPortEmpty(i))
            return true;
    }
    return false;
}

bool Node::inputShapesDefined() const {
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        if (!getParentEdgesAtPort(i)[0]->getMemory().getDesc().isDefined()) {
            return false;
        }
    }
    return true;
}

bool Node::outputShapesDefined() const {
    for (size_t i = 0; i < outputShapes.size(); i++) {
        if (!getChildEdgesAtPort(i)[0]->getMemory().getDesc().isDefined()) {
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
        if (lastInputDims.empty())
            return true;
        IE_THROW() << "Input dims and parent edges number mismatch!";
    }

    for (size_t i = 0; i < lastInputDims.size(); i++) {
        if (lastInputDims[i] != getParentEdgesAtPort(i)[0]->getMemory().getStaticDims())
            return true;
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
        for (size_t i = 0; i < shapes.size(); i++)
            input_shapes.emplace_back(std::ref(shapes[i].getStaticDims()));

        std::unordered_map<size_t, MemoryPtr> input_values;
        if (input_value_port_mask) {
            for (size_t port = 0; port < inputShapes.size(); ++port) {
                if (input_value_port_mask & (1 << port)) {
                    input_values[port] = getParentEdgesAtPort(port)[0]->getMemoryPtr();
                }
            }
        }

        auto result = shapeInference->infer(input_shapes, input_values);
        if (ShapeInferStatus::success != result.status) {
            IE_THROW(Unexpected) << "Shape inference unexpectedly skipped";
        }

        return std::move(result.dims);
    }
    catch (const std::runtime_error& exp) {
        IE_THROW() << "Shape inference of " << getTypeStr()  << " node with name " << getName() << " failed: " << exp.what();
    }
}

IShapeInfer::Result Node::shapeInfer() const {
    try {
        std::vector<std::reference_wrapper<const VectorDims>> input_shapes;
        auto input_value_port_mask = shapeInference->get_port_mask();

        input_shapes.reserve(inputShapes.size());
        for (size_t port = 0; port < inputShapes.size(); ++port)
            input_shapes.emplace_back(std::ref(getParentEdgesAtPort(port)[0]->getMemory().getStaticDims()));

        std::unordered_map<size_t, MemoryPtr> input_values;
        if (input_value_port_mask) {
            for (size_t port = 0; port < inputShapes.size(); ++port) {
                if (input_value_port_mask & (1 << port)) {
                    input_values[port] = getParentEdgesAtPort(port)[0]->getMemoryPtr();
                }
            }
        }

        return shapeInference->infer(input_shapes, input_values);
    }
    catch (const std::runtime_error& exp) {
        IE_THROW() << "Shape inference of " << getTypeStr()  << " node with name " << getName() << " failed: " << exp.what();
    }
}

void Node::updateLastInputDims() {
    if (lastInputDims.size() != getParentEdges().size()) {
        if (!lastInputDims.empty())
            IE_THROW() << "Input dims and parent edges number mismatch!";
        lastInputDims.resize(getParentEdges().size());
    }

    for (size_t i = 0; i < lastInputDims.size(); i++)
        lastInputDims[i] = getParentEdgesAtPort(i)[0]->getMemory().getStaticDims();
}

bool Node::canFuseSimpleOperation(const NodePtr& node) const {
    if (node->getType() == Type::FakeQuantize) {
        bool ret = node->getAlgorithm() != Algorithm::FQBinarization;
        for (size_t i = 1; i < node->getParentEdges().size(); i++) {
            ret &= node->getParentEdgesAtPort(i)[0]->getParent()->getChildEdges().size() == 1;
        }
        return ret;
    } else if (node->getType() == Type::Eltwise) {
        return one_of(node->getAlgorithm(),
                      Algorithm::EltwiseRelu,
                      Algorithm::EltwiseGelu,
                      Algorithm::EltwiseElu,
                      Algorithm::EltwiseSigmoid,
                      Algorithm::EltwiseClamp,
                      Algorithm::EltwiseTanh,
                      Algorithm::EltwiseSwish,
                      Algorithm::EltwiseHswish,
                      Algorithm::EltwiseMish,
                      Algorithm::EltwiseHsigmoid,
                      Algorithm::EltwiseRoundHalfToEven,
                      Algorithm::EltwiseRoundHalfAwayFromZero,
                      Algorithm::EltwiseAbs,
                      Algorithm::EltwiseSqrt,
                      Algorithm::EltwiseSoftRelu) ||
            node->canBePerformedAsScaleShift(this);
    }
    return false;
}

void Node::addFusedNode(const NodePtr &fusingNode) {
    fusedWith.push_back(fusingNode);
}

void Node::addSupportedPrimDesc(const std::vector<PortConfigurator>& inPortConfigs,
                                const std::vector<PortConfigurator>& outPortConfigs,
                                impl_desc_type implType,
                                bool dynBatchSupport) {
    auto fill_port = [] (const PortConfigurator& portConfigurator, const Shape& shape,
                         InferenceEngine::Precision prc, std::vector<PortConfig>& port) -> bool {
        // In order to simplify particular node initialization logic we just don't add config in case target shape is not supported by blockedDescCreator.
        // This should be suitable for major of scenarios since almost all nodes add `ncsp` blockedDescCreator which supports any shape rank.
        if (shape.getRank() < portConfigurator.blockedDescCreator->getMinimalRank())
            return false;

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
        auto prc = inPortConfigs[i].prc == InferenceEngine::Precision::UNSPECIFIED ? getOriginalInputPrecisionAtPort(i) : inPortConfigs[i].prc;
        if (!fill_port(inPortConfigs[i], shape, prc, config.inConfs))
            return;
    }

    for (size_t i = 0; i < outPortConfigs.size(); i++) {
        auto dims = outPortConfigs[i].shape.getRank() == 0 ? getOutputShapeAtPort(i) : outPortConfigs[i].shape;
        auto prc = outPortConfigs[i].prc == InferenceEngine::Precision::UNSPECIFIED ? getOriginalOutputPrecisionAtPort(i) : outPortConfigs[i].prc;
        if (!fill_port(outPortConfigs[i], dims, prc, config.outConfs))
            return;
    }

    config.dynBatchSupport = dynBatchSupport;
    supportedPrimitiveDescriptors.push_back({config, implType});
}

}   // namespace intel_cpu
}   // namespace ov

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_edge.h"
#include "mkldnn_node.h"
#include "mkldnn_extension_utils.h"
#include <blob_factory.hpp>
#include "utils/cpu_utils.hpp"
#include "cpu_memory_desc_utils.h"

using namespace mkldnn;
namespace MKLDNNPlugin {

MKLDNNEdge::MKLDNNEdge(const MKLDNNNodePtr &parent, const MKLDNNNodePtr &child, int pr_port, int ch_port) :
        parent(parent), child(child), parent_port(pr_port), child_port(ch_port) {}

const MKLDNNNodePtr MKLDNNEdge::getParent() const {
    auto parentPtr = parent.lock();
    if (!parentPtr)
        IE_THROW() << "Edge contains empty parent node";
    return parentPtr;
}

const MKLDNNNodePtr MKLDNNEdge::getChild() const {
    auto childPtr = child.lock();
    if (!childPtr)
        IE_THROW() << "Edge contains empty child node";
    return childPtr;
}

bool MKLDNNEdge::isUseExternalMemory() const {
    return externalMemoryPtr;
}

bool MKLDNNEdge::isDropped() const {
    bool not_in_parent = true;
    bool not_in_child = true;

    auto parent_ptr = parent.lock();
    if (parent_ptr) {
        for (auto &edge : parent_ptr->childEdges)
            if (edge.lock().get() == this)
                not_in_parent = false;
    }

    auto child_ptr = child.lock();
    if (child_ptr) {
        for (auto &edge : child_ptr->parentEdges)
            if (edge.lock().get() == this)
                not_in_child = false;
    }
    return not_in_parent && not_in_child;
}

void MKLDNNEdge::drop() {
    auto _drop_from = [&] (std::vector<MKLDNNEdgeWeakPtr> &list) {
        auto myself = std::find_if(list.begin(), list.end(),
                [&] (MKLDNNEdgeWeakPtr edge) { return edge.lock().get() == this; });

        if (myself != list.end())
            list.erase(myself);
    };

    _drop_from(getParent()->childEdges);
    _drop_from(getChild()->parentEdges);
}


bool MKLDNNEdge::needReorder() {
    bool canBeInPlaceConflicts = false;
    auto parentSPD = getParent()->getSelectedPrimitiveDescriptor();
    auto childSPD = getChild()->getSelectedPrimitiveDescriptor();
    if (!parentSPD || !childSPD)
        IE_THROW() << "Cannot make a decision about reorder. Primitive descriptors weren't selected.";

    int outNumber = getOutputNum();
    int inNumber = getInputNum();
    bool in_place = inPlace();
    bool childCanChangeMem = childSPD->getConfig().outConfs.empty();
    for (const auto& conf : childSPD->getConfig().outConfs) {
        if (conf.inPlace == outNumber && outNumber >= 0)
            childCanChangeMem = true;
    }

    const auto& detectInPlaceChildrenNum = [](const std::vector<MKLDNNEdgePtr>& edges) -> size_t {
        size_t count = 0;
        for (const auto& edge : edges) {
            auto childSPD = edge->getChild()->getSelectedPrimitiveDescriptor();
            int outNumber = edge->getOutputNum();
            if (childSPD->getConfig().outConfs.empty())
                count++;
            for (const auto& conf : childSPD->getConfig().outConfs) {
                if (conf.inPlace == outNumber)
                    count++;
            }
        }
        return count;
    };

    const auto portChildEdges = getParent()->getChildEdgesAtPort(inNumber);
    if (in_place && childCanChangeMem && portChildEdges.size() > 1 && detectInPlaceChildrenNum(portChildEdges) > 1)
        canBeInPlaceConflicts = true;
    if (!canBeInPlaceConflicts && in_place && !getParent()->getChildEdges().empty()) {
        for (auto &p_edge_peer : portChildEdges) {
            if (p_edge_peer.get() == this)
                continue;
            if (p_edge_peer->getChild()->getType() != Reorder && p_edge_peer->inPlace(LOOK_DOWN))
                canBeInPlaceConflicts = true;
        }
    }

    if (in_place) {
        if (inNumber >= 0 && inNumber < parentSPD->getConfig().outConfs.size() && parentSPD->getConfig().outConfs[inNumber].inPlace >= 0 &&
            outNumber >= 0 && outNumber < childSPD->getConfig().inConfs.size() && childSPD->getConfig().inConfs[outNumber].inPlace >= 0)
            canBeInPlaceConflicts = true;
    }
    return canBeInPlaceConflicts || !getInputDesc().isCompatible(getOutputDesc());
}

void MKLDNNEdge::reuse(MKLDNNMemoryPtr ptr) {
    if (status != Status::NeedAllocation)
        return;
    memoryPtr = ptr;
    status = Status::Allocated;
}

const MemoryDesc& MKLDNNEdge::getInputDescRO() const {
    return *inputDesc;
}

const MemoryDesc&  MKLDNNEdge::getInputDesc() {
//    if (inputDesc.getLayout() == InferenceEngine::Layout::ANY) {
//        inputDesc = getSpecifiedInputDesc({});
//    }
    if (!inputDesc) {
        inputDesc = getSpecifiedInputDesc({});
    }

    return *inputDesc;
}

const MemoryDesc& MKLDNNEdge::getOutputDescRO() const {
    return *outputDesc;
}

const MemoryDesc& MKLDNNEdge::getOutputDesc() {
//    if (outputDesc.getLayout() == InferenceEngine::Layout::ANY) {
//        outputDesc = getSpecifiedOutputDesc({});
//    }
    if (!outputDesc) {
        outputDesc = getSpecifiedOutputDesc({});
    }

    return *outputDesc;
}

InferenceEngine::TensorDesc MKLDNNEdge::getTensorDesc() {
    if (!getInputDesc().isCompatible(getOutputDesc()))
        IE_THROW() << "Cannot get descriptor for edge: " << getParent()->getName() << "->"
                           << getChild()->getName();

    return MemoryDescUtils::convertToTensorDesc(getInputDesc());
}

int MKLDNNEdge::getInputNum() const {
    return parent_port;
}

int MKLDNNEdge::getOutputNum() const {
    return child_port;
}

void MKLDNNEdge::allocate(const void* mem_ptr) {
    if (status != Status::NeedAllocation)
        return;

    if (memoryPtr)
        IE_THROW() << "Unexpected behaviour: status == NeedAllocation but memory is already allocated.";

    auto& inputDesc = getInputDesc();
    auto& outputDesc = getOutputDesc();
    if (!inputDesc.isDefined() || !outputDesc.isDefined() || !inputDesc.isCompatible(outputDesc))
        IE_THROW() << "Cannot allocate memory. Nodes have primitive descriptors with different formats.";

//    if (!MKLDNNExtensionUtils::initTensorsAreEqual(outputDesc, inputDesc) ||
//            (inputDesc.getDims().size() > 0 && inputDesc.getDims()[0] != 1 &&
//            (inputDesc.getPrecision() != outputDesc.getPrecision() ||
//             inputDesc.getBlockingDesc() != outputDesc.getBlockingDesc())))
//        IE_THROW() << "Cannot allocate memory. Nodes have primitive descriptors with different formats.";
//    if (inputDesc.getLayout() == InferenceEngine::Layout::ANY)
//        IE_THROW() << "Cannot get input descriptor!";

    auto parentPtr = getParent();
    memoryPtr.reset(new MKLDNNMemory(parentPtr->getEngine()));

    memoryPtr->Create(MemoryDescUtils::convertToMKLDNNMemoryDesc(inputDesc), mem_ptr, false);  // no pads zeroing
    status = Status::Allocated;
}

std::string MKLDNNEdge::name() {
//    auto tensorDescToStr = [](InferenceEngine::TensorDesc const & desc) {
//        std::string name = desc.getPrecision().name();
//
//        auto blockingDesc = desc.getBlockingDesc();
//        auto dims = blockingDesc.getBlockDims();
//
//        if (!dims.empty()) {
//            name += "[";
//            for (size_t i = 1; i < dims.size(); ++i) {
//                name += std::to_string(dims[i - 1]) + ",";
//            }
//            name += std::to_string(dims.back()) + "]";
//        }
//
//        return name;
//    };

    auto parentPtr = getParent();
    auto childPtr = getChild();

    // TODO [DS]: why do we need blockedDims to identify the edge?
//    return parentPtr->getName() + std::to_string(parent_port) + tensorDescToStr(getInputDesc())
//            + "<->" + childPtr->getName() + std::to_string(child_port);
    return parentPtr->getName() + std::to_string(parent_port)
           + "<->" + childPtr->getName() + std::to_string(child_port);
}

void MKLDNNEdge::externalAllocate(MKLDNNWeightsSharing::Ptr weightsCache) {
    if (status != Status::NeedAllocation)
        return;

    if (weightsCache) {
        auto alloc = [this] () {
            allocate();
            return memoryPtr;
        };

        auto ptr = weightsCache->findOrCreate(name(), alloc, false);
        memoryPtr = *ptr;
        externalMemoryPtr = true;
        status = Status::Allocated;
    } else {
        allocate();
    }
}

void MKLDNNEdge::changeStatus(MKLDNNEdge::Status state) {
    if (state == Status::NotAllocated) {
        IE_THROW() << "Incorrect behaviour! Use method sharedMemFrom()";
    }
    if (state == Status::Validated) {
        IE_THROW() << "Incorrect behaviour! Use method validate()";
    }
    if (status != Status::Uninitialized && state == Status::NeedAllocation)
        return;
    if (status == Status::NotAllocated)
        memoryFromEdge.reset();
    status = state;
}

const Shape& MKLDNNEdge::getShape() {
//    if (!shape.getRank()) {
//        Shape inShape;
//        Shape outShape;
//        auto childPtr = getChild();
//        auto parentPtr = getParent();
//
//        int inNum = getOutputNum();
//        if (inNum < 0) {
//            IE_THROW() << "Error cannot find input data for " << child.lock()->getName()
//                               << " from " << parent.lock()->getName();
//        }
//        if (inNum < childPtr->inputShapes.size()) {
//            outShape = childPtr->inputShapes[inNum];
//        }
//
//        int outNum = getInputNum();
//        if (outNum < 0) {
//            IE_THROW() << "Error cannot find output data for " << parent.lock()->getName()
//                               << " to " << child.lock()->getName();
//        }
//        if (outNum >= parentPtr->outputShapes.size())
//            outNum = 0;
//        if (outNum < parentPtr->outputShapes.size()) {
//            inShape = parentPtr->outputShapes[outNum];
//        }
//
//        if (inShape.getRank() && outShape.getRank() && inShape.getRank() != outShape.getRank() && inShape.size() != outShape.size())
//            IE_THROW() << "Nodes " << getParent()->getName() << " and " << getChild()->getName()
//                               << " have incompatible dimensions!";
//
//        if (outShape.getRank() != 0) {
//            shape = outShape;
//        } else if (inShape.getRank() != 0) {
//            shape = inShape;
//        } else {
//            shape = Shape(InferenceEngine::SizeVector({1}));
//        }
//
//
//        if (!(outShape.getRank() == 0 && inShape.getRank() == 0) && !shape.getRank())
//            IE_THROW() << "Cannot detect right dims for nodes " << getParent()->getName()
//                               << " and " << getChild()->getName();
//    }

    // TODO [DS]: How should we validate shape compatibility?
    // TODO [DS]: Why do we allow uninitialized shape?
    if (!shape.getRank()) {
        auto childPtr = getChild();

        int inNum = getOutputNum();
        if (inNum < 0) {
            IE_THROW() << "Error cannot find input data for " << child.lock()->getName()
                       << " from " << parent.lock()->getName();
        }
        if (inNum < childPtr->inputShapes.size()) {
            shape = childPtr->inputShapes[inNum];
        }
    }

    return shape;
}

const MemoryDesc& MKLDNNEdge::getDesc() {
    if (!getInputDesc().isCompatible(getOutputDesc()))
        IE_THROW() << "Cannot get descriptor for edge: " << getParent()->getName() << "->"
                   << getChild()->getName();

    return getInputDesc();
}

bool MKLDNNEdge::nodeCanChangeDesc(const MKLDNNNodePtr &node) const {
//    NodeDesc * selectedPd = node->getSelectedPrimitiveDescriptor();
//    if (selectedPd == nullptr)
//        IE_THROW() << "Primitive descriptor for node " << node->getName() << " is not selected.";

    return true;
    // TODO [DS]: since we don't allow ANY layout on ports do we need this method at all?
//    for (auto &inputDesc : selectedPd->getConfig().inConfs) {
//        if (inputDesc.desc.getLayout() != InferenceEngine::Layout::ANY) {
//            return true;
//        }
//    }
//
//    for (auto &outDesc : selectedPd->getConfig().outConfs) {
//        if (outDesc.desc.getLayout() != InferenceEngine::Layout::ANY) {
//            return true;
//        }
//    }

    Shape inputShape;
    for (size_t i = 0; i < node->getParentEdges().size(); i++) {
        if (inputShape.getSize() == 1 && inputShape.getRank() == 0) {
            inputShape = node->getParentEdgeAt(i)->getShape();
            continue;
        }

        if (inputShape.getRank() != node->getParentEdgeAt(i)->getShape().getRank()) {
            return true;
        }
    }
    for (size_t i = 0; i < node->getChildEdges().size(); i++) {
        if (inputShape.getSize() == 1 && inputShape.getRank() == 0) {
            inputShape = node->getChildEdgeAt(i)->getShape();
            continue;
        }

        if (inputShape.getRank() != node->getChildEdgeAt(i)->getShape().getRank()) {
            return true;
        }
    }

    return false;
}

/// In we have {any, any, any} -> {any} or {any} -> {any, any, any} or {any} -> {any} it means that
/// layer doesn't change memory format
/// We don't support {any, any, nchw} -> {any}
std::unique_ptr<MemoryDesc> MKLDNNEdge::getSpecifiedInputDesc(std::map<memory::format_tag, size_t> formats, size_t enterCountUp, size_t enterCountDown) {
//    InferenceEngine::TensorDesc inDesc;
//
//    if (inputDesc.getLayout() != InferenceEngine::Layout::ANY) {
//        return inputDesc;
//    }

    auto parentPtr = getParent();
    if (parentPtr->getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Primitive descriptor for node " << parentPtr->getName() << " is not selected.";

    int inputIdx = getInputNum();
    if (inputIdx < 0)
        IE_THROW() << "Edge cannot be found for node" << parentPtr->getName() << ".";

    if (inputIdx >= parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size())
        inputIdx = 0;

    return parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc->clone();
//    inDesc = parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc;
//
//    if (inDesc.getLayout() != InferenceEngine::Layout::ANY) {
//        return inDesc;
//    }
//
//    bool isFormatChanging = nodeCanChangeDesc(parentPtr);
//
//    if (!isFormatChanging && inputIdx < parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size() &&
//            parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc.getLayout() != InferenceEngine::Layout::ANY) {
//        inDesc = parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc;
//        parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc = inDesc;
//        return inDesc;
//    }
//
//    for (size_t i = 0; i < parentPtr->getChildEdges().size(); i++) {
//        auto childEdge = parentPtr->getChildEdgeAt(i);
//        auto child = childEdge->getChild();
//        int childIdx = childEdge->getOutputNum();
//        if (!child->getSelectedPrimitiveDescriptor() || childIdx < 0 ||
//                childEdge->getShape().getRank() != getShape().getRank()) {
//            continue;
//        }
//        if (child->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size() <= childIdx)
//            childIdx = 0;
//        memory::format_tag childInDesc = MKLDNNMemoryDesc(child->getSelectedPrimitiveDescriptor()->getConfig().inConfs[childIdx].desc).getFormat();
//        if (childInDesc != memory::format_tag::any && childInDesc != memory::format_tag::undef) {
//            if (formats.find(childInDesc) == formats.end())
//                formats[childInDesc] = 1;
//            else
//                formats[childInDesc] += 1;
//            continue;
//        }
//        if (nodeCanChangeDesc(child))
//            continue;
//
//        if (enterCountUp < 2) {
//            childInDesc = MKLDNNMemoryDesc(childEdge->getSpecifiedOutputDesc(formats, enterCountUp, ++enterCountDown)).getFormat();
//            if (childInDesc != memory::format_tag::any && childInDesc != memory::format_tag::undef) {
//                if (formats.find(childInDesc) == formats.end())
//                    formats[childInDesc] = 1;
//                else
//                    formats[childInDesc] += 1;
//            }
//        }
//    }
//
//    if (!isFormatChanging) {
//        for (size_t i = 0; i < parentPtr->getParentEdges().size(); i++) {
//            auto parentEdge = parentPtr->getParentEdgeAt(i);
//            auto parent = parentEdge->getParent();
//            int parentIdx = parentEdge->getInputNum();
//            if (!parent->getSelectedPrimitiveDescriptor() || parentIdx < 0 ||
//                    parentEdge->getShape().getRank() != getShape().getRank()) {
//                continue;
//            }
//            if (parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() <= parentIdx) {
//                parentIdx = 0;
//            }
//            memory::format_tag parentOutDesc = MKLDNNMemoryDesc(parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs[parentIdx].desc).getFormat();
//            if (parentOutDesc != memory::format_tag::any && parentOutDesc != memory::format_tag::undef) {
//                if (formats.find(parentOutDesc) == formats.end())
//                    formats[parentOutDesc] = 1;
//                else
//                    formats[parentOutDesc] += 1;
//                continue;
//            }
//            if (nodeCanChangeDesc(parent))
//                continue;
//
//            if (enterCountUp < 2) {
//                parentOutDesc = MKLDNNMemoryDesc(parentEdge->getSpecifiedInputDesc(formats, ++enterCountUp, enterCountDown)).getFormat();
//                if (parentOutDesc != memory::format_tag::any && parentOutDesc != memory::format_tag::undef) {
//                    if (formats.find(parentOutDesc) == formats.end())
//                        formats[parentOutDesc] = 1;
//                    else
//                        formats[parentOutDesc] += 1;
//                }
//            }
//        }
//    }
//
//    size_t maxFormatCount = 0;
//    memory::format_tag desc =  MKLDNNMemory::GetPlainFormatByRank(getShape().getRank());
//    for (auto &it : formats) {
//        if (maxFormatCount < it.second && MKLDNNMemory::isConsistant(getShape(), it.first)) {
//            maxFormatCount = it.second;
//            desc = it.first;
//        }
//    }
//
//    auto inDataType = MKLDNNMemoryDesc(parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc).getDataType();
//    parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc = MKLDNNMemoryDesc(getShape(), inDataType, desc);
//    if (!isFormatChanging && inputIdx < parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size() &&
//            parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc.getLayout() == InferenceEngine::Layout::ANY) {
//        parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc =
//                MKLDNNExtensionUtils::getUninitTensorDesc(MKLDNNMemoryDesc(getShape(), inDataType, desc));
//    }
//
//    return MKLDNNMemoryDesc(getShape(), inDataType, desc);
}

std::unique_ptr<MemoryDesc> MKLDNNEdge::getSpecifiedOutputDesc(std::map<memory::format_tag, size_t> formats, size_t enterCountUp, size_t enterCountDown) {
//    if (outputDesc.getLayout() != InferenceEngine::Layout::ANY) {
//        return outputDesc;
//    }

    auto childPtr = getChild();
    auto parentPtr = getParent();

    if (childPtr->getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Primitive descriptor for node " << childPtr->getName() << " is not selected.";

    int outputIdx = getOutputNum();
//    int inputIdx = getInputNum();
    if (outputIdx < 0) {
        IE_THROW() << "Edge cannot be found for node" << childPtr->getName() << ".";
    }
    if (outputIdx >= childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size())
        outputIdx = 0;

    return childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc->clone();
//    outDesc = childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc;
//
//    if (outDesc.getLayout() != InferenceEngine::Layout::ANY) {
//        return outDesc;
//    }
//
//    if (inputIdx >= parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size())
//        inputIdx = 0;
//
//    bool isFormatChanging = nodeCanChangeDesc(childPtr);
//
//    if ((!isFormatChanging && outputIdx < childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() &&
//            childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc.getLayout() != InferenceEngine::Layout::ANY) ||
//            (isFormatChanging && inputIdx >= 0 &&
//                    parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc.getLayout() != InferenceEngine::Layout::ANY)) {
//        auto inputDataType = childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc.getPrecision();
//        if (!isFormatChanging)
//            outDesc = childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc;
//        else
//            outDesc = parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc;
//        childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc = InferenceEngine::TensorDesc(inputDataType,
//                                                                                                                      getShape().getDims(),
//                                                    {outDesc.getBlockingDesc().getBlockDims(),
//                                                     outDesc.getBlockingDesc().getOrder()});
//        return childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc;
//    }
//
//    for (size_t i = 0; i < childPtr->getParentEdges().size(); i++) {
//        auto parentEdge = childPtr->getParentEdgeAt(i);
//        auto parent = parentEdge->getParent();
//        int parentIdx = parentEdge->getInputNum();
//        if (!parent->getSelectedPrimitiveDescriptor() || parentIdx < 0 ||
//                parentEdge->getShape().getRank() != getShape().getRank()) {
//            continue;
//        }
//        if (parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() <= parentIdx) {
//            parentIdx = 0;
//        }
//        memory::format_tag parentOutDesc = MKLDNNMemoryDesc(parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs[parentIdx].desc).getFormat();
//        if (parentOutDesc != memory::format_tag::any && parentOutDesc != memory::format_tag::undef) {
//            if (formats.find(parentOutDesc) == formats.end())
//                formats[parentOutDesc] = 1;
//            else
//                formats[parentOutDesc] += 1;
//            continue;
//        }
//        if (nodeCanChangeDesc(parent))
//            continue;
//
//        if (enterCountDown < 2) {
//            parentOutDesc = MKLDNNMemoryDesc(parentEdge->getSpecifiedInputDesc(formats, ++enterCountUp, enterCountDown)).getFormat();
//            if (parentOutDesc != memory::format_tag::any && parentOutDesc != memory::format_tag::undef) {
//                if (formats.find(parentOutDesc) == formats.end())
//                    formats[parentOutDesc] = 1;
//                else
//                    formats[parentOutDesc] += 1;
//            }
//        }
//    }
//
//    if (!isFormatChanging) {
//        for (size_t i = 0; i < childPtr->getChildEdges().size(); i++) {
//            auto childEdge = childPtr->getChildEdgeAt(i);
//            auto child = childEdge->getChild();
//            int childIdx = childEdge->getOutputNum();
//            if (!child->getSelectedPrimitiveDescriptor() || childIdx < 0 ||
//                    childEdge->getShape().getRank() != getShape().getRank()) {
//                continue;
//            }
//            if (child->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size() <= childIdx) {
//                childIdx = 0;
//            }
//            memory::format_tag childInDesc = MKLDNNMemoryDesc(child->getSelectedPrimitiveDescriptor()->getConfig().inConfs[childIdx].desc).getFormat();
//            if (childInDesc != memory::format_tag::any && childInDesc != memory::format_tag::undef) {
//                if (formats.find(childInDesc) == formats.end())
//                    formats[childInDesc] = 1;
//                else
//                    formats[childInDesc] += 1;
//                continue;
//            }
//            if (nodeCanChangeDesc(child))
//                continue;
//
//            if (enterCountDown < 2) {
//                childInDesc = MKLDNNMemoryDesc(childEdge->getSpecifiedOutputDesc(formats, enterCountUp, ++enterCountDown)).getFormat();
//                if (childInDesc != memory::format_tag::any && childInDesc != memory::format_tag::undef) {
//                    if (formats.find(childInDesc) == formats.end())
//                        formats[childInDesc] = 1;
//                    else
//                        formats[childInDesc] += 1;
//                }
//            }
//        }
//    }
//
//    size_t maxFormatCount = 0;
//    memory::format_tag format =  MKLDNNMemory::GetPlainFormatByRank(getShape().getRank());
//    for (auto &it : formats) {
//        if (maxFormatCount < it.second && MKLDNNMemory::isConsistant(getShape(), it.first)) {
//            maxFormatCount = it.second;
//            format = it.first;
//        }
//    }
//
//    auto inDataType = MKLDNNMemoryDesc(childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[getOutputNum()].desc).getDataType();
//    childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc = MKLDNNMemoryDesc(getShape(), inDataType, format);
//    if (!isFormatChanging && outputIdx < childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() &&
//            childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc.getLayout() == InferenceEngine::Layout::ANY) {
//        childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc =
//                MKLDNNExtensionUtils::getUninitTensorDesc(MKLDNNMemoryDesc(getShape(), inDataType, format));
//    }
//
//    return childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc;
}

const MKLDNNMemory &MKLDNNEdge::getMemory() {
    if (status == Status::NotAllocated) {
        memoryPtr.reset(new MKLDNNMemory(getParent()->getEngine()));
        memoryPtr->Create(MemoryDescUtils::convertToMKLDNNMemoryDesc(getDesc()), getSharedEdge()->getMemoryPtr()->GetData());
        memoryFromEdge.reset();
        changeStatus(Status::Allocated);
    }

    return *memoryPtr;
}

MKLDNNMemoryPtr &MKLDNNEdge::getMemoryPtr() {
    if (status == Status::NotAllocated) {
        memoryPtr.reset(new MKLDNNMemory(getParent()->getEngine()));
        memoryPtr->Create(MemoryDescUtils::convertToMKLDNNMemoryDesc(getDesc()), getSharedEdge()->getMemoryPtr()->GetData());
        memoryFromEdge.reset();
        changeStatus(Status::Allocated);
    }

    return memoryPtr;
}

InferenceEngine::Blob::Ptr MKLDNNEdge::getBlob() {
    if (!memoryPtr)
        IE_THROW() << "Cannot get blob! Edge isn't initialized.";
    InferenceEngine::TensorDesc desc = getTensorDesc();

    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        desc = InferenceEngine::TensorDesc(desc.getPrecision(), getShape().getStaticDims(), desc.getLayout());
    else
        desc = InferenceEngine::TensorDesc(desc.getPrecision(), getShape().getStaticDims(), desc.getBlockingDesc());

    return isEmptyTensorDesc(desc) ? make_blob_with_precision(desc) : make_blob_with_precision(desc, memoryPtr->GetData());
}

void MKLDNNEdge::sharedMemFrom(const MKLDNNEdgePtr &edge) {
    memoryFromEdge = edge;
    status = Status::NotAllocated;
}

void MKLDNNEdge::validate() {
    if (status == Status::Validated)
        return;
    getMemory();
    getParent();
    getChild();
    getShape();

    if (status != Status::Allocated) {
        IE_THROW() << "Error memory is not allocated!";
    }
    status = Status::Validated;
}

MKLDNNEdgePtr MKLDNNEdge::getSharedEdge() const {
    auto memoryFromEdgePtr = memoryFromEdge.lock();
    if (!memoryFromEdgePtr) {
        IE_THROW() << "Cannot get memory ptr for edge(" << getParent()->getName() << "->"
                           << getChild()->getName() << "). The pointer on the edge with memory is empty!";
    }
    return memoryFromEdgePtr;
}

MKLDNNEdgePtr MKLDNNEdge::getSharedEdge(std::nothrow_t) const {
    return memoryFromEdge.lock();
}

void MKLDNNEdge::init() {
    if (status != Status::NeedAllocation && status != Status::Uninitialized)
        return;
    MKLDNNEdgePtr edgePtr = getBaseEdge();
    if (edgePtr.get() == this) {
        changeStatus(Status::NeedAllocation);
    } else {
        if (edgePtr->getParent()->isConstant() && !edgePtr->getChild()->isConstant()) {
            changeStatus(Status::NeedAllocation);
            return;
        }
        sharedMemFrom(edgePtr);
    }

    auto port = getInputNum();
    if (port < 0)
        return;
    auto edges_at_same_port = getParent()->getChildEdgesAtPort(static_cast<size_t>(port));
    for (auto edge : edges_at_same_port) {
        if (edge->getStatus() != Status::NeedAllocation && edge->getStatus() != Status::Uninitialized) {
            if (edge->getSharedEdge() != edgePtr)
                IE_THROW() << "Unsupported behavior. Cannot mark edge "
                                   << getParent()->getChildEdgeAt(0)->getParent()->getName() << "->"
                                   << getParent()->getChildEdgeAt(0)->getChild()->getName() << " as not allocated!";
        } else {
            if (edge != edgePtr)
                edge->sharedMemFrom(edgePtr);
        }
    }
}

/**
 * Should analyze graph node dependencies, inplace node information and return root memory(edge) it view on
 *
 * @param type some magic enum values... description needed
 * @return root of view-on-memory subgraph
 */
MKLDNNEdgePtr MKLDNNEdge::getBaseEdge(int look) {
    auto parentConfig = getParent()->getSelectedPrimitiveDescriptor()->getConfig();
    auto childConfig = getChild()->getSelectedPrimitiveDescriptor()->getConfig();
    int inputNum = getInputNum();
    int outputNum = getOutputNum();

    if (childConfig.inConfs[outputNum].inPlace >= 0 && parentConfig.outConfs[inputNum].inPlace >= 0) {
        inputNum = getInputNum();
        return getParent()->getChildEdgeAt(inputNum);
    }

    if (childConfig.inConfs[outputNum].inPlace >= 0 && (look & LOOK_DOWN)) {
        int next_port_idx = childConfig.inConfs[outputNum].inPlace;
        if (childConfig.outConfs[next_port_idx].inPlace >= 0) {
            childConfig.outConfs[next_port_idx].inPlace = -1;
            getChild()->initDescriptor(childConfig);
        }

        auto ch_edges = getChild()->getChildEdgesAtPort(next_port_idx);
        auto &next_ch_edge = ch_edges[0];

        // Multiple connection to some out port
        // Will try to find inplace consumer
        for (auto &ch_edge : ch_edges) {
            auto &chch_conf = ch_edge->getChild()->getSelectedPrimitiveDescriptor()->getConfig();

            if (chch_conf.inConfs[ch_edge->getOutputNum()].inPlace >= 0)
                next_ch_edge = ch_edge;
        }
        return next_ch_edge->getBaseEdge(LOOK_DOWN);
    } else if (parentConfig.outConfs[inputNum].inPlace >= 0 && (look & LOOK_UP)) {
        int next_port_idx = parentConfig.outConfs[inputNum].inPlace;
        if (parentConfig.inConfs[next_port_idx].inPlace >= 0) {
            parentConfig.inConfs[next_port_idx].inPlace = -1;
            getParent()->initDescriptor(parentConfig);
        }
        return getParent()->getParentEdgesAtPort(next_port_idx)[0]->getBaseEdge(LOOK_UP);
    }

    auto edges_for_same_port = getParent()->getChildEdgesAtPort(inputNum);
    if (!(look & LOOK_NO_RECURRENT)) {
        for (auto edge : edges_for_same_port) {
            if (edge.get() != this) {
                auto base = edge->getBaseEdge(LOOK_BOTH | LOOK_NO_RECURRENT);
                if (base != edge && base != edges_for_same_port[0]) return base;
            }
        }
    }
    return edges_for_same_port[0];
}

bool MKLDNNEdge::inPlace(LOOK look) {
    auto parentSPD = getParent()->getSelectedPrimitiveDescriptor();
    auto childSPD = getChild()->getSelectedPrimitiveDescriptor();
    if (!parentSPD || !childSPD)
        IE_THROW() << "Cannot make a decision about reorder. Primitive descriptors weren't selected.";
    int inputNum = getInputNum();
    int outputNum = getOutputNum();
    if (inputNum >= parentSPD->getConfig().outConfs.size())
        inputNum = 0;
    if (outputNum >= childSPD->getConfig().inConfs.size())
        outputNum = 0;

    if (look & LOOK_UP) {
        if (parentSPD->getConfig().outConfs[inputNum].inPlace >= 0)
            return true;
    }
    if (look & LOOK_DOWN) {
        if (childSPD->getConfig().inConfs[outputNum].inPlace >= 0)
            return true;
    }
    return false;
}

}  // namespace MKLDNNPlugin

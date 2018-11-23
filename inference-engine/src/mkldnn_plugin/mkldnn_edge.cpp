// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_edge.h"
#include "mkldnn_node.h"
#include "mkldnn_extension_utils.h"
#include <blob_factory.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;

MKLDNNPlugin::MKLDNNEdge::MKLDNNEdge(const std::shared_ptr<MKLDNNPlugin::MKLDNNNode> &parent,
                                     const std::shared_ptr<MKLDNNPlugin::MKLDNNNode> &child) {
    this->parent = parent;
    this->child = child;
}

const std::shared_ptr<MKLDNNPlugin::MKLDNNNode> MKLDNNPlugin::MKLDNNEdge::getParent() const {
    auto parentPtr = parent.lock();
    if (!parentPtr)
        THROW_IE_EXCEPTION << "Edge contains empty parent node";
    return parentPtr;
}

const std::shared_ptr<MKLDNNPlugin::MKLDNNNode> MKLDNNPlugin::MKLDNNEdge::getChild() const {
    auto childPtr = child.lock();
    if (!childPtr)
        THROW_IE_EXCEPTION << "Edge contains empty child node";
    return childPtr;
}

bool MKLDNNPlugin::MKLDNNEdge::isDropped() {
    return getInputNum() == -1 && getOutputNum() == -1;
}

bool MKLDNNPlugin::MKLDNNEdge::needReorder() {
    bool canBeInPlaceConflicts = false;
    auto parentSPD = getParent()->getSelectedPrimitiveDescriptor();
    auto childSPD = getChild()->getSelectedPrimitiveDescriptor();
    if (!parentSPD || !childSPD)
        THROW_IE_EXCEPTION << "Cannot make a decision about reorder. Primitive descriptors weren't selected.";

    int inputNum = getInputNum();
    bool in_place = inPlace();
    if (in_place && !getParent()->getChildEdges().empty()) {
        for (size_t i = 0; i < getParent()->getChildEdges().size(); i++) {
            if (i == inputNum)
                continue;
            if (getParent()->getChildEdgeAt(i)->getChild()->getType() != Reorder && getParent()->getChildEdgeAt(i)->inPlace(LOOK_DOWN))
                canBeInPlaceConflicts = true;
        }
    }

    if (in_place) {
        int outNumber = getOutputNum();
        int inNumber = getInputNum();
        if (inNumber >= 0 && inNumber < parentSPD->getConfig().outConfs.size() && parentSPD->getConfig().outConfs[inNumber].inPlace >= 0 &&
            outNumber >= 0 && outNumber < childSPD->getConfig().inConfs.size() && childSPD->getConfig().inConfs[outNumber].inPlace >= 0)
            canBeInPlaceConflicts = true;
    }
    return !MKLDNNExtensionUtils::initTensorsAreEqual(getInputDesc(), getOutputDesc()) || canBeInPlaceConflicts;
}

InferenceEngine::TensorDesc MKLDNNPlugin::MKLDNNEdge::getInputDesc() {
    if (inputDesc.getLayout() == InferenceEngine::Layout::ANY) {
        inputDesc = getSpecifiedInputDesc({});
    }
    return inputDesc;
}

InferenceEngine::TensorDesc MKLDNNPlugin::MKLDNNEdge::getOutputDesc() {
    if (outputDesc.getLayout() == InferenceEngine::Layout::ANY) {
        outputDesc = getSpecifiedOutputDesc({});
    }
    return outputDesc;
}

InferenceEngine::TensorDesc MKLDNNPlugin::MKLDNNEdge::getDesc() {
    if (!MKLDNNExtensionUtils::initTensorsAreEqual(getInputDesc(), getOutputDesc()))
        THROW_IE_EXCEPTION << "Cannot get descriptor for edge: " << getParent()->getName() << "->"
                           << getChild()->getName();
    return getInputDesc();
}

int MKLDNNPlugin::MKLDNNEdge::getInputNum() {
    return getAllInputNums()[0];
}

std::vector<int> MKLDNNPlugin::MKLDNNEdge::getAllInputNums() {
    auto parentPtr = parent.lock();
    if (!parentPtr)
        return {-1};

    std::vector<int> res;
    for (size_t i = 0; i < parentPtr->getChildEdges().size(); i++) {
        auto childEdge = parentPtr->getChildEdges()[i].lock();
        if (childEdge && childEdge.get() == this) {
            res.push_back(static_cast<int>(i));
        }
    }
    return res.empty() ? std::vector<int>{-1} : res;
}

int MKLDNNPlugin::MKLDNNEdge::getOutputNum() {
    return getAllOutputNums()[0];
}

std::vector<int> MKLDNNPlugin::MKLDNNEdge::getAllOutputNums() {
    auto childPtr = child.lock();
    if (!childPtr)
        return {-1};

    std::vector<int> res;
    for (size_t i = 0; i < childPtr->getParentEdges().size(); i++) {
        auto parentEdge = childPtr->getParentEdges()[i].lock();
        if (parentEdge && parentEdge.get() == this) {
            res.push_back(static_cast<int>(i));
        }
    }
    return res.empty() ? std::vector<int>{-1} : res;
}

void MKLDNNPlugin::MKLDNNEdge::allocate(const void* mem_ptr) {
    if (status != Status::NeedAllocation)
        return;

    if (memoryPtr)
        THROW_IE_EXCEPTION << "Unexpected behaviour: status == NeedAllocation but memory is already allocated.";

    auto inputDesc = getInputDesc();
    auto outputDesc = getOutputDesc();
    if (!MKLDNNExtensionUtils::initTensorsAreEqual(outputDesc, inputDesc) ||
            (inputDesc.getDims()[0] != 1 && inputDesc != outputDesc))
        THROW_IE_EXCEPTION << "Cannot allocate memory. Nodes have primitive descriptors with different formats.";
    if (inputDesc.getLayout() == InferenceEngine::Layout::ANY)
        THROW_IE_EXCEPTION << "Cannot get input descriptor!";

    auto parentPtr = getParent();
    memoryPtr.reset(new MKLDNNMemory(parentPtr->getEngine()));
    memoryPtr->Create(MKLDNNMemoryDesc(inputDesc), mem_ptr);
    status = Status::Allocated;
}

void MKLDNNPlugin::MKLDNNEdge::changeStatus(MKLDNNPlugin::MKLDNNEdge::Status state) {
    if (state == Status::NotAllocated) {
        THROW_IE_EXCEPTION << "Incorrect behaviour! Use method sharedMemFrom()";
    }
    if (state == Status::Validated) {
        THROW_IE_EXCEPTION << "Incorrect behaviour! Use method validate()";
    }
    if (status != Status::Uninitialized && state == Status::NeedAllocation)
        return;
    if (status == Status::NotAllocated)
        memoryFromEdge.reset();
    status = state;
}

MKLDNNPlugin::MKLDNNDims &MKLDNNPlugin::MKLDNNEdge::getDims() {
    if (!dims.ndims()) {
        MKLDNNDims outDims;
        MKLDNNDims inDims;
        auto childPtr = getChild();
        auto parentPtr = getParent();

        int inNum = getOutputNum();
        if (inNum < 0) {
            THROW_IE_EXCEPTION << "Error cannot find input data for " << child.lock()->getName()
                               << " from " << parent.lock()->getName();
        }
        if (inNum < childPtr->inDims.size()) {
            outDims = childPtr->inDims[inNum];
        }

        int outNum = getInputNum();
        if (outNum < 0) {
            THROW_IE_EXCEPTION << "Error cannot find output data for " << parent.lock()->getName()
                               << " to " << child.lock()->getName();
        }
        if (outNum >= parentPtr->outDims.size())
            outNum = 0;
        if (outNum < parentPtr->outDims.size()) {
            inDims = parentPtr->outDims[outNum];
        }

        if (inDims.ndims() && outDims.ndims() && inDims.ndims() != outDims.ndims() && inDims.size() != outDims.size())
            THROW_IE_EXCEPTION << "Nodes " << getParent()->getName() << " and " << getChild()->getName()
                               << " have incompatible dimensions!";

        dims = outDims.ndims() ? outDims : inDims;

        if (!dims.ndims())
            THROW_IE_EXCEPTION << "Cannot detect right dims for nodes " << getParent()->getName()
                               << " and " << getChild()->getName();
    }
    return dims;
}

void MKLDNNPlugin::MKLDNNEdge::setDims(MKLDNNPlugin::MKLDNNDims &dims) {
    this->dims = dims;
}

bool MKLDNNPlugin::MKLDNNEdge::nodeCanChangeDesc(const std::shared_ptr<MKLDNNPlugin::MKLDNNNode> &node) const {
    PrimitiveDescInfo * selectedPd = node->getSelectedPrimitiveDescriptor();
    if (selectedPd == nullptr)
        THROW_IE_EXCEPTION << "Primitive descriptor for node " << node->getName() << " is not selected.";

    for (auto &inputDesc : selectedPd->getConfig().inConfs) {
        if (inputDesc.desc.getLayout() != InferenceEngine::Layout::ANY) {
            return true;
        }
    }

    for (auto &outDesc : selectedPd->getConfig().outConfs) {
        if (outDesc.desc.getLayout() != InferenceEngine::Layout::ANY) {
            return true;
        }
    }

    MKLDNNDims inputDims;
    for (size_t i = 0; i < node->getParentEdges().size(); i++) {
        if (inputDims.size() == 1 && inputDims.ndims() == 0) {
            inputDims = node->getParentEdgeAt(i)->getDims();
            continue;
        }

        if (inputDims.ndims() != node->getParentEdgeAt(i)->getDims().ndims()) {
            return true;
        }
    }
    for (size_t i = 0; i < node->getChildEdges().size(); i++) {
        if (inputDims.size() == 1 && inputDims.ndims() == 0) {
            inputDims = node->getChildEdgeAt(i)->getDims();
            continue;
        }

        if (inputDims.ndims() != node->getChildEdgeAt(i)->getDims().ndims()) {
            return true;
        }
    }

    return false;
}

/// In we have {any, any, any} -> {any} or {any} -> {any, any, any} or {any} -> {any} it means that
/// layer doesn't change memory format
/// We don't support {any, any, nchw} -> {any}
InferenceEngine::TensorDesc MKLDNNPlugin::MKLDNNEdge::getSpecifiedInputDesc(std::map<mkldnn::memory::format, size_t> formats) {
    InferenceEngine::TensorDesc inDesc;
    static int enterCount = 0;
    enterCount++;

    if (inputDesc.getLayout() != InferenceEngine::Layout::ANY) {
        --enterCount;
        return inputDesc;
    }

    auto parentPtr = getParent();
    if (parentPtr->getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Primitive descriptor for node " << parentPtr->getName() << " is not selected.";

    int inputIdx = getInputNum();
    if (inputIdx < 0)
        THROW_IE_EXCEPTION << "Edge cannot be found for node" << parentPtr->getName() << ".";

    if (inputIdx >= parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size())
        inputIdx = 0;
    inDesc = parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc;

    if (inDesc.getLayout() != InferenceEngine::Layout::ANY) {
        --enterCount;
        return inDesc;
    }

    bool isFormatChanging = nodeCanChangeDesc(parentPtr);

    if (!isFormatChanging && inputIdx < parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size() &&
            parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc.getLayout() != InferenceEngine::Layout::ANY) {
        inDesc = parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc;
        parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc = inDesc;
        --enterCount;
        return inDesc;
    }

    for (size_t i = 0; i < parentPtr->getChildEdges().size(); i++) {
        auto childEdge = parentPtr->getChildEdgeAt(i);
        auto child = childEdge->getChild();
        int childIdx = childEdge->getOutputNum();
        if (!child->getSelectedPrimitiveDescriptor() || childIdx < 0 ||
                childEdge->getDims().ndims() != getDims().ndims()) {
            continue;
        }
        if (child->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size() <= childIdx)
            childIdx = 0;
        memory::format childInDesc = MKLDNNMemoryDesc(child->getSelectedPrimitiveDescriptor()->getConfig().inConfs[childIdx].desc).getFormat();
        if (childInDesc != memory::any && childInDesc != memory::format_undef) {
            if (formats.find(childInDesc) == formats.end())
                formats[childInDesc] = 1;
            else
                formats[childInDesc] += 1;
            continue;
        }
        if (nodeCanChangeDesc(child))
            continue;

        if (enterCount < 2) {
            childInDesc = MKLDNNMemoryDesc(childEdge->getSpecifiedOutputDesc(formats)).getFormat();
            if (childInDesc != memory::any && childInDesc != memory::format_undef) {
                if (formats.find(childInDesc) == formats.end())
                    formats[childInDesc] = 1;
                else
                    formats[childInDesc] += 1;
            }
        }
    }

    if (!isFormatChanging) {
        for (size_t i = 0; i < parentPtr->getParentEdges().size(); i++) {
            auto parentEdge = parentPtr->getParentEdgeAt(i);
            auto parent = parentEdge->getParent();
            int parentIdx = parentEdge->getInputNum();
            if (!parent->getSelectedPrimitiveDescriptor() || parentIdx < 0 ||
                    parentEdge->getDims().ndims() != getDims().ndims()) {
                continue;
            }
            if (parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() <= parentIdx) {
                parentIdx = 0;
            }
            memory::format parentOutDesc = MKLDNNMemoryDesc(parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs[parentIdx].desc).getFormat();
            if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {
                if (formats.find(parentOutDesc) == formats.end())
                    formats[parentOutDesc] = 1;
                else
                    formats[parentOutDesc] += 1;
                continue;
            }
            if (nodeCanChangeDesc(parent))
                continue;

            if (enterCount < 2) {
                parentOutDesc = MKLDNNMemoryDesc(parentEdge->getSpecifiedInputDesc(formats)).getFormat();
                if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {
                    if (formats.find(parentOutDesc) == formats.end())
                        formats[parentOutDesc] = 1;
                    else
                        formats[parentOutDesc] += 1;
                }
            }
        }
    }

    size_t maxFormatCount = 0;
    memory::format desc =  MKLDNNMemory::GetPlainFormat(getDims());
    for (auto &it : formats) {
        if (maxFormatCount < it.second && MKLDNNMemory::isConsistant(getDims(), it.first)) {
            maxFormatCount = it.second;
            desc = it.first;
        }
    }

    auto inDataType = MKLDNNMemoryDesc(parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc).getDataType();
    parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc = MKLDNNMemoryDesc(getDims(), inDataType, desc);
    if (!isFormatChanging && inputIdx < parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size() &&
            parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc.getLayout() == InferenceEngine::Layout::ANY) {
        parentPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[inputIdx].desc =
                MKLDNNExtensionUtils::getUninitTensorDesc(MKLDNNMemoryDesc(getDims(), inDataType, desc));
    }

    --enterCount;
    return MKLDNNMemoryDesc(getDims(), inDataType, desc);
}

InferenceEngine::TensorDesc MKLDNNPlugin::MKLDNNEdge::getSpecifiedOutputDesc(std::map<mkldnn::memory::format, size_t> formats) {
    static int enterCount = 0;
    enterCount++;
    InferenceEngine::TensorDesc outDesc;

    if (outputDesc.getLayout() != InferenceEngine::Layout::ANY) {
        enterCount--;
        return outputDesc;
    }

    auto childPtr = getChild();
    auto parentPtr = getParent();

    if (childPtr->getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Primitive descriptor for node " << childPtr->getName() << " is not selected.";

    int outputIdx = getOutputNum();
    int inputIdx = getInputNum();
    if (outputIdx < 0) {
        THROW_IE_EXCEPTION << "Edge cannot be found for node" << childPtr->getName() << ".";
    }
    if (outputIdx >= childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size())
        outputIdx = 0;
    outDesc = childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc;

    if (outDesc.getLayout() != InferenceEngine::Layout::ANY) {
        enterCount--;
        return outDesc;
    }

    if (inputIdx >= parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size())
        inputIdx = 0;

    bool isFormatChanging = nodeCanChangeDesc(childPtr);

    if ((!isFormatChanging && outputIdx < childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() &&
            childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc.getLayout() != InferenceEngine::Layout::ANY) ||
            (isFormatChanging && inputIdx >= 0 &&
                    parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc.getLayout() != InferenceEngine::Layout::ANY)) {
        auto inputDataType = childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc.getPrecision();
        if (!isFormatChanging)
            outDesc = childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc;
        else
            outDesc = parentPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[inputIdx].desc;
        childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc = InferenceEngine::TensorDesc(inputDataType, getDims().ToSizeVector(),
                                                    {outDesc.getBlockingDesc().getBlockDims(),
                                                     outDesc.getBlockingDesc().getOrder()});
        enterCount--;
        return childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc;
    }

    for (size_t i = 0; i < childPtr->getParentEdges().size(); i++) {
        auto parentEdge = childPtr->getParentEdgeAt(i);
        auto parent = parentEdge->getParent();
        int parentIdx = parentEdge->getInputNum();
        if (!parent->getSelectedPrimitiveDescriptor() || parentIdx < 0 ||
                parentEdge->getDims().ndims() != getDims().ndims()) {
            continue;
        }
        if (parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() <= parentIdx) {
            parentIdx = 0;
        }
        memory::format parentOutDesc = MKLDNNMemoryDesc(parent->getSelectedPrimitiveDescriptor()->getConfig().outConfs[parentIdx].desc).getFormat();
        if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {
            if (formats.find(parentOutDesc) == formats.end())
                formats[parentOutDesc] = 1;
            else
                formats[parentOutDesc] += 1;
            continue;
        }
        if (nodeCanChangeDesc(parent))
            continue;

        if (enterCount < 2) {
            parentOutDesc = MKLDNNMemoryDesc(parentEdge->getSpecifiedInputDesc(formats)).getFormat();
            if (parentOutDesc != memory::any && parentOutDesc != memory::format_undef) {
                if (formats.find(parentOutDesc) == formats.end())
                    formats[parentOutDesc] = 1;
                else
                    formats[parentOutDesc] += 1;
            }
        }
    }

    if (!isFormatChanging) {
        for (size_t i = 0; i < childPtr->getChildEdges().size(); i++) {
            auto childEdge = childPtr->getChildEdgeAt(i);
            auto child = childEdge->getChild();
            int childIdx = childEdge->getOutputNum();
            if (!child->getSelectedPrimitiveDescriptor() || childIdx < 0 ||
                    childEdge->getDims().ndims() != getDims().ndims()) {
                continue;
            }
            if (child->getSelectedPrimitiveDescriptor()->getConfig().inConfs.size() <= childIdx) {
                childIdx = 0;
            }
            memory::format childInDesc = MKLDNNMemoryDesc(child->getSelectedPrimitiveDescriptor()->getConfig().inConfs[childIdx].desc).getFormat();
            if (childInDesc != memory::any && childInDesc != memory::format_undef) {
                if (formats.find(childInDesc) == formats.end())
                    formats[childInDesc] = 1;
                else
                    formats[childInDesc] += 1;
                continue;
            }
            if (nodeCanChangeDesc(child))
                continue;

            if (enterCount < 2) {
                childInDesc = MKLDNNMemoryDesc(childEdge->getSpecifiedOutputDesc(formats)).getFormat();
                if (childInDesc != memory::any && childInDesc != memory::format_undef) {
                    if (formats.find(childInDesc) == formats.end())
                        formats[childInDesc] = 1;
                    else
                        formats[childInDesc] += 1;
                }
            }
        }
    }

    size_t maxFormatCount = 0;
    memory::format format =  MKLDNNMemory::GetPlainFormat(getDims());
    for (auto &it : formats) {
        if (maxFormatCount < it.second && MKLDNNMemory::isConsistant(getDims(), it.first)) {
            maxFormatCount = it.second;
            format = it.first;
        }
    }

    auto inDataType = MKLDNNMemoryDesc(childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[getOutputNum()].desc).getDataType();
    childPtr->getSelectedPrimitiveDescriptor()->getConfig().inConfs[outputIdx].desc = MKLDNNMemoryDesc(getDims(), inDataType, format);
    if (!isFormatChanging && outputIdx < childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() &&
            childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc.getLayout() == InferenceEngine::Layout::ANY) {
        childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc =
                MKLDNNExtensionUtils::getUninitTensorDesc(MKLDNNMemoryDesc(getDims(), inDataType, format));
    }

    enterCount--;
    return childPtr->getSelectedPrimitiveDescriptor()->getConfig().outConfs[outputIdx].desc;
}

const MKLDNNPlugin::MKLDNNMemory &MKLDNNPlugin::MKLDNNEdge::getMemory() {
    if (status == Status::NotAllocated) {
        memoryPtr.reset(new MKLDNNMemory(getParent()->getEngine()));
        memoryPtr->Create(MKLDNNMemoryDesc(getDesc()), getSharedEdge()->getMemoryPtr()->GetData());
        memoryFromEdge.reset();
        changeStatus(Status::Allocated);
    }

    return *memoryPtr;
}

MKLDNNPlugin::MKLDNNMemoryPtr &MKLDNNPlugin::MKLDNNEdge::getMemoryPtr() {
    if (status == Status::NotAllocated) {
        memoryPtr.reset(new MKLDNNMemory(getParent()->getEngine()));
        memoryPtr->Create(MKLDNNMemoryDesc(getDesc()), getSharedEdge()->getMemoryPtr()->GetData());
        memoryFromEdge.reset();
        changeStatus(Status::Allocated);
    }

    return memoryPtr;
}

InferenceEngine::Blob::Ptr MKLDNNEdge::getBlob() {
    if (!memoryPtr || !dims.ndims())
        THROW_IE_EXCEPTION << "Cannot get blob! Edge isn't initialized.";
    InferenceEngine::TensorDesc desc = getDesc();

    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        desc = InferenceEngine::TensorDesc(desc.getPrecision(), dims.ToSizeVector(), desc.getLayout());
    else
        desc = InferenceEngine::TensorDesc(desc.getPrecision(), dims.ToSizeVector(), desc.getBlockingDesc());

    return make_blob_with_precision(desc, memoryPtr->GetData());
}

void MKLDNNPlugin::MKLDNNEdge::sharedMemFrom(const MKLDNNPlugin::MKLDNNEdgePtr &edge) {
    memoryFromEdge = edge;
    status = Status::NotAllocated;
}

void MKLDNNPlugin::MKLDNNEdge::validate() {
    if (status == Status::Validated)
        return;
    getMemory();
    getParent();
    getChild();
    getDims();
    if (status != Status::Allocated) {
        THROW_IE_EXCEPTION << "Error memory is not allocated!";
    }
    status = Status::Validated;
}

MKLDNNPlugin::MKLDNNEdgePtr MKLDNNPlugin::MKLDNNEdge::getSharedEdge() const {
    auto memoryFromEdgePtr = memoryFromEdge.lock();
    if (!memoryFromEdgePtr) {
        THROW_IE_EXCEPTION << "Cannot get memory ptr for edge(" << getParent()->getName() << "->"
                           << getChild()->getName() << "). The pointer on the edge with memory is empty!";
    }
    return memoryFromEdgePtr;
}

void MKLDNNEdge::init() {
    if (status != Status::NeedAllocation && status != Status::Uninitialized)
        return;
    MKLDNNEdgePtr edgePtr = getBaseEdge();
    if (edgePtr.get() == this) {
        changeStatus(Status::NeedAllocation);
        if (getInputNum() > 0 && getParent()->getSelectedPrimitiveDescriptor() &&
            getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() <= getInputNum() &&
            edgePtr != getParent()->getChildEdgeAt(0)) {
            sharedMemFrom(getParent()->getChildEdgeAt(0));
        }
    } else {
        sharedMemFrom(edgePtr);
        if (getInputNum() > 0 && getParent()->getSelectedPrimitiveDescriptor() &&
                getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs.size() <= getInputNum() &&
                edgePtr != getParent()->getChildEdgeAt(0)) {
            if (getParent()->getChildEdgeAt(0)->getStatus() != Status::NeedAllocation &&
                    getParent()->getChildEdgeAt(0)->getStatus() != Status::Uninitialized) {
                if (getParent()->getChildEdgeAt(0)->getSharedEdge() != edgePtr)
                    THROW_IE_EXCEPTION << "Unsupported behavior. Cannot mark edge "
                                       << getParent()->getChildEdgeAt(0)->getParent()->getName() << "->"
                                       << getParent()->getChildEdgeAt(0)->getChild()->getName() << " as not allocated!";
            } else {
                getParent()->getChildEdgeAt(0)->sharedMemFrom(edgePtr);
            }
        }
    }
}

/**
 * Should analize graph node dependensies, inplace node information and return root memory(edge) it view on
 *
 * @param type some magic enum values... description needed
 * @return root of view-on-memory subgraph
 */
MKLDNNEdgePtr MKLDNNEdge::getBaseEdge(LOOK look) {
    auto parentConfig = getParent()->getSelectedPrimitiveDescriptor()->getConfig();
    auto childConfig = getChild()->getSelectedPrimitiveDescriptor()->getConfig();
    int inputNum = getInputNum();
    int outputNum = getOutputNum();
    if (inputNum >= parentConfig.outConfs.size())
        inputNum = 0;
    if (outputNum >= childConfig.inConfs.size())
        outputNum = 0;

    if (childConfig.inConfs[outputNum].inPlace >= 0 && parentConfig.outConfs[inputNum].inPlace >= 0) {
        inputNum = getInputNum();
        return getParent()->getChildEdgeAt(inputNum);
    }

    if (childConfig.inConfs[outputNum].inPlace >= 0 && (look & LOOK_DOWN)) {
        int next_edge_ind = childConfig.inConfs[outputNum].inPlace;
        if (childConfig.outConfs[next_edge_ind].inPlace >= 0) {
            childConfig.outConfs[next_edge_ind].inPlace = -1;
            getChild()->initDescriptor(childConfig);
        }

        // this is a WA ... :-(
        if (childConfig.outConfs.size() <= getChild()->getChildEdges().size()) {
            // Multiple connection to some out port.
            // Will try to find implace consumer.
            for (int i = 0; i< getChild()->getChildEdges().size(); i++) {
                auto chch_edge = getChild()->getChildEdgeAt(i);
                auto chch_conf = chch_edge->getChild()->getSelectedPrimitiveDescriptor()->getConfig();


                if (chch_conf.inConfs[chch_edge->getOutputNum()].inPlace >= 0) {
                    next_edge_ind = i;
                }
            }
        }
        return getChild()->getChildEdgeAt(next_edge_ind)->getBaseEdge(LOOK_DOWN);
    } else if (parentConfig.outConfs[inputNum].inPlace >= 0 && (look & LOOK_UP)) {
        if (parentConfig.inConfs[parentConfig.outConfs[inputNum].inPlace].inPlace >= 0) {
            parentConfig.inConfs[parentConfig.outConfs[inputNum].inPlace].inPlace = -1;
            getParent()->initDescriptor(parentConfig);
        }
        return getParent()->getParentEdgeAt(parentConfig.outConfs[inputNum].inPlace)->getBaseEdge(LOOK_UP);
    }

    inputNum = getInputNum();
    return getParent()->getChildEdgeAt(inputNum);
}

bool MKLDNNEdge::inPlace(LOOK look) {
    auto parentSPD = getParent()->getSelectedPrimitiveDescriptor();
    auto childSPD = getChild()->getSelectedPrimitiveDescriptor();
    if (!parentSPD || !childSPD)
        THROW_IE_EXCEPTION << "Cannot make a decision about reorder. Primitive descriptors weren't selected.";
    int inputNum = getInputNum();
    int outputNum = getOutputNum();
    if (inputNum >= parentSPD->getConfig().outConfs.size())
        inputNum = 0;
    if (outputNum >= childSPD->getConfig().inConfs.size())
        outputNum = 0;

    if (look & LOOK_UP) {
        if (parentSPD->getConfig().outConfs[inputNum].inPlace >= 0)
            return true;
        for (const auto &inConf : parentSPD->getConfig().inConfs) {
            if (inConf.inPlace == inputNum)
                return true;
        }
    }
    if (look & LOOK_DOWN) {
        if (childSPD->getConfig().inConfs[outputNum].inPlace >= 0)
            return true;
        for (const auto &outConf : childSPD->getConfig().outConfs) {
            if (outConf.inPlace == inputNum)
                return true;
        }
    }
    return false;
}

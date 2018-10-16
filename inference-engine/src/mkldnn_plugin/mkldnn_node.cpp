// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_node.h"
#include "mkldnn_extension_mngr.h"

#include "caseless.hpp"
#include <vector>
#include <string>
#include <limits>

#include <nodes/mkldnn_batchnorm_node.h>
#include <nodes/mkldnn_concat_node.h>
#include <nodes/mkldnn_conv_node.h>
#include <nodes/mkldnn_crop_node.h>
#include <nodes/mkldnn_deconv_node.h>
#include <nodes/mkldnn_eltwise_node.h>
#include <nodes/mkldnn_fullyconnected_node.h>
#include <nodes/mkldnn_generic_node.h>
#include <nodes/mkldnn_input_node.h>
#include <nodes/mkldnn_lrn_node.h>
#include <nodes/mkldnn_pooling_node.h>
#include <nodes/mkldnn_power_node.h>
#include <nodes/mkldnn_activation_node.h>
#include <nodes/mkldnn_reorder_node.h>
#include <nodes/mkldnn_reshape_node.h>
#include <nodes/mkldnn_roi_pooling_node.h>
#include <nodes/mkldnn_depthwise_node.h>
#include <nodes/mkldnn_softmax_node.h>
#include <nodes/mkldnn_tile_node.h>
#include <nodes/mkldnn_split_node.h>
#include <nodes/mkldnn_permute_node.h>
#include <nodes/mkldnn_memory_node.hpp>
#include <mkldnn_types.h>

#include "mkldnn_extension_utils.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;

std::vector<MKLDNNNode::Registry::CreatorByLayerFunction> MKLDNNNode::Registry::_dataByLayer;

MKLDNNNode::Register<MKLDNNGenericNode> MKLDNNGenericNode::reg;
MKLDNNNode::Register<MKLDNNBatchNormalizationNode> MKLDNNBatchNormalizationNode::reg;
MKLDNNNode::Register<MKLDNNConcatNode> MKLDNNConcatNode::reg;
MKLDNNNode::Register<MKLDNNConvolutionNode> MKLDNNConvolutionNode::reg;
MKLDNNNode::Register<MKLDNNCropNode> MKLDNNCropNode::reg;
MKLDNNNode::Register<MKLDNNDeconvolutionNode> MKLDNNDeconvolutionNode::reg;
MKLDNNNode::Register<MKLDNNEltwiseNode> MKLDNNEltwiseNode::reg;
MKLDNNNode::Register<MKLDNNFullyConnectedNode> MKLDNNFullyConnectedNode::reg;
MKLDNNNode::Register<MKLDNNInputNode> MKLDNNInputNode::reg;
MKLDNNNode::Register<MKLDNNLrnNode> MKLDNNLrnNode::reg;
MKLDNNNode::Register<MKLDNNPoolingNode> MKLDNNPoolingNode::reg;
MKLDNNNode::Register<MKLDNNPowerNode> MKLDNNPowerNode::reg;
MKLDNNNode::Register<MKLDNNActivationNode> MKLDNNActivationNode::reg;
MKLDNNNode::Register<MKLDNNDepthwiseNode> MKLDNNDepthwiseNode::reg;
MKLDNNNode::Register<MKLDNNReorderNode> MKLDNNReorderNode::reg;
MKLDNNNode::Register<MKLDNNReshapeNode> MKLDNNReshapeNode::reg;
MKLDNNNode::Register<MKLDNNROIPoolingNode> MKLDNNROIPoolingNode::reg;
MKLDNNNode::Register<MKLDNNSoftMaxNode> MKLDNNSoftMaxNode::reg;
MKLDNNNode::Register<MKLDNNSplitNode> MKLDNNSplitNode::reg;
MKLDNNNode::Register<MKLDNNTileNode> MKLDNNTileNode::reg;
MKLDNNNode::Register<MKLDNNPermuteNode> MKLDNNPermuteNode::reg;
MKLDNNNode::Register<MKLDNNMemoryInputNode> MKLDNNMemoryInputNode::reg;
MKLDNNNode::Register<MKLDNNMemoryOutputNode> MKLDNNMemoryOutputNode::reg;

MKLDNNNode::MKLDNNNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng)
        : cnnLayer(layer), name(layer->name), typeStr(layer->type), type(TypeFromName(layer->type)), engine(eng),
          selectedPrimitiveDescriptorIndex(-1), permanent(false), temporary(false), constant(ConstantType::Unknown),
          profilingTask(name) {
    if (!layer->outData.empty()) {
        for (const auto& outData : layer->outData) {
            outDims.emplace_back(outData->getDims());
        }
    } else {
        if (!(CaselessEq<std::string>()(layer->type, "memory") ||
            CaselessEq<std::string>()(layer->type, "memoryinput") ||
            CaselessEq<std::string>()(layer->type, "output") ||
            CaselessEq<std::string>()(layer->type, "reorder"))) {
            THROW_IE_EXCEPTION << "Inappropriate layer type: " << layer->type << " name: " << layer->name;
        }
    }

    parentEdges.resize(layer->insData.size());
    for (const auto& inData : layer->insData) {
        inDims.emplace_back(inData.lock()->getDims());
    }
    if (layer->params.find("PrimitivesPriority") != layer->params.end()) {
        std::istringstream stream(layer->params["PrimitivesPriority"]);
        std::string str;
        while (getline(stream, str, ',')) {
            if (str.substr(0, 4) != "cpu:")
                continue;
            implPriorities.push_back(parse_impl_name(str));
            if (implPriorities[implPriorities.size() - 1] == impl_desc_type::unknown &&
                    str != "cpu:unknown")
                THROW_IE_EXCEPTION << "Unsupported CPU implementation " << str << " for node " << getName();
        }
    }
}

void MKLDNNNode::addEdge(const MKLDNNEdgeWeakPtr& edge, size_t pIndex, size_t cIndex) {
    auto edgePtr = edge.lock();
    if (!edgePtr)
        return;
    auto parentPtr = edgePtr->getParent();
    auto childPtr = edgePtr->getChild();
    if (!parentPtr || !childPtr)
        return;
    if (cIndex < parentPtr->childEdges.size()) {
        removeEdge(parentPtr->childEdges[cIndex]);
        parentPtr->childEdges[cIndex] = edge;
    } else {
        parentPtr->childEdges.push_back(edge);
    }
    if (pIndex < childPtr->parentEdges.size()) {
        removeEdge(childPtr->parentEdges[pIndex]);
        childPtr->parentEdges[pIndex] = edge;
    } else {
        childPtr->parentEdges.push_back(edge);
    }
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
            (*it).reset();
            break;
        }
    }
    for (auto it = parentPtr->childEdges.begin(); it != parentPtr->childEdges.end(); it++) {
        auto childEdge = (*it).lock();
        if (childEdge && childEdge->getChild() == childPtr && childEdge->getParent() == parentPtr) {
            (*it).reset();
            break;
        }
    }
}

void MKLDNNNode::remove() {
    for (const auto &parentEdge : parentEdges) {
        removeEdge(parentEdge);
    }
    for (const auto &childEdge : childEdges) {
        removeEdge(childEdge);
    }
}

MKLDNNNode* MKLDNNNode::CreateNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng,
                                   const MKLDNNExtensionManager::Ptr& extMgr) {
    MKLDNNNode* newNode = Registry::CreateNode(layer, eng, extMgr);
    if (!newNode)
        THROW_IE_EXCEPTION << "Unsupported primitive of type: " << layer->type << " name: " << layer->name;

    return newNode;
}

bool MKLDNNNode::isEdgesEmpty(const std::vector<MKLDNNEdgeWeakPtr>& edges) const {
    for (auto &edge : edges) {
        if (edge.lock())
            return false;
    }
    return true;
}

void MKLDNNNode::selectOptimalPrimitiveDescriptor() {
    selectPreferPrimitiveDescriptor(getPrimitivesPriority());
}

void MKLDNNNode::selectPreferPrimitiveDescriptor(const std::vector<impl_desc_type>& priority) {
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
                    auto parent_spd = parentPtr->getSelectedPrimitiveDescriptor();

                    if (parent_spd != nullptr && !parent_spd->getConfig().outConfs.empty()) {
                        int inNum = parentEdge->getInputNum();
                        if (inNum < 0 || inNum >= parent_spd->getConfig().outConfs.size()) {
                            inNum = 0;
                        }
                        if (MKLDNNExtensionUtils::initTensorsAreEqual(
                                getSupportedPrimitiveDescriptors()[i].getConfig().inConfs[j].desc,
                                parent_spd->getConfig().outConfs[inNum].desc)) {
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
        THROW_IE_EXCEPTION << "Supported primitive descriptors list is empty for node: " << getName();
    // fallback. If there are no primitives from priority list just select a first
    selectPrimitiveDescriptorByIndex(0);
}

bool MKLDNNNode::canBeInPlace() const {
    if (getParentEdges().size() != 1 || getParentEdgeAt(0)->getParent()->getChildEdges().size() != 1 ||
            (getParentEdgeAt(0)->getParent()->isConstant() && !getParentEdgeAt(0)->getChild()->isConstant()))
        return false;

    MKLDNNDims dims = getParentEdgeAt(0)->getDims();
    for (size_t cIdx = 0; cIdx < getChildEdges().size(); cIdx++) {
        if (getChildEdgeAt(cIdx)->getDims() != dims) {
            return false;
        }
    }
    return true;
}

void MKLDNNNode::resolveNotAllocatedEdges() {
    const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
    if (!selected_pd)
        THROW_IE_EXCEPTION << "Cannot find selected primitive descriptor for node: " << getName();
    for (size_t i = 0; i < getParentEdges().size() && i < selected_pd->getConfig().inConfs.size(); i++) {
        auto parentEdge = getParentEdgeAt(i);

        if (parentEdge->getStatus() != MKLDNNEdge::Status::NotAllocated || selected_pd->getConfig().inConfs[i].inPlace < 0)
            continue;

        auto * memPtr = reinterpret_cast<char*>(parentEdge->getMemory().GetData());
        parentEdge->getMemoryPtr().reset(new MKLDNNMemory(getEngine()));
        parentEdge->getMemoryPtr()->Create(MKLDNNMemoryDesc(selected_pd->getConfig().inConfs[i].desc), memPtr);

        parentEdge->changeStatus(MKLDNNEdge::Status::Allocated);
    }
    for (size_t i = 0; i < getChildEdges().size() && i < selected_pd->getConfig().outConfs.size(); i++) {
        auto childEdge = getChildEdgeAt(i);

        if (childEdge->getStatus() != MKLDNNEdge::Status::NotAllocated || selected_pd->getConfig().outConfs[i].inPlace < 0)
            continue;

        auto * memPtr = reinterpret_cast<char*>(childEdge->getMemory().GetData());
        childEdge->getMemoryPtr().reset(new MKLDNNMemory(getEngine()));
        childEdge->getMemoryPtr()->Create(MKLDNNMemoryDesc(selected_pd->getConfig().outConfs[i].desc), memPtr);

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
    SEARCH_TYPE(sse42);
    SEARCH_TYPE(blas);
    SEARCH_TYPE(any);

    SEARCH_TYPE(winograd);
    SEARCH_TYPE(_dw);
    SEARCH_TYPE(_1x1);

    if (type == impl_desc_type::unknown)
        str_type = "unknown";
    else if (str_type.empty())
        str_type = "undef";
    return str_type;
}

const MKLDNNEdgePtr MKLDNNNode::getParentEdgeAt(size_t idx) const {
    if (idx >= parentEdges.size())
        THROW_IE_EXCEPTION << "Node " << getName() << " contains less parent edges than " << idx;
    auto parentEdgePtr = parentEdges[idx].lock();
    if (!parentEdgePtr)
        THROW_IE_EXCEPTION << "Node " << getName() << " contains empty parent edge for index " << idx;
    return parentEdgePtr;
}

const MKLDNNEdgePtr MKLDNNNode::getChildEdgeAt(size_t idx) const {
    if (idx >= childEdges.size())
        THROW_IE_EXCEPTION << "Node " << getName() << " contains less child edges than " << idx;
    auto childEdgePtr = childEdges[idx].lock();
    if (!childEdgePtr)
        THROW_IE_EXCEPTION << "Node " << getName() << " contains empty child edge for index " << idx;
    return childEdgePtr;
}

std::vector<memory::format> MKLDNNNode::getAvailableFormatsForDims(const MKLDNNDims &dims) const {
    if (dims.ndims() == 1)
        return {memory::format::x};
    else if (dims.ndims() == 2)
        return {memory::format::nc};
    else if (dims.ndims() == 4)
        return {memory::format::nchw, memory::format::nChw8c, memory::format::nChw16c};
    return {memory::format::any};
}

void MKLDNNNode::execute(mkldnn::stream strm) {
    if (prim) {
        strm.submit({*prim});
    }
}

void MKLDNNNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    for (auto& desc : descs) {
        try {
            primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(engine);
            do {
                InferenceEngine::LayerConfig config;
                config.dynBatchSupport = true;
                for (size_t i = 0; i < desc.inputNumbers(); i++) {
                    InferenceEngine::DataConfig dataConfig;
                    dataConfig.inPlace = -1;
                    dataConfig.constant = false;
                    dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(getSrcMemDesc(itpd, i));
                    config.inConfs.push_back(dataConfig);
                }

                for (size_t i = 0; i < desc.outputNumbers(); i++) {
                    InferenceEngine::DataConfig dataConfig;
                    dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                    dataConfig.constant = false;
                    dataConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(getDstMemDesc(itpd, i));
                    config.outConfs.push_back(dataConfig);
                }
                impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());

                supportedPrimitiveDescriptors.emplace_back(config, impl_type);
            } while (itpd.next());
        } catch (std::exception& e) {
            // it throw exception in case of no implementation found
            continue;
        }
    }
}

void MKLDNNNode::initDescriptor(const InferenceEngine::LayerConfig &config) {
    auto* selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD) {
        return;
    }
    std::vector<InferenceEngine::TensorDesc> inDescs;
    for (const auto& inConf : config.inConfs)
        inDescs.push_back(inConf.desc);
    std::vector<InferenceEngine::TensorDesc> outDescs;
    for (const auto& outConf : config.outConfs)
        outDescs.push_back(outConf.desc);
    createDescriptor({inDescs}, {outDescs});

    InferenceEngine::LayerConfig rightConfig = getSelectedPrimitiveDescriptor()->getConfig();
    size_t selected_count = 0;
    for (size_t j = 0; j < descs.size(); j++) {
        try {
            const auto &desc = descs[j];
            primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(engine);
            do {
                InferenceEngine::LayerConfig cfg;
                cfg.dynBatchSupport = true;
                for (size_t i = 0; i < desc.inputNumbers(); i++) {
                    InferenceEngine::DataConfig dataConfig;
                    dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                    dataConfig.constant = false;
                    dataConfig.desc = getSrcMemDesc(itpd, i);
                    cfg.inConfs.push_back(dataConfig);
                }

                for (size_t i = 0; i < desc.outputNumbers(); i++) {
                    InferenceEngine::DataConfig dataConfig;
                    dataConfig.inPlace = -1;
                    dataConfig.constant = false;
                    dataConfig.desc = getDstMemDesc(itpd, i);
                    cfg.outConfs.push_back(dataConfig);
                }
                impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str().c_str());
                if (selected_count == selectedPrimitiveDescriptorIndex) {
                    if (impl_type != selectedPD->getImplementationType()) {
                        THROW_IE_EXCEPTION << "Cannot get the original layer configuration!";
                    }
                    rightConfig = cfg;
                }
                if (j == descs.size() - 1) {
                    if (impl_type == selectedPD->getImplementationType()) {
                        rightConfig = config;
                    }
                }
                selected_count++;
            } while (itpd.next());
        } catch(...) {}
    }

    if (descs.empty()) {
        const auto& selectedConfig = selectedPD->getConfig();
        if (selectedConfig.inConfs.size() != config.inConfs.size() || selectedConfig.outConfs.size() != config.outConfs.size())
            return;

        for (size_t i = 0; i < selectedConfig.inConfs.size(); i++) {
            if (selectedConfig.inConfs[i].desc.getLayout() != InferenceEngine::Layout::ANY &&
                !MKLDNNExtensionUtils::initTensorsAreEqual(selectedConfig.inConfs[i].desc, config.inConfs[i].desc))
                THROW_IE_EXCEPTION << "Incorrect descriptor for node: " << getName();
        }

        for (size_t i = 0; i < selectedConfig.outConfs.size(); i++) {
            if (selectedConfig.outConfs[i].desc.getLayout() != InferenceEngine::Layout::ANY &&
                !MKLDNNExtensionUtils::initTensorsAreEqual(selectedConfig.outConfs[i].desc, config.outConfs[i].desc))
                THROW_IE_EXCEPTION << "Incorrect descriptor for node: " << getName();
        }
        rightConfig = config;
    }

    selectedPD->getConfig() = rightConfig;
}

InferenceEngine::Blob::Ptr MKLDNNNode::createInternalBlob(InferenceEngine::SizeVector dims, bool weights) {
    auto checkSize = [](size_t dst_size, size_t src_size) {
        if (dst_size < src_size) {
            THROW_IE_EXCEPTION << "Cannot create internal buffer. Buffer can be overrun.";
        }
    };
    auto * wLayer = dynamic_cast<InferenceEngine::WeightableLayer*>(getCnnLayer().get());
    if (wLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get weightable layer for node " << getName() << ".";

    InferenceEngine::Blob::Ptr blb = weights ? wLayer->_weights : wLayer->_biases;

    if (blb == nullptr)
        THROW_IE_EXCEPTION << "Cannot get internal blob layer for node " << getName() << ".";

    InferenceEngine::TensorDesc desc(blb->precision(), dims, InferenceEngine::TensorDesc::getLayoutByDims(dims));
    InferenceEngine::TBlob<float>::Ptr internalBlob = InferenceEngine::make_shared_blob<float>(desc);
    internalBlob->allocate();
    char *data = internalBlob->buffer();
    size_t intBuffSize = internalBlob->byteSize();

    size_t offset = blb->byteSize();
    checkSize(intBuffSize, offset);
    memcpy(data, blb->buffer(), blb->byteSize());
    data += blb->byteSize();
    for (const auto &merged : getMergeWith()) {
        wLayer = dynamic_cast<InferenceEngine::WeightableLayer*>(merged->getCnnLayer().get());
        if (wLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot convert merged weightable layer for node "
                               << getName() << ".";
        blb = weights ? wLayer->_weights : wLayer->_biases;

        if (blb == nullptr)
            THROW_IE_EXCEPTION << "Cannot get internal blob layer for node " << getName() << ".";
        offset += blb->byteSize();
        checkSize(intBuffSize, offset);
        memcpy(data, blb->buffer(), blb->byteSize());
        data += blb->byteSize();
    }

    return internalBlob;
}

void MKLDNNNode::prepareMemory(const PrimitiveDescInfo *selected_pd, mkldnn::primitive_desc_iterator& itpd) {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto &dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " to node " << getChildEdgeAt(i)->getChild()->getName() << ".";
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto &srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            THROW_IE_EXCEPTION << "Destination memory didn't allocate for node " << getName()
                               << " from node " << getParentEdgeAt(i)->getParent()->getName() << ".";
    }
    std::vector<MKLDNNMemoryDesc> intDescs;
    for (auto &it : internalBlobDesc)
        intDescs.push_back(it(itpd, 0));

    internalBlobMemory.clear();
    for (size_t i = 0; i < internalBlobs.size(); i++) {
        auto& internalBlob = internalBlobs[i];
        internalBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(engine)));
        MKLDNNDims blobDims = MKLDNNDims(internalBlob->getTensorDesc().getDims());
        memory::format format = memory::oihw;

        if (blobDims.ndims() == 1) {
            format = memory::x;
        } else if (blobDims.ndims() == 2) {
            format = memory::oi;
        } else if (blobDims.ndims() == 5) {
            format = memory::goihw;
        }
        auto inDataType = MKLDNNMemoryDesc(getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc).getDataType();

        MKLDNNDims real_dims = intDescs[i].getDims();
        if (blobDims == real_dims) {  // No auto blocking
            // TODO: Cannot create memory from intDescs[i] because ScaleShift changes dims
            internalBlobMemory[i]->Create(blobDims, inDataType, intDescs[i].getFormat());
            internalBlobMemory[i]->SetData(inDataType, format, internalBlob->buffer(),
                                           blobDims.size() * MKLDNNExtensionUtils::sizeOfDataType(inDataType));
        } else {  // Auto blocking, logic and real dims are different
            if (blobDims.ndims() != real_dims.ndims() || blobDims.ndims() > 5)
                THROW_IE_EXCEPTION << getName() << " Error: CPU plugin supports auto blocking only "
                                   << "for blobs with a number of dimensions less than 6!";
            InferenceEngine::Blob::Ptr tmp_wght =
                    InferenceEngine::make_shared_blob<float>(InferenceEngine::Precision::FP32, real_dims.ToSizeVector());

            tmp_wght->allocate();

            int with_group = 0;
            if (blobDims.ndims() == 5)
                with_group = 1;

            // Logic dims
            int L_G = blobDims.ndims() > 0 && with_group ? blobDims[0] : 1;
            int L_N = blobDims.ndims() > 0 ? blobDims[0 + with_group] : 1;
            int L_C = blobDims.ndims() > 1 ? blobDims[1 + with_group] : 1;
            int L_H = blobDims.ndims() > 2 ? blobDims[2 + with_group] : 1;
            int L_W = blobDims.ndims() > 3 ? blobDims[3 + with_group] : 1;

            // Ref
            int R_G = real_dims.ndims() > 0 && with_group ? real_dims[0] : 1;
            int R_N = real_dims.ndims() > 0 ? real_dims[0 + with_group] : 1;
            int R_C = real_dims.ndims() > 1 ? real_dims[1 + with_group] : 1;
            int R_H = real_dims.ndims() > 2 ? real_dims[2 + with_group] : 1;
            int R_W = real_dims.ndims() > 3 ? real_dims[3 + with_group] : 1;

            if (L_H != R_H || L_W != R_W)
                THROW_IE_EXCEPTION << "Unsuported mode of auto blocking tensors";

            auto * tmp_data = tmp_wght->buffer().as<float*>();
            auto * in_data = internalBlob->buffer().as<float*>();
            memset(tmp_data, 0,  real_dims.size()* sizeof(float));

            for (int g = 0; g < L_G; g++)
            for (int n = 0; n < L_N; n++)
            for (int c = 0; c < L_C; c++)
            for (int h = 0; h < L_H; h++)
            for (int w = 0; w < L_W; w++) {
                int l_indx = g * L_N * L_C * L_H * L_W +
                        n * L_C * L_H * L_W +
                        c * L_H * L_W + h * L_W + w;
                int r_indx = g * R_N * R_C * R_H * R_W +
                        n * R_C * R_H * R_W +
                        c * R_H * R_W + h * R_W + w;

                tmp_data[r_indx] = in_data[l_indx];
            }
            internalBlobMemory[i]->Create(real_dims, inDataType, intDescs[i].getFormat());
            internalBlobMemory[i]->SetData(inDataType, format, tmp_wght->buffer(), tmp_wght->byteSize());
        }
    }
}

bool MKLDNNNode::isInplace() const {
    auto config = getSelectedPrimitiveDescriptor()->getConfig();

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

void MKLDNNNode::cleanup() {
    internalBlobs.clear();
    cnnLayer.reset();

    for (auto it : fusedWith) {
        it->cleanup();
    }

    for (auto it : mergedWith) {
        it->cleanup();
    }
}

std::string MKLDNNNode::typeToStr(Type type) {
    switch (type) {
        case Generic:
            return "Generic";
        case Reorder:
            return "Reorder";
        case Input:
            return "Input";
        case Output:
            return "Output";
        case Convolution:
            return "Convolution";
        case Deconvolution:
            return "Deconvolution";
        case Convolution_Sum:
            return "Convolution_Sum";
        case Convolution_Activation:
            return "Convolution_Activation";
        case Convolution_Sum_Activation:
            return "Convolution_Sum_Activation";
        case Activation:
            return "Activation";
        case Lrn:
            return "Lrn";
        case Pooling:
            return "Pooling";
        case FullyConnected:
            return "FullyConnected";
        case SoftMax:
            return "SoftMax";
        case Split:
            return "Split";
        case Concatenation:
            return "Concatenation";
        case Power:
            return "Power";
        case Depthwise:
            return "Depthwise";
        case Crop:
            return "Crop";
        case Reshape:
            return "Reshape";
        case Tile:
            return "Tile";
        case SimplerNMS:
            return "SimplerNMS";
        case ROIPooling:
            return "ROIPooling";
        case BatchNormalization:
            return "BatchNormalization";
        case Flatten:
            return "Flatten";
        case Permute:
            return "Permute";
        case Copy:
            return "Copy";
        case MemoryOutput:
            return "MemoryOutput";
        case MemoryInput:
            return "MemoryInput";
        default:
            return "Unknown";
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
            impl_desc_type::jit_sse42_dw,
            impl_desc_type::jit_sse42_1x1,
            impl_desc_type::jit_sse42,
            impl_desc_type::gemm_any,
            impl_desc_type::gemm_blas,
            impl_desc_type::gemm_avx512,
            impl_desc_type::gemm_avx2,
            impl_desc_type::gemm_sse42,
            impl_desc_type::ref_any,
            impl_desc_type::ref,
    };
    for (const auto& impl : priorities) {
        if (std::find(implPriorities.begin(), implPriorities.end(), impl) == implPriorities.end())
            implPriorities.push_back(impl);
    }
    return implPriorities;
}

bool MKLDNNNode::isUninitTensorDesc(const InferenceEngine::TensorDesc& desc) const {
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return true;

    if (desc.getBlockingDesc().getOffsetPadding() == std::numeric_limits<size_t>::max())
        return true;

    for (size_t i = 0; i < desc.getBlockingDesc().getOrder().size(); i++) {
        if (desc.getBlockingDesc().getOffsetPaddingToData()[i] == std::numeric_limits<size_t>::max() ||
                desc.getBlockingDesc().getStrides()[i] == std::numeric_limits<size_t>::max())
            return true;
    }

    return false;
}

InferenceEngine::TensorDesc MKLDNNNode::getConfiguredInputDesc(const InferenceEngine::LayerConfig& config, size_t idx) const {
    if (!isUninitTensorDesc(config.inConfs[idx].desc))
        return config.inConfs[idx].desc;

    int num = getParentEdgeAt(idx)->getInputNum();
    auto *selectedPD = getParentEdgeAt(idx)->getParent()->getSelectedPrimitiveDescriptor();
    if (!selectedPD)
        THROW_IE_EXCEPTION << "Cannot get selected primitive descriptor for node: " << getParentEdgeAt(idx)->getParent()->getName();

    if (selectedPD->getConfig().outConfs.size() <= num)
        num = 0;

    if (config.inConfs[idx].inPlace >= 0) {
        return getConfiguredOutputDesc(config, static_cast<size_t>(config.inConfs[idx].inPlace));
    }

    if (num >= 0) {
        auto parentConf = selectedPD->getConfig().outConfs[num];
        parentConf.desc.setPrecision(config.inConfs[idx].desc.getPrecision());
        if (isUninitTensorDesc(parentConf.desc) && parentConf.inPlace >= 0)
            getParentEdgeAt(idx)->getParent()->initOptimalPrimitiveDescriptor();
        parentConf = getParentEdgeAt(idx)->getParent()->getSelectedPrimitiveDescriptor()->getConfig().outConfs[num];
        if (!isUninitTensorDesc(parentConf.desc) &&
            MKLDNNExtensionUtils::initTensorsAreEqual(parentConf.desc, config.inConfs[idx].desc)) {
            return parentConf.desc;
        }

        if (config.inConfs[idx].desc.getLayout() == InferenceEngine::Layout::ANY &&
            parentConf.desc.getLayout() != InferenceEngine::Layout::ANY) {
            return InferenceEngine::TensorDesc(parentConf.desc.getPrecision(),
                                               parentConf.desc.getDims(), {
                                                       parentConf.desc.getBlockingDesc().getBlockDims(),
                                                       parentConf.desc.getBlockingDesc().getOrder()
                                               });
        }
    }

    if (config.inConfs[idx].desc.getLayout() != InferenceEngine::Layout::ANY) {
        return InferenceEngine::TensorDesc(config.inConfs[idx].desc.getPrecision(),
                                           config.inConfs[idx].desc.getDims(), {
                                                   config.inConfs[idx].desc.getBlockingDesc().getBlockDims(),
                                                   config.inConfs[idx].desc.getBlockingDesc().getOrder()
                                           });
    }

    return InferenceEngine::TensorDesc(config.inConfs[idx].desc.getPrecision(),
                                       config.inConfs[idx].desc.getDims(),
                                       InferenceEngine::TensorDesc::getLayoutByDims(config.inConfs[idx].desc.getDims()));
}

InferenceEngine::TensorDesc MKLDNNNode::getConfiguredOutputDesc(const InferenceEngine::LayerConfig& config, size_t idx) const {
    if (!isUninitTensorDesc(config.outConfs[idx].desc))
        return config.outConfs[idx].desc;

    int num = getChildEdgeAt(idx)->getOutputNum();
    auto *selectedPD = getChildEdgeAt(idx)->getChild()->getSelectedPrimitiveDescriptor();
    if (!selectedPD)
        THROW_IE_EXCEPTION << "Cannot get selected primitive descriptor for node: " << getChildEdgeAt(idx)->getChild()->getName();

    if (selectedPD->getConfig().inConfs.size() <= num)
        num = 0;

    if (config.outConfs[idx].inPlace >= 0) {
        return getConfiguredInputDesc(config, static_cast<size_t>(config.outConfs[idx].inPlace));
    }

    if (num >= 0) {
        auto childConf = selectedPD->getConfig().inConfs[num];
        childConf.desc.setPrecision(config.outConfs[idx].desc.getPrecision());
        if (isUninitTensorDesc(childConf.desc) && childConf.inPlace >= 0)
            getChildEdgeAt(idx)->getChild()->initOptimalPrimitiveDescriptor();
        childConf = getChildEdgeAt(idx)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num];
        if (!isUninitTensorDesc(childConf.desc) &&
            MKLDNNExtensionUtils::initTensorsAreEqual(childConf.desc, config.outConfs[idx].desc)) {
            return childConf.desc;
        }
        if (config.outConfs[idx].desc.getLayout() == InferenceEngine::Layout::ANY &&
            childConf.desc.getLayout() != InferenceEngine::Layout::ANY) {
            return InferenceEngine::TensorDesc(childConf.desc.getPrecision(),
                                               childConf.desc.getDims(), {
                                                       childConf.desc.getBlockingDesc().getBlockDims(),
                                                       childConf.desc.getBlockingDesc().getOrder()
                                               });
        }
    }

    if (config.outConfs[idx].desc.getLayout() != InferenceEngine::Layout::ANY) {
        return InferenceEngine::TensorDesc(config.outConfs[idx].desc.getPrecision(),
                                                                config.outConfs[idx].desc.getDims(), {
                                                                        config.outConfs[idx].desc.getBlockingDesc().getBlockDims(),
                                                                        config.outConfs[idx].desc.getBlockingDesc().getOrder()
                                                                });
    }

    return InferenceEngine::TensorDesc(config.outConfs[idx].desc.getPrecision(),
                                       config.outConfs[idx].desc.getDims(),
                                       InferenceEngine::TensorDesc::getLayoutByDims(config.outConfs[idx].desc.getDims()));
}

void MKLDNNNode::initOptimalPrimitiveDescriptor() {
    auto config = getSelectedPrimitiveDescriptor()->getConfig();
    if (isInitConfig(config))
        return;

    for (size_t i = 0; i < config.inConfs.size(); i++) {
        config.inConfs[i].desc = getConfiguredInputDesc(config, i);
    }

    for (size_t i = 0; i < config.outConfs.size(); i++) {
        config.outConfs[i].desc = getConfiguredOutputDesc(config, i);
    }
    initDescriptor(config);
}

bool MKLDNNNode::isInitConfig(const InferenceEngine::LayerConfig& config) const {
    for (const auto& configs : {config.inConfs, config.outConfs}) {
        for (const auto &dc : configs) {
            if (isUninitTensorDesc(dc.desc))
                return false;
        }
    }
    return true;
}

MKLDNNMemoryDesc MKLDNNNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.src_primitive_desc(idx).desc());
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
}

MKLDNNMemoryDesc MKLDNNNode::getDstMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = MKLDNNMemoryDesc(primitive_desc_it.dst_primitive_desc(idx).desc());
    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getChildEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
}

int MKLDNNNode::batchToProcess() {
    return dynBatchLim == 0 ? getMaxBatch() : std::min<int>(getMaxBatch(), dynBatchLim);
}

int MKLDNNNode::getMaxBatch() {
    // FIXME: batch != 0 dims number
    if (!inDims.empty())
        return inDims[0][0];
    if (!outDims.empty())
        return outDims[0][0];
    return 0;
}

void MKLDNNNode::setDynamicBatchLim(int lim) {
    dynBatchLim = lim;
    if (prim) {
        prim.setBatchLimit(batchToProcess(), getParentEdges().size(), getChildEdges().size());
    }
}

MKLDNNNode *MKLDNNNode::Registry::CreateNode(const InferenceEngine::CNNLayerPtr &layer, const mkldnn::engine& eng,
                                             const MKLDNNExtensionManager::Ptr& extMgr) {
    for (auto maker : _dataByLayer) {
        std::unique_ptr<MKLDNNNode> ol(maker(layer, eng));
        if (ol != nullptr && ol->created(extMgr))
            return ol.release();
    }
    return nullptr;
}

void MKLDNNNode::Registry::RegisterNode(MKLDNNNode::Registry::CreatorByLayerFunction f) {
    _dataByLayer.push_back(f);
}

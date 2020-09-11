// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_depthwise_node.h"
#include "desc_iterator.hpp"
#include <legacy/ie_layers.h>
#include <string>
#include <vector>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "caseless.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

MKLDNNDepthwiseNode::MKLDNNDepthwiseNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (!isWithBiases())
            return MKLDNNMemoryDesc();
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(1).desc());
    });
}

void MKLDNNDepthwiseNode::getSupportedDescriptors() {
    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    auto parentOutDims = getParentEdgeAt(0)->getDims();

    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Cannot create layer " << getName() << ": Incorrect number of inputs!";
    if (parentOutDims != getChildEdgeAt(0)->getDims())
        THROW_IE_EXCEPTION << "Cannot create layer " << getName() << ": Incorrect dimensions!";

    auto size = static_cast<size_t>(parentOutDims.ndims() == 1 ? parentOutDims[0] : parentOutDims[1]);
    SizeVector weightDims = { size };
    MKLDNNDims blocked_weightDims(weightDims);

    auto * wLayer = dynamic_cast<InferenceEngine::WeightableLayer*>(getCnnLayer().get());
    if (wLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get weightable layer for node " << getName() << ".";

    InferenceEngine::Blob::Ptr blb = wLayer->_weights;
    if (blb)
        realWeightSize = blb->size();
    internalBlobs.push_back(createInternalBlob(weightDims, true));
    if (isWithBiases()) {
        InferenceEngine::Blob::Ptr blb = wLayer->_biases;
        if (blb)
            realBiasSize = blb->size();
        internalBlobs.push_back(createInternalBlob(weightDims, false));
    }

    for (auto format : getAvailableFormatsForDims(parentOutDims)) {
        MKLDNNMemoryDesc in_candidate{parentOutDims, inputDataType, format};
        createDescriptor({in_candidate}, {});
    }
}

void MKLDNNDepthwiseNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto parentOutDims = getParentEdgeAt(0)->getDims();
    if (parentOutDims.ndims() <= 5) {
        MKLDNNNode::initSupportedPrimitiveDescriptors();
    } else {
        createSpecificDescriptor5D();
        if (specificDesc5DPtr == nullptr)
            THROW_IE_EXCEPTION << "Cannot create specific MKLDNNDescriptor for depthwise node " << getName();
        const auto& desc = *specificDesc5DPtr;
        auto itpd = desc.createPrimitiveDescriptorIterator(getEngine());
        while (itpd.is_not_end()) {
            InferenceEngine::LayerConfig config;
            config.dynBatchSupport = true;
            for (size_t i = 0; i < descInputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNMemoryDesc(InferenceEngine::TensorDesc(Precision::FP32, parentOutDims.ToSizeVector(), Layout::ANY));
                config.inConfs.push_back(dataConfig);
            }

            std::vector<mkldnn::memory::format> outFormats;
            for (size_t i = 0; i < descOutputNumbers(desc); i++) {
                InferenceEngine::DataConfig dataConfig;
                dataConfig.inPlace = canBeInPlace() ? 0 : -1;
                dataConfig.constant = false;
                dataConfig.desc = MKLDNNMemoryDesc(InferenceEngine::TensorDesc(Precision::FP32, parentOutDims.ToSizeVector(), Layout::ANY));
                config.outConfs.push_back(dataConfig);

                auto primDesc = itpd.fetch();
                auto dstPrimDesc = mkldnn_primitive_desc_query_pd(primDesc.get(), mkldnn::convert_to_c(dst_pd), 0);
                if (dstPrimDesc) {
                    outFormats.emplace_back(static_cast<memory::format>(itpd.dst_primitive_desc().desc().data.format));
                } else {
                    // This path is needed to correctly handle Deconvolution node
                    auto diffSrcPrimDesc = mkldnn_primitive_desc_query_pd(primDesc.get(), mkldnn::convert_to_c(diff_src_pd), 0);
                    if (diffSrcPrimDesc) {
                        outFormats.emplace_back(static_cast<memory::format>(itpd.diff_src_primitive_desc().desc().data.format));
                    }
                }
            }
            impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());

            supportedPrimitiveDescriptors.emplace_back(config, impl_type, outFormats);
            itpd++;
        }
    }
}

void MKLDNNDepthwiseNode::createPrimitive() {
    if (prim)
        return;

    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    auto createRightPrimitiveDescriptor = [&]() -> depthwise_forward::primitive_desc {
        auto parentOutDims = getParentEdgeAt(0)->getDims();
        if (parentOutDims.ndims() <= 5) {
            return createPrimitiveDescriptor<depthwise_forward::primitive_desc, depthwise_forward::desc>();
        } else {
            const PrimitiveDescInfo *selected_pd = getSelectedPrimitiveDescriptor();
            auto& desc = *specificDesc5DPtr;
            auto itpd = desc.createPrimitiveDescriptorIterator(getEngine(), mkldnn::primitive_attr());

            while (itpd.is_not_end())  {
                impl_desc_type impl_type = parse_impl_name(itpd.get_impl_info_str());
                if (impl_type == getSelectedPrimitiveDescriptor()->getImplementationType()) {
                    specificPrepareMemory5D(itpd);
                    std::shared_ptr<depthwise_forward::desc> selected_desc_ptr = desc;
                    depthwise_forward::primitive_desc prim_desc = depthwise_forward::primitive_desc(*selected_desc_ptr, getEngine());
                    return prim_desc;
                }
                itpd++;
            }
            THROW_IE_EXCEPTION << "Cannot create specific primitive descriptor for depthwise node " << getName() << ".";
        }
    };

    auto prim_desc = createRightPrimitiveDescriptor();

    if (isBroadcast()) {
        float broadcastValue = static_cast<float*>(internalBlobMemory[0]->GetData())[0];
        size_t blbSize = internalBlobMemory[0]->GetPrimitiveDescriptor().desc().data.dims[0];
        for (int i = 1; i < blbSize && realWeightSize != blbSize; i++) {
            static_cast<float*>(internalBlobMemory[0]->GetData())[i] = broadcastValue;
        }

        if (isWithBiases()) {
            blbSize = internalBlobMemory[1]->GetPrimitiveDescriptor().desc().data.dims[0];
            broadcastValue = static_cast<float*>(internalBlobMemory[1]->GetData())[0];
            for (int i = 1; i < blbSize && realBiasSize != blbSize; i++) {
                static_cast<float*>(internalBlobMemory[1]->GetData())[i] = broadcastValue;
            }
        }
    } else {
        size_t blbSize = internalBlobMemory[0]->GetPrimitiveDescriptor().desc().data.dims[0];
        if (realWeightSize != blbSize)
            THROW_IE_EXCEPTION << "Cannot create layer " << getName() << ": Incorrect weights!";
        if (isWithBiases()) {
            blbSize = internalBlobMemory[1]->GetPrimitiveDescriptor().desc().data.dims[0];
            if (realBiasSize != blbSize)
                THROW_IE_EXCEPTION << "Cannot create layer " << getName() << ": Incorrect biases!";
        }
    }

    if (isWithBiases()) {
        prim.reset(new depthwise_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                         internalBlobMemory[0]->GetPrimitive(),
                                         internalBlobMemory[1]->GetPrimitive(),
                                         getChildEdgeAt(0)->getMemory().GetPrimitive()));
    } else {
        prim.reset(new depthwise_forward(prim_desc, getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                         internalBlobMemory[0]->GetPrimitive(),
                                         getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }
}

bool MKLDNNDepthwiseNode::created() const {
    return getType() == Depthwise;
}

void MKLDNNDepthwiseNode::init() {
    GenericLayer* depthwiseLayer = getCnnLayer().get();
    if (depthwiseLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot get CNNLayer.";

    CaselessEq<std::string> comparator;
    if (comparator(depthwiseLayer->type, "ScaleShift")) {
        auto *scshLayer = dynamic_cast<ScaleShiftLayer*>(getCnnLayer().get());
        if (scshLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot get scale shift layer " << getName();
        if (scshLayer->_weights == nullptr)
            THROW_IE_EXCEPTION << "ScaleShift without weights is not supported";

        algorithm = depthwise_scale_shift;
        withBiases = scshLayer->_biases != nullptr;
        broadcast = static_cast<bool>(scshLayer->_broadcast);
    } else if (comparator(depthwiseLayer->type, "PReLU")) {
        auto *preluLayer = dynamic_cast<PReLULayer*>(getCnnLayer().get());
        if (preluLayer == nullptr)
            THROW_IE_EXCEPTION << "Cannot get PReLU layer " << getName();
        if (preluLayer->_weights == nullptr)
            THROW_IE_EXCEPTION << "PReLU without weights is not supported";

        algorithm = depthwise_prelu;
        withBiases = false;
        broadcast = preluLayer->_channel_shared;
    } else {
        THROW_IE_EXCEPTION << "Unsupported depthwise operation";
    }
}

void MKLDNNDepthwiseNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                           const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    MKLDNNMemoryDesc in_candidate(inputDesc[0]);
    MKLDNNMemoryDesc out_candidate(inputDesc[0]);
    MKLDNNDims weightDims({in_candidate.getDims().ndims() == 1 ? in_candidate.getDims()[0] : in_candidate.getDims()[1]});

    MKLDNNMemoryDesc wgh_candidate{weightDims, in_candidate.getDataType(), memory::x};

    if (isWithBiases()) {
        MKLDNNMemoryDesc bias_candidate{weightDims, in_candidate.getDataType(), memory::x};
        MKLDNNDescriptor desc(std::shared_ptr<depthwise_forward::desc>(
                new depthwise_forward::desc(prop_kind::forward_scoring, getAlgorithm(), in_candidate, out_candidate, wgh_candidate, bias_candidate)));
        descs.push_back(desc);
    } else {
        MKLDNNDescriptor desc(std::shared_ptr<depthwise_forward::desc>(
                new depthwise_forward::desc(prop_kind::forward_scoring, getAlgorithm(), in_candidate, out_candidate, wgh_candidate)));
        descs.push_back(desc);
    }
}

void MKLDNNDepthwiseNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
    auto config = selected_pd->getConfig();
    if (isInitConfig(config))
        return;

    if (config.inConfs.size() != 1 || config.outConfs.size() != 1 || (!isUninitTensorDesc(config.inConfs[0].desc) &&
            !isUninitTensorDesc(config.outConfs[0].desc) && config.inConfs[0].desc != config.outConfs[0].desc))
        THROW_IE_EXCEPTION << "Layer " << getName() << " has incorrect selected config!";

    if (getParentEdgeAt(0)->getDims().ndims() > 5)
        return;

    if (!isUninitTensorDesc(config.inConfs[0].desc)) {
        config.outConfs[0].desc = config.inConfs[0].desc;
    } else if (!isUninitTensorDesc(config.outConfs[0].desc)) {
        config.inConfs[0].desc = config.outConfs[0].desc;
    } else {
        config.outConfs[0].desc = config.inConfs[0].desc = getConfiguredInputDesc(config, 0);
    }

    initDescriptor(config);
}

void MKLDNNDepthwiseNode::createSpecificDescriptor5D() {
    auto parentOutDims = getParentEdgeAt(0)->getDims();
    MKLDNNDims newDims;
    for (int i = 0; i < 4; i++)
        newDims.push_back(parentOutDims[i]);
    int lastDim = 1;
    for (int i = 4; i < parentOutDims.ndims(); i++) {
        lastDim *= parentOutDims[i];
    }
    newDims.push_back(lastDim);

    MKLDNNMemoryDesc in_candidate{newDims, MKLDNNExtensionUtils::IEPrecisionToDataType(InferenceEngine::Precision::FP32), mkldnn::memory::ncdhw};
    MKLDNNMemoryDesc out_candidate(in_candidate);
    MKLDNNDims weightDims({in_candidate.getDims()[1]});

    MKLDNNMemoryDesc wgh_candidate{weightDims, in_candidate.getDataType(), memory::x};

    if (isWithBiases()) {
        MKLDNNMemoryDesc bias_candidate{weightDims, in_candidate.getDataType(), memory::x};
        MKLDNNDescriptor desc(std::shared_ptr<depthwise_forward::desc>(
                new depthwise_forward::desc(prop_kind::forward_scoring, getAlgorithm(), in_candidate, out_candidate, wgh_candidate, bias_candidate)));
        specificDesc5DPtr = std::make_shared<MKLDNNDescriptor>(desc);
    } else {
        MKLDNNDescriptor desc(std::shared_ptr<depthwise_forward::desc>(
                new depthwise_forward::desc(prop_kind::forward_scoring, getAlgorithm(), in_candidate, out_candidate, wgh_candidate)));
        specificDesc5DPtr = std::make_shared<MKLDNNDescriptor>(desc);
    }
}

void MKLDNNDepthwiseNode::specificPrepareMemory5D(mkldnn::primitive_desc_iterator& itpd) {
    std::vector<MKLDNNMemoryDesc> intDescs;
    for (auto &it : internalBlobDesc)
        intDescs.push_back(it(itpd, 0));

    internalBlobMemory.clear();
    for (size_t i = 0; i < internalBlobs.size(); i++) {
        const auto &internalBlob = internalBlobs[i];

        auto create = [&] () {
            auto newDesc = MKLDNNMemoryDesc(internalBlob->getTensorDesc());
            auto newFormat = newDesc.getFormat();
            if (newFormat == mkldnn::memory::ncdhw) {
                newFormat = mkldnn::memory::goihw;
            }
            if (newFormat == mkldnn::memory::nchw) {
                newFormat = mkldnn::memory::oihw;
            }

            MKLDNNMemory memory{ getEngine() };
            memory.Create(MKLDNNMemoryDesc(newDesc.getDims(), newDesc.getDataType(), newFormat), internalBlob->buffer());

            MKLDNNMemoryPtr _ptr = MKLDNNMemoryPtr(new MKLDNNMemory(getEngine()));
            _ptr->Create(intDescs[i]);
            _ptr->SetData(memory);

            return _ptr;
        };

        MKLDNNMemoryPtr ptr;
        if (weightCache != nullptr) {
            const uint64_t data_hash = weightCache->GetHashFunc().hash(
                    internalBlob->buffer(), internalBlob->byteSize());

            const std::string string_hash = getName() + "_" + std::to_string(i)
                                            + "_" + std::to_string(internalBlob->byteSize())
                                            + "_" + std::to_string(data_hash);

            ptr = weightCache->findOrCreate(string_hash, create);
        } else {
            ptr = create();
        }

        internalBlobMemory.push_back(ptr);
    }
}

REG_MKLDNN_PRIM_FOR(MKLDNNDepthwiseNode, Depthwise);

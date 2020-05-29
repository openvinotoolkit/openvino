// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_fullyconnected_node.h"
#include "mkldnn_activation_node.h"
#include "mkldnn_depthwise_node.h"
#include "mkldnn_quantize_node.h"
#include "desc_iterator.hpp"
#include <ie_layers.h>
#include <string>
#include <vector>
#include <mkldnn_extension_utils.h>
#include <mkldnn.hpp>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNFullyConnectedNode::MKLDNNFullyConnectedNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache), withBiases(false), baseInputsNumber(0) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(0).desc());
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (internalBlobs.size() <= 1)
            return MKLDNNMemoryDesc();
        return MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(1).desc());
    });

    auto ws = layer->blobs.find("w-scale");
    if (ws != layer->blobs.end()) {
        wScale = ws->second;
    }

    if (getCnnLayer()->type == "FullyConnected" || getCnnLayer()->type == "InnerProduct") {
        baseInputsNumber = getCnnLayer().get()->insData.size();
    }

    // Trying to find oi-scale
    if (getCnnLayer()->type == "FullyConnected" && getCnnLayer()->precision == Precision::I8) {
        if (baseInputsNumber != 1) {
            THROW_IE_EXCEPTION << "Unsupported number of inputs for quantized FullyConnected " << getCnnLayer()->name;
        }

        auto ois = layer->blobs.find("oi-scale");
        if ((getCnnLayer()->outData[0]->getPrecision() == Precision::I8 || getCnnLayer()->outData[0]->getPrecision() == Precision::U8)
            && ois == layer->blobs.end()) {
            THROW_IE_EXCEPTION << "Internal error of graph quantization - mismatch of intermediate scales and next layer type for fully connected "
                << getCnnLayer()->name;
        }
        if (ois != layer->blobs.end()) {
            // If we can find an oi-scale, then the next layer has to be an INT8.
            oScale = ois->second;
        }
    }
}

void MKLDNNFullyConnectedNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    if (inputDataType == memory::f32) {
        outputDataType = memory::f32;
    }

    if (baseInputsNumber > 1) {
        if (!fusedWith.empty()) {
            auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
            if (lastFusedLayer) {
                outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(lastFusedLayer->outData[0]->getPrecision());
            }
        }
        auto weightsDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getCnnLayer()->insData[1].lock()->getPrecision());

        // TODO(amalyse) what are the cases when we have non i8 weights and have to overide the precisions?
        if (((inputDataType != memory::u8 && inputDataType != memory::s8) || weightsDataType != memory::s8) && inputDataType != memory::bf16) {
            inputDataType = memory::f32;
            outputDataType = memory::f32;
        }
    }

    auto * fcLayer = dynamic_cast<FullyConnectedLayer*>(getCnnLayer().get());
    if (fcLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert fully connected layer.";
    if (fcLayer->_weights == nullptr && baseInputsNumber == 1) {
        THROW_IE_EXCEPTION << "Weights are empty for layer: " << fcLayer->name
                           << " used in MKLDNN node: " << getName() << "\n"
                           << "Use the second argumemt of InferenceEngine::Core::ReadNetwork"
                           << " to load them from .bin part of the IR";
    }

    if (getParentEdges().size() != baseInputsNumber)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    MKLDNNDims inDims(fcLayer->input()->getDims());

    if (inDims.ndims() == 2) {
        weightsDims = {fcLayer->_out_num, static_cast<size_t>(inDims.size(1))};
    } else if (inDims.ndims() == 4) {
        weightsDims = {fcLayer->_out_num, static_cast<size_t>(inDims[1]), static_cast<size_t>(inDims[2]),
                       static_cast<size_t>(inDims[3])};
    } else if (inDims.ndims() == 5) {
        weightsDims = {fcLayer->_out_num, static_cast<size_t>(inDims[1]), static_cast<size_t>(inDims[2]),
                       static_cast<size_t>(inDims[3]), static_cast<size_t>(inDims[4])};
    } else {
        THROW_IE_EXCEPTION << "Unsupported source format for FC layer. Expected 5, 4 or 2, got: "
                           << inDims.ndims() << " dims.";
    }

    if (baseInputsNumber == 1) {
        internalBlobs.push_back(createInternalBlob(weightsDims, true));
    }

    withBiases = (fcLayer->_biases != nullptr && fcLayer->_biases->size() != 0) || baseInputsNumber == 3;
    biasesDims.push_back(static_cast<int>(fcLayer->_out_num));
    if (withBiases && baseInputsNumber == 1) {
        internalBlobs.push_back(createInternalBlob(biasesDims, false));
    }

    if (this->getCnnLayer()->blobs.find("weights") != this->getCnnLayer()->blobs.end()) {
        Blob::Ptr weights = this->getCnnLayer()->blobs.find("weights")->second;
        if (weights->getTensorDesc().getPrecision() == Precision::I8) {
            // The weights blob has incorrect dims, so we have to fix it
            TensorDesc wdesc = internalBlobs[0]->getTensorDesc();
            wdesc.setPrecision(Precision::I8);
            InferenceEngine::TBlob<int8_t>::Ptr reshapedInt8Weights =
                    InferenceEngine::TBlob<int8_t>::Ptr(
                            new InferenceEngine::TBlob<int8_t>(wdesc, static_cast<int8_t *>(weights->buffer()),
                                                               weights->byteSize()));

            internalBlobs[0] = reshapedInt8Weights;
            if (withBiases) {
                Blob::Ptr biases = this->getCnnLayer()->blobs.find("biases")->second;
                TensorDesc bdesc = internalBlobs[1]->getTensorDesc();
                bdesc.setPrecision(Precision::I32);
                InferenceEngine::TBlob<int32_t>::Ptr reshapedInt32Biases =
                        InferenceEngine::TBlob<int32_t>::Ptr(
                                new InferenceEngine::TBlob<int32_t>(bdesc, static_cast<int32_t *>(biases->buffer()),
                                                                    biases->byteSize()));
                internalBlobs[1] = reshapedInt32Biases;
            }
        }
    }

    for (auto format : getAvailableFormatsForDims(getParentEdgeAt(0)->getDims())) {
        MKLDNNMemoryDesc in_candidate(inDims, inputDataType, format);
        MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType, memory::any);

        createDescriptor({in_candidate}, {out_candidate});
    }
}

void MKLDNNFullyConnectedNode::createPrimitive() {
    if (prim)
        return;

    std::shared_ptr<mkldnn::primitive_attr> attr = initPrimitiveAttr();
    std::shared_ptr<inner_product_forward::primitive_desc> prim_desc;
    prim_desc = std::make_shared<inner_product_forward::primitive_desc>(
            createPrimitiveDescriptor<inner_product_forward::primitive_desc, inner_product_forward::desc>(*attr));

    if (withBiases) {
        prim.reset(new inner_product_forward(*prim_desc,
                                             getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                             getWeights(),
                                             getBias(),
                                             getChildEdgeAt(0)->getMemory().GetPrimitive()));
    } else {
        prim.reset(new inner_product_forward(*prim_desc,
                                             getParentEdgeAt(0)->getMemory().GetPrimitive(),
                                             getWeights(),
                                             getChildEdgeAt(0)->getMemory().GetPrimitive()));
    }
}

void MKLDNNFullyConnectedNode::setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false) {
    int blob_idx = 0;
    mkldnn::post_ops ops;

    for (auto &node : fusedWith) {
        auto* quantizeNode = dynamic_cast<MKLDNNQuantizeNode *>(node.get());
        if (quantizeNode) {
            quantizeNode->appendPostOps(ops);
            continue;
        }

        auto* depthwiseNode = dynamic_cast<MKLDNNDepthwiseNode *>(node.get());
        if (depthwiseNode) {
            if (initWeights) {
                auto* depthwiseLayer = reinterpret_cast<WeightableLayer*>(depthwiseNode->getCnnLayer().get());
                MKLDNNDims depthwiseDims({static_cast<ptrdiff_t>(rnd_up(getChildEdgeAt(0)->getDims()[1], 16))});

                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                PostOpsIntBlobMemory[blob_idx]->Create(depthwiseDims, memory::data_type::f32, memory::format::x);

                PostOpsIntBlobMemory[blob_idx]->SetData(memory::data_type::f32, memory::x,
                                                        depthwiseLayer->_weights->buffer(),
                                                        depthwiseLayer->_weights->size() *
                                                        MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                if (depthwiseNode->isBroadcast()) {
                    float broadcastValue = static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[0];
                    for (int i = 1; i < PostOpsIntBlobMemory[blob_idx]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
                        static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[i] = broadcastValue;
                    }
                }

                if (depthwiseNode->getAlgorithm() == depthwise_scale_shift) {
                    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                    PostOpsIntBlobMemory[blob_idx + 1]->Create(depthwiseDims, memory::data_type::f32,
                                                               memory::format::x);
                    PostOpsIntBlobMemory[blob_idx + 1]->SetData(memory::data_type::f32, memory::x,
                                                                depthwiseLayer->_biases->buffer(),
                                                                depthwiseLayer->_biases->size() *
                                                                MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));

                    if (depthwiseNode->isBroadcast()) {
                        float broadcastValue = static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[0];
                        for (int i = 1; i < PostOpsIntBlobMemory[blob_idx + 1]->GetPrimitiveDescriptor().desc().data.dims[0]; i++) {
                            static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[i] = broadcastValue;
                        }
                    }

                    ops.append_depthwise(depthwiseNode->getAlgorithm(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx + 1]->GetData());

                    blob_idx += 2;
                } else {
                    ops.append_depthwise(depthwiseNode->getAlgorithm(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
                                         nullptr);

                    blob_idx += 1;
                }
            } else {
                ops.append_depthwise(depthwiseNode->getAlgorithm(),
                                     nullptr,
                                     nullptr);
            }

            continue;
        }

        auto* activationNode = dynamic_cast<MKLDNNActivationNode *>(node.get());
        if (activationNode) {
            ops.append_eltwise(1.0, activationNode->getAlgorithm(), activationNode->getAlpha(), activationNode->getBeta());

            continue;
        }
    }

    attr.set_post_ops(ops);
}

bool MKLDNNFullyConnectedNode::created() const {
    return getType() == FullyConnected;
}

memory::format MKLDNNFullyConnectedNode::weightsFormatForSrcFormat(memory::format sourceFormat) {
    switch (sourceFormat) {
        case memory::format::x:
            return memory::format::x;
        case memory::format::nc:
            return memory::format::oi;
        case memory::format::nchw:
            return memory::format::oihw;
        case memory::format::ncdhw:
            return memory::format::oidhw;
        case memory::format::nChw8c:
            return memory::format::oIhw8i;
        case memory::format::nCdhw8c:
            return memory::format::oIdhw8i;
        case memory::format::nChw16c:
            return memory::format::oIhw16i;
        case memory::format::nCdhw16c:
            return memory::format::oIdhw16i;
        default:
            THROW_IE_EXCEPTION << "Unsupported source format for node " << getName();
    }
}

const std::vector<impl_desc_type>& MKLDNNFullyConnectedNode::getPrimitivesPriority() {
    std::vector<impl_desc_type> priorities = {
            impl_desc_type::unknown,
            impl_desc_type::gemm_blas,
            impl_desc_type::gemm_avx512,
            impl_desc_type::gemm_avx2,
            impl_desc_type::gemm_avx,
            impl_desc_type::gemm_sse42,
            impl_desc_type::gemm_any,
            impl_desc_type::gemm,
            impl_desc_type::jit_gemm,
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
            impl_desc_type::ref,
    };
    for (const auto& impl : priorities) {
        if (std::find(implPriorities.begin(), implPriorities.end(), impl) == implPriorities.end())
            implPriorities.push_back(impl);
    }
    return implPriorities;
}

std::shared_ptr<mkldnn::primitive_attr> MKLDNNFullyConnectedNode::initPrimitiveAttr() {
    auto attr = std::make_shared<mkldnn::primitive_attr>(mkldnn::primitive_attr());
    if (wScale != nullptr) {
       float* wScaleData = static_cast<float*>(wScale->buffer());

       std::vector<float> oScaleDataVector;
       if (getCnnLayer()->precision == Precision::I8 && getCnnLayer()->outData[0]->getPrecision() != Precision::FP32) {
           float *oScaleData = static_cast<float *>(oScale->buffer());

           for (size_t c = 0; c < wScale->size(); c++) {
               oScaleDataVector.push_back(wScaleData[c] / oScaleData[c]);
           }
       } else {
           for (size_t c = 0; c < wScale->size(); c++) {
               oScaleDataVector.push_back(wScaleData[c]);
           }
       }

       attr->set_int_output_round_mode(mkldnn::round_nearest);
       attr->set_output_scales(1 << 1 /*through C dim*/, oScaleDataVector);
    }

    setPostOps(*attr, true);

    return attr;
}

void MKLDNNFullyConnectedNode::createDescriptor(const std::vector<InferenceEngine::TensorDesc> &inputDesc,
                                                const std::vector<InferenceEngine::TensorDesc> &outputDesc) {
    TensorDesc inDesc = inputDesc[0], outDesc = outputDesc[0];
    mkldnn::memory::data_type wdt = MKLDNNExtensionUtils::IEPrecisionToDataType(inDesc.getPrecision());
    mkldnn::memory::data_type bdt = MKLDNNExtensionUtils::IEPrecisionToDataType(inDesc.getPrecision());
    if (inDesc.getPrecision() == Precision::BF16) {
        bdt = mkldnn::memory::data_type::f32;
    }

    if (inDesc.getPrecision() == Precision::U8 || inDesc.getPrecision() == Precision::I8) {
        wdt = memory::s8;
        bdt = baseInputsNumber == 3 ? MKLDNNExtensionUtils::IEPrecisionToDataType(getCnnLayer()->insData[2].lock()->getPrecision()) : memory::f32;
    }

    if (this->getCnnLayer()->blobs.find("weights") != this->getCnnLayer()->blobs.end()) {
        Blob::Ptr weights = this->getCnnLayer()->blobs.find("weights")->second;

        if (weights->getTensorDesc().getPrecision() == Precision::I8) {
            wdt = memory::s8;
            bdt = memory::s32;

            Precision outPrec;
            if (getCnnLayer()->outData[0]->getPrecision() == Precision::FP32) {
                outPrec = Precision::FP32;
            } else {
                // define precision accordninly normalizer
                // TODO(amalyshe) do we need to have separate flow for last in int8 chain or not?
                outPrec = outDesc.getPrecision();
            }

            inDesc = TensorDesc(inDesc.getPrecision(), inputDesc[0].getDims(), inputDesc[0].getBlockingDesc());
            outDesc = TensorDesc(outPrec, outputDesc[0].getDims(), Layout::NC/*, outputDesc[0].getBlockingDesc()*/);
        }
    }

    MKLDNNMemoryDesc in_candidate(inDesc);
    MKLDNNMemoryDesc out_candidate(outDesc);

    memory::format weights_fmt = weightsFormatForSrcFormat(in_candidate.getFormat());

    MKLDNNMemoryDesc wgh_candidate(MKLDNNDims(weightsDims), wdt, weights_fmt);

    if (withBiases) {
        MKLDNNMemoryDesc bias_candidate(MKLDNNDims(biasesDims), bdt, memory::any);
        MKLDNNDescriptor desc(std::shared_ptr<inner_product_forward::desc>(
                new inner_product_forward::desc(prop_kind::forward_scoring, in_candidate, wgh_candidate,
                                                bias_candidate, out_candidate)));
        descs.push_back(desc);
    } else {
        MKLDNNDescriptor desc(std::shared_ptr<inner_product_forward::desc>(
                new inner_product_forward::desc(prop_kind::forward_scoring, in_candidate, wgh_candidate,
                                                out_candidate)));
        descs.push_back(desc);
    }
}

MKLDNNMemoryDesc MKLDNNFullyConnectedNode::getSrcMemDesc(mkldnn::primitive_desc_iterator &primitive_desc_it, size_t idx) {
    InferenceEngine::TensorDesc desc = idx > 0 ? MKLDNNMemoryDesc(primitive_desc_it.weights_primitive_desc(idx - 1).desc())
                                               : MKLDNNMemoryDesc(primitive_desc_it.src_primitive_desc(idx).desc());

    if (desc.getLayout() == InferenceEngine::Layout::ANY)
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getLayout()));
    else
        return MKLDNNMemoryDesc(InferenceEngine::TensorDesc(desc.getPrecision(),
                                                            getParentEdgeAt(idx)->getDims().ToSizeVector(),
                                                            desc.getBlockingDesc()));
}

const mkldnn::memory& MKLDNNFullyConnectedNode::getWeights() const {
    return baseInputsNumber > 1 ? getParentEdgeAt(1)->getMemory().GetPrimitive() : internalBlobMemory[0]->GetPrimitive();
}

const mkldnn::memory& MKLDNNFullyConnectedNode::getBias() const {
    return baseInputsNumber > 2 ? getParentEdgeAt(2)->getMemory().GetPrimitive() : internalBlobMemory[1]->GetPrimitive();
}

#if GraphGen(Gen_FullyConnected)
REG_MKLDNN_PRIM_FOR(MKLDNNFullyConnectedNode, FullyConnected);
#endif

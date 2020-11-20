// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_fullyconnected_node.h"
#include "mkldnn_eltwise_node.h"
#include "mkldnn_quantize_node.h"

#include <legacy/ie_layers.h>
#include <string>
#include <vector>
#include <mkldnn_extension_utils.h>
#include <mkldnn.hpp>
#include "utils/general_utils.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNFullyConnectedNode::MKLDNNFullyConnectedNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache), withBiases(false), baseInputsNumber(0) {
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        return MKLDNNMemoryDesc(primitive_desc_it.weights_desc(0));
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator &primitive_desc_it, size_t idx) -> MKLDNNMemoryDesc {
        if (internalBlobs.size() <= 1)
            return MKLDNNMemoryDesc();
        return MKLDNNMemoryDesc(primitive_desc_it.weights_desc(1));
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

    if (inputDataType == memory::data_type::f32) {
        outputDataType = memory::data_type::f32;
    }

    if (baseInputsNumber > 1) {
        if (!fusedWith.empty()) {
            auto lastFusedLayer = fusedWith[fusedWith.size() - 1].get()->getCnnLayer();
            if (lastFusedLayer) {
                outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(lastFusedLayer->outData[0]->getPrecision());
            }
        }
        auto weightsDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(getCnnLayer()->insData[1].lock()->getPrecision());

        // TODO (amalyse) what are the cases when we have non i8 weights and have to overide the precisions?
        if ((!one_of(inputDataType , memory::data_type::u8, memory::data_type::s8) || weightsDataType != memory::data_type::s8) &&
                inputDataType != memory::data_type::bf16) {
            inputDataType = memory::data_type::f32;
            outputDataType = memory::data_type::f32;
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
    MKLDNNDims outDims(fcLayer->outData[0]->getDims());

    if (inDims.ndims() == 2) {
        weightsDims = {fcLayer->_out_num, static_cast<size_t>(inDims[1])};
    } else if (inDims.ndims() == 3) {
        weightsDims = {static_cast<size_t>(outDims[2]), static_cast<size_t>(inDims[2])};
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
    if (inDims.ndims() == 3) {
        biasesDims.push_back(static_cast<int>(outDims[2]));
    } else {
        biasesDims.push_back(static_cast<int>(fcLayer->_out_num));
    }
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
        MKLDNNMemoryDesc out_candidate(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::any);

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

    prim.reset(new inner_product_forward(*prim_desc));
}

void MKLDNNFullyConnectedNode::execute(mkldnn::stream strm) {
    if (prim) {
        auto src = getParentEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();
        auto dst = getChildEdgesAtPort(0)[0]->getMemoryPtr()->GetPrimitive();

        if (withBiases)
            (*prim).execute(strm, {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, getWeights()}, {DNNL_ARG_BIAS, getBias()},
                                   {DNNL_ARG_DST, dst}});
        else
            (*prim).execute(strm, {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, getWeights()},
                                   {DNNL_ARG_DST, dst}});
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

        auto* eltwiseNode = dynamic_cast<MKLDNNEltwiseNode *>(node.get());
        if (eltwiseNode && (eltwiseNode->getOpType() == MulAdd || eltwiseNode->getOpType() == Prelu)) {
            if (initWeights) {
                auto* depthwiseLayer = reinterpret_cast<WeightableLayer*>(eltwiseNode->getCnnLayer().get());
                int ndims = getParentEdgeAt(0)->getDims().ndims();
                MKLDNNDims depthwiseDims({static_cast<ptrdiff_t>(rnd_up(ndims == 3 ? getChildEdgeAt(0)->getDims()[2] : getChildEdgeAt(0)->getDims()[1], 16))});

                PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                PostOpsIntBlobMemory[blob_idx]->Create(depthwiseDims, memory::data_type::f32, memory::format_tag::x);
                PostOpsIntBlobMemory[blob_idx]->FillZero();

                // In case ndims == 3 graph optimizer allows fusing only if all weights values are the same
                if (depthwiseLayer->blobs["weights"]->size() == 1 || ndims == 3) {
                    float broadcastValue = static_cast<float *>(depthwiseLayer->_weights->buffer())[0];
                    for (int i = 0; i < PostOpsIntBlobMemory[blob_idx]->GetDesc().getDims()[0]; i++) {
                        static_cast<float *>(PostOpsIntBlobMemory[blob_idx]->GetData())[i] = broadcastValue;
                    }
                } else {
                    PostOpsIntBlobMemory[blob_idx]->SetData(memory::data_type::f32, memory::format_tag::x,
                                                            depthwiseLayer->_weights->buffer(),
                                                            depthwiseLayer->_weights->size() *
                                                            MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));
                }

                if (eltwiseNode->getAlgorithm() == algorithm::depthwise_scale_shift) {
                    PostOpsIntBlobMemory.push_back(MKLDNNMemoryPtr(new MKLDNNMemory(getEngine())));
                    PostOpsIntBlobMemory[blob_idx + 1]->Create(depthwiseDims, memory::data_type::f32, memory::format_tag::x);
                    PostOpsIntBlobMemory[blob_idx + 1]->FillZero();

                    // In case ndims == 3 graph optimizer allows fusing only if all biases values are the same
                    if (depthwiseLayer->blobs["biases"]->size() == 1 || ndims == 3) {
                        float broadcastValue = static_cast<float *>(depthwiseLayer->_biases->buffer())[0];
                        for (int i = 0; i < PostOpsIntBlobMemory[blob_idx + 1]->GetDesc().getDims()[0]; i++) {
                            static_cast<float *>(PostOpsIntBlobMemory[blob_idx + 1]->GetData())[i] = broadcastValue;
                        }
                    } else {
                        PostOpsIntBlobMemory[blob_idx + 1]->SetData(memory::data_type::f32, memory::format_tag::x,
                                                                    depthwiseLayer->_biases->buffer(),
                                                                    depthwiseLayer->_biases->size() *
                                                                    MKLDNNExtensionUtils::sizeOfDataType(memory::data_type::f32));
                    }

                    ops.append_depthwise(eltwiseNode->getAlgorithm(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx + 1]->GetData());

                    blob_idx += 2;
                } else {
                    ops.append_depthwise(eltwiseNode->getAlgorithm(),
                                         (const float *) PostOpsIntBlobMemory[blob_idx]->GetData(),
                                         nullptr);

                    blob_idx += 1;
                }
            } else {
                ops.append_depthwise(eltwiseNode->getAlgorithm(),
                                     nullptr,
                                     nullptr);
            }

            continue;
        }

        if (eltwiseNode) {
            eltwiseNode->appendPostOps(ops);
        }
    }

    attr.set_post_ops(ops);
}

bool MKLDNNFullyConnectedNode::created() const {
    return getType() == FullyConnected;
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

       // TODO [oneDNN] : where is set_int_output_round_mode??
//       attr->set_int_output_round_mode(mkldnn::round_nearest);
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
        wdt = memory::data_type::s8;
        bdt = baseInputsNumber == 3 ? MKLDNNExtensionUtils::IEPrecisionToDataType(getCnnLayer()->insData[2].lock()->getPrecision()) : memory::data_type::f32;
    }

    if (this->getCnnLayer()->blobs.find("weights") != this->getCnnLayer()->blobs.end()) {
        Blob::Ptr weights = this->getCnnLayer()->blobs.find("weights")->second;

        if (weights->getTensorDesc().getPrecision() == Precision::I8) {
            wdt = memory::data_type::s8;
            bdt = memory::data_type::s32;

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

    MKLDNNMemoryDesc wgh_candidate(MKLDNNDims(weightsDims), wdt, mkldnn::memory::format_tag::any);

    if (withBiases) {
        MKLDNNMemoryDesc bias_candidate(MKLDNNDims(biasesDims), bdt, memory::format_tag::any);
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
    InferenceEngine::TensorDesc desc = idx > 0 ? MKLDNNMemoryDesc(primitive_desc_it.weights_desc(idx - 1))
                                               : MKLDNNMemoryDesc(primitive_desc_it.src_desc(idx));

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

REG_MKLDNN_PRIM_FOR(MKLDNNFullyConnectedNode, FullyConnected);

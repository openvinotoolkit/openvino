// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_concat_node.h"

#include <map>
#include <utility>
#include <vector>
#include <mkldnn_extension_utils.h>

#include "details/ie_exception.hpp"
#include "ie_layers.h"
#include "mkldnn.hpp"
#include "mkldnn/iml_type_mapper.h"
#include "mkldnn_dims.h"
#include "mkldnn_edge.h"
#include "mkldnn_memory.h"
#include "ie_parallel.hpp"
#include "mkldnn_conv_node.h"
#include "mkldnn_quantize_node.h"
#include "mkldnn_pooling_node.h"
#include <limits>

#include "jit_generator.hpp"
using namespace mkldnn::impl::cpu;
using namespace mkldnn::impl::utils;

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNConcatNode::MKLDNNConcatNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNConcatNode::getSupportedDescriptors() {
    auto * conLayer = dynamic_cast<ConcatLayer*>(getCnnLayer().get());

    if (conLayer == nullptr)
        THROW_IE_EXCEPTION << "Cannot convert concat layer.";

    axis = conLayer->_axis;

    if (getParentEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (getChildEdges().empty())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    auto& dstDims = getChildEdgeAt(0)->getDims();
    size_t axisDim = 0;

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto& dims = getParentEdgeAt(i)->getDims();

        if (dims.ndims() != dstDims.ndims()) {
            THROW_IE_EXCEPTION << "Number of input and output dimensions are not equal for concat node " << getName();
        }

        axisDim += dims[axis];

        for (size_t j = 0; j < dstDims.ndims(); j++) {
            if (j == axis)
                continue;
            if (dstDims[j] != dims[j]) {
                THROW_IE_EXCEPTION << "Incorrect input dimensions for concat node "
                    << getName() << " axis " << axis << " src " << dims[j]
                    << " dst " << dstDims[j];
            }
        }
    }

    if (dstDims[axis] != axisDim) {
        THROW_IE_EXCEPTION << "Incorrect output dimensions or axis for concat node " << getName();
    }
}

void MKLDNNConcatNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    inputPrecision = getCnnLayer()->insData[0].lock()->getPrecision();

    bool isMixedPrecision = false;
    for (int i = 1; i < getCnnLayer()->insData.size(); i++) {
        if (getCnnLayer()->insData[0].lock()->getPrecision() != getCnnLayer()->insData[i].lock()->getPrecision()) {
            isMixedPrecision = true;
        }
    }

    if (isMixedPrecision) {
        for (int i = 0; i < getCnnLayer()->insData.size(); i++) {
            auto inPrc = getCnnLayer()->insData[0].lock()->getPrecision();
            if (inPrc != Precision::U8 && inPrc != Precision::I8 && inPrc != Precision::FP32) {
                THROW_IE_EXCEPTION << "Unsupported mixed precision case for concat node " << getName();
            }
        }
        // We can upscale U8/I8 precisions to FP32 to enable such mixed precision cases
        inputPrecision = Precision::FP32;
    }

    // Concat node supports int8 implementations only for NC, NHWC and NDHWC layouts
    if (inputPrecision == Precision::U8 || inputPrecision == Precision::I8) {
        int ndims = getChildEdgeAt(0)->getDims().ndims();
        if (ndims != 2 && ndims != 4 && ndims != 5)
            inputPrecision = Precision::FP32;
    }

    // MKLDNN supports only equal precisions for inputs and output
    outputPrecision = inputPrecision;

    bool isInt8 = (outputPrecision == Precision::U8 || outputPrecision == Precision::I8);

    size_t blockSize = mayiuse(avx512_common) ? 16 : 8;

    auto outDims = getChildEdgeAt(0)->getDims();
    auto numOfDim = static_cast<size_t>(outDims.ndims());

    mkldnn::memory::format outputFormat;

    if (!isInt8) {
        if (blockSize == 8) {
            std::map<int, mkldnn::memory::format> formats =
                { { 2, mkldnn::memory::nc },
                  { 4, mkldnn::memory::nChw8c },
                  { 5, mkldnn::memory::nCdhw8c }};
            if (formats.find(numOfDim) != formats.end()) {
                outputFormat = formats[numOfDim];
            } else {
                outputFormat = MKLDNNMemory::GetPlainFormat(outDims);
            }
        } else if (blockSize == 16) {
            std::map<int, mkldnn::memory::format> formats =
                { { 2, mkldnn::memory::nc },
                  { 4, mkldnn::memory::nChw16c },
                  { 5, mkldnn::memory::nCdhw16c }};
            if (formats.find(numOfDim) != formats.end()) {
                outputFormat = formats[numOfDim];
            } else {
                outputFormat = MKLDNNMemory::GetPlainFormat(outDims);
            }
        } else {
            THROW_IE_EXCEPTION << "Not supported block size for concat node " << getName();
        }
    } else {
        std::map<int, mkldnn::memory::format> formats =
                { { 2, mkldnn::memory::nc },
                  { 4, mkldnn::memory::nhwc },
                  { 5, mkldnn::memory::ndhwc }};
        if (formats.find(numOfDim) != formats.end()) {
            outputFormat = formats[numOfDim];
        } else {
            outputFormat = mkldnn::memory::any;
        }
    }

    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(inputPrecision);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(outputPrecision);

    SizeVector strides(numOfDim);
    SizeVector order(numOfDim);
    SizeVector offsets(numOfDim, 0);
    size_t offset = (std::numeric_limits<size_t>::max)();

    bool hasDoubleConnection = false;
    for (int i = 0; i < getParentEdges().size(); i++) {
        for (int j = i + 1; j < getParentEdges().size(); j++) {
            if (getParentEdgeAt(i)->getParent() == getParentEdgeAt(j)->getParent()) {
                hasDoubleConnection = true;
                break;
            }
        }
    }

    size_t inputDimsAcc = 0;
    for (int i = 0; i < getParentEdges().size(); i++) {
        inputDimsAcc += getParentEdgeAt(i)->getDims()[axis];
    }

    if (axis != 1 || (numOfDim < 4 || numOfDim > 5) ||
        hasDoubleConnection || inputDimsAcc < blockSize) {
        if (!isInt8) {
            outputFormat = MKLDNNMemory::GetPlainFormat(outDims);
        }

        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = true;

        for (size_t i = 0; i < getParentEdges().size(); i++) {
            auto parentEdge = getParentEdgeAt(i);
            InferenceEngine::DataConfig inConfig;
            inConfig.inPlace = -1;
            inConfig.constant = false;
            inConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(MKLDNNMemoryDesc(parentEdge->getDims(), inputDataType, outputFormat));
            config.inConfs.push_back(inConfig);
        }

        InferenceEngine::DataConfig outConfig;
        outConfig.inPlace = -1;
        outConfig.constant = false;
        outConfig.desc = MKLDNNExtensionUtils::getUninitTensorDesc(
                MKLDNNMemoryDesc(outDims, outputDataType, outputFormat));

        config.outConfs.push_back(outConfig);

        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::ref, outputFormat);
        return;
    }

    if (!isInt8) {
        SizeVector blkDims = outDims.ToSizeVector();
        blkDims[1] = blkDims[1] / blockSize + (blkDims[1] % blockSize ? 1lu : 0lu);
        blkDims.push_back(blockSize);

        size_t blkDimsLen = blkDims.size();

        for (size_t i = 0; i < numOfDim; i++) {
            order[i] = i;
        }
        order.push_back(1);
        offsets.push_back(0);

        strides.resize(blkDimsLen);
        strides[blkDimsLen - 1] = 1;
        for (size_t i = 2lu; i <= blkDimsLen; i++) {
            if (blkDimsLen - i < axis) {
                strides[blkDimsLen - i] = (std::numeric_limits<size_t>::max)();
            } else {
                strides[blkDimsLen - i] = strides[blkDimsLen - i + 1] * blkDims[blkDimsLen - i + 1];
            }
        }

        InferenceEngine::LayerConfig config;
        config.dynBatchSupport = true;

        InferenceEngine::DataConfig outConfig;
        outConfig.inPlace = -1;
        outConfig.constant = false;
        outConfig.desc = TensorDesc(outputPrecision,
            outDims.ToSizeVector(), { blkDims, order, offset, offsets, strides });

        config.outConfs.push_back(outConfig);

        bool canBeInPlace = true;

        size_t oc = 0;
        for (size_t i = 0;  canBeInPlace && i < getParentEdges().size(); i++) {
            auto parentEdge = getParentEdgeAt(i);
            auto parentDims = parentEdge->getDims();
            const size_t ic = parentDims[1];

            if (ic % blockSize == 0 && (oc == 0 || oc % blockSize == 0)) {
                for (size_t j = 0; j <  parentEdge->getParent()->getChildEdges().size(); j++) {
                    auto child = parentEdge->getParent()->getChildEdgeAt(j)->getChild();
                    const MKLDNNConcatNode* childConcat = dynamic_cast<MKLDNNConcatNode *>(child.get());

                    if (childConcat && childConcat != this) {
                        canBeInPlace = false;
                        break;
                    }
                }
            } else {
                canBeInPlace = false;
                break;
            }

            oc += ic;
        }

        for (size_t i = 0; i < getParentEdges().size(); i++) {
            auto parentEdge = getParentEdgeAt(i);
            auto inDims = parentEdge->getDims();

            SizeVector blkDims = inDims.ToSizeVector();
            blkDims[1] = blkDims[1] / blockSize + (blkDims[1] % blockSize ? 1lu : 0lu);
            blkDims.push_back(blockSize);
            size_t blkDimsLen = blkDims.size();

            strides[blkDimsLen - 1] = 1;
            for (size_t i = 2lu; i <= blkDimsLen; i++) {
                if (blkDimsLen - i < axis) {
                    strides[blkDimsLen - i] = (std::numeric_limits<size_t>::max)();
                } else {
                    strides[blkDimsLen - i] = strides[blkDimsLen - i + 1] * blkDims[blkDimsLen - i + 1];
                }
            }

            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = canBeInPlace ? 0 : -1;
            dataConfig.constant = false;

            dataConfig.desc = TensorDesc(outputPrecision, inDims.ToSizeVector(),
                {blkDims, order, offset, offsets, strides});

            config.inConfs.push_back(dataConfig);
        }

        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::simple, outputFormat);
    } else {
        SizeVector blkDims = outDims.ToSizeVector();

        // Here we assume NHWC layout (channels are the last)
        if (numOfDim == 2) {
            order = {0, 1};
            blkDims = { blkDims[0], blkDims[1] };
        } else if (numOfDim == 4) {
            order = {0, 2, 3, 1};
            blkDims = { blkDims[0], blkDims[2], blkDims[3], blkDims[1] };
        } else {
            order = {0, 2, 3, 4, 1};
            blkDims = { blkDims[0], blkDims[2], blkDims[3], blkDims[4], blkDims[1] };
        }

        // C is the last in NHWC, so all strides are max()
        for (size_t i = 0; i < numOfDim; i++) {
            strides[i] = (std::numeric_limits<size_t>::max)();
        }

        InferenceEngine::LayerConfig config;

        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        dataConfig.desc = TensorDesc(outputPrecision,
            outDims.ToSizeVector(), { blkDims, order, offset, offsets, strides });

        config.outConfs.push_back(dataConfig);

        for (size_t i = 0; i < getParentEdges().size(); i++) {
            auto parentEdge = getParentEdgeAt(i);

            SizeVector inDims = parentEdge->getDims().ToSizeVector();
            SizeVector blkDims;

            for (size_t ord : order) {
                blkDims.push_back(inDims[ord]);
            }

            InferenceEngine::DataConfig dataConfig;
            dataConfig.inPlace = -1;
            dataConfig.constant = false;
            dataConfig.desc = TensorDesc(outputPrecision, inDims,
                    {blkDims, order, offset, offsets, strides});

            config.inConfs.push_back(dataConfig);
        }

        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::simple, outputFormat);
    }
}

void MKLDNNConcatNode::selectOptimalPrimitiveDescriptor() {
    selectPrimitiveDescriptorByIndex(0);
}

bool MKLDNNConcatNode::created() const {
    return getType() == Concatenation;
}

bool MKLDNNConcatNode::isOptimized() const {
    return getSelectedPrimitiveDescriptor() && getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].inPlace >= 0;
}

void MKLDNNConcatNode::createPrimitive() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    impl_desc_type impl_type = selected_pd->getImplementationType();

    if (impl_type == impl_desc_type::simple || prim || isOptimized()) {
        return;
    }

    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    std::vector<memory::primitive_desc> srcs_pd;
    std::vector<primitive::at> srcs_p;

    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto& srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr()) {
            auto parent = getParentEdgeAt(i)->getParent();
            THROW_IE_EXCEPTION << "Source memory from " << parent->getName() << " didn't allocate for node "
                               << getName() << ".";
        }

        auto desc = srcMemPtr->GetDescriptor();
        auto dims = getParentEdgeAt(i)->getDims();
        for (size_t j = 0; j < dims.ndims(); j++) {
            desc.data.dims[j] = dims[j];
        }

        srcs_pd.emplace_back(desc, srcMemPtr->GetPrimitiveDescriptor().get_engine());
        srcs_p.emplace_back(srcMemPtr->GetPrimitive());
    }

    auto desc = getChildEdgeAt(0)->getMemory().GetDescriptor();
    auto dims = getChildEdgeAt(0)->getDims();
    for (size_t i = 0; i < dims.ndims(); i++) {
        desc.data.dims[i] = dims[i];
        desc.data.layout_desc.blocking.padding_dims[i] = dims[i];
    }

    auto primitive_desc = concat::primitive_desc(desc, static_cast<int>(axis), srcs_pd);

    prim.reset(new concat(primitive_desc, srcs_p, getChildEdgeAt(0)->getMemory().GetPrimitive()));
}

size_t MKLDNNConcatNode::inverseOrder(const SizeVector& order, size_t axis) {
    for (size_t i = 0; i < order.size(); i++) {
        if (axis == order[i]) {
            return i;
        }
    }
    return -1;
}

void MKLDNNConcatNode::initOptimalPrimitiveDescriptor() {
    auto selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";

    if (!isOptimized()) {
        auto config = selected_pd->getConfig();
        if (!isInitConfig(config)) {
            for (size_t i = 0; i < config.inConfs.size(); i++) {
                config.inConfs[i].desc = getConfiguredInputDesc(config, i);
                // MKLDNN doesn't support different precision on inputs
                config.inConfs[i].desc.setPrecision(inputPrecision);
            }

            for (size_t i = 0; i < config.outConfs.size(); i++) {
                config.outConfs[i].desc = getConfiguredOutputDesc(config, i);
                config.outConfs[i].desc.setPrecision(outputPrecision);
            }

            initDescriptor(config);
        }

        return;
    }

    auto config = selected_pd->getConfig();
    if (isInitConfig(config))
        return;

    for (size_t i = 0; i < config.outConfs.size(); i++) {
        if (config.outConfs[i].desc.getLayout() == InferenceEngine::Layout::ANY ||
                !isUninitTensorDesc(config.outConfs[i].desc))
            continue;

        int num = getChildEdgeAt(i)->getOutputNum();
        if (num >= 0) {
            auto childConf = getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()->getConfig().inConfs[num];
            childConf.desc.setPrecision(config.outConfs[i].desc.getPrecision());

            if (getChildEdgeAt(i)->getChild()->getSelectedPrimitiveDescriptor()) {
                if (isUninitTensorDesc(childConf.desc) && childConf.inPlace >= 0)
                    getChildEdgeAt(i)->getChild()->initOptimalPrimitiveDescriptor();

                if (!isUninitTensorDesc(childConf.desc) &&
                        MKLDNNExtensionUtils::initTensorsAreEqual(childConf.desc, config.outConfs[i].desc)) {
                    config.outConfs[i].desc = childConf.desc;
                    continue;
                }
            }
        }
        config.outConfs[i].desc = InferenceEngine::TensorDesc(config.outConfs[i].desc.getPrecision(),
                                                              config.outConfs[i].desc.getDims(), {
                                                                      config.outConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                      config.outConfs[i].desc.getBlockingDesc().getOrder()
                                                              });
    }
    size_t offset = 0;
    for (size_t i = 0; i < config.inConfs.size(); i++) {
        config.inConfs[i].desc = InferenceEngine::TensorDesc(config.inConfs[i].desc.getPrecision(),
                                                             config.inConfs[i].desc.getDims(), {
                                                                  config.inConfs[i].desc.getBlockingDesc().getBlockDims(),
                                                                  config.inConfs[i].desc.getBlockingDesc().getOrder(),
                                                                  config.outConfs[0].desc.getBlockingDesc().getOffsetPadding() + offset,
                                                                  config.outConfs[0].desc.getBlockingDesc().getOffsetPaddingToData(),
                                                                  config.outConfs[0].desc.getBlockingDesc().getStrides()
                                                             });
        size_t axisSize = 1;

        if (config.inConfs[0].desc.getLayout() == Layout::NHWC) {
            // This is more general and works for any "direct" Layout (such as nchw or nhwc), but it doesn't work for nchw8c
            size_t realAxis = inverseOrder(config.inConfs[0].desc.getBlockingDesc().getOrder(), axis);
            for (size_t j = realAxis; j < config.inConfs[i].desc.getBlockingDesc().getBlockDims().size(); j++) {
                size_t jj = config.inConfs[0].desc.getBlockingDesc().getOrder()[j];
                axisSize *= config.inConfs[i].desc.getBlockingDesc().getBlockDims()[jj];
            }
        } else {
            // This works for nchw and nchw8c/nchw16c
            for (size_t j = axis; j < config.inConfs[i].desc.getBlockingDesc().getBlockDims().size(); j++) {
                axisSize *= config.inConfs[i].desc.getBlockingDesc().getBlockDims()[j];
            }
        }
        offset += axisSize;
    }
    initDescriptor(config);
}

void MKLDNNConcatNode::execute(mkldnn::stream strm) {
    if (isOptimized()) {
        return;
    }

    if (prim) {
        MKLDNNNode::execute(strm);
        return;
    }

    const MKLDNNMemory& dst_memory = getChildEdgeAt(0)->getMemory();
    const mkldnn::memory::data_type data_type = dst_memory.GetDataType();
    const bool isInt8 = (data_type == mkldnn_s8 || data_type == mkldnn_u8);

    if (isInt8) {
        uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst_memory.GetData());

        const size_t num_src = getParentEdges().size();

        std::vector<size_t> channels;
        size_t channels_size = 0;
        std::vector<const uint8_t*> src_ptrs;
        std::vector<uint8_t*> dst_ptrs;

        for (size_t i = 0; i < num_src; i++) {
            const MKLDNNMemory& src_mem = getParentEdgeAt(i)->getMemory();
            const size_t num_channels = src_mem.GetDims()[1];

            channels.push_back(num_channels);
            src_ptrs.push_back(reinterpret_cast<const uint8_t*>(src_mem.GetData()));
            dst_ptrs.push_back(dst_ptr + channels_size);
            channels_size += num_channels;
        }

        const size_t iter_count = getParentEdgeAt(0)->getMemory().GetSize() / channels[0];
        parallel_for(iter_count, [&](int i) {
            const size_t dst_off = i * channels_size;
            for (int j = 0; j < num_src; j++) {
                memcpy(dst_ptrs[j] + dst_off, src_ptrs[j] + i * channels[j], channels[j]);
            }
        });
    } else {
        if (_tasks.empty()) {
            const size_t threads = parallel_get_max_threads();

            uint8_t* dst_data = reinterpret_cast<uint8_t*>(dst_memory.GetData());

            TensorDesc dst_td = getChildEdgeAt(0)->getDesc();
            const size_t data_size = dst_td.getPrecision().size();
            Layout dst_layout = dst_td.getLayout();

            BlockingDesc dst_bd = dst_td.getBlockingDesc();
            SizeVector dst_dims = dst_td.getDims();

            const size_t dst_ndims = dst_dims.size();
            const size_t D = (dst_ndims == 5) ? dst_dims[dst_ndims - 3] : 1;
            const size_t H = (dst_ndims > 2) ? dst_dims[dst_ndims - 2] : 1;
            const size_t W = (dst_ndims > 3) ? dst_dims[dst_ndims - 1] : 1;
            const size_t HW = H * W * D;

            const size_t HW_rounded = (HW / threads) * threads;
            const size_t HW_tail = HW - HW_rounded;
            const size_t HW_per_thread = HW_rounded / threads;

            const size_t num_src = getParentEdges().size();

            size_t output_size = dst_bd.getStrides()[0] * data_size;
            std::vector<size_t> input_size;
            for (int i = 0; i < num_src; i++) {
                const TensorDesc& src_td = getParentEdgeAt(i)->getDesc();
                input_size.push_back(src_td.getBlockingDesc().getStrides()[0] * data_size);
            }

            uint8_t* dst = nullptr;
            const uint8_t* src = nullptr;

            const size_t dst_block = (dst_layout == BLOCKED) ? dst_bd.getBlockDims()[dst_ndims] : HW;
            const size_t dst_stride = dst_block * data_size;

            size_t OC = 0;

            for (size_t i = 0; i < num_src; i++) {
                const MKLDNNMemory& src_mem = getParentEdgeAt(i)->getMemory();
                const uint8_t* src_data = reinterpret_cast<const uint8_t*>(src_mem.GetData());

                size_t IC = getParentEdgeAt(i)->getDesc().getDims()[1];

                TensorDesc src_td = getChildEdgeAt(0)->getDesc();
                BlockingDesc src_bd = src_td.getBlockingDesc();
                Layout src_layout = src_td.getLayout();

                const size_t src_block = (src_layout == BLOCKED) ? src_bd.getBlockDims()[dst_ndims] : HW;
                const size_t src_stride = src_block * data_size;

                BlockingDesc dst_bd = dst_td.getBlockingDesc();

                bool dst_aligned = (dst_layout == NCHW) || (OC % dst_block == 0);

                if (dst_aligned && IC > src_block) {
                    src = src_data;
                    dst = dst_data + OC * HW * data_size;

                    if (src != dst) {
                        size_t IC_to_copy = (src_layout == BLOCKED) ? (IC / src_block) * src_block : IC;

                        size_t bytes_to_copy = HW * IC_to_copy * data_size;
                        size_t rounded_bytes_to_copy = (bytes_to_copy / threads) * threads;
                        size_t bytes_tail = bytes_to_copy - rounded_bytes_to_copy;
                        size_t bytes_per_thread = rounded_bytes_to_copy / threads;

                        for (size_t t = 0; t < threads; t++) {
                            size_t byte_size = ((t + 1) < threads) ? bytes_per_thread : bytes_per_thread + bytes_tail;

                            CopyMemTask task;

                            task.input_size = input_size[i];

                            task.src = src + t * bytes_per_thread;
                            task.dst = dst + t * bytes_per_thread;
                            task.src_stride = 0;
                            task.dst_stride = 0;
                            task.byte_size = byte_size;
                            task.iters = 1;

                            _tasks.push_back(task);
                        }

                        OC += IC_to_copy;

                        size_t IC_tail = IC - IC_to_copy;

                        if (IC_tail != 0) {
                            dst = dst_data + OC * HW * data_size;
                            src = src_data + IC_to_copy * HW * data_size;

                            size_t bytes_to_copy = IC_tail * data_size;
                            size_t offset = HW_per_thread * src_block * data_size;

                            for (size_t t = 0; t < threads; t++) {
                                size_t HW_to_copy = ((t + 1) < threads) ? HW_per_thread : HW_per_thread + HW_tail;

                                CopyMemTask task;

                                task.input_size = input_size[i];

                                task.src = src + t * offset;
                                task.dst = dst + t * offset;
                                task.src_stride = src_stride;
                                task.dst_stride = dst_stride;
                                task.byte_size = bytes_to_copy;
                                task.iters = HW_to_copy;

                                _tasks.push_back(task);
                            }
                        }

                        OC += IC_tail;
                    } else {
                        OC += IC;
                    }
                } else {
                    size_t IC_acc = 0;
                    size_t OC_tail = (OC > dst_block) ? OC % dst_block : OC;
                    size_t IC_tail = (IC < src_block) ? src_block - IC : 0;

                    while (IC_acc < IC) {
                        size_t C_to_copy = (OC_tail != 0) ? (dst_block - OC_tail) : (src_block - IC_tail);

                        if (C_to_copy + IC_acc > IC) {
                            C_to_copy = IC - IC_acc;
                        }

                        size_t bytes_to_copy = C_to_copy * data_size;
                        size_t offset = HW_per_thread * src_block * data_size;

                        size_t src_off = IC_acc == 0 ? 0 : ((IC_acc - IC_tail) * HW + IC_tail) * data_size;
                        size_t dst_off = OC == 0 ? 0 : ((OC - OC_tail) * HW + OC_tail) * data_size;

                        src = src_data + src_off;
                        dst = dst_data + dst_off;

                        for (size_t t = 0; t < threads; t++) {
                            size_t HW_to_copy = ((t + 1) < threads) ? HW_per_thread : HW_per_thread + HW_tail;

                            CopyMemTask task;

                            task.input_size = input_size[i];

                            task.src = src + t * offset;
                            task.dst = dst + t * offset;
                            task.src_stride = src_stride;
                            task.dst_stride = dst_stride;
                            task.byte_size = bytes_to_copy;
                            task.iters = HW_to_copy;

                            _tasks.push_back(task);
                        }

                        OC += C_to_copy;
                        IC_acc += C_to_copy;

                        OC_tail = OC % dst_block;
                        IC_tail = IC_acc % src_block;
                    }
                }
            }
        }

        const size_t N = batchToProcess();

        std::vector<CopyMemTask> batch_tasks = _tasks;

        if (N > 1) {
            const size_t data_size = getChildEdgeAt(0)->getDesc().getPrecision().size();
            const size_t output_size = getChildEdgeAt(0)->getDesc().getBlockingDesc().getStrides()[0] * data_size;

            for (size_t n = 1; n < N; n++) {
                for (size_t i = 0; i < _tasks.size(); i++) {
                    CopyMemTask task = _tasks[i];

                    task.src += n * task.input_size;
                    task.dst += n * output_size;

                    batch_tasks.push_back(task);
                }
            }
        }

        parallel_for(batch_tasks.size(), [&](int i) {
            CopyMemTask task = batch_tasks[i];
            const uint8_t* src = task.src;
            uint8_t* dst = task.dst;

            for (size_t j = 0; j < task.iters; j++, dst += task.dst_stride, src += task.src_stride) {
                std::memcpy(dst, src, task.byte_size);
            }
        });
    }
}

REG_MKLDNN_PRIM_FOR(MKLDNNConcatNode, Concat);
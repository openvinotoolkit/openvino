// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_permute_node.h"
#include <ie_layers.h>
#include <string>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

MKLDNNPermuteNode::MKLDNNPermuteNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket)
        : MKLDNNNode(layer, eng, socket) {}

void MKLDNNPermuteNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 1)
        THROW_IE_EXCEPTION << "Incorrect number of input edges for layer " << getName();
    if (!getChildEdges().size())
        THROW_IE_EXCEPTION << "Incorrect number of output edges for layer " << getName();

    auto& layer = getCnnLayer();
    if (!layer) {
        THROW_IE_EXCEPTION << "Cannot get CNNLayer.";
    }

    order.clear();
    std::vector<int> layerOrder = layer->GetParamAsInts("order");
    for (auto ord : layerOrder)
        order.push_back(static_cast<size_t>(ord));
}

void MKLDNNPermuteNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getCnnLayer()->insData[0].lock()->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getCnnLayer()->outData[0]->getPrecision();
    if (precision != InferenceEngine::Precision::FP32)
        precision = InferenceEngine::Precision::FP32;
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;
    if (getParentEdgeAt(0)->getDims().ndims() == 4) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nchw);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::nchw);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nchw});

        auto srcDims = getParentEdgeAt(0)->getDims();
        if (srcDims[1] % 8 == 0) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nChw8c);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nChw8c});
        }

        if (srcDims[1] % 16 == 0) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nChw16c);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nChw16c});
        }
    } else if (getParentEdgeAt(0)->getDims().ndims() == 5) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::ncdhw);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::ncdhw);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::ncdhw});

        auto srcDims = getParentEdgeAt(0)->getDims();
        if (srcDims[1] % 8 == 0) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nCdhw8c);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nCdhw8c});
        }

        if (srcDims[1] % 16 == 0) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::nCdhw16c);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::nCdhw16c});
        }
    } else {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::any);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType,
                                                   MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims()));
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, MKLDNNMemory::GetPlainFormat(getChildEdgeAt(0)->getDims())});
    }
}

void MKLDNNPermuteNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
}

static void permute_to_0231(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    // Supports only NCHW to NHWC
    int block_size = 1;
    if (!MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat())) {
        block_size = srcMemPtr->GetDescriptor().data.layout_desc.blocking.block_dims[1];
    }

    const int C = srcMemPtr->GetDims()[1];
    const int H = srcMemPtr->GetDims()[2];
    const int W = srcMemPtr->GetDims()[3];

    // NHWC
    const int src_stride = H * W * block_size;

    parallel_for3d(MB, H, W, [&](int n, int h, int w) {
        int src_off = n * C * H * W + (h * W + w) * block_size;
        int dst_off = n * H * W * C + h * W * C + w * C;

        for (int c = 0; c < C; c += block_size) {
            for (int b = 0; b < block_size; b++) {
                dst_data[dst_off] = src_data[src_off + b];
                dst_off++;
            }

            src_off += src_stride;
        }
    });
}

static void permute_to_0213(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    int block_size = 1;
    if (!MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat())) {
        block_size = srcMemPtr->GetDescriptor().data.layout_desc.blocking.block_dims[1];
    }

    const int C = srcMemPtr->GetDims()[1];
    const int H = srcMemPtr->GetDims()[2];
    const int W = srcMemPtr->GetDims()[3];

    parallel_for3d(MB, C/block_size, H, [&](int n, int c, int h) {
        for (int w = 0; w < W; w++) {
            int src_off = n*C*H*W + (c*H*W + h*W + w)*block_size;
            int dst_off = n*C*H*W + (h*C*W + w + c*W)*block_size;
            for (int b = 0; b < block_size; b++) {
                dst_data[dst_off + b] = src_data[src_off + b];
            }
        }
    });
}

static void permute_to_0312(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;

    const int C = srcMemPtr->GetDims()[1];
    const int H = srcMemPtr->GetDims()[2];
    const int W = srcMemPtr->GetDims()[3];

    parallel_for3d(MB, C, H, [&](int n, int c, int h) {
        for (int w = 0; w < W; w++) {
            int src_off = n*C*H*W + c*H*W + h*W + w;
            int dst_off = n*W*C*H + w*C*H + c*H + h;
            dst_data[dst_off] = src_data[src_off];
        }
    });
}

template <size_t scale_H = 0, size_t scale_W = 0>
static void permute_to_014253(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;

    const int C  = srcMemPtr->GetDims()[1];
    const int CH  = scale_H > 0 ? static_cast<int>(scale_H) : srcMemPtr->GetDims()[2];
    const int CW  = scale_W > 0 ? static_cast<int>(scale_W) : srcMemPtr->GetDims()[3];
    const int H  = srcMemPtr->GetDims()[4];
    const int W  = srcMemPtr->GetDims()[5];

    int src_off = 0;
    int dst_off = 0;

    for (int n = 0; n < MB; n++) {
        for (int c = 0; c < C; c++) {
            for (int h = 0; h < H; h++) {
                for (int ch = 0; ch < CH; ch++) {
                    for (int w = 0; w < W; w++) {
                        for (int cw = 0; cw < CW; cw++) {
                            src_off = n * C * CH * CW * H * W +
                                      c * CH * CW * H * W +
                                      ch * CW * H * W +
                                      cw * H * W +
                                      h * W +
                                      w;

                            dst_data[dst_off] = src_data[src_off];
                            dst_off++;
                        }
                    }
                }
            }
        }
    }
}

static void permute_to_3012(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;

    const int C  = srcMemPtr->GetDims()[1];
    const int H  = srcMemPtr->GetDims()[2];
    const int W  = srcMemPtr->GetDims()[3];

    int src_off = 0;
    int dst_off = 0;

    for (int w = 0; w < W; w++) {
        for (int n = 0; n < MB; n++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                     src_off = n * C * H * W +
                               c * H * W +
                               h * W +
                               w;

                     dst_data[dst_off] = src_data[src_off];
                     dst_off++;
                }
            }
        }
    }
}

static void permute_to_021(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;

    const int C  = srcMemPtr->GetDims()[1];
    const int S  = srcMemPtr->GetDims()[2];

    parallel_for2d(MB, S, [&](int n, int s) {
        int src_off = 0;
        int dst_off = 0;

        for (int c = 0; c < C; c++) {
            src_off = n * C * S +
                      c * S +
                      s;
            dst_off = n * S * C +
                      s * C +
                      c;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

static void permute_to_034152(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;

    const int DIM1 = srcMemPtr->GetDims()[1];
    const int DIM2 = srcMemPtr->GetDims()[2];
    const int DIM3 = srcMemPtr->GetDims()[3];
    const int DIM4 = srcMemPtr->GetDims()[4];
    const int DIM5 = srcMemPtr->GetDims()[5];

    int src_off = 0;
    int dst_off = 0;

    for (int n = 0; n < MB; n++) {
        for (int dim3 = 0; dim3 < DIM3; dim3++) {
            for (int dim4 = 0; dim4 < DIM4; dim4++) {
                for (int dim1 = 0; dim1 < DIM1; dim1++) {
                    for (int dim5 = 0; dim5 < DIM5; dim5++) {
                        for (int dim2 = 0; dim2 < DIM2; dim2++) {
                            src_off = n * DIM1 * DIM2 * DIM3 * DIM4 * DIM5 +
                                      dim1 * DIM2 * DIM3 * DIM4 * DIM5 +
                                      dim2 * DIM3 * DIM4 * DIM5 +
                                      dim3 * DIM4 * DIM5 +
                                      dim4 * DIM5 +
                                      dim5;

                            dst_data[dst_off] = src_data[src_off];
                            dst_off++;
                        }
                    }
                }
            }
        }
    }
}

static void permute_to_0132(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    int src_block_size = 1;
    if (!MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat())) {
        src_block_size = srcMemPtr->GetDescriptor().data.layout_desc.blocking.block_dims[1];
    }

    const int C = srcMemPtr->GetDims()[1];
    const int H = srcMemPtr->GetDims()[2];
    const int W = srcMemPtr->GetDims()[3];

    parallel_for3d(MB, C/src_block_size, H, [&](int n, int c, int h) {
        for (int w = 0; w < W; w++) {
            int src_off = n*C*H*W + (c*H*W + h*W + w)*src_block_size;
            int dst_off = n*C*H*W + c*H*W*src_block_size + w*H + h;
            for (int b = 0; b < src_block_size; b++) {
                dst_data[dst_off + b*H*W] = src_data[src_off + b];
            }
        }
    });
}

static void permute_to_03142(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;

    const int DIM1 = srcMemPtr->GetDims()[1];
    const int DIM2 = srcMemPtr->GetDims()[2];
    const int DIM3 = srcMemPtr->GetDims()[3];
    const int DIM4 = srcMemPtr->GetDims()[4];

    int src_off = 0;
    int dst_off = 0;

    for (int n = 0; n < MB; n++) {
        for (int dim3 = 0; dim3 < DIM3; dim3++) {
            for (int dim1 = 0; dim1 < DIM1; dim1++) {
                for (int dim4 = 0; dim4 < DIM4; dim4++) {
                    for (int dim2 = 0; dim2 < DIM2; dim2++) {
                        src_off = n * DIM1 * DIM2 * DIM3 * DIM4 +
                                  dim1 * DIM2 * DIM3 * DIM4 +
                                  dim2 * DIM3 * DIM4 +
                                  dim3 * DIM4 +
                                  dim4;

                        dst_data[dst_off] = src_data[src_off];
                        dst_off++;
                    }
                }
            }
        }
    }
}

static void permute_to_1203(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;

    const int C = srcMemPtr->GetDims()[1];
    const int H = srcMemPtr->GetDims()[2];
    const int W = srcMemPtr->GetDims()[3];

    parallel_for3d(MB, C, H, [&](int n, int c, int h) {
        for (int w = 0; w < W; w++) {
            int src_off = n * C * H * W + c * H * W + h * W + w;
            int dst_off = c * H * MB * W + h * MB * W + n * W + w;
            dst_data[dst_off] = src_data[src_off];
        }
    });
}

static void permute_to_02134(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;

    const int DIM1 = srcMemPtr->GetDims()[1];
    const int DIM2 = srcMemPtr->GetDims()[2];
    const int DIM3 = srcMemPtr->GetDims()[3];
    const int DIM4 = srcMemPtr->GetDims()[4];

    parallel_for4d(MB, DIM2, DIM1, DIM3, [&](int n, int dim2, int dim1, int dim3) {
        for (int dim4 = 0; dim4 < DIM4; dim4++) {
            int src_off = n * DIM1 * DIM2 * DIM3 * DIM4 +
                          dim1 * DIM2 * DIM3 * DIM4 +
                          dim2 * DIM3 * DIM4 +
                          dim3 * DIM4 +
                          dim4;
            int dst_off = n * DIM2 * DIM1 * DIM3 * DIM4 +
                          dim2 * DIM1 * DIM3 * DIM4 +
                          dim1 * DIM3 * DIM4 +
                          dim3 * DIM4 +
                          dim4;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

static void permute_to_02431(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;

    const int DIM1 = srcMemPtr->GetDims()[1];
    const int DIM2 = srcMemPtr->GetDims()[2];
    const int DIM3 = srcMemPtr->GetDims()[3];
    const int DIM4 = srcMemPtr->GetDims()[4];

    parallel_for4d(MB, DIM2, DIM4, DIM3, [&](int n, int dim2, int dim4, int dim3) {
        for (int dim1 = 0; dim1 < DIM1; dim1++) {
            int src_off = n * DIM1 * DIM2 * DIM3 * DIM4 +
                          dim1 * DIM2 * DIM3 * DIM4 +
                          dim2 * DIM3 * DIM4 +
                          dim3 * DIM4 +
                          dim4;
            int dst_off = n * DIM2 * DIM4 * DIM3 * DIM1 +
                          dim2 * DIM4 * DIM3 * DIM1 +
                          dim4 * DIM3 * DIM1 +
                          dim3 * DIM1 +
                          dim1;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

static void permute_to_04231(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;

    const int DIM1 = srcMemPtr->GetDims()[1];
    const int DIM2 = srcMemPtr->GetDims()[2];
    const int DIM3 = srcMemPtr->GetDims()[3];
    const int DIM4 = srcMemPtr->GetDims()[4];

    parallel_for4d(MB, DIM4, DIM2, DIM3, [&](int n, int dim4, int dim2, int dim3) {
        for (int dim1 = 0; dim1 < DIM1; dim1++) {
            int src_off = n * DIM1 * DIM2 * DIM3 * DIM4 +
                          dim1 * DIM2 * DIM3 * DIM4 +
                          dim2 * DIM3 * DIM4 +
                          dim3 * DIM4 +
                          dim4;
            int dst_off = n * DIM4 * DIM2 * DIM3 * DIM1 +
                          dim4 * DIM2 * DIM3 * DIM1 +
                          dim2 * DIM3 * DIM1 +
                          dim3 * DIM1 +
                          dim1;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

static void permute_to_102(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;

    const int C  = srcMemPtr->GetDims()[1];
    const int S  = srcMemPtr->GetDims()[2];

    parallel_for2d(MB, S, [&](int n, int s) {
        int src_off = 0;
        int dst_off = 0;

        for (int c = 0; c < C; c++) {
            src_off = n * C * S +
                      c * S +
                      s;
            dst_off = c * MB * S +
                      n * S +
                      s;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

static void permute_to_02341(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;

    const int DIM1 = srcMemPtr->GetDims()[1];
    const int DIM2 = srcMemPtr->GetDims()[2];
    const int DIM3 = srcMemPtr->GetDims()[3];
    const int DIM4 = srcMemPtr->GetDims()[4];

    parallel_for4d(MB, DIM2, DIM3, DIM4, [&](int n, int dim2, int dim3, int dim4) {
        for (int dim1 = 0; dim1 < DIM1; dim1++) {
            int src_off = n * DIM1 * DIM2 * DIM3 * DIM4 +
                          dim1 * DIM2 * DIM3 * DIM4 +
                          dim2 * DIM3 * DIM4 +
                          dim3 * DIM4 +
                          dim4;
            int dst_off = n * DIM2 * DIM3 * DIM4 * DIM1 +
                          dim2 * DIM3 * DIM4 * DIM1 +
                          dim3 * DIM4 * DIM1 +
                          dim4 * DIM1 +
                          dim1;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

static void permute_to_04123(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());
    src_data += srcMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;
    dst_data += dstMemPtr->GetDescriptor().data.layout_desc.blocking.offset_padding;

    const int DIM1 = srcMemPtr->GetDims()[1];
    const int DIM2 = srcMemPtr->GetDims()[2];
    const int DIM3 = srcMemPtr->GetDims()[3];
    const int DIM4 = srcMemPtr->GetDims()[4];

    parallel_for4d(MB, DIM4, DIM1, DIM2, [&](int n, int dim4, int dim1, int dim2) {
        for (int dim3 = 0; dim3 < DIM3; dim3++) {
            int src_off = n * DIM1 * DIM2 * DIM3 * DIM4 +
                          dim1 * DIM2 * DIM3 * DIM4 +
                          dim2 * DIM3 * DIM4 +
                          dim3 * DIM4 +
                          dim4;
            int dst_off = n * DIM4 * DIM1 * DIM2 * DIM3 +
                          dim4 * DIM1 * DIM2 * DIM3 +
                          dim1 * DIM2 * DIM3 +
                          dim2 * DIM3 +
                          dim3;

            dst_data[dst_off] = src_data[src_off];
        }
    });
}

std::multimap<InferenceEngine::SizeVector, MKLDNNPermuteNode::PermuteImpl> MKLDNNPermuteNode::OptimizedCases = {
        {{0, 2, 3, 1}, MKLDNNPermuteNode::PermuteImpl(permute_to_0231, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return true;
        })},  // NCHW -> NHWC case
        {{0, 1, 4, 2, 5, 3}, MKLDNNPermuteNode::PermuteImpl(permute_to_014253<2, 2>, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat()) && srcMemPtr->GetDims()[2] == 2 && srcMemPtr->GetDims()[3] == 2;
        })},  // Dense upsample convolution case (scale = 2)
        {{0, 1, 4, 2, 5, 3}, MKLDNNPermuteNode::PermuteImpl(permute_to_014253<0, 0>, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat());
        })},  // Dense upsample convolution case (generic)
        {{3, 0, 1, 2}, MKLDNNPermuteNode::PermuteImpl(permute_to_3012, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat()) && MB == srcMemPtr->GetDims()[0];
        })},  // LPR case
        {{0, 2, 1, 3}, MKLDNNPermuteNode::PermuteImpl(permute_to_0213, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat());
        })},  // shufflenet
        {{0, 2, 1}, MKLDNNPermuteNode::PermuteImpl(permute_to_021, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat());
        })},  // self attention block
        {{0, 3, 4, 1, 5, 2}, MKLDNNPermuteNode::PermuteImpl(permute_to_034152, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat());
        })},  // learning-to-see-in-the-dark-sony
        {{0, 1, 3, 2}, MKLDNNPermuteNode::PermuteImpl(permute_to_0132, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return true;
        })},
        {{0, 3, 1, 4, 2}, MKLDNNPermuteNode::PermuteImpl(permute_to_03142, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat());
        })},
        {{1, 2, 0, 3}, MKLDNNPermuteNode::PermuteImpl(permute_to_1203, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat()) && MB == srcMemPtr->GetDims()[0];
        })},
        {{0, 2, 1, 3, 4}, MKLDNNPermuteNode::PermuteImpl(permute_to_02134, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat());
        })},
        {{0, 2, 4, 3, 1}, MKLDNNPermuteNode::PermuteImpl(permute_to_02431, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat());
        })},
        {{0, 4, 2, 3, 1}, MKLDNNPermuteNode::PermuteImpl(permute_to_04231, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat());
        })},
        {{0, 3, 1, 2}, MKLDNNPermuteNode::PermuteImpl(permute_to_0312, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat());
        })},
        {{1, 0, 2}, MKLDNNPermuteNode::PermuteImpl(permute_to_102, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat()) && MB == srcMemPtr->GetDims()[0];
        })},
        {{0, 2, 3, 4, 1}, MKLDNNPermuteNode::PermuteImpl(permute_to_02341, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat());
        })},
        {{0, 4, 1, 2, 3}, MKLDNNPermuteNode::PermuteImpl(permute_to_04123, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return MKLDNNMemory::IsPlainFormat(srcMemPtr->GetFormat());
        })},
};

void MKLDNNPermuteNode::execute(mkldnn::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetData());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetData());

    for (const auto &impl : OptimizedCases) {
        if (impl.first == order && impl.second.isValidParams(batchToProcess(), srcMemPtr, dstMemPtr)) {
            impl.second.execute(batchToProcess(), srcMemPtr, dstMemPtr);
            return;
        }
    }

    auto srcBlob = getParentEdgeAt(0)->getBlob();
    TensorDesc srcDesc = srcBlob->getTensorDesc();

    SizeVector& dims = srcDesc.getDims();
    InferenceEngine::SizeVector orderedDims;
    for (auto ord : order) {
        orderedDims.push_back(dims[ord]);
    }
    TensorDesc dstDesc(InferenceEngine::Precision::FP32, dims, {orderedDims, order});

    int dataSize = srcBlob->size() / srcDesc.getDims()[0] * batchToProcess();

    parallel_for(dataSize, [&](int i) {
        dst_data[dstDesc.offset(i)] = src_data[srcDesc.offset(i)];
    });
}

bool MKLDNNPermuteNode::created() const {
    return getType() == Permute;
}

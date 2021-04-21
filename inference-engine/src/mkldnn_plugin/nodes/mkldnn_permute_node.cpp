// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_permute_node.h"
#include <legacy/ie_layers.h>
#include <string>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"

#include <algorithm>

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;


MKLDNNPermuteNode::MKLDNNPermuteNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache)
        : MKLDNNNode(layer, eng, cache) {}

void MKLDNNPermuteNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 1)
        IE_THROW() << "Incorrect number of input edges for layer " << getName();
    if (!getChildEdges().size())
        IE_THROW() << "Incorrect number of output edges for layer " << getName();

    auto& layer = getCnnLayer();
    if (!layer) {
        IE_THROW() << "Cannot get CNNLayer.";
    }

    order.clear();
    std::vector<int> layerOrder = layer->GetParamAsInts("order");
    for (auto ord : layerOrder)
        order.push_back(static_cast<size_t>(ord));

    if (order.empty()) {
        size_t rank = getParentEdgeAt(0)->getDims().ndims();
        for (size_t i = 1; i <= rank; ++i) {
            order.emplace_back(rank - i);
        }
    }
}

void MKLDNNPermuteNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    prec = getCnnLayer()->insData[0].lock()->getPrecision();
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(prec);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(prec);

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(1);
    config.outConfs.resize(1);
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;
    if (getParentEdgeAt(0)->getDims().ndims() == 4) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nchw);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nchw);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::nchw});

        auto srcDims = getParentEdgeAt(0)->getDims();
        if (srcDims[1] % 8 == 0) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nChw8c);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::nChw8c});
        }

        if (srcDims[1] % 16 == 0) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nChw16c);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::nChw16c});
        }

        if (prec == Precision::FP32 || prec == Precision::I8 || prec == Precision::U8) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nhwc);
            config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::nhwc);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::nhwc});
        }
    } else if (getParentEdgeAt(0)->getDims().ndims() == 5) {
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::ncdhw);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::ncdhw);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::ncdhw});

        auto srcDims = getParentEdgeAt(0)->getDims();
        if (srcDims[1] % 8 == 0) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nCdhw8c);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::nCdhw8c});
        }

        if (srcDims[1] % 16 == 0) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::nCdhw16c);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::nCdhw16c});
        }

        if (prec == Precision::FP32 || prec == Precision::I8 || prec == Precision::U8) {
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, memory::format_tag::ndhwc);
            config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, memory::format_tag::ndhwc);
            supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown, memory::format_tag::ndhwc});
        }
    } else {
        // general plain case
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType);
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::unknown});
    }
}

void MKLDNNPermuteNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        IE_THROW() << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        IE_THROW() << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set.";

    PermuteParams params;
    params.data_size = getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc.getPrecision().size();
    params.order = order;

    auto srcDesc = getParentEdgeAt(0)->getDesc();
    params.src_block_dims = srcDesc.getBlockingDesc().getBlockDims();
    params.src_block_order = srcDesc.getBlockingDesc().getOrder();

    auto dstDesc = getChildEdgeAt(0)->getDesc();
    params.dst_block_dims = dstDesc.getBlockingDesc().getBlockDims();
    params.dst_block_order = dstDesc.getBlockingDesc().getOrder();

    permuteKernel = std::unique_ptr<PermuteKernel>(new PermuteKernel(params));
}

static void permute_to_0231(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());
    // Supports only NCHW to NHWC
    int block_size = 1;
    if (!srcMemPtr->GetDesc().isPlainFormat()) {
        const auto &blk_desc = srcMemPtr->GetDescriptor().data.format_desc.blocking;
        auto found = std::find(blk_desc.inner_idxs, blk_desc.inner_idxs + blk_desc.inner_nblks, 1);
        auto pos = std::distance(found, blk_desc.inner_idxs);
        block_size = blk_desc.inner_blks[pos];
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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());
    int block_size = 1;
    if (!srcMemPtr->GetDesc().isPlainFormat()) {
        const auto &blk_desc = srcMemPtr->GetDescriptor().data.format_desc.blocking;
        auto found = std::find(blk_desc.inner_idxs, blk_desc.inner_idxs + blk_desc.inner_nblks, 1);
        auto pos = std::distance(found, blk_desc.inner_idxs);
        block_size = blk_desc.inner_blks[pos];
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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());

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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());

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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());

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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());

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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());

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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());
    int src_block_size = 1;
    if (!srcMemPtr->GetDesc().isPlainFormat()) {
        const auto &blk_desc = srcMemPtr->GetDescriptor().data.format_desc.blocking;
        auto found = std::find(blk_desc.inner_idxs, blk_desc.inner_idxs + blk_desc.inner_nblks, 1);
        auto pos = std::distance(found, blk_desc.inner_idxs);
        src_block_size = blk_desc.inner_blks[pos];
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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());

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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());

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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());

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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());

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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());

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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());

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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());

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
    auto src_data = reinterpret_cast<const float *>(srcMemPtr->GetPtr());
    auto dst_data = reinterpret_cast<float *>(dstMemPtr->GetPtr());

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

const std::multimap<InferenceEngine::SizeVector, MKLDNNPermuteNode::PermuteImpl> MKLDNNPermuteNode::OptimizedCases = {
        {{0, 2, 3, 1}, MKLDNNPermuteNode::PermuteImpl(permute_to_0231, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return true;
        })},  // NCHW -> NHWC case
        {{0, 1, 4, 2, 5, 3}, MKLDNNPermuteNode::PermuteImpl(permute_to_014253<2, 2>, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat() && srcMemPtr->GetDims()[2] == 2 && srcMemPtr->GetDims()[3] == 2;
        })},  // Dense upsample convolution case (scale = 2)
        {{0, 1, 4, 2, 5, 3}, MKLDNNPermuteNode::PermuteImpl(permute_to_014253<0, 0>, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat();
        })},  // Dense upsample convolution case (generic)
        {{3, 0, 1, 2}, MKLDNNPermuteNode::PermuteImpl(permute_to_3012, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat() && MB == srcMemPtr->GetDims()[0];
        })},  // LPR case
        {{0, 2, 1, 3}, MKLDNNPermuteNode::PermuteImpl(permute_to_0213, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat();
        })},  // shufflenet
        {{0, 2, 1}, MKLDNNPermuteNode::PermuteImpl(permute_to_021, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat();
        })},  // self attention block
        {{0, 3, 4, 1, 5, 2}, MKLDNNPermuteNode::PermuteImpl(permute_to_034152, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat();
        })},  // learning-to-see-in-the-dark-sony
        {{0, 1, 3, 2}, MKLDNNPermuteNode::PermuteImpl(permute_to_0132, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return true;
        })},
        {{0, 3, 1, 4, 2}, MKLDNNPermuteNode::PermuteImpl(permute_to_03142, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat();
        })},
        {{1, 2, 0, 3}, MKLDNNPermuteNode::PermuteImpl(permute_to_1203, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat() && MB == srcMemPtr->GetDims()[0];
        })},
        {{0, 2, 1, 3, 4}, MKLDNNPermuteNode::PermuteImpl(permute_to_02134, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat();
        })},
        {{0, 2, 4, 3, 1}, MKLDNNPermuteNode::PermuteImpl(permute_to_02431, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat();
        })},
        {{0, 4, 2, 3, 1}, MKLDNNPermuteNode::PermuteImpl(permute_to_04231, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat();
        })},
        {{0, 3, 1, 2}, MKLDNNPermuteNode::PermuteImpl(permute_to_0312, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat();
        })},
        {{1, 0, 2}, MKLDNNPermuteNode::PermuteImpl(permute_to_102, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat() && MB == srcMemPtr->GetDims()[0];
        })},
        {{0, 2, 3, 4, 1}, MKLDNNPermuteNode::PermuteImpl(permute_to_02341, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat();
        })},
        {{0, 4, 1, 2, 3}, MKLDNNPermuteNode::PermuteImpl(permute_to_04123, [](int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr) {
            return srcMemPtr->GetDesc().isPlainFormat();
        })},
};

void MKLDNNPermuteNode::execute(mkldnn::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    int MB = batchToProcess();

    if (prec == Precision::FP32 && !getParentEdgeAt(0)->getMemory().GetDesc().isTailCFormat()) {
        for (const auto &impl : OptimizedCases) {
            if (impl.first == order && impl.second.isValidParams(MB, srcMemPtr, dstMemPtr)) {
                impl.second.execute(MB, srcMemPtr, dstMemPtr);
                return;
            }
        }
    }

    const uint8_t* srcData = reinterpret_cast<const uint8_t*>(srcMemPtr->GetPtr());
    uint8_t* dstData = reinterpret_cast<uint8_t*>(dstMemPtr->GetPtr());
    permuteKernel->execute(srcData, dstData, MB);
}

bool MKLDNNPermuteNode::created() const {
    return getType() == Permute;
}
REG_MKLDNN_PRIM_FOR(MKLDNNPermuteNode, Permute);

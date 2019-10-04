// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <functional>
#include <vpu/utils/ie_helpers.hpp>

#include <precision_utils.h>
#include <details/ie_exception.hpp>
#include <blob_transform.hpp>
#include <blob_factory.hpp>
#include <ie_profiling.hpp>

#include <vpu/utils/extra.hpp>
#include <vpu/utils/numeric.hpp>

namespace vpu {

InferenceEngine::Layout deviceLayout(InferenceEngine::Layout const& layout,
                                       vpu::LayoutPreference const& layoutPreference) {
    using namespace InferenceEngine;
    auto ChannelMajor = vpu::LayoutPreference::ChannelMajor;
    auto ChannelMinor = vpu::LayoutPreference::ChannelMinor;

    if (layoutPreference == ChannelMajor) {
        if (layout == NHWC)
            return NCHW;
        if (layout == NDHWC)
            return NCDHW;
    }

    if (layoutPreference == ChannelMinor) {
        if (layout == NCHW)
            return NHWC;
        if (layout == NCDHW)
            return NDHWC;
    }

    return layout;
}

ie::Blob::Ptr getBlobFP16(const ie::Blob::Ptr& in) {
    IE_PROFILING_AUTO_SCOPE(getBlobFP16);

    auto inDesc = in->getTensorDesc();

    auto precision = inDesc.getPrecision();

    if (precision == ie::Precision::FP16)
        return in;

    if (precision != ie::Precision::FP32) {
        VPU_THROW_EXCEPTION << "Unsupported precision " << precision.name();
    }

    // TODO: TensorDesc doesn't update internal BlockingDesc and strides when setLayout is called
    ie::TensorDesc outDesc(inDesc.getPrecision(), inDesc.getDims(), inDesc.getLayout());
    auto out = make_blob_with_precision(outDesc);
    out->allocate();

    ie::PrecisionUtils::f32tof16Arrays(out->buffer().as<fp16_t*>(), in->cbuffer().as<float*>(), in->size());

    return out;
}

ie::Blob::Ptr copyBlob(const ie::Blob::Ptr& in, ie::Layout outLayout) {
    auto inDesc = in->getTensorDesc();

    // TODO: TensorDesc doesn't update internal BlockingDesc and strides when setLayout is called
    ie::TensorDesc outDesc(inDesc.getPrecision(), inDesc.getDims(), outLayout);
    auto out = make_blob_with_precision(outDesc);
    out->allocate();

    copyBlob(in, out);

    return out;
}

void copyBlob(const ie::Blob::Ptr& in, const ie::Blob::Ptr& out) {
    const auto inLayout = in->getTensorDesc().getLayout();
    const auto outLayout = out->getTensorDesc().getLayout();

    const auto& inDims = in->getTensorDesc().getDims();
    const auto& outDims = out->getTensorDesc().getDims();

    IE_ASSERT(inDims == outDims);

    if (inLayout != outLayout) {
        if (outDims.size() == 4) {
            IE_ASSERT(inLayout == ie::Layout::NCHW || inLayout == ie::Layout::NHWC);
            IE_ASSERT(outLayout == ie::Layout::NCHW || outLayout == ie::Layout::NHWC);

            if (outDims[1] != 1 && (outDims[2] != 1 || outDims[3] != 1)) {
                ie::blob_copy(in, out);
                return;
            }
        }

        if (outDims.size() == 5) {
            IE_ASSERT(inLayout == ie::Layout::NCDHW || inLayout == ie::Layout::NDHWC);
            IE_ASSERT(outLayout == ie::Layout::NCDHW || outLayout == ie::Layout::NDHWC);

            if (outDims[1] != 1 && (outDims[2] != 1 || outDims[3] != 1 || outDims[4] != 1)) {
                ie::blob_copy(in, out);
                return;
            }
        }
    }

    auto inPtr = in->cbuffer().as<uint8_t *>();
    IE_ASSERT(inPtr != nullptr);

    auto outPtr = out->cbuffer().as<uint8_t *>();
    IE_ASSERT(outPtr != nullptr);

    std::copy_n(
        in->cbuffer().as<uint8_t *>(),
        in->byteSize(),
        out->buffer().as<uint8_t *>());
}

}  // namespace vpu

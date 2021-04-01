// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/extra.hpp>
#include <vpu/utils/error.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/utils/ie_itt.hpp>

#include <precision_utils.h>
#include <details/ie_exception.hpp>
#include <blob_transform.hpp>
#include <blob_factory.hpp>

#include <vector>
#include <functional>
#include <algorithm>

namespace vpu {

InferenceEngine::Layout deviceLayout(InferenceEngine::Layout const& layout,
                                     LayoutPreference const& layoutPreference) {
    using namespace InferenceEngine;

    if (layoutPreference == LayoutPreference::ChannelMajor) {
        if (layout == NHWC)
            return NCHW;
        if (layout == NDHWC)
            return NCDHW;
    }

    if (layoutPreference == LayoutPreference::ChannelMinor) {
        if (layout == NCHW)
            return NHWC;
        if (layout == NCDHW)
            return NDHWC;
    }

    return layout;
}

ie::Blob::Ptr convertBlobFP32toFP16(const ie::Blob::CPtr& in) {
    OV_ITT_SCOPED_TASK(itt::domains::VPU, "convertBlobFP32toFP16");

    auto inDesc = in->getTensorDesc();

    auto precision = inDesc.getPrecision();

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

ie::Blob::Ptr copyBlob(const ie::Blob::Ptr& original) {
    auto copied = make_blob_with_precision(original->getTensorDesc());
    copied->allocate();
    copyBlob(original, copied);
    return copied;
}

ie::Blob::Ptr copyBlob(const ie::Blob::Ptr& in, ie::Layout outLayout, void* ptr) {
    auto inDesc = in->getTensorDesc();

    // TODO: TensorDesc doesn't update internal BlockingDesc and strides when setLayout is called
    ie::TensorDesc outDesc(inDesc.getPrecision(), inDesc.getDims(), outLayout);
    auto out = make_blob_with_precision(outDesc, ptr);
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

void printTo(DotLabel& lbl, const ie::DataPtr& ieData) {
    VPU_INTERNAL_CHECK(ieData != nullptr, "NULL pointer");

    const auto& desc = ieData->getTensorDesc();

    DotLabel subLbl(lbl);
    subLbl.appendPair("name", ieData->getName());
    subLbl.appendPair("precision", desc.getPrecision().name());
    subLbl.appendPair("dims", desc.getDims());
    subLbl.appendPair("layout", desc.getLayout());
}

void printTo(DotLabel& lbl, const ie::Blob::Ptr& ieBlob) {
    VPU_INTERNAL_CHECK(ieBlob != nullptr, "NULL pointer");

    const auto& desc = ieBlob->getTensorDesc();

    DotLabel subLbl(lbl);
    subLbl.appendPair("precision", desc.getPrecision().name());
    subLbl.appendPair("dims", desc.getDims());
    subLbl.appendPair("layout", desc.getLayout());

    if (desc.getPrecision() == ie::Precision::FP32) {
        auto contentPtr = ieBlob->cbuffer().as<const uint8_t*>();
        auto count = ieBlob->size();

        SmallVector<uint8_t, 8> temp(
            contentPtr,
            contentPtr + std::min<size_t>(count, 8));

        subLbl.appendPair("content", temp);
    } else if (desc.getPrecision() == ie::Precision::FP16) {
        auto contentPtr = ieBlob->cbuffer().as<const fp16_t*>();
        auto count = ieBlob->size();

        auto temp = SmallVector<float, 8>(std::min<size_t>(count, 8));
        ie::PrecisionUtils::f16tof32Arrays(temp.data(), contentPtr, temp.size());

        lbl.appendPair("content", temp);
    }
}

void printTo(DotLabel& lbl, const ie::CNNLayerPtr& ieLayer) {
    VPU_INTERNAL_CHECK(ieLayer != nullptr, "NULL pointer");

    DotLabel subLbl(lbl);
    subLbl.appendPair("name", ieLayer->name);
    subLbl.appendPair("type", ieLayer->type);
    subLbl.appendPair("precision", ieLayer->precision.name());
}

}  // namespace vpu

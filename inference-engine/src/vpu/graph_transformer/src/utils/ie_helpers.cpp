// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/ie_helpers.hpp>

#include <precision_utils.h>
#include <details/ie_exception.hpp>
#include <blob_transform.hpp>
#include <blob_factory.hpp>

#include <vpu/utils/extra.hpp>
#include <vpu/utils/numeric.hpp>
#include <vpu/compile_env.hpp>

namespace vpu {

ie::Blob::Ptr getBlobFP16(const ie::Blob::Ptr& in) {
    VPU_PROFILE(getBlobFP16);

    const auto& env = CompileEnv::get();

    auto inDesc = in->getTensorDesc();

    auto precision = inDesc.getPrecision();

    if (precision == ie::Precision::FP16)
        return in;

    if (precision != ie::Precision::FP32 || !env.config.allowFP32Models) {
        VPU_THROW_EXCEPTION << "Unsupported precision " << precision.name();
    }

    // TODO: TensorDesc doesn't update internal BlockingDesc and strides when setLayout is called
    ie::TensorDesc outDesc(inDesc.getPrecision(), inDesc.getDims(), inDesc.getLayout());
    auto out = make_blob_with_precision(outDesc);
    out->allocate();

    ie::PrecisionUtils::f32tof16Arrays(out->buffer().as<fp16_t*>(), in->cbuffer().as<float*>(), in->size());

    return out;
}

ie::Blob::Ptr copyBlob(const ie::Blob::Ptr& in) {
    return copyBlob(in, in->getTensorDesc().getLayout());
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
    auto inLayout = in->getTensorDesc().getLayout();
    auto outLayout = out->getTensorDesc().getLayout();

    if (inLayout != outLayout) {
        IE_ASSERT(inLayout == ie::Layout::NCHW || inLayout == ie::Layout::NHWC);
        IE_ASSERT(outLayout == ie::Layout::NCHW || outLayout == ie::Layout::NHWC);

        const auto& dims = out->getTensorDesc().getDims();

        if ((dims[0] != 1 || dims[1] != 1) && (dims[2] != 1 || dims[3] != 1)) {
            ie::blob_copy(in, out);
            return;
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

// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/deconvolution_contents.hpp>

#include <vpu/utils/profiling.hpp>
#include <vpu/middleend/sw/utility.hpp>

#include <ie_parallel.hpp>

namespace vpu {

//
// DeconvolutionToConvolutionContent
//

DeconvolutionToConvolutionContent::DeconvolutionToConvolutionContent(
        const DataContent::Ptr& origContent, const DataDesc& desc) :
        _origContent(origContent), _desc(desc) {
}

size_t DeconvolutionToConvolutionContent::byteSize() const {
    return checked_cast<size_t>(_desc.totalDimSize()) *
           checked_cast<size_t>(_desc.elemSize());
}

void DeconvolutionToConvolutionContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(DeconvolutionToConvolutionContent);

    IE_ASSERT(_desc.type() == DataType::FP16);

    deconv_to_conv(_origContent->get<fp16_t>(), static_cast<fp16_t*>(tempBuf), _desc);
}

//
// DepthDeconvolutionCHWWeightsContent
//

void depthDeconvolutionRelayoutCHW(
        const fp16_t* src, int src_size,
        fp16_t* dst, int dst_size,
        int KX, int KY,
        int channels) {
    ie::parallel_for3d(channels, KY, KX, [=](int c, int ky, int kx) {
        int iidx = c * KX * KY + ky * KX + kx;
        IE_ASSERT(iidx >= 0 && iidx < src_size);

        int inv_kx = KX - kx - 1;
        int inv_ky = KY - ky - 1;
        int oidx = c * KX * KY + inv_ky * KX + inv_kx;
        IE_ASSERT(oidx >= 0 && oidx < dst_size);

        dst[oidx] = src[iidx];
    });
}

DepthDeconvolutionCHWWeightsContent::DepthDeconvolutionCHWWeightsContent(
        const DataContent::Ptr& origContent,
        int KX, int KY, int channels) :
        _origContent(origContent),
        _KX(KX), _KY(KY), _channels(channels) {}

void DepthDeconvolutionCHWWeightsContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(DepthDeconvolutionCHWWeightsContent);
    depthDeconvolutionRelayoutCHW(
            _origContent->get<fp16_t>(), _origContent->byteSize() / sizeof(fp16_t),
            static_cast<fp16_t*>(tempBuf), _origContent->byteSize() / sizeof(fp16_t),
            _KX, _KY, _channels);
}

size_t DepthDeconvolutionCHWWeightsContent::byteSize() const {
    return _origContent->byteSize();
}

//
// DepthDeconvolutionHWCWeightsContent
//

void depthDeconvolutionRelayoutHWC(
        const fp16_t* src, int src_size,
        fp16_t* dst, int dst_size,
        int KX, int KY,
        int channels) {
    ie::parallel_for3d(channels, KY, KX, [=](int c, int ky, int kx) {
        int iidx = c * KX * KY + ky * KX + kx;
        IE_ASSERT(iidx < src_size);

        int inv_kx = KX - kx - 1;
        int inv_ky = KY - ky - 1;
        int oidx = inv_ky * KX * channels + inv_kx * channels + c;
        IE_ASSERT(oidx < dst_size);

        dst[oidx] = src[iidx];
    });
}

DepthDeconvolutionHWCWeightsContent::DepthDeconvolutionHWCWeightsContent(
        const DataContent::Ptr& origContent,
        int KX, int KY, int channels) :
        _origContent(origContent),
        _KX(KX), _KY(KY), _channels(channels) {
}

void DepthDeconvolutionHWCWeightsContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(DepthDeconvolutionHWCWeightsContent);
    depthDeconvolutionRelayoutHWC(
            _origContent->get<fp16_t>(), _origContent->byteSize() / sizeof(fp16_t),
            static_cast<fp16_t*>(tempBuf), _origContent->byteSize() / sizeof(fp16_t),
            _KX, _KY, _channels);
}

size_t DepthDeconvolutionHWCWeightsContent::byteSize() const {
    return _origContent->byteSize();
}

//
// DeconvolutionWeightsContent
//

void deconvolutionRelayout(
        const fp16_t* src, int src_size,
        fp16_t* dst, int dst_size,
        int KX, int KY,
        int IC, int OC) {
    ie::parallel_for4d(OC, IC, KY, KX, [=](int oc, int ic, int ky, int kx) {
        int iidx = ic * OC * KY * KX
                   + oc * KY * KX
                   + ky * KX
                   + kx;
        IE_ASSERT(iidx >= 0 && iidx < src_size);

        int inv_kx = KX - kx - 1;
        int inv_ky = KY - ky - 1;
        int oidx = oc * IC * KY * KX
                   + ic * KY * KX
                   + inv_ky * KX
                   + inv_kx;
        IE_ASSERT(oidx >=  0 && oidx < dst_size);

        dst[oidx] = src[iidx];
    });
}

DeconvolutionWeightsContent::DeconvolutionWeightsContent(
        const DataContent::Ptr& origContent,
        DataDesc desc,
        int KX, int KY,
        int IC, int OC) :
        _origContent(origContent), _desc(desc),
        _intermBuf(_desc.totalDimSize()),
        _KX(KX), _KY(KY),
        _IC(IC), _OC(OC) {
}

size_t DeconvolutionWeightsContent::byteSize() const {
    return _desc.totalDimSize() * sizeof(fp16_t);
}

void DeconvolutionWeightsContent::fillTempBuf(void* tempBuf) const {
    VPU_PROFILE(DeconvolutionWeightsContent);

    auto dstPtr = static_cast<fp16_t*>(tempBuf);

    deconvolutionRelayout(
            _origContent->get<fp16_t>(), _desc.totalDimSize(),
            _intermBuf.data(), _desc.totalDimSize(),
            _KX, _KY,
            _IC, _OC);

    kchw_to_hwkc(_intermBuf.data(), dstPtr, _desc);
}

} // namespace vpu

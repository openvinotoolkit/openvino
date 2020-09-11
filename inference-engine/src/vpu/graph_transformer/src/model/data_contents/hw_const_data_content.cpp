// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/hw_const_data_content.hpp>

#include <vpu/utils/profiling.hpp>

#include <ie_parallel.hpp>

namespace vpu {

HwConstData::HwConstData(
        const DataContent::Ptr& origContent,
        const DataDesc& origDesc,
        const DataDesc& resDesc,
        const std::map<Dim, Slice> dimSlices) :
        _origContent(origContent),
        _origDesc(origDesc),
        _resDesc(resDesc),
        _dimSlices(dimSlices) {}

size_t HwConstData::byteSize() const {
    return checked_cast<size_t>(_resDesc.totalDimSize()) *
           checked_cast<size_t>(_resDesc.elemSize());
}

void HwConstData::fillTempBuf(void* outBuf) const {
    VPU_PROFILE(HwConstData);

    VPU_THROW_UNLESS(
        _resDesc.type() == DataType::FP16,
        "Constant data has {} data type while only {} is supported",
        _resDesc.type(), DataType::FP16);

    const auto srcData = _origContent->get<fp16_t>();
    auto dstData = static_cast<fp16_t*>(outBuf);

    VPU_THROW_UNLESS(srcData != nullptr,
        "Source buffer for constant data has null address");

    auto getDimSlice = [this](const Dim dim) {
        auto it = _dimSlices.find(dim);
        if (it != _dimSlices.end()) {
            return it->second;
        }

        const int startInd = 0;
        const size_t size = _origDesc.dim(dim);

        return Slice(startInd, size);
    };

    if (_origDesc.numDims() == 4) {
        Slice slice = getDimSlice(Dim::N);

        int startOC = slice.start;
        size_t numOC = slice.size;

        const auto IC = _origDesc.dim(Dim::C);
        const auto K = _origDesc.dim(Dim::H);
        const auto V = _origDesc.dim(Dim::W);

        const auto kernelStride     = V;
        const auto inChannelStride  = K * kernelStride;
        const auto outerStride      = IC * inChannelStride;

        ie::parallel_for(numOC, [=](int oc) {
            const auto ocSlice = oc;
            oc += startOC;

            const auto ocInner = oc % V;
            const auto ocOuter = oc / V;
            const auto ocSliceInner = ocSlice % V;
            const auto ocSliceOuter = ocSlice / V;

            const auto ocSrc = ocInner + ocOuter * outerStride;
            const auto ocDst = ocSliceInner + ocSliceOuter * outerStride;

            for (int ic = 0; ic < IC; ++ic)
                for (int k = 0; k < K; ++k) {
                    const auto srcInd = ocSrc +
                                        k * kernelStride +
                                        ic * inChannelStride;
                    const auto dstInd = ocDst +
                                        k * kernelStride +
                                        ic * inChannelStride;

                    dstData[dstInd] = srcData[srcInd];
                }
        });
    } else if (_origDesc.numDims() == 1) {
        Slice slice = getDimSlice(Dim::C);

        std::copy(srcData + slice.start, srcData + slice.start + slice.size, dstData);
    } else {
        THROW_IE_EXCEPTION << "Invalid number of dimensions " << _origDesc.numDims();
    }
}

} // namespace vpu

// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/prelu_blob_content.hpp>

#include <vpu/utils/ie_helpers.hpp>
#include <vpu/utils/profiling.hpp>

#include <ie_parallel.hpp>

namespace vpu {

PReLUBlobContent::PReLUBlobContent(const ie::Blob::CPtr& blob, const DataDesc& desc, int repeat) :
        _blob(blob), _desc(desc), _repeat(repeat) {
    VPU_INTERNAL_CHECK(repeat >= 1,
        "PReLUBlobContent only supports repeat value more than 1, actual is {}", repeat);
}

size_t PReLUBlobContent::byteSize() const {
    return checked_cast<size_t>(_desc.totalDimSize()) *
           checked_cast<size_t>(_desc.elemSize());
}

const void* PReLUBlobContent::getRaw() const {
    if (_blobFp16 == nullptr) {
        _blobFp16 = _blob->getTensorDesc().getPrecision() == ie::Precision::FP16 ?
                    _blob : convertBlobFP32toFP16(_blob);
    }

    if (_repeat == 1) {
        return _blobFp16->cbuffer();
    }

    if (_tempFp16.empty()) {
        VPU_PROFILE(PReLUBlobContent);

        IE_ASSERT(_desc.totalDimSize() % _repeat == 0);

        auto origNumElems = _desc.totalDimSize() / _repeat;
        IE_ASSERT(checked_cast<size_t>(origNumElems) <= _blobFp16->size());

        auto origPtr = _blobFp16->cbuffer().as<const fp16_t*>();
        IE_ASSERT(origPtr != nullptr);

        _tempFp16.resize(checked_cast<size_t>(_desc.totalDimSize()));

        ie::parallel_for(_repeat, [this, origPtr, origNumElems](int i) {
            std::copy_n(origPtr, origNumElems, _tempFp16.data() + i * origNumElems);
        });
    }

    return _tempFp16.data();
}

} // namespace vpu

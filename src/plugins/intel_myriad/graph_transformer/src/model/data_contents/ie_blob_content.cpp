// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/ie_blob_content.hpp>

#include <vpu/utils/ie_helpers.hpp>

namespace vpu {

IeBlobContent::IeBlobContent(const ie::Blob::CPtr& blob, DataType resultDataType) : _blob(blob), _resultDataType(resultDataType) {
    VPU_THROW_UNLESS(_resultDataType == DataType::FP16 || _resultDataType == DataType::S32,
                     "IeBlobContent creation error: {} result type is unsupported, only {} and {} are supported",
                     _resultDataType, DataType::FP16, DataType::S32);
}

size_t IeBlobContent::byteSize() const {
    // Result can be converted into type with another size
    const auto elementSize = _resultDataType == DataType::FP16 ? sizeof(fp16_t) : sizeof(int32_t);
    return elementSize * _blob->size();
}

const void* IeBlobContent::getRaw() const {
    if (_resultDataType == DataType::FP16) {
        if (_blobFp16 == nullptr) {
            _blobFp16 = _blob->getTensorDesc().getPrecision() == ie::Precision::FP16 ?
                        _blob : convertBlobFP32toFP16(_blob);
        }
        return _blobFp16->cbuffer();
    } else { // S32
        return _blob->cbuffer();
    }
}

DataContent::Ptr ieBlobContent(const ie::Blob::CPtr& blob, DataType resultDataType) {
    return std::make_shared<IeBlobContent>(blob, resultDataType);
}

} // namespace vpu

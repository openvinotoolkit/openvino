// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/data_content.hpp>

#include <vpu/model/data.hpp>

namespace vpu {

class IeBlobContent final : public DataContent {
public:
    IeBlobContent(const ie::Blob::CPtr& blob, DataType resultDataType);

    size_t byteSize() const override;

protected:
    const void* getRaw() const override;

private:
    DataType _resultDataType;
    mutable ie::Blob::CPtr _blob;
    mutable ie::Blob::CPtr _blobFp16;
};

DataContent::Ptr ieBlobContent(const ie::Blob::CPtr& blob, DataType resultPrecision = DataType::FP16);

} // namespace vpu

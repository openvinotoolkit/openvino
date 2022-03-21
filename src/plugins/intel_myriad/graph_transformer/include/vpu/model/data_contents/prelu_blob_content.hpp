// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vpu/model/data_contents/data_content.hpp>
#include <vpu/model/data_desc.hpp>

#include <ie_blob.h>

namespace vpu {

class PReLUBlobContent final : public DataContent {
public:
    PReLUBlobContent(const InferenceEngine::Blob::CPtr& blob, const DataDesc& desc, int repeat);

    size_t byteSize() const override;

protected:
    const void* getRaw() const override;

private:
    InferenceEngine::Blob::CPtr _blob;
    int _repeat = 0;
    DataDesc _desc;

    mutable InferenceEngine::Blob::CPtr _blobFp16;
    mutable std::vector<fp16_t> _tempFp16;
};

} // namespace vpu

// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/mtcnn_blob_content.hpp>

namespace vpu {

MTCNNBlobContent::MTCNNBlobContent(std::vector<char> blob) : _blob(std::move(blob)) {
    IE_ASSERT(!_blob.empty());
}

size_t MTCNNBlobContent::byteSize() const {
    return _blob.size();
}

const void* MTCNNBlobContent::getRaw() const {
    return _blob.data();
}

} // namespace vpu

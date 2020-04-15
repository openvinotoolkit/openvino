// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/model/data_contents/kernel_binary_content.hpp>

#include <string>

namespace vpu {

KernelBinaryContent::KernelBinaryContent(const std::string& blob) : _blob(blob) {
    IE_ASSERT(!_blob.empty());
}

size_t KernelBinaryContent::byteSize() const {
    return _blob.size();
}

const void* KernelBinaryContent::getRaw() const {
    return _blob.data();
}

} // namespace vpu

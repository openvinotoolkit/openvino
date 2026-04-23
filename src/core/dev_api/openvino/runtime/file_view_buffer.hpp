// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>

#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {
/// \brief FileViewBuffer is an AlignedBuffer which provides a view on a file.
class OPENVINO_API FileViewBuffer : public AlignedBuffer {
public:
    FileViewBuffer(std::filesystem::path file_path, size_t offset, size_t byte_size);
    ~FileViewBuffer() override = default;

    void load() const override;
    void release() const override;

private:
    std::filesystem::path m_file_path;
    const size_t m_lazy_offset;
    const size_t m_lazy_byte_size;
    mutable std::vector<char> m_lazy_buffer;
};
}  // namespace ov

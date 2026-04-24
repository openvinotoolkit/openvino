// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <filesystem>

#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {
/** \brief LazyBuffer is lazy loaded AlignedBuffer which provides a view on a file w/o memory mapping. */
class OPENVINO_API LazyBuffer : public AlignedBuffer {
public:
    /**
     * \brief Constructs a LazyBuffer which provides a view on a file. The file content is loaded to memory when
     * get_ptr() is called for the first time. The file content is loaded at aligned addresses, so the actual allocated
     * memory may be larger than the requested byte size.     *
     * \param file_path Path to the file to load
     * \param offset Offset in the file to start the view
     * \param byte_size Size of the view in bytes
     * \param alignment Alignment for the loaded buffer.     *
     * \throws AssertFailure if the file does not exist or the file size is smaller than the requested view.
     */
    LazyBuffer(std::filesystem::path file_path,
               size_t offset,
               size_t byte_size,
               size_t alignment = AlignedBuffer::s_default_alignment);

    ~LazyBuffer() override = default;

    /**
     * \brief Loads the file content to memory if it is not loaded yet. The content is loaded at aligned addresses, so
     * the actual allocated memory may be larger than the requested byte size.
     * \throws AssertFailure if the file cannot be opened or read. In this case, the buffer remains unloaded.
     */
    void load() const override;

private:
    std::filesystem::path m_file_path;
    const size_t m_offset;
    const size_t m_alignment;
    mutable std::vector<char> m_lazy_buffer;
};
}  // namespace ov

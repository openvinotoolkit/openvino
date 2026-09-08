// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>

#include "openvino/core/deprecated.hpp"
#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {

/** \brief LazyBuffer is lazy loaded AlignedBuffer which provides a view on a file w/o memory mapping. */
class OPENVINO_API OPENVINO_DEPRECATED("LazyBuffer is deprecated and will be removed in 2026.5 release") LazyBuffer
    : public AlignedBuffer {
public:
    /**
     * @brief Constructs a LazyBuffer which provides a view on a file. The file content is loaded to memory when
     * get_ptr() is called for the first time after object creation. The file content is loaded at aligned addresses,
     * so the actual allocated memory may be larger than the requested byte size.
     * @param file_path Path to the file to load
     * @param offset Offset in the file to start the view
     * @param byte_size Size of the view in bytes
     * @throws AssertFailure if the file does not exist or the file size is smaller than the requested view.
     */
    LazyBuffer(std::filesystem::path file_path, size_t offset, size_t byte_size);

    LazyBuffer(LazyBuffer&&) noexcept;
    LazyBuffer& operator=(LazyBuffer&&) noexcept;
    ~LazyBuffer() override;

    LazyBuffer(const LazyBuffer&) = delete;
    LazyBuffer& operator=(const LazyBuffer&) = delete;

    /**
     * @brief Gets aligned pointer to reserved buffer without loading data into it.
     */
    void* get_reserved_ptr() const noexcept {
        return m_aligned_buffer;
    }

    /**
     * @brief No-op by design: once loaded, LazyBuffer keeps file content resident until destroyed.
     */
    void hint_evict() noexcept override;

    /**
     * @brief Loads the file content if it is not loaded yet. The content is loaded at aligned addresses,
     * so the actual allocated memory may be larger than the requested byte size.
     * @throws AssertFailure if the file cannot be opened or read. In this case, the buffer remains unloaded.
     */
    void hint_prefetch() const override;

protected:
    void hint_evict(size_t offset, size_t size) noexcept override;

private:
    std::filesystem::path m_file_path;
    size_t m_offset{0};

    mutable std::atomic<bool> m_loaded{false};
    mutable std::mutex m_loading;
};
}  // namespace ov

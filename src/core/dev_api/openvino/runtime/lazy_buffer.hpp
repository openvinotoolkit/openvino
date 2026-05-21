// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>

#include "openvino/runtime/aligned_buffer.hpp"

namespace ov {
namespace util {
class ReservableBuffer;
}

/** \brief LazyBuffer is lazy loaded AlignedBuffer which provides a view on a file w/o memory mapping. */
class OPENVINO_API LazyBuffer : public AlignedBuffer {
public:
    /**
     * \brief Constructs a LazyBuffer which provides a view on a file. The file content is loaded to memory when
     * get_ptr() is called for the first time after object creation or after hint_evict() is called. The file content is
     * loaded at aligned addresses, so the actual allocated memory may be larger than the requested byte size.
     * \param file_path Path to the file to load
     * \param offset Offset in the file to start the view
     * \param byte_size Size of the view in bytes
     * \throws AssertFailure if the file does not exist or the file size is smaller than the requested view.
     */
    LazyBuffer(std::filesystem::path file_path, size_t offset, size_t byte_size);

    LazyBuffer(LazyBuffer&&) noexcept;
    LazyBuffer& operator=(LazyBuffer&&) noexcept;
    ~LazyBuffer() override;

    LazyBuffer(const LazyBuffer&) = delete;
    LazyBuffer& operator=(const LazyBuffer&) = delete;

    /**
     * \brief Gets aligned pointer to reserved buffer without loading data into it.
     */
    void* get_reserved_ptr() const noexcept {
        return m_aligned_buffer;
    }

    /**
     * \brief Evicts the buffer from memory. After this call, next call to hint_prefetch() will load the file content
     * again.
     */
    void hint_evict() noexcept override;

    /**
     * \brief Loads the file content if it is not loaded yet. The content is loaded at aligned addresses,
     * so the actual allocated memory may be larger than the requested byte size.
     * \throws AssertFailure if the file cannot be opened or read. In this case, the buffer remains unloaded.
     */
    void hint_prefetch() const override;

protected:
    void hint_evict(size_t offset, size_t size) noexcept override;

private:
    std::filesystem::path m_file_path;
    size_t m_offset{0};

    mutable std::atomic<bool> m_loaded{false};
    mutable std::mutex m_loading;

    std::unique_ptr<util::ReservableBuffer> m_buffer;
};
}  // namespace ov

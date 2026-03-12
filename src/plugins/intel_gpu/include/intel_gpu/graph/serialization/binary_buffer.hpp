// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "buffer.hpp"
#include "helpers.hpp"
#include "bind.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "intel_gpu/runtime/itt.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#endif

namespace cldnn {
struct memory;

class BinaryOutputBuffer : public OutputBuffer<BinaryOutputBuffer> {
public:
    BinaryOutputBuffer(std::ostream& stream)
    : OutputBuffer<BinaryOutputBuffer>(this), stream(stream), _impl_params(nullptr), _strm(nullptr) {}

    virtual ~BinaryOutputBuffer() = default;

    virtual void write(void const* data, std::streamsize size) {
        auto const written_size = stream.rdbuf()->sputn(reinterpret_cast<const char*>(data), size);
        OPENVINO_ASSERT(written_size == size,
                        "[GPU] Failed to write " + std::to_string(size) + " bytes to stream! Wrote " +
                            std::to_string(written_size));
    }

    virtual void flush() {}

    void setKernelImplParams(void* impl_params) { _impl_params = impl_params; }
    void* getKernelImplParams() const { return _impl_params; }
    void set_stream(void* strm) { _strm = strm; }
    void* get_stream() const { return _strm; }

private:
    std::ostream& stream;
    void* _impl_params;
    void* _strm;
};

class BinaryInputBuffer : public InputBuffer<BinaryInputBuffer> {
public:
    BinaryInputBuffer(std::istream& stream, engine& engine)
    : InputBuffer<BinaryInputBuffer>(this, engine), _stream(stream), _impl_params(nullptr) {}

    virtual ~BinaryInputBuffer() = default;

    virtual void read(void* const data, std::streamsize size) {
        // For large reads backed by mmap (SharedStreamBuffer), use parallel memcpy
        // to trigger concurrent page faults and saturate NVMe bandwidth.
        constexpr std::streamsize PARALLEL_THRESHOLD = 4 * 1024 * 1024;  // 4MB
        if (size >= PARALLEL_THRESHOLD) {
            if (auto* ssb = dynamic_cast<ov::SharedStreamBuffer*>(_stream.rdbuf())) {
                const auto cur_pos = _stream.tellg();
                OPENVINO_ASSERT(cur_pos != std::streampos(-1),
                    "[GPU] Failed to get stream position for parallel memcpy");
                auto* dst = reinterpret_cast<char*>(data);
                const char* src = ssb->get_ptr() + static_cast<std::streamoff>(cur_pos);
                const size_t total = static_cast<size_t>(size);
                // openvino::itt::ScopedTask<ov::intel_gpu::itt::domains::intel_gpu_plugin> ittScopedTask(
                //     openvino::itt::handle("BinaryInputBuffer::mmap_read(size=" + std::to_string(total) +
                //                           ",off=" + std::to_string(static_cast<size_t>(cur_pos)) + ")"));
                // std::cout << "[GPU] Using parallel memcpy for large read of " << total << " bytes from stream at offset "
                //           << cur_pos << std::endl;

#ifdef _WIN32
                // On Windows, concurrent memcpy threads on cold mmap pages all serialize
                // on the Section Object VMM lock (one page fault resolved at a time),
                // dropping NVMe queue depth to 1 and halving effective bandwidth.
                // PrefetchVirtualMemory issues all page reads concurrently (high queue depth),
                // populating the page cache before parallel memcpy so memcpy only sees warm pages.
                WIN32_MEMORY_RANGE_ENTRY prefetch_range{const_cast<char*>(src), total};
                // const auto t_prefetch_start = std::chrono::steady_clock::now();
                PrefetchVirtualMemory(GetCurrentProcess(), 1, &prefetch_range, 0);
                // const auto t_prefetch_end = std::chrono::steady_clock::now();
                // std::cout << "[GPU] PrefetchVirtualMemory took "
                //           << std::chrono::duration_cast<std::chrono::milliseconds>(t_prefetch_end - t_prefetch_start).count()
                //           << " ms for " << (total / 1024 / 1024) << " MB" << std::endl;
#endif

                constexpr size_t CHUNK_SIZE = 2 * 1024 * 1024;  // 2MB chunks
                const size_t num_chunks = (total + CHUNK_SIZE - 1) / CHUNK_SIZE;
                const auto t_memcpy_start = std::chrono::steady_clock::now();
                ov::parallel_for(num_chunks, [&](size_t i) {
                    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "BinaryInputBuffer::mmap_parallel_memcpy");
                    const size_t off = i * CHUNK_SIZE;
                    const size_t bytes = std::min(CHUNK_SIZE, total - off);
                    std::memcpy(dst + off, src + off, bytes);
                });
                // const auto t_memcpy_end = std::chrono::steady_clock::now();
                // const double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(t_memcpy_end - t_memcpy_start).count() / 1000.0;
                // const double throughput_gbps = (total / 1024.0 / 1024.0 / 1024.0) / (elapsed_ms / 1000.0);
                // std::cout << "[GPU] parallel memcpy " << (total / 1024 / 1024) << " MB in "
                //           << elapsed_ms << " ms (" << throughput_gbps << " GB/s)" << std::endl;
#ifdef _WIN32
                // Release the now-consumed source pages from the process working set.
                // Without this, every 96 MB block's prefetched pages stay resident and
                // the cumulative source footprint reaches ~14 GB by end of load — matching
                // the growing destination allocations and starving physical RAM.  Once
                // pressure peaks, Windows evicts pages that were just prefetched and the
                // next memcpy blocks on a re-fault from disk, causing the observed 500–2900 ms
                // stalls.  MEM_RESET marks pages as evictable without a page-file write
                // (they're clean, file-backed).
                // VirtualFree/MEM_RESET requires page-aligned addresses:
                //   start — round DOWN (already-consumed bytes in that page are safe to release)
                //   end   — round DOWN (do not release the last partial page; the next read
                //            starts inside it)
                {
                    constexpr uintptr_t PAGE_MASK = ~static_cast<uintptr_t>(4095u);
                    const char* reset_begin =
                        reinterpret_cast<const char*>(reinterpret_cast<uintptr_t>(src) & PAGE_MASK);
                    const char* reset_end =
                        reinterpret_cast<const char*>((reinterpret_cast<uintptr_t>(src) + total) & PAGE_MASK);
                    if (reset_begin < reset_end) {
                        VirtualFree(const_cast<char*>(reset_begin),
                                    static_cast<SIZE_T>(reset_end - reset_begin),
                                    MEM_RESET);
                    }
                }
#endif
                _stream.seekg(static_cast<std::streamoff>(size), std::ios_base::cur);
                return;
            }
        }
        auto const read_size = _stream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size);
        OPENVINO_ASSERT(read_size == size,
            "[GPU] Failed to read " + std::to_string(size) + " bytes from stream! Read " + std::to_string(read_size));
    }

    void setKernelImplParams(void* impl_params) { _impl_params = impl_params; }
    void* getKernelImplParams() const { return _impl_params; }

private:
    std::istream& _stream;
    void* _impl_params;
};

class EncryptedBinaryOutputBuffer : public BinaryOutputBuffer {
public:
    EncryptedBinaryOutputBuffer(std::ostream& stream, std::function<std::string(const std::string&)> encrypt)
        : BinaryOutputBuffer(stream),
          encrypt(encrypt) {
        OPENVINO_ASSERT(encrypt);
    }

    ~EncryptedBinaryOutputBuffer() override = default;

    void write(void const* data, std::streamsize size) override {
        plaintext_str.append(reinterpret_cast<const char*>(data), size);
    }

    void flush() override {
        auto encrypted_str = encrypt(plaintext_str);
        size_t bytes = encrypted_str.size();
        BinaryOutputBuffer::write(make_data(&bytes, sizeof(bytes)).data, sizeof(bytes));
        BinaryOutputBuffer::write(make_data(encrypted_str.c_str(), encrypted_str.size()).data, encrypted_str.size());
    }

private:
    std::string
        plaintext_str;  // Not using stringstream here because passing to encrypt() would produce an additional copy.
    std::function<std::string(const std::string&)> encrypt;
};

class EncryptedBinaryInputBuffer : public BinaryInputBuffer {
public:
    EncryptedBinaryInputBuffer(std::istream& stream,
                               engine& engine,
                               std::function<std::string(const std::string&)> decrypt)
        : BinaryInputBuffer(stream, engine),
          decrypt(decrypt) {
        OPENVINO_ASSERT(decrypt);

        size_t bytes;
        BinaryInputBuffer::read(make_data(&bytes, sizeof(bytes)).data, sizeof(bytes));

        // Not reading directly to plaintext_stream because decrypt(plaintext_stream.str()) would create an additional
        // copy.
        std::string str(bytes, 0);
        BinaryInputBuffer::read(
            make_data(const_cast<void*>(reinterpret_cast<const void*>(str.c_str())), str.size()).data,
            str.size());
        plaintext_stream.str(decrypt(str));
    }

    ~EncryptedBinaryInputBuffer() override = default;

    void read(void* const data, std::streamsize size) override {
        auto const read_size = plaintext_stream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size);
        OPENVINO_ASSERT(
            read_size == size,
            "[GPU] Failed to read " + std::to_string(size) + " bytes from stream! Read " + std::to_string(read_size));
    }

private:
    std::stringstream plaintext_stream;
    std::function<std::string(const std::string&)> decrypt;
};

template <typename T>
class Serializer<BinaryOutputBuffer, T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
    static void save(BinaryOutputBuffer& buffer, const T& object) {
        buffer.write(std::addressof(object), sizeof(object));
    }
};

template <typename T>
class Serializer<BinaryInputBuffer, T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
    static void load(BinaryInputBuffer& buffer, T& object) {
        buffer.read(std::addressof(object), sizeof(object));
    }
};

template <typename T>
class Serializer<BinaryOutputBuffer, Data<T>> {
public:
    static void save(BinaryOutputBuffer& buffer, const Data<T>& bin_data) {
        buffer.write(bin_data.data, static_cast<std::streamsize>(bin_data.number_of_bytes));
    }
};

template <typename T>
class Serializer<BinaryInputBuffer, Data<T>> {
public:
    static void load(BinaryInputBuffer& buffer, Data<T>& bin_data) {
        buffer.read(bin_data.data, static_cast<std::streamsize>(bin_data.number_of_bytes));
    }
};

}  // namespace cldnn

#define ASSIGN_TYPE_NAME(cls_name) \
            namespace cldnn {                            \
            }

#define BIND_BINARY_BUFFER_WITH_TYPE(cls_name) \
            namespace cldnn {                            \
            BIND_TO_BUFFER(BinaryOutputBuffer, cls_name) \
            BIND_TO_BUFFER(BinaryInputBuffer, cls_name)  \
            }

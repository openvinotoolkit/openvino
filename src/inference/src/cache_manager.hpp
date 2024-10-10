// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for the OpenVINO Cache Manager class C++ API
 *
 * @file cache_manager.hpp
 */
#pragma once

#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <windows.h>

#include "openvino/util/file_util.hpp"

namespace ov {

/**
 * @brief This class limits the locale env to a special value in sub-scope
 *
 */
class ScopedLocale {
public:
    ScopedLocale(int category, std::string newLocale) : m_category(category) {
        m_oldLocale = setlocale(category, nullptr);
        setlocale(m_category, newLocale.c_str());
    }
    ~ScopedLocale() {
        setlocale(m_category, m_oldLocale.c_str());
    }

private:
    int m_category;
    std::string m_oldLocale;
};

/**
 * @brief This class represents private interface for Cache Manager
 *
 */

class ICacheManager {

public:
    /**
     * @brief Default destructor
     */
    virtual ~ICacheManager() = default;

    /**
     * @brief Function passing created output stream
     *
     */
    using StreamWriter = std::function<void(std::ostream&)>;
    /**
     * @brief Callback when OpenVINO intends to write model to cache
     *
     * Client needs to call create std::ostream object and call writer(ostream)
     * Otherwise, model will not be cached
     *
     * @param id Id of cache (hash of the model)
     * @param writer Lambda function to be called when stream is created
     */
    virtual void write_cache_entry(const std::string& id, StreamWriter writer) = 0;

    /**
     * @brief Function passing created input stream
     */
    using StreamReader = std::function<void(std::istream&)>;

    /**
     * @brief Callback when OpenVINO intends to read model from cache
     *
     * Client needs to call create std::istream object and call reader(istream)
     * Otherwise, model will not be read from cache and will be loaded as usual
     *
     * @param id Id of cache (hash of the model)
     * @param reader Lambda function to be called when input stream is created
     */
    virtual void read_cache_entry(const std::string& id, StreamReader reader) = 0;

    /**
     * @brief Callback when OpenVINO intends to remove cache entry
     *
     * Client needs to perform appropriate cleanup (e.g. delete a cache file)
     *
     * @param id Id of cache (hash of the model)
     */
    virtual void remove_cache_entry(const std::string& id) = 0;
};

/**
 * @brief File storage-based Implementation of ICacheManager
 *
 * Uses simple file for read/write cached models.
 *
 */

class Readfile_istreambuf : public std::streambuf {
public:
    Readfile_istreambuf(const std::wstring& file_path) {
        m_handle = ::CreateFileW(file_path.c_str(),
                                GENERIC_READ,
                                FILE_SHARE_READ,
                                0,
                                OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL,
                                0);

        LARGE_INTEGER file_size_large;
        if (::GetFileSizeEx(m_handle, &file_size_large) == 0) {
            throw std::runtime_error("Can not get file size for ");
        }

        m_file_size = static_cast<uint64_t>(file_size_large.QuadPart);
        read_buffer.resize(m_max_read_length);

        read_more();
    }

    ~Readfile_istreambuf() {
        ::CloseHandle(m_handle);
    }

protected:
    void read_more(){
        OPENVINO_ASSERT(m_offset_in_buffer == m_buffer_size);

        DWORD chunk_size = static_cast<DWORD>(std::min<size_t>(m_file_size - m_bytes_read_without_buffer - m_buffer_size, m_max_read_length));
        DWORD chunk_read = 0;
        BOOL result = ReadFile(m_handle, read_buffer.data(), chunk_size, &chunk_read, NULL);

        OPENVINO_ASSERT(result, "read error: ", GetLastError());
        OPENVINO_ASSERT(chunk_read == chunk_size, "unexpectedly reached end of file");

        m_offset_in_buffer = 0;
        m_bytes_read_without_buffer += m_buffer_size;
        m_buffer_size = chunk_read;

    }

    std::streamsize xsgetn(char* s, std::streamsize count) override{
        OPENVINO_ASSERT(m_file_size >= (m_bytes_read_without_buffer + m_offset_in_buffer) + count);
        if (m_offset_in_buffer == m_max_read_length)
            read_more();

        std::streamsize got_bytes = 0;
        while (got_bytes < count) {
            size_t bytes_to_copy = std::min<size_t>(count - got_bytes, m_buffer_size - m_offset_in_buffer);
            std::memcpy(s + got_bytes, read_buffer.data() + m_offset_in_buffer, bytes_to_copy);
            got_bytes += bytes_to_copy;
            m_offset_in_buffer += bytes_to_copy;
            if (m_offset_in_buffer == m_max_read_length)
                read_more();
        }
        return got_bytes;
    }

    int_type underflow() override {
        if (m_offset_in_buffer == m_max_read_length)
            read_more();
        if (m_bytes_read_without_buffer + m_offset_in_buffer == m_file_size)
            return traits_type::eof();

        return traits_type::to_int_type(*(read_buffer.data() + m_offset_in_buffer));
    }

    int_type uflow() override {
        if (m_offset_in_buffer == m_max_read_length)
            read_more();
        if (m_bytes_read_without_buffer + m_offset_in_buffer == m_file_size)
            return traits_type::eof();

        return traits_type::to_int_type(*(read_buffer.data() + m_offset_in_buffer++));
    }

    int_type pbackfail(int_type ch) override {
        OPENVINO_THROW("NOT IMPLEMENTED");
    }

    std::streamsize showmanyc() override {
        return m_file_size - m_bytes_read_without_buffer - m_offset_in_buffer;
    }

    const size_t m_max_read_length = 64 * 1024 * 1024;
    std::vector<char> read_buffer;
    size_t m_buffer_size = 0;
    size_t m_offset_in_buffer = 0;
    size_t m_bytes_read_without_buffer = 0;

    HANDLE m_handle = nullptr;
    size_t m_file_size = 0;
};


class FileStorageCacheManager final : public ICacheManager {
    std::string m_cachePath;
#if defined(_WIN32) && defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
    std::wstring getBlobFile(const std::string& blobHash) const {
        return ov::util::string_to_wstring(ov::util::make_path(m_cachePath, blobHash + ".blob"));
    }
#else
    std::string getBlobFile(const std::string& blobHash) const {
        return ov::util::make_path(m_cachePath, blobHash + ".blob");
    }
#endif

public:
    /**
     * @brief Constructor
     *
     */
    FileStorageCacheManager(std::string cachePath) : m_cachePath(std::move(cachePath)) {}

    /**
     * @brief Destructor
     *
     */
    ~FileStorageCacheManager() override = default;

private:
    void write_cache_entry(const std::string& id, StreamWriter writer) override {
        // Fix the bug caused by pugixml, which may return unexpected results if the locale is different from "C".
        ScopedLocale plocal_C(LC_ALL, "C");
        std::ofstream stream(getBlobFile(id), std::ios_base::binary | std::ofstream::out);
        writer(stream);
    }

    void read_cache_entry(const std::string& id, StreamReader reader) override {
        // Fix the bug caused by pugixml, which may return unexpected results if the locale is different from "C".
        ScopedLocale plocal_C(LC_ALL, "C");
        auto blobFileName = getBlobFile(id);
        if (ov::util::file_exists(blobFileName)) {

            Readfile_istreambuf buffer(blobFileName);
            std::istream stream(&buffer);

            //std::ifstream stream(blobFileName, std::ios_base::binary);
            reader(stream);
        }
    }

    void remove_cache_entry(const std::string& id) override {
        auto blobFileName = getBlobFile(id);

        if (ov::util::file_exists(blobFileName)) {
#if defined(_WIN32) && defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
            _wremove(blobFileName.c_str());
#else
            std::remove(blobFileName.c_str());
#endif
        }
    }
};

}  // namespace ov

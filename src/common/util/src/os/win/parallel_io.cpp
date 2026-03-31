// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/parallel_io.hpp"

// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on

#include <cstdint>
#include <stdexcept>

#include "openvino/util/file_util.hpp"

namespace {

/**
 * @brief Convert a kernel device path (\Device\HarddiskVolume3\foo\bar) to a
 *        Win32 drive path (C:\foo\bar).
 */
static bool resolve_device_path(const wchar_t* dev_path, wchar_t* out, DWORD out_len) {
    const size_t MAX_DRIVES_LEN = 512;
    wchar_t drives[MAX_DRIVES_LEN] = {};
    if (!GetLogicalDriveStringsW(MAX_DRIVES_LEN, drives))
        return false;
    for (const wchar_t* d = drives; *d; d += wcslen(d) + 1) {
        wchar_t drive[3] = {d[0], d[1], L'\0'};
        wchar_t dev_name[MAX_PATH] = {};
        if (!QueryDosDeviceW(drive, dev_name, MAX_PATH))
            continue;
        const size_t dev_name_len = wcslen(dev_name);
        if (wcsncmp(dev_path, dev_name, dev_name_len) == 0 &&
            (dev_path[dev_name_len] == L'\\' || dev_path[dev_name_len] == L'\0')) {
            swprintf_s(out, out_len, L"%s%s", drive, dev_path + dev_name_len);
            return true;
        }
    }
    return false;
}

}  // namespace

namespace ov::util {

void get_file_handle_and_size(const std::filesystem::path& path,
                              std::streamoff file_offset,
                              FileHandle& out_handle,
                              std::streamoff& out_size) {
    out_handle = CreateFileW(path.native().c_str(),
                             GENERIC_READ,
                             FILE_SHARE_READ | FILE_SHARE_WRITE,
                             nullptr,
                             OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL,
                             nullptr);
    if (out_handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Cannot open file: " + ov::util::path_to_string(path));
    }
    LARGE_INTEGER file_size = {};
    if (!GetFileSizeEx(out_handle, &file_size)) {
        CloseHandle(out_handle);
        out_handle = INVALID_HANDLE_VALUE;
        throw std::runtime_error("Cannot get file size: " + ov::util::path_to_string(path));
    }
    out_size = static_cast<std::streamoff>(file_size.QuadPart);
    if (file_offset < 0 || file_offset > out_size) {
        CloseHandle(out_handle);
        out_handle = INVALID_HANDLE_VALUE;
        throw std::out_of_range("header_offset is out of range for file: " + ov::util::path_to_string(path));
    }
}

void close_file_handle(FileHandle handle) {
    if (handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
    }
}

FileHandle open_file_for_read(const std::filesystem::path& path) {
    return CreateFileW(path.native().c_str(),
                       GENERIC_READ,
                       FILE_SHARE_READ | FILE_SHARE_WRITE,
                       nullptr,
                       OPEN_EXISTING,
                       FILE_ATTRIBUTE_NORMAL,
                       nullptr);
}

bool positional_read(FileHandle handle, char* dst, size_t size, size_t file_offset) {
    char* cur = dst;
    size_t remaining = size;
    size_t cur_offset = file_offset;
    while (remaining > 0) {
        const DWORD to_read = static_cast<DWORD>(std::min(remaining, static_cast<size_t>(UINT_MAX - 1024u)));
        LARGE_INTEGER li;
        li.QuadPart = static_cast<LONGLONG>(cur_offset);
        if (!SetFilePointerEx(handle, li, nullptr, FILE_BEGIN)) {
            return false;
        }
        DWORD bytes_read = 0;
        if (!ReadFile(handle, cur, to_read, &bytes_read, nullptr)) {
            return false;
        }
        if (bytes_read == 0) {
            return false;
        }
        cur += bytes_read;
        cur_offset += bytes_read;
        remaining -= bytes_read;
    }
    return true;
}

bool get_mmap_file_info(const void* addr, std::filesystem::path& out_path, std::streamoff& out_offset) {
    MEMORY_BASIC_INFORMATION mbi{};
    if (!VirtualQuery(addr, &mbi, sizeof(mbi)) || mbi.Type != MEM_MAPPED) {
        return false;
    }
    wchar_t dev_path[MAX_PATH] = {};
    if (GetMappedFileNameW(GetCurrentProcess(), const_cast<void*>(addr), dev_path, MAX_PATH) == 0) {
        return false;
    }
    wchar_t win32_path[MAX_PATH] = {};
    if (!resolve_device_path(dev_path, win32_path, MAX_PATH)) {
        return false;
    }
    out_path = std::filesystem::path(win32_path);
    out_offset = reinterpret_cast<const char*>(addr) - reinterpret_cast<const char*>(mbi.AllocationBase);
    return true;
}

void prefetch_memory(const void* addr, size_t size) {
    WIN32_MEMORY_RANGE_ENTRY prefetch_range{const_cast<void*>(addr), size};
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &prefetch_range, 0);
}

}  // namespace ov::util
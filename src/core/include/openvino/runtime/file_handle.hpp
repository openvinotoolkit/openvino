// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>

namespace ov {
#ifdef _WIN32
// Windows uses HANDLE (void*) for file handles
using FileHandle = void*;
#else
// Linux/Unix uses int for file descriptors
using FileHandle = int;
#endif

/**
 * @brief Type definition for file handle provider callback (cross-platform).
 * Function that takes no arguments and returns a platform-specific file handle.
 * The callback implementation must release ownership, caller should close the FileHandle.
 * On Linux/Unix: returns int (file descriptor)
 * On Windows: returns void* (HANDLE cast to void*)
 * This is useful for scenarios where file access needs to be controlled externally,
 * such as Android content providers or Windows restricted file access scenarios.
 *
 * The data of interest does not have to start at byte 0 of the file: the provider may seek the
 * handle it returns (lseek / SetFilePointerEx) to the first byte it wants the consumer to see.
 * Consumers read that position back with ov::util::get_file_position() and treat it as the base
 * of the region, so a file embedding the data at some offset needs no extra out-of-band offset.
 * A handle left at position 0 - the usual case - means "the whole file".
 * @ingroup ov_runtime_cpp_api
 */
using FileHandleProvider = std::function<FileHandle()>;
}  // namespace ov

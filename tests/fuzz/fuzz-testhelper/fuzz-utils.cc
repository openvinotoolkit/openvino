// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuzz-utils.h"
#include <stdexcept>
#include <stdlib.h>
#include <string.h>
#include <string>
#ifndef _WIN32
#include <unistd.h>
#endif  // _WIN32

MemoryFile::MemoryFile(const void *data, size_t size) {
#ifdef _WIN32
    throw std::exception("MemoryFile is not implemented for Windows");
#else  // _WIN32
    m_name = strdup("/dev/shm/fuzz-XXXXXX");
    if (!m_name)
        throw std::bad_alloc();
    int fd = mkstemp(m_name);
    if (size) {
        size_t nbytes = write(fd, data, size);
        if (nbytes != size) {
            free(m_name);
            close(fd);
            throw std::runtime_error("Failed to write " + std::to_string(size) +
                                     " bytes to " + m_name);
        }
    }
    close(fd);
#endif  // _WIN32
}

MemoryFile::~MemoryFile() {
#ifndef _WIN32
    unlink(m_name);
    free(m_name);
#endif  // _WIN32
}

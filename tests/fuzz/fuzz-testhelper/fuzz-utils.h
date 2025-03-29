// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stddef.h>

class MemoryFile {
  public:
    /// Create a memory backed file
    MemoryFile(const void *data, size_t size);
    /// Delete memory backed file
    ~MemoryFile();

    /// Get path to a file.
    const char *name() { return m_name; }

  private:
    char *m_name;
};

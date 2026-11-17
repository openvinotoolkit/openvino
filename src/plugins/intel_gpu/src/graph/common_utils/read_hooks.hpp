// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ios>

namespace ov::intel_gpu {

/// Optional capabilities of a stream buffer that delegates reads to another transport.
class ReadHooks {
public:
    virtual ~ReadHooks() = default;

    /// Ask the delegated transport to bulk-load the upcoming reads.
    virtual bool prefetch(std::streamsize size) = 0;

    /// Keep already read pages resident when the blob memory is used as weights storage.
    virtual void keep_read_data() noexcept = 0;
};

}  // namespace ov::intel_gpu

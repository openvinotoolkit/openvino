// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Compiler Interface

#pragma once

#include <memory>
#include <vector>

#include "intel_npu/al/config/common.hpp"
#include "intel_npu/al/icompiler.hpp"

namespace intel_npu {

class IGraph {
public:
    IGraph(void* handle, NetworkMetadata metadata) : _handle(handle), _metadata(std::move(metadata)) {}

    virtual CompiledNetwork export_blob() const = 0;

    virtual std::vector<ov::ProfilingInfo> process_profiling_output() const = 0;

    virtual void set_argument_value(uint32_t argi, const void* argv) const = 0;

    virtual void initialize() const = 0;

    virtual ~IGraph() = default;

    NetworkMetadata get_metadata() const {
        return _metadata;
    }

    void* get_handle() const {
        return _handle;
    }

protected:
    void* _handle = nullptr;
    NetworkMetadata _metadata;
};

}  // namespace intel_npu

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstddef>

#include "intel_npu/utils/vm/npu_vm_runtime.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

inline bool use_npu_vm_runtime_v2_api(npu_vm_runtime_version_t apiVersion) {
    return apiVersion >= NPU_VM_RUNTIME_VERSION_2_0;
}

class NpuVMRuntimeConfigChain final {
public:
    NpuVMRuntimeConfigChain() = default;
    NpuVMRuntimeConfigChain(const NpuVMRuntimeConfigChain&) = delete;
    NpuVMRuntimeConfigChain& operator=(const NpuVMRuntimeConfigChain&) = delete;

    void clear() {
        _size = 0;
    }

    void append(npu_vm_runtime_config_type_t type, npu_vm_runtime_config_value_t value) {
        OPENVINO_ASSERT(_size < _descs.size(), "VM runtime config descriptor chain capacity exceeded");
        OPENVINO_ASSERT(_size == 0 || _descs[_size - 1].type < type,
                        "VM runtime config descriptor chain must be ordered by increasing type");
        _descs[_size] = npu_vm_runtime_config_desc_t{type, value, nullptr};
        if (_size > 0) {
            _descs[_size - 1].pNext = &_descs[_size];
        }
        ++_size;
    }

    const npu_vm_runtime_config_desc_t* head() const {
        return _size == 0 ? nullptr : &_descs[0];
    }

private:
    static constexpr size_t MAX_RUNTIME_CONFIG_DESCS = 3;

    std::array<npu_vm_runtime_config_desc_t, MAX_RUNTIME_CONFIG_DESCS> _descs = {};
    size_t _size = 0;
};

}  // namespace intel_npu

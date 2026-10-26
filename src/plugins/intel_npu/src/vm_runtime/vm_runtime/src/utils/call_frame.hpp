//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "function.hpp"
#include "vm_export.hpp"

#include <cstdint>
#include <vector>

namespace intel_npu::vm {

constexpr const uint8_t* EXIT_RETURN_ADDR = nullptr;

class NPU_VM_EXPORT CallFrame {
    const Function* _function;        // The function executing in this call frame
    std::vector<int64_t> _registers;  // The register set for the current function call, indexed by register number
    const uint8_t* _returnAddress;    // The return address to jump to after this function call completes
    std::vector<int64_t*> _results;   // Sinks that RETV writes its return values into (owned by the caller)

public:
    CallFrame(const Function& function, const uint8_t* returnAddress, std::vector<int64_t*> results);

    // Returns the function executing in this call frame
    const Function& getFunction() const;

    // Returns a mutable reference to the register at the given index.
    // Throws a runtime error if the index exceeds the register count.
    int64_t& getReg(int16_t index);

    // Sets the register at the given index to the specified value.
    // Throws a runtime error if the index exceeds the register count.
    void setReg(int16_t index, int64_t value);

    // Returns the address to return to after the function call completes
    const uint8_t* getReturnAddress() const;

    // Returns the result sinks that RETV writes its return values into
    std::vector<int64_t*>& getResults();
};

}  // namespace intel_npu::vm

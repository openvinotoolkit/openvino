//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "call_frame.hpp"
#include "function.hpp"
#include "npu_bytecode_utils/except.hpp"

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

intel_npu::vm::CallFrame::CallFrame(const Function& function, const uint8_t* returnAddress,
                                    std::vector<int64_t*> results)
        : _function(&function),
          _registers(function.getNumGeneralRegisters(), 0),
          _returnAddress(returnAddress),
          _results(std::move(results)) {
}

const intel_npu::vm::Function& intel_npu::vm::CallFrame::getFunction() const {
    return *_function;
}

int64_t& intel_npu::vm::CallFrame::getReg(int16_t index) {
    if (index < 0 || static_cast<size_t>(index) >= _registers.size()) {
        NPU_VM_THROW("Register index {} is out of bounds for register count {}", index, _registers.size());
    }
    return _registers.at(index);
}

void intel_npu::vm::CallFrame::setReg(int16_t index, int64_t value) {
    if (index < 0 || static_cast<size_t>(index) >= _registers.size()) {
        NPU_VM_THROW("Register index {} is out of bounds for register count {}", index, _registers.size());
    }
    _registers.at(index) = value;
}

const uint8_t* intel_npu::vm::CallFrame::getReturnAddress() const {
    return _returnAddress;
}

std::vector<int64_t*>& intel_npu::vm::CallFrame::getResults() {
    return _results;
}

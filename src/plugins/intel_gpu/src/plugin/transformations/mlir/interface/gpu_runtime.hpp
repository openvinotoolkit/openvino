// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "../common/typedefs.hpp"
#include "intel_gpu/runtime/event.hpp"

namespace cldnn {
class stream;
class engine;
struct network;
class primitive_inst;
}  // namespace cldnn

namespace ov {
class Node;
}  // namespace ov

namespace ov::intel_gpu::mlir {

struct MLIRGpuRuntime {
public:
    virtual ~MLIRGpuRuntime() = default;
    MLIRGpuRuntime(const MLIRGpuRuntime&) = delete;
    MLIRGpuRuntime& operator=(const MLIRGpuRuntime&) = delete;

protected:
    MLIRGpuRuntime() = default;
    friend struct ::cldnn::network;
    inline static std::unique_ptr<MLIRGpuRuntime> (*create)(cldnn::stream&, cldnn::engine&) = nullptr;
};

class MLIRGpuProgram {
public:
    virtual ~MLIRGpuProgram() = default;
    MLIRGpuProgram(const MLIRGpuProgram&) = delete;
    MLIRGpuProgram& operator=(const MLIRGpuProgram&) = delete;

    virtual void wait_compiled() = 0;
    virtual cldnn::event::ptr execute(MLIRGpuRuntime& runtime,
                                      const ov::Node& op,
                                      cldnn::primitive_inst& instance,
                                      const std::vector<cldnn::event::ptr>& deps,
                                      bool need_event) = 0;

    const InputOrders& get_input_orders() const {
        return input_orders;
    }

protected:
    explicit MLIRGpuProgram(InputOrders input_orders) : input_orders(std::move(input_orders)) {}

private:
    // Describes physical memory order for certain memref inputs of the compiled MLIR program.
    // 'input_orders[1] = {0, 2, 1}' equals to the following memref in mlir (actual shapes can be any):
    //                     ^  ^  ^
    //                     E  G  N
    // memref<256x512x32xf16, strided<[16384, 1, 512]>>; (N is innermost, stride 1)
    //          ^   ^  ^                   ^  ^    ^
    //          E   N  G                   E  N    G
    // If an input is not present in this map means its physical order is default one (row-major)
    InputOrders input_orders;
};

void register_mlir_gpu_runtime();

}  // namespace ov::intel_gpu::mlir

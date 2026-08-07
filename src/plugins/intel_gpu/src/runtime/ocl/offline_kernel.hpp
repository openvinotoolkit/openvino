// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cldnn {
namespace ocl {

// HW-free placeholder kernel used ONLY on the offline (GPU_OFFLINE_COMPILE) export path.
// It holds just an entry-point id and a pre-compiled (ocloc) binary; it carries NO cl::Kernel
// handle and is NEVER executed (offline compile never builds a cldnn::network). It exists so
// build_batch can produce serializable kernels without driver-JIT'ing against a real cl::Context.
// The compiler/serialization side only reads get_id() (entry point) and get_binary() (bytes);
// import reconstructs real ocl_kernels from these bytes via build_kernels(NATIVE_BIN).
class offline_kernel : public kernel {
    std::string _id;
    std::vector<uint8_t> _binary;

public:
    offline_kernel(std::string id, std::vector<uint8_t> binary)
        : _id(std::move(id))
        , _binary(std::move(binary)) { }

    std::string get_id() const override { return _id; }
    std::vector<uint8_t> get_binary() const override { return _binary; }
    std::string get_build_log() const override { return {}; }

    std::shared_ptr<kernel> clone(bool /*reuse_kernel_handle*/ = false) const override {
        return std::make_shared<offline_kernel>(_id, _binary);
    }

    bool is_same(const kernel& other) const override {
        auto other_ptr = dynamic_cast<const offline_kernel*>(&other);
        return other_ptr != nullptr && other_ptr->_id == _id && other_ptr->_binary == _binary;
    }
};

}  // namespace ocl
}  // namespace cldnn
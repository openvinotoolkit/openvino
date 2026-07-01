// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "sycl_common.hpp"
#include "sycl_memory.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/kernel.hpp"

#include <memory>
#include <vector>

namespace cldnn {
namespace sycl {

class sycl_kernel : public kernel {
    sycl_kernel_type _compiled_kernel;
    std::string _kernel_id;

public:
    sycl_kernel(sycl_kernel_type compiled_kernel, const std::string& kernel_id)
        : _compiled_kernel(compiled_kernel)
        , _kernel_id(kernel_id) { }

    std::string get_id() const override { return _kernel_id; }
    const sycl_kernel_type& get_handle() const { return _compiled_kernel; }
    sycl_kernel_type& get_handle() { return _compiled_kernel; }
    std::shared_ptr<kernel> clone(bool reuse_kernel_handle = false) const override {
        if (reuse_kernel_handle)
            return std::make_shared<sycl_kernel>(get_handle(), _kernel_id);

        OPENVINO_THROW("SYCL kernel cloning with reuse_kernel_handle=false is not supported.");
    }
    bool is_same(const kernel &other) const override {
        auto other_ptr = dynamic_cast<const sycl_kernel*>(&other);
        if (other_ptr == nullptr) {
            return false;
        }
        return get_handle() == other_ptr->get_handle();
    }

    std::vector<uint8_t> get_binary() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::string get_build_log() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};

}  // namespace sycl
}  // namespace cldnn

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "sycl_base_kernel.hpp"
#include "sycl_common.hpp"
#include "intel_gpu/runtime/kernel_args.hpp"
#include "intel_gpu/runtime/kernel.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace cldnn {
namespace sycl {

class sycl_kernel : public sycl_base_kernel {
public:
    sycl_kernel(sycl_kernel_type compiled_kernel, const std::string& kernel_id,
                std::shared_ptr<const std::vector<std::byte>> spirv_binary,
                std::shared_ptr<const std::string> build_log = {})
        : _compiled_kernel(std::move(compiled_kernel))
        , _kernel_id(kernel_id)
        , _spirv_binary(std::move(spirv_binary))
        , _build_log(std::move(build_log)) { }

    std::string get_id() const override { return _kernel_id; }
    std::shared_ptr<kernel> clone(bool /*reuse_kernel_handle*/ = false) const override {
        return std::make_shared<sycl_kernel>(_compiled_kernel, _kernel_id, _spirv_binary, _build_log);
    }
    bool is_same(const kernel &other) const override {
        auto other_ptr = dynamic_cast<const sycl_kernel*>(&other);
        if (other_ptr == nullptr) {
            return false;
        }
        return _compiled_kernel == other_ptr->_compiled_kernel;
    }

    std::vector<uint8_t> get_binary() const override;
    std::string get_build_log() const override;

    void launch(::sycl::handler& cgh, const kernel_arguments_desc& args_desc) override;

    static void create_kernels(const ::sycl::context& ctx,
                               std::shared_ptr<const std::vector<std::byte>> spirv_binary,
                               std::shared_ptr<const std::string> build_log,
                               std::vector<kernel::ptr>& out);

private:
    sycl_kernel_type _compiled_kernel;
    std::string _kernel_id;
    std::shared_ptr<const std::vector<std::byte>> _spirv_binary;
    std::shared_ptr<const std::string> _build_log;
};

}  // namespace sycl
}  // namespace cldnn

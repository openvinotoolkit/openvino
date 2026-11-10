// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#include "sycl_kernel_builder.hpp"
#include "sycl_ptx_kernel.hpp"
#include "ocl/ocl_device.hpp"
#include "ocl/ocl_kernel_builder.hpp"
#include "openvino/core/except.hpp"

#include <regex>
#include <string>
#include <utility>
#include <vector>

namespace cldnn {
namespace sycl {
namespace nvidia {

namespace {

// The PTX that clBuildProgram on the NVIDIA OpenCL runtime produces is meant for that runtime; the patches below make
// it loadable and launchable through the CUDA driver API (cuModuleLoadData / cuLaunchKernel) instead. They are applied
// before the PTX is cached, so a cached binary needs none of them.

// NVVM decorates __local pointer parameters as '.param .u64 .ptr .shared .align N', which cuModuleLoadData rejects with
// CUDA_ERROR_INVALID_IMAGE. Turn the decoration into a comment, leaving a plain .u64 parameter. The comment keeps N -
// the alignment the kernel assumes for that argument (16, or more for wide vector types) - in the PTX, cached binaries
// included, so the launch can place the argument at an offset into shared memory that honours it (see ptx_program in
// sycl_ptx_kernel.hpp).
void patch_ptx_local(std::string& ptx) {
    static const std::regex shared_param_re(
        R"(\.param \.u64 \.ptr \.shared \.align (\d+))");
    ptx = std::regex_replace(ptx, shared_param_re, ".param .u64 /* .ptr .shared .align $1 */");
}

// NVVM maps OpenCL work-item functions to %envreg special registers that the OpenCL runtime sets up before a launch.
// cuLaunchKernel never sets them, so they hold garbage when the kernel is launched through CUDA. The encoding, as found
// in NVVM-generated PTX:
//     %envreg0-2  global_work_offset[0-2]  (the get_global_id offset addend)
//     %envreg3-5  global_work_offset[0-2]  (the get_group_id offset addend)
//     %envreg6-8  num_groups[0-2]          (get_global_size / get_num_groups)
// The offsets are always 0 in OpenVINO and the group counts are what cuLaunchKernel puts in %nctaid.x/y/z, so
// %envreg0-5 become 0 and %envreg6-8 become %nctaid.x/y/z.
void patch_ptx_envreg(std::string& ptx) {
    // One pass for all nine registers. (?!\d) keeps them whole: the prelude's %envreg27-31 are not touched.
    static const std::regex envreg_re(R"(%envreg([0-8])(?!\d))");
    static const char* const replacements[] = {"0", "0", "0", "0", "0", "0", "%nctaid.x", "%nctaid.y", "%nctaid.z"};
    std::string out;
    out.reserve(ptx.size());
    auto last = ptx.cbegin();
    for (std::sregex_iterator it(ptx.cbegin(), ptx.cend(), envreg_re), end; it != end; ++it) {
        out.append(last, (*it)[0].first);
        out += replacements[*(*it)[1].first - '0'];
        last = (*it)[0].second;
    }
    out.append(last, ptx.cend());
    ptx = std::move(out);
}

// Build sycl_ptx_kernel objects from a compiled PTX string, one per entry point. Every entry point of the PTX is a
// kernel of the program: clBuildProgram on the NVIDIA OpenCL runtime emits exactly the program's kernels, whatever
// their names (copy_with_offset included). The global work offset is not a separate entry point but read from
// %envreg0-5, which patch_ptx_envreg() replaces with 0. Cached binaries are this same PTX, so the same holds when they
// are loaded. The kernels share one ptx_program, so the PTX is parsed once and JIT-compiled into a module once for the
// whole batch.
void create_ptx_kernels(std::string ptx, std::vector<kernel::ptr>& out) {
    const auto program = std::make_shared<ptx_program>(std::move(ptx));
    if (program->entry_points().empty())
        OPENVINO_THROW("[GPU] No entry points found in PTX");

    for (const auto& ep : program->entry_points())
        out.push_back(std::make_shared<sycl_ptx_kernel>(program, ep, ep));
}

}  // namespace

sycl_kernel_builder::sycl_kernel_builder(const sycl_device& device)
    : _device(device) {}

void sycl_kernel_builder::init_ocl_builder() const {
    // The matching OpenCL device is looked up once per SYCL device (see sycl_device::get_ocl_device()).
    auto ocl_dev = _device.get_ocl_device();
    OPENVINO_ASSERT(ocl_dev != nullptr,
        "[GPU] Could not find a matching OCL device for the NVIDIA SYCL device");
    _ocl_builder = std::make_shared<::cldnn::ocl::ocl_kernel_builder>(*ocl_dev);
}

void sycl_kernel_builder::build_kernels(
        const void* src, size_t src_bytes, KernelFormat src_format,
        const std::string& options,
        std::vector<kernel::ptr>& out) const {
    if (!src || src_bytes == 0)
        OPENVINO_THROW("[GPU] sycl::nvidia::sycl_kernel_builder: empty kernel source");

    if (src_format == KernelFormat::NATIVE_BIN) {
        // A cached binary is PTX this function has already patched.
        create_ptx_kernels(std::string(static_cast<const char*>(src), src_bytes), out);
        return;
    }

    // Should the initialization throw (no matching OpenCL device), the next call tries again: call_once only records a
    // call that returns.
    std::call_once(_ocl_builder_once, [this]() { init_ocl_builder(); });
    std::vector<kernel::ptr> tmp;
    _ocl_builder->build_kernels(src, src_bytes, src_format, options, tmp);
    OPENVINO_ASSERT(!tmp.empty(), "[GPU] OCL builder returned no kernels");
    // NVIDIA OCL runtime returns PTX as the device binary
    const auto binary = tmp[0]->get_binary();
    std::string ptx_str(binary.begin(), binary.end());
    patch_ptx_envreg(ptx_str);
    patch_ptx_local(ptx_str);
    create_ptx_kernels(std::move(ptx_str), out);
}

}  // namespace nvidia
}  // namespace sycl
}  // namespace cldnn

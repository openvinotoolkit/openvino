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

// Build sycl_ptx_kernel objects from a compiled PTX string, one per entry point. Every entry point of the PTX is a
// kernel of the program: clBuildProgram on the NVIDIA OpenCL runtime emits exactly the program's kernels, whatever
// their names (copy_with_offset included). The global work offset is not a separate entry point but read from
// %envreg0-5, which patch_ptx_envreg() replaces with 0. Cached binaries are this same PTX, so the same holds when they
// are loaded. The kernels share one ptx_program, so the PTX is parsed once and JIT-compiled into a module once for the
// whole batch.
void create_ptx_kernels(std::string ptx, std::vector<kernel::ptr>& out) {
    const auto program = std::make_shared<ptx_program>(std::move(ptx));
    if (program->entry_points().empty())
        OPENVINO_THROW("[GPU][NVIDIA] No entry points found in PTX");

    for (const auto& ep : program->entry_points())
        out.push_back(std::make_shared<sycl_ptx_kernel>(program, ep, ep));
}

}  // namespace

// ---------------------------------------------------------------------------
// PTX post-processing helpers
// ---------------------------------------------------------------------------
namespace {

// Apply PTX text-level patches to fix incompatibilities between the NVIDIA OCL
// compiler output and the CUDA driver API (cuModuleLoadData / cuLaunchKernel).
//
// (1) '.param .u64 .ptr .shared .align N' parameters
//     The NVIDIA OCL compiler (NVVM) emits this decoration for __local pointer
//     arguments.  cuModuleLoadData rejects modules that contain it with
//     CUDA_ERROR_INVALID_IMAGE, so the decoration is turned into a comment and
//     the parameter becomes a plain .u64.  The comment keeps N - the alignment
//     the kernel assumes for that argument (16, or more for wide vector types) -
//     in the PTX, cached binaries included, so the launch can place the argument
//     at an offset into shared memory that honours it.
//
// (2) '%envreg[0-8]' special registers
//     The NVIDIA OCL compiler (NVVM) maps OpenCL work-item functions to a set
//     of '%envreg' special registers that the OpenCL runtime initialises before
//     the kernel launch.  cuLaunchKernel never sets these registers, so they
//     hold garbage when kernels are driven via the CUDA driver API.
//
//     NVVM encoding (confirmed by inspecting NVVM-generated PTX):
//       %envreg0-2  global_work_offset[0-2]  (get_global_id offset addend)
//       %envreg3-5  global_work_offset[0-2]  (get_group_id  offset addend)
//       %envreg6    num_groups[0] == %nctaid.x  (get_global_size / get_num_groups)
//       %envreg7    num_groups[1] == %nctaid.y
//       %envreg8    num_groups[2] == %nctaid.z
//
//     Patches applied when launching via CUDA:
//       %envreg[0-5] → 0         (global_work_offset is always 0 in OpenVINO)
//       %envreg6     → %nctaid.x (grid X dimension set by cuLaunchKernel)
//       %envreg7     → %nctaid.y
//       %envreg8     → %nctaid.z
// Turn '.ptr .shared .align N' into a comment - cuModuleLoadData rejects it (CUDA_ERROR_INVALID_IMAGE) - keeping
// the alignment for the launch (see ptx_program in sycl_ptx_kernel.hpp). NVVM emits it for __local pointer parameters.
static void patch_ptx_local(std::string& ptx) {
    static const std::regex shared_param_re(
        R"(\.param \.u64 \.ptr \.shared \.align (\d+))");
    ptx = std::regex_replace(ptx, shared_param_re, ".param .u64 /* .ptr .shared .align $1 */");
}

// Replace '%envreg[0-5]' with 0 and '%envreg[6-8]' with the corresponding
// %nctaid register.  NVVM emits these for get_global_id/get_global_size;
// cuLaunchKernel never writes %envreg so they hold garbage without this patch.
static void patch_ptx_envreg(std::string& ptx) {
    // global_work_offset registers (always 0 in OpenVINO)
    static const std::regex envreg_offset_re(R"(%envreg[0-5](?!\d))");
    ptx = std::regex_replace(ptx, envreg_offset_re, "0");
    // num_groups registers: replace with the %nctaid.{x,y,z} registers that
    // cuLaunchKernel correctly populates from the grid dimensions.
    static const std::regex envreg6_re(R"(%envreg6(?!\d))");
    static const std::regex envreg7_re(R"(%envreg7(?!\d))");
    static const std::regex envreg8_re(R"(%envreg8(?!\d))");
    ptx = std::regex_replace(ptx, envreg6_re, "%nctaid.x");
    ptx = std::regex_replace(ptx, envreg7_re, "%nctaid.y");
    ptx = std::regex_replace(ptx, envreg8_re, "%nctaid.z");
}

}  // namespace (patch_ptx)

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
sycl_kernel_builder::sycl_kernel_builder(const sycl_device& device)
    : _device(device) {}

// ---------------------------------------------------------------------------
// OCL builder lazy initialisation
// ---------------------------------------------------------------------------
void sycl_kernel_builder::init_ocl_builder() const {
    // The matching OpenCL device is looked up once per SYCL device (see sycl_device::get_ocl_device()).
    auto ocl_dev = _device.get_ocl_device();
    OPENVINO_ASSERT(ocl_dev != nullptr,
        "[GPU][NVIDIA] Could not find a matching OCL device for the NVIDIA SYCL device");
    _ocl_builder = std::make_shared<::cldnn::ocl::ocl_kernel_builder>(*ocl_dev);
}

// ---------------------------------------------------------------------------
// build_kernels
// ---------------------------------------------------------------------------
void sycl_kernel_builder::build_kernels(
        const void* src, size_t src_bytes, KernelFormat src_format,
        const std::string& options,
        std::vector<kernel::ptr>& out) const {
    if (!src || src_bytes == 0)
        OPENVINO_THROW("[GPU][NVIDIA] sycl::nvidia::sycl_kernel_builder: empty kernel source");

    if (src_format == KernelFormat::NATIVE_BIN) {
        create_ptx_kernels(std::string(static_cast<const char*>(src), src_bytes), out);
        return;
    }

    {
        std::lock_guard lock(_mutex);
        if (!_ocl_builder)
            init_ocl_builder();
    }
    std::vector<kernel::ptr> tmp;
    _ocl_builder->build_kernels(src, src_bytes, src_format, options, tmp);
    OPENVINO_ASSERT(!tmp.empty(), "[GPU][NVIDIA] OCL builder returned no kernels");
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

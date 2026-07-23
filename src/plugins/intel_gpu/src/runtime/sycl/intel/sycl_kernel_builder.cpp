// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#include "sycl_kernel_builder.hpp"
#include "../sycl_kernel.hpp"
#include "../sycl_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/util/shared_object.hpp"

#include <ocloc_api.h>

#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

#if defined(_WIN32)
constexpr const char* OCLOC_LIBRARY_NAME = "ocloc64.dll";
#else
constexpr const char* OCLOC_LIBRARY_NAME = "libocloc.so";
#endif

struct OclocApi {
    std::shared_ptr<void> lib;
    pOclocInvoke invoke;
    decltype(&oclocFreeOutput) free_output;
};

// Load ocloc and resolve its entry points exactly once per process.
const OclocApi& get_ocloc_api() {
    static const OclocApi api = [] {
        OclocApi a;
        try {
            a.lib = ov::util::load_shared_object(OCLOC_LIBRARY_NAME);
            a.invoke = reinterpret_cast<pOclocInvoke>(ov::util::get_symbol(a.lib, "oclocInvoke"));
            a.free_output = reinterpret_cast<decltype(&oclocFreeOutput)>(
                ov::util::get_symbol(a.lib, "oclocFreeOutput"));
        } catch (const std::exception& e) {
            OPENVINO_THROW("[GPU] failed to load ocloc library '", OCLOC_LIBRARY_NAME, "': ", e.what());
        }
        return a;
    }();
    return api;
}

uint32_t query_device_ip_version(::sycl::device device) {
#if SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE
    namespace syclex = ::sycl::ext::oneapi::experimental;

    // The experimental `architecture` info query relies on backend support
    // (e.g. `cl_intel_device_attribute_query` on the OpenCL backend). It may
    // throw on drivers that do not implement it. In that case, fall back to 0
    // so that ocloc is invoked without `-device` and still produces target
    // independent SPIR-V.
    try {
        // arch = 0xAABBBBCCCCCCCCDD
        // for Intel GPU, AA=0x00, BBBB=reserved, CCCCCCCC=GMDID, DD=reserved
        const auto arch = device.get_info<syclex::info::device::architecture>();

        // extract GMDID from arch
        return static_cast<uint32_t>(static_cast<uint64_t>(arch) >> 8);
    } catch (const ::sycl::exception&) {
        return 0;
    }
#else
    // If the device architecture extension is not available, fall back to 0.
    (void)device;
    return 0;
#endif
}

std::vector<std::byte> opencl_c_to_spirv(const std::string& source,
                                         const std::string& options,
                                         uint32_t device_ip_version,
                                         std::string& build_log) {
    const auto& ocloc = get_ocloc_api();
    auto* invoke_fn = ocloc.invoke;
    auto* free_fn = ocloc.free_output;

    // Build the ocloc argument list:
    //   ocloc -spv_only [-device <ip_ver>] [-options <options>] -file kernel.cl
    std::vector<const char*> args = {"ocloc", "-spv_only"};

    // Pass the device IP version so that ocloc enables the right extension set
    // (including cl_khr_fp16, sub-groups, etc.).
    std::string device_str;
    if (device_ip_version != 0) {
        device_str = std::to_string(device_ip_version);
        args.push_back("-device");
        args.push_back(device_str.c_str());
    }

    if (!options.empty()) {
        args.push_back("-options");
        args.push_back(options.c_str());
    }
    args.push_back("-file");
    args.push_back("kernel.cl");

    const uint8_t* src_data[]  = {reinterpret_cast<const uint8_t*>(source.c_str())};
    const uint64_t src_sizes[] = {static_cast<uint64_t>(source.size() + 1)};
    const char*    src_names[] = {"kernel.cl"};

    uint32_t num_outputs  = 0;
    uint8_t** outputs     = nullptr;
    uint64_t* out_sizes   = nullptr;
    char**    out_names   = nullptr;

    int ret = invoke_fn(static_cast<uint32_t>(args.size()), args.data(),
                        1, src_data, src_sizes, src_names,
                        0, nullptr, nullptr, nullptr,
                        &num_outputs, &outputs, &out_sizes, &out_names);

    // Collect SPIR-V output and any error log
    std::vector<std::byte> spirv;
    build_log.clear();
    for (uint32_t i = 0; i < num_outputs; ++i) {
        if (out_names[i] && ::strstr(out_names[i], ".spv") != nullptr &&
                outputs[i] != nullptr) {
            spirv.assign(reinterpret_cast<std::byte*>(outputs[i]),
                         reinterpret_cast<std::byte*>(outputs[i]) + out_sizes[i]);
        } else if (out_names[i] &&
                   ::strcmp(out_names[i], "stdout.log") == 0 &&
                   out_sizes[i] > 0) {
            build_log.assign(reinterpret_cast<const char*>(outputs[i]), out_sizes[i]);
        }
    }
    free_fn(&num_outputs, &outputs, &out_sizes, &out_names);

    if (ret != 0 || spirv.empty()) {
        GPU_DEBUG_INFO << "-------- Kernel build error" << std::endl;
        GPU_DEBUG_INFO << build_log;
        GPU_DEBUG_INFO << "-------- End of Kernel build error" << std::endl;

        OPENVINO_THROW("[GPU] ocloc SPIR-V compilation failed");
    }

    return spirv;
}

}  // namespace

namespace cldnn {
namespace sycl {
namespace intel {

sycl_kernel_builder::sycl_kernel_builder(const sycl_device& device)
    : _device(device) {}

    void sycl_kernel_builder::build_kernels(
        const void* src, size_t src_bytes, KernelFormat src_format,
        const std::string& options,
        std::vector<kernel::ptr>& out) const {
    OPENVINO_ASSERT(src && src_bytes > 0,
                    "[GPU] sycl::intel::sycl_kernel_builder: empty kernel source");

    std::vector<std::byte> spirv;
    std::string build_log;

    if (src_format == KernelFormat::SOURCE) {
        const std::string source(static_cast<const char*>(src), src_bytes);
        spirv = opencl_c_to_spirv(source, options, query_device_ip_version(_device.get_device()), build_log);
    } else if (src_format == KernelFormat::NATIVE_BIN) {
        // For Intel SYCL, the "native binary" stored in the kernel cache is
        // the SPIR-V binary returned by sycl_kernel::get_binary().
        spirv.assign(static_cast<const std::byte*>(src),
                     static_cast<const std::byte*>(src) + src_bytes);
    } else {
        OPENVINO_THROW("[GPU] sycl::intel::sycl_kernel_builder: unsupported kernel format");
    }

    cldnn::sycl::sycl_kernel::create_kernels(_device.get_context(), spirv, build_log, out);
}

}  // namespace intel
}  // namespace sycl
}  // namespace cldnn

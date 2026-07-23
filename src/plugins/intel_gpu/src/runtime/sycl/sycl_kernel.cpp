// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#include "sycl_kernel.hpp"
#include "sycl_memory.hpp"
#include "openvino/core/except.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

namespace cldnn {
namespace sycl {

namespace {

// SPIR-V magic number (little-endian layout in memory)
constexpr uint32_t SPIRV_MAGIC = 0x07230203u;

// OpEntryPoint opcode
constexpr uint16_t SPIRV_OP_ENTRY_POINT = 15u;

// Execution model value for OpenCL kernels
constexpr uint32_t SPIRV_EXEC_MODEL_KERNEL = 6u;

// Read a uint32_t from an unaligned/potentially-aliased byte pointer via
// memcpy.  Compilers (GCC/Clang/MSVC) fold this into a single load
// instruction when the pointer is in fact aligned, so there is no
// runtime overhead in the common case.
inline uint32_t read_u32(const void* p) noexcept {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// Validates the SPIR-V module header: non-null pointer, at least the 5-word
// header (magic, version, generator, bound, schema) present, and correct magic.
bool is_spirv(const void* data, size_t bytes) {
    if (!data || bytes < 5 * sizeof(uint32_t))
        return false;
    return read_u32(data) == SPIRV_MAGIC;
}

// Extracts entry point names (OpEntryPoint with ExecutionModel=Kernel) from a
// SPIR-V module. Throws if the input is not a valid SPIR-V binary.
std::vector<std::string> extract_spirv_kernel_names(const void* data, size_t bytes) {
    if (!is_spirv(data, bytes)) {
        OPENVINO_THROW("[GPU] extract_spirv_kernel_names: invalid SPIR-V binary");
    }

    const size_t word_count = bytes / sizeof(uint32_t);
    const auto* p = static_cast<const uint8_t*>(data);

    std::vector<std::string> names;
    size_t i = 5;  // skip the five-word header
    while (i < word_count) {
        const uint32_t first_word  = read_u32(p + i * sizeof(uint32_t));
        const uint16_t opcode      = static_cast<uint16_t>(first_word & 0xFFFFu);
        const uint16_t instr_words = static_cast<uint16_t>(first_word >> 16u);

        // An instruction must contain at least the opcode word and must not
        // extend past the end of the module.  (word_count - i) is safe from
        // underflow because the loop guard ensures i < word_count.
        if (instr_words == 0 || instr_words > word_count - i)
            break;  // malformed

        // OpEntryPoint layout:
        //   word 0: opcode | wordcount   (already consumed above)
        //   word 1: execution model
        //   word 2: entry point <id>
        //   word 3..: name (null-terminated UTF-8, packed into 4-byte words)
        // The minimum valid wordcount is therefore 4 (empty name still needs
        // one null-terminator word).
        if (opcode == SPIRV_OP_ENTRY_POINT &&
            instr_words >= 4 &&
            read_u32(p + (i + 1) * sizeof(uint32_t)) == SPIRV_EXEC_MODEL_KERNEL) {
            const char* name_start =
                reinterpret_cast<const char*>(p + (i + 3) * sizeof(uint32_t));
            const size_t max_name_bytes =
                (static_cast<size_t>(instr_words) - 3) * sizeof(uint32_t);
            const char* name_end = static_cast<const char*>(
                std::memchr(name_start, '\0', max_name_bytes));
            const size_t name_len =
                name_end ? static_cast<size_t>(name_end - name_start) : max_name_bytes;
            names.emplace_back(name_start, name_len);
        }

        i += instr_words;
    }
    return names;
}

// Entry point names present in a SPIR-V module but not usable as executable kernels.
// These are compiler-internal (vendor-specific) synthetic entries and cannot be
// looked up via ext_oneapi_get_kernel().
//   - "Intel_Symbol_Table_Void_Program": IGC bookkeeping entry for externally-linked
//     / indirectly-called symbols.
bool is_ignored_kernel_name(const std::string& name) {
    return name == "Intel_Symbol_Table_Void_Program";
}

} // namespace

std::vector<uint8_t> sycl_kernel::get_binary() const {
    OPENVINO_ASSERT(!_spirv_binary.empty(),
                    "[GPU] sycl_kernel::get_binary: SPIR-V binary is empty");

    const auto* p = reinterpret_cast<const uint8_t*>(_spirv_binary.data());
    return std::vector<uint8_t>(p, p + _spirv_binary.size());
}

std::string sycl_kernel::get_build_log() const {
    return _build_log;
}

void sycl_kernel::launch(::sycl::handler& cgh,
                         const kernel_arguments_desc& args_desc) {
    // Get stored_args by value to avoid data race
    const auto args = stored_args();

    const auto& gws = args_desc.workGroups.global;
    const auto& lws = args_desc.workGroups.local;

    // Associate the kernel bundle with this command group so that set_arg and
    // parallel_for know which kernel to target.
    cgh.use_kernel_bundle(_compiled_kernel.get_kernel_bundle());

    // Bind each argument.
    int arg_idx = 0;
    for (const auto& arg : args) {
        if (arg.kind == arg_t::kind_t::BUFFER) {
            if (!arg.mem) {
                OPENVINO_THROW("[GPU] sycl_kernel::launch: null memory for BUFFER arg at index ", arg_idx);
            }
            if (auto* gb = dynamic_cast<gpu_buffer*>(arg.mem.get())) {
                // SYCL buffer: create a read-write accessor on the
                // (sub-)buffer so the runtime registers the dependency and
                // passes the cl_mem to the OpenCL kernel argument.
                auto acc = gb->get_buffer().template get_access<::sycl::access::mode::read_write>(cgh);
                cgh.set_arg(arg_idx, acc);
            } else if (auto* gu = dynamic_cast<gpu_usm*>(arg.mem.get())) {
                // USM pointer: pass the device pointer directly.
                void* ptr = gu->buffer_ptr();
                cgh.set_arg(arg_idx, ptr);
            } else {
                OPENVINO_THROW("[GPU] sycl_kernel::launch: unknown memory type at arg index ", arg_idx);
            }
        } else if (arg.kind == arg_t::kind_t::LOCAL_MEM) {
            ::sycl::local_accessor<std::byte, 1> local_acc(arg.local_size, cgh);
            cgh.set_arg(arg_idx, local_acc);
        } else if (arg.kind == arg_t::kind_t::SCALAR) {
            // Pass the scalar value with the correct C++ type so SYCL can
            // determine the correct kernel argument size.
            switch (arg.scalar.t) {
                case scalar_desc::Types::UINT8:
                    cgh.set_arg(arg_idx, arg.scalar.v.u8);   break;
                case scalar_desc::Types::UINT16:
                    cgh.set_arg(arg_idx, arg.scalar.v.u16);  break;
                case scalar_desc::Types::UINT32:
                    cgh.set_arg(arg_idx, arg.scalar.v.u32);  break;
                case scalar_desc::Types::UINT64:
                    cgh.set_arg(arg_idx, arg.scalar.v.u64);  break;
                case scalar_desc::Types::INT8:
                    cgh.set_arg(arg_idx, arg.scalar.v.s8);   break;
                case scalar_desc::Types::INT16:
                    cgh.set_arg(arg_idx, arg.scalar.v.s16);  break;
                case scalar_desc::Types::INT32:
                    cgh.set_arg(arg_idx, arg.scalar.v.s32);  break;
                case scalar_desc::Types::INT64:
                    cgh.set_arg(arg_idx, arg.scalar.v.s64);  break;
                case scalar_desc::Types::FLOAT32:
                    cgh.set_arg(arg_idx, arg.scalar.v.f32);  break;
                case scalar_desc::Types::FLOAT64:
                    cgh.set_arg(arg_idx, arg.scalar.v.f64);  break;
                default:
                    OPENVINO_THROW("[GPU] sycl_kernel::launch: "
                                   "unknown scalar type at arg index ", arg_idx);
            }
        }
        ++arg_idx;
    }

    // Build the nd_range from the work-group sizes stored in args_desc.
    // OpenVINO follows OpenCL convention: gws[0]=X (fastest), gws[1]=Y, gws[2]=Z (slowest).
    // SYCL nd_range<3>(r0,r1,r2) uses the opposite convention (dim 0 = slowest) and
    // passes OpenCL globalWorkSize={r2, r1, r0}.  To preserve the OpenCL IDs seen by
    // the kernel we must reverse the dimensions:
    //   SYCL range<3>(g2, g1, g0) → OpenCL globalWorkSize = {g0, g1, g2}
    const size_t g0 = gws.size() > 0 ? gws[0] : 1;
    const size_t g1 = gws.size() > 1 ? gws[1] : 1;
    const size_t g2 = gws.size() > 2 ? gws[2] : 1;
    const size_t l0 = lws.size() > 0 ? lws[0] : 1;
    const size_t l1 = lws.size() > 1 ? lws[1] : 1;
    const size_t l2 = lws.size() > 2 ? lws[2] : 1;

    ::sycl::nd_range<3> ndr(
        ::sycl::range<3>(g2, g1, g0),
        ::sycl::range<3>(l2, l1, l0));

    cgh.parallel_for(ndr, _compiled_kernel);
}

void sycl_kernel::create_kernels(const ::sycl::context& ctx,
                                 const std::vector<std::byte>& spirv_binary,
                                 const std::string& build_log,
                                 std::vector<kernel::ptr>& out) {
    namespace syclex = ::sycl::ext::oneapi::experimental;

    // extract_spirv_kernel_names throws if the binary is not a valid SPIR-V module.
    auto kernel_names = extract_spirv_kernel_names(spirv_binary.data(), spirv_binary.size());

    // Drop kernel names that are not executable
    kernel_names.erase(std::remove_if(kernel_names.begin(), kernel_names.end(),
                                      is_ignored_kernel_name),
                       kernel_names.end());

    if (kernel_names.empty()) {
        OPENVINO_THROW("[GPU] sycl_kernel::create_kernels: no executable kernel entry points found in SPIR-V binary");
    }

    try {
        auto src_bundle = syclex::create_kernel_bundle_from_source(ctx,
                                                                   syclex::source_language::spirv,
                                                                   spirv_binary);
        auto exec_bundle = syclex::build(src_bundle);

        for (const auto& kernel_name : kernel_names) {
            ::sycl::kernel k = exec_bundle.ext_oneapi_get_kernel(kernel_name);
            out.push_back(std::make_shared<sycl_kernel>(k, kernel_name, spirv_binary, build_log));
        }
    } catch (const ::sycl::exception& e) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(e));
    }
}

}  // namespace sycl
}  // namespace cldnn

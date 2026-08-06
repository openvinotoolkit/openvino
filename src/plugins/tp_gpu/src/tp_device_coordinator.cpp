// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tp_gpu/tp_device_coordinator.hpp"

#include <atomic>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#define CL_TARGET_OPENCL_VERSION 220
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "openvino/core/except.hpp"
#include "openvino/zero_api.hpp"
#include "tp_embedded_kernels.h"
#include "tp_l0_shared_context.hpp"

// ---- Inline declarations for the Intel L0 counter-based event extension
//      (zexCounterBasedEventCreate2).  Mirrors the public headers shipped
//      in intel/compute-runtime; vendored here to avoid build-time
//      dependency on the intel_gpu plugin source tree.
namespace {
constexpr uint32_t kZexStructureCounterBasedEventDesc = 0x0003001C;

constexpr uint32_t kZexCbEventFlagImmediate    = 1u << 0;
constexpr uint32_t kZexCbEventFlagNonImmediate = 1u << 1;
constexpr uint32_t kZexCbEventFlagHostVisible  = 1u << 2;

struct zex_counter_based_event_desc_t {
    uint32_t                stype;
    const void*             pNext;
    uint32_t                flags;
    ze_event_scope_flags_t  signalScope;
    ze_event_scope_flags_t  waitScope;
};
}  // namespace

namespace ov {
namespace tp_gpu {

namespace {

inline void ze_throw(ze_result_t r, const char* what) {
    if (r != ZE_RESULT_SUCCESS) {
        OPENVINO_THROW("[TP][L0] ", what, " failed: 0x", std::hex, r);
    }
}
#define ZE_THROW(expr) ::ov::tp_gpu::ze_throw((expr), #expr)

// Lazily resolved Intel L0 extension entry point for counter-based events.
// Resolved on first init_rank when m_use_immediate is true.
using pfn_zexCounterBasedEventCreate2 =
    ze_result_t (*)(ze_context_handle_t, ze_device_handle_t,
                    const zex_counter_based_event_desc_t*, ze_event_handle_t*);
pfn_zexCounterBasedEventCreate2 g_zexCounterBasedEventCreate2 = nullptr;
std::once_flag g_cb_ev_init_flag;

void resolve_counter_based_event_create(ze_driver_handle_t driver) {
    void* fp = nullptr;
    ze_result_t r = ov::zeDriverGetExtensionFunctionAddress(
        driver, "zexCounterBasedEventCreate2", &fp);
    OPENVINO_ASSERT(r == ZE_RESULT_SUCCESS && fp,
                    "[TP][L0] zexCounterBasedEventCreate2 extension not available");
    g_zexCounterBasedEventCreate2 = reinterpret_cast<pfn_zexCounterBasedEventCreate2>(fp);
}

// Per-process gate for kernel-timestamp event creation/reset/query.  When
// disabled (the default), allreduce skips per-call zeEventHostReset on
// timestamp events and avoids signalling the kernel timestamp probe in
// AppendLaunchKernel.  Enabled by setting any non-empty TP_PROF env var.
bool tp_profiling_enabled() {
    static const bool on = std::getenv("TP_PROF") != nullptr;
    return on;
}

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, const char* what) {
    OPENVINO_ASSERT(rhs == 0 || lhs <= std::numeric_limits<std::size_t>::max() / rhs,
                    "[TP][L0] ", what, " byte count overflow: ", lhs, " * ", rhs);
    return lhs * rhs;
}

std::size_t collective_payload_bytes(std::size_t n, ov::element::Type dtype) {
    OPENVINO_ASSERT(dtype == ov::element::f16 || dtype == ov::element::f32,
                    "[TP][L0] AllReduce supports f16/f32 only, got ", dtype);
    OPENVINO_ASSERT(n > 0, "[TP][L0] zero-sized AllReduce is not supported");
    OPENVINO_ASSERT(n <= std::numeric_limits<uint32_t>::max(),
                    "[TP][L0] AllReduce element count exceeds Level Zero launch range: ", n);
    return checked_multiply(n, dtype.size(), "collective payload");
}

void select_compute_ordinal(ze_device_handle_t dev, uint32_t& ordinal) {
    uint32_t qg_count = 0;
    ZE_THROW(ov::zeDeviceGetCommandQueueGroupProperties(dev, &qg_count, nullptr));
    std::vector<ze_command_queue_group_properties_t> qgp(
        qg_count, {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr});
    ZE_THROW(ov::zeDeviceGetCommandQueueGroupProperties(dev, &qg_count, qgp.data()));
    ordinal = 0;
    for (uint32_t g = 0; g < qg_count; ++g) {
        if (qgp[g].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            ordinal = g;
            return;
        }
    }
    OPENVINO_THROW("[TP][L0] no compute-capable queue group found");
}

// Pick a copy-only queue group (dedicated DMA engine).  Returns false when
// the device exposes no copy-only group, in which case the caller should
// route memcpy onto the compute queue.
bool select_copy_ordinal(ze_device_handle_t dev, uint32_t& ordinal) {
    uint32_t qg_count = 0;
    ZE_THROW(ov::zeDeviceGetCommandQueueGroupProperties(dev, &qg_count, nullptr));
    std::vector<ze_command_queue_group_properties_t> qgp(
        qg_count, {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr});
    ZE_THROW(ov::zeDeviceGetCommandQueueGroupProperties(dev, &qg_count, qgp.data()));
    for (uint32_t g = 0; g < qg_count; ++g) {
        const bool copy    = (qgp[g].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COPY) != 0;
        const bool compute = (qgp[g].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) != 0;
        if (copy && !compute) {
            ordinal = g;
            return true;
        }
    }
    return false;
}

bool tp_use_copy_engine() {
    // Default: disabled.  On dual-Arc the dedicated copy engine yields a
    // small (~5%) drop in PCIe memcpy time for large prefill transfers,
    // but each call also costs an extra zeCommandQueueExecuteCommandLists
    // (~15us host overhead per rank) which dominates on small decode
    // transfers (n=1) and ends up net-negative for a typical 1-prefill /
    // many-decodes workload.  Set TP_COPY_ENGINE=1 to opt in (useful for
    // prefill-bound benchmarks or workloads with large per-call transfers).
    static const bool on = std::getenv("TP_COPY_ENGINE") != nullptr;
    return on;
}

// cl_khr_device_uuid: clGetDeviceInfo with this name returns 16 bytes.
#ifndef CL_DEVICE_UUID_KHR
#  define CL_DEVICE_UUID_KHR 0x106A
#endif

// Build the kernel module via OpenCL on the device whose UUID matches the
// supplied L0 device, then return the native binary.
//
// We use OpenCL because Intel's L0 driver does not expose the OCLC compiler
// extension (zeModuleCreate with format=3 returns INVALID_ENUMERATION on
// every context configuration we tried).  intel_gpu's ze_kernel_builder
// applies the same fallback when check_l0_build_support() fails.
std::vector<uint8_t> compile_via_ocl(ze_device_handle_t ze_dev,
                                     const char* src,
                                     size_t /*src_bytes*/) {
    ze_device_properties_t zp{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, nullptr};
    ZE_THROW(ov::zeDeviceGetProperties(ze_dev, &zp));
    // ze_device_uuid_t is 16 raw bytes (matches cl_khr_device_uuid layout).

    cl_uint num_platforms = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    OPENVINO_ASSERT(err == CL_SUCCESS && num_platforms > 0,
                    "[TP][OCL] clGetPlatformIDs failed: ", err);
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    OPENVINO_ASSERT(err == CL_SUCCESS, "[TP][OCL] clGetPlatformIDs: ", err);

    cl_device_id matched = nullptr;
    for (auto p : platforms) {
        cl_uint nd = 0;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || nd == 0) {
            continue;
        }
        std::vector<cl_device_id> devs(nd);
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, devs.data(), nullptr) != CL_SUCCESS) {
            continue;
        }
        for (auto d : devs) {
            uint8_t uuid[16] = {0};
            size_t got = 0;
            if (clGetDeviceInfo(d, CL_DEVICE_UUID_KHR, sizeof(uuid), uuid, &got) != CL_SUCCESS ||
                got != sizeof(uuid)) {
                continue;
            }
            if (std::memcmp(uuid, &zp.uuid, 16) == 0) {
                matched = d;
                break;
            }
        }
        if (matched) break;
    }
    OPENVINO_ASSERT(matched,
                    "[TP][OCL] no OpenCL device matches the L0 device UUID");

    cl_context_properties cprops[] = {0};
    cl_context cctx = clCreateContext(cprops, 1, &matched, nullptr, nullptr, &err);
    OPENVINO_ASSERT(err == CL_SUCCESS && cctx,
                    "[TP][OCL] clCreateContext failed: ", err);

    const char* sources[] = {src};
    cl_program prog = clCreateProgramWithSource(cctx, 1, sources, nullptr, &err);
    if (err != CL_SUCCESS || !prog) {
        clReleaseContext(cctx);
        OPENVINO_THROW("[TP][OCL] clCreateProgramWithSource failed: ", err);
    }

    err = clBuildProgram(prog, 1, &matched, "", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t lsz = 0;
        clGetProgramBuildInfo(prog, matched, CL_PROGRAM_BUILD_LOG, 0, nullptr, &lsz);
        std::string log(lsz, ' ');
        if (lsz) {
            clGetProgramBuildInfo(prog, matched, CL_PROGRAM_BUILD_LOG, lsz, log.data(), nullptr);
        }
        clReleaseProgram(prog);
        clReleaseContext(cctx);
        OPENVINO_THROW("[TP][OCL] clBuildProgram failed: ", err, " log: ", log);
    }

    cl_uint nbins = 0;
    clGetProgramInfo(prog, CL_PROGRAM_NUM_DEVICES, sizeof(nbins), &nbins, nullptr);
    OPENVINO_ASSERT(nbins == 1, "[TP][OCL] expected 1 program binary, got ", nbins);

    size_t bin_size = 0;
    err = clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(bin_size), &bin_size, nullptr);
    OPENVINO_ASSERT(err == CL_SUCCESS && bin_size > 0,
                    "[TP][OCL] CL_PROGRAM_BINARY_SIZES failed: ", err);

    std::vector<uint8_t> bin(bin_size);
    uint8_t* bins[1] = {bin.data()};
    err = clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(bins), bins, nullptr);

    clReleaseProgram(prog);
    clReleaseContext(cctx);

    OPENVINO_ASSERT(err == CL_SUCCESS, "[TP][OCL] CL_PROGRAM_BINARIES failed: ", err);
    return bin;
}

}  // namespace

TPDeviceCoordinator::TPDeviceCoordinator(TPL0SharedContextPtr shared,
                                         int world_size,
                                         int num_collectives)
    : m_shared(std::move(shared)),
      m_world_size(world_size),
      m_num_collectives(num_collectives) {
    OPENVINO_ASSERT(m_shared && m_shared->context && static_cast<int>(m_shared->devices.size()) == world_size,
                    "[TP][L0] coordinator requires a valid shared L0 context with ", world_size, " devices");
    OPENVINO_ASSERT(world_size >= 2, "[TP][L0] coordinator requires at least 2 ranks");

    if (const char* v = std::getenv("TP_USE_IMMEDIATE")) {
        m_use_immediate = std::atoi(v) != 0;
    }

    m_ranks.resize(world_size);
    for (int r = 0; r < world_size; ++r) {
        m_ranks[r].device = m_shared->devices[r];
    }
    m_scratch.allocations.assign(world_size, nullptr);
    m_scratch.bytes_per_rank.assign(world_size, 0);

    try {
        for (int r = 0; r < world_size; ++r) {
            init_rank(m_ranks[r]);
        }
    } catch (...) {
        for (auto& rs : m_ranks) destroy_rank(rs);
        throw;
    }

    m_rendezvous.resize(num_collectives);
    m_plans.resize(num_collectives);
    for (int i = 0; i < num_collectives; ++i) {
        auto rdz = std::make_unique<Rendezvous>();
        rdz->in_ptrs.assign(world_size, nullptr);
        rdz->out_ptrs.assign(world_size, nullptr);
        m_rendezvous[i] = std::move(rdz);
    }

    m_ready = true;
    if (tp_profiling_enabled()) {
        std::cerr << "[TP][L0] coordinator ready: " << world_size << " ranks, "
                  << num_collectives << " collective slots"
                  << (m_use_immediate ? " (immediate cmdlists)" : " (regular cmdlists)")
                  << std::endl;
    }
}

TPDeviceCoordinator::~TPDeviceCoordinator() {
    for (auto& p : m_plans) {
        if (p) destroy_plan(*p);
    }
    destroy_scratch();
    for (auto& rs : m_ranks) {
        destroy_rank(rs);
    }
}

TPDeviceCoordinator::ScratchStats TPDeviceCoordinator::get_scratch_stats() const {
    std::lock_guard<std::mutex> lock(m_scratch_mutex);
    ScratchStats stats;
    stats.payload_capacity_bytes = m_scratch.payload_capacity_bytes;
    stats.total_allocated_bytes = m_scratch.total_allocated_bytes;
    stats.allocated_bytes_per_rank = m_scratch.bytes_per_rank;
    stats.generation = m_scratch.generation;
    stats.growth_count = m_scratch.growth_count;
    stats.allocation_count = m_scratch.allocation_count;
    return stats;
}

void TPDeviceCoordinator::init_rank(RankState& rs) {
    auto ctx = m_shared->context;
    auto dev = rs.device;

    select_compute_ordinal(dev, rs.compute_ordinal);

    // Probe device timer for kernel-timestamp -> ns conversion.
    // ze_device_properties_t::timerResolution is documented as either
    // cycles/sec or ns/tick depending on the API revision; on Intel GPUs
    // (BMG, ARC, Xe) it is reported as ns/tick directly (e.g. 52 ns/tick
    // for the standard 19.2 MHz GT timer).
    {
        ze_device_properties_t dp{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, nullptr};
        ZE_THROW(ov::zeDeviceGetProperties(dev, &dp));
        rs.timer_ns_per_tick = dp.timerResolution > 0 ? dp.timerResolution : 1;
        const uint32_t bits = dp.kernelTimestampValidBits;
        rs.timestamp_mask = (bits == 0 || bits >= 64) ? ~uint64_t{0}
                                                       : ((uint64_t{1} << bits) - 1);
    }

    ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr};
    qd.ordinal  = rs.compute_ordinal;
    qd.mode     = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    if (m_use_immediate) {
        // Immediate cmdlist: appended commands begin executing as soon as
        // they are recorded; there is no queue and no ExecuteCommandLists
        // submission step.  The qd descriptor is reused to specify ordinal
        // and async semantics for the underlying execution engine.
        ZE_THROW(ov::zeCommandListCreateImmediate(ctx, dev, &qd, &rs.compute_list));
    } else {
        ZE_THROW(ov::zeCommandQueueCreate(ctx, dev, &qd, &rs.compute_queue));
        ze_command_list_desc_t ld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr};
        ld.commandQueueGroupOrdinal = rs.compute_ordinal;
        ZE_THROW(ov::zeCommandListCreate(ctx, dev, &ld, &rs.compute_list));

        // Optional dedicated copy engine for the cross-device memcpy step.
        if (tp_use_copy_engine() && select_copy_ordinal(dev, rs.copy_ordinal)) {
            rs.has_dedicated_copy = true;
            ze_command_queue_desc_t cqd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr};
            cqd.ordinal  = rs.copy_ordinal;
            cqd.mode     = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
            cqd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
            ZE_THROW(ov::zeCommandQueueCreate(ctx, dev, &cqd, &rs.copy_queue));
            ze_command_list_desc_t cld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr};
            cld.commandQueueGroupOrdinal = rs.copy_ordinal;
            ZE_THROW(ov::zeCommandListCreate(ctx, dev, &cld, &rs.copy_list));
        }
    }

    // Build kernel module.
    //
    // Intel's L0 driver does not expose the OCLC compiler extension on this
    // configuration (zeModuleCreate with format=3 returns
    // ZE_RESULT_ERROR_INVALID_ENUMERATION regardless of context arity).
    // intel_gpu observes the same and falls back to OpenCL; we do the same:
    //   1. Match the L0 device to its OpenCL counterpart by UUID.
    //   2. Build from CL source via clBuildProgram.
    //   3. Pull the native binary via CL_PROGRAM_BINARIES.
    //   4. Load the native binary into the shared multi-device L0 context as
    //      ZE_MODULE_FORMAT_NATIVE.
    // Native binaries are bound to (driver, device); reloading into a
    // different L0 context on the same device is valid.
    const char* src = ov::tp_gpu::kernels::allreduce_sum_cl;
    size_t src_bytes = std::strlen(src) + 1;

    std::vector<uint8_t> native_bin = compile_via_ocl(dev, src, src_bytes);

    // Reload into shared context as native binary.
    {
        ze_module_desc_t md{ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr};
        md.format        = ZE_MODULE_FORMAT_NATIVE;
        md.inputSize     = native_bin.size();
        md.pInputModule  = native_bin.data();
        md.pBuildFlags   = "";
        md.pConstants    = nullptr;

        ze_module_build_log_handle_t blog = nullptr;
        ze_result_t br = ov::zeModuleCreate(ctx, dev, &md, &rs.module, &blog);
        if (br != ZE_RESULT_SUCCESS) {
            std::string log;
            if (blog) {
                size_t lsz = 0;
                ov::zeModuleBuildLogGetString(blog, &lsz, nullptr);
                if (lsz) {
                    log.resize(lsz, ' ');
                    ov::zeModuleBuildLogGetString(blog, &lsz, log.data());
                }
                ov::zeModuleBuildLogDestroy(blog);
            }
            OPENVINO_THROW("[TP][L0] zeModuleCreate(NATIVE, shared-ctx) failed: 0x",
                           std::hex, br, " log: ", log);
        }
        if (blog) ov::zeModuleBuildLogDestroy(blog);
    }

    ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr};
    kd.pKernelName = "allreduce_sum_f16";
    ZE_THROW(ov::zeKernelCreate(rs.module, &kd, &rs.kernel_f16));

    kd.pKernelName = "allreduce_sum_f32";
    ZE_THROW(ov::zeKernelCreate(rs.module, &kd, &rs.kernel_f32));

    if (m_use_immediate) {
        std::call_once(g_cb_ev_init_flag,
                       resolve_counter_based_event_create, m_shared->driver);
        zex_counter_based_event_desc_t cd{};
        cd.stype = kZexStructureCounterBasedEventDesc;
        cd.pNext = nullptr;
        // IMMEDIATE flag selects the immediate-cmdlist signaling fast path
        // in the driver; HOST_VISIBLE makes the event observable by host.
        cd.flags = kZexCbEventFlagImmediate | kZexCbEventFlagHostVisible;
        cd.signalScope = ZE_EVENT_SCOPE_FLAG_HOST;
        cd.waitScope   = ZE_EVENT_SCOPE_FLAG_DEVICE;
        ZE_THROW(g_zexCounterBasedEventCreate2(ctx, dev, &cd, &rs.cb_event_done));
    }
}

void TPDeviceCoordinator::destroy_rank(RankState& rs) {
    if (rs.cb_event_done){ ov::zeEventDestroy(rs.cb_event_done);  rs.cb_event_done = nullptr; }
    if (rs.kernel_f16)   { ov::zeKernelDestroy(rs.kernel_f16);   rs.kernel_f16 = nullptr; }
    if (rs.kernel_f32)   { ov::zeKernelDestroy(rs.kernel_f32);   rs.kernel_f32 = nullptr; }
    if (rs.module)       { ov::zeModuleDestroy(rs.module);       rs.module = nullptr; }
    if (rs.copy_list)    { ov::zeCommandListDestroy(rs.copy_list);     rs.copy_list = nullptr; }
    if (rs.copy_queue)   { ov::zeCommandQueueDestroy(rs.copy_queue);   rs.copy_queue = nullptr; }
    if (rs.compute_list) { ov::zeCommandListDestroy(rs.compute_list);  rs.compute_list = nullptr; }
    if (rs.compute_queue){ ov::zeCommandQueueDestroy(rs.compute_queue); rs.compute_queue = nullptr; }
}

void TPDeviceCoordinator::destroy_plan(Plan& plan) {
    // Before destroying any GPU resources, make sure no in-flight work
    // from the previous execute_plan is still using them.  Without this
    // sync, zeCommandListReset / zeMemFree on a still-pending cmdlist
    // can deadlock on shared multi-device contexts.
    if (m_ready) {
        for (auto& rs : m_ranks) {
            if (rs.compute_queue) {
                ov::zeCommandQueueSynchronize(rs.compute_queue, UINT64_MAX);
            }
            if (rs.copy_queue) {
                ov::zeCommandQueueSynchronize(rs.copy_queue, UINT64_MAX);
            }
        }
    }

    for (auto& e : plan.ev_recv)  if (e) ov::zeEventDestroy(e);
    for (auto& e : plan.ev_bcast) if (e) ov::zeEventDestroy(e);
    if (plan.ev_reduce) ov::zeEventDestroy(plan.ev_reduce);
    if (plan.pool)      ov::zeEventPoolDestroy(plan.pool);
    for (auto& e : plan.ev_ts_copy)   if (e) ov::zeEventDestroy(e);
    for (auto& e : plan.ev_ts_kernel) if (e) ov::zeEventDestroy(e);
    if (plan.ts_pool) ov::zeEventPoolDestroy(plan.ts_pool);
    if (m_recorded_plan == &plan) {
        m_recorded_plan = nullptr;
        m_recorded_scratch_generation = 0;
    }
    plan.ev_recv.clear();
    plan.ev_bcast.clear();
    plan.ev_reduce = nullptr;
    plan.pool = nullptr;
    plan.ev_ts_copy.clear();
    plan.ev_ts_kernel.clear();
    plan.ts_pool = nullptr;
    plan.in_ptrs.clear();
    plan.out_ptrs.clear();
    plan.n = 0;
    plan.dtype = ov::element::dynamic;
    plan.max_payload_bytes = 0;
}

bool TPDeviceCoordinator::ensure_scratch_capacity(std::size_t payload_bytes) {
    std::lock_guard<std::mutex> lock(m_scratch_mutex);
    if (payload_bytes <= m_scratch.payload_capacity_bytes) {
        return false;
    }

    std::vector<void*> new_allocations(static_cast<std::size_t>(m_world_size), nullptr);
    std::vector<std::size_t> new_bytes_per_rank(static_cast<std::size_t>(m_world_size), 0);
    ze_device_mem_alloc_desc_t mad{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr};

    try {
        if (m_world_size == 2) {
            for (int rank = 0; rank < m_world_size; ++rank) {
                ZE_THROW(ov::zeMemAllocDevice(m_shared->context,
                                              &mad,
                                              payload_bytes,
                                              64,
                                              m_ranks[rank].device,
                                              &new_allocations[rank]));
                new_bytes_per_rank[rank] = payload_bytes;
            }
        } else {
            const auto packed_bytes = checked_multiply(payload_bytes,
                                                       static_cast<std::size_t>(m_world_size - 1),
                                                       "packed scratch");
            ZE_THROW(ov::zeMemAllocDevice(m_shared->context,
                                          &mad,
                                          packed_bytes,
                                          64,
                                          m_ranks[0].device,
                                          &new_allocations[0]));
            new_bytes_per_rank[0] = packed_bytes;
        }
    } catch (...) {
        for (auto* ptr : new_allocations) {
            if (ptr) {
                ov::zeMemFree(m_shared->context, ptr);
            }
        }
        throw;
    }

    try {
        // Every execute path synchronizes before returning.  Synchronize
        // again before replacing addresses embedded in recorded lists.  If
        // synchronization fails, discard only the newly allocated arena and
        // keep the old one intact.
        for (auto& rs : m_ranks) {
            if (rs.compute_queue) {
                ZE_THROW(ov::zeCommandQueueSynchronize(rs.compute_queue, UINT64_MAX));
            }
            if (rs.copy_queue) {
                ZE_THROW(ov::zeCommandQueueSynchronize(rs.copy_queue, UINT64_MAX));
            }
        }
    } catch (...) {
        for (auto* ptr : new_allocations) {
            if (ptr) {
                ov::zeMemFree(m_shared->context, ptr);
            }
        }
        throw;
    }

    for (auto* ptr : m_scratch.allocations) {
        if (ptr) {
            ov::zeMemFree(m_shared->context, ptr);
        }
    }

    m_scratch.allocations = std::move(new_allocations);
    m_scratch.bytes_per_rank = std::move(new_bytes_per_rank);
    m_scratch.payload_capacity_bytes = payload_bytes;
    m_scratch.total_allocated_bytes = 0;
    for (const auto bytes : m_scratch.bytes_per_rank) {
        m_scratch.total_allocated_bytes += bytes;
    }
    ++m_scratch.generation;
    ++m_scratch.growth_count;
    m_scratch.allocation_count += m_world_size == 2 ? 2 : 1;
    m_recorded_plan = nullptr;
    m_recorded_scratch_generation = 0;

    if (tp_profiling_enabled()) {
        std::cerr << "[TP][MEM] scratch grow generation=" << m_scratch.generation
                  << " payload_capacity=" << m_scratch.payload_capacity_bytes
                  << " total=" << m_scratch.total_allocated_bytes;
        for (int rank = 0; rank < m_world_size; ++rank) {
            std::cerr << " r" << rank << "=" << m_scratch.bytes_per_rank[rank];
        }
        std::cerr << std::endl;
    }
    return true;
}

void TPDeviceCoordinator::destroy_scratch() {
    std::lock_guard<std::mutex> lock(m_scratch_mutex);
    if (m_shared && m_shared->context) {
        for (auto*& ptr : m_scratch.allocations) {
            if (ptr) {
                ov::zeMemFree(m_shared->context, ptr);
                ptr = nullptr;
            }
        }
    }
    std::fill(m_scratch.bytes_per_rank.begin(), m_scratch.bytes_per_rank.end(), 0);
    m_scratch.payload_capacity_bytes = 0;
    m_scratch.total_allocated_bytes = 0;
    m_recorded_plan = nullptr;
    m_recorded_scratch_generation = 0;
}

void* TPDeviceCoordinator::scratch_buffer(int index) const {
    if (m_world_size == 2) {
        OPENVINO_ASSERT(index >= 0 && index < m_world_size,
                        "[TP][L0] scratch rank out of range: ", index);
        return m_scratch.allocations[static_cast<std::size_t>(index)];
    }

    OPENVINO_ASSERT(index >= 0 && index < m_world_size - 1,
                    "[TP][L0] scratch worker out of range: ", index);
    auto* base = static_cast<uint8_t*>(m_scratch.allocations[0]);
    return base + static_cast<std::size_t>(index) * m_scratch.payload_capacity_bytes;
}

void TPDeviceCoordinator::build_plan(int /*collective_id*/,
                                     const std::vector<void*>& in_ptrs,
                                     const std::vector<void*>& out_ptrs,
                                     std::size_t n,
                                     ov::element::Type dtype,
                                     Plan& plan) {
    OPENVINO_ASSERT(dtype == ov::element::f16 || dtype == ov::element::f32,
                    "[TP][L0] AllReduce supports f16/f32 only, got ", dtype);

    auto ctx = m_shared->context;
    const int N = m_world_size;
    const std::size_t payload_bytes = collective_payload_bytes(n, dtype);

    plan.in_ptrs  = in_ptrs;
    plan.out_ptrs = out_ptrs;
    plan.n        = n;
    plan.dtype    = dtype;

    OPENVINO_ASSERT(m_scratch.payload_capacity_bytes >= payload_bytes,
                    "[TP][L0] scratch arena is smaller than collective payload");

    // Event resources are pointer/shape independent and remain cached per
    // collective.  Device staging is coordinator-owned and shared.
    const bool resources_already_built = plan.pool != nullptr;
    if (resources_already_built) {
        return;
    }

    if (N == 2) {
        // Symmetric N=2: every rank pushes its `in` to peer's local staging,
        // then runs its own reduce kernel `out = in + staging` on its queue.
        // Both queues fire in parallel — critical path is one cross-device
        // memcpy + one kernel, vs three-step funnel through rank 0.
        // We use a single KERNEL_TIMESTAMP pool for everything: the recv
        // events double as memcpy-end probes, plus one timestamp event per
        // rank for the reduce kernel.
        ze_event_pool_desc_t epd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr};
        // Keep KERNEL_TIMESTAMP on the pool unconditionally: some Intel L0
        // drivers route cross-device device-scope event waits differently
        // for timestamp pools, and the configuration that is exercised in
        // CI uses this flag for ev_recv even when only used as a plain
        // signal/wait pair.  The cost of the flag is per-signal timestamp
        // recording in the GPU command processor — small but non-zero.
        // The actual P-6 win comes from skipping ev_ts_kernel events.
        epd.flags = ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
        // ev_recv[2] always; ev_ts_kernel[2] only when profiling is enabled.
        const bool prof_on = tp_profiling_enabled();
        epd.count = prof_on ? 4u : 2u;
        std::vector<ze_device_handle_t> devs_nc(m_shared->devices.begin(), m_shared->devices.end());
        ZE_THROW(ov::zeEventPoolCreate(ctx, &epd,
                                       static_cast<uint32_t>(devs_nc.size()),
                                       devs_nc.data(), &plan.pool));

        plan.ev_recv.resize(2);   // [r] = "rank r finished pushing"
        plan.ev_bcast.clear();
        plan.ev_reduce = nullptr;
        for (int r = 0; r < 2; ++r) {
            ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr};
            ed.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
            ed.wait   = ZE_EVENT_SCOPE_FLAG_DEVICE;
            ed.index  = static_cast<uint32_t>(r);
            ZE_THROW(ov::zeEventCreate(plan.pool, &ed, &plan.ev_recv[r]));
        }

        // Reuse plan.pool for kernel-end timestamp events.
        plan.ts_pool = nullptr;  // single pool path
        plan.ev_ts_copy.clear();  // ev_recv[] already serves as copy-end probe
        // Always size to 2 so record_plan can index plan.ev_ts_kernel[r] —
        // entries stay null when profiling is disabled and act as a "no
        // signal" sentinel for AppendLaunchKernel.
        plan.ev_ts_kernel.assign(2, nullptr);
        if (prof_on) {
            for (int r = 0; r < 2; ++r) {
                ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr};
                ed.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
                ed.wait   = ZE_EVENT_SCOPE_FLAG_HOST;
                ed.index  = static_cast<uint32_t>(2 + r);
                ZE_THROW(ov::zeEventCreate(plan.pool, &ed, &plan.ev_ts_kernel[r]));
            }
        }

        return;
    }

    // ---- Legacy N>2 main-funnel path ----
    const int W = N - 1;
    // Event pool: W recv + 1 reduce + W bcast.
    const uint32_t total_events = static_cast<uint32_t>(2 * W + 1);
    ze_event_pool_desc_t epd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr};
    epd.flags = 0;
    epd.count = total_events;
    std::vector<ze_device_handle_t> devs_nc(m_shared->devices.begin(), m_shared->devices.end());
    ZE_THROW(ov::zeEventPoolCreate(ctx, &epd,
                                   static_cast<uint32_t>(devs_nc.size()),
                                   devs_nc.data(), &plan.pool));

    plan.ev_recv.resize(W);
    plan.ev_bcast.resize(W);
    uint32_t idx = 0;
    auto mk = [&](ze_event_handle_t& e) {
        ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr};
        ed.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
        ed.wait   = ZE_EVENT_SCOPE_FLAG_DEVICE;
        ed.index  = idx++;
        ZE_THROW(ov::zeEventCreate(plan.pool, &ed, &e));
    };
    for (int w = 0; w < W; ++w) mk(plan.ev_recv[w]);
    mk(plan.ev_reduce);
    for (int w = 0; w < W; ++w) mk(plan.ev_bcast[w]);

}

void TPDeviceCoordinator::record_plan(Plan& plan) {
    constexpr uint32_t kGroupSize = 256;
    const int N = m_world_size;
    const std::size_t n = plan.n;
    const std::size_t bytes = collective_payload_bytes(n, plan.dtype);

    // On the immediate path nothing is recorded ahead of time \u2014 cmdlists
    // are appended to and consumed inside execute_plan.  zeCommandListReset
    // is also not allowed on immediate cmdlists.
    if (m_use_immediate) {
        return;
    }

    // Reset all rank command lists (must be done before re-recording).
    // zeCommandListReset on a list that still has work in-flight on its
    // queue is undefined; on shared multi-device L0 contexts it can
    // deadlock.  Sync each queue first so the previous submission has
    // fully drained before we wipe the recorded commands.
    for (auto& rs : m_ranks) {
        if (rs.compute_queue) {
            ZE_THROW(ov::zeCommandQueueSynchronize(rs.compute_queue, UINT64_MAX));
        }
        if (rs.copy_queue) {
            ZE_THROW(ov::zeCommandQueueSynchronize(rs.copy_queue, UINT64_MAX));
        }
        ZE_THROW(ov::zeCommandListReset(rs.compute_list));
        if (rs.copy_list) {
            ZE_THROW(ov::zeCommandListReset(rs.copy_list));
        }
    }

    if (N == 2) {
        uint32_t items = static_cast<uint32_t>(n);
        ze_group_count_t gc{(items + kGroupSize - 1) / kGroupSize, 1, 1};
        uint64_t cn64 = n;

        for (int r = 0; r < 2; ++r) {
            auto& self = m_ranks[r];
            const int peer = 1 - r;

            ze_kernel_handle_t kernel =
                (plan.dtype == ov::element::f16) ? self.kernel_f16 : self.kernel_f32;
            ZE_THROW(ov::zeKernelSetGroupSize(kernel, kGroupSize, 1, 1));

            // 1. Push our `in` to peer's local staging (source-side memcpy).
            //    Route onto the dedicated copy engine when available so
            //    cross-device DMA does not contend with the reduce kernel
            //    on the compute engine.  ev_recv[r] is signaled by the
            //    copy queue; the cross-device wait below resolves on the
            //    peer's compute queue regardless of which engine signaled.
            ze_command_list_handle_t copy_target =
                self.copy_list ? self.copy_list : self.compute_list;
            ZE_THROW(ov::zeCommandListAppendMemoryCopy(
                copy_target,
                scratch_buffer(peer),
                plan.in_ptrs[r],
                bytes,
                plan.ev_recv[r],
                0, nullptr));

            // 2. Wait for peer's push to land in our staging.
            ZE_THROW(ov::zeCommandListAppendWaitOnEvents(self.compute_list,
                                                         1, &plan.ev_recv[peer]));

            // 3. Reduce: out_self = in_self + staging_self.
            void* dst   = plan.out_ptrs[r];
            void* src0  = plan.in_ptrs[r];
            void* src1  = scratch_buffer(r);
            ZE_THROW(ov::zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &dst));
            ZE_THROW(ov::zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &src0));
            ZE_THROW(ov::zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &src1));
            ZE_THROW(ov::zeKernelSetArgumentValue(kernel, 3, sizeof(cn64), &cn64));
            ZE_THROW(ov::zeCommandListAppendLaunchKernel(self.compute_list,
                                                         kernel, &gc,
                                                         plan.ev_ts_kernel[r], 0, nullptr));

            ZE_THROW(ov::zeCommandListClose(self.compute_list));
            if (self.copy_list) {
                ZE_THROW(ov::zeCommandListClose(self.copy_list));
            }
        }
        return;
    }

    // ---- Legacy N>2 main-funnel path ----
    const int W = N - 1;
    auto& main_rs = m_ranks[0];
    ze_kernel_handle_t main_kernel =
        (plan.dtype == ov::element::f16) ? main_rs.kernel_f16 : main_rs.kernel_f32;

    // --- Worker lists: copy local input to rank-0 scratch, wait scatter ---
    for (int w = 0; w < W; ++w) {
        auto& worker = m_ranks[w + 1];
        ZE_THROW(ov::zeCommandListAppendMemoryCopy(
            worker.compute_list,
            scratch_buffer(w),
            plan.in_ptrs[w + 1],
            bytes,
            plan.ev_recv[w],
            0, nullptr));
        // Wait for our scatter to land before the queue-sync returns.
        ZE_THROW(ov::zeCommandListAppendWaitOnEvents(worker.compute_list, 1, &plan.ev_bcast[w]));
    }

    // --- Main compute list: wait recvs, run accumulate kernels, scatter ---
    ZE_THROW(ov::zeKernelSetGroupSize(main_kernel, kGroupSize, 1, 1));

    if (W > 0) {
        ZE_THROW(ov::zeCommandListAppendWaitOnEvents(main_rs.compute_list,
                                                     static_cast<uint32_t>(W),
                                                     plan.ev_recv.data()));
    }

    uint32_t items = static_cast<uint32_t>(n);
    ze_group_count_t gc{(items + kGroupSize - 1) / kGroupSize, 1, 1};
    uint64_t cn64 = n;

    void* dst_main = plan.out_ptrs[0];
    void* src0_main = plan.in_ptrs[0];

    for (int w = 0; w < W; ++w) {
        void* a = (w == 0) ? src0_main : dst_main;
        void* b = scratch_buffer(w);
        ZE_THROW(ov::zeKernelSetArgumentValue(main_kernel, 0, sizeof(void*), &dst_main));
        ZE_THROW(ov::zeKernelSetArgumentValue(main_kernel, 1, sizeof(void*), &a));
        ZE_THROW(ov::zeKernelSetArgumentValue(main_kernel, 2, sizeof(void*), &b));
        ZE_THROW(ov::zeKernelSetArgumentValue(main_kernel, 3, sizeof(cn64), &cn64));
        ze_event_handle_t signal = (w == W - 1) ? plan.ev_reduce : nullptr;
        ZE_THROW(ov::zeCommandListAppendLaunchKernel(main_rs.compute_list,
                                                     main_kernel, &gc,
                                                     signal, 0, nullptr));
        // Every accumulation but the first reads the result of the previous
        // one from dst_main and writes back to it.  A command list created
        // without ZE_COMMAND_LIST_FLAG_IN_ORDER gives no ordering between
        // appended kernels, so without this barrier consecutive launches
        // overlap and the sum loses the contributions still in flight.
        if (w + 1 < W) {
            ZE_THROW(ov::zeCommandListAppendBarrier(main_rs.compute_list, nullptr, 0, nullptr));
        }
    }

    // --- Main scatter: copy result to each worker's out_ptr ---
    for (int w = 0; w < W; ++w) {
        ZE_THROW(ov::zeCommandListAppendWaitOnEvents(main_rs.compute_list, 1, &plan.ev_reduce));
        ZE_THROW(ov::zeCommandListAppendMemoryCopy(
            main_rs.compute_list,
            plan.out_ptrs[w + 1],
            dst_main,
            bytes,
            plan.ev_bcast[w],
            0, nullptr));
    }

    // Close all lists.
    for (auto& rs : m_ranks) {
        ZE_THROW(ov::zeCommandListClose(rs.compute_list));
    }
}

void TPDeviceCoordinator::execute_plan(Plan& plan, ExecStats* stats) {
    static const bool dbg = std::getenv("TP_DBG") != nullptr;
    auto trace = [&](const char* msg) {
        if (dbg) std::cerr << "[TP][L0] " << msg << std::endl;
    };

    using clk = std::chrono::steady_clock;
    auto tr0 = clk::now();
    // Reset events on host (they are signal-once, must be reset between calls).
    trace("execute: reset events");
    for (auto& e : plan.ev_recv)  ZE_THROW(ov::zeEventHostReset(e));
    for (auto& e : plan.ev_bcast) ZE_THROW(ov::zeEventHostReset(e));
    for (auto& e : plan.ev_ts_kernel) if (e) ZE_THROW(ov::zeEventHostReset(e));
    if (plan.ev_reduce) ZE_THROW(ov::zeEventHostReset(plan.ev_reduce));
    auto tr1 = clk::now();
    if (stats) {
        stats->reset = tr1 - tr0;
        // Cross-device memcpy size per rank (each rank pushes its full
        // input to the peer's staging buffer; same on both sides for N=2).
        stats->copy_bytes = collective_payload_bytes(plan.n, plan.dtype);
    }

    if (m_world_size == 2 && m_use_immediate) {
        // Immediate path: append memcpy + wait + kernel + barrier-signal
        // directly into each rank's immediate cmdlist.  The cmdlist
        // executes commands as they are appended; the host waits on the
        // counter-based event signaled by the tail barrier on each rank.
        constexpr uint32_t kGroupSize = 256;
        const std::size_t bytes = collective_payload_bytes(plan.n, plan.dtype);
        const uint32_t items = static_cast<uint32_t>(plan.n);
        ze_group_count_t gc{(items + kGroupSize - 1) / kGroupSize, 1, 1};
        uint64_t cn64 = plan.n;

        auto step = [](const char* what) {
            if (dbg) {
                std::cerr << "[TP][IMM] " << what << std::endl << std::flush;
            }
        };

        auto ts0 = clk::now();
        // Phase A: each rank pushes its `in` to peer's local staging.
        // We append both copies before any wait so they overlap maximally.
        for (int r = 0; r < 2; ++r) {
            auto& self = m_ranks[r];
            const int peer = 1 - r;
            step(r == 0 ? "memcpy r0" : "memcpy r1");
            ZE_THROW(ov::zeCommandListAppendMemoryCopy(
                self.compute_list,
                scratch_buffer(peer),
                plan.in_ptrs[r],
                bytes,
                plan.ev_recv[r],
                0, nullptr));
        }
        // Phase B: wait for peer's push, run reduce, signal tail.
        for (int r = 0; r < 2; ++r) {
            auto& self = m_ranks[r];
            const int peer = 1 - r;
            ze_kernel_handle_t kernel =
                (plan.dtype == ov::element::f16) ? self.kernel_f16 : self.kernel_f32;
            ZE_THROW(ov::zeKernelSetGroupSize(kernel, kGroupSize, 1, 1));

            step(r == 0 ? "wait r0" : "wait r1");
            ZE_THROW(ov::zeCommandListAppendWaitOnEvents(self.compute_list,
                                                         1, &plan.ev_recv[peer]));

            void* dst   = plan.out_ptrs[r];
            void* src0  = plan.in_ptrs[r];
            void* src1  = scratch_buffer(r);
            ZE_THROW(ov::zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &dst));
            ZE_THROW(ov::zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &src0));
            ZE_THROW(ov::zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &src1));
            ZE_THROW(ov::zeKernelSetArgumentValue(kernel, 3, sizeof(cn64), &cn64));
            // Signal the counter-based host-sync event directly from the
            // reduce kernel completion.  Avoids AppendBarrier, which on
            // shared multi-device contexts can attempt cross-device
            // synchronization and deadlock on immediate cmdlists.  Drop
            // ev_ts_kernel on the immediate path \u2014 a single-event signal
            // is what counter-based events are designed for.
            step(r == 0 ? "kernel r0 (signal cb)" : "kernel r1 (signal cb)");
            ZE_THROW(ov::zeCommandListAppendLaunchKernel(self.compute_list,
                                                         kernel, &gc,
                                                         self.cb_event_done,
                                                         0, nullptr));
        }
        step("all appended; host-sync r0");
        auto ts1 = clk::now();
        // 1-second timeout instead of UINT64_MAX so a deadlock on the
        // immediate path surfaces as a thrown error rather than a freeze.
        constexpr uint64_t kSyncTimeoutNs = 1'000'000'000ull;
        ze_result_t s0 = ov::zeEventHostSynchronize(m_ranks[0].cb_event_done, kSyncTimeoutNs);
        if (s0 == ZE_RESULT_NOT_READY) {
            OPENVINO_THROW("[TP][L0] immediate path: rank0 cb_event_done sync timeout (1s)");
        }
        ZE_THROW(s0);
        auto ts2 = clk::now();
        ze_result_t s1 = ov::zeEventHostSynchronize(m_ranks[1].cb_event_done, kSyncTimeoutNs);
        if (s1 == ZE_RESULT_NOT_READY) {
            OPENVINO_THROW("[TP][L0] immediate path: rank1 cb_event_done sync timeout (1s)");
        }
        ZE_THROW(s1);
        auto ts3 = clk::now();
        if (stats) {
            stats->submit = ts1 - ts0;
            stats->sync_a = ts2 - ts1;
            stats->sync_b = ts3 - ts2;

            auto tq0 = clk::now();
            auto ticks_to_ns = [](uint64_t start, uint64_t end,
                                  uint64_t mask, uint64_t ns_per_tick) -> uint64_t {
                const uint64_t s = start & mask;
                const uint64_t e = end   & mask;
                const uint64_t delta = (e >= s) ? (e - s) : ((mask + 1 - s) + e);
                return delta * ns_per_tick;
            };
            uint64_t copy_ns_max = 0;
            uint64_t kern_ns_max = 0;
            for (int r = 0; r < 2; ++r) {
                ze_kernel_timestamp_result_t kt{};
                if (ov::zeEventQueryKernelTimestamp(plan.ev_recv[r], &kt) == ZE_RESULT_SUCCESS) {
                    uint64_t ns = ticks_to_ns(kt.global.kernelStart, kt.global.kernelEnd,
                                              m_ranks[r].timestamp_mask,
                                              m_ranks[r].timer_ns_per_tick);
                    if (ns > copy_ns_max) copy_ns_max = ns;
                }
                if (plan.ev_ts_kernel[r] &&
                    ov::zeEventQueryKernelTimestamp(plan.ev_ts_kernel[r], &kt) == ZE_RESULT_SUCCESS) {
                    uint64_t ns = ticks_to_ns(kt.global.kernelStart, kt.global.kernelEnd,
                                              m_ranks[r].timestamp_mask,
                                              m_ranks[r].timer_ns_per_tick);
                    if (ns > kern_ns_max) kern_ns_max = ns;
                }
            }
            stats->dev_copy   = std::chrono::nanoseconds{copy_ns_max};
            stats->dev_kernel = std::chrono::nanoseconds{kern_ns_max};
            auto tq1 = clk::now();
            stats->dev_ts_query = tq1 - tq0;
        }
        trace("execute: done (immediate)");
        return;
    }

    if (m_world_size == 2) {
        // Symmetric: just submit both queues, then sync.
        // When a dedicated copy engine is present per rank, also submit
        // copy_list to copy_queue.  The compute queue's first command is
        // AppendWaitOnEvents on ev_recv[peer], which is signaled by the
        // peer's copy queue completing — so we only need to host-sync the
        // compute queue here; the copy queue is implicitly drained.
        auto ts0 = clk::now();
        for (int r = 0; r < 2; ++r) {
            if (m_ranks[r].copy_queue) {
                ZE_THROW(ov::zeCommandQueueExecuteCommandLists(
                    m_ranks[r].copy_queue, 1, &m_ranks[r].copy_list, nullptr));
            }
            ZE_THROW(ov::zeCommandQueueExecuteCommandLists(
                m_ranks[r].compute_queue, 1, &m_ranks[r].compute_list, nullptr));
        }
        auto ts1 = clk::now();
        ZE_THROW(ov::zeCommandQueueSynchronize(m_ranks[0].compute_queue, UINT64_MAX));
        auto ts2 = clk::now();
        ZE_THROW(ov::zeCommandQueueSynchronize(m_ranks[1].compute_queue, UINT64_MAX));
        auto ts3 = clk::now();
        if (stats) {
            stats->submit = ts1 - ts0;
            stats->sync_a = ts2 - ts1;
            stats->sync_b = ts3 - ts2;

            // Device-side breakdown via kernel timestamps.
            auto tq0 = clk::now();
            auto ticks_to_ns = [](uint64_t start, uint64_t end,
                                  uint64_t mask, uint64_t ns_per_tick) -> uint64_t {
                const uint64_t s = start & mask;
                const uint64_t e = end   & mask;
                const uint64_t delta = (e >= s) ? (e - s) : ((mask + 1 - s) + e);
                return delta * ns_per_tick;
            };
            uint64_t copy_ns_max = 0;
            uint64_t kern_ns_max = 0;
            for (int r = 0; r < 2; ++r) {
                ze_kernel_timestamp_result_t kt{};
                if (ov::zeEventQueryKernelTimestamp(plan.ev_recv[r], &kt) == ZE_RESULT_SUCCESS) {
                    uint64_t ns = ticks_to_ns(kt.global.kernelStart, kt.global.kernelEnd,
                                              m_ranks[r].timestamp_mask,
                                              m_ranks[r].timer_ns_per_tick);
                    if (ns > copy_ns_max) copy_ns_max = ns;
                }
                if (plan.ev_ts_kernel[r] &&
                    ov::zeEventQueryKernelTimestamp(plan.ev_ts_kernel[r], &kt) == ZE_RESULT_SUCCESS) {
                    uint64_t ns = ticks_to_ns(kt.global.kernelStart, kt.global.kernelEnd,
                                              m_ranks[r].timestamp_mask,
                                              m_ranks[r].timer_ns_per_tick);
                    if (ns > kern_ns_max) kern_ns_max = ns;
                }
            }
            stats->dev_copy   = std::chrono::nanoseconds{copy_ns_max};
            stats->dev_kernel = std::chrono::nanoseconds{kern_ns_max};
            auto tq1 = clk::now();
            stats->dev_ts_query = tq1 - tq0;
        }
        trace("execute: done (sym)");
        return;
    }

    // Submit workers first so their gather copies get going.
    trace("execute: submit workers");
    for (int w = 0; w < m_world_size - 1; ++w) {
        auto& worker = m_ranks[w + 1];
        ZE_THROW(ov::zeCommandQueueExecuteCommandLists(
            worker.compute_queue, 1, &worker.compute_list, nullptr));
    }
    trace("execute: submit main");
    auto& main_rs = m_ranks[0];
    ZE_THROW(ov::zeCommandQueueExecuteCommandLists(
        main_rs.compute_queue, 1, &main_rs.compute_list, nullptr));

    // Sync all queues.
    for (int r = 0; r < m_world_size; ++r) {
        if (dbg) std::cerr << "[TP][L0] execute: sync rank " << r << std::endl;
        ZE_THROW(ov::zeCommandQueueSynchronize(m_ranks[r].compute_queue, UINT64_MAX));
    }
    trace("execute: done");
}

void TPDeviceCoordinator::allreduce(int collective_id,
                                    int rank,
                                    void* in_dev,
                                    void* out_dev,
                                    std::size_t n,
                                    ov::element::Type dtype) {
    OPENVINO_ASSERT(m_ready, "[TP][L0] coordinator not initialized");
    OPENVINO_ASSERT(collective_id >= 0 && collective_id < m_num_collectives,
                    "[TP][L0] collective_id out of range: ", collective_id);
    OPENVINO_ASSERT(rank >= 0 && rank < m_world_size,
                    "[TP][L0] rank out of range: ", rank);

    static const bool dbg = std::getenv("TP_DBG") != nullptr;
    auto trace = [&](const char* msg) {
        if (dbg) std::cerr << "[TP][L0] r" << rank << " " << msg << std::endl;
    };

    // ---- Profiling (env TP_PROF=N: dump aggregated timings every N calls of rank 0) ----
    static const int prof_every = []() -> int {
        if (const char* v = std::getenv("TP_PROF")) {
            int x = std::atoi(v);
            return x > 0 ? x : 0;
        }
        return 0;
    }();
    using clk = std::chrono::steady_clock;
    static thread_local std::chrono::nanoseconds t_phase1{}, t_phase2{}, t_exec{},
                                                  t_record{}, t_phase3{};
    static thread_local std::chrono::nanoseconds t_reset{}, t_submit{},
                                                  t_sync_a{}, t_sync_b{};
    static thread_local std::chrono::nanoseconds t_dev_copy{}, t_dev_kernel{},
                                                  t_dev_ts_query{};
    static thread_local std::atomic<uint64_t> n_calls{0}, n_rebuild{0};
    static thread_local uint64_t t_copy_bytes{0};
    auto t0 = clk::now();

    auto& rdz = *m_rendezvous[collective_id];

    // Phase 1 (enter barrier): deposit pointers, wait until all ranks arrive.
    // Uses a generation counter so the wait predicate is monotonic and the
    // last-in resets `arrived` immediately for the next epoch.
    trace("phase1: enter");
    {
        std::unique_lock<std::mutex> lk(rdz.mtx);
        rdz.in_ptrs[rank]  = in_dev;
        rdz.out_ptrs[rank] = out_dev;
        if (rank == 0) {
            rdz.n     = n;
            rdz.dtype = dtype;
        }
        const uint64_t my_gen = rdz.enter_gen;
        if (++rdz.arrived == m_world_size) {
            rdz.arrived = 0;
            rdz.enter_gen++;
            rdz.cv.notify_all();
        } else {
            rdz.cv.wait(lk, [&] { return rdz.enter_gen != my_gen; });
        }
    }
    trace("phase1: passed");
    auto t1 = clk::now();

    // Phase 2: rank 0 orchestrates; non-zero ranks wait for `done`.
    if (rank == 0) {
        OPENVINO_ASSERT(rdz.n == n && rdz.dtype == dtype,
                        "[TP][L0] inconsistent (n,dtype) across ranks for collective ",
                        collective_id);

        auto& slot = m_plans[collective_id];
        if (!slot) slot = std::make_unique<Plan>();

        const auto payload_bytes = collective_payload_bytes(n, dtype);
        const bool scratch_grew = ensure_scratch_capacity(payload_bytes);
        const bool resources_missing = slot->pool == nullptr;
        const bool signature_matches = slot->matches(rdz.in_ptrs, rdz.out_ptrs, n, dtype);
        const bool recorded_matches = !m_use_immediate &&
                                      m_recorded_plan == slot.get() &&
                                      m_recorded_scratch_generation == m_scratch.generation &&
                                      signature_matches;
        const bool need_record = !m_use_immediate && !recorded_matches;

        trace(resources_missing ? "phase2: build resources"
                                : scratch_grew ? "phase2: grow scratch and re-record"
                                : need_record ? "phase2: re-record" : "phase2: reuse recording");
        auto tr0 = clk::now();
        if (resources_missing || scratch_grew) {
            ++n_rebuild;
        }

        const auto previous_max_payload = slot->max_payload_bytes;
        if (tp_profiling_enabled() && payload_bytes > previous_max_payload) {
            std::cerr << "[TP][MEM] collective cid=" << collective_id
                      << " n=" << n
                      << " dtype=" << dtype
                      << " payload=" << payload_bytes
                      << " previous_max=" << previous_max_payload
                      << std::endl;
        }

        if (resources_missing) {
            build_plan(collective_id, rdz.in_ptrs, rdz.out_ptrs, n, dtype, *slot);
        } else if (!signature_matches) {
            slot->in_ptrs = rdz.in_ptrs;
            slot->out_ptrs = rdz.out_ptrs;
            slot->n = n;
            slot->dtype = dtype;
        }
        slot->max_payload_bytes = std::max(previous_max_payload, payload_bytes);

        if (need_record) {
            record_plan(*slot);
            m_recorded_plan = slot.get();
            m_recorded_scratch_generation = m_scratch.generation;
        }
        auto tr1 = clk::now();

        trace("phase2: execute");
        auto te0 = clk::now();
        ExecStats es{};
        execute_plan(*slot, &es);
        auto te1 = clk::now();
        trace("phase2: executed");

        t_record += tr1 - tr0;
        t_exec   += te1 - te0;
        t_reset  += es.reset;
        t_submit += es.submit;
        t_sync_a += es.sync_a;
        t_sync_b += es.sync_b;
        t_dev_copy     += es.dev_copy;
        t_dev_kernel   += es.dev_kernel;
        t_dev_ts_query += es.dev_ts_query;
        t_copy_bytes   += es.copy_bytes;

        std::unique_lock<std::mutex> lk(rdz.mtx);
        rdz.done = true;
        rdz.cv.notify_all();
    } else {
        std::unique_lock<std::mutex> lk(rdz.mtx);
        rdz.cv.wait(lk, [&] { return rdz.done; });
    }
    trace("phase2: passed");
    auto t2 = clk::now();

    // Phase 3 (exit barrier): wait until all ranks have left, then the
    // last-out clears state for the next epoch.
    {
        std::unique_lock<std::mutex> lk(rdz.mtx);
        const uint64_t my_gen = rdz.exit_gen;
        if (++rdz.departed == m_world_size) {
            rdz.departed = 0;
            rdz.done = false;
            std::fill(rdz.in_ptrs.begin(), rdz.in_ptrs.end(), nullptr);
            std::fill(rdz.out_ptrs.begin(), rdz.out_ptrs.end(), nullptr);
            rdz.exit_gen++;
            rdz.cv.notify_all();
        } else {
            rdz.cv.wait(lk, [&] { return rdz.exit_gen != my_gen; });
        }
    }
    trace("phase3: passed (return)");

    auto t3 = clk::now();
    t_phase1 += t1 - t0;
    t_phase2 += t2 - t1;
    t_phase3 += t3 - t2;
    if (prof_every > 0 && rank == 0) {
        uint64_t c = ++n_calls;
        if ((c % prof_every) == 0) {
            using ms = std::chrono::duration<double, std::milli>;
            const double cf = static_cast<double>(c);
            std::cerr << "[TP][PROF] r0 calls=" << c
                      << " rebuilds=" << n_rebuild.load()
                      << "  totals: ph1=" << ms(t_phase1).count() << "ms"
                      << " ph2=" << ms(t_phase2).count() << "ms"
                      << " (record=" << ms(t_record).count() << "ms"
                      << ", exec=" << ms(t_exec).count() << "ms)"
                      << " ph3=" << ms(t_phase3).count() << "ms"
                      << "  per-call: ph1=" << ms(t_phase1).count() / cf << "ms"
                      << " exec=" << ms(t_exec).count() / cf << "ms"
                      << " ph3=" << ms(t_phase3).count() / cf << "ms"
                      << std::endl;
            // exec breakdown: reset events / submit / sync(rank0) / sync(rank1)
            // sync_a is the time the first sync blocks (waits for slower
            // rank up to that point); sync_b > 0 means rank0 returned
            // early and rank1 was the straggler.
            const double exec_total = ms(t_exec).count();
            const double r_pct = exec_total > 0 ? 100.0 * ms(t_reset).count()  / exec_total : 0;
            const double s_pct = exec_total > 0 ? 100.0 * ms(t_submit).count() / exec_total : 0;
            const double a_pct = exec_total > 0 ? 100.0 * ms(t_sync_a).count() / exec_total : 0;
            const double b_pct = exec_total > 0 ? 100.0 * ms(t_sync_b).count() / exec_total : 0;
            std::cerr << "[TP][PROF]   exec breakdown:"
                      << " reset="  << ms(t_reset).count()  / cf << "ms (" << r_pct << "%)"
                      << " submit=" << ms(t_submit).count() / cf << "ms (" << s_pct << "%)"
                      << " sync_r0=" << ms(t_sync_a).count() / cf << "ms (" << a_pct << "%)"
                      << " sync_r1=" << ms(t_sync_b).count() / cf << "ms (" << b_pct << "%)"
                      << std::endl;
            // Device-side per-step timing (max across the two ranks for
            // each step — the slower side defines the critical path).
            const double ts_query_per = ms(t_dev_ts_query).count() / cf;
            std::cerr << "[TP][PROF]   device steps (max across ranks):"
                      << " memcpy=" << ms(t_dev_copy).count()   / cf << "ms"
                      << " kernel=" << ms(t_dev_kernel).count() / cf << "ms"
                      << " (sum=" << (ms(t_dev_copy).count() + ms(t_dev_kernel).count()) / cf << "ms)"
                      << " ts_query_overhead=" << ts_query_per << "ms"
                      << std::endl;
            // PCIe throughput: bytes-per-call / device-side memcpy time.
            // Each rank transfers `copy_bytes` to its peer; the two transfers
            // run concurrently in opposite directions on the PCIe link, so
            // we report per-direction (1x) and aggregate full-duplex (2x) BW.
            const double bytes_per_call = cf > 0 ? double(t_copy_bytes) / cf : 0.0;
            const double dev_copy_s     = ms(t_dev_copy).count() / cf / 1.0e3;
            const double bw_per_dir_gbs = dev_copy_s > 0 ? (bytes_per_call / 1.0e9) / dev_copy_s : 0.0;
            const double exec_per_s     = ms(t_exec).count()    / cf / 1.0e3;
            const double bw_walltime_gbs= exec_per_s > 0 ? (bytes_per_call / 1.0e9) / exec_per_s : 0.0;
            std::cerr << "[TP][PROF]   throughput:"
                      << " bytes/call=" << (bytes_per_call / (1024.0 * 1024.0)) << "MB"
                      << "  per-dir(dev_copy)=" << bw_per_dir_gbs << " GB/s"
                      << "  full-duplex(dev_copy)=" << 2.0 * bw_per_dir_gbs << " GB/s"
                      << "  effective(exec)=" << bw_walltime_gbs << " GB/s"
                      << std::endl;
            const auto scratch = get_scratch_stats();
            std::cerr << "[TP][PROF]   scratch: payload_capacity="
                      << (scratch.payload_capacity_bytes / (1024.0 * 1024.0)) << "MB"
                      << " total=" << (scratch.total_allocated_bytes / (1024.0 * 1024.0)) << "MB"
                      << " generation=" << scratch.generation
                      << " grows=" << scratch.growth_count
                      << " allocations=" << scratch.allocation_count
                      << std::endl;
        }
    }
}

}  // namespace tp_gpu
}  // namespace ov

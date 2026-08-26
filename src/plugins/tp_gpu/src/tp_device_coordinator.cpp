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
#include <sstream>
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

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// The ov::ze* wrappers call ZeroApi::get_instance() on every single call, and
// that takes a process-global mutex, locks a weak_ptr and returns a shared_ptr
// by value.  Every Level Zero call made anywhere in OpenVINO therefore
// serializes on one lock.  Resolving the table once and calling through it
// keeps the entry points but drops the per-call rendezvous.
const std::shared_ptr<ov::ZeroApi>& ze_api() {
    static const std::shared_ptr<ov::ZeroApi> api = ov::ZeroApi::get_instance();
    return api;
}

// Element granularity of a ring chunk.  128 elements is 256 bytes for f16 and
// 512 for f32, which covers any vector width the OpenCL back end may pick for
// the reduce kernel.
constexpr std::size_t kRingAlignElems = 128;

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
                                         int num_collectives,
                                         std::chrono::milliseconds collective_timeout)
    : m_shared(std::move(shared)),
      m_world_size(world_size),
      m_num_collectives(num_collectives),
      m_collective_timeout(collective_timeout) {
    OPENVINO_ASSERT(m_shared && m_shared->context && static_cast<int>(m_shared->devices.size()) == world_size,
                    "[TP][L0] coordinator requires a valid shared L0 context with ", world_size, " devices");
    OPENVINO_ASSERT(world_size >= 2, "[TP][L0] coordinator requires at least 2 ranks");
    OPENVINO_ASSERT(m_collective_timeout.count() >= 0, "[TP][L0] collective timeout must not be negative");

    if (const char* v = std::getenv("TP_USE_IMMEDIATE")) {
        m_use_immediate = std::atoi(v) != 0;
    }
    m_profiling_enabled = tp_profiling_enabled();
    if (const char* v = std::getenv("TP_DEVICE_EVENT_RESET")) {
        m_device_event_reset = std::atoi(v) != 0;
    }
    // The ring only exists for N>2; two ranks already exchange directly,
    // which is what the ring degenerates to minus a round of latency.
    m_use_ring = world_size > 2;
    if (const char* v = std::getenv("TP_RING")) {
        m_use_ring = m_use_ring && std::atoi(v) != 0;
    }
    if (const char* v = std::getenv("TP_RING_IN_ORDER")) {
        m_ring_in_order = std::atoi(v) != 0;
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
    try {
        for (int i = 0; i < num_collectives; ++i) {
            auto rdz = std::make_unique<Rendezvous>();
            rdz->in_ptrs.assign(world_size, nullptr);
            rdz->out_ptrs.assign(world_size, nullptr);
            m_rendezvous[i] = std::move(rdz);

            // Command lists are cheap to keep but not to create: making them
            // lazily put ~50 ms of driver work on the first inference, which
            // is the one whose latency users measure as time to first token.
            m_plans[i] = std::make_unique<Plan>();
            ensure_plan_lists(*m_plans[i]);
        }
    } catch (...) {
        for (auto& plan : m_plans) {
            if (plan) destroy_plan(*plan);
        }
        for (auto& rs : m_ranks) destroy_rank(rs);
        throw;
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
    for (auto& plan : m_plans) {
        if (plan) destroy_plan(*plan);
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

        // No rank-wide command list on the regular path: each collective owns
        // its own, so a recording is not clobbered by the next collective.

        // Optional dedicated copy engine for the cross-device memcpy step.
        if (tp_use_copy_engine() && select_copy_ordinal(dev, rs.copy_ordinal)) {
            rs.has_dedicated_copy = true;
            ze_command_queue_desc_t cqd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr};
            cqd.ordinal  = rs.copy_ordinal;
            cqd.mode     = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
            cqd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
            ZE_THROW(ov::zeCommandQueueCreate(ctx, dev, &cqd, &rs.copy_queue));
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
                ov::zeCommandQueueSynchronize(rs.compute_queue, timeout_ns());
            }
            if (rs.copy_queue) {
                ov::zeCommandQueueSynchronize(rs.copy_queue, timeout_ns());
            }
        }
    }

    for (auto& e : plan.ev_recv)  if (e) ov::zeEventDestroy(e);
    for (auto& e : plan.ev_bcast) if (e) ov::zeEventDestroy(e);
    for (auto& e : plan.ev_ring)  if (e) ov::zeEventDestroy(e);
    if (plan.ev_reduce) ov::zeEventDestroy(plan.ev_reduce);
    if (plan.pool)      ov::zeEventPoolDestroy(plan.pool);
    for (auto& e : plan.ev_ts_copy)   if (e) ov::zeEventDestroy(e);
    for (auto& e : plan.ev_ts_kernel) if (e) ov::zeEventDestroy(e);
    if (plan.ts_pool) ov::zeEventPoolDestroy(plan.ts_pool);
    for (auto& l : plan.compute_lists) if (l) ov::zeCommandListDestroy(l);
    for (auto& l : plan.copy_lists)    if (l) ov::zeCommandListDestroy(l);
    plan.ev_recv.clear();
    plan.ev_bcast.clear();
    plan.ev_ring.clear();
    plan.ev_reduce = nullptr;
    plan.pool = nullptr;
    plan.ev_ts_copy.clear();
    plan.ev_ts_kernel.clear();
    plan.ts_pool = nullptr;
    plan.compute_lists.clear();
    plan.copy_lists.clear();
    plan.recorded = false;
    plan.recorded_scratch_generation = 0;
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

    // Ring: N slots per rank, one per chunk, so concurrent steps never share
    // a slot.  The stride is the largest chunk rounded up for alignment; the
    // total lands within one payload per rank.
    const std::size_t chunk_stride =
        m_use_ring ? align_up((payload_bytes + m_world_size - 1) / m_world_size, 256) : 0;

    try {
        if (m_use_ring) {
            const auto ring_bytes = checked_multiply(chunk_stride,
                                                     static_cast<std::size_t>(m_world_size),
                                                     "ring scratch");
            for (int rank = 0; rank < m_world_size; ++rank) {
                ZE_THROW(ov::zeMemAllocDevice(m_shared->context,
                                              &mad,
                                              ring_bytes,
                                              256,
                                              m_ranks[rank].device,
                                              &new_allocations[rank]));
                new_bytes_per_rank[rank] = ring_bytes;
            }
        } else if (m_world_size == 2) {
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
                sync_queue(rs.compute_queue, "scratch grow: compute queue drain");
            }
            if (rs.copy_queue) {
                sync_queue(rs.copy_queue, "scratch grow: copy queue drain");
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
    m_scratch.chunk_capacity_bytes = chunk_stride;
    m_scratch.total_allocated_bytes = 0;
    for (const auto bytes : m_scratch.bytes_per_rank) {
        m_scratch.total_allocated_bytes += bytes;
    }
    ++m_scratch.generation;
    ++m_scratch.growth_count;
    m_scratch.allocation_count += (m_use_ring || m_world_size == 2) ? m_world_size : 1;

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
    m_scratch.chunk_capacity_bytes = 0;
    m_scratch.total_allocated_bytes = 0;
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

void TPDeviceCoordinator::ring_chunk(std::size_t n,
                                     int chunk,
                                     std::size_t& offset,
                                     std::size_t& count) const {
    const std::size_t N = static_cast<std::size_t>(m_world_size);
    const std::size_t c = static_cast<std::size_t>(chunk);
    // Chunk boundaries have to be aligned, not merely even: the reduce kernel
    // is compiled from scalar OpenCL but the vector back end widens it, and a
    // base pointer that is not naturally aligned corrupts a handful of
    // elements at the head of the chunk -- measured as tens of wrong values
    // out of 64K, varying run to run.  Splitting by aligned units instead of
    // by element count keeps every offset a multiple of kRingAlignElems.
    // A payload smaller than N units leaves trailing chunks empty, which the
    // schedule handles by signalling the step without moving anything.
    const std::size_t align = kRingAlignElems;
    const std::size_t units = (n + align - 1) / align;
    const std::size_t base = units / N;
    const std::size_t rem = units % N;
    const std::size_t unit_off = c * base + std::min(c, rem);
    const std::size_t unit_cnt = base + (c < rem ? 1u : 0u);

    offset = unit_off * align;
    // The last populated chunk is clipped to the payload; alignment rounds
    // the unit count up, so the tail must not run past n.
    count = (offset >= n) ? 0u : std::min(unit_cnt * align, n - offset);
}

void* TPDeviceCoordinator::ring_slot(int rank, int chunk) const {
    OPENVINO_ASSERT(rank >= 0 && rank < m_world_size, "[TP][L0] ring slot rank out of range: ", rank);
    OPENVINO_ASSERT(chunk >= 0 && chunk < m_world_size, "[TP][L0] ring slot chunk out of range: ", chunk);
    auto* base = static_cast<uint8_t*>(m_scratch.allocations[static_cast<std::size_t>(rank)]);
    return base + static_cast<std::size_t>(chunk) * m_scratch.chunk_capacity_bytes;
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

    // Command lists are per collective, so a recording is not clobbered by
    // the next collective's.  Normally already created at setup.
    ensure_plan_lists(plan);

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

    // ---- Ring: reduce-scatter + all-gather ----
    if (m_use_ring) {
        const int steps = 2 * (N - 1);
        ze_event_pool_desc_t epd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr};
        epd.flags = tp_profiling_enabled() ? ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP : 0;
        // One event per (step, rank), plus one "reduce-scatter finished" per
        // rank in a trailing row.
        epd.count = static_cast<uint32_t>((steps + 1) * N);
        std::vector<ze_device_handle_t> devs_nc(m_shared->devices.begin(), m_shared->devices.end());
        ZE_THROW(ov::zeEventPoolCreate(ctx, &epd,
                                       static_cast<uint32_t>(devs_nc.size()),
                                       devs_nc.data(), &plan.pool));

        plan.ev_recv.clear();
        plan.ev_bcast.clear();
        plan.ev_reduce = nullptr;
        plan.ev_ts_copy.clear();
        plan.ev_ts_kernel.clear();
        plan.ts_pool = nullptr;

        // One event per (step, rank): rank r's outgoing copy at that step
        // signals it and its successor waits on it.  Device scope on both
        // ends -- no rank ever blocks the host from inside the ring.
        plan.ev_ring.assign(static_cast<std::size_t>(epd.count), nullptr);
        for (uint32_t i = 0; i < epd.count; ++i) {
            ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr};
            ed.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
            ed.wait   = ZE_EVENT_SCOPE_FLAG_DEVICE;
            ed.index  = i;
            ZE_THROW(ov::zeEventCreate(plan.pool, &ed, &plan.ev_ring[i]));
        }
        return;
    }

    // ---- Legacy N>2 main-funnel path ----
    const int W = N - 1;
    // Event pool: W recv + 1 reduce + W bcast, plus W-1 kernel probes when
    // profiling.  The recv/bcast events double as memcpy-end probes and the
    // reduce event times the last kernel, so the timestamp flag has to be on
    // the pool for the funnel to be measurable at all.
    const bool prof_on = tp_profiling_enabled();
    const uint32_t ts_kernels = prof_on && W > 1 ? static_cast<uint32_t>(W - 1) : 0u;
    const uint32_t total_events = static_cast<uint32_t>(2 * W + 1) + ts_kernels;
    ze_event_pool_desc_t epd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr};
    epd.flags = prof_on ? ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP : 0;
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

    // Probes for every accumulate kernel but the last one, which is timed by
    // ev_reduce.  Without them dev_kernel would report one of W kernels and
    // understate rank 0's serialized reduce by a factor of W.
    plan.ev_ts_kernel.assign(static_cast<std::size_t>(ts_kernels), nullptr);
    for (uint32_t i = 0; i < ts_kernels; ++i) {
        mk(plan.ev_ts_kernel[i]);
    }
}

void TPDeviceCoordinator::ensure_plan_lists(Plan& plan) {
    // The immediate path records nothing ahead of time and keeps using the
    // rank's immediate list.
    if (m_use_immediate || !plan.compute_lists.empty()) {
        return;
    }

    auto ctx = m_shared->context;
    plan.compute_lists.assign(static_cast<std::size_t>(m_world_size), nullptr);
    plan.copy_lists.assign(static_cast<std::size_t>(m_world_size), nullptr);
    for (int r = 0; r < m_world_size; ++r) {
        auto& rs = m_ranks[r];
        ze_command_list_desc_t ld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr};
        ld.commandQueueGroupOrdinal = rs.compute_ordinal;
        // The ring is a linear dependency chain inside each rank's list: the
        // copy of step s+1 reads what the kernel of step s produced.  Asking
        // the driver for in-order execution expresses that for free, instead
        // of a barrier per step whose cost would scale with 2*(N-1).
        if (m_use_ring) {
            ld.flags = m_ring_in_order ? ZE_COMMAND_LIST_FLAG_IN_ORDER : 0;
        }
        ZE_THROW(ov::zeCommandListCreate(ctx, rs.device, &ld, &plan.compute_lists[r]));
        if (rs.has_dedicated_copy) {
            ze_command_list_desc_t cld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr};
            cld.commandQueueGroupOrdinal = rs.copy_ordinal;
            ZE_THROW(ov::zeCommandListCreate(ctx, rs.device, &cld, &plan.copy_lists[r]));
        }
    }
}

void TPDeviceCoordinator::record_ring_plan(Plan& plan) {
    constexpr uint32_t kGroupSize = 256;
    const int N = m_world_size;
    const int steps = N - 1;
    const std::size_t n = plan.n;
    const std::size_t elem = plan.dtype.size();
    OPENVINO_ASSERT(n > 0, "[TP][L0] ring requires a non-empty payload");

    auto ev = [&](int step, int rank) -> ze_event_handle_t {
        return plan.ev_ring[static_cast<std::size_t>(step) * static_cast<std::size_t>(N) +
                            static_cast<std::size_t>(rank)];
    };
    auto byte_at = [elem](void* base, std::size_t offset_elems) -> void* {
        return static_cast<uint8_t*>(base) + offset_elems * elem;
    };
    // Without in-order lists the chain has to be spelled out: the send of the
    // next step reads what this step's kernel produced, and the tail resets
    // must not overtake the waits that consumed those events.
    auto order = [&](ze_command_list_handle_t list) {
        if (!m_ring_in_order) {
            ZE_THROW(ze_api()->zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
        }
    };

    for (int r = 0; r < N; ++r) {
        auto& self = m_ranks[r];
        const int next = (r + 1) % N;
        const int prev = (r + N - 1) % N;
        ze_command_list_handle_t list = plan.compute_lists[r];
        ze_kernel_handle_t kernel =
            (plan.dtype == ov::element::f16) ? self.kernel_f16 : self.kernel_f32;
        ZE_THROW(ze_api()->zeKernelSetGroupSize(kernel, kGroupSize, 1, 1));

        // ---- Reduce-scatter: N-1 steps ----
        // At step s rank r forwards chunk (r-s) and folds the incoming chunk
        // (r-s-1) with its own contribution.  Each chunk is accumulated
        // exactly once per rank, and the incoming buffer already carries the
        // partial sum of every rank before us on the ring, so the reduction
        // is always "my input plus what arrived" -- the same two-source
        // kernel the direct N=2 exchange uses.
        for (int s = 0; s < steps; ++s) {
            const int send_chunk = ((r - s) % N + N) % N;
            const int recv_chunk = ((r - s - 1) % N + N) % N;

            std::size_t send_off = 0, send_cnt = 0, recv_off = 0, recv_cnt = 0;
            ring_chunk(n, send_chunk, send_off, send_cnt);
            ring_chunk(n, recv_chunk, recv_off, recv_cnt);

            // Step 0 forwards our own untouched input; later steps forward
            // the chunk this rank reduced in the previous step.  An empty
            // chunk still has to signal, or the successor waits forever.
            if (send_cnt == 0) {
                ZE_THROW(ze_api()->zeCommandListAppendSignalEvent(list, ev(s, r)));
            } else {
                void* send_src = (s == 0) ? byte_at(plan.in_ptrs[r], send_off)
                                          : byte_at(plan.out_ptrs[r], send_off);
                ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(
                    list,
                    ring_slot(next, send_chunk),
                    send_src,
                    send_cnt * elem,
                    ev(s, r),
                    0, nullptr));
            }

            ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &plan.ev_ring[
                static_cast<std::size_t>(s) * static_cast<std::size_t>(N) +
                static_cast<std::size_t>(prev)]));

            if (recv_cnt == 0) {
                continue;
            }
            void* dst  = byte_at(plan.out_ptrs[r], recv_off);
            void* src0 = byte_at(plan.in_ptrs[r], recv_off);
            void* src1 = ring_slot(r, recv_chunk);
            uint64_t cnt = recv_cnt;
            ze_group_count_t gc{
                (static_cast<uint32_t>(recv_cnt) + kGroupSize - 1) / kGroupSize, 1, 1};
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &dst));
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &src0));
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &src1));
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 3, sizeof(cnt), &cnt));
            ZE_THROW(ze_api()->zeCommandListAppendLaunchKernel(list, kernel, &gc, nullptr, 0, nullptr));
            order(list);
        }

        // ---- All-gather: N-1 steps ----
        // Rank r now owns the finished chunk (r+1).  Each step passes a
        // finished chunk along the same ring, straight into the successor's
        // output -- no staging and no kernel, because there is nothing left
        // to reduce.
        //
        // Writing into the successor's output is only safe once that rank has
        // stopped writing there itself.  Its reduce-scatter touches every
        // chunk but its own, including the ones we are about to deliver, and
        // nothing in the per-step chain orders the two: a rank whose
        // predecessors ran ahead could still be in reduce-scatter when the
        // first all-gather copy lands, and its partial sum would then
        // overwrite our final one.  One handshake per rank closes that.
        ZE_THROW(ze_api()->zeCommandListAppendSignalEvent(list, ev(2 * steps, r)));
        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &plan.ev_ring[
            static_cast<std::size_t>(2 * steps) * static_cast<std::size_t>(N) +
            static_cast<std::size_t>(next)]));

        for (int s = 0; s < steps; ++s) {
            const int send_chunk = ((r + 1 - s) % N + N) % N;
            std::size_t off = 0, cnt = 0;
            ring_chunk(n, send_chunk, off, cnt);

            const int step = steps + s;
            if (cnt == 0) {
                ZE_THROW(ze_api()->zeCommandListAppendSignalEvent(list, ev(step, r)));
            } else {
                ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(
                    list,
                    byte_at(plan.out_ptrs[next], off),
                    byte_at(plan.out_ptrs[r], off),
                    cnt * elem,
                    ev(step, r),
                    0, nullptr));
            }

            ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &plan.ev_ring[
                static_cast<std::size_t>(step) * static_cast<std::size_t>(N) +
                static_cast<std::size_t>(prev)]));
        }

        // Every ring event has exactly one waiter, so each rank clears the
        // ones it consumed: the per-step events of its predecessor and the
        // reduce-scatter handshake of its successor.  The in-order list
        // already orders these behind the waits above.
        if (use_device_event_reset()) {
            order(list);
            for (int s = 0; s < 2 * steps; ++s) {
                ZE_THROW(ze_api()->zeCommandListAppendEventReset(list, ev(s, prev)));
            }
            ZE_THROW(ze_api()->zeCommandListAppendEventReset(list, ev(2 * steps, next)));
        }

        close_list(list);
    }
}

void TPDeviceCoordinator::close_list(ze_command_list_handle_t list) {
    const auto t0 = std::chrono::steady_clock::now();
    ZE_THROW(ze_api()->zeCommandListClose(list));
    m_rec_close += std::chrono::steady_clock::now() - t0;
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

    // Reset this plan's command lists (must be done before re-recording).
    // zeCommandListReset on a list that still has work in-flight on its
    // queue is undefined; on shared multi-device L0 contexts it can
    // deadlock.  Sync each queue first so the previous submission has
    // fully drained before we wipe the recorded commands.
    //
    // Recording is only 0.4% of collective calls but all of its misses land
    // in time to first token, so the three stages are timed separately: the
    // drains, the resets, and the appends that follow.
    using rec_clk = std::chrono::steady_clock;
    const auto t_rec0 = rec_clk::now();
    for (int r = 0; r < N; ++r) {
        auto& rs = m_ranks[r];
        if (rs.compute_queue) {
            sync_queue(rs.compute_queue, "re-record: compute queue drain");
        }
        if (rs.copy_queue) {
            sync_queue(rs.copy_queue, "re-record: copy queue drain");
        }
    }
    const auto t_rec1 = rec_clk::now();
    for (int r = 0; r < N; ++r) {
        ZE_THROW(ze_api()->zeCommandListReset(plan.compute_lists[r]));
        if (plan.copy_lists[r]) {
            ZE_THROW(ze_api()->zeCommandListReset(plan.copy_lists[r]));
        }
    }
    const auto t_rec2 = rec_clk::now();
    m_rec_drain += t_rec1 - t_rec0;
    m_rec_reset += t_rec2 - t_rec1;
    struct BuildTimer {
        TPDeviceCoordinator* self;
        rec_clk::time_point start{rec_clk::now()};
        ~BuildTimer() { self->m_rec_build += rec_clk::now() - start; }
    } build_timer{this};

    if (m_use_ring) {
        record_ring_plan(plan);
        return;
    }

    if (N == 2) {
        uint32_t items = static_cast<uint32_t>(n);
        ze_group_count_t gc{(items + kGroupSize - 1) / kGroupSize, 1, 1};
        uint64_t cn64 = n;

        for (int r = 0; r < 2; ++r) {
            auto& self = m_ranks[r];
            const int peer = 1 - r;
            ze_command_list_handle_t self_compute = plan.compute_lists[r];

            ze_kernel_handle_t kernel =
                (plan.dtype == ov::element::f16) ? self.kernel_f16 : self.kernel_f32;
            ZE_THROW(ze_api()->zeKernelSetGroupSize(kernel, kGroupSize, 1, 1));

            // 1. Push our `in` to peer's local staging (source-side memcpy).
            //    Route onto the dedicated copy engine when available so
            //    cross-device DMA does not contend with the reduce kernel
            //    on the compute engine.  ev_recv[r] is signaled by the
            //    copy queue; the cross-device wait below resolves on the
            //    peer's compute queue regardless of which engine signaled.
            ze_command_list_handle_t copy_target =
                plan.copy_lists[r] ? plan.copy_lists[r] : self_compute;
            ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(
                copy_target,
                scratch_buffer(peer),
                plan.in_ptrs[r],
                bytes,
                plan.ev_recv[r],
                0, nullptr));

            // 2. Wait for peer's push to land in our staging.
            ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(self_compute,
                                                              1, &plan.ev_recv[peer]));

            // Rank r is the only consumer of ev_recv[peer], and once the wait
            // above is satisfied the event has done its job -- clearing it
            // does not touch the staged data the kernel is about to read.
            // Placing the reset here rather than after the kernel is what
            // makes it free: the wait already orders everything appended
            // after it, so no barrier is needed.
            if (use_device_event_reset()) {
                ZE_THROW(ze_api()->zeCommandListAppendEventReset(self_compute, plan.ev_recv[peer]));
            }

            // 3. Reduce: out_self = in_self + staging_self.
            void* dst   = plan.out_ptrs[r];
            void* src0  = plan.in_ptrs[r];
            void* src1  = scratch_buffer(r);
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &dst));
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &src0));
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &src1));
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 3, sizeof(cn64), &cn64));
            ZE_THROW(ze_api()->zeCommandListAppendLaunchKernel(self_compute,
                                                              kernel, &gc,
                                                              plan.ev_ts_kernel[r], 0, nullptr));

            close_list(self_compute);
            if (plan.copy_lists[r]) {
                close_list(plan.copy_lists[r]);
            }
        }
        return;
    }

    // ---- Legacy N>2 main-funnel path ----
    const int W = N - 1;
    auto& main_rs = m_ranks[0];
    ze_command_list_handle_t main_list = plan.compute_lists[0];
    // Route cross-device DMA onto the dedicated copy engine where the device
    // exposes one, so the gather and the scatter stop competing with the
    // reduce kernel for rank 0's compute engine.  Falls back to the rank's
    // compute list when there is no copy engine.
    ze_command_list_handle_t main_copy_list =
        plan.copy_lists[0] ? plan.copy_lists[0] : main_list;
    ze_kernel_handle_t main_kernel =
        (plan.dtype == ov::element::f16) ? main_rs.kernel_f16 : main_rs.kernel_f32;
    const bool device_reset = use_device_event_reset();

    // --- Worker lists: copy local input to rank-0 scratch, wait scatter ---
    for (int w = 0; w < W; ++w) {
        ze_command_list_handle_t worker_list = plan.compute_lists[w + 1];
        ze_command_list_handle_t worker_copy_list =
            plan.copy_lists[w + 1] ? plan.copy_lists[w + 1] : worker_list;
        ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(
            worker_copy_list,
            scratch_buffer(w),
            plan.in_ptrs[w + 1],
            bytes,
            plan.ev_recv[w],
            0, nullptr));
        // Wait for our scatter to land before the queue-sync returns.
        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(worker_list, 1, &plan.ev_bcast[w]));
        // Worker w is the only consumer of ev_bcast[w], so it also clears it
        // for the next call.  The wait above orders the reset, so this costs
        // one command-processor slot and no barrier.
        if (device_reset) {
            ZE_THROW(ze_api()->zeCommandListAppendEventReset(worker_list, plan.ev_bcast[w]));
        }
    }

    // --- Main compute list: wait recvs, run accumulate kernels, scatter ---
    ZE_THROW(ze_api()->zeKernelSetGroupSize(main_kernel, kGroupSize, 1, 1));

    if (W > 0) {
        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(main_list,
                                                          static_cast<uint32_t>(W),
                                                          plan.ev_recv.data()));
        // main_list is the only consumer of ev_recv; the wait above orders
        // the resets and the staged data stays untouched, so the kernels
        // below still read what the gather delivered.
        if (device_reset) {
            for (int w = 0; w < W; ++w) {
                ZE_THROW(ze_api()->zeCommandListAppendEventReset(main_list, plan.ev_recv[w]));
            }
        }
    }

    uint32_t items = static_cast<uint32_t>(n);
    ze_group_count_t gc{(items + kGroupSize - 1) / kGroupSize, 1, 1};
    uint64_t cn64 = n;

    void* dst_main = plan.out_ptrs[0];
    void* src0_main = plan.in_ptrs[0];

    for (int w = 0; w < W; ++w) {
        void* a = (w == 0) ? src0_main : dst_main;
        void* b = scratch_buffer(w);
        ZE_THROW(ze_api()->zeKernelSetArgumentValue(main_kernel, 0, sizeof(void*), &dst_main));
        ZE_THROW(ze_api()->zeKernelSetArgumentValue(main_kernel, 1, sizeof(void*), &a));
        ZE_THROW(ze_api()->zeKernelSetArgumentValue(main_kernel, 2, sizeof(void*), &b));
        ZE_THROW(ze_api()->zeKernelSetArgumentValue(main_kernel, 3, sizeof(cn64), &cn64));
        // The last kernel signals ev_reduce, which the scatter waits on and
        // which doubles as its timestamp probe.  The earlier ones get their
        // own probe under profiling and signal nothing otherwise.
        ze_event_handle_t signal = nullptr;
        if (w == W - 1) {
            signal = plan.ev_reduce;
        } else if (static_cast<std::size_t>(w) < plan.ev_ts_kernel.size()) {
            signal = plan.ev_ts_kernel[w];
        }
        ZE_THROW(ze_api()->zeCommandListAppendLaunchKernel(main_list,
                                                          main_kernel, &gc,
                                                          signal, 0, nullptr));
        // Every accumulation but the first reads the result of the previous
        // one from dst_main and writes back to it.  A command list created
        // without ZE_COMMAND_LIST_FLAG_IN_ORDER gives no ordering between
        // appended kernels, so without this barrier consecutive launches
        // overlap and the sum loses the contributions still in flight.
        if (w + 1 < W) {
            ZE_THROW(ze_api()->zeCommandListAppendBarrier(main_list, nullptr, 0, nullptr));
        }
    }

    // --- Main scatter: copy result to each worker's out_ptr ---
    // One wait covers the whole loop.  ev_reduce is signaled once and every
    // copy below consumes the same result, so repeating the wait per copy
    // only burned W-1 command-processor slots.
    if (W > 0) {
        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(main_copy_list, 1, &plan.ev_reduce));
        // main_copy_list is the only consumer of ev_reduce.  dst_main is not
        // affected by clearing the event, so the copies below are unchanged.
        if (device_reset) {
            ZE_THROW(ze_api()->zeCommandListAppendEventReset(main_copy_list, plan.ev_reduce));
        }
    }
    for (int w = 0; w < W; ++w) {
        ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(
            main_copy_list,
            plan.out_ptrs[w + 1],
            dst_main,
            bytes,
            plan.ev_bcast[w],
            0, nullptr));
    }

    // Close all lists.
    for (auto& list : plan.compute_lists) {
        close_list(list);
    }
    for (auto& list : plan.copy_lists) {
        if (list) {
            close_list(list);
        }
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
    // When the recorded lists clear them on the device there is nothing left
    // to do here; the immediate and profiling paths still need the host reset.
    trace("execute: reset events");
    const bool device_reset = use_device_event_reset();
    if (!device_reset) {
        for (auto& e : plan.ev_recv)  ZE_THROW(ze_api()->zeEventHostReset(e));
        for (auto& e : plan.ev_bcast) ZE_THROW(ze_api()->zeEventHostReset(e));
        for (auto& e : plan.ev_ring)  ZE_THROW(ze_api()->zeEventHostReset(e));
        if (plan.ev_reduce) ZE_THROW(ze_api()->zeEventHostReset(plan.ev_reduce));
    }
    for (auto& e : plan.ev_ts_kernel) if (e) ZE_THROW(ze_api()->zeEventHostReset(e));
    auto tr1 = clk::now();
    if (stats) {
        stats->reset = tr1 - tr0;
        // Payload of a single cross-device transfer, and everything that
        // crosses a device boundary during the collective.  The direct N=2
        // exchange runs two full-payload transfers; the funnel runs W
        // gathers plus W scatters, all through rank 0; the ring runs
        // 2*(N-1) steps of one chunk on every one of the N links.
        const auto payload = collective_payload_bytes(plan.n, plan.dtype);
        const std::size_t links = static_cast<std::size_t>(m_world_size);
        if (m_use_ring) {
            stats->copy_bytes_total = 2u * (links - 1u) * payload;
            stats->copy_bytes = stats->copy_bytes_total / links;
        } else {
            const std::size_t transfers = m_world_size == 2 ? 2u : 2u * (links - 1u);
            stats->copy_bytes = payload;
            stats->copy_bytes_total = payload * transfers;
        }
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
            ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(
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
            ZE_THROW(ze_api()->zeKernelSetGroupSize(kernel, kGroupSize, 1, 1));

            step(r == 0 ? "wait r0" : "wait r1");
            ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(self.compute_list,
                                                              1, &plan.ev_recv[peer]));

            void* dst   = plan.out_ptrs[r];
            void* src0  = plan.in_ptrs[r];
            void* src1  = scratch_buffer(r);
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &dst));
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &src0));
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &src1));
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 3, sizeof(cn64), &cn64));
            // Signal the counter-based host-sync event directly from the
            // reduce kernel completion.  Avoids AppendBarrier, which on
            // shared multi-device contexts can attempt cross-device
            // synchronization and deadlock on immediate cmdlists.  Drop
            // ev_ts_kernel on the immediate path \u2014 a single-event signal
            // is what counter-based events are designed for.
            step(r == 0 ? "kernel r0 (signal cb)" : "kernel r1 (signal cb)");
            ZE_THROW(ze_api()->zeCommandListAppendLaunchKernel(self.compute_list,
                                                              kernel, &gc,
                                                              self.cb_event_done,
                                                              0, nullptr));
        }
        step("all appended; host-sync r0");
        auto ts1 = clk::now();
        // Bounded wait instead of UINT64_MAX so a deadlock on the immediate
        // path surfaces as a thrown error rather than a freeze.
        const uint64_t sync_timeout_ns = timeout_ns();
        ze_result_t s0 = ze_api()->zeEventHostSynchronize(m_ranks[0].cb_event_done, sync_timeout_ns);
        if (s0 == ZE_RESULT_NOT_READY) {
            OPENVINO_THROW("[TP][L0] immediate path: rank 0 cb_event_done did not signal within ",
                           m_collective_timeout.count(), " ms");
        }
        ZE_THROW(s0);
        auto ts2 = clk::now();
        ze_result_t s1 = ze_api()->zeEventHostSynchronize(m_ranks[1].cb_event_done, sync_timeout_ns);
        if (s1 == ZE_RESULT_NOT_READY) {
            OPENVINO_THROW("[TP][L0] immediate path: rank 1 cb_event_done did not signal within ",
                           m_collective_timeout.count(), " ms");
        }
        ZE_THROW(s1);
        auto ts3 = clk::now();
        if (stats) {
            stats->submit = ts1 - ts0;
            stats->sync_first = ts2 - ts1;
            stats->sync_rest  = ts3 - ts2;

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
                if (ze_api()->zeEventQueryKernelTimestamp(plan.ev_recv[r], &kt) == ZE_RESULT_SUCCESS) {
                    uint64_t ns = ticks_to_ns(kt.global.kernelStart, kt.global.kernelEnd,
                                              m_ranks[r].timestamp_mask,
                                              m_ranks[r].timer_ns_per_tick);
                    if (ns > copy_ns_max) copy_ns_max = ns;
                }
                if (plan.ev_ts_kernel[r] &&
                    ze_api()->zeEventQueryKernelTimestamp(plan.ev_ts_kernel[r], &kt) == ZE_RESULT_SUCCESS) {
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

    if (m_use_ring) {
        // Every rank runs the same self-contained chain, so there is no
        // launch order to get right and nothing to submit on rank 0's behalf.
        trace("execute: submit ring");
        auto tg0 = clk::now();
        for (int r = 0; r < m_world_size; ++r) {
            ZE_THROW(ze_api()->zeCommandQueueExecuteCommandLists(
                m_ranks[r].compute_queue, 1, &plan.compute_lists[r], nullptr));
        }
        auto tg1 = clk::now();

        auto tprev = tg1;
        for (int r = 0; r < m_world_size; ++r) {
            // Name the rank: a ring that stalls does so on one specific link,
            // and "which queue never drained" is the first thing worth
            // knowing when it happens.
            const std::string what = "allreduce: ring queue rank " + std::to_string(r) +
                                     " (n=" + std::to_string(plan.n) +
                                     (m_ring_in_order ? ", in-order" : ", barriers") + ")";
            sync_queue(m_ranks[r].compute_queue, what.c_str());
            if (stats) {
                const auto tnow = clk::now();
                (r == 0 ? stats->sync_first : stats->sync_rest) += tnow - tprev;
                tprev = tnow;
            }
        }

        if (stats) {
            stats->submit = tg1 - tg0;
        }
        if (stats && m_profiling_enabled) {
            // The steps are sequential and every rank runs them in lockstep,
            // so the critical path is the sum over steps of the slowest rank.
            auto tq0 = clk::now();
            uint64_t total_ns = 0;
            const int steps = 2 * (m_world_size - 1);
            for (int s = 0; s < steps; ++s) {
                uint64_t step_ns = 0;
                for (int r = 0; r < m_world_size; ++r) {
                    ze_kernel_timestamp_result_t kt{};
                    auto* e = plan.ev_ring[static_cast<std::size_t>(s) *
                                           static_cast<std::size_t>(m_world_size) +
                                           static_cast<std::size_t>(r)];
                    if (!e || ze_api()->zeEventQueryKernelTimestamp(e, &kt) != ZE_RESULT_SUCCESS) {
                        continue;
                    }
                    const uint64_t mask = m_ranks[r].timestamp_mask;
                    const uint64_t a = kt.global.kernelStart & mask;
                    const uint64_t b = kt.global.kernelEnd   & mask;
                    const uint64_t d = (b >= a) ? (b - a) : ((mask + 1 - a) + b);
                    step_ns = std::max(step_ns, d * m_ranks[r].timer_ns_per_tick);
                }
                total_ns += step_ns;
            }
            stats->dev_copy = std::chrono::nanoseconds{total_ns};
            stats->dev_kernel = std::chrono::nanoseconds{0};
            stats->dev_ts_query = clk::now() - tq0;
        }
        trace("execute: done (ring)");
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
                ZE_THROW(ze_api()->zeCommandQueueExecuteCommandLists(
                    m_ranks[r].copy_queue, 1, &plan.copy_lists[r], nullptr));
            }
            ZE_THROW(ze_api()->zeCommandQueueExecuteCommandLists(
                m_ranks[r].compute_queue, 1, &plan.compute_lists[r], nullptr));
        }
        auto ts1 = clk::now();
        sync_queue(m_ranks[0].compute_queue, "allreduce: rank 0 queue");
        auto ts2 = clk::now();
        sync_queue(m_ranks[1].compute_queue, "allreduce: rank 1 queue");
        auto ts3 = clk::now();
        if (stats) {
            stats->submit = ts1 - ts0;
            stats->sync_first = ts2 - ts1;
            stats->sync_rest  = ts3 - ts2;

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
                if (ze_api()->zeEventQueryKernelTimestamp(plan.ev_recv[r], &kt) == ZE_RESULT_SUCCESS) {
                    uint64_t ns = ticks_to_ns(kt.global.kernelStart, kt.global.kernelEnd,
                                              m_ranks[r].timestamp_mask,
                                              m_ranks[r].timer_ns_per_tick);
                    if (ns > copy_ns_max) copy_ns_max = ns;
                }
                if (plan.ev_ts_kernel[r] &&
                    ze_api()->zeEventQueryKernelTimestamp(plan.ev_ts_kernel[r], &kt) == ZE_RESULT_SUCCESS) {
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
    auto tf0 = clk::now();
    for (int w = 0; w < m_world_size - 1; ++w) {
        const int r = w + 1;
        if (m_ranks[r].copy_queue && plan.copy_lists[r]) {
            ZE_THROW(ze_api()->zeCommandQueueExecuteCommandLists(
                m_ranks[r].copy_queue, 1, &plan.copy_lists[r], nullptr));
        }
        ZE_THROW(ze_api()->zeCommandQueueExecuteCommandLists(
            m_ranks[r].compute_queue, 1, &plan.compute_lists[r], nullptr));
    }
    trace("execute: submit main");
    ZE_THROW(ze_api()->zeCommandQueueExecuteCommandLists(
        m_ranks[0].compute_queue, 1, &plan.compute_lists[0], nullptr));
    if (m_ranks[0].copy_queue && plan.copy_lists[0]) {
        ZE_THROW(ze_api()->zeCommandQueueExecuteCommandLists(
            m_ranks[0].copy_queue, 1, &plan.copy_lists[0], nullptr));
    }
    auto tf1 = clk::now();

    // Sync all queues.  With a dedicated copy engine the scatter lives on the
    // copy queue, so draining the compute queue alone would return before the
    // result reached the workers.
    auto tprev = tf1;
    for (int r = 0; r < m_world_size; ++r) {
        if (dbg) std::cerr << "[TP][L0] execute: sync rank " << r << std::endl;
        sync_queue(m_ranks[r].compute_queue, "allreduce: rank queue");
        if (m_ranks[r].copy_queue && plan.copy_lists[r]) {
            sync_queue(m_ranks[r].copy_queue, "allreduce: rank copy queue");
        }
        if (stats) {
            const auto tnow = clk::now();
            (r == 0 ? stats->sync_first : stats->sync_rest) += tnow - tprev;
            tprev = tnow;
        }
    }

    if (stats) {
        stats->submit = tf1 - tf0;
    }

    // Device-side breakdown.  The W gathers run concurrently on separate
    // links and so do the W scatters, but the two phases are separated by
    // the reduce, so the critical path is max(gather) + max(scatter).  The
    // accumulate kernels are strictly serialized on rank 0, hence the sum.
    // Gated on profiling: without it the pool carries no timestamp flag, so
    // every query below would be a driver round-trip that is guaranteed to
    // fail — 2W+1 of them per collective, 64 collectives per model step.
    if (stats && m_profiling_enabled) {
        auto tq0 = clk::now();
        auto ticks_to_ns = [this](ze_event_handle_t ev) -> uint64_t {
            ze_kernel_timestamp_result_t kt{};
            if (!ev || ze_api()->zeEventQueryKernelTimestamp(ev, &kt) != ZE_RESULT_SUCCESS) {
                return 0;
            }
            const uint64_t mask = m_ranks[0].timestamp_mask;
            const uint64_t s = kt.global.kernelStart & mask;
            const uint64_t e = kt.global.kernelEnd   & mask;
            const uint64_t delta = (e >= s) ? (e - s) : ((mask + 1 - s) + e);
            return delta * m_ranks[0].timer_ns_per_tick;
        };
        uint64_t gather_ns = 0, scatter_ns = 0, kernel_ns = 0;
        for (int w = 0; w < m_world_size - 1; ++w) {
            gather_ns  = std::max(gather_ns,  ticks_to_ns(plan.ev_recv[w]));
            scatter_ns = std::max(scatter_ns, ticks_to_ns(plan.ev_bcast[w]));
        }
        for (auto& e : plan.ev_ts_kernel) {
            kernel_ns += ticks_to_ns(e);
        }
        kernel_ns += ticks_to_ns(plan.ev_reduce);
        stats->dev_copy   = std::chrono::nanoseconds{gather_ns + scatter_ns};
        stats->dev_kernel = std::chrono::nanoseconds{kernel_ns};
        stats->dev_ts_query = clk::now() - tq0;
    }
    trace("execute: done");
}

namespace {
enum class WaitOutcome { ready, timed_out, aborted };

/// Blocks until `ready()` holds, the group is aborted, or the timeout expires.
/// A zero timeout means "wait forever" and is only reachable through explicit
/// configuration.
template <class Ready>
WaitOutcome wait_for_condition(std::unique_lock<std::mutex>& lk,
                               std::condition_variable& cv,
                               std::chrono::milliseconds timeout,
                               const std::atomic<bool>& aborted,
                               Ready ready) {
    const auto pred = [&] { return ready() || aborted.load(std::memory_order_acquire); };
    if (timeout.count() == 0) {
        cv.wait(lk, pred);
    } else if (!cv.wait_for(lk, timeout, pred)) {
        return WaitOutcome::timed_out;
    }
    // A rank that reached its goal is allowed to finish even if another rank
    // aborted in the meantime; tearing down a completed round buys nothing.
    return ready() ? WaitOutcome::ready : WaitOutcome::aborted;
}
}  // namespace

uint64_t TPDeviceCoordinator::timeout_ns() const {
    if (m_collective_timeout.count() == 0) {
        return UINT64_MAX;
    }
    return static_cast<uint64_t>(m_collective_timeout.count()) * 1'000'000ull;
}

void TPDeviceCoordinator::sync_queue(ze_command_queue_handle_t queue, const char* what) const {
    const ze_result_t r = ze_api()->zeCommandQueueSynchronize(queue, timeout_ns());
    if (r == ZE_RESULT_NOT_READY) {
        OPENVINO_THROW("[TP][L0] ", what, " did not complete within ",
                       m_collective_timeout.count(), " ms");
    }
    ZE_THROW(r);
}

void TPDeviceCoordinator::abort_all(const std::string& reason) {
    {
        std::lock_guard<std::mutex> lock(m_abort_mutex);
        if (m_abort_reason.empty()) {
            m_abort_reason = reason;
        }
    }
    m_aborted.store(true, std::memory_order_release);

    // Take each rendezvous lock before notifying: a rank that has just
    // evaluated its predicate and is about to sleep would otherwise miss the
    // wakeup and keep waiting for a group that no longer exists.
    for (auto& rdz : m_rendezvous) {
        if (!rdz) {
            continue;
        }
        std::lock_guard<std::mutex> lock(rdz->mtx);
        rdz->cv.notify_all();
    }
}

void TPDeviceCoordinator::throw_if_aborted() const {
    if (!m_aborted.load(std::memory_order_acquire)) {
        return;
    }
    std::lock_guard<std::mutex> lock(m_abort_mutex);
    OPENVINO_THROW("[TP][L0] collective group is no longer usable: ", m_abort_reason);
}

void TPDeviceCoordinator::fail_collective(bool timed_out, int collective_id, int rank, const char* stage) {
    if (timed_out) {
        std::ostringstream oss;
        oss << "[TP][L0] rank " << rank << " timed out after " << m_collective_timeout.count()
            << " ms at the " << stage << " of collective " << collective_id
            << "; the group is aborted. Either another rank never reached this collective or its "
               "device work never completed.";
        abort_all(oss.str());
    }
    throw_if_aborted();
    // throw_if_aborted always throws once m_aborted is set, which abort_all
    // guarantees above.
    OPENVINO_THROW("[TP][L0] collective ", collective_id, " failed on rank ", rank, " at the ", stage);
}

void TPDeviceCoordinator::allreduce(int collective_id,                                    int rank,
                                    void* in_dev,
                                    void* out_dev,
                                    std::size_t n,
                                    ov::element::Type dtype) {
    OPENVINO_ASSERT(m_ready, "[TP][L0] coordinator not initialized");
    OPENVINO_ASSERT(collective_id >= 0 && collective_id < m_num_collectives,
                    "[TP][L0] collective_id out of range: ", collective_id);
    OPENVINO_ASSERT(rank >= 0 && rank < m_world_size,
                    "[TP][L0] rank out of range: ", rank);
    // A null device pointer reaches the driver as a request to wrap a host
    // allocation at address 0 and makes the Level-Zero runtime abort the whole
    // process, so it has to be rejected here while it is still diagnosable.
    OPENVINO_ASSERT(in_dev != nullptr && out_dev != nullptr,
                    "[TP][L0] null device buffer passed to allreduce (collective ", collective_id,
                    ", rank ", rank, ", in=", in_dev, ", out=", out_dev, ")");
    throw_if_aborted();

    // Diagnostic escape hatch: skip the collective entirely so a run measures
    // only what each rank's GPU does on its own shard.  The difference against
    // a normal run is the whole cost of the collective -- rendezvous, submit,
    // sync and transfer -- which is otherwise impossible to separate from the
    // per-rank execution time.  Output buffers are left untouched, so results
    // are meaningless and only timings may be read from such a run.
    static const bool skip_collective = std::getenv("TP_SKIP_COLLECTIVE") != nullptr;
    if (skip_collective) {
        static std::once_flag warned;
        std::call_once(warned, [] {
            std::cerr << "[TP][L0] TP_SKIP_COLLECTIVE is set: AllReduce is a no-op, "
                         "outputs are invalid. Timing-only mode." << std::endl;
        });
        return;
    }

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
    // Not thread_local: InferRequest launches every rank through std::async,
    // so rank 0's collectives run on a fresh thread each inference.  With
    // thread-local counters the totals reset every step and a dump period
    // larger than the number of collectives per step never fires.  Outer
    // inferences are serialized by CompiledModel::lock_inference(), and
    // future::get() orders the writes, so plain statics are safe here.
    static std::chrono::nanoseconds t_phase1{}, t_phase2{}, t_exec{},
                                    t_record{}, t_phase3{};
    static std::chrono::nanoseconds t_reset{}, t_submit{},
                                    t_sync_first{}, t_sync_rest{};
    static std::chrono::nanoseconds t_dev_copy{}, t_dev_kernel{},
                                    t_dev_ts_query{};
    // ph2 minus record minus exec turned out to be the largest single item in
    // continuous batching, and none of the code in between looks expensive.
    // Split it: t_prep is barrier-exit to start of the plan work, t_tail is
    // end of execution to releasing the other ranks.
    static std::chrono::nanoseconds t_prep{}, t_tail{};
    static std::atomic<uint64_t> n_calls{0}, n_rebuild{0}, n_record{0};
    static uint64_t t_copy_bytes{0}, t_copy_bytes_total{0};
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
            const auto outcome = wait_for_condition(lk, rdz.cv, m_collective_timeout, m_aborted,
                                                    [&] { return rdz.enter_gen != my_gen; });
            if (outcome != WaitOutcome::ready) {
                lk.unlock();
                fail_collective(outcome == WaitOutcome::timed_out, collective_id, rank, "enter barrier");
            }
        }
    }
    trace("phase1: passed");
    auto t1 = clk::now();

    // Phase 2: rank 0 orchestrates; non-zero ranks wait for `done`.
    if (rank == 0) {
        // Rank 0 owns the launch.  Every other rank is parked on `done`, so if
        // anything below throws they would wait for a result that will never
        // be produced.  Releasing them is the whole point of the guard.
        struct AbortOnFailure {
            TPDeviceCoordinator* self;
            int collective_id;
            bool armed{true};
            ~AbortOnFailure() {
                if (armed) {
                    self->abort_all("[TP][L0] rank 0 failed while running collective " +
                                    std::to_string(collective_id));
                }
            }
        } abort_guard{this, collective_id};

        OPENVINO_ASSERT(rdz.n == n && rdz.dtype == dtype,
                        "[TP][L0] inconsistent (n,dtype) across ranks for collective ",
                        collective_id);
        for (int i = 0; i < m_world_size; ++i) {
            OPENVINO_ASSERT(rdz.in_ptrs[i] && rdz.out_ptrs[i],
                            "[TP][L0] collective ", collective_id, " has no buffers for rank ", i,
                            "; the barrier was released by ", m_world_size,
                            " arrivals that did not cover every rank");
        }

        Plan* const slot = m_plans[collective_id].get();

        const auto payload_bytes = collective_payload_bytes(n, dtype);
        const bool scratch_grew = ensure_scratch_capacity(payload_bytes);
        const bool resources_missing = slot->pool == nullptr;
        const bool signature_matches = slot->matches(rdz.in_ptrs, rdz.out_ptrs, n, dtype);
        const bool recorded_matches = !m_use_immediate &&
                                      slot->recorded &&
                                      slot->recorded_scratch_generation == m_scratch.generation &&
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
            slot->recorded = true;
            slot->recorded_scratch_generation = m_scratch.generation;
            ++n_record;
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
        t_sync_first += es.sync_first;
        t_sync_rest  += es.sync_rest;
        t_dev_copy     += es.dev_copy;
        t_dev_kernel   += es.dev_kernel;
        t_dev_ts_query += es.dev_ts_query;
        t_copy_bytes   += es.copy_bytes;
        t_copy_bytes_total += es.copy_bytes_total;
        t_prep         += tr0 - t1;

        {
            std::unique_lock<std::mutex> lk(rdz.mtx);
            abort_guard.armed = false;
            rdz.done = true;
            rdz.cv.notify_all();
        }
        t_tail += clk::now() - te1;
    } else {
        std::unique_lock<std::mutex> lk(rdz.mtx);
        const auto outcome = wait_for_condition(lk, rdz.cv, m_collective_timeout, m_aborted,
                                                [&] { return rdz.done; });
        if (outcome != WaitOutcome::ready) {
            lk.unlock();
            fail_collective(outcome == WaitOutcome::timed_out, collective_id, rank, "execute phase");
        }
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
            const auto outcome = wait_for_condition(lk, rdz.cv, m_collective_timeout, m_aborted,
                                                    [&] { return rdz.exit_gen != my_gen; });
            if (outcome != WaitOutcome::ready) {
                lk.unlock();
                fail_collective(outcome == WaitOutcome::timed_out, collective_id, rank, "exit barrier");
            }
        }
    }
    trace("phase3: passed (return)");

    auto t3 = clk::now();
    // Only rank 0's phases are accumulated, to match record/exec/prep/tail and
    // the call counter below.  Adding every rank here would double the phase
    // totals while the inner breakdown stayed single-rank.
    if (rank == 0) {
        t_phase1 += t1 - t0;
        t_phase2 += t2 - t1;
        t_phase3 += t3 - t2;
    }
    if (prof_every > 0 && rank == 0) {
        uint64_t c = ++n_calls;
        if ((c % prof_every) == 0) {
            using ms = std::chrono::duration<double, std::milli>;
            const double cf = static_cast<double>(c);
            std::cerr << "[TP][PROF] r0 calls=" << c
                      << " rebuilds=" << n_rebuild.load()
                      << " records=" << n_record.load()
                      << "  totals: ph1=" << ms(t_phase1).count() << "ms"
                      << " ph2=" << ms(t_phase2).count() << "ms"
                      << " (record=" << ms(t_record).count() << "ms"
                      << ", exec=" << ms(t_exec).count() << "ms)"
                      << " ph3=" << ms(t_phase3).count() << "ms"
                      << "  per-call: ph1=" << ms(t_phase1).count() / cf << "ms"
                      << " prep=" << ms(t_prep).count() / cf << "ms"
                      << " exec=" << ms(t_exec).count() / cf << "ms"
                      << " tail=" << ms(t_tail).count() / cf << "ms"
                      << " ph3=" << ms(t_phase3).count() / cf << "ms"
                      << std::endl;
            // exec breakdown: reset events / submit / sync of the first queue
            // we waited on / sum of the remaining ones.  sync_first absorbs
            // the wait for whichever rank was slowest up to that point;
            // sync_rest > 0 means a later rank was the straggler.
            const double exec_total = ms(t_exec).count();
            const double r_pct = exec_total > 0 ? 100.0 * ms(t_reset).count()  / exec_total : 0;
            const double s_pct = exec_total > 0 ? 100.0 * ms(t_submit).count() / exec_total : 0;
            const double a_pct = exec_total > 0 ? 100.0 * ms(t_sync_first).count() / exec_total : 0;
            const double b_pct = exec_total > 0 ? 100.0 * ms(t_sync_rest).count()  / exec_total : 0;
            std::cerr << "[TP][PROF]   exec breakdown (world=" << m_world_size << "):"
                      << " reset="  << ms(t_reset).count()  / cf << "ms (" << r_pct << "%)"
                      << " submit=" << ms(t_submit).count() / cf << "ms (" << s_pct << "%)"
                      << " sync_first=" << ms(t_sync_first).count() / cf << "ms (" << a_pct << "%)"
                      << " sync_rest="  << ms(t_sync_rest).count()  / cf << "ms (" << b_pct << "%)"
                      << std::endl;
            // Device-side per-step timing along the critical path.
            const double ts_query_per = ms(t_dev_ts_query).count() / cf;
            std::cerr << "[TP][PROF]   device steps:"
                      << " memcpy=" << ms(t_dev_copy).count()   / cf << "ms"
                      << " kernel=" << ms(t_dev_kernel).count() / cf << "ms"
                      << " (sum=" << (ms(t_dev_copy).count() + ms(t_dev_kernel).count()) / cf << "ms)"
                      << " ts_query_overhead=" << ts_query_per << "ms"
                      << std::endl;
            // Link bandwidth is per single transfer: on N=2 the two transfers
            // run concurrently in opposite directions, on the funnel the W
            // gathers run concurrently on separate links.  The aggregate line
            // is what actually crosses device boundaries per collective and
            // is the number the funnel bottlenecks on.
            const double bytes_per_link  = cf > 0 ? double(t_copy_bytes) / cf : 0.0;
            const double bytes_total     = cf > 0 ? double(t_copy_bytes_total) / cf : 0.0;
            const double dev_copy_s      = ms(t_dev_copy).count() / cf / 1.0e3;
            const double bw_per_dir_gbs  = dev_copy_s > 0 ? (bytes_per_link / 1.0e9) / dev_copy_s : 0.0;
            const double bw_aggregate_gbs= dev_copy_s > 0 ? (bytes_total / 1.0e9) / dev_copy_s : 0.0;
            const double exec_per_s      = ms(t_exec).count() / cf / 1.0e3;
            const double bw_walltime_gbs = exec_per_s > 0 ? (bytes_total / 1.0e9) / exec_per_s : 0.0;
            std::cerr << "[TP][PROF]   throughput:"
                      << " bytes/link=" << (bytes_per_link / (1024.0 * 1024.0)) << "MB"
                      << " bytes/call=" << (bytes_total / (1024.0 * 1024.0)) << "MB"
                      << "  per-link(dev_copy)=" << bw_per_dir_gbs << " GB/s"
                      << "  aggregate(dev_copy)=" << bw_aggregate_gbs << " GB/s"
                      << "  effective(exec)=" << bw_walltime_gbs << " GB/s"
                      << std::endl;
            // Where re-recording time goes.  Divided by the number of
            // recordings, not by calls: recording is rare but every miss
            // lands in time to first token.
            const double rc = std::max<double>(1.0, static_cast<double>(n_record.load()));
            std::cerr << "[TP][PROF]   record breakdown: records=" << n_record.load()
                      << " per-record=" << ms(t_record).count() / rc << "ms"
                      << " (drain=" << ms(m_rec_drain).count() / rc << "ms"
                      << " reset=" << ms(m_rec_reset).count() / rc << "ms"
                      << " append=" << ms(m_rec_build - m_rec_close).count() / rc << "ms"
                      << " close=" << ms(m_rec_close).count() / rc << "ms)"
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

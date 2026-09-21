// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tp_gpu/tp_device_coordinator.hpp"

#include <algorithm>
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
#include <thread>

#define CL_TARGET_OPENCL_VERSION 220
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "openvino/core/except.hpp"
#include "openvino/zero_api.hpp"
#include "tp_embedded_kernels.h"
#include "tp_l0_shared_context.hpp"
#include "tp_ze_throw.hpp"

namespace ov {
namespace tp_gpu {

namespace {

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

// How much a halving step may exceed half of the range it splits.
//
// halving_mid snaps the boundary down to kRingAlignElems, so the two halves are
// not equal: whoever keeps the lower one sends the upper, which can be longer
// than half by almost a full alignment unit.  Sizing a step's staging region at
// exactly half therefore lets the transfer run into the next step's region,
// where a partner delivering the following step overwrites it.  That showed up
// as a payload of 1021 f32 on four ranks losing its last 125 elements -- and
// only sometimes, because it is a race between two devices.
constexpr std::size_t kHalvingSlackBytes = kRingAlignElems * 4;  // f32 is the widest element

// Payload ceiling for recursive halving, in bytes.
//
// Halving and the ring move the same 1.5*S per rank, but they place it
// differently: the ring sends one way around the loop, so every link carries
// one transfer, while halving has both partners of a pair pushing at each
// other across the same link at once.  At decode sizes that costs nothing and
// the two saved round trips dominate; at prompt sizes it measured 128 ms to
// first token against the ring's 90.  So the schedule is chosen by payload,
// which is also where the two algorithms genuinely differ: latency-bound
// versus bandwidth-bound.

// Elements folded by one work item of the reduce kernel; must match TP_VEC in
// kernels/allreduce_sum.cl.
constexpr std::size_t kElemsPerItem = 8;

// Work groups needed to fold `n` elements.
uint32_t launch_groups(std::size_t n, uint32_t group_size) {
    const std::size_t items = (n + kElemsPerItem - 1) / kElemsPerItem;
    return static_cast<uint32_t>((items + group_size - 1) / group_size);
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
std::vector<uint8_t> compile_via_ocl(ze_device_handle_t ze_dev, const char* src) {
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
                                         std::chrono::milliseconds collective_timeout,
                                         const TPConfig& config)
    : m_shared(std::move(shared)),
      m_world_size(world_size),
      m_num_collectives(num_collectives),
      m_collective_timeout(collective_timeout),
      m_config(config) {
    OPENVINO_ASSERT(m_shared && m_shared->context && static_cast<int>(m_shared->devices.size()) == world_size,
                    "[TP][L0] coordinator requires a valid shared L0 context with ", world_size, " devices");
    OPENVINO_ASSERT(world_size >= 2, "[TP][L0] coordinator requires at least 2 ranks");
    OPENVINO_ASSERT(m_collective_timeout.count() >= 0, "[TP][L0] collective timeout must not be negative");

    // The ring only exists for N>2; two ranks already exchange directly,
    // which is what the ring degenerates to minus a round of latency.
    m_use_ring = world_size > 2;

    m_ranks.resize(world_size);
    for (int r = 0; r < world_size; ++r) {
        m_ranks[r].device = m_shared->devices[r];
    }
    m_scratch.allocations.assign(world_size, nullptr);
    m_scratch.bytes_per_rank.assign(world_size, 0);

    m_started = std::make_unique<std::atomic<uint64_t>[]>(world_size);
    for (int r = 0; r < world_size; ++r) {
        m_started[r].store(0, std::memory_order_relaxed);
    }

    const auto n_ranks = static_cast<std::size_t>(world_size);
    for (auto* c : {&m_skew.ph1_ns,       &m_skew.ph2_ns,        &m_skew.late_ns,
                    &m_skew.last_count,   &m_skew.seg_ns,        &m_skew.seg_max_ns,
                    &m_skew.seg_count,    &m_skew.p2_gate_ns,    &m_skew.p2_rec_ns,
                    &m_skew.p2_wait_ns,   &m_skew.p2_reset_ns,   &m_skew.p2_append_ns,
                    &m_skew.p2_rec_count, &m_skew.p2_wait_count, &m_skew.p2_block_count,
                    &m_skew.p2_dev_ns,    &m_skew.p2_dev_max_ns, &m_skew.p2_dev_count,
                    &m_skew.p2_dev_max_cid, &m_skew.dev_gap_ns,  &m_skew.dev_gap_count,
                    &m_skew.dev_bytes,    &m_skew.n_pair,        &m_skew.n_ring,
                    &m_skew.n_halving,    &m_skew.gate_fast,     &m_skew.gate_slow,
                    &m_gather.barrier_ns, &m_gather.append_ns,   &m_gather.calls,
                    &m_gather.spliced,    &m_gather.records,     &m_gather.dev_ns,
                    &m_gather.dev_max_ns, &m_gather.dev_count,   &m_gather.bytes,
                    &m_ahead.now,         &m_ahead.sum,          &m_ahead.max,
                    &m_ahead.count}) {
        c->assign(n_ranks, Counter{});
    }
    m_skew.p2_dev_hist.assign(n_ranks * static_cast<std::size_t>(kDevBuckets), Counter{});
    m_skew.dev_last_end_ticks.assign(n_ranks, 0);
    m_skew.dev_last_cid.assign(n_ranks, -1);
    // The only counter that is a running minimum, so it cannot start at zero.
    m_skew.seg_min_ns.assign(n_ranks, Counter{~uint64_t{0}});
    m_skew.last_exit.assign(n_ranks, std::chrono::steady_clock::time_point{});

    try {
        for (int r = 0; r < world_size; ++r) {
            init_rank(m_ranks[r]);
        }
    } catch (...) {
        for (auto& rs : m_ranks) destroy_rank(rs);
        throw;
    }

    m_rendezvous.resize(num_collectives);
    m_plans.resize(static_cast<std::size_t>(num_collectives) *
                   static_cast<std::size_t>(plan_buffers()));
    try {
        for (int i = 0; i < num_collectives; ++i) {
            auto rdz = std::make_unique<Rendezvous>();
            rdz->in_ptrs[0].assign(world_size, nullptr);
            rdz->in_ptrs[1].assign(world_size, nullptr);
            rdz->out_ptrs[0].assign(world_size, nullptr);
            rdz->out_ptrs[1].assign(world_size, nullptr);
            m_rendezvous[i] = std::move(rdz);

            // Command lists are cheap to keep but not to create: making them
            // lazily put ~50 ms of driver work on the first inference, which
            // is the one whose latency users measure as time to first token.
            // The events are built here for the same reason.
            for (int b = 0; b < plan_buffers(); ++b) {
                auto plan = std::make_unique<Plan>();
                plan->buffer = b;
                create_plan_events(*plan);
                m_plans[static_cast<std::size_t>(i) *
                            static_cast<std::size_t>(plan_buffers()) +
                        static_cast<std::size_t>(b)] = std::move(plan);
            }
        }
    } catch (...) {
        for (auto& plan : m_plans) {
            if (plan) destroy_plan(*plan);
        }
        for (auto& rs : m_ranks) destroy_rank(rs);
        throw;
    }

    m_ready = true;
    TP_LOG_INFO << "[TP][L0] coordinator ready: " << world_size << " ranks, "
                << num_collectives << " collective slots" << std::endl;
}

TPDeviceCoordinator::~TPDeviceCoordinator() {
    // Before anything it watches goes away.
    stop_watchdog();
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

    ZE_THROW(ov::zeCommandQueueCreate(ctx, dev, &qd, &rs.compute_queue));

    // No rank-wide command list: each collective owns its own, so a
    // recording is not clobbered by the next collective.

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
    std::vector<uint8_t> native_bin = compile_via_ocl(dev, src);

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
}

void TPDeviceCoordinator::destroy_rank(RankState& rs) {
    if (rs.kernel_f16)   { ov::zeKernelDestroy(rs.kernel_f16);   rs.kernel_f16 = nullptr; }
    if (rs.kernel_f32)   { ov::zeKernelDestroy(rs.kernel_f32);   rs.kernel_f32 = nullptr; }
    if (rs.module)       { ov::zeModuleDestroy(rs.module);       rs.module = nullptr; }
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
        }
    }

    for (auto& e : plan.ev_recv)  if (e) ov::zeEventDestroy(e);
    for (auto& e : plan.ev_ring)  if (e) ov::zeEventDestroy(e);
    for (auto& e : plan.ev_gather) if (e) ov::zeEventDestroy(e);
    if (plan.gather_pool) ov::zeEventPoolDestroy(plan.gather_pool);
    if (plan.pool)      ov::zeEventPoolDestroy(plan.pool);
    for (auto& e : plan.ev_ts_kernel) if (e) ov::zeEventDestroy(e);
    for (auto& e : plan.ev_done) if (e) ov::zeEventDestroy(e);
    if (plan.done_pool) ov::zeEventPoolDestroy(plan.done_pool);
    for (auto& l : plan.compute_lists) if (l) ov::zeCommandListDestroy(l);
    plan.ev_recv.clear();
    plan.ev_ring.clear();
    plan.ev_gather.clear();
    plan.gather_pool = nullptr;
    plan.pool = nullptr;
    plan.ev_ts_kernel.clear();
    plan.ev_done.clear();
    plan.spliced_bytes.clear();
    plan.in_flight.reset();
    plan.done_pool = nullptr;
    plan.compute_lists.clear();
    std::fill(plan.recorded.begin(), plan.recorded.end(), 0);
    std::fill(plan.recorded_scratch_generation.begin(),
              plan.recorded_scratch_generation.end(), 0);
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
    const bool measure = m_config.profiling_host();
    const auto stall_start = measure ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};

    std::vector<void*> new_allocations(static_cast<std::size_t>(m_world_size), nullptr);
    std::vector<std::size_t> new_bytes_per_rank(static_cast<std::size_t>(m_world_size), 0);
    ze_device_mem_alloc_desc_t mad{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr};

    // Ring: N slots per rank, one per chunk, so concurrent steps never share
    // a slot.  The stride is the largest chunk rounded up for alignment; the
    // total lands within one payload per rank.  The halving slack rides along:
    // halving lays its steps out inside the same region, and its steps are
    // slightly larger than half of what they split (see kHalvingSlackBytes).
    const std::size_t chunk_stride =
        m_use_ring ? align_up((payload_bytes + m_world_size - 1) / m_world_size + kHalvingSlackBytes,
                              256)
                   : 0;

    try {
        if (m_use_ring) {
            // One set of N chunk slots per buffer: the recordings of two
            // consecutive instances of the same collective must not land in
            // the same bytes, because a neighbour may still be reading them.
            const auto slots = checked_multiply(static_cast<std::size_t>(m_world_size),
                                                static_cast<std::size_t>(plan_buffers()),
                                                "ring scratch buffers");
            const auto ring_bytes = checked_multiply(chunk_stride, slots, "ring scratch");
            for (int rank = 0; rank < m_world_size; ++rank) {
                ZE_THROW(ov::zeMemAllocDevice(m_shared->context,
                                              &mad,
                                              ring_bytes,
                                              256,
                                              m_ranks[rank].device,
                                              &new_allocations[rank]));
                new_bytes_per_rank[rank] = ring_bytes;
            }
        } else {
            const auto pair_bytes = checked_multiply(payload_bytes,
                                                     static_cast<std::size_t>(plan_buffers()),
                                                     "pair scratch");
            for (int rank = 0; rank < m_world_size; ++rank) {
                ZE_THROW(ov::zeMemAllocDevice(m_shared->context,
                                              &mad,
                                              pair_bytes,
                                              64,
                                              m_ranks[rank].device,
                                              &new_allocations[rank]));
                new_bytes_per_rank[rank] = pair_bytes;
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

    try {
        // Every execute path synchronizes before returning.  Synchronize
        // again before replacing addresses embedded in recorded lists.  If
        // synchronization fails, discard only the newly allocated arena and
        // keep the old one intact.
        for (auto& rs : m_ranks) {
            if (rs.compute_queue) {
                sync_queue(rs.compute_queue, "scratch grow: compute queue drain");
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
    m_scratch.allocation_count += m_world_size;

    if (TP_VERBOSE_AT_LEAST(ov::log::Level::INFO)) {
        ov::tp_gpu::log_stream() << "[TP][MEM] scratch grow generation=" << m_scratch.generation
                                 << " payload_capacity=" << m_scratch.payload_capacity_bytes
                                 << " total=" << m_scratch.total_allocated_bytes;
        for (int rank = 0; rank < m_world_size; ++rank) {
            ov::tp_gpu::log_stream() << " r" << rank << "=" << m_scratch.bytes_per_rank[rank];
        }
        ov::tp_gpu::log_stream() << std::endl;
    }
    if (measure) {
        m_scratch_stall_ns.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() - stall_start)
                                      .count()),
            std::memory_order_relaxed);
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

void* TPDeviceCoordinator::scratch_buffer(int index, int buffer) const {
    if (m_world_size == 2) {
        OPENVINO_ASSERT(index >= 0 && index < m_world_size,
                        "[TP][L0] scratch rank out of range: ", index);
        auto* base = static_cast<uint8_t*>(m_scratch.allocations[static_cast<std::size_t>(index)]);
        return base + static_cast<std::size_t>(buffer) * m_scratch.payload_capacity_bytes;
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

void* TPDeviceCoordinator::ring_slot(int rank, int chunk, int buffer) const {
    OPENVINO_ASSERT(rank >= 0 && rank < m_world_size, "[TP][L0] ring slot rank out of range: ", rank);
    OPENVINO_ASSERT(chunk >= 0 && chunk < m_world_size, "[TP][L0] ring slot chunk out of range: ", chunk);
    OPENVINO_ASSERT(buffer >= 0 && buffer < plan_buffers(),
                    "[TP][L0] ring slot buffer out of range: ", buffer);
    auto* base = static_cast<uint8_t*>(m_scratch.allocations[static_cast<std::size_t>(rank)]);
    const auto slot = static_cast<std::size_t>(buffer) * static_cast<std::size_t>(m_world_size) +
                      static_cast<std::size_t>(chunk);
    return base + slot * m_scratch.chunk_capacity_bytes;
}

// Where the payload splits at one level of the recursion.  Both partners
// compute this from the same range, so they always agree on who keeps which
// half.  The boundary is snapped down to the kernel's alignment for the same
// reason ring_chunk snaps: a source pointer that is not naturally aligned
// corrupts a handful of elements at the head of the range.  A range shorter
// than one alignment unit collapses to an empty half, which the schedule
// handles by signalling the step without moving anything.
static std::size_t halving_mid(std::size_t lo, std::size_t hi) {
    const std::size_t align = kRingAlignElems;
    if (hi <= lo) {
        return lo;
    }
    std::size_t mid = lo + (hi - lo) / 2;
    mid = (mid / align) * align;
    if (mid < lo) {
        mid = lo;
    }
    if (mid > hi) {
        mid = hi;
    }
    return mid;
}

void TPDeviceCoordinator::halving_range(std::size_t n, int rank, int level,
                                        std::size_t& lo, std::size_t& hi) const {
    lo = 0;
    hi = n;
    for (int j = 0; j < level; ++j) {
        const std::size_t mid = halving_mid(lo, hi);
        if (((static_cast<unsigned>(rank) >> j) & 1u) != 0u) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
}

void* TPDeviceCoordinator::halving_stage(int rank, int step, int buffer) const {
    // Each step stages at most half of what the previous one left, plus the
    // alignment slack halving_mid can introduce.  Laying the steps end to end
    // still costs about one payload, which the ring arena reserves per buffer
    // -- ensure_scratch_capacity adds the slack on top.
    std::size_t offset = 0;
    std::size_t span = m_scratch.payload_capacity_bytes;
    auto step_bytes = [](std::size_t range) {
        return (range + 1) / 2 + kHalvingSlackBytes;
    };
    for (int j = 0; j < step; ++j) {
        const std::size_t staged = step_bytes(span);
        offset += align_up(staged, 256);
        span = staged;
    }
    const std::size_t region =
        static_cast<std::size_t>(m_world_size) * m_scratch.chunk_capacity_bytes;
    OPENVINO_ASSERT(offset + align_up(step_bytes(span), 256) <= region,
                    "[TP][L0] halving staging does not fit the ring arena");
    return static_cast<uint8_t*>(ring_slot(rank, 0, buffer)) + offset;
}

void TPDeviceCoordinator::create_plan_events(Plan& plan) {
    auto ctx = m_shared->context;
    const int N = m_world_size;

    plan.recorded.assign(static_cast<std::size_t>(N), 0);
    plan.recorded_scratch_generation.assign(static_cast<std::size_t>(N), 0);

    if (plan.pool != nullptr) {
        return;
    }

    // Completion events for the spliced path.  Their own pool because they
    // are the only host-visible events here: everything else is signalled and
    // waited on entirely by devices.
    {
        ze_event_pool_desc_t epd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr};
        // The timestamp flag is what makes the device side of a spliced
        // collective measurable at all, and it makes the command processor
        // record a start and an end on every signal, so it is only asked for
        // when someone is going to read them.
        epd.flags = ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
        if (m_config.profiling_device()) {
            epd.flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
        }
        epd.count = static_cast<uint32_t>(N);
        std::vector<ze_device_handle_t> devs_nc(m_shared->devices.begin(), m_shared->devices.end());
        ZE_THROW(ov::zeEventPoolCreate(ctx, &epd,
                                       static_cast<uint32_t>(devs_nc.size()),
                                       devs_nc.data(), &plan.done_pool));
        plan.ev_done.assign(static_cast<std::size_t>(N), nullptr);
        plan.spliced_bytes.assign(static_cast<std::size_t>(N), 0);
        plan.in_flight = std::make_unique<std::atomic<uint8_t>[]>(static_cast<std::size_t>(N));
        for (int r = 0; r < N; ++r) {
            plan.in_flight[r].store(0, std::memory_order_relaxed);
        }
        for (int r = 0; r < N; ++r) {
            ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr};
            ed.signal = ZE_EVENT_SCOPE_FLAG_HOST;
            ed.wait   = ZE_EVENT_SCOPE_FLAG_HOST;
            ed.index  = static_cast<uint32_t>(r);
            ZE_THROW(ov::zeEventCreate(plan.done_pool, &ed, &plan.ev_done[r]));
        }
    }

    // Command lists are per collective, so a recording is not clobbered by
    // the next collective's.
    ensure_plan_lists(plan);

    if (N == 2) {
        // Symmetric N=2: every rank pushes its `in` to peer's local staging,
        // then runs its own reduce kernel `out = in + staging` on its queue.
        // Both queues fire in parallel — critical path is one cross-device
        // memcpy + one kernel.
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
        const bool prof_on = m_config.profiling_device();
        epd.count = prof_on ? 4u : 2u;
        std::vector<ze_device_handle_t> devs_nc(m_shared->devices.begin(), m_shared->devices.end());
        ZE_THROW(ov::zeEventPoolCreate(ctx, &epd,
                                       static_cast<uint32_t>(devs_nc.size()),
                                       devs_nc.data(), &plan.pool));

        plan.ev_recv.resize(2);   // [r] = "rank r finished pushing"
        for (int r = 0; r < 2; ++r) {
            ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr};
            ed.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
            ed.wait   = ZE_EVENT_SCOPE_FLAG_DEVICE;
            ed.index  = static_cast<uint32_t>(r);
            ZE_THROW(ov::zeEventCreate(plan.pool, &ed, &plan.ev_recv[r]));
        }

        // Reuse plan.pool for kernel-end timestamp events; ev_recv[] already
        // serves as the copy-end probe.
        // Always size to 2 so record_pair_rank can index plan.ev_ts_kernel[r] —
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
    // Everything past the N==2 early return is a ring: the pair exchange is
    // the only other schedule there is.
    const int steps = 2 * (N - 1);
    ze_event_pool_desc_t epd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr};
    epd.flags = m_config.profiling_device() ? ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP : 0;
    // One event per (step, rank), plus one "reduce-scatter finished" per
    // rank in a trailing row.
    epd.count = static_cast<uint32_t>((steps + 1) * N);
    std::vector<ze_device_handle_t> devs_nc(m_shared->devices.begin(), m_shared->devices.end());
    ZE_THROW(ov::zeEventPoolCreate(ctx, &epd,
                                   static_cast<uint32_t>(devs_nc.size()),
                                   devs_nc.data(), &plan.pool));

    plan.ev_recv.clear();
    plan.ev_ts_kernel.clear();

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
}

void TPDeviceCoordinator::ensure_plan_lists(Plan& plan) {
    if (!plan.compute_lists.empty()) {
        return;
    }

    auto ctx = m_shared->context;
    plan.compute_lists.assign(static_cast<std::size_t>(m_world_size), nullptr);
    for (int r = 0; r < m_world_size; ++r) {
        auto& rs = m_ranks[r];
        ze_command_list_desc_t ld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr};
        ld.commandQueueGroupOrdinal = rs.compute_ordinal;
        ZE_THROW(ov::zeCommandListCreate(ctx, rs.device, &ld, &plan.compute_lists[r]));
    }
}

void TPDeviceCoordinator::record_ring_rank(Plan& plan, int r) {
    constexpr uint32_t kGroupSize = 256;
    const int N = m_world_size;
    const int steps = N - 1;
    const std::size_t n = plan.n;
    const std::size_t elem = plan.dtype.size();
    OPENVINO_ASSERT(n > 0, "[TP][L0] ring requires a non-empty payload");

    auto ev = [&](int step, int who) -> ze_event_handle_t {
        return plan.ev_ring[static_cast<std::size_t>(step) * static_cast<std::size_t>(N) +
                            static_cast<std::size_t>(who)];
    };
    auto byte_at = [elem](void* base, std::size_t offset_elems) -> void* {
        return static_cast<uint8_t*>(base) + offset_elems * elem;
    };
    // The chain has to be spelled out: the send of the next step reads what
    // this step's kernel produced, and the tail resets must not overtake the
    // waits that consumed those events.
    auto order = [&](ze_command_list_handle_t list) {
        ZE_THROW(ze_api()->zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    };

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
                ring_slot(next, send_chunk, plan.buffer),
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
        void* src1 = ring_slot(r, recv_chunk, plan.buffer);
        uint64_t cnt = recv_cnt;
        ze_group_count_t gc{launch_groups(recv_cnt, kGroupSize), 1, 1};
        ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &dst));
        ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &src0));
        ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &src1));
        ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 3, sizeof(cnt), &cnt));
        ZE_THROW(ze_api()->zeCommandListAppendLaunchKernel(list, kernel, &gc, nullptr, 0, nullptr));
        order(list);
    }

    // ---- All-gather: N-1 steps ----
    // Rank r now owns the finished chunk (r+1).  Each step passes a
    // finished chunk along the same ring -- no kernel, because there is
    // nothing left to reduce.
    //
    // Chunks travel through the successor's staging rather than straight
    // into its output buffer, and each rank copies what arrives into its
    // own output itself.  Writing into a peer's output would mean baking
    // that peer's address into this recording, which is the one thing
    // keeping every rank from recording and submitting independently of
    // the others.  Someone has to write those chunks into our output --
    // either the producer or us -- so the local copy is the price of that
    // independence, not an accident: (N-1)/N of the payload per rank.
    //
    // Writing into the successor's staging is only safe once that rank has
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
        const int recv_chunk = ((r - s) % N + N) % N;

        std::size_t send_off = 0, send_cnt = 0, recv_off = 0, recv_cnt = 0;
        ring_chunk(n, send_chunk, send_off, send_cnt);
        ring_chunk(n, recv_chunk, recv_off, recv_cnt);

        const int step = steps + s;
        if (send_cnt == 0) {
            ZE_THROW(ze_api()->zeCommandListAppendSignalEvent(list, ev(step, r)));
        } else {
            // Step 0 forwards the chunk this rank reduced into its own
            // output; every later step forwards what the previous step
            // delivered into our staging, which the wait below ordered.
            void* send_src = (s == 0) ? byte_at(plan.out_ptrs[r], send_off)
                                      : ring_slot(r, send_chunk, plan.buffer);
            ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(
                list,
                ring_slot(next, send_chunk, plan.buffer),
                send_src,
                send_cnt * elem,
                ev(step, r),
                0, nullptr));
        }

        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &plan.ev_ring[
            static_cast<std::size_t>(step) * static_cast<std::size_t>(N) +
            static_cast<std::size_t>(prev)]));

        // Deliver what just arrived into our own output.  Nothing else
        // reads that range, and the wait above already places this after
        // the transfer that produced it, so it needs no ordering of its own.
        if (recv_cnt > 0) {
            ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(
                list,
                byte_at(plan.out_ptrs[r], recv_off),
                ring_slot(r, recv_chunk, plan.buffer),
                recv_cnt * elem,
                nullptr,
                0, nullptr));
        }
    }

    // Every ring event has exactly one waiter, so each rank clears the
    // ones it consumed: the per-step events of its predecessor and the
    // reduce-scatter handshake of its successor.  The in-order list
    // already orders these behind the waits above.
    order(list);
    for (int s = 0; s < 2 * steps; ++s) {
        ZE_THROW(ze_api()->zeCommandListAppendEventReset(list, ev(s, prev)));
    }
    ZE_THROW(ze_api()->zeCommandListAppendEventReset(list, ev(2 * steps, next)));

    close_list(list, m_rec_close_ns);
}

void TPDeviceCoordinator::record_halving_rank(Plan& plan, int r) {
    constexpr uint32_t kGroupSize = 256;
    const int N = m_world_size;
    const std::size_t n = plan.n;
    const std::size_t elem = plan.dtype.size();
    OPENVINO_ASSERT(n > 0, "[TP][L0] halving requires a non-empty payload");

    int levels = 0;
    while ((1 << levels) < N) {
        ++levels;
    }
    OPENVINO_ASSERT((1 << levels) == N, "[TP][L0] halving requires a power-of-two world size");

    // Steps are numbered 0..2*levels-1: the first half scatters, the second
    // gathers.  ev(step, who) is signalled by `who` and waited on by its
    // partner at that step, so every event has exactly one waiter.
    auto ev = [&](int step, int who) -> ze_event_handle_t {
        return plan.ev_ring[static_cast<std::size_t>(step) * static_cast<std::size_t>(N) +
                            static_cast<std::size_t>(who)];
    };
    auto byte_at = [elem](void* base, std::size_t offset_elems) -> void* {
        return static_cast<uint8_t*>(base) + offset_elems * elem;
    };
    auto order = [&](ze_command_list_handle_t list) {
        ZE_THROW(ze_api()->zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
    };

    auto& self = m_ranks[r];
    ze_command_list_handle_t list = plan.compute_lists[r];
    ze_kernel_handle_t kernel =
        (plan.dtype == ov::element::f16) ? self.kernel_f16 : self.kernel_f32;
    ZE_THROW(ze_api()->zeKernelSetGroupSize(kernel, kGroupSize, 1, 1));

    // ---- Scatter: levels steps, each moving half of what is left ----
    for (int j = 0; j < levels; ++j) {
        const int partner = r ^ (1 << j);
        std::size_t lo = 0, hi = 0;
        halving_range(n, r, j, lo, hi);
        const std::size_t mid = halving_mid(lo, hi);
        const bool keep_upper = ((static_cast<unsigned>(r) >> j) & 1u) != 0u;
        const std::size_t keep_lo = keep_upper ? mid : lo;
        const std::size_t keep_hi = keep_upper ? hi : mid;
        const std::size_t send_lo = keep_upper ? lo : mid;
        const std::size_t send_hi = keep_upper ? mid : hi;

        // At the first step our contribution is still the untouched input;
        // afterwards it is the partial sum the previous step left behind.
        void* mine = (j == 0) ? plan.in_ptrs[r] : plan.out_ptrs[r];

        if (send_hi > send_lo) {
            // The partner keeps this half, so it lands in the partner's
            // staging for this step.  It goes at the start of that region,
            // not at the offset it occupies in the payload: the region is
            // only as large as one half, and the receiving kernel reads it
            // from the start too.  Starting at the region base also keeps
            // the source naturally aligned, which the vectorized kernel
            // needs.
            ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(
                list,
                halving_stage(partner, j, plan.buffer),
                byte_at(mine, send_lo),
                (send_hi - send_lo) * elem,
                ev(j, r),
                0, nullptr));
        } else {
            ZE_THROW(ze_api()->zeCommandListAppendSignalEvent(list, ev(j, r)));
        }

        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &plan.ev_ring[
            static_cast<std::size_t>(j) * static_cast<std::size_t>(N) +
            static_cast<std::size_t>(partner)]));

        if (keep_hi > keep_lo) {
            void* dst  = byte_at(plan.out_ptrs[r], keep_lo);
            void* src0 = byte_at(mine, keep_lo);
            void* src1 = halving_stage(r, j, plan.buffer);
            uint64_t cnt = keep_hi - keep_lo;
            ze_group_count_t gc{launch_groups(keep_hi - keep_lo, kGroupSize), 1, 1};
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &dst));
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &src0));
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &src1));
            ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 3, sizeof(cnt), &cnt));
            ZE_THROW(ze_api()->zeCommandListAppendLaunchKernel(list, kernel, &gc, nullptr, 0, nullptr));
        }
        // Nothing but a barrier orders a kernel before the copy that reads
        // what it wrote; the waits order everything else.
        order(list);
    }

    // ---- Gather: the same partners in reverse, writing into their output ----
    // Each rank owns a distinct range, so the writers never overlap and no
    // staging -- and no local delivery copy -- is needed at all.
    for (int j = levels - 1; j >= 0; --j) {
        const int partner = r ^ (1 << j);
        const int step = 2 * levels - 1 - j;
        std::size_t lo = 0, hi = 0;
        halving_range(n, r, j + 1, lo, hi);

        if (hi > lo) {
            ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(
                list,
                byte_at(plan.out_ptrs[partner], lo),
                byte_at(plan.out_ptrs[r], lo),
                (hi - lo) * elem,
                ev(step, r),
                0, nullptr));
        } else {
            ZE_THROW(ze_api()->zeCommandListAppendSignalEvent(list, ev(step, r)));
        }

        // Waiting here also orders the next step's send, which reads the
        // range the partner has just filled in.
        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &plan.ev_ring[
            static_cast<std::size_t>(step) * static_cast<std::size_t>(N) +
            static_cast<std::size_t>(partner)]));
    }

    order(list);
    for (int step = 0; step < 2 * levels; ++step) {
        const int j = (step < levels) ? step : (2 * levels - 1 - step);
        ZE_THROW(ze_api()->zeCommandListAppendEventReset(list, ev(step, r ^ (1 << j))));
    }

    close_list(list, m_rec_close_ns);
}

void TPDeviceCoordinator::close_list(ze_command_list_handle_t list, std::atomic<uint64_t>& into) {
    if (!m_config.profiling_host()) {
        ZE_THROW(ze_api()->zeCommandListClose(list));
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    ZE_THROW(ze_api()->zeCommandListClose(list));
    into.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count()),
        std::memory_order_relaxed);
}

void TPDeviceCoordinator::submit_rank(Plan& plan, int rank) {
    auto& rs = m_ranks[rank];
    ZE_THROW(ze_api()->zeCommandQueueExecuteCommandLists(
        rs.compute_queue, 1, &plan.compute_lists[rank], nullptr));
}

void TPDeviceCoordinator::sync_rank(Plan& plan, int rank) {
    auto& rs = m_ranks[rank];
    // Name the rank: a collective that stalls does so on one specific link,
    // and "which queue never drained" is the first thing worth knowing when
    // it happens.
    const std::string what = "allreduce: queue rank " + std::to_string(rank) +
                             " (n=" + std::to_string(plan.n) +
                             (m_use_ring ? ", ring" : "") +
                             ")";
    sync_queue(rs.compute_queue, what.c_str());
}

void TPDeviceCoordinator::record_rank(Plan& plan, int rank) {
    // Reset this rank's command lists (must be done before re-recording).
    // zeCommandListReset on a list that still has work in-flight on its queue
    // is undefined; on shared multi-device L0 contexts it can deadlock.  Sync
    // the queue first so the previous submission has fully drained before we
    // wipe the recorded commands.
    //
    // Recording is only 0.4% of collective calls but all of its misses land in
    // time to first token, so the stages are timed separately: the drain, the
    // reset, and the appends that follow.
    using rec_clk = std::chrono::steady_clock;
    const bool measure = m_config.profiling_host();
    auto stamp = [measure]() -> rec_clk::time_point {
        return measure ? rec_clk::now() : rec_clk::time_point{};
    };
    auto& rs = m_ranks[rank];
    const auto t0 = stamp();
    if (rs.compute_queue) {
        sync_queue(rs.compute_queue, "re-record: compute queue drain");
    }
    const auto t1 = stamp();
    ZE_THROW(ze_api()->zeCommandListReset(plan.compute_lists[rank]));
    const auto t2 = stamp();

    if (m_use_ring) {
        // Halving needs the world to be a power of two; anything else stays
        // on the ring, which has no such requirement.  Large payloads stay on
        // the ring too -- see halving_max_bytes.
        const bool power_of_two = (m_world_size & (m_world_size - 1)) == 0;
        const std::size_t payload = plan.n * plan.dtype.size();
        if (m_config.get_enable_halving() && power_of_two && payload <= m_config.get_halving_max_bytes()) {
            plan.schedule = Plan::Schedule::halving;
            record_halving_rank(plan, rank);
        } else {
            plan.schedule = Plan::Schedule::ring;
            record_ring_rank(plan, rank);
        }
    } else {
        plan.schedule = Plan::Schedule::pair;
        record_pair_rank(plan, rank);
    }
    const auto t3 = stamp();

    if (!measure) {
        return;
    }
    switch (plan.schedule) {
    case Plan::Schedule::pair:    m_skew.n_pair[rank].bump();    break;
    case Plan::Schedule::ring:    m_skew.n_ring[rank].bump();    break;
    case Plan::Schedule::halving: m_skew.n_halving[rank].bump(); break;
    }

    m_rec_drain_ns.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()),
        std::memory_order_relaxed);
    m_rec_reset_ns.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()),
        std::memory_order_relaxed);
    m_rec_build_ns.fetch_add(
        static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count()),
        std::memory_order_relaxed);
}

void TPDeviceCoordinator::record_pair_rank(Plan& plan, int r) {
    constexpr uint32_t kGroupSize = 256;
    const std::size_t n = plan.n;
    const std::size_t bytes = collective_payload_bytes(n, plan.dtype);
    ze_group_count_t gc{launch_groups(n, kGroupSize), 1, 1};
    uint64_t cn64 = n;

    auto& self = m_ranks[r];
    const int peer = 1 - r;
    ze_command_list_handle_t self_compute = plan.compute_lists[r];

    ze_kernel_handle_t kernel =
        (plan.dtype == ov::element::f16) ? self.kernel_f16 : self.kernel_f32;
    ZE_THROW(ze_api()->zeKernelSetGroupSize(kernel, kGroupSize, 1, 1));

    // 1. Push our `in` to peer's local staging (source-side memcpy).
    //    ev_recv[r] tells the peer the bytes have landed.  The destination is
    //    the coordinator's staging, never the peer's own buffer, which is what
    //    lets each rank record on its own.
    ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(
        self_compute,
        scratch_buffer(peer, plan.buffer),
        plan.in_ptrs[r],
        bytes,
        plan.ev_recv[r],
        0, nullptr));

    // 2. Wait for peer's push to land in our staging.
    ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(self_compute,
                                                      1, &plan.ev_recv[peer]));

    // Rank r is the only consumer of ev_recv[peer], and once the wait above is
    // satisfied the event has done its job -- clearing it does not touch the
    // staged data the kernel is about to read.  Placing the reset here rather
    // than after the kernel is what makes it free: the wait already orders
    // everything appended after it, so no barrier is needed.
    ZE_THROW(ze_api()->zeCommandListAppendEventReset(self_compute, plan.ev_recv[peer]));

    // 3. Reduce: out_self = in_self + staging_self.
    void* dst   = plan.out_ptrs[r];
    void* src0  = plan.in_ptrs[r];
    void* src1  = scratch_buffer(r, plan.buffer);
    ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 0, sizeof(void*), &dst));
    ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 1, sizeof(void*), &src0));
    ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 2, sizeof(void*), &src1));
    ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 3, sizeof(cn64), &cn64));
    ZE_THROW(ze_api()->zeCommandListAppendLaunchKernel(self_compute,
                                                      kernel, &gc,
                                                      plan.ev_ts_kernel[r], 0, nullptr));

    close_list(self_compute, m_rec_close_ns);
}

// Duration between two kernel timestamps of one device, in nanoseconds.
// The counter is narrower than 64 bits on Intel GPUs, so it wraps.
static uint64_t ticks_to_ns(uint64_t start, uint64_t end, uint64_t mask, uint64_t ns_per_tick) {
    const uint64_t s = start & mask;
    const uint64_t e = end & mask;
    const uint64_t delta = (e >= s) ? (e - s) : ((mask + 1 - s) + e);
    return delta * ns_per_tick;
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

void TPDeviceCoordinator::note_collective_started(int rank) {
    std::call_once(m_watchdog_once, [this] { start_watchdog(); });
    m_started[rank].fetch_add(1, std::memory_order_release);
}

void TPDeviceCoordinator::start_watchdog() {
    // A zero timeout means the caller asked for no deadline at all, and there
    // is then nothing for a watchdog to enforce.
    if (m_collective_timeout.count() <= 0) {
        return;
    }
    m_watchdog = std::thread([this] { watchdog_loop(); });
}

void TPDeviceCoordinator::stop_watchdog() {
    if (!m_watchdog.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(m_watchdog_mutex);
        m_watchdog_stop = true;
    }
    m_watchdog_cv.notify_all();
    m_watchdog.join();
}

void TPDeviceCoordinator::watchdog_loop() {
    // Poll several times within the deadline: the check is a handful of atomic
    // loads, and a coarser tick would push the reported timeout well past what
    // was asked for.
    const auto tick = std::max(std::chrono::milliseconds(50), m_collective_timeout / 4);

    // What counts as progress, and why it is not simply "a collective
    // finished".  The host runs ahead of the devices on purpose now -- that is
    // the entire point of splicing -- so at any moment a rank may have dozens
    // of collectives handed over and none of them accounted for.  A 32k
    // prefill does exactly that: 64 collectives go into the queue in one pass
    // and the bookkeeping only catches up two tokens later, when a buffer is
    // reused.  Counting only that reported a hang after 52 splices with
    // nothing wrong.
    //
    // So ask the devices instead.  Every splice signals an event, and a group
    // that is merely behind keeps turning those events green.  A group that is
    // stuck stops, and the host stops handing over new work as well.
    uint64_t seen = 0;
    auto since = std::chrono::steady_clock::now();

    for (;;) {
        {
            std::unique_lock<std::mutex> lock(m_watchdog_mutex);
            m_watchdog_cv.wait_for(lock, tick, [this] { return m_watchdog_stop; });
            if (m_watchdog_stop) {
                return;
            }
        }
        if (m_aborted.load(std::memory_order_acquire)) {
            return;
        }

        uint64_t signature = 0;
        uint64_t outstanding = 0;
        for (int r = 0; r < m_world_size; ++r) {
            signature += m_started[r].load(std::memory_order_acquire);
        }
        for (const auto& plan : m_plans) {
            if (!plan || plan->ev_done.empty()) {
                continue;
            }
            for (int r = 0; r < m_world_size; ++r) {
                if (!plan->in_flight[r].load(std::memory_order_acquire)) {
                    continue;
                }
                ++outstanding;
                if (ze_api()->zeEventQueryStatus(plan->ev_done[r]) == ZE_RESULT_SUCCESS) {
                    ++signature;
                }
            }
        }

        const auto now = std::chrono::steady_clock::now();
        // Idle, or something moved since the last tick.  A signature that goes
        // down counts too: it means a rank consumed an event and spliced again.
        if (outstanding == 0 || signature != seen) {
            seen = signature;
            since = now;
            continue;
        }
        if (now - since < m_collective_timeout) {
            continue;
        }

        std::ostringstream oss;
        oss << "[TP][L0] no collective has completed on any device for " << m_collective_timeout.count()
            << " ms while " << outstanding
            << " are outstanding. A rank's queue is waiting on a peer that never signalled.";
        abort_all(oss.str());
        release_all_waits();
        return;
    }
}

void TPDeviceCoordinator::release_all_waits() {
    // Signalling from the host is a lie to the device -- whatever was waiting
    // proceeds on data that never arrived -- but the group is already dead and
    // the alternative is a process that never returns.
    const auto& api = ze_api();
    auto signal = [&](ze_event_handle_t ev) {
        if (ev != nullptr) {
            api->zeEventHostSignal(ev);
        }
    };
    for (const auto& plan : m_plans) {
        if (!plan) {
            continue;
        }
        for (auto ev : plan->ev_recv) {
            signal(ev);
        }
        for (auto ev : plan->ev_ring) {
            signal(ev);
        }
    }
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

void TPDeviceCoordinator::ensure_gather_events(Plan& plan) {
    if (!plan.ev_gather.empty()) {
        return;
    }
    // Plain signal/wait events: nothing here is timed, and the timestamp flag
    // costs a write in the command processor on every signal.
    ze_event_pool_desc_t epd{ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr};
    epd.flags = 0;
    epd.count = static_cast<uint32_t>(m_world_size);
    std::vector<ze_device_handle_t> devs(m_shared->devices.begin(), m_shared->devices.end());
    ZE_THROW(ov::zeEventPoolCreate(m_shared->context, &epd,
                                   static_cast<uint32_t>(devs.size()),
                                   devs.data(), &plan.gather_pool));

    plan.ev_gather.assign(static_cast<std::size_t>(m_world_size), nullptr);
    for (int r = 0; r < m_world_size; ++r) {
        ze_event_desc_t ed{ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr};
        ed.signal = ZE_EVENT_SCOPE_FLAG_DEVICE;
        ed.wait   = ZE_EVENT_SCOPE_FLAG_DEVICE;
        ed.index  = static_cast<uint32_t>(r);
        ZE_THROW(ov::zeEventCreate(plan.gather_pool, &ed, &plan.ev_gather[r]));
    }
}

void TPDeviceCoordinator::record_gather_rank(Plan& plan, int rank) {
    const std::size_t elem = plan.dtype.size();
    const std::size_t slice_bytes = checked_multiply(plan.slice_elems, elem, "gather slice");
    const std::size_t full_bytes =
        checked_multiply(slice_bytes, static_cast<std::size_t>(m_world_size), "gather row");

    // A rank's slice is contiguous in its own buffer but strided in the
    // root's: every row of the destination holds world_size slices side by
    // side, and this rank owns one column band of it.  One region copy
    // expresses that; the alternative is `rows` separate transfers, which at
    // prompt length is hundreds of driver calls for the same bytes.
    OPENVINO_ASSERT(full_bytes <= std::numeric_limits<uint32_t>::max(),
                    "[TP][L0] gather row of ", full_bytes,
                    " bytes exceeds the pitch a Level Zero region copy can express");
    OPENVINO_ASSERT(plan.rows <= std::numeric_limits<uint32_t>::max(),
                    "[TP][L0] gather of ", plan.rows, " rows exceeds the region copy height");

    ze_copy_region_t dst_region{};
    dst_region.originX = static_cast<uint32_t>(static_cast<std::size_t>(rank) * slice_bytes);
    dst_region.originY = 0;
    dst_region.originZ = 0;
    dst_region.width   = static_cast<uint32_t>(slice_bytes);
    dst_region.height  = static_cast<uint32_t>(plan.rows);
    dst_region.depth   = 1;

    ze_copy_region_t src_region{};
    src_region.originX = 0;
    src_region.originY = 0;
    src_region.originZ = 0;
    src_region.width   = static_cast<uint32_t>(slice_bytes);
    src_region.height  = static_cast<uint32_t>(plan.rows);
    src_region.depth   = 1;

    ze_command_list_handle_t list = plan.compute_lists[rank];
    // The root is the only waiter, so it is the only rank whose slice nobody
    // has to be told about.
    ze_event_handle_t signal = (rank == 0) ? nullptr : plan.ev_gather[rank];
    ZE_THROW(ze_api()->zeCommandListAppendMemoryCopyRegion(
        list,
        plan.out_ptrs[0],
        &dst_region,
        static_cast<uint32_t>(full_bytes),
        0,
        plan.in_ptrs[rank],
        &src_region,
        static_cast<uint32_t>(slice_bytes),
        0,
        signal,
        0,
        nullptr));

    if (rank == 0 && m_world_size > 1) {
        // Hold everything queued behind this recording until the other ranks
        // have written their columns.  On the spliced path "everything queued
        // behind" is the rest of the model on the root, which is exactly what
        // used to be held by draining the queue on the host.
        std::vector<ze_event_handle_t> waits;
        waits.reserve(static_cast<std::size_t>(m_world_size - 1));
        for (int r = 1; r < m_world_size; ++r) {
            waits.push_back(plan.ev_gather[r]);
        }
        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(
            list, static_cast<uint32_t>(waits.size()), waits.data()));
        // Each of these has exactly one waiter -- this rank, just above -- so
        // clearing them here cannot race with anyone, and the wait orders the
        // resets behind the copies that signaled them.  Unlike the ring's
        // resets this is not behind use_device_event_reset(): a host reset
        // would need a point where the root knows the events are consumed,
        // and on the spliced path there is no such point.
        for (auto* e : waits) {
            ZE_THROW(ze_api()->zeCommandListAppendEventReset(list, e));
        }
    }

    close_list(list, m_gather_close_ns);
}

void TPDeviceCoordinator::await_previous_splice(Plan& plan, int rank, int collective_id) {
    if (!plan.in_flight[rank].load(std::memory_order_acquire)) {
        return;
    }
    // Query before waiting: the status check is a memory read, while
    // zeEventHostSynchronize measured 7.2 us a call even with nothing to wait
    // for.  The blocking wait is kept for when the device really is behind.
    ze_result_t r = ze_api()->zeEventQueryStatus(plan.ev_done[rank]);
    if (r == ZE_RESULT_NOT_READY) {
        r = ze_api()->zeEventHostSynchronize(plan.ev_done[rank], timeout_ns());
        if (r == ZE_RESULT_NOT_READY) {
            fail_collective(true, collective_id, rank, "previous splice of this recording");
        }
    }
    ZE_THROW(r);
    ZE_THROW(ze_api()->zeEventHostReset(plan.ev_done[rank]));
    plan.in_flight[rank].store(0, std::memory_order_release);
}

void TPDeviceCoordinator::gather_to_root(int collective_id,
                                         int rank,
                                         void* in_dev,
                                         void* out_dev,
                                         std::size_t rows,
                                         std::size_t slice_elems,
                                         ov::element::Type dtype,
                                         ze_command_list_handle_t model_queue) {
    OPENVINO_ASSERT(m_ready, "[TP][L0] coordinator not initialized");
    OPENVINO_ASSERT(collective_id >= 0 && collective_id < m_num_collectives,
                    "[TP][L0] collective_id out of range: ", collective_id);
    OPENVINO_ASSERT(rank >= 0 && rank < m_world_size, "[TP][L0] rank out of range: ", rank);
    OPENVINO_ASSERT(in_dev != nullptr, "[TP][L0] gather called with a null source on rank ", rank);
    OPENVINO_ASSERT(rank != 0 || out_dev != nullptr,
                    "[TP][L0] gather called with a null destination on the root rank");
    OPENVINO_ASSERT(rows > 0 && slice_elems > 0,
                    "[TP][L0] gather of an empty slice (rows=", rows, ", slice=", slice_elems, ")");
    throw_if_aborted();

    using gclk = std::chrono::steady_clock;
    const bool g_host = m_config.profiling_host();
    const bool g_dev = m_config.profiling_device();
    auto g_stamp = [g_host]() -> gclk::time_point {
        return g_host ? gclk::now() : gclk::time_point{};
    };
    auto g_ns = [](gclk::time_point a, gclk::time_point b) -> uint64_t {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
    };
    const auto g_t0 = g_stamp();

    auto& rdz = *m_rendezvous[collective_id];
    // The gather touches no staging of its own -- every rank writes straight
    // into the root's output, at an offset nobody else uses -- so one set of
    // resources is enough and it always takes the first.
    Plan* const slot = &plan_at(collective_id, 0);

    // Enter barrier: publish this rank's slice and, on the root, the buffer
    // everyone writes into.
    {
        std::unique_lock<std::mutex> lk(rdz.mtx);
        rdz.in_ptrs[0][rank]  = in_dev;
        rdz.out_ptrs[0][rank] = out_dev;
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
                fail_collective(outcome == WaitOutcome::timed_out, collective_id, rank,
                                "gather enter barrier");
            }
        }
    }
    if (g_host) {
        m_gather.barrier_ns[rank].add(g_ns(g_t0, gclk::now()));
        m_gather.calls[rank].bump();
    }

    struct AbortOnFailure {
        TPDeviceCoordinator* self;
        int collective_id;
        int rank;
        bool armed{true};
        ~AbortOnFailure() {
            if (armed) {
                self->abort_all("[TP][L0] rank " + std::to_string(rank) +
                                " failed while gathering collective " +
                                std::to_string(collective_id));
            }
        }
    } abort_guard{this, collective_id, rank};

    // Rank 0 records for everyone, because a recording needs the root's
    // destination pointer, which only the barrier above has made visible.
    if (rank == 0) {
        OPENVINO_ASSERT(rdz.out_ptrs[0][0] != nullptr,
                        "[TP][L0] gather ", collective_id, " has no destination on the root");
        const bool recorded_matches =
            slot->matches_gather(rdz.in_ptrs[0], rdz.out_ptrs[0], rows, slice_elems, dtype) &&
            std::all_of(slot->recorded.begin(), slot->recorded.end(),
                        [](uint8_t v) { return v != 0; });
        if (!recorded_matches) {
            ensure_gather_events(*slot);
            slot->kind = Plan::Kind::gather;
            slot->in_ptrs = rdz.in_ptrs[0];
            slot->out_ptrs = rdz.out_ptrs[0];
            slot->rows = rows;
            slot->slice_elems = slice_elems;
            slot->n = checked_multiply(rows, slice_elems, "gather slice");
            slot->dtype = dtype;
            for (int r = 0; r < m_world_size; ++r) {
                auto& rs = m_ranks[r];
                // Resetting a list the device is still working through is
                // undefined, and where that work lives depends on how the
                // last recording was handed over: its own queue, or somebody
                // else's queue via a splice.
                if (model_queue != nullptr && run_spliced()) {
                    await_previous_splice(*slot, r, collective_id);
                } else if (rs.compute_queue) {
                    sync_queue(rs.compute_queue, "gather re-record: compute queue drain");
                }
                ZE_THROW(ze_api()->zeCommandListReset(slot->compute_lists[r]));
                record_gather_rank(*slot, r);
            }
            std::fill(slot->recorded.begin(), slot->recorded.end(), 1);
            std::fill(slot->recorded_scratch_generation.begin(),
                      slot->recorded_scratch_generation.end(), m_scratch.generation);
            if (g_host) {
                m_gather.records[rank].bump();
            }
        }
        std::unique_lock<std::mutex> lk(rdz.mtx);
        rdz.done = true;
        rdz.cv.notify_all();
    } else {
        std::unique_lock<std::mutex> lk(rdz.mtx);
        const auto outcome = wait_for_condition(lk, rdz.cv, m_collective_timeout, m_aborted,
                                                [&] { return rdz.done; });
        if (outcome != WaitOutcome::ready) {
            lk.unlock();
            fail_collective(outcome == WaitOutcome::timed_out, collective_id, rank,
                            "gather record phase");
        }
    }

    // Ranks write disjoint columns, so there is nothing to order between them
    // beyond the root's wait, which the recording carries.
    if (model_queue != nullptr && run_spliced()) {
        // The completion event of the previous instance still holds its
        // timestamps; read them before await_previous_splice clears it.  The
        // wait has to be blocking: with the host running a hundred splices
        // ahead the event is usually still pending, and a query-only check
        // would skip almost every sample.
        if (g_dev && slot->in_flight[rank].load(std::memory_order_acquire)) {
            if (ze_api()->zeEventHostSynchronize(slot->ev_done[rank], timeout_ns()) ==
                ZE_RESULT_SUCCESS) {
                ze_kernel_timestamp_result_t kt{};
                if (ze_api()->zeEventQueryKernelTimestamp(slot->ev_done[rank], &kt) ==
                    ZE_RESULT_SUCCESS) {
                    const uint64_t ns = ticks_to_ns(kt.global.kernelStart, kt.global.kernelEnd,
                                                    m_ranks[rank].timestamp_mask,
                                                    m_ranks[rank].timer_ns_per_tick);
                    m_gather.dev_ns[rank].add(ns);
                    m_gather.dev_max_ns[rank].keep_max(ns);
                    m_gather.dev_count[rank].bump();
                    m_gather.bytes[rank].add(
                        slot->spliced_bytes[static_cast<std::size_t>(rank)]);
                }
            }
        }
        await_previous_splice(*slot, rank, collective_id);
        // See allreduce(): the flag has to be up before the driver call so a
        // block inside it still looks like outstanding work to the watchdog.
        const auto g_a0 = g_stamp();
        slot->in_flight[rank].store(1, std::memory_order_release);
        ZE_THROW(ze_api()->zeCommandListImmediateAppendCommandListsExp(
            model_queue, 1, &slot->compute_lists[rank], slot->ev_done[rank], 0, nullptr));
        if (g_host) {
            m_gather.append_ns[rank].add(g_ns(g_a0, gclk::now()));
            m_gather.spliced[rank].bump();
        }
        if (g_dev) {
            // A rank copies its own slice into the root: rows * slice_elems.
            slot->spliced_bytes[static_cast<std::size_t>(rank)] =
                checked_multiply(checked_multiply(rows, slice_elems, "gather slice"),
                                 dtype.size(), "gather bytes");
        }
        note_collective_started(rank);
    } else {
        submit_rank(*slot, rank);
        sync_rank(*slot, rank);
    }
    abort_guard.armed = false;

    // Exit barrier: no rank may re-enter this slot while another is still
    // here, or it would overwrite the pointers the recording was built from.
    // Keeping the root from reading early is no longer this barrier's job on
    // the spliced path -- the recording's wait does that on the device -- but
    // it still is on the blocking path, where the drain above has already
    // happened by the time anyone gets here.
    {
        std::unique_lock<std::mutex> lk(rdz.mtx);
        const uint64_t my_gen = rdz.exit_gen;
        if (++rdz.departed == m_world_size) {
            rdz.departed = 0;
            rdz.done = false;
            std::fill(rdz.in_ptrs[0].begin(), rdz.in_ptrs[0].end(), nullptr);
            std::fill(rdz.out_ptrs[0].begin(), rdz.out_ptrs[0].end(), nullptr);
            rdz.exit_gen++;
            rdz.cv.notify_all();
        } else {
            const auto outcome = wait_for_condition(lk, rdz.cv, m_collective_timeout, m_aborted,
                                                    [&] { return rdz.exit_gen != my_gen; });
            if (outcome != WaitOutcome::ready) {
                lk.unlock();
                fail_collective(outcome == WaitOutcome::timed_out, collective_id, rank,
                                "gather exit barrier");
            }
        }
    }

    note_collective_done(rank);
}

bool TPDeviceCoordinator::async_supported() const {
    return ze_api()->zeCommandListImmediateAppendCommandListsExp != nullptr;
}

void TPDeviceCoordinator::allreduce(int collective_id,
                                    int rank,
                                    void* in_dev,
                                    void* out_dev,
                                    std::size_t n,
                                    ov::element::Type dtype,
                                    ze_command_list_handle_t model_queue) {
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
    if (m_config.skip_collective()) {
        static std::once_flag warned;
        std::call_once(warned, [] {
            TP_WARN_ALWAYS << "[TP_GPU] " << ov::tp_gpu::skip_collective.name()
                           << " is set: AllReduce is a no-op, outputs are invalid. Timing-only mode.";
        });
        return;
    }

    auto trace = [rank](const char* msg) {
        TP_LOG_TRACE << "[TP][L0] r" << rank << " " << msg << std::endl;
    };

    using clk = std::chrono::steady_clock;

    // The two halves are independent: HOST measures what the host spends,
    // DEVICE what the GPU spends, and neither prints the other's numbers.
    // ALL is how you ask for both.  Keeping them apart is what makes a HOST
    // run comparable with a plain one -- the kernel-timestamp query that
    // DEVICE adds lands in the middle of the host phases and shifts them.
    const bool measure_host = m_config.profiling_host();
    const bool measure_dev = m_config.profiling_device();

    // Every host clock read below goes through this.  Not "cheap when
    // profiling is off" but absent: with ENABLE_TP_GPU_DEBUG_CAPS off the
    // whole option chain is a literal, `measure_host` folds to false and the
    // compiler drops the reads, the accumulators and the reports.  With debug
    // caps on and TP_PROFILING unset it costs one predictable branch per site
    // and no clock_gettime, which at 65 collectives a token is what matters.
    auto stamp = [measure_host]() -> clk::time_point {
        return measure_host ? clk::now() : clk::time_point{};
    };

    // Not thread_local and no longer function statics: the ranks run on the
    // persistent worker threads of RankWorkers, so a thread-local total would
    // be split across whichever workers happened to serve rank 0, and a
    // function static would be shared by every coordinator in the process.
    // Outer inferences are serialized by CompiledModel::lock_inference() and
    // only rank 0 writes these.
    const auto t0 = stamp();

    auto& rdz = *m_rendezvous[collective_id];

    // Phase 1 (enter barrier): deposit pointers, wait until all ranks arrive.
    // Uses a generation counter so the wait predicate is monotonic and the
    // last-in resets `arrived` immediately for the next epoch.
    trace("phase1: enter");
    auto elapsed_ns = [](clk::time_point a, clk::time_point b) -> uint64_t {
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
    };
    uint64_t entry_gen = 0;
    int rdz_set = 0;
    uint64_t rec_gen_at_entry = 0;
    {
        std::unique_lock<std::mutex> lk(rdz.mtx);
        // Read the generation before publishing: it selects which of the two
        // pointer sets this instance owns, and every rank of one instance
        // reads the same value because only the last arrival bumps it.
        const uint64_t my_gen = rdz.enter_gen;
        entry_gen = my_gen;
        rdz_set = static_cast<int>(my_gen & 1ull);
        rec_gen_at_entry = rdz.record_gen;
        rdz.in_ptrs[rdz_set][rank]  = in_dev;
        rdz.out_ptrs[rdz_set][rank] = out_dev;
        if (rank == 0) {
            rdz.n     = n;
            rdz.dtype = dtype;
        }
        if (measure_host) {
            const auto t_arrive = clk::now();
            if (rdz.arrived == 0) {
                rdz.first_arrival = t_arrive;
            }
            m_skew.late_ns[rank].add(elapsed_ns(rdz.first_arrival, t_arrive));
            if (rdz.arrived + 1 == m_world_size) {
                m_skew.spread_ns += elapsed_ns(rdz.first_arrival, t_arrive);
                m_skew.last_count[rank].bump();
            }
            // How long this rank's own model work took since it left the
            // previous collective.  Only this rank touches these slots.
            if (m_skew.last_exit[rank] != std::chrono::steady_clock::time_point{}) {
                const uint64_t seg = elapsed_ns(m_skew.last_exit[rank], t_arrive);
                m_skew.seg_ns[rank].add(seg);
                m_skew.seg_min_ns[rank].keep_min(seg);
                m_skew.seg_max_ns[rank].keep_max(seg);
                m_skew.seg_count[rank].bump();
            }
        }
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
    const auto t1 = stamp();

    // Which set of resources this instance uses.  The generation is read
    // before the last rank bumps it, so every rank of one instance picks the
    // same set and consecutive instances pick different ones.
    const int buffer = static_cast<int>(entry_gen & 1ull) % plan_buffers();

    // Phase 2.  Recording needs the signature and the staging arena to be
    // settled, and only rank 0 can see every rank's pointers at once, so it
    // settles them and the record gate releases the others.
    //
    // Execution is a different matter.  On a per-rank schedule each rank owns
    // a queue of its own, and having a single thread submit four of them
    // serialized what the hardware can do at once while three threads slept --
    // submit alone measured 0.6 ms per token at four ranks.  Every rank drives
    // its own queue, which is why the gate means "recorded, go" rather than
    // "finished".
    //
    // Profiling does not change any of this.  It used to: device profiling
    // needed the events to survive until their timestamps were read, that was
    // expressed as a host reset, and the host reset was only reachable from a
    // single-threaded rank-0 schedule -- so asking for numbers replaced the
    // system being measured.  The reset is now deferred to
    // harvest_previous_instance() instead, which runs where the host already
    // knows the recording finished.

    // Whoever fails leaves the rest of the group waiting for something that
    // will never arrive, so every rank arms the guard, not just rank 0.
    struct AbortOnFailure {
        TPDeviceCoordinator* self;
        int collective_id;
        int rank;
        bool armed{true};
        ~AbortOnFailure() {
            if (armed) {
                self->abort_all("[TP][L0] rank " + std::to_string(rank) +
                                " failed while running collective " +
                                std::to_string(collective_id));
            }
        }
    } abort_guard{this, collective_id, rank};

    Plan* const slot = &plan_at(collective_id, buffer);
    auto tr0 = stamp();
    auto tr1 = tr0;

    if (rank == 0) {
        OPENVINO_ASSERT(rdz.n == n && rdz.dtype == dtype,
                        "[TP][L0] inconsistent (n,dtype) across ranks for collective ",
                        collective_id);
        for (int i = 0; i < m_world_size; ++i) {
            OPENVINO_ASSERT(rdz.in_ptrs[rdz_set][i] && rdz.out_ptrs[rdz_set][i],
                            "[TP][L0] collective ", collective_id, " has no buffers for rank ", i,
                            "; the barrier was released by ", m_world_size,
                            " arrivals that did not cover every rank");
        }

        const auto payload_bytes = collective_payload_bytes(n, dtype);
        const bool scratch_grew = ensure_scratch_capacity(payload_bytes);
        const bool signature_matches = slot->matches(rdz.in_ptrs[rdz_set], rdz.out_ptrs[rdz_set], n, dtype);
        // Every rank has to be recorded against the current signature and the
        // current staging arena.  They move together today, but they are kept
        // per rank because that is what lets a rank re-record on its own.
        const bool all_ranks_recorded =
            std::all_of(slot->recorded.begin(), slot->recorded.end(),
                        [](uint8_t v) { return v != 0; }) &&
            std::all_of(slot->recorded_scratch_generation.begin(),
                        slot->recorded_scratch_generation.end(),
                        [this](uint64_t g) { return g == m_scratch.generation; });
        const bool recorded_matches = all_ranks_recorded && signature_matches;
        const bool need_record = !recorded_matches;

        trace(scratch_grew ? "phase2: grow scratch and re-record"
                           : need_record ? "phase2: re-record" : "phase2: reuse recording");
        if (scratch_grew) {
            m_totals.rebuilds.bump();
        }

        const auto previous_max_payload = slot->max_payload_bytes;
        if (payload_bytes > previous_max_payload) {
            TP_LOG_INFO << "[TP][MEM] collective cid=" << collective_id
                        << " n=" << n
                        << " dtype=" << dtype
                        << " payload=" << payload_bytes
                        << " previous_max=" << previous_max_payload
                        << std::endl;
        }

        if (!signature_matches) {
            slot->in_ptrs = rdz.in_ptrs[rdz_set];
            slot->out_ptrs = rdz.out_ptrs[rdz_set];
            slot->n = n;
            slot->dtype = dtype;
        }
        slot->max_payload_bytes = std::max(previous_max_payload, payload_bytes);

        if (need_record) {
            // Only invalidate here.  A ring or pair recording reads none of
            // its neighbours' pointers -- just its own and the staging arena
            // -- so each rank can lay down its own commands, and doing it
            // here would serialize four recordings behind rank 0 for no
            // reason.  The arena and the signature are settled by the time
            // the others are released, which is what they need.
            std::fill(slot->recorded.begin(), slot->recorded.end(), 0);
            m_totals.records.bump();
        }
        tr1 = stamp();

        {
            std::unique_lock<std::mutex> lk(rdz.mtx);
            rdz.record_gen++;
            rdz.cv.notify_all();
        }
    } else {
        // The gate exists so that nobody records against a signature or an
        // arena rank 0 is still settling.  When this rank can see for itself
        // that neither moved -- the published pointers are the ones its own
        // recording was built from, and the arena generation still matches --
        // there is nothing to settle and nothing to wait for.  That is the
        // common case by a wide margin: 768 recordings across 130944 calls.
        const bool nothing_to_settle =
            slot->matches(rdz.in_ptrs[rdz_set], rdz.out_ptrs[rdz_set], n, dtype) &&
            slot->recorded[rank] != 0 &&
            slot->recorded_scratch_generation[rank] == m_scratch.generation &&
            !scratch_needs_growth(collective_payload_bytes(n, dtype));
        if (measure_host) {
            (nothing_to_settle ? m_skew.gate_fast[rank] : m_skew.gate_slow[rank]).bump();
        }
        if (!nothing_to_settle) {
            std::unique_lock<std::mutex> lk(rdz.mtx);
            const auto outcome = wait_for_condition(lk, rdz.cv, m_collective_timeout, m_aborted,
                                                    [&] { return rdz.record_gen != rec_gen_at_entry; });
            if (outcome != WaitOutcome::ready) {
                lk.unlock();
                fail_collective(outcome == WaitOutcome::timed_out, collective_id, rank, "record phase");
            }
        }
        tr1 = stamp();
    }

    auto te1 = tr1;
    {
        // Lay down this rank's commands if they are not already there.  The
        // check is per rank because the invalidation is: rank 0 cleared the
        // flags for everyone when the signature or the arena moved.
        if (!slot->recorded[rank] ||
            slot->recorded_scratch_generation[rank] != m_scratch.generation) {
            trace("phase2: record own commands");
            const auto rec0 = stamp();
            record_rank(*slot, rank);
            slot->recorded[rank] = 1;
            slot->recorded_scratch_generation[rank] = m_scratch.generation;
            if (measure_host) {
                m_skew.p2_rec_ns[rank].add(elapsed_ns(rec0, clk::now()));
                m_skew.p2_rec_count[rank].bump();
            }
        }

        if (model_queue != nullptr && run_spliced()) {
            // Splice the recording into the queue the model runs on and
            // leave.  Nothing here waits for the device.
            //
            // The one thing the host must wait for is the previous splice of
            // this very list: the extension forbids re-appending a list that
            // has not finished.  Two buffers alternate, so this is a wait for
            // work handed over two collectives ago and is normally already
            // satisfied -- but "normally" is not a guarantee, and a stalled
            // peer would otherwise turn into memory corruption instead of an
            // error.
            if (slot->in_flight[rank].load(std::memory_order_acquire)) {
                const auto w0 = stamp();
                // Query before waiting.  Two buffers alternate, so this asks
                // about work handed over two collectives ago, which the device
                // has long finished: the status query is a memory read and
                // returns ready essentially every time, while
                // zeEventHostSynchronize measured 7.2 us a call even when it
                // had nothing to wait for.  The blocking wait is kept for the
                // case the device really is behind.
                ze_result_t r = ze_api()->zeEventQueryStatus(slot->ev_done[rank]);
                if (r == ZE_RESULT_NOT_READY) {
                    if (measure_host) {
                        m_skew.p2_block_count[rank].bump();
                    }
                    r = ze_api()->zeEventHostSynchronize(slot->ev_done[rank], timeout_ns());
                    if (r == ZE_RESULT_NOT_READY) {
                        fail_collective(true, collective_id, rank, "previous splice of this recording");
                    }
                }
                ZE_THROW(r);
                const auto w1 = stamp();
                if (measure_dev) {
                    // The event still holds the timestamps of the splice that
                    // just finished; the reset below clears them.  Gated on
                    // device profiling rather than on the dump period because
                    // that is what put ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP on
                    // the pool: asking an event without it for a timestamp
                    // measured 77 ms a call on the validated driver.
                    ze_kernel_timestamp_result_t kt{};
                    if (ze_api()->zeEventQueryKernelTimestamp(slot->ev_done[rank], &kt) ==
                        ZE_RESULT_SUCCESS) {
                        const uint64_t mask = m_ranks[rank].timestamp_mask;
                        const uint64_t tick = m_ranks[rank].timer_ns_per_tick;
                        const uint64_t ns = ticks_to_ns(kt.global.kernelStart, kt.global.kernelEnd,
                                                        mask, tick);
                        m_skew.p2_dev_ns[rank].add(ns);
                        if (ns > m_skew.p2_dev_max_ns[rank].get()) {
                            m_skew.p2_dev_max_ns[rank].keep_max(ns);
                            m_skew.p2_dev_max_cid[rank].v.store(
                                static_cast<uint64_t>(collective_id), std::memory_order_relaxed);
                        }
                        m_skew.p2_dev_count[rank].bump();

                        // Powers of two in microseconds.  Anything under 1us
                        // and anything over the top bucket is clamped into an
                        // end bucket rather than dropped.
                        const uint64_t us = ns / 1000u;
                        int bucket = 0;
                        for (uint64_t v = us; v > 1 && bucket < kDevBuckets - 1; v >>= 1) {
                            ++bucket;
                        }
                        m_skew.p2_dev_hist[static_cast<std::size_t>(rank) *
                                               static_cast<std::size_t>(kDevBuckets) +
                                           static_cast<std::size_t>(bucket)]
                            .bump();

                        // Device time between the end of the previous
                        // collective and the start of this one -- the model's
                        // own work.  The rule is the tick order itself rather
                        // than collective ids: a window that starts before the
                        // previous one ended means the two readings did not
                        // come back in device order, and there is no gap to
                        // speak of.  That self-check also covers the counter
                        // wrapping, which is indistinguishable from it here.
                        const uint64_t prev_end = m_skew.dev_last_end_ticks[rank] & mask;
                        const uint64_t cur_start = kt.global.kernelStart & mask;
                        if (m_skew.dev_last_cid[rank] >= 0 && cur_start >= prev_end) {
                            m_skew.dev_gap_ns[rank].add((cur_start - prev_end) * tick);
                            m_skew.dev_gap_count[rank].bump();
                        }
                        m_skew.dev_last_end_ticks[rank] = kt.global.kernelEnd;
                        m_skew.dev_last_cid[rank] = collective_id;

                        // The bytes of the very splice this timestamp belongs
                        // to, recorded when it was handed over.
                        m_skew.dev_bytes[rank].add(
                            slot->spliced_bytes[static_cast<std::size_t>(rank)]);
                    }
                }
                ZE_THROW(ze_api()->zeEventHostReset(slot->ev_done[rank]));
                slot->in_flight[rank].store(0, std::memory_order_release);
                if (measure_host) {
                    m_ahead.now[rank].v.fetch_sub(1, std::memory_order_relaxed);
                    const auto w2 = clk::now();
                    m_skew.p2_wait_ns[rank].add(elapsed_ns(w0, w1));
                    m_skew.p2_reset_ns[rank].add(elapsed_ns(w1, w2));
                    m_skew.p2_wait_count[rank].bump();
                }
            }

            trace("phase2: splice into the model queue");
            const auto a0 = stamp();
            slot->in_flight[rank].store(1, std::memory_order_release);
            ZE_THROW(ze_api()->zeCommandListImmediateAppendCommandListsExp(
                model_queue, 1, &slot->compute_lists[rank], slot->ev_done[rank], 0, nullptr));
            if (measure_host) {
                m_skew.p2_append_ns[rank].add(elapsed_ns(a0, clk::now()));
                // How many of this rank's splices the devices have not caught
                // up with.  Sampled here rather than scanned at dump time,
                // which would cost a pass over every plan.
                const uint64_t depth = m_ahead.now[rank].v.fetch_add(
                                           1, std::memory_order_relaxed) + 1;
                m_ahead.sum[rank].add(depth);
                m_ahead.max[rank].keep_max(depth);
                m_ahead.count[rank].bump();
            }
            if (measure_dev) {
                // What this rank is about to push across a device boundary.
                // The schedule fixes it exactly: the pair exchange sends one
                // payload, and both the ring and recursive halving send
                // 2*(N-1)/N of it -- the bandwidth optimum, which is why the
                // two agree.  Counting rather than measuring keeps this free
                // and exact.  Parked on the plan, not added to the total:
                // it is only counted once the duration of this same splice
                // comes back, or the two would advance at different rates.
                const auto payload = collective_payload_bytes(n, dtype);
                slot->spliced_bytes[static_cast<std::size_t>(rank)] =
                    m_use_ring ? 2u * static_cast<std::size_t>(m_world_size - 1) * payload /
                                     static_cast<std::size_t>(m_world_size)
                               : payload;
            }
            note_collective_started(rank);
            te1 = stamp();
        } else {
            // Submit before syncing, on every rank: a rank's list blocks on
            // events its neighbours only signal once they run, so draining one
            // queue before the others are submitted would deadlock.  Nothing
            // here waits on the host, so all four submissions happen at once.
            trace("phase2: submit own queue");
            submit_rank(*slot, rank);
            sync_rank(*slot, rank);
            te1 = stamp();
            trace("phase2: own queue drained");
        }
    }
    abort_guard.armed = false;

    if (measure_host && rank == 0) {
        m_totals.record_ns.add(elapsed_ns(tr0, tr1));
        m_totals.splice_ns.add(elapsed_ns(tr1, te1));
        m_totals.prep_ns.add(elapsed_ns(t1, tr0));
        m_totals.tail_ns.add(elapsed_ns(te1, clk::now()));
    }
    trace("phase2: passed");
    const auto t2 = stamp();

    // There used to be an exit barrier here, and a phase-3 counter to measure
    // it.  It guarded one thing: a rank that had already left could reach this
    // collective again and overwrite the published pointers a slower peer was
    // still reading.  The two pointer sets, picked by the parity of the entry
    // generation, guard that directly -- reaching the same set again means
    // passing the enter barrier twice, which the slow peer has to take part
    // in.  Everything else the barrier appeared to protect is already
    // device-side: a rank does not re-splice a recording before its previous
    // splice signalled completion, and the order between ranks is held by the
    // ring's own events.
    //
    // Only rank 0's phases are accumulated, to match record/exec/prep/tail and
    // the call counter below.  Adding every rank here would double the phase
    // totals while the inner breakdown stayed single-rank.
    if (measure_host && rank == 0) {
        m_totals.ph1_ns.add(elapsed_ns(t0, t1));
        m_totals.ph2_ns.add(elapsed_ns(t1, t2));
    }
    if (measure_host) {
        m_skew.ph1_ns[rank].add(elapsed_ns(t0, t1));
        m_skew.ph2_ns[rank].add(elapsed_ns(t1, t2));
        m_skew.p2_gate_ns[rank].add(elapsed_ns(tr0, tr1));
        m_skew.last_exit[rank] = t2;
    }
    note_collective_done(rank);
}

void TPDeviceCoordinator::note_collective_done(int rank) {
    const bool measure_host = m_config.profiling_host();
    const bool measure_dev = m_config.profiling_device();
    if (!(measure_host || measure_dev) || rank != 0) {
        return;
    }
    // One counter for every kind of collective, advanced in one place.
    ++m_skew.calls;
    const auto period = static_cast<uint64_t>(m_config.dump_period());
    if (m_skew.calls - m_skew.last_dump < period) {
        return;
    }
    m_skew.last_dump = m_skew.calls;
    emit_report();
}

void TPDeviceCoordinator::emit_report() {
    const bool measure_host = m_config.profiling_host();
    const bool measure_dev = m_config.profiling_device();
    {
        const double c = static_cast<double>(m_skew.calls);
        auto us = [c](uint64_t v) { return static_cast<double>(v) / 1.0e3 / c; };

        if (measure_host) {
            TP_REPORT << "[TP][RANK|host] calls=" << m_skew.calls
                      << " arrival spread=" << us(m_skew.spread_ns) << "us/call"
                      << std::endl;
            for (int r = 0; r < m_world_size; ++r) {
                const double last_share =
                    100.0 * static_cast<double>(m_skew.last_count[r].get()) / c;
                const double sc =
                    std::max<double>(1.0, static_cast<double>(m_skew.seg_count[r].get()));
                TP_REPORT << "[TP][RANK|host]   rank " << r
                          << ": late=" << us(m_skew.late_ns[r].get()) << "us"
                          << " arrived_last=" << last_share << "%"
                          << "  ph1=" << us(m_skew.ph1_ns[r].get()) << "us"
                          << " ph2=" << us(m_skew.ph2_ns[r].get()) << "us"
                          << "  segment mean="
                          << (static_cast<double>(m_skew.seg_ns[r].get()) / 1.0e3 / sc) << "us"
                          << " min=" << (static_cast<double>(m_skew.seg_min_ns[r].get()) / 1.0e3)
                          << "us max=" << (static_cast<double>(m_skew.seg_max_ns[r].get()) / 1.0e3)
                          << "us" << std::endl;
                const uint64_t p2_known =
                    m_skew.p2_gate_ns[r].get() + m_skew.p2_rec_ns[r].get() +
                    m_skew.p2_wait_ns[r].get() + m_skew.p2_reset_ns[r].get() +
                    m_skew.p2_append_ns[r].get();
                TP_REPORT << "[TP][RANK|host]     ph2 split: gate=" << us(m_skew.p2_gate_ns[r].get()) << "us"
                          << " record=" << us(m_skew.p2_rec_ns[r].get()) << "us"
                          << "(" << m_skew.p2_rec_count[r].get() << "x)"
                          << " evt_wait=" << us(m_skew.p2_wait_ns[r].get()) << "us"
                          << " evt_reset=" << us(m_skew.p2_reset_ns[r].get()) << "us"
                          << "(" << m_skew.p2_wait_count[r].get() << "x, blocked "
                          << m_skew.p2_block_count[r].get() << "x)"
                          << " append=" << us(m_skew.p2_append_ns[r].get()) << "us"
                          << " rest="
                          << us(m_skew.ph2_ns[r].get() - std::min(p2_known, m_skew.ph2_ns[r].get()))
                          << "us" << std::endl;
                const uint64_t ahead_n = m_ahead.count[r].get();
                const uint64_t gate_all = m_skew.gate_fast[r].get() + m_skew.gate_slow[r].get();
                if (ahead_n > 0 || gate_all > 0) {
                    TP_REPORT << "[TP][RANK|host]     run-ahead: mean="
                              << (ahead_n > 0 ? static_cast<double>(m_ahead.sum[r].get()) /
                                                    static_cast<double>(ahead_n)
                                              : 0.0)
                              << " splices max=" << m_ahead.max[r].get()
                              << "  record gate: fast="
                              << (gate_all > 0 ? 100.0 * static_cast<double>(m_skew.gate_fast[r].get()) /
                                                     static_cast<double>(gate_all)
                                               : 0.0)
                              << "% of " << gate_all << std::endl;
                }
            }
            auto tms = [](uint64_t ns) { return static_cast<double>(ns) / 1.0e6; };
            TP_REPORT << "[TP][TOTAL|host] r0 calls=" << m_skew.calls
                      << " rebuilds=" << m_totals.rebuilds.get()
                      << " records=" << m_totals.records.get()
                      << "  totals: ph1=" << tms(m_totals.ph1_ns.get()) << "ms"
                      << " ph2=" << tms(m_totals.ph2_ns.get()) << "ms"
                      << " (record=" << tms(m_totals.record_ns.get()) << "ms"
                      << ", splice=" << tms(m_totals.splice_ns.get()) << "ms)"
                      << "  per-call: ph1=" << tms(m_totals.ph1_ns.get()) / c << "ms"
                      << " prep=" << tms(m_totals.prep_ns.get()) / c << "ms"
                      << " splice=" << tms(m_totals.splice_ns.get()) / c << "ms"
                      << " tail=" << tms(m_totals.tail_ns.get()) / c << "ms"
                      << std::endl;
            // Which schedule the recordings chose.  halving is default-on and
            // bounded by payload, so the split is what says whether that bound
            // is anywhere near right.
            uint64_t pair = 0, ring = 0, halving = 0;
            for (int r = 0; r < m_world_size; ++r) {
                pair += m_skew.n_pair[r].get();
                ring += m_skew.n_ring[r].get();
                halving += m_skew.n_halving[r].get();
            }
            TP_REPORT << "[TP][TOTAL|host]   schedule of " << (pair + ring + halving)
                      << " recordings: pair=" << pair
                      << " ring=" << ring
                      << " halving=" << halving
                      << std::endl;
            // Where re-recording time goes.  Divided by the number of
            // recordings, not by calls: recording is rare but every miss
            // lands in time to first token.
            const double rc = std::max<double>(1.0, static_cast<double>(m_totals.records.get()));
            auto rec_ms = [](const std::atomic<uint64_t>& v) {
                return static_cast<double>(v.load(std::memory_order_relaxed)) / 1.0e6;
            };
            TP_REPORT << "[TP][TOTAL|host]   record breakdown: records=" << m_totals.records.get()
                      << " per-record=" << tms(m_totals.record_ns.get()) / rc << "ms"
                      << " (drain=" << rec_ms(m_rec_drain_ns) / rc << "ms"
                      << " reset=" << rec_ms(m_rec_reset_ns) / rc << "ms"
                      << " append="
                      << (rec_ms(m_rec_build_ns) - rec_ms(m_rec_close_ns)) / rc << "ms"
                      << " close=" << rec_ms(m_rec_close_ns) / rc << "ms)"
                      << std::endl;
            const auto scratch = get_scratch_stats();
            TP_REPORT << "[TP][TOTAL|host]   scratch: payload_capacity="
                      << (scratch.payload_capacity_bytes / (1024.0 * 1024.0)) << "MB"
                      << " total=" << (scratch.total_allocated_bytes / (1024.0 * 1024.0)) << "MB"
                      << " generation=" << scratch.generation
                      << " grows=" << scratch.growth_count
                      << " allocations=" << scratch.allocation_count
                      << " stall="
                      << (static_cast<double>(m_scratch_stall_ns.load(std::memory_order_relaxed)) /
                          1.0e6)
                      << "ms"
                      << std::endl;
            // The gather, kept out of the averages above: one collective, but
            // sized by the vocabulary rather than the hidden dimension.
            for (int r = 0; r < m_world_size; ++r) {
                const uint64_t gc_calls = m_gather.calls[r].get();
                if (gc_calls == 0) {
                    continue;
                }
                const auto gus = [gc_calls](uint64_t v) {
                    return static_cast<double>(v) / 1.0e3 / static_cast<double>(gc_calls);
                };
                TP_REPORT << "[TP][TOTAL|host]   gather rank " << r
                          << ": calls=" << gc_calls
                          << " spliced=" << m_gather.spliced[r].get()
                          << " records=" << m_gather.records[r].get()
                          << " barrier=" << gus(m_gather.barrier_ns[r].get()) << "us"
                          << " append=" << gus(m_gather.append_ns[r].get()) << "us"
                          << " close="
                          << (static_cast<double>(m_gather_close_ns.load(std::memory_order_relaxed)) /
                              1.0e6)
                          << "ms total"
                          << std::endl;
            }
        }

        if (measure_dev) {
            uint64_t total_samples = 0;
            for (int r = 0; r < m_world_size; ++r) {
                total_samples += m_skew.p2_dev_count[r].get();
            }
            if (total_samples == 0) {
                // A completion event is only read when the same recording is
                // spliced again, and the two plan buffers alternate, so the
                // first device numbers appear on the third instance of a
                // collective.  Saying so beats an empty heading.
                TP_REPORT << "[TP][RANK|dev] calls=" << m_skew.calls
                          << ": no samples yet -- device timings start after a"
                             " collective has run three times" << std::endl;
            } else {
            TP_REPORT << "[TP][RANK|dev] calls=" << m_skew.calls
                      << " schedule=" << (m_use_ring ? "ring/halving" : "pair")
                      << " world=" << m_world_size << std::endl;
            uint64_t all_bytes = 0;
            uint64_t all_ns = 0;
            for (int r = 0; r < m_world_size; ++r) {
                const uint64_t samples = m_skew.p2_dev_count[r].get();
                if (samples == 0) {
                    continue;
                }
                const uint64_t ns = m_skew.p2_dev_ns[r].get();
                const uint64_t bytes = m_skew.dev_bytes[r].get();
                all_bytes += bytes;
                all_ns += ns;
                const double busy_us = static_cast<double>(ns) / 1.0e3 /
                                       static_cast<double>(samples);
                const double mb = static_cast<double>(bytes) / (1024.0 * 1024.0) /
                                  static_cast<double>(samples);
                // Bytes over the time the recording owned the queue.  Not the
                // link rate: that window also contains the reduce kernel and
                // every wait on a peer, and separating those needs per-step
                // timestamps the production path cannot hand back.  It is the
                // rate the model actually sees, which is the one that decides
                // whether the collective is worth optimizing.
                const double gbs = ns > 0 ? (static_cast<double>(bytes) / 1.0e9) /
                                                (static_cast<double>(ns) / 1.0e9)
                                          : 0.0;
                TP_REPORT << "[TP][RANK|dev]   rank " << r
                          << ": queue busy mean=" << busy_us << "us"
                          << " max=" << (static_cast<double>(m_skew.p2_dev_max_ns[r].get()) / 1.0e3)
                          << "us(cid " << m_skew.p2_dev_max_cid[r].get() << ")"
                          << "  sent=" << mb << "MB/call"
                          << " effective=" << gbs << " GB/s"
                          << " (" << samples << " samples)" << std::endl;

                // Percentiles off the histogram.  The mean above mixes prefill
                // and decode, which differ by two orders of magnitude; this is
                // where they separate.  Each figure is the upper bound of the
                // bucket the percentile falls in, so it reads "at most".
                auto pct_us = [&](double frac) -> uint64_t {
                    const auto want = static_cast<uint64_t>(
                        static_cast<double>(samples) * frac);
                    uint64_t seen = 0;
                    for (int b = 0; b < kDevBuckets; ++b) {
                        seen += m_skew.p2_dev_hist[static_cast<std::size_t>(r) *
                                                       static_cast<std::size_t>(kDevBuckets) +
                                                   static_cast<std::size_t>(b)]
                                    .get();
                        if (seen >= want) {
                            return uint64_t{1} << (b + 1);
                        }
                    }
                    return uint64_t{1} << kDevBuckets;
                };
                TP_REPORT << "[TP][RANK|dev]     queue busy under: p50=" << pct_us(0.50) << "us"
                          << " p90=" << pct_us(0.90) << "us"
                          << " p99=" << pct_us(0.99) << "us" << std::endl;

                // What the GPU was doing between two collectives: the model.
                const uint64_t gaps = m_skew.dev_gap_count[r].get();
                if (gaps > 0) {
                    const double gap_us = static_cast<double>(m_skew.dev_gap_ns[r].get()) / 1.0e3 /
                                          static_cast<double>(gaps);
                    const double duty = 100.0 * busy_us / (busy_us + gap_us);
                    TP_REPORT << "[TP][RANK|dev]     between collectives: model=" << gap_us << "us"
                              << " -> collectives own " << duty << "% of device time"
                              << " (" << gaps << " gaps)" << std::endl;
                }
            }
            if (all_ns > 0) {
                TP_REPORT << "[TP][TOTAL|dev] across " << m_world_size << " ranks: sent="
                          << (static_cast<double>(all_bytes) / (1024.0 * 1024.0) / c) << "MB/call"
                          << " aggregate="
                          << ((static_cast<double>(all_bytes) / 1.0e9) /
                              (static_cast<double>(all_ns) / 1.0e9 / m_world_size))
                          << " GB/s"
                          << "  (time is queue occupancy, kernel and peer waits included)"
                          << std::endl;
            }
            for (int r = 0; r < m_world_size; ++r) {
                const uint64_t gd = m_gather.dev_count[r].get();
                if (gd == 0) {
                    continue;
                }
                const double busy_us = static_cast<double>(m_gather.dev_ns[r].get()) / 1.0e3 /
                                       static_cast<double>(gd);
                const double mb = static_cast<double>(m_gather.bytes[r].get()) /
                                  (1024.0 * 1024.0) / static_cast<double>(gd);
                TP_REPORT << "[TP][RANK|dev]   gather rank " << r
                          << ": queue busy mean=" << busy_us << "us"
                          << " max=" << (static_cast<double>(m_gather.dev_max_ns[r].get()) / 1.0e3)
                          << "us  sent=" << mb << "MB/call"
                          << " (" << gd << " samples)" << std::endl;
            }
            }
        }
    }
}

}  // namespace tp_gpu
}  // namespace ov

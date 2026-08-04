// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Standalone test for ov::tp::TPDeviceCoordinator.
//
// Goal: validate the L0 device-side AllReduce in isolation — without the
// graph rewriter, intel_gpu plugin, or the TENSOR_PARALLEL plugin's
// rendezvous of compile_model().  Catches problems with:
//   * cross-device USM visibility in a shared multi-device L0 context
//   * native-binary load into a different L0 context on the same device
//   * event signal/wait across devices
//   * rendezvous protocol reset between iterations
//   * plan rebuild on (ptr, n, dtype) change
//   * teardown order / leaks
//
// Usage: ./tp_coordinator_test  [N]  [iters]
// Defaults: N = 64*1024 elements, iters = 5.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/core/except.hpp"
#define ZERO_API_KEEP_SYMBOLS_LIST_MACRO
#include "openvino/zero_api.hpp"

#include "tensor_parallel/tp_device_coordinator.hpp"
#include "tp_l0_shared_context.hpp"

using ov::tp::TPDeviceCoordinator;
using ov::tp::TPDeviceCoordinatorPtr;
using ov::tp::TPL0SharedContext;
using ov::tp::TPL0SharedContextPtr;

namespace {

#define ZE(expr) do { \
    ze_result_t _r = (expr); \
    if (_r != ZE_RESULT_SUCCESS) { \
        std::cerr << "[FAIL] L0 call '" << #expr << "' failed: 0x" << std::hex << _r << std::dec << std::endl; \
        std::exit(1); \
    } \
} while (false)

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "[FAIL] " << msg << "  (cond: " << #cond << ")" << std::endl; \
        std::exit(1); \
    } \
} while (false)

// ---------------------------------------------------------------------------
// Pick the first 2 GPUs on the first L0 driver and build a shared context.
// ---------------------------------------------------------------------------
// Returns nullptr when no driver has >= min_gpus discrete GPUs
// (caller should treat as "skip"). zeInit is invoked unconditionally.
TPL0SharedContextPtr make_shared_ctx(int min_gpus) {
    ZE(ov::zeInit(0));

    uint32_t dcount = 0;
    ZE(ov::zeDriverGet(&dcount, nullptr));
    if (dcount == 0) return nullptr;
    std::vector<ze_driver_handle_t> drivers(dcount);
    ZE(ov::zeDriverGet(&dcount, drivers.data()));

    for (auto drv : drivers) {
        uint32_t cnt = 0;
        ov::zeDeviceGet(drv, &cnt, nullptr);
        std::vector<ze_device_handle_t> all(cnt);
        ov::zeDeviceGet(drv, &cnt, all.data());

        std::vector<ze_device_handle_t> gpus;
        for (auto dh : all) {
            ze_device_properties_t p{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES, nullptr};
            ov::zeDeviceGetProperties(dh, &p);
            // Prefer discrete GPUs; an integrated GPU shares system memory and
            // device USM allocations behave differently (P2P paths the
            // coordinator exercises do not apply meaningfully).
            const bool is_dgpu =
                (p.type == ZE_DEVICE_TYPE_GPU) &&
                (p.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) == 0;
            if (is_dgpu) gpus.push_back(dh);
        }
        if (static_cast<int>(gpus.size()) >= min_gpus) {
            if (ov::zeContextCreateEx == nullptr) return nullptr;
            ze_context_desc_t cd{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
            ze_context_handle_t ctx = nullptr;
            ZE(ov::zeContextCreateEx(drv, &cd,
                                     static_cast<uint32_t>(min_gpus),
                                     gpus.data(), &ctx));

            auto shared = std::make_shared<TPL0SharedContext>();
            shared->driver = drv;
            shared->devices.assign(gpus.begin(), gpus.begin() + min_gpus);
            shared->context = ctx;
            return shared;
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Per-rank scratch resources used to allocate and prime device USM buffers.
// (The coordinator owns its own queues; these helpers are independent.)
// ---------------------------------------------------------------------------
struct RankScratch {
    ze_context_handle_t       ctx{nullptr};
    ze_device_handle_t        dev{nullptr};
    uint32_t                  ord{0};
    ze_command_queue_handle_t queue{nullptr};
    ze_command_list_handle_t  list{nullptr};

    static uint32_t pick_compute_ordinal(ze_device_handle_t d) {
        uint32_t qg = 0;
        ov::zeDeviceGetCommandQueueGroupProperties(d, &qg, nullptr);
        std::vector<ze_command_queue_group_properties_t> qgp(
            qg, {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr});
        ov::zeDeviceGetCommandQueueGroupProperties(d, &qg, qgp.data());
        for (uint32_t g = 0; g < qg; ++g) {
            if (qgp[g].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) return g;
        }
        return 0;
    }

    void init(ze_context_handle_t c, ze_device_handle_t d) {
        ctx = c; dev = d;
        ord = pick_compute_ordinal(d);
        ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr};
        qd.ordinal = ord;
        qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        ZE(ov::zeCommandQueueCreate(ctx, dev, &qd, &queue));
        ze_command_list_desc_t ld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr};
        ld.commandQueueGroupOrdinal = ord;
        ZE(ov::zeCommandListCreate(ctx, dev, &ld, &list));
    }

    void* alloc(size_t bytes) {
        ze_device_mem_alloc_desc_t mad{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr};
        void* p = nullptr;
        ZE(ov::zeMemAllocDevice(ctx, &mad, bytes, 64, dev, &p));
        return p;
    }

    void copy_from_host(void* dev_ptr, const void* host, size_t bytes) {
        ZE(ov::zeCommandListReset(list));
        ZE(ov::zeCommandListAppendMemoryCopy(list, dev_ptr, host, bytes,
                                             nullptr, 0, nullptr));
        ZE(ov::zeCommandListClose(list));
        ZE(ov::zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
        ZE(ov::zeCommandQueueSynchronize(queue, UINT64_MAX));
    }

    void copy_to_host(void* host, const void* dev_ptr, size_t bytes) {
        ZE(ov::zeCommandListReset(list));
        ZE(ov::zeCommandListAppendMemoryCopy(list, host, dev_ptr, bytes,
                                             nullptr, 0, nullptr));
        ZE(ov::zeCommandListClose(list));
        ZE(ov::zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
        ZE(ov::zeCommandQueueSynchronize(queue, UINT64_MAX));
    }

    void destroy() {
        if (list) ov::zeCommandListDestroy(list);
        if (queue) ov::zeCommandQueueDestroy(queue);
        list = nullptr; queue = nullptr;
    }
};

// ---------------------------------------------------------------------------
// Watchdog: if a test does not finish within `seconds`, abort the process so
// the user gets a clear signal which test hung instead of a silent freeze.
// ---------------------------------------------------------------------------
struct Watchdog {
    std::atomic<bool> done{false};
    std::thread t;
    explicit Watchdog(int seconds, std::string label) {
        t = std::thread([this, seconds, label = std::move(label)] {
            for (int i = 0; i < seconds * 10; ++i) {
                if (done.load()) return;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            std::cerr << "\n[FAIL] watchdog: '" << label
                      << "' did not finish in " << seconds << " s — hung."
                      << std::endl;
            std::abort();
        });
    }
    ~Watchdog() {
        done.store(true);
        if (t.joinable()) t.join();
    }
};

// ---------------------------------------------------------------------------
// Drive the coordinator from N concurrent threads.
// ---------------------------------------------------------------------------
void run_collective(TPDeviceCoordinator& coord,
                    int collective_id,
                    const std::vector<void*>& ins,
                    const std::vector<void*>& outs,
                    size_t n, ov::element::Type dtype) {
    const int N = static_cast<int>(ins.size());
    std::vector<std::thread> ths;
    ths.reserve(N);
    std::atomic<int> failed{0};
    for (int r = 0; r < N; ++r) {
        ths.emplace_back([&, r] {
            try {
                coord.allreduce(collective_id, r, ins[r], outs[r], n, dtype);
            } catch (const std::exception& e) {
                std::cerr << "[FAIL] rank " << r << " threw: " << e.what() << std::endl;
                failed.fetch_add(1);
            }
        });
    }
    for (auto& t : ths) t.join();
    CHECK(failed.load() == 0, "one or more ranks failed");
}

// ---------------------------------------------------------------------------
// Test 1: init / teardown stability
// ---------------------------------------------------------------------------
void test_init_teardown() {
    std::cout << "[T1] init/teardown x3 ..." << std::flush;
    Watchdog wd(30, "T1 init/teardown");
    for (int i = 0; i < 3; ++i) {
        auto shared = make_shared_ctx(2);
        auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, /*nc=*/4);
        CHECK(coord->is_ready(), "coordinator not ready");
        coord.reset();
        shared.reset();
    }
    std::cout << "  OK" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 2: AllReduce correctness (f16, single iter)
// ---------------------------------------------------------------------------
void test_allreduce_f16(size_t n) {
    std::cout << "[T2] f16 AllReduce n=" << n << " ..." << std::flush;
    Watchdog wd(30, "T2 f16 AllReduce");
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 1);

    std::vector<RankScratch> rs(2);
    for (int r = 0; r < 2; ++r) rs[r].init(shared->context, shared->devices[r]);

    const size_t bytes = n * sizeof(ov::float16);
    std::vector<void*> ins(2), outs(2);
    for (int r = 0; r < 2; ++r) {
        ins[r]  = rs[r].alloc(bytes);
        outs[r] = rs[r].alloc(bytes);
    }

    // rank 0 -> 1.0, rank 1 -> 2.0; expected sum 3.0
    std::vector<ov::float16> h0(n, ov::float16(1.0f));
    std::vector<ov::float16> h1(n, ov::float16(2.0f));
    rs[0].copy_from_host(ins[0], h0.data(), bytes);
    rs[1].copy_from_host(ins[1], h1.data(), bytes);

    run_collective(*coord, 0, ins, outs, n, ov::element::f16);
    std::cerr << "[T2] collective returned, copying back ..." << std::endl;

    std::vector<ov::float16> r0(n), r1(n);
    rs[0].copy_to_host(r0.data(), outs[0], bytes);
    std::cerr << "[T2] copy_to_host rank0 done" << std::endl;
    rs[1].copy_to_host(r1.data(), outs[1], bytes);
    std::cerr << "[T2] copy_to_host rank1 done" << std::endl;

    int bad0 = 0, bad1 = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(static_cast<float>(r0[i]) - 3.0f) > 1e-3f) ++bad0;
        if (std::fabs(static_cast<float>(r1[i]) - 3.0f) > 1e-3f) ++bad1;
    }
    CHECK(bad0 == 0 && bad1 == 0,
          "values mismatch: rank0_bad=" << bad0 << " rank1_bad=" << bad1);

    for (int r = 0; r < 2; ++r) {
        ov::zeMemFree(shared->context, ins[r]);
        ov::zeMemFree(shared->context, outs[r]);
        rs[r].destroy();
    }
    coord.reset();
    shared.reset();
    std::cout << "  OK" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 3: Multi-iteration with stable pointers (plan should be reused)
//   iter k:  rank0 = k+1, rank1 = k+2  →  expected = 2k+3
// ---------------------------------------------------------------------------
void test_multi_iter(size_t n, int iters) {
    std::cout << "[T3] multi-iter f16 n=" << n << " iters=" << iters << " ..." << std::flush;
    Watchdog wd(60, "T3 multi-iter");
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 1);

    std::vector<RankScratch> rs(2);
    for (int r = 0; r < 2; ++r) rs[r].init(shared->context, shared->devices[r]);

    const size_t bytes = n * sizeof(ov::float16);
    std::vector<void*> ins(2), outs(2);
    for (int r = 0; r < 2; ++r) {
        ins[r]  = rs[r].alloc(bytes);
        outs[r] = rs[r].alloc(bytes);
    }

    std::vector<ov::float16> h0(n), h1(n), r0(n), r1(n);

    for (int k = 0; k < iters; ++k) {
        for (size_t i = 0; i < n; ++i) {
            h0[i] = ov::float16(static_cast<float>(k + 1));
            h1[i] = ov::float16(static_cast<float>(k + 2));
        }
        rs[0].copy_from_host(ins[0], h0.data(), bytes);
        rs[1].copy_from_host(ins[1], h1.data(), bytes);

        run_collective(*coord, 0, ins, outs, n, ov::element::f16);

        rs[0].copy_to_host(r0.data(), outs[0], bytes);
        rs[1].copy_to_host(r1.data(), outs[1], bytes);

        const float exp = static_cast<float>(2 * k + 3);
        int bad = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(static_cast<float>(r0[i]) - exp) > 1e-3f) ++bad;
            if (std::fabs(static_cast<float>(r1[i]) - exp) > 1e-3f) ++bad;
        }
        CHECK(bad == 0, "iter " << k << " mismatch (" << bad << " elems, expected " << exp << ")");
    }

    for (int r = 0; r < 2; ++r) {
        ov::zeMemFree(shared->context, ins[r]);
        ov::zeMemFree(shared->context, outs[r]);
        rs[r].destroy();
    }
    coord.reset();
    shared.reset();
    std::cout << "  OK" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 4: Shared scratch growth on size change
// ---------------------------------------------------------------------------
void test_plan_rebuild() {
    std::cout << "[T4] plan rebuild on size change ..." << std::flush;
    Watchdog wd(60, "T4 plan rebuild");
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 1);

    std::vector<RankScratch> rs(2);
    for (int r = 0; r < 2; ++r) rs[r].init(shared->context, shared->devices[r]);

    auto sizes = std::vector<size_t>{1024, 2048, 4096, 1024};
    for (size_t n : sizes) {
        const size_t bytes = n * sizeof(ov::float16);
        std::vector<void*> ins(2), outs(2);
        for (int r = 0; r < 2; ++r) {
            ins[r]  = rs[r].alloc(bytes);
            outs[r] = rs[r].alloc(bytes);
        }
        std::vector<ov::float16> h0(n, ov::float16(5.0f));
        std::vector<ov::float16> h1(n, ov::float16(7.0f));
        rs[0].copy_from_host(ins[0], h0.data(), bytes);
        rs[1].copy_from_host(ins[1], h1.data(), bytes);

        run_collective(*coord, 0, ins, outs, n, ov::element::f16);

        std::vector<ov::float16> r0(n), r1(n);
        rs[0].copy_to_host(r0.data(), outs[0], bytes);
        rs[1].copy_to_host(r1.data(), outs[1], bytes);
        int bad = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(static_cast<float>(r0[i]) - 12.0f) > 1e-3f) ++bad;
            if (std::fabs(static_cast<float>(r1[i]) - 12.0f) > 1e-3f) ++bad;
        }
        CHECK(bad == 0, "n=" << n << " mismatch (" << bad << " elems)");

        for (int r = 0; r < 2; ++r) {
            ov::zeMemFree(shared->context, ins[r]);
            ov::zeMemFree(shared->context, outs[r]);
        }
    }

        const auto stats = coord->get_scratch_stats();
        const size_t expected_capacity = 4096 * sizeof(ov::float16);
        CHECK(stats.payload_capacity_bytes == expected_capacity,
            "unexpected scratch capacity: " << stats.payload_capacity_bytes);
        CHECK(stats.total_allocated_bytes == 2 * expected_capacity,
            "scratch should contain one max-sized allocation per rank");
        CHECK(stats.growth_count == 3, "scratch must grow only for 1024, 2048 and 4096");
        CHECK(stats.allocation_count == 6, "TP=2 must allocate two buffers per growth");

    for (int r = 0; r < 2; ++r) rs[r].destroy();
    coord.reset();
    shared.reset();
    std::cout << "  OK" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 5: f32 dtype
// ---------------------------------------------------------------------------
void test_allreduce_f32(size_t n) {
    std::cout << "[T5] f32 AllReduce n=" << n << " ..." << std::flush;
    Watchdog wd(30, "T5 f32 AllReduce");
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 1);

    std::vector<RankScratch> rs(2);
    for (int r = 0; r < 2; ++r) rs[r].init(shared->context, shared->devices[r]);

    const size_t bytes = n * sizeof(float);
    std::vector<void*> ins(2), outs(2);
    for (int r = 0; r < 2; ++r) {
        ins[r]  = rs[r].alloc(bytes);
        outs[r] = rs[r].alloc(bytes);
    }
    std::vector<float> h0(n, 1.5f), h1(n, 2.25f);
    rs[0].copy_from_host(ins[0], h0.data(), bytes);
    rs[1].copy_from_host(ins[1], h1.data(), bytes);

    run_collective(*coord, 0, ins, outs, n, ov::element::f32);

    std::vector<float> r0(n), r1(n);
    rs[0].copy_to_host(r0.data(), outs[0], bytes);
    rs[1].copy_to_host(r1.data(), outs[1], bytes);

    int bad = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(r0[i] - 3.75f) > 1e-5f) ++bad;
        if (std::fabs(r1[i] - 3.75f) > 1e-5f) ++bad;
    }
    CHECK(bad == 0, "f32 mismatch: " << bad << " elems");

    for (int r = 0; r < 2; ++r) {
        ov::zeMemFree(shared->context, ins[r]);
        ov::zeMemFree(shared->context, outs[r]);
        rs[r].destroy();
    }
    coord.reset();
    shared.reset();
    std::cout << "  OK" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 6: Multiple collective slots used sequentially
//
// The coordinator currently shares one compute_list/compute_queue per rank
// across all collective_id slots, so concurrent use of different slots is
// NOT supported.  The realistic usage pattern (per-layer AllReduce ops in a
// transformer forward pass) calls slots one after the other; this test
// reproduces that and verifies each slot re-records the coordinator-wide
// command lists while sharing one bounded scratch arena.
// ---------------------------------------------------------------------------
void test_multi_slot(size_t n) {
    std::cout << "[T6] multi-slot sequential n=" << n << " ..." << std::flush;
    Watchdog wd(60, "T6 multi-slot sequential");
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 2);

    std::vector<RankScratch> rs(2);
    for (int r = 0; r < 2; ++r) rs[r].init(shared->context, shared->devices[r]);

    const size_t bytes = n * sizeof(ov::float16);
    // Two independent sets of buffers (one per slot).
    std::vector<std::vector<void*>> ins(2, std::vector<void*>(2)),
                                    outs(2, std::vector<void*>(2));
    for (int s = 0; s < 2; ++s) {
        for (int r = 0; r < 2; ++r) {
            ins [s][r] = rs[r].alloc(bytes);
            outs[s][r] = rs[r].alloc(bytes);
        }
    }
    auto check = [&](const std::vector<void*>& outs_set, float exp, const char* tag) {
        std::vector<ov::float16> r0(n), r1(n);
        rs[0].copy_to_host(r0.data(), outs_set[0], bytes);
        rs[1].copy_to_host(r1.data(), outs_set[1], bytes);
        int bad = 0;
        for (size_t i = 0; i < n; ++i) {
            if (std::fabs(static_cast<float>(r0[i]) - exp) > 1e-3f) ++bad;
            if (std::fabs(static_cast<float>(r1[i]) - exp) > 1e-3f) ++bad;
        }
        CHECK(bad == 0, "slot " << tag << " mismatch (" << bad << " elems)");
    };

    // Change values every repetition and check immediately.  This catches
    // executing a stale command list from the other collective slot.
    for (int rep = 0; rep < 3; ++rep) {
        const float s0_r0 = static_cast<float>(rep + 1);
        const float s0_r1 = static_cast<float>(rep + 2);
        const float s1_r0 = static_cast<float>(rep + 4);
        const float s1_r1 = static_cast<float>(rep + 5);
        std::vector<ov::float16> h0(n, ov::float16(s0_r0));
        std::vector<ov::float16> h1(n, ov::float16(s0_r1));
        std::vector<ov::float16> h4(n, ov::float16(s1_r0));
        std::vector<ov::float16> h5(n, ov::float16(s1_r1));
        rs[0].copy_from_host(ins[0][0], h0.data(), bytes);
        rs[1].copy_from_host(ins[0][1], h1.data(), bytes);
        rs[0].copy_from_host(ins[1][0], h4.data(), bytes);
        rs[1].copy_from_host(ins[1][1], h5.data(), bytes);

        run_collective(*coord, 0, ins[0], outs[0], n, ov::element::f16);
        check(outs[0], s0_r0 + s0_r1, "0");
        run_collective(*coord, 1, ins[1], outs[1], n, ov::element::f16);
        check(outs[1], s1_r0 + s1_r1, "1");
    }

    const auto stats = coord->get_scratch_stats();
    CHECK(stats.payload_capacity_bytes == bytes,
          "scratch capacity must equal the largest collective payload");
    CHECK(stats.total_allocated_bytes == 2 * bytes,
          "scratch size must be independent of collective slot count");
    CHECK(stats.growth_count == 1, "equal-size slots must share the first allocation");
    CHECK(stats.allocation_count == 2, "TP=2 must own one allocation per rank");

    for (int s = 0; s < 2; ++s)
        for (int r = 0; r < 2; ++r) {
            ov::zeMemFree(shared->context, ins [s][r]);
            ov::zeMemFree(shared->context, outs[s][r]);
        }
    for (int r = 0; r < 2; ++r) rs[r].destroy();
    coord.reset();
    shared.reset();
    std::cout << "  OK" << std::endl;
}

// ---------------------------------------------------------------------------
// Test 7: Input/output aliasing supported by the coordinator API.
// ---------------------------------------------------------------------------
void test_in_place(size_t n) {
    std::cout << "[T7] in-place f16 n=" << n << " ..." << std::flush;
    Watchdog wd(30, "T7 in-place");
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 1);

    std::vector<RankScratch> rs(2);
    for (int r = 0; r < 2; ++r) rs[r].init(shared->context, shared->devices[r]);

    const size_t bytes = n * sizeof(ov::float16);
    std::vector<void*> buffers(2);
    for (int r = 0; r < 2; ++r) buffers[r] = rs[r].alloc(bytes);

    std::vector<ov::float16> h0(n, ov::float16(1.0f));
    std::vector<ov::float16> h1(n, ov::float16(2.0f));
    rs[0].copy_from_host(buffers[0], h0.data(), bytes);
    rs[1].copy_from_host(buffers[1], h1.data(), bytes);

    run_collective(*coord, 0, buffers, buffers, n, ov::element::f16);

    std::vector<ov::float16> out0(n), out1(n);
    rs[0].copy_to_host(out0.data(), buffers[0], bytes);
    rs[1].copy_to_host(out1.data(), buffers[1], bytes);
    int bad = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::fabs(static_cast<float>(out0[i]) - 3.0f) > 1e-3f) ++bad;
        if (std::fabs(static_cast<float>(out1[i]) - 3.0f) > 1e-3f) ++bad;
    }
    CHECK(bad == 0, "in-place result mismatch (" << bad << " elems)");

    for (int r = 0; r < 2; ++r) {
        ov::zeMemFree(shared->context, buffers[r]);
        rs[r].destroy();
    }
    coord.reset();
    shared.reset();
    std::cout << "  OK" << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    size_t n     = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : (64ULL * 1024);
    int    iters = (argc > 2) ? std::atoi(argv[2]) : 5;

    std::cout << "tp_coordinator_test  n=" << n << "  iters=" << iters << std::endl;

    // Hold the ZeroApi singleton alive for the whole run, otherwise the
    // weak_ptr gets dropped and ze_loader is reloaded between every call,
    // wiping the driver list initialized by zeInit.
    auto zero_api = ov::ZeroApi::get_instance();
    (void)zero_api;

    // Probe hardware once: if we cannot get >=2 discrete GPUs under a single
    // L0 driver, skip the entire suite gracefully (exit 0).
    {
        ZE(ov::zeInit(0));
        auto probe = make_shared_ctx(2);
        if (!probe) {
            std::cout << "[SKIP] need >= 2 discrete Intel GPUs under one L0 driver" << std::endl;
            return 0;
        }
        // probe is destroyed here; tests construct their own shared ctx.
    }

    try {
        test_init_teardown();
        test_allreduce_f16(n);
        test_multi_iter(n, iters);
        test_plan_rebuild();
        test_allreduce_f32(n);
        test_multi_slot(n);
        test_in_place(n);
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] uncaught: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "ALL TESTS PASSED" << std::endl;
    return 0;
}

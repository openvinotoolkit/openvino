// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Tests for ov::tp_gpu::TPDeviceCoordinator.
//
// Goal: validate the L0 device-side AllReduce in isolation — without the
// graph rewriter, intel_gpu plugin, or the TP_GPU plugin's
// rendezvous of compile_model().  Catches problems with:
//   * cross-device USM visibility in a shared multi-device L0 context
//   * native-binary load into a different L0 context on the same device
//   * event signal/wait across devices
//   * rendezvous protocol reset between iterations
//   * plan rebuild on (ptr, n, dtype) change
//   * teardown order / leaks
//
// These need two discrete Intel GPUs under one L0 driver; without them the
// whole suite skips.

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#define ZERO_API_KEEP_SYMBOLS_LIST_MACRO
#include "openvino/zero_api.hpp"

#include "tp_gpu/tp_device_coordinator.hpp"
#include "tp_l0_shared_context.hpp"

using ov::tp_gpu::TPDeviceCoordinator;
using ov::tp_gpu::TPL0SharedContext;
using ov::tp_gpu::TPL0SharedContextPtr;

namespace {

/// Non-fatal so it also works in helpers that return a value; the failing
/// expression is reported verbatim.
#define EXPECT_ZE(expr) EXPECT_EQ(ZE_RESULT_SUCCESS, (expr)) << #expr

// ---------------------------------------------------------------------------
// Pick the first `min_gpus` GPUs on the first L0 driver and build a shared
// context.  Returns nullptr when no driver has enough discrete GPUs, which the
// fixture treats as "skip".
// ---------------------------------------------------------------------------
TPL0SharedContextPtr make_shared_ctx(int min_gpus) {
    EXPECT_ZE(ov::zeInit(0));

    uint32_t dcount = 0;
    EXPECT_ZE(ov::zeDriverGet(&dcount, nullptr));
    if (dcount == 0)
        return nullptr;
    std::vector<ze_driver_handle_t> drivers(dcount);
    EXPECT_ZE(ov::zeDriverGet(&dcount, drivers.data()));

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
            // device USM allocations behave differently (the P2P paths the
            // coordinator exercises do not apply meaningfully).
            const bool is_dgpu =
                (p.type == ZE_DEVICE_TYPE_GPU) && (p.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) == 0;
            if (is_dgpu)
                gpus.push_back(dh);
        }
        if (static_cast<int>(gpus.size()) >= min_gpus) {
            if (ov::zeContextCreateEx == nullptr)
                return nullptr;
            ze_context_desc_t cd{ZE_STRUCTURE_TYPE_CONTEXT_DESC, nullptr, 0};
            ze_context_handle_t ctx = nullptr;
            EXPECT_ZE(ov::zeContextCreateEx(drv, &cd, static_cast<uint32_t>(min_gpus), gpus.data(), &ctx));

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
    ze_context_handle_t ctx{nullptr};
    ze_device_handle_t dev{nullptr};
    uint32_t ord{0};
    ze_command_queue_handle_t queue{nullptr};
    ze_command_list_handle_t list{nullptr};

    static uint32_t pick_compute_ordinal(ze_device_handle_t d) {
        uint32_t qg = 0;
        ov::zeDeviceGetCommandQueueGroupProperties(d, &qg, nullptr);
        std::vector<ze_command_queue_group_properties_t> qgp(
            qg,
            {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES, nullptr});
        ov::zeDeviceGetCommandQueueGroupProperties(d, &qg, qgp.data());
        for (uint32_t g = 0; g < qg; ++g) {
            if (qgp[g].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE)
                return g;
        }
        return 0;
    }

    void init(ze_context_handle_t c, ze_device_handle_t d) {
        ctx = c;
        dev = d;
        ord = pick_compute_ordinal(d);
        ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr};
        qd.ordinal = ord;
        qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        EXPECT_ZE(ov::zeCommandQueueCreate(ctx, dev, &qd, &queue));
        ze_command_list_desc_t ld{ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr};
        ld.commandQueueGroupOrdinal = ord;
        EXPECT_ZE(ov::zeCommandListCreate(ctx, dev, &ld, &list));
    }

    void* alloc(size_t bytes) {
        ze_device_mem_alloc_desc_t mad{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr};
        void* p = nullptr;
        EXPECT_ZE(ov::zeMemAllocDevice(ctx, &mad, bytes, 64, dev, &p));
        return p;
    }

    void copy_from_host(void* dev_ptr, const void* host, size_t bytes) {
        EXPECT_ZE(ov::zeCommandListReset(list));
        EXPECT_ZE(ov::zeCommandListAppendMemoryCopy(list, dev_ptr, host, bytes, nullptr, 0, nullptr));
        EXPECT_ZE(ov::zeCommandListClose(list));
        EXPECT_ZE(ov::zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
        EXPECT_ZE(ov::zeCommandQueueSynchronize(queue, UINT64_MAX));
    }

    void copy_to_host(void* host, const void* dev_ptr, size_t bytes) {
        EXPECT_ZE(ov::zeCommandListReset(list));
        EXPECT_ZE(ov::zeCommandListAppendMemoryCopy(list, host, dev_ptr, bytes, nullptr, 0, nullptr));
        EXPECT_ZE(ov::zeCommandListClose(list));
        EXPECT_ZE(ov::zeCommandQueueExecuteCommandLists(queue, 1, &list, nullptr));
        EXPECT_ZE(ov::zeCommandQueueSynchronize(queue, UINT64_MAX));
    }

    void destroy() {
        if (list)
            ov::zeCommandListDestroy(list);
        if (queue)
            ov::zeCommandQueueDestroy(queue);
        list = nullptr;
        queue = nullptr;
    }
};

/// A stalled rendezvous would hang the runner forever, so give every test a
/// deadline.  gtest has already printed the test name by the time this fires.
struct Watchdog {
    std::atomic<bool> done{false};
    std::thread thread;

    explicit Watchdog(int seconds) {
        thread = std::thread([this, seconds] {
            for (int i = 0; i < seconds * 10; ++i) {
                if (done.load())
                    return;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            GTEST_LOG_(FATAL) << "watchdog: test did not finish in " << seconds << " s — hung";
        });
    }
    ~Watchdog() {
        done.store(true);
        if (thread.joinable())
            thread.join();
    }
};

/// Drive the coordinator from one thread per rank, the way the plugin's
/// per-rank infer requests do.
void run_collective(TPDeviceCoordinator& coord,
                    int collective_id,
                    const std::vector<void*>& ins,
                    const std::vector<void*>& outs,
                    size_t n,
                    ov::element::Type dtype) {
    const int ranks = static_cast<int>(ins.size());
    std::vector<std::thread> threads;
    threads.reserve(ranks);
    for (int r = 0; r < ranks; ++r) {
        threads.emplace_back([&, r] {
            EXPECT_NO_THROW(coord.allreduce(collective_id, r, ins[r], outs[r], n, dtype)) << "rank " << r;
        });
    }
    for (auto& t : threads)
        t.join();
}

/// Same, for the gather.  Only the root's buffer is written, so the other
/// ranks pass their own (ignored) destination.
void run_gather(TPDeviceCoordinator& coord,
                int collective_id,
                const std::vector<void*>& ins,
                const std::vector<void*>& outs,
                size_t rows,
                size_t slice_elems,
                ov::element::Type dtype) {
    const int ranks = static_cast<int>(ins.size());
    std::vector<std::thread> threads;
    threads.reserve(ranks);
    for (int r = 0; r < ranks; ++r) {
        threads.emplace_back([&, r] {
            EXPECT_NO_THROW(
                coord.gather_to_root(collective_id, r, ins[r], outs[r], rows, slice_elems, dtype))
                << "rank " << r;
        });
    }
    for (auto& t : threads)
        t.join();
}

/// Number of elements differing from `expected` by more than `tolerance`.
template <typename T>
size_t count_mismatches(const std::vector<T>& values, float expected, float tolerance) {
    return static_cast<size_t>(std::count_if(values.begin(), values.end(), [&](const T& value) {
        return std::fabs(static_cast<float>(value) - expected) > tolerance;
    }));
}

// ---------------------------------------------------------------------------
// Fixture: probe the hardware once and skip the suite when it is not there.
// ---------------------------------------------------------------------------
class TPDeviceCoordinatorTest : public ::testing::Test {
protected:
    /// Payload size for the cases that do not vary it themselves.
    static constexpr size_t elements = 64 * 1024;
    static constexpr float f16_tolerance = 1e-3f;
    static constexpr float f32_tolerance = 1e-5f;

    static void SetUpTestSuite() {
        // Hold the ZeroApi singleton alive for the whole run, otherwise the
        // weak_ptr gets dropped and ze_loader is reloaded between every call,
        // wiping the driver list initialized by zeInit.
        zero_api = ov::ZeroApi::get_instance();
        if (ov::zeInit(0) != ZE_RESULT_SUCCESS)
            return;
        // Probe upwards so the wider cases know whether they can run.
        for (int n = 2; n <= 8; ++n) {
            if (make_shared_ctx(n) == nullptr)
                break;
            available_gpus = n;
        }
    }

    static void TearDownTestSuite() {
        zero_api.reset();
    }

    void SetUp() override {
        if (available_gpus < 2)
            GTEST_SKIP() << "need >= 2 discrete Intel GPUs under one L0 driver";
    }

    static std::shared_ptr<ov::ZeroApi> zero_api;
    static int available_gpus;
};

std::shared_ptr<ov::ZeroApi> TPDeviceCoordinatorTest::zero_api;
int TPDeviceCoordinatorTest::available_gpus = 0;

TEST_F(TPDeviceCoordinatorTest, InitTeardownIsStable) {
    Watchdog watchdog(30);

    for (int attempt = 0; attempt < 3; ++attempt) {
        auto shared = make_shared_ctx(2);
        auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, /*collectives=*/4);
        EXPECT_TRUE(coord->is_ready()) << "attempt " << attempt;
        coord.reset();
        shared.reset();
    }
}

// The gather writes a strided band of the root's buffer, one band per rank.
// The failure it is meant to catch is a wrong pitch or origin, which shows up
// as a band landing at the wrong column rather than as garbage, so the check
// is per element against the exact value that column should hold.
TEST_F(TPDeviceCoordinatorTest, GatherToRootPlacesEveryRankSlice) {
    Watchdog watchdog(30);
    const int ranks = std::min(available_gpus, 4);
    if (ranks < 2) {
        GTEST_SKIP() << "needs at least 2 GPUs";
    }

    auto shared = make_shared_ctx(ranks);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, ranks, 1);

    // Deliberately not a round number of rows, and a slice that is not a
    // multiple of anything convenient.  The value at each position is unique
    // and stays below 2048, where f16 still counts integers exactly -- above
    // that its spacing is 2 and the comparison would flag rounding, not
    // misplacement.
    constexpr size_t rows = 5;
    constexpr size_t slice = 48;
    const size_t full = slice * static_cast<size_t>(ranks);

    std::vector<RankScratch> rs(ranks);
    for (int r = 0; r < ranks; ++r)
        rs[r].init(shared->context, shared->devices[r]);

    std::vector<void*> ins(ranks), outs(ranks);
    for (int r = 0; r < ranks; ++r) {
        ins[r] = rs[r].alloc(rows * slice * sizeof(ov::float16));
        outs[r] = rs[r].alloc(rows * full * sizeof(ov::float16));
    }

    // Rank r, row y, column x carries a value unique to that position, so a
    // band landing at the wrong column or row is visible as such.
    auto expected_at = [&](int r, size_t y, size_t x) {
        return static_cast<float>(r * 256 + static_cast<int>(y) * 48 + static_cast<int>(x));
    };
    for (int r = 0; r < ranks; ++r) {
        std::vector<ov::float16> host(rows * slice);
        for (size_t y = 0; y < rows; ++y) {
            for (size_t x = 0; x < slice; ++x) {
                host[y * slice + x] = ov::float16(expected_at(r, y, x));
            }
        }
        rs[r].copy_from_host(ins[r], host.data(), host.size() * sizeof(ov::float16));
    }

    std::vector<ov::float16> poison(rows * full, ov::float16(-1.0f));
    rs[0].copy_from_host(outs[0], poison.data(), poison.size() * sizeof(ov::float16));

    run_gather(*coord, 0, ins, outs, rows, slice, ov::element::f16);

    std::vector<ov::float16> got(rows * full);
    rs[0].copy_to_host(got.data(), outs[0], got.size() * sizeof(ov::float16));

    size_t wrong = 0;
    for (int r = 0; r < ranks; ++r) {
        for (size_t y = 0; y < rows; ++y) {
            for (size_t x = 0; x < slice; ++x) {
                const float want = expected_at(r, y, x);
                const float have = static_cast<float>(got[y * full + r * slice + x]);
                if (std::fabs(have - want) > 0.5f) {
                    ++wrong;
                }
            }
        }
    }
    EXPECT_EQ(wrong, 0u) << "gathered matrix differs in " << wrong << " of " << (rows * full)
                         << " elements";

    for (int r = 0; r < ranks; ++r) {
        ov::zeMemFree(shared->context, ins[r]);
        ov::zeMemFree(shared->context, outs[r]);
        rs[r].destroy();
    }
}

TEST_F(TPDeviceCoordinatorTest, AllReduceF16) {
    Watchdog watchdog(30);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 1);

    std::vector<RankScratch> rs(2);
    for (int r = 0; r < 2; ++r)
        rs[r].init(shared->context, shared->devices[r]);

    const size_t bytes = elements * sizeof(ov::float16);
    std::vector<void*> ins(2), outs(2);
    for (int r = 0; r < 2; ++r) {
        ins[r] = rs[r].alloc(bytes);
        outs[r] = rs[r].alloc(bytes);
    }

    // rank 0 -> 1.0, rank 1 -> 2.0; expected sum 3.0
    std::vector<ov::float16> h0(elements, ov::float16(1.0f));
    std::vector<ov::float16> h1(elements, ov::float16(2.0f));
    rs[0].copy_from_host(ins[0], h0.data(), bytes);
    rs[1].copy_from_host(ins[1], h1.data(), bytes);

    run_collective(*coord, 0, ins, outs, elements, ov::element::f16);

    std::vector<ov::float16> r0(elements), r1(elements);
    rs[0].copy_to_host(r0.data(), outs[0], bytes);
    rs[1].copy_to_host(r1.data(), outs[1], bytes);

    EXPECT_EQ(count_mismatches(r0, 3.0f, f16_tolerance), 0u) << "rank 0";
    EXPECT_EQ(count_mismatches(r1, 3.0f, f16_tolerance), 0u) << "rank 1";

    for (int r = 0; r < 2; ++r) {
        ov::zeMemFree(shared->context, ins[r]);
        ov::zeMemFree(shared->context, outs[r]);
        rs[r].destroy();
    }
}

TEST_F(TPDeviceCoordinatorTest, AllReduceF32) {
    Watchdog watchdog(30);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 1);

    std::vector<RankScratch> rs(2);
    for (int r = 0; r < 2; ++r)
        rs[r].init(shared->context, shared->devices[r]);

    const size_t bytes = elements * sizeof(float);
    std::vector<void*> ins(2), outs(2);
    for (int r = 0; r < 2; ++r) {
        ins[r] = rs[r].alloc(bytes);
        outs[r] = rs[r].alloc(bytes);
    }

    std::vector<float> h0(elements, 1.5f), h1(elements, 2.25f);
    rs[0].copy_from_host(ins[0], h0.data(), bytes);
    rs[1].copy_from_host(ins[1], h1.data(), bytes);

    run_collective(*coord, 0, ins, outs, elements, ov::element::f32);

    std::vector<float> r0(elements), r1(elements);
    rs[0].copy_to_host(r0.data(), outs[0], bytes);
    rs[1].copy_to_host(r1.data(), outs[1], bytes);

    EXPECT_EQ(count_mismatches(r0, 3.75f, f32_tolerance), 0u) << "rank 0";
    EXPECT_EQ(count_mismatches(r1, 3.75f, f32_tolerance), 0u) << "rank 1";

    for (int r = 0; r < 2; ++r) {
        ov::zeMemFree(shared->context, ins[r]);
        ov::zeMemFree(shared->context, outs[r]);
        rs[r].destroy();
    }
}

// Pointers stay stable across iterations, so the plan is built once and reused.
// A stale plan or a rendezvous that does not reset would surface as the
// previous iteration's values.
//   iteration k:  rank0 = k+1, rank1 = k+2  ->  expected = 2k+3
TEST_F(TPDeviceCoordinatorTest, ReusesPlanAcrossIterations) {
    Watchdog watchdog(60);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 1);

    std::vector<RankScratch> rs(2);
    for (int r = 0; r < 2; ++r)
        rs[r].init(shared->context, shared->devices[r]);

    const size_t bytes = elements * sizeof(ov::float16);
    std::vector<void*> ins(2), outs(2);
    for (int r = 0; r < 2; ++r) {
        ins[r] = rs[r].alloc(bytes);
        outs[r] = rs[r].alloc(bytes);
    }

    std::vector<ov::float16> h0(elements), h1(elements), r0(elements), r1(elements);
    for (int k = 0; k < 5; ++k) {
        std::fill(h0.begin(), h0.end(), ov::float16(static_cast<float>(k + 1)));
        std::fill(h1.begin(), h1.end(), ov::float16(static_cast<float>(k + 2)));
        rs[0].copy_from_host(ins[0], h0.data(), bytes);
        rs[1].copy_from_host(ins[1], h1.data(), bytes);

        run_collective(*coord, 0, ins, outs, elements, ov::element::f16);

        rs[0].copy_to_host(r0.data(), outs[0], bytes);
        rs[1].copy_to_host(r1.data(), outs[1], bytes);

        const float expected = static_cast<float>(2 * k + 3);
        EXPECT_EQ(count_mismatches(r0, expected, f16_tolerance), 0u) << "iteration " << k << " rank 0";
        EXPECT_EQ(count_mismatches(r1, expected, f16_tolerance), 0u) << "iteration " << k << " rank 1";
    }

    for (int r = 0; r < 2; ++r) {
        ov::zeMemFree(shared->context, ins[r]);
        ov::zeMemFree(shared->context, outs[r]);
        rs[r].destroy();
    }
}

// The scratch arena grows to the largest payload seen and never shrinks, so the
// trailing smaller size must not trigger a fourth reallocation.
TEST_F(TPDeviceCoordinatorTest, GrowsScratchOnSizeChange) {
    Watchdog watchdog(60);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 1);

    std::vector<RankScratch> rs(2);
    for (int r = 0; r < 2; ++r)
        rs[r].init(shared->context, shared->devices[r]);

    for (size_t n : {size_t{1024}, size_t{2048}, size_t{4096}, size_t{1024}}) {
        const size_t bytes = n * sizeof(ov::float16);
        std::vector<void*> ins(2), outs(2);
        for (int r = 0; r < 2; ++r) {
            ins[r] = rs[r].alloc(bytes);
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
        EXPECT_EQ(count_mismatches(r0, 12.0f, f16_tolerance), 0u) << "n=" << n << " rank 0";
        EXPECT_EQ(count_mismatches(r1, 12.0f, f16_tolerance), 0u) << "n=" << n << " rank 1";

        for (int r = 0; r < 2; ++r) {
            ov::zeMemFree(shared->context, ins[r]);
            ov::zeMemFree(shared->context, outs[r]);
        }
    }

    const auto stats = coord->get_scratch_stats();
    const size_t expected_capacity = 4096 * sizeof(ov::float16);
    EXPECT_EQ(stats.payload_capacity_bytes, expected_capacity);
    EXPECT_EQ(stats.total_allocated_bytes, 2 * expected_capacity)
        << "scratch should hold one max-sized allocation per rank";
    EXPECT_EQ(stats.growth_count, 3u) << "scratch must grow only for 1024, 2048 and 4096";
    EXPECT_EQ(stats.allocation_count, 6u) << "TP=2 must allocate two buffers per growth";

    for (int r = 0; r < 2; ++r)
        rs[r].destroy();
}

// The coordinator shares one compute list/queue per rank across all
// collective_id slots, so slots are used one after another — the way per-layer
// AllReduce ops appear in a transformer forward pass.  Each slot must re-record
// the command lists while sharing one bounded scratch arena.
TEST_F(TPDeviceCoordinatorTest, ReusesCollectiveSlotsSequentially) {
    Watchdog watchdog(60);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 2);

    std::vector<RankScratch> rs(2);
    for (int r = 0; r < 2; ++r)
        rs[r].init(shared->context, shared->devices[r]);

    const size_t bytes = elements * sizeof(ov::float16);
    std::vector<std::vector<void*>> ins(2, std::vector<void*>(2)), outs(2, std::vector<void*>(2));
    for (int s = 0; s < 2; ++s) {
        for (int r = 0; r < 2; ++r) {
            ins[s][r] = rs[r].alloc(bytes);
            outs[s][r] = rs[r].alloc(bytes);
        }
    }

    // Change the values every repetition and check immediately.  This catches
    // executing a stale command list left behind by the other slot.
    for (int rep = 0; rep < 3; ++rep) {
        const float operands[2][2] = {{static_cast<float>(rep + 1), static_cast<float>(rep + 2)},
                                      {static_cast<float>(rep + 4), static_cast<float>(rep + 5)}};
        for (int s = 0; s < 2; ++s) {
            for (int r = 0; r < 2; ++r) {
                std::vector<ov::float16> host(elements, ov::float16(operands[s][r]));
                rs[r].copy_from_host(ins[s][r], host.data(), bytes);
            }
        }

        for (int s = 0; s < 2; ++s) {
            run_collective(*coord, s, ins[s], outs[s], elements, ov::element::f16);

            const float expected = operands[s][0] + operands[s][1];
            for (int r = 0; r < 2; ++r) {
                std::vector<ov::float16> host(elements);
                rs[r].copy_to_host(host.data(), outs[s][r], bytes);
                EXPECT_EQ(count_mismatches(host, expected, f16_tolerance), 0u)
                    << "repetition " << rep << " slot " << s << " rank " << r;
            }
        }
    }

    const auto stats = coord->get_scratch_stats();
    EXPECT_EQ(stats.payload_capacity_bytes, bytes) << "capacity must equal the largest collective payload";
    EXPECT_EQ(stats.total_allocated_bytes, 2 * bytes) << "scratch size must not depend on the slot count";
    EXPECT_EQ(stats.growth_count, 1u) << "equal-size slots must share the first allocation";
    EXPECT_EQ(stats.allocation_count, 2u) << "TP=2 must own one allocation per rank";

    for (int s = 0; s < 2; ++s) {
        for (int r = 0; r < 2; ++r) {
            ov::zeMemFree(shared->context, ins[s][r]);
            ov::zeMemFree(shared->context, outs[s][r]);
        }
    }
    for (int r = 0; r < 2; ++r)
        rs[r].destroy();
}

// Input and output may alias: a row-parallel MatMul output is reduced in place
// rather than through a second buffer.
TEST_F(TPDeviceCoordinatorTest, SupportsInPlaceAllReduce) {
    Watchdog watchdog(30);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 1);

    std::vector<RankScratch> rs(2);
    for (int r = 0; r < 2; ++r)
        rs[r].init(shared->context, shared->devices[r]);

    const size_t bytes = elements * sizeof(ov::float16);
    std::vector<void*> buffers(2);
    for (int r = 0; r < 2; ++r)
        buffers[r] = rs[r].alloc(bytes);

    std::vector<ov::float16> h0(elements, ov::float16(1.0f));
    std::vector<ov::float16> h1(elements, ov::float16(2.0f));
    rs[0].copy_from_host(buffers[0], h0.data(), bytes);
    rs[1].copy_from_host(buffers[1], h1.data(), bytes);

    run_collective(*coord, 0, buffers, buffers, elements, ov::element::f16);

    std::vector<ov::float16> r0(elements), r1(elements);
    rs[0].copy_to_host(r0.data(), buffers[0], bytes);
    rs[1].copy_to_host(r1.data(), buffers[1], bytes);
    EXPECT_EQ(count_mismatches(r0, 3.0f, f16_tolerance), 0u) << "rank 0";
    EXPECT_EQ(count_mismatches(r1, 3.0f, f16_tolerance), 0u) << "rank 1";

    for (int r = 0; r < 2; ++r) {
        ov::zeMemFree(shared->context, buffers[r]);
        rs[r].destroy();
    }
}

// Every other case runs two ranks.  The reduction tree and the rendezvous both
// depend on the rank count, so exercise wider worlds as well.
TEST_F(TPDeviceCoordinatorTest, AllReduceAcrossWiderWorlds) {
    Watchdog watchdog(120);

    for (int ranks : {3, 4}) {
        SCOPED_TRACE("ranks " + std::to_string(ranks));
        if (available_gpus < ranks)
            continue;

        auto shared = make_shared_ctx(ranks);
        auto coord = std::make_shared<TPDeviceCoordinator>(shared, ranks, 1);

        std::vector<RankScratch> rs(ranks);
        for (int r = 0; r < ranks; ++r)
            rs[r].init(shared->context, shared->devices[r]);

        const size_t bytes = elements * sizeof(ov::float16);
        std::vector<void*> ins(ranks), outs(ranks);
        for (int r = 0; r < ranks; ++r) {
            ins[r] = rs[r].alloc(bytes);
            outs[r] = rs[r].alloc(bytes);
        }

        // Several iterations on the same collective slot: the funnel clears
        // its recv/reduce/bcast events from the recorded command lists, and a
        // reset that lands in the wrong list only shows up from the second
        // call on, when a stale signal lets a wait through early.
        for (int k = 0; k < 5; ++k) {
            SCOPED_TRACE("iteration " + std::to_string(k));
            float expected = 0.0f;
            for (int r = 0; r < ranks; ++r) {
                // A distinct value per rank, so a rank missing from the sum is
                // visible in the total rather than hidden by equal operands.
                const float value = static_cast<float>(r + 1 + k);
                expected += value;
                std::vector<ov::float16> host(elements, ov::float16(value));
                rs[r].copy_from_host(ins[r], host.data(), bytes);
            }

            run_collective(*coord, 0, ins, outs, elements, ov::element::f16);

            for (int r = 0; r < ranks; ++r) {
                std::vector<ov::float16> host(elements);
                rs[r].copy_to_host(host.data(), outs[r], bytes);
                EXPECT_EQ(count_mismatches(host, expected, f16_tolerance), 0u) << "rank " << r;
            }
        }

        for (int r = 0; r < ranks; ++r) {
            ov::zeMemFree(shared->context, ins[r]);
            ov::zeMemFree(shared->context, outs[r]);
            rs[r].destroy();
        }
    }
}

// A collective only completes if every rank shows up.  These two cases cover
// what happens when that does not hold: the group must fail loudly instead of
// parking the calling threads forever.

TEST_F(TPDeviceCoordinatorTest, MissingRankTimesOut) {
    Watchdog watchdog(30);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 1, std::chrono::milliseconds{300});

    // Only rank 0 calls, so the enter barrier can never be satisfied.  The
    // buffers are real but never read: execution is not reached.
    RankScratch rank0;
    rank0.init(shared->context, shared->devices[0]);
    const size_t bytes = elements * sizeof(float);
    void* in = rank0.alloc(bytes);
    void* out = rank0.alloc(bytes);

    EXPECT_THROW(coord->allreduce(0, 0, in, out, elements, ov::element::f32), ov::Exception);
    EXPECT_TRUE(coord->is_aborted());

    // The device queues are left in an unknown state, so the coordinator stays
    // failed and rejects further work immediately rather than after a timeout.
    const auto before = std::chrono::steady_clock::now();
    EXPECT_THROW(coord->allreduce(0, 0, in, out, elements, ov::element::f32), ov::Exception);
    EXPECT_LT(std::chrono::steady_clock::now() - before, std::chrono::milliseconds{300});

    ov::zeMemFree(shared->context, in);
    ov::zeMemFree(shared->context, out);
    rank0.destroy();
}

TEST_F(TPDeviceCoordinatorTest, AbortReleasesWaitingRank) {
    Watchdog watchdog(30);
    auto shared = make_shared_ctx(2);
    // No timeout: the only thing that can release the waiter is the abort.
    auto coord = std::make_shared<TPDeviceCoordinator>(shared, 2, 1, std::chrono::milliseconds{0});

    std::thread waiter([&] {
        EXPECT_THROW(coord->allreduce(0, 1, nullptr, nullptr, elements, ov::element::f32), ov::Exception);
    });

    // Give the waiter time to park on the enter barrier before aborting.
    std::this_thread::sleep_for(std::chrono::milliseconds{200});
    coord->abort_all("test-initiated abort");

    waiter.join();
    EXPECT_TRUE(coord->is_aborted());
}

}  // namespace

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Tests for the TP collectives: cldnn::ze::tp::GroupState and the recorders of
// the tp_allreduce and tp_gather impls (impls/ocl_v2/tp/).
//
// Goal: validate the L0 device-side AllReduce in isolation -- without the
// graph rewriter, the GPU graph, or the TP plugin's
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

#ifdef ENABLE_TP

// First: ze_common.hpp keeps the Level Zero symbol list it needs, which a
// plain include of openvino/zero_api.hpp before it would drop.
#include "impls/ocl_v2/tp/tp_allreduce_recorder.hpp"
#include "impls/ocl_v2/tp/tp_gather_recorder.hpp"
#include "ze/tp/tp_l0_shared_context.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <regex>
#include <set>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

#include "impls/ocl_v2/utils/kernels_db.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"

using cldnn::ze::tp::Channel;
using cldnn::ze::tp::CollectiveOptions;
using cldnn::ze::tp::GroupState;
using cldnn::ze::tp::TPL0SharedContext;
using cldnn::ze::tp::TPL0SharedContextPtr;
using ov::intel_gpu::ocl::TPAllReduceRecorder;
using ov::intel_gpu::ocl::TPGatherRecorder;

namespace {

/// A group with one channel per rank and, per collective and rank, the
/// recorder a primitive instance would hold -- driven the way the impls drive
/// them, one thread per rank.
class Collectives {
public:
    Collectives(TPL0SharedContextPtr shared,
                int world_size,
                int num_collectives,
                std::chrono::milliseconds timeout = std::chrono::milliseconds{5000},
                const CollectiveOptions& options = CollectiveOptions{})
        : m_group(std::make_shared<GroupState>(std::move(shared), world_size, num_collectives, timeout, options)),
          m_world_size(world_size) {
        for (int r = 0; r < world_size; ++r) {
            m_channels.push_back(std::make_shared<Channel>(m_group, r));
        }
        for (int i = 0; i < num_collectives * world_size; ++i) {
            m_allreduce.push_back(std::make_unique<TPAllReduceRecorder>());
            m_gather.push_back(std::make_unique<TPGatherRecorder>());
        }
    }

    void allreduce(int collective_id,
                   int rank,
                   void* in,
                   void* out,
                   size_t n,
                   ov::element::Type dtype,
                   ze_kernel_handle_t kernel,
                   ze_command_list_handle_t model_queue = nullptr) {
        m_allreduce[slot(collective_id, rank)]->run(*m_channels[rank], collective_id, in, out, n, dtype, kernel,
                                                    model_queue);
    }

    void gather_to_root(int collective_id,
                        int rank,
                        void* in,
                        void* out,
                        size_t rows,
                        size_t slice_elems,
                        ov::element::Type dtype,
                        ze_command_list_handle_t model_queue = nullptr) {
        m_gather[slot(collective_id, rank)]->run(*m_channels[rank], collective_id, in, out, rows, slice_elems, dtype,
                                                 model_queue);
    }

    const TPL0SharedContextPtr& shared_context() const {
        return m_group->shared_context();
    }
    bool is_ready() const {
        return m_group->is_ready();
    }
    bool is_aborted() const {
        return m_group->is_aborted();
    }
    bool run_spliced() const {
        return m_group->run_spliced();
    }
    void abort_all(const std::string& reason) {
        m_group->abort_all(reason);
    }
    void note_collective_started(int rank) {
        m_group->note_collective_started(rank);
    }
    GroupState::ScratchStats get_scratch_stats() const {
        return m_group->get_scratch_stats();
    }

private:
    size_t slot(int collective_id, int rank) const {
        // Out-of-range ids reach the recorder, which is what rejects them.
        const int id = std::max(0, std::min(collective_id, static_cast<int>(m_allreduce.size()) / m_world_size - 1));
        return static_cast<size_t>(id * m_world_size + rank);
    }

    // Recorders last: they go first and hold the group until they have.
    std::shared_ptr<GroupState> m_group;
    int m_world_size;
    std::vector<std::shared_ptr<Channel>> m_channels;
    std::vector<std::unique_ptr<TPAllReduceRecorder>> m_allreduce;
    std::vector<std::unique_ptr<TPGatherRecorder>> m_gather;
};

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
            // collectives exercise do not apply meaningfully).
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
// (The group owns its own queues; these helpers are independent.)
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
        if (immediate)
            ov::zeCommandListDestroy(immediate);
        if (list)
            ov::zeCommandListDestroy(list);
        if (queue)
            ov::zeCommandQueueDestroy(queue);
        immediate = nullptr;
        list = nullptr;
        queue = nullptr;
    }

    /// The in-order immediate list intel_gpu runs a model on, which is what
    /// the collectives are spliced into in production.  Created on demand:
    /// most cases take the blocking path and do not need one.
    ze_command_list_handle_t model_queue() {
        if (immediate == nullptr) {
            ze_command_queue_desc_t qd{ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr};
            qd.ordinal = ord;
            qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
            qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
            EXPECT_ZE(ov::zeCommandListCreateImmediate(ctx, dev, &qd, &immediate));
        }
        return immediate;
    }

    void wait_for_model_queue() {
        if (immediate != nullptr) {
            EXPECT_ZE(ov::zeCommandListHostSynchronize(immediate, UINT64_MAX));
        }
    }

    ze_command_list_handle_t immediate{nullptr};
};

/// The model's reduction kernel (ocl_v2/tp_allreduce.cl), specialized the way
/// the GPU plugin specializes it and built in the shared context for each rank.
///
/// A context's kernels live until release_all(), which also keeps the context
/// itself alive until then: recordings name the kernels, so they have to
/// outlive every group of the test.
class ReduceKernels {
public:
    static ze_kernel_handle_t get(const TPL0SharedContextPtr& shared, int rank, ov::element::Type dtype) {
        std::lock_guard<std::mutex> lock(mutex());
        auto& built = registry()[shared.get()];
        built.shared = shared;
        auto& entry = built.kernels[{rank, dtype}];
        if (entry.kernel == nullptr) {
            entry = build(shared->context, shared->devices[static_cast<size_t>(rank)], dtype);
        }
        return entry.kernel;
    }

    static void release_all() {
        std::lock_guard<std::mutex> lock(mutex());
        for (auto& [ctx, built] : registry()) {
            for (auto& [key, entry] : built.kernels) {
                if (entry.kernel)
                    ov::zeKernelDestroy(entry.kernel);
                if (entry.module)
                    ov::zeModuleDestroy(entry.module);
            }
        }
        registry().clear();
    }

private:
    struct Entry {
        ze_module_handle_t module{nullptr};
        ze_kernel_handle_t kernel{nullptr};
    };
    struct Built {
        TPL0SharedContextPtr shared;
        std::map<std::pair<int, ov::element::Type_t>, Entry> kernels;
    };

    static std::mutex& mutex() {
        static std::mutex m;
        return m;
    }
    static std::map<const TPL0SharedContext*, Built>& registry() {
        static std::map<const TPL0SharedContext*, Built> r;
        return r;
    }

    /// Inlines the batch headers the template includes, as the kernels cache does.
    static std::string inline_headers(const std::string& code, std::set<std::string>& seen) {
        static const std::regex include(R"(#include\s+\"([^\"]+)\")");
        std::string out;
        auto begin = code.cbegin();
        std::smatch m;
        while (std::regex_search(begin, code.cend(), m, include)) {
            out.append(begin, m[0].first);
            std::string name = m[1].str();
            name = name.substr(name.find_last_of('/') + 1);
            name = name.substr(0, name.find_last_of('.'));
            if (seen.insert(name).second) {
                out += inline_headers(std::string(ov::intel_gpu::ocl::SourcesDB::get_kernel_header(name)), seen) + "\n";
            }
            begin = m[0].second;
        }
        out.append(begin, code.cend());
        return out;
    }

    static Entry build(ze_context_handle_t ctx, ze_device_handle_t dev, ov::element::Type dtype) {
        std::string source = "#define KERNEL(name) __kernel void name\n";
        if (dtype == ov::element::bf16) {
            source += "#define TP_BF16 1\n";
        } else if (dtype == ov::element::f16) {
            source += "#define TP_F16 1\n";
        }
        std::set<std::string> seen;
        source += inline_headers(std::string(ov::intel_gpu::ocl::SourcesDB::get_kernel_template("tp_allreduce")), seen);

        ze_module_desc_t md{ZE_STRUCTURE_TYPE_MODULE_DESC, nullptr};
        md.format = cldnn::ze::ze_module_format_oclc;
        md.inputSize = source.size() + 1;
        md.pInputModule = reinterpret_cast<const uint8_t*>(source.c_str());
        md.pBuildFlags = "-cl-mad-enable -cl-std=CL2.0";

        Entry entry;
        ze_module_build_log_handle_t log = nullptr;
        const ze_result_t built = ov::zeModuleCreate(ctx, dev, &md, &entry.module, &log);
        if (built != ZE_RESULT_SUCCESS) {
            size_t size = 0;
            ov::zeModuleBuildLogGetString(log, &size, nullptr);
            std::string text(size, ' ');
            ov::zeModuleBuildLogGetString(log, &size, text.data());
            ov::zeModuleBuildLogDestroy(log);
            ADD_FAILURE() << "building tp_allreduce.cl for " << dtype << " failed: " << text;
            return {};
        }
        ov::zeModuleBuildLogDestroy(log);

        ze_kernel_desc_t kd{ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr};
        kd.pKernelName = "tp_allreduce";
        EXPECT_ZE(ov::zeKernelCreate(entry.module, &kd, &entry.kernel));
        return entry;
    }
};

ze_kernel_handle_t reduce_kernel(const Collectives& coord, int rank, ov::element::Type dtype) {
    return ReduceKernels::get(coord.shared_context(), rank, dtype);
}

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

/// Drive the collectives from one thread per rank, the way the plugin's
/// per-rank infer requests do.
///
/// `queues` is empty for the blocking path and holds one immediate command
/// list per rank for the spliced one.
void run_collective(Collectives& coord,
                    int collective_id,
                    const std::vector<void*>& ins,
                    const std::vector<void*>& outs,
                    size_t n,
                    ov::element::Type dtype,
                    const std::vector<ze_command_list_handle_t>& queues = {}) {
    const int ranks = static_cast<int>(ins.size());
    std::vector<std::thread> threads;
    threads.reserve(ranks);
    for (int r = 0; r < ranks; ++r) {
        threads.emplace_back([&, r] {
            auto* queue = queues.empty() ? nullptr : queues[r];
            EXPECT_NO_THROW(coord.allreduce(collective_id, r, ins[r], outs[r], n, dtype,
                                            reduce_kernel(coord, r, dtype), queue))
                << "rank " << r;
        });
    }
    for (auto& t : threads)
        t.join();
}

/// Same, for the gather.  Only the root's buffer is written, so the other
/// ranks pass their own (ignored) destination.
void run_gather(Collectives& coord,
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
class TPCollectivesTest : public ::testing::Test {
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

    void TearDown() override {
        ReduceKernels::release_all();
    }

    static std::shared_ptr<ov::ZeroApi> zero_api;
    static int available_gpus;
};

std::shared_ptr<ov::ZeroApi> TPCollectivesTest::zero_api;
int TPCollectivesTest::available_gpus = 0;

TEST_F(TPCollectivesTest, InitTeardownIsStable) {
    Watchdog watchdog(30);

    for (int attempt = 0; attempt < 3; ++attempt) {
        auto shared = make_shared_ctx(2);
        auto coord = std::make_shared<Collectives>(shared, 2, /*collectives=*/4);
        EXPECT_TRUE(coord->is_ready()) << "attempt " << attempt;
        coord.reset();
        shared.reset();
    }
}

// The gather writes a strided band of the root's buffer, one band per rank.
// The failure it is meant to catch is a wrong pitch or origin, which shows up
// as a band landing at the wrong column rather than as garbage, so the check
// is per element against the exact value that column should hold.
TEST_F(TPCollectivesTest, GatherToRootPlacesEveryRankSlice) {
    Watchdog watchdog(30);
    const int ranks = std::min(available_gpus, 4);
    if (ranks < 2) {
        GTEST_SKIP() << "needs at least 2 GPUs";
    }

    auto shared = make_shared_ctx(ranks);
    auto coord = std::make_shared<Collectives>(shared, ranks, 1);

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

TEST_F(TPCollectivesTest, AllReduceF16) {
    Watchdog watchdog(30);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<Collectives>(shared, 2, 1);

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

TEST_F(TPCollectivesTest, AllReduceF32) {
    Watchdog watchdog(30);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<Collectives>(shared, 2, 1);

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
TEST_F(TPCollectivesTest, ReusesPlanAcrossIterations) {
    Watchdog watchdog(60);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<Collectives>(shared, 2, 1);

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
TEST_F(TPCollectivesTest, GrowsScratchOnSizeChange) {
    Watchdog watchdog(60);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<Collectives>(shared, 2, 1);

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
    // Per rank, and per set of staging: two ranks holding two sets each.
    EXPECT_EQ(stats.total_allocated_bytes, 2 * 2 * expected_capacity)
        << "scratch should hold one max-sized allocation per rank and buffer";
    EXPECT_EQ(stats.growth_count, 3u) << "scratch must grow only for 1024, 2048 and 4096";
    EXPECT_EQ(stats.allocation_count, 6u) << "TP=2 must allocate two buffers per growth";

    for (int r = 0; r < 2; ++r)
        rs[r].destroy();
}

// Collective slots are used one after another -- the way per-layer AllReduce
// ops appear in a transformer forward pass -- each with its own recording, all
// sharing one bounded staging per rank.
TEST_F(TPCollectivesTest, ReusesCollectiveSlotsSequentially) {
    Watchdog watchdog(60);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<Collectives>(shared, 2, 2);

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
    // Two ranks, and two sets of staging per rank so that consecutive
    // instances of a collective never share bytes.  What must not appear in
    // this figure is the number of collective slots.
    EXPECT_EQ(stats.total_allocated_bytes, 2 * 2 * bytes) << "scratch size must not depend on the slot count";
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
TEST_F(TPCollectivesTest, SupportsInPlaceAllReduce) {
    Watchdog watchdog(30);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<Collectives>(shared, 2, 1);

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
TEST_F(TPCollectivesTest, AllReduceAcrossWiderWorlds) {
    Watchdog watchdog(120);

    for (int ranks : {3, 4}) {
        SCOPED_TRACE("ranks " + std::to_string(ranks));
        if (available_gpus < ranks)
            continue;

        auto shared = make_shared_ctx(ranks);
        auto coord = std::make_shared<Collectives>(shared, ranks, 1);

        std::vector<RankScratch> rs(ranks);
        for (int r = 0; r < ranks; ++r)
            rs[r].init(shared->context, shared->devices[r]);

        const size_t bytes = elements * sizeof(ov::float16);
        std::vector<void*> ins(ranks), outs(ranks);
        for (int r = 0; r < ranks; ++r) {
            ins[r] = rs[r].alloc(bytes);
            outs[r] = rs[r].alloc(bytes);
        }

        // Several iterations on the same collective slot: the recordings clear
        // their own events from the recorded command lists, and a reset that
        // lands in the wrong list only shows up from the second call on, when
        // a stale signal lets a wait through early.
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

TEST_F(TPCollectivesTest, MissingRankTimesOut) {
    Watchdog watchdog(30);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<Collectives>(shared, 2, 1, std::chrono::milliseconds{300});

    // Only rank 0 calls, so the enter barrier can never be satisfied.  The
    // buffers are real but never read: execution is not reached.
    RankScratch rank0;
    rank0.init(shared->context, shared->devices[0]);
    const size_t bytes = elements * sizeof(float);
    void* in = rank0.alloc(bytes);
    void* out = rank0.alloc(bytes);
    auto kernel = reduce_kernel(*coord, 0, ov::element::f32);

    EXPECT_THROW(coord->allreduce(0, 0, in, out, elements, ov::element::f32, kernel), ov::Exception);
    EXPECT_TRUE(coord->is_aborted());

    // The device queues are left in an unknown state, so the group stays
    // failed and rejects further work immediately rather than after a timeout.
    const auto before = std::chrono::steady_clock::now();
    EXPECT_THROW(coord->allreduce(0, 0, in, out, elements, ov::element::f32, kernel), ov::Exception);
    EXPECT_LT(std::chrono::steady_clock::now() - before, std::chrono::milliseconds{300});

    ov::zeMemFree(shared->context, in);
    ov::zeMemFree(shared->context, out);
    rank0.destroy();
}

TEST_F(TPCollectivesTest, AbortReleasesWaitingRank) {
    Watchdog watchdog(30);
    auto shared = make_shared_ctx(2);
    // No timeout: the only thing that can release the waiter is the abort.
    auto coord = std::make_shared<Collectives>(shared, 2, 1, std::chrono::milliseconds{0});

    std::thread waiter([&] {
        EXPECT_THROW(coord->allreduce(0, 1, nullptr, nullptr, elements, ov::element::f32, nullptr), ov::Exception);
    });

    // Give the waiter time to park on the enter barrier before aborting.
    std::this_thread::sleep_for(std::chrono::milliseconds{200});
    coord->abort_all("test-initiated abort");

    waiter.join();
    EXPECT_TRUE(coord->is_aborted());
}

// Once a collective rides in the model's queue there is no host-side wait left
// to time out: a rank that never signals leaves its peers waiting on the
// device, where Level Zero has no deadline.  The watchdog is what notices, and
// what it must not do is mistake a host that is deliberately running ahead for
// a group that has died -- a 32k prefill hands over every collective of a pass
// before a single one is accounted for.

TEST_F(TPCollectivesTest, WatchdogLeavesAGroupWithNothingOutstandingAlone) {
    Watchdog watchdog(30);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<Collectives>(shared, 2, 1, std::chrono::milliseconds{200});

    // Collectives handed over and none of them reported back, for several
    // times the deadline.  Nothing was actually spliced, so nothing is
    // outstanding on any device and the group is idle, not stuck.
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds{800};
    while (std::chrono::steady_clock::now() < until) {
        coord->note_collective_started(0);
        coord->note_collective_started(1);
        std::this_thread::sleep_for(std::chrono::milliseconds{30});
    }
    EXPECT_FALSE(coord->is_aborted());
}

TEST_F(TPCollectivesTest, WatchdogLeavesARunningGroupAlone) {
    Watchdog watchdog(30);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<Collectives>(shared, 2, 1, std::chrono::milliseconds{200});

    RankScratch rs[2];
    std::vector<void*> ins(2);
    std::vector<void*> outs(2);
    const size_t bytes = elements * sizeof(ov::float16);
    for (int r = 0; r < 2; ++r) {
        rs[r].init(shared->context, shared->devices[r]);
        ins[r] = rs[r].alloc(bytes);
        outs[r] = rs[r].alloc(bytes);
    }

    // Real collectives, run for several times the deadline.  The watchdog
    // polls the devices while this happens and must find them progressing.
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds{800};
    while (std::chrono::steady_clock::now() < until) {
        run_collective(*coord, 0, ins, outs, elements, ov::element::f16);
    }
    EXPECT_FALSE(coord->is_aborted());

    for (int r = 0; r < 2; ++r) {
        ov::zeMemFree(shared->context, ins[r]);
        ov::zeMemFree(shared->context, outs[r]);
        rs[r].destroy();
    }
}

// ---------------------------------------------------------------------------
// Payload shapes.
//
// Every case above reduces 64K elements, which is a whole number of the
// 128-element alignment unit the schedules split by.  The branches that handle
// what is left over -- a chunk that comes out empty, a range shorter than one
// unit -- are never reached by that size, and they are exactly the ones a
// misplaced offset hides in.
// ---------------------------------------------------------------------------

/// Device buffers for one collective, kept alive for as long as the caller
/// needs them.
///
/// Holding them matters: a plan is identified by the addresses it was recorded
/// from, so freeing a buffer and allocating another that lands on the same
/// address makes the recording look current while its residency list points at
/// an allocation the driver has destroyed.  Anything reusing one collective
/// across calls has to keep its buffers, which is also what the GPU plugin's
/// memory pool does in a real model.
struct CollectiveBuffers {
    std::vector<RankScratch> rs;
    std::vector<void*> ins, outs;
    ze_context_handle_t ctx{nullptr};

    CollectiveBuffers(const TPL0SharedContextPtr& shared, int ranks, size_t bytes)
        : rs(ranks),
          ins(ranks),
          outs(ranks),
          ctx(shared->context) {
        for (int r = 0; r < ranks; ++r) {
            rs[r].init(shared->context, shared->devices[r]);
            ins[r] = rs[r].alloc(bytes);
            outs[r] = rs[r].alloc(bytes);
        }
    }

    ~CollectiveBuffers() {
        for (size_t r = 0; r < rs.size(); ++r) {
            ov::zeMemFree(ctx, ins[r]);
            ov::zeMemFree(ctx, outs[r]);
            rs[r].destroy();
        }
    }
};

/// One all-reduce of `n` elements with rank r contributing (r + 1), checked on
/// every rank against the expected sum.  Values stay small so f16 counts them
/// exactly and a mismatch means misplacement, not rounding.
template <typename T>
void expect_allreduce_sum(Collectives& coord,
                          CollectiveBuffers& buffers,
                          int collective_id,
                          size_t n,
                          ov::element::Type dtype,
                          float tolerance) {
    const int ranks = static_cast<int>(buffers.rs.size());
    const size_t bytes = n * sizeof(T);
    float expected = 0.0f;
    for (int r = 0; r < ranks; ++r) {
        const float value = static_cast<float>(r + 1);
        expected += value;
        std::vector<T> host(n, static_cast<T>(value));
        buffers.rs[r].copy_from_host(buffers.ins[r], host.data(), bytes);
    }

    run_collective(coord, collective_id, buffers.ins, buffers.outs, n, dtype);

    for (int r = 0; r < ranks; ++r) {
        std::vector<T> host(n);
        buffers.rs[r].copy_to_host(host.data(), buffers.outs[r], bytes);
        EXPECT_EQ(count_mismatches(host, expected, tolerance), 0u)
            << "n=" << n << " ranks=" << ranks << " rank " << r;
    }
}

// A payload shorter than the alignment unit leaves every chunk but the first
// empty, and one shorter than the world size leaves whole ranks with nothing to
// send.  Both have to signal their step anyway or a peer waits forever.
TEST_F(TPCollectivesTest, AllReduceHandlesPayloadsBelowTheAlignmentUnit) {
    Watchdog watchdog(120);
    const int ranks = std::min(available_gpus, 4);
    auto shared = make_shared_ctx(ranks);

    for (size_t n : {size_t{1}, size_t{2}, size_t{63}, size_t{127}, size_t{128}, size_t{129}}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        auto coord = std::make_shared<Collectives>(shared, ranks, 1);
        CollectiveBuffers buffers(shared, ranks, n * sizeof(ov::float16));
        expect_allreduce_sum<ov::float16>(*coord, buffers, 0, n, ov::element::f16, f16_tolerance);
        EXPECT_FALSE(coord->is_aborted());
    }
}

// Sizes chosen so that neither the world size nor the alignment divides them:
// the last populated chunk is clipped to the payload and the tail must not run
// past it.
TEST_F(TPCollectivesTest, AllReduceHandlesPayloadsThatDoNotDivide) {
    Watchdog watchdog(120);

    for (int ranks : {2, 3, 4}) {
        if (available_gpus < ranks)
            continue;
        SCOPED_TRACE("ranks " + std::to_string(ranks));
        auto shared = make_shared_ctx(ranks);

        for (bool halving : {true, false}) {
            SCOPED_TRACE(halving ? "halving allowed" : "ring only");
            for (size_t n : {size_t{1021}, size_t{4099}}) {
                SCOPED_TRACE("n=" + std::to_string(n));
                CollectiveOptions config;
                config.enable_halving = halving;
                auto coord = std::make_shared<Collectives>(shared,
                                                                   ranks,
                                                                   1,
                                                                   std::chrono::milliseconds{5000},
                                                                   config);
                CollectiveBuffers buffers(shared, ranks, n * sizeof(float));
                expect_allreduce_sum<float>(*coord, buffers, 0, n, ov::element::f32, f32_tolerance);
                EXPECT_FALSE(coord->is_aborted());
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Schedule selection.
//
// record_rank picks halving over the ring when the world is a power of two and
// the payload fits under halving_max_bytes.  Both arms have to produce the same
// numbers, and the existing cases only ever take one of them for a given world.
// ---------------------------------------------------------------------------

TEST_F(TPCollectivesTest, RingAndHalvingAgree) {
    Watchdog watchdog(120);
    if (available_gpus < 4)
        GTEST_SKIP() << "halving needs a power-of-two world larger than a pair";

    auto shared = make_shared_ctx(4);
    constexpr size_t n = 8 * 1024;

    for (bool halving : {true, false}) {
        SCOPED_TRACE(halving ? "halving" : "ring");
        CollectiveOptions config;
        config.enable_halving = halving;
        auto coord = std::make_shared<Collectives>(shared,
                                                           4,
                                                           1,
                                                           std::chrono::milliseconds{5000},
                                                           config);
        CollectiveBuffers buffers(shared, 4, n * sizeof(float));
        expect_allreduce_sum<float>(*coord, buffers, 0, n, ov::element::f32, f32_tolerance);
        EXPECT_FALSE(coord->is_aborted());
    }
}

// The ceiling is a per-collective decision, so one recorder can record
// halving for a small payload and the ring for a large one.  Lowering the
// ceiling puts the flip within reach of a test-sized payload.
TEST_F(TPCollectivesTest, ScheduleFlipsAtTheHalvingCeiling) {
    Watchdog watchdog(120);
    if (available_gpus < 4)
        GTEST_SKIP() << "halving needs a power-of-two world larger than a pair";

    auto shared = make_shared_ctx(4);
    constexpr size_t ceiling_elems = 2048;  // f32 -> 8 KB
    CollectiveOptions config;
    config.halving_max_bytes = ceiling_elems * sizeof(float);
    auto coord =
        std::make_shared<Collectives>(shared, 4, 1, std::chrono::milliseconds{5000}, config);

    // Under, exactly at, and over the ceiling, on the same collective slot and
    // the same buffers: the last call has to re-record from halving to the ring
    // in place.
    CollectiveBuffers buffers(shared, 4, ceiling_elems * 2 * sizeof(float));
    for (size_t n : {ceiling_elems / 2, ceiling_elems, ceiling_elems * 2}) {
        SCOPED_TRACE("n=" + std::to_string(n));
        expect_allreduce_sum<float>(*coord, buffers, 0, n, ov::element::f32, f32_tolerance);
    }
    EXPECT_FALSE(coord->is_aborted());
}

// ---------------------------------------------------------------------------
// Re-recording in place.
// ---------------------------------------------------------------------------

// Prefill and decode alternate between a large and a tiny payload on the very
// same collective, which re-records the command lists each time and grows the
// arena on the first large one only.
TEST_F(TPCollectivesTest, AlternatesPayloadSizesInOneSlot) {
    Watchdog watchdog(120);
    const int ranks = std::min(available_gpus, 4);
    auto shared = make_shared_ctx(ranks);
    auto coord = std::make_shared<Collectives>(shared, ranks, 1);

    constexpr size_t large = 16 * 1024;
    constexpr size_t small = 64;
    CollectiveBuffers buffers(shared, ranks, large * sizeof(ov::float16));

    for (int cycle = 0; cycle < 5; ++cycle) {
        SCOPED_TRACE("cycle " + std::to_string(cycle));
        expect_allreduce_sum<ov::float16>(*coord, buffers, 0, large, ov::element::f16, f16_tolerance);
        expect_allreduce_sum<ov::float16>(*coord, buffers, 0, small, ov::element::f16, f16_tolerance);
    }

    // The arena is sized by the largest payload ever seen and never shrinks, so
    // the small half of every cycle must not reallocate.
    EXPECT_EQ(coord->get_scratch_stats().growth_count, 1u)
        << "only the first large payload should have grown the arena";
    EXPECT_FALSE(coord->is_aborted());
}

// The element type is part of the recorded signature -- it picks the kernel and
// the byte stride -- so switching it has to re-record rather than reinterpret
// the buffer.
TEST_F(TPCollectivesTest, SwitchesDtypeInOneSlot) {
    Watchdog watchdog(120);
    const int ranks = std::min(available_gpus, 4);
    auto shared = make_shared_ctx(ranks);
    auto coord = std::make_shared<Collectives>(shared, ranks, 1);

    constexpr size_t n = 4096;
    CollectiveBuffers buffers(shared, ranks, n * sizeof(float));
    expect_allreduce_sum<ov::float16>(*coord, buffers, 0, n, ov::element::f16, f16_tolerance);
    expect_allreduce_sum<float>(*coord, buffers, 0, n, ov::element::f32, f32_tolerance);
    expect_allreduce_sum<ov::bfloat16>(*coord, buffers, 0, n, ov::element::bf16, f16_tolerance);
    expect_allreduce_sum<ov::float16>(*coord, buffers, 0, n, ov::element::f16, f16_tolerance);
    EXPECT_FALSE(coord->is_aborted());
}

// ---------------------------------------------------------------------------
// Op contract: the sum is accumulated in f32 and rounded once per reduction
// step, and every rank ends up with the same bits.
// ---------------------------------------------------------------------------

/// f32 -> bf16 bits, round to nearest even; the inputs here are finite.
uint16_t bf16_bits_rte(float value) {
    uint32_t u = 0;
    std::memcpy(&u, &value, sizeof(u));
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

uint16_t bits_of(ov::float16 v) {
    return v.to_bits();
}
uint16_t bits_of(ov::bfloat16 v) {
    return v.to_bits();
}
uint32_t bits_of(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, sizeof(u));
    return u;
}

/// Random operands in [0.5, 4): normal in every type, and their sums too, so
/// each step is one ordinary rounding.
template <typename T>
std::vector<T> random_operands(size_t n, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.5f, 4.0f);
    std::vector<T> v(n);
    for (auto& x : v)
        x = static_cast<T>(dist(gen));
    return v;
}

/// Two ranks reduce in one step, so the result has to be exactly the f32 sum
/// of the operands rounded once to nearest even.
template <typename T>
void expect_single_rounding(Collectives& coord, CollectiveBuffers& buffers, ov::element::Type dtype) {
    constexpr size_t n = 4099;  // a scalar tail, and an odd byte offset for the second staging set
    const size_t bytes = n * sizeof(T);
    const auto a = random_operands<T>(n, 1);
    const auto b = random_operands<T>(n, 2);
    buffers.rs[0].copy_from_host(buffers.ins[0], a.data(), bytes);
    buffers.rs[1].copy_from_host(buffers.ins[1], b.data(), bytes);

    run_collective(coord, 0, buffers.ins, buffers.outs, n, dtype);

    for (int r = 0; r < 2; ++r) {
        std::vector<T> got(n);
        buffers.rs[r].copy_to_host(got.data(), buffers.outs[r], bytes);
        size_t wrong = 0;
        for (size_t i = 0; i < n; ++i) {
            const float sum = static_cast<float>(a[i]) + static_cast<float>(b[i]);
            T want;
            if constexpr (std::is_same_v<T, ov::bfloat16>) {
                want = ov::bfloat16::from_bits(bf16_bits_rte(sum));
            } else {
                want = static_cast<T>(sum);
            }
            wrong += bits_of(got[i]) != bits_of(want);
        }
        EXPECT_EQ(wrong, 0u) << dtype << " rank " << r;
    }
}

TEST_F(TPCollectivesTest, ReductionRoundsOnceToNearestEven) {
    Watchdog watchdog(60);
    auto shared = make_shared_ctx(2);
    auto coord = std::make_shared<Collectives>(shared, 2, 1);
    CollectiveBuffers buffers(shared, 2, 4099 * sizeof(float));

    expect_single_rounding<ov::float16>(*coord, buffers, ov::element::f16);
    expect_single_rounding<ov::bfloat16>(*coord, buffers, ov::element::bf16);
    expect_single_rounding<float>(*coord, buffers, ov::element::f32);
    EXPECT_FALSE(coord->is_aborted());
}

/// Every rank's output carries exactly the same bits, whatever the schedule.
template <typename T>
void expect_ranks_agree(Collectives& coord, CollectiveBuffers& buffers, size_t n, ov::element::Type dtype) {
    const int ranks = static_cast<int>(buffers.rs.size());
    const size_t bytes = n * sizeof(T);
    for (int r = 0; r < ranks; ++r) {
        const auto host = random_operands<T>(n, static_cast<uint32_t>(10 + r));
        buffers.rs[r].copy_from_host(buffers.ins[r], host.data(), bytes);
    }

    run_collective(coord, 0, buffers.ins, buffers.outs, n, dtype);

    std::vector<T> reference(n);
    buffers.rs[0].copy_to_host(reference.data(), buffers.outs[0], bytes);
    for (int r = 1; r < ranks; ++r) {
        std::vector<T> got(n);
        buffers.rs[r].copy_to_host(got.data(), buffers.outs[r], bytes);
        EXPECT_EQ(std::memcmp(got.data(), reference.data(), bytes), 0)
            << dtype << " n=" << n << " rank " << r << " differs from rank 0";
    }
}

TEST_F(TPCollectivesTest, RanksAgreeBitwise) {
    Watchdog watchdog(180);

    for (int ranks : {2, 3, 4}) {
        if (available_gpus < ranks)
            continue;
        SCOPED_TRACE("ranks " + std::to_string(ranks));
        auto shared = make_shared_ctx(ranks);

        for (bool halving : {true, false}) {
            SCOPED_TRACE(halving ? "halving allowed" : "ring only");
            CollectiveOptions config;
            config.enable_halving = halving;
            auto coord =
                std::make_shared<Collectives>(shared, ranks, 1, std::chrono::milliseconds{5000}, config);
            constexpr size_t n = 8 * 1024 + 5;
            CollectiveBuffers buffers(shared, ranks, n * sizeof(float));
            expect_ranks_agree<ov::float16>(*coord, buffers, n, ov::element::f16);
            expect_ranks_agree<ov::bfloat16>(*coord, buffers, n, ov::element::bf16);
            expect_ranks_agree<float>(*coord, buffers, n, ov::element::f32);
            EXPECT_FALSE(coord->is_aborted());
        }
    }
}

// ---------------------------------------------------------------------------
// Stale recordings.
//
// A plan is identified by the addresses it was recorded from and nothing else.
// intel_gpu frees and reallocates network buffers between inferences, so an
// allocation can land on an address a recording still refers to: matches()
// reports a hit, the recording is reused, and its residency list points at a
// GraphicsAllocation the driver has destroyed.  The next submit walks that
// list.
//
// In a model this needs the allocator to hand back the same address, which is
// why it shows up as roughly one crash in fifty runs.  Here it is forced.
// ---------------------------------------------------------------------------

TEST_F(TPCollectivesTest, SurvivesABufferAddressBeingReused) {
    Watchdog watchdog(120);
    const int ranks = std::min(available_gpus, 4);
    auto shared = make_shared_ctx(ranks);
    auto coord = std::make_shared<Collectives>(shared, ranks, 1);

    constexpr size_t n = 4096;
    const size_t bytes = n * sizeof(ov::float16);

    // Same size and same collective every round, so whatever the allocator
    // returns after the free has a good chance of being the address the
    // previous recording was built from.
    std::vector<void*> seen;
    for (int round = 0; round < 16; ++round) {
        SCOPED_TRACE("round " + std::to_string(round));
        auto buffers = std::make_unique<CollectiveBuffers>(shared, ranks, bytes);
        seen.push_back(buffers->ins[0]);
        expect_allreduce_sum<ov::float16>(*coord, *buffers, 0, n, ov::element::f16, f16_tolerance);
        buffers.reset();  // frees the device USM the recording still names
    }

    // Report whether the reuse actually happened: without it the case proves
    // nothing, and that is worth knowing when it passes.
    std::vector<void*> distinct = seen;
    std::sort(distinct.begin(), distinct.end());
    const bool reused = std::unique(distinct.begin(), distinct.end()) != distinct.end();
    EXPECT_TRUE(reused) << "the allocator never returned a previous address; "
                           "this run did not exercise the stale-recording path";
    EXPECT_FALSE(coord->is_aborted());
}

// The same hazard as above, driven the way a model drives it: several
// collectives per step, a large prefill followed by short decodes, and the
// buffers reallocated whenever the prompt length changes.  That last part is
// what makes the crash rare in practice -- the allocator has to hand an old
// address back while a recording still names it -- so the loop runs long
// enough to give it many chances.  TP_STRESS_ROUNDS raises the count for
// hunting, the default keeps the case affordable in CI.
TEST_F(TPCollectivesTest, SurvivesAModelShapedAllocationChurn) {
    const int rounds = [] {
        const char* env = std::getenv("TP_STRESS_ROUNDS");
        return env ? std::max(1, std::atoi(env)) : 40;
    }();
    Watchdog watchdog(60 + rounds * 8);
    const int ranks = std::min(available_gpus, 4);
    auto shared = make_shared_ctx(ranks);

    // More than one collective, because a plan holds its own command lists
    // and a stale one only hurts when something else ran in between.
    constexpr int collectives = 8;
    auto coord = std::make_shared<Collectives>(shared, ranks, collectives);

    // Stand-ins for the prompt lengths the crash was seen with.
    const std::vector<size_t> prompt_elems{32, 128, 1024, 8192, 32768};
    constexpr size_t decode_elems = 128;

    for (int round = 0; round < rounds; ++round) {
        SCOPED_TRACE("round " + std::to_string(round));
        const size_t prefill = prompt_elems[static_cast<size_t>(round) % prompt_elems.size()];

        // Prefill: one pass over every collective at the prompt size.
        {
            auto buffers = std::make_unique<CollectiveBuffers>(shared, ranks,
                                                               prefill * sizeof(ov::float16));
            for (int cid = 0; cid < collectives; ++cid) {
                expect_allreduce_sum<ov::float16>(*coord, *buffers, cid, prefill,
                                                  ov::element::f16, f16_tolerance);
            }
        }

        // Decode: a different size on the same slots, with the buffers
        // reallocated each step so addresses keep being recycled.
        for (int step = 0; step < 4; ++step) {
            auto buffers = std::make_unique<CollectiveBuffers>(shared, ranks,
                                                               decode_elems * sizeof(ov::float16));
            for (int cid = 0; cid < collectives; ++cid) {
                expect_allreduce_sum<ov::float16>(*coord, *buffers, cid, decode_elems,
                                                  ov::element::f16, f16_tolerance);
            }
        }

        ASSERT_FALSE(coord->is_aborted()) << "group aborted in round " << round;
    }
}

// ---------------------------------------------------------------------------
// Gather shapes.
// ---------------------------------------------------------------------------

// A single row makes the region copy degenerate to one contiguous run, and a
// slice that is not a multiple of anything keeps the destination pitch from
// accidentally lining up.
TEST_F(TPCollectivesTest, GatherHandlesASingleRow) {
    Watchdog watchdog(60);
    const int ranks = std::min(available_gpus, 4);
    auto shared = make_shared_ctx(ranks);
    auto coord = std::make_shared<Collectives>(shared, ranks, 1);

    constexpr size_t rows = 1;
    constexpr size_t slice = 7;
    const size_t full = slice * static_cast<size_t>(ranks);

    std::vector<RankScratch> rs(ranks);
    std::vector<void*> ins(ranks), outs(ranks);
    for (int r = 0; r < ranks; ++r) {
        rs[r].init(shared->context, shared->devices[r]);
        ins[r] = rs[r].alloc(rows * slice * sizeof(ov::float16));
        outs[r] = rs[r].alloc(rows * full * sizeof(ov::float16));
        std::vector<ov::float16> host(rows * slice);
        for (size_t x = 0; x < slice; ++x)
            host[x] = ov::float16(static_cast<float>(r * 16 + static_cast<int>(x)));
        rs[r].copy_from_host(ins[r], host.data(), host.size() * sizeof(ov::float16));
    }

    std::vector<ov::float16> poison(rows * full, ov::float16(-1.0f));
    rs[0].copy_from_host(outs[0], poison.data(), poison.size() * sizeof(ov::float16));

    run_gather(*coord, 0, ins, outs, rows, slice, ov::element::f16);

    std::vector<ov::float16> got(rows * full);
    rs[0].copy_to_host(got.data(), outs[0], got.size() * sizeof(ov::float16));
    for (int r = 0; r < ranks; ++r) {
        for (size_t x = 0; x < slice; ++x) {
            EXPECT_NEAR(static_cast<float>(got[r * slice + x]),
                        static_cast<float>(r * 16 + static_cast<int>(x)),
                        0.5f)
                << "rank " << r << " column " << x;
        }
    }

    for (int r = 0; r < ranks; ++r) {
        ov::zeMemFree(shared->context, ins[r]);
        ov::zeMemFree(shared->context, outs[r]);
        rs[r].destroy();
    }
}

// ---------------------------------------------------------------------------
// Execution paths.
//
// Everything above hands the collective to the group's own queues and
// waits for them.  That is the fallback.  What a model actually runs is the
// spliced path: the recording goes into the in-order list intel_gpu is already
// dispatching on, and the call returns without waiting.  Until now nothing
// below the plugin exercised it.
// ---------------------------------------------------------------------------

/// Same contributions and checks as expect_allreduce_sum, but the collectives
/// ride the per-rank immediate lists and the host waits for those instead.
template <typename T>
void expect_spliced_allreduce_sum(Collectives& coord,
                                  CollectiveBuffers& buffers,
                                  int collective_id,
                                  size_t n,
                                  ov::element::Type dtype,
                                  float tolerance) {
    const int ranks = static_cast<int>(buffers.rs.size());
    const size_t bytes = n * sizeof(T);
    std::vector<ze_command_list_handle_t> queues(ranks);
    float expected = 0.0f;
    for (int r = 0; r < ranks; ++r) {
        const float value = static_cast<float>(r + 1);
        expected += value;
        std::vector<T> host(n, static_cast<T>(value));
        buffers.rs[r].copy_from_host(buffers.ins[r], host.data(), bytes);
        queues[r] = buffers.rs[r].model_queue();
    }

    run_collective(coord, collective_id, buffers.ins, buffers.outs, n, dtype, queues);

    // The call returned before the devices were done, which is the point of
    // splicing; the results are only there once the model queues drain.
    for (int r = 0; r < ranks; ++r)
        buffers.rs[r].wait_for_model_queue();

    for (int r = 0; r < ranks; ++r) {
        std::vector<T> host(n);
        buffers.rs[r].copy_to_host(host.data(), buffers.outs[r], bytes);
        EXPECT_EQ(count_mismatches(host, expected, tolerance), 0u)
            << "spliced n=" << n << " rank " << r;
    }
}

TEST_F(TPCollectivesTest, SplicedAllReduceMatchesTheBlockingPath) {
    Watchdog watchdog(120);
    const int ranks = std::min(available_gpus, 4);
    auto shared = make_shared_ctx(ranks);
    auto coord = std::make_shared<Collectives>(shared, ranks, 1);
    if (!coord->run_spliced())
        GTEST_SKIP() << "driver has no immediate-append extension";

    constexpr size_t n = 8 * 1024;
    CollectiveBuffers buffers(shared, ranks, n * sizeof(ov::float16));

    // Several instances of the same collective in a row: the recording is
    // handed over while the previous one may still be in flight, and the
    // completion event of that previous splice has to be consumed before the
    // list can be appended again.  One call would not reach that path.
    for (int k = 0; k < 6; ++k) {
        SCOPED_TRACE("instance " + std::to_string(k));
        expect_spliced_allreduce_sum<ov::float16>(*coord, buffers, 0, n, ov::element::f16, f16_tolerance);
    }
    EXPECT_FALSE(coord->is_aborted());
}

// force_sync_collective exists so a suspected splice problem can be ruled out
// without rebuilding.  It has to be an equivalence, not merely a slower path.
TEST_F(TPCollectivesTest, ForcingTheSyncPathKeepsTheSameResult) {
    Watchdog watchdog(120);
    const int ranks = std::min(available_gpus, 4);
    auto shared = make_shared_ctx(ranks);

    CollectiveOptions config;
    config.force_sync_collective = true;
    auto coord =
        std::make_shared<Collectives>(shared, ranks, 1, std::chrono::milliseconds{5000}, config);
    EXPECT_FALSE(coord->run_spliced()) << "the option must win over the driver's capability";

    constexpr size_t n = 8 * 1024;
    CollectiveBuffers buffers(shared, ranks, n * sizeof(ov::float16));

    // A model queue is passed anyway.  The recorder is what decides whether
    // to use it, and with the option set it must not -- otherwise a caller
    // that ignored run_spliced() would silently keep splicing.
    std::vector<ze_command_list_handle_t> queues(ranks);
    for (int r = 0; r < ranks; ++r)
        queues[r] = buffers.rs[r].model_queue();

    float expected = 0.0f;
    for (int r = 0; r < ranks; ++r) {
        const float value = static_cast<float>(r + 1);
        expected += value;
        std::vector<ov::float16> host(n, ov::float16(value));
        buffers.rs[r].copy_from_host(buffers.ins[r], host.data(), n * sizeof(ov::float16));
    }

    run_collective(*coord, 0, buffers.ins, buffers.outs, n, ov::element::f16, queues);

    // No wait on the model queues: the blocking path must have finished the
    // work before it returned.
    for (int r = 0; r < ranks; ++r) {
        std::vector<ov::float16> host(n);
        buffers.rs[r].copy_to_host(host.data(), buffers.outs[r], n * sizeof(ov::float16));
        EXPECT_EQ(count_mismatches(host, expected, f16_tolerance), 0u) << "rank " << r;
    }
    EXPECT_FALSE(coord->is_aborted());
}

// A group that fails while work is still outstanding on the devices must still
// surface as an exception on every later call rather than as a hang.
TEST_F(TPCollectivesTest, AbortAfterSplicingFailsSubsequentCalls) {
    Watchdog watchdog(60);
    const int ranks = 2;
    auto shared = make_shared_ctx(ranks);
    auto coord = std::make_shared<Collectives>(shared, ranks, 1);
    if (!coord->run_spliced())
        GTEST_SKIP() << "driver has no immediate-append extension";

    constexpr size_t n = 4096;
    CollectiveBuffers buffers(shared, ranks, n * sizeof(ov::float16));
    expect_spliced_allreduce_sum<ov::float16>(*coord, buffers, 0, n, ov::element::f16, f16_tolerance);

    coord->abort_all("test-initiated abort after a splice");
    EXPECT_TRUE(coord->is_aborted());

    // Immediately, not after the timeout: the group is single-use once
    // it has failed.
    const auto before = std::chrono::steady_clock::now();
    EXPECT_THROW(coord->allreduce(0, 0, buffers.ins[0], buffers.outs[0], n, ov::element::f16,
                                  reduce_kernel(*coord, 0, ov::element::f16)),
                 ov::Exception);
    EXPECT_LT(std::chrono::steady_clock::now() - before, std::chrono::seconds{1});
}

}  // namespace

#endif  // ENABLE_TP

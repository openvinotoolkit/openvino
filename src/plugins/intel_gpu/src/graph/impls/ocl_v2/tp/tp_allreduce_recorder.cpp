// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP

#include "tp_allreduce_recorder.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <string>

#include "openvino/core/except.hpp"

#define ZE_THROW(expr) OV_ZE_EXPECT(expr)

namespace ov::intel_gpu::ocl {

using cldnn::ze::tp::alloc_id_of;
using cldnn::ze::tp::Counter;
using cldnn::ze::tp::ScopedTime;
using cldnn::ze::tp::ze_api;

namespace {

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs, const char* what) {
    OPENVINO_ASSERT(rhs == 0 || lhs <= std::numeric_limits<std::size_t>::max() / rhs,
                    "[TP][L0] ",
                    what,
                    " byte count overflow: ",
                    lhs,
                    " * ",
                    rhs);
    return lhs * rhs;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Distance between the two pair staging sets: the reduce kernel's vector loads
// need every set to start aligned, whatever the payload size.
std::size_t pair_stride(std::size_t payload_bytes) {
    return align_up(payload_bytes, 256);
}

// Element granularity of a ring chunk: 256 bytes for f16, 512 for f32, which
// covers any vector width the reduce kernel may use.
constexpr std::size_t kRingAlignElems = 128;

// How much a halving step may exceed half of the range it splits: halving_mid
// snaps the boundary down to kRingAlignElems, so the sent half can be longer
// than half by almost a full unit. Without the slack a transfer ran into the
// next step's region (1021 f32 on four ranks lost its last 125 elements).
constexpr std::size_t kHalvingSlackBytes = kRingAlignElems * 4;

std::size_t collective_payload_bytes(std::size_t n, ov::element::Type dtype) {
    OPENVINO_ASSERT(dtype == ov::element::f16 || dtype == ov::element::bf16 || dtype == ov::element::f32,
                    "[TP][L0] AllReduce supports f16/bf16/f32 only, got ",
                    dtype);
    OPENVINO_ASSERT(n > 0, "[TP][L0] zero-sized AllReduce is not supported");
    OPENVINO_ASSERT(n <= std::numeric_limits<uint32_t>::max(),
                    "[TP][L0] AllReduce element count exceeds Level Zero launch range: ",
                    n);
    return checked_multiply(n, dtype.size(), "collective payload");
}

// Launches the model's reduction kernel (ocl_v2/tp_allreduce.cl):
// dst = src0 + src1 over `count` elements, 8 elements per work item.
void append_reduce(ze_command_list_handle_t list,
                   ze_kernel_handle_t kernel,
                   void* dst,
                   void* src0,
                   void* src1,
                   std::size_t count) {
    constexpr std::size_t elems_per_item = 8;
    constexpr uint32_t group_size = 256;
    const uint64_t n = count;
    const std::size_t items = (count + elems_per_item - 1) / elems_per_item;
    const ze_group_count_t groups{static_cast<uint32_t>((items + group_size - 1) / group_size), 1, 1};
    ZE_THROW(ze_api()->zeKernelSetGroupSize(kernel, group_size, 1, 1));
    ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 0, sizeof(dst), &dst));
    ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 1, sizeof(src0), &src0));
    ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 2, sizeof(src1), &src1));
    ZE_THROW(ze_api()->zeKernelSetArgumentValue(kernel, 3, sizeof(n), &n));
    ZE_THROW(ze_api()->zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0, nullptr));
}

// Where the payload splits at one level of the recursion. Both partners compute
// it from the same range, so they agree on who keeps which half; snapped down
// to the kernel's alignment, like ring chunks.
std::size_t halving_mid(std::size_t lo, std::size_t hi) {
    if (hi <= lo) {
        return lo;
    }
    std::size_t mid = lo + (hi - lo) / 2;
    mid = (mid / kRingAlignElems) * kRingAlignElems;
    return std::min(std::max(mid, lo), hi);
}

int halving_levels(int world_size) {
    int levels = 0;
    while ((1 << levels) < world_size) {
        ++levels;
    }
    return levels;
}

// Bytes of one halving staging region at `level`: half of what `level` halvings
// left, plus the slack halving_mid may add at every split.
std::size_t halving_region_bytes(std::size_t payload_bytes, int level) {
    std::size_t span = payload_bytes;
    for (int j = 0; j < level; ++j) {
        span = (span + 1) / 2 + kHalvingSlackBytes;
    }
    return align_up((span + 1) / 2 + kHalvingSlackBytes, 256);
}

// Duration between two kernel timestamps of one device, in nanoseconds.
uint64_t ticks_to_ns(uint64_t start, uint64_t end, uint64_t mask, uint64_t ns_per_tick) {
    const uint64_t s = start & mask;
    const uint64_t e = end & mask;
    const uint64_t delta = (e >= s) ? (e - s) : ((mask + 1 - s) + e);
    return delta * ns_per_tick;
}

void barrier(ze_command_list_handle_t list) {
    ZE_THROW(ze_api()->zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
}

void* byte_offset(void* base, std::size_t bytes) {
    return static_cast<uint8_t*>(base) + bytes;
}

}  // namespace

bool TPAllReduceRecorder::uses_halving(std::size_t payload_bytes) const {
    const auto& options = m_rec.group().options();
    const bool power_of_two = (m_world_size & (m_world_size - 1)) == 0;
    return m_world_size > 2 && power_of_two && options.enable_halving && payload_bytes <= options.halving_max_bytes;
}

std::size_t TPAllReduceRecorder::staging_bytes(std::size_t payload_bytes, bool halving) const {
    if (!halving) {
        return payload_bytes;
    }
    const int levels = halving_levels(m_world_size);
    std::size_t total = 0;
    for (int k = 0; k < 2 * levels; ++k) {
        total += halving_region_bytes(payload_bytes, k % levels);
    }
    return std::max(total, payload_bytes);
}

std::size_t TPAllReduceRecorder::chunk_stride(std::size_t payload_bytes, bool halving) const {
    // The largest chunk rounded up for alignment, with the slack halving_mid
    // may add; halving lays its regions out inside the same N slots.
    const std::size_t need = staging_bytes(payload_bytes, halving);
    const auto N = static_cast<std::size_t>(m_world_size);
    return align_up((need + N - 1) / N + kHalvingSlackBytes, 256);
}

std::size_t TPAllReduceRecorder::half_bytes(std::size_t payload_bytes, bool halving) const {
    return m_world_size > 2 ? checked_multiply(chunk_stride(payload_bytes, halving),
                                               static_cast<std::size_t>(m_world_size),
                                               "ring staging")
                            : pair_stride(payload_bytes);
}

void TPAllReduceRecorder::ring_chunk(std::size_t n, int chunk, std::size_t& offset, std::size_t& count) const {
    // Chunk boundaries are aligned, not merely even: the kernel's vload8/vstore8
    // expect aligned bases, and a misaligned chunk corrupted a handful of
    // elements at its head. Trailing chunks may come out empty.
    const auto N = static_cast<std::size_t>(m_world_size);
    const auto c = static_cast<std::size_t>(chunk);
    const std::size_t units = (n + kRingAlignElems - 1) / kRingAlignElems;
    const std::size_t base = units / N;
    const std::size_t rem = units % N;
    const std::size_t unit_off = c * base + std::min(c, rem);
    const std::size_t unit_cnt = base + (c < rem ? 1u : 0u);
    offset = unit_off * kRingAlignElems;
    count = (offset >= n) ? 0u : std::min(unit_cnt * kRingAlignElems, n - offset);
}

void* TPAllReduceRecorder::ring_slot(const Signature& sig, const Piece& piece, int rank, int chunk) const {
    OPENVINO_ASSERT(chunk >= 0 && chunk < m_world_size, "[TP][L0] ring slot chunk out of range: ", chunk);
    return byte_offset(sig.staging[piece.parity][static_cast<std::size_t>(rank)],
                       static_cast<std::size_t>(chunk) * chunk_stride(piece.n * sig.dtype.size(), false));
}

void TPAllReduceRecorder::halving_range(std::size_t n, int rank, int level, std::size_t& lo, std::size_t& hi) const {
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

void* TPAllReduceRecorder::halving_stage(const Signature& sig, const Piece& piece, int rank, int region) const {
    // Regions [0, levels) take the scatter steps, [levels, 2 * levels) the
    // gather steps: a partner may deliver the next step while this rank still
    // reads the last.
    const int levels = halving_levels(m_world_size);
    OPENVINO_ASSERT(region >= 0 && region < 2 * levels, "[TP][L0] halving staging region out of range: ", region);
    const std::size_t payload_bytes = piece.n * sig.dtype.size();
    std::size_t offset = 0;
    for (int k = 0; k < region; ++k) {
        offset += halving_region_bytes(payload_bytes, k % levels);
    }
    OPENVINO_ASSERT(offset + halving_region_bytes(payload_bytes, region % levels) <= half_bytes(payload_bytes, true),
                    "[TP][L0] halving staging does not fit the staging half");
    return byte_offset(sig.staging[piece.parity][static_cast<std::size_t>(rank)], offset);
}

TPAllReduceRecorder::ChunkPlan TPAllReduceRecorder::plan_chunks(std::size_t n, ov::element::Type dtype) {
    const std::size_t elem = dtype.size();
    const std::size_t payload = n * elem;
    const std::size_t limit = m_rec.group().staging_half_limit();
    ChunkPlan plan;
    plan.chunk_elems = n;
    plan.halving = uses_halving(payload);
    if (limit == 0 || half_bytes(payload, plan.halving) <= limit) {
        return plan;
    }
    plan.halving = false;
    if (half_bytes(payload, false) <= limit) {
        return plan;
    }

    // The largest chunk the staging holds, once per element size. Chunks of a
    // split payload always take the ring or the pair exchange: their staging
    // grows with the chunk, so a shorter last chunk fits as well.
    std::size_t& max_elems = m_max_chunk_elems[elem == 2 ? 0 : 1];
    if (max_elems == 0) {
        std::size_t c = limit / elem / kRingAlignElems * kRingAlignElems;
        while (c > 0) {
            const std::size_t need = half_bytes(c * elem, false);
            if (need <= limit) {
                break;
            }
            c -= std::min(c, align_up((need - limit + elem - 1) / elem, kRingAlignElems));
        }
        OPENVINO_ASSERT(c > 0,
                        "[TP][L0] the fixed staging of ",
                        m_rec.group().fixed_staging_bytes(),
                        " bytes cannot hold a chunk of ",
                        kRingAlignElems,
                        " ",
                        dtype,
                        " elements on ",
                        m_world_size,
                        " ranks");
        max_elems = c;
    }
    plan.count = (n + max_elems - 1) / max_elems;
    plan.chunk_elems = std::min(max_elems, align_up((n + plan.count - 1) / plan.count, kRingAlignElems));
    return plan;
}

void TPAllReduceRecorder::record(int parity) {
    auto& group = m_rec.group();
    auto& st = group.stats();
    const int rank = m_rec.rank();
    const bool measure = group.options().profiling_host;
    const Signature& sig = m_sig[parity];
    m_rec.reset(parity, true);
    ze_command_list_handle_t list = m_rec.list(parity);

    Counter* chosen = nullptr;
    {
        ScopedTime build(measure, st.rec_build_ns);
        const std::size_t elem = sig.dtype.size();
        // The first chunk has the call's parity, the others alternate like
        // the calls they stand for.
        int chunk_parity = parity;
        for (std::size_t off = 0; off < sig.n; off += sig.chunk_elems) {
            Piece piece;
            piece.in = byte_offset(sig.in, off * elem);
            piece.out = byte_offset(sig.out, off * elem);
            piece.n = std::min(sig.chunk_elems, sig.n - off);
            piece.parity = chunk_parity;
            piece.halving = sig.halving;
            chunk_parity ^= 1;
            if (off != 0) {
                // A chunk starts once the previous one, resets included, is
                // done, as a call does after the call before it.
                barrier(list);
            }
            if (m_world_size == 2) {
                chosen = &st.n_pair[rank];
                record_pair(list, sig, piece);
            } else if (piece.halving) {
                // Both move 1.5*S per rank, but halving has both partners of a pair
                // pushing across one link at once: free at decode sizes, where the
                // saved round trips dominate, a loss at prompt sizes.
                chosen = &st.n_halving[rank];
                record_halving(list, sig, piece);
            } else {
                chosen = &st.n_ring[rank];
                record_ring(list, sig, piece);
            }
        }
        ScopedTime close(measure, st.rec_close_ns);
        ZE_THROW(ze_api()->zeCommandListClose(list));
    }
    if (measure) {
        chosen->bump();
    }
}

void TPAllReduceRecorder::record_pair(ze_command_list_handle_t list, const Signature& sig, const Piece& piece) {
    auto& ev_recv = *m_call.recv[piece.parity];
    const auto& staging = sig.staging[piece.parity];
    const int r = m_rec.rank();
    const int peer = 1 - r;
    const std::size_t bytes = piece.n * sig.dtype.size();

    // Push our input into the peer's staging, never into the peer's own buffer,
    // which is what lets each rank record on its own. ev_recv[r] tells the peer.
    ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                     staging[static_cast<std::size_t>(peer)],
                                                     piece.in,
                                                     bytes,
                                                     ev_recv[r],
                                                     0,
                                                     nullptr));
    ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ev_recv[peer]));
    // This rank is the only waiter; the wait already orders what follows, so
    // the reset needs no barrier.
    ZE_THROW(ze_api()->zeCommandListAppendEventReset(list, ev_recv[peer]));
    append_reduce(list, sig.kernel, piece.out, piece.in, staging[static_cast<std::size_t>(r)], piece.n);
}

void TPAllReduceRecorder::record_ring(ze_command_list_handle_t list, const Signature& sig, const Piece& piece) {
    auto& ev_ring = *m_call.step[piece.parity];
    const int N = m_world_size;
    const int steps = N - 1;
    const int r = m_rec.rank();
    const std::size_t n = piece.n;
    const std::size_t elem = sig.dtype.size();

    auto ev = [&](int step, int who) -> ze_event_handle_t& {
        return ev_ring[static_cast<std::size_t>(step) * static_cast<std::size_t>(N) + static_cast<std::size_t>(who)];
    };
    auto byte_at = [elem](void* base, std::size_t offset_elems) -> void* {
        return static_cast<uint8_t*>(base) + offset_elems * elem;
    };

    const int next = (r + 1) % N;
    const int prev = (r + N - 1) % N;

    // Reduce-scatter: at step s rank r forwards chunk (r-s) and folds the
    // incoming chunk (r-s-1), which already carries the partial sum of every
    // rank before it, into its own input.
    for (int s = 0; s < steps; ++s) {
        const int send_chunk = ((r - s) % N + N) % N;
        const int recv_chunk = ((r - s - 1) % N + N) % N;
        std::size_t send_off = 0, send_cnt = 0, recv_off = 0, recv_cnt = 0;
        ring_chunk(n, send_chunk, send_off, send_cnt);
        ring_chunk(n, recv_chunk, recv_off, recv_cnt);

        // An empty chunk still has to signal, or the successor waits forever.
        if (send_cnt == 0) {
            ZE_THROW(ze_api()->zeCommandListAppendSignalEvent(list, ev(s, r)));
        } else {
            void* send_src = (s == 0) ? byte_at(piece.in, send_off) : byte_at(piece.out, send_off);
            ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                             ring_slot(sig, piece, next, send_chunk),
                                                             send_src,
                                                             send_cnt * elem,
                                                             ev(s, r),
                                                             0,
                                                             nullptr));
        }
        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ev(s, prev)));
        if (recv_cnt == 0) {
            continue;
        }
        append_reduce(list,
                      sig.kernel,
                      byte_at(piece.out, recv_off),
                      byte_at(piece.in, recv_off),
                      ring_slot(sig, piece, r, recv_chunk),
                      recv_cnt);
        // The next send reads what this kernel wrote.
        barrier(list);
    }

    // All-gather: finished chunks travel through the successor's staging, and
    // each rank copies what arrives into its own output -- writing into a
    // peer's output would bake the peer's address into this recording. The
    // successor may still be in its reduce-scatter, writing the slots we are
    // about to deliver into, so one handshake per rank goes first.
    ZE_THROW(ze_api()->zeCommandListAppendSignalEvent(list, ev(2 * steps, r)));
    ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ev(2 * steps, next)));

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
            void* send_src = (s == 0) ? byte_at(piece.out, send_off) : ring_slot(sig, piece, r, send_chunk);
            ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                             ring_slot(sig, piece, next, send_chunk),
                                                             send_src,
                                                             send_cnt * elem,
                                                             ev(step, r),
                                                             0,
                                                             nullptr));
        }
        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ev(step, prev)));
        if (recv_cnt > 0) {
            ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                             byte_at(piece.out, recv_off),
                                                             ring_slot(sig, piece, r, recv_chunk),
                                                             recv_cnt * elem,
                                                             nullptr,
                                                             0,
                                                             nullptr));
        }
    }

    // Every event has exactly one waiter, which clears it: the predecessor's
    // step events and the successor's handshake.
    barrier(list);
    for (int s = 0; s < 2 * steps; ++s) {
        ZE_THROW(ze_api()->zeCommandListAppendEventReset(list, ev(s, prev)));
    }
    ZE_THROW(ze_api()->zeCommandListAppendEventReset(list, ev(2 * steps, next)));
}

void TPAllReduceRecorder::record_halving(ze_command_list_handle_t list, const Signature& sig, const Piece& piece) {
    auto& ev_ring = *m_call.step[piece.parity];
    const int N = m_world_size;
    const int r = m_rec.rank();
    const std::size_t n = piece.n;
    const std::size_t elem = sig.dtype.size();
    const int levels = halving_levels(N);
    OPENVINO_ASSERT((1 << levels) == N, "[TP][L0] halving requires a power-of-two world size");

    // Steps 0..2*levels-1: the first half scatters, the second gathers.
    // ev(step, who) is signalled by `who` and waited on by its partner.
    auto ev = [&](int step, int who) -> ze_event_handle_t& {
        return ev_ring[static_cast<std::size_t>(step) * static_cast<std::size_t>(N) + static_cast<std::size_t>(who)];
    };
    auto byte_at = [elem](void* base, std::size_t offset_elems) -> void* {
        return static_cast<uint8_t*>(base) + offset_elems * elem;
    };

    // Scatter: each step moves half of what is left.
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
        void* mine = (j == 0) ? piece.in : piece.out;

        if (send_hi > send_lo) {
            // At the start of the partner's region for this step, which is only
            // as large as one half and keeps the source aligned.
            ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                             halving_stage(sig, piece, partner, j),
                                                             byte_at(mine, send_lo),
                                                             (send_hi - send_lo) * elem,
                                                             ev(j, r),
                                                             0,
                                                             nullptr));
        } else {
            ZE_THROW(ze_api()->zeCommandListAppendSignalEvent(list, ev(j, r)));
        }
        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ev(j, partner)));
        if (keep_hi > keep_lo) {
            append_reduce(list,
                          sig.kernel,
                          byte_at(piece.out, keep_lo),
                          byte_at(mine, keep_lo),
                          halving_stage(sig, piece, r, j),
                          keep_hi - keep_lo);
        }
        barrier(list);
    }

    // Gather: the same partners in reverse, through the partner's staging. A
    // remote write into a peer's output lost against the peer's own kernels
    // now and then (about 1 decode run in 12 on four ranks).
    for (int j = levels - 1; j >= 0; --j) {
        const int partner = r ^ (1 << j);
        const int step = 2 * levels - 1 - j;
        const int region = levels + j;
        std::size_t lo = 0, hi = 0;
        halving_range(n, r, j + 1, lo, hi);
        if (hi > lo) {
            ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                             halving_stage(sig, piece, partner, region),
                                                             byte_at(piece.out, lo),
                                                             (hi - lo) * elem,
                                                             ev(step, r),
                                                             0,
                                                             nullptr));
        } else {
            ZE_THROW(ze_api()->zeCommandListAppendSignalEvent(list, ev(step, r)));
        }
        ZE_THROW(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ev(step, partner)));

        std::size_t partner_lo = 0, partner_hi = 0;
        halving_range(n, partner, j + 1, partner_lo, partner_hi);
        if (partner_hi > partner_lo) {
            ZE_THROW(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                             byte_at(piece.out, partner_lo),
                                                             halving_stage(sig, piece, r, region),
                                                             (partner_hi - partner_lo) * elem,
                                                             nullptr,
                                                             0,
                                                             nullptr));
        }
        barrier(list);
    }

    barrier(list);
    for (int step = 0; step < 2 * levels; ++step) {
        const int j = (step < levels) ? step : (2 * levels - 1 - step);
        ZE_THROW(ze_api()->zeCommandListAppendEventReset(list, ev(step, r ^ (1 << j))));
    }
}

void TPAllReduceRecorder::run(const cldnn::ze::tp::Channel& channel,
                              int collective_id,
                              void* in,
                              void* out,
                              std::size_t n,
                              ov::element::Type dtype,
                              ze_kernel_handle_t kernel,
                              ze_command_list_handle_t model_queue) {
    auto& group = *channel.group();
    const int rank = channel.rank();
    OPENVINO_ASSERT(collective_id >= 0 && collective_id < group.num_collectives(),
                    "[TP][L0] collective_id out of range: ",
                    collective_id);
    // A null device pointer reaches the driver as a request to wrap a host
    // allocation at address 0 and aborts the process.
    OPENVINO_ASSERT(in != nullptr && out != nullptr,
                    "[TP][L0] null device buffer passed to allreduce (collective ",
                    collective_id,
                    ", rank ",
                    rank,
                    ")");
    OPENVINO_ASSERT(kernel != nullptr,
                    "[TP][L0] allreduce called without a reduction kernel (collective ",
                    collective_id,
                    ", rank ",
                    rank,
                    ")");
    group.throw_if_aborted();

    // Timing-only mode: the difference against a normal run is the whole cost
    // of the collective. Outputs are left untouched.
    if (group.options().skip_collective) {
        static std::once_flag warned;
        std::call_once(warned, [] {
            TP_WARN_ALWAYS << "[TP] skip_collective is set: AllReduce is a no-op, outputs are invalid. "
                              "Timing-only mode.";
        });
        return;
    }

    collective_payload_bytes(n, dtype);
    m_world_size = channel.world_size();
    m_rec.bind(channel, collective_id, true);

    // A message larger than the fixed staging goes in chunks, all of them in
    // the one recording: no rendezvous, list or re-recording per chunk.
    const ChunkPlan plan = plan_chunks(n, dtype);
    const std::size_t payload_bytes = n * dtype.size();

    using clk = std::chrono::steady_clock;
    auto& st = group.stats();
    const bool measure_host = group.options().profiling_host;
    const bool measure_dev = group.options().profiling_device;
    auto stamp = [measure_host]() -> clk::time_point {
        return measure_host ? clk::now() : clk::time_point{};
    };
    auto elapsed_ns = [](clk::time_point a, clk::time_point b) -> uint64_t {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
    };

    const auto t0 = stamp();
    group.begin_allreduce(rank,
                          collective_id,
                          n,
                          dtype,
                          half_bytes(plan.chunk_elems * dtype.size(), plan.halving),
                          plan.count,
                          m_call);
    const auto t1 = stamp();
    const int parity = m_call.parity;
    if (measure_host && rank == 0 && plan.count > 1) {
        st.chunked_calls.bump();
        st.chunks.add(plan.count);
        st.max_chunks.keep_max(plan.count);
    }

    // Whoever fails leaves the rest of the group waiting for something that
    // will never arrive.
    struct AbortOnFailure {
        cldnn::ze::tp::GroupState& group;
        int collective_id;
        int rank;
        bool armed{true};
        ~AbortOnFailure() {
            if (armed) {
                group.abort_all("[TP][L0] rank " + std::to_string(rank) + " failed while running collective " +
                                std::to_string(collective_id));
            }
        }
    } abort_guard{group, collective_id, rank};

    Signature sig;
    sig.valid = true;
    sig.in = in;
    sig.out = out;
    sig.in_id = alloc_id_of(group.context(), in);
    sig.out_id = alloc_id_of(group.context(), out);
    sig.kernel = kernel;
    sig.n = n;
    sig.chunk_elems = plan.chunk_elems;
    sig.halving = plan.halving;
    sig.dtype = dtype;
    sig.staging[0] = m_call.staging[0];
    sig.staging[1] = m_call.staging[1];
    sig.staging_ids = m_call.staging_ids;

    // Runs only once the previous splice of the list has been consumed.
    auto record_if_stale = [&] {
        if (m_sig[parity] == sig) {
            return;
        }
        const auto rec0 = stamp();
        m_sig[parity] = std::move(sig);
        record(parity);
        if (measure_host) {
            const uint64_t ns = elapsed_ns(rec0, clk::now());
            st.p2_rec_ns[rank].add(ns);
            st.p2_rec_count[rank].bump();
            if (rank == 0) {
                st.records.bump();
                st.record_ns.add(ns);
            }
        }
    };

    auto te1 = t1;
    auto& splice = m_rec.splice(parity);
    if (model_queue != nullptr && channel.run_spliced()) {
        // The extension forbids re-appending a list that has not finished.
        // Two parities alternate, so this asks about work handed over two
        // collectives ago, which is normally long done.
        if (splice.in_flight.load(std::memory_order_acquire)) {
            const auto w0 = stamp();
            ze_result_t res = ze_api()->zeEventQueryStatus(splice.done);
            if (res == ZE_RESULT_NOT_READY) {
                if (measure_host) {
                    st.p2_block_count[rank].bump();
                }
                res = ze_api()->zeEventHostSynchronize(splice.done, group.timeout_ns());
                if (res == ZE_RESULT_NOT_READY) {
                    group.fail_collective(true, collective_id, rank, "previous splice of this recording");
                }
            }
            ZE_THROW(res);
            const auto w1 = stamp();
            if (measure_dev) {
                // The event still holds the timestamps of the splice that just
                // finished; the reset below clears them.
                ze_kernel_timestamp_result_t kt{};
                if (ze_api()->zeEventQueryKernelTimestamp(splice.done, &kt) == ZE_RESULT_SUCCESS) {
                    const uint64_t mask = group.timestamp_mask(rank);
                    const uint64_t tick = group.timer_ns_per_tick(rank);
                    const uint64_t ns = ticks_to_ns(kt.global.kernelStart, kt.global.kernelEnd, mask, tick);
                    st.p2_dev_ns[rank].add(ns);
                    if (ns > st.p2_dev_max_ns[rank].get()) {
                        st.p2_dev_max_ns[rank].keep_max(ns);
                        st.p2_dev_max_cid[rank].v.store(static_cast<uint64_t>(collective_id),
                                                        std::memory_order_relaxed);
                    }
                    st.p2_dev_count[rank].bump();

                    // Powers of two in microseconds, clamped into the end buckets.
                    const uint64_t us = ns / 1000u;
                    int bucket = 0;
                    for (uint64_t v = us; v > 1 && bucket < cldnn::ze::tp::CollectiveStats::kDevBuckets - 1;
                         v >>= 1) {
                        ++bucket;
                    }
                    st.p2_dev_hist[static_cast<std::size_t>(rank) *
                                       static_cast<std::size_t>(cldnn::ze::tp::CollectiveStats::kDevBuckets) +
                                   static_cast<std::size_t>(bucket)]
                        .bump();

                    // Device time between the previous collective and this one:
                    // the model's own work. A window starting before the previous
                    // one ended means out-of-order readings or a wrap; skipped.
                    const uint64_t prev_end = st.dev_last_end_ticks[rank] & mask;
                    const uint64_t cur_start = kt.global.kernelStart & mask;
                    if (st.dev_last_cid[rank] >= 0 && cur_start >= prev_end) {
                        st.dev_gap_ns[rank].add((cur_start - prev_end) * tick);
                        st.dev_gap_count[rank].bump();
                    }
                    st.dev_last_end_ticks[rank] = kt.global.kernelEnd;
                    st.dev_last_cid[rank] = collective_id;
                    st.dev_bytes[rank].add(splice.bytes);
                }
            }
            ZE_THROW(ze_api()->zeEventHostReset(splice.done));
            splice.in_flight.store(0, std::memory_order_release);
            if (measure_host) {
                st.ahead_now[rank].v.fetch_sub(1, std::memory_order_relaxed);
                const auto w2 = clk::now();
                st.p2_wait_ns[rank].add(elapsed_ns(w0, w1));
                st.p2_reset_ns[rank].add(elapsed_ns(w1, w2));
                st.p2_wait_count[rank].bump();
            }
        }

        record_if_stale();
        const auto a0 = stamp();
        m_rec.splice_into(model_queue, parity);
        if (measure_host) {
            st.p2_append_ns[rank].add(elapsed_ns(a0, clk::now()));
            const uint64_t depth = st.ahead_now[rank].v.fetch_add(1, std::memory_order_relaxed) + 1;
            st.ahead_sum[rank].add(depth);
            st.ahead_max[rank].keep_max(depth);
            st.ahead_count[rank].bump();
        }
        if (measure_dev) {
            // The pair exchange sends one payload; ring and halving 2*(N-1)/N of it.
            splice.bytes = m_world_size > 2 ? 2u * static_cast<std::size_t>(m_world_size - 1) * payload_bytes /
                                                  static_cast<std::size_t>(m_world_size)
                                            : payload_bytes;
        }
        te1 = stamp();
    } else {
        // Every rank submits before it syncs: a rank's list blocks on events
        // its neighbours only signal once they run.
        record_if_stale();
        const std::string what = "allreduce: queue rank " + std::to_string(rank) + " (n=" + std::to_string(n) +
                                 (m_world_size > 2 ? ", ring" : "") + ")";
        group.execute_and_wait(rank, m_rec.list(parity), what.c_str());
        te1 = stamp();
    }
    abort_guard.armed = false;

    if (measure_host) {
        const auto t2 = clk::now();
        if (rank == 0) {
            st.splice_ns.add(elapsed_ns(t1, te1));
            st.tail_ns.add(elapsed_ns(te1, t2));
        }
        st.ph1_ns[rank].add(elapsed_ns(t0, t1));
        st.ph2_ns[rank].add(elapsed_ns(t1, t2));
        st.last_exit[rank] = t2;
    }
    group.note_collective_done(rank);
}

}  // namespace ov::intel_gpu::ocl

#endif  // ENABLE_TP

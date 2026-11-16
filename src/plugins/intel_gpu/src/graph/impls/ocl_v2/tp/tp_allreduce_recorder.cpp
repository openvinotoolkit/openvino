// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_TP

#include "tp_allreduce_recorder.hpp"
#include "tp_common.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>

#include "openvino/core/except.hpp"
#include "intel_gpu/runtime/itt.hpp"

namespace ov::intel_gpu::ocl {

using cldnn::ze::tp::alloc_id_of;
using cldnn::ze::tp::Counter;
using cldnn::ze::tp::ScopedTime;
using cldnn::ze::tp::ze_api;

namespace {

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
    OV_ZE_EXPECT(ze_api()->zeKernelSetGroupSize(kernel, group_size, 1, 1));
    OV_ZE_EXPECT(ze_api()->zeKernelSetArgumentValue(kernel, 0, sizeof(dst), &dst));
    OV_ZE_EXPECT(ze_api()->zeKernelSetArgumentValue(kernel, 1, sizeof(src0), &src0));
    OV_ZE_EXPECT(ze_api()->zeKernelSetArgumentValue(kernel, 2, sizeof(src1), &src1));
    OV_ZE_EXPECT(ze_api()->zeKernelSetArgumentValue(kernel, 3, sizeof(n), &n));
    OV_ZE_EXPECT(ze_api()->zeCommandListAppendLaunchKernel(list, kernel, &groups, nullptr, 0, nullptr));
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

uint32_t halving_levels(uint32_t world_size) {
    uint32_t levels = 0;
    while ((1u << levels) < world_size) {
        ++levels;
    }
    return levels;
}

// Bytes of one halving staging region at `level`: half of what `level` halvings
// left, plus the slack halving_mid may add at every split.
std::size_t halving_region_bytes(std::size_t payload_bytes, uint32_t level) {
    std::size_t span = payload_bytes;
    for (uint32_t j = 0; j < level; ++j) {
        span = (span + 1) / 2 + kHalvingSlackBytes;
    }
    return align_up((span + 1) / 2 + kHalvingSlackBytes, 256);
}

void barrier(ze_command_list_handle_t list) {
    OV_ZE_EXPECT(ze_api()->zeCommandListAppendBarrier(list, nullptr, 0, nullptr));
}

void* byte_offset(void* base, std::size_t bytes) {
    return static_cast<uint8_t*>(base) + bytes;
}

// A primitive id may hold a comma or a quote.
std::string csv_quoted(const std::string& field) {
    std::string out = "\"";
    for (const char c : field) {
        out += c;
        if (c == '"')
            out += '"';
    }
    return out + "\"";
}

// Device statistics of a finished splice whose event still holds its timestamps.
void sample_device_time(cldnn::ze::tp::GroupState& group,
                        const cldnn::ze::tp::Splice& splice,
                        uint32_t rank,
                        uint64_t collective_id,
                        const std::string& name) {
    ze_kernel_timestamp_result_t kt{};
    if (ze_api()->zeEventQueryKernelTimestamp(splice.done, &kt) != ZE_RESULT_SUCCESS) {
        return;
    }
    auto& st = group.stats();
    const uint64_t mask = group.timestamp_mask(rank);
    const uint64_t tick = group.timer_ns_per_tick(rank);
    const uint64_t ns = ticks_to_ns(kt.global.kernelStart, kt.global.kernelEnd, mask, tick);
    st.p2_dev_ns[rank].add(ns);
    if (ns > st.p2_dev_max_ns[rank].get()) {
        st.p2_dev_max_ns[rank].keep_max(ns);
        st.p2_dev_max_cid[rank].v.store(collective_id, std::memory_order_relaxed);
    }
    st.p2_dev_count[rank].bump();

    // Powers of two in microseconds, clamped into the end buckets.
    const uint64_t us = ns / 1000u;
    uint32_t bucket = 0;
    for (uint64_t v = us; v > 1 && bucket < cldnn::ze::tp::CollectiveStats::kDevBuckets - 1; v >>= 1) {
        ++bucket;
    }
    st.p2_dev_hist[std::size_t{rank} * cldnn::ze::tp::CollectiveStats::kDevBuckets + bucket].bump();

    // Device time between the previous collective and this one: the model's
    // own work. A window starting before the previous one ended means
    // out-of-order readings or a wrap; skipped.
    const uint64_t prev_end = st.dev_last_end_ticks[rank] & mask;
    const uint64_t cur_start = kt.global.kernelStart & mask;
    std::string gap_us;
    if (st.dev_has_last[rank] && cur_start >= prev_end) {
        st.dev_gap_ns[rank].add((cur_start - prev_end) * tick);
        st.dev_gap_count[rank].bump();
        gap_us = std::to_string(static_cast<double>((cur_start - prev_end) * tick) / 1.0e3);
    }
    st.dev_last_end_ticks[rank] = kt.global.kernelEnd;
    st.dev_has_last[rank] = 1;
    st.dev_bytes[rank].add(splice.bytes);
    st.dev_call_min.keep_min(splice.call);
    st.dev_call_max.keep_max(splice.call);
    if (group.profiling_csv()) {
        std::ostringstream row;
        row << "device," << rank << "," << collective_id << "," << csv_quoted(name) << "," << splice.call << ",,"
            << splice.bytes << ",,,," << static_cast<double>(ns) / 1.0e3 << "," << gap_us;
        group.write_csv_row(row.str());
    }
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
    const uint32_t levels = halving_levels(m_world_size);
    std::size_t total = 0;
    for (uint32_t k = 0; k < 2 * levels; ++k) {
        total += halving_region_bytes(payload_bytes, k % levels);
    }
    return std::max(total, payload_bytes);
}

std::size_t TPAllReduceRecorder::chunk_stride(std::size_t payload_bytes, bool halving) const {
    // The largest chunk rounded up for alignment, with the slack halving_mid
    // may add; halving lays its regions out inside the same N slots.
    const std::size_t need = staging_bytes(payload_bytes, halving);
    const std::size_t N = m_world_size;
    return align_up((need + N - 1) / N + kHalvingSlackBytes, 256);
}

std::size_t TPAllReduceRecorder::half_bytes(std::size_t payload_bytes, bool halving) const {
    return m_world_size > 2 ? checked_multiply(chunk_stride(payload_bytes, halving), m_world_size, "ring staging")
                            : pair_stride(payload_bytes);
}

void TPAllReduceRecorder::ring_chunk(std::size_t n, uint32_t chunk, std::size_t& offset, std::size_t& count) const {
    // Chunk boundaries are aligned, not merely even: the kernel's vload8/vstore8
    // expect aligned bases, and a misaligned chunk corrupted a handful of
    // elements at its head. Trailing chunks may come out empty.
    const std::size_t N = m_world_size;
    const std::size_t c = chunk;
    const std::size_t units = (n + kRingAlignElems - 1) / kRingAlignElems;
    const std::size_t base = units / N;
    const std::size_t rem = units % N;
    const std::size_t unit_off = c * base + std::min(c, rem);
    const std::size_t unit_cnt = base + (c < rem ? 1u : 0u);
    offset = unit_off * kRingAlignElems;
    count = (offset >= n) ? 0u : std::min(unit_cnt * kRingAlignElems, n - offset);
}

void* TPAllReduceRecorder::ring_slot(const Signature& sig, const Piece& piece, uint32_t rank, uint32_t chunk) const {
    OPENVINO_ASSERT(chunk < m_world_size, "[TP][L0] ring slot chunk out of range: ", chunk);
    return byte_offset(sig.staging[piece.parity][rank], chunk * chunk_stride(piece.n * sig.dtype.size(), false));
}

void TPAllReduceRecorder::halving_range(std::size_t n,
                                        uint32_t rank,
                                        uint32_t level,
                                        std::size_t& lo,
                                        std::size_t& hi) const {
    lo = 0;
    hi = n;
    for (uint32_t j = 0; j < level; ++j) {
        const std::size_t mid = halving_mid(lo, hi);
        if (((rank >> j) & 1u) != 0u) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
}

void* TPAllReduceRecorder::halving_stage(const Signature& sig,
                                         const Piece& piece,
                                         uint32_t rank,
                                         uint32_t region) const {
    // Regions [0, levels) take the scatter steps, [levels, 2 * levels) the
    // gather steps: a partner may deliver the next step while this rank still
    // reads the last.
    const uint32_t levels = halving_levels(m_world_size);
    OPENVINO_ASSERT(region < 2 * levels, "[TP][L0] halving staging region out of range: ", region);
    const std::size_t payload_bytes = piece.n * sig.dtype.size();
    std::size_t offset = 0;
    for (uint32_t k = 0; k < region; ++k) {
        offset += halving_region_bytes(payload_bytes, k % levels);
    }
    OPENVINO_ASSERT(offset + halving_region_bytes(payload_bytes, region % levels) <= half_bytes(payload_bytes, true),
                    "[TP][L0] halving staging does not fit the staging half");
    return byte_offset(sig.staging[piece.parity][rank], offset);
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

void TPAllReduceRecorder::record(uint32_t parity) {
    auto& group = m_rec.group();
    auto& st = group.stats();
    const uint32_t rank = m_rec.rank();
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
        uint32_t chunk_parity = parity;
        for (std::size_t off = 0; off < sig.n; off += sig.chunk_elems) {
            Piece piece;
            piece.in = byte_offset(sig.in, off * elem);
            piece.out = byte_offset(sig.out, off * elem);
            piece.n = std::min(sig.chunk_elems, sig.n - off);
            piece.parity = chunk_parity;
            piece.halving = sig.halving;
            chunk_parity ^= 1u;
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
        OV_ZE_EXPECT(ze_api()->zeCommandListClose(list));
    }
    if (measure) {
        chosen->bump();
    }
}

void TPAllReduceRecorder::record_pair(ze_command_list_handle_t list, const Signature& sig, const Piece& piece) {
    auto& ev_recv = *m_call.recv[piece.parity];
    const auto& staging = sig.staging[piece.parity];
    const uint32_t r = m_rec.rank();
    const uint32_t peer = 1 - r;
    const std::size_t bytes = piece.n * sig.dtype.size();

    // Push our input into the peer's staging, never into the peer's own buffer,
    // which is what lets each rank record on its own. ev_recv[r] tells the peer.
    OV_ZE_EXPECT(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                         staging[peer],
                                                         piece.in,
                                                         bytes,
                                                         ev_recv[r],
                                                         0,
                                                         nullptr));
    OV_ZE_EXPECT(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ev_recv[peer]));
    // This rank is the only waiter; the wait already orders what follows, so
    // the reset needs no barrier.
    OV_ZE_EXPECT(ze_api()->zeCommandListAppendEventReset(list, ev_recv[peer]));
    append_reduce(list, sig.kernel, piece.out, piece.in, staging[r], piece.n);
}

void TPAllReduceRecorder::record_ring(ze_command_list_handle_t list, const Signature& sig, const Piece& piece) {
    auto& ev_ring = *m_call.step[piece.parity];
    const uint32_t N = m_world_size;
    const uint32_t steps = N - 1;
    const uint32_t r = m_rec.rank();
    const std::size_t n = piece.n;
    const std::size_t elem = sig.dtype.size();

    auto ev = [&](uint32_t step, uint32_t who) -> ze_event_handle_t& {
        return ev_ring[std::size_t{step} * N + who];
    };
    auto byte_at = [elem](void* base, std::size_t offset_elems) -> void* {
        return static_cast<uint8_t*>(base) + offset_elems * elem;
    };

    const uint32_t next = (r + 1) % N;
    const uint32_t prev = (r + N - 1) % N;

    // Reduce-scatter: at step s rank r forwards chunk (r-s) and folds the
    // incoming chunk (r-s-1), which already carries the partial sum of every
    // rank before it, into its own input. s < N - 1 keeps the sums positive.
    for (uint32_t s = 0; s < steps; ++s) {
        const uint32_t send_chunk = (r + N - s) % N;
        const uint32_t recv_chunk = (r + N - s - 1) % N;
        std::size_t send_off = 0, send_cnt = 0, recv_off = 0, recv_cnt = 0;
        ring_chunk(n, send_chunk, send_off, send_cnt);
        ring_chunk(n, recv_chunk, recv_off, recv_cnt);

        // An empty chunk still has to signal, or the successor waits forever.
        if (send_cnt == 0) {
            OV_ZE_EXPECT(ze_api()->zeCommandListAppendSignalEvent(list, ev(s, r)));
        } else {
            void* send_src = (s == 0) ? byte_at(piece.in, send_off) : byte_at(piece.out, send_off);
            OV_ZE_EXPECT(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                                 ring_slot(sig, piece, next, send_chunk),
                                                                 send_src,
                                                                 send_cnt * elem,
                                                                 ev(s, r),
                                                                 0,
                                                                 nullptr));
        }
        OV_ZE_EXPECT(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ev(s, prev)));
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
    OV_ZE_EXPECT(ze_api()->zeCommandListAppendSignalEvent(list, ev(2 * steps, r)));
    OV_ZE_EXPECT(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ev(2 * steps, next)));

    for (uint32_t s = 0; s < steps; ++s) {
        const uint32_t send_chunk = (r + 1 + N - s) % N;
        const uint32_t recv_chunk = (r + N - s) % N;
        std::size_t send_off = 0, send_cnt = 0, recv_off = 0, recv_cnt = 0;
        ring_chunk(n, send_chunk, send_off, send_cnt);
        ring_chunk(n, recv_chunk, recv_off, recv_cnt);

        const uint32_t step = steps + s;
        if (send_cnt == 0) {
            OV_ZE_EXPECT(ze_api()->zeCommandListAppendSignalEvent(list, ev(step, r)));
        } else {
            void* send_src = (s == 0) ? byte_at(piece.out, send_off) : ring_slot(sig, piece, r, send_chunk);
            OV_ZE_EXPECT(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                                 ring_slot(sig, piece, next, send_chunk),
                                                                 send_src,
                                                                 send_cnt * elem,
                                                                 ev(step, r),
                                                                 0,
                                                                 nullptr));
        }
        OV_ZE_EXPECT(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ev(step, prev)));
        if (recv_cnt > 0) {
            OV_ZE_EXPECT(ze_api()->zeCommandListAppendMemoryCopy(list,
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
    for (uint32_t s = 0; s < 2 * steps; ++s) {
        OV_ZE_EXPECT(ze_api()->zeCommandListAppendEventReset(list, ev(s, prev)));
    }
    OV_ZE_EXPECT(ze_api()->zeCommandListAppendEventReset(list, ev(2 * steps, next)));
}

void TPAllReduceRecorder::record_halving(ze_command_list_handle_t list, const Signature& sig, const Piece& piece) {
    auto& ev_ring = *m_call.step[piece.parity];
    const uint32_t N = m_world_size;
    const uint32_t r = m_rec.rank();
    const std::size_t n = piece.n;
    const std::size_t elem = sig.dtype.size();
    const uint32_t levels = halving_levels(N);
    OPENVINO_ASSERT((1u << levels) == N, "[TP][L0] halving requires a power-of-two world size");

    // Steps 0..2*levels-1: the first half scatters, the second gathers.
    // ev(step, who) is signalled by `who` and waited on by its partner.
    auto ev = [&](uint32_t step, uint32_t who) -> ze_event_handle_t& {
        return ev_ring[std::size_t{step} * N + who];
    };
    auto byte_at = [elem](void* base, std::size_t offset_elems) -> void* {
        return static_cast<uint8_t*>(base) + offset_elems * elem;
    };

    // Scatter: each step moves half of what is left.
    for (uint32_t j = 0; j < levels; ++j) {
        const uint32_t partner = r ^ (1u << j);
        std::size_t lo = 0, hi = 0;
        halving_range(n, r, j, lo, hi);
        const std::size_t mid = halving_mid(lo, hi);
        const bool keep_upper = ((r >> j) & 1u) != 0u;
        const std::size_t keep_lo = keep_upper ? mid : lo;
        const std::size_t keep_hi = keep_upper ? hi : mid;
        const std::size_t send_lo = keep_upper ? lo : mid;
        const std::size_t send_hi = keep_upper ? mid : hi;
        void* mine = (j == 0) ? piece.in : piece.out;

        if (send_hi > send_lo) {
            // At the start of the partner's region for this step, which is only
            // as large as one half and keeps the source aligned.
            OV_ZE_EXPECT(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                                 halving_stage(sig, piece, partner, j),
                                                                 byte_at(mine, send_lo),
                                                                 (send_hi - send_lo) * elem,
                                                                 ev(j, r),
                                                                 0,
                                                                 nullptr));
        } else {
            OV_ZE_EXPECT(ze_api()->zeCommandListAppendSignalEvent(list, ev(j, r)));
        }
        OV_ZE_EXPECT(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ev(j, partner)));
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
    for (uint32_t j = levels; j-- > 0;) {
        const uint32_t partner = r ^ (1u << j);
        const uint32_t step = 2 * levels - 1 - j;
        const uint32_t region = levels + j;
        std::size_t lo = 0, hi = 0;
        halving_range(n, r, j + 1, lo, hi);
        if (hi > lo) {
            OV_ZE_EXPECT(ze_api()->zeCommandListAppendMemoryCopy(list,
                                                                 halving_stage(sig, piece, partner, region),
                                                                 byte_at(piece.out, lo),
                                                                 (hi - lo) * elem,
                                                                 ev(step, r),
                                                                 0,
                                                                 nullptr));
        } else {
            OV_ZE_EXPECT(ze_api()->zeCommandListAppendSignalEvent(list, ev(step, r)));
        }
        OV_ZE_EXPECT(ze_api()->zeCommandListAppendWaitOnEvents(list, 1, &ev(step, partner)));

        std::size_t partner_lo = 0, partner_hi = 0;
        halving_range(n, partner, j + 1, partner_lo, partner_hi);
        if (partner_hi > partner_lo) {
            OV_ZE_EXPECT(ze_api()->zeCommandListAppendMemoryCopy(list,
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
    for (uint32_t step = 0; step < 2 * levels; ++step) {
        const uint32_t j = (step < levels) ? step : (2 * levels - 1 - step);
        OV_ZE_EXPECT(ze_api()->zeCommandListAppendEventReset(list, ev(step, r ^ (1u << j))));
    }
}

void TPAllReduceRecorder::run(const cldnn::ze::tp::Channel& channel,
                              uint64_t collective_id,
                              const std::string& name,
                              void* in,
                              void* out,
                              std::size_t n,
                              ov::element::Type dtype,
                              ze_kernel_handle_t kernel,
                              ze_command_list_handle_t model_queue) {
    auto& group = *channel.group();
    const uint32_t rank = channel.rank();
    // A null device pointer reaches the driver as a request to wrap a host
    // allocation at address 0 and aborts the process.
    OPENVINO_ASSERT(in != nullptr && out != nullptr,
                    "[TP][L0] null device buffer passed to allreduce ",
                    name,
                    " (collective ",
                    collective_id,
                    ", rank ",
                    rank,
                    ")");
    OPENVINO_ASSERT(kernel != nullptr,
                    "[TP][L0] allreduce ",
                    name,
                    " called without a reduction kernel (collective ",
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
    m_rec.bind(channel, true);
    const uint64_t call = m_calls++;

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
    {
        OV_ITT_SCOPED_TASK_BASE(ov::intel_gpu::itt::domains::tp_collective, "tp_allreduce_rendezvous");
        group.begin_allreduce(rank,
                              collective_id,
                              n,
                              dtype,
                              half_bytes(plan.chunk_elems * dtype.size(), plan.halving),
                              plan.count,
                              m_call);
    }
    const auto t1 = stamp();
    const uint32_t parity = m_call.parity;
    if (measure_host && rank == 0 && plan.count > 1) {
        st.chunked_calls.bump();
        st.chunks.add(plan.count);
        st.max_chunks.keep_max(plan.count);
    }

    // Whoever fails leaves the rest of the group waiting for something that
    // will never arrive.
    AbortOnFailure abort_guard{group, collective_id, name, rank, "running"};

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
        OV_ITT_SCOPED_TASK_BASE(ov::intel_gpu::itt::domains::tp_collective, "tp_allreduce_record");
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
            OV_ITT_SCOPED_TASK_BASE(ov::intel_gpu::itt::domains::tp_collective, "tp_allreduce_wait_previous_splice");
            const auto w0 = stamp();
            bool blocked = false;
            group.wait_splice(splice, rank, collective_id, &blocked);
            const auto w1 = stamp();
            if (measure_dev) {
                // The event still holds the timestamps of the splice that just
                // finished; the release below clears them.
                sample_device_time(group, splice, rank, collective_id, name);
            }
            group.release_splice(splice, rank);
            if (measure_host) {
                const auto w2 = clk::now();
                if (blocked) {
                    st.p2_block_count[rank].bump();
                }
                st.p2_wait_ns[rank].add(elapsed_ns(w0, w1));
                st.p2_reset_ns[rank].add(elapsed_ns(w1, w2));
                st.p2_wait_count[rank].bump();
            }
        }

        record_if_stale();
        const auto a0 = stamp();
        {
            OV_ITT_SCOPED_TASK_BASE(ov::intel_gpu::itt::domains::tp_collective, "tp_allreduce_append");
            m_rec.splice_into(model_queue, parity);
        }
        if (measure_host) {
            st.p2_append_ns[rank].add(elapsed_ns(a0, clk::now()));
            const uint64_t depth = st.ahead_now[rank].v.fetch_add(1, std::memory_order_relaxed) + 1;
            st.ahead_sum[rank].add(depth);
            st.ahead_max[rank].keep_max(depth);
            st.ahead_count[rank].bump();
        }
        if (measure_dev) {
            // The pair exchange sends one payload; ring and halving 2*(N-1)/N of it.
            splice.bytes = m_world_size > 2 ? std::size_t{2} * (m_world_size - 1) * payload_bytes / m_world_size
                                            : payload_bytes;
            splice.call = call;
        }
        te1 = stamp();
    } else {
        // Every rank submits before it syncs: a rank's list blocks on events
        // its neighbours only signal once they run.
        record_if_stale();
        const std::string what = "allreduce: queue rank " + std::to_string(rank) + " (n=" + std::to_string(n) +
                                 (m_world_size > 2 ? ", ring" : "") + ")";
        OV_ITT_SCOPED_TASK_BASE(ov::intel_gpu::itt::domains::tp_collective, "tp_allreduce_execute_sync");
        group.execute_and_wait(rank, m_rec.list(parity), what.c_str());
        te1 = stamp();
    }
    abort_guard.armed = false;

    if (measure_host || measure_dev) {
        st.ar_calls[rank].bump();
        if (rank == 0) {
            st.ar_bytes.add(payload_bytes);
        }
    }
    if (measure_host) {
        const auto t2 = clk::now();
        if (rank == 0) {
            st.splice_ns.add(elapsed_ns(t1, te1));
            st.tail_ns.add(elapsed_ns(te1, t2));
        }
        st.ph1_ns[rank].add(elapsed_ns(t0, t1));
        st.ph2_ns[rank].add(elapsed_ns(t1, t2));
        st.last_exit[rank] = t2;
        if (group.profiling_csv()) {
            const char* schedule = m_world_size == 2 ? "pair" : plan.halving ? "halving" : "ring";
            std::ostringstream row;
            row << "host," << rank << "," << collective_id << "," << csv_quoted(name) << "," << call << ","
                << schedule << "," << payload_bytes
                << "," << static_cast<double>(group.last_late_ns(rank)) / 1.0e3 << ","
                << static_cast<double>(elapsed_ns(t0, t1)) / 1.0e3 << "," << static_cast<double>(elapsed_ns(t1, t2)) / 1.0e3
                << ",,";
            group.write_csv_row(row.str());
        }
    }
    group.note_collective_done(rank);
}

}  // namespace ov::intel_gpu::ocl

#endif  // ENABLE_TP

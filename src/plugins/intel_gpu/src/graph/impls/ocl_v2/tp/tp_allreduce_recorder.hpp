// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "tp_recording.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "openvino/core/type/element_type.hpp"

namespace ov::intel_gpu::ocl {

/// \brief The Level Zero allreduce-sum of one TPAllReduce on one rank.
///
/// Three schedules, chosen per recording: two ranks exchange directly, larger
/// worlds use a ring, and a power-of-two world carrying a small payload uses
/// recursive halving. Peers write only into each other's staging; every rank
/// delivers into its own output. A rank's recording is kept until its own
/// buffers, its kernel, the shape, or a peer's staging change.
class TPAllReduceRecorder {
public:
    TPAllReduceRecorder() = default;
    TPAllReduceRecorder(const TPAllReduceRecorder&) : TPAllReduceRecorder() {}
    TPAllReduceRecorder& operator=(const TPAllReduceRecorder&) = delete;

    /// Sums `in` over all ranks into `out` (they may alias), each rank calling
    /// with the same `collective_id`, `n` and `dtype`. `kernel` is this rank's
    /// reduction kernel (tp_allreduce.cl) for `dtype`; the recording names it.
    ///
    /// With `model_queue` -- the in-order immediate list the model runs on --
    /// and run_spliced(), the recording is spliced there and the call returns
    /// without waiting. Otherwise it runs on the group's queue and blocks.
    void run(const cldnn::ze::tp::Channel& channel,
             int collective_id,
             void* in,
             void* out,
             std::size_t n,
             ov::element::Type dtype,
             ze_kernel_handle_t kernel,
             ze_command_list_handle_t model_queue);

private:
    // What one parity's recording was built from: this rank's buffers and
    // kernel, the shape, and where the peers' staging is.
    struct Signature {
        bool valid{false};
        void* in{nullptr};
        void* out{nullptr};
        uint64_t in_id{0};
        uint64_t out_id{0};
        ze_kernel_handle_t kernel{nullptr};
        std::size_t n{0};
        ov::element::Type dtype{ov::element::dynamic};
        std::vector<void*> staging;
        std::vector<uint64_t> staging_ids;

        bool operator==(const Signature& o) const {
            return valid == o.valid && in == o.in && out == o.out && in_id == o.in_id && out_id == o.out_id &&
                   kernel == o.kernel && n == o.n && dtype == o.dtype && staging == o.staging &&
                   staging_ids == o.staging_ids;
        }
    };

    void record(int parity);
    void record_pair(int parity);
    void record_ring(int parity);
    void record_halving(int parity);

    // Staging layout, shared by the writer and the reader of every region.
    bool uses_halving(std::size_t payload_bytes) const;
    std::size_t staging_bytes(std::size_t payload_bytes) const;
    std::size_t chunk_stride(std::size_t payload_bytes) const;
    std::size_t half_bytes(std::size_t payload_bytes) const;
    void ring_chunk(std::size_t n, int chunk, std::size_t& offset, std::size_t& count) const;
    void* ring_slot(const Signature& sig, int rank, int chunk) const;
    void halving_range(std::size_t n, int rank, int level, std::size_t& lo, std::size_t& hi) const;
    void* halving_stage(const Signature& sig, int rank, int region) const;

    TPRecording m_rec;
    Signature m_sig[2];
    cldnn::ze::tp::GroupState::AllreduceCall m_call;
    int m_world_size{0};
};

}  // namespace ov::intel_gpu::ocl

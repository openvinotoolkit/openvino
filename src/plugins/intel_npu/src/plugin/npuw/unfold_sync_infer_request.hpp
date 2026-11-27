// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

#include "base_sync_infer_request.hpp"
#include "intel_npu/common/execution_list.hpp"

namespace ov {
namespace npuw {
namespace compiled {
struct HostFlashAttention;
}  // namespace compiled
namespace runtime {
namespace host_flash_attention {
class Selector;
}  // namespace host_flash_attention
}  // namespace runtime

class UnfoldInferRequest final : public IBaseInferRequest {
public:
    explicit UnfoldInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model);

    ////////////////////////////////////
    // implement IBaseInferRequest - nether of these are required here
    // this hierarchy needs revew
    void prepare_for_infer() override {}
    bool valid_subrequest(std::size_t idx) const override;
    void start_subrequest(std::size_t) override {}
    void run_subrequest_for_success(std::size_t) override {}
    void subscribe_subrequest(std::size_t, Completed cb) override {}
    void complete_subrequest(std::size_t) override {}
    void cancel_subrequest(std::size_t) override {}
    bool supports_async_pipeline() const override {
        return false;
    }
    void update_subrequest_links(std::size_t) override {}

private:
    void infer() override;
    bool bind_behavior_input(std::size_t idx,
                             std::size_t real_idx,
                             std::size_t input_idx,
                             const ov::SoPtr<ov::ITensor>& tensor,
                             RqPtr request) override;

    // PROTOTYPE (see ov::npuw::util::exec_list_enabled_for()): all subgraphs [first, last] recorded into
    // one NPU execution list. When set, infer() runs only through it.
    struct ExecList {
        std::size_t first = 0u;
        std::size_t last = 0u;
        std::size_t executes = 0u;
        std::shared_ptr<::intel_npu::IExecutionList> list;
    };
    void create_execution_list();
    ExecList m_exec_list;

    // PROTOTYPE: HFA (host flash attention) calls run here instead of through the attention runtime behavior of
    // JustInferRequest. Every call has its own requests for its regular tiles, bound once: the K/V tiles are whole
    // KV blocks or strided views of the KV cache, and the attention state (ACC, MAX, D) is a set of tensors shared by
    // all calls of a function, initialized by binding the first tile to constant initial values. So nothing has to
    // be done on the host between the tiles, and a prefill chunk runs as one sequence of requests. Only the mask tiles
    // are cut on the host, once per chunk, after the subgraph producing the mask has run.
    struct HFATile {
        std::size_t block = 0u;  // Index of the past KV block
        int64_t offset = 0;      // In the block
    };
    struct HFACall {
        std::size_t idx = 0u;       // Subgraph index of the call
        std::size_t real_idx = 0u;  // Its function body
        const ov::npuw::compiled::HostFlashAttention* hfa = nullptr;
        std::vector<ov::SoPtr<ov::ITensor>> params;  // By function parameter index
        std::optional<std::size_t> mask_producer;    // Subgraph producing the mask, if not a global parameter

        std::vector<const void*> bound_params;  // Data of params when the requests were bound
        std::vector<HFATile> tiles;             // All past tiles the KV blocks hold
        int64_t layer_kv_length = 0;            // Present + all past blocks
        RqPtrs tile_requests;                   // One per past tile, created when a chunk needs it
        std::size_t bound_tile_requests = 0u;   // tile_requests[0, bound_tile_requests) are bound to params
        std::size_t past_tiles = 0u;            // Run by this infer
        std::optional<bool> final_reads_init;   // Which state the final tile request reads now
    };
    struct HFAState {  // Per function body
        ov::SoPtr<ov::ITensor> init_acc, init_max, init_d;
        ov::SoPtr<ov::ITensor> acc, max, d;
    };
    struct HFAMask {  // Tiles of one attention mask, per function body
        ov::SoPtr<ov::ITensor> source;
        std::vector<ov::SoPtr<ov::ITensor>> regular;  // One per past tile; empty if the regular tiles have no mask
        ov::SoPtr<ov::ITensor> final_tile;
        std::size_t needed_regular = 0u;  // By this infer
        bool cut = false;                 // In this infer
    };
    using HFAMaskKey = std::pair<std::size_t /* real_idx */, const void* /* source data */>;

    void bind_hfa_call(HFACall& call);
    void bind_hfa_tile(HFACall& call, std::size_t tile);
    HFAState& hfa_state(const HFACall& call);
    HFAMask& hfa_mask(const HFACall& call);
    void cut_hfa_mask(const HFACall& call, HFAMask& mask);
    void prepare_hfa_calls();
    void run_hfa_segment(RqPtrs& requests, std::size_t segment);
    void infer_with_hfa();

    std::vector<HFACall> m_hfa_calls;
    std::unordered_map<std::size_t, std::size_t> m_hfa_call_of;  // Subgraph index -> m_hfa_calls index
    std::map<std::size_t, HFAState> m_hfa_states;
    std::map<HFAMaskKey, HFAMask> m_hfa_masks;
    std::shared_ptr<ov::npuw::runtime::host_flash_attention::Selector> m_hfa_selector;
    // Lists are reset and filled again for every infer, since the number of tiles grows with every chunk. Without
    // a factory the requests run one by one.
    std::shared_ptr<::intel_npu::IExecutionListFactory> m_hfa_list_factory;
    std::vector<std::shared_ptr<::intel_npu::IExecutionList>> m_hfa_lists;  // One per segment
};

}  // namespace npuw
}  // namespace ov

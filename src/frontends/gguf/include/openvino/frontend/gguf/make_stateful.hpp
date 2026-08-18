// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>
#include <string>

#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::frontend::gguf::pass {

class GGUF_FRONTEND_API MakeStateful : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("gguf::MakeStateful");

    /// \param skip_caches Friendly names of cache Parameters to leave stateless. A sliding-window
    ///        cache is evicted from the front rather than only appended to, so an append-grown
    ///        Variable would not reproduce it.
    /// \param append_axis Cache axis the new rows are appended along (the token axis). -1 infers
    ///        it as the cache Parameter's single dynamic axis. Pass an explicit axis for a fully
    ///        static (preallocated) cache, where there is nothing to infer from.
    /// \param beam_idx_name Name of the beam-reorder input, which this pass ADDS to the model (no
    ///        decoder declares it, since it indexes OpenVINO state that ggml has no counterpart
    ///        for). The past cache is gathered by it along the batch axis before the append
    ///        Concat; with batch 1 that Gather is an identity, but emitting it is what lets CPU's
    ///        stateful_sdpa_fusion match and makes beam search work. A model that already carries
    ///        a Parameter of this name has it reused instead.
    explicit MakeStateful(std::set<std::string> skip_caches = {},
                          int64_t append_axis = -1,
                          std::string beam_idx_name = "beam_idx")
        : m_skip_caches(std::move(skip_caches)),
          m_append_axis(append_axis),
          m_beam_idx_name(std::move(beam_idx_name)) {}

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    std::set<std::string> m_skip_caches;
    int64_t m_append_axis;
    std::string m_beam_idx_name;
};

}  // namespace ov::frontend::gguf::pass

// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/openvino.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace npuw {

namespace function {

// Partition-time spatial information. So far assume spatial execution in 1 dimension only
// Defined at this level to be aligned with other partitioning entities (but needs to be moved)
struct Spatial {
    using PPtr = std::shared_ptr<ov::op::v0::Parameter>;
    struct Param {
        PPtr param;
        std::size_t dim;
    };
    std::size_t _range = 0u;    // Range over which spatial execution is organized, e.g. 1024
    std::size_t _slice = 0u;    // A submission size for a single execution, e.g. 128
    std::size_t _out_dim = 0u;  // Assume it is the same dim for all Results
    std::vector<Param> _inputs;
};

}  // namespace function

namespace compiled {

// Compile-time spatial information. Not much different from the above
struct Spatial {
    struct Param {
        std::size_t idx;  // function input index for this spatial parameter
        std::size_t dim;
    };
    std::vector<Param> params;
    std::size_t range = 0u;    // NB: duplication of the above
    std::size_t nway = 0u;     // NB: duplication of the above
    std::size_t out_dim = 0u;  // NB: duplication of the above

    std::size_t nway_iters = 0u;
    std::size_t tail_size = 0u;

    Spatial(const function::Spatial& s, const std::shared_ptr<ov::Model>& m)
        : range(s._range),
          nway(s._slice),
          out_dim(s._out_dim),
          nway_iters(range / nway),
          tail_size(range % nway) {
        for (auto&& input : s._inputs) {
            std::size_t p_idx = m->get_parameter_index(input.param);
            params.push_back(Param{p_idx, input.dim});
        }
    }
};

}  // namespace compiled

namespace runtime {
namespace spatial {

// A base class to decide the work-scope from some feature
class Selector {
public:
    using Ptr = std::shared_ptr<Selector>;
    virtual ~Selector() = default;
    virtual void prepare() = 0;
    virtual bool need_submit(std::size_t offset, std::size_t len) const = 0;
};

// No dynamic dispatch - just run over the whole range
class All final : public Selector {
    void prepare() override {}
    bool need_submit(std::size_t, std::size_t) const override {
        return true;
    }
};

// Define work scope based on attention mask
class AttentionMask final : public Selector {
    std::size_t m_attn_mask_param_idx = 0u;
    std::size_t m_valid_range_begin = 0u;
    std::size_t m_valid_range_end = 0u;

    const ov::ISyncInferRequest& m_rq;

    AttentionMask(std::size_t param_idx, const ov::ISyncInferRequest& rq);
    void prepare() override;
    bool need_submit(std::size_t offset, std::size_t len) const override;

public:
    static Selector::Ptr find(const ov::ISyncInferRequest& rq);
};

}  // namespace spatial
}  // namespace runtime

}  // namespace npuw
}  // namespace ov

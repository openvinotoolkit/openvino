// Copyright (C) 2025 Intel Corporation
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

// Partition-time dynamic information. So far assume dynamic execution in 1 dimension only
// Defined at this level to be aligned with other partitioning entities (but needs to be moved?)
struct Dynamic {
    using PPtr = std::shared_ptr<ov::op::v0::Parameter>;
    struct Param {
        PPtr param;
        std::size_t dim;
    };
    std::vector<Param> _inputs;
};

}  // namespace function

namespace compiled {

// Compile-time dynamic information. Not much different from the above
struct Dynamic {
    struct Param {
        std::size_t idx;  // function input index for this spatial parameter
        std::size_t dim;
    };
    std::vector<Param> params;
    Dynamic() = default;
    Dynamic(const function::Dynamic& s, const std::shared_ptr<ov::Model>& m) {
        for (auto&& input : s._inputs) {
            std::size_t p_idx = m->get_parameter_index(input.param);
            params.push_back(Param{p_idx, input.dim});
        }
    }
};

}  // namespace compiled

namespace runtime {
namespace dynamic {

// A selector class to land here

}  // namespace dynamic
}  // namespace runtime

}  // namespace npuw
}  // namespace ov

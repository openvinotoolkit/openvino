// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/pass.hpp"


namespace ov {
namespace snippets {
namespace lowered {
namespace pass {

void PassPipeline::register_pass(const std::shared_ptr<Pass>& pass) {
    m_passes.push_back(pass);
}

void PassPipeline::run(LinearIR& linear_ir) const {
    for (const auto& pass : m_passes) {
        pass->run(linear_ir);
    }
}

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ov

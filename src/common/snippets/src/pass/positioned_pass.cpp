// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/positioned_pass.hpp"

#include <cstddef>

#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets::pass {

PassPosition::PassPosition(Place pass_place) : m_place(pass_place) {
    OPENVINO_ASSERT(utils::any_of(m_place, Place::PipelineStart, Place::PipelineEnd),
                    "Invalid arg: pass_type_info and pass_instance args could be omitted only for "
                    "Place::PipelineStart/Place::PipelineEnd");
}

PassPosition::PassPosition(Place pass_place, const DiscreteTypeInfo& pass_type_info, size_t pass_instance)
    : m_pass_type_info(pass_type_info),
      m_pass_instance(pass_instance),
      m_place(pass_place) {
    OPENVINO_ASSERT(
        utils::any_of(m_place, Place::Before, Place::After) && m_pass_type_info != DiscreteTypeInfo(),
        "Invalid args combination: pass_place must be Place::Before/Place::After and pass_type_info must be non-empty");
}

}  // namespace ov::snippets::pass

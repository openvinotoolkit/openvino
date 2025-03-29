// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/positioned_pass.hpp"


namespace ov {
namespace snippets {
namespace pass {

PassPosition::PassPosition(Place pass_place) : m_place(pass_place) {
    OPENVINO_ASSERT(m_place == Place::PipelineStart || m_place == Place::PipelineEnd,
                    "Invalid arg: pass_type_info and pass_instance args could be omitted only for Place::PipelineStart/Place::PipelineEnd");
}

PassPosition::PassPosition(Place pass_place, const DiscreteTypeInfo& pass_type_info, size_t pass_instance)
: m_pass_type_info(pass_type_info), m_pass_instance(pass_instance), m_place(pass_place) {
    OPENVINO_ASSERT((m_place == Place::Before || m_place == Place::After) && m_pass_type_info != DiscreteTypeInfo(),
                    "Invalid args combination: pass_place must be Place::Before/Place::After and pass_type_info must be non-empty");
}

} // namespace pass
} // namespace snippets
} // namespace ov

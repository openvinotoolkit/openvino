// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/pass_position.hpp"


namespace ov {
namespace snippets {
namespace pass {
PassPosition::PassPosition(Place pass_place) : m_place(pass_place) {
    OPENVINO_ASSERT(m_place == Place::PipelineStart || m_place == Place::PipelineEnd,
                    "Invalid arg: pass_name and pass_instance args could be omitted only for Place::PipelineStart/Place::PipelineEnd");
}
PassPosition::PassPosition(Place pass_place, std::string pass_name, size_t pass_instance)
: m_pass_name(std::move(pass_name)), m_pass_instance(pass_instance), m_place(pass_place) {
    OPENVINO_ASSERT((m_place == Place::Before || m_place == Place::After) && !m_pass_name.empty(),
                    "Invalid args combination: pass_place must be Place::Before/Place::After and pass_name must be non-empty");
}
} // namespace pass
} // namespace snippets
} // namespace ov

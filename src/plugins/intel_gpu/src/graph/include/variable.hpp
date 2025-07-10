// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "openvino/core/type/element_type.hpp"

namespace cldnn {
namespace memory_state {

class variable {
public:
    explicit variable(const std::string& variable_id, ov::element::Type user_specified_type = ov::element::dynamic)
        : m_variable_id{variable_id},
          m_user_specified_type(user_specified_type) {}

    const std::string& variable_id() const { return m_variable_id; }
    ov::element::Type get_user_specified_type() const { return m_user_specified_type; }

private:
    std::string m_variable_id;
    ov::element::Type m_user_specified_type;
};

} // namespace memory_state
} // namespace cldnn

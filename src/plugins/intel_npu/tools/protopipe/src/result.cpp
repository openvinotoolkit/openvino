//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "result.hpp"
#include "utils/error.hpp"

Result::Result(const Error& error): m_status(error){};
Result::Result(const Success& success): m_status(success){};

Result::operator bool() const {
    return std::holds_alternative<Success>(m_status);
}

std::string Result::str() const {
    if (std::holds_alternative<Success>(m_status)) {
        return std::get<Success>(m_status).msg;
    }
    ASSERT(std::holds_alternative<Error>(m_status));
    return std::get<Error>(m_status).reason;
}

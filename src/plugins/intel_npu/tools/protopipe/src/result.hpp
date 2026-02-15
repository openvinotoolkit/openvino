//
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <variant>

struct Success {
    std::string msg;
};
struct Error {
    std::string reason;
};

class Result {
public:
    Result() = default;  // monostate (empty)
    Result(const Error& error);
    Result(const Success& success);

    operator bool() const;
    std::string str() const;

private:
    using Status = std::variant<std::monostate, Error, Success>;
    Status m_status;
};

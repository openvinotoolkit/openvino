// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "executor.hpp"

#include <string>

namespace ov::intel_cpu {

std::string ExecutorTypeToString(const ExecutorType type) {
#define CASE(_type)           \
    case ExecutorType::_type: \
        return #_type;
    switch (type) {
        CASE(Undefined);
        CASE(Graph);
        CASE(Common);
        CASE(jit_x64);
        CASE(Dnnl);
        CASE(Acl);
        CASE(Mlas);
        CASE(jit_aarch64);
        CASE(Shl);
        CASE(Kleidiai);
    }
#undef CASE
    return "Undefined";
}

ExecutorType ExecutorTypeFromString(const std::string& typeStr) {
#define CASE(_type)                 \
    if (typeStr == #_type) {        \
        return ExecutorType::_type; \
    }
    CASE(Undefined);
    CASE(Graph);
    CASE(Common);
    CASE(jit_x64);
    CASE(Dnnl);
    CASE(Acl);
    CASE(Mlas);
    CASE(jit_aarch64);
    CASE(Shl);
    CASE(Kleidiai);
#undef CASE
    return ExecutorType::Undefined;
}

}  // namespace ov::intel_cpu

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>

#include "executor.hpp"

namespace ov {
namespace intel_cpu {

std::string ExecutorTypeToString(const ExecutorType type) {
#define CASE(_type) case ExecutorType::_type: return #_type;
    switch (type) {
        CASE(Undefined);
        CASE(Graph);
        CASE(Common);
        CASE(x64);
        CASE(Dnnl);
        CASE(Acl);
        CASE(Mlas);
    }
#undef CASE
    return "Undefined";
}

ExecutorType ExecutorTypeFromString(const std::string& typeStr) {
#define CASE(_type) if (typeStr == #_type) { return ExecutorType::_type; }
    CASE(Undefined);
    CASE(Graph);
    CASE(Common);
    CASE(x64);
    CASE(Dnnl);
    CASE(Acl);
    CASE(Mlas);
#undef CASE
    return ExecutorType::Undefined;
}

}   // namespace intel_cpu
}   // namespace ov

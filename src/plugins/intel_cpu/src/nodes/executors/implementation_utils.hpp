// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include "cpu_types.h"
#include "memory_desc/cpu_memory_desc.h"
#include "nodes/executors/memory_arguments.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace intel_cpu {

template<typename Config, int idx>
ov::element::Type memoryDescType(const Config& config) {
    return config.descs.at(idx)->getPrecision();
}

template<typename Config>
ov::element::Type srcType(const Config& config) {
    return memoryDescType<Config, ARG_SRC>(config);
}

template<typename Config>
ov::element::Type weiType(const Config& config) {
    return memoryDescType<Config, ARG_WEI>(config);
}

template<typename Config>
ov::element::Type biaType(const Config& config) {
    return memoryDescType<Config, ARG_BIAS>(config);
}

template<typename Config, int idx = 0>
ov::element::Type dstType(const Config& config) {
    return memoryDescType<Config, ARG_DST>(config);
}

template<typename Config, int idx>
ov::element::Type dims(const Config& config) {
    return config.descs.at(idx)->getShape().getDims();
}

template<typename Config>
const VectorDims& srcDims(const Config& config) {
    return dims<Config, ARG_SRC>(config);
}

template<typename Config>
const VectorDims& weiDims(const Config& config) {
    return dims<Config, ARG_WEI>(config);
}

template<typename Config, int idx>
size_t rank(const Config& config) {
    return config.descs.at(idx)->getShape().getRank();
}

template<typename Config>
size_t srcRank(const Config& config) {
    return rank<Config, ARG_SRC>(config);
}

template<typename Config>
size_t weiRank(const Config& config) {
    return rank<Config, ARG_WEI>(config);
}

template<typename Config, int idx>
size_t memSize(const Config& config) {
    return config.descs.at(idx)->getCurrentMemSize();
}

template<typename Config>
size_t srcMemSize(const Config& config) {
    return memSize<Config, ARG_SRC>(config);
}

template<typename Config>
size_t weiMemSize(const Config& config) {
    return memSize<Config, ARG_WEI>(config);
}

template<typename Config>
size_t postOpsNumbers(const Config& config) {
    return config.postOps.size();
}

}   // namespace intel_cpu
}   // namespace ov

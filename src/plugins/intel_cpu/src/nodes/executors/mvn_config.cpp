// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_config.hpp"

#include <utility>

#include "cpu_types.h"
#include "nodes/executors/executor.hpp"
#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

legacy::MVNExecutor::MVNExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

legacy::MVNExecutorBase::MVNExecutorBase(const MVNAttrs& mvnAttrs)
    : mvnAttrs(mvnAttrs),
      src_data_size(mvnAttrs.src_prc.size()),
      dst_data_size(mvnAttrs.dst_prc.size()) {}

}  // namespace ov::intel_cpu
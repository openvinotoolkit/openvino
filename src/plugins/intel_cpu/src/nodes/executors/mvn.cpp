// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn.hpp"

namespace ov {
namespace intel_cpu {

MVNExecutorBase::MVNExecutorBase(const MVNAttrs& mvnAttrs)
        : mvnAttrs(mvnAttrs),
          src_data_size(mvnAttrs.src_prc.size()),
          dst_data_size(mvnAttrs.dst_prc.size()) {}

MVNExecutor::MVNExecutor(const ExecutorContext::CPtr context) : context(context) {}

}   // namespace intel_cpu
}   // namespace ov
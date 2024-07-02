// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <tuple>

#include "nodes/executors/executor.hpp"
#include "nodes/executors/mvn.hpp"
#include "onednn/iml_type_mapper.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

class MVNRefExecutor : public MVNExecutorBase {
public:
    MVNRefExecutor(const MVNAttrs& mvnAttrs);

    void exec(const uint8_t *in_ptr_, uint8_t *out_ptr_, const void *post_ops_data_, const VectorDims& shape5d) override;

private:
    void mvn_ref(const uint8_t *in_ptr_, uint8_t *out_ptr_, const VectorDims& shape5d);
};

}   // namespace intel_cpu
}   // namespace ov
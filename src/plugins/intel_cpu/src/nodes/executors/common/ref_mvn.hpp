// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/mvn_config.hpp"
#include "cpu_types.h"


namespace ov::intel_cpu::legacy {

class MVNRefExecutor : public legacy::MVNExecutorBase {
public:
    MVNRefExecutor(const MVNAttrs& mvnAttrs);

    void exec(const uint8_t* src_data,
              uint8_t* dst_data,
              const void* post_ops_data_,
              const VectorDims& shape5d) override;

private:
    void mvn_ref(const uint8_t* src_data, uint8_t* dst_data, const VectorDims& shape5d);
};

} // namespace ov::intel_cpu::legacy

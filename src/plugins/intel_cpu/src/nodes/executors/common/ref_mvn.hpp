// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include "cpu_types.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/mvn_config.hpp"
#include "onednn/iml_type_mapper.h"

namespace ov::intel_cpu {

namespace legacy {

class MVNRefExecutor : public legacy::MVNExecutorBase {
public:
    MVNRefExecutor(const MVNAttrs& mvnAttrs, const dnnl::primitive_attr& attr);

    void exec(const uint8_t* src_data,
              uint8_t* dst_data,
              const void* post_ops_data_,
              const VectorDims& shape5d) override;

private:
    void mvn_ref(const uint8_t* src_data, uint8_t* dst_data, const VectorDims& shape5d);
};

}  // namespace legacy

class CommonMVNExecutor : public Executor {
public:
    CommonMVNExecutor(const MVNAttrs& attrs, const MemoryArgs& memory, const ExecutorContext::CPtr context)
        : refContext(context),
          refMVNAttrs(attrs) {
        postOpsDataPtrs = attrs.postOpsDataPtrs;
    }

    void execute(const MemoryArgs& memory) override;

    [[nodiscard]] impl_desc_type implType() const override {
        return impl_desc_type::ref;
    }

    // offloads execution data preparation from the exec call
    bool update(const MemoryArgs& memory) override;

    static bool supports(const MVNConfig& config);

private:
    ExecutorContext::CPtr refContext;
    MVNAttrs refMVNAttrs;
    std::vector<const void*> postOpsDataPtrs;
    std::shared_ptr<legacy::MVNRefExecutor> oldMVNRefExecutor;
    VectorDims shape5D;
};
}  // namespace ov::intel_cpu

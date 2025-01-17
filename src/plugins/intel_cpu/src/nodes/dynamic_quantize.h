// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

// #include <memory>
// #include <oneapi/dnnl/dnnl.hpp>
// #include <string>
// #include <vector>

// #include "cpu_memory.h"
// #include "nodes/executors/executor_factory.hpp"
// #include "nodes/executors/memory_arguments.hpp"
// #include "nodes/executors/DynamicQuantize_config.hpp"
// #include "post_ops.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

struct DQAttrs {
    size_t groupSize;
};

struct dynamic_quantization_compile_params_t {
    size_t ic_quant_block;
    ov::element::Type src_dt;
    ov::element::Type qsrc_dt;
};

struct dynamic_quantization_runtime_params_t {
    const void *src_ptr;
    const void *qsrc_ptr;
    const void *src_scales_ptr;
    size_t ic_size;
};

struct dynamic_quantization_kernel_t {
    void operator()(const dynamic_quantization_runtime_params_t *args) { assert(ker_);
        ker_(args);
    }

    dynamic_quantization_kernel_t(const dynamic_quantization_compile_params_t& jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~dynamic_quantization_kernel_t() {}
protected:
    void (*ker_)(const dynamic_quantization_runtime_params_t *);

    dynamic_quantization_compile_params_t jcp_;
};

class DynamicQuantize : public Node {
public:
    DynamicQuantize(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    
    void getSupportedDescriptors() override {}
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool canBeInPlace() const override { return false; }

    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;

    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;

private:
    static const size_t DATA_ID = 0;
    static const size_t SCALES_ID = 1;
    static const size_t ZERO_POINTS_ID = 2;

    DQAttrs attrs;
    ov::element::Type_t inputPrecision;

    std::unique_ptr<dynamic_quantization_kernel_t> pKernel;

    uint64_t groupSize = 0;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
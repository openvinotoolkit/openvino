// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "node.h"
#include "nodes/common/softmax.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

struct jit_args_logistic {
    const void* src;
    void* dst;
    size_t work_amount;
};

struct jit_logistic_config_params {
    ov::element::Type src_dt;
    ov::element::Type dst_dt;
    unsigned src_data_size = 0;
    unsigned dst_data_size = 0;
};

struct jit_uni_logistic_kernel {
    void (*ker_)(const jit_args_logistic*) = nullptr;

    void operator()(const jit_args_logistic* args) const {
        assert(ker_);
        ker_(args);
    }

    virtual void create_ker() = 0;

    jit_uni_logistic_kernel() = default;
    virtual ~jit_uni_logistic_kernel() = default;
};

class RegionYolo : public Node {
public:
    RegionYolo(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    [[nodiscard]] bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }

private:
    int classes;
    int coords;
    int num;
    float do_softmax;
    std::vector<int64_t> mask;
    ov::element::Type input_prec, output_prec;

    int block_size;
    std::shared_ptr<jit_uni_logistic_kernel> logistic_kernel = nullptr;
    std::shared_ptr<SoftmaxGeneric> softmax_kernel;

    static inline float logistic_scalar(float src);
    inline void calculate_logistic(size_t start_index, int count, uint8_t* dst_data);
};

}  // namespace ov::intel_cpu::node

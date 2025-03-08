// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "nodes/common/softmax.h"

namespace ov {
namespace intel_cpu {
namespace node {

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
    void (*ker_)(const jit_args_logistic*);

    void operator()(const jit_args_logistic* args) {
        assert(ker_);
        ker_(args);
    }

    virtual void create_ker() = 0;

    jit_uni_logistic_kernel() : ker_(nullptr) {}
    virtual ~jit_uni_logistic_kernel() {}
};

class RegionYolo : public Node {
public:
    RegionYolo(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    bool needPrepareParams() const override;
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

    union U {
        float as_float_value;
        int as_int_value;
    };

    inline float logistic_scalar(float src);
    inline void calculate_logistic(size_t start_index, int count, uint8_t* dst_data);
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov

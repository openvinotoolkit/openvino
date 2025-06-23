// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "ov_ops/moe.hpp"

struct onednn_matmul;

namespace ov::intel_cpu::node {

class MOEExpert : public Node {
public:
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    MOEExpert(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    bool created() const override {
        return getType() == Type::MOE;
    }

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override{};
    void prepareParams() override;
    void execute(const dnnl::stream&) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

private:
    void extractConsts();
    std::shared_ptr<ov::op::internal::MOE> m_moe_op;
    op::internal::MOE::Config m_config;

    struct Weight {
        uint8_t* data;
        uint8_t* scale;
        uint8_t* zp;
        int oc;
        int ic;
        int qg_size; // quantization group size
        dnnl::memory::data_type raw_data_dtype;
        dnnl::memory::data_type raw_scale_dtype;
        dnnl::memory::data_type raw_zp_dtype;

        dnnl::memory mem_data;
        dnnl::memory mem_scale;
        dnnl::memory mem_zp;
    };
    struct ExpertWeights {
        Weight gate;
        Weight up;
        Weight down;
    };
    std::vector<ExpertWeights> m_weights;

    struct ExecutorBase {
        virtual void execute(const dnnl::stream&, MOEExpert*) = 0;
        virtual void reorder_weights(const dnnl::engine&, ExpertWeights * pweight) = 0;
        virtual ~ExecutorBase() = default;
    };
    struct Executor;
    std::shared_ptr<ExecutorBase> m_executor;
};

}  // namespace ov::intel_cpu::node
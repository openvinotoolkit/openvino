// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <unordered_set>
#include <vector>

#include "cpu_memory.h"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "post_ops.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

// tensor parallel config
struct FCTensorParallelConfig {
    int w_rank = -1;
    int w_size = -1;
    int id = 0;
    bool enable_tensor_parallel = false;
    std::shared_ptr<SubMemoryManager> sub_memory = nullptr;
    MemoryPtr cached_splited_weight = nullptr;
    MemoryPtr cached_splited_bias = nullptr;
    MemoryPtr cached_scale = nullptr;
    MemoryPtr cached_zeropoint = nullptr;
    MemoryPtr cached_dst = nullptr;
};

class FullyConnected : public Node {
public:
    FullyConnected(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool canBeInPlace() const override {
        return false;
    }

    int getFusingAxis() const override {
        return getOutputShapeAtPort(0).getRank() == 3 ? 2 : 1;
    }

    const std::vector<impl_desc_type>& getDefaultImplPriority() override;

    size_t descInputNumbers() override {
        return static_cast<size_t>(getOriginalInputsNumber());
    }

    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;

    ov::element::Type getRuntimePrecision() const override;

    bool canFuse(const NodePtr& node) const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    static bool isSupportedCompressedOperation(const std::shared_ptr<ov::Node>& op,
                                               size_t IC,
                                               size_t OC,
                                               size_t G,
                                               ov::element::Type inferencePrecision) noexcept;
    static ov::element::TypeVector getSupportedCompressedWeightsTypes(bool apply_fp8 = false);
    static ov::element::TypeVector getSupportedCompressedActivationsTypes();

    bool isExecutable() const override {
        return !isInputTensorAtPortEmpty(0);
    }

    void prepareParams() override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    bool canBeExecutedInInt8() const override;
    void keepWeightsNonTransposed(bool weightsNonTransposed) {
        this->attrs.weightsNonTransposed = weightsNonTransposed;
    }

    void fuseDecompressionMultiply(const MemoryCPtr& memory);
    void fuseDecompressionSubtract(const MemoryCPtr& memory);

protected:
    void toNumaNodeImpl(int numaID) override;

private:
    enum InputId : size_t {
        DATA = 0,
        WEIGHTS,
        BIAS,
        WEIGHT_SCALES,
        WEIGHT_ZERO_POINTS,
        INPUT_SCALES,
        INPUT_ZERO_POINTS,
        OUTPUT_SCALES,
        OUTPUT_ZERO_POINTS,
    };

    static bool isConstantInput(const std::shared_ptr<const ov::Node>& op, InputId port);

    std::unordered_map<size_t, size_t> m_atoi;  // memory argument id to input id

    void fuseDecompressionConstant(const MemoryCPtr& memory, MemoryCPtr& decompressionValuesPtr);

    void initTensorParallelConfig(const GraphContext::CPtr& context);
    void needUpdateTensorParalelConfig();
    void needPrepareParamsForTensorParallel();
    void initTensorParallelSync();
    void execTensorParallelSync();
    void needSplitMemoryForTensorParallel();

    FCAttrs attrs;
    PostOps postOps;
    MemoryArgs memory;
    ExecutorFactoryPtr<FCAttrs> factory;
    ExecutorPtr executor = nullptr;

    FCTensorParallelConfig tp_cfg;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov

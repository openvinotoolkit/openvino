// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "post_ops.hpp"
#include "openvino/runtime/threading/cpu_message.hpp"

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
    FullyConnected(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override{};
    void execute(dnnl::stream strm) override;
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

    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool canBeExecutedInInt8() const override;
    void keepWeightsNonTransposed(bool weightsNonTransposed) {
        this->attrs.weightsNonTransposed = weightsNonTransposed;
    }

    void fuseDecompressionMultiply(const MemoryCPtr& memory);
    void fuseDecompressionSubtract(const MemoryCPtr& memory);

protected:
    void toNumaNodeImpl(int numaID) override;

private:
    static const size_t DATA_ID = 0;
    static const size_t WEIGHTS_ID = 1;
    static const size_t BIAS_ID = 2;

    ExecutorPtr createExecutor();
    void fuseDecompressionConstant(const MemoryCPtr& memory, MemoryCPtr& decompressionValuesPtr);

    void initTensorParallelConfig(const GraphContext::CPtr context);
    void needUpdateTensorParalelConfig();
    void needPrepareParamsForTensorParallel();
    void initTensorParallelSync();
    void execTensorParallelSync();
    void needSplitMemoryForTensorParallel();
    void needSplitScaleForTensorParallel(const MemoryCPtr& memory);
    void needUpdateScaleForTensorParallel();
    void needSplitZeroPointForTensorParallel(const MemoryCPtr& memory);
    void needUpdateZeroPointForTensorParallel();
    void needUpdateDQScaleForTensorParallel(std::vector<float>& dequantizationScales);

    FCAttrs attrs;
    PostOps postOps;
    MemoryArgs memory;
    ExecutorFactoryPtr<FCAttrs, node::FullyConnected> factory;
    ExecutorPtr executor = nullptr;
    std::string errorPrefix;

    FCTensorParallelConfig tp_cfg;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov

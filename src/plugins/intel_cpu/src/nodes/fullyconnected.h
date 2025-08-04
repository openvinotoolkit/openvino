// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "config.h"
#include "cpu_memory.h"
#include "graph_context.h"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "sub_memory_manager.hpp"

namespace ov::intel_cpu::node {

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

    void getSupportedDescriptors() override {};
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
        return getOriginalInputsNumber();
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
                                               const Config& config) noexcept;
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
    enum InputId : uint8_t {
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
    MemoryArgs memory;
    ExecutorFactoryPtr<FCAttrs> factory;
    ExecutorPtr executor = nullptr;

    FCTensorParallelConfig tp_cfg;
};

}  // namespace ov::intel_cpu::node

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#ifdef NPU_LLVM_BACKEND
#pragma once

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>

#include "intel_npu/common/igraph.hpp"
#include "intel_npu/utils/zero/zero_utils.hpp"
#include "intel_npu/utils/zero/zero_wrappers.hpp"
#include "zero_memory.hpp"
#include "zero_pipeline.hpp"
#include "zero_profiling.hpp"
#include "zero_tensor.hpp"

namespace intel_npu {

struct DynamicPipeline : public Pipeline {
public:
    DynamicPipeline(const Config& config,
                    const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                    const std::shared_ptr<IGraph>& graph,
                    const std::vector<std::vector<std::shared_ptr<ov::ITensor>>>& input_tensors,
                    const std::vector<std::shared_ptr<ov::ITensor>>& output_tensors);

    DynamicPipeline(const DynamicPipeline&) = delete;
    DynamicPipeline& operator=(const DynamicPipeline&) = delete;
    virtual ~DynamicPipeline() = default;

    virtual void push() override;
    virtual void pull() override;
    virtual void reset() const override;

    virtual void update_graph_arguments(uint32_t arg_index, const void* arg_data, size_t byte_size);
    virtual void update_graph_arguments_batching(uint32_t arg_index, const void* arg_data, size_t batch_index);

    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const;

protected:
    const std::vector<std::vector<std::shared_ptr<ov::ITensor>>> _levelZeroInputTensors;
    const std::vector<std::shared_ptr<ov::ITensor>> _levelZeroOutputTensors;
    std::unique_ptr<mlir::ExecutionEngine> _engine;
};

}  // namespace intel_npu
#endif
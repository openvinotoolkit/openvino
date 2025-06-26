// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_memory.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

enum class ScatterUpdateMode : uint8_t { ScatterUpdate, ScatterNDUpdate, ScatterElementsUpdate };

namespace scatter_reductions {
enum class CommonReduction : uint8_t { NONE, SUM, SUB, PROD, MIN, MAX, MEAN };
class ReduceMultiply {
public:
    template <typename DT>
    void operator()(DT* dst_data, const DT* src_data) const {
        *dst_data *= *src_data;
    }
};

class ReduceAdd {
public:
    template <typename DT>
    void operator()(DT* dst_data, const DT* src_data) const {
        *dst_data += *src_data;
    }
};

class ReduceSub {
public:
    template <typename DT>
    void operator()(DT* dst_data, const DT* src_data) const {
        *dst_data -= *src_data;
    }
};

class ReduceMean {
public:
    template <typename DT>
    void operator()(DT* dst_data, const DT* src_data) const {
        *dst_data += *src_data;
    }
};

class ReduceMaximum {
public:
    template <typename DT>
    void operator()(DT* dst_data, const DT* src_data) const {
        *dst_data = std::max(*dst_data, *src_data);
    }
};

class ReduceMinimum {
public:
    template <typename DT>
    void operator()(DT* dst_data, const DT* src_data) const {
        *dst_data = std::min(*dst_data, *src_data);
    }
};

class ReduceNone {
public:
    template <typename DT>
    void operator()(DT* dst_data, const DT* src_data) const {
        *dst_data = *src_data;
    }
};
};  // namespace scatter_reductions

class ScatterUpdate : public Node {
public:
    ScatterUpdate(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    [[nodiscard]] bool created() const override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool canBeInPlace() const override {
        return false;
    }

    [[nodiscard]] bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    [[nodiscard]] bool neverExecute() const override;
    [[nodiscard]] bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    using Reduction = scatter_reductions::CommonReduction;
    template <typename DataType, typename KernelType>
    void scatterElementsUpdate(const MemoryPtr& mem_data,
                               const MemoryPtr& mem_indices,
                               const MemoryPtr& mem_updates,
                               int axis,
                               const KernelType& kernel);
    template <typename DataType>
    void scatterElementsUpdate(const MemoryPtr& mem_data,
                               const MemoryPtr& mem_indices,
                               const MemoryPtr& mem_updates,
                               int axis,
                               const scatter_reductions::ReduceMean& kernel);
    template <typename DataType, typename KernelType>
    void scatterNDUpdate(const MemoryPtr& mem_data,
                         const MemoryPtr& mem_indices,
                         const MemoryPtr& mem_updates,
                         const KernelType& kernel);
    void scatterNDUpdate(const MemoryPtr& mem_data,
                         const MemoryPtr& mem_indices,
                         const MemoryPtr& mem_updates,
                         const scatter_reductions::ReduceNone& kernel);

private:
    void scatterUpdate(uint8_t* indicesPtr, uint8_t* updatePtr, int axis, uint8_t* dstDataPtr);
    void scatterNDUpdate(const MemoryPtr& dstMemPtr, const MemoryPtr& indicesMemPtr, const MemoryPtr& updateMemPtr);
    void scatterElementsUpdate(const MemoryPtr& dstMemPtr,
                               const MemoryPtr& indicesMemPtr,
                               const MemoryPtr& updateMemPtr,
                               int axis);
    inline int64_t getIndicesValue(uint8_t* indices, size_t offset) const;

    ScatterUpdateMode scatterUpdateMode = ScatterUpdateMode::ScatterUpdate;
    enum : uint8_t { DATA_ID, INDICES_ID, UPDATE_ID, AXIS_ID };

    Reduction reduction_type;
    bool use_init_val = true;

    // if axis can be set other than default 0.
    bool axisRelaxed = false;
    size_t dataSize{0LU}, indicesSize{0LU}, axisSize{0LU};
    ov::element::Type dataPrec, indicesPrec, axisPrec;
    // In ov::PartialShape with rank 0 (scalars) is converted to ov::intel_cpu::Shape with rank 1.
    // Add flag set in constructor for workaround for ScatterNDUpdates
    bool isUpdateScalar = false;
};

}  // namespace ov::intel_cpu::node

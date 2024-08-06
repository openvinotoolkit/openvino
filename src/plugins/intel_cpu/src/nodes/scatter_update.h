// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"
#include "openvino/op/scatter_elements_update.hpp"
#include <utility>

namespace ov {
namespace intel_cpu {
namespace node {

enum class ScatterUpdateMode {
    ScatterUpdate,
    ScatterNDUpdate,
    ScatterElementsUpdate
};

namespace scatter_reductions {
enum class CommonReduction { NONE, SUM, SUB, PROD, MIN, MAX, MEAN };
class ReduceMultiply {
public:
    template <typename DT>
    void operator() (DT* dst_data, const DT* src_data) const {
        *dst_data *= *src_data;
    }
};

class ReduceAdd {
public:
    template <typename DT>
    void operator() (DT* dst_data, const DT* src_data) const {
        *dst_data += *src_data;
    }
};

class ReduceSub {
public:
    template <typename DT>
    void operator() (DT* dst_data, const DT* src_data) const {
        *dst_data -= *src_data;
    }
};

class ReduceMean {
public:
    template <typename DT>
    void operator() (DT* dst_data, const DT* src_data) const {
        *dst_data += *src_data;
    }
};

class ReduceMaximum {
public:
    template <typename DT>
    void operator() (DT* dst_data, const DT* src_data) const {
        *dst_data = std::max(*dst_data, *src_data);
    }
};

class ReduceMinimum {
public:
    template <typename DT>
    void operator() (DT* dst_data, const DT* src_data) const {
        *dst_data = std::min(*dst_data, *src_data);
    }
};

class ReduceNone {
public:
    template <typename DT>
    void operator() (DT* dst_data, const DT* src_data) const {
        *dst_data = *src_data;
    }
};
};  // namespace scatter_reductions

class ScatterUpdate : public Node {
public:
    ScatterUpdate(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

    bool needPrepareParams() const override;
    void executeDynamicImpl(dnnl::stream strm) override;

    bool isExecutable() const override;
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    using Reduction = scatter_reductions::CommonReduction;
    template <typename DataType, typename KernelType>
    void scatterElementsUpdate(const MemoryPtr& mem_data, const MemoryPtr& mem_indices, const MemoryPtr& mem_updates, int axis, const KernelType& kernel);
    template <typename DataType>
    void scatterElementsUpdate(const MemoryPtr& mem_data, const MemoryPtr& mem_indices, const MemoryPtr& mem_updates,
                                int axis, const scatter_reductions::ReduceMean& kernel);
    template <typename DataType, typename KernelType>
    void scatterNDUpdate(const MemoryPtr& mem_data, const MemoryPtr& mem_indices, const MemoryPtr& mem_updates, const KernelType& kernel);
    void scatterNDUpdate(const MemoryPtr& mem_data, const MemoryPtr& mem_indices, const MemoryPtr& mem_updates,
                         const scatter_reductions::ReduceNone& kernel);

private:
    void scatterUpdate(uint8_t *indicesPtr, uint8_t *updatePtr, int axis, uint8_t *dstDataPtr);
    void scatterNDUpdate(const MemoryPtr& dstMemPtr, const MemoryPtr& indicesMemPtr, const MemoryPtr& updateMemPtr);
    void scatterElementsUpdate(const MemoryPtr& dstMemPtr, const MemoryPtr& indicesMemPtr, const MemoryPtr& updateMemPtr, int axis);
    inline int64_t getIndicesValue(uint8_t *indices, size_t offset);

    ScatterUpdateMode scatterUpdateMode = ScatterUpdateMode::ScatterUpdate;
    enum { DATA_ID, INDICES_ID, UPDATE_ID, AXIS_ID };

    Reduction reduction_type;
    bool use_init_val = true;

    // if axis can be set other than default 0.
    bool axisRelaxed = false;
    size_t dataSize, indicesSize, axisSize;
    ov::element::Type dataPrec, indicesPrec, axisPrec;
    // In ov::PartialShape with rank 0 (scalars) is converted to ov::intel_cpu::Shape with rank 1.
    // Add flag set in constructor for workaround for ScatterNDUpdates
    bool isUpdateScalar = false;

    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

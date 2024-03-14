// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfloat>
#include <cmath>
#include "node.h"
#include "openvino/op/scatter_elements_update.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

using Reduction = ov::op::v12::ScatterElementsUpdate::Reduction;

enum class ScatterUpdateMode {
    ScatterUpdate,
    ScatterNDUpdate,
    ScatterElementsUpdate
};

namespace math {

template <typename reduced_t>
struct AccumulativeType {
  using type = reduced_t;
};
template <>
struct AccumulativeType<ov::bfloat16> {
  using type = float;
};
template <>
struct AccumulativeType<ov::float16> {
  using type = float;
};

template <typename T>
using AccType = typename AccumulativeType<T>::type;


class ReduceMultiply {
public:
    template <typename DT>
    void operator() (AccType<DT>* dst_data, const DT* src_data) const {
        *dst_data *= AccType<DT>(*src_data);
    }
};

class ReduceAdd {
public:
    template <typename DT>
    void operator() (AccType<DT>* dst_data, const DT* src_data) const {
        *dst_data += AccType<DT>(*src_data);
    }
};

class ReduceMean {
public:
    template <typename DT>
    void operator() (AccType<DT>* dst_data, const DT* src_data) const {
        *dst_data += AccType<DT>(*src_data);
    }
};

class ReduceMaximum {
public:
    template <typename DT>
    void operator() (AccType<DT>* dst_data, const DT* src_data) const {
        *dst_data = std::isnan(*src_data) ? AccType<DT>(*src_data) : std::max(*dst_data, AccType<DT>(*src_data));
    }
};

class ReduceMinimum {
public:
    template <typename DT>
    void operator() (AccType<DT>* dst_data, const DT* src_data) const {
        *dst_data = std::isnan(*src_data) ? AccType<DT>(*src_data) : std::min(*dst_data, AccType<DT>(*src_data));
    }
};

class ReduceNone {
public:
    template <typename DT>
    void operator() (AccType<DT>* dst_data, const DT* src_data) const {
        *dst_data = AccType<DT>(*src_data);
    }
};

};  // namespace math

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

private:
    void scatterUpdate(uint8_t *indicesPtr, uint8_t *updatePtr, int axis, uint8_t *dstDataPtr);
    void scatterNDUpdate(uint8_t *indicesPtr, uint8_t *updatePtr, uint8_t *dstDataPtr);

    template <typename DT, typename reduce_func>
    void scatterElementsUpdate(const MemoryPtr& mem_data, const MemoryPtr& mem_indices, const MemoryPtr& mem_updates, int axis, reduce_func& kernel_func);
    template <typename DT>
    void scatterElementsUpdate(const MemoryPtr& mem_data, const MemoryPtr& mem_indices, const MemoryPtr& mem_updates, int axis, math::ReduceMean& kernel_func);
    inline int64_t getIndicesValue(uint8_t *indices, size_t offset);

    ScatterUpdateMode scatterUpdateMode = ScatterUpdateMode::ScatterUpdate;
    enum { DATA_ID, INDICES_ID, UPDATE_ID, AXIS_ID };

    Reduction reduction_type;
    bool use_init_val = true;

    // if axis can be set other than default 0.
    bool axisRelaxed = false;
    size_t dataSize, indicesSize, axisSize;
    ov::element::Type dataPrec, indicesPrec, axisPrec;

    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

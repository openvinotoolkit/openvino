// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cmath>
#include <common/float16.hpp>
#include <cstddef>
#include <cstdint>
#include <oneapi/dnnl/dnnl.hpp>
#include <type_traits>
#include <vector>

#include "cpu_types.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/eltwise_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/jit/eltwise.h"
#include "nodes/kernels/jit_eltwise_common.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

struct EltwiseRefKey {
    std::vector<VectorDims> inpDims;
    VectorDims outBlkDims;
    ov::element::Type outPrc;
    std::vector<EltwiseData> eltwise_data;

    [[nodiscard]] size_t hash() const;
    bool operator==(const EltwiseRefKey& rhs) const;
};

template <typename T>
class EltwiseRefBaseExecutor : public IEltwiseExecutor {
public:
    EltwiseRefBaseExecutor(const EltwiseRefKey& key);

    [[nodiscard]] const VectorDims& getOutDims() const override;

    [[nodiscard]] size_t getBatchDimIdx() const override;

    void exec(const jit_eltwise_call_args_ptrs& args_ptrs, const VectorDims& dims_out) override;

protected:
    void init_ptr(const jit_eltwise_call_args_ptrs& args_ptrs,
                  const VectorDims& dims_out,
                  std::vector<size_t>& counters,
                  size_t iwork,
                  std::vector<T>& src_f,
                  T*& dst_ptr_f);

    EltwiseData m_opData;
    VectorDims m_dims;
    VectorDims m_src_offsets[MAX_ELTWISE_INPUTS];
    VectorDims m_dst_offsets;
    std::vector<VectorDims> m_inpDims;
    size_t m_fullWorkAmount = 0;
    size_t m_inputNum = 0;
    size_t m_batchDimIdx = 0;

private:
    void initializeDimsAndOffsets(const VectorDims& outBlkDims);
    static void offset_out_calc(VectorDims& offset, const VectorDims& dims);
    static void offset_in_calc(VectorDims& offset, const VectorDims& dims_in, const VectorDims& dims_out);
};

template <typename T, typename... Ts>
constexpr bool one_of_v = (std::is_same_v<T, Ts> || ...);

template <typename T>
constexpr bool supported_eltwise_ref_types_v = one_of_v<T, float, dnnl::impl::float16_t>;

template <typename T, typename Enable = std::enable_if_t<supported_eltwise_ref_types_v<T>>>
class EltwiseRefExecutor : public EltwiseRefBaseExecutor<T> {
public:
    EltwiseRefExecutor(const EltwiseRefKey& key);

    void exec(const jit_eltwise_call_args_ptrs& args_ptrs, const VectorDims& dims_out) override;

    static bool supports([[maybe_unused]] const EltwiseConfig& config);
};

template <typename T>
constexpr bool supported_bitwise_ref_types_v = one_of_v<T, int8_t, uint8_t, int16_t, uint16_t, int32_t>;

template <typename T, typename Enable = std::enable_if_t<supported_bitwise_ref_types_v<T>>>
class BitwiseRefExecutor : public EltwiseRefBaseExecutor<T> {
public:
    BitwiseRefExecutor(const EltwiseRefKey& key);

    void exec(const jit_eltwise_call_args_ptrs& args_ptrs, const VectorDims& dims_out) override;

    static bool isSupportedConfiguration(const EltwiseConfig& config);
};

EltwiseExecutorPtr createEltwiseRefExecutor(const std::vector<VectorDims>& inDims,
                                            const VectorDims& outBlkDims,
                                            const ov::element::Type& outPrc,
                                            const ExecutorContext::CPtr& context,
                                            const EltwiseShapeAgnosticData& shapeAgnosticData);

}  // namespace ov::intel_cpu

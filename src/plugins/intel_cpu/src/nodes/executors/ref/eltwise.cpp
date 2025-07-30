// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise.hpp"

#include <oneapi/dnnl/dnnl_types.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <common/float16.hpp>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "cpu/primitive_attr_postops.hpp"
#include "cpu_types.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/eltwise_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/jit/eltwise.h"
#include "nodes/kernels/jit_eltwise_common.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu {

size_t EltwiseRefKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;
    size_t seed = 0;
    auto hash_combine_eltwiseData = [](size_t seed, const EltwiseData& eltwiseData) {
        seed = hash_combine(seed, eltwiseData.algo);
        seed = hash_combine(seed, eltwiseData.onednnAlgorithm);
        seed = hash_combine(seed, eltwiseData.alpha);
        seed = hash_combine(seed, eltwiseData.beta);
        seed = hash_combine(seed, eltwiseData.gamma);
        return seed;
    };
    std::for_each(eltwise_data.begin(), eltwise_data.end(), [&](const EltwiseData& item) {
        seed = hash_combine_eltwiseData(seed, item);
    });

    seed = get_vector_hash(seed, outBlkDims);
    seed = hash_combine(seed, outPrc.hash());
    for (auto&& item : inpDims) {
        seed = get_vector_hash(seed, item);
    }
    return seed;
}

bool EltwiseRefKey::operator==(const EltwiseRefKey& rhs) const {
    if (inpDims.size() != rhs.inpDims.size()) {
        return false;
    }

    bool result = eltwise_data == rhs.eltwise_data && outPrc == rhs.outPrc;

    if (result) {
        result = result && outBlkDims == rhs.outBlkDims;
        for (size_t i = 0; i < inpDims.size() && result; ++i) {
            result = result && (inpDims[i] == rhs.inpDims[i]);
        }
    }

    return result;
}

static EltwiseExecutorPtr createRefExecutorByPrecision(const EltwiseRefKey& key) {
    switch (key.outPrc) {
    case ov::element::i8:
        return std::make_shared<BitwiseRefExecutor<int8_t>>(key);
    case ov::element::u8:
        return std::make_shared<BitwiseRefExecutor<uint8_t>>(key);
    case ov::element::i16:
        return std::make_shared<BitwiseRefExecutor<int16_t>>(key);
    case ov::element::u16:
        return std::make_shared<BitwiseRefExecutor<uint16_t>>(key);
    case ov::element::i32:
        return std::make_shared<BitwiseRefExecutor<int32_t>>(key);
    case ov::element::f16:
        return std::make_shared<EltwiseRefExecutor<dnnl::impl::float16_t>>(key);
    default:
        // Use float reference executor for any other precision
        return std::make_shared<EltwiseRefExecutor<float>>(key);
    }
}

EltwiseExecutorPtr createEltwiseRefExecutor(const std::vector<VectorDims>& inDims,
                                            const VectorDims& outBlkDims,
                                            const ov::element::Type& outPrc,
                                            const ExecutorContext::CPtr& context,
                                            const EltwiseShapeAgnosticData& shapeAgnosticData) {
    EltwiseRefKey key = {inDims, outBlkDims, outPrc, shapeAgnosticData.eltwise_data};

    auto builder = [&](const EltwiseRefKey& key) {
        return createRefExecutorByPrecision(key);
    };

    auto runtimeCache = context->getRuntimeCache();
    const auto result = runtimeCache->getOrCreate(key, builder);
    const auto& executor = result.first;
    OPENVINO_DEBUG_ASSERT(executor, "Failed to create Eltwise reference executor");

    return executor;
}

// EltwiseRefBaseExecutor implementation
template <typename T>
EltwiseRefBaseExecutor<T>::EltwiseRefBaseExecutor(const EltwiseRefKey& key)
    : m_opData(key.eltwise_data.front()),
      m_inpDims(key.inpDims),
      m_inputNum(m_inpDims.size()) {
    initializeDimsAndOffsets(key.outBlkDims);
}

template <typename T>
const VectorDims& EltwiseRefBaseExecutor<T>::getOutDims() const {
    return m_dims;
}

template <typename T>
size_t EltwiseRefBaseExecutor<T>::getBatchDimIdx() const {
    return m_batchDimIdx;
}

template <typename T>
void EltwiseRefBaseExecutor<T>::exec(const jit_eltwise_call_args_ptrs& args_ptrs, const VectorDims& dims_out) {}

template <typename T>
void EltwiseRefBaseExecutor<T>::init_ptr(const jit_eltwise_call_args_ptrs& args_ptrs,
                                         const VectorDims& dims_out,
                                         std::vector<size_t>& counters,
                                         size_t iwork,
                                         std::vector<T>& src_f,
                                         T*& dst_ptr_f) {
    size_t tmp = iwork;
    for (ptrdiff_t j = dims_out.size() - 1; j >= 0; j--) {
        counters[j] = tmp % dims_out[j];
        tmp /= dims_out[j];
    }

    size_t index_in[MAX_ELTWISE_INPUTS] = {0};
    for (size_t i = 0; i < m_inputNum; i++) {
        index_in[i] = 0;
        for (size_t j = 0; j < counters.size(); j++) {
            index_in[i] += counters[j] * m_src_offsets[i][j];
        }
        index_in[i] /= sizeof(T);
    }

    size_t index_out = 0;
    for (size_t j = 0; j < counters.size(); j++) {
        index_out += counters[j] * m_dst_offsets[j];
    }
    index_out /= sizeof(T);

    for (size_t i = 0; i < m_inputNum; i++) {
        src_f[i] = (reinterpret_cast<const T*>(args_ptrs.src_ptr[i]) + index_in[i])[0];
    }
    dst_ptr_f = reinterpret_cast<T*>(args_ptrs.dst_ptr) + index_out;
}

template <typename T>
void EltwiseRefBaseExecutor<T>::initializeDimsAndOffsets(const VectorDims& outBlkDims) {
    OPENVINO_ASSERT(!m_inpDims.empty() && !m_inpDims.front().empty() && !outBlkDims.empty(),
                    "Can not make Eltwise executor from empty input dims array");

    size_t input_size = m_inpDims.front().size();
    m_batchDimIdx = input_size - outBlkDims.size();

    m_dims.resize(input_size, 1);
    for (size_t i = 0; i < outBlkDims.size(); i++) {
        m_dims[m_dims.size() - 1 - i] = outBlkDims[outBlkDims.size() - 1 - i];
    }

    m_fullWorkAmount = 1;
    for (size_t dim : m_dims) {
        m_fullWorkAmount *= dim;
    }

    // Initialize offsets
    m_dst_offsets.resize(input_size, 1);
    offset_out_calc(m_dst_offsets, m_dims);
    for (size_t j = 0; j < input_size; j++) {
        m_dst_offsets[j] *= sizeof(T);
    }

    for (size_t i = 0; i < m_inputNum; i++) {
        m_src_offsets[i].resize(input_size, 1);
        offset_in_calc(m_src_offsets[i], m_inpDims[i], m_dims);
        for (size_t j = 0; j < input_size; j++) {
            m_src_offsets[i][j] *= sizeof(T);
        }
    }
}

template <typename T>
void EltwiseRefBaseExecutor<T>::offset_out_calc(VectorDims& offset, const VectorDims& dims) {
    int k = 1;
    for (int i = offset.size() - 1; i >= 0; i--) {
        offset[i] = k;
        k *= dims[i];
    }
}

template <typename T>
void EltwiseRefBaseExecutor<T>::offset_in_calc(VectorDims& offset,
                                               const VectorDims& dims_in,
                                               const VectorDims& dims_out) {
    int k = 1;
    for (int i = offset.size() - 1; i >= 0; i--) {
        offset[i] = (dims_in[i] == dims_out[i]) ? k : 0;
        k *= dims_in[i];
    }
}

// EltwiseRefExecutor implementation
template <typename T, typename Enable>
EltwiseRefExecutor<T, Enable>::EltwiseRefExecutor(const EltwiseRefKey& key) : EltwiseRefBaseExecutor<T>(key) {}

template <typename T, typename Enable>
void EltwiseRefExecutor<T, Enable>::exec(const jit_eltwise_call_args_ptrs& args_ptrs, const VectorDims& dims_out) {
    // Handle special cases first
    if (this->m_opData.algo == Algorithm::EltwiseLog) {
        const T* src_ptr_f = reinterpret_cast<const T*>(args_ptrs.src_ptr[0]);
        T* dst_ptr_f = reinterpret_cast<T*>(args_ptrs.dst_ptr);
        parallel_for(this->m_fullWorkAmount, [&](size_t i) {
            dst_ptr_f[i] = logf(src_ptr_f[i]);
        });
        return;
    }

    if (this->m_opData.algo == Algorithm::EltwisePowerStatic) {
        const T* src_ptr_f = reinterpret_cast<const T*>(args_ptrs.src_ptr[0]);
        T* dst_ptr_f = reinterpret_cast<T*>(args_ptrs.dst_ptr);
        if (this->m_opData.alpha == 2) {
            parallel_for(this->m_fullWorkAmount, [&](size_t i) {
                dst_ptr_f[i] = (this->m_opData.beta * src_ptr_f[i] + this->m_opData.gamma) *
                               (this->m_opData.beta * src_ptr_f[i] + this->m_opData.gamma);
            });
        } else {
            parallel_for(this->m_fullWorkAmount, [&](size_t i) {
                dst_ptr_f[i] = powf(this->m_opData.beta * src_ptr_f[i] + this->m_opData.gamma, this->m_opData.alpha);
            });
        }
        return;
    }

    // Generic execution
    std::shared_ptr<dnnl::impl::cpu::ref_eltwise_scalar_fwd_t> ref_eltwise_injector = nullptr;
    if (this->m_opData.onednnAlgorithm != dnnl::algorithm::undef) {
        ref_eltwise_injector = std::make_shared<dnnl::impl::cpu::ref_eltwise_scalar_fwd_t>(
            static_cast<dnnl_alg_kind_t>(this->m_opData.onednnAlgorithm),
            this->m_opData.alpha,
            this->m_opData.beta,
            1.0F);
    }

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0;
        size_t end = 0;
        splitter(this->m_fullWorkAmount, nthr, ithr, start, end);

        std::vector<size_t> counters(dims_out.size(), 0);

        for (size_t iwork = start; iwork < end; ++iwork) {
            std::vector<T> src_f(this->m_inputNum);
            T* dst_ptr_f = nullptr;
            this->init_ptr(args_ptrs, dims_out, counters, iwork, src_f, dst_ptr_f);

            switch (this->m_opData.algo) {
            case Algorithm::EltwiseRelu:
            case Algorithm::EltwiseGeluErf:
            case Algorithm::EltwiseGeluTanh:
            case Algorithm::EltwiseElu:
            case Algorithm::EltwiseTanh:
            case Algorithm::EltwiseSigmoid:
            case Algorithm::EltwiseAbs:
            case Algorithm::EltwiseSqrt:
            case Algorithm::EltwiseSoftRelu:
            case Algorithm::EltwiseClamp:
            case Algorithm::EltwiseSwish:
            case Algorithm::EltwiseHswish:
            case Algorithm::EltwiseMish:
            case Algorithm::EltwiseHsigmoid:
            case Algorithm::EltwiseRoundHalfToEven:
            case Algorithm::EltwiseRoundHalfAwayFromZero:
                *dst_ptr_f = ref_eltwise_injector->compute_scalar(src_f[0]);
                break;
            case Algorithm::EltwiseAdd:
                *dst_ptr_f = src_f[0] + src_f[1];
                break;
            case Algorithm::EltwiseMulAdd:
                *dst_ptr_f = src_f[0] * src_f[1] + src_f[2];
                break;
            case Algorithm::EltwiseSubtract:
                *dst_ptr_f = src_f[0] - src_f[1];
                break;
            case Algorithm::EltwiseMultiply:
                *dst_ptr_f = src_f[0] * src_f[1];
                break;
            case Algorithm::EltwiseDivide:
                *dst_ptr_f = src_f[0] / src_f[1];
                break;
            case Algorithm::EltwiseCeiling:
                *dst_ptr_f = ceilf(src_f[0]);
                break;
            case Algorithm::EltwiseFloor:
                *dst_ptr_f = floorf(src_f[0]);
                break;
            case Algorithm::EltwiseNegative:
                *dst_ptr_f = -src_f[0];
                break;
            case Algorithm::EltwiseFloorMod:
                *dst_ptr_f = src_f[0] - floorf(src_f[0] / src_f[1]) * src_f[1];
                break;
            case Algorithm::EltwiseMod:
                *dst_ptr_f = src_f[0] - truncf(src_f[0] / src_f[1]) * src_f[1];
                break;
            case Algorithm::EltwiseMaximum:
                *dst_ptr_f = std::max(src_f[0], src_f[1]);
                break;
            case Algorithm::EltwiseMinimum:
                *dst_ptr_f = std::min(src_f[0], src_f[1]);
                break;
            case Algorithm::EltwiseExp:
                *dst_ptr_f = expf(src_f[0]);
                break;
            case Algorithm::EltwiseSquaredDifference:
                *dst_ptr_f = powf((src_f[0] - src_f[1]), 2.F);
                break;
            case Algorithm::EltwisePowerDynamic:
                *dst_ptr_f = powf(src_f[0], src_f[1]);
                break;
            case Algorithm::EltwiseEqual:
                *dst_ptr_f = src_f[0] == src_f[1];
                break;
            case Algorithm::EltwiseNotEqual:
                *dst_ptr_f = src_f[0] != src_f[1];
                break;
            case Algorithm::EltwiseGreater:
                *dst_ptr_f = src_f[0] > src_f[1];
                break;
            case Algorithm::EltwiseGreaterEqual:
                *dst_ptr_f = src_f[0] >= src_f[1];
                break;
            case Algorithm::EltwiseLess:
                *dst_ptr_f = src_f[0] < src_f[1];
                break;
            case Algorithm::EltwiseLessEqual:
                *dst_ptr_f = src_f[0] <= src_f[1];
                break;
            case Algorithm::EltwiseLogicalAnd:
                *dst_ptr_f = src_f[0] && src_f[1];
                break;
            case Algorithm::EltwiseLogicalOr:
                *dst_ptr_f = src_f[0] || src_f[1];
                break;
            case Algorithm::EltwiseLogicalXor:
                *dst_ptr_f = (src_f[0] || src_f[1]) - (src_f[0] && src_f[1]);
                break;
            case Algorithm::EltwiseLogicalNot:
                *dst_ptr_f = !src_f[0];
                break;
            case Algorithm::EltwisePrelu:
                *dst_ptr_f = src_f[0] > 0 ? src_f[0] : static_cast<T>(src_f[0] * src_f[1]);
                break;
            case Algorithm::EltwiseErf:
                *dst_ptr_f = std::erf(src_f[0]);
                break;
            case Algorithm::EltwiseSoftSign:
                *dst_ptr_f = src_f[0] / (1 + std::fabs(src_f[0]));
                break;
            // @todo implement proper isinfinite for non-float precisions
            case Algorithm::EltwiseIsFinite:
                *dst_ptr_f = std::isfinite(static_cast<float>(src_f[0]));
                break;
            case Algorithm::EltwiseIsInf:
                *dst_ptr_f = (this->m_opData.alpha && (src_f[0] == -std::numeric_limits<T>::infinity())) ||
                             (this->m_opData.beta && (src_f[0] == std::numeric_limits<T>::infinity()));
                break;
            case Algorithm::EltwiseIsNaN:
                *dst_ptr_f = std::isnan(src_f[0]);
                break;
            case Algorithm::EltwiseSelect:
                *dst_ptr_f = src_f[0] ? src_f[1] : src_f[2];
                break;
            default:
                OPENVINO_THROW("Unsupported operation type for Eltwise executor");
            }
        }
    });
}

template <typename T, typename Enable>
bool EltwiseRefExecutor<T, Enable>::supports([[maybe_unused]] const EltwiseConfig& config) {
    return true;
}

// BitwiseRefExecutor implementation
template <typename T, typename Enable>
BitwiseRefExecutor<T, Enable>::BitwiseRefExecutor(const EltwiseRefKey& key) : EltwiseRefBaseExecutor<T>(key) {}

template <typename T, typename Enable>
void BitwiseRefExecutor<T, Enable>::exec(const jit_eltwise_call_args_ptrs& args_ptrs, const VectorDims& dims_out) {
    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t start = 0;
        size_t end = 0;
        splitter(this->m_fullWorkAmount, nthr, ithr, start, end);

        std::vector<size_t> counters(dims_out.size(), 0);

        for (size_t iwork = start; iwork < end; ++iwork) {
            std::vector<T> src_f(this->m_inputNum);
            T* dst_ptr_f = nullptr;
            this->init_ptr(args_ptrs, dims_out, counters, iwork, src_f, dst_ptr_f);

            switch (this->m_opData.algo) {
            case Algorithm::EltwiseBitwiseAnd:
                *dst_ptr_f = src_f[0] & src_f[1];
                break;
            case Algorithm::EltwiseBitwiseNot:
                *dst_ptr_f = ~src_f[0];
                break;
            case Algorithm::EltwiseBitwiseOr:
                *dst_ptr_f = src_f[0] | src_f[1];
                break;
            case Algorithm::EltwiseBitwiseXor:
                *dst_ptr_f = src_f[0] ^ src_f[1];
                break;
            case Algorithm::EltwiseBitwiseLeftShift:
                *dst_ptr_f = src_f[0] << src_f[1];
                break;
            case Algorithm::EltwiseBitwiseRightShift:
                *dst_ptr_f = src_f[0] >> src_f[1];
                break;
            default:
                OPENVINO_THROW("Unsupported operation type for Bitwise Eltwise executor: ",
                               algToString(this->m_opData.algo));
            }
        }
    });
}

template <typename T, typename Enable>
bool BitwiseRefExecutor<T, Enable>::isSupportedConfiguration(const EltwiseConfig& config) {
    const auto algorithm = config.attrs.data.algo;
    return any_of(algorithm,
                  Algorithm::EltwiseBitwiseAnd,
                  Algorithm::EltwiseBitwiseNot,
                  Algorithm::EltwiseBitwiseOr,
                  Algorithm::EltwiseBitwiseXor,
                  Algorithm::EltwiseBitwiseLeftShift,
                  Algorithm::EltwiseBitwiseRightShift);
}

// Explicit template instantiations
template class EltwiseRefBaseExecutor<float>;
template class EltwiseRefBaseExecutor<dnnl::impl::float16_t>;
template class EltwiseRefBaseExecutor<int8_t>;
template class EltwiseRefBaseExecutor<uint8_t>;
template class EltwiseRefBaseExecutor<int16_t>;
template class EltwiseRefBaseExecutor<uint16_t>;
template class EltwiseRefBaseExecutor<int32_t>;

template class EltwiseRefExecutor<float>;
template class EltwiseRefExecutor<dnnl::impl::float16_t>;

template class BitwiseRefExecutor<int8_t>;
template class BitwiseRefExecutor<uint8_t>;
template class BitwiseRefExecutor<int16_t>;
template class BitwiseRefExecutor<uint16_t>;
template class BitwiseRefExecutor<int32_t>;

}  // namespace ov::intel_cpu

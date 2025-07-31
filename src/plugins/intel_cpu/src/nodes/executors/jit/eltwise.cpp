// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/jit/eltwise.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "cpu_types.h"
#include "memory_desc/blocked_memory_desc.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/implementation_utils.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/jit_eltwise_common.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/general_utils.h"

using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;

#if defined(OPENVINO_ARCH_X86_64)
#    include <cpu/x64/cpu_isa_traits.hpp>

#    include "nodes/kernels/x64/jit_uni_eltwise_generic.hpp"
using namespace ov::intel_cpu::x64;
using namespace dnnl::impl::cpu::x64;
#endif

#if defined(OPENVINO_ARCH_ARM64)
#    include <any>
#    include <cpu/aarch64/cpu_isa_traits.hpp>

#    include "nodes/kernels/aarch64/jit_uni_eltwise_generic.hpp"
#    include "post_ops.hpp"
using namespace ov::intel_cpu::aarch64;
using namespace dnnl::impl::cpu::aarch64;
#endif

#if defined(OPENVINO_ARCH_RISCV64)
#    include "nodes/kernels/riscv64/cpu_isa_traits.hpp"
#    include "nodes/kernels/riscv64/jit_uni_eltwise_generic.hpp"
using namespace ov::intel_cpu::riscv64;
#endif

namespace ov::intel_cpu {

EltwiseJitExecutor::EltwiseJitExecutor(const Key& key)
    : m_useRuntimePtrs(key.implType == EltwiseImplType::optimizedShapeAgnostic) {
    const auto& outBlkDims = key.outBlkDims;
    const auto& outOrder = key.outOrder;

    auto inpDims = key.inpDims;
    const auto& inpPrc = key.inpPrc;
    const auto outPrc = key.outPrc;

    auto collapseLastDims = [](VectorDims& dims, int dimsToCollapse) {
        for (size_t i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
            dims[dims.size() - 1] *= dims[i];
        }

        for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
            dims[i] = dims[i - dimsToCollapse];
        }

        for (int i = dimsToCollapse - 1; i >= 0; i--) {
            dims[i] = 1;
        }
    };

    auto collapseLastOffsets = [](VectorDims& dims, int dimsToCollapse) {
        for (size_t i = dims.size() - 2; i > dims.size() - dimsToCollapse - 2; i--) {
            if (dims[dims.size() - 1] > 0 || dims[i] > 0) {
                dims[dims.size() - 1] =
                    std::max(dims[dims.size() - 1], static_cast<size_t>(1)) * std::max(dims[i], static_cast<size_t>(1));
            } else {
                dims[dims.size() - 1] *= dims[i];
            }
        }

        for (int i = dims.size() - 2; i >= dimsToCollapse; i--) {
            dims[i] = dims[i - dimsToCollapse];
        }

        for (int i = dimsToCollapse - 1; i >= 0; i--) {
            dims[i] = 0;
        }
    };

    if (inpDims.empty()) {
        OPENVINO_THROW("Can not make Eltwise executor from empty input dims array");
    } else if (inpDims.front().empty()) {
        OPENVINO_THROW("Can not make Eltwise executor from empty input dims members");
    }

    jit_eltwise_params jep = {};
    size_t inputsNumber = inpDims.size();

    jep.use_runtime_ptrs = m_useRuntimePtrs;

    jep.input_size = inpDims.front().size();

    jep.dims.resize(jep.input_size, 1);

    if (outBlkDims.empty()) {
        OPENVINO_THROW("Can not make Eltwise executor from empty block dims vector");
    }

    size_t outRank = outBlkDims.size();
    for (size_t i = 0; i < outRank; i++) {
        jep.dims[jep.dims.size() - 1 - i] = outBlkDims[outRank - 1 - i];
    }

    for (const auto& inpDim : inpDims) {
        for (size_t j = 0; j < inpDim.size(); j++) {
            if (inpDim[j] != jep.dims[j] && inpDim[j] != 1) {
                OPENVINO_THROW("Eltwise executor got invalid input/output dims configuration.");
            }
        }
    }

    if (outBlkDims.size() != outOrder.size()) {
        OPENVINO_THROW("Can not make Eltwise executor due to out blocked dims and out order vectors size mismatch.");
    }

    int lastUnchangedAxis = 0;
    size_t oc_size = 0;
    jep.oc_offsets.resize(jep.input_size, 0);
    std::fill(jep.oc_offsets.begin(), jep.oc_offsets.end(), 0);

    auto isFusedWith = [&](Type type_) {
        auto start_itr = key.ops_list.begin();
        std::advance(start_itr, 1);  // apply offset since the first op in the list is the op itself
        return any_of(start_itr, key.ops_list.end(), [=](Type type) {
            return type == type_;
        });
    };

    if (isFusedWith(Type::FakeQuantize)) {
        size_t offset_oc = 1;
        for (int i = outOrder.size() - 1; i >= 0; i--) {
            if (outOrder[i] == 1) {
                int oc_dim_idx = i + (jep.input_size - outOrder.size());
                jep.oc_offsets[oc_dim_idx] = offset_oc;
                offset_oc *= jep.dims[oc_dim_idx];
                if (oc_dim_idx + 1 !=
                    static_cast<int>(jep.input_size)) {  // since in nspc case we can safely collapse the last axis
                    lastUnchangedAxis = oc_dim_idx;
                }
            }
        }
        oc_size = jep.oc_offsets[jep.dims.size() - 1] != 0 ? jep.dims[jep.dims.size() - 1] : 1;
    }

    int maxCollapsedDims = static_cast<int>(jep.dims.size()) - lastUnchangedAxis - 2;

    size_t fullWorkAmount = 1;
    for (size_t dim : jep.dims) {
        fullWorkAmount *= dim;
    }

    m_threadsNum = static_cast<size_t>(parallel_get_max_threads());
    static constexpr size_t minimalJitWorkAmount = 256;
    size_t currentJitWorkAmount = jep.dims[jep.dims.size() - 1];
    int collapsedDims = 0;

    bool hasDifferentDims = false;
    while (!m_useRuntimePtrs && currentJitWorkAmount < minimalJitWorkAmount && currentJitWorkAmount < fullWorkAmount) {
        if (collapsedDims >= maxCollapsedDims) {
            break;
        }

        for (size_t j = 1; j < inpDims.size(); j++) {
            if (inpDims[j].back() != inpDims[0].back()) {
                hasDifferentDims = true;
                break;
            }
        }

        if (oc_size > 1 && oc_size != inpDims[0][inpDims[0].size() - 1]) {
            hasDifferentDims = true;
        }

        bool canCollapse = true;
        for (const auto& inpDim : inpDims) {
            if (inpDim[inpDim.size() - 2] != 1) {
                if (hasDifferentDims) {
                    canCollapse = false;
                    break;
                }
            }
        }

        if (!canCollapse) {
            break;
        }

        size_t nextJitWorkAmount = currentJitWorkAmount * jep.dims[jep.dims.size() - 2];
        if (fullWorkAmount / nextJitWorkAmount >= m_threadsNum) {
            currentJitWorkAmount = nextJitWorkAmount;
            collapsedDims++;

            for (auto& inpDim : inpDims) {
                collapseLastDims(inpDim, 1);
            }
            collapseLastDims(jep.dims, 1);

            if (isFusedWith(Type::FakeQuantize)) {
                collapseLastOffsets(jep.oc_offsets, 1);
            }
        } else {
            break;
        }
    }

    if (inpPrc.size() != inputsNumber) {
        OPENVINO_THROW("Can not make Eltwise executor. Wrong input precisions vector size.");
    }

    if (!m_useRuntimePtrs) {
        m_batchDimIdx = jep.input_size - outBlkDims.size() + collapsedDims;
        m_schedulerWorkAmount = fullWorkAmount / jep.dims[jep.dims.size() - 1];

        auto offset_out_calc = [](VectorDims& offset, const VectorDims& dims) {
            int k = 1;
            for (int i = offset.size() - 1; i >= 0; i--) {
                offset[i] = k;
                k *= dims[i];
            }
        };

        auto offset_in_calc = [](VectorDims& offset, const VectorDims& dims_in, const VectorDims& dims_out) {
            int k = 1;
            for (int i = offset.size() - 1; i >= 0; i--) {
                offset[i] = (dims_in[i] == dims_out[i]) ? k : 0;
                k *= dims_in[i];
            }
        };

        // init offset
        jep.dst_offsets.resize(jep.input_size, 1);
        offset_out_calc(jep.dst_offsets, jep.dims);
        for (size_t j = 0; j < jep.input_size; j++) {
            jep.dst_offsets[j] *= outPrc.size();
        }

        for (size_t i = 0; i < inputsNumber; i++) {
            jep.src_offsets[i].resize(jep.input_size, 1);
            offset_in_calc(jep.src_offsets[i], inpDims[i], jep.dims);
            for (size_t j = 0; j < jep.input_size; j++) {
                jep.src_offsets[i][j] *= inpPrc[i].size();
            }
        }
    }

    jep.inputs_number = inputsNumber;

    for (size_t i = 0; i < inputsNumber; i++) {
        jep.src_prc[i] = inpPrc[i];
        jep.src_size[i] = inpDims[i][inpDims[i].size() - 1];
    }
    jep.dst_prc = outPrc;
    jep.work_amount = jep.dst_size = jep.dims.back();
    jep.oc_size = oc_size;

    std::transform(jep.oc_offsets.begin(), jep.oc_offsets.end(), jep.oc_offsets.begin(), [](size_t& offset) {
        return offset * sizeof(float);
    });

#if defined(OPENVINO_ARCH_X86_64)
    if (mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        m_kernel = std::make_unique<jit_uni_eltwise_generic<dnnl::impl::cpu::x64::avx512_core>>(jep,
                                                                                                key.eltwise_data,
                                                                                                key.ops_list,
                                                                                                key.postOps);
    } else if (mayiuse(dnnl::impl::cpu::x64::avx2)) {
        m_kernel = std::make_unique<jit_uni_eltwise_generic<dnnl::impl::cpu::x64::avx2>>(jep,
                                                                                         key.eltwise_data,
                                                                                         key.ops_list,
                                                                                         key.postOps);
    } else if (mayiuse(dnnl::impl::cpu::x64::sse41)) {
        m_kernel = std::make_unique<jit_uni_eltwise_generic<dnnl::impl::cpu::x64::sse41>>(jep,
                                                                                          key.eltwise_data,
                                                                                          key.ops_list,
                                                                                          key.postOps);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
#endif

#if defined(OPENVINO_ARCH_ARM64)
    if (mayiuse(aarch64::asimd)) {
        m_kernel =
            std::make_unique<jit_uni_eltwise_generic<aarch64::asimd>>(jep, key.eltwise_data, key.ops_list, key.postOps);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
#endif

#if defined(OPENVINO_ARCH_RISCV64)
    if (mayiuse(ov::intel_cpu::riscv64::gv)) {
        m_kernel = std::make_unique<ov::intel_cpu::riscv64::jit_uni_eltwise_generic<ov::intel_cpu::riscv64::gv>>(
            jep,
            key.eltwise_data);
    } else {
        OPENVINO_THROW("Can't create jit eltwise kernel");
    }
#endif

    OPENVINO_ASSERT(m_kernel, "Failed to create jit Eltwise kernel is not created.");
    m_kernel->create_ker();
}

void EltwiseJitExecutor::exec(const jit_eltwise_call_args_ptrs& args_ptrs, const VectorDims& dims_out) {
    if (!m_kernel) {
        OPENVINO_THROW("Can't execute, kernel for eltwise node is not compiled");
    }

    if (m_kernel->jep_.input_size == optimalTensorRank) {
        // Execute Optimized 6D
        auto d6_loop = [&](size_t i0, size_t i1, size_t i2, size_t i3, size_t i4) {
            auto args = jit_eltwise_call_args_indexes();
            args.indexes[0] = i0;
            args.indexes[1] = i1;
            args.indexes[2] = i2;
            args.indexes[3] = i3;
            args.indexes[4] = i4;
            (*m_kernel)(&args_ptrs, &args);
        };

        parallel_nt_static(m_threadsNum, [&](const int ithr, const int nthr) {
            for_5d(ithr, nthr, dims_out[0], dims_out[1], dims_out[2], dims_out[3], dims_out[4], d6_loop);
        });
    } else {
        // Execute Optimized Generic
        if (m_kernel->jep_.use_runtime_ptrs) {
            updateWorkAmount(dims_out);
        }

        parallel_nt(m_threadsNum, [&](const int ithr, const int nthr) {
            size_t start = 0;
            size_t end = 0;
            splitter(m_schedulerWorkAmount, nthr, ithr, start, end);

            std::vector<size_t> counters(dims_out.size() - 1, 0);
            auto args = jit_eltwise_call_args_indexes();
            for (size_t iwork = start; iwork < end; ++iwork) {
                size_t tmp = iwork;
                for (ptrdiff_t j = dims_out.size() - 2; j >= 0; j--) {
                    counters[j] = tmp % dims_out[j];
                    tmp /= dims_out[j];
                }

                for (size_t j = 0; j < counters.size(); j++) {
                    args.indexes[j] = counters[j];
                }

                (*m_kernel)(&args_ptrs, &args);
            }
        });
    }
}

bool EltwiseJitExecutor::supports(const EltwiseAttrs& attrs,
                                  const size_t rank,
                                  [[maybe_unused]] const std::vector<ov::element::Type>& input_precisions,
                                  [[maybe_unused]] const std::vector<ov::element::Type>& output_precisions) {
#if defined(OPENVINO_ARCH_X86_64)
    const auto isISASupportedByJIT = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::sse41);
#elif defined(OPENVINO_ARCH_ARM64)
    const auto isISASupportedByJIT = dnnl::impl::cpu::aarch64::mayiuse(dnnl::impl::cpu::aarch64::asimd);
#elif defined(OPENVINO_ARCH_RISCV64)
    const auto isISASupportedByJIT = ov::intel_cpu::riscv64::mayiuse(ov::intel_cpu::riscv64::gv);
#else
    const auto isISASupportedByJIT = false;
#endif
    // if dim rank is greater than the maximum possible, we should not use JIT execution
    if (rank > MAX_ELTWISE_DIM_RANK) {
        return false;
    }

    if (!isISASupportedByJIT) {
        return false;
    }

    const auto algorithm = attrs.data.algo;
    if (any_of(algorithm,
               Algorithm::EltwiseLog,
               Algorithm::EltwiseBitwiseLeftShift,
               Algorithm::EltwiseBitwiseRightShift)) {
        return false;  // NOLINT(readability-simplify-boolean-expr) since no further checks on x64 are required
    }

#if defined(OPENVINO_ARCH_X86_64)
    return true;

#elif defined(OPENVINO_ARCH_ARM64)
    if (any_of(algorithm,
               Algorithm::EltwiseBitwiseAnd,
               Algorithm::EltwiseBitwiseNot,
               Algorithm::EltwiseBitwiseOr,
               Algorithm::EltwiseBitwiseXor)) {
        return false;
    }

    std::vector<ov::element::Type> supported_input_precisions = std::vector<ov::element::Type>{ov::element::f16,
                                                                                               ov::element::f32,
                                                                                               ov::element::i32,
                                                                                               ov::element::i8,
                                                                                               ov::element::u8};

    std::vector<ov::element::Type> supported_output_precisions = supported_input_precisions;
    if (any_of(algorithm, Algorithm::EltwiseDivide, Algorithm::EltwiseFloor)) {
        supported_input_precisions = std::vector<ov::element::Type>{ov::element::f16, ov::element::f32};
    }

    const auto& postOps = attrs.postOps;
    if (!postOps.empty()) {
        // Divide and Floor (issue #138629) operations are supported for fp32 and fp16 only.
        if (const auto* const scaleShiftPostOp = std::any_cast<ScaleShiftPostOp>(&postOps.back())) {
            if (scaleShiftPostOp->type() == ScaleShiftPostOp::Type::divide) {
                supported_input_precisions = std::vector<ov::element::Type>{ov::element::f16, ov::element::f32};
            }
        } else if (const auto* const activationPostOp = std::any_cast<ActivationPostOp>(&postOps.back())) {
            if (activationPostOp->type() == ActivationPostOp::Type::floor) {
                supported_input_precisions = std::vector<ov::element::Type>{ov::element::f16, ov::element::f32};
            }
        }
    } else {
        supported_output_precisions = supported_input_precisions;
    }

#elif defined(OPENVINO_ARCH_RISCV64)
    if (!any_of(algorithm,
                Algorithm::EltwiseAbs,
                Algorithm::EltwiseAdd,
                Algorithm::EltwiseClamp,
                Algorithm::EltwiseDivide,
                Algorithm::EltwiseElu,
                Algorithm::EltwiseErf,
                Algorithm::EltwiseExp,
                Algorithm::EltwiseFloor,
                Algorithm::EltwiseFloorMod,
                Algorithm::EltwiseGeluErf,
                Algorithm::EltwiseGeluTanh,
                Algorithm::EltwiseHsigmoid,
                Algorithm::EltwiseHswish,
                Algorithm::EltwiseLess,
                Algorithm::EltwiseLogicalOr,
                Algorithm::EltwiseEqual,
                Algorithm::EltwiseLessEqual,
                Algorithm::EltwiseGreaterEqual,
                Algorithm::EltwiseLogicalAnd,
                Algorithm::EltwiseLogicalNot,
                Algorithm::EltwiseLogicalXor,
                Algorithm::EltwiseMaximum,
                Algorithm::EltwiseMinimum,
                Algorithm::EltwiseMish,
                Algorithm::EltwiseMod,
                Algorithm::EltwiseMulAdd,
                Algorithm::EltwiseMultiply,
                Algorithm::EltwiseNegative,
                Algorithm::EltwiseNotEqual,
                Algorithm::EltwisePowerStatic,
                Algorithm::EltwisePrelu,
                Algorithm::EltwiseRelu,
                Algorithm::EltwiseSigmoid,
                Algorithm::EltwiseSqrt,
                Algorithm::EltwiseSubtract)) {
        return false;
    }

    const std::vector<ov::element::Type> supported_input_precisions = {ov::element::f32,
                                                                       ov::element::i32,
                                                                       ov::element::i8,
                                                                       ov::element::u8};
    const auto& supported_output_precisions = supported_input_precisions;
#endif

#if defined(OPENVINO_ARCH_ARM64) || defined(OPENVINO_ARCH_RISCV64)
    const auto check_precisions = [&](const std::vector<ov::element::Type>& input_precisions,
                                      const std::vector<ov::element::Type>& output_precisions) {
        if (!std::all_of(input_precisions.begin(),
                         input_precisions.end(),
                         [&supported_input_precisions](const ov::element::Type& precision) {
                             return contains(supported_input_precisions, precision);
                         })) {
            return false;
        }

        return std::all_of(output_precisions.begin(),
                           output_precisions.end(),
                           [&supported_output_precisions](const ov::element::Type& precision) {
                               return contains(supported_output_precisions, precision);
                           });
    };

    return check_precisions(input_precisions, output_precisions);
#endif

    // Unsupported architectures should return false:
    return false;
}

bool EltwiseJitExecutor::supports(const EltwiseConfig& config) {
    std::vector<ov::element::Type> input_precisions(config.descs.size() - 1);  // -1 for output precision
    std::vector<ov::element::Type> output_precisions{
        config.descs.at(ARG_DST)->getPrecision()};  // -1 for output precision

    for (const auto& [argId, desc] : config.descs) {
        if (argId == ARG_DST) {
            continue;  // Skip output precision
        }

        input_precisions[argId - ARG_SRC] = desc->getPrecision();
    }

    return supports(config.attrs, srcRank(config), input_precisions, output_precisions);
}

const VectorDims& EltwiseJitExecutor::getOutDims() const {
    if (!m_kernel) {
        OPENVINO_THROW("Can't get jit eltwise params, kernel for Eltwise executor is not compiled");
    }
    return m_kernel->jep_.dims;
}

size_t EltwiseJitExecutor::getBatchDimIdx() const {
    return m_batchDimIdx;
}

impl_desc_type EltwiseJitExecutor::implType() {
#if defined(OPENVINO_ARCH_ARM64)
    if (mayiuse(dnnl::impl::cpu::aarch64::asimd)) {
        return impl_desc_type::jit_asimd;
    }
#elif defined(OPENVINO_ARCH_RISCV64)
    if (mayiuse(ov::intel_cpu::riscv64::gv)) {
        return impl_desc_type::jit_gv;
    }
#elif defined(OPENVINO_ARCH_X86_64)
    if (mayiuse(dnnl::impl::cpu::x64::avx512_core)) {
        return impl_desc_type::jit_avx512;
    }

    if (mayiuse(dnnl::impl::cpu::x64::avx2)) {
        return impl_desc_type::jit_avx2;
    }

    if (mayiuse(dnnl::impl::cpu::x64::sse41)) {
        return impl_desc_type::jit_sse42;
    }
#endif
    OPENVINO_THROW("Unsupported architecture for Eltwise JIT executor");
}

std::shared_ptr<EltwiseJitExecutor> EltwiseJitExecutor::create(const MemoryArgs& memory,
                                                               const std::vector<VectorDims>& inDims,
                                                               const VectorDims& outBlkDims,
                                                               const ov::element::Type& outPrc,
                                                               const ExecutorContext::CPtr& context,
                                                               const EltwiseShapeAgnosticData& shapeAgnosticData,
                                                               const EltwiseImplType implType) {
    auto outBlockingDesc = memory.at(ARG_DST)->getDescWithType<BlockedMemoryDesc>();
    const auto& outOrder = outBlockingDesc->getOrder();
    std::vector<ov::element::Type> inpPrc(inDims.size());

    for (const auto& [argId, mem] : memory) {
        if (argId == ARG_DST) {
            continue;
        }

        inpPrc[argId - ARG_SRC] = mem->getPrecision();
    }

    Key key = {shapeAgnosticData.eltwise_data,
               shapeAgnosticData.ops_list,
               outBlkDims,
               outOrder,
               inDims,
               inpPrc,
               outPrc,
               shapeAgnosticData.postOps,
               implType};

    auto builder = [&](const Key& key) {
        return std::make_shared<EltwiseJitExecutor>(key);
    };

    auto runtimeCache = context->getRuntimeCache();
    const auto result = runtimeCache->getOrCreate(key, builder);
    const auto& executor = result.first;
    OPENVINO_DEBUG_ASSERT(executor, "Failed to create Eltwise jit executor");

    return executor;
}

void EltwiseJitExecutor::updateWorkAmount(const VectorDims& dims_out) {
    m_schedulerWorkAmount = 1;
    for (size_t i = 0; i < dims_out.size() - 1; i++) {
        m_schedulerWorkAmount *= dims_out[i];
    }
}

}  // namespace ov::intel_cpu

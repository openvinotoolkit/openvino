// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <common/utils.hpp>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

#include "common/primitive_attr.hpp"
#include "common/primitive_hashing_utils.hpp"
#include "cpu_types.h"
#include "nodes/executors/eltwise_config.hpp"
#include "nodes/executors/eltwise_executor.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/kernels/jit_eltwise_common.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

struct EltwiseShapeAgnosticData {
    std::vector<EltwiseData> eltwise_data;
    std::vector<Type> ops_list;
    dnnl::post_ops postOps;
};

class EltwiseJitExecutor : public IEltwiseExecutor {
    struct Key {
        std::vector<EltwiseData> eltwise_data;
        std::vector<Type> ops_list;
        VectorDims outBlkDims;
        VectorDims outOrder;
        std::vector<VectorDims> inpDims;
        std::vector<ov::element::Type> inpPrc;
        ov::element::Type outPrc;
        dnnl::post_ops postOps;
        EltwiseImplType implType;

        [[nodiscard]] size_t hash() const {
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
            seed = get_vector_hash(seed, ops_list);
            if (implType == EltwiseImplType::optimizedShapeAgnostic) {
                seed = hash_combine(seed, outBlkDims.back() == 1);
                for (auto&& item : inpDims) {
                    seed = hash_combine(seed, item.back() == 1);
                }
            } else {
                seed = get_vector_hash(seed, outOrder);
                seed = get_vector_hash(seed, outBlkDims);
                for (auto&& item : inpDims) {
                    seed = get_vector_hash(seed, item);
                }
            }
            std::for_each(inpPrc.begin(), inpPrc.end(), [&](const ov::element::Type& item) {
                seed = hash_combine(seed, item.hash());
            });
            seed = hash_combine(seed, outPrc.hash());
            seed = get_post_op_hash(seed, *postOps.get());
            seed = hash_combine(seed, implType);
            return seed;
        }

        bool operator==(const Key& rhs) const {
            if (inpDims.size() != rhs.inpDims.size()) {
                return false;
            }

            bool result = eltwise_data == rhs.eltwise_data && ops_list == rhs.ops_list && inpPrc == rhs.inpPrc &&
                          outPrc == rhs.outPrc && *postOps.get() == *rhs.postOps.get() && implType == rhs.implType;

            if (result) {
                if (implType == EltwiseImplType::optimizedShapeAgnostic) {
                    bool broadcast = false;
                    bool rhsBroadcast = false;
                    for (size_t i = 0; i < inpDims.size(); ++i) {
                        broadcast = (inpDims[i].back() == 1);
                        rhsBroadcast = (rhs.inpDims[i].back() == 1);
                        if (broadcast != rhsBroadcast) {
                            return false;
                        }
                    }
                } else {
                    result = result && outOrder == rhs.outOrder && outBlkDims == rhs.outBlkDims;
                    for (size_t i = 0; i < inpDims.size() && result; ++i) {
                        result = result && (inpDims[i] == rhs.inpDims[i]);
                    }
                }
            }

            return result;
        }
    };

public:
    EltwiseJitExecutor(const Key& key);

    void exec(const jit_eltwise_call_args_ptrs& args_ptrs, const VectorDims& dims_out) override;
    [[nodiscard]] const VectorDims& getOutDims() const override;
    [[nodiscard]] size_t getBatchDimIdx() const override;
    [[nodiscard]] static impl_desc_type implType();

    static bool supports(const EltwiseAttrs& attrs,
                         size_t inputRank,
                         const std::vector<ov::element::Type>& input_precisions = {},
                         const std::vector<ov::element::Type>& output_precisions = {});

    static bool supports(const EltwiseConfig& config);

    static std::shared_ptr<EltwiseJitExecutor> create(const MemoryArgs& memory,
                                                      const std::vector<VectorDims>& inDims,
                                                      const VectorDims& outBlkDims,
                                                      const ov::element::Type& outPrc,
                                                      const ExecutorContext::CPtr& context,
                                                      const EltwiseShapeAgnosticData& shapeAgnosticData,
                                                      EltwiseImplType implType);

    static const int optimalTensorRank = 6;

private:
    jit_eltwise_params createKernelParams(const EltwiseAttrs& attrs,
                                          const std::vector<ov::element::Type>& inpPrc,
                                          const ov::element::Type& outPrc);
    void initializeKernel(const EltwiseAttrs& attrs,
                          const std::vector<ov::element::Type>& inpPrc,
                          const ov::element::Type& outPrc,
                          const dnnl::post_ops& post_ops);

    void updateWorkAmount(const VectorDims& dims_out);
    void initializeDimsAndOffsets(const std::vector<VectorDims>& inpDims,
                                  const VectorDims& outBlkDims,
                                  [[maybe_unused]] const VectorDims& outOrder);

    bool m_useRuntimePtrs = false;

    std::unique_ptr<jit_uni_eltwise_kernel> m_kernel;
    size_t m_schedulerWorkAmount = 0;
    size_t m_batchDimIdx = 0;
    size_t m_threadsNum = 0;
};

}  // namespace ov::intel_cpu

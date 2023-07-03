// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "nodes/executors/convert.hpp"
#include "utils/debug_capabilities.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

class ACLConvertExecutor : public ConvertExecutor {
public:
    explicit ACLConvertExecutor(const ExecutorContext::CPtr context);
    bool init(const ConvertParams& convertParams,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst) override;
    impl_desc_type getImplType() const override { return implDescType; };
    ~ACLConvertExecutor() override = default;
protected:
    ConvertParams aclConvertParams;
    bool isCopyOp;
    impl_desc_type implDescType = impl_desc_type::acl;
    arm_compute::Tensor srcTensor, dstTensor;
    std::unique_ptr<arm_compute::NECopy> acl_copy;
    std::unique_ptr<arm_compute::NECast> acl_cast;
};


class ACLConvertExecutorBuilder : public ConvertExecutorBuilder {
public:
    ~ACLConvertExecutorBuilder() = default;
    bool isSupported(const ConvertParams& convertParams,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        if (convertParams.srcPrc != convertParams.dstPrc) {
            if (!one_of(convertParams.srcPrc, InferenceEngine::Precision::I8,
                                              InferenceEngine::Precision::U8,
                                              InferenceEngine::Precision::U16,
                                              InferenceEngine::Precision::I16,
                                              InferenceEngine::Precision::FP16,
                                              InferenceEngine::Precision::I32,
                                              InferenceEngine::Precision::FP32)) {
                DEBUG_LOG("NECast does not support source precision: ", convertParams.srcPrc.name());
                return false;
            }
            if ((convertParams.srcPrc == InferenceEngine::Precision::I8 &&
                !one_of(convertParams.dstPrc, InferenceEngine::Precision::I16,
                                              InferenceEngine::Precision::I32,
                                              InferenceEngine::Precision::FP16,
                                              InferenceEngine::Precision::FP32)) ||
                (convertParams.srcPrc == InferenceEngine::Precision::U8 &&
                !one_of(convertParams.dstPrc, InferenceEngine::Precision::U16,
                                              InferenceEngine::Precision::I16,
                                              InferenceEngine::Precision::I32,
                                              InferenceEngine::Precision::FP16,
                                              InferenceEngine::Precision::FP32)) ||
                (convertParams.srcPrc == InferenceEngine::Precision::U16 &&
                !one_of(convertParams.dstPrc, InferenceEngine::Precision::U8,
                                              InferenceEngine::Precision::U32)) ||
                (convertParams.srcPrc == InferenceEngine::Precision::I16 &&
                !one_of(convertParams.dstPrc, InferenceEngine::Precision::I8,
                                              InferenceEngine::Precision::U8,
                                              InferenceEngine::Precision::I32)) ||
                (convertParams.srcPrc == InferenceEngine::Precision::FP16 &&
                !one_of(convertParams.dstPrc, InferenceEngine::Precision::I8,
                                              InferenceEngine::Precision::FP32,
                                              InferenceEngine::Precision::I32,
                                              InferenceEngine::Precision::U8)) ||
                (convertParams.srcPrc == InferenceEngine::Precision::I32 &&
                !one_of(convertParams.dstPrc, InferenceEngine::Precision::I8,
                                              InferenceEngine::Precision::FP16,
                                              InferenceEngine::Precision::FP32,
                                              InferenceEngine::Precision::U8)) ||
                (convertParams.srcPrc == InferenceEngine::Precision::FP32 &&
                !one_of(convertParams.dstPrc, InferenceEngine::Precision::BF16,
                                              InferenceEngine::Precision::FP16,
                                              InferenceEngine::Precision::I32))) {
                DEBUG_LOG("NECast does not support passed combination of source and destination precisions. ",
                          "source precision: ", convertParams.srcPrc.name(), " destination precsion: " , convertParams.dstPrc.name());
                return false;
            }
        }
        return true;
    }
    ConvertExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<ACLConvertExecutor>(context);
    }
};

} // namespace intel_cpu
} // namespace ov
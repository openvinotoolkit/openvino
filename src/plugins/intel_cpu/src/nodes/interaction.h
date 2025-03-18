// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

struct jit_move_scale_compile_params {
    ov::element::Type src_prc;
    ov::element::Type dst_prc;
    bool with_scales;
    size_t input_size;
    bool broadcast_scales;
};

struct jit_move_scale_call_args {
    const void* p_in;
    void* p_out;
    const void* p_scales;
};

struct jit_uni_move_scale_kernel {
    void (*ker_)(const jit_move_scale_call_args*);

    void operator()(const jit_move_scale_call_args* call_args) {
        assert(ker_);
        ker_(call_args);
    }

    explicit jit_uni_move_scale_kernel(const jit_move_scale_compile_params& jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_move_scale_kernel() {}

    virtual void create_ker() = 0;

    jit_move_scale_compile_params jcp_;
};

class Interaction : public Node {
public:
    Interaction(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);
    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    bool neverExecute() const override;
    bool isExecutable() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override;
    void prepareParams() override;

private:
    void execRef(const dnnl::stream& strm);
    dnnl::primitive prim;
    size_t batchSize = 0;
    size_t featureSize = 0;
    size_t inputSizes = 0;
    size_t outputFeaturesLen = 0;
    size_t interactFeatureSize = 0;
    MemoryPtr inputMemPtr;
    MemoryPtr flatMemPtr;
    MemoryPtr outputMemPtr;
    std::vector<uint32_t> featureSizes;
    ov::element::Type dataPrecision;
    ov::element::Type outputDataType;
    std::vector<float> fqScales;
    std::unique_ptr<jit_uni_move_scale_kernel> moveFeatureKernel;
    std::unique_ptr<jit_uni_move_scale_kernel> moveInteractKernel;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov

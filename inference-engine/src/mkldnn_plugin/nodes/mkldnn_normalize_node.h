// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <mkldnn.hpp>
#include <cassert>

#include <cpu/ref_eltwise.hpp>
#include <cpu/ref_depthwise_injector.hpp>

using namespace InferenceEngine;

namespace MKLDNNPlugin {

struct jit_normalize_config_params {
    bool is_nchw;
    bool is_nhwc;
    bool is_blk;
    bool across_spatial;
    mkldnn::memory::data_type src_dt;
    mkldnn::memory::data_type dst_dt;
    int src_data_size;
    int dst_data_size;
    size_t n, c, h, w;
};

struct jit_normalize_call_args {
    const void *src;
    void *dst;
    const float *modulo;
    const float *fused_factor;
    size_t src_stride;
    size_t dst_stride;
    size_t work_amount;
    size_t oc_off;
};

struct jit_uni_normalize_modulo_kernel {
    void (*ker_)(const jit_normalize_call_args *);

    void operator()(const jit_normalize_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    jit_uni_normalize_modulo_kernel(jit_normalize_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_normalize_modulo_kernel() {}

    virtual void create_ker() = 0;

    jit_normalize_config_params jcp_;
};

struct jit_uni_normalize_kernel {
    void (*ker_)(const jit_normalize_call_args *);

    void operator()(const jit_normalize_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_normalize_kernel(jit_normalize_config_params jcp, const mkldnn_primitive_attr &attr) : ker_(nullptr), jcp_(jcp), attr_(attr) {}
    virtual ~jit_uni_normalize_kernel() {}

    virtual void create_ker() = 0;

    jit_normalize_config_params jcp_;
    const mkldnn_primitive_attr &attr_;
};

class MKLDNNNormalizeL2Node : public MKLDNNNode {
public:
    MKLDNNNormalizeL2Node(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;
    bool canFuse(const MKLDNNNodePtr& node) const override;

private:
    enum class NormEpsMode {
        ADD,
        MAX
    };
    NormEpsMode epsMode = NormEpsMode::ADD;

    float epsApply(const float &modulo) const {
        if (epsMode == NormEpsMode::ADD) {
            return modulo + eps;
        } else if (epsMode == NormEpsMode::MAX) {
            return std::max(modulo, eps);
        } else {
            IE_THROW() << errorPrefix << "has unsupported epsilon mode";
        }
    }

    bool cornerCase = false;

    template<typename T>
    struct NormalizeExecute;

    template <typename in_data_t, typename out_data_t>
    void normalize_nchw(const in_data_t* src_data, out_data_t* dst_data, const InferenceEngine::SizeVector& dims);

    template <typename in_data_t, typename out_data_t>
    void normalize_nchw_ref(const in_data_t* src_data, out_data_t* dst_data, const InferenceEngine::SizeVector& dims);

    template <typename in_data_t, typename out_data_t>
    void normalize_nhwc(const in_data_t* src_data, out_data_t* dst_data, const InferenceEngine::SizeVector& dims);

    template <typename in_data_t, typename out_data_t>
    void normalize_blk(const in_data_t* src_data, out_data_t* dst_data, const InferenceEngine::SizeVector& dims);

    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights = false);
    inline void apply_post_ops_scalar(float &dst_value, int index_c);

    template <typename in_data_t, typename out_data_t>
    void normalize_function(const in_data_t* src_data, out_data_t* dst_data, const InferenceEngine::SizeVector& dims);

    bool across_spatial = true;
    float eps = 1e-10f;

    InferenceEngine::Precision input_prec, output_prec;
    size_t src_data_size, dst_data_size;

    mkldnn::primitive_attr attr;

    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;

    std::shared_ptr<jit_uni_normalize_modulo_kernel> normalize_modulo_kernel;
    std::shared_ptr<jit_uni_normalize_kernel> normalize_kernel;

    std::vector<std::shared_ptr<mkldnn::impl::cpu::ref_eltwise_scalar_fwd_t>> eltwise_injectors_ref;
    std::vector<std::shared_ptr<mkldnn::impl::cpu::ref_depthwise_scalar_fwd_t>> depthwise_injectors_ref;

    jit_normalize_config_params jcp = {};

    static const size_t DATA = 0;
    static const size_t AXES = 1;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin


// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

struct jit_reduce_config_params {
    bool planar_layout;
    Algorithm reduce_mode;
    mkldnn::memory::data_type src_dt;
    mkldnn::memory::data_type dst_dt;
    int src_data_size;
    int dst_data_size;
};

struct jit_reduce_call_args {
    const void *src;
    void *dst;
    size_t work_amount;
    size_t reduce_w = 2;  // only used in planar layout  [1: reduce width dimension]   [0: reduce other dimension] [other value: N/A]
    size_t reduce_c = 2;  // only used in blocked layout [1: reduce channel dimension] [0: reduce other dimension] [other value: N/A]
    const float *divisor; // mean = sum / divisor
};

struct jit_uni_reduce_kernel {
    void (*ker_)(const jit_reduce_call_args *);

    void operator()(const jit_reduce_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_reduce_kernel(jit_reduce_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_reduce_kernel() {}

    virtual void create_ker() = 0;

    jit_reduce_config_params jcp_;
};

struct jit_uni_reduce_post_kernel {
    void (*ker_)(const jit_reduce_call_args *);

    void operator()(const jit_reduce_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    virtual void create_ker() = 0;

    explicit jit_uni_reduce_post_kernel(jit_reduce_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_reduce_post_kernel() {}

    jit_reduce_config_params jcp_;
};

class MKLDNNReduceNode : public MKLDNNNode {
public:
    MKLDNNReduceNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    void reduce_type(const uint8_t *in_ptr, uint8_t *out_ptr, size_t dst_size);
    void reduce_PLN(const uint8_t *in_ptr, uint8_t *out_ptr);
    void reduce_BLK(const uint8_t *in_ptr, uint8_t *out_ptr);
    void reduce_BLK_concern_padding(const uint8_t *in_ptr, uint8_t *out_ptr);
    inline void reduce_kernel_process(const uint8_t *in_p, uint8_t *out_p, size_t work_amount, size_t reduce_w = 2);
    inline void reduce_kernel_post_process(uint8_t *out_ptr);
    inline void init_dst_data(uint8_t *out_ptr, size_t dst_size);
    inline void calc_process_dst_dims(const int32_t *idx_data);
    inline void reduce_ref(const float *in_ptr, float *out_ptr);
    void reduce_ref_process(const float *in_ptr, float *out_ptr, float init_value, std::function<float(float, float)> func);
    inline void reduce_ref_map(float *out_ptr, size_t work_amount_dst, size_t reduced_dims_work_amount);

    size_t blk_size;
    size_t dims_size;
    static const size_t REDUCE_DATA = 0;
    static const size_t REDUCE_INDEXES = 1;
    bool planar_layout = true;
    bool jit_mode = true;
    bool keep_dims = true;
    bool ReduceN, ReduceC, ReduceD, ReduceH, ReduceW;
    size_t IB, IC, ID, IH, IW;
    size_t OB, OC, OD, OH, OW;
    size_t src_data_size, dst_data_size;
    InferenceEngine::Precision input_prec, output_prec;
    InferenceEngine::SizeVector src_dims;
    InferenceEngine::SizeVector src_strides;
    InferenceEngine::SizeVector process_dst_dims;
    InferenceEngine::SizeVector axes_for_reduction;

    std::shared_ptr<jit_uni_reduce_kernel> reduce_kernel;
    std::shared_ptr<jit_uni_reduce_post_kernel> reduce_post_kernel;

    static std::map<const ngraph::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ngraph::Node>& op, MKLDNNReduceNode& node)>> initializers;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin


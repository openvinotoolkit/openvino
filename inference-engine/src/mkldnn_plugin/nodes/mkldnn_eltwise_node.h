// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>
#include <c_types_map.hpp>
#include <memory>

namespace MKLDNNPlugin {

struct jit_eltwise_fq_params {
    int src0_step;
    int src1_step;
    int dst_step;
    mkldnn::memory::data_type src0_dt;
    mkldnn::memory::data_type src1_dt;
    mkldnn::memory::data_type dst_dt;
    int src0_data_size;
    int src1_data_size;
    int dst_data_size;

    InferenceEngine::EltwiseLayer::eOperation eltwise_op;
};

struct jit_eltwise_fq_call_args {
    const void *src0;
    const void *src1;
    void *dst;
    size_t work_amount;
};

struct jit_uni_eltwise_fq_kernel {
    void (*ker_)(const jit_eltwise_fq_call_args *);

    void operator()(const jit_eltwise_fq_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_eltwise_fq_kernel(jit_eltwise_fq_params jep, const mkldnn_primitive_attr &attr) : ker_(nullptr), jep_(jep), attr_(attr) {}
    virtual ~jit_uni_eltwise_fq_kernel() {}

    jit_eltwise_fq_params jep_;
    const mkldnn_primitive_attr &attr_;
};

class MKLDNNEltwiseNode : public MKLDNNNode {
public:
    MKLDNNEltwiseNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, int socket);
    ~MKLDNNEltwiseNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override;

    bool isSum();
    bool isUnitScales();
    bool isWithBroadcast();
    void initOptimalPrimitiveDescriptor() override;

private:
    InferenceEngine::EltwiseLayer::eOperation op;
    std::vector<float> sum_scales;
    bool broadcast = false;
    int batch_dim = 5;
    std::vector<MKLDNNMemoryPtr> PostOpsIntBlobMemory;
    mkldnn::primitive_attr attr;

    std::shared_ptr<jit_uni_eltwise_fq_kernel> eltiwse_fq_kernel;
    jit_eltwise_fq_params jep;

    void jit_eltwise_fq();
    void setPostOps(mkldnn::primitive_attr &attr, bool initWeights);

    template <typename T0, typename T1> void ref_eltwise(int in0, int in1);
    template <typename T0, typename T1, typename T2> void ref_eltwise2(int in0, int in1);
    void dims_calc(int *dims, const MKLDNNDims &edge_dims, bool channels_first);
    void offset_out_calc(int *offset, int *dims);
    void offset_in_calc(int *offset, int *dims_in, int *dims_out);

    template <typename T0, typename T1> void eltwise_add(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_prod(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_max(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_sub(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_min(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_div(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_squared_diff(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_floor_mod(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_pow(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_logical_and(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_logical_or(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_logical_xor(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);

    template <typename T0, typename T1, typename T2> void eltwise_equal(const T0 *src0_ptr, const T1 *src1_ptr, T2 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1, typename T2> void eltwise_not_equal(const T0 *src0_ptr, const T1 *src1_ptr, T2 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1, typename T2> void eltwise_less(const T0 *src0_ptr, const T1 *src1_ptr, T2 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1, typename T2> void eltwise_less_equal(const T0 *src0_ptr, const T1 *src1_ptr, T2 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1, typename T2> void eltwise_greater(const T0 *src0_ptr, const T1 *src1_ptr, T2 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1, typename T2> void eltwise_greater_equal(const T0 *src0_ptr, const T1 *src1_ptr, T2 *dst_ptr, size_t dst_data_size);
};

}  // namespace MKLDNNPlugin


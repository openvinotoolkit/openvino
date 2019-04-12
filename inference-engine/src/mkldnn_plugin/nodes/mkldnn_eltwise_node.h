// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNEltwiseNode : public MKLDNNNode {
public:
    MKLDNNEltwiseNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng);
    ~MKLDNNEltwiseNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override;

    bool isSum();
    bool isUnitScales();
    void initOptimalPrimitiveDescriptor() override;

private:
    static Register<MKLDNNEltwiseNode> reg;
    InferenceEngine::EltwiseLayer::eOperation op;
    std::vector<float> sum_scales;
    bool broadcast = false;

    template <typename T0, typename T1> void ref_eltwise(int in0, int in1);
    void dims_calc(int *dims, const MKLDNNDims &edge_dims);
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
    template <typename T0, typename T1> void eltwise_equal(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_not_equal(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_less(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_less_equal(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_greater(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_greater_equal(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_logical_and(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_logical_or(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
    template <typename T0, typename T1> void eltwise_logical_xor(const T0 *src0_ptr, const T1 *src1_ptr, T0 *dst_ptr, size_t dst_data_size);
};

}  // namespace MKLDNNPlugin


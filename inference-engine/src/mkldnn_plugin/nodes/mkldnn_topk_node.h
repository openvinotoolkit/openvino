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

enum TopKLayoutType {
    topk_ncsp,
    topk_nspc,
    topk_blocked
};

enum TopKAlgorithm {
    topk_bubble_sort,
    topk_bitonic_sort,
    topk_heap_sort
};

struct jit_topk_config_params {
    bool mode_max;
    bool sort_index;
    bool topk_innermost;
    TopKLayoutType layout;
    TopKAlgorithm algorithm;
    mkldnn::memory::data_type data_type;
    InferenceEngine::Precision precision;
    int data_size;
    int blk_size;
    int top_k;
    int work_amout;
    int axis_dim;       // size of topk axis
    int bitonic_size;   // only used for bitonic sort, the smallest power of 2 number bigger than axis_dim
    int bitonic_k_size; // only used for bitonic sort, the smallest power of 2 number bigger than top_k
    int sort_stride;    // memory stride of adjacent elements in sorting
    int blk_stride;     // stride of channel blocks at the same space coordinate, only used in blocked layout with topk on channel
};

struct jit_topk_call_args {
    const void *src;
    void *process;
    void *process_index;
    void *dst;
    void *index;
    size_t work_amount;
};

struct jit_uni_topk_kernel {
    void (*ker_)(const jit_topk_call_args *);

    void operator()(const jit_topk_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_topk_kernel(jit_topk_config_params jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_topk_kernel() {}

    virtual void create_ker() = 0;

    jit_topk_config_params jcp_;
};

class MKLDNNTopKNode : public MKLDNNNode {
public:
    MKLDNNTopKNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNTopKNode() override = default;

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
    void topk_PLN(const uint8_t *in_ptr, uint8_t *out_ptr, uint8_t *dst_idx,
                  uint8_t *process_ptr = NULL, uint8_t *process_idx_ptr = NULL);
    void topk_BLK(const uint8_t *in_ptr, uint8_t *out_ptr, uint8_t *dst_idx,
                  uint8_t *process_ptr = NULL, uint8_t *process_idx_ptr = NULL);
    void topk_ref(const float *in_ptr, float *out_ptr, int32_t *dst_idx);
    inline void topk_kernel_process(const uint8_t *in_p, uint8_t *out_p, uint8_t *src_idx,
                                    uint8_t *process_p, uint8_t *process_idx_p, size_t work_amount);
    inline static int count(InferenceEngine::SizeVector dims, size_t start_ind, size_t end_ind);
    inline static int count(InferenceEngine::SizeVector dims, size_t start_ind = 0);
    void top1_axis(const float* src_data, float* dst_data, int32_t* dst_idx,
                   const InferenceEngine::SizeVector &in_dims, std::function<float(float, float)> compare) const;
    void top1(const float* src_data, float* dst_data, int32_t* dst_idx,
                   const InferenceEngine::SizeVector &in_dims, std::function<float(float, float)> compare) const;
    void topk_axis(const float* src_data, float* dst_data, int32_t* dst_idx,
                   const InferenceEngine::SizeVector &in_dims, std::function<float(float, float)> compare) const;
    void topk(const float* src_data, float* dst_data, int32_t* dst_idx,
                   const InferenceEngine::SizeVector &in_dims, std::function<float(float, float)> compare) const;

    bool topk_innermost = false;
    bool jit_mode = true;
    bool sort_index;
    bool mode_max;
    int axis;
    static const size_t TOPK_DATA = 0;
    static const size_t TOPK_K = 1;
    static const size_t TOPK_INDEX = 1;
    size_t O, A, I;
    size_t N, ICB, OCB, D, H, W;
    size_t blk_size;
    size_t bitonic_size = 1;
    size_t bitonic_k_size = 1;
    size_t count_xmm;
    size_t src_dims_size;
    size_t dst_dims_size;
    size_t data_size;
    size_t axis_dim;
    int top_k = 1;
    int dim, before_num;
    bool is_last_dim = false;

    InferenceEngine::SizeVector src_dims;
    InferenceEngine::SizeVector dst_dims;
    InferenceEngine::SizeVector dst_idx_dims;
    TopKLayoutType layout;
    TopKAlgorithm algorithm;
    mkldnn::memory::data_type data_type;

    std::shared_ptr<jit_uni_topk_kernel> topk_kernel;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin

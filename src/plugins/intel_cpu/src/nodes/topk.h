// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <memory>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

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
    bool mode_max;           // which of the two elements to select. ture: max; false: min
    bool sort_index;         // sort by value or index. true: index; false: value
    bool topk_innermost;     // if topk sorting is applied on innermost dimension or other dimension
    bool bubble_inplace;     // all the elements in sorting is right in the register, no need to load and store for each comparison
    TopKLayoutType layout;   // memory layout
    TopKAlgorithm algorithm; // topk sorting algorithm
    InferenceEngine::Precision precision; // precision
    int data_size;           // data size
    int blk_size;            // block size
    int top_k;               // number of the output elements in the sorting dimension
    int work_amount;         // how many elements are processed when call jit kernel once
    int axis_dim;            // size of topk axis
    int sort_stride;         // memory stride of adjacent elements in sorting
    int bitonic_idx_cnt;     // the repeatedly counted total number of elements in sorting, which equal the total number of comparison x 2
    int bitonic_k_idx_cnt;   // the counterpart of bitonic_idx_cnt, when sort_index == true

    bool operator!=(const jit_topk_config_params& jcp) {
        //these 6 params usually changed when model reshaped, we check it first
        if ( sort_index != jcp.sort_index || algorithm != jcp.algorithm || top_k != jcp.top_k
            || work_amount != jcp.work_amount || axis_dim != jcp.axis_dim || sort_stride != jcp.sort_stride) {
            return true;
        } else if ( mode_max != jcp.mode_max || topk_innermost != jcp.topk_innermost || bubble_inplace != jcp.bubble_inplace
            || layout != jcp.layout || precision != jcp.precision || data_size != jcp.data_size || blk_size != jcp.blk_size
            || bitonic_idx_cnt != jcp.bitonic_idx_cnt || bitonic_k_idx_cnt != jcp.bitonic_k_idx_cnt ) {
            return true;
        }
        return false;
    }

    const jit_topk_config_params& operator=(const jit_topk_config_params& jcp) {
        mode_max = jcp.mode_max;
        sort_index = jcp.sort_index;
        topk_innermost = jcp.topk_innermost;
        bubble_inplace = jcp.bubble_inplace;
        layout = jcp.layout;
        algorithm = jcp.algorithm;
        precision = jcp.precision;
        data_size = jcp.data_size;
        blk_size = jcp.blk_size;
        top_k = jcp.top_k;
        work_amount = jcp.work_amount;
        axis_dim = jcp.axis_dim;
        sort_stride = jcp.sort_stride;
        bitonic_idx_cnt = jcp.bitonic_idx_cnt;
        bitonic_k_idx_cnt = jcp.bitonic_k_idx_cnt;
        return *this;
    }
};

struct jit_topk_call_args {
    const void *src;
    void *process;
    void *process_index;
    void *dst;
    void *index;
    const int *bitonic_idx_buf;
    const int *bitonic_k_idx_buf;
    const int *idx_block_buf;// original idx sequence, repeated by block (eg. 00000000,11111111,...,77777777), only used in bubble sort
    const int *idx_seq_buf;  // original idx sequence (eg. 01234567), only used in bubble sort and heap sort
    size_t axis_dim;         // point to axis_dim, only used in heap sort with dynamic shapes to achieve axis_dim agnosic
    size_t top_k;
    size_t work_amount;
    size_t sort_stride;
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

class TopK : public Node {
public:
    TopK(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);
    ~TopK() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool needShapeInfer() const override;
    std::vector<VectorDims> shapeInfer() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node> &op, std::string &errorMessage) noexcept;

private:
    void topk_process(const uint8_t *in_ptr, uint8_t *out_ptr, uint8_t *dst_idx);
    void topk_ref(const float *in_ptr, float *out_ptr, int32_t *dst_idx);
    inline void topk_kernel_process(const uint8_t *in_p, uint8_t *out_p, uint8_t *src_idx,
                                    uint8_t *process_p, uint8_t *process_idx_p, size_t work_amount);
    inline static int count(InferenceEngine::SizeVector dims, size_t start_ind, size_t end_ind);
    inline static int count(InferenceEngine::SizeVector dims, size_t start_ind = 0);
    inline void bitonic_push_idx(int p, int n, std::vector<int> &vec, int &cnt, bool cmp_val = true);
    void calc_bitonic_idx(size_t n, int &cnt, bool cmp_val);
    void calc_dims_size(const InferenceEngine::SizeVector &layout_dims);
    void topk_ref_process(const float* src_data, float* dst_data, int32_t* dst_idx,
                   const InferenceEngine::SizeVector &in_dims, std::function<float(float, float)> compare) const;
    void preset_params();
    void prepare_original_idx();
    void prepare_JitKernel();

    jit_topk_config_params m_jcp;

    bool topk_innermost;
    bool jit_mode;
    bool sort_index;
    bool mode_max;
    int axis;
    static const size_t TOPK_DATA = 0;
    static const size_t TOPK_K = 1;
    static const size_t TOPK_INDEX = 1;
    size_t O, A, I;
    size_t blk_size;
    size_t data_size;
    size_t axis_dim;
    int top_k;
    int dim, before_num;
    bool bubble_inplace;
    bool preset_params_done;

    InferenceEngine::SizeVector src_dims, dst_dims;
    TopKLayoutType layout;
    TopKAlgorithm algorithm;

    std::vector<int> vec_bitonic_idx;
    std::vector<int> vec_bitonic_k_idx;

    std::vector<int> vec_idx_seq;
    std::vector<int> vec_idx_block;

    std::vector<uint8_t> vec_process_ptr;
    std::vector<uint8_t> vec_process_idx_ptr;

    std::shared_ptr<jit_uni_topk_kernel> topk_kernel;

    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

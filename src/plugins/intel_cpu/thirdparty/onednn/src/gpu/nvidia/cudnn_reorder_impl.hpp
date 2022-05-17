/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2020 Codeplay Software Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_NVIDIA_CUDNN_REORDER_IMPL_HPP
#define GPU_NVIDIA_CUDNN_REORDER_IMPL_HPP

#include "common/type_helpers.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_reorder_generic_t {
public:
    virtual status_t init(const reorder_pd_t *pd) = 0;

    virtual void execute(cudnnHandle_t handle, void *src, void *dst) const = 0;

    virtual ~cudnn_reorder_generic_t() {
        CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, src_desc_);
        CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, dst_desc_);
    }

    int dst_offset_in_bytes() { return dst_offset_in_bytes_; }
    int src_offset_in_bytes() { return src_offset_in_bytes_; }

protected:
    cudnnDataType_t src_data_type_;
    cudnnDataType_t dst_data_type_;
    int ndims_;
    int dims_[DNNL_MAX_NDIMS];
    cudnnTensorDescriptor_t src_desc_;
    cudnnTensorDescriptor_t dst_desc_;
    float alpha_, beta_;
    int dst_offset_in_bytes_ = 0;
    int src_offset_in_bytes_ = 0;
};

// This structure is used when the memory format includes blocking
struct cudnn_reorder_ex_t : public cudnn_reorder_generic_t {
public:
    status_t init(const reorder_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating
        // cudnn descriptors
        memory_desc_wrapper wrap(pd->src_md());
        if (wrap.size() == 0) { return status::success; }
        // Validity checks
        assert(pd->dst_md()->ndims == pd->src_md()->ndims);

        get_format(pd->src_md(), src_format_);
        get_format(pd->dst_md(), dst_format_);
        dst_offset_in_bytes_ = pd->dst_md()->offset0
                * types::data_type_size(pd->dst_md()->data_type);
        src_offset_in_bytes_ = pd->src_md()->offset0
                * types::data_type_size(pd->src_md()->data_type);
        alpha_ = pd->alpha();
        beta_ = pd->beta();

        CHECK(convert_data_type(pd->src_md(), &src_data_type_));
        CHECK(convert_data_type(pd->dst_md(), &dst_data_type_));

        convert_dims(pd->src_md()->padded_dims, dims_, pd->src_md()->ndims);

        ndims_ = pd->dst_md()->ndims > 4 ? pd->dst_md()->ndims : 4;

        // Create and set tensor transform descriptor
        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnCreateTensorTransformDescriptor, &trans_desc_));
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetTensorTransformDescriptor,
                trans_desc_, ndims_, dst_format_, nullptr, nullptr, nullptr,
                cudnnFoldingDirection_t::CUDNN_TRANSFORM_FOLD));
        // Create and set source tensor descriptor
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreateTensorDescriptor, &src_desc_));
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetTensorNdDescriptorEx, src_desc_,
                src_format_, src_data_type_, ndims_, dims_));
        // Create and set destination tensor descriptor
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreateTensorDescriptor, &dst_desc_));
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetTensorNdDescriptorEx, dst_desc_,
                dst_format_, dst_data_type_, ndims_, dims_));
        return status::success;
    }

    void execute(cudnnHandle_t handle, void *src, void *dst) const override {
        // cudnnTransformTensorEx() function is required to support blocking.
        // It requires the output tensor to be in cuDNN supported format.
        CUDNN_EXECUTE_FUNC(cudnnTransformTensorEx, handle, trans_desc_, &alpha_,
                src_desc_, src, &beta_, dst_desc_, dst);
    }

    ~cudnn_reorder_ex_t() {
        CUDNN_EXECUTE_FUNC_V(
                cudnnDestroyTensorTransformDescriptor, trans_desc_);
    }

private:
    cudnnTensorFormat_t src_format_;
    cudnnTensorFormat_t dst_format_;
    cudnnTensorTransformDescriptor_t trans_desc_;

    using cudnn_reorder_generic_t::cudnn_reorder_generic_t;
};

// This structure is used when the memory format does not include blocking
struct cudnn_reorder_stride_t : public cudnn_reorder_generic_t {
public:
    status_t init(const reorder_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating
        // cudnn descriptors
        memory_desc_wrapper wrap(pd->src_md());
        if (wrap.size() == 0) { return status::success; }

        // Validity checks
        assert(pd->dst_md()->ndims == pd->src_md()->ndims);
        dst_offset_in_bytes_ = pd->dst_md()->offset0
                * types::data_type_size(pd->dst_md()->data_type);
        src_offset_in_bytes_ = pd->src_md()->offset0
                * types::data_type_size(pd->src_md()->data_type);
        alpha_ = pd->alpha();
        beta_ = pd->beta();

        convert_dims(pd->dst_md()->dims, dims_, pd->dst_md()->ndims);
        convert_dims(pd->src_md()->format_desc.blocking.strides, src_strides_,
                pd->src_md()->ndims);
        convert_dims(pd->dst_md()->format_desc.blocking.strides, dst_strides_,
                pd->dst_md()->ndims);
        adjust_dim_for_dnn(dims_, pd->dst_md()->ndims, pd->src_md());
        adjust_stride_for_dnn(src_strides_, pd->dst_md()->ndims, pd->src_md());
        adjust_stride_for_dnn(dst_strides_, pd->dst_md()->ndims, pd->dst_md());
        ndims_ = pd->dst_md()->ndims >= 4 ? pd->dst_md()->ndims
                        + pd->dst_md()->format_desc.blocking.inner_nblks
                                          : 4;
        bool vectorized = has_different_block_size(pd->src_md(), pd->dst_md());
        CHECK(convert_data_type(pd->src_md(), &src_data_type_, vectorized));
        CHECK(convert_data_type(pd->dst_md(), &dst_data_type_, vectorized));
        // Create and set source tensor descriptor
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreateTensorDescriptor, &src_desc_));
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetTensorNdDescriptor, src_desc_,
                src_data_type_, ndims_, dims_, src_strides_));
        // Create and set destination tensor descriptor
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreateTensorDescriptor, &dst_desc_));
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetTensorNdDescriptor, dst_desc_,
                dst_data_type_, ndims_, dims_, dst_strides_));
        return status::success;
    }

    void execute(cudnnHandle_t handle, void *src, void *dst) const override {
        // We don't need to specify the format (deducible using the strides)
        // in case of cudnnTransformTensor().
        // For example, this is useful when converting from abcd to bacd
        CUDNN_EXECUTE_FUNC(cudnnTransformTensor, handle, &alpha_, src_desc_,
                src, &beta_, dst_desc_, dst);
    }

private:
    int src_strides_[DNNL_MAX_NDIMS];
    int dst_strides_[DNNL_MAX_NDIMS];

    using cudnn_reorder_generic_t::cudnn_reorder_generic_t;
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

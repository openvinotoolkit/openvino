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

#ifndef GPU_NVIDIA_SYCL_CUDA_ELTWISE_IMPL_HPP
#define GPU_NVIDIA_SYCL_CUDA_ELTWISE_IMPL_HPP

#include "cudnn.h"

#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_eltwise_impl_base_t {

public:
    virtual status_t init(const eltwise_pd_t *pd) = 0;

    virtual void execute(cudnnHandle_t handle, void **x, int size) const = 0;

    virtual status_t create_and_set_act_descriptor() {
        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnCreateActivationDescriptor, &act_desc_));

        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetActivationDescriptor, act_desc_,
                alg_kind, cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN, coef));

        return status::success;
    }

    // Mapping between dnnl algorithm and cuDNN activation mode
    status_t convert_alg_kind(
            alg_kind_t alg_kind, cudnnActivationMode_t *cuda_alg_kind) const {
        switch (alg_kind) {
            case alg_kind::eltwise_relu:
                *cuda_alg_kind = cudnnActivationMode_t::CUDNN_ACTIVATION_RELU;
                break;
            case alg_kind::eltwise_bounded_relu:
                *cuda_alg_kind
                        = cudnnActivationMode_t::CUDNN_ACTIVATION_CLIPPED_RELU;
                break;
            case alg_kind::eltwise_tanh:
                *cuda_alg_kind = cudnnActivationMode_t::CUDNN_ACTIVATION_TANH;
                break;
            case alg_kind::eltwise_elu:
                *cuda_alg_kind = cudnnActivationMode_t::CUDNN_ACTIVATION_ELU;
                break;
            case alg_kind::eltwise_logistic:
                *cuda_alg_kind
                        = cudnnActivationMode_t::CUDNN_ACTIVATION_SIGMOID;
                break;
            default: return status::unimplemented;
        }
        return status::success;
    }

    virtual ~cudnn_eltwise_impl_base_t() {
        if (act_desc_) {
            CUDNN_EXECUTE_FUNC_V(cudnnDestroyActivationDescriptor, act_desc_);
        }
    }

protected:
    int ndims;
    cudnnActivationDescriptor_t act_desc_ = nullptr;
    cudnnActivationMode_t alg_kind;
    // alpha and beta are post operation scaling parameters used by cuDNN
    float alpha = 1;
    float beta = 0;
    // coef in cuDNN is use for Relu (is equal to zero) and BRelu (represents
    // the bound)
    double coef = 0;
};

struct cudnn_eltwise_fwd_impl_t : public cudnn_eltwise_impl_base_t {
public:
    status_t init(const eltwise_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating
        // cudnn descriptors
        if (has_zero_dims(pd->src_md()->dims, pd->ndims())) {
            return status::success;
        }
        if (pd->ndims() > CUDNN_DIM_MAX) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        // Obtain source and destination dimensions, strides and datatype
        convert_dims(pd->src_md()->padded_dims, dims_, pd->ndims());
        convert_dims(pd->src_md()->format_desc.blocking.strides, strides_,
                pd->ndims());
        CHECK(convert_data_type(pd->src_md(), &data_type_));

        // Get cuDNN activation mode
        alg_kind_t alg = pd->desc()->alg_kind;
        auto alg_ok = convert_alg_kind(alg, &alg_kind);
        if (alg_ok != status::success) { return status::unimplemented; }
        coef = pd->desc()->alpha;

        CHECK(create_and_set_tensor_descriptor(
                &tensor_desc_, data_type_, ndims, dims_, strides_));
        CHECK(create_and_set_act_descriptor());
        return status::success;
    }

    void execute(cudnnHandle_t handle, void **x, int size) const override {
        // Confirm that 2 arguments were passed src and dst
        assert(size == 2);
        CUDNN_EXECUTE_FUNC(cudnnActivationForward, handle, act_desc_, &alpha,
                tensor_desc_, x[0], &beta, tensor_desc_, x[1]);
    }

    ~cudnn_eltwise_fwd_impl_t() {
        CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, tensor_desc_);
    }

private:
    int strides_[DNNL_MAX_NDIMS];
    int dims_[DNNL_MAX_NDIMS];
    cudnnDataType_t data_type_;
    cudnnTensorDescriptor_t tensor_desc_;
};

struct cudnn_eltwise_bwd_impl_t : public cudnn_eltwise_impl_base_t {

public:
    status_t init(const eltwise_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with creating
        // cudnn descriptors
        if (memory_desc_wrapper(pd->desc()->data_desc).has_zero_dim())
            return status::success;

        if (pd->ndims() > CUDNN_DIM_MAX) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        // Obtain dimension and strides for the backward eltwise operation
        convert_dims(pd->src_md()->padded_dims, dims_, pd->ndims());

        convert_dims(pd->src_md()->format_desc.blocking.strides, strides_,
                pd->ndims());

        alg_kind_t alg = pd->desc()->alg_kind;
        auto alg_ok = convert_alg_kind(alg, &alg_kind);
        if (alg_ok != status::success) { return status::unimplemented; }
        coef = pd->desc()->alpha;

        // Check validity of input
        assert(pd->diff_dst_md()->data_type == pd->src_md()->data_type);
        assert(pd->diff_dst_md()->data_type == pd->diff_src_md()->data_type);

        CHECK(convert_data_type(pd->src_md(), &data_type_));

        CHECK(create_and_set_tensor_descriptor(
                &tensor_desc_src_, data_type_, ndims, dims_, strides_));
        CHECK(create_and_set_tensor_descriptor(
                &tensor_diff_desc_, data_type_, ndims, dims_, strides_));
        CHECK(create_and_set_act_descriptor());
        return status::success;
    }

    void execute(cudnnHandle_t handle, void **x, int size) const override {
        // Assert that 3 arguments were passed src, diff_dst and diff_src
        assert(size == 3);
        void *dy = x[1];
        void *dx = x[2];
        CUDNN_EXECUTE_FUNC(cudnnActivationBackward, handle, act_desc_, &alpha,
                tensor_desc_src_, x[0], tensor_diff_desc_, dy, tensor_desc_src_,
                x[0], &beta, tensor_diff_desc_, dx);
    }

    ~cudnn_eltwise_bwd_impl_t() {
        CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, tensor_desc_src_);
        CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, tensor_diff_desc_);
    }

private:
    int dims_[DNNL_MAX_NDIMS];
    int strides_[DNNL_MAX_NDIMS];
    cudnnTensorDescriptor_t tensor_diff_desc_;
    cudnnDataType_t data_type_;
    cudnnTensorDescriptor_t tensor_desc_src_;
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

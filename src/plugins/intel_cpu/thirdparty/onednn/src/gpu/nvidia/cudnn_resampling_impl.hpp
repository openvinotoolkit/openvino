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

#ifndef GPU_NVIDIA_CUDNN_RESAMPLING_IMPL_HPP
#define GPU_NVIDIA_CUDNN_RESAMPLING_IMPL_HPP

#include <cudnn.h>

#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_resampling_impl_base_t {
    virtual ~cudnn_resampling_impl_base_t() {
        for (int i = 0; i < NUM_IO; ++i) {
            if (tensor_descs_[i]) {
                CUDNN_EXECUTE_FUNC_V(
                        cudnnDestroyTensorDescriptor, tensor_descs_[i]);
            }
        }

        if (st_desc_) {
            CUDNN_EXECUTE_FUNC_V(
                    cudnnDestroySpatialTransformerDescriptor, st_desc_);
        }
    }

    virtual status_t init(resampling_pd_t *pd) = 0;

    virtual void execute(
            cudnnHandle_t handle, const std::vector<void *> &args) const = 0;

    int ndims() { return ndims_; }

    status_t create_and_set_st_desc() {
        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnCreateSpatialTransformerDescriptor, &st_desc_));

        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetSpatialTransformerNdDescriptor,
                st_desc_, CUDNN_SAMPLER_BILINEAR, data_types_[dst], ndims_,
                dims_[dst]));

        return status::success;
    }

    enum io { src, dst, NUM_IO };
    int dims_[NUM_IO][DNNL_MAX_NDIMS];
    int strides_[NUM_IO][DNNL_MAX_NDIMS];
    cudnnDataType_t data_types_[NUM_IO];
    cudnnTensorDescriptor_t tensor_descs_[NUM_IO] = {};
    cudnnSpatialTransformerDescriptor_t st_desc_;
    int ndims_;
    const float alpha_ = 1.f, beta_ = 0.f;
};

struct cudnn_resampling_fwd_impl_t : public cudnn_resampling_impl_base_t {
    status_t init(resampling_pd_t *pd) override {
        ndims_ = std::max(4, pd->ndims());

        if (ndims_ > 4) return status::unimplemented;

        cudnnTensorFormat_t src_format, dst_format;
        CHECK(get_format(pd->src_md(), dst_format));
        CHECK(get_format(pd->dst_md(), src_format));
        convert_dims(pd->src_md()->padded_dims, dims_[src], pd->ndims());
        convert_dims(pd->src_md()->format_desc.blocking.strides, strides_[src],
                pd->ndims(), 4,
                (dst_format != CUDNN_TENSOR_NHWC ? 1 : dims_[src][1]));
        convert_dims(pd->dst_md()->padded_dims, dims_[dst], pd->ndims());
        convert_dims(pd->dst_md()->format_desc.blocking.strides, strides_[dst],
                pd->ndims(), 4,
                (dst_format != CUDNN_TENSOR_NHWC ? 1 : dims_[dst][1]));

        CHECK(convert_data_type(pd->src_md(), &data_types_[src]));
        CHECK(convert_data_type(pd->dst_md(), &data_types_[dst]));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[src],
                data_types_[src], ndims_, dims_[src], strides_[src]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[dst],
                data_types_[dst], ndims_, dims_[dst], strides_[dst]));

        CHECK(create_and_set_st_desc());
        return status::success;
    }

    void execute(cudnnHandle_t handle,
            const std::vector<void *> &args) const override {

        CUDNN_EXECUTE_FUNC(cudnnSpatialTfSamplerForward, handle, st_desc_,
                &alpha_, tensor_descs_[src], args[0], args[1], &beta_,
                tensor_descs_[dst], args[2]);
    }
};

struct cudnn_resampling_bwd_impl_t : public cudnn_resampling_impl_base_t {

    status_t init(resampling_pd_t *pd) override {
        ndims_ = std::max(4, pd->ndims());

        if (ndims_ > 4) return status::unimplemented;

        cudnnTensorFormat_t src_format, dst_format;
        CHECK(get_format(pd->diff_src_md(), dst_format));
        CHECK(get_format(pd->diff_dst_md(), src_format));
        convert_dims(pd->diff_src_md()->padded_dims, dims_[src], pd->ndims());
        convert_dims(pd->diff_src_md()->format_desc.blocking.strides,
                strides_[src], pd->ndims(), 4,
                (dst_format != CUDNN_TENSOR_NHWC ? 1 : dims_[src][1]));
        convert_dims(pd->diff_dst_md()->padded_dims, dims_[dst], pd->ndims());
        convert_dims(pd->diff_dst_md()->format_desc.blocking.strides,
                strides_[dst], pd->ndims(), 4,
                (dst_format != CUDNN_TENSOR_NHWC ? 1 : dims_[dst][1]));

        CHECK(convert_data_type(pd->diff_src_md(), &data_types_[src]));
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[dst]));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[src],
                data_types_[src], ndims_, dims_[src], strides_[src]));
        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[dst],
                data_types_[dst], ndims_, dims_[dst], strides_[dst]));

        CHECK(create_and_set_st_desc());
        auto wrap = memory_desc_wrapper(pd->diff_src_md());

        auto grid_size = pd->MB() * pd->OH() * pd->OW() * 2;
        auto grid_size_in_byte = grid_size * wrap.data_type_size();
        // cuDNN does not allow the dgrid to be NULL ptr. Although we dont
        // need to compute dgrid since the theta is not comming from a
        // local network, we have to set that since Nvidia does not accept
        // so we allocate an scratchpad for dgrid
        pd->scratchpad_registry().registrar().book(
                memory_tracking::names::key_none, grid_size_in_byte, size_t(1));
        return status::success;
    }

    void execute(cudnnHandle_t handle,
            const std::vector<void *> &args) const override {
        // we are not backpropagating for the grid here.
        // So both alpha and beta are zero and the dgrid value
        //  wont be used
        CUDNN_EXECUTE_FUNC(cudnnSpatialTfSamplerBackward, handle, st_desc_,
                &alpha_, tensor_descs_[src], args[0], &beta_,
                tensor_descs_[src], args[0], &beta_, tensor_descs_[dst],
                args[1], args[2], &beta_, args[3]);
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

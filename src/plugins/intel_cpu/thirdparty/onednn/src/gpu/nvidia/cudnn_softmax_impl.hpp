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

#ifndef GPU_NVIDIA_CUDNN_SOFTMAX_IMPL_HPP
#define GPU_NVIDIA_CUDNN_SOFTMAX_IMPL_HPP

#include "cudnn.h"

#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_softmax_impl_base_t {
    cudnnDataType_t data_type;
    int ndims;
    cudnnSoftmaxAlgorithm_t alg_kind;
    // cuDNN only supports softmax on channel dimension
    cudnnSoftmaxMode_t mode = cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL;
    // oneDNN softmax primitive doesn't support any post-ops or attributes,
    // hence we can set alpha = 1 and beta = 0 for all cases
    float alpha = 1.0f;
    float beta = 0.0f;

    virtual ~cudnn_softmax_impl_base_t() {}

    virtual status_t init(const softmax_pd_t *pd) = 0;

    virtual void execute(cudnnHandle_t handle, void **x, int size) const = 0;

    // Mapping between dnnl algorithm and cuDNN softmax algorithm
    status_t convert_alg_kind(
            bool is_log_softmax, cudnnSoftmaxAlgorithm_t *cuda_alg_kind) const {
        if (is_log_softmax) {
            *cuda_alg_kind = cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG;
        } else {
            *cuda_alg_kind = cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE;
        }
        return status::success;
    }

    status_t convert_dims_softmax(const dims_t &orig_dims, int *modified_dims,
            int axis, int ndims, format_tag_t tag,
            cudnnTensorFormat_t &format) const {

        // Initialise all dims to 1
        for (int i = 0; i < 4; i++) {
            modified_dims[i] = 1;
        }
        if (axis == 1) {
            // Copy dimensions into the new array
            format = tag == dnnl_nhwc ? cudnnTensorFormat_t::CUDNN_TENSOR_NHWC
                                      : cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
            int num_dims = ndims < 4 ? ndims : 4;
            for (int i = 0; i < num_dims; i++) {
                modified_dims[i] = orig_dims[i];
            }
            for (int i = 4; i < ndims; i++) {
                modified_dims[3] *= orig_dims[i];
            }
            return status::success;
        }
        format = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
        switch (tag) {
            case dnnl_cn: {
                modified_dims[0] = orig_dims[1];
                modified_dims[1] = orig_dims[0];
                break;
            }
            case dnnl_nchw: {
                switch (axis) {
                    case 0:
                        modified_dims[1] = orig_dims[axis];
                        modified_dims[2] = orig_dims[1];
                        for (int i = 2; i < ndims; i++) {
                            modified_dims[3] *= orig_dims[i];
                        }
                        break;
                    default: {
                        for (int i = 0; i < axis; i++) {
                            modified_dims[0] *= orig_dims[i];
                        }
                        modified_dims[1] = orig_dims[axis];
                        if (axis == ndims - 1) { return status::success; }
                        for (int i = axis + 1; i < ndims; i++) {
                            modified_dims[2] *= orig_dims[i];
                        }
                        break;
                    }
                }
                break;
            }
            case dnnl_nhwc:
                switch (axis) {
                    case 0:
                        modified_dims[1] = orig_dims[0];
                        for (int i = 1; i < ndims; i++) {
                            modified_dims[2] *= orig_dims[i];
                        }
                        break;
                    case 2:
                        modified_dims[0] = orig_dims[0];
                        modified_dims[1] = orig_dims[2];
                        for (int i = 3; i < ndims; i++) {
                            modified_dims[2] *= orig_dims[i];
                        }
                        modified_dims[3] = orig_dims[1];
                        break;
                    case 3:
                        modified_dims[0] = orig_dims[0] * orig_dims[2];
                        modified_dims[1] = orig_dims[3];
                        modified_dims[2] = ndims == 4 ? 1 : orig_dims[4];
                        modified_dims[3] = orig_dims[1];
                        break;
                }
                break;
            default: return status::unimplemented;
        }
        return status::success;
    }

    status_t convert_tag(const memory_desc_t *md, format_tag_t &tag) const {
        const memory_desc_wrapper mem_wrapper(md);
        if (mem_wrapper.matches_one_of_tag(format_tag::ba)) {
            tag = dnnl_cn;
        } else if (mem_wrapper.matches_one_of_tag(format_tag::ab,
                           format_tag::abc, format_tag::abcd, format_tag::abcde,
                           format_tag::abcdef)) {
            tag = dnnl_nchw;
        } else if (mem_wrapper.matches_one_of_tag(format_tag::acb,
                           format_tag::acdb, format_tag::acdeb)) {
            tag = dnnl_nhwc;
        } else {
            return status::unimplemented;
        }
        return status::success;
    }
};

struct cudnn_softmax_fwd_impl_t : public cudnn_softmax_impl_base_t {
    int dims[DNNL_MAX_NDIMS];
    cudnnTensorDescriptor_t tensor_desc;
    cudnnTensorFormat_t format;

    status_t init(const softmax_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with
        // creating cudnn descriptors
        if (has_zero_dims(pd->src_md(0)->dims, pd->ndims())) {
            return status::success;
        }

        if (pd->ndims() > CUDNN_DIM_MAX) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        format_tag_t tag;
        CHECK(convert_tag(pd->src_md(), tag));
        CHECK(convert_dims_softmax(pd->src_md()->padded_dims, dims, pd->axis(),
                pd->ndims(), tag, format));

        convert_alg_kind(pd->is_logsoftmax(), &alg_kind);

        assert(pd->src_md()->data_type == pd->dst_md()->data_type);

        CHECK(convert_data_type(pd->src_md(), &data_type));

        CHECK(create_and_set_tensor_descriptor_ex(
                &tensor_desc, format, data_type, 4, dims));
        return status::success;
    }

    void execute(cudnnHandle_t handle, void **x, int size) const override {
        // Confirm that 2 arguments were passed, src and dst
        assert(size == 2);
        CUDNN_EXECUTE_FUNC(cudnnSoftmaxForward, handle, alg_kind, mode, &alpha,
                tensor_desc, x[0], &beta, tensor_desc, x[1]);
    }

    ~cudnn_softmax_fwd_impl_t() {
        CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, tensor_desc);
    }
};

struct cudnn_softmax_bwd_impl_t : public cudnn_softmax_impl_base_t {
    int dims[DNNL_MAX_NDIMS];
    int dims_dst[DNNL_MAX_NDIMS];
    cudnnTensorDescriptor_t tensor_dst_desc;
    cudnnTensorDescriptor_t tensor_diff_desc;
    cudnnTensorFormat_t format;

    status_t init(const softmax_pd_t *pd) override {
        // If any of the dimensions are 0 we should not continue with
        // creating cudnn descriptors
        if (memory_desc_wrapper(pd->desc()->diff_desc).has_zero_dim())
            return status::success;

        if (pd->ndims() > CUDNN_DIM_MAX) { return status::invalid_arguments; }
        ndims = pd->ndims() < 4 ? 4 : pd->ndims();

        format_tag_t tag;
        CHECK(convert_tag(pd->dst_md(), tag));
        CHECK(convert_dims_softmax(pd->dst_md()->padded_dims, dims_dst,
                pd->axis(), pd->ndims(), tag, format));
        CHECK(convert_dims_softmax(pd->diff_src_md()->padded_dims, dims,
                pd->axis(), pd->ndims(), tag, format));

        convert_alg_kind(pd->is_logsoftmax(), &alg_kind);

        assert(pd->diff_dst_md()->data_type == pd->dst_md()->data_type);
        assert(pd->diff_dst_md()->data_type == pd->diff_src_md()->data_type);

        CHECK(convert_data_type(pd->dst_md(), &data_type));

        CHECK(create_and_set_tensor_descriptor_ex(
                &tensor_dst_desc, format, data_type, 4, dims_dst));
        CHECK(create_and_set_tensor_descriptor_ex(
                &tensor_diff_desc, format, data_type, 4, dims));
        return status::success;
    }

    void execute(cudnnHandle_t handle, void **x, int size) const override {
        // Assert that 3 arguments were passed src, diff_dst and diff_src
        assert(size == 3);
        CUDNN_EXECUTE_FUNC(cudnnSoftmaxBackward, handle, alg_kind, mode, &alpha,
                tensor_dst_desc, x[0], tensor_diff_desc, x[1], &beta,
                tensor_diff_desc, x[2]);
    }

    ~cudnn_softmax_bwd_impl_t() {
        CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, tensor_dst_desc);
        CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, tensor_diff_desc);
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef GPU_NVIDIA_CUDNN_INNER_PRODUCT_IMPL_HPP
#define GPU_NVIDIA_CUDNN_INNER_PRODUCT_IMPL_HPP

#include "cublas_v2.h"
#include "cudnn.h"

#include "common/type_helpers.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {
namespace {
inline void get_4d_tensor_descriptor(
        const memory_desc_t *mem_desc1, int *dims, int *strides) {
    memory_desc_t mem_desc = *mem_desc1;

    // Forcing tensors dims less than 4 to be 4 {n c h w};
    using namespace format_tag;
    auto set_dim = [&]() {
        if (mem_desc.ndims == 3) {
            mem_desc.ndims = 4;
            mem_desc.dims[3] = mem_desc.dims[2];
            mem_desc.dims[2] = 1;
            mem_desc.padded_dims[3] = mem_desc.padded_dims[2];
            mem_desc.padded_dims[2] = 1;
        } else if (mem_desc.ndims == 2) {
            mem_desc.ndims = 4;
            mem_desc.dims[3] = 1;
            mem_desc.dims[2] = 1;
            mem_desc.padded_dims[3] = 1;
            mem_desc.padded_dims[2] = 1;
        }
    };
    // Forcing strides < 4 to be 4
    if (memory_desc_matches_tag(mem_desc, nwc)) {
        set_dim();
        //  promoting nwc(owi) to NHWC = {wc 1 c} to {wc 1 wc c}
        mem_desc.format_desc.blocking.strides[3]
                = mem_desc.format_desc.blocking.strides[2];
        mem_desc.format_desc.blocking.strides[2]
                = mem_desc.format_desc.blocking.strides[0];
        assert(memory_desc_matches_tag(mem_desc, nhwc)
                && "Tag is not set to NHWC");
    } else if (memory_desc_matches_tag(mem_desc, ncw)) {
        set_dim();
        // promoting ncw(oiw) to NCHW = {wc w 1} to {wc w w 1}
        mem_desc.format_desc.blocking.strides[3]
                = mem_desc.format_desc.blocking.strides[2];
        mem_desc.format_desc.blocking.strides[2]
                = mem_desc.format_desc.blocking.strides[1];
        assert(memory_desc_matches_tag(mem_desc, nchw)
                && "Tag is not set to NCHW");
    } else if (memory_desc_matches_tag(mem_desc, wio)) {
        set_dim();
        // promoting wcn(wio) to HWCN = {1 n nc} to {1 n ncw nc}
        mem_desc.format_desc.blocking.strides[3]
                = mem_desc.format_desc.blocking.strides[2];
        mem_desc.format_desc.blocking.strides[2] *= mem_desc.dims[3];
        assert(memory_desc_matches_tag(mem_desc, hwio)
                && " Tag is not set to HWIO");
    } else if (memory_desc_matches_tag(mem_desc, nc)) {
        set_dim();
        // fixing strides
        // promoting nc(oi) to NCHW = {c 1} to {c 1 1 1}
        mem_desc.format_desc.blocking.strides[2]
                = mem_desc.format_desc.blocking.strides[1];
        mem_desc.format_desc.blocking.strides[3]
                = mem_desc.format_desc.blocking.strides[1];
        assert(memory_desc_matches_tag(mem_desc, nchw)
                && " Tag is not set to NCHW");
    } else if (memory_desc_matches_tag(mem_desc, cn)) {
        set_dim();
        // fixing strides cn(oi) to HWCN = {1 n} to {1 n nc nc}.
        // Note that CHWN exists as well, but for inner product
        // we convert it to HWCN. Other primitives may need
        // different conversion.
        mem_desc.format_desc.blocking.strides[2]
                = mem_desc.format_desc.blocking.strides[1]
                * mem_desc.padded_dims[1];
        mem_desc.format_desc.blocking.strides[3]
                = mem_desc.format_desc.blocking.strides[2];
        assert(memory_desc_matches_tag(mem_desc, hwio)
                && " Tag is not set to NCHW");
    }
    convert_dnnl_dims_array(mem_desc.dims, dims, mem_desc.ndims);
    convert_dnnl_dims_array(
            mem_desc.format_desc.blocking.strides, strides, mem_desc.ndims);
}
} // namespace
struct cudnn_inner_product_impl_base_t {
    // The io enum requires the weights be the last parameter to ensure
    // tensor_descs is contiguous.
    enum io { src = 0, bia, dst, wei, NUM_IO };
    cudnnDataType_t data_types_[NUM_IO + 1]; // +1 data-type for accumulation
    int ndims_;
    int dims_[NUM_IO][DNNL_MAX_NDIMS];
    // one extra stride added for transform filter
    int strides_[NUM_IO + 1][DNNL_MAX_NDIMS];

    cudnnTensorDescriptor_t tensor_descs_[NUM_IO - 1] = {};

    size_t workspace_size_ = 0;
    float alpha_ = 1, beta_ = 0;
    bool with_bias_;
    bool scale_bias_ = false;
    bool with_relu_ = false, with_eltwise_ = false, with_sum_ = false;
    bool filter_using_spatial_format_ = false;

    virtual bool need_to_transform_filter() const {
        return filter_using_spatial_format_;
    }

    virtual bool ip_using_scratchpad() const { return (workspace_size_ > 0); }
    bool conv_using_scale_scratchpad() const { return scale_bias_; }

    void set_bias_dims(cudnnTensorFormat_t format, int ndims, int bias_dim) {
        // Set the dimensions and strides for the bias.
        // Note that the second dimension of bias and the first dimension
        // of filter should be equal, as cuDNN always stores dimensions in
        // NCDHW order. The first dimension of filter must be equal to the
        // second dimension of bias
        for (size_t i = 0; i < ndims; ++i) {
            dims_[io::bia][i] = 1;
            strides_[io::bia][i] = (format != CUDNN_TENSOR_NHWC ? 1 : bias_dim);
        }
        dims_[io::bia][1] = bias_dim;
        strides_[io::bia][1] = 1;
        strides_[io::bia][0] = bias_dim;
    }
    virtual status_t init(engine_t * /*engine*/, inner_product_pd_t * /*pd*/,
            bool /*with_relu*/, bool /*with_eltwise*/, bool /*with_sum */,
            bool /*using_fused_path_for_blocking*/)
            = 0;

    virtual void execute(cudnnHandle_t /*handle*/,
            cublasHandle_t /*cublas_handle*/,
            const std::vector<void *> & /*args*/) const = 0;
};

struct cudnn_inner_product_fwd_base_t : public cudnn_inner_product_impl_base_t {
    float output_scales_; // alpha in gemm
    float sum_scale_; // beta in gemm
    float eltwise_alpha(const inner_product_pd_t *pd) const {
        const int eltwise_idx
                = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return with_eltwise_
                ? pd->attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha
                : 0.0f;
    }
    float sum_scale(const inner_product_pd_t *pd) const {
        const int sum_idx = pd->attr()->post_ops_.find(primitive_kind::sum);
        return with_sum_ ? pd->attr()->post_ops_.entry_[sum_idx].sum.scale
                         : 0.0f;
    }

    dnnl::impl::alg_kind_t eltwise_algorithm_kind(
            const inner_product_pd_t *pd) const {
        const int eltwise_idx
                = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return pd->attr()->post_ops_.entry_[eltwise_idx].eltwise.alg;
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

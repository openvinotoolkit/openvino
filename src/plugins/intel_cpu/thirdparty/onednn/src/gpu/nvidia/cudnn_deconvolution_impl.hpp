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

#ifndef GPU_NVIDIA_CUDNN_DECONVOLUTION_IMPL_HPP
#define GPU_NVIDIA_CUDNN_DECONVOLUTION_IMPL_HPP

#include "cudnn.h"

#include "common/c_types_map.hpp"
#include "common/deconvolution_pd.hpp"
#include "gpu/nvidia/cudnn_convolution_pd.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_deconvolution_bwd_bias_impl_t {
protected:
    enum io { y = 0, bias, NUM_IO };
    memory_desc_t dnnl_descs[NUM_IO];
    cudnnTensorDescriptor_t descs[NUM_IO];
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    int strides[NUM_IO][DNNL_MAX_NDIMS];
    int ndims[NUM_IO];
    cudnnDataType_t data_types[NUM_IO];

public:
    ~cudnn_deconvolution_bwd_bias_impl_t() {
        for (size_t i = 0; i < NUM_IO; i++) {
            if (descs[i]) {
                CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, descs[i]);
            }
        }
    }

    status_t init(const memory_desc_t *dst, const memory_desc_t *bia) {
        dnnl_descs[y] = *dst;
        dnnl_descs[bias] = *bia;

        ndims[y] = dnnl_descs[y].ndims;
        ndims[bias] = dnnl_descs[bias].ndims;
        convert_dims(dnnl_descs[y].padded_dims, dims[y], ndims[y]);
        CHECK(convert_data_type(&dnnl_descs[y], &data_types[y]));
        CHECK(convert_data_type(&dnnl_descs[bias], &data_types[bias]));
        convert_dims(dnnl_descs[y].format_desc.blocking.strides, strides[y],
                ndims[y]);
        ndims[y] = std::max(4, ndims[y]);
        convert_dims(dnnl_descs[bias].format_desc.blocking.strides,
                strides[bias], ndims[bias], ndims[y]);
        convert_dims(dnnl_descs[bias].padded_dims, dims[bias], ndims[bias],
                ndims[y]);
        std::swap(dims[bias][0], dims[bias][1]);
        ndims[bias] = ndims[y];
        CHECK(create_and_set_tensor_descriptor(
                &descs[y], data_types[y], ndims[y], dims[y], strides[y]));
        CHECK(create_and_set_tensor_descriptor(&descs[bias], data_types[bias],
                ndims[bias], dims[bias], strides[bias]));

        return status::success;
    }

    void execute_bias(cudnnHandle_t handle, void *y, void *bias) const {
        const float bias_alpha = 1.0f;
        const float bias_beta = 0.0f;
        CUDNN_EXECUTE_FUNC_V(cudnnConvolutionBackwardBias, handle, &bias_alpha,
                descs[io::y], y, &bias_beta, descs[io::bias], bias);
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

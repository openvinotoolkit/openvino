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

#ifndef GPU_NVIDIA_CUDNN_GEMM_INNER_PRODUCT_IMPL_HPP
#define GPU_NVIDIA_CUDNN_GEMM_INNER_PRODUCT_IMPL_HPP

#include "cublas_v2.h"
#include "cudnn.h"

#include "common/type_helpers.hpp"
#include "gpu/nvidia/cudnn_inner_product_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

// GEMM Implementation
struct cudnn_gemm_inner_product_base_t {
protected:
    int m_, n_, k_, lda_, ldb_, ldc_;
    cublasOperation_t trans_a_, trans_b_;
    // compute_type is always equal to c_type_;
    // if datatype is f16 or s8 and bias is presented the compute type must be
    // f32 and we need to do the operation in f32
    cudaDataType_t a_type_, b_type_, c_type_,
            // Despite the claim in cuBlas
            // (https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmEx)
            // for the support of fp16 computation when all the types are fp16,
            // in cublas 10.1, and 10.2, if the fp16 is chosen as a
            // computation mode, it silently does no computation. So we force
            // computation type to be f32 in order to get the correct result.
            // This can be reverted when the bug in cublas is fixed.
            compute_type_ = CUDA_R_32F;
    cublasGemmAlgo_t algo_ = CUBLAS_GEMM_DEFAULT;
    status_t get_cublas_data_type(
            const cudnnDataType_t &cudnn_dt, cudaDataType_t &blas_dt) const {
        switch (cudnn_dt) {
            case CUDNN_DATA_FLOAT: blas_dt = CUDA_R_32F; return status::success;
            case CUDNN_DATA_HALF: blas_dt = CUDA_R_16F; return status::success;
            case CUDNN_DATA_INT8: blas_dt = CUDA_R_8I; return status::success;
            case CUDNN_DATA_INT8x4: blas_dt = CUDA_R_8I; return status::success;
            default: return status::unimplemented;
        }
        return status::unimplemented;
    }
};

struct cudnn_gemm_inner_product_fwd_impl_t
    : public cudnn_inner_product_fwd_base_t,
      public cudnn_gemm_inner_product_base_t,
      public cudnn_conv_filter_adjustment_base_t {

    cudnnActivationDescriptor_t act_desc_;
    bool use_acc_dst_;
    cudnnTensorDescriptor_t y_acc_desc_;
    bool need_reorder_;

    bool ip_using_scratchpad() const override { return (use_acc_dst_ > 0); }
    virtual bool need_to_transform_filter() const override {
        return need_reorder_;
    }

    virtual status_t init(engine_t *, inner_product_pd_t *pd, bool with_relu,
            bool with_eltwise, bool with_sum, bool need_reorder) override {
        need_reorder_ = need_reorder;
        // GEMM is column major, here the data is row major.
        // By switching the weight and source we convert the row major to
        // column major without transposing matrices.
        // B * A = C, where B is weight, A is src and C is dst
        bool wie_tr = (pd->weights_md()->format_desc.blocking.strides[0] != 1);
        CHECK(convert_data_type(pd->src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->weights_md(0), &data_types_[io::wei]));
        if (need_reorder) {
            cudnnTensorFormat_t source_format;
            CHECK(get_format(pd->src_md(), source_format));
            ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
            get_4d_tensor_descriptor(
                    pd->weights_md(0), dims_[io::wei], strides_[io::wei]);
            set_filter_format(
                    ndims_, dims_[io::wei], strides_[NUM_IO], source_format);
            CHECK(init_filter_transformation(data_types_[io::wei], ndims_,
                    dims_[io::wei], strides_[io::wei], strides_[NUM_IO]));

            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_none,
                    memory_desc_wrapper(pd->weights_md(0)).size(), size_t(1));
            wie_tr = strides_[NUM_IO][0] != 1;
        }

        trans_a_ = wie_tr ? CUBLAS_OP_T : CUBLAS_OP_N;
        trans_b_ = CUBLAS_OP_N;
        int ic = pd->IC_total_padded();
        int oc = pd->OC();
        int mb = pd->MB();
        n_ = mb;
        k_ = ic;
        m_ = oc;
        lda_ = wie_tr ? k_ : m_;
        ldb_ = k_;
        ldc_ = m_;
        with_bias_ = pd->with_bias();
        with_eltwise_ = with_eltwise || with_relu;
        with_relu_ = with_eltwise;
        use_acc_dst_ = ((pd->dst_md()->data_type == data_type::s8)
                || (with_bias_
                        && pd->weights_md(1)->data_type
                                != pd->dst_md()->data_type));
        // this must be applied on bias if exists.
        output_scales_ = pd->attr()->output_scales_.scales_[0]; // alpha
        with_sum_ = with_sum;
        // scaling factor to add the previous destination value to the current
        // computation. This is equivalent of
        sum_scale_ = sum_scale(pd);
        ndims_ = 4;

        bool input_is_blocked
                = pd->src_md()->format_desc.blocking.inner_blks[0] == 4
                && pd->weights_md(0)->format_desc.blocking.inner_blks[0] == 4;
        if (input_is_blocked) { // since we flatten the tensor and use gemm
            // we dont care about the blocked data type
            data_types_[io::src] = CUDNN_DATA_INT8;
            data_types_[io::wei] = CUDNN_DATA_INT8;
            data_types_[io::dst] = CUDNN_DATA_INT8;
        } else {
            CHECK(convert_data_type(pd->dst_md(), &data_types_[io::dst]));
        }
        CHECK(get_cublas_data_type(data_types_[io::wei], a_type_));
        CHECK(get_cublas_data_type(data_types_[io::src], b_type_));

        c_type_ = (data_types_[io::dst] == CUDNN_DATA_HALF && !use_acc_dst_)
                ? CUDA_R_16F
                : CUDA_R_32F;
        get_4d_tensor_descriptor(
                pd->dst_md(), dims_[io::dst], strides_[io::dst]);

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::dst],
                data_types_[io::dst], ndims_, dims_[io::dst],
                strides_[io::dst]));

        if (with_bias_) {
            CHECK(convert_data_type(pd->weights_md(1), &data_types_[io::bia]));
            // format is always nchw
            set_bias_dims(CUDNN_TENSOR_NCHW, ndims_, pd->OC());

            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::bia],
                    data_types_[io::bia], ndims_, dims_[io::bia],
                    strides_[io::bia]));
        }
        if (use_acc_dst_) {
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                    memory_desc_wrapper(pd->dst_md()).size(), size_t(1));
            CHECK(create_and_set_tensor_descriptor(&y_acc_desc_,
                    CUDNN_DATA_FLOAT, ndims_, dims_[io::dst],
                    strides_[io::dst]));
        } else {
            y_acc_desc_ = tensor_descs_[io::dst];
        }
        if (with_eltwise_) { CHECK(create_and_set_op_descriptor(pd)); }
        return status::success;
    }

    void execute(cudnnHandle_t cudnn_handle, cublasHandle_t cublas_handle,
            const std::vector<void *> &args) const override {
        assert(args.size() == 7);
        auto x = args[0], w = args[1], b = args[2], y = args[3],
             workspace = args[4];
        auto w_arg = w;
        if (need_reorder_) {
            void *transformed_w = args[5];
            transform_filter(cudnn_handle, w, transformed_w);
            w_arg = transformed_w;
        }
        auto y_dst = use_acc_dst_ ? workspace : y;
        auto sum_scale = use_acc_dst_ ? 0.0f : sum_scale_;
        // do gemm
        CUBLAS_EXECUTE_FUNC(cublasGemmEx, cublas_handle, trans_a_, trans_b_, m_,
                n_, k_, &output_scales_, w_arg, a_type_, lda_, x, b_type_, ldb_,
                &sum_scale, y_dst, c_type_, ldc_, compute_type_, algo_);

        if (with_bias_) {

            CUDNN_EXECUTE_FUNC(cudnnAddTensor, cudnn_handle, &output_scales_,
                    tensor_descs_[io::bia], b, &alpha_, y_acc_desc_, y_dst);
        }
        if (use_acc_dst_) {
            CUDNN_EXECUTE_FUNC(cudnnTransformTensor, cudnn_handle, &alpha_,
                    y_acc_desc_, y_dst, &sum_scale_, tensor_descs_[io::dst], y);
        }
        if (with_eltwise_) {
            CUDNN_EXECUTE_FUNC(cudnnActivationForward, cudnn_handle, act_desc_,
                    &alpha_, tensor_descs_[io::dst], y, &beta_,
                    tensor_descs_[io::dst], y);
        }
    }

    status_t create_and_set_op_descriptor(const inner_product_pd_t *pd) {

        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnCreateActivationDescriptor, &act_desc_));

        cudnnActivationMode_t act_mode;
        switch (eltwise_algorithm_kind(pd)) {
            case alg_kind::eltwise_tanh:
                act_mode = CUDNN_ACTIVATION_TANH;
                break;
            case alg_kind::eltwise_elu: act_mode = CUDNN_ACTIVATION_ELU; break;
            case alg_kind::eltwise_relu:
                act_mode = CUDNN_ACTIVATION_RELU;
                break;
            case alg_kind::eltwise_logistic:
                act_mode = CUDNN_ACTIVATION_SIGMOID;
                break;
            case alg_kind::eltwise_bounded_relu:
                act_mode = CUDNN_ACTIVATION_CLIPPED_RELU;
                break;
            default: return status::unimplemented;
        }
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetActivationDescriptor, act_desc_,
                act_mode, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                eltwise_alpha(pd)));

        return status::success;
    }
};

struct cudnn_gemm_inner_product_bwd_data_impl_t
    : public cudnn_inner_product_impl_base_t,
      public cudnn_gemm_inner_product_base_t,
      public cudnn_conv_filter_adjustment_base_t {
    bool need_reorder_;

    virtual bool need_to_transform_filter() const override {
        return need_reorder_;
    }

    virtual status_t init(engine_t *, inner_product_pd_t *pd,
            bool /*with_relu*/, bool /*with_eltwise*/, bool /*with_sum */,
            bool need_reorder) override {
        need_reorder_ = need_reorder;

        // GEMM is column major, here the data is row major.
        // By switching the weight and source we convert the row major to
        // column major without transposing matrices.
        // B * A = C, where B is weight, A is d_dst and C is d_src
        bool wie_tr = (pd->weights_md(0)->format_desc.blocking.strides[0] == 1);
        CHECK(convert_data_type(pd->diff_src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->weights_md(0), &data_types_[io::wei]));
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[io::dst]));
        if (need_reorder) {
            cudnnTensorFormat_t diff_source_format_;
            CHECK(get_format(pd->diff_src_md(), diff_source_format_));
            ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
            get_4d_tensor_descriptor(
                    pd->weights_md(0), dims_[io::wei], strides_[io::wei]);
            set_filter_format(ndims_, dims_[io::wei], strides_[NUM_IO],
                    diff_source_format_);
            CHECK(init_filter_transformation(data_types_[io::wei], ndims_,
                    dims_[io::wei], strides_[io::wei], strides_[NUM_IO]));

            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_none,
                    memory_desc_wrapper(pd->weights_md(0)).size(), size_t(1));
            wie_tr = strides_[NUM_IO][0] == 1;
        }
        trans_a_ = wie_tr ? CUBLAS_OP_T : CUBLAS_OP_N;
        trans_b_ = CUBLAS_OP_N;
        int ic = pd->IC_total_padded();
        int oc = pd->OC();
        int mb = pd->MB();
        n_ = mb;
        k_ = oc;
        m_ = ic;
        lda_ = wie_tr ? k_ : m_;
        ldb_ = k_;
        ldc_ = m_;
        CHECK(get_cublas_data_type(data_types_[io::wei], a_type_));
        CHECK(get_cublas_data_type(data_types_[io::dst], b_type_));
        CHECK(get_cublas_data_type(data_types_[io::src], c_type_));
        return status::success;
    }
    void execute(cudnnHandle_t cudnn_handle, cublasHandle_t cublas_handle,
            const std::vector<void *> &args) const override {
        assert(args.size() == 5);
        auto dx = args[0], w = args[1], dy = args[2];
        auto w_arg = w;
        if (need_reorder_) {
            void *transformed_w = args[4];
            transform_filter(cudnn_handle, w, transformed_w);
            w_arg = transformed_w;
        }
        // do gemm
        CUBLAS_EXECUTE_FUNC(cublasGemmEx, cublas_handle, trans_a_, trans_b_, m_,
                n_, k_, &alpha_, w_arg, a_type_, lda_, dy, b_type_, ldb_,
                &beta_, dx, c_type_, ldc_, compute_type_, algo_);
    }
};

struct cudnn_gemm_inner_product_bwd_weights_impl_t
    : public cudnn_inner_product_impl_base_t,
      public cudnn_gemm_inner_product_base_t,
      public cudnn_conv_filter_adjustment_base_t {
    cudnnReduceTensorDescriptor_t reduceTensorDesc_ = nullptr;
    bool wie_tr_;
    bool need_reorder_;

    virtual bool need_to_transform_filter() const override {
        return need_reorder_;
    }

    virtual ~cudnn_gemm_inner_product_bwd_weights_impl_t() {
        if (reduceTensorDesc_) {
            CUDNN_EXECUTE_FUNC_V(
                    cudnnDestroyReduceTensorDescriptor, reduceTensorDesc_);
        }
    }
    status_t create_and_set_reduce_descriptor() {
        CUDNN_EXECUTE_FUNC_S(
                cudnnCreateReduceTensorDescriptor, &reduceTensorDesc_);
        CUDNN_EXECUTE_FUNC_S(cudnnSetReduceTensorDescriptor, reduceTensorDesc_,
                CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
                CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
        return status::success;
    }
    virtual status_t init(engine_t *engine, inner_product_pd_t *pd,
            bool /*with_relu*/, bool /*with_eltwise*/, bool /*with_sum */,
            bool need_reorder) override {
        need_reorder_ = need_reorder;
        with_bias_ = pd->with_bias();

        // GEMM is column major, here the data is row major.
        // By switching the weight and source we convert the row major to
        // column major without transposing matrices.
        // B * A = C.
        // Here backward weight is equivalent of d_dst * src^T when the weight
        // filter is IC*OC. Therefore B is d_dst and A is transposed src, and C
        // is d_wei. However, when the filter format is OC*IC , the backward
        // weight is equivalent to src * d_dst^T. In this case, B is src, A is
        // transposed d_dst and C is d_wei.
        wie_tr_ = (pd->diff_weights_md(0)->format_desc.blocking.strides[0]
                == 1);
        // std::cout << wie_tr_ << std::endl;
        CHECK(convert_data_type(pd->src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->diff_weights_md(0), &data_types_[io::wei]));
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[io::dst]));
        if (need_reorder_) {
            cudnnTensorFormat_t source_format;
            CHECK(get_format(pd->src_md(), source_format));
            ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
            get_4d_tensor_descriptor(
                    pd->diff_weights_md(0), dims_[io::wei], strides_[io::wei]);
            set_filter_format(
                    ndims_, dims_[io::wei], strides_[NUM_IO], source_format);
            CHECK(init_filter_transformation(data_types_[io::wei], ndims_,
                    dims_[io::wei], strides_[NUM_IO], strides_[io::wei]));
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_none,
                    memory_desc_wrapper(pd->diff_weights_md(0)).size(),
                    size_t(1));
            wie_tr_ = (strides_[NUM_IO][0] == 1);
        }
        trans_a_ = CUBLAS_OP_N;
        trans_b_ = CUBLAS_OP_T;
        int ic = pd->IC_total_padded();
        int oc = pd->OC();
        int mb = pd->MB();
        n_ = wie_tr_ ? ic : oc;
        k_ = mb;
        m_ = wie_tr_ ? oc : ic;
        lda_ = m_;
        ldb_ = n_;
        ldc_ = m_;

        CHECK(get_cublas_data_type(
                data_types_[(wie_tr_ ? io::dst : io::src)], a_type_));
        CHECK(get_cublas_data_type(
                data_types_[(wie_tr_ ? io::src : io::dst)], b_type_));
        CHECK(get_cublas_data_type(data_types_[io::wei], c_type_));
        if (with_bias_) {
            ndims_ = 4;
            get_4d_tensor_descriptor(
                    pd->diff_dst_md(), dims_[io::dst], strides_[io::dst]);
            CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[io::dst]));
            set_bias_dims(CUDNN_TENSOR_NCHW, ndims_, pd->OC());
            CHECK(convert_data_type(
                    pd->diff_weights_md(1), &data_types_[io::bia]));
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::dst],
                    data_types_[io::dst], ndims_, dims_[io::dst],
                    strides_[io::dst]));
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::bia],
                    data_types_[io::bia], ndims_, dims_[io::bia],
                    strides_[io::bia]));
            CHECK(create_and_set_reduce_descriptor());

            auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
            stream_t *service_stream;
            CHECK(sycl_engine.get_service_stream(service_stream));

            auto cuda_stream
                    = utils::downcast<sycl_cuda_stream_t *>(service_stream);
            auto handle = cuda_stream->get_cudnn_handle();

            // get the required workspace size
            CUDNN_EXECUTE_FUNC_S(cudnnGetReductionWorkspaceSize, handle,
                    reduceTensorDesc_, tensor_descs_[io::dst],
                    tensor_descs_[io::bia], &workspace_size_);
        }

        if (workspace_size_ > 0) {
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                    workspace_size_, size_t(1));
        }

        return status::success;
    }
    void execute(cudnnHandle_t cudnn_handle, cublasHandle_t cublas_handle,
            const std::vector<void *> &args) const override {
        assert(args.size() == 6);
        auto x = args[0], dy = args[1], dw = args[2], db = args[3],
             workspace = args[4];
        auto dw_arg = need_reorder_ ? args[5] : dw;
        // do gemm
        CUBLAS_EXECUTE_FUNC(cublasGemmEx, cublas_handle, trans_a_, trans_b_, m_,
                n_, k_, &alpha_, (wie_tr_ ? dy : x), a_type_, lda_,
                (wie_tr_ ? x : dy), b_type_, ldb_, &beta_, dw_arg, c_type_,
                ldc_, compute_type_, algo_);

        if (need_reorder_) {
            // The output of weight is in nvida specific format,
            // however a user requires the oneDNN format as an output
            transform_filter(cudnn_handle, dw_arg, dw);
        }
        if (with_bias_) {

            // backward bias for inner product is reduction of dy on dim[0] .
            // So we can use cudnnReduceTensor to partially reduce dy.
            CUDNN_EXECUTE_FUNC(cudnnReduceTensor, cudnn_handle,
                    reduceTensorDesc_, nullptr, 0, workspace, workspace_size_,
                    &alpha_, tensor_descs_[io::dst], dy, &beta_,
                    tensor_descs_[io::bia], db);
        }
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

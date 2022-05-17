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

#ifndef GPU_NVIDIA_CUDNN_CONV_INNER_PRODUCT_IMPL_HPP
#define GPU_NVIDIA_CUDNN_CONV_INNER_PRODUCT_IMPL_HPP

#include "cublas_v2.h"
#include "cudnn.h"

#include "common/type_helpers.hpp"
#include "gpu/nvidia/cudnn_conv_filter_adjustment_base.hpp"
#include "gpu/nvidia/cudnn_inner_product_impl.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_conv_inner_product_impl_base_t
    : public cudnn_inner_product_fwd_base_t,
      public cudnn_conv_filter_adjustment_base_t {

    bool unfold_dimensions_ = false;
    cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
    cudnnFilterDescriptor_t filter_desc_;

    status_t filter_tag(
            const memory_desc_t &md, format_tag_t &weight_tag) const {
        using namespace format_tag;
        weight_tag = memory_desc_matches_one_of_tag(md, oidhw, odhwi, dhwio,
                oihw, ohwi, hwio, oiw, owi, wio, aBcd4b,
                any); // blocked layouts
        if (weight_tag == undef) return status::unimplemented;
        return status::success;
    }

    status_t source_tag(const memory_desc_t &md, format_tag_t &src_tag) const {
        using namespace format_tag;
        src_tag = memory_desc_matches_one_of_tag(
                md, ncdhw, ndhwc, nchw, nhwc, ncw, nwc, aBcd4b, any);
        if (src_tag == undef) return status::unimplemented;
        return status::success;
    }

    virtual ~cudnn_conv_inner_product_impl_base_t() {
        if (conv_desc_) {
            CUDNN_EXECUTE_FUNC_V(cudnnDestroyConvolutionDescriptor, conv_desc_);
        }
        if (filter_desc_) {
            CUDNN_EXECUTE_FUNC_V(cudnnDestroyFilterDescriptor, filter_desc_);
        }
        for (size_t i = 0; i < NUM_IO - 1; i++) {
            if (tensor_descs_[i]) {
                CUDNN_EXECUTE_FUNC_V(
                        cudnnDestroyTensorDescriptor, tensor_descs_[i]);
            }
        }
    }

    void unfold_dims(io memory_index, int *folded_dims, int *folded_strides,
            cudnnTensorFormat_t format, int ndims) {
        folded_dims[0] = dims_[memory_index][0];
        folded_dims[1] = dims_[memory_index][1];
        for (int i = 2; i < ndims; i++) {
            folded_dims[1] *= dims_[memory_index][i];
            folded_dims[i] = 1;
        }
        for (int i = 2; i < ndims; i++) {
            folded_strides[i]
                    = (format == CUDNN_TENSOR_NHWC ? folded_dims[1] : 1);
        }

        folded_strides[1] = 1;
        folded_strides[0] = folded_dims[1];
    }

    virtual void execute(cudnnHandle_t handle, cublasHandle_t,
            const std::vector<void *> &args) const = 0;
};

struct cudnn_conv_inner_product_fwd_impl_t
    : public cudnn_conv_inner_product_impl_base_t {
    bool use_fused_path_for_blocking_ = false;
    bool input_is_blocked_ = false;
    bool filter_is_blocked_ = false;
    cudnnConvolutionFwdAlgo_t algo_;
    cudnnActivationDescriptor_t act_desc_fuse_relu;
    cudnnActivationDescriptor_t act_desc_no_relu_;
    cudnnTensorFormat_t source_format_;

    ~cudnn_conv_inner_product_fwd_impl_t() {
        if (with_bias_) {
            CUDNN_EXECUTE_FUNC_V(
                    cudnnDestroyActivationDescriptor, act_desc_fuse_relu);
        }
        if ((with_eltwise_ && !with_relu_) || (!with_bias_ && with_relu_)) {
            CUDNN_EXECUTE_FUNC_V(
                    cudnnDestroyActivationDescriptor, act_desc_no_relu_);
        }
    }
    virtual status_t init(engine_t *engine, inner_product_pd_t *pd,
            bool with_relu, bool with_eltwise, bool with_sum,
            bool use_fuse_path_for_blocking) override {
        with_bias_ = pd->with_bias();
        with_relu_ = with_relu;
        with_eltwise_ = with_eltwise;
        use_fused_path_for_blocking_ = use_fuse_path_for_blocking;
        output_scales_ = pd->attr()->output_scales_.scales_[0];
        with_sum_ = with_sum;
        scale_bias_ = (output_scales_ != 1) && with_bias_;
        // scaling factor to add the previous destination value to the current
        // computation
        sum_scale_ = sum_scale(pd);
        input_is_blocked_
                = pd->src_md()->format_desc.blocking.inner_blks[0] == 4;
        filter_is_blocked_
                = pd->weights_md(0)->format_desc.blocking.inner_blks[0] == 4;
        // Pad out the dimensions to at least 4.
        if (pd->ndims() > CUDNN_DIM_MAX || pd->ndims() < 2) {
            return status::invalid_arguments;
        }
        ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
        // Initialise meta-data from the descriptors.
        // Convert the padded dimensions to the dimensions expected by cuDNN.
        get_4d_tensor_descriptor(
                pd->src_md(), dims_[io::src], strides_[io::src]);
        get_4d_tensor_descriptor(
                pd->weights_md(), dims_[io::wei], strides_[io::wei]);
        get_4d_tensor_descriptor(
                pd->dst_md(), dims_[io::dst], strides_[io::dst]);

        // Convert oneDNN data types to their cuDNN counterparts.
        CHECK(convert_data_type(pd->src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->weights_md(0), &data_types_[io::wei]));
        if (input_is_blocked_) {
            data_types_[io::dst] = CUDNN_DATA_INT8x4;
        } else {
            CHECK(convert_data_type(pd->dst_md(), &data_types_[io::dst]));
        }

        // Ensure INT8 types are accumulated with INT32.
        if (data_types_[io::src] != CUDNN_DATA_HALF
                && data_types_[io::src] != CUDNN_DATA_FLOAT) {
            data_types_[NUM_IO] = CUDNN_DATA_INT32;
        }

        cudnnTensorFormat_t weights_format;
        format_tag_t w_tag, s_tag;
        CHECK(filter_tag(*pd->weights_md(0), w_tag));
        CHECK(source_tag(*pd->src_md(0), s_tag));
        CHECK(get_format(
                pd->src_md(), source_format_, pd->src_md()->ndims == 2));

        // Currently cuDNN does not support
        // cudnnConvolutionBiasActivationForward
        // for 5D convolution. Therefore we have to unfold the dims for 5d when
        // it is 5d. Also cuDNN does not support s8 type and nhwc format for
        // 5d convolution.
        unfold_dimensions_ = ndims_ > 4
                && ((pd->weights_md(0)->data_type == data_type::s8)
                        || (source_format_ == CUDNN_TENSOR_NHWC) || with_bias_);

        if (!supported_filter_format(pd->weights_md(0))
                || (unfold_dimensions_ && (w_tag != s_tag))
                || ((source_format_ == CUDNN_TENSOR_NCHW)
                        && (w_tag != s_tag))) {
            set_filter_format(
                    ndims_, dims_[io::wei], strides_[NUM_IO], source_format_);
            CHECK(init_filter_transformation(data_types_[io::wei], ndims_,
                    dims_[io::wei], strides_[io::wei], strides_[NUM_IO]));
            filter_using_spatial_format_ = true;
            // we transform the filter based on src format
            weights_format = source_format_;
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_none,
                    memory_desc_wrapper(pd->weights_md(0)).size(), size_t(1));
        } else {
            CHECK(get_format(pd->weights_md(0), weights_format,
                    pd->weights_md(0)->ndims == 2));
        }

        if (scale_bias_) {

            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_adjusted_scales,
                    memory_desc_wrapper(pd->weights_md(1)).size(), size_t(1));
        }

        // Copy over the strides.
        if (with_bias_) {
            CHECK(convert_data_type(pd->weights_md(1), &data_types_[io::bia]));
            set_bias_dims(weights_format, ndims_, pd->OC());
        }

        // cuDNN requires Input and output feature maps to be a multiple of 4
        // for int8. only nhwc is supported for int8// cudnn doesnot support
        // 5d convolution format for int8
        if ((pd->weights_md(0)->data_type == data_type::s8)
                && ((pd->IC() % 4 != 0) || (pd->OC() % 4 != 0))) {
            return status::unimplemented;
        }
        // source format and weight format are the same at this stage
        if (unfold_dimensions_) {
            unfold_dims(io::wei, dims_[io::wei], strides_[io::wei],
                    source_format_, ndims_);
            unfold_dims(io::src, dims_[io::src], strides_[io::src],
                    source_format_, ndims_);
            ndims_ = 4;
        }

        if (input_is_blocked_) {
            CHECK(create_and_set_tensor_descriptor_ex(&tensor_descs_[io::src],
                    CUDNN_TENSOR_NCHW_VECT_C, data_types_[io::src], ndims_,
                    dims_[io::src]));
        } else {
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::src],
                    data_types_[io::src], ndims_, dims_[io::src],
                    strides_[io::src]));
        }
        if (with_bias_) {
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::bia],
                    data_types_[io::bia], ndims_, dims_[io::bia],
                    strides_[io::bia]));
        }
        // If input is blocked, the output needs to be as well.
        if (input_is_blocked_) {
            CHECK(create_and_set_tensor_descriptor_ex(&tensor_descs_[io::dst],
                    CUDNN_TENSOR_NCHW_VECT_C, data_types_[io::dst], ndims_,
                    dims_[io::dst]));
        } else {
            cudnnTensorFormat_t out_format
                    = filter_is_blocked_ ? CUDNN_TENSOR_NCHW : weights_format;
            CHECK(create_and_set_tensor_descriptor_ex(&tensor_descs_[io::dst],
                    out_format, data_types_[io::dst], ndims_, dims_[io::dst]));
        }

        CHECK(create_and_set_filter_descriptor(&filter_desc_, weights_format,
                data_types_[io::wei], ndims_, dims_[io::wei],
                strides_[io::wei]));

        // Set the convolution. For inner product, this means unit strides and
        // dilation, no padding, and with cross-correlation as the mode.
        int conv_dims = ndims_ - 2;
        std::vector<int> unit_strides(conv_dims, 1);
        std::vector<int> unit_dilation(conv_dims, 1);
        std::vector<int> zero_padding(conv_dims, 0);

        CHECK(create_and_set_conv_descriptor(&conv_desc_, conv_dims,
                zero_padding.data(), unit_strides.data(), unit_dilation.data(),
                CUDNN_CROSS_CORRELATION, data_types_[NUM_IO]));

        auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto cuda_stream
                = utils::downcast<sycl_cuda_stream_t *>(service_stream);
        auto handle = cuda_stream->get_cudnn_handle();

        // Inner product can choose whatever algorithm it prefers, although
        // for the identity post-op the IMPLICIT_PRECOMP_GEMM must be used.
        // there is a bug in nvidia that cannot support
        // cudnnGetConvolutionForwardAlgorithm for int8 type
        if (pd->src_md()->data_type != data_type::s8
                && pd->weights_md(0)->data_type != data_type::s8) {
            int num_algos = 0, returned_algo_count = 0;
            CHECK(CUDNN_EXECUTE_FUNC_S(
                    cudnnGetConvolutionForwardAlgorithmMaxCount, handle,
                    &num_algos));
            std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(num_algos);

            CHECK(CUDNN_EXECUTE_FUNC_S(cudnnGetConvolutionForwardAlgorithm_v7,
                    handle, tensor_descs_[io::src], filter_desc_, conv_desc_,
                    tensor_descs_[io::dst], num_algos, &returned_algo_count,
                    &perf_results[0]));

            size_t free_memory, total_memory;
            CHECK(CUDA_EXECUTE_FUNC_S(
                    cuMemGetInfo, &free_memory, &total_memory));

            // perf_results are sorted ascending by compute time, so the first suitable
            // algorithm found is the one with best performance
            for (int i = 0; i < returned_algo_count; i++) {
                if (perf_results[i].status == CUDNN_STATUS_SUCCESS
                        && perf_results[i].memory < free_memory) {

                    algo_ = perf_results[i].algo;
                    break;
                }
            }
        } else {
            algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        }
        if (!with_relu_) {
            algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        }

        // Allocate the workspace from the algorithm selection, if applicable.
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnGetConvolutionForwardWorkspaceSize,
                handle, tensor_descs_[io::src], filter_desc_, conv_desc_,
                tensor_descs_[io::dst], algo_, &workspace_size_));
        if (workspace_size_ > 0) {
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                    workspace_size_, size_t(1));
        }

        // Add the eltwise op. Note that this only applies to the forward pass.
        CHECK(create_and_set_op_descriptor(pd));
        return status::success;
    }

    void execute(cudnnHandle_t handle, cublasHandle_t,
            const std::vector<void *> &args) const override {
        auto x = args[0], w = args[1], b = args[2], y = args[3],
             workspace = args[4];
        assert(args.size() == 7);
        auto w_arg = w;
        if (filter_using_spatial_format_) {
            void *transformed_w = args[5];
            transform_filter(handle, w, transformed_w);
            w_arg = transformed_w;
        }

        if (with_bias_) {
            auto scaled_bias = b;
            if (scale_bias_) {
                void *output_scale_workspace = args[6];
                CUDNN_EXECUTE_FUNC(cudnnAddTensor, handle, &output_scales_,
                        tensor_descs_[io::bia], b, &beta_,
                        tensor_descs_[io::bia], output_scale_workspace);
                scaled_bias = output_scale_workspace;
            }

            CUDNN_EXECUTE_FUNC(cudnnConvolutionBiasActivationForward, handle,
                    &output_scales_, tensor_descs_[io::src], x, filter_desc_,
                    w_arg, conv_desc_, algo_, workspace, workspace_size_,
                    &sum_scale_, tensor_descs_[io::dst], y,
                    tensor_descs_[io::bia], scaled_bias, act_desc_fuse_relu,
                    tensor_descs_[io::dst], y);
        } else {
            CUDNN_EXECUTE_FUNC(cudnnConvolutionForward, handle, &output_scales_,
                    tensor_descs_[io::src], x, filter_desc_, w_arg, conv_desc_,
                    algo_, workspace, workspace_size_, &sum_scale_,
                    tensor_descs_[io::dst], y);
        }
        if ((with_eltwise_ && !with_relu_) || (!with_bias_ && with_relu_)) {
            CUDNN_EXECUTE_FUNC(cudnnActivationForward, handle,
                    act_desc_no_relu_, &alpha_, tensor_descs_[io::dst], y,
                    &beta_, tensor_descs_[io::dst], y);
        }
    }

private:
    status_t create_and_set_op_descriptor(inner_product_pd_t *pd) {
        if (with_bias_) {
            auto mode_fuse = with_relu_ ? CUDNN_ACTIVATION_RELU
                                        : CUDNN_ACTIVATION_IDENTITY;
            CHECK(CUDNN_EXECUTE_FUNC_S(
                    cudnnCreateActivationDescriptor, &act_desc_fuse_relu));
            // For ReLU, a ceiling of 0 means no limit.
            CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetActivationDescriptor,
                    act_desc_fuse_relu, mode_fuse,
                    cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                    eltwise_alpha(pd)));
        }
        if ((with_eltwise_ && !with_relu_) || (!with_bias_ && with_relu_)) {
            CHECK(CUDNN_EXECUTE_FUNC_S(
                    cudnnCreateActivationDescriptor, &act_desc_no_relu_));

            cudnnActivationMode_t no_relu_mode;
            switch (eltwise_algorithm_kind(pd)) {
                case alg_kind::eltwise_tanh:
                    no_relu_mode = CUDNN_ACTIVATION_TANH;
                    break;
                case alg_kind::eltwise_elu:
                    no_relu_mode = CUDNN_ACTIVATION_ELU;
                    break;
                case alg_kind::eltwise_relu:
                    no_relu_mode = CUDNN_ACTIVATION_RELU;
                    break;
                case alg_kind::eltwise_logistic:
                    no_relu_mode = CUDNN_ACTIVATION_SIGMOID;
                    break;
                case alg_kind::eltwise_bounded_relu:
                    no_relu_mode = CUDNN_ACTIVATION_CLIPPED_RELU;
                    break;
                default: return status::unimplemented;
            }
            CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetActivationDescriptor,
                    act_desc_no_relu_, no_relu_mode,
                    cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                    eltwise_alpha(pd)));
        }
        return status::success;
    }
};

struct cudnn_conv_inner_product_bwd_data_impl_t
    : public cudnn_conv_inner_product_impl_base_t {
    cudnnConvolutionBwdDataAlgo_t algo_;
    // the type of filter depends on dy, however since dy is nc
    // for nhwc filter the source must be nhwc as well.
    // So we use the src type for transforming the filter.
    cudnnTensorFormat_t diff_source_format_;
    virtual status_t init(engine_t *engine, inner_product_pd_t *pd,
            bool /*with_relu*/, bool /*with_eltwise*/, bool /*with_sum */,
            bool /*using_fused_path_for_blocking*/) override {
        // Pad out the dimensions to 4
        if (pd->ndims() > CUDNN_DIM_MAX || pd->ndims() < 2) {
            return status::invalid_arguments;
        }
        ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();
        // Initialise meta-data from the descriptors.
        // Convert the padded dimensions to the dimensions expected by cuDNN.
        get_4d_tensor_descriptor(
                pd->diff_src_md(), dims_[io::src], strides_[io::src]);
        get_4d_tensor_descriptor(
                pd->weights_md(), dims_[io::wei], strides_[io::wei]);
        get_4d_tensor_descriptor(
                pd->diff_dst_md(), dims_[io::dst], strides_[io::dst]);

        // Convert oneDNN data types to their cuDNN counterparts.
        CHECK(convert_data_type(pd->diff_src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->weights_md(0), &data_types_[io::wei]));
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[io::dst]));

        format_tag_t w_tag, s_tag;
        CHECK(filter_tag(*pd->weights_md(0), w_tag));
        CHECK(source_tag(*pd->diff_src_md(0), s_tag));
        cudnnTensorFormat_t weights_format;
        CHECK(get_format(pd->diff_src_md(), diff_source_format_));
        // Currently nvidia does not support cudnnConvolution
        // for 5D convolution when the filter format is nhwc.
        // Therefore we have to unfold the dims for 5d when it is 5d.
        unfold_dimensions_
                = ndims_ > 4 && ((diff_source_format_ == CUDNN_TENSOR_NHWC));
        // Copy over the strides.
        // weight format and dy format must be the same, since dx is the result
        // here, we check with diff_src, to make sure we get the correct result.
        if (!supported_filter_format(pd->weights_md(0)) || (w_tag != s_tag)) {
            set_filter_format(ndims_, dims_[io::wei], strides_[NUM_IO],
                    diff_source_format_);
            CHECK(init_filter_transformation(data_types_[io::wei], ndims_,
                    dims_[io::wei], strides_[io::wei], strides_[NUM_IO]));
            filter_using_spatial_format_ = true;
            // the type of weight format must match
            weights_format = diff_source_format_;
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_none,
                    memory_desc_wrapper(pd->weights_md(0)).size(), size_t(1));
        } else {
            CHECK(get_format(pd->weights_md(0), weights_format));
        }

        // source format and weight format are the same at this stage
        if (unfold_dimensions_) {
            unfold_dims(io::wei, dims_[io::wei], strides_[io::wei],
                    diff_source_format_, ndims_);
            unfold_dims(io::src, dims_[io::src], strides_[io::src],
                    diff_source_format_, ndims_);
            ndims_ = 4;
        }

        // Set the tensor descriptors from the dimensions and strides.
        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::src],
                data_types_[io::src], ndims_, dims_[io::src],
                strides_[io::src]));

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::dst],
                data_types_[io::dst], ndims_, dims_[io::dst],
                strides_[io::dst]));

        CHECK(create_and_set_filter_descriptor(&filter_desc_, weights_format,
                data_types_[io::wei], ndims_, dims_[io::wei],
                strides_[io::wei]));

        // Set the convolution. For inner product, this means unit strides and
        // dilation, no padding, and with cross-correlation as the mode.
        int conv_dims = ndims_ - 2;
        std::vector<int> unit_strides(conv_dims, 1);
        std::vector<int> unit_dilation(conv_dims, 1);
        std::vector<int> zero_padding(conv_dims, 0);

        CHECK(create_and_set_conv_descriptor(&conv_desc_, conv_dims,
                zero_padding.data(), unit_strides.data(), unit_dilation.data(),
                CUDNN_CROSS_CORRELATION, data_types_[NUM_IO]));
        auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto cuda_stream
                = utils::downcast<sycl_cuda_stream_t *>(service_stream);
        auto handle = cuda_stream->get_cudnn_handle();

        // Inner product can choose whatever algorithm it prefers.
        int num_algos = 0, returned_algo_count = 0;
        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnGetConvolutionBackwardDataAlgorithmMaxCount, handle,
                &num_algos));
        std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results(num_algos);

        CUDNN_EXECUTE_FUNC(cudnnGetConvolutionBackwardDataAlgorithm_v7, handle,
                filter_desc_, tensor_descs_[io::dst], conv_desc_,
                tensor_descs_[io::src], num_algos, &returned_algo_count,
                &perf_results[0]);

        size_t free_memory, total_memory;
        CHECK(CUDA_EXECUTE_FUNC_S(cuMemGetInfo, &free_memory, &total_memory));

        // perf_results are sorted ascending by compute time, so the first suitable
        // algorithm found is the one with best performance
        for (int i = 0; i < returned_algo_count; i++) {
            if (perf_results[i].status == CUDNN_STATUS_SUCCESS
                    && perf_results[i].memory < free_memory) {

                algo_ = perf_results[i].algo;
                break;
            }
        }

        // Allocate the workspace from the algorithm selection, if applicable.
        CUDNN_EXECUTE_FUNC(cudnnGetConvolutionBackwardDataWorkspaceSize, handle,
                filter_desc_, tensor_descs_[io::dst], conv_desc_,
                tensor_descs_[io::src], algo_, &workspace_size_);

        if (workspace_size_ > 0) {
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                    workspace_size_, size_t(1));
        }

        return status::success;
    }

    void execute(cudnnHandle_t handle, cublasHandle_t,
            const std::vector<void *> &args) const override {
        assert(args.size() == 5);
        auto dx = args[0], w = args[1], dy = args[2], workspace = args[3];
        auto w_arg = w;
        if (filter_using_spatial_format_) {
            auto transformed_w = args[4];
            transform_filter(handle, w, transformed_w);
            w_arg = transformed_w;
        }
        CUDNN_EXECUTE_FUNC(cudnnConvolutionBackwardData, handle, &alpha_,
                filter_desc_, w_arg, tensor_descs_[io::dst], dy, conv_desc_,
                algo_, workspace, workspace_size_, &beta_,
                tensor_descs_[io::src], dx);
    }
};

struct cudnn_conv_inner_product_bwd_weights_impl_t
    : public cudnn_conv_inner_product_impl_base_t {
    cudnnConvolutionBwdFilterAlgo_t algo_;
    cudnnTensorFormat_t source_format_;

    virtual status_t init(engine_t *engine, inner_product_pd_t *pd,
            bool /*with_relu*/, bool /*with_eltwise*/, bool /*with_sum */,
            bool /*using_fused_path_for_blocking*/) override {
        // If any of the dimensions are 0 we should not continue with creating
        // cudnn descriptors
        with_bias_ = pd->with_bias();

        // Pad out the dimensions to 4
        if (pd->ndims() > CUDNN_DIM_MAX || pd->ndims() < 2) {
            return status::invalid_arguments;
        }
        ndims_ = pd->ndims() < 4 ? 4 : pd->ndims();

        // Initialise meta-data from the descriptors.
        // Convert the padded dimensions to the dimensions expected by cuDNN.
        get_4d_tensor_descriptor(
                pd->src_md(), dims_[io::src], strides_[io::src]);
        get_4d_tensor_descriptor(
                pd->diff_weights_md(), dims_[io::wei], strides_[io::wei]);
        get_4d_tensor_descriptor(
                pd->diff_dst_md(), dims_[io::dst], strides_[io::dst]);

        format_tag_t w_tag, s_tag;
        CHECK(filter_tag(*pd->diff_weights_md(0), w_tag));
        CHECK(source_tag(*pd->src_md(0), s_tag));

        cudnnTensorFormat_t diff_weights_format;
        CHECK(get_format(pd->src_md(0), source_format_));
        // Currently nvidia does not support cudnnConvolution
        // for 5D convolution when the filter format is nhwc.
        // Therefore we have to unfold the dims for 5d when it is 5d.
        unfold_dimensions_
                = ndims_ > 4 && ((source_format_ == CUDNN_TENSOR_NHWC));
        // weight format and src format must be the same.
        // we check with src, to make sure we get the correct result.
        if (!supported_filter_format(pd->diff_weights_md(0))
                || (w_tag != s_tag)) {
            set_filter_format(
                    ndims_, dims_[io::wei], strides_[NUM_IO], source_format_);
            CHECK(init_filter_transformation(data_types_[io::wei], ndims_,
                    dims_[io::wei], strides_[NUM_IO], strides_[io::wei]));
            filter_using_spatial_format_ = true;
            // the type of weight format must match
            diff_weights_format = source_format_;
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_none,
                    memory_desc_wrapper(pd->diff_weights_md(0)).size(),
                    size_t(1));
        } else {
            CHECK(get_format(pd->diff_weights_md(0), diff_weights_format));
        }

        // Copy over the strides.
        // Convert oneDNN data types to their cuDNN counterparts.
        CHECK(convert_data_type(pd->src_md(), &data_types_[io::src]));
        CHECK(convert_data_type(pd->diff_weights_md(0), &data_types_[io::wei]));
        CHECK(convert_data_type(pd->diff_dst_md(), &data_types_[io::dst]));

        // source format and weight format are the same at this stage
        if (unfold_dimensions_) {
            unfold_dims(io::wei, dims_[io::wei], strides_[io::wei],
                    source_format_, ndims_);
            unfold_dims(io::src, dims_[io::src], strides_[io::src],
                    source_format_, ndims_);
            ndims_ = 4;
        }

        if (with_bias_) {
            set_bias_dims(diff_weights_format, ndims_, pd->OC());
            CHECK(convert_data_type(
                    pd->diff_weights_md(1), &data_types_[io::bia]));
        }
        // Set the tensor descriptors from the dimensions and strides.
        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::src],
                data_types_[io::src], ndims_, dims_[io::src],
                strides_[io::src]));

        CHECK(create_and_set_filter_descriptor(&filter_desc_,
                diff_weights_format, data_types_[io::wei], ndims_,
                dims_[io::wei], strides_[io::wei]));

        // oneDNN does not set unused dimensions and strides in the output, so
        // we do that here. If nhwc filter, then repeat the N stride for the
        // spatial dimensions.

        CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::dst],
                data_types_[io::dst], ndims_, dims_[io::dst],
                strides_[io::dst]));
        if (with_bias_) {
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[io::bia],
                    data_types_[io::bia], ndims_, dims_[io::bia],
                    strides_[io::bia]));
        }
        // Set the convolution. For inner product, this means unit strides and
        // dilation, no padding, and with cross-correlation as the mode.
        int conv_dims = ndims_ - 2;
        std::vector<int> unit_strides(conv_dims, 1);
        std::vector<int> unit_dilation(conv_dims, 1);
        std::vector<int> zero_padding(conv_dims, 0);

        CHECK(create_and_set_conv_descriptor(&conv_desc_, conv_dims,
                zero_padding.data(), unit_strides.data(), unit_dilation.data(),
                CUDNN_CROSS_CORRELATION, data_types_[NUM_IO]));
        auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto cuda_stream
                = utils::downcast<sycl_cuda_stream_t *>(service_stream);
        auto handle = cuda_stream->get_cudnn_handle();

        // Inner product can choose whatever algorithm it prefers.
        int num_algos = 0, returned_algo_count = 0;
        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnGetConvolutionBackwardFilterAlgorithmMaxCount, handle,
                &num_algos));
        std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results(
                num_algos);

        CUDNN_EXECUTE_FUNC(cudnnGetConvolutionBackwardFilterAlgorithm_v7,
                handle, tensor_descs_[io::src], tensor_descs_[io::dst],
                conv_desc_, filter_desc_, num_algos, &returned_algo_count,
                &perf_results[0]);

        size_t free_memory, total_memory;
        CHECK(CUDA_EXECUTE_FUNC_S(cuMemGetInfo, &free_memory, &total_memory));

        // perf_results are sorted ascending by compute time, so the first suitable
        // algorithm found is the one with best performance
        for (int i = 0; i < returned_algo_count; i++) {
            if (perf_results[i].status == CUDNN_STATUS_SUCCESS
                    && perf_results[i].memory < free_memory) {

                algo_ = perf_results[i].algo;
                break;
            }
        }

        // Allocate the workspace from the algorithm selection, if applicable.
        CUDNN_EXECUTE_FUNC_S(cudnnGetConvolutionBackwardFilterWorkspaceSize,
                handle, tensor_descs_[io::src], tensor_descs_[io::dst],
                conv_desc_, filter_desc_, algo_, &workspace_size_);
        if (workspace_size_ > 0) {
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                    workspace_size_, size_t(1));
        }

        return status::success;
    }

    void execute(cudnnHandle_t handle, cublasHandle_t,
            const std::vector<void *> &args) const override {
        assert(args.size() == 6);
        auto x = args[0], dy = args[1], dw = args[2], db = args[3],
             workspace = args[4];

        auto dw_arg = filter_using_spatial_format_ ? args[5] : dw;
        CUDNN_EXECUTE_FUNC(cudnnConvolutionBackwardFilter, handle, &alpha_,
                tensor_descs_[io::src], x, tensor_descs_[io::dst], dy,
                conv_desc_, algo_, workspace, workspace_size_, &beta_,
                filter_desc_, dw_arg);

        if (filter_using_spatial_format_) {
            // The output of weight is in nvida specific format,
            // however a user requires the oneDNN format as an output
            transform_filter(handle, dw_arg, dw);
        }

        if (with_bias_) {
            CUDNN_EXECUTE_FUNC(cudnnConvolutionBackwardBias, handle, &alpha_,
                    tensor_descs_[io::dst], dy, &beta_, tensor_descs_[io::bia],
                    db);
        }
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

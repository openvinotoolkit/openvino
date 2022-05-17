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

#ifndef GPU_NVIDIA_CUDNN_CONVOLUTION_IMPL_HPP
#define GPU_NVIDIA_CUDNN_CONVOLUTION_IMPL_HPP

#include "cudnn.h"

#include "common/c_types_map.hpp"
#include "common/convolution_pd.hpp"
#include "common/utils.hpp"
#include "gpu/nvidia/cudnn_conv_filter_adjustment_base.hpp"
#include "gpu/nvidia/cudnn_convolution_pd.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_convolution_impl_base_t
    : public cudnn_conv_filter_adjustment_base_t {
protected:
    enum io { x = 0, bias, weights, y, NUM_IO };
    memory_desc_t dnnl_descs[NUM_IO];
    cudnnConvolutionDescriptor_t conv_desc;
    int padding[CUDNN_DIM_MAX];
    int dilation[CUDNN_DIM_MAX];
    cudnnTensorDescriptor_t descs[NUM_IO];
    cudnnDataType_t data_types[NUM_IO];
    int ndims[NUM_IO];
    int dims[NUM_IO][DNNL_MAX_NDIMS];
    int strides[NUM_IO + 1][DNNL_MAX_NDIMS];
    int filter_strides[DNNL_MAX_NDIMS];
    cudnnTensorFormat_t formats[NUM_IO];
    bool filter_needs_transform = false;
    cudnnFilterDescriptor_t weights_desc;
    float alpha = 0.f;
    float beta = 0.f;
    int group_count = 1;
    bool with_groups = false;
    size_t scratchpad_size = 0;
    bool with_bias = false;

    bool do_scaling = false;
    float output_scaling = 1.0f;
    bool use_temp_dst_ = false;
    cudnnDataType_t computation_data_type = CUDNN_DATA_FLOAT;
    cudnnDataType_t reorder_type = CUDNN_DATA_INT8;

public:
    virtual ~cudnn_convolution_impl_base_t() {
        CUDNN_EXECUTE_FUNC_V(cudnnDestroyFilterDescriptor, weights_desc);
        CUDNN_EXECUTE_FUNC_V(cudnnDestroyConvolutionDescriptor, conv_desc);
        for (size_t i = 0; i < io::NUM_IO; i++) {
            CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, descs[i]);
        }
    }
    virtual status_t configure_alg_kind(engine_t *, convolution_pd_t *pd) = 0;

    virtual bool supported_filter_format(
            const memory_desc_t *md) const override {
        const memory_desc_wrapper mem_wrapper(md);

        return (mem_wrapper.matches_one_of_tag(format_tag::ab, format_tag::abc,
                        format_tag::abcd, format_tag::abcde, format_tag::abcdef)
                || (with_groups ? mem_wrapper.matches_one_of_tag(
                            format_tag::gowi, format_tag::gohwi,
                            format_tag::godhwi)
                                : mem_wrapper.matches_one_of_tag(
                                        format_tag::owi, format_tag::ohwi,
                                        format_tag::odhwi)));
    }

    bool using_transformed_filter() const { return filter_needs_transform; }
    bool with_scratchpad() const { return scratchpad_size > 0; }

    virtual status_t init(engine_t *engine, convolution_pd_t *pd,
            bool use_scratch_dst = false) {
        CHECK(configure_parameters(pd));
        CHECK(create_cudnn_descs(pd));
        CHECK(check_output_dims());
        CHECK(configure_alg_kind(engine, pd));
        CHECK(init_scratchpad(engine, pd));

        return status::success;
    }

    virtual status_t init_zero_dims(convolution_pd_t *pd) {
        return status::success;
    }
    void get_dims_and_strides(int io) {
        convert_dims(
                dnnl_descs[io].dims, dims[io], dnnl_descs[io].ndims, ndims[io]);
        if (ndims[io] > dnnl_descs[io].ndims) {
            std::swap(dims[io][ndims[io] - 1], dims[io][ndims[io] - 2]);
            if (ndims[io] == 4) {
                if (formats[io] == CUDNN_TENSOR_NHWC) {
                    propagate_strides(strides[io], dims[io], {1, 3, 2, 0});
                } else {
                    propagate_strides(strides[io], dims[io], {3, 2, 1, 0});
                }
            }
        } else {
            convert_dims(dnnl_descs[io].format_desc.blocking.strides,
                    strides[io], dnnl_descs[io].ndims, ndims[io]);
        }
    }
    status_t configure_parameters(const convolution_pd_t *pd) {
        if (pd->ndims() > CUDNN_DIM_MAX) { return status::invalid_arguments; }
        CHECK(set_padding_and_dilation(pd));
        with_groups = pd->with_groups();
        with_bias = pd->with_bias();
        alpha = 1.0f;
        beta = 0.0f;
        output_scaling = pd->attr()->output_scales_.scales_[0];
        do_scaling = output_scaling != 1.f;
        dnnl_descs[x] = *pd->invariant_src_md();
        dnnl_descs[weights] = *pd->invariant_wei_md();
        dnnl_descs[y] = *pd->invariant_dst_md();
        if (with_bias) dnnl_descs[bias] = *pd->invariant_bia_md();

        ndims[x] = std::max(dnnl_descs[x].ndims, 4);
        ndims[weights] = std::max(dnnl_descs[weights].ndims, 4 + with_groups);
        ndims[y] = std::max(dnnl_descs[y].ndims, 4);

        CHECK(convert_data_type(&dnnl_descs[x], &data_types[x]));
        CHECK(convert_data_type(&dnnl_descs[weights], &data_types[weights]));
        CHECK(convert_data_type(&dnnl_descs[y], &data_types[y]));

        CHECK(get_formats());
        set_compute_format();
        get_dims_and_strides(x);
        get_dims_and_strides(weights);
        get_dims_and_strides(y);

        if (!supported_filter_format(&dnnl_descs[weights])) {
            set_filter_format(
                    ndims[weights], dims[weights], strides[NUM_IO], formats[x]);
            CHECK(init_filter_transformation(data_types[weights],
                    ndims[weights], dims[weights], strides[weights],
                    strides[NUM_IO]));
            filter_needs_transform = true;
            // we transform the filter based on src format
            formats[weights] = formats[x];
        } else {
            CHECK(get_filter_format());
            get_dims_and_strides(weights);
        }
        if (with_groups) {
            dims[weights][1] *= pd->G();
            ndims[weights] = std::max(4, ndims[weights] - with_groups);
        }

        if (with_bias) {
            ndims[bias] = dnnl_descs[bias].ndims;
            CHECK(convert_data_type(&dnnl_descs[bias], &data_types[bias]));
            convert_dims(
                    dnnl_descs[bias].dims, dims[bias], ndims[bias], ndims[y]);
            std::swap(dims[bias][0], dims[bias][1]);
            convert_dims(dnnl_descs[bias].format_desc.blocking.strides,
                    strides[bias], ndims[bias], ndims[y]);
            ndims[bias] = ndims[y];
        }

        return status::success;
    }

    status_t create_cudnn_descs(const convolution_pd_t *pd) {
        CHECK(create_and_set_convolution_desc(pd));
        CHECK(create_and_set_tensor_descriptor(
                &descs[x], data_types[x], ndims[x], dims[x], strides[x]));
        CHECK(create_and_set_filter_descriptor(&weights_desc, formats[weights],
                data_types[weights], ndims[weights],
                dims[weights] + with_groups, strides[weights]));
        CHECK(create_and_set_tensor_descriptor(
                &descs[y], data_types[y], ndims[y], dims[y], strides[y]));

        if (with_bias) {
            CHECK(create_and_set_tensor_descriptor(&descs[bias],
                    data_types[bias], ndims[bias], dims[bias], strides[bias]));
        }

        return status::success;
    }
    virtual status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) {
        if (filter_needs_transform) {
            auto sz = memory_desc_wrapper(&dnnl_descs[weights]).size();
            auto data_size
                    = types::data_type_size(pd->invariant_wei_md(0)->data_type);
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cudnn_filter, sz,
                    data_size);
        }
        return status::success;
    };

    status_t create_and_set_convolution_desc(const convolution_pd_t *pd) {
        CUDNN_EXECUTE_FUNC_V(cudnnCreateConvolutionDescriptor, &conv_desc);
        // Allow cuDNN to dispatch into Tensor Core implementations
        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnSetConvolutionMathType, conv_desc, CUDNN_TENSOR_OP_MATH));
        CUDNN_EXECUTE_FUNC_V(cudnnSetConvolutionNdDescriptor, conv_desc,
                ndims[x] - 2, padding, filter_strides, dilation,
                cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                computation_data_type);
        // Check for groups and set group count if necessary
        if (with_groups) {
            group_count = pd->G();
            if (group_count > 1)
                CHECK(CUDNN_EXECUTE_FUNC_S(
                        cudnnSetConvolutionGroupCount, conv_desc, group_count));
        }
        return status::success;
    }

    status_t set_padding_and_dilation(const convolution_pd_t *pd) {
        int actual_ndims = pd->ndims();
        if (actual_ndims == 3) {
            padding[0] = 0;
            padding[1] = static_cast<int>(pd->padL());
            dilation[0] = 1;
            dilation[1] = static_cast<int>(pd->KDW() + 1);

            filter_strides[0] = 1;
            filter_strides[1] = static_cast<int>(pd->KSW());
        } else if (actual_ndims == 4) {
            padding[0] = static_cast<int>(pd->padT());
            padding[1] = static_cast<int>(pd->padL());

            dilation[0] = static_cast<int>(pd->KDH() + 1);
            dilation[1] = static_cast<int>(pd->KDW() + 1);

            filter_strides[0] = static_cast<int>(pd->KSH());
            filter_strides[1] = static_cast<int>(pd->KSW());
        } else {
            padding[0] = static_cast<int>(pd->padFront());
            padding[1] = static_cast<int>(pd->padT());
            padding[2] = static_cast<int>(pd->padL());

            dilation[0] = static_cast<int>(pd->KDD() + 1);
            dilation[1] = static_cast<int>(pd->KDH() + 1);
            dilation[2] = static_cast<int>(pd->KDW() + 1);

            filter_strides[0] = static_cast<int>(pd->KSD());
            filter_strides[1] = static_cast<int>(pd->KSH());
            filter_strides[2] = static_cast<int>(pd->KSW());
        }
        return status::success;
    }

    virtual void execute(
            cudnnHandle_t handle, const std::vector<void *> &args) const = 0;

    void execute_sum(cudnnHandle_t handle, void *x, void *y, float alpha_,
            float beta_) const {
        float alpha = alpha_;
        float beta = beta_;
        CUDNN_EXECUTE_FUNC_V(cudnnAddTensor, handle, &alpha, descs[io::y], x,
                &beta, descs[io::y], y);
    }

    void execute_scale(cudnnHandle_t handle, void *y) const {
        if (do_scaling) {
            CUDNN_EXECUTE_FUNC_V(
                    cudnnScaleTensor, handle, descs[io::y], y, &output_scaling);
        }
    }

    void execute_set_weights_bias(
            cudnnHandle_t handle, void *weights, void *bias, float value) {
        CUDNN_EXECUTE_FUNC_V(
                cudnnSetTensor, handle, descs[io::weights], weights, &value);
        if (bias) {
            CUDNN_EXECUTE_FUNC_V(
                    cudnnSetTensor, handle, descs[io::bias], bias, &value);
        }
    }

    bool with_eltwise(const convolution_pd_t *pd, int position) const {
        return pd->attr()->post_ops_.contain(primitive_kind::eltwise, position);
    }

    status_t check_output_dims() const {
        int expected_dims[CUDNN_DIM_MAX] = {};
        CUDNN_EXECUTE_FUNC_V(cudnnGetConvolutionNdForwardOutputDim, conv_desc,
                descs[x], weights_desc, ndims[y], &expected_dims[0]);
        for (size_t i = 0; i < ndims[y]; i++) {
            if (dims[y][i] != expected_dims[i]) return status::unimplemented;
        }
        return status::success;
    }

    void set_compute_format() {
        if (data_types[x] == CUDNN_DATA_INT8) {
            computation_data_type = CUDNN_DATA_INT32;
        } else {
            computation_data_type = data_types[y];
        }
    }

    status_t get_filter_format() {
        memory_desc_wrapper wrapper(&dnnl_descs[weights]);
        if (wrapper.matches_one_of_tag(format_tag::ab, format_tag::abc,
                    format_tag::abcd, format_tag::abcde, format_tag::abcdef)) {
            formats[weights] = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
        } else if ((!with_groups
                           && wrapper.matches_one_of_tag(format_tag::owi,
                                   format_tag::ohwi, format_tag::odhwi))
                || (with_groups
                        && wrapper.matches_one_of_tag(format_tag::gowi,
                                format_tag::gohwi, format_tag::godhwi))) {
            formats[weights] = cudnnTensorFormat_t::CUDNN_TENSOR_NHWC;
        } else {
            return status::unimplemented;
        }

        return status::success;
    }

    status_t get_formats() {
        CHECK(get_format(&dnnl_descs[x], formats[x]));
        CHECK(get_format(&dnnl_descs[y], formats[y]));
        return status::success;
    }

    void set_filter_nhwc(int filter_ndims, int *transform_filter_strides,
            int *filter_dims) override {
        if (with_groups) {
            switch (filter_ndims) {
                case 4: // Convert to krsc
                    return propagate_strides(transform_filter_strides,
                            filter_dims, {2, 3, 1, 0});
                case 5:
                    return propagate_strides(transform_filter_strides,
                            filter_dims, {2, 4, 3, 1, 0});
                case 6:
                    return propagate_strides(transform_filter_strides,
                            filter_dims, {2, 5, 4, 3, 1, 0});
            }
        } else {
            cudnn_conv_filter_adjustment_base_t::set_filter_nhwc(
                    filter_ndims, transform_filter_strides, filter_dims);
        }
    }

    bool use_temp_dst() const { return use_temp_dst_; }
};

struct cudnn_convolution_impl_fwd_t : public cudnn_convolution_impl_base_t {
protected:
    cudnnActivationDescriptor_t activation_desc = nullptr;
    cudnnActivationDescriptor_t eltwise_desc = nullptr;
    cudnnTensorDescriptor_t reorder_dst_desc = nullptr;
    cudnnConvolutionFwdAlgo_t fwd_alg_kind;
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf;
    int requested_algo_count = 0;
    int returned_algo_count = 0;
    int num_post_ops = 0;
    primitive_kind_t post_ops[2];
    bool need_reorder = false;
    float sum_scale = 1.0f;
    bool conv_bias_eltwise = false;
    bool conv_bias = false;

public:
    virtual ~cudnn_convolution_impl_fwd_t() {
        if (activation_desc)
            CUDNN_EXECUTE_FUNC_V(
                    cudnnDestroyActivationDescriptor, activation_desc);
        if (eltwise_desc)
            CUDNN_EXECUTE_FUNC_V(
                    cudnnDestroyActivationDescriptor, eltwise_desc);
        if (reorder_dst_desc)
            CUDNN_EXECUTE_FUNC_V(
                    cudnnDestroyTensorDescriptor, reorder_dst_desc);
    }

    status_t configure_post_ops(convolution_pd_t *pd) {
        auto &p = pd->attr()->post_ops_;
        num_post_ops = p.len();
        for (size_t i = 0; i < p.len(); i++) {
            post_ops[i] = p.entry_[i].kind;
            if (post_ops[i] == dnnl_eltwise) {
                CHECK(create_and_set_eltwise_descriptor(pd));
            }
            if (post_ops[i] == dnnl_sum) { sum_scale = p.entry_[i].sum.scale; }
        }

        // Try to fuse kernels
        // pattern 1: conv + bias + eltwise
        conv_bias_eltwise = num_post_ops > 0 && post_ops[0] == dnnl_eltwise
                && with_bias && !do_scaling
                && data_types[y] != CUDNN_DATA_INT8
                // XXX: cuDNN has a correctness issue for fusion of group conv
                && pd->G() == 1
                && eltwise_algorithm_kind(pd) == alg_kind::eltwise_relu;
        // pattern 2: conv + bias
        conv_bias = with_bias && !conv_bias_eltwise
                && !do_scaling
                // XXX: cuDNN limitation on algorithm support when activation is
                // equal to CUDNN_ACTIVATION_IDENTITY.
                && fwd_alg_kind
                        == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
                // XXX: cuDNN has a correctness issue for fusion of group conv
                && pd->G() == 1;
        // If the only post-op is fused then there is no need for temp dst
        if (conv_bias_eltwise && num_post_ops == 1) use_temp_dst_ = false;

        if (data_types[y] == CUDNN_DATA_INT8 && use_temp_dst_) {
            data_types[y] = CUDNN_DATA_FLOAT;
            need_reorder = true;
            CHECK(create_and_set_tensor_descriptor_ex(&reorder_dst_desc,
                    formats[y], reorder_type, ndims[y], dims[y]));
        }

        return status::success;
    }

    status_t init(engine_t *engine, convolution_pd_t *pd,
            bool use_scratch_dst) override {
        use_temp_dst_ = use_scratch_dst;
        CHECK(configure_parameters(pd));
        CHECK(create_cudnn_descs(pd));
        CHECK(configure_alg_kind(engine, pd));
        CHECK(configure_post_ops(pd));
        CHECK(init_scratchpad(engine, pd));

        return status::success;
    }

    void execute_reorder(cudnnHandle_t handle, void *src, void *dst,
            bool flip_formats) const {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        if (flip_formats) {
            CUDNN_EXECUTE_FUNC_V(cudnnTransformTensor, handle, &alpha,
                    reorder_dst_desc, src, &beta, descs[y], dst);
        } else {
            CUDNN_EXECUTE_FUNC_V(cudnnTransformTensor, handle, &alpha, descs[y],
                    src, &beta, reorder_dst_desc, dst);
        }
    }

    void execute_eltwise(cudnnHandle_t handle, void *src, void *dst) const {
        float alpha = 1.0f;
        float beta = 0.0f;
        CUDNN_EXECUTE_FUNC_V(cudnnActivationForward, handle, eltwise_desc,
                &alpha, descs[io::y], src, &beta, descs[io::y], dst);
    }

    void execute(cudnnHandle_t handle,
            const std::vector<void *> &args) const override {
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4], post_op_scratch = args[6],
             post_op_reorder = args[7];
        void *output = use_temp_dst_ ? post_op_scratch : y;
        if (using_transformed_filter()) {
            auto w_scratch = args[5];
            transform_filter(handle, weights, w_scratch);
            weights = w_scratch;
        }
        bool fused = conv_bias || conv_bias_eltwise;
        if (fused) {
            auto err = cudnnConvolutionBiasActivationForward(handle, &alpha,
                    descs[io::x], x, weights_desc, weights, conv_desc,
                    fwd_alg_kind, scratchpad, scratchpad_size, &beta,
                    descs[io::y], output, descs[io::bias], bias,
                    conv_bias_eltwise ? eltwise_desc : activation_desc,
                    descs[io::y], output);
            // try to fallback into standalone convolution
            if (err == CUDNN_STATUS_NOT_SUPPORTED) {
                fused = false;
            } else {
                CUDNN_CHECK_V(err);
            }
        }

        if (!fused) {
            const float bias_alpha = 1.0f;
            const float bias_beta = 1.0f;
            CUDNN_EXECUTE_FUNC_V(cudnnConvolutionForward, handle, &alpha,
                    descs[io::x], x, weights_desc, weights, conv_desc,
                    fwd_alg_kind, scratchpad, scratchpad_size, &beta,
                    descs[io::y], output);
            if (with_bias) {
                CUDNN_EXECUTE_FUNC_V(cudnnAddTensor, handle, &bias_alpha,
                        descs[io::bias], bias, &bias_beta, descs[io::y],
                        output);
            }
        }
        execute_scale(handle, output);
        // skip first eltwise in case it is fused into convolution
        const int post_ops_start_pos = fused && conv_bias_eltwise;
        for (int i = post_ops_start_pos; i < num_post_ops; i++) {
            bool last_op = i == num_post_ops - 1 && !need_reorder;
            switch (post_ops[i]) {
                case dnnl_sum:
                    if (need_reorder) {
                        execute_reorder(handle, y, post_op_reorder, true);
                        execute_sum(handle, post_op_reorder, post_op_scratch,
                                sum_scale, 1.0f);
                    } else if (last_op) {
                        execute_sum(
                                handle, post_op_scratch, y, 1.0f, sum_scale);
                    } else {
                        execute_sum(
                                handle, y, post_op_scratch, sum_scale, 1.0f);
                    }

                    break;

                case dnnl_eltwise:
                    if (last_op) {
                        execute_eltwise(handle, output, y);
                    } else {
                        execute_eltwise(handle, output, post_op_scratch);
                    }
                    break;
                default: assert(!"unsupported post op");
            }
        }

        if (need_reorder) {
            execute_reorder(handle, post_op_scratch, y, false);
        }
    }
    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto cuda_stream
                = utils::downcast<sycl_cuda_stream_t *>(service_stream);
        auto handle = cuda_stream->get_cudnn_handle();

        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnGetConvolutionForwardWorkspaceSize,
                handle, descs[x], weights_desc, conv_desc, descs[y],
                fwd_alg_kind, &scratchpad_size));
        if (scratchpad_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cudnn_algo,
                    scratchpad_size, size_t(1));

        return cudnn_convolution_impl_base_t::init_scratchpad(engine, pd);
    }
    status_t configure_alg_kind(
            engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
        cuda_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto cuda_stream
                = utils::downcast<sycl_cuda_stream_t *>(service_stream);
        auto handle = cuda_stream->get_cudnn_handle();

        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnGetConvolutionForwardAlgorithmMaxCount,
                handle, &requested_algo_count));
        perf.resize(requested_algo_count);
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnFindConvolutionForwardAlgorithm, handle,
                descs[x], weights_desc, conv_desc, descs[y],
                requested_algo_count, &returned_algo_count, perf.data()));

        auto submit_status = CUDNN_STATUS_NOT_SUPPORTED;
        for (size_t i = 0; i < returned_algo_count; i++) {
            submit_status = perf[i].status;
            if (submit_status == CUDNN_STATUS_SUCCESS) {
                // cudnnFindConvolutionForwardAlgorithm can erroneously report
                // algorithms for int8 which does not work so ensure that we
                // only allow CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
                // in this case.
                if (computation_data_type == CUDNN_DATA_INT32
                        && perf[i].algo
                                != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
                    continue;
                }
                switch (pd->desc()->alg_kind) {
                    case dnnl_convolution_auto:
                        if (utils::one_of(perf[i].algo,
                                    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)) {
                            utils::downcast<cudnn_convolution_fwd_pd_t *>(pd)
                                    ->set_alg_kind(dnnl_convolution_direct);
                        } else {
                            utils::downcast<cudnn_convolution_fwd_pd_t *>(pd)
                                    ->set_alg_kind(dnnl_convolution_winograd);
                        }
                        break;
                    case dnnl_convolution_direct:
                        if (!utils::one_of(perf[i].algo,
                                    CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM))
                            continue;
                        break;
                    case dnnl_convolution_winograd:
                        if (!utils::one_of(perf[i].algo,
                                    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
                                    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED))
                            continue;
                        break;
                    default: return status::unimplemented;
                }
                fwd_alg_kind = perf[i].algo;
                CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetConvolutionMathType,
                        conv_desc, perf[i].mathType));
                break;
            }
        }

        if (submit_status != CUDNN_STATUS_SUCCESS) return status::unimplemented;

        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnCreateActivationDescriptor, &activation_desc));
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetActivationDescriptor,
                activation_desc,
                cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY,
                CUDNN_NOT_PROPAGATE_NAN, 1.0));

        return status::success;
    }

    status_t create_and_set_eltwise_descriptor(const convolution_pd_t *pd) {

        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnCreateActivationDescriptor, &eltwise_desc));

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
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetActivationDescriptor, eltwise_desc,
                act_mode, cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                eltwise_alpha(pd)));

        return status::success;
    }

    dnnl::impl::alg_kind_t eltwise_algorithm_kind(
            const convolution_pd_t *pd) const {
        const int eltwise_idx
                = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return pd->attr()->post_ops_.entry_[eltwise_idx].eltwise.alg;
    }

    float eltwise_alpha(const convolution_pd_t *pd) const {
        const int eltwise_idx
                = pd->attr()->post_ops_.find(primitive_kind::eltwise);
        return pd->attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha;
    }
};

struct cudnn_convolution_impl_bwd_data_t
    : public cudnn_convolution_impl_base_t {
protected:
    cudnnConvolutionBwdDataAlgo_t bwd_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf;
    int requested_algo_count = 0;
    int returned_algo_count = 0;
    status_t configure_alg_kind(
            engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
        cuda_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto cuda_stream
                = utils::downcast<sycl_cuda_stream_t *>(service_stream);
        auto handle = cuda_stream->get_cudnn_handle();

        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnGetConvolutionBackwardDataAlgorithmMaxCount, handle,
                &requested_algo_count));
        perf.resize(requested_algo_count);
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnFindConvolutionBackwardDataAlgorithm,
                handle, weights_desc, descs[y], conv_desc, descs[x],
                requested_algo_count, &returned_algo_count, perf.data()));
        for (size_t i = 0; i < returned_algo_count; i++) {
            if (perf[i].status == CUDNN_STATUS_SUCCESS) {
                switch (pd->desc()->alg_kind) {
                    case dnnl_convolution_auto:
                        if (utils::one_of(perf[i].algo,
                                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)) {
                            utils::downcast<cudnn_convolution_bwd_data_pd_t *>(
                                    pd)
                                    ->set_alg_kind(dnnl_convolution_direct);
                        } else {
                            utils::downcast<cudnn_convolution_bwd_data_pd_t *>(
                                    pd)
                                    ->set_alg_kind(dnnl_convolution_winograd);
                        }
                        break;
                    case dnnl_convolution_direct:
                        if (!utils::one_of(perf[i].algo,
                                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1))
                            continue;
                        break;
                    case dnnl_convolution_winograd:
                        if (!utils::one_of(perf[i].algo,
                                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
                                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED))
                            continue;
                        break;
                    default: return status::unimplemented;
                }
                bwd_algo = perf[i].algo;
                CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetConvolutionMathType,
                        conv_desc, perf[i].mathType));
                break;
            } else {
                return status::unimplemented;
            }
        }

        return status::success;
    }

    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto cuda_stream
                = utils::downcast<sycl_cuda_stream_t *>(service_stream);
        auto handle = cuda_stream->get_cudnn_handle();

        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnGetConvolutionBackwardDataWorkspaceSize,
                handle, weights_desc, descs[io::y], conv_desc, descs[io::x],
                bwd_algo, &scratchpad_size));
        if (scratchpad_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cudnn_algo,
                    scratchpad_size, size_t(1));

        return cudnn_convolution_impl_base_t::init_scratchpad(engine, pd);
    }

    void execute(cudnnHandle_t handle,
            const std::vector<void *> &args) const override {
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4];
        if (using_transformed_filter()) {
            auto w_scratch = args[5];
            transform_filter(handle, weights, w_scratch);
            weights = w_scratch;
        }
        const float bias_alpha = 1.0f;
        const float bias_beta = 1.0f;
        CUDNN_EXECUTE_FUNC_V(cudnnConvolutionBackwardData, handle, &alpha,
                weights_desc, weights, descs[io::y], y, conv_desc, bwd_algo,
                scratchpad, scratchpad_size, &beta, descs[io::x], x);
        if (with_bias) {
            CUDNN_EXECUTE_FUNC_V(cudnnAddTensor, handle, &bias_alpha,
                    descs[io::bias], bias, &bias_beta, descs[io::x], x);
        }
    }
};

struct cudnn_convolution_impl_bwd_weights_t
    : public cudnn_convolution_impl_base_t {
protected:
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo
            = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf;
    int requested_algo_count = 0;
    int returned_algo_count = 0;

public:
    status_t init_zero_dims(convolution_pd_t *pd) override {
        if (pd->ndims() > CUDNN_DIM_MAX) { return status::invalid_arguments; }
        dnnl_descs[weights] = *pd->invariant_wei_md();
        CHECK(get_format(&dnnl_descs[weights], formats[weights], true));
        ndims[y] = pd->invariant_dst_md()->ndims;
        ndims[weights] = dnnl_descs[weights].ndims - pd->with_groups();
        CHECK(convert_data_type(&dnnl_descs[weights], &data_types[weights]));
        convert_dims(dnnl_descs[weights].dims + pd->with_groups(),
                dims[weights], ndims[weights]);
        ndims[weights] = std::max(4, ndims[weights]);
        convert_dims(dnnl_descs[weights].format_desc.blocking.strides,
                strides[weights], ndims[weights]);
        CHECK(create_and_set_tensor_descriptor(&descs[weights],
                data_types[weights], ndims[weights], dims[weights],
                strides[weights]));

        if (pd->with_bias()) {
            dnnl_descs[bias] = *pd->invariant_bia_md();
            ndims[bias] = dnnl_descs[bias].ndims;
            CHECK(convert_data_type(&dnnl_descs[bias], &data_types[bias]));
            convert_dims(dnnl_descs[bias].padded_dims, dims[bias], ndims[bias],
                    ndims[y]);
            std::swap(dims[bias][0], dims[bias][1]);
            convert_dims(dnnl_descs[bias].format_desc.blocking.strides,
                    strides[bias], ndims[bias], ndims[weights]);
            ndims[bias] = ndims[y];
            CHECK(create_and_set_tensor_descriptor(&descs[bias],
                    data_types[bias], ndims[bias], dims[bias], strides[bias]));
        }
        return status::success;
    }
    virtual status_t configure_alg_kind(
            engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
        cuda_sycl_scoped_context_handler_t sc(sycl_engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto cuda_stream
                = utils::downcast<sycl_cuda_stream_t *>(service_stream);
        auto handle = cuda_stream->get_cudnn_handle();

        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnGetConvolutionBackwardFilterAlgorithmMaxCount, handle,
                &requested_algo_count));
        perf.resize(requested_algo_count);
        CHECK(CUDNN_EXECUTE_FUNC_S(cudnnFindConvolutionBackwardFilterAlgorithm,
                handle, descs[x], descs[y], conv_desc, weights_desc,
                requested_algo_count, &returned_algo_count, perf.data()));
        for (size_t i = 0; i < returned_algo_count; i++) {
            if (perf[i].status == CUDNN_STATUS_SUCCESS) {
                switch (pd->desc()->alg_kind) {
                    case dnnl_convolution_auto:
                        if (utils::one_of(perf[i].algo,
                                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3)) {
                            utils::downcast<
                                    cudnn_convolution_bwd_weights_pd_t *>(pd)
                                    ->set_alg_kind(dnnl_convolution_direct);
                        } else {
                            utils::downcast<
                                    cudnn_convolution_bwd_weights_pd_t *>(pd)
                                    ->set_alg_kind(dnnl_convolution_winograd);
                        }
                        break;
                    case dnnl_convolution_direct:
                        if (!utils::one_of(perf[i].algo,
                                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3))
                            continue;
                        break;
                    case dnnl_convolution_winograd:
                        if (!utils::one_of(perf[i].algo,
                                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD,
                                    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED))
                            continue;
                        break;
                    default: return status::unimplemented;
                }
                bwd_filter_algo = perf[i].algo;
                CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetConvolutionMathType,
                        conv_desc, perf[i].mathType));
                break;
            } else {
                return status::unimplemented;
            }
        }

        return status::success;
    }

    status_t init_scratchpad(engine_t *engine, convolution_pd_t *pd) override {
        auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(engine);
        stream_t *service_stream;
        CHECK(sycl_engine.get_service_stream(service_stream));

        auto cuda_stream
                = utils::downcast<sycl_cuda_stream_t *>(service_stream);
        auto handle = cuda_stream->get_cudnn_handle();

        CHECK(CUDNN_EXECUTE_FUNC_S(
                cudnnGetConvolutionBackwardFilterWorkspaceSize, handle,
                descs[io::x], descs[io::y], conv_desc, weights_desc,
                bwd_filter_algo, &scratchpad_size));
        if (scratchpad_size > 0)
            pd->scratchpad_registry().registrar().book(
                    memory_tracking::names::key_conv_cudnn_algo,
                    scratchpad_size, size_t(1));

        return cudnn_convolution_impl_base_t::init_scratchpad(engine, pd);
    }

    void execute(cudnnHandle_t handle,
            const std::vector<void *> &args) const override {
        auto x = args[0], weights = args[1], y = args[2], bias = args[3],
             scratchpad = args[4];
        auto filter = weights;
        if (using_transformed_filter()) {
            auto w_scratch = args[5];
            transform_filter(handle, weights, w_scratch);
            filter = w_scratch;
        }
        const float bias_alpha = 1.0f;
        const float bias_beta = 0.0f;
        CUDNN_EXECUTE_FUNC_V(cudnnConvolutionBackwardFilter, handle, &alpha,
                descs[io::x], x, descs[io::y], y, conv_desc, bwd_filter_algo,
                scratchpad, scratchpad_size, &beta, weights_desc, filter);
        if (with_bias) {
            CUDNN_EXECUTE_FUNC_V(cudnnConvolutionBackwardBias, handle,
                    &bias_alpha, descs[io::y], y, &bias_beta, descs[io::bias],
                    bias);
        }
        if (using_transformed_filter()) {
            undo_transform_filter(handle, filter, weights);
        }
    }
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

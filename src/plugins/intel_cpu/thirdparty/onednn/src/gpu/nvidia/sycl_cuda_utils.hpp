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

#ifndef GPU_NVIDIA_SYCL_CUDA_UTILS_HPP
#define GPU_NVIDIA_SYCL_CUDA_UTILS_HPP

#include <cuda.h>
#include <cudnn.h>
#include <stdexcept>
#include <cublas_v2.h>

#include "dnnl_sycl.hpp"

#include "common/engine.hpp"
#include "common/z_magic.hpp"

#include <CL/sycl/backend/cuda.hpp>

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

#define CTX_OUT_ACCESSOR(arg) \
    utils::downcast<sycl::sycl_buffer_memory_storage_t *>( \
            &CTX_OUT_STORAGE(arg)) \
            ->buffer() \
            .get_access<cl::sycl::access::mode::write>(cgh)

#define CTX_IN_ACCESSOR(arg) \
    utils::downcast<sycl::sycl_buffer_memory_storage_t *>( \
            &CTX_IN_STORAGE(arg)) \
            ->buffer() \
            .get_access<cl::sycl::access::mode::read>(cgh)

#define CTX_SCRATCH_ACCESSOR(arg) \
    utils::downcast<sycl::sycl_buffer_memory_storage_t *>( \
            ctx.get_scratchpad_grantor().get_memory_storage(arg).get()) \
            ->buffer() \
            .get_access<cl::sycl::access::mode::read_write>(cgh)

bool compare_cuda_devices(
        const cl::sycl::device &lhs, const cl::sycl::device &rhs);

// Check if the device type matches the passed engine kind
inline status_t check_device(dnnl::impl::engine_kind_t eng_kind) {
    return (eng_kind == dnnl::impl::engine_kind::gpu
                    ? status::success
                    : status::invalid_arguments);
}

static void convert_dnnl_dims_array(
        const dnnl_dim_t *dims, int *new_dims, int n_dims) {
    for (size_t i = 0; i < n_dims; i++) {
        new_dims[i] = static_cast<int>(dims[i]);
    }
}

static void convert_dims(const dnnl_dim_t *dims, int *new_dims, int n_dims,
        int adjustment_size = 4, int adjustment_value = 1) {
    convert_dnnl_dims_array(dims, new_dims, n_dims);
    for (size_t i = n_dims; i < adjustment_size; i++) {
        new_dims[i] = adjustment_value;
    }
}
static bool memory_desc_matches_nchw_vect_c(const memory_desc_t *mem_desc) {
    // Only one block is supported for second (C) dimension and the block size
    // must be 4 and the dimension has to be a multiple of block size.
    auto is_int_8 = utils::one_of(mem_desc->data_type, data_type::s8);
    auto &strides = mem_desc->format_desc.blocking.strides;
    if (is_int_8 && mem_desc->format_desc.blocking.inner_nblks == 1
            && mem_desc->format_desc.blocking.inner_idxs[0] == 1
            && mem_desc->format_desc.blocking.inner_blks[0] == 4
            && mem_desc->dims[1] % 4 == 0) {
        for (int d = 0; d < mem_desc->ndims - 1; ++d)
            if (strides[d] < strides[d + 1]) return false;
        return true;
    }
    return false;
}

static bool has_different_block_size(
        const memory_desc_t *src_md, const memory_desc_t *dst_md) {
    return ((src_md->format_desc.blocking.inner_nblks > 0
                    && dst_md->format_desc.blocking.inner_nblks == 0)
            || (src_md->format_desc.blocking.inner_nblks == 0
                    && dst_md->format_desc.blocking.inner_nblks > 0));
}
static bool adjust_dim_for_dnn(
        int *dims, int n_dims, const memory_desc_t *mem_desc) {
    if (memory_desc_matches_nchw_vect_c(mem_desc)) {
        dims[n_dims] = mem_desc->format_desc.blocking.inner_blks[0];
        dims[mem_desc->format_desc.blocking.inner_idxs[0]]
                /= mem_desc->format_desc.blocking.inner_blks[0];
        return true;
    }
    return false;
}

static bool adjust_stride_for_dnn(
        int *stride, int n_dims, const memory_desc_t *mem_desc) {
    if (memory_desc_matches_nchw_vect_c(mem_desc)) {
        stride[n_dims] = mem_desc->format_desc.blocking.inner_nblks;
        return true;
    }
    return false;
}

// Check if the dimensions contain any zeros, returns true if they do.
static bool has_zero_dims(const dnnl_dim_t *dims, int n_dims) {
    for (size_t i = 0; i < n_dims; i++) {
        if (dims[i] == 0) { return true; }
    }
    return false;
}

static status_t get_format(const memory_desc_t *md, cudnnTensorFormat_t &format,
        bool consider_ab_as_nhwc = false) {
    const memory_desc_wrapper mem_wrapper(md);
    if (memory_desc_matches_nchw_vect_c(md)) {
        format = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW_VECT_C;
    } else if (mem_wrapper.matches_one_of_tag(format_tag::ab, format_tag::abc,
                       format_tag::abcd, format_tag::abcde,
                       format_tag::abcdef)) {
        format = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
    } else if (mem_wrapper.matches_one_of_tag(
                       format_tag::acb, format_tag::acdb, format_tag::acdeb)) {
        format = cudnnTensorFormat_t::CUDNN_TENSOR_NHWC;
    } else {
        return status::unimplemented;
    }
    if (consider_ab_as_nhwc && mem_wrapper.matches_one_of_tag(format_tag::ab)) {
        format = cudnnTensorFormat_t::CUDNN_TENSOR_NHWC;
    }
    return status::success;
}

static bool memory_format_ok(const memory_desc_t *mem_desc) {
    return (memory_desc_matches_nchw_vect_c(mem_desc)
            || mem_desc->format_desc.blocking.inner_nblks == 0);
}

static status_t convert_data_type(const memory_desc_t *mem_desc,
        cudnnDataType_t *cudnn_data_type, bool vectorized = true) {
    switch (mem_desc->data_type) {
        case dnnl_data_type_t::dnnl_f16:
            *cudnn_data_type = cudnnDataType_t::CUDNN_DATA_HALF;
            break;
        case dnnl_data_type_t::dnnl_f32:
            *cudnn_data_type = cudnnDataType_t::CUDNN_DATA_FLOAT;
            break;
            // CUDNN_TENSOR_NCHW_VECT_C format is only supported with tensor
            // data types CUDNN_DATA_INT8x4, CUDNN_DATA_INT8x32, and
            // CUDNN_DATA_UINT8x4. oneDNN does not support UINT8 and block size
            // of 32, hence the only valid case is CUDNN_DATA_INT8x4
        case dnnl_data_type_t::dnnl_s8:
            *cudnn_data_type
                    = ((vectorized
                               && mem_desc->format_desc.blocking.inner_blks[0]
                                       == 4)
                                    ? cudnnDataType_t::CUDNN_DATA_INT8x4
                                    : cudnnDataType_t::CUDNN_DATA_INT8);
            break;
        default: return status::unimplemented;
    }
    return status::success;
}

class cublas_error : virtual public std::runtime_error {

protected:
    const char *cublas_error_map(cublasStatus_t error) {
        switch (error) {
            case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";

            case CUBLAS_STATUS_NOT_INITIALIZED:
                return "CUBLAS_STATUS_NOT_INITIALIZED";

            case CUBLAS_STATUS_ALLOC_FAILED:
                return "CUBLAS_STATUS_ALLOC_FAILED";

            case CUBLAS_STATUS_INVALID_VALUE:
                return "CUBLAS_STATUS_INVALID_VALUE";

            case CUBLAS_STATUS_ARCH_MISMATCH:
                return "CUBLAS_STATUS_ARCH_MISMATCH";

            case CUBLAS_STATUS_MAPPING_ERROR:
                return "CUBLAS_STATUS_MAPPING_ERROR";

            case CUBLAS_STATUS_EXECUTION_FAILED:
                return "CUBLAS_STATUS_EXECUTION_FAILED";

            case CUBLAS_STATUS_INTERNAL_ERROR:
                return "CUBLAS_STATUS_INTERNAL_ERROR";

            case CUBLAS_STATUS_NOT_SUPPORTED:
                return "CUBLAS_STATUS_NOT_SUPPORTED";

            case CUBLAS_STATUS_LICENSE_ERROR:
                return "CUBLAS_STATUS_LICENSE_ERROR";

            default: return "<unknown>";
        }
    }

    int error_number_;

public:
    explicit cublas_error(const std::string &message, cublasStatus_t result)
        : std::runtime_error(
                (message + std::string(cublas_error_map(result)))) {
        error_number_ = static_cast<int>(result);
    }

    virtual ~cublas_error() throw() {}

    virtual int get_error_number() const throw() { return error_number_; }
};

class cuda_error : virtual public std::runtime_error {

protected:
    inline const char *cuda_error_map(CUresult result) {
        switch (result) {
            case CUDA_SUCCESS: return "CUDA_SUCCESS";
            case CUDA_ERROR_NOT_PERMITTED: return "CUDA_ERROR_NOT_PERMITTED";
            case CUDA_ERROR_INVALID_CONTEXT:
                return "CUDA_ERROR_INVALID_CONTEXT";
            case CUDA_ERROR_INVALID_DEVICE: return "CUDA_ERROR_INVALID_DEVICE";
            case CUDA_ERROR_INVALID_VALUE: return "CUDA_ERROR_INVALID_VALUE";
            case CUDA_ERROR_OUT_OF_MEMORY: return "CUDA_ERROR_OUT_OF_MEMORY";
            case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
                return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
            default: return "<unknown>";
        }
    }
    int error_number_;

public:
    explicit cuda_error(const std::string &message, CUresult result)
        : std::runtime_error((message + std::string(cuda_error_map(result)))) {
        error_number_ = static_cast<int>(result);
    }

    explicit cuda_error(const std::string &message, cudaError_t result)
        : std::runtime_error(
                (message + std::to_string(static_cast<int>(result)))) {
        error_number_ = static_cast<int>(result);
    }
    virtual ~cuda_error() throw() {}

    virtual int get_error_number() const throw() { return error_number_; }
};

class cudnn_error : virtual public std::runtime_error {

protected:
    inline const char *cudnn_get_error_string(cudnnStatus_t status) {
        switch (status) {
            case CUDNN_STATUS_SUCCESS: return "CUDNN_STATUS_SUCCESS";
            case CUDNN_STATUS_NOT_INITIALIZED:
                return "CUDNN_STATUS_NOT_INITIALIZED";
            case CUDNN_STATUS_ALLOC_FAILED: return "CUDNN_STATUS_ALLOC_FAILED";
            case CUDNN_STATUS_BAD_PARAM: return "CUDNN_STATUS_BAD_PARAM";
            case CUDNN_STATUS_INTERNAL_ERROR:
                return "CUDNN_STATUS_INTERNAL_ERROR";
            case CUDNN_STATUS_INVALID_VALUE:
                return "CUDNN_STATUS_INVALID_VALUE";
            case CUDNN_STATUS_ARCH_MISMATCH:
                return "CUDNN_STATUS_ARCH_MISMATCH";
            case CUDNN_STATUS_MAPPING_ERROR:
                return "CUDNN_STATUS_MAPPING_ERROR";
            case CUDNN_STATUS_EXECUTION_FAILED:
                return "CUDNN_STATUS_EXECUTION_FAILED";
            case CUDNN_STATUS_NOT_SUPPORTED:
                return "CUDNN_STATUS_NOT_SUPPORTED";
            case CUDNN_STATUS_LICENSE_ERROR:
                return "CUDNN_STATUS_LICENSE_ERROR";
            case CUDNN_STATUS_RUNTIME_IN_PROGRESS:
                return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
            case CUDNN_STATUS_RUNTIME_FP_OVERFLOW:
                return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
            default: return "<unknown>";
        }
    }
    int error_number_;

public:
    explicit cudnn_error(const std::string &message, cudnnStatus_t result)
        : std::runtime_error(
                (message + std::string(cudnn_get_error_string(result)))) {
        error_number_ = static_cast<int>(result);
    }

    virtual ~cudnn_error() throw() {}

    virtual int get_error_number() const throw() { return error_number_; }
};

template <typename T>
cl::sycl::event copy(cl::sycl::queue &q, T *src, cl::sycl::buffer<T, 1> &dst) {

    auto event = q.submit([&, src](cl::sycl::handler &cgh) {
        // Retrieve a  write accessor to a global buffer
        auto acc = dst.template get_access<cl::sycl::access::mode::write,
                cl::sycl::access::target::global_buffer>(cgh);
        // Copy from the input pointer into the buffer associated with the
        // accessor
        cgh.copy(src, acc);
    });
    return event;
}

template <typename T>
cl::sycl::event copy(cl::sycl::queue &q, cl::sycl::buffer<T, 1> &src, T *dst) {

    auto event = q.submit([&, dst](cl::sycl::handler &cgh) {
        // Retrieve a read accessor to a global buffer
        auto acc = src.template get_access<cl::sycl::access::mode::read,
                cl::sycl::access::target::global_buffer>(cgh);
        // Copy from the buffer associated with the accessor into the output
        // pointer
        cgh.copy(acc, dst);
    });

    return event;
}

template <typename T>
cl::sycl::event copy(cl::sycl::queue &q, cl::sycl::buffer<T, 1> &src,
        cl::sycl::buffer<T, 1> &dst) {
    auto event = q.submit([&](cl::sycl::handler &cgh) {
        auto src_acc
                = src.template get_access<cl::sycl::access::mode::read_write>(
                        cgh);
        auto dst_acc
                = dst.template get_access<cl::sycl::access::mode::read_write>(
                        cgh);
        cgh.copy(src_acc, dst_acc);
    });
    return event;
}

static status_t cudnn_to_dnnl_status(cudnnStatus_t cu_status) {
    switch (cu_status) {
        case CUDNN_STATUS_SUCCESS: return status::success;
        case CUDNN_STATUS_BAD_PARAM: return status::invalid_arguments;
        case CUDNN_STATUS_NOT_SUPPORTED: return status::unimplemented;
        default: return status::runtime_error;
    }
}

static status_t cublas_to_dnnl_status(cublasStatus_t cu_status) {
    switch (cu_status) {
        case CUBLAS_STATUS_SUCCESS: return status::success;
        default: return status::runtime_error;
    }
}

static status_t cuda_to_dnnl_status(CUresult cu_result) {
    switch (cu_result) {
        case CUDA_SUCCESS: return status::success;
        default: return status::runtime_error;
    }
}

#define CUDA_ERROR_LOCATION __FILE__ " : " STRINGIFY(__LINE__)

#define CUDA_EXECUTE_FUNC(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != CUDA_SUCCESS) { \
            throw cuda_error(std::string("At :") \
                            + std::string(CUDA_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err); \
        } \
    }

#define CUBLAS_EXECUTE_FUNC(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            throw cublas_error(std::string("At :") \
                            + std::string(CUDA_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err); \
        } \
    }

#define CUDNN_EXECUTE_FUNC(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != CUDNN_STATUS_SUCCESS) { \
            throw cudnn_error(std::string("At :") \
                            + std::string(CUDA_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err); \
        } \
    }

#define CUDA_EXECUTE_FUNC_V(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != CUDA_SUCCESS) { \
            std::cout << cuda_error(std::string("At :") \
                            + std::string(CUDA_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define CUDNN_EXECUTE_FUNC_V(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != CUDNN_STATUS_SUCCESS) { \
            std::cout << cudnn_error(std::string("At :") \
                            + std::string(CUDA_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define CUBLAS_EXECUTE_FUNC_V(name, ...) \
    { \
        auto err = name(__VA_ARGS__); \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::cout << cublas_error(std::string("At :") \
                            + std::string(CUDA_ERROR_LOCATION) \
                            + std::string(#name) + std::string(" : "), \
                    err) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define CUDNN_CHECK_V(e) \
    { \
        auto status = (e); \
        if (status != CUDNN_STATUS_SUCCESS) { \
            std::cout << cudnn_error(std::string("At :") \
                            + std::string(CUDA_ERROR_LOCATION) \
                            + std::string(" : "), \
                    status) \
                                 .what() \
                      << std::endl; \
        } \
    }

#define CUDA_EXECUTE_FUNC_S(name, ...) \
    [&]() { \
        auto err = name(__VA_ARGS__); \
        return cuda_to_dnnl_status(err); \
    }()

#define CUBLAS_EXECUTE_FUNC_S(name, ...) \
    [&]() { \
        auto err = name(__VA_ARGS__); \
        return cublas_to_dnnl_status(err); \
    }()

#define CUDNN_EXECUTE_FUNC_S(name, ...) \
    [&]() { \
        auto err = name(__VA_ARGS__); \
        if (err != CUDNN_STATUS_SUCCESS) { return cudnn_to_dnnl_status(err); } \
        return status::success; \
    }()

static status_t create_and_set_tensor_descriptor(
        cudnnTensorDescriptor_t *tensor_desc, cudnnDataType_t data_type,
        int ndims, int *dims, int *strides) {

    CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreateTensorDescriptor, tensor_desc));

    CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetTensorNdDescriptor, *tensor_desc,
            data_type, ndims, dims, strides));

    return status::success;
}

static status_t create_and_set_tensor_descriptor_ex(
        cudnnTensorDescriptor_t *tensor_desc, cudnnTensorFormat_t format,
        cudnnDataType_t data_type, int ndims, int *dims) {

    CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreateTensorDescriptor, tensor_desc));

    CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetTensorNdDescriptorEx, *tensor_desc,
            format, data_type, ndims, dims));

    return status::success;
}

static status_t create_and_set_filter_descriptor(
        cudnnFilterDescriptor_t *filter_desc, cudnnTensorFormat_t format,
        cudnnDataType_t data_type, int ndims, int *dims, int *) {
    CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreateFilterDescriptor, filter_desc));

    CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetFilterNdDescriptor, *filter_desc,
            data_type, format, ndims, dims));

    return status::success;
}

static status_t create_and_set_conv_descriptor(
        cudnnConvolutionDescriptor_t *conv_desc, int ndims, int *padding,
        int *strides, int *dilation, cudnnConvolutionMode_t mode,
        cudnnDataType_t data_type) {
    CHECK(CUDNN_EXECUTE_FUNC_S(cudnnCreateConvolutionDescriptor, conv_desc));

    CHECK(CUDNN_EXECUTE_FUNC_S(cudnnSetConvolutionNdDescriptor, *conv_desc,
            ndims, padding, strides, dilation, mode, data_type));

    return status::success;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

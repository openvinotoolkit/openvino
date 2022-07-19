// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <cmath>
#include <dnnl_extension_utils.h>
#include <onednn/dnnl.h>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <common/primitive_hashing_utils.hpp>

#include "rdft.h"
#include "ie_parallel.hpp"
#include "ie_precision.hpp"

#include "utils/general_utils.h"
#include "common/cpu_memcpy.h"
#include <openvino/op/rdft.hpp>
#include <openvino/op/irdft.hpp>

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {


static const size_t DATA_INDEX = 0;
static const size_t AXES_INDEX = 1;
static const size_t SIGNAL_SIZE_INDEX = 2;
static constexpr double PI = 3.14159265358979323846;


bool RDFT::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (isDynamicNgraphNode(op)) {
            errorMessage = "Doesn't support op with dynamic shapes";
            return false;
        }
        const bool is_rdft = is_type<const ov::op::v9::RDFT>(op);
        const bool is_irdft = is_type<const ov::op::v9::IRDFT>(op);

        if (!is_rdft && !is_irdft) {
            errorMessage = "Only opset9 RDFT/IRDFT operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

RDFT::RDFT(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache) :
               Node(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    std::string error_msg_prefix = "RDFT layer with name '" + op->get_name() + "'";
    const size_t num_inputs = getOriginalInputsNumber();
    if (num_inputs != 2 && num_inputs != 3) {
        IE_THROW() << error_msg_prefix << " has invalid number of input/output edges: " << num_inputs;
    }

    const auto axes_rank = inputShapes[AXES_INDEX].getRank();
    if (axes_rank != 1) {
        IE_THROW() << error_msg_prefix << " has invalid 'axes' input tensor with rank: " << axes_rank;
    }

    if (num_inputs > SIGNAL_SIZE_INDEX) {
        const auto signal_size_rank = inputShapes[SIGNAL_SIZE_INDEX].getRank();
        if (signal_size_rank != 1) {
            IE_THROW() << error_msg_prefix << " has invalid 'signal_size' input tensor with rank: " << signal_size_rank;
        }
    }

    inverse = ov::is_type<ov::op::v9::IRDFT>(op);
}

void RDFT::getSupportedDescriptors() {}

void RDFT::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    const auto& data_precision = getOriginalInputPrecisionAtPort(DATA_INDEX);
    if (!data_precision.is_float()) {
        IE_THROW() << error_msg_prefix << " has unsupported 'data' input precision: " << data_precision.name();
    }

    const auto& axes_precision = getOriginalInputPrecisionAtPort(AXES_INDEX);
    if (axes_precision != Precision::I32 && axes_precision != Precision::I64) {
        IE_THROW() << error_msg_prefix << " has unsupported 'axes' input precision: " << axes_precision.name();
    }

    if (inputShapes.size() > SIGNAL_SIZE_INDEX) {
        const auto& signal_size_precision = getOriginalInputPrecisionAtPort(SIGNAL_SIZE_INDEX);
        if (signal_size_precision != Precision::I32 && signal_size_precision != Precision::I64) {
            IE_THROW() << error_msg_prefix << " has unsupported 'signal_size' input precision: " << signal_size_precision.name();
        }
    }

    std::vector<PortConfigurator> configurators({{LayoutType::ncsp, Precision::FP32},
                                                 {LayoutType::ncsp, Precision::I32}});
    if (inputShapes.size() > SIGNAL_SIZE_INDEX)
        configurators.push_back({LayoutType::ncsp, Precision::I32});

    addSupportedPrimDesc(configurators, {{LayoutType::ncsp, Precision::FP32}}, impl_desc_type::ref_any);
}

static void normalize_axes(std::vector<int>& axes, size_t rank) {
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += rank;
        }
    }
}

void RDFT::execute(dnnl::stream strm) {
    const auto& input_mem = getParentEdgeAt(DATA_INDEX)->getMemory();
    const auto& output_mem = getChildEdgeAt(0)->getMemory();
    const auto& input_shape = input_mem.getStaticDims();
    const auto& output_shape = output_mem.getStaticDims();

    auto input_ptr = reinterpret_cast<float*>(input_mem.GetPtr());
    auto output_ptr = reinterpret_cast<float*>(output_mem.GetPtr());

    auto rank = inverse ? output_shape.size() : input_shape.size();
    const auto& axes_mem = getParentEdgeAt(AXES_INDEX)->getMemoryPtr();
    auto axes_ptr = reinterpret_cast<const int32_t*>(axes_mem->GetPtr());
    std::vector<int> axes(axes_ptr, axes_ptr + axes_mem->getStaticDims()[0]);

    std::vector<int> signal_sizes;
    if (SIGNAL_SIZE_INDEX < getOriginalInputsNumber()) {
        const auto& signal_size_mem = getParentEdgeAt(SIGNAL_SIZE_INDEX)->getMemoryPtr();
        auto signal_ptr = reinterpret_cast<const int32_t*>(signal_size_mem->GetPtr());
        signal_sizes = std::vector<int>(signal_ptr, signal_ptr + signal_size_mem->getStaticDims()[0]);
    }

    normalize_axes(axes, rank);

    const auto& input_strides = input_mem.GetDescWithType<BlockedMemoryDesc>()->getStrides();
    const auto& output_strides = output_mem.GetDescWithType<BlockedMemoryDesc>()->getStrides();

    executor->execute(input_ptr, output_ptr, rank,
                      axes, signal_sizes,
                      input_shape, output_shape,
                      input_strides, output_strides);
}

bool RDFT::created() const {
    return getType() == Type::RDFT;
}

void RDFT::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

bool RDFT::needPrepareParams() const {
    return true;
}


static void adjust_signal_sizes(VectorDims& input_shape,
                                std::vector<int>& signal_sizes,
                                const VectorDims& output_shape,
                                const std::vector<int>& axes,
                                bool is_inverse) {
    if (signal_sizes.size() == 0) {
        for (auto axis : axes) {
            signal_sizes.push_back(input_shape[axis]);
        }
        if (is_inverse) {
            signal_sizes[signal_sizes.size() - 1] = 2 * (input_shape[axes.back()] - 1);
        }
        return;
    }

    for (size_t i = 0; i < axes.size(); i++) {
        auto axis = axes[i];
        size_t input_size = input_shape[axis];
        size_t signal_size = signal_sizes[i];
        if (signal_size <= input_size) {
            input_shape[axis] = signal_size;
        } else if (!is_inverse) {
            IE_THROW() << "Signal size greater than input size is not supported yet";
        }
    }
    if (is_inverse) {
        input_shape[axes.back()] = signal_sizes.back() / 2 + 1;
    }
}

void RDFTExecutor::execute(float* input_ptr, float* output_ptr,
                           size_t rank, const std::vector<int>& axes,
                           std::vector<int> signal_sizes,
                           VectorDims input_shape, const VectorDims& output_shape,
                           const VectorDims& input_strides, const VectorDims& output_strides) {
    adjust_signal_sizes(input_shape, signal_sizes, output_shape, axes, is_inverse);
    generate_twiddles(signal_sizes, output_shape, axes);

    if (rank == 1) {
        auto twiddles_ptr = twiddles[0].data();
        dft_common(input_ptr, twiddles_ptr, output_ptr,
                   input_shape[0], signal_sizes[0], output_shape[0],
                   is_inverse ? complex_to_real : real_to_complex,
                   can_use_fft(signal_sizes[0]), true);
    } else {
        if (!is_inverse)
            rdft_nd(input_ptr, output_ptr, axes, signal_sizes, input_shape, input_strides, output_shape, output_strides);
        else
            irdft_nd(input_ptr, output_ptr, axes, signal_sizes, input_shape, input_strides, output_shape, output_strides);
    }
}

static void coords_from_index(size_t index, std::vector<size_t>& coords, const std::vector<size_t>& shape, int exclude_axis) {
    for (size_t i = coords.size(); i > 0; i--) {
        if (exclude_axis == i - 1) {
            coords[i - 1] = 0;
            continue;
        }
        coords[i - 1] = index % shape[i - 1];
        index /= shape[i - 1];
    }
}

static size_t get_offset(const std::vector<size_t>& coords, const std::vector<size_t>& strides) {
    size_t offset = 0;
    for (size_t i = 0; i < coords.size(); ++i) {
        offset += coords[i] * strides[i];
    }
    return offset;
}

static void gather_real(float* output, const float* input, size_t axis,
                        const std::vector<size_t>& coords,
                        size_t size, const std::vector<size_t>& strides) {
    size_t input_offset = get_offset(coords, strides);

    for (size_t i = 0; i < size; i++) {
        output[i] = input[input_offset];
        input_offset += strides[axis];
    }
}

static void gather_complex(float* output, const float* input, size_t axis,
                                     const std::vector<size_t>& coords,
                                     size_t size, const std::vector<size_t>& strides) {
    size_t input_offset = get_offset(coords, strides);

    for (size_t i = 0; i < 2 * size; i += 2) {
        output[i] = input[input_offset];
        output[i + 1] = input[input_offset + 1];
        input_offset += strides[axis];
    }
}

static void scatter_real(float* output, const float* input, size_t axis,
                         const std::vector<size_t>& coords,
                         size_t size, const std::vector<size_t>& strides) {
    size_t offset = get_offset(coords, strides);

    for (size_t i = 0; i < size; i++) {
        output[offset] = input[i];
        offset += strides[axis];
    }
}

static void scatter_complex(float* output, const float* input, size_t axis,
                            const std::vector<size_t>& coords,
                            size_t size, const std::vector<size_t>& strides) {
    size_t offset = get_offset(coords, strides);

    for (size_t i = 0; i < 2 * size; i += 2) {
        output[offset] = input[i];
        output[offset + 1] = input[i + 1];
        offset += strides[axis];
    }
}

static bool is_power_of_two(size_t n) {
    return (n != 0) && (n & (n - 1)) == 0;
}

static size_t dft_simd_size(int vlen) {
    return vlen / (2 * sizeof(float));
}

void RDFTExecutor::dft_common(float* input_ptr, float* twiddles_ptr, float* output_ptr,
                              size_t input_size, size_t signal_size, size_t output_size,
                              enum dft_type type, bool use_fft, bool parallelize) {
    if (use_fft) {
        fft(input_ptr, twiddles_ptr, output_ptr,
            input_size, signal_size, output_size,
            type, parallelize);
    } else {
        dft(input_ptr, twiddles_ptr, output_ptr,
            input_size, signal_size, output_size,
            type, parallelize);
    }
}

void RDFTExecutor::dft_on_axis(enum dft_type type,
                               float* input_ptr, float* output_ptr,
                               float* twiddles_ptr, int axis,
                               size_t signal_size,
                               const VectorDims& input_shape,
                               const VectorDims& input_strides,
                               const VectorDims& output_shape,
                               const VectorDims& output_strides,
                               const std::vector<size_t>& iteration_range) {
    size_t input_size = input_shape[axis];
    size_t output_size = output_shape[axis];

    void (*gather)(float* output, const float* input,
                   size_t axis, const std::vector<size_t>& coords,
                   size_t size, const std::vector<size_t>& strides) = nullptr;
    void (*scatter)(float* output, const float* input,
                    size_t axis, const std::vector<size_t>& coords,
                    size_t size, const std::vector<size_t>& strides) = nullptr;

    size_t gather_size = 0;
    size_t scatter_size = 0;

    switch (type) {
    case real_to_complex:
        scatter = scatter_complex;
        gather = gather_real;
        gather_size = input_size;
        scatter_size = output_size * 2;
        break;
    case complex_to_complex:
        gather = gather_complex;
        scatter = scatter_complex;
        gather_size = input_size * 2;
        scatter_size = output_size * 2;
        break;
    case complex_to_real:
        gather = gather_complex;
        scatter = scatter_real;
        gather_size = input_size * 2;
        scatter_size = output_size;
        break;
    }

    bool use_fft = can_use_fft(signal_size);

    size_t total_work_size = std::accumulate(iteration_range.begin(),
                                             iteration_range.end(),
                                             1, std::multiplies<size_t>()) / iteration_range[axis];
    bool parallelize_outer_axes = total_work_size > signal_size;

    if (parallelize_outer_axes) {
        parallel_for(total_work_size, [&] (size_t i) {
            std::vector<size_t> coords(iteration_range.size(), 0);
            std::vector<float> gather_scatter_buffer(gather_size + scatter_size);
            float* gather_buffer = &gather_scatter_buffer[0];
            float* scatter_buffer = &gather_scatter_buffer[gather_size];
            coords_from_index(i, coords, iteration_range, axis);
            gather(gather_buffer, input_ptr,
                   axis, coords,
                   input_size, input_strides);
            dft_common(gather_buffer, twiddles_ptr, scatter_buffer,
                       input_size, signal_size, output_size,
                       type, use_fft, !parallelize_outer_axes);
            scatter(output_ptr, scatter_buffer, axis, coords, output_size, output_strides);
        });
    } else {
        std::vector<size_t> coords(iteration_range.size(), 0);
        std::vector<float> gather_scatter_buffer(gather_size + scatter_size);
        float* gather_buffer = &gather_scatter_buffer[0];
        float* scatter_buffer = &gather_scatter_buffer[gather_size];
        for (size_t i = 0; i < total_work_size; i++) {
            coords_from_index(i, coords, iteration_range, axis);
            gather(gather_buffer, input_ptr,
                   axis, coords,
                   input_size, input_strides);
            dft_common(gather_buffer, twiddles_ptr, scatter_buffer,
                       input_size, signal_size, output_size,
                       type, use_fft, !parallelize_outer_axes);
            scatter(output_ptr, scatter_buffer, axis, coords, output_size, output_strides);
        }
    }
}

// N-dimensional real DFT
void RDFTExecutor::rdft_nd(float* input_ptr, float* output_ptr,
                           const std::vector<int>& axes,
                           const std::vector<int>& signal_sizes,
                           const VectorDims& input_shape,
                           const VectorDims& input_strides,
                           const VectorDims& output_shape,
                           const VectorDims& output_strides) {
    const std::vector<size_t> iteration_range(output_shape.begin(), output_shape.end() - 1);

    dft_on_axis(real_to_complex, input_ptr, output_ptr,
                twiddles.back().data(), axes.back(),
                signal_sizes.back(),
                input_shape, input_strides,
                output_shape, output_strides,
                iteration_range);
    input_ptr = output_ptr;

    for (size_t i = 0; i < axes.size() - 1; i++) {
        auto axis = axes[i];
        dft_on_axis(complex_to_complex, input_ptr, output_ptr,
                    twiddles[i].data(), axis,
                    signal_sizes[i],
                    output_shape, output_strides,
                    output_shape, output_strides,
                    iteration_range);
    }
}

// N-dimensional real inverse DFT
void RDFTExecutor::irdft_nd(float* input_ptr, float* output_ptr,
                            const std::vector<int>& axes,
                            const std::vector<int>& signal_sizes,
                            const VectorDims& input_shape,
                            const VectorDims& input_strides,
                            const VectorDims& output_shape,
                            const VectorDims& output_strides) {
    const std::vector<size_t> iteration_range(input_shape.begin(), input_shape.end() - 1);

    if (axes.size() == 1) {
        dft_on_axis(complex_to_real, input_ptr, output_ptr,
                    twiddles[0].data(), axes[0],
                    signal_sizes[0],
                    input_shape, input_strides,
                    output_shape, output_strides,
                    iteration_range);
        return;
    }

    float* output = output_ptr;
    std::vector<float> tmp;
    size_t input_shape_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<size_t>());
    size_t output_shape_size = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<size_t>());
    if (input_shape_size > output_shape_size) {
        tmp.resize(input_shape_size);
        output = &tmp[0];
    }

    std::vector<size_t> adjusted_input_strides(input_strides.size());
    adjusted_input_strides[adjusted_input_strides.size() - 1] = 1;
    for (size_t i = adjusted_input_strides.size() - 1; i > 0; i--) {
        adjusted_input_strides[i - 1] = adjusted_input_strides[i] * input_shape[i];
    }

    for (size_t i = 0; i < axes.size() - 1; i++) {
        auto axis = axes[i];
        dft_on_axis(complex_to_complex, input_ptr, output,
                    twiddles[i].data(), axis,
                    signal_sizes[i],
                    input_shape, input_strides,
                    input_shape, adjusted_input_strides,
                    iteration_range);
        input_ptr = output;
    }
    dft_on_axis(complex_to_real, input_ptr, output_ptr,
                twiddles.back().data(), axes.back(),
                signal_sizes.back(),
                input_shape, adjusted_input_strides,
                output_shape, output_strides,
                iteration_range);
}

std::vector<float> RDFTExecutor::generate_twiddles_fft(size_t N) {
    std::vector<float> twiddles;
    for (size_t num_blocks = 1; num_blocks < N; num_blocks *= 2) {
        for (size_t block = 0; block < num_blocks; block++) {
            double angle = 2 * PI * block / (num_blocks * 2);
            twiddles.push_back(std::cos(angle));
            twiddles.push_back(-std::sin(angle));
        }
    }
    return twiddles;
}

std::vector<float> RDFTExecutor::generate_twiddles_common(size_t signal_size, size_t output_size,
                                                          enum dft_type type, bool use_fft) {
    if (use_fft) {
        return generate_twiddles_fft(signal_size);
    }
    return generate_twiddles_dft(signal_size, output_size, type);
}

void RDFTExecutor::generate_twiddles(const std::vector<int>& signal_sizes,
                                     const std::vector<size_t>& output_shape,
                                     const std::vector<int>& axes) {
    for (size_t i = 0; i < axes.size(); i++) {
        auto axis = axes[i];
        size_t N = signal_sizes[i];
        size_t K = output_shape[axis];
        auto type = complex_to_complex;
        if (i == axes.size() - 1)
            type = is_inverse ? complex_to_real : real_to_complex;
        twiddles.push_back(generate_twiddles_common(N, K, type, can_use_fft(N)));
    }
}

static std::vector<int> generate_fft_indices(int vlen) {
    size_t simd_size = dft_simd_size(vlen);
    std::vector<int> indices;
    for (int num_blocks = 1; num_blocks < simd_size; num_blocks *= 2) {
        int block_size = simd_size / num_blocks;
        for (int block = 0; block < simd_size / (block_size / 2); block++) {
            int in_base = block * block_size;
            for (int pair = 0; pair < block_size / 2; pair++) {
                int even = (in_base + pair);
                indices.push_back(even * 2);
                indices.push_back(even * 2 + 1);
            }
        }
        for (int block = 0; block < simd_size / (block_size / 2); block++) {
            int in_base = block * block_size;
            for (int pair = 0; pair < block_size / 2; pair++) {
                int odd = (in_base + block_size / 2 + pair);
                indices.push_back(odd * 2);
                indices.push_back(odd * 2 + 1);
            }
        }
        for (int block = 0; block < simd_size / (block_size / 2); block++) {
            for (int pair = 0; pair < block_size / 2; pair++) {
                indices.push_back(block * 2);
                indices.push_back(block * 2);
            }
        }
    }
    return indices;
}

static void fft_copy_inverse_input_data(float* dst, float* src, size_t input_size, size_t signal_size, bool parallelize) {
    if (!parallelize) {
        cpu_memcpy(dst, src, input_size * complex_type_size<float>());
        src = src + 2 * input_size - 4;
        for (size_t i = input_size; i < signal_size; i++, src -= 2) {
            dst[2 * i] = src[0];
            dst[2 * i + 1] = -src[1];
        }
    } else {
        parallel_for(signal_size, [&] (size_t i) {
                if (i < input_size) {
                    dst[2 * i] = src[2 * i];
                    dst[2 * i + 1] = src[2 * i + 1];
                } else {
                    size_t src_idx = 2 * input_size - 2 - i;
                    dst[2 * i] = src[2 * src_idx];
                    dst[2 * i + 1] = -src[2 * src_idx + 1];
                }
        });
    }
}

static void fft_copy_real_input_data(float* dst, float* src, size_t input_size, bool parallelize) {
    if (!parallelize) {
        for (size_t i = 0; i < input_size; i++) {
            dst[2 * i] = src[i];
            dst[2 * i + 1] = 0;
        }
    } else {
        parallel_for(input_size, [&] (size_t i) {
            dst[2 * i] = src[i];
            dst[2 * i + 1] = 0;
        });
    }
}

static void fft_copy_inverse_real_output(float* dst, float* src, size_t signal_size, bool parallelize) {
    if (!parallelize) {
        for (size_t i = 0; i < signal_size; i++) {
            dst[i] = src[2 * i];
        }
    } else {
        parallel_for(signal_size, [&] (size_t i) {
            dst[i] = src[2 * i];
        });
    }
}

struct RDFTJitExecutor : public RDFTExecutor {
    RDFTJitExecutor(bool inverse) : RDFTExecutor(inverse) {
        enum dft_type rdft_type = is_inverse ? complex_to_real : real_to_complex;
        if (mayiuse(cpu::x64::avx512_core)) {
            rdft_kernel.reset(new jit_dft_kernel_f32<cpu::x64::avx512_core>(is_inverse, rdft_type));
            dft_kernel.reset(new jit_dft_kernel_f32<cpu::x64::avx512_core>(is_inverse, complex_to_complex));
            fft_kernel.reset(new jit_fft_kernel_f32<cpu::x64::avx512_core>(is_inverse));
            vlen = cpu_isa_traits<cpu::x64::avx512_core>::vlen;
        } else if (mayiuse(cpu::x64::avx2)) {
            rdft_kernel.reset(new jit_dft_kernel_f32<cpu::x64::avx2>(is_inverse, rdft_type));
            dft_kernel.reset(new jit_dft_kernel_f32<cpu::x64::avx2>(is_inverse, complex_to_complex));
            fft_kernel.reset(new jit_fft_kernel_f32<cpu::x64::avx2>(is_inverse));
            vlen = cpu_isa_traits<cpu::x64::avx2>::vlen;
        } else if (mayiuse(cpu::x64::sse41)) {
            rdft_kernel.reset(new jit_dft_kernel_f32<cpu::x64::sse41>(is_inverse, rdft_type));
            dft_kernel.reset(new jit_dft_kernel_f32<cpu::x64::sse41>(is_inverse, complex_to_complex));
            vlen = cpu_isa_traits<cpu::x64::sse41>::vlen;
        } else {
            IE_THROW() << "Can't create RDFT kernel";
        }

        if (rdft_kernel)
            rdft_kernel->create_ker();
        if (dft_kernel)
            dft_kernel->create_ker();
        if (fft_kernel)
            fft_kernel->create_ker();

        fft_indices = generate_fft_indices(vlen);
    }

    bool can_use_fft(size_t dim) override {
        return is_power_of_two(dim) &&
               fft_kernel &&
               dim >= 4 * dft_simd_size(vlen);
    }

    std::vector<float> generate_twiddles_dft(size_t input_size, size_t output_size, enum dft_type type) override {
        std::vector<float> twiddles(input_size * output_size * 2);
        int simd_size = vlen / sizeof(float);
        if (type == real_to_complex || type == complex_to_complex) {
            simd_size /= 2; // there are two floats per one complex element in the output
        }

        parallel_for(output_size / simd_size, [&] (size_t K) {
            for (size_t n = 0; n < input_size; n++) {
                if (type == real_to_complex) {
                    for (size_t k = 0; k < simd_size; k++) {
                        double angle = 2 * PI * (K * simd_size + k) * n / input_size;
                        twiddles[((K * input_size + n) * simd_size + k) * 2] = std::cos(angle);
                        twiddles[((K * input_size + n) * simd_size + k) * 2 + 1] = -std::sin(angle);
                    }
                } else if (type == complex_to_real || type == complex_to_complex) {
                    for (size_t k = 0; k < simd_size; k++) {
                        double angle = 2 * PI * (K * simd_size + k) * n / input_size;
                        twiddles[(K * input_size + n) * 2 * simd_size + k] = std::cos(angle);
                    }
                    for (size_t k = 0; k < simd_size; k++) {
                        double angle = 2 * PI * (K * simd_size + k) * n / input_size;
                        twiddles[((K * input_size + n) * 2 + 1) * simd_size + k] = -std::sin(angle);
                    }
                }
            }
        });
        if ((output_size % simd_size) != 0) {
            size_t start = (output_size / simd_size) * simd_size;
            parallel_for(output_size - start, [&] (size_t k) {
                k += start;
                for (size_t n = 0; n < input_size; n++) {
                    double angle = 2 * PI * k * n / input_size;
                    twiddles[2 * (k * input_size + n)] = std::cos(angle);
                    twiddles[2 * (k * input_size + n) + 1] = -std::sin(angle);
                }
            });
        }
        return twiddles;
    }

    void dft(float* input_ptr, float* twiddles_ptr, float* output_ptr,
             size_t input_size, size_t signal_size, size_t output_size,
             enum dft_type type, bool parallelize) override {
        jit_dft_kernel* kernel = type == complex_to_complex ? dft_kernel.get() : rdft_kernel.get();
        if (parallelize) {
            const int cacheline_size = 64;
            size_t block_size = 4 * cacheline_size / sizeof(float);
            size_t num_blocks = (output_size + block_size - 1) / block_size;
            parallel_nt(num_blocks, [&] (size_t i, size_t nthr) {
                if (num_blocks > nthr) {
                    auto new_block_size = (((output_size / nthr) + block_size - 1) / block_size) * block_size;
                    block_size = new_block_size;
                    num_blocks = nthr;
                }
                jit_dft_args args{};
                args.input = input_ptr,
                args.twiddles = twiddles_ptr,
                args.output = output_ptr,
                args.input_size = input_size,
                args.signal_size = signal_size,
                args.output_start = i * block_size,
                args.output_end = std::min(output_size - i * block_size, block_size),
                (*kernel)(&args);
            });
        } else {
            jit_dft_args args{};
            args.input = input_ptr,
            args.twiddles = twiddles_ptr,
            args.output = output_ptr,
            args.input_size = input_size,
            args.signal_size = signal_size,
            args.output_start = 0,
            args.output_end = output_size,
            (*kernel)(&args);
        }
    }

    void fft(float* input, float* twiddles_ptr, float* output,
             size_t input_size, size_t signal_size, size_t output_size,
             enum dft_type type, bool parallelize) override {
        std::vector<float> scratch_space(4 * signal_size, 0);

        float* input_ptr = input;
        float* output_ptr = &scratch_space[2 * signal_size];
        int* indices_ptr = &fft_indices[0];
        size_t simd_size = dft_simd_size(vlen);
        size_t input_stride = simd_size;

        if (input_size < signal_size || type == real_to_complex) {
            if (is_inverse)
                fft_copy_inverse_input_data(&scratch_space[0], input, input_size, signal_size, parallelize);
            else if (type == real_to_complex)
                fft_copy_real_input_data(&scratch_space[0], input, input_size, parallelize);
            input_ptr = &scratch_space[0];
        }

        size_t work_divide_factor = simd_size * 2;
        size_t block_size = 0;

        auto block_iteration = [&] (size_t i, size_t nthr) {
            size_t offset = i * work_divide_factor;
            size_t block = offset / block_size;
            size_t input_offset = block * block_size * 2 + offset % block_size;
            size_t output_offset = offset;
            struct jit_fft_args params{};
            params.input = input_ptr + input_offset,
            params.twiddles = twiddles_ptr,
            params.output = output_ptr + output_offset,
            params.signal_size = signal_size,
            params.block = block,
            params.block_size = block_size,
            params.subblock_start = i * work_divide_factor / 2,
            params.subblock_end = (i + 1) * work_divide_factor / 2,
            (*fft_kernel)(&params);
        };

        for (size_t num_blocks = 1; num_blocks < signal_size / simd_size; num_blocks *= 2) {
            block_size = signal_size / num_blocks;
            if (parallelize) {
                parallel_nt(signal_size / work_divide_factor, block_iteration);
            } else {
                for (size_t i = 0; i < signal_size / work_divide_factor; i++) {
                    block_iteration(i, 0);
                }
            }
            input_stride = simd_size * 2;
            twiddles_ptr += num_blocks * 2;
            std::swap(input_ptr, output_ptr);
        }

        auto small_block_iteration = [&] (size_t i, size_t nthr) {
            size_t offset = i * work_divide_factor;
            size_t input_offset = offset;
            size_t output_offset = offset;
            struct jit_fft_args params{};
            params.input = input_ptr + input_offset * 2,
            params.twiddles = twiddles_ptr + i * (simd_size / block_size) * 2 * 2,
            params.output = output_ptr + output_offset,
            params.indices = indices_ptr,
            params.signal_size = signal_size,
            params.block_size = block_size,
            params.subblock_start = i * work_divide_factor / 2,
            params.subblock_end = (i + 1) * work_divide_factor / 2,
            (*fft_kernel)(&params);
        };

        for (size_t num_blocks = signal_size / simd_size; num_blocks < signal_size; num_blocks *= 2) {
            block_size = signal_size / num_blocks;
            if (num_blocks == signal_size / 2 && output_size == signal_size && type != complex_to_real) {
                output_ptr = output;
            }
            if (parallelize) {
                parallel_nt(signal_size / work_divide_factor, small_block_iteration);
            } else {
                for (size_t i = 0; i < signal_size / work_divide_factor; i++) {
                    small_block_iteration(i, 0);
                }
            }
            indices_ptr += 3 * 2 * simd_size;
            twiddles_ptr += num_blocks * 2;
            std::swap(input_ptr, output_ptr);
        }

        if (type == complex_to_real) {
            fft_copy_inverse_real_output(output, input_ptr, signal_size, parallelize);
        } else if (output_size != signal_size) {
            cpu_memcpy(output, input_ptr, output_size * complex_type_size<float>());
        }
    }

    std::unique_ptr<jit_dft_kernel> rdft_kernel = nullptr;
    std::unique_ptr<jit_dft_kernel> dft_kernel = nullptr;
    std::unique_ptr<jit_fft_kernel> fft_kernel = nullptr;

    std::vector<int> fft_indices;
    int vlen;
};


struct RDFTRefExecutor : public RDFTExecutor {
    RDFTRefExecutor(bool inverse) : RDFTExecutor(inverse) {}

    private:
        bool can_use_fft(size_t dim) override {
            return is_power_of_two(dim) && dim > 1;
        }

        std::vector<float> generate_twiddles_dft(size_t input_size, size_t output_size, enum dft_type type) override {
            std::vector<float> twiddles(input_size * output_size * 2);
            parallel_for(output_size, [&] (size_t k) {
                for (size_t n = 0; n < input_size; n++) {
                    double angle = 2 * PI * k * n / input_size;
                    if (!is_inverse)
                        angle = -angle;
                    twiddles[(k * input_size + n) * 2] = std::cos(angle);
                    twiddles[(k * input_size + n) * 2 + 1] = std::sin(angle);
                }
            });
            return twiddles;
        }

        void dft_r2c(float* input_ptr, float* twiddles_ptr, float* output_ptr,
                     size_t input_size, size_t output_size, bool parallelize) {
            auto dft_iteration = [&] (size_t k) {
                 float real = 0, imag = 0;
                for (size_t n = 0; n < input_size; n++) {
                    float cos = twiddles_ptr[2 * (k * input_size + n)];
                    float sin = twiddles_ptr[2 * (k * input_size + n) + 1];
                    real += input_ptr[n] * cos;
                    imag += input_ptr[n] * sin;
                }
                output_ptr[2 * k] = real;
                output_ptr[2 * k + 1] = imag;
            };
            if (parallelize) {
                parallel_for(output_size, dft_iteration);
            } else {
                for (size_t k = 0; k < output_size; k++) {
                    dft_iteration(k);
                }
            }
        }

        void dft_c2c(float* input_ptr, float* twiddles_ptr, float* output_ptr,
                     size_t input_size, size_t signal_size, size_t output_size, bool parallelize) {
            auto dft_iteration = [&] (size_t k) {
                 float real = 0, imag = 0;
                for (size_t n = 0; n < input_size; n++) {
                    float cos = twiddles_ptr[2 * (k * output_size + n)];
                    float sin = twiddles_ptr[2 * (k * output_size + n) + 1];
                    float input_real = input_ptr[2 * n];
                    float input_imag = input_ptr[2 * n + 1];
                    real += input_real * cos - input_imag * sin;
                    imag += input_imag * cos + input_real * sin;
                }
                if (is_inverse) {
                    float* inp = input_ptr + 2 * (input_size - 2 + output_size % 2);
                    for (int n = input_size; n < signal_size; n++, inp -= 2) {
                        float cos = twiddles_ptr[2 * (k * output_size + n)];
                        float sin = twiddles_ptr[2 * (k * output_size + n) + 1];
                        float input_real = inp[0];
                        float input_imag = -inp[1];
                        real += input_real * cos - input_imag * sin;
                        imag += input_imag * cos + input_real * sin;
                    }
                    real /= output_size;
                    imag /= output_size;
                }
                output_ptr[2 * k] = real;
                output_ptr[2 * k + 1] = imag;
            };
            if (parallelize) {
                parallel_for(output_size, dft_iteration);
            } else {
                for (size_t k = 0; k < output_size; k++) {
                    dft_iteration(k);
                }
            }
        }

        void dft_c2r(float* input_ptr, float* twiddles_ptr, float* output_ptr,
                     size_t input_size, size_t signal_size, size_t output_size, bool parallelize) {
            auto dft_iteration = [&] (size_t k) {
                float real = 0;
                for (size_t n = 0; n < input_size; n++) {
                    float cos = twiddles_ptr[2 * (k * output_size + n)];
                    float sin = twiddles_ptr[2 * (k * output_size + n) + 1];
                    float input_real = input_ptr[2 * n];
                    float input_imag = input_ptr[2 * n + 1];
                    real += input_real * cos - input_imag * sin;
                }
                if (is_inverse) {
                    float* inp = input_ptr + 2 * (input_size - 2 + output_size % 2);
                    for (size_t n = input_size; n < signal_size; n++, inp -= 2) {
                        float cos = twiddles_ptr[2 * (k * output_size + n)];
                        float sin = twiddles_ptr[2 * (k * output_size + n) + 1];
                        float input_real = inp[0];
                        float input_imag = inp[1];
                        real += input_real * cos + input_imag * sin;
                    }
                    real /= output_size;
                }
                output_ptr[k] = real;
            };
            if (parallelize) {
                parallel_for(output_size, dft_iteration);
            } else {
                for (int k = 0; k < output_size; k++) {
                    dft_iteration(k);
                }
            }
        }

        void dft(float* input_ptr, float* twiddles_ptr, float* output_ptr,
                 size_t input_size, size_t signal_size, size_t output_size,
                 enum dft_type type, bool parallelize) override {
            if (type == real_to_complex) {
                dft_r2c(input_ptr, twiddles_ptr, output_ptr, input_size, output_size, parallelize);
            } else if (type == complex_to_complex) {
                dft_c2c(input_ptr, twiddles_ptr, output_ptr, input_size, signal_size, output_size, parallelize);
            } else if (type == complex_to_real) {
                dft_c2r(input_ptr, twiddles_ptr, output_ptr, input_size, signal_size, output_size, parallelize);
            }
        }

        void fft(float* input, float* twiddles_ptr, float* output,
                 size_t input_size, size_t signal_size, size_t output_size,
                 enum dft_type type, bool parallelize) override {
            std::vector<float> scratch_space(4 * signal_size, 0);

            float* input_ptr = input;
            float* output_ptr = &scratch_space[2 * signal_size];

            if (input_size < signal_size || type == real_to_complex) {
                if (is_inverse)
                    fft_copy_inverse_input_data(&scratch_space[0], input, input_size, signal_size, parallelize);
                else if (type == real_to_complex)
                    fft_copy_real_input_data(&scratch_space[0], input, input_size, parallelize);
                input_ptr = &scratch_space[0];
            }

            size_t num_blocks = 0;
            size_t block_size = 0;

            auto block_iteration = [&] (size_t block) {
                size_t input_offset = block * block_size;
                size_t output_offset = block * block_size / 2;
                float cos = twiddles_ptr[2 * block];
                float sin = twiddles_ptr[2 * block + 1];
                if (is_inverse)
                    sin = -sin;
                for (size_t pair = 0; pair < block_size / 2; pair++) {
                    float even_real = input_ptr[2 * (input_offset + pair)];
                    float even_imag = input_ptr[2 * (input_offset + pair) + 1];
                    float odd_real = input_ptr[2 * (input_offset + block_size / 2 + pair)];
                    float odd_imag = input_ptr[2 * (input_offset + block_size / 2 + pair) + 1];
                    output_ptr[2 * (output_offset + pair)] = even_real + cos * odd_real - sin * odd_imag;
                    output_ptr[2 * (output_offset + pair) + 1] = even_imag + cos * odd_imag + sin * odd_real;
                    output_ptr[2 * (output_offset + signal_size / 2 + pair)] = even_real - cos * odd_real + sin * odd_imag;
                    output_ptr[2 * (output_offset + signal_size / 2 + pair) + 1] = even_imag - cos * odd_imag - sin * odd_real;
                    if (is_inverse && num_blocks == signal_size / 2) {
                        output_ptr[2 * (output_offset + pair)] /= signal_size;
                        output_ptr[2 * (output_offset + pair) + 1] /= signal_size;
                        output_ptr[2 * (output_offset + signal_size / 2 + pair)] /= signal_size;
                        output_ptr[2 * (output_offset + signal_size / 2 + pair) + 1] /= signal_size;
                    }
                }
            };

            for (num_blocks = 1; num_blocks < signal_size; num_blocks *= 2) {
                block_size = signal_size / num_blocks;
                if (num_blocks == signal_size / 2 && output_size == signal_size && type != complex_to_real) {
                    output_ptr = output;
                }
                if (parallelize) {
                    parallel_for(num_blocks, block_iteration);
                } else {
                    for (size_t block = 0; block < num_blocks; block++) {
                        block_iteration(block);
                    }
                }
                twiddles_ptr += num_blocks * 2;
                if (num_blocks == 1 && input_ptr == input)
                    input_ptr = &scratch_space[0];
                std::swap(input_ptr, output_ptr);
            }

            if (type == complex_to_real) {
                fft_copy_inverse_real_output(output, input_ptr, signal_size, parallelize);
            } else if (output_size != signal_size) {
                cpu_memcpy(output, input_ptr, output_size * complex_type_size<float>());
            }
        }
};

struct RDFTKey {
    bool is_inverse;

    size_t hash() const {
        using namespace dnnl::impl::primitive_hashing;

        size_t seed = 0;
        seed = hash_combine(seed, is_inverse);
        return seed;
    }

    bool operator==(const RDFTKey& rhs) const {
        return is_inverse == rhs.is_inverse;
    }
};

void RDFT::prepareParams() {
    RDFTKey key = { .is_inverse = inverse };

    auto build_executor = [&] (const RDFTKey& key) -> std::shared_ptr<RDFTExecutor> {
        std::shared_ptr<RDFTExecutor> executor;
        if (mayiuse(cpu::x64::sse41)) {
            executor = std::make_shared<RDFTJitExecutor>(key.is_inverse);
        } else {
            executor = std::make_shared<RDFTRefExecutor>(key.is_inverse);
        }
        return executor;
    };

    auto cache = getRuntimeCache();
    auto result = cache->getOrCreate(key, build_executor);
    executor = result.first;
}
}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

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
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;

namespace ov {
namespace intel_cpu {
namespace node {


static const size_t DATA_INDEX = 0;
static const size_t AXES_INDEX = 1;
static const size_t SIGNAL_SIZE_INDEX = 2;


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

static void normalize_axes(std::vector<int>& axes, size_t rank, bool inverse) {
    for (auto& axis : axes) {
        if (axis < 0) {
            axis += rank;
        }
    }
    if (!inverse)
        std::sort(axes.begin(), axes.end(), std::greater<int>());
    else
        std::sort(axes.begin(), axes.end());
}

static std::vector<int> generate_fft_indices(int vlen);

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
    normalize_axes(axes, rank, inverse);

    const auto& input_strides = input_mem.GetDescWithType<BlockedMemoryDesc>()->getStrides();
    const auto& output_strides = output_mem.GetDescWithType<BlockedMemoryDesc>()->getStrides();

    executor->execute(input_ptr, output_ptr, rank, axes,
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


template <typename T>
size_t complex_type_size() {
    return sizeof(T) * 2;
}

void RDFTExecutor::execute(float* input_ptr, float* output_ptr, size_t rank, const std::vector<int>& axes,
                           const VectorDims& input_shape, const VectorDims& output_shape,
                           const VectorDims& input_strides, const VectorDims& output_strides) {
    generate_twiddles(input_shape, output_shape, axes);

    if (rank == 1) {
        size_t num_samples = is_inverse ? output_shape[0] : input_shape[0];
        auto twiddles_ptr = twiddles[0].data();
        dft_common(input_ptr, twiddles_ptr, output_ptr,
                   input_shape[0], num_samples, output_shape[0],
                   is_inverse ? complex_to_real : real_to_complex,
                   can_use_fft(num_samples), true);
    } else {
        if (!is_inverse)
            rdft_nd(input_ptr, output_ptr, axes, input_shape, input_strides, output_shape, output_strides);
        else
            irdft_nd(input_ptr, output_ptr, axes, input_shape, input_strides, output_shape, output_strides);
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
                              size_t input_size, size_t num_samples, size_t output_size,
                              enum dft_type type, bool use_fft, bool parallelize) {
    if (use_fft) {
        fft(input_ptr, twiddles_ptr, output_ptr,
            input_size, num_samples, output_size,
            type, parallelize);
    } else {
        dft(input_ptr, twiddles_ptr, output_ptr,
            input_size, num_samples, output_size,
            type, parallelize);
    }
}

void RDFTExecutor::dft_on_axis(enum dft_type type,
                               float* input_ptr, float* output_ptr,
                               float* twiddles_ptr, int axis,
                               size_t num_samples,
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

    bool use_fft = can_use_fft(num_samples);
    bool parallelize_outer_axes = use_fft;

    size_t total_work_size = std::accumulate(iteration_range.begin(),
                                             iteration_range.end(),
                                             1, std::multiplies<size_t>()) / iteration_range[axis];

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
                       input_size, num_samples, output_size,
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
                       input_size, num_samples, output_size,
                       type, use_fft, !parallelize_outer_axes);
            scatter(output_ptr, scatter_buffer, axis, coords, output_size, output_strides);
        }
    }
}

// N-dimensional real DFT
void RDFTExecutor::rdft_nd(float* input_ptr, float* output_ptr,
                           const std::vector<int>& axes,
                           const VectorDims& input_shape,
                           const VectorDims& input_strides,
                           const VectorDims& output_shape,
                           const VectorDims& output_strides) {
    const std::vector<size_t> iteration_range(output_shape.begin(), output_shape.end() - 1);

    dft_on_axis(real_to_complex, input_ptr, output_ptr,
                twiddles[0].data(), axes[0],
                input_shape[axes[0]],
                input_shape, input_strides,
                output_shape, output_strides,
                iteration_range);
    input_ptr = output_ptr;

    for (size_t i = 1; i < axes.size(); i++) {
        auto axis = axes[i];
        dft_on_axis(complex_to_complex, input_ptr, output_ptr,
                    twiddles[i].data(), axis,
                    input_shape[axis],
                    output_shape, output_strides,
                    output_shape, output_strides,
                    iteration_range);
    }
}

// N-dimensional real inverse DFT
void RDFTExecutor::irdft_nd(float* input_ptr, float* output_ptr,
                            const std::vector<int>& axes,
                            const VectorDims& input_shape,
                            const VectorDims& input_strides,
                            const VectorDims& output_shape,
                            const VectorDims& output_strides) {
    const std::vector<size_t> iteration_range(input_shape.begin(), input_shape.end() - 1);

    if (axes.size() == 1) {
        dft_on_axis(complex_to_real, input_ptr, output_ptr,
                    twiddles[0].data(), axes[0],
                    output_shape[axes[0]],
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

    for (size_t i = 0; i < axes.size() - 1; i++) {
        auto axis = axes[i];
        dft_on_axis(complex_to_complex, input_ptr, output,
                    twiddles[i].data(), axis,
                    output_shape[axis],
                    input_shape, input_strides,
                    input_shape, input_strides,
                    iteration_range);
        input_ptr = output;
    }
    dft_on_axis(complex_to_real, input_ptr, output_ptr,
                twiddles.back().data(), axes.back(),
                output_shape[axes.back()],
                input_shape, input_strides,
                output_shape, output_strides,
                iteration_range);
}

std::vector<float> RDFTExecutor::generate_twiddles_fft(size_t N) {
    std::vector<float> twiddles;
    for (size_t num_blocks = 1; num_blocks < N; num_blocks *= 2) {
        for (size_t block = 0; block < num_blocks; block++) {
            float angle = 2 * M_PI * block / (num_blocks * 2);
            twiddles.push_back(std::cos(angle));
            twiddles.push_back(-std::sin(angle));
        }
    }
    return twiddles;
}

std::vector<float> RDFTExecutor::generate_twiddles_common(size_t input_size, size_t output_size,
                                                          enum dft_type type, bool use_fft) {
    if (use_fft) {
        return generate_twiddles_fft(input_size);
    }
    return generate_twiddles_dft(input_size, output_size, type);
}

void RDFTExecutor::generate_twiddles(const std::vector<size_t>& input_shape,
                                     const std::vector<size_t>& output_shape,
                                     const std::vector<int>& axes) {
    auto it = axes.begin();
    auto axes_end = axes.end();
    auto axis = *it;
    size_t K = output_shape[axis];
    size_t N = input_shape[axis];
    if (is_inverse)
        N = K;
    if (axes.size() == 1) {
        twiddles.push_back(generate_twiddles_common(N, K, is_inverse ? complex_to_real : real_to_complex, can_use_fft(N)));
        return;
    }
    twiddles.push_back(generate_twiddles_common(N, K, is_inverse ? complex_to_complex : real_to_complex, can_use_fft(N)));
    it++;
    for (; it != axes_end - 1; it++) {
        axis = *it;
        K = output_shape[axis];
        twiddles.push_back(generate_twiddles_common(K, K, complex_to_complex, can_use_fft(K)));
    }
    if (it != axes_end) {
        axis = *it;
        K = output_shape[axis];
        N = input_shape[axis];
        if (is_inverse)
            N = K;
        twiddles.push_back(generate_twiddles_common(N, K, is_inverse ? complex_to_real : complex_to_complex, can_use_fft(N)));
    }
}


#define GET_OFF(field) offsetof(jit_dft_args, field)

struct jit_dft_args {
    const void* input;
    const void* twiddles;
    void* output;
    size_t input_size;
    size_t num_samples;
    size_t output_start;
    size_t output_end;
};

struct jit_dft_kernel {
    jit_dft_kernel(bool is_inverse, enum dft_type type) : is_inverse_(is_inverse), kernel_type_(type) {}

    void (*ker_)(const jit_dft_args*);

    void operator()(const jit_dft_args* args) {
        assert(ker_);
        ker_(args);
    }

    jit_dft_kernel() : ker_(nullptr) {}
    virtual ~jit_dft_kernel() {}

    virtual void create_ker() = 0;

    bool is_inverse_;
    enum dft_type kernel_type_;
};


template <cpu_isa_t isa>
struct jit_dft_kernel_f32 : public jit_dft_kernel, public jit_generator {
    public:
        DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_dft_kernel_f32)

        jit_dft_kernel_f32(bool is_inverse, enum dft_type type) : jit_dft_kernel(is_inverse, type), jit_generator() {}

        void create_ker() override {
            jit_generator::create_kernel();
            ker_ = (decltype(ker_))jit_ker();
        }

        void generate() override {
            using namespace Xbyak::util;
            using Xbyak::Label;
            using Xbyak::Xmm;
            using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm,
                                              isa == cpu::x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

            this->preamble();

            const int type_size = sizeof(float);
            int input_type_size = 0;
            int output_type_size = 0;

            switch (kernel_type_) {
            case real_to_complex:
                input_type_size = type_size;
                output_type_size = complex_type_size<float>();
                break;
            case complex_to_complex:
                input_type_size = complex_type_size<float>();
                output_type_size = complex_type_size<float>();
                break;
            case complex_to_real:
                input_type_size = complex_type_size<float>();
                output_type_size = type_size;
                break;
            }
            int vlen = cpu_isa_traits<isa>::vlen;
            const int simd_size = vlen / output_type_size;

            mov(input_ptr, ptr[param + GET_OFF(input)]);
            mov(input_size, ptr[param + GET_OFF(input_size)]);
            mov(twiddles_ptr, ptr[param + GET_OFF(twiddles)]);
            mov(output_start, ptr[param + GET_OFF(output_start)]);
            mov(output_end, ptr[param + GET_OFF(output_end)]);

            // offset twiddles_ptr by input_size * complex_type_size<float>() * output_start bytes
            mov(num_samples, ptr[param + GET_OFF(num_samples)]);
            mov(rax, num_samples);
            lea(rax, ptr[rax * complex_type_size<float>()]);
            xor_(rdx, rdx);
            mul(output_start);
            if (kernel_type_ == complex_to_complex) {
                shl(rax, 1);
            }
            add(twiddles_ptr, rax);

            // offset output_ptr by output_start * output_type_size bytes
            mov(output_ptr, ptr[param + GET_OFF(output)]);
            lea(output_ptr, ptr[output_ptr + output_type_size * output_start]);

            size_t reg_idx = 0;
            Xmm xmm_num_samples = Xmm(reg_idx);
            Vmm vmm_num_samples = Vmm(reg_idx);
            if (is_inverse_) {
                reg_idx++;
                uni_vbroadcastss(Vmm(reg_idx), ptr[param + GET_OFF(num_samples)]);
                uni_vcvtdq2ps(vmm_num_samples, Vmm(reg_idx));
            }


            Vmm vmm_neg_mask = Vmm(reg_idx);
            Xmm xmm_neg_mask = Xmm(reg_idx);
            if (kernel_type_ == complex_to_complex) {
                reg_idx++;
                if (!is_inverse_) {
                    mov(rax, 1ULL << 31);
                } else {
                    mov(rax, 1ULL << 63);
                }
                uni_vmovq(xmm_neg_mask, rax);
                uni_vbroadcastsd(vmm_neg_mask, xmm_neg_mask);
            }

            mov(rax, num_samples);
            and_(rax, 1);
            setz(is_num_samples_even);

            Label loop_over_output;
            Label loop_over_output_continue;
            Label loop_simd;
            Label loop_nonsimd;

            auto simd_loop = [this, vlen, simd_size,
                              input_type_size, &reg_idx,
                              &vmm_num_samples,
                              &xmm_neg_mask,
                              &vmm_neg_mask] {
                Vmm result = Vmm(reg_idx++);
                Vmm inp_real = Vmm(reg_idx++);
                Vmm inp_imag = Vmm(reg_idx++);
                const Vmm& input = inp_real;
                const Vmm& input_perm = inp_imag;
                Vmm twiddles = Vmm(reg_idx++);
                const Vmm& cos = twiddles;
                Vmm sin = Vmm(reg_idx++);

                uni_vpxor(result, result, result);

                if (kernel_type_ == complex_to_complex && is_inverse_) {
                    mov(rdx, 1ULL << 63);
                    uni_vmovq(xmm_neg_mask, rdx);
                    uni_vbroadcastsd(vmm_neg_mask, xmm_neg_mask);
                }

                Label loop;
                L(loop);
                {
                    if (kernel_type_ == real_to_complex) {
                        uni_vbroadcastss(inp_real, ptr[input_ptr]);
                        uni_vmovups(twiddles, ptr[twiddles_ptr]);
                        uni_vfmadd231ps(result, inp_real, twiddles);

                        add(twiddles_ptr, vlen);
                    } else if (kernel_type_ == complex_to_real) {
                        uni_vbroadcastss(inp_real, ptr[input_ptr]);
                        uni_vbroadcastss(inp_imag, ptr[input_ptr + type_size]);
                        uni_vmovups(cos, ptr[twiddles_ptr]);
                        uni_vmovups(sin, ptr[twiddles_ptr + vlen]);
                        uni_vfmadd231ps(result, inp_real, cos);
                        uni_vfmadd231ps(result, inp_imag, sin);

                        add(twiddles_ptr, 2 * vlen);
                    } else if (kernel_type_ == complex_to_complex) {
                        // output_real += input_real * cos(..) - input_imag * sin(..)
                        // output_imag += input_imag * cos(..) + input_real * sin(..)
                        uni_vbroadcastsd(input, ptr[input_ptr]);
                        vpermilps(input_perm, input, 0b10110001); // swap real with imag
                        uni_vpxor(input_perm, input_perm, vmm_neg_mask); // negate imag part (or real part if is_inverse == true)
                        uni_vmovups(cos, ptr[twiddles_ptr]);
                        uni_vmovups(sin, ptr[twiddles_ptr + vlen]);
                        uni_vfmadd231ps(result, input, cos);
                        uni_vfmadd231ps(result, input_perm, sin);

                        add(twiddles_ptr, 2 * vlen);
                    }

                    add(input_ptr, input_type_size);

                    dec(input_size);
                    cmp(input_size, 0);
                    jne(loop, T_NEAR);
                }

                if (is_inverse_) {
                    Label loop_backwards;
                    Label loop_backwards_exit;

                    mov(input_size, num_samples);
                    sub(input_size, ptr[param + GET_OFF(input_size)]);

                    if (kernel_type_ == complex_to_complex) {
                        mov(rdx, 1ULL << 31);
                        vmovq(xmm_neg_mask, rdx);
                        uni_vbroadcastsd(vmm_neg_mask, xmm_neg_mask);
                    }

                    test(is_num_samples_even, 1);
                    jz(loop_backwards);

                    sub(input_ptr, input_type_size);

                    L(loop_backwards);
                    {
                        cmp(input_size, 0);
                        je(loop_backwards_exit, T_NEAR);

                        sub(input_ptr, input_type_size);
                        if (kernel_type_ == complex_to_real) {
                            uni_vbroadcastss(inp_real, ptr[input_ptr]);
                            uni_vbroadcastss(inp_imag, ptr[input_ptr + type_size]);
                            uni_vmovups(cos, ptr[twiddles_ptr]);
                            uni_vmovups(sin, ptr[twiddles_ptr + vlen]);

                            uni_vfmadd231ps(result, inp_real, cos);
                            uni_vfnmadd231ps(result, inp_imag, sin);
                        } else if (kernel_type_ == complex_to_complex) {
                            // output_real += input_real * cos(..) - input_imag * sin(..)
                            // output_imag += input_imag * cos(..) + input_real * sin(..)
                            uni_vbroadcastsd(input, ptr[input_ptr]);
                            vpermilps(input_perm, input, 0b10110001); // swap real with imag
                            uni_vpxor(input_perm, input_perm, vmm_neg_mask); // negate imag part
                            uni_vmovups(cos, ptr[twiddles_ptr]);
                            uni_vmovups(sin, ptr[twiddles_ptr + vlen]);
                            uni_vfmadd231ps(result, input, cos);
                            uni_vfmadd231ps(result, input_perm, sin);
                        }

                        add(twiddles_ptr, 2 * vlen);
                        dec(input_size);
                        jmp(loop_backwards, T_NEAR);
                    }
                    L(loop_backwards_exit);
                }

                if (is_inverse_) {
                    uni_vdivps(result, result, vmm_num_samples);
                }
                // store the results
                uni_vmovups(ptr[output_ptr], result);

                add(output_ptr, vlen);
                sub(output_end, simd_size);
            };

            auto nonsimd_loop = [this, type_size,
                                 input_type_size,
                                 output_type_size,
                                 &xmm_num_samples,
                                 &reg_idx] {
                Xmm xmm_inp_real = Xbyak::Xmm(reg_idx++);
                Xmm xmm_inp_imag = Xbyak::Xmm(reg_idx++);
                Xmm xmm_real = Xbyak::Xmm(reg_idx++);
                Xmm xmm_imag = Xbyak::Xmm(reg_idx++);
                Xmm xmm_cos = Xbyak::Xmm(reg_idx++);
                Xmm xmm_sin = Xbyak::Xmm(reg_idx++);

                if (kernel_type_ != complex_to_real) {
                    xorps(xmm_real, xmm_real);
                    xorps(xmm_imag, xmm_imag);
                } else {
                    xorps(xmm_real, xmm_real);
                }

                Label loop;
                L(loop);
                {
                    movss(xmm_cos, ptr[twiddles_ptr]);
                    movss(xmm_sin, ptr[twiddles_ptr + type_size]);
                    if (kernel_type_ == real_to_complex) {
                        movss(xmm_inp_real, ptr[input_ptr]);

                        // output_real += input_real * cos(..)
                        mulss(xmm_cos, xmm_inp_real);
                        addss(xmm_real, xmm_cos);

                        // output_imag += input_real * sin(..)
                        mulss(xmm_sin, xmm_inp_real);
                        addss(xmm_imag, xmm_sin);
                    } else if (kernel_type_ == complex_to_real) {
                        movss(xmm_inp_real, ptr[input_ptr]);
                        movss(xmm_inp_imag, ptr[input_ptr + type_size]);

                        // output += real * cos(..) + imag * sin(..)
                        mulss(xmm_cos, xmm_inp_real);
                        mulss(xmm_sin, xmm_inp_imag);
                        addss(xmm_cos, xmm_sin);
                        addss(xmm_real, xmm_cos);
                    } else if (kernel_type_ == complex_to_complex) {
                        // output_real += input_real * cos(..) - input_imag * sin(..)
                        movss(xmm_inp_real, ptr[input_ptr]);
                        movss(xmm_inp_imag, ptr[input_ptr + type_size]);
                        mulss(xmm_inp_real, xmm_cos);
                        mulss(xmm_inp_imag, xmm_sin);
                        if (!is_inverse_) {
                            subss(xmm_inp_real, xmm_inp_imag);
                        } else {
                            addss(xmm_inp_real, xmm_inp_imag);
                        }
                        addss(xmm_real, xmm_inp_real);

                        // output_imag += input_imag * cos(..) + input_real * sin(..)
                        movss(xmm_inp_real, ptr[input_ptr]);
                        movss(xmm_inp_imag, ptr[input_ptr + type_size]);
                        mulss(xmm_inp_imag, xmm_cos);
                        mulss(xmm_inp_real, xmm_sin);
                        if (!is_inverse_) {
                            addss(xmm_inp_imag, xmm_inp_real);
                        } else {
                            subss(xmm_inp_imag, xmm_inp_real);
                        }
                        addss(xmm_imag, xmm_inp_imag);
                    }

                    // increment indexes for next iteration
                    add(twiddles_ptr, complex_type_size<float>());
                    add(input_ptr, input_type_size);
                    dec(input_size);

                    // continue if input_size > 0
                    cmp(input_size, 0);
                    jg(loop, T_NEAR);
                }
                if (is_inverse_) {
                    Label loop_backwards;
                    Label loop_backwards_exit;

                    mov(input_size, num_samples);
                    sub(input_size, ptr[param + GET_OFF(input_size)]);

                    test(is_num_samples_even, 1);
                    jz(loop_backwards);

                    sub(input_ptr, input_type_size);

                    L(loop_backwards);
                    {
                        cmp(input_size, 0);
                        je(loop_backwards_exit);

                        sub(input_ptr, input_type_size);

                        movss(xmm_cos, ptr[twiddles_ptr]);
                        movss(xmm_sin, ptr[twiddles_ptr + type_size]);
                        movss(xmm_inp_real, ptr[input_ptr]);
                        movss(xmm_inp_imag, ptr[input_ptr + type_size]);

                        if (kernel_type_ == complex_to_real) {
                            // output += real * cos(..) - imag * sin(..)
                            mulss(xmm_cos, xmm_inp_real);
                            mulss(xmm_sin, xmm_inp_imag);
                            subss(xmm_cos, xmm_sin);
                            addss(xmm_real, xmm_cos);
                        } else if (kernel_type_ == complex_to_complex) {
                            // output_real += input_real * cos(..) - input_imag * sin(..)
                            movss(xmm_inp_real, ptr[input_ptr]);
                            movss(xmm_inp_imag, ptr[input_ptr + type_size]);
                            mulss(xmm_inp_real, xmm_cos);
                            mulss(xmm_inp_imag, xmm_sin);
                            subss(xmm_inp_real, xmm_inp_imag);
                            addss(xmm_real, xmm_inp_real);

                            // output_imag += input_imag * cos(..) + input_real * sin(..)
                            movss(xmm_inp_real, ptr[input_ptr]);
                            movss(xmm_inp_imag, ptr[input_ptr + type_size]);
                            mulss(xmm_inp_imag, xmm_cos);
                            mulss(xmm_inp_real, xmm_sin);
                            addss(xmm_inp_imag, xmm_inp_real);
                            addss(xmm_imag, xmm_inp_imag);
                        }

                        add(twiddles_ptr, complex_type_size<float>());
                        dec(input_size);
                        jmp(loop_backwards);
                    }
                    L(loop_backwards_exit);
                }

                if (kernel_type_ == complex_to_real) {
                    if (is_inverse_) {
                        divss(xmm_real, xmm_num_samples);
                    }
                    // store the result
                    movss(ptr[output_ptr], xmm_real);
                } else {
                    if (is_inverse_) {
                        divss(xmm_real, xmm_num_samples);
                        divss(xmm_imag, xmm_num_samples);
                    }
                    // store the results
                    movss(ptr[output_ptr], xmm_real);
                    movss(ptr[output_ptr + type_size], xmm_imag);
                }

                add(output_ptr, output_type_size);
                dec(output_end);
            };

            L(loop_over_output);
            {
                mov(input_ptr, ptr[param + GET_OFF(input)]);
                mov(input_size, ptr[param + GET_OFF(input_size)]);

                cmp(output_end, simd_size);
                jae(loop_simd, T_NEAR);

                jmp(loop_nonsimd, T_NEAR);

                L(loop_simd);
                    simd_loop();
                    jmp(loop_over_output_continue, T_NEAR);

                L(loop_nonsimd);
                    nonsimd_loop();

                L(loop_over_output_continue);
                    cmp(output_end, 0);
                    ja(loop_over_output, T_NEAR);
            }

            this->postamble();
        }

    private:
        void uni_vbroadcastsd(const Xbyak::Xmm& x, const Xbyak::Operand& op) {
            movsd(x, op);
            shufps(x, x, 0x0);
        }

        void uni_vbroadcastsd(const Xbyak::Ymm& x, const Xbyak::Operand& op) {
            vbroadcastsd(x, op);
        }

        Xbyak::Reg64 param = abi_param1;
        Xbyak::Reg8 is_num_samples_even = al;
        Xbyak::Reg64 input_ptr = rbx;
        Xbyak::Reg64 input_size = rcx;
        Xbyak::Reg64 output_ptr = r8;
        Xbyak::Reg64 twiddles_ptr = r9;
        Xbyak::Reg64 num_samples = r10;
        Xbyak::Reg64 output_start = r11;
        Xbyak::Reg64 output_end = r12;
};

struct jit_fft_args {
    const float* input;
    const float* twiddles;
    float* output;
    int* indices;
    size_t num_samples;
    size_t block;
    size_t block_size;
    size_t subblock_start;
    size_t subblock_end;
};

#define GET_OFF_FFT(field) offsetof(jit_fft_args, field)

struct jit_fft_kernel {
    jit_fft_kernel(bool is_inverse) : is_inverse_(is_inverse) {}

    void (*ker_)(const jit_fft_args*);

    void operator()(const jit_fft_args* args) {
        assert(ker_);
        ker_(args);
    }

    jit_fft_kernel() : ker_(nullptr) {}
    virtual ~jit_fft_kernel() {}

    virtual void create_ker() = 0;

    bool is_inverse_;
};

template <cpu_isa_t isa>
struct jit_fft_kernel_f32 : public jit_fft_kernel, public jit_generator {
    public:
        DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_fft_kernel_f32)

        jit_fft_kernel_f32(bool is_inverse) : jit_fft_kernel(is_inverse), jit_generator() {}

        void create_ker() override {
            jit_generator::create_kernel();
            ker_ = (decltype(ker_))jit_ker();
        }

        void generate() override {
            using namespace Xbyak::util;
            using Xbyak::Label;
            using Xbyak::Address;
            using Xbyak::Xmm;
            using Xbyak::Ymm;
            using Xbyak::RegExp;
            using Vmm = typename conditional3<isa == cpu_isa_t::sse41, Xbyak::Xmm,
                                              isa == cpu_isa_t::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
            const int type_size = sizeof(float);
            const int vlen = cpu_isa_traits<isa>::vlen;
            const int simd_size = vlen / complex_type_size<float>();

            int reg_idx = 0;
            Vmm result_bottom = Vmm(reg_idx++);
            Vmm result_top = Vmm(reg_idx++);
            Vmm even = Vmm(reg_idx++);
            Vmm odd = Vmm(reg_idx++);
            Vmm odd_perm = Vmm(reg_idx++);
            Vmm cos = Vmm(reg_idx++);
            Vmm sin = Vmm(reg_idx++);
            Vmm neg_mask = Vmm(reg_idx++);
            Xmm xmm_neg_mask = Xmm(neg_mask.getIdx());
            Xmm tmp = Xmm(reg_idx++);
            Vmm even_indices = Vmm(reg_idx++);
            Vmm odd_indices = Vmm(reg_idx++);
            Vmm twiddles_indices = Vmm(reg_idx++);
            Vmm gather_mask = Vmm(reg_idx++);
            Vmm zero = Vmm(reg_idx++);
            Vmm vmm_num_samples = Vmm(reg_idx++);

            auto complex_to_complex_multiply = [this, &odd_perm, &neg_mask]
                                                 (const Vmm& even, const Vmm& odd,
                                                  const Vmm& cos, const Vmm& sin,
                                                  const Vmm& result_bottom,
                                                  const Vmm& result_top) {
                vpermilps(odd_perm, odd, 0b10110001); // swap real with imag
                uni_vxorps(odd_perm, odd_perm, neg_mask);

                uni_vmovups(result_bottom, even);
                uni_vfmadd231ps(result_bottom, cos, odd);
                uni_vfmadd231ps(result_bottom, sin, odd_perm);

                uni_vmovups(result_top, even);
                uni_vfnmadd231ps(result_top, cos, odd);
                uni_vfnmadd231ps(result_top, sin, odd_perm);
            };

            Label handle_small_block_size;
            Label loop_subblocks;
            Label loop_subblocks_continue;
            Label exit;

            this->preamble();

            if (is_inverse_) {
                mov(rax, 1ULL << 63);
                uni_vbroadcastss(Vmm(0), ptr[param1 + GET_OFF(num_samples)]);
                uni_vcvtdq2ps(vmm_num_samples, Vmm(0));
            } else {
                mov(rax, 1ULL << 31);
            }
            uni_vmovq(xmm_neg_mask, rax);
            vbroadcastsd(neg_mask, xmm_neg_mask);

            mov(block_size, ptr[param1 + GET_OFF_FFT(block_size)]);
            cmp(block_size, simd_size);
            jbe(handle_small_block_size, T_NEAR);

            mov(input_base_ptr, ptr[param1 + GET_OFF_FFT(input)]);
            mov(output_ptr, ptr[param1 + GET_OFF_FFT(output)]);
            mov(twiddles_ptr, ptr[param1 + GET_OFF_FFT(twiddles)]);
            mov(num_samples, ptr[param1 + GET_OFF_FFT(num_samples)]);
            mov(block, ptr[param1 + GET_OFF_FFT(block)]);
            mov(subblock_start, ptr[param1 + GET_OFF_FFT(subblock_start)]);
            mov(block_size_mask, block_size);
            dec(block_size_mask);

            uni_vpxor(zero, zero, zero);

            mov(input_ptr, input_base_ptr);

            L(loop_subblocks);
            {
                uni_vmovups(even, ptr[input_ptr]);
                uni_vmovups(odd, ptr[input_ptr + (complex_type_size<float>() / 2) * block_size]);
                add(input_ptr, vlen);

                uni_vbroadcastss(cos, ptr[twiddles_ptr + block * complex_type_size<float>()]);
                uni_vbroadcastss(sin, ptr[twiddles_ptr + block * complex_type_size<float>() + type_size]);

                complex_to_complex_multiply(even, odd, cos, sin, result_bottom, result_top);

                uni_vmovups(ptr[output_ptr], result_bottom);
                uni_vmovups(ptr[output_ptr + num_samples * (complex_type_size<float>() / 2)], result_top);

                add(output_ptr, vlen);
                add(subblock_start, simd_size);

                mov(rax, subblock_start);
                and_(rax, block_size_mask);
                jnz(loop_subblocks_continue);

                inc(block);
                lea(input_base_ptr, ptr[input_base_ptr + block_size * complex_type_size<float>()]);
                mov(input_ptr, input_base_ptr);

                L(loop_subblocks_continue);
                cmp(subblock_start, ptr[param1 + GET_OFF_FFT(subblock_end)]);
                jne(loop_subblocks);
            }

            jmp(exit, T_NEAR);

            auto gather = [this, &gather_mask] (const Vmm& dst, const Xbyak::Reg64& base, const Vmm& indices, int offset = 0) {
                if (isa == cpu_isa_t::avx2) {
                    vpcmpeqd(gather_mask, gather_mask, gather_mask);
                    vgatherdps(dst, ptr[base + indices * type_size + offset], gather_mask);
                } else if (isa == cpu_isa_t::avx512_core) {
                    mov(eax, 0xffff);
                    kmovw(k1, eax);
                    vgatherdps(dst | k1, ptr[base + indices * type_size + offset]);
                }
            };

            L(handle_small_block_size);

            mov(input_ptr, ptr[param1 + GET_OFF_FFT(input)]);
            mov(output_ptr, ptr[param1 + GET_OFF_FFT(output)]);
            mov(indices_ptr, ptr[param1 + GET_OFF_FFT(indices)]);
            mov(twiddles_ptr, ptr[param1 + GET_OFF_FFT(twiddles)]);
            mov(num_samples, ptr[param1 + GET_OFF_FFT(num_samples)]);
            mov(block, ptr[param1 + GET_OFF_FFT(block)]);
            mov(subblock_start, ptr[param1 + GET_OFF_FFT(subblock_start)]);

            xor_(rdx, rdx);
            mov(rax,  simd_size);
            div(block_size);
            shl(rax, 1);
            lea(twiddles_offset, ptr[rax * complex_type_size<float>()]);

            uni_vmovdqu(even_indices, ptr[indices_ptr]);
            add(indices_ptr, vlen);
            uni_vmovdqu(odd_indices, ptr[indices_ptr]);
            add(indices_ptr, vlen);
            uni_vmovdqu(twiddles_indices, ptr[indices_ptr]);
            add(indices_ptr, vlen);

            Label loop_small_block;
            Label loop_small_subblock;
            Label store;
            L(loop_small_block);
            {
                gather(even, input_ptr, even_indices);
                gather(odd, input_ptr, odd_indices);
                add(input_ptr, vlen * 2);
                gather(cos, twiddles_ptr, twiddles_indices);
                gather(sin, twiddles_ptr, twiddles_indices, type_size);
                add(twiddles_ptr, twiddles_offset);

                complex_to_complex_multiply(even, odd, cos, sin, result_bottom, result_top);

                if (is_inverse_) {
                    cmp(block_size, 2);
                    ja(store);
                    vdivps(result_bottom, vmm_num_samples);
                    vdivps(result_top, vmm_num_samples);
                }

                L(store);
                uni_vmovups(ptr[output_ptr], result_bottom);
                uni_vmovups(ptr[output_ptr + num_samples * (complex_type_size<float>() / 2)], result_top);

                add(output_ptr, vlen);

                add(subblock_start, simd_size);
                cmp(subblock_start, ptr[param1 + GET_OFF_FFT(subblock_end)]);
                ja(loop_small_block);
            }

            L(exit);

            this->postamble();
        }

    private:
        Xbyak::Reg64 input_ptr = rbx;
        Xbyak::Reg64 input_base_ptr = rcx;
        Xbyak::Reg64 output_ptr = rsi;
        Xbyak::Reg64 block_size_mask = rdx;
        Xbyak::Reg64 twiddles_ptr = r8;
        Xbyak::Reg64 indices_ptr = r9;
        Xbyak::Reg64 block_size = r10;
        Xbyak::Reg64 num_samples = r11;
        Xbyak::Reg64 block = r12;
        Xbyak::Reg64 subblock_start = r13;
        Xbyak::Reg64 twiddles_offset = r14;
};

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

static void fft_copy_inverse_input_data(float* dst, float* src, size_t input_size, size_t num_samples, bool parallelize) {
    if (!parallelize) {
        cpu_memcpy(dst, src, input_size * complex_type_size<float>());
        src = src + 2 * input_size - 4;
        for (size_t i = input_size; i < num_samples; i++, src -= 2) {
            dst[2 * i] = src[0];
            dst[2 * i + 1] = -src[1];
        }
    } else {
        parallel_for(num_samples, [&] (size_t i) {
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

static void fft_copy_inverse_real_output(float* dst, float* src, size_t num_samples, bool parallelize) {
    if (!parallelize) {
        for (size_t i = 0; i < num_samples; i++) {
            dst[i] = src[2 * i];
        }
    } else {
        parallel_for(num_samples, [&] (size_t i) {
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
        std::vector<float> twiddles;
        twiddles.reserve(input_size * output_size * 2);
        int simd_size = vlen / sizeof(float);
        if (type == real_to_complex || type == complex_to_complex) {
            simd_size /= 2; // there are two floats per one complex element in the output
        }

        for (size_t K = 0; K < output_size / simd_size; K++) {
            for (size_t n = 0; n < input_size; n++) {
                if (type == real_to_complex) {
                    for (size_t k = 0; k < simd_size; k++) {
                        twiddles.push_back(std::cos(2 * M_PI * (K * simd_size + k) * n / input_size));
                        twiddles.push_back(-std::sin(2 * M_PI * (K * simd_size + k) * n / input_size));
                    }
                } else if (type == complex_to_real) {
                    for (size_t k = 0; k < simd_size; k++) {
                        twiddles.push_back(std::cos(2 * M_PI * (K * simd_size + k) * n / input_size));
                    }
                    for (size_t k = 0; k < simd_size; k++) {
                        twiddles.push_back(-std::sin(2 * M_PI * (K * simd_size + k) * n / input_size));
                    }
                } else if (type == complex_to_complex) {
                    for (size_t k = 0; k < simd_size; k++) {
                        twiddles.push_back(std::cos(2 * M_PI * (K * simd_size + k) * n / input_size));
                        twiddles.push_back(std::cos(2 * M_PI * (K * simd_size + k) * n / input_size));
                    }
                    for (size_t k = 0; k < simd_size; k++) {
                        twiddles.push_back(-std::sin(2 * M_PI * (K * simd_size + k) * n / input_size));
                        twiddles.push_back(-std::sin(2 * M_PI * (K * simd_size + k) * n / input_size));
                    }
                }
            }
        }
        if ((output_size % simd_size) != 0) {
            size_t start = (output_size / simd_size) * simd_size;
            for (size_t k = start; k < output_size; k++) {
                for (size_t n = 0; n < input_size; n++) {
                    twiddles.push_back(std::cos(2 * M_PI * k * n / input_size));
                    twiddles.push_back(-std::sin(2 * M_PI * k * n / input_size));
                }
            }
        }
        return twiddles;
    }

    void dft(float* input_ptr, float* twiddles_ptr, float* output_ptr,
             size_t input_size, size_t num_samples, size_t output_size,
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
                jit_dft_args args {
                    .input = input_ptr,
                    .twiddles = twiddles_ptr,
                    .output = output_ptr,
                    .input_size = input_size,
                    .num_samples = num_samples,
                    .output_start = i * block_size,
                    .output_end = std::min(output_size - i * block_size, block_size),
                };
                (*kernel)(&args);
            });
        } else {
             jit_dft_args args {
                .input = input_ptr,
                .twiddles = twiddles_ptr,
                .output = output_ptr,
                .input_size = input_size,
                .num_samples = num_samples,
                .output_start = 0,
                .output_end = output_size,
            };
            (*kernel)(&args);
        }
    }

    void fft(float* input, float* twiddles_ptr, float* output,
             size_t input_size, size_t num_samples, size_t output_size,
             enum dft_type type, bool parallelize) override {
        std::vector<float> scratch_space(4 * num_samples, 0);

        float* input_ptr = input;
        float* output_ptr = &scratch_space[2 * num_samples];
        int* indices_ptr = &fft_indices[0];
        size_t simd_size = dft_simd_size(vlen);
        size_t input_stride = simd_size;

        if (input_size < num_samples || type == real_to_complex) {
            if (is_inverse)
                fft_copy_inverse_input_data(&scratch_space[0], input, input_size, num_samples, parallelize);
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
            struct jit_fft_args params{
                .input = input_ptr + input_offset,
                .twiddles = twiddles_ptr,
                .output = output_ptr + output_offset,
                .num_samples = num_samples,
                .block = block,
                .block_size = block_size,
                .subblock_start = i * work_divide_factor / 2,
                .subblock_end = (i + 1) * work_divide_factor / 2,
            };
            (*fft_kernel)(&params);
        };

        for (size_t num_blocks = 1; num_blocks < num_samples / simd_size; num_blocks *= 2) {
            block_size = num_samples / num_blocks;
            if (parallelize) {
                parallel_nt(num_samples / work_divide_factor, block_iteration);
            } else {
                for (size_t i = 0; i < num_samples / work_divide_factor; i++) {
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
            struct jit_fft_args params{
                .input = input_ptr + input_offset * 2,
                .twiddles = twiddles_ptr + i * (simd_size / block_size) * 2 * 2,
                .output = output_ptr + output_offset,
                .indices = indices_ptr,
                .num_samples = num_samples,
                .block_size = block_size,
                .subblock_start = i * work_divide_factor / 2,
                .subblock_end = (i + 1) * work_divide_factor / 2,
            };
            (*fft_kernel)(&params);
        };

        for (size_t num_blocks = num_samples / simd_size; num_blocks < num_samples; num_blocks *= 2) {
            block_size = num_samples / num_blocks;
            if (num_blocks == num_samples / 2 && output_size == num_samples && type != complex_to_real) {
                output_ptr = output;
            }
            if (parallelize) {
                parallel_nt(num_samples / work_divide_factor, small_block_iteration);
            } else {
                for (size_t i = 0; i < num_samples / work_divide_factor; i++) {
                    small_block_iteration(i, 0);
                }
            }
            indices_ptr += 3 * 2 * simd_size;
            twiddles_ptr += num_blocks * 2;
            std::swap(input_ptr, output_ptr);
        }

        if (type == complex_to_real) {
            fft_copy_inverse_real_output(output, input_ptr, num_samples, parallelize);
        } else if (output_size != num_samples) {
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
            std::vector<float> twiddles;
            twiddles.reserve(input_size * output_size * 2);
            for (size_t k = 0; k < output_size; k++) {
                for (size_t n = 0; n < input_size; n++) {
                    float angle = 2 * M_PI * k * n / input_size;
                    if (!is_inverse)
                        angle = -angle;
                    twiddles.push_back(std::cos(angle));
                    twiddles.push_back(std::sin(angle));
                }
            }
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
                     size_t input_size, size_t output_size, bool parallelize) {
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
                    for (int n = input_size; n < output_size; n++, inp -= 2) {
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
                     size_t input_size, size_t output_size, bool parallelize) {
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
                    for (size_t n = input_size; n < output_size; n++, inp -= 2) {
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
                 size_t input_size, size_t num_samples, size_t output_size,
                 enum dft_type type, bool parallelize) override {
            if (type == real_to_complex) {
                dft_r2c(input_ptr, twiddles_ptr, output_ptr, input_size, output_size, parallelize);
            } else if (type == complex_to_complex) {
                dft_c2c(input_ptr, twiddles_ptr, output_ptr, input_size, output_size, parallelize);
            } else if (type == complex_to_real) {
                dft_c2r(input_ptr, twiddles_ptr, output_ptr, input_size, output_size, parallelize);
            }
        }

        void fft(float* input, float* twiddles_ptr, float* output,
                 size_t input_size, size_t num_samples, size_t output_size,
                 enum dft_type type, bool parallelize) override {
            std::vector<float> scratch_space(4 * num_samples, 0);

            float* input_ptr = input;
            float* output_ptr = &scratch_space[2 * num_samples];

            if (input_size < num_samples || type == real_to_complex) {
                if (is_inverse)
                    fft_copy_inverse_input_data(&scratch_space[0], input, input_size, num_samples, parallelize);
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
                    output_ptr[2 * (output_offset + num_samples / 2 + pair)] = even_real - cos * odd_real + sin * odd_imag;
                    output_ptr[2 * (output_offset + num_samples / 2 + pair) + 1] = even_imag - cos * odd_imag - sin * odd_real;
                    if (is_inverse && num_blocks == num_samples / 2) {
                        output_ptr[2 * (output_offset + pair)] /= num_samples;
                        output_ptr[2 * (output_offset + pair) + 1] /= num_samples;
                        output_ptr[2 * (output_offset + num_samples / 2 + pair)] /= num_samples;
                        output_ptr[2 * (output_offset + num_samples / 2 + pair) + 1] /= num_samples;
                    }
                }
            };

            for (num_blocks = 1; num_blocks < num_samples; num_blocks *= 2) {
                block_size = num_samples / num_blocks;
                if (num_blocks == num_samples / 2 && output_size == num_samples && type != complex_to_real) {
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
                fft_copy_inverse_real_output(output, input_ptr, num_samples, parallelize);
            } else if (output_size != num_samples) {
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

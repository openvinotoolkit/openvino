// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dynamic_quantize.h"

#include "cpu/x64/jit_generator.hpp"
#include "openvino/core/parallel.hpp"

#include "ov_ops/dynamic_quantize.hpp"

using namespace ov::element;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::cpu;
using namespace dnnl::impl::cpu::x64;

namespace ov {
namespace intel_cpu {
namespace node {

#define GET_OFF(field) offsetof(dynamic_quantization_runtime_params_t, field)

template <cpu_isa_t isa>
struct jit_dynamic_quantization_kernel_t : public dynamic_quantization_kernel_t, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_dynamic_quantization_kernel_t)

    jit_dynamic_quantization_kernel_t(const dynamic_quantization_compile_params_t& jcp)
        : dynamic_quantization_kernel_t(jcp), jit_generator(jit_name()) {
        vec_size = cpu_isa_traits<isa>::vlen / sizeof(float);

        create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

private:
    using Vmm = typename conditional3<isa == x64::sse41, Xbyak::Xmm, isa == x64::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;

    static constexpr int n_vregs = cpu_isa_traits<isa>::n_vregs;

    void generate() override {
        preamble();

        mov(reg_src, ptr[param1 + GET_OFF(src_ptr)]);
        mov(reg_qsrc, ptr[param1 + GET_OFF(qsrc_ptr)]);
        mov(reg_src_scales, ptr[param1 + GET_OFF(src_scales_ptr)]);
        mov(reg_ic_size, ptr[param1 + GET_OFF(ic_size)]);

        Xbyak::Label ic_loop_label;
        Xbyak::Label ic_end_label;

        size_t src_dt_size = jcp_.src_dt.size();
        size_t qsrc_dt_size = jcp_.qsrc_dt.size();
        size_t src_scales_dt_size = sizeof(float);

        static const float negative_zero[16] = {
            -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f,
            -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f
        };

        static const float positive_one[16] = {
            1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f,
            1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f
        };

        static const float int8_max[16] = {
            127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f,
            127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f
        };

        mov(reg_tmp, (size_t)negative_zero);
        uni_vmovups(vmm_sign_bit_mask(), ptr[reg_tmp]);

        mov(reg_tmp, (size_t)positive_one);
        uni_vmovups(vmm_one(), ptr[reg_tmp]);

        mov(reg_tmp, (size_t)int8_max);
        uni_vmovups(vmm_int8_max(), ptr[reg_tmp]);

        L(ic_loop_label);
        {
            cmp(reg_ic_size, jcp_.ic_quant_block);
            jl(ic_end_label, T_NEAR);

            assert(!(jcp_.ic_quant_block % vec_size));

            int ic_blocks = jcp_.ic_quant_block / vec_size;
            uni_vpxor(vmm_max(), vmm_max(), vmm_max());
            for (int icb = 0; icb < ic_blocks; icb++) {
                load_src(vmm_src(), ptr[reg_src + icb * vec_size * src_dt_size]);
                vandnps(vmm_src(), vmm_sign_bit_mask(), vmm_src());
                uni_vmaxps(vmm_max(), vmm_max(), vmm_src());
            }

            if (isa == x64::avx512_core) {
                Xbyak::Zmm max_zmm = Xbyak::Zmm(vmm_max().getIdx());
                Xbyak::Zmm aux_zmm = Xbyak::Zmm(vmm_aux().getIdx());
                vshuff32x4(aux_zmm, max_zmm, max_zmm, 0x4E);
                uni_vmaxps(max_zmm, max_zmm, aux_zmm);
                vshuff32x4(aux_zmm, max_zmm, max_zmm, 0xB1);
                uni_vmaxps(max_zmm, max_zmm, aux_zmm);
            } else if (isa == x64::avx2) {
                Xbyak::Ymm max_ymm = Xbyak::Ymm(vmm_max().getIdx());
                Xbyak::Ymm aux_ymm = Xbyak::Ymm(vmm_aux().getIdx());
                vperm2i128(aux_ymm, max_ymm, max_ymm, 0x01);
                uni_vmaxps(max_ymm, max_ymm, aux_ymm);
            } else {
                assert(!"unsupported isa");
            }
            uni_vshufps(vmm_aux(), vmm_max(), vmm_max(), 0x4E);
            uni_vmaxps(vmm_max(), vmm_max(), vmm_aux());
            uni_vshufps(vmm_aux(), vmm_max(), vmm_max(), 0xB1);
            uni_vmaxps(vmm_max(), vmm_max(), vmm_aux());

            auto vmm_dscale = vmm_max();
            uni_vbroadcastss(vmm_dscale, Xbyak::Xmm(vmm_dscale.getIdx()));
            uni_vdivps(vmm_dscale, vmm_dscale, vmm_int8_max());

            // todo: check zero case ( (dscale != 0) ? (1.0f / dscale) : 0;)
            uni_vdivps(vmm_qscale(), vmm_one(), vmm_dscale);

            uni_vmovss(ptr[reg_src_scales], Xbyak::Xmm(vmm_dscale.getIdx()));
            for (int icb = 0; icb < ic_blocks; icb++) {
                load_src(vmm_src(), ptr[reg_src + icb * vec_size * src_dt_size]);
                uni_vmulps(vmm_src(), vmm_src(), vmm_qscale());
                uni_vcvtps2dq(vmm_src(), vmm_src());

                if (isa == avx512_core) {
                    vpmovsdb(ptr[reg_qsrc + icb * vec_size * qsrc_dt_size], vmm_src());
                } else {
                    uni_vpackssdw(vmm_src(), vmm_src(), vmm_src());
                    vpermq(Xbyak::Ymm(vmm_src().getIdx()), Xbyak::Ymm(vmm_src().getIdx()), 0x08);
                    uni_vpacksswb(vmm_src(), vmm_src(), vmm_src());
                    vmovq(ptr[reg_qsrc + icb * vec_size * qsrc_dt_size], Xbyak::Xmm(vmm_src().getIdx()));
                }
            }

            sub(reg_ic_size, jcp_.ic_quant_block);
            add(reg_src, src_dt_size * jcp_.ic_quant_block);
            add(reg_qsrc, qsrc_dt_size * jcp_.ic_quant_block);
            add(reg_src_scales, src_scales_dt_size);

            jmp(ic_loop_label, T_NEAR);
        }
        L(ic_end_label);

        postamble();
    }

    void load_src(Vmm vmm_load, const Xbyak::Address& addr) {
        switch (jcp_.src_dt) {
            case ov::element::f32: {
                uni_vmovups(vmm_load, addr);
                break;
            }
            default: assert(!"unsupported data type");
        }
    }

    Vmm vmm_src() {
        return Vmm(0);
    }

    Vmm vmm_max() {
        return Vmm(1);
    }

    Vmm vmm_sign_bit_mask() {
        return Vmm(2);
    }

    Vmm vmm_aux() {
        return Vmm(3);
    }

    Vmm vmm_int8_max() {
        return Vmm(4);
    }

    Vmm vmm_qscale() {
        return Vmm(5);
    }

    Vmm vmm_one() {
        return Vmm(6);
    }

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_qsrc = r9;
    Xbyak::Reg64 reg_src_scales = r10;
    Xbyak::Reg64 reg_ic_size = r11;
    Xbyak::Reg64 reg_tmp = r12;

    size_t vec_size;
};

bool DynamicQuantize::isSupportedOperation(const std::shared_ptr<const ov::Node>& op,
                                          std::string& errorMessage) noexcept {
    try {
        const auto dq = std::dynamic_pointer_cast<const op::internal::DynamicQuantize>(op);
        if (!dq) {
            errorMessage = "Only DynamicQuantize operation from internal opset is supported";
            return false;
        }

        // Supported quantization precisions
        if (!one_of(dq->get_output_element_type(DATA_ID), ov::element::i8)) {
            return false;
        }

        if (dq->get_output_size() == 3 &&
            dq->get_output_element_type(DATA_ID) != dq->get_output_element_type(ZERO_POINTS_ID)) {
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

DynamicQuantize::DynamicQuantize(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    const auto dq = std::dynamic_pointer_cast<const op::internal::DynamicQuantize>(op);
    auto groupSizes = dq->get_group_sizes();
    groupSize = groupSizes.back();
}

void DynamicQuantize::prepareParams() {
}

void DynamicQuantize::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

bool DynamicQuantize::created() const {
    return getType() == Type::DynamicQuantize;
}

void DynamicQuantize::initSupportedPrimitiveDescriptors() {
    inputPrecision = getOriginalInputPrecisionAtPort(DATA_ID);
    if (!one_of(inputPrecision, f32/*, bf16, f16*/))
        inputPrecision = f32;

    auto quantizedDataPrecision = getOriginalOutputPrecisionAtPort(DATA_ID);

    std::vector<PortConfigurator> inPortConfigs;
    inPortConfigs.push_back({LayoutType::ncsp, inputPrecision});
    std::vector<PortConfigurator> outPortConfigs;
    outPortConfigs.push_back({LayoutType::ncsp, quantizedDataPrecision}); // Quantized Data
    outPortConfigs.push_back({LayoutType::ncsp, inputPrecision}); // Scales
    if (getOriginalOutputsNumber() == 3) {
        outPortConfigs.push_back({LayoutType::ncsp, quantizedDataPrecision}); // Zero points
    }

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void DynamicQuantize::createPrimitive() {
}

template <typename SRC_T, typename QDST_T>
static void quantize_symmetric(const SRC_T* psrc, QDST_T* pdst, SRC_T* pscales, const VectorDims& srcDims, size_t groupSize) {
    auto MB = srcDims[0];
    auto IC = srcDims[1];

    auto groups = div_up(IC, groupSize);
    auto vec_loop_end = (groups - 1) * groupSize;

    for (size_t mb = 0; mb < MB; mb++) {
        float amax = -FLT_MAX;
        float amin = FLT_MAX;
        for (size_t ic = 0; ic < IC; ic++) {
            amax = std::max(amax, psrc[mb * IC + ic]);
            amin = std::min(amin, psrc[mb * IC + ic]);
        }

        float max_abs = std::max(std::abs(amin), std::abs(amax));

        const float qscale = 255 / max_abs;
        const float dscale = 1.f / qscale;

        pscales[mb] = dscale;
        for (size_t ic = 0; ic < IC; ic++) {
            pdst[mb * IC + ic] = std::max(std::min(std::round(psrc[mb * IC + ic] * qscale), 127.f), -128.f);
        }

        // for (size_t ic = 0; ic < IC; ic++) {
        //     float dequant = (pdst[mb * IC + ic]) * pscales[mb];
        //     if (std::abs(psrc[mb * IC + ic] - dequant) > 5.f) {
        //         std::cerr << "amin=" << amin << " amax=" << amax << std::endl;
        //         std::cerr << "dscale=" << dscale << " zp=" << zp << std::endl;
        //         std::cerr << "psrc=" << psrc[mb * IC + ic] << " quant=" << static_cast<float>(pdst[mb * IC + ic]) << " dequant=" << dequant << std::endl;
        //     }
        // }
    }
}

template <typename SRC_T, typename QDST_T>
static void quantize_asymmetric(const SRC_T* psrc, QDST_T* pdst, SRC_T* pscales, QDST_T* pzp, const VectorDims& srcDims, size_t groupSize) {
    auto MB = srcDims[0];
    auto IC = srcDims[1];

    auto groups = div_up(IC, groupSize);
    auto vec_loop_end = (groups - 1) * groupSize;

    for (size_t mb = 0; mb < MB; mb++) {
        float amax = -FLT_MAX;
        float amin = FLT_MAX;
        for (size_t ic = 0; ic < IC; ic++) {
            amax = std::max(amax, psrc[mb * IC + ic]);
            amin = std::min(amin, psrc[mb * IC + ic]);
        }

        const float qscale = 255 / (amax - amin);
        const float dscale = 1.f / qscale;

        // const float qscale  = (dscale != 0) ? (1.0f / dscale) : 0;

        float zp = -std::round(qscale * amin) - 128;
        zp = std::max(std::min(zp, 127.f), -128.f);

        pscales[mb] = dscale;
        pzp[mb] = zp;
        for (size_t ic = 0; ic < IC; ic++) {
            pdst[mb * IC + ic] = std::max(std::min(std::round(psrc[mb * IC + ic] * qscale + zp), 127.f), -128.f);
        }

        for (size_t ic = 0; ic < IC; ic++) {
            float dequant = (pdst[mb * IC + ic] - zp) * pscales[mb];
            if (std::abs(psrc[mb * IC + ic] - dequant) > 5.f) {
                std::cerr << "amin=" << amin << " amax=" << amax << std::endl;
                std::cerr << "dscale=" << dscale << " zp=" << zp << std::endl;
                std::cerr << "psrc=" << psrc[mb * IC + ic] << " quant=" << static_cast<float>(pdst[mb * IC + ic]) << " dequant=" << dequant << std::endl;
            }
        }
    }
}

template <typename T>
static std::vector<T> normalizeDimsTo2D(const std::vector<T>& dims) {
    return {std::accumulate(dims.begin(), dims.end() - 1, (T)1, std::multiplies<T>()), dims[dims.size() - 1]};
}

void DynamicQuantize::execute(dnnl::stream strm) {
    auto srcMemory = getSrcMemoryAtPort(DATA_ID);
    auto dstMemory = getDstMemoryAtPort(DATA_ID);
    auto scaleMemory = getDstMemoryAtPort(SCALES_ID);


    auto srcDims = normalizeDimsTo2D(srcMemory->getDesc().getShape().getDims());

    auto psrc = srcMemory->getDataAs<const float>();
    auto pdst = dstMemory->getDataAs<int8_t>();
    auto pscales = scaleMemory->getDataAs<float>();
    auto pzp = getOriginalOutputsNumber() == 3 ? getDstMemoryAtPort(ZERO_POINTS_ID)->getDataAs<int8_t>() : nullptr;

    if (!pzp) {
        quantize_symmetric(psrc, pdst, pscales, srcDims, groupSize);
    } else {
        OPENVINO_THROW_NOT_IMPLEMENTED("not supported yet");
    }

    // auto pzp = zeroPointMemory->getDataAs<int8_t>();

    // auto groups = div_up(IC, groupSize);
    // auto vec_loop_end = (groups - 1) * groupSize;

    // for (size_t mb = 0; mb < MB; mb++) {
    //     float amax = -FLT_MAX;
    //     float amin = FLT_MAX;
    //     for (size_t ic = 0; ic < IC; ic++) {
    //         amax = std::max(amax, psrc[mb * IC + ic]);
    //         amin = std::min(amin, psrc[mb * IC + ic]);
    //     }

    //     const float qscale = 255 / (amax - amin);
    //     const float dscale = 1.f / qscale;

    //     // const float qscale  = (dscale != 0) ? (1.0f / dscale) : 0;

    //     float zp = -std::round(qscale * amin) - 128;
    //     zp = std::max(std::min(zp, 127.f), -128.f);

    //     pscales[mb] = dscale;
    //     pzp[mb] = zp;
    //     for (size_t ic = 0; ic < IC; ic++) {
    //         pdst[mb * IC + ic] = std::max(std::min(std::round(psrc[mb * IC + ic] * qscale + zp), 127.f), -128.f);
    //     }

    //     for (size_t ic = 0; ic < IC; ic++) {
    //         float dequant = (pdst[mb * IC + ic] - zp) * pscales[mb];
    //         if (std::abs(psrc[mb * IC + ic] - dequant) > 5.f) {
    //             std::cerr << "amin=" << amin << " amax=" << amax << std::endl;
    //             std::cerr << "dscale=" << dscale << " zp=" << zp << std::endl;
    //             std::cerr << "psrc=" << psrc[mb * IC + ic] << " quant=" << static_cast<float>(pdst[mb * IC + ic]) << " dequant=" << dequant << std::endl;
    //         }
    //     }
    // }

    // parallel_for(MB, [&](size_t mb) {
    //     dynamic_quantization_runtime_params_t rt_params = {};
    //     rt_params.src_ptr = psrc + mb * IC;
    //     rt_params.qsrc_ptr = pdst + mb * IC;
    //     rt_params.src_scales_ptr = pscales + mb * groups;
    //     rt_params.ic_size = vec_loop_end;
    //     (*pKernel)(&rt_params);

    //     if (vec_loop_end != IC) {
    //         float amax = 0;
    //         for (size_t ic = vec_loop_end; ic < IC; ic++) {
    //             amax = std::max(amax, std::abs(psrc[mb * IC + ic]));
    //         }

    //         const float dscale = amax / 127;
    //         const float qscale  = (dscale != 0) ? (1.0f / dscale) : 0;

    //         pscales[mb * groups + groups - 1] = dscale;
    //         for (size_t ic = vec_loop_end; ic < IC; ic++) {
    //             pdst[mb * IC + ic] = std::round(psrc[mb * IC + ic] * qscale);
    //         }
    //     }
    // });
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
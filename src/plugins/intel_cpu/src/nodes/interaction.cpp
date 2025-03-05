// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "interaction.h"

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "dnnl_extension_utils.h"
#include "emitters/plugin/x64/jit_dnnl_emitters.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "onednn/dnnl.h"
#include "transformations/cpu_opset/x64/op/interaction.hpp"

using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov::intel_cpu::node {

#if defined(OPENVINO_ARCH_X86_64)

template <cpu_isa_t isa>
struct jit_move_scale_kernel : public jit_uni_move_scale_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_move_scale_kernel)

    explicit jit_move_scale_kernel(const jit_move_scale_compile_params& jcp)
        : jit_uni_move_scale_kernel(jcp),
          jit_generator(jit_name()) {
        runtime_prc = jcp_.src_prc == ov::element::bf16 ? ov::element::bf16 : ov::element::f32;
        if (jcp_.dst_prc == ov::element::i8 || jcp_.dst_prc == ov::element::u8) {
            runtime_prc = ov::element::f32;
        }
        vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen / runtime_prc.size();
    }
    ~jit_move_scale_kernel() override = default;

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

private:
    using Vmm =
        typename dnnl::impl::utils::conditional3<isa == cpu_isa_t::sse41, Xmm, isa == cpu_isa_t::avx2, Ymm, Zmm>::type;

    void generate() override {
        this->preamble();

#    define GET_OFF(field) offsetof(jit_move_scale_call_args, field)
        mov(reg_in, ptr[reg_params + GET_OFF(p_in)]);
        mov(reg_out, ptr[reg_params + GET_OFF(p_out)]);
        mov(reg_work_amount, jcp_.input_size);

        if (jcp_.with_scales) {
            mov(reg_scales, ptr[reg_params + GET_OFF(p_scales)]);
        }

        Xbyak::Label move_scale_loop_label;
        Xbyak::Label move_scale_end_label;

        if (jcp_.with_scales && jcp_.broadcast_scales) {
            uni_vmovss(Xmm(vmm_scales.getIdx()), ptr[reg_scales]);
            uni_vbroadcastss(vmm_scales, Xmm(vmm_scales.getIdx()));
        }

        mov(reg_in_aux, reg_in);
        mov(reg_out_aux, reg_out);

        size_t tail_size = jcp_.input_size % vec_size;
        L(move_scale_loop_label);
        {
            cmp(reg_work_amount, vec_size);
            jl(move_scale_end_label, T_NEAR);

            load_scale_store(vec_size);

            sub(reg_work_amount, vec_size);

            jmp(move_scale_loop_label, T_NEAR);
        }
        L(move_scale_end_label);
        if (tail_size) {
            load_scale_store(tail_size);
        }

        this->postamble();

        for (const auto& emitter : emitters) {
            if (emitter.second) {
                emitter.second->emit_data();
            }
        }
    }

    void load_scale_store(size_t step) {
        bool is_tail = step < vec_size;
        load(vmm_in, reg_in_aux, jcp_.src_prc, runtime_prc, step, false);

        if (jcp_.with_scales) {
            if (!jcp_.broadcast_scales) {
                load(vmm_scales, reg_scales, ov::element::f32, ov::element::f32, step, false);
                add(reg_scales, sizeof(float) * step);
            }
            uni_vmulps(vmm_in, vmm_in, vmm_scales);
        }

        store(reg_out_aux, vmm_in, runtime_prc, jcp_.dst_prc, step);

        if (!is_tail) {
            add(reg_in_aux, jcp_.src_prc.size() * step);
            add(reg_out_aux, jcp_.dst_prc.size() * step);
        }
    }
#    undef GET_OFF

    inline void load(const Vmm& vmm_dst,
                     const Xbyak::Reg64& reg_src,
                     ov::element::Type src_prc,
                     ov::element::Type dst_prc,
                     const int& elt_num,
                     bool fill) {
        const auto seed = load_emitter_params(src_prc, dst_prc, elt_num, fill, "float_min").hash();
        if (!emitters[seed]) {
            emitters[seed] =
                std::make_unique<jit_load_emitter>(this, isa, src_prc, dst_prc, elt_num, src_prc, fill, "float_min");
        }

        emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), 0},
                                  {static_cast<size_t>(vmm_dst.getIdx())},
                                  pool_aux_vmm_idxs,
                                  pool_aux_gpr_idxs);
    }
    inline void store(const Xbyak::Reg64& reg_dst,
                      const Vmm& vmm_src,
                      ov::element::Type src_prc,
                      ov::element::Type dst_prc,
                      const int& elt_num) {
        const auto seed = store_emitter_params(src_prc, dst_prc, elt_num).hash();
        if (!emitters[seed]) {
            emitters[seed] = std::make_unique<jit_store_emitter>(this, isa, src_prc, dst_prc, elt_num);
        }

        emitters[seed]->emit_code({static_cast<size_t>(vmm_src.getIdx())},
                                  {static_cast<size_t>(reg_dst.getIdx())},
                                  pool_aux_vmm_idxs,
                                  pool_aux_gpr_idxs);
    }

    size_t vec_size;
    ov::element::Type runtime_prc;

    Xmm xmm_tmp = Xmm(2);
    Vmm vmm_scales = Vmm(0);
    Vmm vmm_in = Vmm(1);

    Reg64 reg_in = r8;
    Reg64 reg_in_aux = r9;
    Reg64 reg_out = r10;
    Reg64 reg_out_aux = r11;
    Reg64 reg_scales = r12;
    Reg64 reg_work_amount = r14;
    Reg64 reg_params = abi_param1;

    const std::vector<size_t> pool_aux_gpr_idxs = {static_cast<size_t>(rsi.getIdx()),
                                                   static_cast<size_t>(rbp.getIdx())};
    const std::vector<size_t> pool_aux_vmm_idxs = {static_cast<size_t>(xmm_tmp.getIdx())};

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
};

#endif  // OPENVINO_ARCH_X86_64

Interaction::Interaction(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, NgraphShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }
    const auto interaction = ov::as_type_ptr<const InteractionNode>(op);
    const std::vector<float>& scales = interaction->get_output_scales();
    if (!scales.empty()) {
        fqScales = scales;
        outputDataType = interaction->get_output_element_type(0);
    }
}

void Interaction::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }
    dataPrecision = getOriginalInputPrecisionAtPort(0);
    if (dataPrecision != ov::element::f32 && dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx512_core_bf16)) {
        dataPrecision = ov::element::bf16;
    } else {
        dataPrecision = ov::element::f32;
    }

    if (fqScales.empty()) {
        outputDataType = dataPrecision;
    }
    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    for (size_t i = 0; i < getParentEdges().size(); ++i) {
        inPortConfigs.emplace_back(LayoutType::ncsp, dataPrecision, getInputShapeAtPort(i), false, -1);
    }
    // initialize output port
    std::vector<PortConfigurator> outPortConfigs = {
        PortConfigurator{LayoutType::ncsp, outputDataType, getOutputShapeAtPort(0), false, -1}};
    // add descriptor
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

static inline void cat(uint8_t* out,
                       const std::vector<const uint8_t*>& in,
                       const std::vector<uint32_t>& feature_sizes,
                       int64_t bs,
                       size_t elemSize) {
    size_t offset = 0;
    for (size_t j = 0; j < feature_sizes.size(); j++) {
        cpu_memcpy(out + offset * elemSize, in[j] + bs * feature_sizes[j] * elemSize, feature_sizes[j] * elemSize);
        offset += feature_sizes[j];
    }
}

static inline void flat_triangle(const uint8_t* in, uint8_t* out, size_t size, size_t elemSize) {
    size_t offset = 0;
    for (size_t i = 1; i < size; i++) {
        cpu_memcpy(out + offset * elemSize, in + i * size * elemSize, i * elemSize);
        offset += i;
    }
}

void Interaction::execRef(const dnnl::stream& strm) {
    using namespace dnnl;
    auto* outFeaturesPtr = getDstDataAtPortAs<uint8_t>(0);
    std::vector<const uint8_t*> inputPtrs(inputSizes);
    for (uint32_t n = 0; n < inputSizes; n++) {
        auto inPtr = getSrcDataAtPortAs<const uint8_t>(n);
        inputPtrs[n] = inPtr;
    }
    std::unordered_map<int, memory> mem_ags{{DNNL_ARG_SRC, inputMemPtr->getPrimitive()},
                                            {DNNL_ARG_WEIGHTS, inputMemPtr->getPrimitive()},
                                            {DNNL_ARG_DST, outputMemPtr->getPrimitive()}};
    float* scales = fqScales.empty() ? nullptr : fqScales.data();
    for (int64_t start = 0; start < static_cast<int64_t>(batchSize); start++) {
        cat(inputMemPtr->getDataAs<uint8_t>(), inputPtrs, featureSizes, start, dataPrecision.size());
        prim.execute(strm, mem_ags);
        flat_triangle(outputMemPtr->getDataAs<const uint8_t>(),
                      flatMemPtr->getDataAs<uint8_t>(),
                      inputSizes,
                      dataPrecision.size());
        // in1 dense feature
        // in2 flatted interaction features
        if (moveFeatureKernel) {
            jit_move_scale_call_args featArgs;
            featArgs.p_in = inputPtrs[0] + start * featureSize * dataPrecision.size();
            featArgs.p_out = outFeaturesPtr + start * outputFeaturesLen * outputDataType.size();
            featArgs.p_scales = scales;
            (*moveFeatureKernel)(&featArgs);
        }
        if (moveInteractKernel) {
            jit_move_scale_call_args interArgs;
            interArgs.p_in = flatMemPtr->getData();
            interArgs.p_out = outFeaturesPtr + (start * outputFeaturesLen + featureSize) * outputDataType.size();
            interArgs.p_scales = scales;
            (*moveInteractKernel)(&interArgs);
        }
    }
}

void Interaction::execute(const dnnl::stream& strm) {
    execRef(strm);
}

bool Interaction::created() const {
    return getType() == Type::Interaction;
}

void Interaction::prepareParams() {
    using namespace dnnl;
    const auto& denseFeatureDims = getParentEdgeAt(0)->getMemory().getStaticDims();
    batchSize = denseFeatureDims[0];
    featureSize = denseFeatureDims[1];
    inputSizes = inputShapes.size();
    interactFeatureSize = inputSizes * (inputSizes - 1) / 2;
    outputFeaturesLen = interactFeatureSize + featureSize;
    std::vector<int64_t> lhsShape({static_cast<int64_t>(inputSizes), static_cast<int64_t>(featureSize)});
    std::vector<int64_t> lhsStride({static_cast<int64_t>(featureSize), 1});
    std::vector<int64_t> rhsShape({static_cast<int64_t>(featureSize), static_cast<int64_t>(inputSizes)});
    std::vector<int64_t> rhsStride({1, static_cast<int64_t>(featureSize)});
    std::vector<int64_t> resShape({static_cast<int64_t>(inputSizes), static_cast<int64_t>(inputSizes)});
    std::vector<int64_t> resStride({static_cast<int64_t>(inputSizes), 1});
    auto dataType = DnnlExtensionUtils::ElementTypeToDataType(dataPrecision);
    auto src_md = memory::desc(lhsShape, dataType, lhsStride);
    auto weights_md = memory::desc(rhsShape, dataType, rhsStride);
    auto dst_md = memory::desc(resShape, dataType, resStride);
    primitive_attr matmul_attr;
    auto matmul_pd = matmul::primitive_desc(getEngine(), src_md, weights_md, dst_md, matmul_attr);
    prim = matmul(matmul_pd);
    featureSizes.assign(inputSizes, featureSize);
    auto initMemoryPtr = [&](const ov::element::Type& prc, const intel_cpu::Shape& shape, MemoryPtr& ptr) {
        ptr = std::make_shared<Memory>(getEngine(), intel_cpu::DnnlBlockedMemoryDesc(prc, shape));
    };
    initMemoryPtr(dataPrecision, intel_cpu::Shape{inputSizes, featureSize}, inputMemPtr);
    initMemoryPtr(dataPrecision, intel_cpu::Shape{inputShapes.size(), inputShapes.size()}, outputMemPtr);
    initMemoryPtr(dataPrecision, intel_cpu::Shape{interactFeatureSize}, flatMemPtr);

    jit_move_scale_compile_params jcp;
    jcp.src_prc = dataPrecision;
    jcp.dst_prc = outputDataType;
    jcp.with_scales = !fqScales.empty();
    jcp.broadcast_scales = fqScales.size() == 1;
    jcp.input_size = featureSize;

    jit_move_scale_compile_params interJcp;
    interJcp.src_prc = dataPrecision;
    interJcp.dst_prc = outputDataType;
    interJcp.with_scales = !fqScales.empty();
    interJcp.broadcast_scales = fqScales.size() == 1;
    interJcp.input_size = interactFeatureSize;

#if defined(OPENVINO_ARCH_X86_64)
    if (mayiuse(cpu_isa_t::avx512_core)) {
        moveFeatureKernel = std::make_unique<jit_move_scale_kernel<cpu_isa_t::avx512_core>>(jcp);
        moveInteractKernel = std::make_unique<jit_move_scale_kernel<cpu_isa_t::avx512_core>>(interJcp);
    } else if (mayiuse(cpu_isa_t::avx2)) {
        moveFeatureKernel = std::make_unique<jit_move_scale_kernel<cpu_isa_t::avx2>>(jcp);
        moveInteractKernel = std::make_unique<jit_move_scale_kernel<cpu_isa_t::avx2>>(interJcp);
    } else if (mayiuse(cpu_isa_t::sse41)) {
        moveFeatureKernel = std::make_unique<jit_move_scale_kernel<cpu_isa_t::sse41>>(jcp);
        moveInteractKernel = std::make_unique<jit_move_scale_kernel<cpu_isa_t::sse41>>(interJcp);
    }
#endif  // OPENVINO_ARCH_X86_64

    if (moveFeatureKernel && moveInteractKernel) {
        moveFeatureKernel->create_ker();
        moveInteractKernel->create_ker();
    } else {
        THROW_CPU_NODE_ERR("cannot create jit eltwise kernel");
    }
#ifdef CPU_DEBUG_CAPS
    if (prim) {
        auto pd = prim.get_primitive_desc();
        DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
    }
#endif
}

void Interaction::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

bool Interaction::neverExecute() const {
    return false;
}

bool Interaction::isExecutable() const {
    return true;
}

bool Interaction::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto interaction = ov::as_type_ptr<const InteractionNode>(op);
        if (!interaction) {
            errorMessage = "Only Interaction operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace ov::intel_cpu::node

// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "roi_pooling.h"

#include "onednn/dnnl.h"
#include "dnnl_extension_utils.h"
#include "selective_build.h"

#include <openvino/opsets/opset2.hpp>

#include "openvino/core/parallel.hpp"
#include "utils/bfloat16.hpp"
#include "emitters/plugin/x64/jit_load_store_emitters.hpp"

#include "cpu/x64/jit_generator.hpp"
#include "common/primitive_hashing_utils.hpp"

#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

using namespace dnnl;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(field) offsetof(jit_roi_pooling_call_args, field)

namespace ov {
namespace intel_cpu {
namespace node {

#if defined(OPENVINO_ARCH_X86_64)
template <cpu_isa_t isa>
struct jit_uni_roi_pooling_kernel_f32 : public jit_uni_roi_pooling_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_roi_pooling_kernel_f32);

    explicit jit_uni_roi_pooling_kernel_f32(jit_roi_pooling_params jcp) : jit_uni_roi_pooling_kernel(jcp), jit_generator(jit_name()) {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    };

    void generate() override {
        load_emitter.reset(new jit_load_emitter(this, isa, jpp_.src_prc, ov::element::f32, step));
        store_emitter.reset(new jit_store_emitter(this, isa, ov::element::f32, jpp_.dst_prc, step));
        store_empty_roi_emitter.reset(new jit_store_emitter(this, isa, jpp_.src_prc, jpp_.dst_prc, step));

        this->preamble();

        Label exit_label;
        Label tail_label;

        mov(reg_input, ptr[this->param1 + GET_OFF(src)]);
        mov(reg_output, ptr[this->param1 + GET_OFF(dst)]);
        mov(reg_bin_area, ptr[this->param1 + GET_OFF(bin_area)]);
        mov(reg_c_blocks, ptr[this->param1 + GET_OFF(c_blocks)]);

        if (jpp_.alg == Algorithm::ROIPoolingMax) {
            mov(reg_kh, ptr[this->param1 + GET_OFF(kh)]);
            mov(reg_kw, ptr[this->param1 + GET_OFF(kw)]);
        } else {
            mov(reg_yf, ptr[this->param1 + GET_OFF(yf)]);
            mov(reg_xf, ptr[this->param1 + GET_OFF(xf)]);
            mov(reg_yoff, ptr[this->param1 + GET_OFF(yoff)]);
            mov(reg_xoff, ptr[this->param1 + GET_OFF(xoff)]);
        }

        load_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx()), static_cast<size_t>(reg_load_table.getIdx())};
        store_pool_gpr_idxs = {static_cast<size_t>(reg_load_store_mask.getIdx())};
        store_pool_vec_idxs = {static_cast<size_t>(vmm_zero.getIdx())};

        int nb_c_tail = jpp_.nb_c % jpp_.nb_c_blocking;
        cmp(reg_c_blocks, jpp_.nb_c_blocking);
        jne(nb_c_tail ? tail_label : exit_label, T_NEAR);

        loop_body(jpp_.nb_c_blocking);
        jmp(exit_label, T_NEAR);

        if (nb_c_tail) {
            L(tail_label);
            loop_body(nb_c_tail);
        }

        L(exit_label);

        this->postamble();

        load_emitter->emit_data();
        store_emitter->emit_data();
        store_empty_roi_emitter->emit_data();
    }

private:
    using Vmm = typename conditional3<isa == cpu::x64::sse41, Xbyak::Xmm, isa == cpu::x64::avx2,
            Xbyak::Ymm, Xbyak::Zmm>::type;

    const int vlen = cpu_isa_traits<isa>::vlen;
    const int step = vlen / sizeof(float);

    Vmm vmm_mask = Vmm(0);
    Vmm vmm_zero = Vmm(2); // avoid using xmm0 (reserved as mask reg in sse41-instruction blendvps)

    Xmm xmm_yf = Xmm(0);
    Vmm vmm_yf = Vmm(0);
    Xmm xmm_xf = Xmm(1);
    Vmm vmm_xf = Vmm(1);

    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
    std::vector<size_t> load_pool_gpr_idxs;

    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;
    std::unique_ptr<jit_store_emitter> store_empty_roi_emitter = nullptr;
    std::vector<size_t> store_pool_gpr_idxs;
    std::vector<size_t> store_pool_vec_idxs;

    Vmm get_acc_reg(int idx) { return Vmm(2*idx + 1); }
    Vmm get_src_reg(int idx) { return Vmm(2*idx + 2); }

    Opmask k_store_mask = Opmask(7);

    const unsigned char _cmp_lt_os = 1;

    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input     = r8;
    reg64_t aux_reg_input = rax;
    reg64_t aux_reg_input1 = rdx;
    reg64_t reg_output    = r9;
    reg64_t reg_kh    = r10;
    reg64_t reg_kw    = r11;

    reg64_t h_iter = r13;
    reg64_t w_iter = r14;

    reg64_t reg_c_blocks = rbx;
    reg64_t reg_bin_area = rdx;

    reg64_t reg_yf = reg_kh;
    reg64_t reg_xf = reg_kw;

    reg64_t reg_yoff = h_iter;
    reg64_t reg_xoff = r12;

    Xbyak::Reg64 reg_load_table = r15;
    Xbyak::Reg64 reg_load_store_mask = abi_param1;

    std::vector<size_t> get_local_store_pool_vec_idxs(Vmm vmm) const {
        std::vector<size_t> local_store_pool_vec_idxs = { static_cast<size_t>(vmm.getIdx()) };
        local_store_pool_vec_idxs.insert(local_store_pool_vec_idxs.begin(), store_pool_vec_idxs.begin(), store_pool_vec_idxs.end());
        return local_store_pool_vec_idxs;
    }

    void roi_pool_max(int c_blocks) {
        Label h_loop_label;
        Label w_loop_label;

        mov(aux_reg_input, reg_input);

        const int src_c_off = jpp_.ih * jpp_.iw * jpp_.c_block * jpp_.src_prc.size();
        for (int i = 0; i < c_blocks; i++) {
            Vmm vmm_max = get_acc_reg(i);

            load_emitter->emit_code({static_cast<size_t>(reg_input.getIdx()), static_cast<size_t>(i * src_c_off)}, {static_cast<size_t>(vmm_max.getIdx())},
                                    {}, load_pool_gpr_idxs);
        }

        xor_(h_iter, h_iter);
        L(h_loop_label); {
            xor_(w_iter, w_iter);
            mov(aux_reg_input1, aux_reg_input);
            L(w_loop_label); {
                for (int i = 0; i < c_blocks; i++) {
                    Vmm vmm_max = get_acc_reg(i);
                    Vmm vmm_src = get_src_reg(i);

                    load_emitter->emit_code({static_cast<size_t>(aux_reg_input1.getIdx()), static_cast<size_t>(i * src_c_off)},
                                            {static_cast<size_t>(vmm_src.getIdx())}, {}, load_pool_gpr_idxs);

                    if (isa == cpu::x64::sse41) {
                        movups(vmm_mask, vmm_max);
                        cmpps(vmm_mask, vmm_src, _cmp_lt_os);
                        blendvps(vmm_max, vmm_src);
                    } else if (isa == cpu::x64::avx2) {
                        vcmpps(vmm_mask, vmm_max, vmm_src, _cmp_lt_os);
                        vblendvps(vmm_max, vmm_max, vmm_src, vmm_mask);
                    } else if (isa == cpu::x64::avx512_core) {
                        vcmpps(k_store_mask,  vmm_max,  vmm_src, _cmp_lt_os);
                        vblendmps(vmm_max| k_store_mask, vmm_max, vmm_src);
                    }
                }

                add(aux_reg_input1, jpp_.c_block * jpp_.src_prc.size());

                inc(w_iter);
                cmp(w_iter, reg_kw);
                jl(w_loop_label, T_NEAR);
            }

            add(aux_reg_input, jpp_.iw * jpp_.c_block * jpp_.src_prc.size());

            inc(h_iter);
            cmp(h_iter, reg_kh);
            jl(h_loop_label, T_NEAR);
        }

        const int dst_c_off = jpp_.oh * jpp_.ow * jpp_.c_block * jpp_.dst_prc.size();
        for (int i = 0; i < c_blocks; i++) {
            Vmm vmm_dst = get_acc_reg(i);

            store_emitter->emit_code({static_cast<size_t>(vmm_dst.getIdx())},
                                     {static_cast<size_t>(reg_output.getIdx()), static_cast<size_t>(i * dst_c_off)},
                                     get_local_store_pool_vec_idxs(vmm_dst), store_pool_gpr_idxs);
        }
    }

    void roi_pool_bilinear(int c_blocks) {
        uni_vmovq(xmm_yf, reg_yf);
        uni_vbroadcastss(vmm_yf, xmm_yf);
        uni_vmovq(xmm_xf, reg_xf);
        uni_vbroadcastss(vmm_xf, xmm_xf);

        Vmm vmm_src00 = get_src_reg(0);
        Vmm vmm_src01 = get_src_reg(1);
        Vmm vmm_src10 = get_src_reg(2);
        Vmm vmm_src11 = get_src_reg(3);

        for (int i = 0; i < c_blocks; i++) {
            const int src_c_off = i * jpp_.ih * jpp_.iw * jpp_.c_block * jpp_.src_prc.size();

            mov(aux_reg_input, reg_input);

            load_emitter->emit_code({static_cast<size_t>(aux_reg_input.getIdx()), static_cast<size_t>(src_c_off)}, {static_cast<size_t>(vmm_src00.getIdx())},
                                    {}, load_pool_gpr_idxs);
            add(aux_reg_input, reg_xoff);

            load_emitter->emit_code({static_cast<size_t>(aux_reg_input.getIdx()), static_cast<size_t>(src_c_off)}, {static_cast<size_t>(vmm_src01.getIdx())},
                                    {}, load_pool_gpr_idxs);

            add(aux_reg_input, reg_yoff);
            load_emitter->emit_code({static_cast<size_t>(aux_reg_input.getIdx()), static_cast<size_t>(src_c_off)}, {static_cast<size_t>(vmm_src11.getIdx())},
                                    {}, load_pool_gpr_idxs);
            sub(aux_reg_input, reg_xoff);

            load_emitter->emit_code({static_cast<size_t>(aux_reg_input.getIdx()), static_cast<size_t>(src_c_off)}, {static_cast<size_t>(vmm_src10.getIdx())},
                                    {}, load_pool_gpr_idxs);

            uni_vsubps(vmm_src01, vmm_src01, vmm_src00);
            uni_vfmadd213ps(vmm_src01, vmm_xf, vmm_src00);

            uni_vsubps(vmm_src11, vmm_src11, vmm_src10);
            uni_vfmadd213ps(vmm_src11, vmm_xf, vmm_src10);

            uni_vsubps(vmm_src11, vmm_src11, vmm_src01);
            uni_vfmadd213ps(vmm_src11, vmm_yf, vmm_src01);

            const int dst_c_off = i * jpp_.oh * jpp_.ow * jpp_.c_block * jpp_.dst_prc.size();

            store_emitter->emit_code({static_cast<size_t>(vmm_src11.getIdx())},
                                     {static_cast<size_t>(reg_output.getIdx()), static_cast<size_t>(dst_c_off)},
                                     get_local_store_pool_vec_idxs(vmm_src11), store_pool_gpr_idxs);
        }
    }

    void empty_roi(int c_blocks) {
        uni_vpxor(vmm_zero, vmm_zero, vmm_zero);

        const int dst_c_off = jpp_.oh * jpp_.ow * jpp_.c_block * jpp_.dst_prc.size();
        for (int i = 0; i < c_blocks; i++) {
            store_empty_roi_emitter->emit_code({static_cast<size_t>(vmm_zero.getIdx())},
                                               {static_cast<size_t>(reg_output.getIdx()), static_cast<size_t>(i * dst_c_off)},
                                               store_pool_vec_idxs, store_pool_gpr_idxs);
        }
    }

    void loop_body(int c_blocks) {
        Label empty_roi_label;
        Label exit_label;

        cmp(reg_bin_area, 0);
        je(empty_roi_label, T_NEAR);

        if (jpp_.alg == Algorithm::ROIPoolingMax)
            roi_pool_max(c_blocks);
        else
            roi_pool_bilinear(c_blocks);

        if (isa == cpu::x64::sse41) {
            add(reg_input, 4 * jpp_.src_prc.size());
            add(reg_output, 4 * jpp_.dst_prc.size());

            if (jpp_.alg == Algorithm::ROIPoolingMax)
                roi_pool_max(c_blocks);
            else
                roi_pool_bilinear(c_blocks);
        }
        jmp(exit_label, T_NEAR);

        L(empty_roi_label);
        empty_roi(c_blocks);
        if (isa == cpu::x64::sse41) {
            add(reg_output, 4 * jpp_.dst_prc.size());
            empty_roi(c_blocks);
        }

        L(exit_label);
    }
};
#endif

namespace {
struct RoiPoolingKey {
    jit_roi_pooling_params refParams;

    size_t hash() const;
    bool operator==(const RoiPoolingKey& rhs) const;
};

size_t RoiPoolingKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    seed = hash_combine(seed, refParams.mb);
    seed = hash_combine(seed, refParams.c);
    seed = hash_combine(seed, refParams.nb_c);
    seed = hash_combine(seed, refParams.c_block);
    seed = hash_combine(seed, refParams.nb_c_blocking);
    seed = hash_combine(seed, refParams.ih);
    seed = hash_combine(seed, refParams.iw);
    seed = hash_combine(seed, refParams.oh);
    seed = hash_combine(seed, refParams.ow);
    seed = hash_combine(seed, refParams.alg);
    seed = hash_combine(seed, refParams.src_prc.hash());
    seed = hash_combine(seed, refParams.dst_prc.hash());
    seed = hash_combine(seed, refParams.spatial_scale);
    seed = hash_combine(seed, refParams.pooled_h);
    seed = hash_combine(seed, refParams.pooled_w);

    return seed;
}

bool RoiPoolingKey::operator==(const RoiPoolingKey &rhs) const {
    return refParams == rhs.refParams;
}
} // namespace

bool jit_roi_pooling_params::operator==(const jit_roi_pooling_params &rhs) const noexcept {
    return mb == rhs.mb &&
           c == rhs.c &&
           ih == rhs.ih &&
           iw == rhs.iw &&
           oh == rhs.oh &&
           ow == rhs.ow &&
           c_block == rhs.c_block &&
           nb_c == rhs.nb_c &&
           nb_c_blocking == rhs.nb_c_blocking &&
           spatial_scale == rhs.spatial_scale &&
           pooled_h == rhs.pooled_h &&
           pooled_w == rhs.pooled_w &&
           src_prc == rhs.src_prc &&
           dst_prc == rhs.dst_prc &&
           alg == rhs.alg;
}

bool ROIPooling::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto roiPooling = ov::as_type_ptr<const ov::opset2::ROIPooling>(op);
        if (!roiPooling) {
            errorMessage = "Only opset2 ROIPooling operation is supported";
            return false;
        }
        const std::string mode = roiPooling->get_method();
        if (mode != "max" && mode != "bilinear") {
            errorMessage = "Doesn't support method: " + mode;
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

ROIPooling::ROIPooling(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    std::string errorPrefix = "ROIPooling layer with name '" + getName() + "' ";

    auto roiPooling = ov::as_type_ptr<const ov::opset2::ROIPooling>(op);
    refParams.pooled_h = roiPooling->get_output_roi()[0];
    refParams.pooled_w = roiPooling->get_output_roi()[1];
    refParams.spatial_scale = roiPooling->get_spatial_scale();
    const auto& m = roiPooling->get_method();
    if (m == "max") {
        algorithm = Algorithm::ROIPoolingMax;
    } else if (m == "bilinear") {
        algorithm = Algorithm::ROIPoolingBilinear;
    }
}

void ROIPooling::getSupportedDescriptors() {
    if (getParentEdges().size() != 2)
        OPENVINO_THROW(errorPrefix, "has incorrect number of input edges: ", getParentEdges().size());
    if (getChildEdges().empty())
        OPENVINO_THROW(errorPrefix, "has incorrect number of output edges: ", getChildEdges().size());

    if (getInputShapeAtPort(0).getRank() != 4) {
        OPENVINO_THROW(errorPrefix, "doesn't support 0th input with rank: ", getInputShapeAtPort(0).getRank());
    }

    if (getInputShapeAtPort(1).getRank() != 2) {
        OPENVINO_THROW(errorPrefix, "doesn't support 1st input with rank: ", getInputShapeAtPort(1).getRank());
    }

    if (getOutputShapeAtPort(0).getRank() != 4) {
        OPENVINO_THROW(errorPrefix, "doesn't support output with rank: ", getOutputShapeAtPort(0).getRank());
    }

    const auto& dims = getInputShapeAtPort(1).getDims();
    if (dims[1] != 5) {
        OPENVINO_THROW(errorPrefix, "has invalid shape on 1st input: [", dims[0], ",", dims[1], "]");
    }
}

void ROIPooling::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto format = mayiuse(avx512_core) ? LayoutType::nCsp16c : LayoutType::nCsp8c;
    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_core)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    refParams.src_prc = getOriginalInputPrecisionAtPort(0);

    if (!mayiuse(avx512_core)) {
        if (refParams.src_prc == ov::element::bf16)
            refParams.src_prc = ov::element::f32;
    }

    if (impl_type != impl_desc_type::ref && refParams.src_prc == ov::element::f16) {
        refParams.src_prc = ov::element::f32;
    }

    addSupportedPrimDesc({{format, refParams.src_prc},
                          {LayoutType::ncsp, refParams.src_prc}},
                         {{format, refParams.src_prc}},
                          impl_type);
}

void ROIPooling::createPrimitive() {
    auto selectedPD = getSelectedPrimitiveDescriptor();
    if (!selectedPD)
        OPENVINO_THROW("CPU ROI Pooling node with name '", getName(), "' doesn't have primitive descriptors.");

    refParams.c_block = mayiuse(cpu::x64::avx512_core) ? 16 : 8;;
    refParams.nb_c_blocking = mayiuse(cpu::x64::avx512_core) ? 15 : 7;
    refParams.alg = getAlgorithm();

    const auto& config = selectedPD->getConfig();
    refParams.src_prc = config.inConfs[0].getMemDesc()->getPrecision();
    refParams.dst_prc = config.outConfs[0].getMemDesc()->getPrecision();

    if (inputShapesDefined()) {
        if (needPrepareParams() && isExecutable())
            prepareParams();
        updateLastInputDims();
    }
}

void ROIPooling::execute(dnnl::stream strm) {
    if (execPtr) {
        const auto &srcMemory0 = getParentEdgeAt(0)->getMemory();
        const auto &srcMemory1 = getParentEdgeAt(1)->getMemory();
        const auto &dstMemory = getChildEdgeAt(0)->getMemory();
        execPtr->exec(srcMemory0, srcMemory1, dstMemory);
    } else {
        OPENVINO_THROW("Can't execute ROI Pooling node. Primitive wasn't created");
    }
}

void ROIPooling::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void ROIPooling::prepareParams() {
    const auto& srcMemPtr0 = getSrcMemoryAtPort(0);
    const auto& srcMemPtr1 = getSrcMemoryAtPort(0);
    const auto& dstMemPtr = getDstMemoryAtPort(0);
    if (!srcMemPtr0 || !srcMemPtr0->isDefined())
        OPENVINO_THROW("Input memory is undefined.");
    if (!srcMemPtr1 || !srcMemPtr1->isDefined())
        OPENVINO_THROW("Input memory is undefined.");
    if (!dstMemPtr || !dstMemPtr->isDefined())
        OPENVINO_THROW("Destination is undefined.");
    if (getSelectedPrimitiveDescriptor() == nullptr)
        OPENVINO_THROW("Preferable primitive descriptor is not set.");

    const auto& inDims = getParentEdgeAt(0)->getMemory().getStaticDims();
    const auto& outDims = getChildEdgeAt(0)->getMemory().getStaticDims();

    refParams.mb = outDims[0];
    refParams.c = rnd_up(inDims[1], refParams.c_block);
    refParams.nb_c = refParams.c / refParams.c_block;
    refParams.ih = inDims[2];
    refParams.iw = inDims[3];
    refParams.oh = outDims[2];
    refParams.ow = outDims[3];

    RoiPoolingKey key = {refParams};
    auto builder = [](const RoiPoolingKey& key) {
        return ROIPoolingExecutor::createROIPoolingNewExecutor(key.refParams);
    };
    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    execPtr = result.first;
}

template <typename T>
class ROIPooling::ROIPoolingJitExecutor : public ROIPooling::ROIPoolingExecutor {
public:
    ROIPoolingJitExecutor(const jit_roi_pooling_params &jpp) {
#if defined(OPENVINO_ARCH_X86_64)
        if (mayiuse(cpu::x64::avx512_core)) {
            roi_pooling_kernel.reset(new jit_uni_roi_pooling_kernel_f32<cpu::x64::avx512_core>(jpp));
        } else if (mayiuse(cpu::x64::avx2)) {
            roi_pooling_kernel.reset(new jit_uni_roi_pooling_kernel_f32<cpu::x64::avx2>(jpp));
        } else if (mayiuse(cpu::x64::sse41)) {
            roi_pooling_kernel.reset(new jit_uni_roi_pooling_kernel_f32<cpu::x64::sse41>(jpp));
        } else {
            OPENVINO_THROW("Can't create jit RoiPooling kernel");
        }

        if (roi_pooling_kernel)
            roi_pooling_kernel->create_ker();
#endif
    }

    void exec(
        const IMemory& srcData,
        const IMemory& srcRoi,
        const IMemory& dst) override {
        if (!roi_pooling_kernel)
            OPENVINO_THROW("Could not execute. Kernel for RoiPooling node was not compiled.");

        auto src_strides = srcData.getDescWithType<BlockedMemoryDesc>()->getStrides();
        auto src_roi_step = srcRoi.getDescWithType<BlockedMemoryDesc>()->getStrides()[0];
        auto dst_strides = dst.getDescWithType<BlockedMemoryDesc>()->getStrides();
        const auto* src_ptr = srcData.getDataAs<const T>();
        const auto* roi_ptr = srcRoi.getDataAs<const T>();
        auto* dst_ptr = dst.getDataAs<T>();
        executeOptimizedGeneric(src_ptr, roi_ptr, dst_ptr, src_strides, dst_strides, src_roi_step);
    }

private:
    void executeOptimizedGeneric(
        const T* src_data,
        const T* src_roi,
        T* dst,
        const VectorDims& src_strides,
        const VectorDims& dst_strides,
        const size_t src_roi_step) {
        const auto& jpp = roi_pooling_kernel->jpp_;
        int cb_work = impl::utils::div_up(jpp.nb_c, jpp.nb_c_blocking);
        int MB = jpp.mb;

        int real_rois = 0;
        for (; real_rois < MB; real_rois++) {
            size_t roi_off = real_rois * src_roi_step;

            const auto *src_roi_ptr = &src_roi[roi_off];
            int roi_batch_ind = static_cast<int>(src_roi_ptr[0]);
            if (roi_batch_ind == -1) {
                break;
            }
        }

        parallel_for4d(MB, cb_work, jpp.oh, jpp.ow, [&](int n, int cbb, int oh, int ow) {
            auto arg = jit_roi_pooling_call_args();
            int cb = cbb * jpp.nb_c_blocking;
            int cb_num = jpp.nb_c_blocking;
            arg.c_blocks = std::min(cb + cb_num, jpp.nb_c) - cb;

            if (n >= real_rois) {
                arg.bin_area = 0;
                arg.dst = &dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]];
                (*roi_pooling_kernel)(&arg);
            } else {
                size_t roi_off = n * src_roi_step;
                const auto *src_roi_ptr = &src_roi[roi_off];

                int roi_batch_ind = static_cast<int>(src_roi_ptr[0]);

                if (jpp.alg == Algorithm::ROIPoolingMax) {
                    int roi_start_w = static_cast<int>(round(src_roi_ptr[1] * jpp.spatial_scale));
                    int roi_start_h = static_cast<int>(round(src_roi_ptr[2] * jpp.spatial_scale));
                    int roi_end_w = static_cast<int>(round(src_roi_ptr[3] * jpp.spatial_scale));
                    int roi_end_h = static_cast<int>(round(src_roi_ptr[4] * jpp.spatial_scale));

                    int hstart, hend, wstart, wend;
                    std::tie(hstart, hend, wstart, wend) = getBordersForMaxMode(
                        roi_start_h, roi_end_h, roi_start_w, roi_end_w, jpp.ih, oh, jpp.iw, ow, jpp.pooled_h, jpp.pooled_w);

                    arg.src = &src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] + hstart * src_strides[2] + wstart * src_strides[3]];
                    arg.dst = &dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]];

                    arg.bin_area = (hend - hstart) * (wend - wstart);
                    arg.kh = hend - hstart;
                    arg.kw = wend - wstart;
                } else {
                    float roi_start_w_ = src_roi_ptr[1];
                    float roi_start_h_ = src_roi_ptr[2];
                    float roi_end_w_   = src_roi_ptr[3];
                    float roi_end_h_   = src_roi_ptr[4];

                    float in_x, in_y;
                    std::tie(in_x, in_y) = getXYForBilinearMode(
                        roi_start_h_, roi_end_h_, roi_start_w_, roi_end_w_, jpp.ih, oh, jpp.iw, ow, jpp.pooled_h, jpp.pooled_w);

                    if (in_y < 0 || in_y > jpp.ih - 1 || in_x < 0 || in_x > jpp.iw - 1) {
                        arg.bin_area = 0;
                        arg.dst = &dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]];
                    } else {
                        int top_y_index    = static_cast<int>(floorf(in_y));
                        int bottom_y_index = static_cast<int>(ceilf(in_y));
                        int left_x_index   = static_cast<int>(floorf(in_x));
                        int right_x_index  = static_cast<int>(ceilf(in_x));

                        if (right_x_index > jpp.iw - 1)
                            right_x_index = jpp.iw - 1;

                        if (bottom_y_index > jpp.ih - 1)
                            bottom_y_index = jpp.ih - 1;

                        arg.dst = &dst[n * dst_strides[0] + cb * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3]];

                        arg.xf = in_x - left_x_index;
                        arg.yf = in_y - top_y_index;

                        arg.xoff = sizeof(T) * (right_x_index - left_x_index) * jpp.c_block;
                        arg.yoff = sizeof(T) * (bottom_y_index - top_y_index) * jpp.iw * jpp.c_block;

                        arg.src = &src_data[roi_batch_ind * src_strides[0] + cb * src_strides[1] +
                                            top_y_index * src_strides[2] + left_x_index * src_strides[3]];

                        arg.bin_area = 1;
                    }
                }

                (*roi_pooling_kernel)(&arg);
            }
        });
    }

    std::shared_ptr<jit_uni_roi_pooling_kernel> roi_pooling_kernel;
};

template <typename T>
class ROIPooling::ROIPoolingRefExecutor : public ROIPooling::ROIPoolingExecutor {
public:
    ROIPoolingRefExecutor(const jit_roi_pooling_params &_jpp) : jpp(_jpp) {}
    void exec(
        const IMemory& srcData,
        const IMemory& srcRoi,
        const IMemory& dst) override {
        auto src_strides = srcData.getDescWithType<BlockedMemoryDesc>()->getStrides();
        auto src_roi_step = srcRoi.getDescWithType<BlockedMemoryDesc>()->getStrides()[0];
        auto dst_strides = dst.getDescWithType<BlockedMemoryDesc>()->getStrides();
        const auto* src_ptr = srcData.getDataAs<const T>();
        const auto* roi_ptr = srcRoi.getDataAs<const T>();
        auto* dst_ptr = dst.getDataAs<T>();
        executeReference(src_ptr, roi_ptr, dst_ptr, src_strides, dst_strides, src_roi_step);
    }

    void executeReference(
        const T* src_data,
        const T* src_roi,
        T* dst,
        const VectorDims& src_strides,
        const VectorDims& dst_strides,
        const size_t src_roi_step) {
        int cb_work = impl::utils::div_up(jpp.nb_c, jpp.nb_c_blocking);
        int MB = jpp.mb;

        int real_rois = 0;
        for (; real_rois < MB; real_rois++) {
            size_t roi_off = real_rois * src_roi_step;

            const auto *src_roi_ptr = &src_roi[roi_off];
            int roi_batch_ind = static_cast<int>(src_roi_ptr[0]);
            if (roi_batch_ind == -1) {
                break;
            }
        }

        parallel_for4d(MB, cb_work, jpp.oh, jpp.ow, [&](int n, int cbb, int oh, int ow) {
            int cb_num = jpp.nb_c_blocking;
            int c_block = jpp.c_block;

            if (n >= real_rois) {
                for (int cbb_cur = 0; cbb_cur < cb_num; cbb_cur++) {
                    int ch_blk_cur = cbb * cb_num + cbb_cur;
                    if (ch_blk_cur >= jpp.nb_c) {
                        break; // current block work is done
                    }
                    for (int c = 0; c < c_block; c++) {
                        dst[n * dst_strides[0] + ch_blk_cur * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3] + c] = 0;
                    }
                }
            } else {
                size_t roi_off = n * src_roi_step;
                const auto *src_roi_ptr = &src_roi[roi_off];

                int roi_batch_ind = static_cast<int>(src_roi_ptr[0]);

                if (jpp.alg == Algorithm::ROIPoolingMax) {
                    int roi_start_w = static_cast<int>(round(src_roi_ptr[1] * jpp.spatial_scale));
                    int roi_start_h = static_cast<int>(round(src_roi_ptr[2] * jpp.spatial_scale));
                    int roi_end_w = static_cast<int>(round(src_roi_ptr[3] * jpp.spatial_scale));
                    int roi_end_h = static_cast<int>(round(src_roi_ptr[4] * jpp.spatial_scale));

                    int hstart, hend, wstart, wend;
                    std::tie(hstart, hend, wstart, wend) = getBordersForMaxMode(
                        roi_start_h, roi_end_h, roi_start_w, roi_end_w, jpp.ih, oh, jpp.iw, ow, jpp.pooled_h, jpp.pooled_w);

                    for (int cbb_cur = 0; cbb_cur < cb_num; cbb_cur++) {
                        int ch_blk_cur = cbb * cb_num + cbb_cur;
                        if (ch_blk_cur >= jpp.nb_c) {
                            break;  // current block work is done
                        }
                        for (int c = 0; c < c_block; c++) {
                            const size_t pool_index = n * dst_strides[0] + ch_blk_cur * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3] + c;
                            if ((hend <= hstart) || (wend <= wstart)) {
                                dst[pool_index] = 0;
                            } else {
                                dst[pool_index] =  src_data[roi_batch_ind * src_strides[0] + ch_blk_cur * src_strides[1] +
                                                            hstart * src_strides[2] + wstart * src_strides[3] + c];
                                for (int h = hstart; h < hend; ++h) {
                                    for (int w = wstart; w < wend; ++w) {
                                        float batch_data = src_data[roi_batch_ind * src_strides[0] + ch_blk_cur * src_strides[1] +
                                                                    h * src_strides[2] + w * src_strides[3] + c];
                                        dst[pool_index] = std::fmax(batch_data, dst[pool_index]);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    float roi_start_w_ = src_roi_ptr[1];
                    float roi_start_h_ = src_roi_ptr[2];
                    float roi_end_w_   = src_roi_ptr[3];
                    float roi_end_h_   = src_roi_ptr[4];

                    float in_x, in_y;
                    std::tie(in_x, in_y) = getXYForBilinearMode(
                        roi_start_h_, roi_end_h_, roi_start_w_, roi_end_w_, jpp.ih, oh, jpp.iw, ow, jpp.pooled_h, jpp.pooled_w);

                    if (in_y < 0 || in_y > jpp.ih - 1 || in_x < 0 || in_x > jpp.iw - 1) {
                        for (int cbb_cur = 0; cbb_cur < cb_num; cbb_cur++) {
                            int ch_blk_cur = cbb * cb_num + cbb_cur;
                            if (ch_blk_cur >= jpp.nb_c) {
                                break;  // current block work is done
                            }
                            for (int c = 0; c < c_block; c++) {
                                dst[n * dst_strides[0] + ch_blk_cur * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3] + c] = 0;
                            }
                        }
                    } else {
                        int top_y_index    = static_cast<int>(floorf(in_y));
                        int bottom_y_index = static_cast<int>(ceilf(in_y));
                        int left_x_index   = static_cast<int>(floorf(in_x));
                        int right_x_index  = static_cast<int>(ceilf(in_x));

                        if (right_x_index > jpp.iw - 1)
                            right_x_index = jpp.iw - 1;

                        if (bottom_y_index > jpp.ih - 1)
                            bottom_y_index = jpp.ih - 1;

                        for (int cbb_cur = 0; cbb_cur < cb_num; cbb_cur++) {
                            int ch_blk_cur = cbb * cb_num + cbb_cur;
                            if (ch_blk_cur >= jpp.nb_c) {
                                break;  // current block work is done
                            }
                            for (int c = 0; c < c_block; c++) {
                                const float top_left     = src_data[roi_batch_ind * src_strides[0] + ch_blk_cur * src_strides[1] +
                                                                    top_y_index * src_strides[2] + left_x_index * src_strides[3] + c];
                                const float top_right    = src_data[roi_batch_ind * src_strides[0] + ch_blk_cur * src_strides[1] +
                                                                    top_y_index * src_strides[2] + right_x_index * src_strides[3] + c];
                                const float bottom_left  = src_data[roi_batch_ind * src_strides[0] + ch_blk_cur * src_strides[1] +
                                                                    bottom_y_index * src_strides[2] + left_x_index * src_strides[3] + c];
                                const float bottom_right = src_data[roi_batch_ind * src_strides[0] + ch_blk_cur * src_strides[1] +
                                                                    bottom_y_index * src_strides[2] + right_x_index * src_strides[3] + c];

                                const float top    = top_left + (top_right - top_left) * (in_x - left_x_index);
                                const float bottom = bottom_left + (bottom_right - bottom_left) * (in_x - left_x_index);

                                dst[n * dst_strides[0] + ch_blk_cur * dst_strides[1] + oh * dst_strides[2] + ow * dst_strides[3] + c] =
                                        top + (bottom - top) * (in_y - top_y_index);
                            }
                        }
                    }
                }
            }
        });
    }

private:
    jit_roi_pooling_params jpp;
};

std::shared_ptr<ROIPooling::ROIPoolingExecutor> ROIPooling::ROIPoolingExecutor::createROIPoolingNewExecutor(
    const jit_roi_pooling_params& jpp) {
    ROIPoolingContext ctx = { nullptr, jpp };

    OV_SWITCH(intel_cpu, ROIPoolingExecutorCreation, ctx, jpp.src_prc,
              OV_CASE(ov::element::f32, float),
              OV_CASE(ov::element::bf16, bfloat16_t),
              OV_CASE(ov::element::f16, dnnl::impl::float16_t))

    return ctx.executor;
}

std::tuple<int, int, int, int> ROIPooling::ROIPoolingExecutor::getBordersForMaxMode(
    const int roi_start_h, const int roi_end_h, const int roi_start_w, const int roi_end_w,
    const int ih, const int oh, const int iw, const int ow, const int pooled_h, const int pooled_w) {
    int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);

    int hstart = (oh * roi_height) / pooled_h;
    if ((hstart * pooled_h) > (oh * roi_height)) {
        --hstart;
    }

    int wstart = (ow * roi_width) / pooled_w;
    if ((wstart * pooled_w) > (ow * roi_width)) {
        --wstart;
    }

    int hend = ((oh + 1) * roi_height) / pooled_h;
    if ((hend * pooled_h) < ((oh + 1) * roi_height)) {
        ++hend;
    }

    int wend = ((ow + 1) * roi_width) / pooled_w;
    if ((wend * pooled_w) < ((ow + 1) * roi_width)) {
        ++wend;
    }

    hstart = std::min(std::max(hstart + roi_start_h, 0), ih);
    hend = std::min(std::max(hend + roi_start_h, 0), ih);
    wstart = std::min(std::max(wstart + roi_start_w, 0), iw);
    wend = std::min(std::max(wend + roi_start_w, 0), iw);
    return std::make_tuple(hstart, hend, wstart, wend);
}

std::pair<float, float> ROIPooling::ROIPoolingExecutor::getXYForBilinearMode(
    const float roi_start_h, const float roi_end_h, const float roi_start_w, const float roi_end_w,
    const int ih, const int oh, const int iw, const int ow, const int pooled_h, const int pooled_w) {
    float height_scale = (pooled_h > 1 ? ((roi_end_h - roi_start_h) * (ih - 1)) / (pooled_h - 1) : 0);
    float width_scale  = (pooled_w > 1 ? ((roi_end_w - roi_start_w) * (iw - 1)) / (pooled_w - 1) : 0);

    float in_y, in_x;
    // because of nonalgebraic character of floating point operation, some proposals can cause violation of inequality:
    // ((end_h - start_h) * (input_h - 1) / (pooled_h - 1)) * (pooled_h - 1) <= (end_h - start_h) * (input_h - 1),
    // and as result excess of right limit for proposal value,
    // if the border case (current_h == pooled_h - 1) will not be handled explicitly
    if (pooled_h > 1) {
        in_y = (oh == pooled_h - 1 ? roi_end_h * (ih - 1) : (oh * height_scale + roi_start_h * (ih - 1)));
    } else {
        in_y = 0.5 * (roi_start_h + roi_end_h) * (ih - 1);
    }
    if (pooled_w > 1) {
        in_x = (ow == pooled_w - 1 ? roi_end_w * (iw - 1) : (ow * width_scale  + roi_start_w * (iw - 1)));
    } else {
        in_x = 0.5 * (roi_start_w + roi_end_w) * (iw - 1);
    }

    return std::make_pair(in_x, in_y);
}

template <typename T>
std::shared_ptr<ROIPooling::ROIPoolingExecutor> ROIPooling::ROIPoolingExecutor::makeExecutor(
    const jit_roi_pooling_params& jpp) {
#if defined(OPENVINO_ARCH_X86_64)
    if (mayiuse(cpu::x64::sse41))
        return std::make_shared<ROIPoolingJitExecutor<T>>(jpp);
#endif

    return std::make_shared<ROIPoolingRefExecutor<T>>(jpp);
}

bool ROIPooling::created() const {
    return getType() == Type::ROIPooling;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

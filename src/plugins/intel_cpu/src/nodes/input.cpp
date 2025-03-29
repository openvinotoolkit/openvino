// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input.h"

#include "cpu/x64/jit_generator.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/node_config.h"
#include "openvino/core/parallel.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "shape_inference/shape_inference_pass_through.hpp"
#include "transformations/cpu_opset/common/op/read_value_with_subgraph.hpp"

using namespace dnnl;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov::intel_cpu::node {

#if defined(OPENVINO_ARCH_X86_64)
namespace {
struct jit_has_special_value_base : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_has_special_value_base)

    using args_t = struct {
        const float* src;
        const size_t count;
        bool hasTargetValues;
    };

    using fn_t = void (*)(const args_t*);

    jit_has_special_value_base() : jit_generator(jit_name()) {
        jit_ker_ = nullptr;
    }

    fn_t get() {
        return jit_ker() || create_kernel() == dnnl::impl::status::success ? (fn_t)jit_ker() : nullptr;
    }

protected:
    void foreach (const Xbyak::Reg64& idx,
                  size_t step,
                  const Xbyak::Reg64& end,
                  std::function<void(const Xbyak::Reg64&)> && fn) {
        Label loop, exit;

        L(loop);
        cmp(idx, end);
        jge(exit);

        fn(idx);

        add(idx, step);
        jmp(loop);
        L(exit);
    }

    void copy_floats(const Xbyak::Reg64& dst, const Xbyak::Reg64& src, const Xbyak::Reg64& size) {
        push(rsi);
        push(r15);

        xor_(rsi, rsi);

        foreach (rsi, 1, size, [&, this](const Xbyak::Reg64& idx) {
            mov(r15d, dword[src + idx * sizeof(float)]);
            mov(dword[dst + idx * sizeof(float)], r15d);
        })
            ;

        pop(r15);
        pop(rsi);
    }

    void check_subnormals(const Xbyak::Reg64& src,
                          const Xbyak::Ymm& exponent_mask,
                          const Xbyak::Ymm& mantissa_mask,
                          const Xbyak::Ymm& zero) {
        auto a = ymm1;
        auto b = ymm2;
        auto c = ymm3;

        vmovdqu(a, yword[src]);      // load 8 floats
        vpand(b, a, mantissa_mask);  // b = a & 00000000011111111111111111111111
        vpcmpeqd(b, b, zero);        // if (b == 0) b = 1 else b = 0
        vpand(c, a, exponent_mask);  // c = a & 01111111100000000000000000000000
        vpcmpeqd(c, c, zero);        // if (c == 0) c = 1 else c = 0
        vptest(b, c);                // if ((!b & c) == 0) CF = 1 else CF = 0
    }

    void check_subnormals(const Xbyak::Reg64& src,
                          const Xbyak::Xmm& exponent_mask,
                          const Xbyak::Xmm& mantissa_mask,
                          const Xbyak::Xmm& zero) {
        auto a = xmm1;
        auto b = xmm2;
        auto c = xmm3;

        uni_vmovdqu(a, xword[src]);      // load 4 floats
        uni_vmovdqu(b, a);               // b = a
        uni_vmovdqu(c, a);               // c = a
        uni_vpand(b, b, mantissa_mask);  // b = a & 00000000011111111111111111111111
        uni_vpcmpeqd(b, b, zero);        // if (b == 0) b = 1 else b = 0
        uni_vpand(c, c, exponent_mask);  // c = a & 01111111100000000000000000000000
        uni_vpcmpeqd(c, c, zero);        // if (c == 0) c = 1 else c = 0
        uni_vtestps(b, c);               // if ((!b & c) == 0) CF = 1 else CF = 0
    }

    void check_bf16_saturations(const Xbyak::Reg64& src,
                                const Xbyak::Ymm& bf16_max_mask,
                                const Xbyak::Ymm& bf16_min_mask) {
        auto a = ymm1;
        auto b = ymm2;
        auto c = ymm3;
        vmovdqu(a, yword[src]);             // load 8 floats
        vcmpps(b, a, bf16_max_mask, 0x1e);  // b = (a > bf16_max) ? 1 : 0
        vcmpps(c, a, bf16_min_mask, 0x11);  // c = (a < bf16_min) ? 1 : 0
        vorps(b, b, c);                     // b = b | c
        vptest(b, b);                       // if (b != 0) CF = 1 else CF = 0
    }

    void check_bf16_saturations(const Xbyak::Reg64& src,
                                const Xbyak::Xmm& bf16_max_mask,
                                const Xbyak::Xmm& bf16_min_mask) {
        auto a = xmm1;
        auto b = xmm2;
        auto c = xmm3;

        uni_vmovdqu(a, xword[src]);             // load 4 floats
        uni_vcmpps(b, a, bf16_max_mask, 0x1e);  // b = (a > bf16_max) ? 1 : 0
        uni_vcmpps(c, a, bf16_max_mask, 0x11);  // c = (a < bf16_min) ? 1 : 0
        uni_vorps(b, b, c);                     // b = b | c
        uni_vtestps(b, b);                      // if (b != 0) CF = 1 else CF = 0
    }

protected:
    Label exit, has_target_values, no_target_values;

    const Reg64& reg_src = rax;
    const Reg64& reg_dst = rbx;
    const Reg64& reg_sz = rdx;
    const Reg64& reg_idx = rsi;
    const Reg64& reg_mask_addr = r15;

    static const uint32_t exponent_mask_data[8];
    static const uint32_t mantissa_mask_data[8];
    static const float bf16_max_mask_data[8];
    static const float bf16_min_mask_data[8];
};

const uint32_t jit_has_special_value_base::exponent_mask_data[8] =
    {0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000};

const uint32_t jit_has_special_value_base::mantissa_mask_data[8] =
    {0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff};

const float jit_has_special_value_base::bf16_max_mask_data[8] = {std::numeric_limits<ov::bfloat16>::max(),
                                                                 std::numeric_limits<ov::bfloat16>::max(),
                                                                 std::numeric_limits<ov::bfloat16>::max(),
                                                                 std::numeric_limits<ov::bfloat16>::max(),
                                                                 std::numeric_limits<ov::bfloat16>::max(),
                                                                 std::numeric_limits<ov::bfloat16>::max(),
                                                                 std::numeric_limits<ov::bfloat16>::max(),
                                                                 std::numeric_limits<ov::bfloat16>::max()};

const float jit_has_special_value_base::bf16_min_mask_data[8] = {std::numeric_limits<ov::bfloat16>::lowest(),
                                                                 std::numeric_limits<ov::bfloat16>::lowest(),
                                                                 std::numeric_limits<ov::bfloat16>::lowest(),
                                                                 std::numeric_limits<ov::bfloat16>::lowest(),
                                                                 std::numeric_limits<ov::bfloat16>::lowest(),
                                                                 std::numeric_limits<ov::bfloat16>::lowest(),
                                                                 std::numeric_limits<ov::bfloat16>::lowest(),
                                                                 std::numeric_limits<ov::bfloat16>::lowest()};
template <cpu_isa_t isa>
struct jit_has_subnormals : public jit_has_special_value_base {
    using Vmm = typename dnnl::impl::utils::conditional<isa == sse41, Xbyak::Xmm, Xbyak::Ymm>::type;

    const Vmm rmm4 = Vmm(4);
    const Vmm rmm5 = Vmm(5);
    const Vmm rmm6 = Vmm(6);
    const int length = isa == sse41 ? 4 : 8;

    void generate() override final {
        size_t const vlen = length;
        const int sh_bits = std::ilogb(vlen);

        auto zero = rmm4;
        auto exponent_mask = rmm5;
        auto mantissa_mask = rmm6;

        preamble();

        // Get arguments addresses
        mov(reg_src, ptr[param1 + offsetof(args_t, src)]);
        lea(reg_dst, ptr[param1 + offsetof(args_t, hasTargetValues)]);
        mov(reg_sz, ptr[param1 + offsetof(args_t, count)]);

        // Initialize necessary consts
        uni_vpxor(zero, zero, zero);
        mov(reg_mask_addr, (size_t)exponent_mask_data);
        uni_vmovdqu(exponent_mask, ptr[reg_mask_addr]);
        mov(reg_mask_addr, (size_t)mantissa_mask_data);
        uni_vmovdqu(mantissa_mask, ptr[reg_mask_addr]);

        // Main loop
        xor_(reg_idx, reg_idx);
        mov(r8, reg_sz);
        shr(r8, sh_bits);

        foreach (reg_idx, 1, r8, [&, this](const Xbyak::Reg64& idx) {
            check_subnormals(reg_src, exponent_mask, mantissa_mask, zero);
            jnc(has_target_values);
            add(reg_src, sizeof(float) * vlen);
        })
            ;

        // Tail
        shl(reg_idx, sh_bits);
        sub(reg_sz, reg_idx);
        test(reg_sz, reg_sz);
        jz(exit);

        // use space on stack for 4 or 8 floats
        sub(rsp, vlen * sizeof(float));
        mov(r8, rsp);

        uni_vmovdqu(ptr[r8], zero);

        copy_floats(r8, reg_src, reg_sz);
        check_subnormals(r8, exponent_mask, mantissa_mask, zero);
        jc(no_target_values);
        add(rsp, vlen * sizeof(float));

        L(has_target_values);

        mov(rax, 1);
        mov(byte[reg_dst], al);
        jmp(exit);

        L(no_target_values);
        add(rsp, vlen * sizeof(float));

        L(exit);

        postamble();
    }
};
template <cpu_isa_t isa>
struct jit_has_bf16_overflows : public jit_has_special_value_base {
    using Vmm = typename dnnl::impl::utils::conditional<isa == sse41, Xbyak::Xmm, Xbyak::Ymm>::type;

    const Vmm rmm4 = Vmm(4);
    const Vmm rmm5 = Vmm(5);
    const Vmm rmm6 = Vmm(6);
    const int length = isa == sse41 ? 4 : 8;

    void generate() override final {
        size_t const vlen = length;
        const int sh_bits = std::ilogb(vlen);

        auto zero = rmm4;
        auto bf16_max_mask = rmm5;
        auto bf16_min_mask = rmm6;

        preamble();

        // Get arguments addresses
        mov(reg_src, ptr[param1 + offsetof(args_t, src)]);
        lea(reg_dst, ptr[param1 + offsetof(args_t, hasTargetValues)]);
        mov(reg_sz, ptr[param1 + offsetof(args_t, count)]);

        // Initialize necessary consts
        uni_vpxor(zero, zero, zero);
        mov(reg_mask_addr, (size_t)bf16_max_mask_data);
        uni_vmovdqu(bf16_max_mask, ptr[reg_mask_addr]);
        mov(reg_mask_addr, (size_t)bf16_min_mask_data);
        uni_vmovdqu(bf16_min_mask, ptr[reg_mask_addr]);

        // Main loop
        xor_(reg_idx, reg_idx);
        mov(r8, reg_sz);
        shr(r8, sh_bits);

        foreach (reg_idx, 1, r8, [&, this](const Xbyak::Reg64& idx) {
            check_bf16_saturations(reg_src, bf16_max_mask, bf16_min_mask);
            jnz(has_target_values, T_NEAR);
            add(reg_src, sizeof(float) * vlen);
        })
            ;

        // Tail
        shl(reg_idx, sh_bits);
        sub(reg_sz, reg_idx);
        test(reg_sz, reg_sz);
        jz(exit);

        // use space on stack for 4 or 8 floats
        sub(rsp, vlen * sizeof(float));
        mov(r8, rsp);

        uni_vmovdqu(ptr[r8], zero);

        copy_floats(r8, reg_src, reg_sz);
        check_bf16_saturations(r8, bf16_max_mask, bf16_min_mask);
        jz(no_target_values, T_NEAR);
        add(rsp, vlen * sizeof(float));

        L(has_target_values);

        mov(rax, 1);
        mov(byte[reg_dst], al);
        jmp(exit);

        L(no_target_values);
        add(rsp, vlen * sizeof(float));

        L(exit);

        postamble();
    }
};
jit_has_special_value_base::fn_t jit_has_subnormals_function() {
    if (mayiuse(cpu_isa_t::avx2)) {
        static jit_has_subnormals<cpu_isa_t::avx2> generator;
        static auto fn = generator.get();
        return fn;
    }
    if (mayiuse(cpu_isa_t::sse41)) {
        static jit_has_subnormals<cpu_isa_t::sse41> generator;
        static auto fn = generator.get();
        return fn;
    }
    return nullptr;
}
jit_has_special_value_base::fn_t jit_has_bf16_overflows_function() {
    if (mayiuse(cpu_isa_t::avx2)) {
        static jit_has_bf16_overflows<cpu_isa_t::avx2> generator;
        static auto fn = generator.get();
        return fn;
    }
    if (mayiuse(cpu_isa_t::sse41)) {
        static jit_has_bf16_overflows<cpu_isa_t::sse41> generator;
        static auto fn = generator.get();
        return fn;
    }
    return nullptr;
}

}  // namespace
#endif

Input::Input(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, PassThroughShapeInferFactory()) {
    if (!one_of(op->get_type_info(),
                op::v0::Parameter::get_type_info_static(),
                op::v0::Constant::get_type_info_static(),
                op::v0::Result::get_type_info_static(),
                op::v3::ReadValue::get_type_info_static(),
                op::v6::ReadValue::get_type_info_static(),
                ov::intel_cpu::ReadValueWithSubgraph::get_type_info_static())) {
        OPENVINO_THROW_NOT_IMPLEMENTED("CPU Input node doesn't support ngraph operation ",
                                       op->get_type_name(),
                                       " with name ",
                                       op->get_friendly_name());
    }
    if (auto constOp = ov::as_type_ptr<op::v0::Constant>(op)) {
        constant = ConstantType::Const;
        m_constOp = constOp;
        cloneBlobIfRequired();
    } else {
        constant = ConstantType::StrictNoConst;
    }
}

void Input::cloneBlobIfRequired() {
    const auto prec = m_constOp->get_element_type();
    if (prec == ov::element::dynamic && shape_size(m_constOp->get_shape()) == 0) {
        memoryPtr = MemoryDescUtils::makeEmptyMemory(context);
        return;
    }

    Shape shape(m_constOp->get_shape().empty() ? ov::Shape(1, 1) : m_constOp->get_shape());
    const size_t size = shape.getElementsCount();
    CpuBlockedMemoryDesc memDesc(prec, shape);

    bool needFlushDenormalsToZero = true;
    if (context->getConfig().DAZOn) {
        // DAZ has been set, processor automatically converts all denormal source operands
        // to a zero with the sign of the original operand before performing any
        // computations on them, thus no need to flush them to zero manually
        needFlushDenormalsToZero = false;
    }

    // The presence of subnormals is better to determined at IR read time.
    auto checkSubnormalsAndBF16Overflows = [&](bool& has_subnormals, bool& has_bf16_overflows) {
        if (prec == ov::element::f32) {
            auto const* u32data = m_constOp->get_data_ptr<uint32_t>();
            auto const* f32data = m_constOp->get_data_ptr<float>();

            if (!size) {
                return;
            }
            // Only bf16 inferencePrecision cases need to be checked for saturation
            const bool do_bf16_saturation_check =
                (context->getConfig().inferencePrecision == ov::element::bf16) ? true : false;

#if defined(OPENVINO_ARCH_X86_64)
            auto fn = jit_has_subnormals_function();
            auto fn_bf16_check = jit_has_bf16_overflows_function();
            if (fn && fn_bf16_check) {
                static const size_t batch_size = 2048;
                const size_t iterations_num = size / batch_size + 1;

                std::atomic<bool> has_subnormals_local(false);
                std::atomic<bool> has_bf16_overflows_local(false);
                if (needFlushDenormalsToZero || do_bf16_saturation_check) {
                    parallel_for(iterations_num, [&](int n) {
                        auto ptr = u32data + n * batch_size;
                        jit_has_special_value_base::args_t args = {reinterpret_cast<float const*>(ptr),
                                                                   std::min(batch_size, (size_t)(u32data + size - ptr)),
                                                                   false};

                        if (needFlushDenormalsToZero && !has_subnormals_local) {
                            fn(&args);
                            if (args.hasTargetValues) {
                                has_subnormals_local = true;
                            }
                        }

                        if (do_bf16_saturation_check && !has_bf16_overflows_local) {
                            // batch_size is small enough, so source data are still cache-hot
                            args.hasTargetValues = false;
                            fn_bf16_check(&args);
                            if (args.hasTargetValues) {
                                has_bf16_overflows_local = true;
                            }
                        }
                    });
                }

                has_subnormals = has_subnormals_local;
                has_bf16_overflows = has_bf16_overflows_local;

                return;
            }
#endif

            uint32_t mantissaMask = 0x007fffff;
            uint32_t exponentMask = 0x7f800000;
            const float bf16_max = std::numeric_limits<ov::bfloat16>::max();
            for (size_t i = 0; i < size; ++i) {
                if (needFlushDenormalsToZero && (u32data[i] & exponentMask) == 0 && (u32data[i] & mantissaMask) != 0) {
                    has_subnormals = true;
                }

                if (do_bf16_saturation_check && (f32data[i] < -bf16_max || f32data[i] > bf16_max)) {
                    has_bf16_overflows = true;
                }

                if ((!needFlushDenormalsToZero || has_subnormals) &&
                    (!do_bf16_saturation_check || has_bf16_overflows)) {
                    return;
                }
            }
        }
    };

    bool has_subnormals = false;
    bool has_bf16_overflows = false;

    checkSubnormalsAndBF16Overflows(has_subnormals, has_bf16_overflows);

    auto cloneBlob = [&, this]() {
        MemoryPtr memory;

        // CVS-74980
        // oneDNN always allocate 1byte for element type with bitWidth < 8 (u4,u1...)
        // but ngraph Constant uses actual bitWidth for data storage allocation
        // in that case we make a copy to avoid overflow
        if (m_constOp->get_byte_size() >= memDesc.getCurrentMemSize()) {
            if (m_constOp->get_element_type() == element::string) {
                memory =
                    std::make_shared<StringMemory>(getEngine(), memDesc, m_constOp->get_data_ptr<element::string>());
            } else {
                memory = std::make_shared<Memory>(getEngine(), memDesc, m_constOp->get_data_ptr());
            }
        } else {
            if (m_constOp->get_element_type() == element::string) {
                memory = std::make_shared<StringMemory>(getEngine(), memDesc);
                auto src = m_constOp->get_data_ptr<StringMemory::OvString>();
                auto dst = memory->getDataAs<StringMemory::OvString>();
                std::copy(src, src + size, dst);
            } else {
                memory = std::make_shared<Memory>(getEngine(), memDesc);
                memcpy(memory->getData(), m_constOp->get_data_ptr(), m_constOp->get_byte_size());
            }
        }

        MemoryPtr ptr;
        if (memDesc.getPrecision() == element::string) {
            ptr = std::make_shared<StringMemory>(getEngine(), memDesc);
        } else {
            ptr = std::make_shared<StaticMemory>(getEngine(), memDesc);
        }
        ptr->load(*memory.get(), has_subnormals, has_bf16_overflows);

        return ptr;
    };

    auto isBlobAligned = [](const std::shared_ptr<ov::op::v0::Constant>& constant) {
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
        // Majority of arithmetic and data processing instructions in legacy SSE isa requires
        // the memory address in the operands must be aligned on 16-byte boundary. To ensure
        // safely reusing ngraph const blob memory, need to check address alignment.
        const void* ptr = constant->get_data_ptr();
        return mayiuse(cpu_isa_t::avx2) || ((reinterpret_cast<uintptr_t>(ptr) & 15) == 0);
#else
        return true;
#endif
    };

    auto blobKey = [&]() {
        char ptr[32];
        snprintf(ptr, sizeof ptr, "%p", m_constOp->get_data_ptr());
        return getName() + "_" + std::to_string(size * prec.size()) + "_" + ptr;
    };

    const auto weightCache = context->getWeightsCache();
    const bool clone_is_not_needed =
        prec != element::string &&
        // IRs already have all subnormals flushed to zero, but in
        // read_model scenario with directly loaded original model still can have subnormals
        isBlobAligned(m_constOp) && !has_subnormals && !has_bf16_overflows &&
        // Blob should be cloned in cache only if original weights are stored on other numa node.
        // This is possible only in multistream case on multisocket machine.
        // TODO: don't clone blob for multisocket + multistream case if current stream is run on the numa node where
        // original weights are stored.
        (!weightCache || context->getNumNumaNodes() == 1 || context->getCPUStreamExecutor()->get_streams_num() == 1);

    memoryPtr = clone_is_not_needed ? std::make_shared<Memory>(getEngine(), memDesc, m_constOp->get_data_ptr())
                                    : std::const_pointer_cast<const IMemory>(
                                          weightCache ? *weightCache->findOrCreate(blobKey(), cloneBlob) : cloneBlob());
}

static std::vector<Shape> createInputShapes(const Shape& shape, const Type type) {
    if (type == Type::Output) {
        return {shape};
    }
    return {};
}

static std::vector<Shape> createOutputShapes(const Shape& shape, const Type type) {
    if (type == Type::Input) {
        return {shape};
    }
    return {};
}

static std::vector<ov::element::Type> createInputPrecisions(const ov::element::Type& prc, const Type type) {
    if (type == Type::Output) {
        return {prc};
    }
    return {};
}

static std::vector<ov::element::Type> createOutputPrecisions(const ov::element::Type& prc, const Type type) {
    if (type == Type::Input) {
        return {prc};
    }
    return {};
}

Input::Input(const Shape& shape,
             const ov::element::Type& prc,
             const std::string& name,
             const std::string& type,
             const GraphContext::CPtr& context)
    : Node(type,
           createInputShapes(shape, TypeFromName(type)),
           createOutputShapes(shape, TypeFromName(type)),
           createInputPrecisions(prc, TypeFromName(type)),
           createOutputPrecisions(prc, TypeFromName(type)),
           name,
           context) {
    constant = ConstantType::NoConst;
    isDynamic = shape.isDynamic();
    if (isDynamic) {
        shapeInference = PassThroughShapeInferFactory().makeShapeInfer();
    }
}

Input::Input(const MemoryDescPtr& memDesc,
             const std::string& name,
             const std::string& type,
             const GraphContext::CPtr& context)
    : Input(memDesc->getShape(), memDesc->getPrecision(), name, type, context) {
    extMemDesc = memDesc;
}

Input::Input(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context, const InputConfig& config)
    : Input(op, context) {
    extMemDesc = config.desc;
    m_isInPlace = config.inPlace;
}

Input::Input(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context, const OutputConfig& config)
    : Input(op, context) {
    extMemDesc = config.desc;
    m_useParentMemoryDescForOutput = config.useParentMemoryDescForOutput;
    m_isInPlace = config.inPlace;
}

MemoryCPtr Input::getMemoryPtr() const {
    return memoryPtr;
}

void Input::getSupportedDescriptors() {
    if (getType() == Type::Input) {
        if (!getParentEdges().empty()) {
            THROW_CPU_NODE_ERR("has incorrect number of input edges.");
        }
        if (getChildEdges().empty()) {
            THROW_CPU_NODE_ERR("has incorrect number of output edges.");
        }
    } else if (getType() == Type::Output) {
        if (getParentEdges().size() != 1) {
            THROW_CPU_NODE_ERR("has incorrect number of input edges.");
        }
        if (!getChildEdges().empty()) {
            THROW_CPU_NODE_ERR("has incorrect number of output edges.");
        }
    }
}

void Input::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty()) {
        return;
    }

    if (extMemDesc) {
        initSupportedPdFromMemDesc();
    } else {
        initSupportedPdDefault();
    }
}

void Input::initOptimalPrimitiveDescriptor() {
    if (m_useParentMemoryDescForOutput || extMemDesc) {
        return;
    }

    Node::initOptimalPrimitiveDescriptor();
}

void Input::selectOptimalPrimitiveDescriptor() {
    if (!(m_useParentMemoryDescForOutput && getType() == Type::Output)) {
        return Node::selectOptimalPrimitiveDescriptor();
    }

    // ignore previous configuration
    supportedPrimitiveDescriptors.clear();

    const int inPlacePort = m_isInPlace ? 0 : -1;
    // and just use parent memory descriptor for Output node to avoid reorders insertion
    std::vector<PortConfig> inConfs;
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        inConfs.push_back(
            {PortConfig(getParentOutputMemDesc(getParentEdgeAt(0)), BlockedMemoryDesc::FULL_MASK, inPlacePort)});
    }
    NodeConfig config(inConfs, {});

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    selectPrimitiveDescriptorByIndex(0);
}

void Input::createPrimitive() {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto dstMemPtr = getDstMemoryAtPort(i);
        if (!dstMemPtr) {
            THROW_CPU_NODE_ERR("has null memory object at port ",
                               i,
                               " to node ",
                               getChildEdgeAt(i)->getChild()->getName(),
                               ".");
        }
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto srcMemPtr = getSrcMemoryAtPort(i);
        if (!srcMemPtr) {
            THROW_CPU_NODE_ERR("has null memory object at port ",
                               i,
                               " from node ",
                               getParentEdgeAt(i)->getParent()->getName(),
                               ".");
        }
    }

    const NodeDesc* selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr) {
        THROW_CPU_NODE_ERR("doesn't have selected primitive descriptor.");
    }
}

bool Input::created() const {
    return getType() == Type::Input || getType() == Type::Output;
}

void Input::initSupportedPdDefault() {
    std::vector<PortConfigurator> inPortConfs;
    std::vector<PortConfigurator> outPortConfs;

    if (getType() == Type::Input || getType() == Type::MemoryInput) {
        auto precision = getOriginalOutputPrecisionAtPort(0);

        outPortConfs.emplace_back(LayoutType::ncsp, precision);
        if (!getParentEdges().empty()) {
            inPortConfs.emplace_back(LayoutType::ncsp, precision, true);
        }
    } else if (getType() == Type::Output) {
        auto precision = getOriginalInputPrecisionAtPort(0);

        inPortConfs.emplace_back(LayoutType::ncsp, precision);
    }

    addSupportedPrimDesc(inPortConfs, outPortConfs, impl_desc_type::unknown);
}

void Input::initSupportedPdFromMemDesc() {
    NodeConfig config;
    PortConfig portConfig(extMemDesc, BlockedMemoryDesc::FULL_MASK, m_isInPlace ? 0 : -1, false);

    if (getType() == Type::Input || getType() == Type::MemoryInput) {
        config.outConfs.push_back(portConfig);
    } else if (getType() == Type::Output) {
        config.inConfs.push_back(portConfig);
    }

    supportedPrimitiveDescriptors.emplace_back(std::move(config), impl_desc_type::unknown);
}

void Input::resolveInPlaceEdges(Edge::LOOK look) {
    if (!m_isInPlace) {
        return Node::resolveInPlaceEdges(look);
    }

    if (look & Edge::LOOK_UP) {
        auto edges = getChildEdgesAtPort(0);
        for (const auto& edge : edges) {
            EdgePtr sharedEdge = edge;

            while (sharedEdge->getSharedEdge(std::nothrow)) {
                sharedEdge = sharedEdge->getSharedEdge(std::nothrow);
            }

            edge->reuse(sharedEdge->getMemoryPtr());
        }
    }

    if (look & Edge::LOOK_DOWN) {
        for (size_t i = 0; i < getParentEdges().size(); i++) {
            auto edge = getParentEdgeAt(i);
            EdgePtr sharedEdge = edge;

            while (sharedEdge->getSharedEdge(std::nothrow)) {
                sharedEdge = sharedEdge->getSharedEdge(std::nothrow);
            }

            edge->reuse(sharedEdge->getMemoryPtr());
        }
    }
}

}  // namespace ov::intel_cpu::node

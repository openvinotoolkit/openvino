// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input.h"

#include "cpu/x64/jit_generator.hpp"
#include "nodes/node_config.h"
#include "openvino/core/parallel.hpp"
#include "shape_inference/shape_inference_pass_through.hpp"

using namespace dnnl;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {

#if defined(OPENVINO_ARCH_X86_64)
namespace {
struct jit_has_subnormals_base : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_has_subnormals_base)

    typedef struct {
        const float* src;
        const size_t count;
        bool hasSubnormals;
    } args_t;

    typedef void (*fn_t)(const args_t*);

    jit_has_subnormals_base() : jit_generator(jit_name()) {
        jit_ker_ = nullptr;
    }

    fn_t get() {
        return jit_ker() || create_kernel() == dnnl::impl::status::success
                ? (fn_t)jit_ker()
                : nullptr;
    }

protected:
    void foreach(const Xbyak::Reg64& idx,
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

    void copy_floats(const Xbyak::Reg64& dst,
                     const Xbyak::Reg64& src,
                     const Xbyak::Reg64& size) {
        push(rsi);
        push(r15);

        xor_(rsi, rsi);

        foreach(rsi, 1, size, [&, this](const Xbyak::Reg64& idx) {
            mov(r15d, dword[src + idx * sizeof(float)]);
            mov(dword[dst + idx * sizeof(float)], r15d);
        });

        pop(r15);
        pop(rsi);
    }

    void check_subnormals(const Xbyak::Reg64& src, const Xbyak::Ymm &exponent_mask, const Xbyak::Ymm &mantissa_mask, const Xbyak::Ymm &zero) {
        auto a = ymm1;
        auto b = ymm2;
        auto c = ymm3;

        vmovdqu(a, yword[src]);         // load 8 floats
        vpand(b, a, mantissa_mask);     // b = a & 00000000011111111111111111111111
        vpcmpeqd(b, b, zero);           // if (b == 0) b = 1 else b = 0
        vpand(c, a, exponent_mask);     // c = a & 01111111100000000000000000000000
        vpcmpeqd(c, c, zero);           // if (c == 0) c = 1 else c = 0
        vptest(b, c);                   // if ((!b & c) == 0) CF = 1 else CF = 0
    }

    void check_subnormals(const Xbyak::Reg64& src, const Xbyak::Xmm &exponent_mask, const Xbyak::Xmm &mantissa_mask, const Xbyak::Xmm &zero) {
        auto a = xmm1;
        auto b = xmm2;
        auto c = xmm3;

        uni_vmovdqu(a, xword[src]);          // load 4 floats
        uni_vmovdqu(b, a);                   // b = a
        uni_vmovdqu(c, a);                   // c = a
        uni_vpand(b, b, mantissa_mask);      // b = a & 00000000011111111111111111111111
        uni_vpcmpeqd(b, b, zero);            // if (b == 0) b = 1 else b = 0
        uni_vpand(c, c, exponent_mask);      // c = a & 01111111100000000000000000000000
        uni_vpcmpeqd(c, c, zero);            // if (c == 0) c = 1 else c = 0
        uni_vtestps(b, c);                   // if ((!b & c) == 0) CF = 1 else CF = 0
    }

protected:
    Label exit, has_subnormals, no_subnormals;

    const Reg64 &reg_src = rax;
    const Reg64 &reg_dst = rbx;
    const Reg64 &reg_sz = rdx;
    const Reg64 &reg_idx = rsi;
    const Reg64 &reg_mask_addr = r15;

    static const uint32_t exponent_mask_data[8];
    static const uint32_t mantissa_mask_data[8];
};

const uint32_t jit_has_subnormals_base::exponent_mask_data[8] = {
    0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000,
    0x7f800000, 0x7f800000, 0x7f800000, 0x7f800000
};

const uint32_t jit_has_subnormals_base::mantissa_mask_data[8] = {
    0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff,
    0x007fffff, 0x007fffff, 0x007fffff, 0x007fffff
};

template<cpu_isa_t isa>
struct jit_has_subnormals : public jit_has_subnormals_base {
    using Vmm = typename dnnl::impl::utils::conditional<isa == sse41, Xbyak::Xmm, Xbyak::Ymm>::type;

    const Vmm rmm4 = Vmm(4);
    const Vmm rmm5 = Vmm(5);
    const Vmm rmm6 = Vmm(6);
    const int length = isa == sse41 ? 4 : 8;

    void generate() override final { // NOLINT
        size_t const vlen = length;
        const int sh_bits = std::ilogb(vlen);

        auto zero = rmm4;
        auto exponent_mask = rmm5;
        auto mantissa_mask = rmm6;

        preamble();

        // Get arguments addresses
        mov(reg_src, ptr[param1 + offsetof(args_t, src)]);
        lea(reg_dst, ptr[param1 + offsetof(args_t, hasSubnormals)]);
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

        foreach(reg_idx, 1, r8, [&, this](const Xbyak::Reg64& idx) {
            check_subnormals(reg_src, exponent_mask, mantissa_mask, zero);
            jnc(has_subnormals);
            add(reg_src, sizeof(float) * vlen);
        });

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
        jc(no_subnormals);
        add(rsp, vlen * sizeof(float));

        L(has_subnormals);

        mov(rax, 1);
        mov(byte[reg_dst], al);
        jmp(exit);

        L(no_subnormals);
        add(rsp, vlen * sizeof(float));

        L(exit);

        postamble();
    }
};

jit_has_subnormals_base::fn_t jit_has_subnormals_function() {
    if (mayiuse(cpu_isa_t::avx2)) {
        static jit_has_subnormals<cpu_isa_t::avx2> generator;
        static auto fn = generator.get();
        return fn;
    } else if (mayiuse(cpu_isa_t::sse41)) {
        static jit_has_subnormals<cpu_isa_t::sse41> generator;
        static auto fn = generator.get();
        return fn;
    }
    return nullptr;
}

}   // namespace
#endif

Input::Input(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, PassThroughShapeInferFactory()) {
    if (!one_of(op->get_type_info(),
                op::v0::Parameter::get_type_info_static(),
                op::v0::Constant::get_type_info_static(),
                op::v0::Result::get_type_info_static(),
                op::v3::ReadValue::get_type_info_static(),
                op::v6::ReadValue::get_type_info_static()))
        OPENVINO_THROW_NOT_IMPLEMENTED("CPU Input node doesn't support ngraph operation ",
                                       op->get_type_name(),
                                       " with name ",
                                       op->get_friendly_name());
    constOp = ov::as_type_ptr<op::v0::Constant>(op);
    if (constOp) {
        constant = ConstantType::Const;
        cloneBlobIfRequired();
    } else {
        constant = ConstantType::StrictNoConst;
    }
}

void Input::cloneBlobIfRequired() {
    Shape shape(constOp->get_shape().empty() ? ov::Shape(1, 1) : constOp->get_shape());
    const auto prec = constOp->get_element_type();
    const size_t size = shape.getElementsCount();
    CpuBlockedMemoryDesc memDesc(prec, shape);

    bool needFlushDenormalsToZero = true;
    if (context->getConfig().DAZOn) {
        // DAZ has been set, processor automatically converts all denormal source operands
        // to a zero with the sign of the original operand before performing any
        // computations on them, thus no need to flush them to zero manually
        needFlushDenormalsToZero = false;
    }

    auto cloneBlob = [&, this] () {
        MemoryPtr memory;

        // CVS-74980
        // oneDNN always allocate 1byte for element type with bitWidth < 8 (u4,u1...)
        // but ngraph Constant uses actual bitWidth for data storage allocation
        // in that case we make a copy to avoid overflow
        if (constOp->get_byte_size() >= memDesc.getCurrentMemSize()) {
            if (constOp->get_element_type() == element::string) {
                memory = std::make_shared<StringMemory>(getEngine(), memDesc, constOp->get_data_ptr<element::string>());
            } else {
                memory = std::make_shared<Memory>(getEngine(), memDesc, constOp->get_data_ptr());
            }
        } else {
            if (constOp->get_element_type() == element::string) {
                memory = std::make_shared<StringMemory>(getEngine(), memDesc);
                auto src = constOp->get_data_ptr<StringMemory::OvString>();
                auto dst = memory->getDataAs<StringMemory::OvString>();
                std::copy(src, src + size, dst);
            } else {
                memory = std::make_shared<Memory>(getEngine(), memDesc);
                memcpy(memory->getData(), constOp->get_data_ptr(), constOp->get_byte_size());
            }
        }

        MemoryPtr ptr;
        if (memDesc.getPrecision() == element::string) {
            ptr = std::make_shared<StringMemory>(getEngine(), memDesc);
        } else {
            ptr = std::make_shared<StaticMemory>(getEngine(), memDesc);
        }
        ptr->load(*memory.get(), needFlushDenormalsToZero);

        return ptr;
    };

    auto isBlobAligned = [&] () {
        bool blobAlignedOnSSE = true;
#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
        // Majority of arithmetic and data processing instructions in legacy SSE isa requires
        // the memory address in the operands must be aligned on 16-byte boundary. To ensure
        // safely reusing ngraph const blob memory, need to check address alignment.
        const void *ptr = constOp->get_data_ptr();
        blobAlignedOnSSE = mayiuse(cpu_isa_t::avx2) || ((reinterpret_cast<uintptr_t>(ptr) & 15) == 0);
#endif
        return blobAlignedOnSSE;
    };

    // The presence of subnormals is better to determined at IR read time.
    auto hasSubnormals = [&] () {
        if (prec == ov::element::f32) {
            uint32_t const *u32data = constOp->get_data_ptr<uint32_t>();

            if (!size)
                return false;

#if defined(OPENVINO_ARCH_X86_64)
            if (auto fn = jit_has_subnormals_function()) {
                static const size_t batch_size = 2048;
                const size_t iterations_num = size / batch_size + 1;

                volatile bool has_subnormals = false;

                parallel_for(iterations_num, [&](int n) {
                    auto ptr = u32data + n * batch_size;
                    const jit_has_subnormals_base::args_t args = {
                        reinterpret_cast<float const *>(ptr),
                        std::min(batch_size, (size_t)(u32data + size - ptr)),
                        false
                    };

                    fn(&args);

                    if (args.hasSubnormals)
                        has_subnormals = true;
                });

                return has_subnormals;
            }
#endif

            uint32_t mantissaMask = 0x007fffff;
            uint32_t exponentMask = 0x7f800000;
            for (size_t i = 0; i < size; ++i) {
                if ((u32data[i] & exponentMask) == 0 && (u32data[i] & mantissaMask) != 0) {
                    return true;
                }
            }
        }
        return false;
    };

    auto blobKey = [&] () {
        char ptr[32];
        snprintf(ptr, sizeof ptr, "%p", constOp->get_data_ptr());
        return getName()
                + "_" + std::to_string(size * prec.size())
                + "_" + ptr;
    };

    const auto weightCache = context->getWeightsCache();
    const bool clone_is_not_needed =
        prec != element::string &&
        // IRs already have all subnormals flushed to zero, but in
        // read_model scenario with directly loaded original model still can have subnormals
        isBlobAligned() && (!needFlushDenormalsToZero || !hasSubnormals()) &&
        // Blob should be cloned in cache only if original weights are stored on other numa node.
        // This is possible only in multistream case on multisocket machine.
        // TODO: don't clone blob for multisocket + multistream case if current stream is run on the numa node where original weights are stored.
        (!weightCache || context->getNumNumaNodes() == 1 || context->getCPUStreamExecutor()->get_streams_num() == 1);
    memoryPtr = clone_is_not_needed ? std::make_shared<Memory>(getEngine(), memDesc, constOp->get_data_ptr())
                                    : std::const_pointer_cast<const IMemory>(
                                          weightCache ? *weightCache->findOrCreate(blobKey(), cloneBlob) : cloneBlob());
}

static std::vector<Shape> createInputShapes(const Shape& shape,
                                            const Type type) {
    if (type == Type::Output)
        return {shape};
    return {};
}

static std::vector<Shape> createOutputShapes(const Shape& shape,
                                             const Type type) {
    if (type == Type::Input)
        return {shape};
    return {};
}

static std::vector<ov::element::Type> createInputPrecisions(const ov::element::Type& prc,
                                                         const Type type) {
    if (type == Type::Output)
        return {prc};
    return {};
}

static std::vector<ov::element::Type> createOutputPrecisions(const ov::element::Type& prc,
                                                          const Type type) {
    if (type == Type::Input)
        return {prc};
    return {};
}

Input::Input(const Shape& shape,
             const ov::element::Type& prc,
             const std::string& name,
             const std::string& type,
             const GraphContext::CPtr context)
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

Input::Input(MemoryDescPtr memDesc, const std::string& name, const std::string& type, const GraphContext::CPtr context)
    : Input(memDesc->getShape(), memDesc->getPrecision(), name, type, context) {
    extMemDesc = memDesc;
}

Input::Input(const std::shared_ptr<ov::Node>& op,
             const GraphContext::CPtr context,
             InputConfig config)
    : Input(op, context) {
    extMemDesc = config.desc;
    m_isInPlace = config.inPlace;
}

Input::Input(const std::shared_ptr<ov::Node>& op,
             const GraphContext::CPtr context,
             OutputConfig config)
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
        if (!getParentEdges().empty())
            THROW_CPU_NODE_ERR("has incorrect number of input edges.");
        if (getChildEdges().empty())
            THROW_CPU_NODE_ERR("has incorrect number of output edges.");
    } else if (getType() == Type::Output) {
        if (getParentEdges().size() != 1)
            THROW_CPU_NODE_ERR("has incorrect number of input edges.");
        if (!getChildEdges().empty())
            THROW_CPU_NODE_ERR("has incorrect number of output edges.");
    }
}

void Input::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    if (extMemDesc) {
        initSupportedPdFromMemDesc();
    } else {
        initSupportedPdDefault();
    }
}

void Input::initOptimalPrimitiveDescriptor() {
    if (m_useParentMemoryDescForOutput || extMemDesc)
        return;

    Node::initOptimalPrimitiveDescriptor();
}

void Input::selectOptimalPrimitiveDescriptor() {
    if (!(m_useParentMemoryDescForOutput && getType() == Type::Output))
        return Node::selectOptimalPrimitiveDescriptor();

    // ignore previous configuration
    supportedPrimitiveDescriptors.clear();

    // and just use parent memory descriptor for Output node to avoid reorders insertion
    NodeConfig config({PortConfig(getParentOutputMemDesc(getParentEdgeAt(0)), BlockedMemoryDesc::FULL_MASK, 0)}, {});

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown);
    selectPrimitiveDescriptorByIndex(0);
}

void Input::createPrimitive() {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto dstMemPtr = getDstMemoryAtPort(i);
        if (!dstMemPtr)
            THROW_CPU_NODE_ERR("has null memory object at port ", i,
                              " to node ", getChildEdgeAt(i)->getChild()->getName(), ".");
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto srcMemPtr = getSrcMemoryAtPort(i);
        if (!srcMemPtr)
            THROW_CPU_NODE_ERR("has null memory object at port ", i,
                              " from node ", getParentEdgeAt(i)->getParent()->getName(), ".");
    }

    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        THROW_CPU_NODE_ERR("doesn't have selected primitive descriptor.");
}

bool Input::created() const {
    return getType() == Type::Input || getType() == Type::Output;
}

void Input::initSupportedPdDefault() {
    std::vector<PortConfigurator> inPortConfs;
    std::vector<PortConfigurator> outPortConfs;

    if (getType() == Type::Input || getType() == Type::MemoryInput) {
        auto precision = getOriginalOutputPrecisionAtPort(0);

        outPortConfs.push_back({LayoutType::ncsp, precision});
        if (!getParentEdges().empty()) {
            inPortConfs.push_back({LayoutType::ncsp, precision, true});
        }
    } else if (getType() == Type::Output) {
        auto precision = getOriginalInputPrecisionAtPort(0);

        inPortConfs.push_back({LayoutType::ncsp, precision});
    }

    addSupportedPrimDesc(inPortConfs,
                         outPortConfs,
                         impl_desc_type::unknown);
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

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

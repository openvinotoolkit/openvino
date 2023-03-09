// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input.h"
#include "common/cpu_memcpy.h"
#include <dnnl_extension_utils.h>

#include <string>
#include <tuple>
#include <algorithm>
#include <cmath>
#include <utils/general_utils.h>
#include <ngraph/ops.hpp>
#include <ie_parallel.hpp>
#include <ie_ngraph_utils.hpp>
#include <blob_factory.hpp>
#include "caseless.hpp"
#include "common/cpu_memcpy.h"
#include "common/cpu_convert.h"
#include "utils/cpu_utils.hpp"
#include <cpu/x64/jit_generator.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/shape_inference/shape_inference_pass_through.hpp"

using namespace dnnl;
using namespace InferenceEngine;
using namespace details;
using namespace ngraph::op;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

namespace ov {
namespace intel_cpu {
namespace node {
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

    void check_subnormals(const Xbyak::Reg64& src, const Xbyak::Ymm &mask, const Xbyak::Ymm &zero) {
        auto a = ymm1;
        auto b = ymm2;
        auto c = ymm3;

        vmovdqu(a, yword[src]);         // load 8 floats
        vpcmpeqd(b, a, zero);           // if (a == 0) b = 1 else b = 0
        vpand(c, a, mask);              // c = a & 01111111100000000000000000000000
        vpcmpeqd(c, c, zero);           // if (c == 0) c = 1 else c = 0
        vptest(b, c);                   // if ((!b & c) == 0) CF = 1 else CF = 0
    }

    void check_subnormals(const Xbyak::Reg64& src, const Xbyak::Xmm &mask, const Xbyak::Xmm &zero) {
        auto a = xmm1;
        auto b = xmm2;
        auto c = xmm3;

        uni_vmovdqu(a, xword[src]);          // load 4 floats
        uni_vmovdqu(b, a);                   // b = a
        uni_vmovdqu(c, a);                   // c = a
        uni_vpcmpeqd(b, b, zero);               // if (a == 0) b = 1 else b = 0
        uni_vpand(c, c, mask);                  // c = a & 01111111100000000000000000000000
        uni_vpcmpeqd(c, c, zero);               // if (c == 0) c = 1 else c = 0
        uni_vtestps(b, c);                    // if ((!b & c) == 0) CF = 1 else CF = 0
    }

    template<cpu_isa_t isa>
    struct reg;

protected:
    Label exit, has_subnormals, no_subnormals;

    const Reg64 &reg_src = rax;
    const Reg64 &reg_dst = rbx;
    const Reg64 &reg_sz = rdx;
    const Reg64 &reg_idx = rsi;
    const Reg64 &reg_mask_addr = r15;

    static const uint32_t mask_data[8];
};

const uint32_t jit_has_subnormals_base::mask_data[8] = {
    0xFF << 23, 0xFF << 23, 0xFF << 23, 0xFF << 23,
    0xFF << 23, 0xFF << 23, 0xFF << 23, 0xFF << 23
};

template<>
struct jit_has_subnormals_base::reg<cpu_isa_t::avx2> {
    constexpr static uint32_t length = 8;
    constexpr static const Xbyak::Ymm & rmm4 = Xbyak::util::ymm4;
    constexpr static const Xbyak::Ymm & rmm5 = Xbyak::util::ymm5;
};

template<>
struct jit_has_subnormals_base::reg<cpu_isa_t::sse41> {
    constexpr static uint32_t length = 4;
    constexpr static const Xbyak::Xmm & rmm4 = Xbyak::util::xmm4;
    constexpr static const Xbyak::Xmm & rmm5 = Xbyak::util::xmm5;
};

template<cpu_isa_t isa>
struct jit_has_subnormals : public jit_has_subnormals_base {
    void generate() override final { // NOLINT
        size_t const vlen = reg<isa>::length;
        const int sh_bits = std::ilogb(vlen);

        auto zero = reg<isa>::rmm4;
        auto mask = reg<isa>::rmm5;

        preamble();

        // Get arguments addresses
        mov(reg_src, ptr[param1 + offsetof(args_t, src)]);
        lea(reg_dst, ptr[param1 + offsetof(args_t, hasSubnormals)]);
        mov(reg_sz, ptr[param1 + offsetof(args_t, count)]);
        mov(reg_mask_addr, (size_t)mask_data);

        // Initialize necessary consts
        uni_vpxor(zero, zero, zero);
        uni_vmovdqu(mask, ptr[reg_mask_addr]);

        // Main loop
        xor_(reg_idx, reg_idx);
        mov(r8, reg_sz);
        shr(r8, sh_bits);

        foreach(reg_idx, 1, r8, [&, this](const Xbyak::Reg64& idx) {
            check_subnormals(reg_src, mask, zero);
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
        check_subnormals(r8, mask, zero);
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

Input::Input(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, PassThroughShapeInferFactory()) {
    if (!one_of(op->get_type_info(),
            v0::Parameter::get_type_info_static(),
            v0::Constant::get_type_info_static(),
            v0::Result::get_type_info_static(),
            v3::ReadValue::get_type_info_static(),
            v6::ReadValue::get_type_info_static()))
        IE_THROW(NotImplemented) << "CPU Input node doesn't support ngraph operation " << op->get_type_name() << " with name " << op->get_friendly_name();

    constant = ConstantType::NoConst;

    constOp = ngraph::as_type_ptr<ngraph::op::Constant>(op);
    if (constOp) {
        constant = ConstantType::Const;
        cloneBlobIfRequired();
    }
}

void Input::cloneBlobIfRequired() {
    Shape shape(constOp->get_shape().empty() ? ngraph::Shape(1, 1) : constOp->get_shape());
    const auto prec = convertPrecision(constOp->get_element_type());
    const size_t size = shape.getElementsCount();
    DnnlBlockedMemoryDesc memDesc(prec, shape);

    auto cloneBlob = [&, this] () {
        Memory memory{ getEngine() };

        // CVS-74980
        // oneDNN always allocate 1byte for element type with bitWidth < 8 (u4,u1...)
        // but ngraph Constant uses actual bitWidth for data storage allocation
        // in that case we make a copy to avoid overflow
        if (constOp->get_byte_size() >= memDesc.getCurrentMemSize()) {
            memory.Create(memDesc, constOp->get_data_ptr());
        } else {
            memory.Create(memDesc);
            memcpy(memory.GetPtr(), constOp->get_data_ptr(), constOp->get_byte_size());
        }

        MemoryPtr ptr = MemoryPtr(new Memory(getEngine()));
        ptr->Create(memDesc);
        ptr->SetData(memory);

        return ptr;
    };

    auto isBlobAligned = [&, this] () {
        const void *ptr = constOp->get_data_ptr();
        return prec.size() > 1 ? (reinterpret_cast<size_t>(ptr) % prec.size()) == 0 : true;
    };

    // The presence of subnormals is better to determined at IR read time.
    auto hasSubnormals = [&, this] () {
        if (prec == InferenceEngine::Precision::FP32) {
            uint32_t const *u32data = constOp->get_data_ptr<uint32_t>();

            if (!size)
                return false;

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
            } else {
                for (size_t i = 0; i < size; ++i) {
                    if (u32data[i] && (u32data[i] & (0xFF << 23)) == 0) {
                        return true;
                    }
                }
            }
        }
        return false;
    };

    // WA for CVS-46304
    auto isWA = [&, this] () {
        auto outputs = constOp->outputs();
        for (const auto& output : outputs) {
            auto node = output.get_node();
            if (!node
                || TypeFromName(node->get_type_name()) != Type::FullyConnected)
                continue;
            if (mayiuse(cpu_isa_t::avx512_core)) {
                if (size % 16)
                    return true;
            } else if (mayiuse(cpu_isa_t::avx)) {
                if (size % 8)
                    return true;
            } else if (mayiuse(cpu_isa_t::sse41)) {
                if (size % 4)
                    return true;
            }
        }
        return false;
    };

    auto blobKey = [&, this] () {
        char ptr[32];
        snprintf(ptr, sizeof ptr, "%p", constOp->get_data_ptr());
        return getName()
                + "_" + std::to_string(size * prec.size())
                + "_" + ptr;
    };

    auto weightCache = context->getWeightsCache();
    if (weightCache) {
        MemoryPtr ptr = *weightCache->findOrCreate(blobKey(), cloneBlob);
        memoryPtr = std::const_pointer_cast<const Memory>(ptr);
    // IRs already have all subnormals flushed to zero, but in
    // read_model scenario with directly loaded original model still can have subnormals
    } else if (isBlobAligned() && !hasSubnormals() && !isWA()) {
        auto ptr = new Memory(getEngine());
        ptr->Create(memDesc, constOp->get_data_ptr());
        memoryPtr = MemoryCPtr(ptr);
    } else {
        memoryPtr = std::const_pointer_cast<const Memory>(cloneBlob());
    }
}

Input::Input(const Shape& shape,
             const InferenceEngine::Precision& prc,
             const std::string& name,
             const std::string& type,
             const GraphContext::CPtr context)
    : Node(type, name, context) {
    constant = ConstantType::NoConst;
    if (getType() == Type::Input) {
        outputShapes.emplace_back(shape);
        addOriginalOutputPrecision(prc);
    }  else if (getType() == Type::Output) {
        inputShapes.emplace_back(shape);
        addOriginalInputPrecision(prc);
    }
}

Input::Input(MemoryDescPtr memDesc, const std::string& name, const std::string& type, const GraphContext::CPtr context)
    : Input(memDesc->getShape(), memDesc->getPrecision(), name, type, context) {
    extMemDesc = memDesc;
}

void Input::withMeanImage() {
    isMeanImage = true;
}

MemoryCPtr Input::getMemoryPtr() const {
    return memoryPtr;
}

void Input::getSupportedDescriptors() {
    if (getType() == Type::Input) {
        if (!getParentEdges().empty())
            IE_THROW() << "Incorrect number of input edges for layer " << getName();
        if (getChildEdges().empty())
            IE_THROW() << "Incorrect number of output edges for layer " << getName();
    } else if (getType() == Type::Output) {
        if (getParentEdges().size() != 1)
            IE_THROW() << "Incorrect number of input edges for layer " << getName();
        if (!getChildEdges().empty())
            IE_THROW() << "Incorrect number of output edges for layer " << getName();
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

void Input::createPrimitive() {
    for (size_t i = 0; i < getChildEdges().size(); i++) {
        auto &dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->isAllocated())
            IE_THROW() << "Destination memory didn't allocate for node " << getName()
                               << " to node " << getChildEdgeAt(i)->getChild()->getName() << ".";
    }
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        auto &srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->isAllocated())
            IE_THROW() << "Destination memory didn't allocate for node " << getName()
                               << " from node " << getParentEdgeAt(i)->getParent()->getName() << ".";
    }

    const NodeDesc *selected_pd = getSelectedPrimitiveDescriptor();
    if (selected_pd == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for node " << getName() << ".";
}

bool Input::created() const {
    return getType() == Type::Input || getType() == Type::Output;
}

void Input::initSupportedPdDefault() {
    std::vector<PortConfigurator> inPortConfs;
    std::vector<PortConfigurator> outPortConfs;

    if (getType() == Type::Input || getType() == Type::MemoryInput) {
        auto precision = getOriginalOutputPrecisionAtPort(0);
        if (precision == Precision::U16 || isMeanImage) {
            precision = Precision::FP32;
        }

        outPortConfs.push_back({LayoutType::ncsp, precision});
        if (!getParentEdges().empty()) {
            inPortConfs.push_back({LayoutType::ncsp, precision, true});
        }
    } else if (getType() == Type::Output) {
        auto precision = getOriginalInputPrecisionAtPort(0);
        if (precision == Precision::U16) precision = Precision::FP32;

        inPortConfs.push_back({LayoutType::ncsp, precision});
    }

    addSupportedPrimDesc(inPortConfs,
                         outPortConfs,
                         impl_desc_type::unknown);
}

void Input::initSupportedPdFromMemDesc() {
    NodeConfig config;
    PortConfig portConfig;
    portConfig.inPlace(-1);
    portConfig.constant(false);
    portConfig.setMemDesc(extMemDesc);
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

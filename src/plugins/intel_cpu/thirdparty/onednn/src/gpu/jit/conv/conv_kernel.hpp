/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef GPU_JIT_CONV_CONV_KERNEL_HPP
#define GPU_JIT_CONV_CONV_KERNEL_HPP

#include <exception>

#include "gpu/jit/conv/config.hpp"
#include "gpu/jit/conv/fma_support.hpp"
#include "gpu/jit/conv/ir.hpp"
#include "gpu/jit/conv/kernel_arg_info.hpp"
#include "gpu/jit/conv/kernel_builder.hpp"
#include "gpu/jit/conv/message_support.hpp"
#include "gpu/jit/conv/ngen_proxy.hpp"
#include "gpu/jit/conv/post_op_support.hpp"
#include "gpu/jit/conv/reduce_support.hpp"
#include "gpu/jit/conv/reorder_support.hpp"
#include "gpu/jit/jit_eltwise_injector.hpp"
#include "gpu/jit/jit_generator.hpp"
#include "gpu/jit/ngen/ngen.hpp"
#include "gpu/jit/ngen/ngen_register_allocator.hpp"

#include "gpu/jit/gemm/emulation.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

template <typename T>
T to_cpp(const ngen::Immediate &imm) {
    auto u64 = uint64_t(imm);
    switch (imm.getType()) {
        case ngen::DataType::w:
            return (T)utils::bit_cast<std::array<int16_t, 4>>(u64)[0];
        case ngen::DataType::uw:
            return (T)utils::bit_cast<std::array<uint16_t, 4>>(u64)[0];
        case ngen::DataType::d:
            return (T)utils::bit_cast<std::array<int32_t, 2>>(u64)[0];
        case ngen::DataType::ud:
            return (T)utils::bit_cast<std::array<uint32_t, 2>>(u64)[0];
        case ngen::DataType::q: return (T)utils::bit_cast<int64_t>(u64);
        case ngen::DataType::uq: return (T)utils::bit_cast<uint64_t>(u64);
        default: ir_error_not_expected();
    }
    return 0;
}

// type_t to ngen::DataType convertor.
ngen::DataType to_ngen(const type_t &type) {
    ir_assert(type.is_scalar()) << "Expected scalar type.";

#define CASE(_kind, ngen_enum) \
    if (type.kind() == type_kind_t::_kind) return ngen::DataType::ngen_enum

    CASE(bf16, bf);
    CASE(f16, hf);
    CASE(f32, f);
    CASE(s16, w);
    CASE(s32, d);
    CASE(s64, q);
    CASE(s8, b);
    CASE(u16, uw);
    CASE(u32, ud);
    CASE(u64, uq);
    CASE(u8, ub);

    if (type == type_t::byte_ptr()) return ngen::DataType::uq;

#undef CASE
    ir_error_not_expected();
    return ngen::DataType::invalid;
}

ngen::Immediate to_ngen(
        const expr_t &expr, const type_t &type = type_t::undef()) {
    ir_assert(expr.type().is_scalar()) << "Vector types are not supported.";
    if (expr.is<int_imm_t>()) {
        auto &imm = expr.as<int_imm_t>();
        // No conversion.
        if (utils::one_of(type, type_t::undef(), expr.type()))
            return ngen::Immediate(imm.value);
            // Do conversion.
#define CASE(cpp_type) \
    if (type.is_cpp<cpp_type>()) return ngen::Immediate(cpp_type(imm.value))

        CASE(int16_t);
        CASE(int32_t);
        CASE(int64_t);
        CASE(uint16_t);
        CASE(uint32_t);
        CASE(uint64_t);

#undef CASE
        ir_error_not_expected() << "Can't convert expression: " << expr;
    } else if (expr.is<float_imm_t>()) {
        ir_assert(utils::one_of(type, type_t::undef(), type_t::f32()))
                << "Conversion is not supported.";
        auto &imm = expr.as<float_imm_t>();
        return ngen::Immediate(imm.value);
    }
    ir_error_not_expected() << "Can't convert expression: " << expr;
    return ngen::Immediate();
}

ngen::Bundle to_ngen(const ngen_proxy::Bundle &bundle) {
    return ngen::Bundle(bundle.bank_id, bundle.bundle_id);
}

ngen::InstructionModifier to_ngen(
        const ngen_proxy::InstructionModifier &mod_proxy) {
    ngen::InstructionModifier mod;
    if (mod_proxy.is_atomic) mod |= ngen::ThreadCtrl::Atomic;
    if (!mod_proxy.sbid.is_empty()) mod |= ngen::SBID(mod_proxy.sbid.token).set;
    return mod;
}

ngen::AtomicOp to_ngen(ngen_proxy::AtomicOp atomic_op) {
    switch (atomic_op) {
        case ngen_proxy::AtomicOp::fadd: return ngen::AtomicOp::fadd;
        default: ir_error_not_expected();
    }
    return ngen::AtomicOp(std::numeric_limits<uint16_t>::max());
}

ngen::ConditionModifier cmp_op_to_ngen(op_kind_t op_kind) {
    ir_assert(is_cmp_op(op_kind));
    switch (op_kind) {
        case op_kind_t::_eq: return ngen::ConditionModifier::eq;
        case op_kind_t::_ne: return ngen::ConditionModifier::ne;
        case op_kind_t::_ge: return ngen::ConditionModifier::ge;
        case op_kind_t::_gt: return ngen::ConditionModifier::gt;
        case op_kind_t::_le: return ngen::ConditionModifier::le;
        case op_kind_t::_lt: return ngen::ConditionModifier::lt;
        default: ir_error_not_expected();
    }
    return ngen::ConditionModifier::none;
}

ngen::RegData ngen_reg_data(ngen::HW hw, const ngen::RegData &base,
        int off_bytes, ngen::DataType type, int width, int hstride = 1) {
    if (type == ngen::DataType::invalid) type = base.getType();
    auto grf_size = ngen::GRF::bytes(hw);
    auto new_off = base.getByteOffset() + off_bytes;
    auto new_grf_off = (new_off % grf_size);
    auto type_size = ngen::getBytes(type);
    auto grf = ngen::GRF(base.getBase() + new_off / grf_size).retype(type);

    ir_assert(new_grf_off % type_size == 0);

    if (width == 1) {
        hstride = 0;
    } else if (hstride == 0) {
        ir_assert(width == 1);
    } else {
        int max_width = 32 / type_size;
        width = std::min(width, max_width / hstride);
        width = std::min(width, 16);
    }
    int vstride = width * hstride;
    return grf[new_grf_off / type_size](vstride, width, hstride);
}

ngen::Subregister ngen_subregister(ngen::HW hw, const ngen::RegData &base,
        int off_bytes, ngen::DataType type = ngen::DataType::invalid) {
    if (type == ngen::DataType::invalid) type = base.getType();
    auto rd = ngen_reg_data(hw, base, off_bytes, type, 1, 0);
    return ngen::Subregister(rd, rd.getOffset(), rd.getType());
}

ngen::Immediate ngen_negate(const ngen::Immediate &imm) {
    switch (imm.getType()) {
        case ngen::DataType::w: return ngen::Immediate(-to_cpp<int16_t>(imm));
        case ngen::DataType::d: return ngen::Immediate(-to_cpp<int32_t>(imm));
        case ngen::DataType::f: return ngen::Immediate(-to_cpp<float>(imm));
        default: ir_error_not_expected();
    }
    return ngen::Immediate();
}

bool ngen_is_qw(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::q, ngen::DataType::uq);
}

bool ngen_is_dw(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::d, ngen::DataType::ud);
}

bool ngen_is_w(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::w, ngen::DataType::uw);
}

bool ngen_is_b(ngen::DataType type) {
    return utils::one_of(type, ngen::DataType::b, ngen::DataType::ub);
}

bool ngen_is_xf(ngen::DataType type) {
    return utils::one_of(
            type, ngen::DataType::bf, ngen::DataType::hf, ngen::DataType::f);
}

enum class ngen_operand_kind_t { invalid, immediate, reg_data, flag_register };

// Wrapper to generalize ngen::FlagRegister, ngen::RegData and ngen::Immediate
// operands.
class ngen_operand_t {
public:
    ngen_operand_t() : kind_(ngen_operand_kind_t::invalid) {}

    ngen_operand_t(const ngen::FlagRegister &flag)
        : kind_(ngen_operand_kind_t::flag_register)
        , ptr_(new ngen::FlagRegister(flag),
                  destroy<ngen_operand_kind_t::flag_register>) {}

    ngen_operand_t(const ngen::RegData &reg_data)
        : kind_(ngen_operand_kind_t::reg_data)
        , ptr_(new ngen::RegData(reg_data),
                  destroy<ngen_operand_kind_t::reg_data>) {}

    ngen_operand_t(const ngen::Immediate &imm)
        : kind_(ngen_operand_kind_t::immediate)
        , ptr_(new ngen::Immediate(imm),
                  destroy<ngen_operand_kind_t::immediate>) {}

    template <typename T>
    ngen_operand_t(const T &other, const ngen::InstructionModifier &mod)
        : ngen_operand_t(other) {
        mod_ = mod;
    }

    const ngen::Immediate &immediate() const {
        ir_assert(is_immediate());
        return *(const ngen::Immediate *)ptr_.get();
    }

    const ngen::RegData &reg_data() const {
        ir_assert(is_reg_data());
        return *(const ngen::RegData *)ptr_.get();
    }

    const ngen::FlagRegister &flag_register() const {
        ir_assert(is_flag_register());
        return *(const ngen::FlagRegister *)ptr_.get();
    }

    ngen::InstructionModifier flag_register_mod() const {
        ngen::InstructionModifier mod;
        mod |= flag_register();
        return !is_negated() ? mod : ~mod;
    }

    const ngen::InstructionModifier &mod() const { return mod_; }

    bool is_invalid() const { return kind_ == ngen_operand_kind_t::invalid; }

    bool is_immediate() const {
        return kind_ == ngen_operand_kind_t::immediate;
    }

    bool is_reg_data() const { return kind_ == ngen_operand_kind_t::reg_data; }

    bool is_flag_register() const {
        return kind_ == ngen_operand_kind_t::flag_register;
    }

    bool is_negated() const { return is_negated_; }

    ngen::DataType type() const {
        if (is_immediate()) return immediate().getType();
        if (is_reg_data()) return reg_data().getType();
        ir_error_not_expected();
        return ngen::DataType::invalid;
    }

    ngen_operand_t operator-() const {
        if (is_immediate()) { return ngen_operand_t(ngen_negate(immediate())); }
        if (is_reg_data()) { return ngen_operand_t(-reg_data()); }
        if (is_flag_register()) {
            auto ret = *this;
            ret.is_negated_ = !ret.is_negated_;
            return ret;
        }
        ir_error_not_expected();
        return ngen_operand_t();
    }

    ngen_operand_t reinterpret(ngen::HW hw, const type_t &new_type) const {
        ir_assert(is_reg_data());
        ir_assert(new_type.is_scalar());
        return ngen_operand_t(
                ngen_reg_data(hw, reg_data(), 0, to_ngen(new_type), 1), mod_);
    }

    // Creates an operand with the requested register region based on the
    // existing region. off - offset in elements of the region data type.
    ngen_operand_t sub_reg_data(ngen::HW hw, int off, int exec_size) const {
        ir_assert(is_reg_data());
        auto rd = reg_data();
        int new_base = rd.getBase();
        int new_off = rd.getByteOffset() + off * rd.getBytes() * rd.getHS();
        int grf_size = ngen::GRF::bytes(hw);
        new_base += (new_off / grf_size);
        new_off = (new_off % grf_size);

        rd.setBase(new_base);
        rd.setOffset(new_off / rd.getBytes());
        rd.setRegion(1, exec_size, rd.getHS());
        rd.fixup(exec_size, ngen::DataType::invalid, false, 1);
        return ngen_operand_t(rd, exec_size);
    }

    bool operator==(const ngen_operand_t &other) const {
        if (kind_ != other.kind_) return false;
        if (mod_.getAll() != other.mod_.getAll()) return false;
        switch (kind_) {
            case ngen_operand_kind_t::immediate: {
                auto &this_imm = immediate();
                auto &other_imm = other.immediate();
                return (this_imm.getType() == other_imm.getType())
                        && (uint64_t(this_imm) == uint64_t(other_imm));
            }
            case ngen_operand_kind_t::reg_data:
                return reg_data() == other.reg_data();
            case ngen_operand_kind_t::flag_register:
                return flag_register() == other.flag_register();
            default: ir_error_not_expected();
        }
        return false;
    }

private:
    template <ngen_operand_kind_t kind>
    static void destroy(void *ptr) {
        if (!ptr) return;

        switch (kind) {
            case ngen_operand_kind_t::immediate:
                delete (ngen::Immediate *)ptr;
                break;
            case ngen_operand_kind_t::reg_data:
                delete (ngen::RegData *)ptr;
                break;
            case ngen_operand_kind_t::flag_register:
                delete (ngen::FlagRegister *)ptr;
                break;
            default: ir_error_not_expected();
        }
    }

    ngen_operand_kind_t kind_;
    std::shared_ptr<void> ptr_;
    ngen::InstructionModifier mod_;

    // Whether the operand is negated. Applicable to flag registers only.
    // Negation of register data and immediate operands is directly supported
    // through nGEN API.
    bool is_negated_ = false;
};

template <typename T>
T to_cpp(ngen::HW hw, const ngen_operand_t &op) {
    ir_assert(op.is_immediate());
    return to_cpp<T>(op.immediate());
}

// Maintains scoped allocations which are automatically released when the scope
// is destructed.
class ngen_register_scope_t {
public:
    ngen_register_scope_t(ngen::RegisterAllocator &ra) : ra_(&ra) {}

    ngen_register_scope_t(const ngen_register_scope_t &) = delete;

    ngen_register_scope_t(ngen_register_scope_t &&other)
        : ra_(other.ra_)
        , grf_ranges_(std::move(other.grf_ranges_))
        , subregisters_(std::move(other.subregisters_)) {
        other.ra_ = nullptr;
    }

    ngen::RegisterAllocator &register_allocator() { return *ra_; }

    ~ngen_register_scope_t() {
        for (auto &r : grf_ranges_)
            ra_->safeRelease(r);

        for (auto &s : subregisters_)
            ra_->safeRelease(s);
        for (auto &f : flags_)
            ra_->safeRelease(f);
    }

    ngen::GRFRange try_alloc_range(
            int regs, ngen::Bundle base_bundle = ngen::Bundle()) {
        auto ret = ra_->try_alloc_range(regs, base_bundle);
        if (!ret.isInvalid()) grf_ranges_.push_back(ret);
        return ret;
    }

    ngen::GRFRange alloc_range(
            int regs, ngen::Bundle base_bundle = ngen::Bundle()) {
        auto ret = ra_->alloc_range(regs, base_bundle);
        grf_ranges_.push_back(ret);
        return ret;
    }

    ngen::GRF alloc(ngen::Bundle bundle = ngen::Bundle()) {
        return alloc_range(1, bundle)[0];
    }

    ngen::Subregister alloc_sub(
            ngen::DataType type, ngen::Bundle bundle = ngen::Bundle()) {
        auto ret = ra_->alloc_sub(type, bundle);
        subregisters_.push_back(ret);
        return ret;
    }

    ngen::RegData alloc_reg_data(const type_t &type, int stride_bytes = -1,
            ngen::Bundle bundle = ngen::Bundle()) {
        if (type.is_scalar()) return alloc_sub(to_ngen(type), bundle);

        if (stride_bytes == -1) stride_bytes = type.scalar().size();

        ir_assert(stride_bytes > 0);
        ir_assert(stride_bytes % type.scalar().size() == 0);

        int grf_size = ngen::GRF::bytes(ra_->hardware());
        int regs = utils::div_up(type.elems() * stride_bytes, grf_size);
        auto sub = alloc_range(regs, bundle)[0].retype(
                to_ngen(type.scalar()))[0];
        auto ret = sub(stride_bytes / type.scalar().size());
        ret.fixup(type.elems(), ngen::DataType::invalid, false, 1);
        return ret;
    }

    ngen::FlagRegister alloc_flag() {
        auto ret = ra_->alloc_flag();
        flags_.push_back(ret);
        return ret;
    }

    template <typename T>
    void safeRelease(T &t) {
        ra_->safeRelease(t);
    }

private:
    ngen::RegisterAllocator *ra_;

    std::vector<ngen::GRFRange> grf_ranges_;
    std::vector<ngen::Subregister> subregisters_;
    std::vector<ngen::FlagRegister> flags_;
};

class expr_binding_t {
public:
    expr_binding_t(ngen::HW hw) : hw_(hw) {}

    ~expr_binding_t() {
        if (!std::uncaught_exception()) {
            ir_assert(expr2dst_.empty()) << "Detected missing unbind_dst().";
        }
    }

    bool is_dst_bound(const expr_t &expr) const {
        return expr2dst_.count(expr) == 1;
    }

    ngen_operand_t get_dst(const expr_t &expr) const {
        ir_assert(is_dst_bound(expr)) << "Destination is not bound: " << expr;
        return expr2dst_.at(expr);
    }

    void bind_dst(const expr_t &expr, const ngen_operand_t &operand) {
        ir_assert(!expr.is_empty());
        auto ret = expr2dst_.insert({expr, operand});
        ir_assert(ret.second) << "Already bound: " << expr;
    }

    void unbind_dst(const expr_t &expr) {
        ir_assert(!expr.is_empty());
        auto it = expr2dst_.find(expr);
        ir_assert(it != expr2dst_.end());
        expr2dst_.erase(it);
    }

    bool is_bound(const expr_t &expr) const {
        return expr2operand_.count(expr) == 1;
    }

    ngen_operand_t get(const expr_t &expr, bool allow_empty = false) const {
        if (expr.is_empty()) return ngen_operand_t();
        if (!is_bound(expr)) {
            if (!allow_empty)
                ir_assert(false) << "Operand is not bound: " << expr;
            return ngen_operand_t();
        }
        return expr2operand_.at(expr);
    }

    void bind(const expr_t &expr, const ngen_operand_t &operand) {
        if (is_dst_bound(expr)) unbind_dst(expr);

        auto op_to_bind = operand;

        // Operand is with predicate - can't bind.
        if (operand.mod().getPredCtrl() != ngen::PredCtrl::None) return;

        int esize = operand.mod().getExecSize();
        if (esize == 0) esize = 1;
        if (esize != expr.type().elems()) {
            ir_assert(expr.type().is_scalar() || esize == 1)
                    << "Expected broadcast.";
            // Can't bind scalar to vector, extract scalar and bind.
            if (operand.is_reg_data() && esize != 1) {
                op_to_bind = ngen_reg_data(
                        hw_, operand.reg_data(), 0, ngen::DataType::invalid, 1);
            }
        }

        auto ret = expr2operand_.insert({expr, op_to_bind});
        ir_assert(ret.second) << "Already bound: " << expr;
    }

    void unbind(const expr_t &expr) {
        ir_assert(!expr.is_empty());

        auto it = expr2operand_.find(expr);
        ir_assert(it != expr2operand_.end());
        expr2operand_.erase(it);
    }

private:
    ngen::HW hw_;
    object_map_t<expr_t, ngen_operand_t> expr2dst_;
    object_map_t<expr_t, ngen_operand_t> expr2operand_;
};

template <ngen::HW hw>
class expr_evaluator_t;

template <ngen::HW hw>
class ir_to_ngen_t;

template <ngen::HW hw>
class conv_kernel_t : public jit_generator<hw> {
public:
    NGEN_FORWARD_OPENCL(hw);

    friend class expr_evaluator_t<hw>;
    friend class ir_to_ngen_t<hw>;
    friend class send_impl_t;

    conv_kernel_t(const conv_config_t &cfg, const convolution_pd_t *pd,
            const kernel_arg_info_t &kernel_arg_info);

    void setup_interface(const stmt_t &kernel_body,
            const kernel_arg_info_t &kernel_arg_info) {
        externalName("gen_conv");
        requireLocalID(3);
        requireLocalSize();
        requireGRF(cfg_.regs);
        requireSIMD(cfg_.simd_size);
        requireBarrier();
        if (utils::one_of(cfg_.fma_kind, fma_kind_t::dpas, fma_kind_t::dpasw))
            requireDPAS();
        if (cfg_.do_atomic_update) requireGlobalAtomics();

        for (int i = 0; i < kernel_arg_info.nargs(); i++) {
            auto &name = kernel_arg_info.arg_name(i);
            auto &type = kernel_arg_info.arg_type(i);
            if (type.is_ptr()) {
                newArgument(name, ngen::ExternalArgumentType::GlobalPtr);
            } else {
                newArgument(name, to_ngen(type));
            }
        }

        int slm_size
                = alloc_manager_t(kernel_body).total_size(alloc_kind_t::slm);
        requireSLM(slm_size);

        finalizeInterface();
    }

    // Kernel padding for instruction prefetch.
    void pad_kernel() {
        for (int rep = 0; rep < 8; rep++)
            nop();
    }

    void emov(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0) {
        if (dst.is_reg_data()) {
            if (src0.is_reg_data()) {
                emov(mod, dst.reg_data(), src0.reg_data());
            } else if (src0.is_immediate()) {
                emov(mod, dst.reg_data(), src0.immediate());
            } else {
                emov(mod | src0.flag_register_mod(), dst.reg_data(), 1);
                emov(mod | ~src0.flag_register_mod(), dst.reg_data(), 0);
            }
        } else {
            // dst is a flag register.
            ir_assert(!dst.is_negated());
            if (src0.is_reg_data()) {
                emov(mod, dst.flag_register(), src0.reg_data());
            } else {
                emov(mod, dst.flag_register(), src0.immediate());
            }
        }
    }

    void eadd(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            eadd(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            eadd(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void emul(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            emul(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            auto &src1_imm = src1.immediate();
            if (ngen_is_qw(dst.type()) || ngen_is_w(src1_imm.getType())) {
                emul(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
                return;
            }
            if (ngen_is_dw(src1_imm.getType())) {
                ir_assert(mod.getExecSize() == 1);
                auto tmp = ra_.alloc_sub<int64_t>();
                emul(mod, tmp.q(0), src0.reg_data(), src1_imm);
                emov(mod, dst.reg_data(), tmp.reinterpret(0, dst.type()));
                ra_.safeRelease(tmp);
                return;
            }
            emul(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void eadd3(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1,
            const ngen_operand_t &src2) {
        if (hw >= ngen::HW::XeHP) {
            if (src2.is_reg_data()) {
                add3(mod, dst.reg_data(), src0.reg_data(), src1.reg_data(),
                        src2.reg_data());
            } else {
                add3(mod, dst.reg_data(), src0.reg_data(), src1.reg_data(),
                        src2.immediate());
            }
            return;
        }
        add(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        if (src2.is_reg_data()) {
            add(mod, dst.reg_data(), dst.reg_data(), src2.reg_data());
        } else {
            add(mod, dst.reg_data(), dst.reg_data(), src2.immediate());
        }
    }

    void emad(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1,
            const ngen_operand_t &src2) {
        if (src2.is_reg_data()) {
            mad(mod, dst.reg_data(), src0.reg_data(), src1.reg_data(),
                    src2.reg_data());
        } else if (hw < ngen::HW::XeLP) {
            mul(mod, dst.reg_data(), src1.reg_data(), src2.immediate());
            add(mod, dst.reg_data(), dst.reg_data(), src0.reg_data());
        } else {
            mad(mod, dst.reg_data(), src0.reg_data(), src1.reg_data(),
                    src2.immediate());
        }
    }

    void ediv(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (!src1.is_immediate()) {
            efdiv(mod, dst, src0, src1);
        } else {
            auto &src1_imm = src1.immediate();
            int32_t src1_value = to_cpp<int32_t>(src1_imm);
            ir_assert(0 < src1_value && src1_value <= INT32_MAX) << src1_value;
            eidiv(dst.reg_data(), ngen::Subregister(), src0.reg_data(),
                    src1_value);
        }
    }

    void efdiv(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        ir_assert(!src1.is_immediate());
        auto one = ra_.alloc().f();
        auto zero = ra_.alloc().f();
        auto src0_tmp = ra_.alloc_range(2);
        auto tmp = ra_.alloc_range(4);
        auto alt_mod = ngen::InstructionModifier(mod);

        const int width = mod.getExecSize() >= 8 ? 8 : mod.getExecSize();
        alt_mod.setExecSize(width);

        mov(alt_mod, one, ngen::Immediate(1));
        mov(alt_mod, zero, ngen::Immediate(0));
        mov(mod, src0_tmp[0].f(), dst.reg_data());
        mov(mod, src1.reg_data(), src1.reg_data());

        const ngen::FlagRegister flag;
        setDefaultNoMask(false);

        for (int i = 0; i < mod.getExecSize(); i += width) {
            fdiv_ieee(alt_mod, flag, dst.sub_reg_data(hw, i, width).reg_data(),
                    src0_tmp[i / width].f(),
                    src1.sub_reg_data(hw, i, width).reg_data(), zero, one, tmp);
        }

        ra_.safeRelease(one);
        ra_.safeRelease(zero);
        ra_.safeRelease(src0_tmp);
        ra_.safeRelease(tmp);
        setDefaultNoMask(true);
    }

    void emod(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        ir_assert(src1.is_immediate());
        auto &src1_imm = src1.immediate();
        int32_t src1_value = to_cpp<int32_t>(src1_imm);
        ir_assert(0 < src1_value && src1_value <= INT32_MAX) << src1_value;
        eidiv(ngen::Subregister(), dst.reg_data(), src0.reg_data(), src1_value);
    }

    void eshl(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            shl(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            shl(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void eshr(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            shr(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            shr(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void emin(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            min_(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            min_(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void emax(const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
            const ngen_operand_t &src0, const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            max_(mod, dst.reg_data(), src0.reg_data(), src1.reg_data());
        } else {
            max_(mod, dst.reg_data(), src0.reg_data(), src1.immediate());
        }
    }

    void ecmp(const ngen::InstructionModifier &mod, const ngen_operand_t &src0,
            const ngen_operand_t &src1) {
        if (src1.is_reg_data()) {
            cmp(mod, src0.reg_data(), src1.reg_data());
        } else {
            cmp(mod, src0.reg_data(), src1.immediate());
        }
    }

    // Adapted version of magicgu function from Hacker's Delight 10-15.
    static void eidiv_magicgu(uint32_t d, uint32_t &m, uint32_t &p) {
        uint32_t s32_max = std::numeric_limits<int32_t>::max();
        ir_assert(d != 0 && d <= s32_max);
        uint64_t nc = (s32_max / d) * d - 1;
        for (p = 32; p < 64; p++) {
            uint64_t _2p = 1LL << p;
            if (_2p > nc * (d - 1 - (_2p - 1) % d)) {
                m = (_2p + d - 1 - (_2p - 1) % d) / d;
                return;
            }
        }
        ir_error_not_expected();
    }

    // Emulates integer division by a constant.
    // Requirements:
    //     0 <= x <= UINT32_MAX
    //     0 <  y <= INT32_MAX
    // Computes:
    //     qot = x / y
    //     rem = x % y
    void eidiv(const ngen::RegData &qot, const ngen::RegData &rem,
            const ngen::RegData &x, uint32_t y) {
        if (ngen::utils::is_zero_or_pow2(y)) {
            if (!qot.isInvalid()) shr(1, qot, x, ngen::utils::log2(y));
            if (!rem.isInvalid()) and_(1, rem, x, y - 1);
            return;
        }

        uint32_t m = 0, p = 0;
        eidiv_magicgu(y, m, p);

        auto _x = ra_.alloc().ud();
        auto _qot = ra_.alloc().ud();
        mov(1, _x, x);

        // qot = (x * m) >> p
        mul(1, acc0.ud(0), _x, m & 0xFFFF);
        mach(1, _qot, _x, m);
        shr<uint32_t>(1, _qot, _qot, p - 32);
        if (!qot.isInvalid()) mov(1, qot, _qot);

        if (!rem.isInvalid()) {
            // rem = x - qot * y
            bool y_is_16_bit = (y <= static_cast<uint32_t>(
                                        std::numeric_limits<int16_t>::max()));
            if (hw >= ngen::HW::XeLP && y_is_16_bit) {
                mad(1, rem, x, _qot, -int16_t(y));
            } else {
                auto tmp = ra_.alloc_sub<uint64_t>();
                mul(1, tmp, _qot, y);
                add(1, rem, x, -tmp.ud(0));
                ra_.safeRelease(tmp);
            }
        }

        ra_.safeRelease(_x);
        ra_.safeRelease(_qot);
    }

    friend struct dnnl::impl::gpu::jit::EmulationImplementation;
    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0) {
        EmulationImplementation::emov<DT>(*this, mod, dst, src0, emu_strategy);
    }
    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::Immediate src0) {
        EmulationImplementation::emov<DT>(*this, mod, dst, src0, emu_strategy);
    }
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1) {
        EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1) {
        EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1) {
        if (ngen_is_xf(dst.getType())) {
            mul(mod, dst, src0, src1);
            return;
        }
        EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1) {
        if (ngen_is_xf(dst.getType())) {
            mul(mod, dst, src0, src1);
            return;
        }
        EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void eshl(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1) {
        EmulationImplementation::eshl<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }
    template <typename DT = void>
    void eshr(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1) {
        EmulationImplementation::eshr<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }

private:
    const conv_config_t &cfg_;
    ngen::RegisterAllocator ra_;
    ngen::GRF signal_header_;

    EmulationStrategy emu_strategy = EmulationStrategy(hw);
    EmulationState emu_state;
};

inline ngen::Subregister get_subregister(
        ngen::HW hw, ngen::DataType type, const ngen::GRFRange &r, int idx) {
    int grf_size = ngen::GRF::bytes(hw);
    int type_size = ngen::getBytes(type);
    int off = idx * type_size;
    return r[off / grf_size].sub((off % grf_size) / type_size, type);
}

inline ngen::Subregister get_subregister(const ngen::RegData &rd) {
    return ngen::Subregister(rd, rd.getOffset(), rd.getType());
}

template <ngen::HW hw = ngen::HW::Unknown>
class zero_out_kernel_t : public jit_generator<hw> {
public:
    NGEN_FORWARD_OPENCL(hw);

    zero_out_kernel_t(int simd, int regs, bool with_dpas) : ra_(hw) {
        externalName("zero_out");
        requireLocalID(1);
        requireLocalSize();
        requireGRF(regs);
        requireSIMD(simd);
        if (with_dpas) requireDPAS();

        newArgument("ptr", ngen::ExternalArgumentType::GlobalPtr);
        newArgument("size", ngen::DataType::ud);

        finalizeInterface();

        // Claim registers.
        ra_.claim(r0);
        ra_.claim(getLocalID(0));
        ra_.claim(getLocalSize(0));
        ra_.claim(getArgument("ptr"));
        ra_.claim(getArgument("size"));

        setDefaultNoMask();
        setDefaultAutoSWSB(true);

        bool use_a64 = false;

        prologue();

        if (emu_strategy.emulate64) {
            emu_state.temp[0] = ra_.alloc();
            emu_state.temp[1] = ra_.alloc();
        }

        auto ptr = getArgument("ptr");
        auto surf = Surface(getArgumentSurface("ptr"));
        auto size = getArgument("size");
        auto global_id = ra_.alloc_sub<uint32_t>();
        auto off0 = ra_.alloc_sub<uint32_t>();

        mul(1, global_id, r0.ud(1), getLocalSize(0).uw());
        add(1, global_id, global_id, getLocalID(0));
        shl(1, off0, global_id, math::ilog2q(bytes_per_thr / simd));

        int grf_size = ngen::GRF::bytes(hw);
        int bytes_per_store = 16;
        int ud_size = sizeof(uint32_t);
        int uq_size = sizeof(uint64_t);

        auto zero = ra_.alloc_range(bytes_per_store * ud_size / grf_size);
        auto off_vec = ra_.alloc_range(bytes_per_thr * ud_size / grf_size);
        auto ptr_vec = ra_.alloc_range(bytes_per_thr * uq_size / grf_size);

        for (int i = 0; i < bytes_per_store * ud_size; i += 64) {
            auto z = get_subregister(hw, ngen::DataType::ud, zero, i);
            mov(16, z, 0);
        }

        auto idx_vec = ra_.alloc().uw();
        mov(8, idx_vec, ngen::Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));

        for (int i = 0; i < bytes_per_thr; i += 8) {
            auto off_sub_vec
                    = get_subregister(hw, ngen::DataType::ud, off_vec, i)(1);
            add3(8, off_sub_vec, off0, idx_vec, i);
            if (use_a64) {
                auto ptr_sub_vec = get_subregister(
                        hw, ngen::DataType::uq, ptr_vec, i)(1);
                eadd(8, ptr_sub_vec, ptr, off_sub_vec);
            }
        }

        for (int i = 0; i < bytes_per_thr; i += bytes_per_store) {
            auto off_sub_vec
                    = get_subregister(hw, ngen::DataType::ud, off_vec, i)(1);
            cmp(16 | lt | f0[0], off_sub_vec, size);
            if (use_a64) {
                auto h_a64
                        = get_subregister(hw, ngen::DataType::uq, ptr_vec, i);
                store(16 | f0[0], ngen::scattered_byte(), A64, h_a64, zero[0]);
            } else {
                auto h_bts = off_sub_vec;
                store(16 | f0[0], ngen::scattered_byte(), surf, h_bts, zero[0]);
            }
        }

        epilogue();
    }

    friend struct dnnl::impl::gpu::jit::EmulationImplementation;

    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0) {
        EmulationImplementation::emov<DT>(*this, mod, dst, src0, emu_strategy);
    }

    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1) {
        EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }

    static const int bytes_per_thr;

private:
    ngen::RegisterAllocator ra_;
    EmulationStrategy emu_strategy = EmulationStrategy(hw);
    EmulationState emu_state;
};

template <ngen::HW hw>
const int zero_out_kernel_t<hw>::bytes_per_thr = 128;

template <ngen::HW hw = ngen::HW::Unknown>
class reorder_kernel_t : public jit_generator<hw> {
public:
    NGEN_FORWARD_OPENCL(hw);

    reorder_kernel_t(int simd, int regs, bool with_dpas) : ra_(hw) {
        externalName("reorder");
        requireLocalID(1);
        requireLocalSize();
        requireGRF(regs);
        requireSIMD(simd);
        if (with_dpas) requireDPAS();

        newArgument("src", ngen::ExternalArgumentType::GlobalPtr);
        newArgument("dst", ngen::ExternalArgumentType::GlobalPtr);
        newArgument("elems", ngen::DataType::ud);

        finalizeInterface();

        // Claim registers.
        ra_.claim(r0);
        ra_.claim(getLocalID(0));
        ra_.claim(getLocalSize(0));
        ra_.claim(getArgument("src"));
        ra_.claim(getArgument("dst"));
        ra_.claim(getArgument("elems"));

        setDefaultNoMask();
        setDefaultAutoSWSB(true);

        bool use_a64 = false;

        prologue();

        if (emu_strategy.emulate64) {
            emu_state.temp[0] = ra_.alloc();
            emu_state.temp[1] = ra_.alloc();
        }

        int grf_size = ngen::GRF::bytes(hw);
        int ud_size = sizeof(uint32_t);
        int uq_size = sizeof(uint64_t);
        int f_size = sizeof(float);
        int bf_size = sizeof(uint16_t);

        auto src = getArgument("src");
        auto dst = getArgument("dst");
        auto src_surf = Surface(getArgumentSurface("src"));
        auto dst_surf = Surface(getArgumentSurface("dst"));
        auto elems = getArgument("elems");
        auto global_id = ra_.alloc_sub<uint32_t>();
        auto elem_vec = ra_.alloc_range(elems_per_thr * ud_size / grf_size);
        auto src_ptr_vec = ra_.alloc_range(elems_per_thr * uq_size / grf_size);

        auto get_elem = [&](int i) {
            return get_subregister(hw, ngen::DataType::ud, elem_vec, i);
        };

        auto S = ra_.alloc_range(elems_per_thr * f_size / grf_size);
        // D is for bf16 but allocated as dword-strided to use with
        // scattered_byte(2) messages.
        auto D = ra_.alloc_range(elems_per_thr * f_size / grf_size);

        auto get_src_reg = [&](int i) {
            return get_subregister(hw, ngen::DataType::f, S, i);
        };

        auto get_dst_reg = [&](int i) {
            return get_subregister(hw, ngen::DataType::bf, D, i);
        };

        mul(1, global_id, r0.ud(1), getLocalSize(0).uw());
        add(1, global_id, global_id, getLocalID(0));

        auto idx_vec = ra_.alloc().uw();
        mov(8, idx_vec, ngen::Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        for (int i = 0; i < elems_per_thr; i += 8)
            shl(8, get_elem(i), global_id, math::ilog2q(elems_per_thr / simd));
        for (int i = 0; i < elems_per_thr; i += 8) {
            add3(8, get_elem(i), get_elem(i), idx_vec, i);
            if (use_a64) {
                auto src_ptr_sub_vec = get_subregister(
                        hw, ngen::DataType::uq, src_ptr_vec, i)(1);
                eshl(8, src_ptr_sub_vec, get_elem(i)(1), math::ilog2q(f_size));
                eadd(8, src_ptr_sub_vec, src_ptr_sub_vec, src);
            }
        }

        int elems_per_load = 16;
        for (int i = 0; i < elems_per_thr; i += elems_per_load) {
            cmp(16 | lt | f0[0], get_elem(i), elems);
            if (use_a64) {
                auto h_a64 = get_subregister(
                        hw, ngen::DataType::uq, src_ptr_vec, i);
                load(16 | f0[0], get_src_reg(i), ngen::scattered_dword(), A64,
                        h_a64);
            } else {
                auto h_bts = get_elem(i);
                load(16 | f0[0], get_src_reg(i), ngen::scattered_dword(),
                        src_surf, h_bts);
            }
        }

        int mov_step = (grf_size == 32 ? 8 : 16);
        for (int i = 0; i < elems_per_thr; i += mov_step) {
            // dst is dword-strided.
            mov(mov_step, get_dst_reg(i * 2)(2), get_src_reg(i)(1));
        }

        auto dst_header = ra_.alloc_range(
                elems_per_load * (use_a64 ? uq_size : ud_size) / grf_size);
        for (int i = 0; i < elems_per_thr; i += elems_per_load) {
            for (int j = 0; j < elems_per_load; j += 8) {
                ngen::RegData h;
                if (use_a64) {
                    int off = j * uq_size;
                    h = dst_header[off / grf_size].uq(
                            (off % grf_size) / uq_size)(1);
                } else {
                    int off = j * ud_size;
                    h = dst_header[off / grf_size].ud(
                            (off % grf_size) / ud_size)(1);
                }
                eshl(8, h, get_elem(i + j)(1), math::ilog2q(bf_size));
                if (use_a64) eadd(8, h, h, dst);
            }

            cmp(16 | lt | f0[0], get_elem(i), elems);
            if (use_a64) {
                store(16 | f0[0], ngen::scattered_byte(2), A64, dst_header[0],
                        get_dst_reg(i * 2));
            } else {
                store(16 | f0[0], ngen::scattered_byte(2), dst_surf,
                        dst_header[0], get_dst_reg(i * 2));
            }
        }

        epilogue();
    }

    friend struct dnnl::impl::gpu::jit::EmulationImplementation;

    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0) {
        EmulationImplementation::emov<DT>(*this, mod, dst, src0, emu_strategy);
    }

    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1) {
        EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }

    template <typename DT = void>
    void eshl(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1) {
        EmulationImplementation::eshl<DT>(
                *this, mod, dst, src0, src1, emu_strategy, emu_state);
    }

    static const int elems_per_thr;

private:
    ngen::RegisterAllocator ra_;
    EmulationStrategy emu_strategy = EmulationStrategy(hw);
    EmulationState emu_state;
};

template <ngen::HW hw>
const int reorder_kernel_t<hw>::elems_per_thr = 32;

// Evaluates expression by emitting instructions with nGEN.
template <ngen::HW hw>
class expr_evaluator_t : public ir_visitor_t {
public:
    expr_evaluator_t(conv_kernel_t<hw> *host,
            const expr_binding_t &expr_binding, ngen_register_scope_t &scope)
        : host_(host), expr_binding_(expr_binding), scope_(scope) {}

    // If `dst_operand` is not empty, use its pre-allocated location for the
    // result.
    ngen_operand_t eval(const expr_t &e,
            const ngen_operand_t &dst_operand = ngen_operand_t()) {
        if (!dst_operand.is_invalid()) {
            ir_assert(dst_operand.mod().getExecSize() != 0);
        }
        if (expr_binding_.is_bound(e)) {
            if (!dst_operand.is_invalid()) {
                host_->emov(
                        dst_operand.mod(), dst_operand, expr_binding_.get(e));
                return dst_operand;
            }
        } else {
            if (!dst_operand.is_invalid())
                expr_binding_.bind_dst(e, dst_operand);
            visit(e);
        }

        return expr_binding_.get(e, /*allow_empty=*/true);
    }

    std::vector<ngen_operand_t> eval(const std::vector<expr_t> &exprs) {
        std::vector<ngen_operand_t> ret;
        for (auto &e : exprs) {
            if (!expr_binding_.is_bound(e)) visit(e);
            ret.push_back(expr_binding_.get(e));
        }
        return ret;
    }

    void _visit(const binary_op_t *obj) override {
        auto dst_op = alloc_dst_op(obj);
        auto mod = dst_op.mod();

        switch (obj->op_kind) {
            case op_kind_t::_and: {
                eval(obj->a, dst_op);
                eval(obj->b,
                        ngen_operand_t(
                                dst_op, mod | dst_op.flag_register_mod()));
                break;
            }
            default: {
                // Some cases require pre-allocated register regions with
                // special strides for a/b.
                auto a_out_op = maybe_alloc_strided_op(obj->type, obj->a);
                auto b_out_op = maybe_alloc_strided_op(obj->type, obj->b);
                auto src0_op = eval(obj->a, a_out_op);
                auto src1_op = eval(obj->b, b_out_op);
                ebinary(obj, mod, dst_op, src0_op, src1_op);
                break;
            }
        }

        bind(obj, dst_op);
    }

    void _visit(const bool_imm_t *obj) override {
        // Scalar booleans must never be directly lowered:
        // - Booleans are mapped to flag registers
        // - Flag register stores vector of boolean vectors
        // - All boolean values in IR must be expressed by shuffle_t objects
        // - _visit(shuffle_t *) must properly handle vector of booleans -> flag
        //   register lowering
        ir_error_not_expected();
    }

    void _visit(const cast_t *obj) override {
        auto &from_type = obj->expr.type();
        auto &to_type = obj->type;

        ir_assert(from_type != to_type) << "Equal types are not expected.";

        if (is_const(obj->expr)) {
            bind(obj, to_ngen(obj->expr, to_type));
            return;
        }

        auto dst_op = alloc_dst_op(obj);

        // Handle ptr -> u64 and u64 -> ptr casts.
        if (utils::one_of(obj->type, type_t::u64(), type_t::byte_ptr())
                && utils::one_of(
                        obj->expr.type(), type_t::u64(), type_t::byte_ptr())) {
            eval(obj->expr, dst_op);
            bind(obj, dst_op);
            return;
        }

        // Handle integer (down-)conversion, assume bitwise equality in this
        // case. Examples: d <-> ud, d -> w, q -> d.
        bool is_int_convert = (from_type.is_scalar() && to_type.is_scalar()
                && from_type.is_int() && to_type.is_int()
                && from_type.size() >= to_type.size());
        if (is_int_convert) {
            eval(obj->expr, dst_op.reinterpret(hw, from_type));
            bind(obj, dst_op);
            return;
        }

        auto expr_op = eval(obj->expr);
        auto mod = dst_op.mod();
        if (obj->saturate) mod |= host_->sat;
        host_->emov(mod, dst_op, expr_op);
        bind(obj, dst_op);
    }

    void _visit(const float_imm_t *obj) override { bind(obj, to_ngen(obj)); }

    void _visit(const int_imm_t *obj) override { bind(obj, to_ngen(obj)); }

    void _visit(const load_t *obj) override {
        auto &type = obj->type;
        auto buf_op = eval(obj->buf);
        auto off_op = eval(obj->off);
        int stride;
        if (obj->has_default_stride()) {
            stride = 1;
        } else {
            ir_assert(obj->stride % type.scalar().size() == 0);
            stride = obj->stride / type.scalar().size();
        }
        auto load_reg_data = ngen_reg_data(hw, buf_op.reg_data(),
                to_cpp<int>(off_op.immediate()), to_ngen(type.scalar()),
                type.elems(), stride);
        bind(obj, load_reg_data);
    }

    void _visit(const ptr_t *obj) override {
        auto base_op = eval(obj->base);

        if (is_zero(obj->off)) {
            bind(obj, base_op);
            return;
        }

        ir_assert(base_op.is_reg_data());

        int off = to_cpp<int>(obj->off);
        int base = base_op.reg_data().getBase();
        int grf_size = ngen::GRF::bytes(hw);
        auto grf = ngen::GRF(base + off / grf_size).retype(ngen::DataType::ub);
        if (off % grf_size == 0)
            bind(obj, grf);
        else
            bind(obj, grf[off % grf_size]);
    }

    void _visit(const shuffle_t *obj) override {
        int elems = obj->elems();
        if (obj->type.is_bool() && is_shuffle_const(obj)) {
            auto dst_op = alloc_dst_op(obj);
            auto e_shuffle = expr_t(obj);
            ir_assert(dst_op.is_flag_register()) << e_shuffle;
            ir_assert(!dst_op.is_negated()) << e_shuffle;
            uint16_t flag_mask = 0;
            for (int i = elems - 1; i >= 0; i--) {
                flag_mask <<= 1;
                flag_mask |= (to_cpp<bool>(e_shuffle[i]) ? 1 : 0);
            }
            if (dst_op.mod().getPredCtrl() == ngen::PredCtrl::None) {
                host_->emov(1, dst_op, ngen::Immediate(flag_mask));
            } else {
                ir_assert(dst_op.mod().getFlagReg() == dst_op.flag_register());
                host_->and_(1, dst_op.flag_register(), dst_op.flag_register(),
                        ngen::Immediate(flag_mask));
            }
            bind(obj, dst_op);
            return;
        }

        if (obj->is_broadcast()) {
            if (obj->type.is_bool()) {
                auto dst_op = alloc_dst_op(obj);
                eval(obj->vec[0], dst_op);
                bind(obj, dst_op);
            } else {
                auto scalar_op = eval(obj->vec[0]);
                bind(obj, scalar_op);
            }
            return;
        }

        // tuples: <offset, length, idx>
        std::vector<std::tuple<int, int, int>> chunks;
        for (int i = 0; i < elems; i++) {
            int idx = obj->idx[i];
            if (chunks.empty() || std::get<2>(chunks.back()) != idx) {
                chunks.emplace_back(i, 1, idx);
            } else {
                std::get<1>(chunks.back())++;
            }
        }

        auto dst_op = alloc_dst_op(obj);
        for (auto &chunk : chunks) {
            int off = std::get<0>(chunk);
            int length = std::get<1>(chunk);
            int idx = std::get<2>(chunk);
            // Split length into powers of two.
            while (length > 0) {
                int exec_size = (1 << math::ilog2q(length));
                auto chunk_op = dst_op.sub_reg_data(hw, off, exec_size);
                eval(obj->vec[idx], ngen_operand_t(chunk_op, exec_size));
                length -= exec_size;
                off += exec_size;
            }
        }
        bind(obj, dst_op);
    }

    void _visit(const ternary_op_t *obj) override {
        switch (obj->op_kind) {
            case op_kind_t::_add3:
            case op_kind_t::_mad: {
                auto dst_op = alloc_dst_op(obj);
                auto mod = dst_op.mod();
                auto src0_op = eval(obj->a);
                auto src1_op = eval(obj->b);
                auto src2_op = eval(obj->c);
                if (obj->op_kind == op_kind_t::_add3) {
                    host_->eadd3(mod, dst_op, src0_op, src1_op, src2_op);
                } else {
                    host_->emad(mod, dst_op, src0_op, src1_op, src2_op);
                }
                bind(obj, dst_op);
                break;
            }
            default: ir_error_not_expected();
        }
    }

    void _visit(const unary_op_t *obj) override {
        ir_assert(obj->op_kind == op_kind_t::_minus);
        auto a_op = eval(obj->a);
        bind(obj, -a_op);
    }

    void _visit(const var_t *obj) override {
        ir_assert(expr_binding_.is_bound(obj))
                << "Variable is not defined: " << expr_t(obj);
    }

private:
    ngen_operand_t alloc_dst_op(const expr_t &e) {
        ir_assert(!expr_binding_.is_bound(e)) << "Already evaluated: " << e;
        if (expr_binding_.is_dst_bound(e)) return expr_binding_.get_dst(e);

        // Expression is not bound yet, allocate new storage and bind.
        ngen_operand_t op;
        if (e.type().is_bool()) {
            op = ngen_operand_t(scope_.alloc_flag(), e.type().elems());
        } else {
            op = ngen_operand_t(
                    scope_.alloc_reg_data(e.type()), e.type().elems());
        }
        expr_binding_.bind_dst(e, op);
        return op;
    }

    ngen_operand_t alloc_tmp(const expr_t &e) {
        return ngen_operand_t(
                scope_.alloc_reg_data(e.type()), e.type().elems());
    }

    ngen::GRFRange alloc_tmp_range(const expr_t &e, size_t n = 0) {
        return scope_.alloc_range(n);
    }

    // Pre-allocates a strided register region for expression `e` if needed.
    ngen_operand_t maybe_alloc_strided_op(
            const type_t &res_type, const expr_t &e) {
        // Need q-strided region for `e` if res_type is q/uq and `e` is of a
        // sub-q data type and not a scalar.
        if (e.type().is_scalar()) return ngen_operand_t();
        if (!utils::one_of(res_type.scalar(), type_t::s64(), type_t::u64()))
            return ngen_operand_t();
        if (utils::one_of(e.type().scalar(), type_t::s64(), type_t::u64()))
            return ngen_operand_t();

        auto *shuffle = e.as_ptr<shuffle_t>();
        if (shuffle && shuffle->is_broadcast()) return ngen_operand_t();

        return ngen_operand_t(
                scope_.alloc_reg_data(e.type(), res_type.scalar().size()),
                e.type().elems());
    }

    void bind(const expr_t &e, const ngen_operand_t &op) {
        if (!expr_binding_.is_dst_bound(e)) {
            expr_binding_.bind(e, op);
            return;
        }
        auto dst_op = expr_binding_.get_dst(e);
        if (dst_op == op) {
            expr_binding_.bind(e, op);
            return;
        }
        // Expression is already bound, move to the location it was bound to.
        // This is required for immediate values - they are bound as is but
        // sometimes we need them to be moved to registers.
        host_->emov(dst_op.mod(), dst_op, op);
        expr_binding_.bind(e, dst_op);
    }

    void ebinary(const binary_op_t *obj, const ngen::InstructionModifier &mod,
            const ngen_operand_t &dst, const ngen_operand_t &src0,
            const ngen_operand_t &src1) {
        switch (obj->op_kind) {
            case op_kind_t::_add: host_->eadd(mod, dst, src0, src1); break;
            case op_kind_t::_sub: host_->eadd(mod, dst, src0, -src1); break;
            case op_kind_t::_mul: host_->emul(mod, dst, src0, src1); break;
            case op_kind_t::_div: host_->ediv(mod, dst, src0, src1); break;
            case op_kind_t::_mod: host_->emod(mod, dst, src0, src1); break;
            case op_kind_t::_shl: host_->eshl(mod, dst, src0, src1); break;
            case op_kind_t::_shr: host_->eshr(mod, dst, src0, src1); break;
            case op_kind_t::_min: host_->emin(mod, dst, src0, src1); break;
            case op_kind_t::_max: host_->emax(mod, dst, src0, src1); break;
            case op_kind_t::_ge:
            case op_kind_t::_gt:
            case op_kind_t::_le:
            case op_kind_t::_lt:
            case op_kind_t::_eq:
            case op_kind_t::_ne: {
                ir_assert(!dst.is_negated()) << "Destination can't be negated.";
                ngen::InstructionModifier cmp_mod = mod;
                cmp_mod |= cmp_op_to_ngen(obj->op_kind);
                cmp_mod |= dst.flag_register();
                host_->ecmp(cmp_mod, src0, src1);
                break;
            }
            default:
                ir_error_not_expected()
                        << "Unknown kind: " << to_string(obj->op_kind);
        }
    }

    conv_kernel_t<hw> *host_;
    expr_binding_t expr_binding_;
    ngen_register_scope_t &scope_;
};

template <typename DataSpecT, typename = void>
struct atomic_helper_t {
    template <typename GeneratorT>
    static void call(GeneratorT *, ngen::AtomicOp,
            const ngen::InstructionModifier &, const DataSpecT &,
            ngen::AddressBase, const ngen::RegData &, const ngen::RegData &) {
        ir_error_not_expected()
                << "Unknown DataSpec: atomics are not supported.";
    }
};

template <typename DataSpecT>
struct atomic_helper_t<DataSpecT,
        typename std::enable_if<
                std::is_same<DataSpecT, ngen::scattered_dword>::value>::type> {
    template <typename GeneratorT>
    static void call(GeneratorT *host, ngen::AtomicOp atomic_op,
            const ngen::InstructionModifier &mod, const DataSpecT &spec,
            ngen::AddressBase base, const ngen::RegData &addr,
            const ngen::RegData &data) {
        host->atomic(atomic_op, mod, spec, base, addr, data);
    }
};

// Helper to emit send instructions.
class send_impl_t {
public:
    send_impl_t(ngen::HW hw, const send_t &send) : hw_(hw), send_(send) {
        MAYBE_UNUSED(hw_);
    }

    template <typename GeneratorT, typename T>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const ngen::InstructionModifier &mod,
            const ngen::RegData &surf_base_addr, int surf_bti,
            const ngen::RegData &header, const T &data) {

        auto access_type = send_.access_type;
        auto data_type = send_.data_type;
        auto data_elems = send_.data_elems;
        auto address_model = send_.address_model;
        auto atomic_op = send_.atomic_op;

        bool is_read = (access_type == ngen_proxy::Access::Read);
        ngen::AddressBase address_base;
        if (address_model == ngen_proxy::AddressModel::ModelBTS) {
            address_base = ngen::AddressBase::createBTS(surf_bti);
        } else if (address_model == ngen_proxy::AddressModel::ModelA64) {
            address_base = ngen::AddressBase::createA64(true);
        } else if (address_model == ngen_proxy::AddressModel::ModelSLM) {
            address_base = ngen::AddressBase::createSLM();
        } else {
            ir_error_not_expected();
        }

        if (data_type == type_t::byte()) {
            emit_load_or_store(is_read, atomic_op, host, mod,
                    ngen::scattered_byte(data_elems), address_base, header,
                    data);
        } else if (data_type == type_t::dword()) {
            emit_load_or_store(is_read, atomic_op, host, mod,
                    ngen::scattered_dword(data_elems), address_base, header,
                    data);
        } else if (data_type == type_t::qword()) {
            emit_load_or_store(is_read, atomic_op, host, mod,
                    ngen::scattered_qword(data_elems), address_base, header,
                    data);
        } else if (data_type == type_t::oword()) {
            emit_load_or_store(is_read, atomic_op, host, mod,
                    ngen::block_oword(data_elems), address_base, header, data);
        } else if (data_type == type_t::hword()) {
            emit_load_or_store(is_read, atomic_op, host, mod,
                    ngen::block_hword(data_elems), address_base, header, data);
        } else {
            ir_error_not_expected();
        }
    }

private:
    template <typename GeneratorT, typename DataSpecT>
    void emit_load_or_store(bool is_read, ngen_proxy::AtomicOp atomic_op,
            GeneratorT *host, const ngen::InstructionModifier &mod,
            const DataSpecT &spec, ngen::AddressBase base,
            const ngen::RegData &addr, const ngen::RegData &data) {
        bool is_atomic = (atomic_op != ngen_proxy::AtomicOp::undef);

        ir_assert(!send_.is_prefetch) << "Prefetches are not supported.";

        if (is_read) {
            ir_assert(!is_atomic) << "Unexpected atomic loads.";
            host->load(mod, data, spec, base, addr);
        } else {
            if (is_atomic) {
                atomic_helper_t<DataSpecT>::call(
                        host, to_ngen(atomic_op), mod, spec, base, addr, data);
            } else {
                host->store(mod, spec, base, addr, data);
            }
        }
    }
    ngen::HW hw_;
    const send_t &send_;
};

// Reinterprets layouts to wider data type (up to 4 bytes).
// Example: 16a16b (s8 type) -> 16a4b (s32 type)
static bool try_reinterpret_to_wider_type(layout_t &src, layout_t &dst,
        const tensor_t &tile = {}, bool do_update = true,
        int *new_size_out = nullptr) {
    if (src.blocks().empty() || dst.blocks().empty()) return false;
    if (src.type() != dst.type()) return false;

    auto &s0 = src.blocks()[0];
    auto &d0 = dst.blocks()[0];
    if (s0.dim_idx != d0.dim_idx) return false;

    int old_size = src.type().size();
    int s0_old_size = int(s0.block) * old_size;
    int d0_old_size = int(d0.block) * old_size;

    int new_size = math::gcd(s0_old_size, d0_old_size);
    new_size = math::gcd(new_size, 4); // Try types up to 4 bytes.
    if (new_size <= old_size) return false;

    auto tile_ok = [&](const layout_t &l) {
        if (tile.is_empty()) return true;
        int factor = new_size / old_size;
        if (tile(l.blocks()[0].dim_idx) % factor != 0) return false;
        return true;
    };

    auto strides_ok = [&](const layout_t &l) {
        for (int i = 1; i < int(l.blocks().size()); i++) {
            auto &b = l.blocks()[i];
            if (b.block * old_size % new_size != 0) return false;
        }
        return true;
    };

    while (new_size > old_size) {
        bool ok = true;
        ok &= (tile_ok(src) && tile_ok(dst));
        ok &= (strides_ok(src) && strides_ok(dst));
        if (ok) {
            if (do_update) {
                src = src.reinterpret(type_t::s(new_size * 8));
                dst = dst.reinterpret(type_t::s(new_size * 8));
            }
            if (new_size_out) *new_size_out = new_size;
            return true;
        }
        new_size /= 2;
    }
    return false;
}

// Implementation of GRF reorder between 2D dense layouts.
// Requirements for A -> B reorder:
// - A and B must have the same data type
// - Layouts must be 2D and dense
// Reorder may require several steps, in this case a temporary buffer T is
// allocated. For example: A -> T -> B or A -> B -> T -> B
class reorder_2d_impl_t {
public:
    reorder_2d_impl_t(
            ngen::HW hw, const layout_t &src_layout, const layout_t &dst_layout)
        : hw_(hw), src_(src_layout), dst_(dst_layout) {
        ir_assert(src_.type() == dst_.type());
        tile_ = find_2d_tile(src_, dst_);
    }

    const tensor_t &tile() const { return tile_; }

    template <typename GeneratorT>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const grf_permutator_t &grf_perm, const ngen::RegData &src_rd,
            const ngen::RegData &dst_rd) {
        int a_idx, b_idx;
        int tile_a, tile_b;
        tile_to_2d_dims(tile_, a_idx, b_idx, tile_a, tile_b);

        // Convert src/dst to 2D layouts.
        dim_assignment_t to_ab(src_.ndims(), 2);
        to_ab.assign(a_idx, 0);
        to_ab.assign(b_idx, 1);
        auto src_ab = to_ab.map(src_);
        auto dst_ab = to_ab.map(dst_);

        // Find minimal cost reorder path between layouts.
        auto path = find_min_cost_path(hw_, src_ab, dst_ab, tile_a, tile_b);

        // Allocate a temporary GRF buffer if needed.
        ngen::GRFRange tmp;
        if (path.size() > 1) {
            const int grf_size = ngen::GRF::bytes(hw_);
            tmp = scope.alloc_range(utils::div_up(dst_ab.size(), grf_size));
        }

        // Iterate through found reorders.
        auto *prev_layout = &src_ab;
        auto prev_rd = src_rd;
        int path_len = int(path.size());
        auto &orig_type = src_ab.type();
        for (int i = 0; i < path_len; i++) {
            auto &step = path[i];
            auto &tile = step.tile;
            auto &type = step.type;
            auto *next_layout = &step.layout;

            // x -> y reorder.
            auto x = prev_layout->map(tile).reinterpret(type);
            auto y = next_layout->map(tile).reinterpret(type);

            bool use_dst = ((path_len - i) % 2 == 1);
            auto next_rd = (use_dst ? dst_rd : tmp[0].retype(to_ngen(type)));
            auto &x_blocks = x.blocks();
            auto &y_blocks = y.blocks();
            ir_assert(x_blocks.size() <= 1);
            ir_assert(y_blocks.size() <= 1);
            int x_stride = (x_blocks.empty() ? 1 : int(x_blocks[0].stride));
            int y_stride = (y_blocks.empty() ? 1 : int(y_blocks[0].stride));
            int width = int(tile.elems()) * orig_type.size() / type.size();
            next_layout->for_each_tile(
                    tile, [&](const std::vector<dim_t> &start) {
                        int prev_off = int(prev_layout->offset_in_bytes(start));
                        int next_off = int(next_layout->offset_in_bytes(start));
                        auto x_sub = ngen_subregister(
                                hw_, prev_rd, prev_off, to_ngen(type));
                        auto y_sub = ngen_subregister(
                                hw_, next_rd, next_off, to_ngen(type));
                        emit_reorder_1d_tile(hw_, host, scope, grf_perm, width,
                                x_sub, x_stride, y_sub, y_stride);
                    });
            prev_layout = next_layout;
            prev_rd = next_rd;
        }
    }

private:
    // Helper class to incrementally increase a sub-layout of the given layout.
    // One step - adding the minimal factor of the next remaining block. Used
    // to find the minimal tile between two layouts that is innermost for both
    // layouts.
    struct layout_iterator_t {
        layout_iterator_t(const layout_t &l) : l(l), block_idx(-1), block(1) {}

        bool has_next() const {
            dim_t b = block;
            int b_idx = block_idx;
            while (b == 1) {
                b_idx++;
                if (b_idx >= int(l.blocks().size())) return false;
                b = int(l.blocks()[b_idx].block);
            }
            return true;
        }

        layout_iterator_t &operator++() {
            ir_assert(has_next());
            while (block == 1) {
                block_idx++;
                block = int(l.blocks()[block_idx].block);
            }
            // Find smallest factor.
            for (int factor = 2; factor <= int(block); factor++) {
                if (block % factor == 0) {
                    block /= factor;
                    return *this;
                }
            }

            ir_error_not_expected();
            return *this;
        }

        tensor_t tile() const {
            std::vector<dim_t> dims(l.ndims(), 1);
            for (int i = 0; i <= block_idx; i++) {
                auto &b = l.blocks()[i];
                int b_block = b.block;
                if (i == block_idx) b_block /= block;
                dims[b.dim_idx] *= b_block;
            }
            return tensor_t(dims);
        }

        const layout_t &l;

        int block_idx;
        dim_t block;
    };

    // Represents 2D reorder corresponding to (a x b) tile.
    struct edge_t {
        edge_t() = default;
        edge_t(int idx, int a, int b) : idx(idx), a(a), b(b) {}

        tensor_t tile() const { return tensor_t({a, b}); }

        std::string str() const {
            std::ostringstream oss;
            oss << "edge(idx = " << idx << ", a = " << a << ", b = " << b
                << ")";
            return oss.str();
        }

        int idx; // Identifier of the edge.
        int a = 0, b = 0; // Specify tile (a x b).
    };

    // Represents GRF layout between edges-reorders.
    struct vertex_t {
        vertex_t(ngen::HW hw, int idx, const layout_t &layout)
            : hw(hw), idx(idx), layout(layout) {}

        std::string str() const {
            std::ostringstream oss;
            oss << "vertex(idx = " << idx << ", layout = " << layout << ")";
            return oss.str();
        }

        void set_edges(const std::vector<edge_t> &edges) {
            adj_edge_type_masks.resize(edges.size());
            int type_size = layout.type().size();
            for (int i = 0; i < int(edges.size()); i++) {
                auto &e = edges[i];
                auto tile = e.tile();
                int max_type_size;
                bool ok = try_reinterpret_to_wider_type(
                        layout, layout, tile, false, &max_type_size);
                if (!ok) max_type_size = type_size;
                int from = math::ilog2q(type_size);
                int to = math::ilog2q(max_type_size);
                for (int j = from; j <= to; j++) {
                    type_t type = type_t::u(8 << j);
                    if (can_reorder(tile, type))
                        adj_edge_type_masks[i] |= (1 << j);
                }
            }
        }

        void add_neighbor(const vertex_t *v) { adj_vertices.push_back(v); }

        bool is_neighbor(const vertex_t &v) const {
            for (auto *n : adj_vertices)
                if (n == &v) return true;
            return false;
        }

        // Check the following limitations:
        // - Assume at most one block (maybe with non-dense stride)
        // - Horizontal stride must be <= 4 for GRF region
        // - GRF region can't span more than 2 registers
        bool can_reorder(const tensor_t &tile, const type_t &type) const {
            auto ab_layout = layout.map(tile).reinterpret(type);
            int nblocks = int(ab_layout.blocks().size());
            if (nblocks == 0) return true;
            if (nblocks > 1) return false;
            auto &last = ab_layout.blocks().back();
            int max_stride = int(last.stride * last.block);
            if (last.stride > 4) return false;
            int max_stride_bytes = max_stride * type.size();
            int grf_size = ngen::GRF::bytes(hw);
            if (max_stride_bytes > 2 * grf_size) return false;
            return true;
        }

        // Finds the minimal cost of reordering from this vertex to vertex v.
        int cost(const vertex_t &v, const std::vector<edge_t> &edges,
                edge_t &min_edge, type_t &min_type) const {
            int min_cost = std::numeric_limits<int>::max();
            for (int i = 0; i < int(edges.size()); i++) {
                type_t i_min_type;
                int new_cost = cost(edges[i], v, i_min_type);
                if (new_cost < min_cost) {
                    min_cost = new_cost;
                    min_edge = edges[i];
                    min_type = i_min_type;
                }
            }
            return min_cost;
        }

        // Finds the minimal cost of reordering from this vertex to vertex `v`
        // through edge `e`. If the reorder is possible, `type` contains the
        // reorder type with the minimal cost.
        int cost(const edge_t &e, const vertex_t &v, type_t &type) const {
            uint32_t mask = (adj_edge_type_masks[e.idx]
                    & v.adj_edge_type_masks[e.idx]);
            if (mask == 0) return std::numeric_limits<int>::max();
            int cur_size = layout.type().size();
            int cur_cost = layout.elems() / (e.a * e.b);
            int min_log_bytes = math::ilog2q(cur_size);
            int max_log_bytes = 3;
            int min_cost = std::numeric_limits<int>::max();
            for (int i = min_log_bytes; i <= max_log_bytes; i++) {
                if ((mask & (1 << i)) == 0) continue;
                min_cost = cur_cost;
                type = type_t::u(8 << i);
                break;
            }
            return min_cost;
        }

        ngen::HW hw;
        int idx; // Identifier of the vertex.
        layout_t layout; // Layout of the vertex.
        // Specifies a bitmask for every edge: if adj_edge_type_masks[E_idx]
        // has b-th bit set then this vertex can be reordered through E edge
        // using the data type with size 2^b bytes.
        std::vector<uint32_t> adj_edge_type_masks;
        std::vector<const vertex_t *> adj_vertices; // Adjacent vertices.
    };

    // Represents a reorder step.
    struct reorder_step_t {
        reorder_step_t() = default;
        reorder_step_t(const layout_t &layout, const tensor_t &tile,
                const type_t &type)
            : layout(layout), tile(tile), type(type) {}

        layout_t layout; // Destination layout.
        tensor_t tile; // Tile corresponding to one instruction.
        type_t type; // Registers should be reinterpreted to `type` for reorder.
    };

    // Returns the biggest common 2D tile that is innermost for both layouts.
    static tensor_t find_2d_tile(const layout_t &a, const layout_t &b) {
        std::vector<dim_t> tile_dims(a.ndims(), 1);
        if (a.blocks().empty() || b.blocks().empty())
            return tensor_t(tile_dims);

        auto non_one_ndims = [](const tensor_t &t) {
            int ret = 0;
            for (dim_t d : t.dims())
                ret += (d != 1 ? 1 : 0);
            return ret;
        };

        layout_iterator_t a_it(a);
        layout_iterator_t b_it(b);

        tensor_t max_tile;
        for (;;) {
            auto a_tile = a_it.tile();
            auto b_tile = b_it.tile();
            if (non_one_ndims(a_tile) > 2 || non_one_ndims(b_tile) > 2) break;
            dim_t a_elems = a_tile.elems();
            dim_t b_elems = b_tile.elems();
            if (a_tile.is_equal(b_tile)) {
                max_tile = a_tile;
                if (!a_it.has_next() || !b_it.has_next()) break;
                ++a_it;
                ++b_it;
            } else if (a_elems <= b_elems) {
                if (!a_it.has_next()) break;
                ++a_it;
            } else {
                if (!b_it.has_next()) break;
                ++b_it;
            }
        }
        return max_tile;
    }

    // Extracts dimension sizes and their indices from a multidimensional
    // tensor.
    static void tile_to_2d_dims(
            const tensor_t &tile, int &a_idx, int &b_idx, int &a, int &b) {
        a_idx = -1;
        b_idx = -1;
        for (int i = 0; i < tile.ndims(); i++) {
            if (tile.dims()[i] == 1) continue;
            if (a_idx == -1) {
                a_idx = i;
                continue;
            }
            if (b_idx == -1) {
                b_idx = i;
                continue;
            }
            ir_error_not_expected();
        }

        for (int i = 0; i < tile.ndims(); i++) {
            if (utils::one_of(i, a_idx, b_idx)) continue;
            if (a_idx == -1) {
                a_idx = i;
                continue;
            }
            if (b_idx == -1) {
                b_idx = i;
                continue;
            }
        }

        if (a_idx > b_idx) std::swap(a_idx, b_idx);

        a = tile.dims()[a_idx];
        b = tile.dims()[b_idx];
    }

    // Finds the optimal sequence of reorders between src and dst layouts.
    static std::vector<reorder_step_t> find_min_cost_path(ngen::HW hw,
            const layout_t &src, const layout_t &dst, int tile_a, int tile_b) {
        // Create all possible edges - 2D reorders.
        std::vector<edge_t> edges;
        for (int a = 1; a <= tile_a; a *= 2) {
            for (int b = 1; b <= tile_b; b *= 2) {
                int idx = int(edges.size());
                edges.emplace_back(idx, a, b);
            }
        }

        int nedges = int(edges.size());

        // Create all possible layouts for tile_a x tile_b tensor.
        std::vector<vertex_t> vertices;
        std::vector<std::vector<std::pair<int, uint32_t>>> edge_vertices(
                nedges);
        auto all_layouts = generate_all_layouts(src.type(), tile_a, tile_b);
        for (auto &l : all_layouts) {
            // Skip if too many blocks.
            if (l.blocks().size() > 4) continue;
            int v_idx = int(vertices.size());
            vertices.emplace_back(hw, v_idx, l);
            auto &v = vertices.back();
            // Pass all known reorders, the vertex/layout will filter out
            // incompatible reorders.
            v.set_edges(edges);
            // Store all vertices adjacent to a specific edge.
            for (int i = 0; i < nedges; i++) {
                uint32_t mask = v.adj_edge_type_masks[i];
                if (mask != 0) edge_vertices[i].emplace_back(v_idx, mask);
            }
        }

        // Find neighbors between all vertices.
        int nvertices = int(vertices.size());
        for (int i = 0; i < nvertices; i++) {
            auto &v = vertices[i];
            for (int j = 0; j < nedges; j++) {
                uint32_t mask = v.adj_edge_type_masks[j];
                if (mask != 0) {
                    for (auto &idx_mask : edge_vertices[j]) {
                        int v_idx = idx_mask.first;
                        if (v_idx == i) continue;
                        uint32_t common_mask = (mask
                                & vertices[v_idx].adj_edge_type_masks[j]);
                        if (common_mask != 0) v.add_neighbor(&vertices[v_idx]);
                    }
                }
            }
        }

        // Identify source and destination vertices.
        int src_idx = -1;
        int dst_idx = -1;
        for (int i = 0; i < nvertices; i++) {
            auto &v = vertices[i];
            if (src_idx == -1
                    && v.layout.is_strictly_equal(
                            src, /*compare_offset=*/false))
                src_idx = i;
            if (dst_idx == -1
                    && v.layout.is_strictly_equal(
                            dst, /*compare_offset=*/false))
                dst_idx = i;
        }

        ir_assert(src_idx != -1);
        ir_assert(dst_idx != -1);

        // Layouts are the same, just copy.
        if (src_idx == dst_idx) {
            auto &v = vertices[src_idx];
            edge_t min_edge;
            type_t min_type;
            v.cost(v, edges, min_edge, min_type);
            reorder_step_t step(v.layout, min_edge.tile(), min_type);
            return {step};
        }

        // Dijkstra's algorithm, find the minimal cost path between src and
        // dst. Use the number of instructions to estimate the cost.
        int inf_cost = std::numeric_limits<int>::max();
        std::vector<int> cost(nvertices, inf_cost);
        std::vector<int> prev(nvertices);
        std::vector<reorder_step_t> reorder_steps(nvertices);
        std::vector<bool> seen(nvertices, false);
        cost[src_idx] = 0;
        for (int i = 0; i < nvertices; i++) {
            int min_idx = -1;
            int min_cost = inf_cost;
            for (int j = 0; j < nvertices; j++) {
                if (seen[j]) continue;
                if (cost[j] < min_cost) {
                    min_idx = j;
                    min_cost = cost[j];
                }
            }
            seen[min_idx] = true;
            auto &v_min = vertices[min_idx];
            for (auto *v : v_min.adj_vertices) {
                edge_t min_edge;
                type_t min_type;
                int new_cost = cost[min_idx]
                        + v_min.cost(*v, edges, min_edge, min_type);
                if (new_cost < cost[v->idx]) {
                    cost[v->idx] = new_cost;
                    prev[v->idx] = min_idx;
                    reorder_steps[v->idx] = reorder_step_t(
                            v->layout, min_edge.tile(), min_type);
                }
            }
        }

        // Sanity check, ensure the reorder sequence is not too long.
        int max_cost = 256;
        ir_assert(cost[dst_idx] <= max_cost);
        MAYBE_UNUSED(max_cost);

        // Restore the shortest reorder path.
        std::vector<reorder_step_t> ret;
        int idx = dst_idx;
        while (idx != src_idx) {
            ret.push_back(reorder_steps[idx]);
            idx = prev[idx];
        }
        std::reverse(ret.begin(), ret.end());
        return ret;
    }

    // Returns all possible layouts for (a x b) tensor.
    static std::vector<layout_t> generate_all_layouts(
            const type_t &type, int a, int b) {
        std::vector<layout_t> ret;
        std::vector<block_t> blocks;
        generate_all_layouts_impl(ret, blocks, type, a, b, 1);
        return ret;
    }

    static void generate_all_layouts_impl(std::vector<layout_t> &layouts,
            std::vector<block_t> &blocks, const type_t &type, int a, int b,
            int stride) {
        if (a == 1 && b == 1) {
            layouts.emplace_back(type, 2, 0, blocks);
            return;
        }
        bool iterate_a = true;
        bool iterate_b = true;

        // Avoid repeating indices to keep only unique layouts.
        if (!blocks.empty()) {
            auto &last = blocks.back();
            iterate_a &= (last.dim_idx != 0);
            iterate_b &= (last.dim_idx != 1);
        }

        if (iterate_a) {
            for (int a_blk = 2; a_blk <= a; a_blk++) {
                if (a % a_blk != 0) continue;
                blocks.emplace_back(0, a_blk, stride);
                generate_all_layouts_impl(
                        layouts, blocks, type, a / a_blk, b, stride * a_blk);
                blocks.pop_back();
            }
        }
        if (iterate_b) {
            for (int b_blk = 2; b_blk <= b; b_blk++) {
                if (b % b_blk != 0) continue;
                blocks.emplace_back(1, b_blk, stride);
                generate_all_layouts_impl(
                        layouts, blocks, type, a, b / b_blk, stride * b_blk);
                blocks.pop_back();
            }
        }
    }

    ngen::HW hw_;

    layout_t src_;
    layout_t dst_;

    tensor_t tile_;
};

ngen::Subregister get_subregister(ngen::HW hw, const grf_permutator_t &grf_perm,
        const ngen::Subregister &base_sub, int off, int width, int stride_bytes,
        ngen::DataType type = ngen::DataType::invalid) {
    if (type == ngen::DataType::invalid) type = base_sub.getType();
    int off_bytes = off * stride_bytes;
    auto rd = ngen_reg_data(hw, base_sub, off_bytes, type, 1, 0);
    auto ret = ngen::Subregister(rd, rd.getOffset(), rd.getType());
    if (grf_perm.is_empty()) return ret;

    // Ensure no GRF boundary crossing.
    int off0 = ret.getByteOffset();
    int off1 = off0 + stride_bytes * (width - 1);

    int grf_size = ngen::GRF::bytes(hw);

    int base0 = ret.getBase();
    int base1 = base0 + off1 / grf_size;

    int new_base = grf_perm.map(base0);

    for (int i = 1; i <= base1 - base0; i++) {
        ir_assert(grf_perm.map(base0 + i) == new_base + i)
                << "Unexpected mapping.";
    }

    ret.setBase(new_base);
    return ret;
}

// Performs 1D reorder, possibly with strides and type conversion.
template <typename GeneratorT>
void emit_reorder_1d_tile(ngen::HW hw, GeneratorT *host,
        ngen_register_scope_t &scope, const grf_permutator_t &grf_perm,
        int width, const ngen::Subregister &_src, int src_stride,
        const ngen::Subregister &_dst, int dst_stride) {
    auto src = _src;
    auto dst = _dst;
    ngen::DataType src_type = src.getType();
    ngen::DataType dst_type = dst.getType();
    // Replace (float -> float) by (int -> int) as word/dword moves have less
    // restrictions.
    if (src_type == dst_type && ngen_is_xf(src_type)) {
        src_type = to_ngen(type_t::u(ngen::getBytes(src_type) * 8));
        dst_type = src_type;
        src = src.reinterpret(0, src_type);
        dst = dst.reinterpret(0, dst_type);
    }

    int src_stride_bytes = src_stride * ngen::getBytes(src_type);
    int dst_stride_bytes = dst_stride * ngen::getBytes(dst_type);
    bool dst_b = ngen_is_b(dst_type);
    bool dst_bf = (dst_type == ngen::DataType::bf);
    bool dst_d = ngen_is_dw(dst_type) || (dst_type == ngen::DataType::f);
    bool dst_f = (dst_type == ngen::DataType::f);
    bool dst_hf = (dst_type == ngen::DataType::hf);
    bool src_b = ngen_is_b(src_type);
    bool src_hf = (src_type == ngen::DataType::hf);
    bool src_bf = (src_type == ngen::DataType::bf);
    bool src_d = ngen_is_dw(src_type) || (src_type == ngen::DataType::f);
    bool src_f = (src_type == ngen::DataType::f);
    bool f_to_xf = (src_f && (dst_bf || dst_hf));

    auto get_step = [&]() {
        int step = (width < 16 ? 8 : 16);

        // f32 -> bf16 or f32 -> f16: SIMD16 does not support mixed mode move.
        if (f_to_xf) step = std::min(step, 8);

        // Max supported stride is 4.
        if (src_stride > 4 || dst_stride > 4) step = 1;

        return step;
    };

    // bf16 -> f32:
    // - bf16 must be packed: use left shift instead.
    if (src_bf && dst_f) {
        int step = get_step();
        for (int i = 0; i < width; i += step) {
            int esize = std::min(step, width - i);
            ir_assert(math::is_pow2(esize));
            auto s = get_subregister(hw, grf_perm, src, i, esize,
                    src_stride_bytes, ngen::DataType::uw);
            auto d = get_subregister(hw, grf_perm, dst, i, esize,
                    dst_stride_bytes, ngen::DataType::ud);
            host->eshl(
                    esize, d(dst_stride), s(src_stride), ngen::Immediate(16));
        }
        return;
    }

    // f32/f16/s32 -> s8/u8 and s8/u8 -> f32/s32
    // - Use saturation
    // - s8/u8 must be DW-strided: use temporary
    if ((src_d && dst_b) || (src_hf && dst_b) || (src_b && dst_d)) {
        if (dst_d) ir_assert(dst_stride_bytes == 4);
        if (src_d) ir_assert(src_stride_bytes == 4);
        if (src_hf) ir_assert(src_stride_bytes == 2);
        if (dst_b) ir_assert(utils::one_of(dst_stride_bytes, 1, 4));
        if (src_b) ir_assert(utils::one_of(src_stride_bytes, 1, 4));
        int step = get_step();
        const int grf_size = ngen::GRF::bytes(hw);
        auto tmp = scope.alloc_range(
                utils::div_up(int(step * sizeof(uint32_t)), grf_size));
        for (int i = 0; i < width; i += step) {
            int esize = std::min(step, width - i);
            ir_assert(math::is_pow2(esize));

            auto s = get_subregister(
                    hw, grf_perm, src, i, esize, src_stride_bytes);
            auto d = get_subregister(
                    hw, grf_perm, dst, i, esize, dst_stride_bytes);
            if (src_d || src_hf) {
                // d -> b.
                if (dst_stride_bytes == 1) {
                    auto t = tmp[0].retype(dst_type)[0](4);
                    host->emov(esize | host->sat, t, s(1));
                    host->emov(esize, d(1), t);
                } else {
                    host->emov(esize | host->sat, d(4), s(1));
                }
            } else {
                // b -> d.
                // hf -> d.
                if (esize == 1) {
                    // Direct x8 -> x32 scalar cast is not always
                    // supported. Use intermediate cast to s16.
                    auto t = tmp[0].uw();
                    host->emov(esize, t, s);
                    host->emov(esize, d, t);
                } else if (src_stride_bytes == 1) {
                    auto t = tmp[0].retype(src_type)[0](4);
                    host->emov(esize, t, s(1));
                    host->emov(esize, d(1), t);
                } else {
                    host->emov(esize, d(1), s(4));
                }
            }
        }
        return;
    }

    // Perform regular move.
    int step = get_step();
    for (int i = 0; i < width; i += step) {
        int esize = std::min(step, width - i);
        ir_assert(math::is_pow2(esize));
        ir_assert(!f_to_xf || utils::one_of(i % 16, 0, 8))
                << "Not always supported in HW.";
        auto s = get_subregister(hw, grf_perm, src, i, esize, src_stride_bytes);
        auto d = get_subregister(hw, grf_perm, dst, i, esize, dst_stride_bytes);
        host->emov(esize, d(dst_stride), s(src_stride));
    }
}

class reorder_impl_t {
public:
    reorder_impl_t(ngen::HW hw, const reorder_t &reorder,
            const grf_permutator_t &grf_perm)
        : hw_(hw)
        , src_layout_(reorder.src_layout)
        , dst_layout_(reorder.dst_layout)
        , grf_perm_(grf_perm) {
        try_reinterpret_to_wider_type(src_layout_, dst_layout_);

        // Pure bf moves are not supported.
        if (utils::everyone_is(
                    type_t::bf16(), src_layout_.type(), dst_layout_.type())) {
            src_layout_ = src_layout_.retype(type_t::u16());
            dst_layout_ = dst_layout_.retype(type_t::u16());
        }

        with_permutation_ = !grf_perm_.is_empty();
    }

    template <typename GeneratorT>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const ngen::RegData &_src, const ngen::RegData &_dst) {
        if (with_permutation_) {
            ir_assert(_src.getOffset() == 0)
                    << "Must be aligned to GRF boundary.";
            ir_assert(_dst.getOffset() == 0)
                    << "Must be aligned to GRF boundary.";
            grf_perm_.set_grf_base(_src.getBase());
        }

        auto &src_type = src_layout_.type();
        auto &dst_type = dst_layout_.type();
        auto src = ngen_reg_data(hw_, _src, 0, to_ngen(src_type), 1);
        auto dst = ngen_reg_data(hw_, _dst, 0, to_ngen(dst_type), 1);

        if (try_emit_2d(host, scope, src, dst)) return;
        emit_1d(host, scope, src, dst);
    }

private:
    template <typename GeneratorT>
    void emit_1d(GeneratorT *host, ngen_register_scope_t &scope,
            const ngen::RegData &src_rd, const ngen::RegData &dst_rd) {
        int src_stride;
        int dst_stride;
        auto tile = find_max_tile_with_fixed_stride(
                src_layout_, dst_layout_, src_stride, dst_stride);

        int tile_elems = int(tile.elems());
        auto &src_type = src_layout_.type();
        auto &dst_type = dst_layout_.type();
        dst_layout_.for_each_tile(tile, [&](const std::vector<dim_t> &start) {
            int src_off = int(src_layout_(start) * src_type.size());
            int dst_off = int(dst_layout_(start) * dst_type.size());
            auto sub_src = ngen_subregister(hw_, src_rd, src_off);
            auto sub_dst = ngen_subregister(hw_, dst_rd, dst_off);

            emit_reorder_1d_tile(hw_, host, scope, grf_perm_, tile_elems,
                    sub_src, src_stride, sub_dst, dst_stride);
        });
    }

    static tensor_t find_max_2d_dense_tile(const layout_t &a_layout,
            const layout_t &b_layout, dim_t _max_elems) {
        dim_t max_elems = _max_elems;
        for (auto &l : {&a_layout, &b_layout}) {
            dim_t stride = 1;
            dim_t elems = 1;
            int non_one_ndims = 0;
            std::vector<bool> seen(l->ndims());
            for (auto &b : l->blocks()) {
                // Tile is not dense anymore, break.
                if (dim_t(b.stride) != stride) break;
                stride = dim_t(b.stride) * b.block;

                if (b.block == 1) continue;
                if (!seen[b.dim_idx]) {
                    seen[b.dim_idx] = true;
                    non_one_ndims++;
                }
                // Tile is not 2D anymore, break.
                if (non_one_ndims > 2) break;
                elems *= b.block;
            }
            max_elems = std::min(max_elems, elems);
        }
        return a_layout.split_into_max_tile(max_elems, /*is_dense=*/true);
    }

    template <typename GeneratorT>
    bool try_emit_2d(GeneratorT *host, ngen_register_scope_t &scope,
            const ngen::RegData &src_rd, const ngen::RegData &dst_rd) {
        if (src_layout_.type() != dst_layout_.type()) return false;
        if (!src_layout_.is_dense()) return false;
        if (!dst_layout_.is_dense()) return false;

        int max_tile_size = 512;
        int max_tile_elems = max_tile_size / src_layout_.type().size();
        auto tile = find_max_2d_dense_tile(
                src_layout_, dst_layout_, max_tile_elems);

        // Couldn't find tile, 2D reorder is not supported.
        if (tile.is_empty()) return false;

        auto src_tile_layout = src_layout_.map(tile);
        auto dst_tile_layout = dst_layout_.map(tile);
        if (!dst_tile_layout.is_dense()) return false;

        // Set layout offset to 0 since the offset is handled by fixing up the
        // register input to try_emit_2d_impl
        src_tile_layout.set_offset(0);
        dst_tile_layout.set_offset(0);

        bool ok = true;
        src_layout_.for_each_tile(tile, [&](const std::vector<dim_t> &start) {
            auto src_off = src_layout_.offset_in_bytes<dim_t>(start);
            auto dst_off = dst_layout_.offset_in_bytes<dim_t>(start);
            auto src_tile_rd = ngen_reg_data(
                    hw_, src_rd, int(src_off), ngen::DataType::invalid, 1);
            auto dst_tile_rd = ngen_reg_data(
                    hw_, dst_rd, int(dst_off), ngen::DataType::invalid, 1);

            ngen_register_scope_t tile_scope(scope.register_allocator());
            ok &= try_emit_2d_impl(host, tile_scope, src_tile_layout,
                    dst_tile_layout, src_tile_rd, dst_tile_rd);
        });
        return ok;
    }

    template <typename GeneratorT>
    bool try_emit_2d_impl(GeneratorT *host, ngen_register_scope_t &scope,
            const layout_t &src_layout, const layout_t &dst_layout,
            const ngen::RegData &src_rd, const ngen::RegData &dst_rd) {
        // Try to allocate/release a temporary buffer to avoid out_of_registers
        // exception.
        const int grf_size = ngen::GRF::bytes(hw_);
        auto dummy = scope.try_alloc_range(
                utils::div_up(dst_layout.size(), grf_size));
        if (dummy.isInvalid()) return false;

        // Allocation succeeded, can proceed further.
        scope.safeRelease(dummy);

        reorder_2d_impl_t r(hw_, src_layout, dst_layout);
        int tile_elems = int(r.tile().elems());
        if (tile_elems < 16 || tile_elems > 512) return false;

        r.emit(host, scope, grf_perm_, src_rd, dst_rd);
        return true;
    }

    static tensor_t find_max_tile_with_fixed_stride(const layout_t &src,
            const layout_t &dst, int &src_stride, int &dst_stride) {
        // 1. Split layouts to have aligned blocks.
        auto a = src;
        auto b = dst;
        layout_t::align_layouts(a, b);

        // 2. Find the max innermost tile.
        auto a_blocks = a.blocks();
        auto b_blocks = b.blocks();

        std::vector<dim_t> tile_dims(a.ndims(), 1);
        src_stride = (a_blocks.empty() ? 1 : int(a_blocks[0].stride));
        dst_stride = (b_blocks.empty() ? 1 : int(b_blocks[0].stride));
        int src_cur_stride = src_stride;
        int dst_cur_stride = dst_stride;

        int min_blocks = int(std::min(a_blocks.size(), b_blocks.size()));
        for (int i = 0; i < min_blocks; i++) {
            auto &ab = a_blocks[i];
            auto &bb = b_blocks[i];
            if (ab.dim_idx != bb.dim_idx || ab.block != bb.block) break;

            // Strides are supported for the innermost block only.
            if (src_cur_stride != int(ab.stride)) break;
            if (dst_cur_stride != int(bb.stride)) break;

            src_cur_stride = int(ab.block * ab.stride);
            dst_cur_stride = int(bb.block * bb.stride);
            tile_dims[ab.dim_idx] *= ab.block;
        }
        return tensor_t(tile_dims);
    }

    ngen::HW hw_;
    layout_t src_layout_;
    layout_t dst_layout_;
    grf_permutator_t grf_perm_;
    bool with_permutation_ = false;
};

class reduce_impl_t {
public:
    reduce_impl_t(ngen::HW hw, const reduce_t &reduce)
        : hw_(hw)
        , src_layout_(reduce.src_layout)
        , dst_layout_(reduce.dst_layout) {}

    template <typename GeneratorT>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const ngen::RegData &_src_rd, const ngen::RegData &_dst_rd) {
        auto &src_type = src_layout_.type();
        auto &dst_type = dst_layout_.type();
        auto src_rd = ngen_reg_data(hw_, _src_rd, 0, to_ngen(src_type), 1);
        auto dst_rd = ngen_reg_data(hw_, _dst_rd, 0, to_ngen(dst_type), 1);

        tensor_t tile = find_1d_tile();
        src_layout_.for_each_tile(
                tile, [&](const std::vector<dim_t> &src_start) {
                    auto dst_start = src_start;
                    for (int i = 0; i < dst_layout_.ndims(); i++) {
                        if (dst_layout_.dims()[i] == 1) dst_start[i] = 0;
                    }
                    int src_off = int(src_layout_(src_start) * src_type.size());
                    int dst_off = int(dst_layout_(dst_start) * dst_type.size());
                    auto sub_src = ngen_subregister(hw_, src_rd, src_off);
                    auto sub_dst = ngen_subregister(hw_, dst_rd, dst_off);
                    host->add(int(tile.elems()), sub_dst(1), sub_dst(1),
                            sub_src(1));
                });
    }

private:
    tensor_t find_1d_tile() const {
        auto a = src_layout_;
        auto b = src_layout_;
        layout_t::align_layouts(a, b);

        ir_assert(!a.blocks().empty());
        ir_assert(!b.blocks().empty());

        auto &a0 = a.blocks()[0];
        auto &b0 = b.blocks()[0];

        ir_assert(a0.is_equal(b0)) << "Incompatible layouts for reduction.";
        ir_assert(dim_t(a0.stride) == 1) << "Reduction is not supported.";
        ir_assert(utils::one_of(a0.block, 8, 16))
                << "Reduction is not supported.";

        std::vector<dim_t> tile_dims(src_layout_.ndims(), 1);
        tile_dims[a0.dim_idx] = a0.block;

        return tensor_t(tile_dims);
    }

    ngen::HW hw_;
    layout_t src_layout_;
    layout_t dst_layout_;
};

// Lowers IR to nGEN.
template <ngen::HW hw>
class ir_to_ngen_t : public ir_visitor_t {
public:
    ir_to_ngen_t(conv_kernel_t<hw> *host, const expr_binding_t &expr_binding)
        : host_(host)
        , expr_binding_(expr_binding)
        , simd_size_(host->cfg_.simd_size) {}

    void _visit(const alloc_t *obj) override {
        auto scope = register_scope();
        bool do_alloc = (obj->kind == alloc_kind_t::grf);
        if (do_alloc) {
            int grf_size = ngen::GRF::bytes(hw);
            int regs = utils::div_up(obj->size, grf_size);
            ngen::Bundle bundle;
            auto *grf_attr = obj->attr.as_ptr<grf_alloc_attr_t>();
            if (grf_attr) bundle = to_ngen(grf_attr->bundle);
            auto reg_range = scope.alloc_range(regs, bundle);
            expr_binding_.bind(obj->buf, reg_range[0]);
        }
        visit(obj->body);
        if (do_alloc) expr_binding_.unbind(obj->buf);
    }

    void _visit(const for_t *obj) override {
        auto scope = register_scope();
        auto var_op = scope.alloc_sub(to_ngen(obj->var.type()));
        auto init_op = eval(obj->init, scope);
        auto bound_op = eval(obj->bound, scope);
        ngen::Label loop_label;
        host_->emov(1, var_op, init_op);
        expr_binding_.bind(obj->var, var_op);
        host_->mark(loop_label);
        visit(obj->body);
        host_->eadd(1, var_op, var_op, ngen::Immediate(1));
        host_->ecmp(1 | host_->lt | host_->f0[0], var_op, bound_op);
        host_->jmpi(1 | host_->f0[0], loop_label);
        expr_binding_.unbind(obj->var);
    }

    void _visit(const func_call_t *obj) override {
        auto scope = register_scope();
        auto &func = obj->func;
        if (func.is<dpas_t>()) {
            auto arg_ops = eval(obj->args, scope);
            dpas(func.as<dpas_t>(), arg_ops, obj->attr);
        } else if (func.is<mad_t>()) {
            auto arg_ops = eval(obj->args, scope);
            mad(scope, func.as<mad_t>(), arg_ops, obj->attr);
        } else if (func.is<reduce_t>()) {
            auto arg_ops = eval(obj->args, scope);
            ir_assert(obj->attr.is_empty()) << "Unexpected attribute.";
            reduce(scope, func.as<reduce_t>(), arg_ops);
        } else if (func.is<reorder_t>()) {
            auto arg_ops = eval(obj->args, scope);
            ir_assert(obj->attr.is_empty()) << "Unexpected attribute.";
            reorder(scope, func.as<reorder_t>(), reorder_t::arg_src_buf(obj),
                    arg_ops);
        } else if (func.is<send_t>()) {
            auto &send_func = func.as<send_t>();
            auto args = obj->args;
            auto &mem_buf = send_t::arg_mem_buf(args);
            auto &mask = send_t::arg_mask(args);
            // If all channels are disabled for writing, quick return.
            if (all_of(mask, expr_t(false))) {
                if (send_func.is_read()) {
                    auto reg_buf_op = eval(send_t::arg_reg_buf(args), scope);
                    zero_out_data_payload(send_func, send_func.eff_mask_count,
                            reg_buf_op.reg_data());
                }
                return;
            }
            // If all channels are enabled, do not use mask.
            if (all_of(mask, expr_t(true))) mask = expr_t();
            auto arg_ops = eval(args, scope);
            send(scope, func.as<send_t>(), mem_buf, arg_ops, obj->attr);
        } else if (func.is<eltwise_t>()) {
            auto &eltwise_func = func.as<eltwise_t>();
            auto arg_ops = eval(obj->args, scope);
            eltwise(scope, eltwise_func, arg_ops);
        } else if (func.is_equal(funcs::barrier_func())) {
            barrier(obj->attr);
        } else if (func.is_equal(funcs::barrier_wait_func())) {
            barrier_wait();
        } else if (func.is_equal(funcs::signal_func())) {
            signal(obj->attr);
        } else if (func.is_equal(funcs::slm_fence_func())) {
            slm_fence(obj->attr);
        } else {
            ir_error_not_expected() << object_t(obj);
        }
    }

    void _visit(const if_t *obj) override {
        ir_assert(obj->cond.is<shuffle_t>());
        ir_assert(obj->cond.as<shuffle_t>().elems() == simd_size_);

        bool has_else = !obj->else_body.is_empty();
        auto scope = register_scope();
        auto cond_op = eval(obj->cond, scope);

        ngen::Label l_else;
        ngen::Label l_endif;
        host_->if_(simd_size_ | cond_op.flag_register(),
                has_else ? l_else : l_endif, l_endif);
        visit(obj->body);
        if (has_else) {
            host_->else_(simd_size_, l_endif, l_endif);
            host_->mark(l_else);
            visit(obj->else_body);
        }
        host_->mark(l_endif);
        host_->endif(simd_size_);
    }

    void _visit(const let_t *obj) override {
        if (obj->value.is_empty()) {
            // External variable, must be already bound.
            ir_assert(expr_binding_.is_bound(obj->var))
                    << "Variable is not defined: " << obj->var;
            visit(obj->body);
            return;
        }

        auto scope = register_scope();
        if (is_const(obj->value) || is_shuffle_const(obj->value)
                || obj->var.type() != obj->value.type()) {
            auto &var_type = obj->var.type();
            auto var_op = scope.alloc_reg_data(var_type);
            eval(obj->value, scope, ngen_operand_t(var_op, var_type.elems()));
            expr_binding_.bind(obj->var, var_op);
        } else {
            auto value_op = eval(obj->value, scope);
            expr_binding_.bind(obj->var, value_op);
        }
        visit(obj->body);
        expr_binding_.unbind(obj->var);
    }

    void _visit(const store_t *obj) override {
        auto scope = register_scope();
        auto buf_op = eval(obj->buf, scope);
        auto off = to_cpp<int>(obj->off);
        auto mask_op = eval(obj->mask, scope);

        auto &type = obj->value.type();

        int stride;
        if (obj->has_default_stride()) {
            stride = 1;
        } else {
            ir_assert(obj->stride % type.scalar().size() == 0);
            stride = obj->stride / type.scalar().size();
        }

        ngen::InstructionModifier mod = type.elems();
        if (!mask_op.is_invalid()) mod |= mask_op.flag_register_mod();
        auto dst_rd = ngen_reg_data(hw, buf_op.reg_data(), off,
                to_ngen(type.scalar()), type.elems(), stride);
        ngen_operand_t dst(dst_rd, mod);
        eval(obj->value, scope, dst);
    }

private:
    ngen_register_scope_t register_scope() {
        return ngen_register_scope_t(host_->ra_);
    }

    void signal(const func_call_attr_t &attr) {
        ngen::InstructionModifier mod;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);
        host_->barriermsg(mod, host_->signal_header_);
    }

    void barrier_wait() { host_->barrierwait(); }

    void slm_fence(const func_call_attr_t &attr) {
        auto scope = register_scope();
        auto tmp = scope.alloc();
        ngen::InstructionModifier mod;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);

        const int dwords = ngen::GRF::bytes(hw) / sizeof(int32_t);
        host_->slmfence(mod, tmp, host_->r0);
        host_->template mov<int32_t>(dwords, host_->null, tmp);
    }

    void barrier(const func_call_attr_t &attr) {
        auto scope = register_scope();
        auto tmp = scope.alloc();
        ngen::InstructionModifier mod;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);

        const int dwords = ngen::GRF::bytes(hw) / sizeof(int32_t);
        host_->slmfence(mod, tmp, host_->r0);
        host_->template mov<int32_t>(dwords, host_->null, tmp);
        host_->barriermsg(mod, host_->signal_header_);
        host_->barrierwait();
    }

    void dpas(const dpas_t &dpas_func, const std::vector<ngen_operand_t> &args,
            const func_call_attr_t &attr) {
        auto dst = dpas_t::arg_dst(args).reg_data();
        auto src1 = dpas_t::arg_src1(args).reg_data();
        auto src2 = dpas_t::arg_src2(args).reg_data();

        ngen::RegData src0;
        auto &src0_op = dpas_t::arg_src0(args);
        if (src0_op.is_reg_data()) {
            src0 = src0_op.reg_data();
        } else {
            ir_assert(src0_op.is_immediate());
            ir_assert(to_cpp<int32_t>(src0_op.immediate()) == 0);
            src0 = host_->null;
        }

        dst.setType(to_ngen(dpas_func.dst_type));
        src0.setType(to_ngen(dpas_func.dst_type));
        src1.setType(to_ngen(dpas_func.src1_type));
        src2.setType(to_ngen(dpas_func.src2_type));
        ngen::InstructionModifier mod = simd_size_;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);
        if (dpas_func.is_dpasw) {
            host_->dpasw(mod, dpas_func.sdepth, dpas_func.rcount, dst, src0,
                    src1, src2);
        } else {
            host_->dpas(mod, dpas_func.sdepth, dpas_func.rcount, dst, src0,
                    src1, src2);
        }
    }

    void mad(ngen_register_scope_t &scope, const mad_t &mad_func,
            const std::vector<ngen_operand_t> &args,
            const func_call_attr_t &attr) {
        auto dst = mad_t::arg_dst(args).reg_data();
        auto src1 = mad_t::arg_src1(args).reg_data();
        auto src2 = mad_t::arg_src2(args).reg_data();

        ngen::RegData src0;
        auto &src0_op = mad_t::arg_src0(args);
        if (src0_op.is_reg_data()) {
            src0 = ngen_reg_data(hw, src0_op.reg_data(), 0,
                    to_ngen(mad_func.dst_type), mad_func.simd_size);
        } else {
            ir_assert(src0_op.is_immediate());
            ir_assert(to_cpp<int32_t>(src0_op.immediate()) == 0);
            src0 = host_->null;
            src0.setType(to_ngen(mad_func.dst_type));
        }

        dst = ngen_reg_data(
                hw, dst, 0, to_ngen(mad_func.dst_type), mad_func.simd_size);

        int src1_width = (mad_func.src1_stride == 0 ? 1 : mad_func.simd_size);
        int src2_width = (mad_func.src2_stride == 0 ? 1 : mad_func.simd_size);
        src1 = ngen_reg_data(hw, src1, 0, to_ngen(mad_func.src1_type),
                src1_width, mad_func.src1_stride);
        src2 = ngen_reg_data(hw, src2, 0, to_ngen(mad_func.src2_type),
                src2_width, mad_func.src2_stride);

        ngen::InstructionModifier mod = simd_size_;
        if (!attr.is_empty())
            mod = mod | to_ngen(attr.as<instruction_modifier_attr_t>().mod);

        if (src0.isNull()) {
            host_->mul(mod, dst, src1, src2);
        } else {
            ir_assert(dst.getByteOffset() == src0.getByteOffset())
                    << "dst/src0 must be aligned to the same GRF offset.";
            auto _src1 = src1;
            auto _src2 = src2;
            maybe_fix_mad_src(dst, _src1, _src2, mad_func, scope);
            host_->mad(mod, dst, src0, _src1, _src2);
        }
    }

    void maybe_fix_mad_src(const ngen::RegData &dst, ngen::RegData &src1,
            ngen::RegData &src2, const mad_t &mad_func,
            ngen_register_scope_t &scope) {
        int grf_size = ngen::GRF::bytes(hw);
        auto dst_off = dst.getByteOffset();
        auto src1_off = src1.getByteOffset();
        auto src2_off = src2.getByteOffset();

        // src1 must be aligned with dst if not broadcasted.
        if (mad_func.src1_stride != 0 && src1_off != dst_off) {
            ngen::RegData new_src1 = scope.alloc_range(
                    utils::div_up(mad_func.src1_size() + dst_off, grf_size))[0];
            new_src1 = ngen_reg_data(hw, new_src1, dst_off, src1.getType(),
                    mad_func.simd_size, mad_func.src1_stride);
            emit_reorder_1d_tile(hw, host_, scope, grf_permutator_t(hw),
                    mad_func.simd_size, get_subregister(src1),
                    mad_func.src1_stride, get_subregister(new_src1),
                    mad_func.src1_stride);
            src1 = new_src1;
        }

        // src2 must be aligned with dst if not broadcasted.
        if (mad_func.src2_stride != 0 && src2_off != dst_off) {
            ngen::RegData new_src2 = scope.alloc_range(
                    utils::div_up(mad_func.src2_size() + dst_off, grf_size))[0];
            new_src2 = ngen_reg_data(hw, new_src2, dst_off, src2.getType(),
                    mad_func.simd_size, mad_func.src2_stride);
            emit_reorder_1d_tile(hw, host_, scope, grf_permutator_t(hw),
                    mad_func.simd_size, get_subregister(src2),
                    mad_func.src2_stride, get_subregister(new_src2),
                    mad_func.src2_stride);
            src2 = new_src2;
        }
    }

    void reduce(ngen_register_scope_t &scope, const reduce_t &reduce_func,
            const std::vector<ngen_operand_t> &args) {
        auto &src_op = reduce_t::arg_src_buf(args);
        auto &dst_op = reduce_t::arg_dst_buf(args);

        reduce_impl_t reduce_impl(hw, reduce_func);
        reduce_impl.emit(host_, scope, src_op.reg_data(), dst_op.reg_data());
    }

    void reorder(ngen_register_scope_t &scope, const reorder_t &reorder_func,
            const expr_t &src_buf, const std::vector<ngen_operand_t> &args) {
        auto &src_op = reorder_t::arg_src_buf(args);
        auto &dst_op = reorder_t::arg_dst_buf(args);

        auto &src_buf_base
                = (src_buf.is<ptr_t>() ? src_buf.as<ptr_t>().base : src_buf);
        grf_permutator_t grf_perm(hw);
        if (reorder_func.grf_perm) {
            auto &perm_base = reorder_func.grf_perm->grf_buf_base();
            if (perm_base.is_equal(src_buf_base)) {
                grf_perm = *reorder_func.grf_perm;
            }
        }

        reorder_impl_t reorder_impl(hw, reorder_func, grf_perm);
        reorder_impl.emit(host_, scope, src_op.reg_data(), dst_op.reg_data());
    }

    void zero_out_data_payload(const send_t &send_func,
            const ngen::InstructionModifier &_mod, const ngen::RegData &rd) {
        bool is_per_slot = (send_func.mask_count() > 1);

        auto get_modifier = [&](int exec_size) {
            if (is_per_slot) {
                ir_assert(_mod.getExecSize() == exec_size);
                auto mod = _mod;
                mod = ~mod;
                mod.setSWSB({});
                return mod;
            }
            return ngen::InstructionModifier(exec_size);
        };

        int ud_size = sizeof(uint32_t);
        int send_size = send_func.register_size();
        int grf_size = ngen::GRF::bytes(hw);
        int step = (is_per_slot ? send_func.mask_count() * ud_size
                                : 2 * grf_size);
        for (int i = 0; i < send_size; i += step) {
            int exec_size;
            if (is_per_slot) {
                exec_size = send_func.eff_mask_count;
            } else {
                exec_size = std::min(step, send_size - i) / ud_size;
            }
            auto sub_rd_mov
                    = ngen_reg_data(hw, rd, i, ngen::DataType::f, exec_size);
            host_->emov(
                    get_modifier(exec_size), sub_rd_mov, ngen::Immediate(0.0f));
        }
    }

    void send(ngen_register_scope_t &scope, const send_t &send_func,
            const expr_t &mem_buf, const std::vector<ngen_operand_t> &args,
            const func_call_attr_t &attr) {
        send_impl_t spec_impl(hw, send_func);
        auto &mem_off_op = send_t::arg_mem_off(args);
        auto &reg_buf_op = send_t::arg_reg_buf(args);
        auto &mask_op = send_t::arg_mask(args);

        ngen::RegData mem_buf_rd;
        int surf_bti = -1;
        switch (send_func.address_model) {
            case ngen_proxy::AddressModel::ModelSLM: break;
            case ngen_proxy::AddressModel::ModelBTS: {
                auto &buf_name = mem_buf.as<var_t>().name;
                surf_bti = host_->getArgumentSurface(buf_name);
                break;
            }
            case ngen_proxy::AddressModel::ModelA64: {
                auto &mem_buf_op = send_t::arg_mem_buf(args);
                mem_buf_rd = mem_buf_op.reg_data();
                break;
            }
            default: ir_error_not_expected();
        }
        ngen::InstructionModifier mod = send_func.eff_mask_count;
        ir_assert(math::is_pow2(mod.getExecSize()));
        if (!attr.is_empty())
            mod |= to_ngen(attr.as<instruction_modifier_attr_t>().mod);
        if (!mask_op.is_invalid()) mod |= mask_op.flag_register_mod();
        auto rd = (send_func.is_prefetch ? host_->null : reg_buf_op.reg_data());

        // Zero-out inactive channels.
        if (send_func.is_read() && !send_func.is_prefetch
                && mod.getPredCtrl() != ngen::PredCtrl::None) {
            zero_out_data_payload(send_func, mod, rd);
        }

        // Emit send instruction.
        spec_impl.emit(host_, scope, mod, mem_buf_rd, surf_bti,
                mem_off_op.reg_data(), rd);
    }

    void eltwise(ngen_register_scope_t &scope, const eltwise_t &func,
            const std::vector<ngen_operand_t> &args) {
        int elems = to_cpp<int>(hw, eltwise_t::arg_elems(args));
        auto &data_op = eltwise_t::arg_data(args);
        auto &data_rd = data_op.reg_data();

        int grf_size = ngen::GRF::bytes(hw);
        ir_assert(elems * sizeof(float) % grf_size == 0)
                << "Partial GRF updates are not supported.";
        ir_assert(data_rd.getOffset() == 0)
                << "Data must be aligned to GRF boundary.";

        jit_eltwise_injector_f32<hw> inj(
                host_, func.alg_kind, func.alpha, func.beta, func.scale);
        auto scratch = scope.alloc_range(inj.preferred_scratch_regs());
        inj.set_scratch(scratch);
        inj.prepare();
        inj.compute(ngen::GRFRange(
                data_rd.getBase(), elems * sizeof(float) / grf_size));
    }

    ngen_operand_t eval(const expr_t &e, ngen_register_scope_t &scope,
            const ngen_operand_t &dst_operand = ngen_operand_t()) {
        expr_evaluator_t<hw> expr_evaluator(host_, expr_binding_, scope);
        return expr_evaluator.eval(e, dst_operand);
    }

    std::vector<ngen_operand_t> eval(
            const std::vector<expr_t> &exprs, ngen_register_scope_t &scope) {
        expr_evaluator_t<hw> expr_evaluator(host_, expr_binding_, scope);
        return expr_evaluator.eval(exprs);
    }

    conv_kernel_t<hw> *host_;
    expr_binding_t expr_binding_;
    int simd_size_;
};

template <ngen::HW hw>
conv_kernel_t<hw>::conv_kernel_t(const conv_config_t &cfg,
        const convolution_pd_t *pd, const kernel_arg_info_t &kernel_arg_info)
    : cfg_(cfg), ra_(hw) {

    // XXX: BWD_W does 32x32 multiplication in the inner loop which may cause
    // hangs when using with split barrier. Switch to emulation to work around
    // the issue.
    if (cfg_.is_bwd_w) emu_strategy.emulate64 = true;

    // Build IR for the kernel.
    kernel_builder_t builder(cfg, pd, kernel_arg_info);
    stmt_t body = builder.stmt();

    alloc_manager_t alloc_mgr(body);

    setup_interface(body, kernel_arg_info);

    setDefaultNoMask();
    setDefaultAutoSWSB(true);

    prologue();

    // Claim registers.
    ra_.claim(r0);
    for (int i = 0; i < 3; i++)
        ra_.claim(getLocalID(i));

    for (int i = 0; i < kernel_arg_info.nargs(); i++) {
        ra_.claim(getArgument(kernel_arg_info.arg_name(i)));
    }

    if (emu_strategy.emulate64) {
        emu_state.temp[0] = ra_.alloc();
        emu_state.temp[1] = ra_.alloc();
    }
    // Enable IEEE f32 -> s32 rounding and f32/f16 denormals.
    or_(1, cr0, cr0, uint16_t(0x1480));

    // Allocate and initialize signal header for future use.
    signal_header_ = ra_.alloc();
    barrierheader(signal_header_);

    // Bind "external" variables.
    expr_binding_t expr_binding(hw);

    // Bind grid indices.
    expr_binding.bind(builder.kernel_grid_idx(0), r0.ud(1));
    expr_binding.bind(builder.kernel_grid_idx(1), r0.ud(6));
    expr_binding.bind(builder.kernel_grid_idx(2), r0.ud(7));

    // Bind local IDs.
    for (int i = 0; i < 3; i++) {
        expr_binding.bind(builder.local_id(i), getLocalID(i).uw(0));
    }

    // Bind arguments.
    for (int i = 0; i < kernel_arg_info.nargs(); i++) {
        auto &arg_var = kernel_arg_info.arg_var(i);
        auto &name = kernel_arg_info.arg_name(i);
        if (arg_var.type().is_ptr()) {
            auto alloc_buf = alloc_mgr.find_buffer(name);
            ir_assert(alloc_buf.is_same(arg_var));
        }
        expr_binding.bind(arg_var, getArgument(name));
    }

    // Bind SLM buffer (SLM loads/stores use 0-based offsets).
    auto slm_buf = alloc_mgr.find_buffer("slm", /*allow_empty=*/true);
    if (!slm_buf.is_empty()) { expr_binding.bind(slm_buf, to_ngen(expr_t(0))); }

    // Generate assembly from IR.
    ir_to_ngen_t<hw> visitor(this, expr_binding);
    visitor.visit(body);

    epilogue();
    pad_kernel();
}

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

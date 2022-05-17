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

#ifndef EMULATION_HPP
#define EMULATION_HPP

#include <exception>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

struct EmulationStrategy {
    // Emulate 64-bit arithmetic (required for GenXLP)
    bool emulate64 = false;
    // Emulate DW x DW -> DW multiplication (required for Gen12)
    bool emulateDWxDW = false;
    // Use 32-bit adds for 64-bit arithmetic, assuming no 2^32 boundaries crossed.
    bool emulate64_add32 = false;

    EmulationStrategy() = default;
    EmulationStrategy(ngen::HW hw_) {
        using namespace ngen;
        if (hw_ == HW::Gen11) emulate64 = true;
        if (hw_ >= HW::Gen11) emulateDWxDW = true;
        if (hw_ == HW::Gen12LP) emulate64 = true;
        if (hw_ == HW::XeHPG) emulate64 = true;
    }
};

struct EmulationState {
    ngen::GRF temp[2]; // Temporary GRFs for use in emulation sequences
    ngen::FlagRegister
            flag; // Flag register for use in emulating 64-bit adds (optional, avoids temporary registers/acc)
    int flagOffset = 0; // Channel offset to use with flag register.
};

// Implementation wrapped as static methods in non-instantiated class.
// Clients should declare EmulationImplementation as a friend.
struct EmulationImplementation {
    [[noreturn]] static void stub() {
        throw std::runtime_error("Unimplemented");
    }

    template <typename DT, typename O>
    static void applyDefaultType(O &op) {
        using namespace ngen;
        if (op.getType() == DataType::invalid) op.setType(getDataType<DT>());
    }

    template <typename O>
    static bool isQW(const O &op) {
        using namespace ngen;
        using dnnl::impl::utils::one_of;
        return one_of(op.getType(), DataType::q, DataType::uq);
    }

    template <typename O>
    static bool isDW(const O &op) {
        using namespace ngen;
        using dnnl::impl::utils::one_of;
        return one_of(op.getType(), DataType::d, DataType::ud);
    }

    template <typename O>
    static bool isW(const O &op) {
        using namespace ngen;
        using dnnl::impl::utils::one_of;
        return one_of(op.getType(), DataType::w, DataType::uw);
    }

    template <typename T1, typename T2>
    static bool equal(const T1 &o1, const T2 &o2) {
        return o1 == o2;
    }
    static bool equal(const ngen::RegData &o1, const ngen::Immediate &o2) {
        return false;
    }

    static void downgradeToDW(ngen::RegData &op) {
        using namespace ngen;
        if (isQW(op)) {
            op.setType(
                    (op.getType() == DataType::q) ? DataType::d : DataType::ud);
            op.setOffset(op.getOffset() * 2);
        }
    }

    static void downgradeToDW(ngen::Immediate &op) {
        using namespace ngen;
        if (isQW(op))
            op.setType(
                    (op.getType() == DataType::q) ? DataType::d : DataType::ud);
    }

    // Get the DW equivalent of a QW region.
    static void makeDWPair(ngen::RegData &op, int esize) {
        if (isQW(op)) {
            downgradeToDW(op);
            if (op.getHS() > 1) {
                if (op.getVS() != op.getHS() * op.getWidth()) stub();
                op.setRegion(op.getHS() * 2, 2, 1);
            } else {
                auto newVS = op.getVS() * 2;
                if (esize == op.getWidth()) newVS = esize * 2;
                op.setRegion(newVS, op.getWidth() * 2, 1);
            }
        }
    }

    // Split a register into DW pairs.
    static void splitToDW(
            ngen::RegData in, ngen::RegData &outLo, ngen::RegData &outHi) {
        using namespace ngen;
        bool isQ = (in.getType() == DataType::q);
        bool isUQ = (in.getType() == DataType::uq);

        if (isQ || isUQ) {
            outLo = in;
            outLo.setRegion(in.getVS() * 2, in.getWidth(), in.getHS() * 2);
            outLo.setOffset(in.getOffset() * 2);
            outLo.setType(DataType::ud);

            outHi = outLo;
            outHi.setOffset(in.getOffset() * 2 + 1);
            outHi.setType(isQ ? DataType::d : DataType::ud);
        } else {
            outLo = in;
            outHi = Subregister {}; // invalid
        }
    }

    // Split an ngen::Immediate into DW pairs.
    static void splitToDW(const ngen::Immediate &in, ngen::Immediate &outLo,
            ngen::Immediate &outHi) {
        using namespace ngen;
        bool isQ = (in.getType() == DataType::q);
        bool isUQ = (in.getType() == DataType::uq);

        if (isQ || isUQ) {
            outLo = uint32_t(static_cast<uint64_t>(in));
            outLo.setType(DataType::ud);

            outHi = uint32_t(static_cast<uint64_t>(in) >> 32);
            outHi.setType(isQ ? DataType::d : DataType::ud);
        } else {
            outLo = in;
            outHi = uint16_t(0);
        }
    }

    static ngen::RegData lowWord(ngen::RegData in) {
        using namespace ngen;
        if (isW(in)) return in;

        auto outLo = in;
        outLo.setRegion(in.getVS() * 2, in.getWidth(), in.getHS() * 2);
        outLo.setOffset(in.getOffset() * 2);
        outLo.setType(DataType::uw);

        return outLo;
    }

    static ngen::Immediate lowWord(const ngen::Immediate &in) {
        return uint16_t(static_cast<uint64_t>(in) & 0xffff);
    }

    static bool isUnitStride(const ngen::RegData &rd) {
        return (rd.getHS() == 1 && rd.getVS() == rd.getWidth());
    }

    // Move, emulating 64-bit moves with 32-bit (generally a good idea).
    template <typename DT = void, typename Generator>
    static void emov(Generator &g, const ngen::InstructionModifier &mod,
            ngen::RegData dst, ngen::RegData src0,
            const EmulationStrategy &strategy) {
        using namespace ngen;
        applyDefaultType<DT>(dst);
        applyDefaultType<DT>(src0);

        bool dstQ = isQW(dst);
        bool s0Q = isQW(src0);

        if ((dstQ || s0Q) && strategy.emulate64) {
            if (dstQ != s0Q) stub();

            auto mod2x = mod;
            mod2x.setExecSize(mod.getExecSize() * 2);

            makeDWPair(dst, mod.getExecSize());
            makeDWPair(src0, mod.getExecSize());
            g.mov(mod2x, dst, src0);
        } else if (dst.getType() == DataType::f
                && src0.getType() == DataType::bf
                && (src0.getHS() != 1 || mod.getExecSize() == 1)) {
            // Emulate bf16->f32 upconversion
            dst.setType(DataType::ud);
            src0.setType(DataType::uw);
            g.shl(mod, dst, src0, 16);
        } else
            g.mov(mod, dst, src0);
    }

    template <typename DT = void, typename Generator>
    static void emov(Generator &g, const ngen::InstructionModifier &mod,
            ngen::RegData dst, ngen::Immediate src0,
            const EmulationStrategy &strategy) {
        using namespace ngen;
        applyDefaultType<DT>(dst);
        applyDefaultType<DT>(src0);

        bool dstQ = isQW(dst);
        bool s0Q = isQW(src0);

        if ((dstQ || s0Q) && strategy.emulate64) {
            if (dstQ != s0Q) stub();

            RegData dstHi, dstLo;
            Immediate s0Hi = 0, s0Lo = 0;

            splitToDW(src0, s0Lo, s0Hi);

            if (static_cast<uint64_t>(s0Lo) == static_cast<uint64_t>(s0Hi)) {
                auto mod2x = mod;
                mod2x.setExecSize(mod.getExecSize() * 2);

                downgradeToDW(dst);
                g.mov(mod2x, dst, s0Lo);
            } else {
                splitToDW(dst, dstLo, dstHi);
                g.mov(mod, dstLo, s0Lo);
                g.mov(mod, dstHi, s0Hi);
            }
        } else
            g.mov(mod, dst, src0);
    }

    template <typename Generator>
    static void eaddSignExtend1(Generator &g,
            const ngen::InstructionModifier &mod, bool &doSub,
            const ngen::Immediate &src1, ngen::Immediate &s1LoPos,
            const ngen::Immediate &s1Lo, const ngen::Immediate &s1Hi, bool &s1Q,
            const ngen::GRF (&temp)[2]) {
        using namespace ngen;
        uint64_t raw = static_cast<uint64_t>(src1);
        if (src1.getType() == DataType::d) {
            int32_t val = raw;
            s1LoPos = uint32_t(std::abs(val));
            doSub = (val < 0);
        } else if (src1.getType() == DataType::w) {
            int16_t val = raw;
            s1LoPos = uint16_t(std::abs(val));
            doSub = (val < 0);
        }
    }

    template <typename Generator>
    static void eaddSignExtend1(Generator &g,
            const ngen::InstructionModifier &mod, bool &doSub,
            const ngen::RegData &src1, ngen::RegData &s1LoPos,
            ngen::RegData &s1Lo, ngen::RegData &s1Hi, bool &s1Q,
            const ngen::GRF (&temp)[2]) {
        using namespace ngen;
        s1Q = true;
        s1Hi = temp[0].d();
        if (s1Lo.getNeg()) {
            g.asr(mod, s1Hi, -s1Lo, uint16_t(31));
            s1Hi = -s1Hi;
        } else
            g.asr(mod, s1Hi, s1Lo, uint16_t(31));
        s1Lo.setType(DataType::ud);
    }

    static void eaddHandleS1Neg(
            bool &doSub, ngen::RegData &s1LoPos, const ngen::RegData &s1Lo) {
        if (isSigned(s1Lo.getType())) stub();
        doSub = s1Lo.getNeg();
        s1LoPos = -s1Lo;
    }

    static void eaddHandleS1Neg(bool &doSub, const ngen::Immediate &s1LoPos,
            const ngen::Immediate &s1Lo) {
        /* no-op */
    }

    template <typename Generator>
    static void eaddFixupQD(Generator &g, const ngen::InstructionModifier &mod,
            const ngen::FlagRegister &flag, const ngen::RegData &dstHi,
            const ngen::RegData &src1) {
        if ((src1.getBytes() < 8) && isSigned(src1.getType())) {
            // Add sign extension of src1 to high 32 bits of dst (inefficient but rarely used path).
            g.cmp(mod | (src1.getNeg() ? g.gt : g.lt) | flag, src1, 0);
            g.add(mod | flag, dstHi, dstHi, -1);
        }
    }

    template <typename Generator>
    static void eaddFixupQD(Generator &g, const ngen::InstructionModifier &mod,
            const ngen::FlagRegister &flag, const ngen::RegData &dstHi,
            const ngen::Immediate &src1) {
        /* no-op */
    }

    static bool eaddIsNegative(const ngen::RegData &r) { return r.getNeg(); }

    static bool eaddIsNegative(const ngen::Immediate &i) {
        return int32_t(uint64_t(i)) < 0;
    }

    // Integer addition, emulating 64-bit arithmetic if configured.
    template <typename DT = void, typename S1, typename Generator>
    static void eaddInternal(Generator &g, const ngen::InstructionModifier &mod,
            ngen::RegData dst, ngen::RegData src0, S1 src1,
            const EmulationStrategy &strategy, const EmulationState &state) {
        using namespace ngen;
        const auto &temp = state.temp;

        applyDefaultType<DT>(dst);
        applyDefaultType<DT>(src0);
        applyDefaultType<DT>(src1);

        bool dstQ = isQW(dst);
        bool s0Q = isQW(src0);
        bool s1Q = isQW(src1);

        if (dstQ && strategy.emulate64_add32) {
            RegData dstHi, dstLo, s0Hi, s0Lo;
            S1 s1Hi, s1Lo;

            splitToDW(dst, dstLo, dstHi);
            splitToDW(src0, s0Lo, s0Hi);
            splitToDW(src1, s1Lo, s1Hi);
            g.add(mod, dstLo, s0Lo, s1Lo);

            if (s0Q && s1Q)
                g.add(mod, dstHi, s0Hi, s1Hi);
            else if (s0Q) {
                if (!equal(dstHi, s0Hi)) g.mov(mod, dstHi, s0Hi);
            } else if (s1Q) {
                if (!equal(dstHi, s1Hi)) g.mov(mod, dstHi, s1Hi);
            } else
                g.mov(mod, dstHi, uint16_t(0));
        } else if (!strategy.emulate64)
            g.add(mod, dst, src0, src1);
        else {
            if (!dstQ) {
                downgradeToDW(src0);
                downgradeToDW(src1);
                g.add(mod, dst, src0, src1);
            } else {
                RegData dstHi, dstLo, s0Hi, s0Lo;
                S1 s1Hi, s1Lo, s1LoPos;
                FlagRegister flag = state.flag;

                splitToDW(dst, dstLo, dstHi);
                splitToDW(src0, s0Lo, s0Hi);
                splitToDW(src1, s1Lo, s1Hi);
                s1LoPos = s1Lo;

                bool s0Signed = isSigned(s0Lo.getType());
                bool s1Signed = isSigned(s1Lo.getType());

                if (flag.isValid() && !eaddIsNegative(s0Lo)) {
                    // Use flag register + ov.
                    auto Mx = g.ExecutionOffset(state.flagOffset);
                    bool neg = eaddIsNegative(s1Lo);

                    auto s0LoUD = s0Lo;
                    auto s1LoMod = s1Lo;
                    s0LoUD.setType(DataType::ud);
                    if (s1Signed
                            && !std::is_base_of<ngen::Immediate, S1>::value) {
                        s1LoMod.setType(DataType::ud);
                        neg = false;
                    }

                    g.add(mod | Mx | g.ov | flag, dstLo, s0LoUD, s1LoMod);
                    if (s0Q && s1Q)
                        g.add(mod, dstHi, s0Hi, s1Hi);
                    else if (s0Q && !equal(dstHi, s0Hi))
                        g.mov(mod, dstHi, s0Hi);
                    else if (s1Q && !equal(dstHi, s1Hi))
                        g.mov(mod, dstHi, s1Hi);
                    else if (!s0Q && !s1Q)
                        g.mov(mod, dstHi, 0);
                    g.add(mod | Mx | flag, dstHi, dstHi, neg ? -1 : +1);
                    eaddFixupQD(g, mod | Mx, flag, dstHi, src1);
                } else {
                    // Slow path: addc/subb + acc.
                    RegData carry = temp[0].ud();
                    bool lateCarry = false;
                    RegData subDstLo;
                    bool doSub = false;

                    // For :uq + :d or :q + :ud, sign extend 32-bit input to 64 bits.
                    if (s0Signed != s1Signed) {
                        if (s0Signed) {
                            s0Q = true;
                            s0Hi = temp[0].d();
                            g.asr(mod, s0Hi, s0Lo, uint16_t(31));
                            s0Lo.setType(DataType::ud);
                            if (s0Lo.getNeg()) s0Hi = -s0Hi;
                        } else
                            eaddSignExtend1(g, mod, doSub, src1, s1LoPos, s1Lo,
                                    s1Hi, s1Q, temp);
                        carry = temp[1].ud();
                        lateCarry = true;
                    }

                    // Handle modifiers.
                    if (s0Lo.getNeg()) stub();
                    eaddHandleS1Neg(doSub, s1LoPos, s1Lo);

                    // Compute low 32 bits, saving carry/borrow.
                    if (dstLo.getOffset() != 0) {
                        doSub ? g.subb(mod, g.null.retype(s0Lo.getType()), s0Lo,
                                s1LoPos)
                              : g.addc(mod, g.null.retype(s0Lo.getType()), s0Lo,
                                      s1Lo);
                        g.add(mod, dstLo, s0Lo, s1Lo);
                    } else if ((mod.getExecSize() > 1)
                            && !isUnitStride(dstLo)) {
                        subDstLo = temp[1].ud();
                        doSub ? g.subb(mod, subDstLo, s0Lo, s1LoPos)
                              : g.addc(mod, subDstLo, s0Lo, s1Lo);
                    } else {
                        doSub ? g.subb(mod, dstLo, s0Lo, s1LoPos)
                              : g.addc(mod, dstLo, s0Lo, s1Lo);
                    }

                    // Retrieve carry from accumulator, unless it conflicts with subDstLo.
                    if (!lateCarry) g.mov(mod, carry, g.acc0.ud());

                    // Move low 32-bits to final resting place, if needed.
                    if (subDstLo.isValid()) g.mov(mod, dstLo, subDstLo);

                    // Retrieve carry from accumulator once subDstLo isn't needed.
                    if (lateCarry) g.mov(mod, carry, g.acc0.ud());

                    if (doSub) carry = -carry;

                    // Compute high 32 bits of sum.
                    if (s0Q && s1Q) {
                        g.add(mod, dstHi, s0Hi, s1Hi);
                        g.add(mod, dstHi, carry, dstHi);
                    } else if (s0Q)
                        g.add(mod, dstHi, carry, s0Hi);
                    else if (s1Q)
                        g.add(mod, dstHi, carry, s1Hi);
                    else
                        g.mov(mod, dstHi, carry);
                }
            }
        }
    }

    template <typename DT = void, typename Generator>
    static void eadd(Generator &g, const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const ngen::RegData &src1, const EmulationStrategy &strategy,
            const EmulationState &state) {
        eaddInternal<DT>(g, mod, dst, src0, src1, strategy, state);
    }

    template <typename DT = void, typename Generator>
    static void eadd(Generator &g, const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            ngen::Immediate src1, const EmulationStrategy &strategy,
            const EmulationState &state) {
        eaddInternal<DT>(g, mod, dst, src0, src1, strategy, state);
    }

    // Integer multiplication, emulating 32x32 multiplication as configured.
    template <typename DT = void, typename S1, typename Generator>
    static void emulInternal(Generator &g, const ngen::InstructionModifier &mod,
            ngen::RegData dst, ngen::RegData src0, S1 src1,
            const EmulationStrategy &strategy, const EmulationState &state) {
        using namespace ngen;
        applyDefaultType<DT>(dst);
        applyDefaultType<DT>(src0);
        applyDefaultType<DT>(src1);

        bool dstD = isDW(dst);
        bool dstQ = isQW(dst);
        bool s0W = isW(src0);
        bool s0D = isDW(src0);
        bool s0Q = isQW(src0);
        bool s1W = isW(src1);
        bool s1D = isDW(src1);
        bool s1Q = isQW(src1);

        bool s0Signed = isSigned(src0.getType());
        bool s1Signed = isSigned(src1.getType());
        auto mulHiType = (s0Signed || s1Signed) ? DataType::d : DataType::ud;

        bool emulate64 = strategy.emulate64;

        if (mod.getExecSize() != 1) stub();

        if (s0Q || s1Q) {
            stub();
        } else if (dstQ && s0W && s1W) {
            RegData dstLo, dstHi;
            splitToDW(dst, dstLo, dstHi);

            g.mul(mod, dstLo, src0, src1);

            dstHi.setType(mulHiType);
            dstLo.setType(mulHiType);

            if (s0Signed || s1Signed)
                g.asr(mod, dstHi, dstLo, 31);
            else
                g.mov(mod, dstHi, 0);
        } else if (dstQ && s0W && s1D) {
            stub();
        } else if (dstQ && s0D && (s1W || (s1D && emulate64))) {
            RegData dstLo, dstHi;
            splitToDW(dst, dstLo, dstHi);

            auto acc = g.acc0.retype(mulHiType)[dstLo.getOffset()];

            g.mul(mod, acc, src0, lowWord(src1));
            if (s1D)
                g.mach(mod, dstLo, src0, src1);
            else
                g.mach(mod, dstLo, src0, int32_t(0));
            g.mov(mod, dstHi, dstLo);
            g.mov(mod, dstLo, acc);
        } else if (dstD && s0D && s1D && strategy.emulateDWxDW) {
            auto acc = g.acc0.retype(mulHiType)[dst.getOffset()];
            auto dummy = g.null.retype(mulHiType)[dst.getOffset()];

            g.mul(mod, acc, src0, lowWord(src1));

            if (g.hardware < HW::Gen10) {
                g.mach(mod, dummy, src0, src1);
                g.mov(mod, dst, acc);
            } else {
                g.macl(mod, dst, src0, src1);
            }
        } else
            g.mul(mod, dst, src0, src1);
    }

    template <typename DT = void, typename Generator>
    static void emul(Generator &g, const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const ngen::RegData &src1, const EmulationStrategy &strategy,
            const EmulationState &state) {
        emulInternal<DT>(g, mod, dst, src0, src1, strategy, state);
    }

    template <typename DT = void, typename Generator>
    static void emul(Generator &g, const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            ngen::Immediate src1, const EmulationStrategy &strategy,
            const EmulationState &state) {
        emulInternal<DT>(g, mod, dst, src0, src1, strategy, state);
    }

    template <typename S1, typename Generator>
    static void emul32High(Generator &g, const ngen::InstructionModifier &mod,
            const ngen::RegData &dstHi, const ngen::RegData &src0,
            const S1 &src1) {
        g.mul(mod, g.acc0.ud(dstHi.getOffset()), src0, lowWord(src1));
        g.mach(mod, dstHi, src0, src1);
    }

    // Shift left, emulating 64-bit arithmetic if configured.
    template <typename DT = void, typename Generator>
    static void eshl(Generator &g, const ngen::InstructionModifier &mod,
            ngen::RegData dst, ngen::RegData src0, uint16_t src1,
            const EmulationStrategy &strategy, const EmulationState &state) {
        using namespace ngen;
        const auto &temp = state.temp;

        applyDefaultType<DT>(dst);
        applyDefaultType<DT>(src0);

        bool dstQ = isQW(dst);
        bool s0Q = isQW(src0);

        if (src1 == 0) {
            emov<DT, Generator>(g, mod, dst, src0, strategy);
            return;
        }

        if (dstQ && strategy.emulate64) {
            if (src1 >= 32) stub();

            RegData dstHi, dstLo, s0Hi, s0Lo;

            auto acc = temp[0].ud();

            splitToDW(dst, dstLo, dstHi);

            if (s0Q) {
                splitToDW(dst, s0Lo, s0Hi);

                g.shr(mod, acc, s0Lo, uint16_t(32 - src1));
                g.shl(mod, dstHi, s0Hi, src1);
                g.shl(mod, dstLo, s0Lo, src1);
                g.or_(mod, dstHi, acc, dstHi);
            } else {
                dstHi.setType(DataType::ud);
                g.shl(mod, dstLo, src0, src1);
                g.shr(mod, dstHi, src0, uint16_t(32 - src1));
            }
        } else {
            if (s0Q && !dstQ) downgradeToDW(src0);
            g.shl(mod, dst, src0, src1);
        }
    }

    // Shift right, emulating 64-bit arithmetic if configured.
    template <typename DT = void, typename Generator>
    static void eshr(Generator &g, const ngen::InstructionModifier &mod,
            ngen::RegData dst, ngen::RegData src0, uint16_t src1,
            const EmulationStrategy &strategy, const EmulationState &state) {
        using namespace ngen;
        const auto &temp = state.temp;

        applyDefaultType<DT>(dst);
        applyDefaultType<DT>(src0);

        bool dstQ = isQW(dst);
        bool s0Q = isQW(src0);

        if (src1 == 0) {
            emov<DT, Generator>(g, mod, dst, src0, strategy);
            return;
        }

        if (dstQ && strategy.emulate64) {
            if (src1 >= 32) stub();

            RegData dstHi, dstLo, s0Hi, s0Lo;

            auto acc = temp[0].ud();

            splitToDW(dst, dstLo, dstHi);

            if (s0Q) {
                splitToDW(dst, s0Lo, s0Hi);

                g.shl(mod, acc, s0Lo, uint16_t(32 - src1));
                g.shr(mod, dstLo, s0Lo, src1);
                isSigned(src0.getType()) ? g.asr(mod, dstHi, s0Hi, src1)
                                         : g.shr(mod, dstHi, s0Hi, src1);
                g.or_(mod, dstLo, acc, dstLo);
            } else {
                dstLo.setType(dstHi.getType());
                isSigned(src0.getType()) ? g.asr(mod, dstLo, src0, src1)
                                         : g.shr(mod, dstLo, src0, src1);
                g.mov(mod, dstHi, uint16_t(0));
            }
        } else {
            if (s0Q && !dstQ) downgradeToDW(src0);
            isSigned(src0.getType()) ? g.asr(mod, dst, src0, src1)
                                     : g.shr(mod, dst, src0, src1);
        }
    }

    // Multiply by a constant, optimizing for power-of-2 constants and emulating 64-bit arithmetic if configured.
    template <typename DT = void, typename Generator>
    static void emulConstant(Generator &g, const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0, int32_t src1,
            const EmulationStrategy &strategy, const EmulationState &state) {
        if (src1 == 0)
            emov<DT>(g, mod, dst, uint16_t(0), strategy);
        else if (src1 == 1) {
            if (dst != src0) emov<DT>(g, mod, dst, src0, strategy);
        } else if (ngen::utils::is_zero_or_pow2(src1))
            eshl<DT>(g, mod, dst, src0, uint16_t(ngen::utils::log2(src1)),
                    strategy, state);
        else if (src1 > 0)
            emul<DT>(g, mod, dst, src0, uint32_t(src1), strategy, state);
        else
            emul<DT>(g, mod, dst, src0, int32_t(src1), strategy, state);
    }
}; // struct EmulationHelper

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#define EMULATION_FORWARD \
    template <typename DT = void> \
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst, \
            ngen::RegData src0, const EmulationStrategy &strategy) { \
        EmulationImplementation::emov<DT>(*this, mod, dst, src0, strategy); \
    } \
    template <typename DT = void> \
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst, \
            ngen::Immediate src0, const EmulationStrategy &strategy) { \
        EmulationImplementation::emov<DT>(*this, mod, dst, src0, strategy); \
    } \
    template <typename DT = void> \
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst, \
            const ngen::RegData &src0, const ngen::RegData &src1, \
            const EmulationStrategy &strategy, const EmulationState &state) { \
        EmulationImplementation::eadd<DT>( \
                *this, mod, dst, src0, src1, strategy, state); \
    } \
    template <typename DT = void> \
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst, \
            const ngen::RegData &src0, ngen::Immediate src1, \
            const EmulationStrategy &strategy, const EmulationState &state) { \
        EmulationImplementation::eadd<DT>( \
                *this, mod, dst, src0, src1, strategy, state); \
    } \
    template <typename DT = void> \
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst, \
            const ngen::RegData &src0, const ngen::RegData &src1, \
            const EmulationStrategy &strategy, const EmulationState &state) { \
        EmulationImplementation::emul<DT>( \
                *this, mod, dst, src0, src1, strategy, state); \
    } \
    template <typename DT = void> \
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst, \
            const ngen::RegData &src0, ngen::Immediate src1, \
            const EmulationStrategy &strategy, const EmulationState &state) { \
        EmulationImplementation::emul<DT>( \
                *this, mod, dst, src0, src1, strategy, state); \
    } \
    template <typename DT = void> \
    void eshl(const ngen::InstructionModifier &mod, ngen::RegData dst, \
            ngen::RegData src0, uint16_t src1, \
            const EmulationStrategy &strategy, const EmulationState &state) { \
        EmulationImplementation::eshl<DT>( \
                *this, mod, dst, src0, src1, strategy, state); \
    } \
    template <typename DT = void> \
    void eshr(const ngen::InstructionModifier &mod, ngen::RegData dst, \
            ngen::RegData src0, uint16_t src1, \
            const EmulationStrategy &strategy, const EmulationState &state) { \
        EmulationImplementation::eshr<DT>( \
                *this, mod, dst, src0, src1, strategy, state); \
    } \
    template <typename DT = void> \
    void emulConstant(const ngen::InstructionModifier &mod, \
            const ngen::RegData &dst, const ngen::RegData &src0, int32_t src1, \
            const EmulationStrategy &strategy, const EmulationState &state) { \
        EmulationImplementation::emulConstant<DT>( \
                *this, mod, dst, src0, src1, strategy, state); \
    } \
    template <typename S1> \
    void emul32High(const ngen::InstructionModifier &mod, \
            const ngen::RegData &dstHi, const ngen::RegData &src0, \
            const S1 &src1) { \
        EmulationImplementation::emul32High(*this, mod, dstHi, src0, src1); \
    }

#endif

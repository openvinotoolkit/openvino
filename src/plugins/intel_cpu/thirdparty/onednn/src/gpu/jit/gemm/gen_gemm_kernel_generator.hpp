/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
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

#ifndef GPU_JIT_GEMM_GEN_GEMM_KERNEL_GENERATOR_HPP
#define GPU_JIT_GEMM_GEN_GEMM_KERNEL_GENERATOR_HPP

/* Embargo support */

#define STANDALONE 0

#include "common/math_utils.hpp"
#include "common/utils.hpp"
#include "gpu/jit/gemm/gen_gemm_kernel_common.hpp"
#include "gpu/jit/gemm/utils.hpp"
#include "gpu/jit/jit_generator.hpp"
#include "gpu/jit/jit_post_op_injector.hpp"

#include "../ngen/ngen_opencl.hpp"

#include "../ngen/ngen_register_allocator.hpp"

#include "gpu/jit/gemm/emulation.hpp"

#include <array>
#include <complex>
#include <cstdint>
#include <exception>
#include <iostream>
#include <sstream>
#include <vector>

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

class Type {
public:
    enum _Type : uint32_t {
        invalid = 0,
        f16 = 0x000201,
        f32 = 0x010402,
        u8 = 0x840100,
        s8 = 0x850100,
        u16 = 0x860201,
        s16 = 0x870201,
        u32 = 0x880402,
        s32 = 0x890402,
        u64 = 0x8A0803,
        s64 = 0x8B0803,
        bf16 = 0x0C0201,
    };

private:
    _Type val;

public:
    constexpr Type() : Type(f32) {}
    constexpr Type(_Type val_) : val(val_) {}
    constexpr operator _Type() const { return val; }

    constexpr Type real() const { return *this; }
    constexpr bool isComplex() const { return false; }
    constexpr bool isInteger() const { return uint32_t(val) & 0x800000; }
    constexpr bool isFP() const { return !isInteger(); }
    constexpr bool isSigned() const {
        return (uint32_t(val) & 0x810000) != 0x800000;
    }
    constexpr int log2Size() const { return uint32_t(val) & 0xFF; }
    constexpr int size() const { return (uint32_t(val) >> 8) & 0xFF; }
    constexpr int components() const { return isComplex() ? 2 : 1; }
    data_type_t get_dnnl_type() const {
        switch (val) {
            case Type::f32: return data_type::f32;
            case Type::f16: return data_type::f16;
            case Type::s32: return data_type::s32;
            case Type::u8: return data_type::u8;
            case Type::s8: return data_type::s8;
            default: assert(!"Unsupported type"); return data_type::undef;
        }
    }

    template <typename U>
    constexpr friend int operator*(U a, Type t) {
        return a << t.log2Size();
    }
    template <typename U>
    constexpr friend int operator/(U a, Type t) {
        return a >> t.log2Size();
    }

    ngen::DataType ngen() const {
        using namespace ngen;
        static const DataType table[16] = {DataType::hf, DataType::f,
                DataType::df, DataType::invalid, DataType::ub, DataType::b,
                DataType::uw, DataType::w, DataType::ud, DataType::d,
                DataType::uq, DataType::q, DataType::bf, DataType::invalid,
                DataType::invalid, DataType::invalid};
        return table[(uint32_t(val) >> 16) & 0xF];
    }
};

enum class MatrixLayout : uint8_t {
    N = 0,
    Nontranspose = 0,
    T = 1,
    Transpose = 1,
    Pc = 2,
    PackedColumns = 2,
    Pr = 3,
    PackedRows = 3
};

static inline bool isPacked(MatrixLayout l) {
    return (l == MatrixLayout::PackedRows)
            || (l == MatrixLayout::PackedColumns);
}

enum class AccessType : uint8_t {
    Scattered, // Use scattered accesses
    ChannelScattered, // Use untyped surface reads
    Block, // Use block messages
    PseudoBlock, // Use scattered accesses to emulate block accesses
};

enum LoopType : uint8_t {
    LoopM = 0,
    LoopN = 1,
    LoopK = 2,
    LoopMNBoustrophedonMNK
    = 0xFB, // Fused m/n indices (boustrophedon ordering), with MNK nested inside
    LoopMNBoustrophedonNMK
    = 0xFC, // Fused n/m indices (boustrophedon ordering), with NMK nested inside
    LoopMNHilbertMNK
    = 0xFD, // Fused m/n indices (Hilbert ordering), with MNK nested inside
    LoopMNHilbertNMK
    = 0xFE, // Fused n/m indices (Hilbert ordering), with NMK nested inside
    LoopAny = 0xFF,
    LoopNone = 0xFF
};

enum class RemainderHandling : uint8_t {
    Ignore, // Assume no remainder, or handled by hardware bounds checking.
    General, // Handle all remainder cases.
    Split, // Generate copies of the kernel with and without remainder handling.
    KnownRemainder, // Assume remainder case; don't create special code for non-remainder case.
};

enum class KernelScheduling : uint8_t {
    Static,
    EUStatic,
    Dynamic,
};

struct GRFMultirange {
    std::vector<ngen::GRFRange> ranges;

    GRFMultirange() {}
    GRFMultirange(ngen::GRFRange range) : ranges {1, range} {}

    ngen::GRF operator[](int idx) const {
        for (auto &r : ranges) {
            if (idx < r.getLen()) return r[idx];
            idx -= r.getLen();
        }
        throw std::runtime_error("Index out of bounds");
    }

    bool contiguous(int start, int count) const {
        for (auto &r : ranges) {
            if (start < r.getLen()) return (start + count) <= r.getLen();
            start -= r.getLen();
        }
        return false;
    }

    void append(ngen::GRFRange r) {
        if (!ranges.empty()) {
            auto &rend = ranges.back();
            if (rend.getBase() + rend.getLen() == r.getBase()) {
                rend = ngen::GRFRange(
                        rend.getBase(), rend.getLen() + r.getLen());
                return;
            }
        }
        ranges.push_back(r);
    }

    uint8_t getLen() const {
        uint8_t len = 0;
        for (auto &r : ranges)
            len += r.getLen();
        return len;
    }

    bool empty() const { return ranges.empty(); }

    GRFMultirange subrange(int off, int len) const {
        GRFMultirange result;
        int cur = 0;
        for (auto &r : ranges) {
            if (cur >= off + len) break;
            int soff = std::max(0, off - cur);
            int send = std::min(r.getLen(), off + len - cur);
            if (soff < r.getLen())
                result.ranges.push_back(
                        ngen::GRFRange(r.getBase() + soff, send - soff));
        }
        return result;
    }
};

template <typename T>
class Scalar {
protected:
    bool fixed_value;
    union {
        ngen::Subregister regs[2];
        T value;
    };

public:
    Scalar() : Scalar(ngen::Subregister()) {}
    explicit Scalar(T value_) : fixed_value(true), value(value_) {}
    Scalar(ngen::Subregister reg0, ngen::Subregister reg1)
        : fixed_value(false), regs {reg0, reg1} {}
    explicit Scalar(ngen::Subregister reg) : Scalar(reg, reg) {}

    Scalar &operator=(T value_) {
        fixed_value = true;
        value = value_;
        return *this;
    }
    Scalar &operator=(ngen::Subregister reg) {
        fixed_value = false;
        regs[0] = regs[1] = reg;
        return *this;
    }

    template <typename U>
    friend inline bool operator==(const Scalar<T> &scalar, const U &val) {
        return scalar.fixed_value && (val == scalar.value);
    }
    template <typename U>
    friend inline bool operator==(const U &val, const Scalar<T> &scalar) {
        return scalar == val;
    }

    template <typename U>
    friend inline bool operator!=(const Scalar<T> &scalar, const U &val) {
        return !(scalar == val);
    }
    template <typename U>
    friend inline bool operator!=(const U &val, const Scalar<T> &scalar) {
        return !(scalar == val);
    }

    operator T() const {
        if (!fixed_value) throw std::runtime_error("Scalar is not fixed.");
        return value;
    }

    bool fixed() const { return fixed_value; }

    ngen::Subregister getReg(int idx) const;
    ngen::Subregister getRegAvoiding(
            ngen::HW hw, const ngen::RegData &rd) const;
};

class MultishiftSubregister {
protected:
    static constexpr int maxShift = 5;
    ngen::Subregister regs[maxShift + 1] = {ngen::Subregister()};
    bool neg = false;

public:
    MultishiftSubregister operator-() const {
        auto copy = *this;
        copy.neg = !copy.neg;
        return copy;
    }

    ngen::Subregister operator>>(int shift) const {
        ngen::RegData sub = ngen::Subregister {};
        if (shift >= 0 && shift <= maxShift) sub = regs[shift];
        if (neg) sub = -sub;
        return *reinterpret_cast<ngen::Subregister *>(&sub);
    }

    void set(int shift, ngen::Subregister reg) { regs[shift] = reg; }
};

struct MatrixAddressing {
    MatrixLayout layout; // Layout type (N/T/Pr/Pc)
    ngen::AddressBase base; // Base for addressing (A64/BTS/...)
    bool padded; // Allow read/write overruns?
    uint8_t packSize; // # of elements in a packed row/column for packed layouts.
    uint8_t crosspack; // Crosspack for packed layouts.
    uint8_t alignment; // Alignment for all addresses, offsets, and leading dimensions.
    uint8_t tileR = 0, tileC = 0; // Tiling (0 if none) for packed layouts.

    void setAlignment(int align) { alignment = sanitizeAlign(align); }
    int defaultAlignment(Type T) const {
        return sanitizeAlign(
                T.size() * (isPacked(layout) ? (packSize * crosspack) : 1));
    }

    ngen::GlobalAccessType getGlobalAccessType() const {
        return base.isStateless() ? ngen::GlobalAccessType::Stateless
                                  : ngen::GlobalAccessType::Surface;
    }

private:
    static int sanitizeAlign(int align) {
        return std::min(128, largest_pow2_divisor(align));
    }
};

struct MatrixAddressingStrategy {
    AccessType accessType; // Block/scattered/etc. access
    uint8_t tileR = 0, tileC = 0; // Desired tiling (0 if none) in registers.
    unsigned atomic : 1; // Atomic access? (only relevant for C)
    unsigned address2D : 1; // Use 2D addressing? (media block-style loads)
    unsigned prefetch : 1; // Prefetch only?
    unsigned newDP : 1; // Use new dataport messages? (XeHPG+)
    ngen::CacheSettingsLSC caching = ngen::CacheSettingsLSC::
            Default; // Cache policy for new dataport messages.

    MatrixAddressingStrategy()
        : atomic(false), address2D(false), prefetch(false), newDP(false) {}

    void preflight(ngen::HW hw);
};

struct VirtualFlag {
    uint8_t idx : 6;
    uint8_t n : 2;

    constexpr VirtualFlag() : idx(0), n(0) {}
    /* implicit */ VirtualFlag(const ngen::FlagRegister &flag)
        : idx(flag.index()), n(flag.getBytes() >> 1) {}
    explicit constexpr VirtualFlag(int idx_, int n_ = 1) : idx(idx_), n(n_) {}

    ngen::FlagRegister toPhysical() const;

    friend inline bool operator==(VirtualFlag vf1, VirtualFlag vf2) {
        return vf1.idx == vf2.idx && vf1.n == vf2.n;
    }
    friend inline bool operator!=(VirtualFlag vf1, VirtualFlag vf2) {
        return !(vf1 == vf2);
    }

    bool operator!() const { return (idx == 0) && (n == 0); }
    explicit operator bool() const { return !!*this; }

    void clear() { *this = VirtualFlag(); }
};

struct MaskInfo {
    union {
        struct {
            uint8_t isFixed : 1; // = false (variable mask)
            uint8_t reverse : 1; // True to reverse mask.
            uint8_t : 6;
            uint8_t rsize; // Maximum remainder value. (e.g. 16 if we need the last 4 bits of the index).
            uint8_t maskRep; // # of repetitions of mask pattern.
            uint8_t bitRep : 5; // # of times each mask bit is repeated.
            uint8_t rdivide : 3; // Amount by which to divide index before forming mask. Fractions are rounded up.
                    // Note maskRep * bitRep * (rsize >> rshift) = # mask bits.
        } variable;
        struct {
            uint8_t isFixed : 1; // = true (fixed mask)
            uint8_t _ : 7;
            uint8_t rsize; // Maximum remainder value.
            uint16_t value; // Mask value.
        } fixed;
        uint32_t raw;
    };

    MaskInfo() : fixed {true, 0, 0, 0xFFFF} {}

    bool operator!() const { return fixed.isFixed && fixed.value == 0xFFFF; }
    explicit operator bool() const { return !!*this; }

    static MaskInfo None() { return MaskInfo(); }

    friend bool operator==(const MaskInfo &i1, const MaskInfo &i2) {
        return i1.raw == i2.raw;
    }
    friend bool operator!=(const MaskInfo &i1, const MaskInfo &i2) {
        return !(i1 == i2);
    }
};

struct MaskAssignment {
    MaskInfo mask; // Associated mask
    LoopType var; // Variable to base mask off of
    uint8_t offset; // Amount to subtract from variable.
    VirtualFlag flag; // Index of virtual flag register to use.

    bool compatible(const MaskAssignment &other) const {
        return mask == other.mask && var == other.var && offset == other.offset;
    }
    void reverse(int width) {
        offset = width - offset - mask.variable.rsize;
        mask.variable.reverse = !mask.variable.reverse;
    }
};

struct RegisterBlock {
    /* Register layout information. */
    uint8_t nr, nc; // Size of this block.
    uint8_t ld; // Leading dimension, in elements.
    uint8_t offsetR, offsetC; // Row and column offset within matrix block
    uint8_t colMajor : 1; // Is this block column-major? (columns stored consecutively inside each register)
    uint8_t crosspack : 7; // Crosspack for this block (1 if none).
    uint16_t bytes; // # of bytes in this block
    uint16_t offsetBytes; // Byte offset within register block

    /* Load/store information. */
    uint8_t remainderR : 1; // Row remaindering enabled?
    uint8_t remainderC : 1; // Column remaindering enabled?
    uint8_t noRowsOK : 1; // Can handle no rows (in mask/descriptor)?
    uint8_t noColsOK : 1; // Can handle no columns (in mask/descriptor)?
    uint8_t descRemR : 1; // Row remainders can be handled by changing the descriptor?
    uint8_t descRemC : 1; // Column remainders can be handled by changing the descriptor?
    uint8_t descAssigned : 1; // True if address registers have been assigned for this block's descriptors.
    uint8_t writable : 1; // True if block is set up for writing.

    uint8_t ebytes; // Size of element in bytes, e.g. 4 for scattered_dword, 16 for block_hword
    uint8_t count; // Element count.
    uint8_t extra; // Extra info. For block accesses, 1 means aligned OWord, 0 unaligned. For scattered accesses, # of consecutive elements.
    uint8_t simdSize; // SIMD size for load/stores (0 indicating no separate load/store needs to be done.)
    VirtualFlag flag; // Assigned flag register index and modifiers, if any.
    uint8_t flagAny : 1; // Use .anyh?
    uint8_t flagAll : 1; // Use .allh?
    uint8_t : 6;
    uint8_t sfid; // SFID for this block.
    uint8_t rowFragment; // If this block needs fragmenting to support row/column remainders, the maximum block size (power of 2) to fragment down to.
    uint8_t colFragment; //     Zero if no fragmenting needed.
    uint8_t addrShift; // log2(address units). e.g. 0 if byte addresses should be used, 4 if oword addresses should be used.
    uint8_t log2GRFBytes; // log2(bytes per GRF).

    MaskInfo rowMask; // Row mask for this block
    MaskInfo colMask; // Column mask for this block

    void calcBytes(Type T); // Auto-calculate # of registers.
    void calcBytes(Type T, const MatrixAddressingStrategy &astrategy);

    void clearFlag() {
        flag.clear();
        flagAll = flagAny = false;
    }
    void eraseMask() {
        clearFlag();
        rowMask = MaskInfo();
        colMask = MaskInfo();
    }

    bool isLoadBlock() const { return simdSize > 0; }

    int nregs() const;
    int offsetReg() const;

    void simplify(Type T);
};

struct VirtualFlagAllocator {
    VirtualFlagAllocator(ngen::HW hw)
        : free((1ul << (ngen::GRF::bytes(hw) >> 1)) - 1)
        , nflag(ngen::FlagRegister::subcount(hw)) {}

    VirtualFlag allocVirtual(int n = 1);
    ngen::FlagRegister alloc(int n = 1);

    void claim(VirtualFlag vflag) { free &= ~mask(vflag); }
    void release(VirtualFlag vflag) { free |= mask(vflag); }
    void release(const ngen::FlagRegister &reg) {
        release(VirtualFlag(reg));
        unlock(reg);
    }
    void safeRelease(VirtualFlag &vflag) {
        if (vflag) release(vflag);
        vflag.clear();
    }
    void safeRelease(ngen::FlagRegister &reg) {
        if (reg.isValid()) release(reg);
        reg.invalidate();
    }

    bool isVirtual(VirtualFlag vflag) { return (vflag.idx >= nflag); }

    bool lock(VirtualFlag vflag) {
        bool wasLocked = isLocked(vflag);
        locked |= mask(vflag);
        return wasLocked;
    }
    void unlock(VirtualFlag vflag) { locked &= ~mask(vflag); }
    bool isLocked(VirtualFlag vflag) const { return (locked & mask(vflag)); }

    ngen::FlagRegister assignPhysical(VirtualFlag vflag);

    static int getBase(int idx) { return idx & 0x1F; }
    static int getN(int idx) { return idx >> 5; }
    static int makeIndex(int base, int n) { return base | (n << 5); }

protected:
    uint32_t free;
    uint8_t locked = 0;
    uint8_t nextPhys = 0;
    uint8_t nflag;

    static uint32_t mask(VirtualFlag vflag) { return mask(vflag.idx, vflag.n); }
    static uint32_t mask(int idx, int n) {
        return (1ul << (idx + n)) - (1ul << idx);
    }
};

// State parameters shared between different kernel types.
struct CommonState {
    ngen::RegisterAllocator ra;
    ngen::GRF signChange, selectImag;
    ngen::GRF vflagStorage;
    std::array<VirtualFlag, 4> activeVFlags;
    VirtualFlagAllocator raVFlag;
    ngen::Subregister readFailures;
    ngen::Subregister fusedID;
    ngen::Subregister lsDescConstant[3];
    ngen::FlagRegister flagSwizzle;
    EmulationState emulate;
    ngen::GRFRange eatomicAddRegs[2];
    ngen::GRFRange remaskRegs;
    VirtualFlag vflagEAtomicAdd;
    ngen::Subregister all1s;
    ngen::RegData r0_info;
    bool movedR0 = false;
    ngen::Subregister lid0;
    struct {
        ngen::GRF zero, one;
        ngen::GRFRange src1Storage;
        ngen::GRF src1, srcR1, srcI1, r, d;
        ngen::GRFRange mathTemp;
        ngen::GRF temp;
        std::array<ngen::FlagRegister, 2> tempFlags;
        ngen::Subregister flagStore; // ud
        ngen::Label label;
        int simd;
        ngen::Subregister callStorageSub, callStorage;
        bool use = false;
    } invertSub;

    CommonState(ngen::HW hw) : ra(hw), raVFlag(hw) {}

    void wipeActiveVFlags() {
        for (int i = 0; i < int(activeVFlags.size()); i++)
            if (!raVFlag.isLocked(VirtualFlag(i))) activeVFlags[i].clear();
    }

    void usePhysicalFlag(ngen::FlagRegister flag) {
        activeVFlags[flag.index()] = flag;
    }

    void allocEmulate64Temp(const EmulationStrategy &estrategy) {
        int ntemp = 0;
        if (estrategy.emulate64) ntemp = std::max(ntemp, 2);
        if (estrategy.emulateDWxDW) ntemp = std::max(ntemp, 1);

        for (int q = 0; q < ntemp; q++)
            emulate.temp[q] = ra.alloc();
    }
};

// Places to store r0 information.
enum class MoveR0 { None, Acc, Addr, GRF };

// Problem parameters shared between kernel types.
struct CommonProblem {
    bool wgSupport
            = true; // Compile kernel with support for nontrivial workgroups?
    bool nonuniformWGs = true; // Support nonuniform workgroups?
    bool gtpinSupport = false; // Support GT-Pin?
    bool fused = false; // Fused kernels?
};

// Strategy parameters shared between different kernel types.
struct CommonStrategy {
    int subgroupSize = 8; // Subgroup size provided to OpenCL runtime.
    bool dualGRF = true; // Enable two-GRF instructions
    bool ieeeDenormals = true; // Enable IEEE-compliant denormals
    bool spf = false; // Enable Single Program Flow (SPF) mode in EUs.
    MoveR0 moveR0 = MoveR0::Acc; // Where to store r0 information.
    bool sipR0WA = false; // Avoid using r0 to avoid clobbering by SIP.
    bool readSuppressionWA
            = true; // Workaround for HW issue with read suppression after fused sends.
    bool wgInSS
            = false; // Pretend to use barriers so that each WG belongs to 1 SS/DSS.
    int GRFs = 128; // # of GRFs to use.
    bool finalFence = false; // Issue global memory fence before EOT.
    int pauseCycles
            = 0x0200; // Number of cycles to pause when waiting in a spin-loop.
    bool simulation = false; // For use in simulator?

    EmulationStrategy emulate;

    void preflight(ngen::HW hw, const CommonProblem &problem);
};

// Driver information, shared by all kernel types.
struct CommonDriverInfo {
    int subgroupSize = 0;
    bool fusedEUs = false;
    int grfCount = 128;
    LoopType loopOrder[3] = {LoopNone, LoopNone, LoopNone};
    int blocking[3] = {0};
    int blockingAlt[3] = {0};
    int unroll[3] = {0};
    int wg[3] = {1, 1, 1};
    int wgExpand = 1;
    bool fixedWG = false;
    bool kRemainderHandling = false;
    bool kParallel = false;
    int kParallelLocal = 0;
    int alignment[3] = {0, 0,
            0}; // Address alignment requirements for A,B,C (gemm) or S,D (copy)

    bool isNMK() const {
        return loopOrder[0] == LoopN || loopOrder[0] == LoopMNHilbertNMK
                || loopOrder[0] == LoopMNBoustrophedonNMK;
    }
    bool isHilbert() const {
        return loopOrder[0] == LoopMNHilbertMNK
                || loopOrder[0] == LoopMNHilbertNMK;
    }
    bool isBoustrophedon() const {
        return loopOrder[0] == LoopMNBoustrophedonMNK
                || loopOrder[0] == LoopMNBoustrophedonNMK;
    }
    bool isLinearOrder() const { return isHilbert() || isBoustrophedon(); }

    int wgTile(LoopType l) const { return unroll[l] * wg[l]; }
};

// Types of updates for GEMM kernels.
enum class UpdateType {
    Full,
    UpperTriangle,
    UpperTriangleHermitian,
    LowerTriangle,
    LowerTriangleHermitian
};

// k loop bounds types for GEMM kernels.
enum class KRange {
    Full,
    ALowerTriangle,
    AUpperTriangle,
    BLowerTriangle,
    BUpperTriangle
};

// Preferences for using scattered accesses.
enum class ScatterSIMD { Default, Wide, Narrow };

// A/B offset mode.
enum class ABOffset {
    None, // No A/B offsets.
    Calc, // Calculate A/B row/column sums in kernel.
    Load, // Use precalculated row/column sums.
};

// C offset mode.
enum class COffset {
    None, // No C offsets.
    Post, // C offset after all other updates.
    Pre, // C offset before all other updates (bias).
};

// Batch mode.
enum class BatchMode { None, Strided, Nonstrided, Variable };

// GEMM kernel problem description.
struct GEMMProblem : public CommonProblem {
    Type Ta, Tb, Tc, Tco, Ts; // Types for A/B/C/C offsets/scalars in registers.
    Type Tc_ext; // Type for C data in memory.

    Scalar<double> alpha_real, alpha_imag; // Alpha value, if fixed.
    Scalar<double> beta_real, beta_imag; // Beta value, if fixed.
    MatrixAddressing A, B, C, CO; // Addressing information for matrices.
    bool kPositive = false; // Can we assume k > 0?
    bool backward = false; // If true, k loop is backwards.
    bool checkBeta0 = true; // If true, check for beta = 0 and handle specially.
    LoopType fusedLoop = LoopM; // Direction of fusing if threads fused.
    ABOffset abOffset = ABOffset::None; // A/B offset mode.
    COffset cOffset = COffset::None; // C offset mode.
    BatchMode batch = BatchMode::None; // Batch mode.
    int batchDims = 0; // # of batch dimensions (strided batch only).
    post_ops_t post_ops;
    bool postOpFwd = true; // Eltwise parameters

    bool hasPostOp() const { return post_ops.len() > 0; }

    bool beta0() const {
        return (beta_real == 0) && (!Tc.isComplex() || (beta_imag == 0));
    }
    bool beta1() const {
        return (beta_real == 1) && (!Tc.isComplex() || (beta_imag == 0));
    }
    bool alpha1() const {
        return (alpha_real == 1) && (!Tc.isComplex() || (alpha_imag == 0));
    }
    bool alphaM1() const {
        return (alpha_real == -1) && (!Tc.isComplex() || (alpha_imag == 0));
    }
    bool fusedM() const { return fused && (fusedLoop == LoopM); }
    bool fusedN() const { return fused && (fusedLoop == LoopN); }

    bool needsTsConvert() const {
        if (!(alpha1() || alphaM1())) return true;
        if (!(beta0() || beta1())) return true;
        if (hasPostOp()) return true;
        return false;
    }
};

struct GEMMState;

// Strategy parameters for GEMM kernels.
struct GEMMStrategy : public CommonStrategy {
    int blocking[3] = {
            0}; // Recommended block size in each dimension (m/n/k) -- for driver.
    int blockingAlt[3] = {
            0}; // Alternate block size in each dimension (m/n/k) -- for driver.
    //     m/n alternates are for Hilbert-ordered kernels when Hilbert ordering disabled.
    //     k alternate is for multi-tile execution with implicit scaling.
    int unroll[3]; // Unrolls in each dimension (m/n/k), indexed by LoopType.
    int unrollK_masked = 0; // k unroll to use when masking.
    LoopType loopOrder[3] = {LoopM, LoopN,
            LoopK}; // Expected order of loops in driver code (in order from innermost to outermost).
    bool hilbertOrder = false; // Use Hilbert-like walk order in C.
    bool boustrophedon = false; // Use panel-boustrophedon walk order in C.
    bool reverse[2] = {false, false}; // Reverse m/n walk order?
    int fmaSIMD; // Vector length for FMA.
    int kChain = 1; // # of FMAs to chain in k dimension.
    int wg[3] = {0, 0,
            1}; // m/n/k workgroup sizes, 0 if unconstrained. Indexed by LoopType.
    bool forceFixedWG
            = false; // If true, always use fixed workgroup size even if not required.
    MatrixAddressingStrategy A, B, C; // Strategies for accessing A/B/C.
    int ka_load, kb_load; // How much of A/B is loaded at once, in k dimension
    int ka_load_masked = 0,
        kb_load_masked = 0; // Same as above, when masking m/n.
    int ka_repack = 0,
        kb_repack = 0; // How often to repack loaded A/B (when crosspacked)
    bool slmA = false, slmB = false; // Whether to copy A/B to SLM.
    bool splitCopy = false; // Separate SLM copy and compute threads?
    int slmBuffers = 0; // # of A/B SLM buffers, 0 for none.
    int unrollKSLM
            = 0; // k unroll for SLM copies (0 = auto = unroll[LoopK]/slmCopies)
    bool slmATrans = false,
         slmBTrans
            = false; // Whether A/B SLM data should be completely crosspacked (transposed).
    int A_copies = 1,
        B_copies = 1; // # of copies of A/B matrices, for latency absorption
    int slmCopies = 1; // # of copies of loaded A/B matrices for SLM copies.
    bool duplicateA = false,
         duplicateB
            = false; // Copy A/B to registers in another bank to avoid conflicts?
    int optAlignAB
            = 0; // Optional alignment for A/B. If > 0, create two versions of k loop, one for A/B aligned to this value, one not.
    int ka_prefetch = 0, kb_prefetch = 0; // Chunk size for prefetching A/B.
    int ka_pfStride = 0, kb_pfStride = 0; // k stride between A/B prefetches.
    bool cooperativePF = true; // Enable WG-cooperative A/B prefetches.
    int prefetchA = 0, prefetchB = 0,
        prefetchC = 0; // Prefetch distances, in units of unrollK.
    MatrixAddressingStrategy A_prefetch, B_prefetch,
            C_prefetch; // Strategies for prefetching A/B/C.
    enum {
        CSeparate, // C stored in its own bundle, A/B in the other bundle.
        ACB, // A, then C, then B
        BCA, // B, then C, then A
        VNC, // A/B (broadcast matrix second), then C
        ABInterleave, // A/B interleaved, then C
        NSeparate, // Broadcast input stored in its own bundle(s)
    } registerScheme
            = CSeparate; // Register layout scheme.
    bool kParallel
            = false; // If true, generate k-parallelized kernel using global memory reduction.
    int kParallelLocal
            = 0; // If > 0, generate k-parallelized kernel using local memory reduction, supporting up to this many threads in k dimension.
    bool doubleWA
            = false; // Use explicit double broadcast instructions? (Gen9 only)
    int barrierFreq = 0; // If > 0, set a barrier every barrierFreq loops
    bool altCRemainder = false; // Use alternative double-loop C remainder code?
    bool cAccumulators
            = false; // Use accumulator registers for part of C (to save a few registers)?
    bool cLoadAhead = false; // Load C before doing FMAs?
    bool forceWideSIMDC = false; // Force wider SIMD for C?
    bool noJumpTables = false; // Disallow jump tables?
    RemainderHandling remHandling[3]; // m, n, k remainder handling.
    bool jointSplit
            = true; // Use remainder kernel for both m and n dimensions if both are split.
    int mSplitThresh = 0,
        nSplitThresh
            = 0; // m/n minimum thresholds for using split remainder handling. 0 means always use split.
    bool atomicFMA = false; // Use {Atomic} FMA chains.
    bool checkAdd32
            = true; // Check inside kernel if inner loop additions can be done in 32-bit.
    bool delayABInc
            = false; // Delay A/B increment a few outer products in the k loop.
    bool slmMBlockSplit
            = false; // Split SLM copies in m/n dimensions instead of the k dimension.
    bool slmNBlockSplit = false;
    bool slmEarlyKMask
            = false; // Prepare A/B reads to use k-masking (when applicable) in main loop, instead of waiting for remainder.
    bool slmAltBarriers = false; // Alternate fenceless SLM buffering algorithm.
    bool skipFence
            = false; // Skip SLM fences that theoretically should be required but HW doesn't need.
    bool systolic = false; // Use systolic array if applicable.
    bool dpasw = false; // Use DPASW (only fixed systolic for now).
    bool fixedSystolic
            = false; // Use hardcoded systolic inner loop for 32x32 or 32x48 unrolls.
    bool xParallel = false; // TRSM: parallelize in x dimension.
    bool checkBeta1
            = false; // If true, check for beta = 1 and handle specially.

    bool insideSK = false; // Inside a superkernel?

    CommonDriverInfo driverInfo(const GEMMProblem &problem) const;

    void preflight(ngen::HW hw, const GEMMProblem &problem);
    bool minimize(ngen::HW hw, const GEMMProblem &problem);

    bool lateExit() const { return (slmBuffers > 0) || kParallelLocal; }

    int maxKSLM(const GEMMState &state, bool isA) const;
    int slmABufBlockSize(Type Ta, const GEMMState &state) const {
        return fixedSystolic
                ? 1152
                : int(slmA) * Ta * unroll[LoopM] * maxKSLM(state, true);
    }
    int slmBBufBlockSize(Type Tb, const GEMMState &state) const {
        return fixedSystolic
                ? 1536
                : int(slmB) * Tb * unroll[LoopN] * maxKSLM(state, false);
    }
    int slmABufSize(Type Ta, const GEMMState &state) const {
        return slmABufBlockSize(Ta, state) * wg[LoopM] * wg[LoopK] * slmBuffers;
    }
    int slmBBufSize(Type Tb, const GEMMState &state) const {
        return slmBBufBlockSize(Tb, state) * wg[LoopN] * wg[LoopK] * slmBuffers;
    }
    int slmSysgemmBlockSize() const {
        return 1152 * wg[LoopM] + 1536 * wg[LoopN];
    }

    int ka_inc() const { return slmA ? unrollKSLM : ka_load; }
    int kb_inc() const { return slmB ? unrollKSLM : kb_load; }

    bool needsMNLocalIDs() const {
        return xParallel || (slmBuffers > 0) || cooperativePF || kParallelLocal;
    }
    bool needsBarrier() const {
        return (barrierFreq > 0) || (slmBuffers > 0) || xParallel;
    }

    bool fixedWG(const GEMMProblem &problem) const {
        return (slmBuffers > 0) || forceFixedWG;
    }

    bool linearOrder() const { return hilbertOrder || boustrophedon; }
};

// State parameters for GEMM kernels.
struct GEMMState : public CommonState {
    struct {
        ngen::Subregister A, B, C[2], CO, base; // q
        ngen::Subregister ao, bo, abo; // w/w/ud
        ngen::Subregister offsetA, offsetB, offsetC[2]; // q
        ngen::Subregister offsetCO; // d
        ngen::Subregister lda, ldb, ldc[2]; // d
        ngen::Subregister m, n, k, k0; // d
        ngen::Subregister alpha_real, alpha_imag; // T_real
        ngen::Subregister beta_real, beta_imag; // T_real
        ngen::Subregister groupIDM, groupIDN, groupIDK; // ud
        ngen::Subregister groupIDMN; // ud
        ngen::GRF localIDM, localIDN, localIDK; // uw
        ngen::Subregister localSizeM, localSizeN, localSizeK; // ud
        ngen::Subregister groupCountM, groupCountN; // ud
        ngen::Subregister hilbertVD, hilbertUVDRecip; // ud
        ngen::Subregister hilbertBail; // ud
        ngen::Subregister bslice, bthresh; // d
        ngen::Subregister mapping; // q
        ngen::Subregister flags; // ud
        ngen::Subregister diagA, diagB, diagC; // q
        uint8_t surfaceA, surfaceB; // BTS indices
        uint8_t surfaceC[2], surfaceCO; // BTS indices
        ngen::Subregister strideA[2], strideB[2],
                strideC[2]; // ud, used for strided batch.
        ngen::Subregister batchSize1, recipBatchSize1; // ud, 2D strided batch
        ngen::Subregister offsetBatch; // ud, used for non-strided batch.
        ngen::Subregister incr_a_array,
                incr_b_array; // ud, used for non-strided variable batch.
        ngen::Subregister incr_alpha,
                incr_beta; // ud, used for non-strided variable batch.
        ngen::Subregister alpha_array,
                beta_array; // q, used for non-strided variable batch.
    } inputs;
    Type Tacc; // Current type in accumulator registers.
    ngen::Subregister batchID[2]; // ud
    ngen::Subregister effA, effB, effC[2],
            effCO; // Offsets to base of A/B/C/CO chunks for loading/storing.
    ngen::Subregister effAi, effBi;
    ngen::Subregister effAo, effBo;
    ngen::Subregister effAp, effBp;
    std::vector<ngen::GRFRange> A_addrs, B_addrs, C_addrs[2];
    std::vector<ngen::GRFRange> Ai_addrs, Bi_addrs;
    std::vector<ngen::GRFRange> Ao_addrs, Bo_addrs;
    std::vector<ngen::GRFRange> Ap_addrs, Bp_addrs, Cp_addrs;
    std::vector<GRFMultirange> A_regs, B_regs, C_regs;
    GRFMultirange A1_regs,
            B1_regs; // Duplicate A/B registers (in opposite banks from A_regs/B_regs).
    GRFMultirange Ar_regs, Br_regs; // Repacked A/B registers.
    std::vector<GRFMultirange> Ai_regs,
            Bi_regs; // Incoming data to copy to SLM.
    GRFMultirange Ao_regs, Bo_regs; // Outgoing data to copy to SLM.
    GRFMultirange As_regs, Bs_regs; // A row sums/B column sums.
    GRFMultirange Ap_regs, Bp_regs, Cp_regs; // A/B/C prefetch registers.
    ngen::GRFRange broadcast_regs;
    std::vector<ngen::GRFRange> tempMul_regs;
    ngen::Subregister i0, j0, h0; // d
    ngen::Subregister remainders[3]; // d (todo: w)
    ngen::Subregister remaindersFused[2]; // w
    ngen::Subregister remaindersWG[2]; // d (todo: w)
    ngen::Subregister remFusedStorage; // d
    ngen::Subregister lda_ka, ldb_kb; // d
    ngen::Subregister lda_ka_prefetch, ldb_kb_prefetch; // d
    int ka_cached, kb_cached; // Multipliers for lda_ka/ldb_kb.
    ngen::Subregister K; // d
    ngen::FlagRegister flagAP;
    ngen::Subregister beta1; // d
    ngen::Subregister add64; // uw
    ngen::Subregister lidM, lidN, lidStorage; // uw, uw, ud
    ngen::Subregister lidK, lszK, lidszKStorage; // uw, uw, ud
    ngen::Subregister ha0_slm, hb0_slm, hab0Storage; // uw, uw, ud
    ngen::Subregister ia0_slm, jb0_slm; // uw
    ngen::Subregister postRemA, postRemB; // ud
    ngen::Subregister postRemAi, postRemBi; // ud
    ngen::Subregister isCompute; // ud
    int ma_slm, ka_slm, kb_slm, nb_slm;
    bool A_slmSplitM = false, B_slmSplitN = false;
    std::vector<RegisterBlock> A_layout, B_layout, C_layout;
    std::vector<RegisterBlock> Ar_layout, Br_layout;
    std::vector<RegisterBlock> Ai_layout, Bi_layout;
    std::vector<RegisterBlock> Ao_layout, Bo_layout;
    std::vector<RegisterBlock> As_layout, Bs_layout;
    std::vector<RegisterBlock> Ap_layout, Bp_layout, Cp_layout;
    std::vector<RegisterBlock> C_layoutExt, C_layoutExtUnmasked;
    bool aioShare, bioShare;
    MatrixAddressing Ai, Bi, Ao, Bo;
    MatrixAddressingStrategy Ai_strategy, Bi_strategy;
    MatrixAddressingStrategy Ao_strategy, Bo_strategy;
    MatrixAddressingStrategy Cext_strategy;
    bool isNested;
    int C_accCount;
    bool cSwapActive = false;
    int C_count = 1;
    bool allocedAo = false, allocedBo = false;
    bool allowEmptyC = false;
    bool copyC = false;
    bool broadcast;

    struct {
        bool active = false;
        uint8_t surfacePlan;
        ngen::Subregister plan;
        ngen::Subregister slotA, slotB;
        ngen::Subregister localIDFlat;
        ngen::FlagRegister needLateGEMMDone;
    } fusedGEMM;

    struct {
        ngen::InstructionModifier depAddr[4];
    } sysgemm;

    GEMMState(ngen::HW hw) : CommonState(hw) {}
};

// GEMM superkernel strategy parameters.
struct GEMMSuperkernelStrategy {
    std::vector<GEMMStrategy> substrategies;
    KernelScheduling schedule;
    bool multiM, multiN;

    void preflight(ngen::HW hw, const GEMMProblem &problem);
    int subgroupSize() const { return substrategies[0].subgroupSize; }
};

// GEMM superkernel state.
struct GEMMSuperkernelState : public GEMMState {
    struct {
        uint8_t surfacePlan;
        ngen::Subregister planCount;
        ngen::GRF localID;
        ngen::Subregister localSize;
    } inputsSK;
    ngen::Subregister last_i0, last_j0, last_h0;

    GEMMSuperkernelState(ngen::HW hw) : GEMMState(hw) {}
};

// Copy kernel problem description: D <- alpha*S
struct CopyProblem : public CommonProblem {
    Type Ts, Td, Tsum;
    Scalar<double> alpha_real, alpha_imag;
    MatrixAddressing S, D;
    bool conjugate;
    bool lower;
    bool unit;
    bool trsm;
    bool sum = false;
    int targetWG = 1;

    struct {
        bool sInc = false; // If true, omit S increments.
        bool sLoad = false; // If true, omit S loads.
        bool dStore = false; // If true, omit D stores.
    } mock;

    bool reflecting() const { return false; }
};

// Strategy parameters for copy kernels.
struct CopyStrategy : public CommonStrategy {
    MatrixAddressingStrategy S, D;
    RemainderHandling remHandlingX,
            remHandlingY; // Remainder handling for X dimension (packed dimension) and Y dimension (length of panel)
    int s_load, d_load; // # of rows/columns to load from S/store to D at once
    int s_load_masked,
            d_load_masked; // Same as s_load/d_load, for use when masking.
    int wgW = 0, wgZ = 0; // Fixed workgroup sizes (0 if variable).

    int unrollX, unrollY; // Unrolls for each dimension.
    bool duplicateAlpha; // True to make two copies of alpha, one for each register bank
    bool xLoop; // True to loop over x, false to loop over y within a kernel

    bool zParallel = false; // Kernel parallelized in z dimension?

    int barrierFreq; // If > 0, set a barrier every barrierFreq loops
    int optionalAlignS; // If > 0, generate code to check if S is aligned to this #elements and branch to specific code for that case.

    CommonDriverInfo driverInfo(const CopyProblem &problem) const;

    void preflight(ngen::HW hw, const CopyProblem &problem);

    int unrollW() const { return xLoop ? unrollY : unrollX; }
    int unrollZ() const { return xLoop ? unrollX : unrollY; }
};

// State parameters for copy kernels.
struct CopyState : public CommonState {
    struct {
        ngen::Subregister S, D; // q
        ngen::Subregister offsetS, offsetD; // q
        ngen::Subregister lds, ldd; // d
        ngen::Subregister m, n; // d
        ngen::Subregister alpha_real; // T_real
        ngen::Subregister alpha_imag; // T_real
        ngen::Subregister groupIDW, groupIDZ; // ud
        ngen::GRF localIDW, localIDZ; // uw
        ngen::Subregister localSizeW, localSizeZ; // ud
        ngen::Subregister diag; // d
        ngen::Subregister blockZ; // ud
        uint8_t surfaceS, surfaceD; // DTS indices
    } inputs;
    ngen::Subregister w0, z0; // ud
    ngen::Subregister effS,
            effD; // Offsets to base of S/D chunks for loading/storing.
    ngen::Subregister offsetS1,
            effS1; // Reflected variants of offsetS/effS for symmetric/Hermitian.
    std::vector<ngen::GRFRange> S_addrs, D_addrs;
    std::vector<ngen::GRFRange> S_addrSrcs[2];
    ngen::GRFRange S_regs, D_regs;
    std::vector<ngen::GRFRange> Ds_regs;
    ngen::Subregister lds_sl; // d
    ngen::Subregister ldd_dl; // d
    ngen::Subregister Z; // d
    ngen::FlagRegister flagAP, flagTri, flagDiag;
    ngen::FlagRegister flagReflect;
    std::vector<RegisterBlock> S_layout, D_layout;
    std::vector<RegisterBlock> Ds_layout;
    ngen::Subregister remainderX, remainderY; // ud
    ngen::GRF indexVec; // w
    ngen::GRFRange complexOne; // T_real

    bool isNested;

    struct {
        bool active = false;
    } fusedGEMM;

    CopyState(ngen::HW hw) : CommonState(hw) {}

    void dump();
};

struct Address2DParams {
    ngen::Subregister rows, cols;
    ngen::Subregister offR, offC;
    ngen::Subregister remR, remC;
    int fixedRows = 0, fixedCols = 0;
};

template <ngen::HW hw>
class gemm_kernel_generator_t : public jit_generator<hw> {
public:
    using super = ngen::OpenCLCodeGenerator<hw>;
    gemm_kernel_generator_t() {}

    NGEN_FORWARD_OPENCL(hw);

    using Injector = jit_post_op_injector<hw>;
    std::unique_ptr<Injector> postOpInjector;

    void gemm(GEMMProblem problem, GEMMStrategy strategy,
            const ngen::NEOInterfaceHandler &interface_);
    void gemmSuperkernel(GEMMProblem problem, GEMMSuperkernelStrategy strategy,
            const ngen::NEOInterfaceHandler &interface_, bool loopless);
    void copy(CopyProblem problem, CopyStrategy strategy,
            const ngen::NEOInterfaceHandler &interface_);

protected:
    ngen::NEOInterfaceHandler
            &interface = ngen::OpenCLCodeGenerator<hw>::interface_;

    std::exception_ptr lastException;

    std::ostream &getOutStream() const { return std::cerr; }

    std::ostream &noteStream() const { return getOutStream(); }

    class status_stream {
    protected:
        char cc;
        std::stringstream line;
        bool lineStart = true;

        gemm_kernel_generator_t<hw> &parent;

        friend class gemm_kernel_generator_t<hw>;

    public:
        status_stream(gemm_kernel_generator_t<hw> &parent_, int color = 1)
            : cc(color + '0'), parent(parent_) {}

        static constexpr struct Endl {
        } endl {};

        template <typename T>
        status_stream &operator<<(const T &obj) {
            return *this;
        }

        status_stream &operator<<(const Endl &e) { return *this; }
    } status {*this};

#ifdef SHOW_DISCARDS
    void discardStream() {
        InstructionStream *s = popStream();
        auto oldCC = status.cc;
        status.cc = '4';
        status << "------- \x1B[32mBEGIN\x1B[34m discarded stream -------"
               << status_stream::endl;
        auto &sbuffer = *reinterpret_cast<std::ostringstream *>(s->getBuffer());
        auto str = sbuffer.str();
        bool lastNL = false;
        for (int l = 0; l < str.length(); l++) {
            char c = str[l];

            if (c == '\n') {
                if (lastNL) status << "//";
                status << status_stream::endl;
                lastNL = true;
            } else {
                status << c;
                lastNL = false;
            }
        }
        status << "-------  \x1B[32mEND\x1B[34m discarded stream  -------"
               << status_stream::endl;
        status.cc = status.cc;
        delete s;
    }
#endif

    enum class HintType {
        Bank0,
        Bank1,
        TempComp0,
        TempComp1,
        LongTerm,
        R0Info,
        A0,
        A0Broadcast,
        A1,
        A1Broadcast,
        B0,
        B0Broadcast,
        B1,
        B1Broadcast,
        C,
        CLoad,
        S,
        D,
        SAddr,
        DAddr
    };
    enum class StdCRemType { Ignore, Mask, Descriptor };
    enum class COperation { Load, Update, UpdateStore };

    friend std::ostream &operator<<(std::ostream &s, StdCRemType rt) {
        const char *names[3] = {"ignore", "mask", "custom descriptor"};
        return (s << names[static_cast<int>(rt)]);
    }

    ngen::FlagRegister getPhysicalFlag(VirtualFlag vflag, CommonState &state);
    void allocVFlagStorage(const CommonStrategy &strategy, CommonState &state);

    ngen::Bundle getHint(HintType type);
    ngen::Bundle getHint(HintType type, const CommonStrategy &strategy);
    ngen::Bundle getHint(HintType type, const GEMMStrategy &strategy);
    ngen::Bundle getHint(HintType type, const CopyStrategy &strategy);

    void goto12(const ngen::InstructionModifier &mod, ngen::Label &jip) {
        goto12(mod, jip, jip);
    }
    void goto12(const ngen::InstructionModifier &mod, ngen::Label &jip,
            ngen::Label &uip, bool branchCtrl = false);

    template <typename DT = void>
    void mulConstant(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0, int32_t src1);

    friend struct EmulationImplementation;
    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, const CommonStrategy &strategy,
            CommonState &state);
    template <typename DT = void>
    void emov(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::Immediate src0, const CommonStrategy &strategy,
            CommonState &state) {
        EmulationImplementation::emov<DT>(
                *this, mod, dst, src0, strategy.emulate);
    }
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const CommonStrategy &strategy, const CommonState &state) {
        EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, strategy.emulate, state.emulate);
    }
    template <typename DT = void>
    void eadd(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1,
            const CommonStrategy &strategy, const CommonState &state) {
        EmulationImplementation::eadd<DT>(
                *this, mod, dst, src0, src1, strategy.emulate, state.emulate);
    }
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const ngen::RegData &src1,
            const CommonStrategy &strategy, const CommonState &state) {
        EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, strategy.emulate, state.emulate);
    }
    template <typename DT = void>
    void emul(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, ngen::Immediate src1,
            const CommonStrategy &strategy, const CommonState &state) {
        EmulationImplementation::emul<DT>(
                *this, mod, dst, src0, src1, strategy.emulate, state.emulate);
    }
    template <typename DT = void>
    void eshl(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1, const CommonStrategy &strategy,
            const CommonState &state) {
        EmulationImplementation::eshl<DT>(
                *this, mod, dst, src0, src1, strategy.emulate, state.emulate);
    }
    template <typename DT = void>
    void eshr(const ngen::InstructionModifier &mod, ngen::RegData dst,
            ngen::RegData src0, uint16_t src1, const CommonStrategy &strategy,
            const CommonState &state) {
        EmulationImplementation::eshr<DT>(
                *this, mod, dst, src0, src1, strategy.emulate, state.emulate);
    }
    template <typename DT = void>
    void emulConstant(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0, int32_t src1,
            const CommonStrategy &strategy, const CommonState &state) {
        EmulationImplementation::emulConstant<DT>(
                *this, mod, dst, src0, src1, strategy.emulate, state.emulate);
    }
    template <typename S1>
    void emul32High(const ngen::InstructionModifier &mod,
            const ngen::RegData &dstHi, const ngen::RegData &src0,
            const S1 &src1) {
        EmulationImplementation::emul32High(*this, mod, dstHi, src0, src1);
    }

    template <typename S0, typename S2>
    void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const S0 &src0, const ngen::RegData &src1, const S2 &src2,
            const CommonStrategy &strategy, CommonState &state);
    template <typename S0>
    void emad(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const S0 &src0, const ngen::RegData &src1, int32_t src2,
            const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void, typename S0, typename S2>
    void eadd3(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const S0 &src0, const ngen::RegData &src1, const S2 &src2);

    template <typename DT = void>
    void emath(const ngen::InstructionModifier &mod, ngen::MathFunction fc,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const GEMMStrategy &strategy, CommonState &state);
    template <typename DT = void>
    void einv(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const GEMMStrategy &strategy,
            CommonState &state) {
        emath<DT>(mod, ngen::MathFunction::inv, dst, src0, strategy, state);
    }
    template <typename DT = void>
    void esqt(const ngen::InstructionModifier &mod, const ngen::RegData &dst,
            const ngen::RegData &src0, const GEMMStrategy &strategy,
            CommonState &state) {
        emath<DT>(mod, ngen::MathFunction::sqt, dst, src0, strategy, state);
    }

    void ejmpi(ngen::InstructionModifier mod, ngen::Label &dst);

    void cmp0(const ngen::InstructionModifier &mod, ngen::RegData src0);
    void syncall();

    void wrdepRanges(const std::vector<GRFMultirange> &rrs) {
        for (auto &rr : rrs)
            for (auto &r : rr.ranges)
                wrdep(r);
    }

    void addScaled(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, int src0, const ngen::RegData &src1,
            int numerator, int denominator, CommonState &state,
            bool exact = false);
    void addScaled(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0,
            const ngen::RegData &src1, int numerator, int denominator,
            CommonState &state, bool exact = false);
    void addScaled(const ngen::InstructionModifier &mod,
            const ngen::RegData &dst, const ngen::RegData &src0, int src1,
            int numerator, int denominator, CommonState &state,
            bool exact = false);

    template <typename DT = void>
    void mod(const ngen::Subregister &dst, const ngen::Subregister &src,
            uint16_t modulus, const CommonStrategy &strategy,
            CommonState &state);
    template <typename DT = void>
    void modExt(const ngen::Subregister &dstMod,
            const ngen::Subregister &dstMultiple, const ngen::Subregister &src,
            uint16_t modulus, const CommonStrategy &strategy,
            CommonState &state);
    template <typename DT = void>
    void alignDown(const ngen::Subregister &dst, const ngen::Subregister &src,
            uint16_t align, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void>
    void alignUp(const ngen::Subregister &dst, const ngen::Subregister &src,
            uint16_t align, const CommonStrategy &strategy, CommonState &state);
    template <typename DT = void>
    void divDown(const ngen::Subregister &dst, const ngen::Subregister &src0,
            const ngen::Subregister &src1, const ngen::Subregister &src1Recip,
            const ngen::FlagRegister &flag, const CommonStrategy &strategy,
            CommonState &state);
    template <typename DT = void>
    void divDown(const ngen::Subregister &dst, const ngen::Subregister &src,
            uint16_t divisor, const CommonStrategy &strategy,
            CommonState &state);

    void simtDoWhileLoop(
            const ngen::InstructionModifier &mod, ngen::Label &dest);
    void slmBarrier(const ngen::GRF &temp, const ngen::GRF &r0_info = r0);
    void globalMemBarrier(const ngen::GRF &temp, const ngen::GRF &r0_info = r0);
    void pause(const CommonStrategy &strategy);

    template <typename T>
    void duplicateScalar(Scalar<T> &val, CommonState &state);
    MultishiftSubregister multishift(const ngen::Subregister &reg,
            unsigned shifts, const CommonStrategy &strategy, CommonState &state,
            ngen::Bundle hint = ngen::Bundle());

    void getFusedID(int scale, const CommonProblem &problem,
            const CommonStrategy &strategy, CommonState &state);
    void moveR0(const CommonStrategy &strategy, CommonState &state);
    void moveR0(const GEMMStrategy &strategy, GEMMState &state);
    template <typename F>
    void useR0(CommonState &state, F f);
    void removeSG(const CommonProblem &problem, const CommonStrategy &strategy,
            const CommonState &state);
    void reorderFusedEUs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    ngen::Subregister copySubregister(const ngen::Subregister &reg,
            CommonState &state,
            ngen::Bundle hint = ngen::Bundle(ngen::Bundle::any, 0));
    void zeroMatrix(const GRFMultirange &r, const CommonStrategy &strategy);
    void releaseFusedRemainders(GEMMState &state);
    void saveMNLocalIDs(const GEMMStrategy &strategy, GEMMState &state);
    void releaseSavedMNLocalIDs(GEMMState &state);

    void doReadSuppressionWA(
            const CommonStrategy &strategy, CommonState &state);

    bool getBlockInfo(Type T, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy, int r, int c,
            bool remainderR, bool remainderC, bool writable, bool avoidFragment,
            ScatterSIMD smode, int maxRBlock, int maxCBlock, int &rblock,
            int &cblock, RegisterBlock &layout);
    bool getSubblock(Type T, RegisterBlock &blockDst,
            const RegisterBlock &blockSrc, bool column, int x1, int x2,
            int x1Unclamped, int x2Unclamped, bool overrunOK,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getSubblocks(Type T, std::vector<RegisterBlock> &sublayout,
            const std::vector<RegisterBlock> &layout, bool column, int x1,
            int x2, bool overrunOK, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getSubblocks(Type T, std::vector<RegisterBlock> &sublayout,
            std::vector<ngen::GRFRange> *subaddrs, std::vector<int> *indices,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> *addrs, bool column, int x1,
            int x2, bool overrunOK, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getSubblocks(Type T, std::vector<RegisterBlock> &sublayout,
            std::vector<ngen::GRFRange> &subaddrs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, bool column, int x1,
            int x2, bool overrunOK, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getSubblocks(Type T, std::vector<RegisterBlock> &sublayout,
            std::vector<int> &indices, const std::vector<RegisterBlock> &layout,
            bool column, int x1, int x2, bool overrunOK,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool reblockLayout(Type Tdst, std::vector<int32_t> &blockMap,
            std::vector<RegisterBlock> &layoutDst,
            const std::vector<RegisterBlock> &layoutRef,
            const std::vector<RegisterBlock> &layoutSrc,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);

    bool tryAddMasking(Type T, RegisterBlock &block, bool remainderR,
            bool remainderC, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool tryAddMasking(Type T, std::vector<RegisterBlock> &layout,
            bool remainderR, bool remainderC, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    void addMasking(Type T, std::vector<RegisterBlock> &layout, bool remainderR,
            bool remainderC, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    void addMasking(Type T, std::vector<RegisterBlock> &layout,
            std::vector<ngen::GRFRange> &addrs, const ngen::Subregister &ld,
            bool remainderR, bool remainderC, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    void adjustSubblockAddrs(Type T,
            const std::vector<RegisterBlock> &sublayout,
            const std::vector<ngen::GRFRange> &subaddrs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, const CommonState &state);

    bool addToRegLayout(Type T, std::vector<RegisterBlock> &layout, int r,
            int c, int roff, int coff, bool remainderR, bool remainderC,
            bool writable, bool avoidFragment, ScatterSIMD smode, int maxRBlock,
            int maxCBlock, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool add1DBlockToRegLayout(Type T, std::vector<RegisterBlock> &layout,
            int r, int c, bool writable, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    bool getRegLayout(Type T, std::vector<RegisterBlock> &layout, int r, int c,
            bool remainderR, bool remainderC, bool writable, bool avoidFragment,
            ScatterSIMD smode, int maxRBlock, int maxCBlock,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy);
    void makeUnbackedRegLayout(Type T, std::vector<RegisterBlock> &layout,
            int r, int c, bool colMajor, int crosspack = 1, int tileR = 0,
            int tileC = 0);

    void setupTeardownLoadStoreDesc(
            bool setup, const CommonStrategy &strategy, CommonState &state);
    void loadLoadStoreDescriptors(bool load, bool store, RegisterBlock &block,
            const ngen::Subregister &count, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);

    static ngen::DataSpecLSC getDataSpecLSC(
            AccessType access, const RegisterBlock &block);
    static ngen::DataSpecLSC getDataSpecLSC(const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const RegisterBlock &block);
    ngen::InstructionModifier getRegisterBlockMask(
            const RegisterBlock &block, CommonState &state);
    void loadMatrixBlock(const ngen::Register &dest,
            const RegisterBlock &layout, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const ngen::GRFRange &addr, const CommonStrategy &strategy,
            CommonState &state, bool zeroMask = false);
    void loadMatrix(const GRFMultirange &dest,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const std::vector<ngen::GRFRange> &addrs,
            const CommonStrategy &strategy, CommonState &state,
            bool zeroMask = false);
    void prefetchMatrix(const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const std::vector<ngen::GRFRange> &addrs,
            const CommonStrategy &strategy, CommonState &state);
    void storeMatrixBlock(const ngen::GRF &src, const RegisterBlock &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const ngen::GRFRange &addr, const CommonStrategy &strategy,
            CommonState &state);
    void storeMatrix(const GRFMultirange &src,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const std::vector<ngen::GRFRange> &addrs,
            const CommonStrategy &strategy, CommonState &state);
    void atomicAddMatrixBlock(Type T, const ngen::GRF &src,
            const RegisterBlock &layout, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const ngen::GRFRange &addr, const CommonProblem &problem,
            const CommonStrategy &strategy, CommonState &state);
    void atomicAddMatrix(Type T, const GRFMultirange &src,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const std::vector<ngen::GRFRange> &addrs,
            const CommonProblem &problem, const CommonStrategy &strategy,
            CommonState &state);

    bool assignMasks(std::vector<RegisterBlock> &layout, LoopType rloop,
            LoopType cloop, std::vector<MaskAssignment> &assignments,
            CommonState &state);
    void loadMask(MaskAssignment assignment, ngen::Subregister index,
            CommonState &state);
    void loadMasks(const std::vector<MaskAssignment> &assignments,
            ngen::Subregister (&indices)[3], CommonState &state, int start = 0);

    void setupTeardownRemask(Type T, bool setup, int nq,
            const ngen::Subregister &remQ, const CommonStrategy &strategy,
            CommonState &state,
            const ngen::Subregister &offQ = ngen::Subregister());
    void remaskLayout(Type T, bool column,
            const std::vector<RegisterBlock> &layout, const GRFMultirange &regs,
            const CommonStrategy &strategy, CommonState &state, int offset = 0);

    ngen::Subregister startShift(
            const MultishiftSubregister &ptr, int shift, CommonState &state);
    template <typename BO>
    typename std::enable_if<!std::is_base_of<ngen::RegData, BO>::value,
            BO>::type
    startShift(const BO &ptr, int shift, CommonState &state);
    template <typename BO>
    typename std::enable_if<std::is_base_of<ngen::RegData, BO>::value, BO>::type
    startShift(const BO &ptr, int shift, CommonState &state);
    template <typename BO, typename BI>
    typename std::enable_if<!std::is_base_of<ngen::RegData, BO>::value>::type
    doneShift(
            const BO &ptr, const BI &ptrShifted, int shift, CommonState &state);
    template <typename BO, typename BI>
    typename std::enable_if<std::is_base_of<ngen::RegData, BO>::value>::type
    doneShift(
            const BO &ptr, const BI &ptrShifted, int shift, CommonState &state);

    template <typename BO>
    void setupAddrShifted(const ngen::GRFRange &addr, const BO &ptr,
            const RegisterBlock &layout, const ngen::Subregister &ld,
            size_t sizeofT, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state,
            const Address2DParams &params = {});
    template <typename BO>
    void setupAddr(const ngen::GRFRange &addr, const BO &ptr,
            const RegisterBlock &layout, const ngen::Subregister &ld,
            size_t sizeofT, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state,
            const Address2DParams &params = {});
    template <typename BO>
    void setupAddr(Type T, const std::vector<ngen::GRFRange> &addr,
            const BO &ptr, const std::vector<RegisterBlock> &layout,
            const ngen::Subregister &ld, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state,
            const Address2DParams &params = {});
    template <typename I, typename Ir, typename Ic>
    void incAddrShifted(const ngen::GRFRange &addrDst,
            const ngen::GRFRange &addrSrc, I inc, Ir incR, Ic incC,
            const RegisterBlock &layoutDst, const RegisterBlock &layoutSrc,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I, typename Ir, typename Ic>
    void incAddrShifted(const std::vector<ngen::GRFRange> &addr, I inc, Ir incR,
            Ic incC, const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I>
    void incAddrShifted(const std::vector<ngen::GRFRange> &addr, I inc,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I, typename Ir, typename Ic>
    void incAddr(const ngen::GRFRange &addrDst, const ngen::GRFRange &addrSrc,
            I inc, Ir incR, Ic incC, const RegisterBlock &layoutDst,
            const RegisterBlock &layoutSrc, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I, typename Ir, typename Ic>
    void incAddr(const std::vector<ngen::GRFRange> &addr, I inc, Ir incR,
            Ic incC, const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I>
    void incAddr(const ngen::GRFRange &addrDst, const ngen::GRFRange &addrSrc,
            I inc, const RegisterBlock &layoutDst,
            const RegisterBlock &layoutSrc, const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename I>
    void incAddr(const std::vector<ngen::GRFRange> &addr, I inc,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state);
    template <typename A, typename I, typename Ir, typename Ic>
    void incDecAddr(const A &addr, I inc, Ir incR, Ic incC,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state, bool decrement);
    template <typename A, typename I>
    void incDecAddr(const A &addr, I inc,
            const std::vector<RegisterBlock> &layout,
            const MatrixAddressing &atype,
            const MatrixAddressingStrategy &astrategy,
            const CommonStrategy &strategy, CommonState &state, bool decrement);

    void setupCAddr0(ngen::GRFRange (&C_addr0)[2],
            ngen::GRFRange (&C_addr0Unmasked)[2],
            const std::vector<RegisterBlock> &C_layout,
            const std::vector<RegisterBlock> &C_layoutUnmasked, int C_count,
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    void outerProductGen9IGEMM(int ha, int hb,
            const std::vector<RegisterBlock> &A_layout,
            const std::vector<RegisterBlock> &B_layout,
            const GRFMultirange &A_regs, const GRFMultirange &B_regs,
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void outerProductSystolic(int h, int ha, int hb,
            const std::vector<RegisterBlock> &A_layout,
            const std::vector<RegisterBlock> &B_layout,
            const GRFMultirange &A_regs, const GRFMultirange &B_regs,
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void outerProduct(int h, int ha, int hb, int opCount,
            const std::vector<RegisterBlock> &A_layout,
            const std::vector<RegisterBlock> &B_layout,
            const GRFMultirange &A_regs, const GRFMultirange &B_regs,
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    void updateC(const GRFMultirange &C_acc, const GRFMultirange &C_accSwap,
            const GRFMultirange &C_load, GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
    void updateCLayout(const std::vector<RegisterBlock> &layoutExt,
            const ngen::GRFRange (&C_addr0)[2], COperation op,
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool doStdCRemainder(std::vector<RegisterBlock> &layoutExt,
            std::vector<RegisterBlock> &layoutExtUnmasked, bool inside,
            bool columns[2], StdCRemType remTypes[2], bool fragments[2],
            bool fragPositives[2], int fragSizes[2],
            const ngen::GRFRange (&C_addr0)[2],
            const ngen::GRFRange (&C_addr0Unmasked)[2], COperation op,
            std::vector<MaskAssignment> &masks, GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState state);
    void doAlternateCRemainder(COperation op, GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);

    void accumulateSum(bool column, Type Tsrc, const GRFMultirange &srcRegs,
            const std::vector<RegisterBlock> &srcLayout, Type Tdst,
            const GRFMultirange &dstRegs,
            const std::vector<RegisterBlock> &dstLayout,
            const CommonStrategy &strategy, CommonState &state);
    void makeSumLayout(bool column, Type Tsrc,
            const std::vector<RegisterBlock> &srcLayout, Type Tdst,
            std::vector<RegisterBlock> &dstLayout,
            const CommonStrategy &strategy, CommonState &state);
    void horizontalAdd(bool column, Type T, const GRFMultirange &regs,
            std::vector<RegisterBlock> &layout);
    bool gemmFinalizeSums(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);

    bool maySLMSplitN(const GEMMProblem &problem, const GEMMStrategy &strategy);
    bool maySLMSplitM(const GEMMProblem &problem, const GEMMStrategy &strategy);

    void convert(const GRFMultirange &range, Type Told, Type Tnew,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    bool gemmConvertC(Type Tnew, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmBetaScale(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmFixedOffsetC(const ngen::Subregister &offset,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void gemmVariableOffsetC(bool column, const GRFMultirange &offsets,
            const ngen::Subregister &scale, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state,
            std::vector<RegisterBlock> CO = std::vector<RegisterBlock>());
    bool gemmLoadABOffset(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmApplyABOffset(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    bool gemmApplyCOffset(bool row, bool column, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    bool gemmApplyCOffsetDispatch(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmKReduce(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);

    void gemmAllocRegs(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmAllocAoBoRegs(
            bool forceAlloc, const GEMMStrategy &strategy, GEMMState &state);
    void doAIncrementInternal(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy, int ka_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void doAIncrementInternal(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy,
            const MultishiftSubregister &ka_inc, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void doAIncrementInternal(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy,
            const ngen::Subregister &ka_inc, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    template <typename I>
    void doAIncrement(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy, I ka_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void doALoad(const GRFMultirange &regs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    template <typename I>
    void doALoadInc(const GRFMultirange &regs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &A,
            const MatrixAddressingStrategy &A_strategy, I ka_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void doBIncrementInternal(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy, int kb_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void doBIncrementInternal(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy,
            const MultishiftSubregister &kb_inc, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void doBIncrementInternal(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy,
            const ngen::Subregister &kb_inc, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    template <typename I>
    void doBIncrement(const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy, I kb_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void doBLoad(const GRFMultirange &regs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    template <typename I>
    void doBLoadInc(const GRFMultirange &regs,
            const std::vector<RegisterBlock> &layout,
            const std::vector<ngen::GRFRange> &addrs, const MatrixAddressing &B,
            const MatrixAddressingStrategy &B_strategy, I kb_inc,
            const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void gemmCalcIncrements(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int ka_load = 0,
            int kb_load = 0);
    void gemmCalcAiOffset(ngen::Subregister &off, ngen::Subregister &offR,
            ngen::Subregister &offC, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmCalcBiOffset(ngen::Subregister &off, ngen::Subregister &offR,
            ngen::Subregister &offC, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    bool gemmPrepMaskedAB(const GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state);

    bool gemmKLoop(int ka_repack, int kb_repack, bool lateKLoopCheck,
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmAccumulateC(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmAccessC(COperation op, GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
    bool gemmUpdateC(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    bool gemmBody(GEMMProblem problem, GEMMStrategy strategy, GEMMState state);
    bool gemmBodyInternal(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    bool wgRemCheck(const GEMMProblem &problem, const GEMMStrategy &strategy);
    template <typename Problem>
    bool mnRemainderHandling(LoopType loop, Problem &problem,
            GEMMStrategy &strategy, GEMMState &state,
            bool (gemm_kernel_generator_t<hw>::*func)(
                    Problem, GEMMStrategy, GEMMState));
    template <typename Problem>
    bool mnJointSplitRemainderHandling(Problem &problem, GEMMStrategy &strategy,
            GEMMState &state,
            bool (gemm_kernel_generator_t<hw>::*func)(
                    Problem, GEMMStrategy, GEMMState));
    bool gemmMEdge(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    bool gemmNEdge(GEMMProblem problem, GEMMStrategy strategy, GEMMState state);

    void gemmHilbertlikeOrder(const GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);
    void gemmBoustrophedonOrder(const GEMMProblem &problem,
            GEMMStrategy &strategy, GEMMState &state);

    void gemmCheck32(const GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state);
    void gemmTypeCheck(Type Ta, Type Tb, Type Tc);
    void gemmGetBatchIDs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmReleaseBatchIDs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmOffsetAk(int h, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmOffsetBk(int h, const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void gemmOffsetABC(bool initial, ngen::Subregister i0, ngen::Subregister j0,
            ngen::Subregister h0, GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state, bool doA = true, bool doB = true,
            bool doC = true);
    void gemmOffsetBatchABC(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);
    void gemmSetupABC(GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state, bool doA = true, bool doB = true,
            bool doC = true);
    void gemmSubkernel(
            GEMMProblem &problem, GEMMStrategy &strategy, GEMMState state);
    size_t gemmSLMSize(const GEMMProblem &problem, const GEMMStrategy &strategy,
            const GEMMState &state);
    void gemmInitInterface(GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state, bool inSK = false);
    void gemmInitState(GEMMProblem &problem, GEMMStrategy &strategy,
            GEMMState &state, bool inSK = false);
    void gemm(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state);

    void gemmSuperkernelInitState(GEMMProblem &problem,
            GEMMSuperkernelStrategy &strategy, GEMMSuperkernelState &state,
            bool loopless);

    bool sysgemmAccumulateC(GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void sysgemmKLoop(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void sysgemmKLoop4(const GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state, bool oddB);
    void sysgemmStoreSignal(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state,
            bool forceFence = false);
    void sysgemmCopyLoad(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int storeBuffer,
            bool useC = false);
    void sysgemmCopyLoad4(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int storeBuffer,
            bool loadB, int useC = 0,
            ngen::RegData flagLoadB = ngen::RegData());
    void sysgemmCopyStore(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int storeBuffer,
            bool first = false);
    void sysgemmCopyStore4(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int storeBuffer,
            bool storeB, int useC = 0, int useC_B = 0);
    void sysgemmMultiply(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int buffer,
            bool lastMultiply = false);
    void sysgemmMultiply4(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int buffer,
            bool firstMultiply = false,
            ngen::RegData flagWaitLoad = ngen::RegData(),
            ngen::RegData flagSignal = ngen::RegData(),
            ngen::Label *labelDone = nullptr);
    void sysgemmMultiplyChunk(const GEMMProblem &problem,
            const GEMMStrategy &strategy, bool first, int ao, int i0,
            bool waitB, bool prepB,
            const ngen::InstructionModifier &swsb0
            = ngen::InstructionModifier(),
            const ngen::InstructionModifier &swsbEnd
            = ngen::InstructionModifier());
    void sysgemmBarrierPrep(
            const ngen::InstructionModifier &swsb, const ngen::GRF &header);
    void sysgemmReorderLocalIDs(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);

    bool sysgemm2AccumulateC(GEMMProblem &problem, const GEMMStrategy &strategy,
            GEMMState &state);
    void sysgemm2KLoopCompute(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void sysgemm2KLoopCopy(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state);
    void sysgemm2Multiply(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int buffer,
            bool cooldown = false,
            ngen::FlagRegister flagWaitLoad = ngen::FlagRegister(),
            ngen::FlagRegister flagSignal = ngen::FlagRegister());
    void sysgemm2MultiplyX32(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int buffer,
            bool cooldown = false,
            ngen::FlagRegister flagWaitLoad = ngen::FlagRegister(),
            ngen::FlagRegister flagSignal = ngen::FlagRegister());
    void sysgemm2MultiplyX48(const GEMMProblem &problem,
            const GEMMStrategy &strategy, GEMMState &state, int buffer,
            bool cooldown = false,
            ngen::FlagRegister flagWaitLoad = ngen::FlagRegister(),
            ngen::FlagRegister flagSignal = ngen::FlagRegister());
    void sysgemm2MultiplyChunkX32(const GEMMProblem &problem,
            const GEMMStrategy &strategy, int chunkA, bool odd);
    void sysgemm2MultiplyChunkX48(const GEMMProblem &problem,
            const GEMMStrategy &strategy, int chunkA);

    bool copyRegisterBlock(Type Ts, Type Td, const RegisterBlock &blockSrc,
            const RegisterBlock &blockDst, const ngen::GRFRange &src,
            const ngen::GRFRange &dst, int dOffR, int dOffC,
            const CommonStrategy &strategy, CommonState &state,
            bool preserveSrc = false);
    bool copyRegisters(Type Ts, Type Td,
            const std::vector<RegisterBlock> &layoutSrc,
            const std::vector<RegisterBlock> &layoutDst,
            const GRFMultirange &src, const GRFMultirange &dst, int dOffR,
            int dOffC, bool conjugate, const CommonStrategy &strategy,
            CommonState &state, bool preserveSrc = false);
    bool copyRegisters(Type Ts, Type Td,
            const std::vector<RegisterBlock> &layoutSrc,
            const std::vector<RegisterBlock> &layoutDst,
            const GRFMultirange &src, const GRFMultirange &dst, int dOffR,
            int dOffC, const Scalar<double> &alpha_real,
            const Scalar<double> &alpha_imag, bool conjugate,
            const CommonStrategy &strategy, CommonState &state,
            bool preserveSrc = false);

    bool copyBody(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);
    bool copyBodyRemCheck(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);
    bool copyBodyInternal(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);
    void copySlice(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);

    void copyCalcIncrements(const CopyProblem &problem,
            const CopyStrategy &strategy, CopyState &state, int s_load = 0,
            int d_load = 0);

    void copyInitInterface(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);
    void copyInitState(
            CopyProblem &problem, CopyStrategy &strategy, CopyState &state);
    void copy(CopyProblem &problem, CopyStrategy &strategy, CopyState &state);

    void prologue(const CommonStrategy &strategy);
    void epilogue(const CommonStrategy &strategy, const CommonState &state);
    void padding();
    void initState(const CommonProblem &problem, const CommonStrategy &strategy,
            CommonState &state);
};

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif /* header guard */

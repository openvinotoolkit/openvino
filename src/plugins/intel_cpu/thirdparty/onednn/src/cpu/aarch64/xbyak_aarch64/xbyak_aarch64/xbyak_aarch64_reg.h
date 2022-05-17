#pragma once
/*******************************************************************************
 * Copyright 2019-2021 FUJITSU LIMITED
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

#include <cassert>

class Operand {
public:
  static const int VL = 4;
  enum Kind { NONE, RREG, VREG_SC, VREG_VEC, ZREG, PREG_Z, PREG_M, OPMASK };

  enum Code {
    X0 = 0,
    X1,
    X2,
    X3,
    X4,
    X5,
    X6,
    X7,
    X8,
    X9,
    X10,
    X11,
    X12,
    X13,
    X14,
    X15,
    X16 = 16,
    X17,
    X18,
    X19,
    X20,
    X21,
    X22,
    X23,
    X24,
    X25,
    X26,
    X27,
    X28,
    X29,
    X30,
    SP = 31,
    XZR = 31,
    W0 = 0,
    W1,
    W2,
    W3,
    W4,
    W5,
    W6,
    W7,
    W8,
    W9,
    W10,
    W11,
    W12,
    W13,
    W14,
    W15,
    W16 = 16,
    W17,
    W18,
    W19,
    W20,
    W21,
    W22,
    W23,
    W24,
    W25,
    W26,
    W27,
    W28,
    W29,
    W30,
    WSP = 31,
    WZR = 31,
  };

private:
  Kind kind_;
  uint32_t bit_;

public:
  explicit Operand(Kind kind, uint32_t bit) : kind_(kind), bit_(bit) {}
  uint32_t getBit() const { return bit_; }
  bool isRReg() const { return is(RREG); }
  bool isVRegSc() const { return is(VREG_SC); }
  bool isVRegVec() const { return is(VREG_VEC); }
  bool isZReg() const { return is(ZREG); }
  bool isPRegZ() const { return is(PREG_Z); }
  bool isPRegM() const { return is(PREG_M); }

private:
  bool is(Kind kind) const { return (kind_ == kind); }
};

class Reg : public Operand {
  uint32_t index_;

public:
  explicit Reg(uint32_t index, Kind kind, uint32_t bit) : Operand(kind, bit), index_(index) {}
  uint32_t getIdx() const { return index_; }
};

// General Purpose Register
class RReg : public Reg {
public:
  explicit RReg(uint32_t index, uint32_t bit) : Reg(index, RREG, bit) {}
};

class XReg : public RReg {
public:
  explicit XReg(uint32_t index) : RReg(index, 64) {}
};

class WReg : public RReg {
public:
  explicit WReg(uint32_t index) : RReg(index, 32) {}
};

// SIMD & FP scalar regisetr
class VRegSc : public Reg {
public:
  explicit VRegSc(uint32_t index, uint32_t bit) : Reg(index, VREG_SC, bit) {}
};

class BReg : public VRegSc {
public:
  explicit BReg(uint32_t index) : VRegSc(index, 8) {}
};
class HReg : public VRegSc {
public:
  explicit HReg(uint32_t index) : VRegSc(index, 16) {}
};
class SReg : public VRegSc {
public:
  explicit SReg(uint32_t index) : VRegSc(index, 32) {}
};
class DReg : public VRegSc {
public:
  explicit DReg(uint32_t index) : VRegSc(index, 64) {}
};
class QReg : public VRegSc {
public:
  explicit QReg(uint32_t index) : VRegSc(index, 128) {}
};

// base for SIMD vector regisetr
class VRegVec : public Reg {
  uint32_t lane_;

public:
  explicit VRegVec(uint32_t index, uint32_t bits, uint32_t lane) : Reg(index, VREG_VEC, bits), lane_(lane){};
  uint32_t getLane() const { return lane_; }
};

// SIMD vector regisetr element
class VRegElem : public VRegVec {
  uint32_t elem_idx_;

public:
  explicit VRegElem(uint32_t index, uint32_t eidx, uint32_t bit, uint32_t lane) : VRegVec(index, bit, lane), elem_idx_(eidx) {}
  uint32_t getElemIdx() const { return elem_idx_; }
};

// base for SIMD Vector Register List
class VRegList : public VRegVec {
  uint32_t len_;

public:
  explicit VRegList(const VRegVec &s) : VRegVec(s.getIdx(), s.getBit(), s.getLane()), len_(s.getIdx() - s.getIdx() + 1) {}
  explicit VRegList(const VRegVec &s, const VRegVec &e) : VRegVec(s.getIdx(), s.getBit(), s.getLane()), len_(((e.getIdx() + 32 - s.getIdx()) % 32) + 1) {}
  uint32_t getLen() const { return len_; }
};

class VReg4B;
class VReg8B;
class VReg16B;
class VReg2H;
class VReg4H;
class VReg8H;
class VReg2S;
class VReg4S;
class VReg1D;
class VReg2D;
class VReg1Q;

class VRegBElem : public VRegElem {
public:
  explicit VRegBElem(uint32_t index, uint32_t eidx, uint32_t lane) : VRegElem(index, eidx, 8, lane) {}
};
class VRegHElem : public VRegElem {
public:
  explicit VRegHElem(uint32_t index, uint32_t eidx, uint32_t lane) : VRegElem(index, eidx, 16, lane) {}
};
class VRegSElem : public VRegElem {
public:
  explicit VRegSElem(uint32_t index, uint32_t eidx, uint32_t lane) : VRegElem(index, eidx, 32, lane) {}
};
class VRegDElem : public VRegElem {
public:
  explicit VRegDElem(uint32_t index, uint32_t eidx, uint32_t lane) : VRegElem(index, eidx, 64, lane) {}
};
class VRegQElem : public VRegElem {
public:
  explicit VRegQElem(uint32_t index, uint32_t eidx, uint32_t lane) : VRegElem(index, eidx, 128, lane) {}
};

class VReg4BList : public VRegList {
public:
  VReg4BList(const VReg4B &s);
  VReg4BList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
  VRegBElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegBElem(getIdx(), i, getLane());
  }
};
class VReg8BList : public VRegList {
public:
  VReg8BList(const VReg8B &s);
  VReg8BList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
  VRegBElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegBElem(getIdx(), i, getLane());
  }
};
class VReg16BList : public VRegList {
public:
  VReg16BList(const VReg16B &s);
  VReg16BList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
  VRegBElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegBElem(getIdx(), i, getLane());
  }
};
class VReg2HList : public VRegList {
public:
  VReg2HList(const VReg2H &s);
  VReg2HList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
  VRegHElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegHElem(getIdx(), i, getLane());
  }
};
class VReg4HList : public VRegList {
public:
  VReg4HList(const VReg4H &s);
  VReg4HList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
  VRegHElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegHElem(getIdx(), i, getLane());
  }
};
class VReg8HList : public VRegList {
public:
  VReg8HList(const VReg8H &s);
  VReg8HList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
  VRegHElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegHElem(getIdx(), i, getLane());
  }
};
class VReg2SList : public VRegList {
public:
  VReg2SList(const VReg2S &s);
  VReg2SList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
  VRegSElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegSElem(getIdx(), i, getLane());
  }
};
class VReg4SList : public VRegList {
public:
  VReg4SList(const VReg4S &s);
  VReg4SList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
  VRegSElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegSElem(getIdx(), i, getLane());
  }
};
class VReg1DList : public VRegList {
public:
  VReg1DList(const VReg1D &s);
  VReg1DList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
  VRegDElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegDElem(getIdx(), i, getLane());
  }
};
class VReg2DList : public VRegList {
public:
  VReg2DList(const VReg2D &s);
  VReg2DList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
  VRegDElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegDElem(getIdx(), i, getLane());
  }
};
class VReg1QList : public VRegList {
public:
  VReg1QList(const VReg1Q &s);
  VReg1QList(const VRegVec &s, const VRegVec &e) : VRegList(s, e) {}
  VRegQElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegQElem(getIdx(), i, getLane());
  }
};

class VReg4B : public VRegVec {
public:
  explicit VReg4B(uint32_t index) : VRegVec(index, 8, 4) {}
  VRegBElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegBElem(getIdx(), i, getLane());
  }
  VReg4BList operator-(const VReg4B &other) const { return VReg4BList(*this, other); }
};
class VReg8B : public VRegVec {
public:
  explicit VReg8B(uint32_t index) : VRegVec(index, 8, 8) {}
  VRegBElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegBElem(getIdx(), i, getLane());
  }
  VReg8BList operator-(const VReg8B &other) const { return VReg8BList(*this, other); }
};
class VReg16B : public VRegVec {
public:
  explicit VReg16B(uint32_t index) : VRegVec(index, 8, 16) {}
  VRegBElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegBElem(getIdx(), i, getLane());
  }
  VReg16BList operator-(const VReg16B &other) const { return VReg16BList(*this, other); }
};
class VReg2H : public VRegVec {
public:
  explicit VReg2H(uint32_t index) : VRegVec(index, 16, 2) {}
  VRegHElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegHElem(getIdx(), i, getLane());
  }
  VReg2HList operator-(const VReg2H &other) const { return VReg2HList(*this, other); }
};
class VReg4H : public VRegVec {
public:
  explicit VReg4H(uint32_t index) : VRegVec(index, 16, 4) {}
  VRegHElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegHElem(getIdx(), i, getLane());
  }
  VReg4HList operator-(const VReg4H &other) const { return VReg4HList(*this, other); }
};
class VReg8H : public VRegVec {
public:
  explicit VReg8H(uint32_t index) : VRegVec(index, 16, 8) {}
  VRegHElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegHElem(getIdx(), i, getLane());
  }
  VReg8HList operator-(const VReg8H &other) const { return VReg8HList(*this, other); }
};
class VReg2S : public VRegVec {
public:
  explicit VReg2S(uint32_t index) : VRegVec(index, 32, 2) {}
  VRegSElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegSElem(getIdx(), i, getLane());
  }
  VReg2SList operator-(const VReg2S &other) const { return VReg2SList(*this, other); }
};
class VReg4S : public VRegVec {
public:
  explicit VReg4S(uint32_t index) : VRegVec(index, 32, 4) {}
  VRegSElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegSElem(getIdx(), i, getLane());
  }
  VReg4SList operator-(const VReg4S &other) const { return VReg4SList(*this, other); }
};
class VReg1D : public VRegVec {
public:
  explicit VReg1D(uint32_t index) : VRegVec(index, 64, 1) {}
  VRegDElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegDElem(getIdx(), i, getLane());
  }
  VReg1DList operator-(const VReg1D &other) const { return VReg1DList(*this, other); }
};
class VReg2D : public VRegVec {
public:
  explicit VReg2D(uint32_t index) : VRegVec(index, 64, 2) {}
  VRegDElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegDElem(getIdx(), i, getLane());
  }
  VReg2DList operator-(const VReg2D &other) const { return VReg2DList(*this, other); }
};
class VReg1Q : public VRegVec {
public:
  explicit VReg1Q(uint32_t index) : VRegVec(index, 128, 1) {}
  VRegQElem operator[](uint32_t i) const {
    assert(getLane() > i);
    return VRegQElem(getIdx(), i, getLane());
  }
  VReg1QList operator-(const VReg1Q &other) const { return VReg1QList(*this, other); }
};

inline VReg4BList::VReg4BList(const VReg4B &s) : VRegList(s, s) {}
inline VReg8BList::VReg8BList(const VReg8B &s) : VRegList(s, s) {}
inline VReg16BList::VReg16BList(const VReg16B &s) : VRegList(s, s) {}
inline VReg2HList::VReg2HList(const VReg2H &s) : VRegList(s, s) {}
inline VReg4HList::VReg4HList(const VReg4H &s) : VRegList(s, s) {}
inline VReg8HList::VReg8HList(const VReg8H &s) : VRegList(s, s) {}
inline VReg2SList::VReg2SList(const VReg2S &s) : VRegList(s, s) {}
inline VReg4SList::VReg4SList(const VReg4S &s) : VRegList(s, s) {}
inline VReg1DList::VReg1DList(const VReg1D &s) : VRegList(s, s) {}
inline VReg2DList::VReg2DList(const VReg2D &s) : VRegList(s, s) {}
inline VReg1QList::VReg1QList(const VReg1Q &s) : VRegList(s, s) {}

// SIMD vector regisetr
class VReg : public VRegVec {
public:
  explicit VReg(uint32_t index) : VRegVec(index, 128, 1), b4(index), b8(index), b16(index), b(index), h2(index), h4(index), h8(index), h(index), s2(index), s4(index), s(index), d1(index), d2(index), d(index), q1(index), q(index) {}

  VReg4B b4;
  VReg8B b8;
  VReg16B b16;
  VReg16B b;
  VReg2H h2;
  VReg4H h4;
  VReg8H h8;
  VReg8H h;
  VReg2S s2;
  VReg4S s4;
  VReg4S s;
  VReg1D d1;
  VReg2D d2;
  VReg2D d;
  VReg1Q q1;
  VReg1Q q;
};

// SVE SIMD Vector Register Base
class _ZReg : public Reg {
public:
  explicit _ZReg(uint32_t index, uint32_t bits = 128 * VL) : Reg(index, ZREG, bits) {}
};

// SVE SIMD Vector Register Element
class ZRegElem : public _ZReg {
  uint32_t elem_idx_;

public:
  explicit ZRegElem(uint32_t index, uint32_t eidx, uint32_t bit) : _ZReg(index, bit), elem_idx_(eidx) {}
  uint32_t getElemIdx() const { return elem_idx_; }
};

// base for SVE SIMD Vector Register List
class ZRegList : public _ZReg {
  uint32_t len_;

public:
  explicit ZRegList(const _ZReg &s) : _ZReg(s.getIdx(), s.getBit()), len_(s.getIdx() - s.getIdx() + 1) {}
  explicit ZRegList(const _ZReg &s, const _ZReg &e) : _ZReg(s.getIdx(), s.getBit()), len_(e.getIdx() - s.getIdx() + 1) {}
  uint32_t getLen() const { return len_; }
};

class ZRegB;
class ZRegH;
class ZRegS;
class ZRegD;
class ZRegQ;

class ZRegBElem : public ZRegElem {
public:
  explicit ZRegBElem(uint32_t index, uint32_t eidx) : ZRegElem(index, eidx, 8) {}
};
class ZRegHElem : public ZRegElem {
public:
  explicit ZRegHElem(uint32_t index, uint32_t eidx) : ZRegElem(index, eidx, 16) {}
};
class ZRegSElem : public ZRegElem {
public:
  explicit ZRegSElem(uint32_t index, uint32_t eidx) : ZRegElem(index, eidx, 32) {}
};
class ZRegDElem : public ZRegElem {
public:
  explicit ZRegDElem(uint32_t index, uint32_t eidx) : ZRegElem(index, eidx, 64) {}
};
class ZRegQElem : public ZRegElem {
public:
  explicit ZRegQElem(uint32_t index, uint32_t eidx) : ZRegElem(index, eidx, 128) {}
};

class ZRegBList : public ZRegList {
public:
  ZRegBList(const ZRegB &s);
  explicit ZRegBList(const _ZReg &s, const _ZReg &e) : ZRegList(s, e) {}
  ZRegBElem operator[](uint32_t i) const { return ZRegBElem(getIdx(), i); }
};
class ZRegHList : public ZRegList {
public:
  ZRegHList(const ZRegH &s);
  explicit ZRegHList(const _ZReg &s, const _ZReg &e) : ZRegList(s, e) {}
  ZRegHElem operator[](uint32_t i) const { return ZRegHElem(getIdx(), i); }
};
class ZRegSList : public ZRegList {
public:
  ZRegSList(const ZRegS &s);
  explicit ZRegSList(const _ZReg &s, const _ZReg &e) : ZRegList(s, e) {}
  ZRegSElem operator[](uint32_t i) const { return ZRegSElem(getIdx(), i); }
};
class ZRegDList : public ZRegList {
public:
  ZRegDList(const ZRegD &s);
  explicit ZRegDList(const _ZReg &s, const _ZReg &e) : ZRegList(s, e) {}
  ZRegDElem operator[](uint32_t i) const { return ZRegDElem(getIdx(), i); }
};
class ZRegQList : public ZRegList {
public:
  ZRegQList(const ZRegQ &s);
  explicit ZRegQList(const _ZReg &s, const _ZReg &e) : ZRegList(s, e) {}
  ZRegQElem operator[](uint32_t i) const { return ZRegQElem(getIdx(), i); }
};

class ZRegB : public _ZReg {
public:
  explicit ZRegB(uint32_t index) : _ZReg(index, 8) {}
  ZRegBElem operator[](uint32_t i) const { return ZRegBElem(getIdx(), i); }
  ZRegBList operator-(const ZRegB &other) const { return ZRegBList(*this, other); }
};
class ZRegH : public _ZReg {
public:
  explicit ZRegH(uint32_t index) : _ZReg(index, 16) {}
  ZRegHElem operator[](uint32_t i) const { return ZRegHElem(getIdx(), i); }
  ZRegHList operator-(const ZRegH &other) const { return ZRegHList(*this, other); }
};
class ZRegS : public _ZReg {
public:
  explicit ZRegS(uint32_t index) : _ZReg(index, 32) {}
  ZRegSElem operator[](uint32_t i) const { return ZRegSElem(getIdx(), i); }
  ZRegSList operator-(const ZRegS &other) const { return ZRegSList(*this, other); }
};
class ZRegD : public _ZReg {
public:
  explicit ZRegD(uint32_t index) : _ZReg(index, 64) {}
  ZRegDElem operator[](uint32_t i) const { return ZRegDElem(getIdx(), i); }
  ZRegDList operator-(const ZRegD &other) const { return ZRegDList(*this, other); }
};
class ZRegQ : public _ZReg {
public:
  explicit ZRegQ(uint32_t index) : _ZReg(index, 128) {}
  ZRegQElem operator[](uint32_t i) const { return ZRegQElem(getIdx(), i); }
  ZRegQList operator-(const ZRegQ &other) const { return ZRegQList(*this, other); }
};

// SIMD Vector Regisetr for SVE
class ZReg : public _ZReg {
public:
  explicit ZReg(uint32_t index) : _ZReg(index), b(index), h(index), s(index), d(index), q(index) {}

  ZRegB b;
  ZRegH h;
  ZRegS s;
  ZRegD d;
  ZRegQ q;
};

class _PReg : public Reg {
public:
  explicit _PReg(uint32_t index, bool M = false, uint32_t bits = 16 * VL) : Reg(index, ((M == 0) ? PREG_Z : PREG_M), bits) {}
  bool isM() const { return isPRegM(); }
  bool isZ() const { return isPRegZ(); }
};

class PRegB : public _PReg {
public:
  explicit PRegB(uint32_t index) : _PReg(index, false, 8) {}
};
class PRegH : public _PReg {
public:
  explicit PRegH(uint32_t index) : _PReg(index, false, 16) {}
};
class PRegS : public _PReg {
public:
  explicit PRegS(uint32_t index) : _PReg(index, false, 32) {}
};
class PRegD : public _PReg {
public:
  explicit PRegD(uint32_t index) : _PReg(index, false, 64) {}
};

enum PredType {
  T_z, // Zeroing predication
  T_m  // Merging predication
};

class PReg : public _PReg {
public:
  explicit PReg(uint32_t index, bool M = false) : _PReg(index, M), b(index), h(index), s(index), d(index) {}
  _PReg operator/(PredType t) const { return (t == T_z) ? _PReg(getIdx(), false) : _PReg(getIdx(), true); }

  PRegB b;
  PRegH h;
  PRegS s;
  PRegD d;
};

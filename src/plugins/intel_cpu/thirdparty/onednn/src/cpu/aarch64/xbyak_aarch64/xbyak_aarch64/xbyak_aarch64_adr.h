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

#include "xbyak_aarch64_err.h"
#include "xbyak_aarch64_reg.h"

enum ShMod { LSL = 0, LSR = 1, ASR = 2, ROR = 3, MSL = 4, NONE = 5 };

enum ExtMod { UXTB = 0, UXTH = 1, UXTW = 2, UXTX = 3, SXTB = 4, SXTH = 5, SXTW = 6, SXTX = 7, UXT = 8, SXT = 9, MUL = 10, MUL_VL = 11, EXT_LSL = 12 };

enum AdrKind {
  // for v8
  BASE_ONLY = 1,     // base register only
  BASE_IMM = 1 << 1, // base plus offset (immediate)
  BASE_REG = 1 << 2, // base plus offset (register)
  BASE_EXT = 1 << 3, // base plus offset (extend register)
  PRE = 1 << 4,      // pre-indexed
  POST_IMM = 1 << 5, // post-indexed (immediate)
  POST_REG = 1 << 6, // post-indexed (register)

  // for SVE
  SC_SC = 1 << 7,        // scalar base plus scalar index
  SC_IMM = 1 << 8,       // scalar base plus immediate
  SC_64VEC = 1 << 9,     // scalar base plus 64-bit vector index
  SC_32VEC64E = 1 << 10, // scalar base plus 32-bit vecotr index (64-bit element)
  SC_32VEC32E = 1 << 11, // scalar base plus 32-bit vecotr index (32-bit element)
  VEC_IMM64E = 1 << 12,  // vector base plus immediate offset (64-bit element)
  VEC_IMM32E = 1 << 13,  // vector base plus immediate offset (32-bit element)

  // for SVE address generator
  VEC_PACK = 1 << 14,   // vector (packed)
  VEC_UNPACK = 1 << 15, // vector (unpacked)
};

class Adr {
  AdrKind kind_;

protected:
  ExtMod trans(ExtMod org) { return ((org == UXT) ? UXTW : (org == SXT) ? SXTW : org); }

  ExtMod trans(const RReg &rm, ExtMod org) {
    if (org == SXT) {
      return (rm.getBit() == 64) ? SXTX : SXTW;
    } else if (org == UXT) {
      return (rm.getBit() == 64) ? UXTX : UXTW;
    }
    return org;
  }

public:
  explicit Adr(AdrKind kind) : kind_(kind) {}
  AdrKind getKind() { return kind_; }
};

// Pre-indexed
class AdrPreImm : public Adr {
  XReg xn_;
  int32_t imm_;

public:
  explicit AdrPreImm(const XReg &xn, int32_t imm) : Adr(PRE), xn_(xn), imm_(imm) {}
  const XReg &getXn() const { return xn_; }
  int32_t getImm() const { return imm_; }
};

// Pos-indexed (immediate offset)
class AdrPostImm : public Adr {
  XReg xn_;
  int32_t imm_;

public:
  explicit AdrPostImm(const XReg &xn, int32_t imm) : Adr(POST_IMM), xn_(xn), imm_(imm) {}
  const XReg &getXn() const { return xn_; }
  int32_t getImm() const { return imm_; }
};

// Pos-indexed (register offset)
class AdrPostReg : public Adr {
  XReg xn_;
  XReg xm_;

public:
  explicit AdrPostReg(const XReg &xn, const XReg &xm) : Adr(POST_REG), xn_(xn), xm_(xm) {}
  const XReg &getXn() const { return xn_; }
  const XReg &getXm() const { return xm_; }
};

// base only
class AdrNoOfs : public Adr {
  XReg xn_;

public:
  explicit AdrNoOfs(const XReg &xn) : Adr(BASE_ONLY), xn_(xn) {}
  const XReg &getXn() const { return xn_; }
};

// base plus offset (signed immediate)
class AdrImm : public Adr {
  XReg xn_;
  int32_t imm_;

public:
  explicit AdrImm(const XReg &xn, int32_t imm) : Adr(BASE_IMM), xn_(xn), imm_(imm) {}
  AdrImm(const AdrNoOfs &a) : Adr(BASE_IMM), xn_(a.getXn()), imm_(0) {}
  const XReg &getXn() const { return xn_; }
  int32_t getImm() const { return imm_; }
};

// base plus offset (unsigned immediate)
class AdrUimm : public Adr {
  XReg xn_;
  uint32_t uimm_;

public:
  explicit AdrUimm(const XReg &xn, uint32_t uimm) : Adr(BASE_IMM), xn_(xn), uimm_(uimm) {}
  AdrUimm(const AdrNoOfs &a) : Adr(BASE_IMM), xn_(a.getXn()), uimm_(0) {}
  AdrUimm(const AdrImm &a) : Adr(BASE_IMM), xn_(a.getXn()), uimm_(a.getImm()) {}
  const XReg &getXn() const { return xn_; }
  uint32_t getImm() const { return uimm_; }
};

// base plus offset (register)
class AdrReg : public Adr {
  XReg xn_;
  XReg xm_;
  ShMod mod_;
  uint32_t sh_;
  bool init_sh_;

public:
  explicit AdrReg(const XReg &xn, const XReg &xm, ShMod mod, uint32_t sh) : Adr(BASE_REG), xn_(xn), xm_(xm), mod_(mod), sh_(sh), init_sh_(true) {}
  explicit AdrReg(const XReg &xn, const XReg &xm, ShMod mod = LSL) : Adr(BASE_REG), xn_(xn), xm_(xm), mod_(mod), sh_(0), init_sh_(false) {}
  const XReg &getXn() const { return xn_; }
  const XReg &getXm() const { return xm_; }
  ShMod getMod() const { return mod_; }
  uint32_t getSh() const { return sh_; }
  bool getInitSh() const { return init_sh_; }
};

// base plus offset (extended register)
class AdrExt : public Adr {
  XReg xn_;
  RReg rm_;
  ExtMod mod_;
  uint32_t sh_;
  bool init_sh_;

public:
  explicit AdrExt(const XReg &xn, const RReg &rm, ExtMod mod, uint32_t sh) : Adr(BASE_EXT), xn_(xn), rm_(rm), mod_(trans(rm, mod)), sh_(sh), init_sh_(true) {}
  explicit AdrExt(const XReg &xn, const RReg &rm, ExtMod mod) : Adr(BASE_EXT), xn_(xn), rm_(rm), mod_(trans(rm, mod)), sh_(0), init_sh_(false) {}
  const XReg &getXn() const { return xn_; }
  const RReg &getRm() const { return rm_; }
  ExtMod getMod() const { return mod_; }
  uint32_t getSh() const { return sh_; }
  bool getInitSh() const { return init_sh_; }
};

///////////////////// for SVE //////////////////////////////////////////

// Scalar plus scalar
class AdrScSc : public Adr {
  XReg xn_;
  XReg xm_;
  ShMod mod_;
  uint32_t sh_;
  bool init_mod_;

public:
  explicit AdrScSc(const XReg &xn, const XReg &xm, ShMod mod, uint32_t sh) : Adr(SC_SC), xn_(xn), xm_(xm), mod_(mod), sh_(sh), init_mod_(true) {}
  explicit AdrScSc(const XReg &xn, const XReg &xm) : Adr(SC_SC), xn_(xn), xm_(xm), mod_(LSL), sh_(0), init_mod_(false) {}
  // AdrScSc(const AdrNoOfs &a) :Adr(SC_SC), xn_(a.getXn()), xm_(XReg(31)),
  // mod_(LSL), sh_(0) {}
  AdrScSc(const AdrReg &a) : Adr(SC_SC), xn_(a.getXn()), xm_(a.getXm()), mod_(a.getMod()), sh_(a.getSh()) {}
  const XReg &getXn() const { return xn_; }
  const XReg &getXm() const { return xm_; }
  uint32_t getSh() const { return sh_; }
  ShMod getMod() const { return mod_; }
  bool getInitMod() const { return init_mod_; }
};

// Scalar plus immediate
class AdrScImm : public Adr {
  XReg xn_;
  int32_t simm_;
  ExtMod mod_;

public:
  explicit AdrScImm(const XReg &xn, int32_t simm = 0, ExtMod mod = MUL_VL) : Adr(SC_IMM), xn_(xn), simm_(simm), mod_(trans(mod)) {}
  // AdrScImm(const AdrNoOfs &a) :Adr(SC_IMM), xn_(a.getXn()), simm_(0),
  // mod_(MUL_VL) {}
  AdrScImm(const AdrImm &a) : Adr(SC_IMM), xn_(a.getXn()), simm_(a.getImm()), mod_(MUL_VL) {}
  const XReg &getXn() const { return xn_; }
  int32_t getSimm() const { return simm_; }
  ExtMod getMod() const { return mod_; }
};

// Scalar plus vector (unscaled 64-bit offset)
class AdrSc64U : public Adr {
  XReg xn_;
  ZRegD zm_;

public:
  explicit AdrSc64U(const XReg &xn, const ZRegD &zm) : Adr(SC_64VEC), xn_(xn), zm_(zm) {}
  const XReg &getXn() const { return xn_; }
  const ZRegD &getZm() const { return zm_; }
};

// Scalar plus vector (scaled 64-bit offset)
class AdrSc64S : public Adr {
  XReg xn_;
  ZRegD zm_;
  ShMod mod_;
  uint32_t sh_;

public:
  explicit AdrSc64S(const XReg &xn, const ZRegD &zm, ShMod mod, uint32_t sh) : Adr(SC_64VEC), xn_(xn), zm_(zm), mod_(mod), sh_(sh) {}
  AdrSc64S(const AdrSc64U &a) : Adr(SC_64VEC), xn_(a.getXn()), zm_(a.getZm()), mod_(LSL), sh_(0) {}
  const XReg &getXn() const { return xn_; }
  const ZRegD &getZm() const { return zm_; }
  ShMod getMod() const { return mod_; }
  uint32_t getSh() const { return sh_; }
};

// Scalar plus vector (unscaled 32-bit offset)
class AdrSc32U : public Adr {
  XReg xn_;
  ZRegS zm_;
  ExtMod mod_;

public:
  explicit AdrSc32U(const XReg &xn, const ZRegS &zm, ExtMod mod) : Adr(SC_32VEC32E), xn_(xn), zm_(zm), mod_(trans(mod)) {}
  const XReg &getXn() const { return xn_; }
  const ZRegS &getZm() const { return zm_; }
  ExtMod getMod() const { return mod_; }
};

// Scalar plus vector (scaled 32-bit offset)
class AdrSc32S : public Adr {
  XReg xn_;
  ZRegS zm_;
  ExtMod mod_;
  uint32_t sh_;

public:
  explicit AdrSc32S(const XReg &xn, const ZRegS &zm, ExtMod mod, uint32_t sh) : Adr(SC_32VEC32E), xn_(xn), zm_(zm), mod_(trans(mod)), sh_(sh) {}
  AdrSc32S(const AdrSc32U &a) : Adr(SC_32VEC32E), xn_(a.getXn()), zm_(a.getZm()), mod_(a.getMod()), sh_(0) {}
  const XReg &getXn() const { return xn_; }
  const ZRegS &getZm() const { return zm_; }
  ExtMod getMod() const { return mod_; }
  uint32_t getSh() const { return sh_; }
};

// Scalar plus vector (unpacked unscaled 32-bit offset)
class AdrSc32UU : public Adr {
  XReg xn_;
  ZRegD zm_;
  ExtMod mod_;

public:
  explicit AdrSc32UU(const XReg &xn, const ZRegD &zm, ExtMod mod) : Adr(SC_32VEC64E), xn_(xn), zm_(zm), mod_(trans(mod)) {}
  const XReg &getXn() const { return xn_; }
  const ZRegD &getZm() const { return zm_; }
  ExtMod getMod() const { return mod_; }
};

// Scalar plus vector (unpacked scaled 32-bit offset)
class AdrSc32US : public Adr {
  XReg xn_;
  ZRegD zm_;
  ExtMod mod_;
  uint32_t sh_;

public:
  explicit AdrSc32US(const XReg &xn, const ZRegD &zm, ExtMod mod, uint32_t sh) : Adr(SC_32VEC64E), xn_(xn), zm_(zm), mod_(trans(mod)), sh_(sh) {}
  AdrSc32US(const AdrSc32UU &a) : Adr(SC_32VEC64E), xn_(a.getXn()), zm_(a.getZm()), mod_(a.getMod()), sh_(0) {}
  const XReg &getXn() const { return xn_; }
  const ZRegD &getZm() const { return zm_; }
  ExtMod getMod() const { return mod_; }
  uint32_t getSh() const { return sh_; }
};

// Vector plus immediate 64-bit element
class AdrVecImm64 : public Adr {
  ZRegD zn_;
  uint32_t imm_;

public:
  explicit AdrVecImm64(const ZRegD &zn, uint32_t imm = 0) : Adr(VEC_IMM64E), zn_(zn), imm_(imm) {}
  const ZRegD &getZn() const { return zn_; }
  uint32_t getImm() const { return imm_; }
};

// Vector plus immediate 32-bit element
class AdrVecImm32 : public Adr {
  ZRegS zn_;
  uint32_t imm_;

public:
  explicit AdrVecImm32(const ZRegS &zn, uint32_t imm = 0) : Adr(VEC_IMM32E), zn_(zn), imm_(imm) {}
  const ZRegS &getZn() const { return zn_; }
  uint32_t getImm() const { return imm_; }
};

// Vector Address (packed offset)
class AdrVec : public Adr {
  _ZReg zn_;
  _ZReg zm_;
  ShMod mod_;
  uint32_t sh_;

public:
  explicit AdrVec(const _ZReg &zn, const _ZReg &zm, ShMod mod = NONE, uint32_t sh = 0) : Adr(VEC_PACK), zn_(zn), zm_(zm), mod_(mod), sh_(sh) {}
  const _ZReg &getZn() const { return zn_; }
  const _ZReg &getZm() const { return zm_; }
  ShMod getMod() const { return mod_; }
  uint32_t getSh() const { return sh_; }
};

// Vector Address (unpacked offset)
class AdrVecU : public Adr {
  _ZReg zn_;
  _ZReg zm_;
  ExtMod mod_;
  uint32_t sh_;

public:
  explicit AdrVecU(const _ZReg &zn, const _ZReg &zm, ExtMod mod, uint32_t sh = 0) : Adr(VEC_UNPACK), zn_(zn), zm_(zm), mod_(trans(mod)), sh_(sh) {}
  const _ZReg &getZn() const { return zn_; }
  const _ZReg &getZm() const { return zm_; }
  ExtMod getMod() const { return mod_; }
  uint32_t getSh() const { return sh_; }
};

AdrNoOfs ptr(const XReg &xn);
AdrImm ptr(const XReg &xn, int32_t imm);
AdrUimm ptr(const XReg &xn, uint32_t uimm);
AdrReg ptr(const XReg &xn, const XReg &xm);
AdrReg ptr(const XReg &xn, const XReg &xm, ShMod mod, uint32_t sh);
AdrReg ptr(const XReg &xn, const XReg &xm, ShMod mod);
AdrExt ptr(const XReg &xn, const RReg &rm, ExtMod mod, uint32_t sh);
AdrExt ptr(const XReg &xn, const RReg &rm, ExtMod mod);
AdrPreImm pre_ptr(const XReg &xn, int32_t imm);
AdrPostImm post_ptr(const XReg &xn, int32_t imm);
AdrPostReg post_ptr(const XReg &xn, XReg xm);
AdrScImm ptr(const XReg &xn, int32_t simm, ExtMod mod);
AdrSc64U ptr(const XReg &xn, const ZRegD &zm);
AdrSc64S ptr(const XReg &xn, const ZRegD &zm, ShMod mod, uint32_t sh);
AdrSc32U ptr(const XReg &xn, const ZRegS &zm, ExtMod mod);
AdrSc32S ptr(const XReg &xn, const ZRegS &zm, ExtMod mod, uint32_t sh);
AdrSc32UU ptr(const XReg &xn, const ZRegD &zm, ExtMod mod);
AdrSc32US ptr(const XReg &xn, const ZRegD &zm, ExtMod mod, uint32_t sh);
AdrVecImm64 ptr(const ZRegD &zn, uint32_t imm = 0);
AdrVecImm32 ptr(const ZRegS &zn, uint32_t imm = 0);
AdrVec ptr(const ZRegS &zn, const ZRegS &zm, ShMod mod = NONE, uint32_t sh = 0);
AdrVec ptr(const ZRegD &zn, const ZRegD &zm, ShMod mod = NONE, uint32_t sh = 0);
AdrVecU ptr(const ZRegD &zn, const ZRegD &zm, ExtMod mod, uint32_t sh = 0);

inline AdrNoOfs ptr(const XReg &xn) { return AdrNoOfs(xn); }

inline AdrImm ptr(const XReg &xn, int32_t imm) { return AdrImm(xn, imm); }

inline AdrUimm ptr(const XReg &xn, uint32_t uimm) { return AdrUimm(xn, uimm); }

inline AdrReg ptr(const XReg &xn, const XReg &xm) { return AdrReg(xn, xm); }

inline AdrReg ptr(const XReg &xn, const XReg &xm, ShMod mod, uint32_t sh) { return AdrReg(xn, xm, mod, sh); }

inline AdrReg ptr(const XReg &xn, const XReg &xm, ShMod mod) { return AdrReg(xn, xm, mod); }

inline AdrExt ptr(const XReg &xn, const RReg &rm, ExtMod mod, uint32_t sh) { return AdrExt(xn, rm, mod, sh); }

inline AdrExt ptr(const XReg &xn, const RReg &rm, ExtMod mod) { return AdrExt(xn, rm, mod); }

inline AdrPreImm pre_ptr(const XReg &xn, int32_t imm) { return AdrPreImm(xn, imm); }

inline AdrPostImm post_ptr(const XReg &xn, int32_t imm) { return AdrPostImm(xn, imm); }

inline AdrPostReg post_ptr(const XReg &xn, XReg xm) { return AdrPostReg(xn, xm); }

inline AdrScImm ptr(const XReg &xn, int32_t simm, ExtMod mod) { return AdrScImm(xn, simm, mod); }

inline AdrSc64U ptr(const XReg &xn, const ZRegD &zm) { return AdrSc64U(xn, zm); }

inline AdrSc64S ptr(const XReg &xn, const ZRegD &zm, ShMod mod, uint32_t sh) { return AdrSc64S(xn, zm, mod, sh); }

inline AdrSc32U ptr(const XReg &xn, const ZRegS &zm, ExtMod mod) { return AdrSc32U(xn, zm, mod); }

inline AdrSc32S ptr(const XReg &xn, const ZRegS &zm, ExtMod mod, uint32_t sh) { return AdrSc32S(xn, zm, mod, sh); }

inline AdrSc32UU ptr(const XReg &xn, const ZRegD &zm, ExtMod mod) { return AdrSc32UU(xn, zm, mod); }

inline AdrSc32US ptr(const XReg &xn, const ZRegD &zm, ExtMod mod, uint32_t sh) { return AdrSc32US(xn, zm, mod, sh); }

inline AdrVecImm64 ptr(const ZRegD &zn, uint32_t imm) { return AdrVecImm64(zn, imm); }

inline AdrVecImm32 ptr(const ZRegS &zn, uint32_t imm) { return AdrVecImm32(zn, imm); }

inline AdrVec ptr(const ZRegS &zn, const ZRegS &zm, ShMod mod, uint32_t sh) { return AdrVec(zn, zm, mod, sh); }

inline AdrVec ptr(const ZRegD &zn, const ZRegD &zm, ShMod mod, uint32_t sh) { return AdrVec(zn, zm, mod, sh); }

inline AdrVecU ptr(const ZRegD &zn, const ZRegD &zm, ExtMod mod, uint32_t sh) { return AdrVecU(zn, zm, mod, sh); }

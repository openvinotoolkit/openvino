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
#ifdef __APPLE__
#include <libkern/OSCacheControl.h>
#endif

#include "xbyak_aarch64_adr.h"
#include "xbyak_aarch64_code_array.h"
#include "xbyak_aarch64_err.h"
#include "xbyak_aarch64_label.h"
#include "xbyak_aarch64_reg.h"

enum BarOpt { SY = 0xf, ST = 0xe, LD = 0xd, ISH = 0xb, ISHST = 0xa, ISHLD = 0x9, NSH = 0x7, NSHST = 0x6, NSHLD = 0x5, OSH = 0x3, OSHST = 0x2, OSHLD = 0x1 };

enum PStateField { SPSel, DAIFSet, DAIFClr, UAO, PAN, DIT };

enum Cond { EQ = 0x0, NE = 0x1, CS = 0x2, HS = 0x2, CC = 0x3, LO = 0x3, MI = 0x4, PL = 0x5, VS = 0x6, VC = 0x7, HI = 0x8, LS = 0x9, GE = 0xa, LT = 0xb, GT = 0xc, LE = 0xd, AL = 0xe, NV = 0xf };

enum Prfop {
  PLDL1KEEP = (0x0 << 3) + (0x0 << 1) + 0x0,
  PLDL1STRM = (0x0 << 3) + (0x0 << 1) + 0x1,
  PLDL2KEEP = (0x0 << 3) + (0x1 << 1) + 0x0,
  PLDL2STRM = (0x0 << 3) + (0x1 << 1) + 0x1,
  PLDL3KEEP = (0x0 << 3) + (0x2 << 1) + 0x0,
  PLDL3STRM = (0x0 << 3) + (0x2 << 1) + 0x1,
  PLIL1KEEP = (0x1 << 3) + (0x0 << 1) + 0x0,
  PLIL1STRM = (0x1 << 3) + (0x0 << 1) + 0x1,
  PLIL2KEEP = (0x1 << 3) + (0x1 << 1) + 0x0,
  PLIL2STRM = (0x1 << 3) + (0x1 << 1) + 0x1,
  PLIL3KEEP = (0x1 << 3) + (0x2 << 1) + 0x0,
  PLIL3STRM = (0x1 << 3) + (0x2 << 1) + 0x1,
  PSTL1KEEP = (0x2 << 3) + (0x0 << 1) + 0x0,
  PSTL1STRM = (0x2 << 3) + (0x0 << 1) + 0x1,
  PSTL2KEEP = (0x2 << 3) + (0x1 << 1) + 0x0,
  PSTL2STRM = (0x2 << 3) + (0x1 << 1) + 0x1,
  PSTL3KEEP = (0x2 << 3) + (0x2 << 1) + 0x0,
  PSTL3STRM = (0x2 << 3) + (0x2 << 1) + 0x1
};

enum PrfopSve { PLDL1KEEP_SVE = 0x0, PLDL1STRM_SVE = 0x1, PLDL2KEEP_SVE = 0x2, PLDL2STRM_SVE = 0x3, PLDL3KEEP_SVE = 0x4, PLDL3STRM_SVE = 0x5, PSTL1KEEP_SVE = 0x8, PSTL1STRM_SVE = 0x9, PSTL2KEEP_SVE = 0xa, PSTL2STRM_SVE = 0xb, PSTL3KEEP_SVE = 0xc, PSTL3STRM_SVE = 0xd };

enum Pattern { POW2 = 0x0, VL1 = 0x1, VL2 = 0x2, VL3 = 0x3, VL4 = 0x4, VL5 = 0x5, VL6 = 0x6, VL7 = 0x7, VL8 = 0x8, VL16 = 0x9, VL32 = 0xa, VL64 = 0xb, VL128 = 0xc, VL256 = 0xd, MUL4 = 0x1d, MUL3 = 0x1e, ALL = 0x1f };

enum IcOp {
  ALLUIS = inner::genSysInstOp(0, 7, 1, 0), // op1=0, CRn=7, CRm=1, op2=0
  ALLU = inner::genSysInstOp(0, 7, 5, 0),   // op1=0, CRn=7, CRm=5, op2=0
  VAU = inner::genSysInstOp(3, 7, 5, 0)     // op1=3, CRn=7, CRm=5, op2=1
};

enum DcOp {
  IVAC = inner::genSysInstOp(0, 7, 0x6, 1),  // op1=0, CRn=7, CRm=0x6, op2=1
  ISW = inner::genSysInstOp(0, 7, 0x6, 2),   // op1=0, CRn=7, CRm=0x6, op2=2
  CSW = inner::genSysInstOp(0, 7, 0xA, 2),   // op1=0, CRn=7, CRm=0xA, op2=2
  CISW = inner::genSysInstOp(0, 7, 0xE, 2),  // op1=0, CRn=7, CRm=0xE, op2=2
  ZVA = inner::genSysInstOp(3, 7, 0x4, 1),   // op1=3, CRn=7, CRm=0x4, op2=1
  CVAC = inner::genSysInstOp(3, 7, 0xA, 1),  // op1=3, CRn=7, CRm=0xA, op2=1
  CVAU = inner::genSysInstOp(3, 7, 0xB, 1),  // op1=3, CRn=7, CRm=0xB, op2=1
  CIVAC = inner::genSysInstOp(3, 7, 0xE, 1), // op1=3, CRn=7, CRm=0xE, op2=1
  CVAP = inner::genSysInstOp(3, 7, 0xC, 1)   // op1=3, CRn=7, CRm=0xC, op2=1
};

enum AtOp {
  S1E1R = inner::genSysInstOp(0, 7, 0x8, 0),  // op1=0, CRn=7, CRm=0x8, op2=0
  S1E1W = inner::genSysInstOp(0, 7, 0x8, 1),  // op1=0, CRn=7, CRm=0x8, op2=1
  S1E0R = inner::genSysInstOp(0, 7, 0x8, 2),  // op1=0, CRn=7, CRm=0x8, op2=2
  S1E0W = inner::genSysInstOp(0, 7, 0x8, 3),  // op1=0, CRn=7, CRm=0x8, op2=3
  S1E2R = inner::genSysInstOp(4, 7, 0x8, 0),  // op1=4, CRn=7, CRm=0x8, op2=0
  S1E2W = inner::genSysInstOp(4, 7, 0x8, 1),  // op1=4, CRn=7, CRm=0x8, op2=1
  S12E1R = inner::genSysInstOp(4, 7, 0x8, 4), // op1=4, CRn=7, CRm=0x8, op2=4
  S12E1W = inner::genSysInstOp(4, 7, 0x8, 5), // op1=4, CRn=7, CRm=0x8, op2=5
  S12E0R = inner::genSysInstOp(4, 7, 0x8, 6), // op1=4, CRn=7, CRm=0x8, op2=6
  S12E0W = inner::genSysInstOp(4, 7, 0x8, 7), // op1=4, CRn=7, CRm=0x8, op2=7
  S1E3R = inner::genSysInstOp(6, 7, 0x8, 0),  // op1=6, CRn=7, CRm=0x8, op2=0
  S1E3W = inner::genSysInstOp(6, 7, 0x8, 1),  // op1=6, CRn=7, CRm=0x8, op2=1
  S1E1RP = inner::genSysInstOp(0, 7, 0x9, 0), // op1=0, CRn=7, CRm=0x9, op2=0
  S1E1WP = inner::genSysInstOp(0, 7, 0x9, 1), // op1=0, CRn=7, CRm=0x9, op2=1
};

enum TlbiOp {
  VMALLE1IS = inner::genSysInstOp(0, 7, 3, 0),    // op1=0, CRn=7, CRm=0x3, op2=0
  VAE1IS = inner::genSysInstOp(0, 7, 3, 1),       // op1=0, CRn=7, CRm=0x3, op2=1
  ASIDE1IS = inner::genSysInstOp(0, 7, 3, 2),     // op1=0, CRn=7, CRm=0x3, op2=2
  VAAE1IS = inner::genSysInstOp(0, 7, 3, 3),      // op1=0, CRn=7, CRm=0x3, op2=3
  VALE1IS = inner::genSysInstOp(0, 7, 3, 5),      // op1=0, CRn=7, CRm=0x3, op2=5
  VAALE1IS = inner::genSysInstOp(0, 7, 3, 7),     // op1=0, CRn=7, CRm=0x3, op2=7
  VMALLE1 = inner::genSysInstOp(0, 7, 7, 0),      // op1=0, CRn=7, CRm=0x7, op2=0
  VAE1 = inner::genSysInstOp(0, 7, 7, 1),         // op1=0, CRn=7, CRm=0x7, op2=1
  ASIDE1 = inner::genSysInstOp(0, 7, 7, 2),       // op1=0, CRn=7, CRm=0x7, op2=2
  VAAE1 = inner::genSysInstOp(0, 7, 7, 3),        // op1=0, CRn=7, CRm=0x7, op2=3
  VALE1 = inner::genSysInstOp(0, 7, 7, 5),        // op1=0, CRn=7, CRm=0x7, op2=5
  VAALE1 = inner::genSysInstOp(0, 7, 7, 7),       // op1=0, CRn=7, CRm=0x7, op2=7
  IPAS2E1IS = inner::genSysInstOp(4, 7, 0, 1),    // op1=4, CRn=7, CRm=0x0, op2=1
  IPAS2LE1IS = inner::genSysInstOp(4, 7, 0, 5),   // op1=4, CRn=7, CRm=0x0, op2=5
  ALLE2IS = inner::genSysInstOp(4, 7, 3, 0),      // op1=4, CRn=7, CRm=0x3, op2=0
  VAE2IS = inner::genSysInstOp(4, 7, 3, 1),       // op1=4, CRn=7, CRm=0x3, op2=1
  ALLE1IS = inner::genSysInstOp(4, 7, 3, 4),      // op1=4, CRn=7, CRm=0x3, op2=4
  VALE2IS = inner::genSysInstOp(4, 7, 3, 5),      // op1=4, CRn=7, CRm=0x3, op2=5
  VMALLS12E1IS = inner::genSysInstOp(4, 7, 3, 6), // op1=4, CRn=7, CRm=0x3, op2=6
  IPAS2E1 = inner::genSysInstOp(4, 7, 4, 1),      // op1=4, CRn=7, CRm=0x4, op2=1
  IPAS2LE1 = inner::genSysInstOp(4, 7, 4, 5),     // op1=4, CRn=7, CRm=0x4, op2=5
  ALLE2 = inner::genSysInstOp(4, 7, 7, 0),        // op1=4, CRn=7, CRm=0x7, op2=0
  VAE2 = inner::genSysInstOp(4, 7, 7, 1),         // op1=4, CRn=7, CRm=0x7, op2=1
  ALLE1 = inner::genSysInstOp(4, 7, 7, 4),        // op1=4, CRn=7, CRm=0x7, op2=4
  VALE2 = inner::genSysInstOp(4, 7, 7, 5),        // op1=4, CRn=7, CRm=0x7, op2=5
  VMALLS12E1 = inner::genSysInstOp(4, 7, 7, 6),   // op1=4, CRn=7, CRm=0x7, op2=6
  ALLE3IS = inner::genSysInstOp(6, 7, 3, 0),      // op1=6, CRn=7, CRm=0x3, op2=0
  VAE3IS = inner::genSysInstOp(6, 7, 3, 1),       // op1=6, CRn=7, CRm=0x3, op2=1
  VALE3IS = inner::genSysInstOp(6, 7, 3, 5),      // op1=6, CRn=7, CRm=0x3, op2=5
  ALLE3 = inner::genSysInstOp(6, 7, 7, 0),        // op1=6, CRn=7, CRm=0x7, op2=0
  VAE3 = inner::genSysInstOp(6, 7, 7, 1),         // op1=6, CRn=7, CRm=0x7, op2=1
  VALE3 = inner::genSysInstOp(6, 7, 7, 5),        // op1=6, CRn=7, CRm=0x7, op2=5
  VMALLE1OS = inner::genSysInstOp(0, 7, 1, 0),    // op1=0, CRn=7, CRm=0x1, op2=0
  VAE1OS = inner::genSysInstOp(0, 7, 1, 1),       // op1=0, CRn=7, CRm=0x1, op2=1
  ASIDE1OS = inner::genSysInstOp(0, 7, 1, 2),     // op1=0, CRn=7, CRm=0x1, op2=2
  VAAE1OS = inner::genSysInstOp(0, 7, 1, 3),      // op1=0, CRn=7, CRm=0x1, op2=3
  VALE1OS = inner::genSysInstOp(0, 7, 1, 5),      // op1=0, CRn=7, CRm=0x1, op2=5
  VAALE1OS = inner::genSysInstOp(0, 7, 1, 7),     // op1=0, CRn=7, CRm=0x1, op2=7
  RVAE1IS = inner::genSysInstOp(0, 7, 2, 1),      // op1=0, CRn=7, CRm=0x2, op2=1
  RVAAE1IS = inner::genSysInstOp(0, 7, 2, 3),     // op1=0, CRn=7, CRm=0x2, op2=3
  RVALE1IS = inner::genSysInstOp(0, 7, 2, 5),     // op1=0, CRn=7, CRm=0x2, op2=5
  RVAALE1IS = inner::genSysInstOp(0, 7, 2, 7),    // op1=0, CRn=7, CRm=0x2, op2=7
  RVAE1OS = inner::genSysInstOp(0, 7, 5, 1),      // op1=0, CRn=7, CRm=0x5, op2=1
  RVAAE1OS = inner::genSysInstOp(0, 7, 5, 3),     // op1=0, CRn=7, CRm=0x5, op2=3
  RVALE1OS = inner::genSysInstOp(0, 7, 5, 5),     // op1=0, CRn=7, CRm=0x5, op2=5
  RVAALE1OS = inner::genSysInstOp(0, 7, 5, 7),    // op1=0, CRn=7, CRm=0x5, op2=7
  RVAE1 = inner::genSysInstOp(0, 7, 6, 1),        // op1=0, CRn=7, CRm=0x6, op2=1
  RVAAE1 = inner::genSysInstOp(0, 7, 6, 3),       // op1=0, CRn=7, CRm=0x6, op2=3
  RVALE1 = inner::genSysInstOp(0, 7, 6, 5),       // op1=0, CRn=7, CRm=0x6, op2=5
  RVAALE1 = inner::genSysInstOp(0, 7, 6, 7),      // op1=0, CRn=7, CRm=0x6, op2=7
  RIPAS2E1IS = inner::genSysInstOp(4, 7, 0, 2),   // op1=4, CRn=7, CRm=0x0, op2=2
  RIPAS2LE1IS = inner::genSysInstOp(4, 7, 0, 6),  // op1=4, CRn=7, CRm=0x0, op2=6
  ALLE2OS = inner::genSysInstOp(4, 7, 1, 0),      // op1=4, CRn=7, CRm=0x1, op2=0
  VAE2OS = inner::genSysInstOp(4, 7, 1, 1),       // op1=4, CRn=7, CRm=0x1, op2=1
  ALLE1OS = inner::genSysInstOp(4, 7, 1, 4),      // op1=4, CRn=7, CRm=0x1, op2=4
  VALE2OS = inner::genSysInstOp(4, 7, 1, 5),      // op1=4, CRn=7, CRm=0x1, op2=5
  VMALLS12E1OS = inner::genSysInstOp(4, 7, 1, 6), // op1=4, CRn=7, CRm=0x1, op2=6
  RVAE2IS = inner::genSysInstOp(4, 7, 2, 1),      // op1=4, CRn=7, CRm=0x2, op2=1
  RVALE2IS = inner::genSysInstOp(4, 7, 2, 5),     // op1=4, CRn=7, CRm=0x2, op2=5
  IPAS2E1OS = inner::genSysInstOp(4, 7, 4, 0),    // op1=4, CRn=7, CRm=0x4, op2=0
  RIPAS2E1 = inner::genSysInstOp(4, 7, 4, 2),     // op1=4, CRn=7, CRm=0x4, op2=2
  RIPAS2E1OS = inner::genSysInstOp(4, 7, 4, 3),   // op1=4, CRn=7, CRm=0x4, op2=3
  IPAS2LE1OS = inner::genSysInstOp(4, 7, 4, 4),   // op1=4, CRn=7, CRm=0x4, op2=4
  RIPAS2LE1 = inner::genSysInstOp(4, 7, 4, 6),    // op1=4, CRn=7, CRm=0x4, op2=6
  RIPAS2LE1OS = inner::genSysInstOp(4, 7, 4, 7),  // op1=4, CRn=7, CRm=0x4, op2=7
  RVAE2OS = inner::genSysInstOp(4, 7, 5, 1),      // op1=4, CRn=7, CRm=0x5, op2=1
  RVALE2OS = inner::genSysInstOp(4, 7, 5, 5),     // op1=4, CRn=7, CRm=0x5, op2=5
  RVAE2 = inner::genSysInstOp(4, 7, 6, 1),        // op1=4, CRn=7, CRm=0x6, op2=1
  RVALE2 = inner::genSysInstOp(4, 7, 6, 5),       // op1=4, CRn=7, CRm=0x6, op2=5
  ALLE3OS = inner::genSysInstOp(6, 7, 1, 0),      // op1=6, CRn=7, CRm=0x1, op2=0
  VAE3OS = inner::genSysInstOp(6, 7, 1, 1),       // op1=6, CRn=7, CRm=0x1, op2=1
  VALE3OS = inner::genSysInstOp(6, 7, 1, 5),      // op1=6, CRn=7, CRm=0x1, op2=5
  RVAE3IS = inner::genSysInstOp(6, 7, 2, 1),      // op1=6, CRn=7, CRm=0x2, op2=1
  RVALE3IS = inner::genSysInstOp(6, 7, 2, 5),     // op1=6, CRn=7, CRm=0x1, op2=5
  RVAE3OS = inner::genSysInstOp(6, 7, 5, 1),      // op1=6, CRn=7, CRm=0x5, op2=1
  RVALE3OS = inner::genSysInstOp(6, 7, 5, 5),     // op1=6, CRn=7, CRm=0x5, op2=5
  RVAE3 = inner::genSysInstOp(6, 7, 6, 1),        // op1=6, CRn=7, CRm=0x6, op2=1
  RVALE3 = inner::genSysInstOp(6, 7, 6, 5)        // op1=6, CRn=7, CRm=0x6, op2=5
};

class CodeGenerator : public CodeArray {

  LabelManager labelMgr_;

  // ################### check function #################
  // check val (list)
  template <typename T> bool chkVal(T val, const std::initializer_list<T> &list) {
    return std::any_of(list.begin(), list.end(), [=](T x) { return x == val; });
  }

  // check val (range)
  template <typename T> bool chkVal(T val, T min, T max) { return (min <= val && val <= max); }

  // check val (condtional func)
  template <typename T> bool chkVal(T val, const std::function<bool(T)> &func) { return func(val); }

  // verify (include range)
  void verifyIncRange(uint64_t val, uint64_t min, uint64_t max, int err_type, bool to_i = false) {
    if (to_i && !chkVal((int64_t)val, (int64_t)min, (int64_t)max)) {
      throw Error(err_type);
    } else if (!to_i && !chkVal(val, min, max)) {
      throw Error(err_type);
    }
  }

  // verify (not include range)
  void verifyNotIncRange(uint64_t val, uint64_t min, uint64_t max, int err_type, bool to_i = false) {
    if (to_i && chkVal((uint64_t)val, (uint64_t)min, (uint64_t)max)) {
      throw Error(err_type);
    } else if (!to_i && chkVal(val, min, max)) {
      throw Error(err_type);
    }
  }

  // verify (include list)
  void verifyIncList(uint64_t val, const std::initializer_list<uint64_t> &list, int err_type) {
    if (!chkVal(val, list)) {
      throw Error(err_type);
    }
  }

  // verify (not include list)
  void verifyNotIncList(uint64_t val, const std::initializer_list<uint64_t> &list, int err_type) {
    if (chkVal(val, list)) {
      throw Error(err_type);
    }
  }

  // verify (conditional function)
  void verifyCond(uint64_t val, const std::function<bool(uint64_t)> &func, int err_type) {
    if (!chkVal(val, func)) {
      throw Error(err_type);
    }
  }

  // verify (conditional function)
  void verifyNotCond(uint64_t val, const std::function<bool(uint64_t)> &func, int err_type) {
    if (chkVal(val, func)) {
      throw Error(err_type);
    }
  }

  // ############### encoding helper function #############
  // generate encoded imm
  uint32_t genNImmrImms(uint64_t imm, uint32_t size);

  // generate relative address for label offset
  uint64_t genLabelOffset(const Label &label, const JmpLabel &jmpL) {
    size_t offset = 0;
    int64_t labelOffset = 0;
    if (labelMgr_.getOffset(&offset, label)) {
      labelOffset = (offset - size_) * CSIZE;
    } else {
      labelMgr_.addUndefinedLabel(label, jmpL);
    }
    return labelOffset;
  }

  // ################## encoding function ##################
  // PC-rel. addressing
  uint32_t PCrelAddrEnc(uint32_t op, const XReg &rd, int64_t labelOffset);
  void PCrelAddr(uint32_t op, const XReg &rd, const Label &label);
  void PCrelAddr(uint32_t op, const XReg &rd, int64_t label);
  void AddSubImm(uint32_t op, uint32_t S, const RReg &rd, const RReg &rn, uint32_t imm, uint32_t sh);
  void LogicalImm(uint32_t opc, const RReg &rd, const RReg &rn, uint64_t imm, bool alias = false);
  void MvWideImm(uint32_t opc, const RReg &rd, uint32_t imm, uint32_t sh);
  void MvImm(const RReg &rd, uint64_t imm);
  void Bitfield(uint32_t opc, const RReg &rd, const RReg &rn, uint32_t immr, uint32_t imms, bool rn_chk = true);
  void Extract(uint32_t op21, uint32_t o0, const RReg &rd, const RReg &rn, const RReg &rm, uint32_t imm);
  uint32_t CondBrImmEnc(uint32_t cond, int64_t labelOffset);
  void CondBrImm(Cond cond, const Label &label);
  void CondBrImm(Cond cond, int64_t label);
  void ExceptionGen(uint32_t opc, uint32_t op2, uint32_t LL, uint32_t imm);
  void Hints(uint32_t CRm, uint32_t op2);
  void Hints(uint32_t imm);
  void BarriersOpt(uint32_t op2, BarOpt opt, uint32_t rt);
  void BarriersNoOpt(uint32_t CRm, uint32_t op2, uint32_t rt);
  void PState(PStateField psfield, uint32_t imm);
  void PState(uint32_t op1, uint32_t CRm, uint32_t op2);
  void SysInst(uint32_t L, uint32_t op1, uint32_t CRn, uint32_t CRm, uint32_t op2, const XReg &rt);
  void SysRegMove(uint32_t L, uint32_t op0, uint32_t op1, uint32_t CRn, uint32_t CRm, uint32_t op2, const XReg &rt);
  void UncondBrNoReg(uint32_t opc, uint32_t op2, uint32_t op3, uint32_t rn, uint32_t op4);
  void UncondBr1Reg(uint32_t opc, uint32_t op2, uint32_t op3, const RReg &rn, uint32_t op4);
  void UncondBr2Reg(uint32_t opc, uint32_t op2, uint32_t op3, const RReg &rn, const RReg &rm);
  uint32_t UncondBrImmEnc(uint32_t op, int64_t labelOffset);
  void UncondBrImm(uint32_t op, const Label &label);
  void UncondBrImm(uint32_t op, int64_t label);
  uint32_t CompareBrEnc(uint32_t op, const RReg &rt, int64_t labelOffset);
  void CompareBr(uint32_t op, const RReg &rt, const Label &label);
  void CompareBr(uint32_t op, const RReg &rt, int64_t label);
  uint32_t TestBrEnc(uint32_t op, const RReg &rt, uint32_t imm, int64_t labelOffset);
  void TestBr(uint32_t op, const RReg &rt, uint32_t imm, const Label &label);
  void TestBr(uint32_t op, const RReg &rt, uint32_t imm, int64_t label);
  void AdvSimdLdStMultiStructExceptLd1St1(uint32_t L, uint32_t opc, const VRegList &vt, const AdrNoOfs &adr);
  void AdvSimdLdStMultiStructForLd1St1(uint32_t L, uint32_t opc, const VRegList &vt, const AdrNoOfs &adr);
  void AdvSimdLdStMultiStructPostRegExceptLd1St1(uint32_t L, uint32_t opc, const VRegList &vt, const AdrPostReg &adr);
  void AdvSimdLdStMultiStructPostRegForLd1St1(uint32_t L, uint32_t opc, const VRegList &vt, const AdrPostReg &adr);
  void AdvSimdLdStMultiStructPostImmExceptLd1St1(uint32_t L, uint32_t opc, const VRegList &vt, const AdrPostImm &adr);
  void AdvSimdLdStMultiStructPostImmForLd1St1(uint32_t L, uint32_t opc, const VRegList &vt, const AdrPostImm &adr);
  void AdvSimdLdStSingleStruct(uint32_t L, uint32_t R, uint32_t num, const VRegElem &vt, const AdrNoOfs &adr);
  void AdvSimdLdRepSingleStruct(uint32_t L, uint32_t R, uint32_t opcode, uint32_t S, const VRegVec &vt, const AdrNoOfs &adr);
  void AdvSimdLdStSingleStructPostReg(uint32_t L, uint32_t R, uint32_t num, const VRegElem &vt, const AdrPostReg &adr);
  void AdvSimdLdStSingleStructRepPostReg(uint32_t L, uint32_t R, uint32_t opcode, uint32_t S, const VRegVec &vt, const AdrPostReg &adr);
  void AdvSimdLdStSingleStructPostImm(uint32_t L, uint32_t R, uint32_t num, const VRegElem &vt, const AdrPostImm &adr);
  void AdvSimdLdRepSingleStructPostImm(uint32_t L, uint32_t R, uint32_t opcode, uint32_t S, const VRegVec &vt, const AdrPostImm &adr);
  void StExclusive(uint32_t size, uint32_t o0, const WReg ws, const RReg &rt, const AdrImm &adr);
  void LdExclusive(uint32_t size, uint32_t o0, const RReg &rt, const AdrImm &adr);
  void StLORelase(uint32_t size, uint32_t o0, const RReg &rt, const AdrImm &adr);
  void LdLOAcquire(uint32_t size, uint32_t o0, const RReg &rt, const AdrImm &adr);
  void Cas(uint32_t size, uint32_t o2, uint32_t L, uint32_t o1, uint32_t o0, const RReg &rs, const RReg &rt, const AdrNoOfs &adr);
  void StExclusivePair(uint32_t L, uint32_t o1, uint32_t o0, const WReg &ws, const RReg &rt1, const RReg &rt2, const AdrImm &adr);
  void LdExclusivePair(uint32_t L, uint32_t o1, uint32_t o0, const RReg &rt1, const RReg &rt2, const AdrImm &adr);
  void CasPair(uint32_t L, uint32_t o1, uint32_t o0, const RReg &rs, const RReg &rt, const AdrNoOfs &adr);
  void LdaprStlr(uint32_t size, uint32_t opc, const RReg &rt, const AdrImm &adr);
  uint32_t LdRegLiteralEnc(uint32_t opc, uint32_t V, const RReg &rt, int64_t labelOffset);
  void LdRegLiteral(uint32_t opc, uint32_t V, const RReg &rt, const Label &label);
  void LdRegLiteral(uint32_t opc, uint32_t V, const RReg &rt, int64_t label);
  uint32_t LdRegSimdFpLiteralEnc(const VRegSc &vt, int64_t labelOffset);
  void LdRegSimdFpLiteral(const VRegSc &vt, const Label &label);
  void LdRegSimdFpLiteral(const VRegSc &vt, int64_t label);
  uint32_t PfLiteralEnc(Prfop prfop, int64_t labelOffset);
  void PfLiteral(Prfop prfop, const Label &label);
  void PfLiteral(Prfop prfop, int64_t label);
  void LdStNoAllocPair(uint32_t L, const RReg &rt1, const RReg &rt2, const AdrImm &adr);
  void LdStSimdFpNoAllocPair(uint32_t L, const VRegSc &vt1, const VRegSc &vt2, const AdrImm &adr);
  void LdStRegPairPostImm(uint32_t opc, uint32_t L, const RReg &rt1, const RReg &rt2, const AdrPostImm &adr);
  void LdStSimdFpPairPostImm(uint32_t L, const VRegSc &vt1, const VRegSc &vt2, const AdrPostImm &adr);
  void LdStRegPair(uint32_t opc, uint32_t L, const RReg &rt1, const RReg &rt2, const AdrImm &adr);
  void LdStSimdFpPair(uint32_t L, const VRegSc &vt1, const VRegSc &vt2, const AdrImm &adr);
  void LdStRegPairPre(uint32_t opc, uint32_t L, const RReg &rt1, const RReg &rt2, const AdrPreImm &adr);
  void LdStSimdFpPairPre(uint32_t L, const VRegSc &vt1, const VRegSc &vt2, const AdrPreImm &adr);
  void LdStRegUnsImm(uint32_t size, uint32_t opc, const RReg &rt, const AdrImm &adr);
  void LdStSimdFpRegUnsImm(uint32_t opc, const VRegSc &vt, const AdrImm &adr);
  void PfRegUnsImm(Prfop prfop, const AdrImm &adr);
  void LdStRegPostImm(uint32_t size, uint32_t opc, const RReg &rt, const AdrPostImm &adr);
  void LdStSimdFpRegPostImm(uint32_t opc, const VRegSc &vt, const AdrPostImm &adr);
  void LdStRegUnpriv(uint32_t size, uint32_t opc, const RReg &rt, const AdrImm &adr);
  void LdStRegPre(uint32_t size, uint32_t opc, const RReg &rt, const AdrPreImm &adr);
  void LdStSimdFpRegPre(uint32_t opc, const VRegSc &vt, const AdrPreImm &adr);
  void AtomicMemOp(uint32_t size, uint32_t V, uint32_t A, uint32_t R, uint32_t o3, uint32_t opc, const RReg &rs, const RReg &rt, const AdrNoOfs &adr);
  void AtomicMemOp(uint32_t size, uint32_t V, uint32_t A, uint32_t R, uint32_t o3, uint32_t opc, const RReg &rs, const RReg &rt, const AdrImm &adr);
  void LdStReg(uint32_t size, uint32_t opc, const RReg &rt, const AdrReg &adr);
  void LdStReg(uint32_t size, uint32_t opc, const RReg &rt, const AdrExt &adr);
  void LdStSimdFpReg(uint32_t opc, const VRegSc &vt, const AdrReg &adr);
  void LdStSimdFpReg(uint32_t opc, const VRegSc &vt, const AdrExt &adr);
  void PfExt(Prfop prfop, const AdrReg &adr);
  void PfExt(Prfop prfop, const AdrExt &adr);
  void LdStRegPac(uint32_t M, uint32_t W, const XReg &xt, const AdrImm &adr);
  void LdStRegPac(uint32_t M, uint32_t W, const XReg &xt, const AdrPreImm &adr);
  void LdStRegUnImm(uint32_t size, uint32_t opc, const RReg &rt, const AdrUimm &adr);
  void LdStSimdFpUnImm(uint32_t opc, const VRegSc &vt, const AdrUimm &adr);
  void PfRegImm(Prfop prfop, const AdrUimm &adr);
  void DataProc2Src(uint32_t opcode, const RReg &rd, const RReg &rn, const RReg &rm);
  void DataProc1Src(uint32_t opcode2, uint32_t opcode, const RReg &rd, const RReg &rn);
  void DataProc1Src(uint32_t opcode2, uint32_t opcode, const RReg &rd);
  void LogicalShiftReg(uint32_t opc, uint32_t N, const RReg &rd, const RReg &rn, const RReg &rm, ShMod shmod, uint32_t sh);
  void MvReg(const RReg &rd, const RReg &rn);
  void AddSubShiftReg(uint32_t opc, uint32_t S, const RReg &rd, const RReg &rn, const RReg &rm, ShMod shmod, uint32_t sh, bool alias = false);
  void AddSubExtReg(uint32_t opc, uint32_t S, const RReg &rd, const RReg &rn, const RReg &rm, ExtMod extmod, uint32_t sh);
  void AddSubCarry(uint32_t op, uint32_t S, const RReg &rd, const RReg &rn, const RReg &rm);
  void RotateR(uint32_t op, uint32_t S, uint32_t o2, const XReg &xn, uint32_t sh, uint32_t mask);
  void Evaluate(uint32_t op, uint32_t S, uint32_t opcode2, uint32_t sz, uint32_t o3, uint32_t mask, const WReg &wn);
  void CondCompReg(uint32_t op, uint32_t S, uint32_t o2, uint32_t o3, const RReg &rn, const RReg &rm, uint32_t nczv, Cond cond);
  void CondCompImm(uint32_t op, uint32_t S, uint32_t o2, uint32_t o3, const RReg &rn, uint32_t imm, uint32_t nczv, Cond cond);
  void CondSel(uint32_t op, uint32_t S, uint32_t op2, const RReg &rd, const RReg &rn, const RReg &rm, Cond cond);
  void DataProc3Reg(uint32_t op54, uint32_t op31, uint32_t o0, const RReg &rd, const RReg &rn, const RReg &rm, const RReg &ra);
  void DataProc3Reg(uint32_t op54, uint32_t op31, uint32_t o0, const RReg &rd, const RReg &rn, const RReg &rm);
  void CryptAES(uint32_t opcode, const VRegVec &vd, const VRegVec &vn);
  void Crypt3RegSHA(uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegVec &vm);
  void Crypt3RegSHA(uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
  void Crypt2RegSHA(uint32_t opcode, const Reg &vd, const Reg &vn);
  void AdvSimdScCopy(uint32_t op, uint32_t imm4, const VRegSc &vd, const VRegElem &vn);
  void AdvSimdSc3SameFp16(uint32_t U, uint32_t a, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm);
  void AdvSimdSc2RegMiscFp16(uint32_t U, uint32_t a, uint32_t opcode, const VRegSc &vd, const VRegSc &vn);
  void AdvSimdSc2RegMiscFp16(uint32_t U, uint32_t a, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, double zero);
  void AdvSimdSc3SameExtra(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm);
  void AdvSimdSc2RegMisc(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn);
  void AdvSimdSc2RegMisc(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, uint32_t zero);
  void AdvSimdSc2RegMiscSz0x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn);
  void AdvSimdSc2RegMiscSz1x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn);
  void AdvSimdSc2RegMiscSz1x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, double zero);
  void AdvSimdScPairwise(uint32_t U, uint32_t size, uint32_t opcode, const VRegSc &vd, const VRegVec &vn);
  void AdvSimdSc3Diff(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm);
  void AdvSimdSc3Same(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm);
  void AdvSimdSc3SameSz0x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm);
  void AdvSimdSc3SameSz1x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm);
  void AdvSimdScShImm(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, uint32_t sh);
  void AdvSimdScXIndElemSz(uint32_t U, uint32_t size, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegElem &vm);
  void AdvSimdScXIndElem(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegElem &vm);
  void AdvSimdTblLkup(uint32_t op2, uint32_t len, uint32_t op, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
  void AdvSimdTblLkup(uint32_t op2, uint32_t op, const VRegVec &vd, const VRegList &vn, const VRegVec &vm);
  void AdvSimdPermute(uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
  void AdvSimdExtract(uint32_t op2, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm, uint32_t index);
  void AdvSimdCopyDupElem(uint32_t op, uint32_t imm4, const VRegVec &vd, const VRegElem &vn);
  void AdvSimdCopyDupGen(uint32_t op, uint32_t imm4, const VRegVec &vd, const RReg &rn);
  void AdvSimdCopyMov(uint32_t op, uint32_t imm4, const RReg &rd, const VRegElem &vn);
  void AdvSimdCopyInsGen(uint32_t op, uint32_t imm4, const VRegElem &vd, const RReg &rn);
  void AdvSimdCopyElemIns(uint32_t op, const VRegElem &vd, const VRegElem &vn);
  void AdvSimd3SameFp16(uint32_t U, uint32_t a, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
  void AdvSimd2RegMiscFp16(uint32_t U, uint32_t a, uint32_t opcode, const VRegVec &vd, const VRegVec &vn);
  void AdvSimd2RegMiscFp16(uint32_t U, uint32_t a, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, double zero);
  void AdvSimd3SameExtra(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
  void AdvSimd3SameExtraRotate(uint32_t U, uint32_t op32, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm, uint32_t rotate);
  void AdvSimd2RegMisc(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn);
  void AdvSimd2RegMisc(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, uint32_t sh);
  void AdvSimd2RegMiscZero(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, uint32_t zero);
  void AdvSimd2RegMiscSz(uint32_t U, uint32_t size, uint32_t opcode, const VRegVec &vd, const VRegVec &vn);
  void AdvSimd2RegMiscSz0x(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn);
  void AdvSimd2RegMiscSz1x(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn);
  void AdvSimd2RegMiscSz1x(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, double zero);
  void AdvSimdAcrossLanes(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegVec &vn);
  void AdvSimdAcrossLanesSz0x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegVec &vn);
  void AdvSimdAcrossLanesSz1x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegVec &vn);
  void AdvSimd3Diff(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
  void AdvSimd3Same(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
  void AdvSimd3SameSz0x(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
  void AdvSimd3SameSz1x(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
  void AdvSimd3SameSz(uint32_t U, uint32_t size, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
  void AdvSimdModiImmMoviMvni(uint32_t op, uint32_t o2, const VRegVec &vd, uint32_t imm, ShMod shmod, uint32_t sh);
  void AdvSimdModiImmMoviMvniEnc(uint32_t Q, uint32_t op, uint32_t o2, const Reg &vd, uint64_t imm);
  void AdvSimdModiImmMoviMvni(uint32_t op, uint32_t o2, const VRegSc &vd, uint64_t imm);
  void AdvSimdModiImmMoviMvni(uint32_t op, uint32_t o2, const VRegVec &vd, uint64_t imm);
  void AdvSimdModiImmOrrBic(uint32_t op, uint32_t o2, const VRegVec &vd, uint32_t imm, ShMod mod, uint32_t sh);
  void AdvSimdModiImmFmov(uint32_t op, uint32_t o2, const VRegVec &vd, double imm);
  void AdvSimdShImm(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, uint32_t sh);
  void AdvSimdVecXindElemEnc(uint32_t Q, uint32_t U, uint32_t size, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegElem &vm);
  void AdvSimdVecXindElem(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegElem &vm);
  void AdvSimdVecXindElem(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegElem &vm, uint32_t rotate);
  void AdvSimdVecXindElemSz(uint32_t U, uint32_t size, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegElem &vm);
  void Crypto3RegImm2(uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegElem &vm);
  void Crypto3RegSHA512(uint32_t O, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegVec &vm);
  void Crypto3RegSHA512(uint32_t O, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm);
  void CryptoSHA(const VRegVec &vd, const VRegVec &vn, const VRegVec &vm, uint32_t imm6);
  void Crypto4Reg(uint32_t Op0, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm, const VRegVec &va);
  void Crypto2RegSHA512(uint32_t opcode, const VRegVec &vd, const VRegVec &vn);
  void ConversionFpFix(uint32_t S, uint32_t type, uint32_t rmode, uint32_t opcode, const VRegSc &vd, const RReg &rn, uint32_t fbits);
  void ConversionFpFix(uint32_t S, uint32_t type, uint32_t rmode, uint32_t opcode, const RReg &rd, const VRegSc &vn, uint32_t fbits);
  void ConversionFpInt(uint32_t sf, uint32_t S, uint32_t type, uint32_t rmode, uint32_t opcode, const RReg &rd, const VRegSc &vn);
  void ConversionFpInt(uint32_t sf, uint32_t S, uint32_t type, uint32_t rmode, uint32_t opcode, const VRegSc &vd, const RReg &rn);
  void ConversionFpInt(uint32_t sf, uint32_t S, uint32_t type, uint32_t rmode, uint32_t opcode, const RReg &rd, const VRegElem &vn);
  void ConversionFpInt(uint32_t sf, uint32_t S, uint32_t type, uint32_t rmode, uint32_t opcode, const VRegElem &vd, const RReg &rn);
  void FpDataProc1Reg(uint32_t M, uint32_t S, uint32_t type, uint32_t opcode, const VRegSc &vd, const VRegSc &vn);
  void FpComp(uint32_t M, uint32_t S, uint32_t type, uint32_t op, uint32_t opcode2, const VRegSc &vn, const VRegSc &vm);
  void FpComp(uint32_t M, uint32_t S, uint32_t type, uint32_t op, uint32_t opcode2, const VRegSc &vn, double imm);
  void FpImm(uint32_t M, uint32_t S, uint32_t type, const VRegSc &vd, double imm);
  void FpCondComp(uint32_t M, uint32_t S, uint32_t type, uint32_t op, const VRegSc &vn, const VRegSc &vm, uint32_t nzcv, Cond cond);
  void FpDataProc2Reg(uint32_t M, uint32_t S, uint32_t type, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm);
  void FpCondSel(uint32_t M, uint32_t S, uint32_t type, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm, Cond cond);
  void FpDataProc3Reg(uint32_t M, uint32_t S, uint32_t type, uint32_t o1, uint32_t o0, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm, const VRegSc &va);
  void InstCache(IcOp icop, const XReg &xt);
  void DataCache(DcOp dcop, const XReg &xt);
  void AddressTrans(AtOp atop, const XReg &xt);
  void TLBInv(TlbiOp tlbiop, const XReg &xt);
  void SveIntBinArPred(uint32_t opc, uint32_t type, const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveBitwiseLOpPred(uint32_t opc, const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveIntAddSubVecPred(uint32_t opc, const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveIntMinMaxDiffPred(uint32_t opc, uint32_t U, const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveIntMultDivVecPred(uint32_t opc, uint32_t U, const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveIntReduction(uint32_t opc, uint32_t type, const Reg &rd, const _PReg &pg, const Reg &rn);
  void SveBitwiseLReductPred(uint32_t opc, const VRegSc &vd, const _PReg &pg, const _ZReg &zn);
  void SveConstPrefPred(uint32_t opc, const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveIntAddReductPred(uint32_t opc, uint32_t U, const VRegSc &vd, const _PReg &pg, const _ZReg &zn);
  void SveIntMinMaxReductPred(uint32_t opc, uint32_t U, const VRegSc &vd, const _PReg &pg, const _ZReg &zn);
  void SveBitShPred(uint32_t opc, uint32_t type, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm);
  void SveBitwiseShByImmPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, uint32_t amount);
  void SveBitwiseShVecPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm);
  void SveBitwiseShWElemPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm);
  void SveIntUnaryArPred(uint32_t opc, uint32_t type, const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveBitwiseUnaryOpPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm);
  void SveIntUnaryOpPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm);
  void SveIntMultAccumPred(uint32_t opc, const _ZReg &zda, const _PReg &pg, const _ZReg &zn, const _ZReg &zm);
  void SveIntMultAddPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm, const _ZReg &za);
  void SveIntAddSubUnpred(uint32_t opc, const _ZReg &zd, const _ZReg &zn, const _ZReg &zm);
  void SveBitwiseLOpUnpred(uint32_t opc, const _ZReg &zd, const _ZReg &zn, const _ZReg &zm);
  void SveIndexGenImmImmInc(const _ZReg &zd, int32_t imm1, int32_t imm2);
  void SveIndexGenImmRegInc(const _ZReg &zd, int32_t imm, const RReg &rm);
  void SveIndexGenRegImmInc(const _ZReg &zd, const RReg &rn, int32_t imm);
  void SveIndexGenRegRegInc(const _ZReg &zd, const RReg &rn, const RReg &rm);
  void SveStackFrameAdjust(uint32_t op, const XReg &xd, const XReg &xn, int32_t imm);
  void SveStackFrameSize(uint32_t op, uint32_t opc2, const XReg &xd, int32_t imm);
  void SveBitwiseShByImmUnpred(uint32_t opc, const _ZReg &zd, const _ZReg &zn, uint32_t amount);
  void SveBitwiseShByWideElemUnPred(uint32_t opc, const _ZReg &zd, const _ZReg &zn, const _ZReg &zm);
  void SveAddressGen(const _ZReg &zd, const AdrVec &adr);
  void SveAddressGen(const _ZReg &zd, const AdrVecU &adr);
  void SveIntMiscUnpred(uint32_t size, uint32_t opc, uint32_t type, const _ZReg &zd, const _ZReg &zn);
  void SveConstPrefUnpred(uint32_t opc, uint32_t opc2, const _ZReg &zd, const _ZReg &zn);
  void SveFpExpAccel(uint32_t opc, const _ZReg &zd, const _ZReg &zn);
  void SveFpTrigSelCoef(uint32_t opc, const _ZReg &zd, const _ZReg &zn, const _ZReg &zm);
  void SveElemCountGrp(uint32_t size, uint32_t op, uint32_t type1, uint32_t type2, const Reg &rd, Pattern pat, ExtMod mod, uint32_t imm);
  void SveElemCount(uint32_t size, uint32_t op, const XReg &xd, Pattern pat, ExtMod mod, uint32_t imm);
  void SveIncDecRegByElemCount(uint32_t size, uint32_t D, const XReg &xd, Pattern pat, ExtMod mod, uint32_t imm);
  void SveIncDecVecByElemCount(uint32_t size, uint32_t D, const _ZReg &zd, Pattern pat, ExtMod mod, uint32_t imm);
  void SveSatuIncDecRegByElemCount(uint32_t size, uint32_t D, uint32_t U, const RReg &rdn, Pattern pat, ExtMod mod, uint32_t imm);
  void SveSatuIncDecVecByElemCount(uint32_t size, uint32_t D, uint32_t U, const _ZReg &zdn, Pattern pat, ExtMod mod, uint32_t imm);
  void SveBitwiseImm(uint32_t opc, const _ZReg &zd, uint64_t imm);
  void SveBitwiseLogicalImmUnpred(uint32_t opc, const _ZReg &zdn, uint64_t imm);
  void SveBcBitmaskImm(const _ZReg &zdn, uint64_t imm);
  void SveCopyFpImmPred(const _ZReg &zd, const _PReg &pg, double imm);
  void SveCopyIntImmPred(const _ZReg &zd, const _PReg &pg, uint32_t imm, ShMod mod, uint32_t sh);
  void SveExtVec(const _ZReg &zdn, const _ZReg &zm, uint32_t imm);
  void SvePerVecUnpred(uint32_t size, uint32_t type1, uint32_t type2, const _ZReg &zd, const Reg &rn);
  void SveBcGeneralReg(const _ZReg &zd, const RReg &rn);
  void SveBcIndexedElem(const _ZReg &zd, const ZRegElem &zn);
  void SveInsSimdFpSclarReg(const _ZReg &zdn, const VRegSc &vm);
  void SveInsGeneralReg(const _ZReg &zdn, const RReg &rm);
  void SveRevVecElem(const _ZReg &zd, const _ZReg &zn);
  void SveTableLookup(const _ZReg &zd, const _ZReg &zn, const _ZReg &zm);
  void SveUnpackVecElem(uint32_t U, uint32_t H, const _ZReg &zd, const _ZReg &zn);
  void SvePermutePredElem(uint32_t opc, uint32_t H, const _PReg &pd, const _PReg &pn, const _PReg &pm);
  void SveRevPredElem(const _PReg &pd, const _PReg &pn);
  void SveUnpackPredElem(uint32_t H, const _PReg &pd, const _PReg &pn);
  void SvePermuteVecElem(uint32_t opc, const _ZReg &zd, const _ZReg &zn, const _ZReg &zm);
  void SveCompressActElem(const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveCondBcElemToVec(uint32_t B, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm);
  void SveCondExtElemToSimdFpScalar(uint32_t B, const VRegSc &vdn, const _PReg &pg, const _ZReg &zm);
  void SveCondExtElemToGeneralReg(uint32_t B, const RReg &rdn, const _PReg &pg, const _ZReg &zm);
  void SveCopySimdFpScalarToVecPred(const _ZReg &zd, const _PReg &pg, const VRegSc &vn);
  void SveCopyGeneralRegToVecPred(const _ZReg &zd, const _PReg &pg, const RReg &rn);
  void SveExtElemToSimdFpScalar(uint32_t B, const VRegSc &vd, const _PReg &pg, const _ZReg &zn);
  void SveExtElemToGeneralReg(uint32_t B, const RReg &rd, const _PReg &pg, const _ZReg &zn);
  void SveRevWithinElem(uint32_t opc, const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveSelVecSplice(const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveSelVecElemPred(const _ZReg &zd, const _PReg &pg, const _ZReg &zn, const _ZReg &zm);
  void SveIntCompVecGrp(uint32_t opc, uint32_t ne, const _PReg &pd, const _PReg &pg, const _ZReg &zn, const _ZReg &zm);
  void SveIntCompVec(uint32_t op, uint32_t o2, uint32_t ne, const _PReg &pd, const _PReg &pg, const _ZReg &zn, const _ZReg &zm);
  void SveIntCompWideElem(uint32_t op, uint32_t o2, uint32_t ne, const _PReg &pd, const _PReg &pg, const _ZReg &zn, const _ZReg &zm);
  void SveIntCompUImm(uint32_t lt, uint32_t ne, const _PReg &pd, const _PReg &pg, const _ZReg &zn, uint32_t imm);
  void SvePredLOp(uint32_t op, uint32_t S, uint32_t o2, uint32_t o3, const _PReg &pd, const _PReg &pg, const _PReg &pn, const _PReg &pm);
  void SvePropagateBreakPrevPtn(uint32_t op, uint32_t S, uint32_t B, const _PReg &pd, const _PReg &pg, const _PReg &pn, const _PReg &pm);
  void SvePartitionBreakCond(uint32_t B, uint32_t S, const _PReg &pd, const _PReg &pg, const _PReg &pn);
  void SvePropagateBreakNextPart(uint32_t S, const _PReg &pdm, const _PReg &pg, const _PReg &pn);
  void SvePredFirstAct(uint32_t op, uint32_t S, const _PReg &pdn, const _PReg &pg);
  void SvePredInit(uint32_t S, const _PReg &pd, Pattern pat);
  void SvePredNextAct(const _PReg &pdn, const _PReg &pg);
  void SvePredReadFFRPred(uint32_t op, uint32_t S, const _PReg &pd, const _PReg &pg);
  void SvePredReadFFRUnpred(uint32_t op, uint32_t S, const _PReg &pd);
  void SvePredTest(uint32_t op, uint32_t S, uint32_t opc2, const _PReg &pg, const _PReg &pn);
  void SvePredZero(uint32_t op, uint32_t S, const _PReg &pd);
  void SveIntCompSImm(uint32_t op, uint32_t o2, uint32_t ne, const _PReg &pd, const _PReg &pg, const _ZReg &zn, int32_t imm);
  void SvePredCount(uint32_t opc, uint32_t o2, const RReg &rd, const _PReg &pg, const _PReg &pn);
  void SveIncDecPredCount(uint32_t size, uint32_t op, uint32_t D, uint32_t opc2, uint32_t type1, uint32_t type2, const Reg &rdn, const _PReg &pg);
  void SveIncDecRegByPredCount(uint32_t op, uint32_t D, uint32_t opc2, const RReg &rdn, const _PReg &pg);
  void SveIncDecVecByPredCount(uint32_t op, uint32_t D, uint32_t opc2, const _ZReg &zdn, const _PReg &pg);
  void SveSatuIncDecRegByPredCount(uint32_t D, uint32_t U, uint32_t op, const RReg &rdn, const _PReg &pg);
  void SveSatuIncDecVecByPredCount(uint32_t D, uint32_t U, uint32_t opc, const _ZReg &zdn, const _PReg &pg);
  void SveFFRInit(uint32_t opc);
  void SveFFRWritePred(uint32_t opc, const _PReg &pn);
  void SveCondTermScalars(uint32_t op, uint32_t ne, const RReg &rn, const RReg &rm);
  void SveIntCompScalarCountAndLimit(uint32_t U, uint32_t lt, uint32_t eq, const _PReg &pd, const RReg &rn, const RReg &rm);
  void SveBcFpImmUnpred(uint32_t opc, uint32_t o2, const _ZReg &zd, double imm);
  void SveBcIntImmUnpred(uint32_t opc, const _ZReg &zd, int32_t imm, ShMod mod, uint32_t sh);
  void SveIntAddSubImmUnpred(uint32_t opc, const _ZReg &zdn, uint32_t imm, ShMod mod, uint32_t sh);
  void SveIntMinMaxImmUnpred(uint32_t opc, uint32_t o2, const _ZReg &zdn, int32_t imm);
  void SveIntMultImmUnpred(uint32_t opc, uint32_t o2, const _ZReg &zdn, int32_t imm);
  void SveIntDotProdcutUnpred(uint32_t U, const _ZReg &zda, const _ZReg &zn, const _ZReg &zm);
  void SveIntDotProdcutIndexed(uint32_t size, uint32_t U, const _ZReg &zda, const _ZReg &zn, const ZRegElem &zm);
  void SveFpComplexAddPred(const _ZReg &zdn, const _PReg &pg, const _ZReg &zm, uint32_t ct);
  void SveFpComplexMultAddPred(const _ZReg &zda, const _PReg &pg, const _ZReg &zn, const _ZReg &zm, uint32_t ct);
  void SveFpMultAddIndexed(uint32_t op, const _ZReg &zda, const _ZReg &zn, const ZRegElem &zm);
  void SveFpComplexMultAddIndexed(const _ZReg &zda, const _ZReg &zn, const ZRegElem &zm, uint32_t ct);
  void SveFpMultIndexed(const _ZReg &zd, const _ZReg &zn, const ZRegElem &zm);
  void SveFpRecurReduct(uint32_t opc, const VRegSc vd, const _PReg &pg, const _ZReg &zn);
  void SveFpReciproEstUnPred(uint32_t opc, const _ZReg &zd, const _ZReg &zn);
  void SveFpCompWithZero(uint32_t eq, uint32_t lt, uint32_t ne, const _PReg &pd, const _PReg &pg, const _ZReg &zn, double zero);
  void SveFpSerialReductPred(uint32_t opc, const VRegSc vdn, const _PReg &pg, const _ZReg &zm);
  void SveFpArithmeticUnpred(uint32_t opc, const _ZReg &zd, const _ZReg &zn, const _ZReg &zm);
  void SveFpArithmeticPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm);
  void SveFpArithmeticImmPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, float ct);
  void SveFpTrigMultAddCoef(const _ZReg &zdn, const _ZReg &zm, uint32_t imm);
  void SveFpCvtPrecision(uint32_t opc, uint32_t opc2, const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveFpCvtToInt(uint32_t opc, uint32_t opc2, uint32_t U, const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveFpRoundToIntegral(uint32_t opc, const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveFpUnaryOp(uint32_t opc, const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveIntCvtToFp(uint32_t opc, uint32_t opc2, uint32_t U, const _ZReg &zd, const _PReg &pg, const _ZReg &zn);
  void SveFpCompVec(uint32_t op, uint32_t o2, uint32_t o3, const _PReg &pd, const _PReg &pg, const _ZReg &zn, const _ZReg &zm);
  void SveFpMultAccumAddend(uint32_t opc, const _ZReg &zda, const _PReg &pg, const _ZReg &zn, const _ZReg &zm);
  void SveFpMultAccumMulti(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm, const _ZReg &za);
  void Sve32GatherLdSc32U(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc32U &adr);
  void Sve32GatherLdVecImm(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrVecImm32 &adr);
  void Sve32GatherLdHSc32S(uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc32S &adr);
  void Sve32GatherLdWSc32S(uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc32S &adr);
  void Sve32GatherPfSc32S(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrSc32S &adr);
  void Sve32GatherPfVecImm(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrVecImm32 &adr);
  void Sve32ContiPfScImm(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrScImm &adr);
  void Sve32ContiPfScImm(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrNoOfs &adr);
  void Sve32ContiPfScSc(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrScSc &adr);
  void SveLoadAndBcElem(uint32_t dtypeh, uint32_t dtypel, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr);
  void SveLoadAndBcElem(uint32_t dtypeh, uint32_t dtypel, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr);
  void SveLoadPredReg(const _PReg &pt, const AdrScImm &adr);
  void SveLoadPredReg(const _PReg &pt, const AdrNoOfs &adr);
  void SveLoadPredVec(const _ZReg &zt, const AdrScImm &adr);
  void SveLoadPredVec(const _ZReg &zt, const AdrNoOfs &adr);
  void SveContiFFLdScSc(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr);
  void SveContiFFLdScSc(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr);
  void SveContiLdScImm(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr);
  void SveContiLdScImm(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr);
  void SveContiLdScSc(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr);
  void SveContiNFLdScImm(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr);
  void SveContiNFLdScImm(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr);
  void SveContiNTLdScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr);
  void SveContiNTLdScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr);
  void SveContiNTLdScSc(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr);
  void SveLdBcQuadScImm(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr);
  void SveLdBcQuadScImm(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr);
  void SveLdBcQuadScSc(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr);
  void SveLdMultiStructScImm(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr);
  void SveLdMultiStructScImm(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr);
  void SveLdMultiStructScSc(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr);
  void Sve64GatherLdSc32US(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc32US &adr);
  void Sve64GatherLdSc64S(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc64S &adr);
  void Sve64GatherLdSc64U(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc64U &adr);
  void Sve64GatherLdSc32UU(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc32UU &adr);
  void Sve64GatherLdVecImm(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrVecImm64 &adr);
  void Sve64GatherPfSc64S(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrSc64S &adr);
  void Sve64GatherPfSc32US(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrSc32US &adr);
  void Sve64GatherPfVecImm(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrVecImm64 &adr);
  void Sve32ScatterStSc32S(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrSc32S &adr);
  void Sve32ScatterStSc32U(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrSc32U &adr);
  void Sve32ScatterStVecImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrVecImm32 &adr);
  void Sve64ScatterStSc64S(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrSc64S &adr);
  void Sve64ScatterStSc64U(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrSc64U &adr);
  void Sve64ScatterStSc32US(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrSc32US &adr);
  void Sve64ScatterStSc32UU(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrSc32UU &adr);
  void Sve64ScatterStVecImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrVecImm64 &adr);
  void SveContiNTStScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr);
  void SveContiNTStScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr);
  void SveContiNTStScSc(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr);
  void SveContiStScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr);
  void SveContiStScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr);
  void SveContiStScSc(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr);
  void SveStMultiStructScImm(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr);
  void SveStMultiStructScImm(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr);
  void SveStMultiStructScSc(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr);
  void SveStorePredReg(const _PReg &pt, const AdrScImm &adr);
  void SveStorePredReg(const _PReg &pt, const AdrNoOfs &adr);
  void SveStorePredVec(const _ZReg &zt, const AdrScImm &adr);
  void SveStorePredVec(const _ZReg &zt, const AdrNoOfs &adr);
  void mov(const XReg &rd, const Label &label) { adr(rd, label); }

  template <class T> void putL_inner(T &label) {
    if (isAutoGrow() && size_ >= maxSize_)
      growMemory();
    UncondBrImm(0, label); // insert nemonic (B <label>)
  }

public:
  const WReg w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12;
  const WReg w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23;
  const WReg w24, w25, w26, w27, w28, w29, w30, wzr, wsp;

  const XReg x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12;
  const XReg x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23;
  const XReg x24, x25, x26, x27, x28, x29, x30, xzr, sp;

  const BReg b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12;
  const BReg b13, b14, b15, b16, b17, b18, b19, b20, b21, b22, b23;
  const BReg b24, b25, b26, b27, b28, b29, b30, b31;

  const HReg h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12;
  const HReg h13, h14, h15, h16, h17, h18, h19, h20, h21, h22, h23;
  const HReg h24, h25, h26, h27, h28, h29, h30, h31;

#ifdef XBYAK_AARCH64_FOR_DNNL
  const SReg s0, s1, s2, s3, s4, s5, s6, s7, s8_, s9, s10, s11, s12;
#else
  const SReg s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12;
#endif
  const SReg s13, s14, s15, s16, s17, s18, s19, s20, s21, s22, s23;
  const SReg s24, s25, s26, s27, s28, s29, s30, s31;

  const DReg d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12;
  const DReg d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23;
  const DReg d24, d25, d26, d27, d28, d29, d30, d31;

  const QReg q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12;
  const QReg q13, q14, q15, q16, q17, q18, q19, q20, q21, q22, q23;
  const QReg q24, q25, q26, q27, q28, q29, q30, q31;

  const VReg v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12;
  const VReg v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23;
  const VReg v24, v25, v26, v27, v28, v29, v30, v31;

  const ZReg z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12;
  const ZReg z13, z14, z15, z16, z17, z18, z19, z20, z21, z22, z23;
  const ZReg z24, z25, z26, z27, z28, z29, z30, z31;

  const PReg p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12;
  const PReg p13, p14, p15;

  CodeGenerator(size_t maxSize = DEFAULT_MAX_CODE_SIZE, void *userPtr = DontSetProtectRWE, Allocator *allocator = 0)
      : CodeArray(maxSize, userPtr, allocator)
#if 1
        ,
        w0(0), w1(1), w2(2), w3(3), w4(4), w5(5), w6(6), w7(7), w8(8), w9(9), w10(10), w11(11), w12(12), w13(13), w14(14), w15(15), w16(16), w17(17), w18(18), w19(19), w20(20), w21(21), w22(22), w23(23), w24(24), w25(25), w26(26), w27(27), w28(28), w29(29), w30(30), wzr(31), wsp(31)

        ,
        x0(0), x1(1), x2(2), x3(3), x4(4), x5(5), x6(6), x7(7), x8(8), x9(9), x10(10), x11(11), x12(12), x13(13), x14(14), x15(15), x16(16), x17(17), x18(18), x19(19), x20(20), x21(21), x22(22), x23(23), x24(24), x25(25), x26(26), x27(27), x28(28), x29(29), x30(30), xzr(31), sp(31)

        ,
        b0(0), b1(1), b2(2), b3(3), b4(4), b5(5), b6(6), b7(7), b8(8), b9(9), b10(10), b11(11), b12(12), b13(13), b14(14), b15(15), b16(16), b17(17), b18(18), b19(19), b20(20), b21(21), b22(22), b23(23), b24(24), b25(25), b26(26), b27(27), b28(28), b29(29), b30(30), b31(31)

        ,
        h0(0), h1(1), h2(2), h3(3), h4(4), h5(5), h6(6), h7(7), h8(8), h9(9), h10(10), h11(11), h12(12), h13(13), h14(14), h15(15), h16(16), h17(17), h18(18), h19(19), h20(20), h21(21), h22(22), h23(23), h24(24), h25(25), h26(26), h27(27), h28(28), h29(29), h30(30), h31(31)

        ,
        s0(0), s1(1), s2(2), s3(3), s4(4), s5(5), s6(6), s7(7),
#ifdef XBYAK_AARCH64_FOR_DNNL
        s8_(8),
#else
        s8(8),
#endif
        s9(9), s10(10), s11(11), s12(12), s13(13), s14(14), s15(15), s16(16), s17(17), s18(18), s19(19), s20(20), s21(21), s22(22), s23(23), s24(24), s25(25), s26(26), s27(27), s28(28), s29(29), s30(30), s31(31)

        ,
        d0(0), d1(1), d2(2), d3(3), d4(4), d5(5), d6(6), d7(7), d8(8), d9(9), d10(10), d11(11), d12(12), d13(13), d14(14), d15(15), d16(16), d17(17), d18(18), d19(19), d20(20), d21(21), d22(22), d23(23), d24(24), d25(25), d26(26), d27(27), d28(28), d29(29), d30(30), d31(31)

        ,
        q0(0), q1(1), q2(2), q3(3), q4(4), q5(5), q6(6), q7(7), q8(8), q9(9), q10(10), q11(11), q12(12), q13(13), q14(14), q15(15), q16(16), q17(17), q18(18), q19(19), q20(20), q21(21), q22(22), q23(23), q24(24), q25(25), q26(26), q27(27), q28(28), q29(29), q30(30), q31(31)

        ,
        v0(0), v1(1), v2(2), v3(3), v4(4), v5(5), v6(6), v7(7), v8(8), v9(9), v10(10), v11(11), v12(12), v13(13), v14(14), v15(15), v16(16), v17(17), v18(18), v19(19), v20(20), v21(21), v22(22), v23(23), v24(24), v25(25), v26(26), v27(27), v28(28), v29(29), v30(30), v31(31)

        ,
        z0(0), z1(1), z2(2), z3(3), z4(4), z5(5), z6(6), z7(7), z8(8), z9(9), z10(10), z11(11), z12(12), z13(13), z14(14), z15(15), z16(16), z17(17), z18(18), z19(19), z20(20), z21(21), z22(22), z23(23), z24(24), z25(25), z26(26), z27(27), z28(28), z29(29), z30(30), z31(31)

        ,
        p0(0), p1(1), p2(2), p3(3), p4(4), p5(5), p6(6), p7(7), p8(8), p9(9), p10(10), p11(11), p12(12), p13(13), p14(14), p15(15)
#endif
  {
    labelMgr_.set(this);
  }

  unsigned int getVersion() const { return VERSION; }

  void L(Label &label) { labelMgr_.defineClabel(label); }
  Label L() {
    Label label;
    L(label);
    return label;
  }
  void inLocalLabel() { /*assert(NULL);*/
  }
  void outLocalLabel() { /*assert(NULL);*/
  }
  /*
          assign src to dst
          require
          dst : does not used by L()
          src : used by L()
  */
  void assignL(Label &dst, const Label &src) { labelMgr_.assign(dst, src); }
  /*
          put address of label to buffer
          @note the put size is 4(32-bit), 8(64-bit)
  */
  void putL(const Label &label) { putL_inner(label); }

  void reset() {
    resetSize();
    labelMgr_.reset();
    labelMgr_.set(this);
  }
  bool hasUndefinedLabel() const { return labelMgr_.hasUndefClabel(); }
  void clearCache(void *begin, void *end) {
#ifdef _WIN32
    (void)begin;
    (void)end;
#elif __APPLE__
    sys_icache_invalidate(begin, ((char *)end) - ((char *)begin));
#else
    __builtin___clear_cache((char *)begin, (char *)end);
#endif
  }
  /*
          MUST call ready() to complete generating code if you use AutoGrow
     mode.
          It is not necessary for the other mode if hasUndefinedLabel() is true.
  */
  void ready(ProtectMode mode = PROTECT_RE) {
    if (hasUndefinedLabel())
      throw Error(ERR_LABEL_IS_NOT_FOUND);
    if (isAutoGrow()) {
      calcJmpAddress();
    }
    if (useProtect())
      setProtectMode(mode);
    clearCache(const_cast<uint8_t *>(getCode()), const_cast<uint8_t *>(getCurr()));
  }
  // set read/exec
  void readyRE() { return ready(PROTECT_RE); }
#ifdef XBYAK_TEST
  void dump(bool doClear = true) {
    CodeArray::dump();
    if (doClear)
      size_ = 0;
  }
#endif

  WReg getTmpWReg() { return w29; }

  XReg getTmpXReg() { return x29; }

  VReg getTmpVReg() { return v31; }

  ZReg getTmpZReg() { return z31; }

  PReg getTmpPReg() { return p7; }

  /* If "imm" is "00..011..100..0" or "11..100..011..1",
     this function returns TRUE, otherwise FALSE. */
  template <typename T> bool isBitMask(T imm) {
    uint64_t bit_ptn = static_cast<uint64_t>(imm);
    int curr, prev = 0;
    uint64_t invCount = 0;

    prev = (bit_ptn & 0x1) ? 1 : 0;

    for (size_t i = 1; i < 8 * sizeof(T); i++) {
      curr = (bit_ptn & (uint64_t(1) << i)) ? 1 : 0;
      if (prev != curr) {
        invCount++;
      }
      prev = curr;
    }

    if (1 <= invCount && invCount <= 2) { // intCount == 0 means all 0 or all 1
      return true;
    }

    return false;
  }

#include "xbyak_aarch64_meta_mnemonic.h"
#include "xbyak_aarch64_mnemonic_def.h"

  void align(size_t x) {
    if (x == 4)
      return; // ARMv8 instructions are always 4 bytes.
    if (x < 4 || (x % 4))
      throw Error(ERR_BAD_ALIGN);

    if (isAutoGrow() && x > inner::getPageSize())
      fprintf(stderr, "warning:autoGrow mode does not support %d align\n", (int)x);

    size_t remain = size_t(getCurr());
    if (remain % 4)
      throw Error(ERR_BAD_ALIGN);
    remain = x - (remain % x);

    while (remain) {
      nop();
      remain -= 4;
    }
  }
};

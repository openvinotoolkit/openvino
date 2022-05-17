/*******************************************************************************
 * Copyright 2020-2021 FUJITSU LIMITED
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
/////////////// bit operation ////////////////////
constexpr uint32_t F(uint32_t val, uint32_t pos) { return val << pos; }
inline uint64_t lsb(uint64_t v) { return v & 0x1; }

uint32_t concat(const std::initializer_list<uint32_t> list) {
  uint32_t result = 0;
  for (auto f : list) {
    result |= f;
  }
  return result;
}

inline uint64_t msb(uint64_t v, uint32_t size) {
  uint32_t shift = (size == 0) ? 0 : size - 1;
  return (v >> shift) & 0x1;
}

inline uint64_t ones(uint32_t size) { return (size == 64) ? 0xffffffffffffffff : ((uint64_t)1 << size) - 1; }

inline uint32_t field(uint64_t v, uint32_t mpos, uint32_t lpos) { return static_cast<uint32_t>((v >> lpos) & ones(mpos - lpos + 1)); }

inline uint64_t rrotate(uint64_t v, uint32_t size, uint32_t num) {
  uint32_t shift = (size == 0) ? 0 : (num % size);
  v &= ones(size);
  return (v >> shift) | ((v & ones(shift)) << (size - shift));
}

inline uint64_t lrotate(uint64_t v, uint32_t size, uint32_t num) {
  uint32_t shift = (size == 0) ? 0 : (num % size);
  v &= ones(size);
  return ((v << shift) | ((v >> (size - shift)))) & ones(size);
}

inline uint64_t replicate(uint64_t v, uint32_t esize, uint32_t size) {
  uint64_t result = 0;
  for (uint32_t i = 0; i < 64 / esize; ++i) {
    result |= v << (esize * i);
  }
  return result & ones(size);
}

/////////////// ARMv8/SVE psuedo code function ////////////////
bool checkPtn(uint64_t v, uint32_t esize, uint32_t size) {
  std::vector<uint64_t> ptns;
  uint32_t max_num = size / esize;
  for (uint32_t i = 0; i < max_num; ++i) {
    ptns.push_back((v >> (esize * i)) & ones(esize));
  }
  return std::all_of(ptns.begin(), ptns.end(), [&ptns](uint64_t x) { return x == ptns[0]; });
}

uint32_t getPtnSize(uint64_t v, uint32_t size) {
  uint32_t esize;
  for (esize = 2; esize <= size; esize <<= 1) {
    if (checkPtn(v, esize, size))
      break;
  }
  return esize;
}

uint32_t getPtnRotateNum(uint64_t ptn, uint32_t ptn_size) {
  assert(ptn != 0 && (ptn & ones(ptn_size)) != ones(ptn_size));
  uint32_t num;
  for (num = 0; msb(ptn, ptn_size) || !lsb(ptn); ++num) {
    ptn = lrotate(ptn, ptn_size, 1);
  }
  return num;
}

uint32_t countOneBit(uint64_t v, uint32_t size) {
  uint64_t num = 0;
  for (uint32_t i = 0; i < size; ++i) {
    num += lsb(v);
    v >>= 1;
  };
  return static_cast<uint32_t>(num);
}

uint32_t countSeqOneBit(uint64_t v, uint32_t size) {
  uint32_t num;
  for (num = 0; num < size && lsb(v); ++num) {
    v >>= 1;
  };
  return num;
}

uint32_t compactImm(double imm, uint32_t size) {
  uint32_t sign = (imm < 0) ? 1 : 0;

  imm = std::abs(imm);
  int32_t max_digit = static_cast<int32_t>(std::floor(std::log2(imm)));

  int32_t n = (size == 16) ? 7 : (size == 32) ? 10 : 13;
  int32_t exp = (max_digit - 1) + (1 << n);

  imm -= pow(2, max_digit);
  uint32_t frac = 0;
  for (int i = 0; i < 4; ++i) {
    if (pow(2, max_digit - 1 - i) <= imm) {
      frac |= 1 << (3 - i);
      imm -= pow(2, max_digit - 1 - i);
    }
  }
  uint32_t imm8 = concat({F(sign, 7), F(field(~exp, n, n), 6), F(field(exp, 1, 0), 4), F(frac, 0)});
  return imm8;
}

uint32_t compactImm(uint64_t imm) {
  uint32_t imm8 = 0;
  for (uint32_t i = 0; i < 64; i += 8) {
    uint32_t bit = (imm >> i) & 0x1;
    imm8 |= bit << (i >> 3);
  }
  return imm8;
}

bool isCompact(uint64_t imm, uint32_t imm8) {
  bool result = true;
  for (uint32_t i = 0; i < 64; ++i) {
    uint32_t bit = (imm >> i) & 0x1;
    uint32_t rbit = (imm8 >> (i >> 3)) & 0x1;
    result &= (bit == rbit);
  }
  return result;
}

uint64_t genMoveMaskPrefferd(uint64_t imm) {
  bool chk_result = true;
  if (field(imm, 7, 0) != 0) {
    if (field(imm, 63, 7) == 0 || field(imm, 63, 7) == ones(57))
      chk_result = false;
    if ((field(imm, 63, 32) == field(imm, 31, 0)) && (field(imm, 31, 7) == 0 || field(imm, 31, 7) == ones(25)))
      chk_result = false;
    if ((field(imm, 63, 32) == field(imm, 31, 0)) && (field(imm, 31, 16) == field(imm, 15, 0)) && (field(imm, 15, 7) == 0 || field(imm, 15, 7) == ones(9)))
      chk_result = false;
    if ((field(imm, 63, 32) == field(imm, 31, 0)) && (field(imm, 31, 16) == field(imm, 15, 0)) && (field(imm, 15, 8) == field(imm, 7, 0)))
      chk_result = false;
  } else {
    if (field(imm, 63, 15) == 0 || field(imm, 63, 15) == ones(49))
      chk_result = false;
    if ((field(imm, 63, 32) == field(imm, 31, 0)) && (field(imm, 31, 7) == 0 || field(imm, 31, 7) == ones(25)))
      chk_result = false;
    if ((field(imm, 63, 32) == field(imm, 31, 0)) && (field(imm, 31, 16) == field(imm, 15, 0)))
      chk_result = false;
  }
  return (chk_result) ? imm : 0;
}

Cond invert(Cond cond) {
  uint32_t inv_val = (uint32_t)cond ^ 1;
  return (Cond)(inv_val & ones(4));
}

uint32_t genHw(uint64_t imm, uint32_t size) {
  if (imm == 0)
    return 0;

  uint32_t hw = 0;
  uint32_t times = (size == 32) ? 1 : 3;
  for (uint32_t i = 0; i < times; ++i) {
    if (field(imm, 15, 0) != 0)
      break;
    ++hw;
    imm >>= 16;
  }
  return hw;
}

/////////////// ARM8/SVE encoding helper function ////////////////

uint32_t genSf(const RReg &Reg) { return (Reg.getBit() == 64) ? 1 : 0; }

uint32_t genQ(const VRegVec &Reg) { return (Reg.getBit() * Reg.getLane() == 128) ? 1 : 0; }

uint32_t genQ(const VRegElem &Reg) {
  uint32_t pos = 0;
  switch (Reg.getBit()) {
  case 8:
    pos = 3;
    break;
  case 16:
    pos = 2;
    break;
  case 32:
    pos = 1;
    break;
  case 64:
    pos = 0;
    break;
  default:
    pos = 0;
  }
  return field(Reg.getElemIdx(), pos, pos);
}

uint32_t genSize(const Reg &Reg) {
  uint32_t size = 0;
  switch (Reg.getBit()) {
  case 8:
    size = 0;
    break;
  case 16:
    size = 1;
    break;
  case 32:
    size = 2;
    break;
  case 64:
    size = 3;
    break;
  default:
    size = 0;
  }
  return size;
}

uint32_t genSizeEnc(const VRegElem &Reg) {
  uint32_t size = 0;
  switch (Reg.getBit()) {
  case 8:
    size = field(Reg.getElemIdx(), 1, 0);
    break;
  case 16:
    size = field(Reg.getElemIdx(), 0, 0) << 1;
    break;
  case 32:
    size = 0;
    break;
  case 64:
    size = 1;
    break;
  default:
    size = 0;
  }
  return size;
}

uint32_t genSize(uint32_t dtype) {
  uint32_t size = (dtype == 0xf) ? 3 : (dtype == 0x4 || dtype == 0xa || dtype == 0xb) ? 2 : (5 <= dtype && dtype <= 9) ? 1 : 0;
  return size;
}

uint32_t genS(const VRegElem &Reg) {
  uint32_t s = 0;
  switch (Reg.getBit()) {
  case 8:
    s = field(Reg.getElemIdx(), 2, 2);
    break;
  case 16:
    s = field(Reg.getElemIdx(), 1, 1);
    break;
  case 32:
    s = field(Reg.getElemIdx(), 0, 0);
    break;
  case 64:
    s = 0;
    break;
  default:
    s = 0;
  }
  return s;
}

uint32_t CodeGenerator::genNImmrImms(uint64_t imm, uint32_t size) {
  // check imm
  if (imm == 0 || imm == ones(size)) {
    throw Error(ERR_ILLEGAL_IMM_VALUE);
  }

  auto ptn_size = getPtnSize(imm, size);
  auto ptn = imm & ones(ptn_size);
  auto rotate_num = getPtnRotateNum(ptn, ptn_size);
  auto rotate_ptn = lrotate(ptn, ptn_size, rotate_num);
  auto one_bit_num = countOneBit(rotate_ptn, ptn_size);
  auto seq_one_bit_num = countSeqOneBit(rotate_ptn, ptn_size);

  // check ptn
  if (one_bit_num != seq_one_bit_num) {
    throw Error(ERR_ILLEGAL_IMM_VALUE);
  }

  uint32_t N = (ptn_size > 32) ? 1 : 0;
  uint32_t immr = rotate_num;
  uint32_t imms = static_cast<uint32_t>((ones(6) & ~ones(static_cast<uint32_t>(std::log2(ptn_size)) + 1)) | (one_bit_num - 1));
  return (N << 12) | (immr << 6) | imms;
}

uint32_t CodeGenerator::PCrelAddrEnc(uint32_t op, const XReg &rd, int64_t labelOffset) {
  int32_t imm = static_cast<uint32_t>((op == 1) ? labelOffset >> 12 : labelOffset);
  uint32_t immlo = field(imm, 1, 0);
  uint32_t immhi = field(imm, 20, 2);
  verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(imm, -1 * (1 << 20), ones(20), ERR_ILLEGAL_IMM_RANGE, true);
  return concat({F(op, 31), F(immlo, 29), F(0x10, 24), F(immhi, 5), F(rd.getIdx(), 0)});
}

void CodeGenerator::PCrelAddr(uint32_t op, const XReg &rd, const Label &label) {
  auto encFunc = [&, op, rd](int64_t labelOffset) { return PCrelAddrEnc(op, rd, labelOffset); };
  JmpLabel jmpL = JmpLabel(encFunc, size_);
  uint32_t code = PCrelAddrEnc(op, rd, genLabelOffset(label, jmpL));
  dd(code);
}

void CodeGenerator::PCrelAddr(uint32_t op, const XReg &rd, int64_t label) {
  uint32_t code = PCrelAddrEnc(op, rd, label);
  dd(code);
}

// Add/subtract (immediate)
void CodeGenerator::AddSubImm(uint32_t op, uint32_t S, const RReg &rd, const RReg &rn, uint32_t imm, uint32_t sh) {
  uint32_t sf = genSf(rd);
  uint32_t imm12 = imm & ones(12);
  uint32_t sh_f = (sh == 12) ? 1 : 0;

  verifyIncRange(imm, 0, ones(12), ERR_ILLEGAL_IMM_RANGE);
  verifyIncList(sh, {0, 12}, ERR_ILLEGAL_CONST_VALUE);

  uint32_t code = concat({F(sf, 31), F(op, 30), F(S, 29), F(0x11, 24), F(sh_f, 22), F(imm12, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Logical (immediate)
void CodeGenerator::LogicalImm(uint32_t opc, const RReg &rd, const RReg &rn, uint64_t imm, bool alias) {
  uint32_t sf = genSf(rd);
  uint32_t n_immr_imms = genNImmrImms(imm, rd.getBit());

  if (!alias && opc == 3)
    verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  if (!alias && opc == 1)
    verifyIncRange(rn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(sf, 31), F(opc, 29), F(0x24, 23), F(n_immr_imms, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Move wide(immediate)
void CodeGenerator::MvWideImm(uint32_t opc, const RReg &rd, uint32_t imm, uint32_t sh) {
  uint32_t sf = genSf(rd);
  uint32_t hw = field(sh, 5, 4);
  uint32_t imm16 = imm & 0xffff;

  if (sf == 0)
    verifyIncList(sh, {0, 16}, ERR_ILLEGAL_CONST_VALUE);
  else
    verifyIncList(sh, {0, 16, 32, 48}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(imm, 0, ones(16), ERR_ILLEGAL_IMM_RANGE);
  verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(sf, 31), F(opc, 29), F(0x25, 23), F(hw, 21), F(imm16, 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Move (immediate) alias of ORR,MOVN,MOVZ
void CodeGenerator::MvImm(const RReg &rd, uint64_t imm) {
  uint32_t rd_bit = rd.getBit();
  uint32_t hw = 0;
  uint32_t inv_hw = 0;
  uint32_t validField[4] = {0};
  uint32_t imm16 = 0;
  uint32_t inv_imm16 = 0;
  uint32_t fieldCount = 0;
  uint32_t invFieldCount = 0;

  if (imm == 0) {
    MvWideImm(2, rd, 0, 0);
    return;
  }

  if ((rd_bit == 64 && imm == ~uint64_t(0)) || (rd_bit == 32 && ((imm & uint64_t(0xffffffff)) == uint64_t(0xffffffff)))) {
    MvWideImm(0, rd, 0, 0);
    return;
  }

  /***** MOVZ *****/
  /* Count how many valid 16-bit field exists. */
  for (uint32_t i = 0; i < rd_bit / 16; ++i) {
    if (field(imm, 15 + i * 16, i * 16)) {
      validField[i] = 1;
      ++fieldCount;
      hw = i;
      imm16 = field(imm, 15 + 16 * i, 16 * i);
    }
  }
  if (fieldCount < 2) {
    if (!(imm16 == 0 && hw != 0)) {
      /* alias of MOVZ
         which set 16-bit immediate, bit position is indicated by (hw * 4). */
      MvWideImm(2, rd, imm16, hw << 4);
      return;
    }
  }

  /***** MOVN *****/
  /* Count how many valid 16-bit field exists. */
  for (uint32_t i = 0; i < rd_bit / 16; ++i) {
    if (field(~imm, 15 + i * 16, i * 16)) {
      ++invFieldCount;
      inv_imm16 = field(~imm, 15 + 16 * i, 16 * i);
      inv_hw = i;
    }
  }
  if (invFieldCount == 1) {
    if ((!(inv_imm16 == 0 && inv_hw != 0) && inv_imm16 != ones(16) && rd_bit == 32) || (!(inv_imm16 == 0 && inv_hw != 0) && rd_bit == 64)) {
      /* alias of MOVN
         which firstly, set 16-bit immediate, bit position is indicated by (hw
         * 4) then, result is inverted (NOT). */
      MvWideImm(0, rd, inv_imm16, inv_hw << 4);
      return;
    }
  }

  /***** ORR *****/
  auto ptn_size = getPtnSize(imm, rd_bit);
  auto ptn = imm & ones(ptn_size);
  auto rotate_num = getPtnRotateNum(ptn, ptn_size);
  auto rotate_ptn = lrotate(ptn, ptn_size, rotate_num);
  auto one_bit_num = countOneBit(rotate_ptn, ptn_size);
  auto seq_one_bit_num = countSeqOneBit(rotate_ptn, ptn_size);
  if (one_bit_num == seq_one_bit_num) {
    // alias of ORR
    LogicalImm(1, rd, RReg(31, rd_bit), imm, true);
    return;
  }

  /**** MOVZ followed by successive MOVK *****/
  bool isFirst = true;
  for (uint32_t i = 0; i < rd_bit / 16; ++i) {
    if (validField[i]) {
      if (isFirst) {
        MvWideImm(2, rd, field(imm, 15 + 16 * i, 16 * i), 16 * i);
        isFirst = false;
      } else {
        MvWideImm(3, rd, field(imm, 15 + 16 * i, 16 * i), 16 * i);
      }
    }
  }
}

// Bitfield
void CodeGenerator::Bitfield(uint32_t opc, const RReg &rd, const RReg &rn, uint32_t immr, uint32_t imms, bool rn_chk) {
  uint32_t sf = genSf(rd);
  uint32_t N = sf;

  verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  if (rn_chk)
    verifyIncRange(rn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(immr, 0, rd.getBit() - 1, ERR_ILLEGAL_IMM_RANGE);
  verifyIncRange(imms, 0, rd.getBit() - 1, ERR_ILLEGAL_IMM_RANGE);

  uint32_t code = concat({F(sf, 31), F(opc, 29), F(0x26, 23), F(N, 22), F(immr, 16), F(imms, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Extract
void CodeGenerator::Extract(uint32_t op21, uint32_t o0, const RReg &rd, const RReg &rn, const RReg &rm, uint32_t imm) {
  uint32_t sf = genSf(rd);
  uint32_t N = sf;

  verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(rn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(rm.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(imm, 0, rd.getBit() - 1, ERR_ILLEGAL_IMM_RANGE);

  uint32_t code = concat({F(sf, 31), F(op21, 29), F(0x27, 23), F(N, 22), F(o0, 21), F(rm.getIdx(), 16), F(imm, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Conditional branch (immediate)
uint32_t CodeGenerator::CondBrImmEnc(uint32_t cond, int64_t labelOffset) {
  uint32_t imm19 = static_cast<uint32_t>((labelOffset >> 2) & ones(19));
  verifyIncRange(labelOffset, -1 * (1 << 20), ones(20), ERR_LABEL_IS_TOO_FAR, true);
  return concat({F(0x2a, 25), F(imm19, 5), F(cond, 0)});
}

void CodeGenerator::CondBrImm(Cond cond, const Label &label) {
  auto encFunc = [&, cond](int64_t labelOffset) { return CondBrImmEnc(cond, labelOffset); };
  JmpLabel jmpL = JmpLabel(encFunc, size_);
  uint32_t code = CondBrImmEnc(cond, genLabelOffset(label, jmpL));
  dd(code);
}

void CodeGenerator::CondBrImm(Cond cond, int64_t label) {
  uint32_t code = CondBrImmEnc(cond, label);
  dd(code);
}

// Exception generation
void CodeGenerator::ExceptionGen(uint32_t opc, uint32_t op2, uint32_t LL, uint32_t imm) {
  uint32_t imm16 = imm & ones(16);
  verifyIncRange(imm, 0, ones(16), ERR_ILLEGAL_IMM_RANGE);
  uint32_t code = concat({F(0xd4, 24), F(opc, 21), F(imm16, 5), F(op2, 2), F(LL, 0)});
  dd(code);
}

// Hints
void CodeGenerator::Hints(uint32_t CRm, uint32_t op2) {
  uint32_t code = concat({F(0xd5032, 12), F(CRm, 8), F(op2, 5), F(0x1f, 0)});
  dd(code);
}

void CodeGenerator::Hints(uint32_t imm) { Hints(field(imm, 6, 3), field(imm, 2, 0)); }

// Barriers (option)
void CodeGenerator::BarriersOpt(uint32_t op2, BarOpt opt, uint32_t rt) {
  if (op2 == 6)
    verifyIncList(opt, {SY}, ERR_ILLEGAL_BARRIER_OPT);
  uint32_t code = concat({F(0xd5033, 12), F(opt, 8), F(op2, 5), F(rt, 0)});
  dd(code);
}

// Barriers (no option)
void CodeGenerator::BarriersNoOpt(uint32_t CRm, uint32_t op2, uint32_t rt) {
  verifyIncRange(CRm, 0, ones(4), ERR_ILLEGAL_IMM_RANGE);
  uint32_t code = concat({F(0xd5033, 12), F(CRm, 8), F(op2, 5), F(rt, 0)});
  dd(code);
}

// pstate
void CodeGenerator::PState(PStateField psfield, uint32_t imm) {
  uint32_t CRm = imm & ones(4);
  uint32_t op1, op2;
  switch (psfield) {
  case SPSel:
    op1 = 0;
    op2 = 5;
    break;
  case DAIFSet:
    op1 = 3;
    op2 = 6;
    break;
  case DAIFClr:
    op1 = 3;
    op2 = 7;
    break;
  case UAO:
    op1 = 0;
    op2 = 3;
    break;
  case PAN:
    op1 = 0;
    op2 = 4;
    break;
  case DIT:
    op1 = 3;
    op2 = 2;
    break;
  default:
    op1 = 0;
    op2 = 0;
  }
  uint32_t code = concat({F(0xd5, 24), F(op1, 16), F(0x4, 12), F(CRm, 8), F(op2, 5), F(0x1f, 0)});
  dd(code);
}

void CodeGenerator::PState(uint32_t op1, uint32_t CRm, uint32_t op2) {
  uint32_t code = concat({F(0xd5, 24), F(op1, 16), F(0x4, 12), F(CRm, 8), F(op2, 5), F(0x1f, 0)});
  dd(code);
}

// Systtem instructions
void CodeGenerator::SysInst(uint32_t L, uint32_t op1, uint32_t CRn, uint32_t CRm, uint32_t op2, const XReg &rt) {
  uint32_t code = concat({F(0xd5, 24), F(L, 21), F(1, 19), F(op1, 16), F(CRn, 12), F(CRm, 8), F(op2, 5), F(rt.getIdx(), 0)});
  dd(code);
}

// System register move
void CodeGenerator::SysRegMove(uint32_t L, uint32_t op0, uint32_t op1, uint32_t CRn, uint32_t CRm, uint32_t op2, const XReg &rt) {
  uint32_t code = concat({F(0xd5, 24), F(L, 21), F(1, 20), F(op0, 19), F(op1, 16), F(CRn, 12), F(CRm, 8), F(op2, 5), F(rt.getIdx(), 0)});
  dd(code);
}

// Unconditional branch
void CodeGenerator::UncondBrNoReg(uint32_t opc, uint32_t op2, uint32_t op3, uint32_t rn, uint32_t op4) {
  uint32_t code = concat({F(0x6b, 25), F(opc, 21), F(op2, 16), F(op3, 10), F(rn, 5), F(op4, 0)});
  dd(code);
}

void CodeGenerator::UncondBr1Reg(uint32_t opc, uint32_t op2, uint32_t op3, const RReg &rn, uint32_t op4) {
  uint32_t code = concat({F(0x6b, 25), F(opc, 21), F(op2, 16), F(op3, 10), F(rn.getIdx(), 5), F(op4, 0)});
  dd(code);
}

void CodeGenerator::UncondBr2Reg(uint32_t opc, uint32_t op2, uint32_t op3, const RReg &rn, const RReg &rm) {
  uint32_t code = concat({F(0x6b, 25), F(opc, 21), F(op2, 16), F(op3, 10), F(rn.getIdx(), 5), F(rm.getIdx(), 0)});
  dd(code);
}

// Unconditional branch (immediate)
uint32_t CodeGenerator::UncondBrImmEnc(uint32_t op, int64_t labelOffset) {
  verifyIncRange(labelOffset, -1 * (1 << 27), ones(27), ERR_LABEL_IS_TOO_FAR, true);
  uint32_t imm26 = static_cast<uint32_t>((labelOffset >> 2) & ones(26));
  return concat({F(op, 31), F(5, 26), F(imm26, 0)});
}

void CodeGenerator::UncondBrImm(uint32_t op, const Label &label) {
  auto encFunc = [&, op](int64_t labelOffset) { return UncondBrImmEnc(op, labelOffset); };
  JmpLabel jmpL = JmpLabel(encFunc, size_);
  uint32_t code = UncondBrImmEnc(op, genLabelOffset(label, jmpL));
  dd(code);
}

void CodeGenerator::UncondBrImm(uint32_t op, int64_t label) {
  uint32_t code = UncondBrImmEnc(op, label);
  dd(code);
}

// Compare and branch (immediate)
uint32_t CodeGenerator::CompareBrEnc(uint32_t op, const RReg &rt, int64_t labelOffset) {
  verifyIncRange(labelOffset, -1 * (1 << 20), ones(20), ERR_LABEL_IS_TOO_FAR, true);

  uint32_t sf = genSf(rt);
  uint32_t imm19 = (static_cast<uint32_t>(labelOffset >> 2)) & ones(19);
  return concat({F(sf, 31), F(0x1a, 25), F(op, 24), F(imm19, 5), F(rt.getIdx(), 0)});
}

void CodeGenerator::CompareBr(uint32_t op, const RReg &rt, const Label &label) {
  auto encFunc = [&, op](int64_t labelOffset) { return CompareBrEnc(op, rt, labelOffset); };
  JmpLabel jmpL = JmpLabel(encFunc, size_);
  uint32_t code = CompareBrEnc(op, rt, genLabelOffset(label, jmpL));
  dd(code);
}

void CodeGenerator::CompareBr(uint32_t op, const RReg &rt, int64_t label) {
  uint32_t code = CompareBrEnc(op, rt, label);
  dd(code);
}

// Test and branch (immediate)
uint32_t CodeGenerator::TestBrEnc(uint32_t op, const RReg &rt, uint32_t imm, int64_t labelOffset) {
  verifyIncRange(labelOffset, -1 * (1 << 15), ones(15), ERR_LABEL_IS_TOO_FAR, true);
  verifyIncRange(imm, 0, ones(6), ERR_ILLEGAL_IMM_RANGE);

  uint32_t b5 = field(imm, 5, 5);
  uint32_t b40 = field(imm, 4, 0);
  uint32_t imm14 = (static_cast<uint32_t>(labelOffset >> 2)) & ones(14);

  if (b5 == 1)
    verifyIncList(rt.getBit(), {64}, ERR_ILLEGAL_IMM_VALUE);

  return concat({F(b5, 31), F(0x1b, 25), F(op, 24), F(b40, 19), F(imm14, 5), F(rt.getIdx(), 0)});
}

void CodeGenerator::TestBr(uint32_t op, const RReg &rt, uint32_t imm, const Label &label) {
  auto encFunc = [&, op, rt, imm](int64_t labelOffset) { return TestBrEnc(op, rt, imm, labelOffset); };
  JmpLabel jmpL = JmpLabel(encFunc, size_);
  uint32_t code = TestBrEnc(op, rt, imm, genLabelOffset(label, jmpL));
  dd(code);
}

void CodeGenerator::TestBr(uint32_t op, const RReg &rt, uint32_t imm, int64_t label) {
  uint32_t code = TestBrEnc(op, rt, imm, label);
  dd(code);
}

// Advanced SIMD load/store multipule structure
void CodeGenerator::AdvSimdLdStMultiStructExceptLd1St1(uint32_t L, uint32_t opc, const VRegList &vt, const AdrNoOfs &adr) {
  uint32_t Q = genQ(vt);
  uint32_t size = genSize(vt);
  uint32_t len = vt.getLen();

  verifyIncRange(len, 1, 4, ERR_ILLEGAL_REG_IDX);

  opc = (opc == 0x2 && len == 1) ? 0x7 : (opc == 0x2 && len == 2) ? 0xa : (opc == 0x2 && len == 3) ? 0x6 : opc;
  uint32_t code = concat({F(Q, 30), F(0x18, 23), F(L, 22), F(opc, 12), F(size, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::AdvSimdLdStMultiStructForLd1St1(uint32_t L, uint32_t opc, const VRegList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(L, opc, vt, adr); }

// Advanced SIMD load/store multple structures (post-indexed register offset)
void CodeGenerator::AdvSimdLdStMultiStructPostRegExceptLd1St1(uint32_t L, uint32_t opc, const VRegList &vt, const AdrPostReg &adr) {
  uint32_t Q = genQ(vt);
  uint32_t size = genSize(vt);

  verifyIncRange(adr.getXm().getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t len = vt.getLen();
  verifyIncRange(len, 1, 4, ERR_ILLEGAL_REG_IDX);
  opc = (opc == 0x2 && len == 1) ? 0x7 : (opc == 0x2 && len == 2) ? 0xa : (opc == 0x2 && len == 3) ? 0x6 : opc;
  uint32_t code = concat({F(Q, 30), F(0x19, 23), F(L, 22), F(adr.getXm().getIdx(), 16), F(opc, 12), F(size, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::AdvSimdLdStMultiStructPostRegForLd1St1(uint32_t L, uint32_t opc, const VRegList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(L, opc, vt, adr); }

// Advanced SIMD load/store multple structures (post-indexed immediate offset)
void CodeGenerator::AdvSimdLdStMultiStructPostImmExceptLd1St1(uint32_t L, uint32_t opc, const VRegList &vt, const AdrPostImm &adr) {
  uint32_t Q = genQ(vt);
  uint32_t size = genSize(vt);
  uint32_t len = vt.getLen();

  verifyIncRange(adr.getImm(), 0, ((8 * len) << Q), ERR_ILLEGAL_IMM_RANGE);
  verifyIncRange(len, 1, 4, ERR_ILLEGAL_REG_IDX);

  opc = (opc == 0x2 && len == 1) ? 0x7 : (opc == 0x2 && len == 2) ? 0xa : (opc == 0x2 && len == 3) ? 0x6 : opc;
  uint32_t code = concat({F(Q, 30), F(0x19, 23), F(L, 22), F(0x1f, 16), F(opc, 12), F(size, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD load/store multple structures (post-indexed immediate offset)
void CodeGenerator::AdvSimdLdStMultiStructPostImmForLd1St1(uint32_t L, uint32_t opc, const VRegList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(L, opc, vt, adr); }

// Advanced SIMD load/store single structures
void CodeGenerator::AdvSimdLdStSingleStruct(uint32_t L, uint32_t R, uint32_t num, const VRegElem &vt, const AdrNoOfs &adr) {
  uint32_t Q = genQ(vt);
  uint32_t S = genS(vt);
  uint32_t size = genSizeEnc(vt);
  uint32_t opc = (vt.getBit() == 8) ? field(num - 1, 1, 1) : (vt.getBit() == 16) ? field(num - 1, 1, 1) + 2 : field(num - 1, 1, 1) + 4;
  uint32_t code = concat({F(Q, 30), F(0x1a, 23), F(L, 22), F(R, 21), F(opc, 13), F(S, 12), F(size, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD load replication single structures
void CodeGenerator::AdvSimdLdRepSingleStruct(uint32_t L, uint32_t R, uint32_t opcode, uint32_t S, const VRegVec &vt, const AdrNoOfs &adr) {
  uint32_t Q = genQ(vt);
  uint32_t size = genSize(vt);
  uint32_t code = concat({F(Q, 30), F(0x1a, 23), F(L, 22), F(R, 21), F(opcode, 13), F(S, 12), F(size, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD load/store single structures (post-indexed register)
void CodeGenerator::AdvSimdLdStSingleStructPostReg(uint32_t L, uint32_t R, uint32_t num, const VRegElem &vt, const AdrPostReg &adr) {
  uint32_t Q = genQ(vt);
  uint32_t S = genS(vt);
  uint32_t size = genSizeEnc(vt);
  uint32_t opc = (vt.getBit() == 8) ? field(num - 1, 1, 1) : (vt.getBit() == 16) ? field(num - 1, 1, 1) + 2 : field(num - 1, 1, 1) + 4;
  uint32_t code = concat({F(Q, 30), F(0x1b, 23), F(L, 22), F(R, 21), F(adr.getXm().getIdx(), 16), F(opc, 13), F(S, 12), F(size, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD load/store single structures (post-indexed register,
// replicate)
void CodeGenerator::AdvSimdLdStSingleStructRepPostReg(uint32_t L, uint32_t R, uint32_t opcode, uint32_t S, const VRegVec &vt, const AdrPostReg &adr) {
  uint32_t Q = genQ(vt);
  uint32_t size = genSize(vt);
  uint32_t code = concat({F(Q, 30), F(0x1b, 23), F(L, 22), F(R, 21), F(adr.getXm().getIdx(), 16), F(opcode, 13), F(S, 12), F(size, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD load/store single structures (post-indexed immediate)
void CodeGenerator::AdvSimdLdStSingleStructPostImm(uint32_t L, uint32_t R, uint32_t num, const VRegElem &vt, const AdrPostImm &adr) {
  uint32_t Q = genQ(vt);
  uint32_t S = genS(vt);
  uint32_t size = genSizeEnc(vt);
  uint32_t opc = (vt.getBit() == 8) ? field(num - 1, 1, 1) : (vt.getBit() == 16) ? field(num - 1, 1, 1) + 2 : field(num - 1, 1, 1) + 4;

  verifyIncList(adr.getImm(), {num * vt.getBit() / 8}, ERR_ILLEGAL_IMM_VALUE);

  uint32_t code = concat({F(Q, 30), F(0x1b, 23), F(L, 22), F(R, 21), F(0x1f, 16), F(opc, 13), F(S, 12), F(size, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD load replication single structures (post-indexed immediate)
void CodeGenerator::AdvSimdLdRepSingleStructPostImm(uint32_t L, uint32_t R, uint32_t opcode, uint32_t S, const VRegVec &vt, const AdrPostImm &adr) {
  uint32_t Q = genQ(vt);
  uint32_t size = genSize(vt);
  uint32_t len = (field(opcode, 0, 0) << 1) + R + 1;

  verifyIncList(adr.getImm(), {len << size}, ERR_ILLEGAL_IMM_VALUE);

  uint32_t code = concat({F(Q, 30), F(0x1b, 23), F(L, 22), F(R, 21), F(0x1f, 16), F(opcode, 13), F(S, 12), F(size, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

// store exclusive
void CodeGenerator::StExclusive(uint32_t size, uint32_t o0, const WReg ws, const RReg &rt, const AdrImm &adr) {
  uint32_t L = 0;
  uint32_t o2 = 0;
  uint32_t o1 = 0;

  verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);
  verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(ws.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(size, 30), F(0x8, 24), F(o2, 23), F(L, 22), F(o1, 21), F(ws.getIdx(), 16), F(o0, 15), F(0x1f, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// load exclusive
void CodeGenerator::LdExclusive(uint32_t size, uint32_t o0, const RReg &rt, const AdrImm &adr) {
  uint32_t L = 1;
  uint32_t o2 = 0;
  uint32_t o1 = 0;

  verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);

  uint32_t code = concat({F(size, 30), F(0x8, 24), F(o2, 23), F(L, 22), F(o1, 21), F(0x1f, 16), F(o0, 15), F(0x1f, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// store LORelease
void CodeGenerator::StLORelase(uint32_t size, uint32_t o0, const RReg &rt, const AdrImm &adr) {
  uint32_t L = 0;
  uint32_t o2 = 1;
  uint32_t o1 = 0;

  verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);
  verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(size, 30), F(0x8, 24), F(o2, 23), F(L, 22), F(o1, 21), F(0x1f, 16), F(o0, 15), F(0x1f, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// load LOAcquire
void CodeGenerator::LdLOAcquire(uint32_t size, uint32_t o0, const RReg &rt, const AdrImm &adr) {
  uint32_t L = 1;
  uint32_t o2 = 1;
  uint32_t o1 = 0;

  verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);
  verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(size, 30), F(0x8, 24), F(o2, 23), F(L, 22), F(o1, 21), F(0x1f, 16), F(o0, 15), F(0x1f, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// compare and swap
void CodeGenerator::Cas(uint32_t size, uint32_t o2, uint32_t L, uint32_t o1, uint32_t o0, const RReg &rs, const RReg &rt, const AdrNoOfs &adr) {
  verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(rs.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(size, 30), F(0x8, 24), F(o2, 23), F(L, 22), F(o1, 21), F(rs.getIdx(), 16), F(o0, 15), F(0x1f, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// load/store exclusive pair
void CodeGenerator::StExclusivePair(uint32_t L, uint32_t o1, uint32_t o0, const WReg &ws, const RReg &rt1, const RReg &rt2, const AdrImm &adr) {
  uint32_t sz = (rt1.getBit() == 64) ? 1 : 0;

  verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);
  verifyIncRange(rt1.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(rt2.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(ws.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(1, 31), F(sz, 30), F(0x8, 24), F(0, 23), F(L, 22), F(o1, 21), F(ws.getIdx(), 16), F(o0, 15), F(rt2.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(rt1.getIdx(), 0)});
  dd(code);
}

// load/store exclusive pair
void CodeGenerator::LdExclusivePair(uint32_t L, uint32_t o1, uint32_t o0, const RReg &rt1, const RReg &rt2, const AdrImm &adr) {
  uint32_t sz = (rt1.getBit() == 64) ? 1 : 0;

  verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);
  verifyIncRange(rt1.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(rt2.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(1, 31), F(sz, 30), F(0x8, 24), F(0, 23), F(L, 22), F(o1, 21), F(0x1f, 16), F(o0, 15), F(rt2.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(rt1.getIdx(), 0)});
  dd(code);
}

// compare and swap pair
void CodeGenerator::CasPair(uint32_t L, uint32_t o1, uint32_t o0, const RReg &rs, const RReg &rt, const AdrNoOfs &adr) {
  uint32_t sz = (rt.getBit() == 64) ? 1 : 0;

  verifyIncRange(rs.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(0, 31), F(sz, 30), F(0x8, 24), F(0, 23), F(L, 22), F(o1, 21), F(rs.getIdx(), 16), F(o0, 15), F(0x1f, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// LDAPR/STLR (unscaled immediate)
void CodeGenerator::LdaprStlr(uint32_t size, uint32_t opc, const RReg &rt, const AdrImm &adr) {
  int32_t simm = adr.getImm();
  uint32_t imm9 = simm & ones(9);

  verifyIncRange(simm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);
  verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(size, 30), F(0x19, 24), F(opc, 22), F(imm9, 12), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// load register (literal)
uint32_t CodeGenerator::LdRegLiteralEnc(uint32_t opc, uint32_t V, const RReg &rt, int64_t labelOffset) {
  verifyIncRange(labelOffset, (-1) * (1 << 20), ones(20), ERR_LABEL_IS_TOO_FAR, true);

  uint32_t imm19 = (static_cast<uint32_t>(labelOffset >> 2)) & ones(19);
  return concat({F(opc, 30), F(0x3, 27), F(V, 26), F(imm19, 5), F(rt.getIdx(), 0)});
}

void CodeGenerator::LdRegLiteral(uint32_t opc, uint32_t V, const RReg &rt, const Label &label) {
  auto encFunc = [&, opc, V, rt](int64_t labelOffset) { return LdRegLiteralEnc(opc, V, rt, labelOffset); };
  JmpLabel jmpL = JmpLabel(encFunc, size_);
  uint32_t code = LdRegLiteralEnc(opc, V, rt, genLabelOffset(label, jmpL));
  dd(code);
}

void CodeGenerator::LdRegLiteral(uint32_t opc, uint32_t V, const RReg &rt, int64_t label) {
  uint32_t code = LdRegLiteralEnc(opc, V, rt, label);
  dd(code);
}

// load register (SIMD&FP, literal)
uint32_t CodeGenerator::LdRegSimdFpLiteralEnc(const VRegSc &vt, int64_t labelOffset) {
  verifyIncRange(labelOffset, -1 * (1 << 20), ones(20), ERR_LABEL_IS_TOO_FAR, true);

  uint32_t opc = (vt.getBit() == 32) ? 0 : (vt.getBit() == 64) ? 1 : 2;
  uint32_t imm19 = (static_cast<uint32_t>(labelOffset >> 2)) & ones(19);
  uint32_t V = 1;
  return concat({F(opc, 30), F(0x3, 27), F(V, 26), F(imm19, 5), F(vt.getIdx(), 0)});
}

void CodeGenerator::LdRegSimdFpLiteral(const VRegSc &vt, const Label &label) {
  auto encFunc = [&, vt](int64_t labelOffset) { return LdRegSimdFpLiteralEnc(vt, labelOffset); };
  JmpLabel jmpL = JmpLabel(encFunc, size_);
  uint32_t code = LdRegSimdFpLiteralEnc(vt, genLabelOffset(label, jmpL));
  dd(code);
}

void CodeGenerator::LdRegSimdFpLiteral(const VRegSc &vt, int64_t label) {
  uint32_t code = LdRegSimdFpLiteralEnc(vt, label);
  dd(code);
}

// prefetch (literal)
uint32_t CodeGenerator::PfLiteralEnc(Prfop prfop, int64_t labelOffset) {
  verifyIncRange(labelOffset, -1 * (1 << 20), ones(20), ERR_LABEL_IS_TOO_FAR, true);

  uint32_t opc = 3;
  uint32_t imm19 = (static_cast<uint32_t>(labelOffset >> 2)) & ones(19);
  uint32_t V = 0;
  return concat({F(opc, 30), F(0x3, 27), F(V, 26), F(imm19, 5), F(prfop, 0)});
}

void CodeGenerator::PfLiteral(Prfop prfop, const Label &label) {
  auto encFunc = [&, prfop](int64_t labelOffset) { return PfLiteralEnc(prfop, labelOffset); };
  JmpLabel jmpL = JmpLabel(encFunc, size_);
  uint32_t code = PfLiteralEnc(prfop, genLabelOffset(label, jmpL));
  dd(code);
}

void CodeGenerator::PfLiteral(Prfop prfop, int64_t label) {
  uint32_t code = PfLiteralEnc(prfop, label);
  dd(code);
}

// Load/store no-allocate pair (offset)
void CodeGenerator::LdStNoAllocPair(uint32_t L, const RReg &rt1, const RReg &rt2, const AdrImm &adr) {
  int32_t imm = adr.getImm();
  int32_t times = (rt1.getBit() == 32) ? 1 : 2;

  verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE, true);
  verifyCond(
      imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); }, ERR_ILLEGAL_IMM_COND);

  uint32_t opc = (rt1.getBit() == 32) ? 0 : 2;
  uint32_t imm7 = (imm >> (times + 1)) & ones(7);
  uint32_t V = 0;

  verifyIncRange(rt1.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(rt2.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(L, 22), F(imm7, 15), F(rt2.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(rt1.getIdx(), 0)});
  dd(code);
}

// Load/store no-allocate pair (offset)
void CodeGenerator::LdStSimdFpNoAllocPair(uint32_t L, const VRegSc &vt1, const VRegSc &vt2, const AdrImm &adr) {
  int32_t imm = adr.getImm();
  int32_t times = vt1.getBit() / 32;

  verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE, true);
  verifyCond(
      imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); }, ERR_ILLEGAL_IMM_COND);

  uint32_t opc = (vt1.getBit() == 32) ? 0 : (vt1.getBit() == 64) ? 1 : 2;
  uint32_t sh = static_cast<uint32_t>(std::log2(4 * times));
  uint32_t imm7 = (imm >> sh) & ones(7);
  uint32_t V = 1;
  uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(L, 22), F(imm7, 15), F(vt2.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(vt1.getIdx(), 0)});
  dd(code);
}

// Load/store pair (post-indexed)
void CodeGenerator::LdStRegPairPostImm(uint32_t opc, uint32_t L, const RReg &rt1, const RReg &rt2, const AdrPostImm &adr) {
  int32_t imm = adr.getImm();
  int32_t times = (opc == 2) ? 2 : 1;

  verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE, true);
  verifyCond(
      imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); }, ERR_ILLEGAL_IMM_COND);

  uint32_t imm7 = (imm >> (times + 1)) & ones(7);
  uint32_t V = 0;

  verifyIncRange(rt1.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(rt2.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(1, 23), F(L, 22), F(imm7, 15), F(rt2.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(rt1.getIdx(), 0)});
  dd(code);
}

// Load/store pair (post-indexed)
void CodeGenerator::LdStSimdFpPairPostImm(uint32_t L, const VRegSc &vt1, const VRegSc &vt2, const AdrPostImm &adr) {
  int32_t imm = adr.getImm();
  int32_t times = vt1.getBit() / 32;

  verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE, true);
  verifyCond(
      imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); }, ERR_ILLEGAL_IMM_COND);

  uint32_t opc = (vt1.getBit() == 32) ? 0 : (vt1.getBit() == 64) ? 1 : 2;
  uint32_t sh = static_cast<uint32_t>(std::log2(4 * times));
  uint32_t imm7 = (imm >> sh) & ones(7);
  uint32_t V = 1;
  uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(1, 23), F(L, 22), F(imm7, 15), F(vt2.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(vt1.getIdx(), 0)});
  dd(code);
}

// Load/store pair (offset)
void CodeGenerator::LdStRegPair(uint32_t opc, uint32_t L, const RReg &rt1, const RReg &rt2, const AdrImm &adr) {
  int32_t imm = adr.getImm();
  int32_t times = (opc == 2) ? 2 : 1;

  verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE, true);
  verifyCond(
      imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); }, ERR_ILLEGAL_IMM_COND);

  uint32_t imm7 = (imm >> (times + 1)) & ones(7);
  uint32_t V = 0;

  verifyIncRange(rt1.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(rt2.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(2, 23), F(L, 22), F(imm7, 15), F(rt2.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(rt1.getIdx(), 0)});
  dd(code);
}

// Load/store pair (offset)
void CodeGenerator::LdStSimdFpPair(uint32_t L, const VRegSc &vt1, const VRegSc &vt2, const AdrImm &adr) {
  int32_t imm = adr.getImm();
  int32_t times = vt1.getBit() / 32;

  verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE, true);
  verifyCond(
      imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); }, ERR_ILLEGAL_IMM_COND);

  uint32_t opc = (vt1.getBit() == 32) ? 0 : (vt1.getBit() == 64) ? 1 : 2;
  uint32_t sh = static_cast<uint32_t>(std::log2(4 * times));
  uint32_t imm7 = (imm >> sh) & ones(7);
  uint32_t V = 1;
  uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(2, 23), F(L, 22), F(imm7, 15), F(vt2.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(vt1.getIdx(), 0)});
  dd(code);
}

// Load/store pair (pre-indexed)
void CodeGenerator::LdStRegPairPre(uint32_t opc, uint32_t L, const RReg &rt1, const RReg &rt2, const AdrPreImm &adr) {
  int32_t imm = adr.getImm();
  int32_t times = (opc == 2) ? 2 : 1;

  verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE, true);
  verifyCond(
      imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); }, ERR_ILLEGAL_IMM_COND);

  uint32_t imm7 = (imm >> (times + 1)) & ones(7);
  uint32_t V = 0;

  verifyIncRange(rt1.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(rt2.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(3, 23), F(L, 22), F(imm7, 15), F(rt2.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(rt1.getIdx(), 0)});
  dd(code);
}

// Load/store pair (pre-indexed)
void CodeGenerator::LdStSimdFpPairPre(uint32_t L, const VRegSc &vt1, const VRegSc &vt2, const AdrPreImm &adr) {
  int32_t imm = adr.getImm();
  int32_t times = vt1.getBit() / 32;

  verifyIncRange(imm, (-256 * times), (252 * times), ERR_ILLEGAL_IMM_RANGE, true);
  verifyCond(
      imm, [=](uint64_t x) { return ((x % (4 * times)) == 0); }, ERR_ILLEGAL_IMM_COND);

  uint32_t opc = (vt1.getBit() == 32) ? 0 : (vt1.getBit() == 64) ? 1 : 2;
  uint32_t sh = static_cast<uint32_t>(std::log2(4 * times));
  uint32_t imm7 = (imm >> sh) & ones(7);
  uint32_t V = 1;
  uint32_t code = concat({F(opc, 30), F(0x5, 27), F(V, 26), F(3, 23), F(L, 22), F(imm7, 15), F(vt2.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(vt1.getIdx(), 0)});
  dd(code);
}

// Load/store register (unscaled immediate)
void CodeGenerator::LdStRegUnsImm(uint32_t size, uint32_t opc, const RReg &rt, const AdrImm &adr) {
  int imm = adr.getImm();
  uint32_t imm9 = imm & ones(9);

  verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);
  verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t V = 0;
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// Load/store register (SIMD&FP, unscaled immediate)
void CodeGenerator::LdStSimdFpRegUnsImm(uint32_t opc, const VRegSc &vt, const AdrImm &adr) {
  uint32_t size = (vt.getBit() == 16) ? 1 : (vt.getBit() == 32) ? 2 : (vt.getBit() == 64) ? 3 : 0;

  int imm = adr.getImm();
  uint32_t imm9 = adr.getImm() & ones(9);

  verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t V = 1;
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

// prefetch register (unscaled immediate)
void CodeGenerator::PfRegUnsImm(Prfop prfop, const AdrImm &adr) {
  uint32_t size = 3;
  uint32_t opc = 2;

  int imm = adr.getImm();
  uint32_t imm9 = imm & ones(9);

  verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t V = 0;
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12), F(adr.getXn().getIdx(), 5), F(prfop, 0)});
  dd(code);
}

// Load/store register (immediate post-indexed)
void CodeGenerator::LdStRegPostImm(uint32_t size, uint32_t opc, const RReg &rt, const AdrPostImm &adr) {
  int imm = adr.getImm();
  uint32_t imm9 = imm & ones(9);

  verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);
  verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t V = 0;
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12), F(1, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// Load/store register (SIMD&FP, immediate post-indexed)
void CodeGenerator::LdStSimdFpRegPostImm(uint32_t opc, const VRegSc &vt, const AdrPostImm &adr) {
  uint32_t size = (vt.getBit() == 16) ? 1 : (vt.getBit() == 32) ? 2 : (vt.getBit() == 64) ? 3 : 0;

  int imm = adr.getImm();
  uint32_t imm9 = imm & ones(9);

  verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t V = 1;
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12), F(1, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

// Load/store register (unprivileged)
void CodeGenerator::LdStRegUnpriv(uint32_t size, uint32_t opc, const RReg &rt, const AdrImm &adr) {
  int imm = adr.getImm();
  uint32_t imm9 = imm & ones(9);

  verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);
  verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t V = 0;
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12), F(2, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// Load/store register (immediate pre-indexed)
void CodeGenerator::LdStRegPre(uint32_t size, uint32_t opc, const RReg &rt, const AdrPreImm &adr) {
  int imm = adr.getImm();
  uint32_t imm9 = imm & ones(9);

  verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);
  verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t V = 0;
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12), F(3, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// Load/store register (SIMD&FP, immediate pre-indexed)
void CodeGenerator::LdStSimdFpRegPre(uint32_t opc, const VRegSc &vt, const AdrPreImm &adr) {
  uint32_t size = (vt.getBit() == 16) ? 1 : (vt.getBit() == 32) ? 2 : (vt.getBit() == 64) ? 3 : 0;

  int imm = adr.getImm();
  uint32_t imm9 = imm & ones(9);

  verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t V = 1;
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(imm9, 12), F(3, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

// Atomic memory oprations
void CodeGenerator::AtomicMemOp(uint32_t size, uint32_t V, uint32_t A, uint32_t R, uint32_t o3, uint32_t opc, const RReg &rs, const RReg &rt, const AdrNoOfs &adr) {
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(A, 23), F(R, 22), F(1, 21), F(rs.getIdx(), 16), F(o3, 15), F(opc, 12), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::AtomicMemOp(uint32_t size, uint32_t V, uint32_t A, uint32_t R, uint32_t o3, uint32_t opc, const RReg &rs, const RReg &rt, const AdrImm &adr) {
  verifyIncList(adr.getImm(), {0}, ERR_ILLEGAL_IMM_VALUE);
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(A, 23), F(R, 22), F(1, 21), F(rs.getIdx(), 16), F(o3, 15), F(opc, 12), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// load/store register (register offset)
void CodeGenerator::LdStReg(uint32_t size, uint32_t opc, const RReg &rt, const AdrReg &adr) {
  uint32_t option = 3;
  uint32_t S = ((adr.getInitSh() && size == 0) || (adr.getSh() != 0 && size != 0)) ? 1 : 0;
  uint32_t V = 0;

  verifyIncList(adr.getSh(), {0, size}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncList(adr.getMod(), {LSL}, ERR_ILLEGAL_SHMOD);

  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(1, 21), F(adr.getXm().getIdx(), 16), F(option, 13), F(S, 12), F(2, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// load/store register (register offset)
void CodeGenerator::LdStReg(uint32_t size, uint32_t opc, const RReg &rt, const AdrExt &adr) {
  uint32_t option = adr.getMod();
  uint32_t S = ((adr.getInitSh() && size == 0) || (adr.getSh() != 0 && size != 0)) ? 1 : 0;
  uint32_t V = 0;

  verifyIncList(adr.getSh(), {0, size}, ERR_ILLEGAL_CONST_VALUE);
  if (adr.getRm().getBit() == 64)
    verifyIncList(option, {SXTX}, ERR_ILLEGAL_EXTMOD);
  else
    verifyIncList(option, {UXTW, SXTW}, ERR_ILLEGAL_EXTMOD);

  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(1, 21), F(adr.getRm().getIdx(), 16), F(option, 13), F(S, 12), F(2, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});
  dd(code);
}

// load/store register (register offset)
void CodeGenerator::LdStSimdFpReg(uint32_t opc, const VRegSc &vt, const AdrReg &adr) {
  uint32_t size = genSize(vt);
  uint32_t option = 3;
  uint32_t vt_bit = vt.getBit();
  uint32_t S = ((adr.getInitSh() && vt_bit == 8) || (adr.getSh() != 0 && vt_bit != 8)) ? 1 : 0;
  uint32_t V = 1;

  verifyIncList(adr.getSh(), {0, size}, ERR_ILLEGAL_CONST_VALUE);

  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(1, 21), F(adr.getXm().getIdx(), 16), F(option, 13), F(S, 12), F(2, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

// load/store register (register offset)
void CodeGenerator::LdStSimdFpReg(uint32_t opc, const VRegSc &vt, const AdrExt &adr) {
  uint32_t size = genSize(vt);
  uint32_t option = adr.getMod();
  uint32_t vt_bit = vt.getBit();
  uint32_t S = ((adr.getInitSh() && vt_bit == 8) || (adr.getSh() != 0 && vt_bit != 8)) ? 1 : 0;
  uint32_t V = 1;

  uint32_t max_sh = (vt.getBit() == 128) ? 4 : size;
  verifyIncList(adr.getSh(), {0, max_sh}, ERR_ILLEGAL_CONST_VALUE);

  if (adr.getRm().getBit() == 64)
    verifyIncList(option, {SXTX}, ERR_ILLEGAL_EXTMOD);
  else
    verifyIncList(option, {UXTW, SXTW}, ERR_ILLEGAL_EXTMOD);

  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(1, 21), F(adr.getRm().getIdx(), 16), F(option, 13), F(S, 12), F(2, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

// load/store register (register offset)
void CodeGenerator::PfExt(Prfop prfop, const AdrReg &adr) {
  uint32_t size = 3;
  uint32_t opc = 2;
  uint32_t option = adr.getMod();
  uint32_t S = ((adr.getInitSh() && size == 0) || (adr.getSh() != 0 && size != 0)) ? 1 : 0;
  uint32_t V = 0;

  verifyIncList(adr.getSh(), {0, 3}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncList(option, {LSL}, ERR_ILLEGAL_SHMOD);

  uint32_t ext_opt = 3;
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(1, 21), F(adr.getXm().getIdx(), 16), F(ext_opt, 13), F(S, 12), F(2, 10), F(adr.getXn().getIdx(), 5), F(prfop, 0)});
  dd(code);
}

void CodeGenerator::PfExt(Prfop prfop, const AdrExt &adr) {
  uint32_t size = 3;
  uint32_t opc = 2;
  uint32_t option = adr.getMod();
  uint32_t S = ((adr.getInitSh() && size == 0) || (adr.getSh() != 0 && size != 0)) ? 1 : 0;
  uint32_t V = 0;

  verifyIncList(adr.getSh(), {0, 3}, ERR_ILLEGAL_CONST_VALUE);

  if (adr.getRm().getBit() == 64)
    verifyIncList(option, {SXTX}, ERR_ILLEGAL_EXTMOD);
  else
    verifyIncList(option, {UXTW, SXTW}, ERR_ILLEGAL_EXTMOD);

  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(opc, 22), F(1, 21), F(adr.getRm().getIdx(), 16), F(option, 13), F(S, 12), F(2, 10), F(adr.getXn().getIdx(), 5), F(prfop, 0)});
  dd(code);
}

// loat/store register (pac)
void CodeGenerator::LdStRegPac(uint32_t M, uint32_t W, const XReg &xt, const AdrImm &adr) {
  uint32_t size = 3;
  uint32_t V = 0;

  int32_t imm = adr.getImm();
  uint32_t S = (imm < 0) ? 1 : 0;
  uint32_t imm9 = (imm >> 3) & ones(9);

  verifyIncRange(imm, -4096, 4088, ERR_ILLEGAL_IMM_RANGE, true);
  verifyCond(
      std::abs(imm), [](uint64_t x) { return ((x % 8) == 0); }, ERR_ILLEGAL_IMM_COND);
  verifyIncRange(xt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(M, 23), F(S, 22), F(1, 21), F(imm9, 12), F(W, 11), F(1, 10), F(adr.getXn().getIdx(), 5), F(xt.getIdx(), 0)});
  dd(code);
}

// loat/store register (pac)
void CodeGenerator::LdStRegPac(uint32_t M, uint32_t W, const XReg &xt, const AdrPreImm &adr) {
  uint32_t size = 3;
  uint32_t V = 0;

  int32_t imm = adr.getImm();
  uint32_t S = (imm < 0) ? 1 : 0;
  uint32_t imm9 = (imm >> 3) & ones(9);

  verifyIncRange(imm, -4096, 4088, ERR_ILLEGAL_IMM_RANGE, true);
  verifyCond(
      std::abs(imm), [](uint64_t x) { return ((x % 8) == 0); }, ERR_ILLEGAL_IMM_COND);
  verifyIncRange(xt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(M, 23), F(S, 22), F(1, 21), F(imm9, 12), F(W, 11), F(1, 10), F(adr.getXn().getIdx(), 5), F(xt.getIdx(), 0)});
  dd(code);
}

// loat/store register (unsigned immediate)
void CodeGenerator::LdStRegUnImm(uint32_t size, uint32_t opc, const RReg &rt, const AdrUimm &adr) {
  int32_t imm = adr.getImm();
  int32_t times = 1 << size;
  uint32_t imm12 = (imm >> size) & ones(12);

  verifyIncRange(imm, 0, 4095 * times, ERR_ILLEGAL_IMM_RANGE);
  verifyCond(
      imm, [=](uint64_t x) { return ((x & ones(size)) == 0); }, ERR_ILLEGAL_IMM_COND);
  verifyIncRange(rt.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t V = 0;
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(1, 24), F(opc, 22), F(imm12, 10), F(adr.getXn().getIdx(), 5), F(rt.getIdx(), 0)});

  dd(code);
}

// loat/store register (unsigned immediate)
void CodeGenerator::LdStSimdFpUnImm(uint32_t opc, const VRegSc &vt, const AdrUimm &adr) {
  int32_t imm = adr.getImm();
  uint32_t times = vt.getBit() / 8;
  uint32_t sh = (uint32_t)std::log2(times);
  uint32_t imm12 = (imm >> sh) & ones(12);

  verifyIncRange(imm, 0, 4095 * times, ERR_ILLEGAL_IMM_RANGE);
  verifyCond(
      imm, [=](uint64_t x) { return ((x & ones(sh)) == 0); }, ERR_ILLEGAL_IMM_COND);

  uint32_t V = 1;
  uint32_t size = genSize(vt);
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(1, 24), F(opc, 22), F(imm12, 10), F(adr.getXn().getIdx(), 5), F(vt.getIdx(), 0)});
  dd(code);
}

// loat/store register (unsigned immediate)
void CodeGenerator::PfRegImm(Prfop prfop, const AdrUimm &adr) {
  int32_t imm = adr.getImm();
  int32_t times = 8;
  uint32_t imm12 = (imm >> 3) & ones(12);

  verifyIncRange(imm, 0, 4095 * times, ERR_ILLEGAL_IMM_RANGE);
  verifyCond(
      imm, [=](uint64_t x) { return ((x & ones(3)) == 0); }, ERR_ILLEGAL_IMM_COND);

  uint32_t size = 3;
  uint32_t opc = 2;
  uint32_t V = 0;
  uint32_t code = concat({F(size, 30), F(0x7, 27), F(V, 26), F(1, 24), F(opc, 22), F(imm12, 10), F(adr.getXn().getIdx(), 5), F(prfop, 0)});
  dd(code);
}

// Data processing (2 source)
void CodeGenerator::DataProc2Src(uint32_t opcode, const RReg &rd, const RReg &rn, const RReg &rm) {
  uint32_t sf = genSf(rm);
  uint32_t S = 0;

  verifyCond(
      SP_IDX, [=](uint64_t x) { return rd.getIdx() < x || rn.getIdx() < x || rm.getIdx() < x; }, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(sf, 31), F(S, 29), F(0xd6, 21), F(rm.getIdx(), 16), F(opcode, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Data processing (1 source)
void CodeGenerator::DataProc1Src(uint32_t opcode2, uint32_t opcode, const RReg &rd, const RReg &rn) {
  uint32_t sf = genSf(rd);
  uint32_t S = 0;

  verifyCond(
      SP_IDX, [=](uint64_t x) { return rd.getIdx() < x || rn.getIdx() < x; }, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(sf, 31), F(1, 30), F(S, 29), F(0xd6, 21), F(opcode2, 16), F(opcode, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Data processing (1 source)
void CodeGenerator::DataProc1Src(uint32_t opcode2, uint32_t opcode, const RReg &rd) {
  uint32_t sf = genSf(rd);
  uint32_t S = 0;

  verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(sf, 31), F(1, 30), F(S, 29), F(0xd6, 21), F(opcode2, 16), F(opcode, 10), F(0x1f, 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Logical (shifted register)
void CodeGenerator::LogicalShiftReg(uint32_t opc, uint32_t N, const RReg &rd, const RReg &rn, const RReg &rm, ShMod shmod, uint32_t sh) {
  uint32_t sf = genSf(rd);
  uint32_t imm6 = sh & ones(6);

  verifyIncRange(sh, 0, (32 << sf) - 1, ERR_ILLEGAL_CONST_RANGE);
  verifyCond(
      SP_IDX, [=](uint64_t x) { return rd.getIdx() < x || rn.getIdx() < x || rm.getIdx() < x; }, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(sf, 31), F(opc, 29), F(0xa, 24), F(shmod, 22), F(N, 21), F(rm.getIdx(), 16), F(imm6, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Move (register) alias of ADD,ORR
void CodeGenerator::MvReg(const RReg &rd, const RReg &rn) {
  if (rd.getIdx() == SP_IDX || rn.getIdx() == SP_IDX) {
    // alias of ADD
    AddSubImm(0, 0, rd, rn, 0, 0);
  } else {
    // alias of ORR
    LogicalShiftReg(1, 0, rd, RReg(SP_IDX, rd.getBit()), rn, LSL, 0);
  }
}

// Add/subtract (shifted register)
void CodeGenerator::AddSubShiftReg(uint32_t opc, uint32_t S, const RReg &rd, const RReg &rn, const RReg &rm, ShMod shmod, uint32_t sh, bool alias) {
  uint32_t rd_sp = (rd.getIdx() == SP_IDX);
  uint32_t rn_sp = (rn.getIdx() == SP_IDX);
  if (((rd_sp + rn_sp) >= 1 + (uint32_t)alias) && shmod == LSL) {
    AddSubExtReg(opc, S, rd, rn, rm, EXT_LSL, sh);
    return;
  }

  if (shmod == NONE)
    shmod = LSL;

  uint32_t sf = genSf(rd);
  uint32_t imm6 = sh & ones(6);

  verifyIncRange(sh, 0, (32 << sf) - 1, ERR_ILLEGAL_CONST_RANGE);
  if (!(alias && S == 1))
    verifyIncRange(rd.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  if (!(alias && opc == 1))
    verifyIncRange(rn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(rm.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(sf, 31), F(opc, 30), F(S, 29), F(0xb, 24), F(shmod, 22), F(rm.getIdx(), 16), F(imm6, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Add/subtract (extended register)
void CodeGenerator::AddSubExtReg(uint32_t opc, uint32_t S, const RReg &rd, const RReg &rn, const RReg &rm, ExtMod extmod, uint32_t sh) {
  uint32_t sf = genSf(rd);
  uint32_t opt = 0;
  uint32_t imm3 = sh & ones(3);

  verifyIncRange(sh, 0, 4, ERR_ILLEGAL_CONST_RANGE);

  uint32_t option = (extmod == EXT_LSL && sf == 0) ? 2 : (extmod == EXT_LSL && sf == 1) ? 3 : extmod;
  uint32_t code = concat({F(sf, 31), F(opc, 30), F(S, 29), F(0xb, 24), F(opt, 22), F(1, 21), F(rm.getIdx(), 16), F(option, 13), F(imm3, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Add/subtract (with carry)
void CodeGenerator::AddSubCarry(uint32_t op, uint32_t S, const RReg &rd, const RReg &rn, const RReg &rm) {
  uint32_t sf = genSf(rd);

  verifyCond(
      SP_IDX, [=](uint64_t x) { return rd.getIdx() < x || rn.getIdx() < x || rm.getIdx() < x; }, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(sf, 31), F(op, 30), F(S, 29), F(0xd, 25), F(rm.getIdx(), 16), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Rotate right into flags
void CodeGenerator::RotateR(uint32_t op, uint32_t S, uint32_t o2, const XReg &xn, uint32_t sh, uint32_t mask) {
  uint32_t sf = genSf(xn);
  uint32_t imm6 = sh & ones(6);

  verifyIncRange(xn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(sh, 0, 63, ERR_ILLEGAL_CONST_RANGE);
  verifyIncRange(mask, 0, 15, ERR_ILLEGAL_CONST_RANGE);

  uint32_t code = concat({F(sf, 31), F(op, 30), F(S, 29), F(0xd, 25), F(imm6, 15), F(0x1, 10), F(xn.getIdx(), 5), F(o2, 4), F(mask, 0)});
  dd(code);
}

// Evaluate into flags
void CodeGenerator::Evaluate(uint32_t op, uint32_t S, uint32_t opcode2, uint32_t sz, uint32_t o3, uint32_t mask, const WReg &wn) {
  uint32_t sf = 0;

  verifyIncRange(wn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(mask, 0, 15, ERR_ILLEGAL_CONST_RANGE);

  uint32_t code = concat({F(sf, 31), F(op, 30), F(S, 29), F(0xd, 25), F(opcode2, 15), F(sz, 14), F(0x2, 10), F(wn.getIdx(), 5), F(o3, 4), F(mask, 0)});
  dd(code);
}

// Conditional compare (register)
void CodeGenerator::CondCompReg(uint32_t op, uint32_t S, uint32_t o2, uint32_t o3, const RReg &rn, const RReg &rm, uint32_t nczv, Cond cond) {
  uint32_t sf = genSf(rn);

  verifyCond(
      SP_IDX, [=](uint64_t x) { return rn.getIdx() < x || rm.getIdx() < x; }, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(nczv, 0, 15, ERR_ILLEGAL_CONST_RANGE);

  uint32_t code = concat({F(sf, 31), F(op, 30), F(S, 29), F(0xd2, 21), F(rm.getIdx(), 16), F(cond, 12), F(o2, 10), F(rn.getIdx(), 5), F(o3, 4), F(nczv, 0)});
  dd(code);
}

// Conditional compare (imm)
void CodeGenerator::CondCompImm(uint32_t op, uint32_t S, uint32_t o2, uint32_t o3, const RReg &rn, uint32_t imm, uint32_t nczv, Cond cond) {
  uint32_t sf = genSf(rn);
  uint32_t imm5 = imm & ones(5);

  verifyIncRange(imm, 0, 31, ERR_ILLEGAL_IMM_RANGE);
  verifyIncRange(nczv, 0, 15, ERR_ILLEGAL_CONST_RANGE);
  verifyIncRange(rn.getIdx(), 0, SP_IDX - 1, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(sf, 31), F(op, 30), F(S, 29), F(0xd2, 21), F(imm5, 16), F(cond, 12), F(1, 11), F(o2, 10), F(rn.getIdx(), 5), F(o3, 4), F(nczv, 0)});
  dd(code);
}

// Conditional select
void CodeGenerator::CondSel(uint32_t op, uint32_t S, uint32_t op2, const RReg &rd, const RReg &rn, const RReg &rm, Cond cond) {
  uint32_t sf = genSf(rn);

  verifyCond(
      SP_IDX, [=](uint64_t x) { return rd.getIdx() < x || rn.getIdx() < x || rm.getIdx() < x; }, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(sf, 31), F(op, 30), F(S, 29), F(0xd4, 21), F(rm.getIdx(), 16), F(cond, 12), F(op2, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Conditional select
void CodeGenerator::DataProc3Reg(uint32_t op54, uint32_t op31, uint32_t o0, const RReg &rd, const RReg &rn, const RReg &rm, const RReg &ra) {
  uint32_t sf = genSf(rd);

  verifyCond(
      SP_IDX, [=](uint64_t x) { return rd.getIdx() < x || rn.getIdx() < x || rm.getIdx() < x || ra.getIdx() < x; }, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(sf, 31), F(op54, 29), F(0x1b, 24), F(op31, 21), F(rm.getIdx(), 16), F(o0, 15), F(ra.getIdx(), 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Conditional select
void CodeGenerator::DataProc3Reg(uint32_t op54, uint32_t op31, uint32_t o0, const RReg &rd, const RReg &rn, const RReg &rm) {
  uint32_t sf = genSf(rd);

  verifyCond(
      SP_IDX, [=](uint64_t x) { return rd.getIdx() < x || rn.getIdx() < x || rm.getIdx() < x; }, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(sf, 31), F(op54, 29), F(0x1b, 24), F(op31, 21), F(rm.getIdx(), 16), F(o0, 15), F(0x1f, 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Cryptographic AES
void CodeGenerator::CryptAES(uint32_t opcode, const VRegVec &vd, const VRegVec &vn) {
  uint32_t size = genSize(vd);
  uint32_t code = concat({F(0x4e, 24), F(size, 22), F(0x14, 17), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Cryptographic three-register SHA
void CodeGenerator::Crypt3RegSHA(uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegVec &vm) {
  uint32_t size = 0;
  uint32_t code = concat({F(0x5e, 24), F(size, 22), F(vm.getIdx(), 16), F(opcode, 12), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Cryptographic three-register SHA
void CodeGenerator::Crypt3RegSHA(uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm) {
  uint32_t size = 0;
  uint32_t code = concat({F(0x5e, 24), F(size, 22), F(vm.getIdx(), 16), F(opcode, 12), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Cryptographic two-register SHA
void CodeGenerator::Crypt2RegSHA(uint32_t opcode, const Reg &vd, const Reg &vn) {
  uint32_t size = 0;
  uint32_t code = concat({F(0x5e, 24), F(size, 22), F(0x14, 17), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD Scalar copy
void CodeGenerator::AdvSimdScCopy(uint32_t op, uint32_t imm4, const VRegSc &vd, const VRegElem &vn) {
  uint32_t sh = genSize(vd);
  uint32_t imm5 = 1 << sh | vn.getElemIdx() << (sh + 1);
  uint32_t code = concat({F(1, 30), F(op, 29), F(0xf, 25), F(imm5, 16), F(imm4, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD Scalar three same FP16
void CodeGenerator::AdvSimdSc3SameFp16(uint32_t U, uint32_t a, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm) {
  uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(a, 23), F(2, 21), F(vm.getIdx(), 16), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD Scalar two-register miscellaneous FP16
void CodeGenerator::AdvSimdSc2RegMiscFp16(uint32_t U, uint32_t a, uint32_t opcode, const VRegSc &vd, const VRegSc &vn) {
  uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(a, 23), F(0xf, 19), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::AdvSimdSc2RegMiscFp16(uint32_t U, uint32_t a, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, double zero) {
  verifyIncList(std::lround(zero * 10), {0}, ERR_ILLEGAL_CONST_VALUE);
  AdvSimdSc2RegMiscFp16(U, a, opcode, vd, vn);
}

// Advanced SIMD Scalar three same extra
void CodeGenerator::AdvSimdSc3SameExtra(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm) {
  uint32_t size = genSize(vd);
  uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22), F(vm.getIdx(), 16), F(1, 15), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD Scalar two-register miscellaneous
void CodeGenerator::AdvSimdSc2RegMisc(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn) {
  uint32_t size = genSize(vd);
  uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22), F(1, 21), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::AdvSimdSc2RegMisc(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, uint32_t zero) {
  verifyIncList(zero, {0}, ERR_ILLEGAL_CONST_VALUE);
  AdvSimdSc2RegMisc(U, opcode, vd, vn);
}

// Advanced SIMD Scalar two-register miscellaneous
void CodeGenerator::AdvSimdSc2RegMiscSz0x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn) {
  uint32_t size = genSize(vn) & 1;
  uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22), F(1, 21), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD Scalar two-register miscellaneous
void CodeGenerator::AdvSimdSc2RegMiscSz1x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn) {
  uint32_t size = genSize(vd);
  uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22), F(1, 21), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::AdvSimdSc2RegMiscSz1x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, double zero) {
  verifyIncList(std::lround(zero * 10), {0}, ERR_ILLEGAL_CONST_VALUE);
  AdvSimdSc2RegMiscSz1x(U, opcode, vd, vn);
}

// Advanced SIMD scalar pairwize
void CodeGenerator::AdvSimdScPairwise(uint32_t U, uint32_t size, uint32_t opcode, const VRegSc &vd, const VRegVec &vn) {
  uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22), F(3, 20), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD scalar three different
void CodeGenerator::AdvSimdSc3Diff(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm) {
  uint32_t size = genSize(vn);
  uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22), F(1, 21), F(vm.getIdx(), 16), F(opcode, 12), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD scalar three same
void CodeGenerator::AdvSimdSc3Same(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm) {
  uint32_t size = genSize(vd);
  uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22), F(1, 21), F(vm.getIdx(), 16), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD scalar three same
void CodeGenerator::AdvSimdSc3SameSz0x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm) {
  uint32_t size = genSize(vd) & 1;
  uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22), F(1, 21), F(vm.getIdx(), 16), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD scalar three same
void CodeGenerator::AdvSimdSc3SameSz1x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm) {
  uint32_t size = genSize(vd);
  uint32_t code = concat({F(1, 30), F(U, 29), F(0xf, 25), F(size, 22), F(1, 21), F(vm.getIdx(), 16), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD scalar shift by immediate
void CodeGenerator::AdvSimdScShImm(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, uint32_t sh) {
  uint32_t size = genSize(vd);

  bool lsh = (opcode == 0xa || opcode == 0xc || opcode == 0xe); // left shift
  uint32_t base = vd.getBit();
  uint32_t imm = (lsh) ? (sh + base) : ((base << 1) - sh);
  uint32_t immh = 1 << size | field(imm, size + 2, 3);
  uint32_t immb = field(imm, 2, 0);

  verifyIncRange(sh, (1 - lsh), (base - lsh), ERR_ILLEGAL_CONST_RANGE);

  uint32_t code = concat({F(1, 30), F(U, 29), F(0x1f, 24), F(immh, 19), F(immb, 16), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD scalar x indexed element
void CodeGenerator::AdvSimdScXIndElemSz(uint32_t U, uint32_t size, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegElem &vm) {
  uint32_t bits = vm.getBit();
  uint32_t eidx = vm.getElemIdx();
  uint32_t H = (bits == 16) ? field(eidx, 2, 2) : (bits == 32) ? field(eidx, 1, 1) : field(eidx, 0, 0);
  uint32_t L = (bits == 16) ? field(eidx, 1, 1) : (bits == 32) ? field(eidx, 0, 0) : 0;
  uint32_t M = (bits == 16) ? field(eidx, 0, 0) : field(vm.getIdx(), 4, 4);
  uint32_t vmidx = vm.getIdx() & ones(4);

  if (bits == 16)
    verifyIncRange(vm.getIdx(), 0, 15, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(1, 30), F(U, 29), F(0x1f, 24), F(size, 22), F(L, 21), F(M, 20), F(vmidx, 16), F(opcode, 12), F(H, 11), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::AdvSimdScXIndElem(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegElem &vm) {
  uint32_t size = genSize(vm);
  AdvSimdScXIndElemSz(U, size, opcode, vd, vn, vm);
}

// Advanced SIMD table lookup
void CodeGenerator::AdvSimdTblLkup(uint32_t op2, uint32_t len, uint32_t op, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm) {
  uint32_t Q = genQ(vd);
  uint32_t code = concat({F(Q, 30), F(0xe, 24), F(op2, 22), F(vm.getIdx(), 16), F(len - 1, 13), F(op, 12), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD table lookup
void CodeGenerator::AdvSimdTblLkup(uint32_t op2, uint32_t op, const VRegVec &vd, const VRegList &vn, const VRegVec &vm) {
  uint32_t Q = genQ(vd);
  uint32_t len = vn.getLen() - 1;
  uint32_t code = concat({F(Q, 30), F(0xe, 24), F(op2, 22), F(vm.getIdx(), 16), F(len, 13), F(op, 12), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD permute
void CodeGenerator::AdvSimdPermute(uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm) {
  uint32_t Q = genQ(vd);
  uint32_t size = genSize(vd);
  uint32_t code = concat({F(Q, 30), F(0xe, 24), F(size, 22), F(vm.getIdx(), 16), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD extract
void CodeGenerator::AdvSimdExtract(uint32_t op2, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm, uint32_t index) {
  uint32_t Q = genQ(vd);
  uint32_t imm4 = index & ones(4);

  verifyIncRange(index, 0, 15, ERR_ILLEGAL_CONST_RANGE);
  if (Q == 0)
    verifyCond(
        imm4, [](int64_t x) { return (x >> 3) == 0; }, ERR_ILLEGAL_CONST_COND);

  uint32_t code = concat({F(Q, 30), F(0x2e, 24), F(op2, 22), F(vm.getIdx(), 16), F(imm4, 11), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD copy
void CodeGenerator::AdvSimdCopyDupElem(uint32_t op, uint32_t imm4, const VRegVec &vd, const VRegElem &vn) {
  uint32_t Q = genQ(vd);
  uint32_t size = genSize(vd);
  uint32_t imm5 = (1 << size) | (vn.getElemIdx() << (size + 1));
  uint32_t code = concat({F(Q, 30), F(op, 29), F(0xe, 24), F(imm5, 16), F(imm4, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD copy
void CodeGenerator::AdvSimdCopyDupGen(uint32_t op, uint32_t imm4, const VRegVec &vd, const RReg &rn) {
  uint32_t Q = genQ(vd);
  uint32_t size = genSize(vd);
  uint32_t imm5 = 1 << size;
  uint32_t code = concat({F(Q, 30), F(op, 29), F(0xe, 24), F(imm5, 16), F(imm4, 11), F(3, 10), F(rn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD copy
void CodeGenerator::AdvSimdCopyMov(uint32_t op, uint32_t imm4, const RReg &rd, const VRegElem &vn) {
  uint32_t Q = genSf(rd);
  uint32_t size = genSize(vn);
  uint32_t imm5 = ((1 << size) | (vn.getElemIdx() << (size + 1))) & ones(5);
  uint32_t code = concat({F(Q, 30), F(op, 29), F(0xe, 24), F(imm5, 16), F(imm4, 11), F(1, 10), F(vn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD copy
void CodeGenerator::AdvSimdCopyInsGen(uint32_t op, uint32_t imm4, const VRegElem &vd, const RReg &rn) {
  uint32_t Q = 1;
  uint32_t size = genSize(vd);
  uint32_t imm5 = ((1 << size) | (vd.getElemIdx() << (size + 1))) & ones(5);
  uint32_t code = concat({F(Q, 30), F(op, 29), F(0xe, 24), F(imm5, 16), F(imm4, 11), F(1, 10), F(rn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD copy
void CodeGenerator::AdvSimdCopyElemIns(uint32_t op, const VRegElem &vd, const VRegElem &vn) {
  uint32_t Q = 1;
  uint32_t size = genSize(vd);
  uint32_t imm5 = ((1 << size) | (vd.getElemIdx() << (size + 1))) & ones(5);
  uint32_t imm4 = (vn.getElemIdx() << size) & ones(4);
  uint32_t code = concat({F(Q, 30), F(op, 29), F(0xe, 24), F(imm5, 16), F(imm4, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD three same (FP16)
void CodeGenerator::AdvSimd3SameFp16(uint32_t U, uint32_t a, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm) {
  uint32_t Q = genQ(vd);
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(a, 23), F(2, 21), F(vm.getIdx(), 16), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD two-register miscellaneous (FP16)
void CodeGenerator::AdvSimd2RegMiscFp16(uint32_t U, uint32_t a, uint32_t opcode, const VRegVec &vd, const VRegVec &vn) {
  uint32_t Q = genQ(vd);
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(a, 23), F(0xf, 19), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::AdvSimd2RegMiscFp16(uint32_t U, uint32_t a, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, double zero) {
  verifyIncList(std::lround(zero * 10), {0}, ERR_ILLEGAL_CONST_VALUE);
  AdvSimd2RegMiscFp16(U, a, opcode, vd, vn);
}

// Advanced SIMD three same extra
void CodeGenerator::AdvSimd3SameExtra(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm) {
  uint32_t Q = genQ(vd);
  uint32_t size = genSize(vd);
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(vm.getIdx(), 16), F(1, 15), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD three same extra
void CodeGenerator::AdvSimd3SameExtraRotate(uint32_t U, uint32_t op32, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm, uint32_t rotate) {
  uint32_t Q = genQ(vd);
  uint32_t size = genSize(vd);
  uint32_t rot = rotate / 90;
  uint32_t opcode = (op32 == 2) ? ((op32 << 2) | rot) : ((op32 << 2) | (rot & 0x2));

  if (op32 == 2)
    verifyIncList(rotate, {0, 90, 180, 270}, ERR_ILLEGAL_CONST_VALUE);
  else if (op32 == 3)
    verifyIncList(rotate, {90, 270}, ERR_ILLEGAL_CONST_VALUE);

  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(vm.getIdx(), 16), F(1, 15), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD two-register miscellaneous
void CodeGenerator::AdvSimd2RegMisc(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn) {
  bool sel_vd = (opcode != 0x2 && opcode != 0x6);
  uint32_t Q = (sel_vd) ? genQ(vd) : genQ(vn);
  uint32_t size = (sel_vd) ? genSize(vd) : genSize(vn);
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(1, 21), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD two-register miscellaneous
void CodeGenerator::AdvSimd2RegMisc(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, uint32_t sh) {
  uint32_t Q = genQ(vn);
  uint32_t size = genSize(vn);

  verifyIncList(sh, {vn.getBit()}, ERR_ILLEGAL_CONST_VALUE);

  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(1, 21), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD two-register miscellaneous
void CodeGenerator::AdvSimd2RegMiscZero(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, uint32_t zero) {
  verifyIncList(zero, {0}, ERR_ILLEGAL_CONST_VALUE);
  AdvSimd2RegMisc(U, opcode, vd, vn);
}

// Advanced SIMD two-register miscellaneous
void CodeGenerator::AdvSimd2RegMiscSz(uint32_t U, uint32_t size, uint32_t opcode, const VRegVec &vd, const VRegVec &vn) {
  uint32_t Q = genQ(vd);
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(1, 21), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD two-register miscellaneous
void CodeGenerator::AdvSimd2RegMiscSz0x(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn) {
  bool sel_vd = (opcode == 0x17);
  uint32_t Q = (!sel_vd) ? genQ(vd) : genQ(vn);
  uint32_t size = (sel_vd) ? genSize(vd) : genSize(vn);
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F((size & 1), 22), F(1, 21), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD two-register miscellaneous
void CodeGenerator::AdvSimd2RegMiscSz1x(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn) {
  uint32_t Q = genQ(vd);
  uint32_t size = genSize(vd);
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(1, 21), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::AdvSimd2RegMiscSz1x(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, double zero) {
  verifyIncList(std::lround(zero * 10), {0}, ERR_ILLEGAL_CONST_VALUE);
  AdvSimd2RegMiscSz1x(U, opcode, vd, vn);
}

// Advanced SIMD across lanes
void CodeGenerator::AdvSimdAcrossLanes(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegVec &vn) {
  uint32_t Q = genQ(vn);
  uint32_t size = genSize(vn);
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(3, 20), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD across lanes
void CodeGenerator::AdvSimdAcrossLanesSz0x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegVec &vn) {
  uint32_t Q = genQ(vn);
  uint32_t size = 0;
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(3, 20), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD across lanes
void CodeGenerator::AdvSimdAcrossLanesSz1x(uint32_t U, uint32_t opcode, const VRegSc &vd, const VRegVec &vn) {
  uint32_t Q = genQ(vn);
  uint32_t size = 2;
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(3, 20), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD three different
void CodeGenerator::AdvSimd3Diff(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm) {
  bool vd_sel = (opcode == 0x4 || opcode == 0x6);
  uint32_t Q = (vd_sel) ? genQ(vd) : genQ(vm);
  uint32_t size = (vd_sel) ? genSize(vd) : genSize(vm);
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(1, 21), F(vm.getIdx(), 16), F(opcode, 12), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD three same
void CodeGenerator::AdvSimd3Same(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm) {
  uint32_t Q = genQ(vd);
  uint32_t size = genSize(vd);
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(1, 21), F(vm.getIdx(), 16), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD three same
void CodeGenerator::AdvSimd3SameSz0x(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm) {
  uint32_t Q = genQ(vd);
  uint32_t size = genSize(vd) & 1;
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(1, 21), F(vm.getIdx(), 16), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD three same
void CodeGenerator::AdvSimd3SameSz1x(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm) {
  uint32_t Q = genQ(vd);
  uint32_t size = genSize(vd);
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(1, 21), F(vm.getIdx(), 16), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD three same
void CodeGenerator::AdvSimd3SameSz(uint32_t U, uint32_t size, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm) {
  uint32_t Q = genQ(vd);
  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xe, 24), F(size, 22), F(1, 21), F(vm.getIdx(), 16), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD modified immediate (vector)
void CodeGenerator::AdvSimdModiImmMoviMvni(uint32_t op, uint32_t o2, const VRegVec &vd, uint32_t imm, ShMod shmod, uint32_t sh) {
  uint32_t Q = genQ(vd);
  uint32_t crmode = (vd.getBit() == 8) ? 0xe : (vd.getBit() == 16) ? 0x8 | (sh >> 2) : (vd.getBit() == 32 && shmod == LSL) ? (sh >> 2) : (vd.getBit() == 32 && shmod == MSL) ? 0xc | (sh >> 4) : 0xe;

  if (vd.getBit() == 8)
    verifyIncList(sh, {0}, ERR_ILLEGAL_CONST_VALUE);
  else if (vd.getBit() == 16)
    verifyIncList(sh, {8 * field(crmode, 1, 1)}, ERR_ILLEGAL_CONST_VALUE);
  else if (vd.getBit() == 32 && shmod == LSL)
    verifyIncList(sh, {8 * field(crmode, 2, 1)}, ERR_ILLEGAL_CONST_VALUE);
  else if (vd.getBit() == 32 && shmod == MSL)
    verifyIncList(sh, {8 * field(crmode, 0, 0) + 8}, ERR_ILLEGAL_CONST_VALUE);

  uint32_t abc = field(imm, 7, 5);
  uint32_t defgh = field(imm, 4, 0);
  uint32_t code = concat({F(Q, 30), F(op, 29), F(0xf, 24), F(abc, 16), F(crmode, 12), F(o2, 11), F(1, 10), F(defgh, 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD modified immediate (scalar)
void CodeGenerator::AdvSimdModiImmMoviMvniEnc(uint32_t Q, uint32_t op, uint32_t o2, const Reg &vd, uint64_t imm) {
  uint32_t crmode = 0xe;
  uint32_t imm8 = compactImm(imm);

  verifyCond(
      imm, [&](uint64_t x) { return isCompact(x, imm8); }, ERR_ILLEGAL_IMM_COND);

  uint32_t abc = field(imm8, 7, 5);
  uint32_t defgh = field(imm8, 4, 0);
  uint32_t code = concat({F(Q, 30), F(op, 29), F(0xf, 24), F(abc, 16), F(crmode, 12), F(o2, 11), F(1, 10), F(defgh, 5), F(vd.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::AdvSimdModiImmMoviMvni(uint32_t op, uint32_t o2, const VRegSc &vd, uint64_t imm) {
  uint32_t Q = 0;
  AdvSimdModiImmMoviMvniEnc(Q, op, o2, vd, imm);
}

void CodeGenerator::AdvSimdModiImmMoviMvni(uint32_t op, uint32_t o2, const VRegVec &vd, uint64_t imm) {
  uint32_t Q = genQ(vd);
  AdvSimdModiImmMoviMvniEnc(Q, op, o2, vd, imm);
}

// Advanced SIMD modified immediate
void CodeGenerator::AdvSimdModiImmOrrBic(uint32_t op, uint32_t o2, const VRegVec &vd, uint32_t imm, ShMod mod, uint32_t sh) {
  uint32_t Q = genQ(vd);
  uint32_t crmode = (vd.getBit() == 16) ? (0x9 | (sh >> 2)) : (1 | (sh >> 2));

  verifyIncList(mod, {LSL}, ERR_ILLEGAL_SHMOD);
  if (vd.getBit() == 16)
    verifyIncList(sh, {8 * field(crmode, 1, 1)}, ERR_ILLEGAL_CONST_VALUE);
  else if (vd.getBit() == 32)
    verifyIncList(sh, {8 * field(crmode, 2, 1)}, ERR_ILLEGAL_CONST_VALUE);

  uint32_t abc = field(imm, 7, 5);
  uint32_t defgh = field(imm, 4, 0);
  uint32_t code = concat({F(Q, 30), F(op, 29), F(0xf, 24), F(abc, 16), F(crmode, 12), F(o2, 11), F(1, 10), F(defgh, 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD modified immediate
void CodeGenerator::AdvSimdModiImmFmov(uint32_t op, uint32_t o2, const VRegVec &vd, double imm) {
  uint32_t Q = genQ(vd);
  uint32_t crmode = 0xf;
  uint32_t imm8 = compactImm(imm, vd.getBit());
  uint32_t abc = field(imm8, 7, 5);
  uint32_t defgh = field(imm8, 4, 0);
  uint32_t code = concat({F(Q, 30), F(op, 29), F(0xf, 24), F(abc, 16), F(crmode, 12), F(o2, 11), F(1, 10), F(defgh, 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD shift by immediate
void CodeGenerator::AdvSimdShImm(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, uint32_t sh) {
  bool vd_sel = (opcode != 0x14);
  uint32_t Q = (vd_sel) ? genQ(vd) : genQ(vn);
  uint32_t size = (vd_sel) ? genSize(vd) : genSize(vn);

  bool lsh = (opcode == 0xa || opcode == 0xc || opcode == 0xe || opcode == 0x14); // left shift
  uint32_t base = (vd_sel) ? vd.getBit() : vn.getBit();
  uint32_t imm = (lsh) ? (sh + base) : ((base << 1) - sh);
  uint32_t immh = 1 << size | field(imm, size + 2, 3);
  uint32_t immb = field(imm, 2, 0);

  verifyIncRange(sh, (1 - lsh), (base - lsh), ERR_ILLEGAL_CONST_RANGE);

  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xf, 24), F(immh, 19), F(immb, 16), F(opcode, 11), F(1, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Advanced SIMD vector x indexed element
void CodeGenerator::AdvSimdVecXindElemEnc(uint32_t Q, uint32_t U, uint32_t size, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegElem &vm) {
  bool ucmla = (U == 1 && (opcode & 0x9) == 1);
  uint32_t bits = (vm.getBit() == 8) ? 32 : (ucmla) ? vm.getBit() * 2 : vm.getBit();
  uint32_t eidx = vm.getElemIdx();
  uint32_t H = (bits == 16) ? field(eidx, 2, 2) : (bits == 32) ? field(eidx, 1, 1) : field(eidx, 0, 0);
  uint32_t L = (bits == 16) ? field(eidx, 1, 1) : (bits == 32) ? field(eidx, 0, 0) : 0;
  uint32_t M = (bits == 16) ? field(eidx, 0, 0) : field(vm.getIdx(), 4, 4);
  uint32_t vmidx = vm.getIdx() & ones(4);

  if (bits == 16)
    verifyIncRange(vm.getIdx(), 0, 15, ERR_ILLEGAL_REG_IDX);
  else
    verifyIncRange(vm.getIdx(), 0, 31, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(Q, 30), F(U, 29), F(0xf, 24), F(size, 22), F(L, 21), F(M, 20), F(vmidx, 16), F(opcode, 12), F(H, 11), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::AdvSimdVecXindElem(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegElem &vm) {
  bool vd_sel = (opcode == 0xe);
  uint32_t Q = (vd_sel) ? genQ(vd) : genQ(vn);
  uint32_t size = (vd_sel) ? genSize(vd) : genSize(vn);
  AdvSimdVecXindElemEnc(Q, U, size, opcode, vd, vn, vm);
}

// Advanced SIMD vector x indexed element
void CodeGenerator::AdvSimdVecXindElem(uint32_t U, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegElem &vm, uint32_t rotate) {
  uint32_t Q = genQ(vd);
  uint32_t size = genSize(vd);
  uint32_t rot = rotate / 90;

  verifyIncList(rotate, {0, 90, 180, 270}, ERR_ILLEGAL_CONST_VALUE);

  AdvSimdVecXindElemEnc(Q, U, size, (rot << 1 | opcode), vd, vn, vm);
}

void CodeGenerator::AdvSimdVecXindElemSz(uint32_t U, uint32_t size, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegElem &vm) {
  uint32_t Q = genQ(vd);
  AdvSimdVecXindElemEnc(Q, U, size, opcode, vd, vn, vm);
}

// Cryptographic three-register, imm2
void CodeGenerator::Crypto3RegImm2(uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegElem &vm) {
  uint32_t imm2 = vm.getElemIdx();
  uint32_t code = concat({F(0x672, 21), F(vm.getIdx(), 16), F(2, 14), F(imm2, 12), F(opcode, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Cryptographic three-register SHA 512
void CodeGenerator::Crypto3RegSHA512(uint32_t O, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegVec &vm) {
  uint32_t code = concat({F(0x673, 21), F(vm.getIdx(), 16), F(1, 15), F(O, 14), F(opcode, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Cryptographic three-register SHA 512
void CodeGenerator::Crypto3RegSHA512(uint32_t O, uint32_t opcode, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm) {
  uint32_t code = concat({F(0x673, 21), F(vm.getIdx(), 16), F(1, 15), F(O, 14), F(opcode, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// XAR
void CodeGenerator::CryptoSHA(const VRegVec &vd, const VRegVec &vn, const VRegVec &vm, uint32_t imm6) {
  verifyIncRange(imm6, 0, ones(6), ERR_ILLEGAL_IMM_RANGE);
  uint32_t code = concat({F(0x674, 21), F(vm.getIdx(), 16), F(imm6, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Cryptographic four-register
void CodeGenerator::Crypto4Reg(uint32_t Op0, const VRegVec &vd, const VRegVec &vn, const VRegVec &vm, const VRegVec &va) {
  uint32_t code = concat({F(0x19c, 23), F(Op0, 21), F(vm.getIdx(), 16), F(va.getIdx(), 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Cryptographic two-register SHA512
void CodeGenerator::Crypto2RegSHA512(uint32_t opcode, const VRegVec &vd, const VRegVec &vn) {
  uint32_t code = concat({F(0xcec08, 12), F(opcode, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// conversion between floating-point and fixed-point
void CodeGenerator::ConversionFpFix(uint32_t S, uint32_t type, uint32_t rmode, uint32_t opcode, const VRegSc &vd, const RReg &rn, uint32_t fbits) {
  uint32_t sf = genSf(rn);
  uint32_t scale = 64 - fbits;

  verifyIncRange(fbits, 1, (32 << sf), ERR_ILLEGAL_CONST_RANGE);

  uint32_t code = concat({F(sf, 31), F(S, 29), F(0xf, 25), F(type, 22), F(rmode, 19), F(opcode, 16), F(scale, 10), F(rn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// conversion between floating-point and fixed-point
void CodeGenerator::ConversionFpFix(uint32_t S, uint32_t type, uint32_t rmode, uint32_t opcode, const RReg &rd, const VRegSc &vn, uint32_t fbits) {
  uint32_t sf = genSf(rd);
  uint32_t scale = 64 - fbits;

  verifyIncRange(fbits, 1, (32 << sf), ERR_ILLEGAL_CONST_RANGE);

  uint32_t code = concat({F(sf, 31), F(S, 29), F(0xf, 25), F(type, 22), F(rmode, 19), F(opcode, 16), F(scale, 10), F(vn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// conversion between floating-point and integer
void CodeGenerator::ConversionFpInt(uint32_t sf, uint32_t S, uint32_t type, uint32_t rmode, uint32_t opcode, const RReg &rd, const VRegSc &vn) {
  uint32_t code = concat({F(sf, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21), F(rmode, 19), F(opcode, 16), F(vn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// conversion between floating-point and integer
void CodeGenerator::ConversionFpInt(uint32_t sf, uint32_t S, uint32_t type, uint32_t rmode, uint32_t opcode, const VRegSc &vd, const RReg &rn) {
  uint32_t code = concat({F(sf, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21), F(rmode, 19), F(opcode, 16), F(rn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// conversion between floating-point and integer
void CodeGenerator::ConversionFpInt(uint32_t sf, uint32_t S, uint32_t type, uint32_t rmode, uint32_t opcode, const RReg &rd, const VRegElem &vn) {
  uint32_t code = concat({F(sf, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21), F(rmode, 19), F(opcode, 16), F(vn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// conversion between floating-point and integer
void CodeGenerator::ConversionFpInt(uint32_t sf, uint32_t S, uint32_t type, uint32_t rmode, uint32_t opcode, const VRegElem &vd, const RReg &rn) {
  uint32_t code = concat({F(sf, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21), F(rmode, 19), F(opcode, 16), F(rn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Floating-piont data-processing (1 source)
void CodeGenerator::FpDataProc1Reg(uint32_t M, uint32_t S, uint32_t type, uint32_t opcode, const VRegSc &vd, const VRegSc &vn) {
  uint32_t code = concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21), F(opcode, 15), F(1, 14), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Floating-piont compare
void CodeGenerator::FpComp(uint32_t M, uint32_t S, uint32_t type, uint32_t op, uint32_t opcode2, const VRegSc &vn, const VRegSc &vm) {
  uint32_t code = concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21), F(vm.getIdx(), 16), F(op, 14), F(1, 13), F(vn.getIdx(), 5), F(opcode2, 0)});
  dd(code);
}

// Floating-piont compare
void CodeGenerator::FpComp(uint32_t M, uint32_t S, uint32_t type, uint32_t op, uint32_t opcode2, const VRegSc &vn, double imm) {
  verifyIncList(std::lround(imm), {0}, ERR_ILLEGAL_CONST_VALUE);
  uint32_t code = concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21), F(op, 14), F(1, 13), F(vn.getIdx(), 5), F(opcode2, 0)});
  dd(code);
}

// Floating-piont immediate
void CodeGenerator::FpImm(uint32_t M, uint32_t S, uint32_t type, const VRegSc &vd, double imm) {
  uint32_t imm8 = compactImm(imm, vd.getBit());
  uint32_t code = concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21), F(imm8, 13), F(1, 12), F(vd.getIdx(), 0)});
  dd(code);
}

// Floating-piont conditional compare
void CodeGenerator::FpCondComp(uint32_t M, uint32_t S, uint32_t type, uint32_t op, const VRegSc &vn, const VRegSc &vm, uint32_t nzcv, Cond cond) {
  uint32_t code = concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21), F(vm.getIdx(), 16), F(cond, 12), F(1, 10), F(vn.getIdx(), 5), F(op, 4), F(nzcv, 0)});
  dd(code);
}

// Floating-piont data-processing (2 source)
void CodeGenerator::FpDataProc2Reg(uint32_t M, uint32_t S, uint32_t type, uint32_t opcode, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm) {
  uint32_t code = concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21), F(vm.getIdx(), 16), F(opcode, 12), F(2, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Floating-piont conditional select
void CodeGenerator::FpCondSel(uint32_t M, uint32_t S, uint32_t type, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm, Cond cond) {
  uint32_t code = concat({F(M, 31), F(S, 29), F(0xf, 25), F(type, 22), F(1, 21), F(vm.getIdx(), 16), F(cond, 12), F(3, 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// Floating-piont data-processing (3 source)
void CodeGenerator::FpDataProc3Reg(uint32_t M, uint32_t S, uint32_t type, uint32_t o1, uint32_t o0, const VRegSc &vd, const VRegSc &vn, const VRegSc &vm, const VRegSc &va) {
  uint32_t code = concat({F(M, 31), F(S, 29), F(0x1f, 24), F(type, 22), F(o1, 21), F(vm.getIdx(), 16), F(o0, 15), F(va.getIdx(), 10), F(vn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// ########################### System instruction
// #################################
// Instruction cache maintenance
void CodeGenerator::InstCache(IcOp icop, const XReg &xt) {
  uint32_t code = concat({F(0xd5, 24), F(1, 19), F(icop, 5), F(xt.getIdx(), 0)});
  dd(code);
}

// Data cache maintenance
void CodeGenerator::DataCache(DcOp dcop, const XReg &xt) {
  uint32_t code = concat({F(0xd5, 24), F(1, 19), F(dcop, 5), F(xt.getIdx(), 0)});
  dd(code);
}

// Addresss Translate
void CodeGenerator::AddressTrans(AtOp atop, const XReg &xt) {
  uint32_t code = concat({F(0xd5, 24), F(1, 19), F(atop, 5), F(xt.getIdx(), 0)});
  dd(code);
}

// TLB Invaidate operation
void CodeGenerator::TLBInv(TlbiOp tlbiop, const XReg &xt) {
  uint32_t code = concat({F(0xd5, 24), F(1, 19), F(tlbiop, 5), F(xt.getIdx(), 0)});
  dd(code);
}

// ################################### SVE
// #########################################

// SVE Integer Binary Arithmetic - Predicated Group
void CodeGenerator::SveIntBinArPred(uint32_t opc, uint32_t type, const _ZReg &zd, const _PReg &pg, const _ZReg &zn) {
  uint32_t size = genSize(zd);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x4, 24), F(size, 22), F(type, 19), F(opc, 16), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE bitwize Logical Operation (predicated)
void CodeGenerator::SveBitwiseLOpPred(uint32_t opc, const _ZReg &zd, const _PReg &pg, const _ZReg &zn) { SveIntBinArPred(opc, 3, zd, pg, zn); }

// SVE Integer add/subtract vectors (predicated)
void CodeGenerator::SveIntAddSubVecPred(uint32_t opc, const _ZReg &zd, const _PReg &pg, const _ZReg &zn) { SveIntBinArPred(opc, 0, zd, pg, zn); }

// SVE Integer min/max/diffrence (predicated)
void CodeGenerator::SveIntMinMaxDiffPred(uint32_t opc, uint32_t U, const _ZReg &zd, const _PReg &pg, const _ZReg &zn) { SveIntBinArPred((opc << 1 | U), 1, zd, pg, zn); }

// SVE Integer multiply/divide vectors (predicated)
void CodeGenerator::SveIntMultDivVecPred(uint32_t opc, uint32_t U, const _ZReg &zd, const _PReg &pg, const _ZReg &zn) { SveIntBinArPred((opc << 1 | U), 2, zd, pg, zn); }

// SVE Integer Reduction Group
void CodeGenerator::SveIntReduction(uint32_t opc, uint32_t type, const Reg &rd, const _PReg &pg, const Reg &rn) {
  uint32_t size = genSize(rn);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x4, 24), F(size, 22), F(type, 19), F(opc, 16), F(1, 13), F(pg.getIdx(), 10), F(rn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// SVE bitwise logical reduction (predicated)
void CodeGenerator::SveBitwiseLReductPred(uint32_t opc, const VRegSc &vd, const _PReg &pg, const _ZReg &zn) { SveIntReduction(opc, 3, vd, pg, zn); }

// SVE constructive prefix (predicated)
void CodeGenerator::SveConstPrefPred(uint32_t opc, const _ZReg &zd, const _PReg &pg, const _ZReg &zn) { SveIntReduction((opc << 1 | (static_cast<uint32_t>(pg.isM()))), 2, zd, pg, zn); }

// SVE integer add reduction (predicated)
void CodeGenerator::SveIntAddReductPred(uint32_t opc, uint32_t U, const VRegSc &vd, const _PReg &pg, const _ZReg &zn) { SveIntReduction((opc << 1 | U), 0, vd, pg, zn); }

// SVE integer min/max reduction (predicated)
void CodeGenerator::SveIntMinMaxReductPred(uint32_t opc, uint32_t U, const VRegSc &vd, const _PReg &pg, const _ZReg &zn) { SveIntReduction((opc << 1 | U), 1, vd, pg, zn); }

// SVE Bitwise Shift - Predicate Group
void CodeGenerator::SveBitShPred(uint32_t opc, uint32_t type, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm) {
  uint32_t size = genSize(zdn);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x4, 24), F(size, 22), F(type, 19), F(opc, 16), F(4, 13), F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(zdn.getIdx(), 0)});
  dd(code);
}

// SVE bitwise shift by immediate (predicated)
void CodeGenerator::SveBitwiseShByImmPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, uint32_t amount) {
  bool lsl = (opc == 3);
  uint32_t size = genSize(zdn);
  uint32_t imm = (lsl) ? (amount + zdn.getBit()) : (2 * zdn.getBit() - amount);
  uint32_t imm3 = imm & ones(3);
  uint32_t tsz = (1 << size) | field(imm, size + 2, 3);
  uint32_t tszh = field(tsz, 3, 2);
  uint32_t tszl = field(tsz, 1, 0);

  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(amount, (1 - lsl), (zdn.getBit() - lsl), ERR_ILLEGAL_CONST_RANGE);

  uint32_t code = concat({F(0x4, 24), F(tszh, 22), F(0, 19), F(opc, 16), F(4, 13), F(pg.getIdx(), 10), F(tszl, 8), F(imm3, 5), F(zdn.getIdx(), 0)});
  dd(code);
}

// SVE bitwise shift by vector (predicated)
void CodeGenerator::SveBitwiseShVecPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm) { SveBitShPred(opc, 2, zdn, pg, zm); }

// SVE bitwise shift by wide elements (predicated)
void CodeGenerator::SveBitwiseShWElemPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm) { SveBitShPred(opc, 3, zdn, pg, zm); }

// SVE Integer Unary Arithmetic - Predicated Group
void CodeGenerator::SveIntUnaryArPred(uint32_t opc, uint32_t type, const _ZReg &zd, const _PReg &pg, const _ZReg &zn) {
  uint32_t size = genSize(zd);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x4, 24), F(size, 22), F(type, 19), F(opc, 16), F(5, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE bitwise unary operations (predicated)
void CodeGenerator::SveBitwiseUnaryOpPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm) { SveIntUnaryArPred(opc, 3, zdn, pg, zm); }

// SVE integer unary operations (predicated)
void CodeGenerator::SveIntUnaryOpPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm) { SveIntUnaryArPred(opc, 2, zdn, pg, zm); }

// SVE integer multiply-accumulate writing addend (predicated)
void CodeGenerator::SveIntMultAccumPred(uint32_t opc, const _ZReg &zda, const _PReg &pg, const _ZReg &zn, const _ZReg &zm) {
  uint32_t size = genSize(zda);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x4, 24), F(size, 22), F(zm.getIdx(), 16), F(1, 14), F(opc, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
  dd(code);
}

// SVE integer multiply-add writeing multiplicand (predicated)
void CodeGenerator::SveIntMultAddPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm, const _ZReg &za) {
  uint32_t size = genSize(zdn);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x4, 24), F(size, 22), F(zm.getIdx(), 16), F(3, 14), F(opc, 13), F(pg.getIdx(), 10), F(za.getIdx(), 5), F(zdn.getIdx(), 0)});
  dd(code);
}

// SVE integer add/subtract vectors (unpredicated)
void CodeGenerator::SveIntAddSubUnpred(uint32_t opc, const _ZReg &zd, const _ZReg &zn, const _ZReg &zm) {
  uint32_t size = genSize(zd);
  uint32_t code = concat({F(0x4, 24), F(size, 22), F(1, 21), F(zm.getIdx(), 16), F(opc, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE bitwise logical operations (unpredicated)
void CodeGenerator::SveBitwiseLOpUnpred(uint32_t opc, const _ZReg &zd, const _ZReg &zn, const _ZReg &zm) {
  uint32_t code = concat({F(0x4, 24), F(opc, 22), F(1, 21), F(zm.getIdx(), 16), F(0xc, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE index generation (immediate start, immediate increment)
void CodeGenerator::SveIndexGenImmImmInc(const _ZReg &zd, int32_t imm1, int32_t imm2) {
  uint32_t size = genSize(zd);
  uint32_t imm5b = imm2 & ones(5);
  uint32_t imm5 = imm1 & ones(5);

  verifyIncRange(imm1, -16, 15, ERR_ILLEGAL_IMM_RANGE, true);
  verifyIncRange(imm2, -16, 15, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t code = concat({F(0x4, 24), F(size, 22), F(1, 21), F(imm5b, 16), F(0x10, 10), F(imm5, 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE index generation (immediate start, register increment)
void CodeGenerator::SveIndexGenImmRegInc(const _ZReg &zd, int32_t imm, const RReg &rm) {
  uint32_t size = genSize(zd);
  uint32_t imm5 = imm & ones(5);

  verifyIncRange(imm, -16, 15, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t code = concat({F(0x4, 24), F(size, 22), F(1, 21), F(rm.getIdx(), 16), F(0x12, 10), F(imm5, 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE index generation (register start, immediate increment)
void CodeGenerator::SveIndexGenRegImmInc(const _ZReg &zd, const RReg &rn, int32_t imm) {
  uint32_t size = genSize(zd);
  uint32_t imm5 = imm & ones(5);

  verifyIncRange(imm, -16, 15, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t code = concat({F(0x4, 24), F(size, 22), F(1, 21), F(imm5, 16), F(0x11, 10), F(rn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE index generation (register start, register increment)
void CodeGenerator::SveIndexGenRegRegInc(const _ZReg &zd, const RReg &rn, const RReg &rm) {
  uint32_t size = genSize(zd);
  uint32_t code = concat({F(0x4, 24), F(size, 22), F(1, 21), F(rm.getIdx(), 16), F(0x13, 10), F(rn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE stack frame adjustment
void CodeGenerator::SveStackFrameAdjust(uint32_t op, const XReg &xd, const XReg &xn, int32_t imm) {
  uint32_t imm6 = imm & ones(6);

  verifyIncRange(imm, -32, 31, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t code = concat({F(0x8, 23), F(op, 22), F(1, 21), F(xn.getIdx(), 16), F(0xa, 11), F(imm6, 5), F(xd.getIdx(), 0)});
  dd(code);
}

// SVE stack frame size
void CodeGenerator::SveStackFrameSize(uint32_t op, uint32_t opc2, const XReg &xd, int32_t imm) {
  uint32_t imm6 = imm & ones(6);

  verifyIncRange(imm, -32, 31, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t code = concat({F(0x9, 23), F(op, 22), F(1, 21), F(opc2, 16), F(0xa, 11), F(imm6, 5), F(xd.getIdx(), 0)});
  dd(code);
}

// SVE bitwise shift by immediate (unpredicated)
void CodeGenerator::SveBitwiseShByImmUnpred(uint32_t opc, const _ZReg &zd, const _ZReg &zn, uint32_t amount) {
  bool lsl = (opc == 3);
  uint32_t size = genSize(zd);
  uint32_t imm = (lsl) ? (amount + zd.getBit()) : (2 * zd.getBit() - amount);
  uint32_t imm3 = imm & ones(3);
  uint32_t tsz = (1 << size) | field(imm, size + 2, 3);
  uint32_t tszh = field(tsz, 3, 2);
  uint32_t tszl = field(tsz, 1, 0);

  verifyIncRange(amount, (1 - lsl), (zd.getBit() - lsl), ERR_ILLEGAL_CONST_RANGE);

  uint32_t code = concat({F(0x4, 24), F(tszh, 22), F(1, 21), F(tszl, 19), F(imm3, 16), F(0x9, 12), F(opc, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE bitwise shift by wide elements (unpredicated)
void CodeGenerator::SveBitwiseShByWideElemUnPred(uint32_t opc, const _ZReg &zd, const _ZReg &zn, const _ZReg &zm) {
  uint32_t size = genSize(zd);
  uint32_t code = concat({F(0x4, 24), F(size, 22), F(1, 21), F(zm.getIdx(), 16), F(0x8, 12), F(opc, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE address generation
void CodeGenerator::SveAddressGen(const _ZReg &zd, const AdrVec &adr) {
  ShMod mod = adr.getMod();
  uint32_t sh = adr.getSh();
  uint32_t opc = 2 | (genSize(zd) & 0x1);
  uint32_t msz = sh & ones(2);

  verifyIncList(mod, {LSL, NONE}, ERR_ILLEGAL_SHMOD);
  verifyIncRange(sh, 0, 3, ERR_ILLEGAL_CONST_RANGE);

  uint32_t code = concat({F(0x4, 24), F(opc, 22), F(1, 21), F(adr.getZm().getIdx(), 16), F(0xa, 12), F(msz, 10), F(adr.getZn().getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE address generation
void CodeGenerator::SveAddressGen(const _ZReg &zd, const AdrVecU &adr) {
  ExtMod mod = adr.getMod();
  uint32_t sh = adr.getSh();
  uint32_t opc = (mod == SXTW) ? 0 : 1;
  uint32_t msz = sh & ones(2);

  verifyIncList(mod, {UXTW, SXTW}, ERR_ILLEGAL_EXTMOD);
  verifyIncRange(sh, 0, 3, ERR_ILLEGAL_CONST_RANGE);

  uint32_t code = concat({F(0x4, 24), F(opc, 22), F(1, 21), F(adr.getZm().getIdx(), 16), F(0xa, 12), F(msz, 10), F(adr.getZn().getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE Integer Misc - Unpredicated Group
void CodeGenerator::SveIntMiscUnpred(uint32_t size, uint32_t opc, uint32_t type, const _ZReg &zd, const _ZReg &zn) {
  uint32_t code = concat({F(0x4, 24), F(size, 22), F(1, 21), F(opc, 16), F(0xb, 12), F(type, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE constructive prefix (unpredicated)
void CodeGenerator::SveConstPrefUnpred(uint32_t opc, uint32_t opc2, const _ZReg &zd, const _ZReg &zn) { SveIntMiscUnpred(opc, opc2, 3, zd, zn); }

// SVE floating-point exponential accelerator
void CodeGenerator::SveFpExpAccel(uint32_t opc, const _ZReg &zd, const _ZReg &zn) {
  uint32_t size = genSize(zd);
  SveIntMiscUnpred(size, opc, 2, zd, zn);
}

// SVE floating-point trig select coefficient
void CodeGenerator::SveFpTrigSelCoef(uint32_t opc, const _ZReg &zd, const _ZReg &zn, const _ZReg &zm) {
  uint32_t size = genSize(zd);
  SveIntMiscUnpred(size, zm.getIdx(), opc, zd, zn);
}

// SVE Element Count Group
void CodeGenerator::SveElemCountGrp(uint32_t size, uint32_t op, uint32_t type1, uint32_t type2, const Reg &rd, Pattern pat, ExtMod mod, uint32_t imm) {
  uint32_t imm4 = (imm - 1) & ones(4);
  verifyIncList(mod, {MUL}, ERR_ILLEGAL_EXTMOD);
  verifyIncRange(imm, 1, 16, ERR_ILLEGAL_IMM_RANGE);
  uint32_t code = concat({F(0x4, 24), F(size, 22), F(type1, 20), F(imm4, 16), F(type2, 11), F(op, 10), F(pat, 5), F(rd.getIdx(), 0)});
  dd(code);
}

// SVE element count
void CodeGenerator::SveElemCount(uint32_t size, uint32_t op, const XReg &xd, Pattern pat, ExtMod mod, uint32_t imm) { SveElemCountGrp(size, op, 2, 0x1c, xd, pat, mod, imm); }

// SVE inc/dec register by element count
void CodeGenerator::SveIncDecRegByElemCount(uint32_t size, uint32_t D, const XReg &xd, Pattern pat, ExtMod mod, uint32_t imm) { SveElemCountGrp(size, D, 3, 0x1c, xd, pat, mod, imm); }

// SVE inc/dec vector by element count
void CodeGenerator::SveIncDecVecByElemCount(uint32_t size, uint32_t D, const _ZReg &zd, Pattern pat, ExtMod mod, uint32_t imm) { SveElemCountGrp(size, D, 3, 0x18, zd, pat, mod, imm); }

// SVE saturating inc/dec register by element count
void CodeGenerator::SveSatuIncDecRegByElemCount(uint32_t size, uint32_t D, uint32_t U, const RReg &rdn, Pattern pat, ExtMod mod, uint32_t imm) {
  uint32_t sf = genSf(rdn);
  SveElemCountGrp(size, U, (2 | sf), (0x1e | D), rdn, pat, mod, imm);
}

// SVE saturating inc/dec vector by element count
void CodeGenerator::SveSatuIncDecVecByElemCount(uint32_t size, uint32_t D, uint32_t U, const _ZReg &zdn, Pattern pat, ExtMod mod, uint32_t imm) { SveElemCountGrp(size, U, 2, (0x18 | D), zdn, pat, mod, imm); }

// SVE Bitwise Immeidate Group
void CodeGenerator::SveBitwiseImm(uint32_t opc, const _ZReg &zd, uint64_t imm) {
  uint32_t imm13 = genNImmrImms(imm, zd.getBit());
  uint32_t code = concat({F(0x5, 24), F(opc, 22), F(imm13, 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE bitwise logical with immediate (unpredicated)
void CodeGenerator::SveBitwiseLogicalImmUnpred(uint32_t opc, const _ZReg &zdn, uint64_t imm) { SveBitwiseImm(opc, zdn, imm); }

// SVE broadcast bitmask immediate
void CodeGenerator::SveBcBitmaskImm(const _ZReg &zdn, uint64_t imm) { SveBitwiseImm(3, zdn, imm); }

// SVE copy floating-point immediate (predicated)
void CodeGenerator::SveCopyFpImmPred(const _ZReg &zd, const _PReg &pg, double imm) {
  uint32_t size = genSize(zd);
  uint32_t imm8 = compactImm(imm, zd.getBit());
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(1, 20), F(pg.getIdx(), 16), F(6, 13), F(imm8, 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE copy integer immediate (predicated)
void CodeGenerator::SveCopyIntImmPred(const _ZReg &zd, const _PReg &pg, uint32_t imm, ShMod mod, uint32_t sh) {
  verifyIncList(mod, {LSL}, ERR_ILLEGAL_SHMOD);
  verifyIncList(sh, {0, 8}, ERR_ILLEGAL_CONST_VALUE);
  uint32_t size = genSize(zd);
  uint32_t imm8 = imm & ones(8);
  uint32_t type = (pg.isM() << 1) | (sh == 8);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(1, 20), F(pg.getIdx(), 16), F(type, 13), F(imm8, 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE extract vector (immediate offset)
void CodeGenerator::SveExtVec(const _ZReg &zdn, const _ZReg &zm, uint32_t imm) {
  uint32_t imm8h = field(imm, 7, 3);
  uint32_t imm8l = field(imm, 2, 0);
  verifyIncRange(imm, 0, 255, ERR_ILLEGAL_IMM_RANGE);
  uint32_t code = concat({F(0x5, 24), F(1, 21), F(imm8h, 16), F(imm8l, 10), F(zm.getIdx(), 5), F(zdn.getIdx(), 0)});
  dd(code);
}

// SVE Permute Vector - Unpredicate Group
void CodeGenerator::SvePerVecUnpred(uint32_t size, uint32_t type1, uint32_t type2, const _ZReg &zd, const Reg &rn) {
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(1, 21), F(type1, 16), F(type2, 10), F(rn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE broadcast general register
void CodeGenerator::SveBcGeneralReg(const _ZReg &zd, const RReg &rn) {
  uint32_t size = genSize(zd);
  SvePerVecUnpred(size, 0, 0xe, zd, rn);
}

// SVE broadcast indexed element
void CodeGenerator::SveBcIndexedElem(const _ZReg &zd, const ZRegElem &zn) {
  uint32_t eidx = zn.getElemIdx();
  uint32_t pos = static_cast<uint32_t>(std::log2(zn.getBit()) - 2);
  uint32_t imm = (eidx << pos) | (1 << (pos - 1));
  uint32_t imm2 = field(imm, 6, 5);
  uint32_t tsz = field(imm, 4, 0);

  if (zd.getBit() == 128)
    verifyIncList(field(tsz, 4, 0), {0x10}, ERR_ILLEGAL_IMM_COND);
  else if (zd.getBit() == 64)
    verifyIncList(field(tsz, 3, 0), {0x8}, ERR_ILLEGAL_IMM_COND);
  else if (zd.getBit() == 32)
    verifyIncList(field(tsz, 2, 0), {0x4}, ERR_ILLEGAL_IMM_COND);
  else if (zd.getBit() == 16)
    verifyIncList(field(tsz, 1, 0), {0x2}, ERR_ILLEGAL_IMM_COND);
  else if (zd.getBit() == 8)
    verifyIncList(field(tsz, 0, 0), {0x1}, ERR_ILLEGAL_IMM_COND);

  SvePerVecUnpred(imm2, tsz, 0x8, zd, zn);
}

// SVE insert SIMD&FP scalar register
void CodeGenerator::SveInsSimdFpSclarReg(const _ZReg &zdn, const VRegSc &vm) {
  uint32_t size = genSize(zdn);
  SvePerVecUnpred(size, 0x14, 0xe, zdn, vm);
}

// SVE insert general register
void CodeGenerator::SveInsGeneralReg(const _ZReg &zdn, const RReg &rm) {
  uint32_t size = genSize(zdn);
  SvePerVecUnpred(size, 0x4, 0xe, zdn, rm);
}

// SVE reverse vector elements
void CodeGenerator::SveRevVecElem(const _ZReg &zd, const _ZReg &zn) {
  uint32_t size = genSize(zd);
  SvePerVecUnpred(size, 0x18, 0xe, zd, zn);
}

// SVE table lookup
void CodeGenerator::SveTableLookup(const _ZReg &zd, const _ZReg &zn, const _ZReg &zm) {
  uint32_t size = genSize(zd);
  SvePerVecUnpred(size, zm.getIdx(), 0xc, zd, zn);
}

// SVE unpack vector elements
void CodeGenerator::SveUnpackVecElem(uint32_t U, uint32_t H, const _ZReg &zd, const _ZReg &zn) {
  uint32_t size = genSize(zd);
  SvePerVecUnpred(size, (0x10 | (U << 1) | H), 0xe, zd, zn);
}

// SVE permute predicate elements
void CodeGenerator::SvePermutePredElem(uint32_t opc, uint32_t H, const _PReg &pd, const _PReg &pn, const _PReg &pm) {
  uint32_t size = genSize(pd);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(1, 21), F(pm.getIdx(), 16), F(2, 13), F(opc, 11), F(H, 10), F(pn.getIdx(), 5), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE reverse predicate elements
void CodeGenerator::SveRevPredElem(const _PReg &pd, const _PReg &pn) {
  uint32_t size = genSize(pd);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(0xd1, 14), F(pn.getIdx(), 5), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE unpack predicate elements
void CodeGenerator::SveUnpackPredElem(uint32_t H, const _PReg &pd, const _PReg &pn) {
  uint32_t code = concat({F(0x5, 24), F(3, 20), F(H, 16), F(1, 14), F(pn.getIdx(), 5), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE permute vector elements
void CodeGenerator::SvePermuteVecElem(uint32_t opc, const _ZReg &zd, const _ZReg &zn, const _ZReg &zm) {
  uint32_t size = genSize(zd);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(1, 21), F(zm.getIdx(), 16), F(3, 13), F(opc, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE compress active elements
void CodeGenerator::SveCompressActElem(const _ZReg &zd, const _PReg &pg, const _ZReg &zn) {
  uint32_t size = genSize(zd);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(1, 21), F(0xc, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE conditionally broaccast element to vector
void CodeGenerator::SveCondBcElemToVec(uint32_t B, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm) {
  uint32_t size = genSize(zdn);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(0x14, 17), F(B, 16), F(0x4, 13), F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(zdn.getIdx(), 0)});
  dd(code);
}

// SVE conditionally extract element to SIMD&FP scalar
void CodeGenerator::SveCondExtElemToSimdFpScalar(uint32_t B, const VRegSc &vdn, const _PReg &pg, const _ZReg &zm) {
  uint32_t size = genSize(vdn);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(0x15, 17), F(B, 16), F(0x4, 13), F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(vdn.getIdx(), 0)});
  dd(code);
}

// SVE conditionally extract element to general Reg
void CodeGenerator::SveCondExtElemToGeneralReg(uint32_t B, const RReg &rdn, const _PReg &pg, const _ZReg &zm) {
  uint32_t size = genSize(zm);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(0x18, 17), F(B, 16), F(0x5, 13), F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(rdn.getIdx(), 0)});
  dd(code);
}

// SVE copy SIMD&FP scalar register to vector (predicated)
void CodeGenerator::SveCopySimdFpScalarToVecPred(const _ZReg &zd, const _PReg &pg, const VRegSc &vn) {
  uint32_t size = genSize(zd);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(0x10, 17), F(0x4, 13), F(pg.getIdx(), 10), F(vn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE copy general register to vector (predicated)
void CodeGenerator::SveCopyGeneralRegToVecPred(const _ZReg &zd, const _PReg &pg, const RReg &rn) {
  uint32_t size = genSize(zd);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(0x14, 17), F(0x5, 13), F(pg.getIdx(), 10), F(rn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE extract element to SIMD&FP scalar register
void CodeGenerator::SveExtElemToSimdFpScalar(uint32_t B, const VRegSc &vd, const _PReg &pg, const _ZReg &zn) {
  uint32_t size = genSize(vd);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(0x11, 17), F(B, 16), F(0x4, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// SVE extract element to general register
void CodeGenerator::SveExtElemToGeneralReg(uint32_t B, const RReg &rd, const _PReg &pg, const _ZReg &zn) {
  uint32_t size = genSize(zn);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(0x10, 17), F(B, 16), F(0x5, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// SVE reverse within elements
void CodeGenerator::SveRevWithinElem(uint32_t opc, const _ZReg &zd, const _PReg &pg, const _ZReg &zn) {
  uint32_t size = genSize(zd);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(0x9, 18), F(opc, 16), F(0x4, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE vector splice
void CodeGenerator::SveSelVecSplice(const _ZReg &zd, const _PReg &pg, const _ZReg &zn) {
  uint32_t size = genSize(zd);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(0xb, 18), F(0x4, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE select vector elements (predicated)
void CodeGenerator::SveSelVecElemPred(const _ZReg &zd, const _PReg &pg, const _ZReg &zn, const _ZReg &zm) {
  uint32_t size = genSize(zd);
  uint32_t code = concat({F(0x5, 24), F(size, 22), F(1, 21), F(zm.getIdx(), 16), F(0x3, 14), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE Integer Compare - Vector Group
void CodeGenerator::SveIntCompVecGrp(uint32_t opc, uint32_t ne, const _PReg &pd, const _PReg &pg, const _ZReg &zn, const _ZReg &zm) {
  uint32_t size = genSize(pd);
  uint32_t code = concat({F(0x24, 24), F(size, 22), F(zm.getIdx(), 16), F(opc, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(ne, 4), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE integer compare vectors
void CodeGenerator::SveIntCompVec(uint32_t op, uint32_t o2, uint32_t ne, const _PReg &pd, const _PReg &pg, const _ZReg &zn, const _ZReg &zm) {
  uint32_t opc = (op << 2) | o2;
  SveIntCompVecGrp(opc, ne, pd, pg, zn, zm);
}

// SVE integer compare with wide elements
void CodeGenerator::SveIntCompWideElem(uint32_t op, uint32_t o2, uint32_t ne, const _PReg &pd, const _PReg &pg, const _ZReg &zn, const _ZReg &zm) {
  uint32_t opc = (op << 2) | 2 | o2;
  SveIntCompVecGrp(opc, ne, pd, pg, zn, zm);
}

// SVE integer compare with unsigned immediate
void CodeGenerator::SveIntCompUImm(uint32_t lt, uint32_t ne, const _PReg &pd, const _PReg &pg, const _ZReg &zn, uint32_t imm) {
  uint32_t size = genSize(pd);
  uint32_t imm7 = imm & ones(7);
  verifyIncRange(imm, 0, 127, ERR_ILLEGAL_IMM_RANGE);
  uint32_t code = concat({F(0x24, 24), F(size, 22), F(1, 21), F(imm7, 14), F(lt, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(ne, 4), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE predicate logical operations
void CodeGenerator::SvePredLOp(uint32_t op, uint32_t S, uint32_t o2, uint32_t o3, const _PReg &pd, const _PReg &pg, const _PReg &pn, const _PReg &pm) {
  uint32_t code = concat({F(0x25, 24), F(op, 23), F(S, 22), F(pm.getIdx(), 16), F(1, 14), F(pg.getIdx(), 10), F(o2, 9), F(pn.getIdx(), 5), F(o3, 4), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE propagate break from previous partition
void CodeGenerator::SvePropagateBreakPrevPtn(uint32_t op, uint32_t S, uint32_t B, const _PReg &pd, const _PReg &pg, const _PReg &pn, const _PReg &pm) {
  uint32_t code = concat({F(0x25, 24), F(op, 23), F(S, 22), F(pm.getIdx(), 16), F(3, 14), F(pg.getIdx(), 10), F(pn.getIdx(), 5), F(B, 4), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE partition break condition
void CodeGenerator::SvePartitionBreakCond(uint32_t B, uint32_t S, const _PReg &pd, const _PReg &pg, const _PReg &pn) {
  uint32_t M = (S == 1) ? 0 : pg.isM();
  uint32_t code = concat({F(0x25, 24), F(B, 23), F(S, 22), F(2, 19), F(1, 14), F(pg.getIdx(), 10), F(pn.getIdx(), 5), F(M, 4), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE propagate break to next partition
void CodeGenerator::SvePropagateBreakNextPart(uint32_t S, const _PReg &pdm, const _PReg &pg, const _PReg &pn) {
  uint32_t code = concat({F(0x25, 24), F(S, 22), F(3, 19), F(1, 14), F(pg.getIdx(), 10), F(pn.getIdx(), 5), F(pdm.getIdx(), 0)});
  dd(code);
}

// SVE predicate first active
void CodeGenerator::SvePredFirstAct(uint32_t op, uint32_t S, const _PReg &pdn, const _PReg &pg) {
  uint32_t code = concat({F(0x25, 24), F(op, 23), F(S, 22), F(3, 19), F(3, 14), F(pg.getIdx(), 5), F(pdn.getIdx(), 0)});
  dd(code);
}

// SVE predicate initialize
void CodeGenerator::SvePredInit(uint32_t S, const _PReg &pd, Pattern pat) {
  uint32_t size = genSize(pd);
  uint32_t code = concat({F(0x25, 24), F(size, 22), F(3, 19), F(S, 16), F(7, 13), F(pat, 5), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE predicate next active
void CodeGenerator::SvePredNextAct(const _PReg &pdn, const _PReg &pg) {
  uint32_t size = genSize(pdn);
  uint32_t code = concat({F(0x25, 24), F(size, 22), F(3, 19), F(0xe, 13), F(1, 10), F(pg.getIdx(), 5), F(pdn.getIdx(), 0)});
  dd(code);
}

// SVE predicate read from FFR (predicate)
void CodeGenerator::SvePredReadFFRPred(uint32_t op, uint32_t S, const _PReg &pd, const _PReg &pg) {
  uint32_t code = concat({F(0x25, 24), F(op, 23), F(S, 22), F(3, 19), F(0xf, 12), F(pg.getIdx(), 5), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE predicate read from FFR (unpredicate)
void CodeGenerator::SvePredReadFFRUnpred(uint32_t op, uint32_t S, const _PReg &pd) {
  uint32_t code = concat({F(0x25, 24), F(op, 23), F(S, 22), F(3, 19), F(0x1f, 12), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE predicate test
void CodeGenerator::SvePredTest(uint32_t op, uint32_t S, uint32_t opc2, const _PReg &pg, const _PReg &pn) {
  uint32_t code = concat({F(0x25, 24), F(op, 23), F(S, 22), F(2, 19), F(3, 14), F(pg.getIdx(), 10), F(pn.getIdx(), 5), F(opc2, 0)});
  dd(code);
}

// SVE predicate zero
void CodeGenerator::SvePredZero(uint32_t op, uint32_t S, const _PReg &pd) {
  uint32_t code = concat({F(0x25, 24), F(op, 23), F(S, 22), F(3, 19), F(7, 13), F(1, 10), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE integer compare with signed immediate
void CodeGenerator::SveIntCompSImm(uint32_t op, uint32_t o2, uint32_t ne, const _PReg &pd, const _PReg &pg, const _ZReg &zn, int32_t imm) {
  uint32_t size = genSize(pd);
  uint32_t imm5 = imm & ones(5);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(imm, -16, 15, ERR_ILLEGAL_IMM_RANGE, true);
  uint32_t code = concat({F(0x25, 24), F(size, 22), F(imm5, 16), F(op, 15), F(o2, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(ne, 4), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE predicate count
void CodeGenerator::SvePredCount(uint32_t opc, uint32_t o2, const RReg &rd, const _PReg &pg, const _PReg &pn) {
  uint32_t size = genSize(pn);
  uint32_t code = concat({F(0x25, 24), F(size, 22), F(1, 21), F(opc, 16), F(2, 14), F(pg.getIdx(), 10), F(o2, 9), F(pn.getIdx(), 5), F(rd.getIdx(), 0)});
  dd(code);
}

// SVE Inc/Dec by Predicate Count Group
void CodeGenerator::SveIncDecPredCount(uint32_t size, uint32_t op, uint32_t D, uint32_t opc2, uint32_t type1, uint32_t type2, const Reg &rdn, const _PReg &pg) {
  uint32_t code = concat({F(0x25, 24), F(size, 22), F(type1, 18), F(op, 17), F(D, 16), F(type2, 11), F(opc2, 9), F(pg.getIdx(), 5), F(rdn.getIdx(), 0)});
  dd(code);
}

// SVE inc/dec register by predicate count
void CodeGenerator::SveIncDecRegByPredCount(uint32_t op, uint32_t D, uint32_t opc2, const RReg &rdn, const _PReg &pg) {
  uint32_t size = genSize(pg);
  SveIncDecPredCount(size, op, D, opc2, 0xb, 0x11, rdn, pg);
}

// SVE inc/dec vector by predicate count
void CodeGenerator::SveIncDecVecByPredCount(uint32_t op, uint32_t D, uint32_t opc2, const _ZReg &zdn, const _PReg &pg) {
  uint32_t size = genSize(zdn);
  SveIncDecPredCount(size, op, D, opc2, 0xb, 0x10, zdn, pg);
}

// SVE saturating inc/dec register by predicate count
void CodeGenerator::SveSatuIncDecRegByPredCount(uint32_t D, uint32_t U, uint32_t op, const RReg &rdn, const _PReg &pg) {
  uint32_t sf = genSf(rdn);
  uint32_t size = genSize(pg);
  SveIncDecPredCount(size, D, U, ((sf << 1) | op), 0xa, 0x11, rdn, pg);
}

// SVE saturating inc/dec vector by predicate count
void CodeGenerator::SveSatuIncDecVecByPredCount(uint32_t D, uint32_t U, uint32_t opc, const _ZReg &zdn, const _PReg &pg) {
  uint32_t size = genSize(zdn);
  SveIncDecPredCount(size, D, U, opc, 0xa, 0x10, zdn, pg);
}

// SVE FFR initialise
void CodeGenerator::SveFFRInit(uint32_t opc) {
  uint32_t code = concat({F(0x25, 24), F(opc, 22), F(0xb, 18), F(0x24, 10)});
  dd(code);
}

// SVE FFR write from predicate
void CodeGenerator::SveFFRWritePred(uint32_t opc, const _PReg &pn) {
  uint32_t code = concat({F(0x25, 24), F(opc, 22), F(0xa, 18), F(0x24, 10), F(pn.getIdx(), 5)});
  dd(code);
}

// SVE conditionally terminate scalars
void CodeGenerator::SveCondTermScalars(uint32_t op, uint32_t ne, const RReg &rn, const RReg &rm) {
  uint32_t sz = genSf(rn);
  uint32_t code = concat({F(0x25, 24), F(op, 23), F(sz, 22), F(1, 21), F(rm.getIdx(), 16), F(0x8, 10), F(rn.getIdx(), 5), F(ne, 4)});
  dd(code);
}

// SVE integer compare scalar count and limit
void CodeGenerator::SveIntCompScalarCountAndLimit(uint32_t U, uint32_t lt, uint32_t eq, const _PReg &pd, const RReg &rn, const RReg &rm) {
  uint32_t size = genSize(pd);
  uint32_t sf = genSf(rn);
  uint32_t code = concat({F(0x25, 24), F(size, 22), F(1, 21), F(rm.getIdx(), 16), F(sf, 12), F(U, 11), F(lt, 10), F(rn.getIdx(), 5), F(eq, 4), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE broadcast floating-point immediate (unpredicated)
void CodeGenerator::SveBcFpImmUnpred(uint32_t opc, uint32_t o2, const _ZReg &zd, double imm) {
  uint32_t size = genSize(zd);
  uint32_t imm8 = compactImm(imm, zd.getBit());
  uint32_t code = concat({F(0x25, 24), F(size, 22), F(7, 19), F(opc, 17), F(7, 14), F(o2, 13), F(imm8, 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE broadcast integer immediate (unpredicated)
void CodeGenerator::SveBcIntImmUnpred(uint32_t opc, const _ZReg &zd, int32_t imm, ShMod mod, uint32_t sh) {
  verifyIncList(mod, {LSL}, ERR_ILLEGAL_SHMOD);
  verifyIncList(sh, {0, 8}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(imm, -128, 127, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t size = genSize(zd);
  uint32_t imm8 = imm & ones(8);
  uint32_t code = concat({F(0x25, 24), F(size, 22), F(7, 19), F(opc, 17), F(3, 14), F((sh == 8), 13), F(imm8, 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE integer add/subtract immediate (unpredicated)
void CodeGenerator::SveIntAddSubImmUnpred(uint32_t opc, const _ZReg &zdn, uint32_t imm, ShMod mod, uint32_t sh) {
  verifyIncList(mod, {LSL}, ERR_ILLEGAL_SHMOD);
  verifyIncList(sh, {0, 8}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(imm, 0, 255, ERR_ILLEGAL_IMM_RANGE);

  uint32_t size = genSize(zdn);
  uint32_t imm8 = imm & ones(8);
  uint32_t code = concat({F(0x25, 24), F(size, 22), F(4, 19), F(opc, 16), F(3, 14), F((sh == 8), 13), F(imm8, 5), F(zdn.getIdx(), 0)});
  dd(code);
}

// SVE integer min/max immediate (unpredicated)
void CodeGenerator::SveIntMinMaxImmUnpred(uint32_t opc, uint32_t o2, const _ZReg &zdn, int32_t imm) {
  if ((opc & 0x1))
    verifyIncRange(imm, 0, 255, ERR_ILLEGAL_IMM_RANGE);
  else
    verifyIncRange(imm, -128, 127, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t size = genSize(zdn);
  uint32_t imm8 = imm & ones(8);
  uint32_t code = concat({F(0x25, 24), F(size, 22), F(5, 19), F(opc, 16), F(3, 14), F(o2, 13), F(imm8, 5), F(zdn.getIdx(), 0)});
  dd(code);
}

// SVE integer multiply immediate (unpredicated)
void CodeGenerator::SveIntMultImmUnpred(uint32_t opc, uint32_t o2, const _ZReg &zdn, int32_t imm) {
  uint32_t size = genSize(zdn);
  uint32_t imm8 = imm & ones(8);
  verifyIncRange(imm, -128, 127, ERR_ILLEGAL_IMM_RANGE, true);
  uint32_t code = concat({F(0x25, 24), F(size, 22), F(6, 19), F(opc, 16), F(3, 14), F(o2, 13), F(imm8, 5), F(zdn.getIdx(), 0)});
  dd(code);
}

// SVE integer dot product (unpredicated)
void CodeGenerator::SveIntDotProdcutUnpred(uint32_t U, const _ZReg &zda, const _ZReg &zn, const _ZReg &zm) {
  uint32_t size = genSize(zda);
  uint32_t code = concat({F(0x44, 24), F(size, 22), F(zm.getIdx(), 16), F(U, 10), F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
  dd(code);
}

// SVE integer dot product (indexed)
void CodeGenerator::SveIntDotProdcutIndexed(uint32_t size, uint32_t U, const _ZReg &zda, const _ZReg &zn, const ZRegElem &zm) {
  uint32_t zm_idx = zm.getIdx();
  uint32_t zm_eidx = zm.getElemIdx();
  uint32_t opc = (size == 2) ? (((zm_eidx & ones(2)) << 3) | zm_idx) : (((zm_eidx & ones(1)) << 4) | zm_idx);

  verifyIncRange(zm_eidx, 0, (size == 2) ? 3 : 1, ERR_ILLEGAL_REG_ELEM_IDX);
  verifyIncRange(zm_idx, 0, (size == 2) ? 7 : 15, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(0x44, 24), F(size, 22), F(1, 21), F(opc, 16), F(U, 10), F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
  dd(code);
}

// SVE floating-point complex add (predicated)
void CodeGenerator::SveFpComplexAddPred(const _ZReg &zdn, const _PReg &pg, const _ZReg &zm, uint32_t ct) {
  uint32_t size = genSize(zdn);
  uint32_t rot = (ct == 270) ? 1 : 0;
  verifyIncList(ct, {90, 270}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x64, 24), F(size, 22), F(rot, 16), F(1, 15), F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(zdn.getIdx(), 0)});
  dd(code);
}

// SVE floating-point complex multiply-add (predicated)
void CodeGenerator::SveFpComplexMultAddPred(const _ZReg &zda, const _PReg &pg, const _ZReg &zn, const _ZReg &zm, uint32_t ct) {
  uint32_t size = genSize(zda);
  uint32_t rot = (ct / 90);
  verifyIncList(ct, {0, 90, 180, 270}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x64, 24), F(size, 22), F(zm.getIdx(), 16), F(rot, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
  dd(code);
}

// SVE floating-point multiply-add (indexed)
void CodeGenerator::SveFpMultAddIndexed(uint32_t op, const _ZReg &zda, const _ZReg &zn, const ZRegElem &zm) {
  uint32_t zm_idx = zm.getIdx();
  uint32_t zm_bit = zm.getBit();
  uint32_t zm_eidx = zm.getElemIdx();
  uint32_t size = (zm_bit == 16) ? (0 | field(zm_eidx, 2, 2)) : genSize(zda);
  uint32_t opc = (zm_bit == 64) ? (((zm_eidx & ones(1)) << 4) | zm_idx) : (((zm_eidx & ones(2)) << 3) | zm_idx);

  verifyIncRange(zm_eidx, 0, ((zm_bit == 16) ? 7 : (zm_bit == 32) ? 3 : 1), ERR_ILLEGAL_REG_ELEM_IDX);
  verifyIncRange(zm_eidx, 0, ((zm_bit == 64) ? 15 : 7), ERR_ILLEGAL_REG_ELEM_IDX);

  uint32_t code = concat({F(0x64, 24), F(size, 22), F(1, 21), F(opc, 16), F(op, 10), F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
  dd(code);
}

// SVE floating-point complex multiply-add (indexed)
void CodeGenerator::SveFpComplexMultAddIndexed(const _ZReg &zda, const _ZReg &zn, const ZRegElem &zm, uint32_t ct) {
  uint32_t size = genSize(zda) + 1;
  uint32_t zm_idx = zm.getIdx();
  uint32_t zm_eidx = zm.getElemIdx();
  uint32_t opc = (size == 2) ? (((zm_eidx & ones(2)) << 3) | zm_idx) : (((zm_eidx & ones(1)) << 4) | zm_idx);

  verifyIncRange(zm_eidx, 0, (size == 2) ? 3 : 1, ERR_ILLEGAL_REG_ELEM_IDX);
  verifyIncRange(zm_idx, 0, (size == 2) ? 7 : 15, ERR_ILLEGAL_REG_IDX);
  verifyIncList(ct, {0, 90, 180, 270}, ERR_ILLEGAL_CONST_VALUE);

  uint32_t rot = (ct / 90);
  uint32_t code = concat({F(0x64, 24), F(size, 22), F(1, 21), F(opc, 16), F(1, 12), F(rot, 10), F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
  dd(code);
}

// SVE floating-point multiply (indexed)
void CodeGenerator::SveFpMultIndexed(const _ZReg &zd, const _ZReg &zn, const ZRegElem &zm) {
  uint32_t zm_idx = zm.getIdx();
  uint32_t zm_bit = zm.getBit();
  uint32_t zm_eidx = zm.getElemIdx();
  uint32_t size = (zm_bit == 16) ? (0 | field(zm_eidx, 2, 2)) : genSize(zd);
  uint32_t opc = (zm_bit == 64) ? (((zm_eidx & ones(1)) << 4) | zm_idx) : (((zm_eidx & ones(2)) << 3) | zm_idx);

  verifyIncRange(zm_eidx, 0, ((zm_bit == 16) ? 7 : (zm_bit == 32) ? 3 : 1), ERR_ILLEGAL_REG_ELEM_IDX);
  verifyIncRange(zm_eidx, 0, ((zm_bit == 64) ? 15 : 7), ERR_ILLEGAL_REG_ELEM_IDX);

  uint32_t code = concat({F(0x64, 24), F(size, 22), F(1, 21), F(opc, 16), F(1, 13), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE floating-point recursive reduction
void CodeGenerator::SveFpRecurReduct(uint32_t opc, const VRegSc vd, const _PReg &pg, const _ZReg &zn) {
  uint32_t size = genSize(vd);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x65, 24), F(size, 22), F(opc, 16), F(1, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(vd.getIdx(), 0)});
  dd(code);
}

// SVE floating-point reciprocal estimate unpredicated
void CodeGenerator::SveFpReciproEstUnPred(uint32_t opc, const _ZReg &zd, const _ZReg &zn) {
  uint32_t size = genSize(zd);
  uint32_t code = concat({F(0x65, 24), F(size, 22), F(1, 19), F(opc, 16), F(3, 12), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE floating-point compare with zero
void CodeGenerator::SveFpCompWithZero(uint32_t eq, uint32_t lt, uint32_t ne, const _PReg &pd, const _PReg &pg, const _ZReg &zn, double zero) {
  uint32_t size = genSize(pd);
  verifyIncList(std::lround(zero * 10), {0}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x65, 24), F(size, 22), F(1, 20), F(eq, 17), F(lt, 16), F(1, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(ne, 4), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE floating-point serial resuction (predicated)
void CodeGenerator::SveFpSerialReductPred(uint32_t opc, const VRegSc vdn, const _PReg &pg, const _ZReg &zm) {
  uint32_t size = genSize(vdn);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x65, 24), F(size, 22), F(3, 19), F(opc, 16), F(1, 13), F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(vdn.getIdx(), 0)});
  dd(code);
}

// SVE floating-point arithmetic (unpredicated)
void CodeGenerator::SveFpArithmeticUnpred(uint32_t opc, const _ZReg &zd, const _ZReg &zn, const _ZReg &zm) {
  uint32_t size = genSize(zd);
  uint32_t code = concat({F(0x65, 24), F(size, 22), F(zm.getIdx(), 16), F(opc, 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE floating-point arithmetic (predicated)
void CodeGenerator::SveFpArithmeticPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm) {
  uint32_t size = genSize(zdn);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x65, 24), F(size, 22), F(opc, 16), F(4, 13), F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(zdn.getIdx(), 0)});
  dd(code);
}

// SVE floating-point arithmetic with immediate (predicated)
void CodeGenerator::SveFpArithmeticImmPred(uint32_t opc, const _ZReg &zdn, const _PReg &pg, float ct) {
  uint32_t size = genSize(zdn);
  uint32_t i1 = (std::lround(ct * 10) < 10) ? 0 : 1;
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

  if (opc == 0 || opc == 1 || opc == 3)
    verifyIncList(std::lround(ct * 10), {5, 10}, ERR_ILLEGAL_CONST_VALUE);
  else if (opc == 2)
    verifyIncList(std::lround(ct * 10), {5, 20}, ERR_ILLEGAL_CONST_VALUE);
  else
    verifyIncList(std::lround(ct * 10), {0, 10}, ERR_ILLEGAL_CONST_VALUE);

  uint32_t code = concat({F(0x65, 24), F(size, 22), F(3, 19), F(opc, 16), F(4, 13), F(pg.getIdx(), 10), F(i1, 5), F(zdn.getIdx(), 0)});
  dd(code);
}

// SVE floating-point trig multiply-add coefficient
void CodeGenerator::SveFpTrigMultAddCoef(const _ZReg &zdn, const _ZReg &zm, uint32_t imm) {
  uint32_t size = genSize(zdn);
  uint32_t imm3 = imm & ones(3);
  verifyIncRange(imm, 0, 7, ERR_ILLEGAL_IMM_RANGE);
  uint32_t code = concat({F(0x65, 24), F(size, 22), F(2, 19), F(imm3, 16), F(1, 15), F(zm.getIdx(), 5), F(zdn.getIdx(), 0)});
  dd(code);
}

// SVE floating-point convert precision
void CodeGenerator::SveFpCvtPrecision(uint32_t opc, uint32_t opc2, const _ZReg &zd, const _PReg &pg, const _ZReg &zn) {
  uint32_t code = concat({F(0x65, 24), F(opc, 22), F(1, 19), F(opc2, 16), F(5, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE floating-point convert to integer
void CodeGenerator::SveFpCvtToInt(uint32_t opc, uint32_t opc2, uint32_t U, const _ZReg &zd, const _PReg &pg, const _ZReg &zn) {
  uint32_t code = concat({F(0x65, 24), F(opc, 22), F(3, 19), F(opc2, 17), F(U, 16), F(5, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE floating-point round to integral value
void CodeGenerator::SveFpRoundToIntegral(uint32_t opc, const _ZReg &zd, const _PReg &pg, const _ZReg &zn) {
  uint32_t size = genSize(zd);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x65, 24), F(size, 22), F(opc, 16), F(5, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE floating-point unary operations
void CodeGenerator::SveFpUnaryOp(uint32_t opc, const _ZReg &zd, const _PReg &pg, const _ZReg &zn) {
  uint32_t size = genSize(zd);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x65, 24), F(size, 22), F(3, 18), F(opc, 16), F(5, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE integer convert to floationg-point
void CodeGenerator::SveIntCvtToFp(uint32_t opc, uint32_t opc2, uint32_t U, const _ZReg &zd, const _PReg &pg, const _ZReg &zn) {
  uint32_t code = concat({F(0x65, 24), F(opc, 22), F(2, 19), F(opc2, 17), F(U, 16), F(5, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zd.getIdx(), 0)});
  dd(code);
}

// SVE floationg-point compare vectors
void CodeGenerator::SveFpCompVec(uint32_t op, uint32_t o2, uint32_t o3, const _PReg &pd, const _PReg &pg, const _ZReg &zn, const _ZReg &zm) {
  uint32_t size = genSize(pd);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x65, 24), F(size, 22), F(zm.getIdx(), 16), F(op, 15), F(1, 14), F(o2, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(o3, 4), F(pd.getIdx(), 0)});
  dd(code);
}

// SVE floationg-point multiply-accumulate writing addend
void CodeGenerator::SveFpMultAccumAddend(uint32_t opc, const _ZReg &zda, const _PReg &pg, const _ZReg &zn, const _ZReg &zm) {
  uint32_t size = genSize(zda);
  uint32_t code = concat({F(0x65, 24), F(size, 22), F(1, 21), F(zm.getIdx(), 16), F(opc, 13), F(pg.getIdx(), 10), F(zn.getIdx(), 5), F(zda.getIdx(), 0)});
  dd(code);
}

// SVE floationg-point multiply-accumulate writing multiplicand
void CodeGenerator::SveFpMultAccumMulti(uint32_t opc, const _ZReg &zdn, const _PReg &pg, const _ZReg &zm, const _ZReg &za) {
  uint32_t size = genSize(zdn);
  uint32_t code = concat({F(0x65, 24), F(size, 22), F(1, 21), F(za.getIdx(), 16), F(1, 15), F(opc, 13), F(pg.getIdx(), 10), F(zm.getIdx(), 5), F(zdn.getIdx(), 0)});
  dd(code);
}

// SVE 32-bit gather load (scalar plus 32-bit unscaled offsets)
void CodeGenerator::Sve32GatherLdSc32U(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc32U &adr) {
  uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x42, 25), F(msz, 23), F(xs, 22), F(adr.getZm().getIdx(), 16), F(U, 14), F(ff, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 32-bit gather load (vector plus immediate)
void CodeGenerator::Sve32GatherLdVecImm(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  uint32_t imm5 = (adr.getImm() >> msz) & ones(5);

  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(adr.getImm(), 0, 31 * (1 << msz), ERR_ILLEGAL_IMM_RANGE);

  uint32_t code = concat({F(0x42, 25), F(msz, 23), F(1, 21), F(imm5, 16), F(1, 15), F(U, 14), F(ff, 13), F(pg.getIdx(), 10), F(adr.getZn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 32-bit gather load halfwords (scalar plus 32-bit scaled offsets)
void CodeGenerator::Sve32GatherLdHSc32S(uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc32S &adr) {
  uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x42, 25), F(1, 23), F(xs, 22), F(1, 21), F(adr.getZm().getIdx(), 16), F(U, 14), F(ff, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 32-bit gather load words (scalar plus 32-bit scaled offsets)
void CodeGenerator::Sve32GatherLdWSc32S(uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc32S &adr) {
  uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x42, 25), F(2, 23), F(xs, 22), F(1, 21), F(adr.getZm().getIdx(), 16), F(U, 14), F(ff, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 32-bit gather prefetch (scalar plus 32-bit scaled offsets)
void CodeGenerator::Sve32GatherPfSc32S(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrSc32S &adr) {
  uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x42, 25), F(xs, 22), F(1, 21), F(adr.getZm().getIdx(), 16), F(msz, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(prfop_sve, 0)});
  dd(code);
}

// SVE 32-bit gather prefetch (vector plus immediate)
void CodeGenerator::Sve32GatherPfVecImm(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrVecImm32 &adr) {
  uint32_t imm5 = (adr.getImm() >> msz) & ones(5);

  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  verifyIncRange(adr.getImm(), 0, 31 * (1 << msz), ERR_ILLEGAL_IMM_RANGE);

  uint32_t code = concat({F(0x42, 25), F(msz, 23), F(imm5, 16), F(7, 13), F(pg.getIdx(), 10), F(adr.getZn().getIdx(), 5), F(prfop_sve, 0)});
  dd(code);
}

// SVE 32-bit contiguous prefetch (scalar plus immediate)
void CodeGenerator::Sve32ContiPfScImm(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrScImm &adr) {
  int32_t simm = adr.getSimm();
  uint32_t imm6 = simm & ones(6);
  verifyIncRange(simm, -32, 31, ERR_ILLEGAL_IMM_RANGE, true);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x42, 25), F(7, 22), F(imm6, 16), F(msz, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(prfop_sve, 0)});
  dd(code);
}

void CodeGenerator::Sve32ContiPfScImm(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrNoOfs &adr) {
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x42, 25), F(7, 22), F(0, 16), F(msz, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(prfop_sve, 0)});
  dd(code);
}

// SVE 32-bit contiguous prefetch (scalar plus scalar)
void CodeGenerator::Sve32ContiPfScSc(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrScSc &adr) {
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  uint32_t code = concat({F(0x42, 25), F(msz, 23), F(adr.getXm().getIdx(), 16), F(6, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(prfop_sve, 0)});
  dd(code);
}

// SVE load and broadcast element
void CodeGenerator::SveLoadAndBcElem(uint32_t dtypeh, uint32_t dtypel, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr) {
  uint32_t uimm = adr.getSimm();
  uint32_t dtype = dtypeh << 2 | dtypel;
  uint32_t size = genSize(dtype);
  uint32_t imm6 = (uimm >> size) & ones(6);

  verifyIncRange(uimm, 0, 63 * (1 << size), ERR_ILLEGAL_IMM_RANGE);
  verifyCond(
      uimm, [=](uint64_t x) { return (x % ((static_cast<uint64_t>(1)) << size)) == 0; }, ERR_ILLEGAL_IMM_COND);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(0x42, 25), F(dtypeh, 23), F(1, 22), F(imm6, 16), F(1, 15), F(dtypel, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveLoadAndBcElem(uint32_t dtypeh, uint32_t dtypel, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr) {
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x42, 25), F(dtypeh, 23), F(1, 22), F(0, 16), F(1, 15), F(dtypel, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE load predicate register
void CodeGenerator::SveLoadPredReg(const _PReg &pt, const AdrScImm &adr) {
  int32_t imm = adr.getSimm();
  uint32_t imm9h = field(imm, 8, 3);
  uint32_t imm9l = field(imm, 2, 0);
  verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);
  uint32_t code = concat({F(0x42, 25), F(3, 23), F(imm9h, 16), F(imm9l, 10), F(adr.getXn().getIdx(), 5), F(pt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveLoadPredReg(const _PReg &pt, const AdrNoOfs &adr) {
  uint32_t code = concat({F(0x42, 25), F(3, 23), F(0, 16), F(0, 10), F(adr.getXn().getIdx(), 5), F(pt.getIdx(), 0)});
  dd(code);
}

// SVE load predicate vector
void CodeGenerator::SveLoadPredVec(const _ZReg &zt, const AdrScImm &adr) {
  int32_t imm = adr.getSimm();
  uint32_t imm9h = field(imm, 8, 3);
  uint32_t imm9l = field(imm, 2, 0);

  verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t code = concat({F(0x42, 25), F(3, 23), F(imm9h, 16), F(1, 14), F(imm9l, 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveLoadPredVec(const _ZReg &zt, const AdrNoOfs &adr) {
  uint32_t code = concat({F(0x42, 25), F(3, 23), F(0, 16), F(1, 14), F(0, 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE contiguous first-fault load (scalar plus scalar)
void CodeGenerator::SveContiFFLdScSc(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr) {
  if (adr.getInitMod()) {
    verifyIncList(adr.getSh(), {genSize(dtype)}, ERR_ILLEGAL_CONST_VALUE);
    verifyIncList(adr.getMod(), {LSL}, ERR_ILLEGAL_SHMOD);
  }

  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(dtype, 21), F(adr.getXm().getIdx(), 16), F(3, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveContiFFLdScSc(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr) {
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(dtype, 21), F(31, 16), F(3, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE contiguous load (scalar plus immediate)
void CodeGenerator::SveContiLdScImm(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr) {
  int32_t imm = adr.getSimm();
  uint32_t imm4 = imm & ones(4);
  verifyIncRange(imm, -8, 7, ERR_ILLEGAL_IMM_RANGE, true);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(dtype, 21), F(imm4, 16), F(5, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveContiLdScImm(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr) {
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(dtype, 21), F(0, 16), F(5, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE contiguous load (scalar plus scalar)
void CodeGenerator::SveContiLdScSc(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr) {
  verifyIncList(adr.getSh(), {genSize(dtype)}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(dtype, 21), F(adr.getXm().getIdx(), 16), F(2, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE contiguous non-fault load (scalar plus immediate)
void CodeGenerator::SveContiNFLdScImm(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr) {
  int32_t imm = adr.getSimm();
  uint32_t imm4 = imm & ones(4);
  verifyIncRange(imm, -8, 7, ERR_ILLEGAL_IMM_RANGE, true);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(dtype, 21), F(1, 20), F(imm4, 16), F(5, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveContiNFLdScImm(uint32_t dtype, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr) {
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(dtype, 21), F(1, 20), F(0, 16), F(5, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE contiguous non-temporal load (scalar plus immediate)
void CodeGenerator::SveContiNTLdScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr) {
  int32_t imm = adr.getSimm();
  uint32_t imm4 = imm & ones(4);
  verifyIncRange(imm, -8, 7, ERR_ILLEGAL_IMM_RANGE, true);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(msz, 23), F(imm4, 16), F(7, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveContiNTLdScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr) {
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(msz, 23), F(0, 16), F(7, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE contiguous non-temporal load (scalar plus scalar)
void CodeGenerator::SveContiNTLdScSc(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr) {
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(msz, 23), F(adr.getXm().getIdx(), 16), F(6, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE load and broadcast quadword (scalar plus immediate)
void CodeGenerator::SveLdBcQuadScImm(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr) {
  int32_t imm = adr.getSimm();
  uint32_t imm4 = (imm >> 4) & ones(4);
  verifyIncRange(imm, -128, 127, ERR_ILLEGAL_IMM_RANGE, true);
  verifyCond(
      imm, [](uint64_t x) { return (x % 16) == 0; }, ERR_ILLEGAL_IMM_COND);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(msz, 23), F(num, 21), F(imm4, 16), F(1, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveLdBcQuadScImm(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr) {
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(msz, 23), F(num, 21), F(0, 16), F(1, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE load and broadcast quadword (scalar plus scalar)
void CodeGenerator::SveLdBcQuadScSc(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr) {
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(msz, 23), F(num, 21), F(adr.getXm().getIdx(), 16), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE load multiple structures (scalar plus immediate)
void CodeGenerator::SveLdMultiStructScImm(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr) {
  int32_t imm = adr.getSimm();
  uint32_t imm4 = (imm / ((int32_t)num + 1)) & ones(4);

  verifyIncRange(imm, -8 * ((int32_t)num + 1), 7 * ((int32_t)num + 1), ERR_ILLEGAL_IMM_RANGE, true);
  verifyCond(
      std::abs(imm), [=](uint64_t x) { return (x % (num + 1)) == 0; }, ERR_ILLEGAL_IMM_COND);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(0x52, 25), F(msz, 23), F(num, 21), F(imm4, 16), F(7, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveLdMultiStructScImm(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr) {
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(msz, 23), F(num, 21), F(0, 16), F(7, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE load multiple structures (scalar plus scalar)
void CodeGenerator::SveLdMultiStructScSc(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr) {
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x52, 25), F(msz, 23), F(num, 21), F(adr.getXm().getIdx(), 16), F(6, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 64-bit gather load (scalar plus unpacked 32-bit scaled offsets)
void CodeGenerator::Sve64GatherLdSc32US(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc32US &adr) {
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x62, 25), F(msz, 23), F(xs, 22), F(1, 21), F(adr.getZm().getIdx(), 16), F(U, 14), F(ff, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 64-bit gather load (scalar plus 64-bit scaled offsets)
void CodeGenerator::Sve64GatherLdSc64S(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc64S &adr) {
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x62, 25), F(msz, 23), F(3, 21), F(adr.getZm().getIdx(), 16), F(1, 15), F(U, 14), F(ff, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 64-bit gather load (scalar plus 64-bit unscaled offsets)
void CodeGenerator::Sve64GatherLdSc64U(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc64U &adr) {
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x62, 25), F(msz, 23), F(2, 21), F(adr.getZm().getIdx(), 16), F(1, 15), F(U, 14), F(ff, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 64-bit gather load (scalar plus unpacked 32-bit unscaled offsets)
void CodeGenerator::Sve64GatherLdSc32UU(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrSc32UU &adr) {
  uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x62, 25), F(msz, 23), F(xs, 22), F(adr.getZm().getIdx(), 16), F(U, 14), F(ff, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 64-bit gather load (vector plus immeidate)
void CodeGenerator::Sve64GatherLdVecImm(uint32_t msz, uint32_t U, uint32_t ff, const _ZReg &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  uint32_t imm = adr.getImm();
  uint32_t imm5 = (imm >> msz) & ones(5);

  verifyIncRange(imm, 0, 31 * (1 << msz), ERR_ILLEGAL_IMM_RANGE);
  verifyCond(
      imm, [=](uint64_t x) { return (x % ((static_cast<uint64_t>(1)) << msz)) == 0; }, ERR_ILLEGAL_IMM_COND);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(0x62, 25), F(msz, 23), F(1, 21), F(imm5, 16), F(1, 15), F(U, 14), F(ff, 13), F(pg.getIdx(), 10), F(adr.getZn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 64-bit gather load (scalar plus 64-bit scaled offsets)
void CodeGenerator::Sve64GatherPfSc64S(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrSc64S &adr) {
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x62, 25), F(3, 21), F(adr.getZm().getIdx(), 16), F(1, 15), F(msz, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(prfop_sve, 0)});
  dd(code);
}

// SVE 64-bit gather load (scalar plus unpacked 32-bit scaled offsets)
void CodeGenerator::Sve64GatherPfSc32US(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrSc32US &adr) {
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x62, 25), F(xs, 22), F(1, 21), F(adr.getZm().getIdx(), 16), F(msz, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(prfop_sve, 0)});
  dd(code);
}

// SVE 64-bit gather load (vector plus immediate)
void CodeGenerator::Sve64GatherPfVecImm(PrfopSve prfop_sve, uint32_t msz, const _PReg &pg, const AdrVecImm64 &adr) {
  uint32_t imm = adr.getImm();
  uint32_t imm5 = (imm >> msz) & ones(5);

  verifyIncRange(imm, 0, 31 * (1 << msz), ERR_ILLEGAL_IMM_RANGE);
  verifyCond(
      imm, [=](uint64_t x) { return (x % ((static_cast<uint64_t>(1)) << msz)) == 0; }, ERR_ILLEGAL_IMM_COND);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(0x62, 25), F(msz, 23), F(imm5, 16), F(7, 13), F(pg.getIdx(), 10), F(adr.getZn().getIdx(), 5), F(prfop_sve, 0)});
  dd(code);
}

// SVE 32-bit scatter store (sclar plus 32-bit scaled offsets)
void CodeGenerator::Sve32ScatterStSc32S(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrSc32S &adr) {
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(3, 21), F(adr.getZm().getIdx(), 16), F(1, 15), F(xs, 14), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 32-bit scatter store (sclar plus 32-bit unscaled offsets)
void CodeGenerator::Sve32ScatterStSc32U(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrSc32U &adr) {
  uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(2, 21), F(adr.getZm().getIdx(), 16), F(1, 15), F(xs, 14), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 32-bit scatter store (vector plus immediate)
void CodeGenerator::Sve32ScatterStVecImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrVecImm32 &adr) {
  uint32_t imm = adr.getImm();
  uint32_t imm5 = (imm >> msz) & ones(5);

  verifyIncRange(imm, 0, 31 * (1 << msz), ERR_ILLEGAL_IMM_RANGE);
  verifyCond(
      imm, [=](uint64_t x) { return (x % ((static_cast<uint64_t>(1)) << msz)) == 0; }, ERR_ILLEGAL_IMM_COND);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(3, 21), F(imm5, 16), F(5, 13), F(pg.getIdx(), 10), F(adr.getZn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 64-bit scatter store (scalar plus 64-bit scaled offsets)
void CodeGenerator::Sve64ScatterStSc64S(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrSc64S &adr) {
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(1, 21), F(adr.getZm().getIdx(), 16), F(5, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 64-bit scatter store (scalar plus 64-bit unscaled offsets)
void CodeGenerator::Sve64ScatterStSc64U(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrSc64U &adr) {
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(adr.getZm().getIdx(), 16), F(5, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 64-bit scatter store (scalar plus unpacked 32-bit scaled offsets)
void CodeGenerator::Sve64ScatterStSc32US(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrSc32US &adr) {
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(1, 21), F(adr.getZm().getIdx(), 16), F(1, 15), F(xs, 14), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 64-bit scatter store (scalar plus unpacked 32-bit unscaled offsets)
void CodeGenerator::Sve64ScatterStSc32UU(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrSc32UU &adr) {
  uint32_t xs = (adr.getMod() == SXTW) ? 1 : 0;
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(adr.getZm().getIdx(), 16), F(1, 15), F(xs, 14), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE 64-bit scatter store (vector plus immediate)
void CodeGenerator::Sve64ScatterStVecImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrVecImm64 &adr) {
  uint32_t imm = adr.getImm();
  uint32_t imm5 = (imm >> msz) & ones(5);

  verifyIncRange(imm, 0, 31 * (1 << msz), ERR_ILLEGAL_IMM_RANGE);
  verifyCond(
      imm, [=](uint64_t x) { return (x % ((static_cast<uint64_t>(1)) << msz)) == 0; }, ERR_ILLEGAL_IMM_COND);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(2, 21), F(imm5, 16), F(5, 13), F(pg.getIdx(), 10), F(adr.getZn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE contiguous non-temporal store (scalar plus immediate)
void CodeGenerator::SveContiNTStScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr) {
  int32_t imm = adr.getSimm();
  uint32_t imm4 = imm & ones(4);
  verifyIncRange(imm, -8, 7, ERR_ILLEGAL_IMM_RANGE, true);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(1, 20), F(imm4, 16), F(7, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveContiNTStScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr) {
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(1, 20), F(0, 16), F(7, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE contiguous non-temporal store (scalar plus scalar)
void CodeGenerator::SveContiNTStScSc(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr) {
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(adr.getXm().getIdx(), 16), F(3, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE contiguous store (scalar plus immediate)
void CodeGenerator::SveContiStScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr) {
  uint32_t size = genSize(zt);
  int32_t imm = adr.getSimm();
  uint32_t imm4 = imm & ones(4);
  verifyIncRange(imm, -8, 7, ERR_ILLEGAL_IMM_RANGE, true);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(size, 21), F(imm4, 16), F(7, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveContiStScImm(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr) {
  uint32_t size = genSize(zt);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(size, 21), F(0, 16), F(7, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE contiguous store (scalar plus scalar)
void CodeGenerator::SveContiStScSc(uint32_t msz, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr) {
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  uint32_t size = genSize(zt);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(size, 21), F(adr.getXm().getIdx(), 16), F(2, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE store multipule structures (scalar plus immediate)
void CodeGenerator::SveStMultiStructScImm(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrScImm &adr) {
  int32_t imm = adr.getSimm();
  uint32_t imm4 = (imm / ((int32_t)num + 1)) & ones(4);

  verifyIncRange(imm, -8 * ((int32_t)num + 1), 7 * ((int32_t)num + 1), ERR_ILLEGAL_IMM_RANGE, true);
  verifyCond(
      std::abs(imm), [=](uint64_t x) { return (x % (num + 1)) == 0; }, ERR_ILLEGAL_IMM_COND);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);

  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(num, 21), F(1, 20), F(imm4, 16), F(7, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveStMultiStructScImm(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrNoOfs &adr) {
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(num, 21), F(1, 20), F(0, 16), F(7, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE store multipule structures (scalar plus scalar)
void CodeGenerator::SveStMultiStructScSc(uint32_t msz, uint32_t num, const _ZReg &zt, const _PReg &pg, const AdrScSc &adr) {
  verifyIncList(adr.getSh(), {msz}, ERR_ILLEGAL_CONST_VALUE);
  verifyIncRange(pg.getIdx(), 0, 7, ERR_ILLEGAL_REG_IDX);
  uint32_t code = concat({F(0x72, 25), F(msz, 23), F(num, 21), F(adr.getXm().getIdx(), 16), F(3, 13), F(pg.getIdx(), 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

// SVE store predicate register
void CodeGenerator::SveStorePredReg(const _PReg &pt, const AdrScImm &adr) {
  int32_t imm = adr.getSimm();
  uint32_t imm9h = field(imm, 8, 3);
  uint32_t imm9l = field(imm, 2, 0);

  verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t code = concat({F(0x72, 25), F(3, 23), F(imm9h, 16), F(imm9l, 10), F(adr.getXn().getIdx(), 5), F(pt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveStorePredReg(const _PReg &pt, const AdrNoOfs &adr) {
  uint32_t code = concat({F(0x72, 25), F(3, 23), F(0, 16), F(0, 10), F(adr.getXn().getIdx(), 5), F(pt.getIdx(), 0)});
  dd(code);
}

// SVE store predicate vector
void CodeGenerator::SveStorePredVec(const _ZReg &zt, const AdrScImm &adr) {
  int32_t imm = adr.getSimm();
  uint32_t imm9h = field(imm, 8, 3);
  uint32_t imm9l = field(imm, 2, 0);

  verifyIncRange(imm, -256, 255, ERR_ILLEGAL_IMM_RANGE, true);

  uint32_t code = concat({F(0x72, 25), F(3, 23), F(imm9h, 16), F(2, 13), F(imm9l, 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

void CodeGenerator::SveStorePredVec(const _ZReg &zt, const AdrNoOfs &adr) {
  uint32_t code = concat({F(0x72, 25), F(3, 23), F(0, 16), F(2, 13), F(0, 10), F(adr.getXn().getIdx(), 5), F(zt.getIdx(), 0)});
  dd(code);
}

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

void CodeGenerator::adr(const XReg &xd, const Label &label) { PCrelAddr(0, xd, label); }
void CodeGenerator::adr(const XReg &xd, const int64_t label) { PCrelAddr(0, xd, label); }
void CodeGenerator::adrp(const XReg &xd, const Label &label) { PCrelAddr(1, xd, label); }
void CodeGenerator::adrp(const XReg &xd, const int64_t label) { PCrelAddr(1, xd, label); }
void CodeGenerator::add(const WReg &rd, const WReg &rn, const uint32_t imm, const uint32_t sh) { AddSubImm(0, 0, rd, rn, imm, sh); }
void CodeGenerator::add(const XReg &rd, const XReg &rn, const uint32_t imm, const uint32_t sh) { AddSubImm(0, 0, rd, rn, imm, sh); }
void CodeGenerator::adds(const WReg &rd, const WReg &rn, const uint32_t imm, const uint32_t sh) { AddSubImm(0, 1, rd, rn, imm, sh); }
void CodeGenerator::adds(const XReg &rd, const XReg &rn, const uint32_t imm, const uint32_t sh) { AddSubImm(0, 1, rd, rn, imm, sh); }
void CodeGenerator::cmn(const WReg &rn, const uint32_t imm, const uint32_t sh) { AddSubImm(0, 1, WReg(31), rn, imm, sh); }
void CodeGenerator::cmn(const XReg &rn, const uint32_t imm, const uint32_t sh) { AddSubImm(0, 1, XReg(31), rn, imm, sh); }
void CodeGenerator::sub(const WReg &rd, const WReg &rn, const uint32_t imm, const uint32_t sh) { AddSubImm(1, 0, rd, rn, imm, sh); }
void CodeGenerator::sub(const XReg &rd, const XReg &rn, const uint32_t imm, const uint32_t sh) { AddSubImm(1, 0, rd, rn, imm, sh); }
void CodeGenerator::subs(const WReg &rd, const WReg &rn, const uint32_t imm, const uint32_t sh) { AddSubImm(1, 1, rd, rn, imm, sh); }
void CodeGenerator::subs(const XReg &rd, const XReg &rn, const uint32_t imm, const uint32_t sh) { AddSubImm(1, 1, rd, rn, imm, sh); }
void CodeGenerator::cmp(const WReg &rn, const uint32_t imm, const uint32_t sh) { AddSubImm(1, 1, WReg(31), rn, imm, sh); }
void CodeGenerator::cmp(const XReg &rn, const uint32_t imm, const uint32_t sh) { AddSubImm(1, 1, XReg(31), rn, imm, sh); }
void CodeGenerator::and_(const WReg &rd, const WReg &rn, const uint64_t imm) { LogicalImm(0, rd, rn, imm); }
void CodeGenerator::and_(const XReg &rd, const XReg &rn, const uint64_t imm) { LogicalImm(0, rd, rn, imm); }
void CodeGenerator::orr(const WReg &rd, const WReg &rn, const uint64_t imm) { LogicalImm(1, rd, rn, imm); }
void CodeGenerator::orr(const XReg &rd, const XReg &rn, const uint64_t imm) { LogicalImm(1, rd, rn, imm); }
void CodeGenerator::eor(const WReg &rd, const WReg &rn, const uint64_t imm) { LogicalImm(2, rd, rn, imm); }
void CodeGenerator::eor(const XReg &rd, const XReg &rn, const uint64_t imm) { LogicalImm(2, rd, rn, imm); }
void CodeGenerator::ands(const WReg &rd, const WReg &rn, const uint64_t imm) { LogicalImm(3, rd, rn, imm); }
void CodeGenerator::ands(const XReg &rd, const XReg &rn, const uint64_t imm) { LogicalImm(3, rd, rn, imm); }
void CodeGenerator::tst(const WReg &rn, const uint64_t imm) { LogicalImm(3, WReg(31), rn, imm, true); }
void CodeGenerator::tst(const XReg &rn, const uint64_t imm) { LogicalImm(3, XReg(31), rn, imm, true); }
void CodeGenerator::movn(const WReg &rd, const uint32_t imm, const uint32_t sh) { MvWideImm(0, rd, imm, sh); }
void CodeGenerator::movn(const XReg &rd, const uint32_t imm, const uint32_t sh) { MvWideImm(0, rd, imm, sh); }
void CodeGenerator::movz(const WReg &rd, const uint32_t imm, const uint32_t sh) { MvWideImm(2, rd, imm, sh); }
void CodeGenerator::movz(const XReg &rd, const uint32_t imm, const uint32_t sh) { MvWideImm(2, rd, imm, sh); }
void CodeGenerator::movk(const WReg &rd, const uint32_t imm, const uint32_t sh) { MvWideImm(3, rd, imm, sh); }
void CodeGenerator::movk(const XReg &rd, const uint32_t imm, const uint32_t sh) { MvWideImm(3, rd, imm, sh); }
void CodeGenerator::mov(const WReg &rd, const uint64_t imm) { MvImm(rd, imm); }
void CodeGenerator::mov(const XReg &rd, const uint64_t imm) { MvImm(rd, imm); }
void CodeGenerator::sbfm(const WReg &rd, const WReg &rn, const uint32_t immr, const uint32_t imms) { Bitfield(0, rd, rn, immr, imms); }
void CodeGenerator::sbfm(const XReg &rd, const XReg &rn, const uint32_t immr, const uint32_t imms) { Bitfield(0, rd, rn, immr, imms); }
void CodeGenerator::sbfiz(const WReg &rd, const WReg &rn, const uint32_t lsb, const uint32_t width) { Bitfield(0, rd, rn, (((-1) * lsb) % 32) & ones(6), width - 1); }
void CodeGenerator::sbfiz(const XReg &rd, const XReg &rn, const uint32_t lsb, const uint32_t width) { Bitfield(0, rd, rn, (((-1) * lsb) % 64) & ones(6), width - 1); }
void CodeGenerator::sbfx(const WReg &rd, const WReg &rn, const uint32_t lsb, const uint32_t width) { Bitfield(0, rd, rn, lsb, lsb + width - 1); }
void CodeGenerator::sbfx(const XReg &rd, const XReg &rn, const uint32_t lsb, const uint32_t width) { Bitfield(0, rd, rn, lsb, lsb + width - 1); }
void CodeGenerator::sxtb(const WReg &rd, const WReg &rn) { Bitfield(0, rd, rn, 0, 7); }
void CodeGenerator::sxtb(const XReg &rd, const WReg &rn) { Bitfield(0, rd, rn, 0, 7); }
void CodeGenerator::sxth(const WReg &rd, const WReg &rn) { Bitfield(0, rd, rn, 0, 15); }
void CodeGenerator::sxth(const XReg &rd, const WReg &rn) { Bitfield(0, rd, rn, 0, 15); }
void CodeGenerator::sxtw(const WReg &rd, const WReg &rn) { Bitfield(0, rd, rn, 0, 31); }
void CodeGenerator::sxtw(const XReg &rd, const WReg &rn) { Bitfield(0, rd, rn, 0, 31); }
void CodeGenerator::asr(const WReg &rd, const WReg &rn, const uint32_t immr) { Bitfield(0, rd, rn, immr, 31); }
void CodeGenerator::asr(const XReg &rd, const XReg &rn, const uint32_t immr) { Bitfield(0, rd, rn, immr, 63); }
void CodeGenerator::bfm(const WReg &rd, const WReg &rn, const uint32_t immr, const uint32_t imms) { Bitfield(1, rd, rn, immr, imms); }
void CodeGenerator::bfm(const XReg &rd, const XReg &rn, const uint32_t immr, const uint32_t imms) { Bitfield(1, rd, rn, immr, imms); }
void CodeGenerator::bfc(const WReg &rd, const uint32_t lsb, const uint32_t width) { Bitfield(1, rd, WReg(31), (((-1) * lsb) % 32) & ones(6), width - 1, false); }
void CodeGenerator::bfc(const XReg &rd, const uint32_t lsb, const uint32_t width) { Bitfield(1, rd, XReg(31), (((-1) * lsb) % 64) & ones(6), width - 1, false); }
void CodeGenerator::bfi(const WReg &rd, const WReg &rn, const uint32_t lsb, const uint32_t width) { Bitfield(1, rd, rn, (((-1) * lsb) % 32) & ones(6), width - 1); }
void CodeGenerator::bfi(const XReg &rd, const XReg &rn, const uint32_t lsb, const uint32_t width) { Bitfield(1, rd, rn, (((-1) * lsb) % 64) & ones(6), width - 1); }
void CodeGenerator::bfxil(const WReg &rd, const WReg &rn, const uint32_t lsb, const uint32_t width) { Bitfield(1, rd, rn, lsb, lsb + width - 1); }
void CodeGenerator::bfxil(const XReg &rd, const XReg &rn, const uint32_t lsb, const uint32_t width) { Bitfield(1, rd, rn, lsb, lsb + width - 1); }
void CodeGenerator::ubfm(const WReg &rd, const WReg &rn, const uint32_t immr, const uint32_t imms) { Bitfield(2, rd, rn, immr, imms); }
void CodeGenerator::ubfm(const XReg &rd, const XReg &rn, const uint32_t immr, const uint32_t imms) { Bitfield(2, rd, rn, immr, imms); }
void CodeGenerator::ubfiz(const WReg &rd, const WReg &rn, const uint32_t lsb, const uint32_t width) { Bitfield(2, rd, rn, (((-1) * lsb) % 32) & ones(6), width - 1); }
void CodeGenerator::ubfiz(const XReg &rd, const XReg &rn, const uint32_t lsb, const uint32_t width) { Bitfield(2, rd, rn, (((-1) * lsb) % 64) & ones(6), width - 1); }
void CodeGenerator::ubfx(const WReg &rd, const WReg &rn, const uint32_t lsb, const uint32_t width) { Bitfield(2, rd, rn, lsb, lsb + width - 1); }
void CodeGenerator::ubfx(const XReg &rd, const XReg &rn, const uint32_t lsb, const uint32_t width) { Bitfield(2, rd, rn, lsb, lsb + width - 1); }
void CodeGenerator::lsl(const WReg &rd, const WReg &rn, const uint32_t sh) { Bitfield(2, rd, rn, (((-1) * sh) % 32) & ones(6), 31 - sh); }
void CodeGenerator::lsl(const XReg &rd, const XReg &rn, const uint32_t sh) { Bitfield(2, rd, rn, (((-1) * sh) % 64) & ones(6), 63 - sh); }
void CodeGenerator::lsr(const WReg &rd, const WReg &rn, const uint32_t sh) { Bitfield(2, rd, rn, sh, 31); }
void CodeGenerator::lsr(const XReg &rd, const XReg &rn, const uint32_t sh) { Bitfield(2, rd, rn, sh, 63); }
void CodeGenerator::uxtb(const WReg &rd, const WReg &rn) { Bitfield(2, rd, rn, 0, 7); }
void CodeGenerator::uxtb(const XReg &rd, const XReg &rn) { Bitfield(2, rd, rn, 0, 7); }
void CodeGenerator::uxth(const WReg &rd, const WReg &rn) { Bitfield(2, rd, rn, 0, 15); }
void CodeGenerator::uxth(const XReg &rd, const XReg &rn) { Bitfield(2, rd, rn, 0, 15); }
void CodeGenerator::extr(const WReg &rd, const WReg &rn, const WReg &rm, const uint32_t imm) { Extract(0, 0, rd, rn, rm, imm); }
void CodeGenerator::extr(const XReg &rd, const XReg &rn, const XReg &rm, const uint32_t imm) { Extract(0, 0, rd, rn, rm, imm); }
void CodeGenerator::ror(const WReg &rd, const WReg &rn, const uint32_t imm) { Extract(0, 0, rd, rn, rn, imm); }
void CodeGenerator::ror(const XReg &rd, const XReg &rn, const uint32_t imm) { Extract(0, 0, rd, rn, rn, imm); }
void CodeGenerator::b(const Cond cond, const Label &label) { CondBrImm(cond, label); }
void CodeGenerator::b(const Cond cond, const int64_t label) { CondBrImm(cond, label); }
void CodeGenerator::svc(const uint32_t imm) { ExceptionGen(0, 0, 1, imm); }
void CodeGenerator::hvc(const uint32_t imm) { ExceptionGen(0, 0, 2, imm); }
void CodeGenerator::smc(const uint32_t imm) { ExceptionGen(0, 0, 3, imm); }
void CodeGenerator::brk(const uint32_t imm) { ExceptionGen(1, 0, 0, imm); }
void CodeGenerator::hlt(const uint32_t imm) { ExceptionGen(2, 0, 0, imm); }
void CodeGenerator::dcps1(const uint32_t imm) { ExceptionGen(5, 0, 1, imm); }
void CodeGenerator::dcps2(const uint32_t imm) { ExceptionGen(5, 0, 2, imm); }
void CodeGenerator::dcps3(const uint32_t imm) { ExceptionGen(5, 0, 3, imm); }
void CodeGenerator::hint(const uint32_t imm) { Hints(imm); }
void CodeGenerator::nop() { Hints(0, 0); }
void CodeGenerator::yield() { Hints(0, 1); }
void CodeGenerator::wfe() { Hints(0, 2); }
void CodeGenerator::wfi() { Hints(0, 3); }
void CodeGenerator::sev() { Hints(0, 4); }
void CodeGenerator::sevl() { Hints(0, 5); }
void CodeGenerator::xpaclri() { Hints(0, 7); }
void CodeGenerator::pacia1716() { Hints(1, 0); }
void CodeGenerator::pacib1716() { Hints(1, 2); }
void CodeGenerator::autia1716() { Hints(1, 4); }
void CodeGenerator::autib1716() { Hints(1, 6); }
void CodeGenerator::esb() { Hints(2, 0); }
void CodeGenerator::psb_csync() { Hints(2, 1); }
void CodeGenerator::tsb_csync() { Hints(2, 2); }
void CodeGenerator::csdb() { Hints(2, 4); }
void CodeGenerator::paciaz() { Hints(3, 0); }
void CodeGenerator::paciasp() { Hints(3, 1); }
void CodeGenerator::pacibz() { Hints(3, 2); }
void CodeGenerator::pacibsp() { Hints(3, 3); }
void CodeGenerator::autiaz() { Hints(3, 4); }
void CodeGenerator::autiasp() { Hints(3, 5); }
void CodeGenerator::autibz() { Hints(3, 6); }
void CodeGenerator::autibsp() { Hints(3, 7); }
void CodeGenerator::dsb(const BarOpt opt) { BarriersOpt(4, opt, 31); }
void CodeGenerator::dmb(const BarOpt opt) { BarriersOpt(5, opt, 31); }
void CodeGenerator::isb(const BarOpt opt) { BarriersOpt(6, opt, 31); }
void CodeGenerator::clrex(const uint32_t imm) { BarriersNoOpt(imm, 2, 31); }
void CodeGenerator::ssbb() { BarriersNoOpt(0, 4, 31); }
void CodeGenerator::pssbb() { BarriersNoOpt(4, 4, 31); }
void CodeGenerator::msr(const PStateField psfield, const uint32_t imm) { PState(psfield, imm); }
void CodeGenerator::cfinv() { PState(0, 0, 0); }
void CodeGenerator::sys(const uint32_t op1, const uint32_t CRn, const uint32_t CRm, const uint32_t op2, const XReg &rt) { SysInst(0, op1, CRn, CRm, op2, rt); }
void CodeGenerator::sysl(const XReg &rt, const uint32_t op1, const uint32_t CRn, const uint32_t CRm, const uint32_t op2) { SysInst(1, op1, CRn, CRm, op2, rt); }
void CodeGenerator::msr(const uint32_t op0, const uint32_t op1, const uint32_t CRn, const uint32_t CRm, const uint32_t op2, const XReg &rt) { SysRegMove(0, op0, op1, CRn, CRm, op2, rt); }
void CodeGenerator::mrs(const XReg &rt, const uint32_t op0, const uint32_t op1, const uint32_t CRn, const uint32_t CRm, const uint32_t op2) { SysRegMove(1, op0, op1, CRn, CRm, op2, rt); }
void CodeGenerator::ret() { UncondBrNoReg(2, 31, 0, 30, 0); }
void CodeGenerator::retaa() { UncondBrNoReg(2, 31, 2, 31, 31); }
void CodeGenerator::retab() { UncondBrNoReg(2, 31, 3, 31, 31); }
void CodeGenerator::eret() { UncondBrNoReg(4, 31, 0, 31, 0); }
void CodeGenerator::eretaa() { UncondBrNoReg(4, 31, 2, 31, 31); }
void CodeGenerator::eretab() { UncondBrNoReg(4, 31, 3, 31, 31); }
void CodeGenerator::drps() { UncondBrNoReg(5, 31, 0, 31, 0); }
void CodeGenerator::br(const XReg &rn) { UncondBr1Reg(0, 31, 0, rn, 0); }
void CodeGenerator::braaz(const XReg &rn) { UncondBr1Reg(0, 31, 2, rn, 31); }
void CodeGenerator::brabz(const XReg &rn) { UncondBr1Reg(0, 31, 3, rn, 31); }
void CodeGenerator::blr(const XReg &rn) { UncondBr1Reg(1, 31, 0, rn, 0); }
void CodeGenerator::blraaz(const XReg &rn) { UncondBr1Reg(1, 31, 2, rn, 31); }
void CodeGenerator::blrabz(const XReg &rn) { UncondBr1Reg(1, 31, 3, rn, 31); }
void CodeGenerator::ret(const XReg &rn) { UncondBr1Reg(2, 31, 0, rn, 0); }
void CodeGenerator::braa(const XReg &rn, const XReg &rm) { UncondBr2Reg(8, 31, 2, rn, rm); }
void CodeGenerator::brab(const XReg &rn, const XReg &rm) { UncondBr2Reg(8, 31, 3, rn, rm); }
void CodeGenerator::blraa(const XReg &rn, const XReg &rm) { UncondBr2Reg(9, 31, 2, rn, rm); }
void CodeGenerator::blrab(const XReg &rn, const XReg &rm) { UncondBr2Reg(9, 31, 3, rn, rm); }
void CodeGenerator::b(const Label &label) { UncondBrImm(0, label); }
void CodeGenerator::b(const int64_t label) { UncondBrImm(0, label); }
void CodeGenerator::bl(const Label &label) { UncondBrImm(1, label); }
void CodeGenerator::bl(const int64_t label) { UncondBrImm(1, label); }
void CodeGenerator::cbz(const WReg &rt, const Label &label) { CompareBr(0, rt, label); }
void CodeGenerator::cbz(const XReg &rt, const Label &label) { CompareBr(0, rt, label); }
void CodeGenerator::cbz(const WReg &rt, const int64_t label) { CompareBr(0, rt, label); }
void CodeGenerator::cbz(const XReg &rt, const int64_t label) { CompareBr(0, rt, label); }
void CodeGenerator::cbnz(const WReg &rt, const Label &label) { CompareBr(1, rt, label); }
void CodeGenerator::cbnz(const XReg &rt, const Label &label) { CompareBr(1, rt, label); }
void CodeGenerator::cbnz(const WReg &rt, const int64_t label) { CompareBr(1, rt, label); }
void CodeGenerator::cbnz(const XReg &rt, const int64_t label) { CompareBr(1, rt, label); }
void CodeGenerator::tbz(const WReg &rt, const uint32_t imm, const Label &label) { TestBr(0, rt, imm, label); }
void CodeGenerator::tbz(const XReg &rt, const uint32_t imm, const Label &label) { TestBr(0, rt, imm, label); }
void CodeGenerator::tbz(const WReg &rt, const uint32_t imm, const int64_t label) { TestBr(0, rt, imm, label); }
void CodeGenerator::tbz(const XReg &rt, const uint32_t imm, const int64_t label) { TestBr(0, rt, imm, label); }
void CodeGenerator::tbnz(const WReg &rt, const uint32_t imm, const Label &label) { TestBr(1, rt, imm, label); }
void CodeGenerator::tbnz(const XReg &rt, const uint32_t imm, const Label &label) { TestBr(1, rt, imm, label); }
void CodeGenerator::tbnz(const WReg &rt, const uint32_t imm, const int64_t label) { TestBr(1, rt, imm, label); }
void CodeGenerator::tbnz(const XReg &rt, const uint32_t imm, const int64_t label) { TestBr(1, rt, imm, label); }
void CodeGenerator::st1(const VReg8BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg4HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg2SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg1DList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg16BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg8HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg4SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg2DList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(0, 2, vt, adr); }
void CodeGenerator::ld1(const VReg8BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg4HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg2SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg1DList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg16BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg8HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg4SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg2DList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructForLd1St1(1, 2, vt, adr); }
void CodeGenerator::st4(const VReg8BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg4HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg2SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg16BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg8HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg4SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg2DList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st3(const VReg8BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg4HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg2SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg16BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg8HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg4SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg2DList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st2(const VReg8BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg4HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg2SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg16BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg8HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg4SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg2DList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::ld4(const VReg8BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg4HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg2SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg16BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg8HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg4SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg2DList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld3(const VReg8BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg4HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg2SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg16BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg8HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg4SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg2DList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld2(const VReg8BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg4HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg2SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg16BList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg8HList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg4SList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg2DList &vt, const AdrNoOfs &adr) { AdvSimdLdStMultiStructExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::st1(const VReg8BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg4HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg2SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg1DList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg16BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg8HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg4SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg2DList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(0, 2, vt, adr); }
void CodeGenerator::ld1(const VReg8BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg4HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg2SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg1DList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg16BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg8HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg4SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg2DList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegForLd1St1(1, 2, vt, adr); }
void CodeGenerator::st4(const VReg8BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg4HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg2SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg16BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg8HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg4SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg2DList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st3(const VReg8BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg4HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg2SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg16BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg8HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg4SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg2DList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st2(const VReg8BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg4HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg2SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg16BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg8HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg4SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg2DList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::ld4(const VReg8BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg4HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg2SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg16BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg8HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg4SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg2DList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld3(const VReg8BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg4HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg2SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg16BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg8HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg4SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg2DList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld2(const VReg8BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg4HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg2SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg16BList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg8HList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg4SList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg2DList &vt, const AdrPostReg &adr) { AdvSimdLdStMultiStructPostRegExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::st1(const VReg8BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg4HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg2SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg1DList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg16BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg8HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg4SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr); }
void CodeGenerator::st1(const VReg2DList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(0, 2, vt, adr); }
void CodeGenerator::ld1(const VReg8BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg4HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg2SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg1DList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg16BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg8HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg4SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr); }
void CodeGenerator::ld1(const VReg2DList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmForLd1St1(1, 2, vt, adr); }
void CodeGenerator::st4(const VReg8BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg4HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg2SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg16BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg8HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg4SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st4(const VReg2DList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 0, vt, adr); }
void CodeGenerator::st3(const VReg8BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg4HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg2SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg16BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg8HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg4SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st3(const VReg2DList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 4, vt, adr); }
void CodeGenerator::st2(const VReg8BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg4HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg2SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg16BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg8HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg4SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::st2(const VReg2DList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(0, 8, vt, adr); }
void CodeGenerator::ld4(const VReg8BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg4HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg2SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg16BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg8HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg4SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld4(const VReg2DList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 0, vt, adr); }
void CodeGenerator::ld3(const VReg8BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg4HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg2SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg16BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg8HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg4SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld3(const VReg2DList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 4, vt, adr); }
void CodeGenerator::ld2(const VReg8BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg4HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg2SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg16BList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg8HList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg4SList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::ld2(const VReg2DList &vt, const AdrPostImm &adr) { AdvSimdLdStMultiStructPostImmExceptLd1St1(1, 8, vt, adr); }
void CodeGenerator::st4(const VRegBElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 1, 4, vt, adr); }
void CodeGenerator::st4(const VRegHElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 1, 4, vt, adr); }
void CodeGenerator::st4(const VRegSElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 1, 4, vt, adr); }
void CodeGenerator::st4(const VRegDElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 1, 4, vt, adr); }
void CodeGenerator::st3(const VRegBElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 0, 3, vt, adr); }
void CodeGenerator::st3(const VRegHElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 0, 3, vt, adr); }
void CodeGenerator::st3(const VRegSElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 0, 3, vt, adr); }
void CodeGenerator::st3(const VRegDElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 0, 3, vt, adr); }
void CodeGenerator::st2(const VRegBElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 1, 2, vt, adr); }
void CodeGenerator::st2(const VRegHElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 1, 2, vt, adr); }
void CodeGenerator::st2(const VRegSElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 1, 2, vt, adr); }
void CodeGenerator::st2(const VRegDElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 1, 2, vt, adr); }
void CodeGenerator::st1(const VRegBElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 0, 1, vt, adr); }
void CodeGenerator::st1(const VRegHElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 0, 1, vt, adr); }
void CodeGenerator::st1(const VRegSElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 0, 1, vt, adr); }
void CodeGenerator::st1(const VRegDElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(0, 0, 1, vt, adr); }
void CodeGenerator::ld4(const VRegBElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 1, 4, vt, adr); }
void CodeGenerator::ld4(const VRegHElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 1, 4, vt, adr); }
void CodeGenerator::ld4(const VRegSElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 1, 4, vt, adr); }
void CodeGenerator::ld4(const VRegDElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 1, 4, vt, adr); }
void CodeGenerator::ld3(const VRegBElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 0, 3, vt, adr); }
void CodeGenerator::ld3(const VRegHElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 0, 3, vt, adr); }
void CodeGenerator::ld3(const VRegSElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 0, 3, vt, adr); }
void CodeGenerator::ld3(const VRegDElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 0, 3, vt, adr); }
void CodeGenerator::ld2(const VRegBElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 1, 2, vt, adr); }
void CodeGenerator::ld2(const VRegHElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 1, 2, vt, adr); }
void CodeGenerator::ld2(const VRegSElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 1, 2, vt, adr); }
void CodeGenerator::ld2(const VRegDElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 1, 2, vt, adr); }
void CodeGenerator::ld1(const VRegBElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 0, 1, vt, adr); }
void CodeGenerator::ld1(const VRegHElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 0, 1, vt, adr); }
void CodeGenerator::ld1(const VRegSElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 0, 1, vt, adr); }
void CodeGenerator::ld1(const VRegDElem &vt, const AdrNoOfs &adr) { AdvSimdLdStSingleStruct(1, 0, 1, vt, adr); }
void CodeGenerator::ld4r(const VReg8BList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg4HList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg2SList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg1DList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg16BList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg8HList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg4SList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg2DList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg8BList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg4HList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg2SList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg1DList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg16BList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg8HList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg4SList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg2DList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg8BList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg4HList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg2SList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg1DList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg16BList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg8HList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg4SList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg2DList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg8BList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg4HList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg2SList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg1DList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg16BList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg8HList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg4SList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg2DList &vt, const AdrNoOfs &adr) { AdvSimdLdRepSingleStruct(1, 0, 6, 0, vt, adr); }
void CodeGenerator::st4(const VRegBElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 1, 4, vt, adr); }
void CodeGenerator::st4(const VRegHElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 1, 4, vt, adr); }
void CodeGenerator::st4(const VRegSElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 1, 4, vt, adr); }
void CodeGenerator::st4(const VRegDElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 1, 4, vt, adr); }
void CodeGenerator::st3(const VRegBElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 0, 3, vt, adr); }
void CodeGenerator::st3(const VRegHElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 0, 3, vt, adr); }
void CodeGenerator::st3(const VRegSElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 0, 3, vt, adr); }
void CodeGenerator::st3(const VRegDElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 0, 3, vt, adr); }
void CodeGenerator::st2(const VRegBElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 1, 2, vt, adr); }
void CodeGenerator::st2(const VRegHElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 1, 2, vt, adr); }
void CodeGenerator::st2(const VRegSElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 1, 2, vt, adr); }
void CodeGenerator::st2(const VRegDElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 1, 2, vt, adr); }
void CodeGenerator::st1(const VRegBElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 0, 1, vt, adr); }
void CodeGenerator::st1(const VRegHElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 0, 1, vt, adr); }
void CodeGenerator::st1(const VRegSElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 0, 1, vt, adr); }
void CodeGenerator::st1(const VRegDElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(0, 0, 1, vt, adr); }
void CodeGenerator::ld4(const VRegBElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 1, 4, vt, adr); }
void CodeGenerator::ld4(const VRegHElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 1, 4, vt, adr); }
void CodeGenerator::ld4(const VRegSElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 1, 4, vt, adr); }
void CodeGenerator::ld4(const VRegDElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 1, 4, vt, adr); }
void CodeGenerator::ld3(const VRegBElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 0, 3, vt, adr); }
void CodeGenerator::ld3(const VRegHElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 0, 3, vt, adr); }
void CodeGenerator::ld3(const VRegSElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 0, 3, vt, adr); }
void CodeGenerator::ld3(const VRegDElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 0, 3, vt, adr); }
void CodeGenerator::ld2(const VRegBElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 1, 2, vt, adr); }
void CodeGenerator::ld2(const VRegHElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 1, 2, vt, adr); }
void CodeGenerator::ld2(const VRegSElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 1, 2, vt, adr); }
void CodeGenerator::ld2(const VRegDElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 1, 2, vt, adr); }
void CodeGenerator::ld1(const VRegBElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 0, 1, vt, adr); }
void CodeGenerator::ld1(const VRegHElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 0, 1, vt, adr); }
void CodeGenerator::ld1(const VRegSElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 0, 1, vt, adr); }
void CodeGenerator::ld1(const VRegDElem &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructPostReg(1, 0, 1, vt, adr); }
void CodeGenerator::ld4r(const VReg8BList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg4HList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg2SList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg1DList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg16BList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg8HList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg4SList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg2DList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg8BList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg4HList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg2SList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg1DList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg16BList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg8HList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg4SList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg2DList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg8BList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg4HList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg2SList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg1DList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg16BList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg8HList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg4SList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg2DList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg8BList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg4HList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg2SList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg1DList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg16BList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg8HList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg4SList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg2DList &vt, const AdrPostReg &adr) { AdvSimdLdStSingleStructRepPostReg(1, 0, 6, 0, vt, adr); }
void CodeGenerator::st4(const VRegBElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 1, 4, vt, adr); }
void CodeGenerator::st4(const VRegHElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 1, 4, vt, adr); }
void CodeGenerator::st4(const VRegSElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 1, 4, vt, adr); }
void CodeGenerator::st4(const VRegDElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 1, 4, vt, adr); }
void CodeGenerator::st3(const VRegBElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 0, 3, vt, adr); }
void CodeGenerator::st3(const VRegHElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 0, 3, vt, adr); }
void CodeGenerator::st3(const VRegSElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 0, 3, vt, adr); }
void CodeGenerator::st3(const VRegDElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 0, 3, vt, adr); }
void CodeGenerator::st2(const VRegBElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 1, 2, vt, adr); }
void CodeGenerator::st2(const VRegHElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 1, 2, vt, adr); }
void CodeGenerator::st2(const VRegSElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 1, 2, vt, adr); }
void CodeGenerator::st2(const VRegDElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 1, 2, vt, adr); }
void CodeGenerator::st1(const VRegBElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 0, 1, vt, adr); }
void CodeGenerator::st1(const VRegHElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 0, 1, vt, adr); }
void CodeGenerator::st1(const VRegSElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 0, 1, vt, adr); }
void CodeGenerator::st1(const VRegDElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(0, 0, 1, vt, adr); }
void CodeGenerator::ld4(const VRegBElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 1, 4, vt, adr); }
void CodeGenerator::ld4(const VRegHElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 1, 4, vt, adr); }
void CodeGenerator::ld4(const VRegSElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 1, 4, vt, adr); }
void CodeGenerator::ld4(const VRegDElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 1, 4, vt, adr); }
void CodeGenerator::ld3(const VRegBElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 0, 3, vt, adr); }
void CodeGenerator::ld3(const VRegHElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 0, 3, vt, adr); }
void CodeGenerator::ld3(const VRegSElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 0, 3, vt, adr); }
void CodeGenerator::ld3(const VRegDElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 0, 3, vt, adr); }
void CodeGenerator::ld2(const VRegBElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 1, 2, vt, adr); }
void CodeGenerator::ld2(const VRegHElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 1, 2, vt, adr); }
void CodeGenerator::ld2(const VRegSElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 1, 2, vt, adr); }
void CodeGenerator::ld2(const VRegDElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 1, 2, vt, adr); }
void CodeGenerator::ld1(const VRegBElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 0, 1, vt, adr); }
void CodeGenerator::ld1(const VRegHElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 0, 1, vt, adr); }
void CodeGenerator::ld1(const VRegSElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 0, 1, vt, adr); }
void CodeGenerator::ld1(const VRegDElem &vt, const AdrPostImm &adr) { AdvSimdLdStSingleStructPostImm(1, 0, 1, vt, adr); }
void CodeGenerator::ld4r(const VReg8BList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg4HList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg2SList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg1DList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg16BList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg8HList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg4SList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld4r(const VReg2DList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg8BList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg4HList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg2SList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg1DList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg16BList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg8HList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg4SList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld3r(const VReg2DList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 7, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg8BList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg4HList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg2SList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg1DList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg16BList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg8HList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg4SList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld2r(const VReg2DList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 1, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg8BList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg4HList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg2SList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg1DList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg16BList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg8HList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg4SList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr); }
void CodeGenerator::ld1r(const VReg2DList &vt, const AdrPostImm &adr) { AdvSimdLdRepSingleStructPostImm(1, 0, 6, 0, vt, adr); }
void CodeGenerator::stxrb(const WReg &ws, const WReg &rt, const AdrImm &adr) { StExclusive(0, 0, ws, rt, adr); }
void CodeGenerator::stlxrb(const WReg &ws, const WReg &rt, const AdrImm &adr) { StExclusive(0, 1, ws, rt, adr); }
void CodeGenerator::stxrh(const WReg &ws, const WReg &rt, const AdrImm &adr) { StExclusive(1, 0, ws, rt, adr); }
void CodeGenerator::stlxrh(const WReg &ws, const WReg &rt, const AdrImm &adr) { StExclusive(1, 1, ws, rt, adr); }
void CodeGenerator::stxr(const WReg &ws, const WReg &rt, const AdrImm &adr) { StExclusive(2, 0, ws, rt, adr); }
void CodeGenerator::stlxr(const WReg &ws, const WReg &rt, const AdrImm &adr) { StExclusive(2, 1, ws, rt, adr); }
void CodeGenerator::stxr(const WReg &ws, const XReg &rt, const AdrImm &adr) { StExclusive(3, 0, ws, rt, adr); }
void CodeGenerator::stlxr(const WReg &ws, const XReg &rt, const AdrImm &adr) { StExclusive(3, 1, ws, rt, adr); }
void CodeGenerator::ldxrb(const WReg &rt, const AdrImm &adr) { LdExclusive(0, 0, rt, adr); }
void CodeGenerator::ldaxrb(const WReg &rt, const AdrImm &adr) { LdExclusive(0, 1, rt, adr); }
void CodeGenerator::ldxrh(const WReg &rt, const AdrImm &adr) { LdExclusive(1, 0, rt, adr); }
void CodeGenerator::ldaxrh(const WReg &rt, const AdrImm &adr) { LdExclusive(1, 1, rt, adr); }
void CodeGenerator::ldxr(const WReg &rt, const AdrImm &adr) { LdExclusive(2, 0, rt, adr); }
void CodeGenerator::ldaxr(const WReg &rt, const AdrImm &adr) { LdExclusive(2, 1, rt, adr); }
void CodeGenerator::ldxr(const XReg &rt, const AdrImm &adr) { LdExclusive(3, 0, rt, adr); }
void CodeGenerator::ldaxr(const XReg &rt, const AdrImm &adr) { LdExclusive(3, 1, rt, adr); }
void CodeGenerator::stllrb(const WReg &rt, const AdrImm &adr) { StLORelase(0, 0, rt, adr); }
void CodeGenerator::stlrb(const WReg &rt, const AdrImm &adr) { StLORelase(0, 1, rt, adr); }
void CodeGenerator::stllrh(const WReg &rt, const AdrImm &adr) { StLORelase(1, 0, rt, adr); }
void CodeGenerator::stlrh(const WReg &rt, const AdrImm &adr) { StLORelase(1, 1, rt, adr); }
void CodeGenerator::stllr(const WReg &rt, const AdrImm &adr) { StLORelase(2, 0, rt, adr); }
void CodeGenerator::stlr(const WReg &rt, const AdrImm &adr) { StLORelase(2, 1, rt, adr); }
void CodeGenerator::stllr(const XReg &rt, const AdrImm &adr) { StLORelase(3, 0, rt, adr); }
void CodeGenerator::stlr(const XReg &rt, const AdrImm &adr) { StLORelase(3, 1, rt, adr); }
void CodeGenerator::ldlarb(const WReg &rt, const AdrImm &adr) { LdLOAcquire(0, 0, rt, adr); }
void CodeGenerator::ldarb(const WReg &rt, const AdrImm &adr) { LdLOAcquire(0, 1, rt, adr); }
void CodeGenerator::ldlarh(const WReg &rt, const AdrImm &adr) { LdLOAcquire(1, 0, rt, adr); }
void CodeGenerator::ldarh(const WReg &rt, const AdrImm &adr) { LdLOAcquire(1, 1, rt, adr); }
void CodeGenerator::ldlar(const WReg &rt, const AdrImm &adr) { LdLOAcquire(2, 0, rt, adr); }
void CodeGenerator::ldar(const WReg &rt, const AdrImm &adr) { LdLOAcquire(2, 1, rt, adr); }
void CodeGenerator::ldlar(const XReg &rt, const AdrImm &adr) { LdLOAcquire(3, 0, rt, adr); }
void CodeGenerator::ldar(const XReg &rt, const AdrImm &adr) { LdLOAcquire(3, 1, rt, adr); }
void CodeGenerator::casb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { Cas(0, 1, 0, 1, 0, rs, rt, adr); }
void CodeGenerator::caslb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { Cas(0, 1, 0, 1, 1, rs, rt, adr); }
void CodeGenerator::casab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { Cas(0, 1, 1, 1, 0, rs, rt, adr); }
void CodeGenerator::casalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { Cas(0, 1, 1, 1, 1, rs, rt, adr); }
void CodeGenerator::cash(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { Cas(1, 1, 0, 1, 0, rs, rt, adr); }
void CodeGenerator::caslh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { Cas(1, 1, 0, 1, 1, rs, rt, adr); }
void CodeGenerator::casah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { Cas(1, 1, 1, 1, 0, rs, rt, adr); }
void CodeGenerator::casalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { Cas(1, 1, 1, 1, 1, rs, rt, adr); }
void CodeGenerator::cas(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { Cas(2, 1, 0, 1, 0, rs, rt, adr); }
void CodeGenerator::casl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { Cas(2, 1, 0, 1, 1, rs, rt, adr); }
void CodeGenerator::casa(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { Cas(2, 1, 1, 1, 0, rs, rt, adr); }
void CodeGenerator::casal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { Cas(2, 1, 1, 1, 1, rs, rt, adr); }
void CodeGenerator::cas(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { Cas(3, 1, 0, 1, 0, rs, rt, adr); }
void CodeGenerator::casl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { Cas(3, 1, 0, 1, 1, rs, rt, adr); }
void CodeGenerator::casa(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { Cas(3, 1, 1, 1, 0, rs, rt, adr); }
void CodeGenerator::casal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { Cas(3, 1, 1, 1, 1, rs, rt, adr); }
void CodeGenerator::stxp(const WReg &ws, const WReg &rt1, const WReg &rt2, const AdrImm &adr) { StExclusivePair(0, 1, 0, ws, rt1, rt2, adr); }
void CodeGenerator::stxp(const WReg &ws, const XReg &rt1, const XReg &rt2, const AdrImm &adr) { StExclusivePair(0, 1, 0, ws, rt1, rt2, adr); }
void CodeGenerator::stlxp(const WReg &ws, const WReg &rt1, const WReg &rt2, const AdrImm &adr) { StExclusivePair(0, 1, 1, ws, rt1, rt2, adr); }
void CodeGenerator::stlxp(const WReg &ws, const XReg &rt1, const XReg &rt2, const AdrImm &adr) { StExclusivePair(0, 1, 1, ws, rt1, rt2, adr); }
void CodeGenerator::ldxp(const WReg &rt1, const WReg &rt2, const AdrImm &adr) { LdExclusivePair(1, 1, 0, rt1, rt2, adr); }
void CodeGenerator::ldxp(const XReg &rt1, const XReg &rt2, const AdrImm &adr) { LdExclusivePair(1, 1, 0, rt1, rt2, adr); }
void CodeGenerator::ldaxp(const WReg &rt1, const WReg &rt2, const AdrImm &adr) { LdExclusivePair(1, 1, 1, rt1, rt2, adr); }
void CodeGenerator::ldaxp(const XReg &rt1, const XReg &rt2, const AdrImm &adr) { LdExclusivePair(1, 1, 1, rt1, rt2, adr); }
void CodeGenerator::casp(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { CasPair(0, 1, 0, rs, rt, adr); }
void CodeGenerator::casp(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { CasPair(0, 1, 0, rs, rt, adr); }
void CodeGenerator::caspl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { CasPair(0, 1, 1, rs, rt, adr); }
void CodeGenerator::caspl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { CasPair(0, 1, 1, rs, rt, adr); }
void CodeGenerator::caspa(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { CasPair(1, 1, 0, rs, rt, adr); }
void CodeGenerator::caspa(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { CasPair(1, 1, 0, rs, rt, adr); }
void CodeGenerator::caspal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { CasPair(1, 1, 1, rs, rt, adr); }
void CodeGenerator::caspal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { CasPair(1, 1, 1, rs, rt, adr); }
void CodeGenerator::stlurb(const WReg &rt, const AdrImm &adr) { LdaprStlr(0, 0, rt, adr); }
void CodeGenerator::ldapurb(const WReg &rt, const AdrImm &adr) { LdaprStlr(0, 1, rt, adr); }
void CodeGenerator::ldapursb(const XReg &rt, const AdrImm &adr) { LdaprStlr(0, 2, rt, adr); }
void CodeGenerator::ldapursb(const WReg &rt, const AdrImm &adr) { LdaprStlr(0, 3, rt, adr); }
void CodeGenerator::stlurh(const WReg &rt, const AdrImm &adr) { LdaprStlr(1, 0, rt, adr); }
void CodeGenerator::ldapurh(const WReg &rt, const AdrImm &adr) { LdaprStlr(1, 1, rt, adr); }
void CodeGenerator::ldapursh(const XReg &rt, const AdrImm &adr) { LdaprStlr(1, 2, rt, adr); }
void CodeGenerator::ldapursh(const WReg &rt, const AdrImm &adr) { LdaprStlr(1, 3, rt, adr); }
void CodeGenerator::stlur(const WReg &rt, const AdrImm &adr) { LdaprStlr(2, 0, rt, adr); }
void CodeGenerator::ldapur(const WReg &rt, const AdrImm &adr) { LdaprStlr(2, 1, rt, adr); }
void CodeGenerator::ldapursw(const XReg &rt, const AdrImm &adr) { LdaprStlr(2, 2, rt, adr); }
void CodeGenerator::stlur(const XReg &rt, const AdrImm &adr) { LdaprStlr(3, 0, rt, adr); }
void CodeGenerator::ldapur(const XReg &rt, const AdrImm &adr) { LdaprStlr(3, 1, rt, adr); }
void CodeGenerator::ldr(const WReg &rt, const Label &label) { LdRegLiteral((rt.getBit() == 64) ? 1 : 0, 0, rt, label); }
void CodeGenerator::ldr(const XReg &rt, const Label &label) { LdRegLiteral((rt.getBit() == 64) ? 1 : 0, 0, rt, label); }
void CodeGenerator::ldr(const WReg &rt, const int64_t label) { LdRegLiteral((rt.getBit() == 64) ? 1 : 0, 0, rt, label); }
void CodeGenerator::ldr(const XReg &rt, const int64_t label) { LdRegLiteral((rt.getBit() == 64) ? 1 : 0, 0, rt, label); }
void CodeGenerator::ldrsw(const WReg &rt, const Label &label) { LdRegLiteral(2, 0, rt, label); }
void CodeGenerator::ldrsw(const XReg &rt, const Label &label) { LdRegLiteral(2, 0, rt, label); }
void CodeGenerator::ldrsw(const WReg &rt, const int64_t label) { LdRegLiteral(2, 0, rt, label); }
void CodeGenerator::ldrsw(const XReg &rt, const int64_t label) { LdRegLiteral(2, 0, rt, label); }
void CodeGenerator::ldr(const SReg &vt, const Label &label) { LdRegSimdFpLiteral(vt, label); }
void CodeGenerator::ldr(const DReg &vt, const Label &label) { LdRegSimdFpLiteral(vt, label); }
void CodeGenerator::ldr(const QReg &vt, const Label &label) { LdRegSimdFpLiteral(vt, label); }
void CodeGenerator::ldr(const SReg &vt, const int64_t label) { LdRegSimdFpLiteral(vt, label); }
void CodeGenerator::ldr(const DReg &vt, const int64_t label) { LdRegSimdFpLiteral(vt, label); }
void CodeGenerator::ldr(const QReg &vt, const int64_t label) { LdRegSimdFpLiteral(vt, label); }
void CodeGenerator::prfm(const Prfop prfop, const Label &label) { PfLiteral(prfop, label); }
void CodeGenerator::prfm(const Prfop prfop, const int64_t label) { PfLiteral(prfop, label); }
void CodeGenerator::stnp(const WReg &rt1, const WReg &rt2, const AdrImm &adr) { LdStNoAllocPair(0, rt1, rt2, adr); }
void CodeGenerator::stnp(const XReg &rt1, const XReg &rt2, const AdrImm &adr) { LdStNoAllocPair(0, rt1, rt2, adr); }
void CodeGenerator::ldnp(const WReg &rt1, const WReg &rt2, const AdrImm &adr) { LdStNoAllocPair(1, rt1, rt2, adr); }
void CodeGenerator::ldnp(const XReg &rt1, const XReg &rt2, const AdrImm &adr) { LdStNoAllocPair(1, rt1, rt2, adr); }
void CodeGenerator::stnp(const SReg &vt1, const SReg &vt2, const AdrImm &adr) { LdStSimdFpNoAllocPair(0, vt1, vt2, adr); }
void CodeGenerator::stnp(const DReg &vt1, const DReg &vt2, const AdrImm &adr) { LdStSimdFpNoAllocPair(0, vt1, vt2, adr); }
void CodeGenerator::stnp(const QReg &vt1, const QReg &vt2, const AdrImm &adr) { LdStSimdFpNoAllocPair(0, vt1, vt2, adr); }
void CodeGenerator::ldnp(const SReg &vt1, const SReg &vt2, const AdrImm &adr) { LdStSimdFpNoAllocPair(1, vt1, vt2, adr); }
void CodeGenerator::ldnp(const DReg &vt1, const DReg &vt2, const AdrImm &adr) { LdStSimdFpNoAllocPair(1, vt1, vt2, adr); }
void CodeGenerator::ldnp(const QReg &vt1, const QReg &vt2, const AdrImm &adr) { LdStSimdFpNoAllocPair(1, vt1, vt2, adr); }
void CodeGenerator::stp(const WReg &rt1, const WReg &rt2, const AdrPostImm &adr) { LdStRegPairPostImm((rt1.getBit() == 32) ? 0 : 2, 0, rt1, rt2, adr); }
void CodeGenerator::stp(const XReg &rt1, const XReg &rt2, const AdrPostImm &adr) { LdStRegPairPostImm((rt1.getBit() == 32) ? 0 : 2, 0, rt1, rt2, adr); }
void CodeGenerator::ldp(const WReg &rt1, const WReg &rt2, const AdrPostImm &adr) { LdStRegPairPostImm((rt1.getBit() == 32) ? 0 : 2, 1, rt1, rt2, adr); }
void CodeGenerator::ldp(const XReg &rt1, const XReg &rt2, const AdrPostImm &adr) { LdStRegPairPostImm((rt1.getBit() == 32) ? 0 : 2, 1, rt1, rt2, adr); }
void CodeGenerator::ldpsw(const XReg &rt1, const XReg &rt2, const AdrPostImm &adr) { LdStRegPairPostImm(1, 1, rt1, rt2, adr); }
void CodeGenerator::stp(const SReg &vt1, const SReg &vt2, const AdrPostImm &adr) { LdStSimdFpPairPostImm(0, vt1, vt2, adr); }
void CodeGenerator::stp(const DReg &vt1, const DReg &vt2, const AdrPostImm &adr) { LdStSimdFpPairPostImm(0, vt1, vt2, adr); }
void CodeGenerator::stp(const QReg &vt1, const QReg &vt2, const AdrPostImm &adr) { LdStSimdFpPairPostImm(0, vt1, vt2, adr); }
void CodeGenerator::ldp(const SReg &vt1, const SReg &vt2, const AdrPostImm &adr) { LdStSimdFpPairPostImm(1, vt1, vt2, adr); }
void CodeGenerator::ldp(const DReg &vt1, const DReg &vt2, const AdrPostImm &adr) { LdStSimdFpPairPostImm(1, vt1, vt2, adr); }
void CodeGenerator::ldp(const QReg &vt1, const QReg &vt2, const AdrPostImm &adr) { LdStSimdFpPairPostImm(1, vt1, vt2, adr); }
void CodeGenerator::stp(const WReg &rt1, const WReg &rt2, const AdrImm &adr) { LdStRegPair((rt1.getBit() == 32) ? 0 : 2, 0, rt1, rt2, adr); }
void CodeGenerator::stp(const XReg &rt1, const XReg &rt2, const AdrImm &adr) { LdStRegPair((rt1.getBit() == 32) ? 0 : 2, 0, rt1, rt2, adr); }
void CodeGenerator::ldp(const WReg &rt1, const WReg &rt2, const AdrImm &adr) { LdStRegPair((rt1.getBit() == 32) ? 0 : 2, 1, rt1, rt2, adr); }
void CodeGenerator::ldp(const XReg &rt1, const XReg &rt2, const AdrImm &adr) { LdStRegPair((rt1.getBit() == 32) ? 0 : 2, 1, rt1, rt2, adr); }
void CodeGenerator::ldpsw(const XReg &rt1, const XReg &rt2, const AdrImm &adr) { LdStRegPair(1, 1, rt1, rt2, adr); }
void CodeGenerator::stp(const SReg &vt1, const SReg &vt2, const AdrImm &adr) { LdStSimdFpPair(0, vt1, vt2, adr); }
void CodeGenerator::stp(const DReg &vt1, const DReg &vt2, const AdrImm &adr) { LdStSimdFpPair(0, vt1, vt2, adr); }
void CodeGenerator::stp(const QReg &vt1, const QReg &vt2, const AdrImm &adr) { LdStSimdFpPair(0, vt1, vt2, adr); }
void CodeGenerator::ldp(const SReg &vt1, const SReg &vt2, const AdrImm &adr) { LdStSimdFpPair(1, vt1, vt2, adr); }
void CodeGenerator::ldp(const DReg &vt1, const DReg &vt2, const AdrImm &adr) { LdStSimdFpPair(1, vt1, vt2, adr); }
void CodeGenerator::ldp(const QReg &vt1, const QReg &vt2, const AdrImm &adr) { LdStSimdFpPair(1, vt1, vt2, adr); }
void CodeGenerator::stp(const WReg &rt1, const WReg &rt2, const AdrPreImm &adr) { LdStRegPairPre((rt1.getBit() == 32) ? 0 : 2, 0, rt1, rt2, adr); }
void CodeGenerator::stp(const XReg &rt1, const XReg &rt2, const AdrPreImm &adr) { LdStRegPairPre((rt1.getBit() == 32) ? 0 : 2, 0, rt1, rt2, adr); }
void CodeGenerator::ldp(const WReg &rt1, const WReg &rt2, const AdrPreImm &adr) { LdStRegPairPre((rt1.getBit() == 32) ? 0 : 2, 1, rt1, rt2, adr); }
void CodeGenerator::ldp(const XReg &rt1, const XReg &rt2, const AdrPreImm &adr) { LdStRegPairPre((rt1.getBit() == 32) ? 0 : 2, 1, rt1, rt2, adr); }
void CodeGenerator::ldpsw(const XReg &rt1, const XReg &rt2, const AdrPreImm &adr) { LdStRegPairPre(1, 1, rt1, rt2, adr); }
void CodeGenerator::stp(const SReg &vt1, const SReg &vt2, const AdrPreImm &adr) { LdStSimdFpPairPre(0, vt1, vt2, adr); }
void CodeGenerator::stp(const DReg &vt1, const DReg &vt2, const AdrPreImm &adr) { LdStSimdFpPairPre(0, vt1, vt2, adr); }
void CodeGenerator::stp(const QReg &vt1, const QReg &vt2, const AdrPreImm &adr) { LdStSimdFpPairPre(0, vt1, vt2, adr); }
void CodeGenerator::ldp(const SReg &vt1, const SReg &vt2, const AdrPreImm &adr) { LdStSimdFpPairPre(1, vt1, vt2, adr); }
void CodeGenerator::ldp(const DReg &vt1, const DReg &vt2, const AdrPreImm &adr) { LdStSimdFpPairPre(1, vt1, vt2, adr); }
void CodeGenerator::ldp(const QReg &vt1, const QReg &vt2, const AdrPreImm &adr) { LdStSimdFpPairPre(1, vt1, vt2, adr); }
void CodeGenerator::sturb(const WReg &rt, const AdrImm &adr) { LdStRegUnsImm(0, 0, rt, adr); }
void CodeGenerator::ldurb(const WReg &rt, const AdrImm &adr) { LdStRegUnsImm(0, 1, rt, adr); }
void CodeGenerator::ldursb(const WReg &rt, const AdrImm &adr) { LdStRegUnsImm(0, 3, rt, adr); }
void CodeGenerator::sturh(const WReg &rt, const AdrImm &adr) { LdStRegUnsImm(1, 0, rt, adr); }
void CodeGenerator::ldurh(const WReg &rt, const AdrImm &adr) { LdStRegUnsImm(1, 1, rt, adr); }
void CodeGenerator::ldursh(const WReg &rt, const AdrImm &adr) { LdStRegUnsImm(1, 3, rt, adr); }
void CodeGenerator::stur(const WReg &rt, const AdrImm &adr) { LdStRegUnsImm(2, 0, rt, adr); }
void CodeGenerator::ldur(const WReg &rt, const AdrImm &adr) { LdStRegUnsImm(2, 1, rt, adr); }
void CodeGenerator::ldursb(const XReg &rt, const AdrImm &adr) { LdStRegUnsImm(0, 2, rt, adr); }
void CodeGenerator::ldursh(const XReg &rt, const AdrImm &adr) { LdStRegUnsImm(1, 2, rt, adr); }
void CodeGenerator::ldursw(const XReg &rt, const AdrImm &adr) { LdStRegUnsImm(2, 2, rt, adr); }
void CodeGenerator::stur(const XReg &rt, const AdrImm &adr) { LdStRegUnsImm(3, 0, rt, adr); }
void CodeGenerator::ldur(const XReg &rt, const AdrImm &adr) { LdStRegUnsImm(3, 1, rt, adr); }
void CodeGenerator::stur(const BReg &vt, const AdrImm &adr) { LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::stur(const HReg &vt, const AdrImm &adr) { LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::stur(const SReg &vt, const AdrImm &adr) { LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::stur(const DReg &vt, const AdrImm &adr) { LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::stur(const QReg &vt, const AdrImm &adr) { LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::ldur(const BReg &vt, const AdrImm &adr) { LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldur(const HReg &vt, const AdrImm &adr) { LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldur(const SReg &vt, const AdrImm &adr) { LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldur(const DReg &vt, const AdrImm &adr) { LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldur(const QReg &vt, const AdrImm &adr) { LdStSimdFpRegUnsImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::prfum(const Prfop prfop, const AdrImm &adr) { PfRegUnsImm(prfop, adr); }
void CodeGenerator::strb(const WReg &rt, const AdrPostImm &adr) { LdStRegPostImm(0, 0, rt, adr); }
void CodeGenerator::ldrb(const WReg &rt, const AdrPostImm &adr) { LdStRegPostImm(0, 1, rt, adr); }
void CodeGenerator::ldrsb(const WReg &rt, const AdrPostImm &adr) { LdStRegPostImm(0, 3, rt, adr); }
void CodeGenerator::strh(const WReg &rt, const AdrPostImm &adr) { LdStRegPostImm(1, 0, rt, adr); }
void CodeGenerator::ldrh(const WReg &rt, const AdrPostImm &adr) { LdStRegPostImm(1, 1, rt, adr); }
void CodeGenerator::ldrsh(const WReg &rt, const AdrPostImm &adr) { LdStRegPostImm(1, 3, rt, adr); }
void CodeGenerator::str(const WReg &rt, const AdrPostImm &adr) { LdStRegPostImm(2, 0, rt, adr); }
void CodeGenerator::ldr(const WReg &rt, const AdrPostImm &adr) { LdStRegPostImm(2, 1, rt, adr); }
void CodeGenerator::ldrsb(const XReg &rt, const AdrPostImm &adr) { LdStRegPostImm(0, 2, rt, adr); }
void CodeGenerator::ldrsh(const XReg &rt, const AdrPostImm &adr) { LdStRegPostImm(1, 2, rt, adr); }
void CodeGenerator::ldrsw(const XReg &rt, const AdrPostImm &adr) { LdStRegPostImm(2, 2, rt, adr); }
void CodeGenerator::str(const XReg &rt, const AdrPostImm &adr) { LdStRegPostImm(3, 0, rt, adr); }
void CodeGenerator::ldr(const XReg &rt, const AdrPostImm &adr) { LdStRegPostImm(3, 1, rt, adr); }
void CodeGenerator::str(const BReg &vt, const AdrPostImm &adr) { LdStSimdFpRegPostImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const HReg &vt, const AdrPostImm &adr) { LdStSimdFpRegPostImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const SReg &vt, const AdrPostImm &adr) { LdStSimdFpRegPostImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const DReg &vt, const AdrPostImm &adr) { LdStSimdFpRegPostImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const QReg &vt, const AdrPostImm &adr) { LdStSimdFpRegPostImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::ldr(const BReg &vt, const AdrPostImm &adr) { LdStSimdFpRegPostImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const HReg &vt, const AdrPostImm &adr) { LdStSimdFpRegPostImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const SReg &vt, const AdrPostImm &adr) { LdStSimdFpRegPostImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const DReg &vt, const AdrPostImm &adr) { LdStSimdFpRegPostImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const QReg &vt, const AdrPostImm &adr) { LdStSimdFpRegPostImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::sttrb(const WReg &rt, const AdrImm &adr) { LdStRegUnpriv(0, 0, rt, adr); }
void CodeGenerator::ldtrb(const WReg &rt, const AdrImm &adr) { LdStRegUnpriv(0, 1, rt, adr); }
void CodeGenerator::ldtrsb(const WReg &rt, const AdrImm &adr) { LdStRegUnpriv(0, 3, rt, adr); }
void CodeGenerator::sttrh(const WReg &rt, const AdrImm &adr) { LdStRegUnpriv(1, 0, rt, adr); }
void CodeGenerator::ldtrh(const WReg &rt, const AdrImm &adr) { LdStRegUnpriv(1, 1, rt, adr); }
void CodeGenerator::ldtrsh(const WReg &rt, const AdrImm &adr) { LdStRegUnpriv(1, 3, rt, adr); }
void CodeGenerator::sttr(const WReg &rt, const AdrImm &adr) { LdStRegUnpriv(2, 0, rt, adr); }
void CodeGenerator::ldtr(const WReg &rt, const AdrImm &adr) { LdStRegUnpriv(2, 1, rt, adr); }
void CodeGenerator::ldtrsb(const XReg &rt, const AdrImm &adr) { LdStRegUnpriv(0, 2, rt, adr); }
void CodeGenerator::ldtrsh(const XReg &rt, const AdrImm &adr) { LdStRegUnpriv(1, 2, rt, adr); }
void CodeGenerator::ldtrsw(const XReg &rt, const AdrImm &adr) { LdStRegUnpriv(2, 2, rt, adr); }
void CodeGenerator::sttr(const XReg &rt, const AdrImm &adr) { LdStRegUnpriv(3, 0, rt, adr); }
void CodeGenerator::ldtr(const XReg &rt, const AdrImm &adr) { LdStRegUnpriv(3, 1, rt, adr); }
void CodeGenerator::strb(const WReg &rt, const AdrPreImm &adr) { LdStRegPre(0, 0, rt, adr); }
void CodeGenerator::ldrb(const WReg &rt, const AdrPreImm &adr) { LdStRegPre(0, 1, rt, adr); }
void CodeGenerator::ldrsb(const WReg &rt, const AdrPreImm &adr) { LdStRegPre(0, 3, rt, adr); }
void CodeGenerator::strh(const WReg &rt, const AdrPreImm &adr) { LdStRegPre(1, 0, rt, adr); }
void CodeGenerator::ldrh(const WReg &rt, const AdrPreImm &adr) { LdStRegPre(1, 1, rt, adr); }
void CodeGenerator::ldrsh(const WReg &rt, const AdrPreImm &adr) { LdStRegPre(1, 3, rt, adr); }
void CodeGenerator::str(const WReg &rt, const AdrPreImm &adr) { LdStRegPre(2, 0, rt, adr); }
void CodeGenerator::ldr(const WReg &rt, const AdrPreImm &adr) { LdStRegPre(2, 1, rt, adr); }
void CodeGenerator::ldrsb(const XReg &rt, const AdrPreImm &adr) { LdStRegPre(0, 2, rt, adr); }
void CodeGenerator::ldrsh(const XReg &rt, const AdrPreImm &adr) { LdStRegPre(1, 2, rt, adr); }
void CodeGenerator::ldrsw(const XReg &rt, const AdrPreImm &adr) { LdStRegPre(2, 2, rt, adr); }
void CodeGenerator::str(const XReg &rt, const AdrPreImm &adr) { LdStRegPre(3, 0, rt, adr); }
void CodeGenerator::ldr(const XReg &rt, const AdrPreImm &adr) { LdStRegPre(3, 1, rt, adr); }
void CodeGenerator::str(const BReg &vt, const AdrPreImm &adr) { LdStSimdFpRegPre((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const HReg &vt, const AdrPreImm &adr) { LdStSimdFpRegPre((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const SReg &vt, const AdrPreImm &adr) { LdStSimdFpRegPre((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const DReg &vt, const AdrPreImm &adr) { LdStSimdFpRegPre((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const QReg &vt, const AdrPreImm &adr) { LdStSimdFpRegPre((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::ldr(const BReg &vt, const AdrPreImm &adr) { LdStSimdFpRegPre((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const HReg &vt, const AdrPreImm &adr) { LdStSimdFpRegPre((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const SReg &vt, const AdrPreImm &adr) { LdStSimdFpRegPre((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const DReg &vt, const AdrPreImm &adr) { LdStSimdFpRegPre((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const QReg &vt, const AdrPreImm &adr) { LdStSimdFpRegPre((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldaddb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclrb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeorb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 2, rs, rt, adr); }
void CodeGenerator::ldsetb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsminb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 6, rs, rt, adr); }
void CodeGenerator::stumaxb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 6, rs, rt, adr); }
void CodeGenerator::lduminb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 7, rs, rt, adr); }
void CodeGenerator::swapb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 1, 0, rs, rt, adr); }
void CodeGenerator::ldaddlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclrlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeorlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 2, rs, rt, adr); }
void CodeGenerator::ldsetlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsminlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 6, rs, rt, adr); }
void CodeGenerator::lduminlb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 7, rs, rt, adr); }
void CodeGenerator::swaplb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 1, 0, rs, rt, adr); }
void CodeGenerator::ldaddab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclrab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeorab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 2, rs, rt, adr); }
void CodeGenerator::ldsetab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsminab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 6, rs, rt, adr); }
void CodeGenerator::lduminab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 7, rs, rt, adr); }
void CodeGenerator::swapab(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 1, 0, rs, rt, adr); }
void CodeGenerator::ldaprb(const WReg &rt, const AdrImm &adr) { AtomicMemOp(0, 0, 1, 0, 1, 4, WReg(31), rt, adr); }
void CodeGenerator::ldaddalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclralb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeoralb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 2, rs, rt, adr); }
void CodeGenerator::ldsetalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsminalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 6, rs, rt, adr); }
void CodeGenerator::lduminalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 7, rs, rt, adr); }
void CodeGenerator::swapalb(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 1, 0, rs, rt, adr); }
void CodeGenerator::ldaddh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclrh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeorh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 2, rs, rt, adr); }
void CodeGenerator::ldseth(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsminh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 6, rs, rt, adr); }
void CodeGenerator::lduminh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 7, rs, rt, adr); }
void CodeGenerator::swaph(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 1, 0, rs, rt, adr); }
void CodeGenerator::ldaddlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclrlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeorlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 2, rs, rt, adr); }
void CodeGenerator::ldsetlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsminlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 6, rs, rt, adr); }
void CodeGenerator::lduminlh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 7, rs, rt, adr); }
void CodeGenerator::swaplh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 1, 0, rs, rt, adr); }
void CodeGenerator::ldaddah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclrah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeorah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 2, rs, rt, adr); }
void CodeGenerator::ldsetah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsminah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 6, rs, rt, adr); }
void CodeGenerator::lduminah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 7, rs, rt, adr); }
void CodeGenerator::swapah(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 1, 0, rs, rt, adr); }
void CodeGenerator::ldaprh(const WReg &rt, const AdrImm &adr) { AtomicMemOp(1, 0, 1, 0, 1, 4, WReg(31), rt, adr); }
void CodeGenerator::ldaddalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclralh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeoralh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 2, rs, rt, adr); }
void CodeGenerator::ldsetalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsminalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 6, rs, rt, adr); }
void CodeGenerator::lduminalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 7, rs, rt, adr); }
void CodeGenerator::swapalh(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 1, 0, rs, rt, adr); }
void CodeGenerator::ldadd(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclr(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeor(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 2, rs, rt, adr); }
void CodeGenerator::ldset(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmax(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsmin(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumax(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 6, rs, rt, adr); }
void CodeGenerator::ldumin(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 7, rs, rt, adr); }
void CodeGenerator::swap(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 1, 0, rs, rt, adr); }
void CodeGenerator::ldaddl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclrl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeorl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 2, rs, rt, adr); }
void CodeGenerator::ldsetl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsminl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 6, rs, rt, adr); }
void CodeGenerator::lduminl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 7, rs, rt, adr); }
void CodeGenerator::swapl(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 1, 0, rs, rt, adr); }
void CodeGenerator::ldadda(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclra(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeora(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 2, rs, rt, adr); }
void CodeGenerator::ldseta(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxa(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsmina(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxa(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 6, rs, rt, adr); }
void CodeGenerator::ldumina(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 7, rs, rt, adr); }
void CodeGenerator::swapa(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 1, 0, rs, rt, adr); }
void CodeGenerator::ldapr(const WReg &rt, const AdrImm &adr) { AtomicMemOp(2, 0, 1, 0, 1, 4, WReg(31), rt, adr); }
void CodeGenerator::ldaddal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclral(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeoral(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 2, rs, rt, adr); }
void CodeGenerator::ldsetal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsminal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 6, rs, rt, adr); }
void CodeGenerator::lduminal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 7, rs, rt, adr); }
void CodeGenerator::swapal(const WReg &rs, const WReg &rt, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 1, 0, rs, rt, adr); }
void CodeGenerator::ldadd(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclr(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeor(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 2, rs, rt, adr); }
void CodeGenerator::ldset(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmax(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsmin(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumax(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 6, rs, rt, adr); }
void CodeGenerator::ldumin(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 7, rs, rt, adr); }
void CodeGenerator::swap(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 1, 0, rs, rt, adr); }
void CodeGenerator::ldaddl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclrl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeorl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 2, rs, rt, adr); }
void CodeGenerator::ldsetl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsminl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 6, rs, rt, adr); }
void CodeGenerator::lduminl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 7, rs, rt, adr); }
void CodeGenerator::swapl(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 1, 0, rs, rt, adr); }
void CodeGenerator::ldadda(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclra(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeora(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 2, rs, rt, adr); }
void CodeGenerator::ldseta(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxa(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsmina(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxa(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 6, rs, rt, adr); }
void CodeGenerator::ldumina(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 7, rs, rt, adr); }
void CodeGenerator::swapa(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 1, 0, rs, rt, adr); }
void CodeGenerator::ldapr(const XReg &rt, const AdrImm &adr) { AtomicMemOp(3, 0, 1, 0, 1, 4, XReg(31), rt, adr); }
void CodeGenerator::ldaddal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 0, rs, rt, adr); }
void CodeGenerator::ldclral(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 1, rs, rt, adr); }
void CodeGenerator::ldeoral(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 2, rs, rt, adr); }
void CodeGenerator::ldsetal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 3, rs, rt, adr); }
void CodeGenerator::ldsmaxal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 4, rs, rt, adr); }
void CodeGenerator::ldsminal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 5, rs, rt, adr); }
void CodeGenerator::ldumaxal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 6, rs, rt, adr); }
void CodeGenerator::lduminal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 7, rs, rt, adr); }
void CodeGenerator::swapal(const XReg &rs, const XReg &rt, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 1, 0, rs, rt, adr); }
void CodeGenerator::staddb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 0, rs, WReg(31), adr); }
void CodeGenerator::stclrb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 1, rs, WReg(31), adr); }
void CodeGenerator::steorb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 2, rs, WReg(31), adr); }
void CodeGenerator::stsetb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 3, rs, WReg(31), adr); }
void CodeGenerator::stsmaxb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 4, rs, WReg(31), adr); }
void CodeGenerator::stsminb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 5, rs, WReg(31), adr); }
void CodeGenerator::stumaxb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 6, rs, WReg(31), adr); }
void CodeGenerator::stuminb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 0, 0, 7, rs, WReg(31), adr); }
void CodeGenerator::staddlb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 0, rs, WReg(31), adr); }
void CodeGenerator::stclrlb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 1, rs, WReg(31), adr); }
void CodeGenerator::steorlb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 2, rs, WReg(31), adr); }
void CodeGenerator::stsetlb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 3, rs, WReg(31), adr); }
void CodeGenerator::stsmaxlb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 4, rs, WReg(31), adr); }
void CodeGenerator::stsminlb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 5, rs, WReg(31), adr); }
void CodeGenerator::stumaxlb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 6, rs, WReg(31), adr); }
void CodeGenerator::stuminlb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 0, 1, 0, 7, rs, WReg(31), adr); }
void CodeGenerator::staddab(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 0, rs, WReg(31), adr); }
void CodeGenerator::stclrab(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 1, rs, WReg(31), adr); }
void CodeGenerator::steorab(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 2, rs, WReg(31), adr); }
void CodeGenerator::stsetab(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 3, rs, WReg(31), adr); }
void CodeGenerator::stsmaxab(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 4, rs, WReg(31), adr); }
void CodeGenerator::stsminab(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 5, rs, WReg(31), adr); }
void CodeGenerator::stumaxab(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 6, rs, WReg(31), adr); }
void CodeGenerator::stuminab(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 0, 0, 7, rs, WReg(31), adr); }
void CodeGenerator::staddalb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 0, rs, WReg(31), adr); }
void CodeGenerator::stclralb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 1, rs, WReg(31), adr); }
void CodeGenerator::steoralb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 2, rs, WReg(31), adr); }
void CodeGenerator::stsetalb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 3, rs, WReg(31), adr); }
void CodeGenerator::stsmaxalb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 4, rs, WReg(31), adr); }
void CodeGenerator::stsminalb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 5, rs, WReg(31), adr); }
void CodeGenerator::stumaxalb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 6, rs, WReg(31), adr); }
void CodeGenerator::stuminalb(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(0, 0, 1, 1, 0, 7, rs, WReg(31), adr); }
void CodeGenerator::staddh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 0, rs, WReg(31), adr); }
void CodeGenerator::stclrh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 1, rs, WReg(31), adr); }
void CodeGenerator::steorh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 2, rs, WReg(31), adr); }
void CodeGenerator::stseth(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 3, rs, WReg(31), adr); }
void CodeGenerator::stsmaxh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 4, rs, WReg(31), adr); }
void CodeGenerator::stsminh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 5, rs, WReg(31), adr); }
void CodeGenerator::stumaxh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 6, rs, WReg(31), adr); }
void CodeGenerator::stuminh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 0, 0, 7, rs, WReg(31), adr); }
void CodeGenerator::staddlh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 0, rs, WReg(31), adr); }
void CodeGenerator::stclrlh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 1, rs, WReg(31), adr); }
void CodeGenerator::steorlh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 2, rs, WReg(31), adr); }
void CodeGenerator::stsetlh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 3, rs, WReg(31), adr); }
void CodeGenerator::stsmaxlh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 4, rs, WReg(31), adr); }
void CodeGenerator::stsminlh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 5, rs, WReg(31), adr); }
void CodeGenerator::stumaxlh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 6, rs, WReg(31), adr); }
void CodeGenerator::stuminlh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 0, 1, 0, 7, rs, WReg(31), adr); }
void CodeGenerator::staddah(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 0, rs, WReg(31), adr); }
void CodeGenerator::stclrah(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 1, rs, WReg(31), adr); }
void CodeGenerator::steorah(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 2, rs, WReg(31), adr); }
void CodeGenerator::stsetah(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 3, rs, WReg(31), adr); }
void CodeGenerator::stsmaxah(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 4, rs, WReg(31), adr); }
void CodeGenerator::stsminah(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 5, rs, WReg(31), adr); }
void CodeGenerator::stumaxah(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 6, rs, WReg(31), adr); }
void CodeGenerator::stuminah(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 0, 0, 7, rs, WReg(31), adr); }
void CodeGenerator::staddalh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 0, rs, WReg(31), adr); }
void CodeGenerator::stclralh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 1, rs, WReg(31), adr); }
void CodeGenerator::steoralh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 2, rs, WReg(31), adr); }
void CodeGenerator::stsetalh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 3, rs, WReg(31), adr); }
void CodeGenerator::stsmaxalh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 4, rs, WReg(31), adr); }
void CodeGenerator::stsminalh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 5, rs, WReg(31), adr); }
void CodeGenerator::stumaxalh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 6, rs, WReg(31), adr); }
void CodeGenerator::stuminalh(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(1, 0, 1, 1, 0, 7, rs, WReg(31), adr); }
void CodeGenerator::stadd(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 0, rs, WReg(31), adr); }
void CodeGenerator::stclr(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 1, rs, WReg(31), adr); }
void CodeGenerator::steor(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 2, rs, WReg(31), adr); }
void CodeGenerator::stset(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 3, rs, WReg(31), adr); }
void CodeGenerator::stsmax(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 4, rs, WReg(31), adr); }
void CodeGenerator::stsmin(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 5, rs, WReg(31), adr); }
void CodeGenerator::stumax(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 6, rs, WReg(31), adr); }
void CodeGenerator::stumin(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 0, 0, 7, rs, WReg(31), adr); }
void CodeGenerator::staddl(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 0, rs, WReg(31), adr); }
void CodeGenerator::stclrl(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 1, rs, WReg(31), adr); }
void CodeGenerator::steorl(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 2, rs, WReg(31), adr); }
void CodeGenerator::stsetl(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 3, rs, WReg(31), adr); }
void CodeGenerator::stsmaxl(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 4, rs, WReg(31), adr); }
void CodeGenerator::stsminl(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 5, rs, WReg(31), adr); }
void CodeGenerator::stumaxl(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 6, rs, WReg(31), adr); }
void CodeGenerator::stuminl(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 0, 1, 0, 7, rs, WReg(31), adr); }
void CodeGenerator::stadda(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 0, rs, WReg(31), adr); }
void CodeGenerator::stclra(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 1, rs, WReg(31), adr); }
void CodeGenerator::steora(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 2, rs, WReg(31), adr); }
void CodeGenerator::stseta(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 3, rs, WReg(31), adr); }
void CodeGenerator::stsmaxa(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 4, rs, WReg(31), adr); }
void CodeGenerator::stsmina(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 5, rs, WReg(31), adr); }
void CodeGenerator::stumaxa(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 6, rs, WReg(31), adr); }
void CodeGenerator::stumina(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 0, 0, 7, rs, WReg(31), adr); }
void CodeGenerator::staddal(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 0, rs, WReg(31), adr); }
void CodeGenerator::stclral(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 1, rs, WReg(31), adr); }
void CodeGenerator::steoral(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 2, rs, WReg(31), adr); }
void CodeGenerator::stsetal(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 3, rs, WReg(31), adr); }
void CodeGenerator::stsmaxal(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 4, rs, WReg(31), adr); }
void CodeGenerator::stsminal(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 5, rs, WReg(31), adr); }
void CodeGenerator::stumaxal(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 6, rs, WReg(31), adr); }
void CodeGenerator::stuminal(const WReg &rs, const AdrNoOfs &adr) { AtomicMemOp(2, 0, 1, 1, 0, 7, rs, WReg(31), adr); }
void CodeGenerator::stadd(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 0, rs, XReg(31), adr); }
void CodeGenerator::stclr(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 1, rs, XReg(31), adr); }
void CodeGenerator::steor(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 2, rs, XReg(31), adr); }
void CodeGenerator::stset(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 3, rs, XReg(31), adr); }
void CodeGenerator::stsmax(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 4, rs, XReg(31), adr); }
void CodeGenerator::stsmin(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 5, rs, XReg(31), adr); }
void CodeGenerator::stumax(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 6, rs, XReg(31), adr); }
void CodeGenerator::stumin(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 0, 0, 7, rs, XReg(31), adr); }
void CodeGenerator::staddl(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 0, rs, XReg(31), adr); }
void CodeGenerator::stclrl(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 1, rs, XReg(31), adr); }
void CodeGenerator::steorl(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 2, rs, XReg(31), adr); }
void CodeGenerator::stsetl(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 3, rs, XReg(31), adr); }
void CodeGenerator::stsmaxl(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 4, rs, XReg(31), adr); }
void CodeGenerator::stsminl(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 5, rs, XReg(31), adr); }
void CodeGenerator::stumaxl(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 6, rs, XReg(31), adr); }
void CodeGenerator::stuminl(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 0, 1, 0, 7, rs, XReg(31), adr); }
void CodeGenerator::stadda(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 0, rs, XReg(31), adr); }
void CodeGenerator::stclra(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 1, rs, XReg(31), adr); }
void CodeGenerator::steora(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 2, rs, XReg(31), adr); }
void CodeGenerator::stseta(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 3, rs, XReg(31), adr); }
void CodeGenerator::stsmaxa(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 4, rs, XReg(31), adr); }
void CodeGenerator::stsmina(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 5, rs, XReg(31), adr); }
void CodeGenerator::stumaxa(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 6, rs, XReg(31), adr); }
void CodeGenerator::stumina(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 0, 0, 7, rs, XReg(31), adr); }
void CodeGenerator::staddal(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 0, rs, XReg(31), adr); }
void CodeGenerator::stclral(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 1, rs, XReg(31), adr); }
void CodeGenerator::steoral(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 2, rs, XReg(31), adr); }
void CodeGenerator::stsetal(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 3, rs, XReg(31), adr); }
void CodeGenerator::stsmaxal(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 4, rs, XReg(31), adr); }
void CodeGenerator::stsminal(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 5, rs, XReg(31), adr); }
void CodeGenerator::stumaxal(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 6, rs, XReg(31), adr); }
void CodeGenerator::stuminal(const XReg &rs, const AdrNoOfs &adr) { AtomicMemOp(3, 0, 1, 1, 0, 7, rs, XReg(31), adr); }
void CodeGenerator::strb(const WReg &rt, const AdrReg &adr) { LdStReg(0, 0, rt, adr); }
void CodeGenerator::strb(const WReg &rt, const AdrExt &adr) { LdStReg(0, 0, rt, adr); }
void CodeGenerator::ldrb(const WReg &rt, const AdrReg &adr) { LdStReg(0, 1, rt, adr); }
void CodeGenerator::ldrb(const WReg &rt, const AdrExt &adr) { LdStReg(0, 1, rt, adr); }
void CodeGenerator::ldrsb(const XReg &rt, const AdrReg &adr) { LdStReg(0, 2, rt, adr); }
void CodeGenerator::ldrsb(const XReg &rt, const AdrExt &adr) { LdStReg(0, 2, rt, adr); }
void CodeGenerator::ldrsb(const WReg &rt, const AdrReg &adr) { LdStReg(0, 3, rt, adr); }
void CodeGenerator::ldrsb(const WReg &rt, const AdrExt &adr) { LdStReg(0, 3, rt, adr); }
void CodeGenerator::strh(const WReg &rt, const AdrReg &adr) { LdStReg(1, 0, rt, adr); }
void CodeGenerator::strh(const WReg &rt, const AdrExt &adr) { LdStReg(1, 0, rt, adr); }
void CodeGenerator::ldrh(const WReg &rt, const AdrReg &adr) { LdStReg(1, 1, rt, adr); }
void CodeGenerator::ldrh(const WReg &rt, const AdrExt &adr) { LdStReg(1, 1, rt, adr); }
void CodeGenerator::ldrsh(const XReg &rt, const AdrReg &adr) { LdStReg(1, 2, rt, adr); }
void CodeGenerator::ldrsh(const XReg &rt, const AdrExt &adr) { LdStReg(1, 2, rt, adr); }
void CodeGenerator::ldrsh(const WReg &rt, const AdrReg &adr) { LdStReg(1, 3, rt, adr); }
void CodeGenerator::ldrsh(const WReg &rt, const AdrExt &adr) { LdStReg(1, 3, rt, adr); }
void CodeGenerator::str(const WReg &rt, const AdrReg &adr) { LdStReg(2, 0, rt, adr); }
void CodeGenerator::str(const WReg &rt, const AdrExt &adr) { LdStReg(2, 0, rt, adr); }
void CodeGenerator::ldr(const WReg &rt, const AdrReg &adr) { LdStReg(2, 1, rt, adr); }
void CodeGenerator::ldr(const WReg &rt, const AdrExt &adr) { LdStReg(2, 1, rt, adr); }
void CodeGenerator::ldrsw(const XReg &rt, const AdrReg &adr) { LdStReg(2, 2, rt, adr); }
void CodeGenerator::ldrsw(const XReg &rt, const AdrExt &adr) { LdStReg(2, 2, rt, adr); }
void CodeGenerator::str(const XReg &rt, const AdrReg &adr) { LdStReg(3, 0, rt, adr); }
void CodeGenerator::str(const XReg &rt, const AdrExt &adr) { LdStReg(3, 0, rt, adr); }
void CodeGenerator::ldr(const XReg &rt, const AdrReg &adr) { LdStReg(3, 1, rt, adr); }
void CodeGenerator::ldr(const XReg &rt, const AdrExt &adr) { LdStReg(3, 1, rt, adr); }
void CodeGenerator::str(const BReg &vt, const AdrReg &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const HReg &vt, const AdrReg &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const SReg &vt, const AdrReg &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const DReg &vt, const AdrReg &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const QReg &vt, const AdrReg &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const BReg &vt, const AdrExt &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const HReg &vt, const AdrExt &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const SReg &vt, const AdrExt &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const DReg &vt, const AdrExt &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const QReg &vt, const AdrExt &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::ldr(const BReg &vt, const AdrReg &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const HReg &vt, const AdrReg &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const SReg &vt, const AdrReg &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const DReg &vt, const AdrReg &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const QReg &vt, const AdrReg &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const BReg &vt, const AdrExt &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const HReg &vt, const AdrExt &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const SReg &vt, const AdrExt &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const DReg &vt, const AdrExt &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const QReg &vt, const AdrExt &adr) { LdStSimdFpReg((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::prfm(const Prfop prfop, const AdrReg &adr) { PfExt(prfop, adr); }
void CodeGenerator::prfm(const Prfop prfop, const AdrExt &adr) { PfExt(prfop, adr); }
void CodeGenerator::ldraa(const XReg &xt, const AdrImm &adr) { LdStRegPac(0, 0, xt, adr); }
void CodeGenerator::ldrab(const XReg &xt, const AdrImm &adr) { LdStRegPac(1, 0, xt, adr); }
void CodeGenerator::ldraa(const XReg &xt, const AdrPreImm &adr) { LdStRegPac(0, 1, xt, adr); }
void CodeGenerator::ldrab(const XReg &xt, const AdrPreImm &adr) { LdStRegPac(1, 1, xt, adr); }
void CodeGenerator::strb(const WReg &rt, const AdrUimm &adr) { LdStRegUnImm(0, 0, rt, adr); }
void CodeGenerator::ldrb(const WReg &rt, const AdrUimm &adr) { LdStRegUnImm(0, 1, rt, adr); }
void CodeGenerator::ldrsb(const XReg &rt, const AdrUimm &adr) { LdStRegUnImm(0, 2, rt, adr); }
void CodeGenerator::ldrsb(const WReg &rt, const AdrUimm &adr) { LdStRegUnImm(0, 3, rt, adr); }
void CodeGenerator::strh(const WReg &rt, const AdrUimm &adr) { LdStRegUnImm(1, 0, rt, adr); }
void CodeGenerator::ldrh(const WReg &rt, const AdrUimm &adr) { LdStRegUnImm(1, 1, rt, adr); }
void CodeGenerator::ldrsh(const XReg &rt, const AdrUimm &adr) { LdStRegUnImm(1, 2, rt, adr); }
void CodeGenerator::ldrsh(const WReg &rt, const AdrUimm &adr) { LdStRegUnImm(1, 3, rt, adr); }
void CodeGenerator::str(const WReg &rt, const AdrUimm &adr) { LdStRegUnImm(2, 0, rt, adr); }
void CodeGenerator::ldr(const WReg &rt, const AdrUimm &adr) { LdStRegUnImm(2, 1, rt, adr); }
void CodeGenerator::ldrsw(const XReg &rt, const AdrUimm &adr) { LdStRegUnImm(2, 2, rt, adr); }
void CodeGenerator::str(const XReg &rt, const AdrUimm &adr) { LdStRegUnImm(3, 0, rt, adr); }
void CodeGenerator::ldr(const XReg &rt, const AdrUimm &adr) { LdStRegUnImm(3, 1, rt, adr); }
void CodeGenerator::str(const BReg &vt, const AdrUimm &adr) { LdStSimdFpUnImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const HReg &vt, const AdrUimm &adr) { LdStSimdFpUnImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const SReg &vt, const AdrUimm &adr) { LdStSimdFpUnImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const DReg &vt, const AdrUimm &adr) { LdStSimdFpUnImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::str(const QReg &vt, const AdrUimm &adr) { LdStSimdFpUnImm((vt.getBit() != 128) ? 0 : 2, vt, adr); }
void CodeGenerator::ldr(const BReg &vt, const AdrUimm &adr) { LdStSimdFpUnImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const HReg &vt, const AdrUimm &adr) { LdStSimdFpUnImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const SReg &vt, const AdrUimm &adr) { LdStSimdFpUnImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const DReg &vt, const AdrUimm &adr) { LdStSimdFpUnImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::ldr(const QReg &vt, const AdrUimm &adr) { LdStSimdFpUnImm((vt.getBit() != 128) ? 1 : 3, vt, adr); }
void CodeGenerator::prfm(const Prfop prfop, const AdrUimm &adr) { PfRegImm(prfop, adr); }
void CodeGenerator::udiv(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(2, rd, rn, rm); }
void CodeGenerator::sdiv(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(3, rd, rn, rm); }
void CodeGenerator::lslv(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(8, rd, rn, rm); }
void CodeGenerator::lsl(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(8, rd, rn, rm); }
void CodeGenerator::lsrv(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(9, rd, rn, rm); }
void CodeGenerator::lsr(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(9, rd, rn, rm); }
void CodeGenerator::asrv(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(10, rd, rn, rm); }
void CodeGenerator::asr(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(10, rd, rn, rm); }
void CodeGenerator::rorv(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(11, rd, rn, rm); }
void CodeGenerator::ror(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(11, rd, rn, rm); }
void CodeGenerator::crc32b(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(16, rd, rn, rm); }
void CodeGenerator::crc32h(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(17, rd, rn, rm); }
void CodeGenerator::crc32w(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(18, rd, rn, rm); }
void CodeGenerator::crc32cb(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(20, rd, rn, rm); }
void CodeGenerator::crc32ch(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(21, rd, rn, rm); }
void CodeGenerator::crc32cw(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc2Src(22, rd, rn, rm); }
void CodeGenerator::udiv(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc2Src(2, rd, rn, rm); }
void CodeGenerator::sdiv(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc2Src(3, rd, rn, rm); }
void CodeGenerator::lslv(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc2Src(8, rd, rn, rm); }
void CodeGenerator::lsl(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc2Src(8, rd, rn, rm); }
void CodeGenerator::lsrv(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc2Src(9, rd, rn, rm); }
void CodeGenerator::lsr(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc2Src(9, rd, rn, rm); }
void CodeGenerator::asrv(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc2Src(10, rd, rn, rm); }
void CodeGenerator::asr(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc2Src(10, rd, rn, rm); }
void CodeGenerator::rorv(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc2Src(11, rd, rn, rm); }
void CodeGenerator::ror(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc2Src(11, rd, rn, rm); }
void CodeGenerator::pacga(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc2Src(12, rd, rn, rm); }
void CodeGenerator::crc32x(const WReg &rd, const WReg &rn, const XReg &rm) { DataProc2Src(19, rd, rn, rm); }
void CodeGenerator::crc32cx(const WReg &rd, const WReg &rn, const XReg &rm) { DataProc2Src(23, rd, rn, rm); }
void CodeGenerator::rbit(const WReg &rd, const WReg &rn) { DataProc1Src(0, 0, rd, rn); }
void CodeGenerator::rev16(const WReg &rd, const WReg &rn) { DataProc1Src(0, 1, rd, rn); }
void CodeGenerator::rev(const WReg &rd, const WReg &rn) { DataProc1Src(0, 2, rd, rn); }
void CodeGenerator::clz(const WReg &rd, const WReg &rn) { DataProc1Src(0, 4, rd, rn); }
void CodeGenerator::cls(const WReg &rd, const WReg &rn) { DataProc1Src(0, 5, rd, rn); }
void CodeGenerator::rbit(const XReg &rd, const XReg &rn) { DataProc1Src(0, 0, rd, rn); }
void CodeGenerator::rev16(const XReg &rd, const XReg &rn) { DataProc1Src(0, 1, rd, rn); }
void CodeGenerator::rev32(const XReg &rd, const XReg &rn) { DataProc1Src(0, 2, rd, rn); }
void CodeGenerator::rev(const XReg &rd, const XReg &rn) { DataProc1Src(0, 3, rd, rn); }
void CodeGenerator::rev64(const XReg &rd, const XReg &rn) { DataProc1Src(0, 3, rd, rn); }
void CodeGenerator::clz(const XReg &rd, const XReg &rn) { DataProc1Src(0, 4, rd, rn); }
void CodeGenerator::cls(const XReg &rd, const XReg &rn) { DataProc1Src(0, 5, rd, rn); }
void CodeGenerator::pacia(const XReg &rd, const XReg &rn) { DataProc1Src(1, 0, rd, rn); }
void CodeGenerator::pacib(const XReg &rd, const XReg &rn) { DataProc1Src(1, 1, rd, rn); }
void CodeGenerator::pacda(const XReg &rd, const XReg &rn) { DataProc1Src(1, 2, rd, rn); }
void CodeGenerator::pacdb(const XReg &rd, const XReg &rn) { DataProc1Src(1, 3, rd, rn); }
void CodeGenerator::autia(const XReg &rd, const XReg &rn) { DataProc1Src(1, 4, rd, rn); }
void CodeGenerator::autib(const XReg &rd, const XReg &rn) { DataProc1Src(1, 5, rd, rn); }
void CodeGenerator::autda(const XReg &rd, const XReg &rn) { DataProc1Src(1, 6, rd, rn); }
void CodeGenerator::autdb(const XReg &rd, const XReg &rn) { DataProc1Src(1, 7, rd, rn); }
void CodeGenerator::paciza(const XReg &rd) { DataProc1Src(1, 8, rd); }
void CodeGenerator::pacizb(const XReg &rd) { DataProc1Src(1, 9, rd); }
void CodeGenerator::pacdza(const XReg &rd) { DataProc1Src(1, 10, rd); }
void CodeGenerator::pacdzb(const XReg &rd) { DataProc1Src(1, 11, rd); }
void CodeGenerator::autiza(const XReg &rd) { DataProc1Src(1, 12, rd); }
void CodeGenerator::autizb(const XReg &rd) { DataProc1Src(1, 13, rd); }
void CodeGenerator::autdza(const XReg &rd) { DataProc1Src(1, 14, rd); }
void CodeGenerator::autdzb(const XReg &rd) { DataProc1Src(1, 15, rd); }
void CodeGenerator::xpaci(const XReg &rd) { DataProc1Src(1, 16, rd); }
void CodeGenerator::xpacd(const XReg &rd) { DataProc1Src(1, 17, rd); }
void CodeGenerator::and_(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(0, 0, rd, rn, rm, shmod, sh); }
void CodeGenerator::and_(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(0, 0, rd, rn, rm, shmod, sh); }
void CodeGenerator::bic(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(0, 1, rd, rn, rm, shmod, sh); }
void CodeGenerator::bic(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(0, 1, rd, rn, rm, shmod, sh); }
void CodeGenerator::orr(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(1, 0, rd, rn, rm, shmod, sh); }
void CodeGenerator::orr(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(1, 0, rd, rn, rm, shmod, sh); }
void CodeGenerator::orn(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(1, 1, rd, rn, rm, shmod, sh); }
void CodeGenerator::orn(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(1, 1, rd, rn, rm, shmod, sh); }
void CodeGenerator::mvn(const WReg &rd, const WReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(1, 1, rd, WReg(31), rm, shmod, sh); }
void CodeGenerator::mvn(const XReg &rd, const XReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(1, 1, rd, XReg(31), rm, shmod, sh); }
void CodeGenerator::eor(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(2, 0, rd, rn, rm, shmod, sh); }
void CodeGenerator::eor(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(2, 0, rd, rn, rm, shmod, sh); }
void CodeGenerator::eon(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(2, 1, rd, rn, rm, shmod, sh); }
void CodeGenerator::eon(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(2, 1, rd, rn, rm, shmod, sh); }
void CodeGenerator::ands(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(3, 0, rd, rn, rm, shmod, sh); }
void CodeGenerator::ands(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(3, 0, rd, rn, rm, shmod, sh); }
void CodeGenerator::tst(const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(3, 0, WReg(31), rn, rm, shmod, sh); }
void CodeGenerator::tst(const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(3, 0, XReg(31), rn, rm, shmod, sh); }
void CodeGenerator::bics(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(3, 1, rd, rn, rm, shmod, sh); }
void CodeGenerator::bics(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { LogicalShiftReg(3, 1, rd, rn, rm, shmod, sh); }
void CodeGenerator::mov(const WReg &rd, const WReg &rn) { MvReg(rd, rn); }
void CodeGenerator::mov(const XReg &rd, const XReg &rn) { MvReg(rd, rn); }
void CodeGenerator::add(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(0, 0, rd, rn, rm, shmod, sh); }
void CodeGenerator::add(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(0, 0, rd, rn, rm, shmod, sh); }
void CodeGenerator::adds(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(0, 1, rd, rn, rm, shmod, sh); }
void CodeGenerator::adds(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(0, 1, rd, rn, rm, shmod, sh); }
void CodeGenerator::cmn(const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(0, 1, WReg(31), rn, rm, shmod, sh, true); }
void CodeGenerator::cmn(const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(0, 1, XReg(31), rn, rm, shmod, sh, true); }
void CodeGenerator::sub(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(1, 0, rd, rn, rm, shmod, sh); }
void CodeGenerator::sub(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(1, 0, rd, rn, rm, shmod, sh); }
void CodeGenerator::neg(const WReg &rd, const WReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(1, 0, rd, WReg(31), rm, shmod, sh, true); }
void CodeGenerator::neg(const XReg &rd, const XReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(1, 0, rd, XReg(31), rm, shmod, sh, true); }
void CodeGenerator::subs(const WReg &rd, const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(1, 1, rd, rn, rm, shmod, sh); }
void CodeGenerator::subs(const XReg &rd, const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(1, 1, rd, rn, rm, shmod, sh); }
void CodeGenerator::negs(const WReg &rd, const WReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(1, 1, rd, WReg(31), rm, shmod, sh, true); }
void CodeGenerator::negs(const XReg &rd, const XReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(1, 1, rd, XReg(31), rm, shmod, sh, true); }
void CodeGenerator::cmp(const WReg &rn, const WReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(1, 1, WReg(31), rn, rm, shmod, sh, true); }
void CodeGenerator::cmp(const XReg &rn, const XReg &rm, const ShMod shmod, const uint32_t sh) { AddSubShiftReg(1, 1, XReg(31), rn, rm, shmod, sh, true); }
void CodeGenerator::add(const WReg &rd, const WReg &rn, const WReg &rm, const ExtMod extmod, const uint32_t sh) { AddSubExtReg(0, 0, rd, rn, rm, extmod, sh); }
void CodeGenerator::add(const XReg &rd, const XReg &rn, const XReg &rm, const ExtMod extmod, const uint32_t sh) { AddSubExtReg(0, 0, rd, rn, rm, extmod, sh); }
void CodeGenerator::adds(const WReg &rd, const WReg &rn, const WReg &rm, const ExtMod extmod, const uint32_t sh) { AddSubExtReg(0, 1, rd, rn, rm, extmod, sh); }
void CodeGenerator::adds(const XReg &rd, const XReg &rn, const XReg &rm, const ExtMod extmod, const uint32_t sh) { AddSubExtReg(0, 1, rd, rn, rm, extmod, sh); }
void CodeGenerator::cmn(const WReg &rn, const WReg &rm, const ExtMod extmod, const uint32_t sh) { AddSubExtReg(0, 1, WReg(31), rn, rm, extmod, sh); }
void CodeGenerator::cmn(const XReg &rn, const XReg &rm, const ExtMod extmod, const uint32_t sh) { AddSubExtReg(0, 1, XReg(31), rn, rm, extmod, sh); }
void CodeGenerator::sub(const WReg &rd, const WReg &rn, const WReg &rm, const ExtMod extmod, const uint32_t sh) { AddSubExtReg(1, 0, rd, rn, rm, extmod, sh); }
void CodeGenerator::sub(const XReg &rd, const XReg &rn, const XReg &rm, const ExtMod extmod, const uint32_t sh) { AddSubExtReg(1, 0, rd, rn, rm, extmod, sh); }
void CodeGenerator::subs(const WReg &rd, const WReg &rn, const WReg &rm, const ExtMod extmod, const uint32_t sh) { AddSubExtReg(1, 1, rd, rn, rm, extmod, sh); }
void CodeGenerator::subs(const XReg &rd, const XReg &rn, const XReg &rm, const ExtMod extmod, const uint32_t sh) { AddSubExtReg(1, 1, rd, rn, rm, extmod, sh); }
void CodeGenerator::cmp(const WReg &rn, const WReg &rm, const ExtMod extmod, const uint32_t sh) { AddSubExtReg(1, 1, WReg(31), rn, rm, extmod, sh); }
void CodeGenerator::cmp(const XReg &rn, const XReg &rm, const ExtMod extmod, const uint32_t sh) { AddSubExtReg(1, 1, XReg(31), rn, rm, extmod, sh); }
void CodeGenerator::adc(const WReg &rd, const WReg &rn, const WReg &rm) { AddSubCarry(0, 0, rd, rn, rm); }
void CodeGenerator::adc(const XReg &rd, const XReg &rn, const XReg &rm) { AddSubCarry(0, 0, rd, rn, rm); }
void CodeGenerator::adcs(const WReg &rd, const WReg &rn, const WReg &rm) { AddSubCarry(0, 1, rd, rn, rm); }
void CodeGenerator::adcs(const XReg &rd, const XReg &rn, const XReg &rm) { AddSubCarry(0, 1, rd, rn, rm); }
void CodeGenerator::sbc(const WReg &rd, const WReg &rn, const WReg &rm) { AddSubCarry(1, 0, rd, rn, rm); }
void CodeGenerator::sbc(const XReg &rd, const XReg &rn, const XReg &rm) { AddSubCarry(1, 0, rd, rn, rm); }
void CodeGenerator::ngc(const WReg &rd, const WReg &rm) { AddSubCarry(1, 0, rd, WReg(31), rm); }
void CodeGenerator::ngc(const XReg &rd, const XReg &rm) { AddSubCarry(1, 0, rd, XReg(31), rm); }
void CodeGenerator::sbcs(const WReg &rd, const WReg &rn, const WReg &rm) { AddSubCarry(1, 1, rd, rn, rm); }
void CodeGenerator::sbcs(const XReg &rd, const XReg &rn, const XReg &rm) { AddSubCarry(1, 1, rd, rn, rm); }
void CodeGenerator::ngcs(const WReg &rd, const WReg &rm) { AddSubCarry(1, 1, rd, WReg(31), rm); }
void CodeGenerator::ngcs(const XReg &rd, const XReg &rm) { AddSubCarry(1, 1, rd, XReg(31), rm); }
void CodeGenerator::rmif(const XReg &xn, const uint32_t sh, const uint32_t mask) { RotateR(0, 1, 0, xn, sh, mask); }
void CodeGenerator::setf8(const WReg &wn) { Evaluate(0, 1, 0, 0, 0, 13, wn); }
void CodeGenerator::setf16(const WReg &wn) { Evaluate(0, 1, 0, 1, 0, 13, wn); }
void CodeGenerator::ccmn(const WReg &rn, const WReg &rm, const uint32_t nczv, const Cond cond) { CondCompReg(0, 1, 0, 0, rn, rm, nczv, cond); }
void CodeGenerator::ccmn(const XReg &rn, const XReg &rm, const uint32_t nczv, const Cond cond) { CondCompReg(0, 1, 0, 0, rn, rm, nczv, cond); }
void CodeGenerator::ccmp(const WReg &rn, const WReg &rm, const uint32_t nczv, const Cond cond) { CondCompReg(1, 1, 0, 0, rn, rm, nczv, cond); }
void CodeGenerator::ccmp(const XReg &rn, const XReg &rm, const uint32_t nczv, const Cond cond) { CondCompReg(1, 1, 0, 0, rn, rm, nczv, cond); }
void CodeGenerator::ccmn(const WReg &rn, const uint32_t imm, const uint32_t nczv, const Cond cond) { CondCompImm(0, 1, 0, 0, rn, imm, nczv, cond); }
void CodeGenerator::ccmn(const XReg &rn, const uint32_t imm, const uint32_t nczv, const Cond cond) { CondCompImm(0, 1, 0, 0, rn, imm, nczv, cond); }
void CodeGenerator::ccmp(const WReg &rn, const uint32_t imm, const uint32_t nczv, const Cond cond) { CondCompImm(1, 1, 0, 0, rn, imm, nczv, cond); }
void CodeGenerator::ccmp(const XReg &rn, const uint32_t imm, const uint32_t nczv, const Cond cond) { CondCompImm(1, 1, 0, 0, rn, imm, nczv, cond); }
void CodeGenerator::csel(const WReg &rd, const WReg &rn, const WReg &rm, const Cond cond) { CondSel(0, 0, 0, rd, rn, rm, cond); }
void CodeGenerator::csel(const XReg &rd, const XReg &rn, const XReg &rm, const Cond cond) { CondSel(0, 0, 0, rd, rn, rm, cond); }
void CodeGenerator::csinc(const WReg &rd, const WReg &rn, const WReg &rm, const Cond cond) { CondSel(0, 0, 1, rd, rn, rm, cond); }
void CodeGenerator::csinc(const XReg &rd, const XReg &rn, const XReg &rm, const Cond cond) { CondSel(0, 0, 1, rd, rn, rm, cond); }
void CodeGenerator::cinc(const WReg &rd, const WReg &rn, const Cond cond) { CondSel(0, 0, 1, rd, rn, rn, invert(cond)); }
void CodeGenerator::cinc(const XReg &rd, const XReg &rn, const Cond cond) { CondSel(0, 0, 1, rd, rn, rn, invert(cond)); }
void CodeGenerator::cset(const WReg &rd, const Cond cond) { CondSel(0, 0, 1, rd, WReg(31), WReg(31), invert(cond)); }
void CodeGenerator::cset(const XReg &rd, const Cond cond) { CondSel(0, 0, 1, rd, XReg(31), XReg(31), invert(cond)); }
void CodeGenerator::csinv(const WReg &rd, const WReg &rn, const WReg &rm, const Cond cond) { CondSel(1, 0, 0, rd, rn, rm, cond); }
void CodeGenerator::csinv(const XReg &rd, const XReg &rn, const XReg &rm, const Cond cond) { CondSel(1, 0, 0, rd, rn, rm, cond); }
void CodeGenerator::cinv(const WReg &rd, const WReg &rn, const Cond cond) { CondSel(1, 0, 0, rd, rn, rn, invert(cond)); }
void CodeGenerator::cinv(const XReg &rd, const XReg &rn, const Cond cond) { CondSel(1, 0, 0, rd, rn, rn, invert(cond)); }
void CodeGenerator::csetm(const WReg &rd, const Cond cond) { CondSel(1, 0, 0, rd, WReg(31), WReg(31), invert(cond)); }
void CodeGenerator::csetm(const XReg &rd, const Cond cond) { CondSel(1, 0, 0, rd, XReg(31), XReg(31), invert(cond)); }
void CodeGenerator::csneg(const WReg &rd, const WReg &rn, const WReg &rm, const Cond cond) { CondSel(1, 0, 1, rd, rn, rm, cond); }
void CodeGenerator::csneg(const XReg &rd, const XReg &rn, const XReg &rm, const Cond cond) { CondSel(1, 0, 1, rd, rn, rm, cond); }
void CodeGenerator::cneg(const WReg &rd, const WReg &rn, const Cond cond) { CondSel(1, 0, 1, rd, rn, rn, invert(cond)); }
void CodeGenerator::cneg(const XReg &rd, const XReg &rn, const Cond cond) { CondSel(1, 0, 1, rd, rn, rn, invert(cond)); }
void CodeGenerator::madd(const WReg &rd, const WReg &rn, const WReg &rm, const WReg &ra) { DataProc3Reg(0, 0, 0, rd, rn, rm, ra); }
void CodeGenerator::madd(const XReg &rd, const XReg &rn, const XReg &rm, const XReg &ra) { DataProc3Reg(0, 0, 0, rd, rn, rm, ra); }
void CodeGenerator::mul(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc3Reg(0, 0, 0, rd, rn, rm, WReg(31)); }
void CodeGenerator::mul(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc3Reg(0, 0, 0, rd, rn, rm, XReg(31)); }
void CodeGenerator::msub(const WReg &rd, const WReg &rn, const WReg &rm, const WReg &ra) { DataProc3Reg(0, 0, 1, rd, rn, rm, ra); }
void CodeGenerator::msub(const XReg &rd, const XReg &rn, const XReg &rm, const XReg &ra) { DataProc3Reg(0, 0, 1, rd, rn, rm, ra); }
void CodeGenerator::mneg(const WReg &rd, const WReg &rn, const WReg &rm) { DataProc3Reg(0, 0, 1, rd, rn, rm, WReg(31)); }
void CodeGenerator::mneg(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc3Reg(0, 0, 1, rd, rn, rm, XReg(31)); }
void CodeGenerator::smaddl(const XReg &rd, const WReg &rn, const WReg &rm, const XReg &ra) { DataProc3Reg(0, 1, 0, rd, rn, rm, ra); }
void CodeGenerator::smull(const XReg &rd, const WReg &rn, const WReg &rm) { DataProc3Reg(0, 1, 0, rd, rn, rm, XReg(31)); }
void CodeGenerator::smsubl(const XReg &rd, const WReg &rn, const WReg &rm, const XReg &ra) { DataProc3Reg(0, 1, 1, rd, rn, rm, ra); }
void CodeGenerator::smnegl(const XReg &rd, const WReg &rn, const WReg &rm) { DataProc3Reg(0, 1, 1, rd, rn, rm, XReg(31)); }
void CodeGenerator::smulh(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc3Reg(0, 2, 0, rd, rn, rm); }
void CodeGenerator::umaddl(const XReg &rd, const WReg &rn, const WReg &rm, const XReg &ra) { DataProc3Reg(0, 5, 0, rd, rn, rm, ra); }
void CodeGenerator::umull(const XReg &rd, const WReg &rn, const WReg &rm) { DataProc3Reg(0, 5, 0, rd, rn, rm, XReg(31)); }
void CodeGenerator::umsubl(const XReg &rd, const WReg &rn, const WReg &rm, const XReg &ra) { DataProc3Reg(0, 5, 1, rd, rn, rm, ra); }
void CodeGenerator::umnegl(const XReg &rd, const WReg &rn, const WReg &rm) { DataProc3Reg(0, 5, 1, rd, rn, rm, XReg(31)); }
void CodeGenerator::umulh(const XReg &rd, const XReg &rn, const XReg &rm) { DataProc3Reg(0, 6, 0, rd, rn, rm); }
void CodeGenerator::aese(const VReg16B &vd, const VReg16B &vn) { CryptAES(4, vd, vn); }
void CodeGenerator::aesd(const VReg16B &vd, const VReg16B &vn) { CryptAES(5, vd, vn); }
void CodeGenerator::aesmc(const VReg16B &vd, const VReg16B &vn) { CryptAES(6, vd, vn); }
void CodeGenerator::aesimc(const VReg16B &vd, const VReg16B &vn) { CryptAES(7, vd, vn); }
void CodeGenerator::sha1c(const QReg &qd, const SReg &sn, const VReg4S &vm) { Crypt3RegSHA(0, qd, sn, vm); }
void CodeGenerator::sha1p(const QReg &qd, const SReg &sn, const VReg4S &vm) { Crypt3RegSHA(1, qd, sn, vm); }
void CodeGenerator::sha1m(const QReg &qd, const SReg &sn, const VReg4S &vm) { Crypt3RegSHA(2, qd, sn, vm); }
void CodeGenerator::sha1su0(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { Crypt3RegSHA(3, vd, vn, vm); }
void CodeGenerator::sha256h(const QReg &qd, const QReg &qn, const VReg4S &vm) { Crypt3RegSHA(4, qd, qn, vm); }
void CodeGenerator::sha256h2(const QReg &qd, const QReg &qn, const VReg4S &vm) { Crypt3RegSHA(5, qd, qn, vm); }
void CodeGenerator::sha256su1(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { Crypt3RegSHA(6, vd, vn, vm); }
void CodeGenerator::sha1h(const SReg &sd, const SReg &sn) { Crypt2RegSHA(0, sd, sn); }
void CodeGenerator::sha1su1(const VReg4S &vd, const VReg4S &vn) { Crypt2RegSHA(1, vd, vn); }
void CodeGenerator::sha256su0(const VReg4S &vd, const VReg4S &vn) { Crypt2RegSHA(2, vd, vn); }
void CodeGenerator::dup(const BReg &vd, const VRegBElem &vn) { AdvSimdScCopy(0, 0, vd, vn); }
void CodeGenerator::dup(const HReg &vd, const VRegHElem &vn) { AdvSimdScCopy(0, 0, vd, vn); }
void CodeGenerator::dup(const SReg &vd, const VRegSElem &vn) { AdvSimdScCopy(0, 0, vd, vn); }
void CodeGenerator::dup(const DReg &vd, const VRegDElem &vn) { AdvSimdScCopy(0, 0, vd, vn); }
void CodeGenerator::mov(const BReg &vd, const VRegBElem &vn) { AdvSimdScCopy(0, 0, vd, vn); }
void CodeGenerator::mov(const HReg &vd, const VRegHElem &vn) { AdvSimdScCopy(0, 0, vd, vn); }
void CodeGenerator::mov(const SReg &vd, const VRegSElem &vn) { AdvSimdScCopy(0, 0, vd, vn); }
void CodeGenerator::mov(const DReg &vd, const VRegDElem &vn) { AdvSimdScCopy(0, 0, vd, vn); }
void CodeGenerator::fmulx(const HReg &hd, const HReg &hn, const HReg &hm) { AdvSimdSc3SameFp16(0, 0, 3, hd, hn, hm); }
void CodeGenerator::fcmeq(const HReg &hd, const HReg &hn, const HReg &hm) { AdvSimdSc3SameFp16(0, 0, 4, hd, hn, hm); }
void CodeGenerator::frecps(const HReg &hd, const HReg &hn, const HReg &hm) { AdvSimdSc3SameFp16(0, 0, 7, hd, hn, hm); }
void CodeGenerator::frsqrts(const HReg &hd, const HReg &hn, const HReg &hm) { AdvSimdSc3SameFp16(0, 1, 7, hd, hn, hm); }
void CodeGenerator::fcmge(const HReg &hd, const HReg &hn, const HReg &hm) { AdvSimdSc3SameFp16(1, 0, 4, hd, hn, hm); }
void CodeGenerator::facge(const HReg &hd, const HReg &hn, const HReg &hm) { AdvSimdSc3SameFp16(1, 0, 5, hd, hn, hm); }
void CodeGenerator::fabd(const HReg &hd, const HReg &hn, const HReg &hm) { AdvSimdSc3SameFp16(1, 1, 2, hd, hn, hm); }
void CodeGenerator::fcmgt(const HReg &hd, const HReg &hn, const HReg &hm) { AdvSimdSc3SameFp16(1, 1, 4, hd, hn, hm); }
void CodeGenerator::facgt(const HReg &hd, const HReg &hn, const HReg &hm) { AdvSimdSc3SameFp16(1, 1, 5, hd, hn, hm); }
void CodeGenerator::fcvtns(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(0, 0, 26, hd, hn); }
void CodeGenerator::fcvtms(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(0, 0, 27, hd, hn); }
void CodeGenerator::fcvtas(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(0, 0, 28, hd, hn); }
void CodeGenerator::scvtf(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(0, 0, 29, hd, hn); }
void CodeGenerator::fcmgt(const HReg &hd, const HReg &hn, const double zero) { AdvSimdSc2RegMiscFp16(0, 1, 12, hd, hn, zero); }
void CodeGenerator::fcmeq(const HReg &hd, const HReg &hn, const double zero) { AdvSimdSc2RegMiscFp16(0, 1, 13, hd, hn, zero); }
void CodeGenerator::fcmlt(const HReg &hd, const HReg &hn, const double zero) { AdvSimdSc2RegMiscFp16(0, 1, 14, hd, hn, zero); }
void CodeGenerator::fcvtps(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(0, 1, 26, hd, hn); }
void CodeGenerator::fcvtzs(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(0, 1, 27, hd, hn); }
void CodeGenerator::frecpe(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(0, 1, 29, hd, hn); }
void CodeGenerator::frecpx(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(0, 1, 31, hd, hn); }
void CodeGenerator::fcvtnu(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(1, 0, 26, hd, hn); }
void CodeGenerator::fcvtmu(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(1, 0, 27, hd, hn); }
void CodeGenerator::fcvtau(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(1, 0, 28, hd, hn); }
void CodeGenerator::ucvtf(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(1, 0, 29, hd, hn); }
void CodeGenerator::fcmge(const HReg &hd, const HReg &hn, const double zero) { AdvSimdSc2RegMiscFp16(1, 1, 12, hd, hn, zero); }
void CodeGenerator::fcmle(const HReg &hd, const HReg &hn, const double zero) { AdvSimdSc2RegMiscFp16(1, 1, 13, hd, hn, zero); }
void CodeGenerator::fcvtpu(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(1, 1, 26, hd, hn); }
void CodeGenerator::fcvtzu(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(1, 1, 27, hd, hn); }
void CodeGenerator::frsqrte(const HReg &hd, const HReg &hn) { AdvSimdSc2RegMiscFp16(1, 1, 29, hd, hn); }
void CodeGenerator::sqrdmlah(const HReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3SameExtra(1, 0, vd, vn, vm); }
void CodeGenerator::sqrdmlah(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3SameExtra(1, 0, vd, vn, vm); }
void CodeGenerator::sqrdmlsh(const HReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3SameExtra(1, 1, vd, vn, vm); }
void CodeGenerator::sqrdmlsh(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3SameExtra(1, 1, vd, vn, vm); }
void CodeGenerator::suqadd(const BReg &vd, const BReg &vn) { AdvSimdSc2RegMisc(0, 3, vd, vn); }
void CodeGenerator::suqadd(const HReg &vd, const HReg &vn) { AdvSimdSc2RegMisc(0, 3, vd, vn); }
void CodeGenerator::suqadd(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMisc(0, 3, vd, vn); }
void CodeGenerator::suqadd(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMisc(0, 3, vd, vn); }
void CodeGenerator::sqabs(const BReg &vd, const BReg &vn) { AdvSimdSc2RegMisc(0, 7, vd, vn); }
void CodeGenerator::sqabs(const HReg &vd, const HReg &vn) { AdvSimdSc2RegMisc(0, 7, vd, vn); }
void CodeGenerator::sqabs(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMisc(0, 7, vd, vn); }
void CodeGenerator::sqabs(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMisc(0, 7, vd, vn); }
void CodeGenerator::cmgt(const DReg &vd, const DReg &vn, const uint32_t zero) { AdvSimdSc2RegMisc(0, 8, vd, vn, zero); }
void CodeGenerator::cmeq(const DReg &vd, const DReg &vn, const uint32_t zero) { AdvSimdSc2RegMisc(0, 9, vd, vn, zero); }
void CodeGenerator::cmlt(const DReg &vd, const DReg &vn, const uint32_t zero) { AdvSimdSc2RegMisc(0, 10, vd, vn, zero); }
void CodeGenerator::abs(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMisc(0, 11, vd, vn); }
void CodeGenerator::sqxtn(const BReg &vd, const HReg &vn) { AdvSimdSc2RegMisc(0, 20, vd, vn); }
void CodeGenerator::sqxtn(const HReg &vd, const SReg &vn) { AdvSimdSc2RegMisc(0, 20, vd, vn); }
void CodeGenerator::sqxtn(const SReg &vd, const DReg &vn) { AdvSimdSc2RegMisc(0, 20, vd, vn); }
void CodeGenerator::usqadd(const BReg &vd, const BReg &vn) { AdvSimdSc2RegMisc(1, 3, vd, vn); }
void CodeGenerator::usqadd(const HReg &vd, const HReg &vn) { AdvSimdSc2RegMisc(1, 3, vd, vn); }
void CodeGenerator::usqadd(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMisc(1, 3, vd, vn); }
void CodeGenerator::usqadd(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMisc(1, 3, vd, vn); }
void CodeGenerator::sqneg(const BReg &vd, const BReg &vn) { AdvSimdSc2RegMisc(1, 7, vd, vn); }
void CodeGenerator::sqneg(const HReg &vd, const HReg &vn) { AdvSimdSc2RegMisc(1, 7, vd, vn); }
void CodeGenerator::sqneg(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMisc(1, 7, vd, vn); }
void CodeGenerator::sqneg(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMisc(1, 7, vd, vn); }
void CodeGenerator::cmge(const DReg &vd, const DReg &vn, const uint32_t zero) { AdvSimdSc2RegMisc(1, 8, vd, vn, zero); }
void CodeGenerator::cmle(const DReg &vd, const DReg &vn, const uint32_t zero) { AdvSimdSc2RegMisc(1, 9, vd, vn, zero); }
void CodeGenerator::neg(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMisc(1, 11, vd, vn); }
void CodeGenerator::sqxtun(const BReg &vd, const HReg &vn) { AdvSimdSc2RegMisc(1, 18, vd, vn); }
void CodeGenerator::sqxtun(const HReg &vd, const SReg &vn) { AdvSimdSc2RegMisc(1, 18, vd, vn); }
void CodeGenerator::sqxtun(const SReg &vd, const DReg &vn) { AdvSimdSc2RegMisc(1, 18, vd, vn); }
void CodeGenerator::uqxtn(const BReg &vd, const HReg &vn) { AdvSimdSc2RegMisc(1, 20, vd, vn); }
void CodeGenerator::uqxtn(const HReg &vd, const SReg &vn) { AdvSimdSc2RegMisc(1, 20, vd, vn); }
void CodeGenerator::uqxtn(const SReg &vd, const DReg &vn) { AdvSimdSc2RegMisc(1, 20, vd, vn); }
void CodeGenerator::fcvtns(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz0x(0, 26, vd, vn); }
void CodeGenerator::fcvtns(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz0x(0, 26, vd, vn); }
void CodeGenerator::fcvtms(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz0x(0, 27, vd, vn); }
void CodeGenerator::fcvtms(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz0x(0, 27, vd, vn); }
void CodeGenerator::fcvtas(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz0x(0, 28, vd, vn); }
void CodeGenerator::fcvtas(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz0x(0, 28, vd, vn); }
void CodeGenerator::scvtf(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz0x(0, 29, vd, vn); }
void CodeGenerator::scvtf(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz0x(0, 29, vd, vn); }
void CodeGenerator::fcvtxn(const SReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz0x(1, 22, vd, vn); }
void CodeGenerator::fcvtnu(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz0x(1, 26, vd, vn); }
void CodeGenerator::fcvtnu(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz0x(1, 26, vd, vn); }
void CodeGenerator::fcvtmu(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz0x(1, 27, vd, vn); }
void CodeGenerator::fcvtmu(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz0x(1, 27, vd, vn); }
void CodeGenerator::fcvtau(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz0x(1, 28, vd, vn); }
void CodeGenerator::fcvtau(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz0x(1, 28, vd, vn); }
void CodeGenerator::ucvtf(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz0x(1, 29, vd, vn); }
void CodeGenerator::ucvtf(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz0x(1, 29, vd, vn); }
void CodeGenerator::fcmgt(const SReg &vd, const SReg &vn, const double zero) { AdvSimdSc2RegMiscSz1x(0, 12, vd, vn, zero); }
void CodeGenerator::fcmgt(const DReg &vd, const DReg &vn, const double zero) { AdvSimdSc2RegMiscSz1x(0, 12, vd, vn, zero); }
void CodeGenerator::fcmeq(const SReg &vd, const SReg &vn, const double zero) { AdvSimdSc2RegMiscSz1x(0, 13, vd, vn, zero); }
void CodeGenerator::fcmeq(const DReg &vd, const DReg &vn, const double zero) { AdvSimdSc2RegMiscSz1x(0, 13, vd, vn, zero); }
void CodeGenerator::fcmlt(const SReg &vd, const SReg &vn, const double zero) { AdvSimdSc2RegMiscSz1x(0, 14, vd, vn, zero); }
void CodeGenerator::fcmlt(const DReg &vd, const DReg &vn, const double zero) { AdvSimdSc2RegMiscSz1x(0, 14, vd, vn, zero); }
void CodeGenerator::fcvtps(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz1x(0, 26, vd, vn); }
void CodeGenerator::fcvtps(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz1x(0, 26, vd, vn); }
void CodeGenerator::fcvtzs(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz1x(0, 27, vd, vn); }
void CodeGenerator::fcvtzs(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz1x(0, 27, vd, vn); }
void CodeGenerator::frecpe(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz1x(0, 29, vd, vn); }
void CodeGenerator::frecpe(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz1x(0, 29, vd, vn); }
void CodeGenerator::frecpx(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz1x(0, 31, vd, vn); }
void CodeGenerator::frecpx(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz1x(0, 31, vd, vn); }
void CodeGenerator::fcmge(const SReg &vd, const SReg &vn, const double zero) { AdvSimdSc2RegMiscSz1x(1, 12, vd, vn, zero); }
void CodeGenerator::fcmge(const DReg &vd, const DReg &vn, const double zero) { AdvSimdSc2RegMiscSz1x(1, 12, vd, vn, zero); }
void CodeGenerator::fcmle(const SReg &vd, const SReg &vn, const double zero) { AdvSimdSc2RegMiscSz1x(1, 13, vd, vn, zero); }
void CodeGenerator::fcmle(const DReg &vd, const DReg &vn, const double zero) { AdvSimdSc2RegMiscSz1x(1, 13, vd, vn, zero); }
void CodeGenerator::fcvtpu(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz1x(1, 26, vd, vn); }
void CodeGenerator::fcvtpu(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz1x(1, 26, vd, vn); }
void CodeGenerator::fcvtzu(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz1x(1, 27, vd, vn); }
void CodeGenerator::fcvtzu(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz1x(1, 27, vd, vn); }
void CodeGenerator::frsqrte(const SReg &vd, const SReg &vn) { AdvSimdSc2RegMiscSz1x(1, 29, vd, vn); }
void CodeGenerator::frsqrte(const DReg &vd, const DReg &vn) { AdvSimdSc2RegMiscSz1x(1, 29, vd, vn); }
void CodeGenerator::addp(const DReg &vd, const VReg2D &vn) { AdvSimdScPairwise(0, 3, 27, vd, vn); }
void CodeGenerator::fmaxnmp(const HReg &vd, const VReg2H &vn) { AdvSimdScPairwise(0, 0, 12, vd, vn); }
void CodeGenerator::faddp(const HReg &vd, const VReg2H &vn) { AdvSimdScPairwise(0, 0, 13, vd, vn); }
void CodeGenerator::fmaxp(const HReg &vd, const VReg2H &vn) { AdvSimdScPairwise(0, 0, 15, vd, vn); }
void CodeGenerator::fminnmp(const HReg &vd, const VReg2H &vn) { AdvSimdScPairwise(0, 2, 12, vd, vn); }
void CodeGenerator::fminp(const HReg &vd, const VReg2H &vn) { AdvSimdScPairwise(0, 2, 15, vd, vn); }
void CodeGenerator::fmaxnmp(const SReg &vd, const VReg2S &vn) { AdvSimdScPairwise(1, (vd.getBit() == 32) ? 0 : 1, 12, vd, vn); }
void CodeGenerator::fmaxnmp(const DReg &vd, const VReg2D &vn) { AdvSimdScPairwise(1, (vd.getBit() == 32) ? 0 : 1, 12, vd, vn); }
void CodeGenerator::faddp(const SReg &vd, const VReg2S &vn) { AdvSimdScPairwise(1, (vd.getBit() == 32) ? 0 : 1, 13, vd, vn); }
void CodeGenerator::faddp(const DReg &vd, const VReg2D &vn) { AdvSimdScPairwise(1, (vd.getBit() == 32) ? 0 : 1, 13, vd, vn); }
void CodeGenerator::fmaxp(const SReg &vd, const VReg2S &vn) { AdvSimdScPairwise(1, (vd.getBit() == 32) ? 0 : 1, 15, vd, vn); }
void CodeGenerator::fmaxp(const DReg &vd, const VReg2D &vn) { AdvSimdScPairwise(1, (vd.getBit() == 32) ? 0 : 1, 15, vd, vn); }
void CodeGenerator::fminnmp(const SReg &vd, const VReg2S &vn) { AdvSimdScPairwise(1, (vd.getBit() == 32) ? 2 : 3, 12, vd, vn); }
void CodeGenerator::fminnmp(const DReg &vd, const VReg2D &vn) { AdvSimdScPairwise(1, (vd.getBit() == 32) ? 2 : 3, 12, vd, vn); }
void CodeGenerator::fminp(const SReg &vd, const VReg2S &vn) { AdvSimdScPairwise(1, (vd.getBit() == 32) ? 2 : 3, 15, vd, vn); }
void CodeGenerator::fminp(const DReg &vd, const VReg2D &vn) { AdvSimdScPairwise(1, (vd.getBit() == 32) ? 2 : 3, 15, vd, vn); }
void CodeGenerator::sqdmlal(const SReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3Diff(0, 9, vd, vn, vm); }
void CodeGenerator::sqdmlal(const DReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3Diff(0, 9, vd, vn, vm); }
void CodeGenerator::sqdmlsl(const SReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3Diff(0, 11, vd, vn, vm); }
void CodeGenerator::sqdmlsl(const DReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3Diff(0, 11, vd, vn, vm); }
void CodeGenerator::sqdmull(const SReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3Diff(0, 13, vd, vn, vm); }
void CodeGenerator::sqdmull(const DReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3Diff(0, 13, vd, vn, vm); }
void CodeGenerator::sqadd(const BReg &vd, const BReg &vn, const BReg &vm) { AdvSimdSc3Same(0, 1, vd, vn, vm); }
void CodeGenerator::sqadd(const HReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3Same(0, 1, vd, vn, vm); }
void CodeGenerator::sqadd(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3Same(0, 1, vd, vn, vm); }
void CodeGenerator::sqadd(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(0, 1, vd, vn, vm); }
void CodeGenerator::sqsub(const BReg &vd, const BReg &vn, const BReg &vm) { AdvSimdSc3Same(0, 5, vd, vn, vm); }
void CodeGenerator::sqsub(const HReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3Same(0, 5, vd, vn, vm); }
void CodeGenerator::sqsub(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3Same(0, 5, vd, vn, vm); }
void CodeGenerator::sqsub(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(0, 5, vd, vn, vm); }
void CodeGenerator::cmgt(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(0, 6, vd, vn, vm); }
void CodeGenerator::cmge(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(0, 7, vd, vn, vm); }
void CodeGenerator::sshl(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(0, 8, vd, vn, vm); }
void CodeGenerator::sqshl(const BReg &vd, const BReg &vn, const BReg &vm) { AdvSimdSc3Same(0, 9, vd, vn, vm); }
void CodeGenerator::sqshl(const HReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3Same(0, 9, vd, vn, vm); }
void CodeGenerator::sqshl(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3Same(0, 9, vd, vn, vm); }
void CodeGenerator::sqshl(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(0, 9, vd, vn, vm); }
void CodeGenerator::srshl(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(0, 10, vd, vn, vm); }
void CodeGenerator::sqrshl(const BReg &vd, const BReg &vn, const BReg &vm) { AdvSimdSc3Same(0, 11, vd, vn, vm); }
void CodeGenerator::sqrshl(const HReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3Same(0, 11, vd, vn, vm); }
void CodeGenerator::sqrshl(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3Same(0, 11, vd, vn, vm); }
void CodeGenerator::sqrshl(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(0, 11, vd, vn, vm); }
void CodeGenerator::add(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(0, 16, vd, vn, vm); }
void CodeGenerator::cmtst(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(0, 17, vd, vn, vm); }
void CodeGenerator::sqdmulh(const HReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3Same(0, 22, vd, vn, vm); }
void CodeGenerator::sqdmulh(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3Same(0, 22, vd, vn, vm); }
void CodeGenerator::uqadd(const BReg &vd, const BReg &vn, const BReg &vm) { AdvSimdSc3Same(1, 1, vd, vn, vm); }
void CodeGenerator::uqadd(const HReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3Same(1, 1, vd, vn, vm); }
void CodeGenerator::uqadd(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3Same(1, 1, vd, vn, vm); }
void CodeGenerator::uqadd(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(1, 1, vd, vn, vm); }
void CodeGenerator::uqsub(const BReg &vd, const BReg &vn, const BReg &vm) { AdvSimdSc3Same(1, 5, vd, vn, vm); }
void CodeGenerator::uqsub(const HReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3Same(1, 5, vd, vn, vm); }
void CodeGenerator::uqsub(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3Same(1, 5, vd, vn, vm); }
void CodeGenerator::uqsub(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(1, 5, vd, vn, vm); }
void CodeGenerator::cmhi(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(1, 6, vd, vn, vm); }
void CodeGenerator::cmhs(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(1, 7, vd, vn, vm); }
void CodeGenerator::ushl(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(1, 8, vd, vn, vm); }
void CodeGenerator::uqshl(const BReg &vd, const BReg &vn, const BReg &vm) { AdvSimdSc3Same(1, 9, vd, vn, vm); }
void CodeGenerator::uqshl(const HReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3Same(1, 9, vd, vn, vm); }
void CodeGenerator::uqshl(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3Same(1, 9, vd, vn, vm); }
void CodeGenerator::uqshl(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(1, 9, vd, vn, vm); }
void CodeGenerator::urshl(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(1, 10, vd, vn, vm); }
void CodeGenerator::uqrshl(const BReg &vd, const BReg &vn, const BReg &vm) { AdvSimdSc3Same(1, 11, vd, vn, vm); }
void CodeGenerator::uqrshl(const HReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3Same(1, 11, vd, vn, vm); }
void CodeGenerator::uqrshl(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3Same(1, 11, vd, vn, vm); }
void CodeGenerator::uqrshl(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(1, 11, vd, vn, vm); }
void CodeGenerator::sub(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(1, 16, vd, vn, vm); }
void CodeGenerator::cmeq(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3Same(1, 17, vd, vn, vm); }
void CodeGenerator::sqrdmulh(const HReg &vd, const HReg &vn, const HReg &vm) { AdvSimdSc3Same(1, 22, vd, vn, vm); }
void CodeGenerator::sqrdmulh(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3Same(1, 22, vd, vn, vm); }
void CodeGenerator::fmulx(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3SameSz0x(0, 27, vd, vn, vm); }
void CodeGenerator::fmulx(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3SameSz0x(0, 27, vd, vn, vm); }
void CodeGenerator::fcmeq(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3SameSz0x(0, 28, vd, vn, vm); }
void CodeGenerator::fcmeq(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3SameSz0x(0, 28, vd, vn, vm); }
void CodeGenerator::frecps(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3SameSz0x(0, 31, vd, vn, vm); }
void CodeGenerator::frecps(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3SameSz0x(0, 31, vd, vn, vm); }
void CodeGenerator::fcmge(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3SameSz0x(1, 28, vd, vn, vm); }
void CodeGenerator::fcmge(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3SameSz0x(1, 28, vd, vn, vm); }
void CodeGenerator::facge(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3SameSz0x(1, 29, vd, vn, vm); }
void CodeGenerator::facge(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3SameSz0x(1, 29, vd, vn, vm); }
void CodeGenerator::frsqrts(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3SameSz1x(0, 31, vd, vn, vm); }
void CodeGenerator::frsqrts(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3SameSz1x(0, 31, vd, vn, vm); }
void CodeGenerator::fabd(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3SameSz1x(1, 26, vd, vn, vm); }
void CodeGenerator::fabd(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3SameSz1x(1, 26, vd, vn, vm); }
void CodeGenerator::fcmgt(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3SameSz1x(1, 28, vd, vn, vm); }
void CodeGenerator::fcmgt(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3SameSz1x(1, 28, vd, vn, vm); }
void CodeGenerator::facgt(const SReg &vd, const SReg &vn, const SReg &vm) { AdvSimdSc3SameSz1x(1, 29, vd, vn, vm); }
void CodeGenerator::facgt(const DReg &vd, const DReg &vn, const DReg &vm) { AdvSimdSc3SameSz1x(1, 29, vd, vn, vm); }
void CodeGenerator::sshr(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 0, vd, vn, imm); }
void CodeGenerator::ssra(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 2, vd, vn, imm); }
void CodeGenerator::srshr(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 4, vd, vn, imm); }
void CodeGenerator::srsra(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 6, vd, vn, imm); }
void CodeGenerator::shl(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 10, vd, vn, imm); }
void CodeGenerator::sqshl(const BReg &vd, const BReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 14, vd, vn, imm); }
void CodeGenerator::sqshl(const HReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 14, vd, vn, imm); }
void CodeGenerator::sqshl(const SReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 14, vd, vn, imm); }
void CodeGenerator::sqshl(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 14, vd, vn, imm); }
void CodeGenerator::sqshl(const BReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 14, vd, vn, imm); }
void CodeGenerator::sqshl(const HReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 14, vd, vn, imm); }
void CodeGenerator::sqshl(const SReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 14, vd, vn, imm); }
void CodeGenerator::sqshrn(const BReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 18, vd, vn, imm); }
void CodeGenerator::sqshrn(const HReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 18, vd, vn, imm); }
void CodeGenerator::sqshrn(const SReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 18, vd, vn, imm); }
void CodeGenerator::sqrshrn(const BReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 19, vd, vn, imm); }
void CodeGenerator::sqrshrn(const HReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 19, vd, vn, imm); }
void CodeGenerator::sqrshrn(const SReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 19, vd, vn, imm); }
void CodeGenerator::scvtf(const HReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 28, vd, vn, imm); }
void CodeGenerator::scvtf(const SReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 28, vd, vn, imm); }
void CodeGenerator::scvtf(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 28, vd, vn, imm); }
void CodeGenerator::fcvtzs(const HReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 31, vd, vn, imm); }
void CodeGenerator::fcvtzs(const SReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 31, vd, vn, imm); }
void CodeGenerator::fcvtzs(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(0, 31, vd, vn, imm); }
void CodeGenerator::ushr(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 0, vd, vn, imm); }
void CodeGenerator::usra(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 2, vd, vn, imm); }
void CodeGenerator::urshr(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 4, vd, vn, imm); }
void CodeGenerator::ursra(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 6, vd, vn, imm); }
void CodeGenerator::sri(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 8, vd, vn, imm); }
void CodeGenerator::sli(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 10, vd, vn, imm); }
void CodeGenerator::sqshlu(const BReg &vd, const BReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 12, vd, vn, imm); }
void CodeGenerator::sqshlu(const HReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 12, vd, vn, imm); }
void CodeGenerator::sqshlu(const SReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 12, vd, vn, imm); }
void CodeGenerator::sqshlu(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 12, vd, vn, imm); }
void CodeGenerator::sqshlu(const BReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 12, vd, vn, imm); }
void CodeGenerator::sqshlu(const HReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 12, vd, vn, imm); }
void CodeGenerator::sqshlu(const SReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 12, vd, vn, imm); }
void CodeGenerator::uqshl(const BReg &vd, const BReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 14, vd, vn, imm); }
void CodeGenerator::uqshl(const HReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 14, vd, vn, imm); }
void CodeGenerator::uqshl(const SReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 14, vd, vn, imm); }
void CodeGenerator::uqshl(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 14, vd, vn, imm); }
void CodeGenerator::uqshl(const BReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 14, vd, vn, imm); }
void CodeGenerator::uqshl(const HReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 14, vd, vn, imm); }
void CodeGenerator::uqshl(const SReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 14, vd, vn, imm); }
void CodeGenerator::sqshrun(const BReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 16, vd, vn, imm); }
void CodeGenerator::sqshrun(const HReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 16, vd, vn, imm); }
void CodeGenerator::sqshrun(const SReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 16, vd, vn, imm); }
void CodeGenerator::sqrshrun(const BReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 17, vd, vn, imm); }
void CodeGenerator::sqrshrun(const HReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 17, vd, vn, imm); }
void CodeGenerator::sqrshrun(const SReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 17, vd, vn, imm); }
void CodeGenerator::uqshrn(const BReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 18, vd, vn, imm); }
void CodeGenerator::uqshrn(const HReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 18, vd, vn, imm); }
void CodeGenerator::uqshrn(const SReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 18, vd, vn, imm); }
void CodeGenerator::uqrshrn(const BReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 19, vd, vn, imm); }
void CodeGenerator::uqrshrn(const HReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 19, vd, vn, imm); }
void CodeGenerator::uqrshrn(const SReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 19, vd, vn, imm); }
void CodeGenerator::ucvtf(const HReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 28, vd, vn, imm); }
void CodeGenerator::ucvtf(const SReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 28, vd, vn, imm); }
void CodeGenerator::ucvtf(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 28, vd, vn, imm); }
void CodeGenerator::fcvtzu(const HReg &vd, const HReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 31, vd, vn, imm); }
void CodeGenerator::fcvtzu(const SReg &vd, const SReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 31, vd, vn, imm); }
void CodeGenerator::fcvtzu(const DReg &vd, const DReg &vn, const uint32_t imm) { AdvSimdScShImm(1, 31, vd, vn, imm); }
void CodeGenerator::sqdmlal(const SReg &vd, const HReg &vn, const VRegHElem &vm) { AdvSimdScXIndElem(0, 3, vd, vn, vm); }
void CodeGenerator::sqdmlal(const DReg &vd, const SReg &vn, const VRegSElem &vm) { AdvSimdScXIndElem(0, 3, vd, vn, vm); }
void CodeGenerator::sqdmlsl(const SReg &vd, const HReg &vn, const VRegHElem &vm) { AdvSimdScXIndElem(0, 7, vd, vn, vm); }
void CodeGenerator::sqdmlsl(const DReg &vd, const SReg &vn, const VRegSElem &vm) { AdvSimdScXIndElem(0, 7, vd, vn, vm); }
void CodeGenerator::sqdmull(const SReg &vd, const HReg &vn, const VRegHElem &vm) { AdvSimdScXIndElem(0, 11, vd, vn, vm); }
void CodeGenerator::sqdmull(const DReg &vd, const SReg &vn, const VRegSElem &vm) { AdvSimdScXIndElem(0, 11, vd, vn, vm); }
void CodeGenerator::sqdmulh(const HReg &vd, const HReg &vn, const VRegHElem &vm) { AdvSimdScXIndElem(0, 12, vd, vn, vm); }
void CodeGenerator::sqdmulh(const SReg &vd, const SReg &vn, const VRegSElem &vm) { AdvSimdScXIndElem(0, 12, vd, vn, vm); }
void CodeGenerator::sqrdmulh(const HReg &vd, const HReg &vn, const VRegHElem &vm) { AdvSimdScXIndElem(0, 13, vd, vn, vm); }
void CodeGenerator::sqrdmulh(const SReg &vd, const SReg &vn, const VRegSElem &vm) { AdvSimdScXIndElem(0, 13, vd, vn, vm); }
void CodeGenerator::sqrdmlah(const HReg &vd, const HReg &vn, const VRegHElem &vm) { AdvSimdScXIndElem(1, 13, vd, vn, vm); }
void CodeGenerator::sqrdmlah(const SReg &vd, const SReg &vn, const VRegSElem &vm) { AdvSimdScXIndElem(1, 13, vd, vn, vm); }
void CodeGenerator::sqrdmlsh(const HReg &vd, const HReg &vn, const VRegHElem &vm) { AdvSimdScXIndElem(1, 15, vd, vn, vm); }
void CodeGenerator::sqrdmlsh(const SReg &vd, const SReg &vn, const VRegSElem &vm) { AdvSimdScXIndElem(1, 15, vd, vn, vm); }
void CodeGenerator::fmla(const HReg &vd, const HReg &vn, const VRegHElem &vm) { AdvSimdScXIndElemSz(0, 0, 1, vd, vn, vm); }
void CodeGenerator::fmls(const HReg &vd, const HReg &vn, const VRegHElem &vm) { AdvSimdScXIndElemSz(0, 0, 5, vd, vn, vm); }
void CodeGenerator::fmul(const HReg &vd, const HReg &vn, const VRegHElem &vm) { AdvSimdScXIndElemSz(0, 0, 9, vd, vn, vm); }
void CodeGenerator::fmla(const SReg &vd, const SReg &vn, const VRegSElem &vm) { AdvSimdScXIndElemSz(0, 2, 1, vd, vn, vm); }
void CodeGenerator::fmls(const SReg &vd, const SReg &vn, const VRegSElem &vm) { AdvSimdScXIndElemSz(0, 2, 5, vd, vn, vm); }
void CodeGenerator::fmul(const SReg &vd, const SReg &vn, const VRegSElem &vm) { AdvSimdScXIndElemSz(0, 2, 9, vd, vn, vm); }
void CodeGenerator::fmla(const DReg &vd, const DReg &vn, const VRegDElem &vm) { AdvSimdScXIndElemSz(0, 3, 1, vd, vn, vm); }
void CodeGenerator::fmls(const DReg &vd, const DReg &vn, const VRegDElem &vm) { AdvSimdScXIndElemSz(0, 3, 5, vd, vn, vm); }
void CodeGenerator::fmul(const DReg &vd, const DReg &vn, const VRegDElem &vm) { AdvSimdScXIndElemSz(0, 3, 9, vd, vn, vm); }
void CodeGenerator::fmulx(const HReg &vd, const HReg &vn, const VRegHElem &vm) { AdvSimdScXIndElemSz(1, 0, 9, vd, vn, vm); }
void CodeGenerator::fmulx(const SReg &vd, const SReg &vn, const VRegSElem &vm) { AdvSimdScXIndElemSz(1, 2, 9, vd, vn, vm); }
void CodeGenerator::fmulx(const DReg &vd, const DReg &vn, const VRegDElem &vm) { AdvSimdScXIndElemSz(1, 3, 9, vd, vn, vm); }
void CodeGenerator::tbl(const VReg8B &vd, const VReg16B &vn, const uint32_t len, const VReg8B &vm) { AdvSimdTblLkup(0, len, 0, vd, vn, vm); }
void CodeGenerator::tbl(const VReg16B &vd, const VReg16B &vn, const uint32_t len, const VReg16B &vm) { AdvSimdTblLkup(0, len, 0, vd, vn, vm); }
void CodeGenerator::tbx(const VReg8B &vd, const VReg16B &vn, const uint32_t len, const VReg8B &vm) { AdvSimdTblLkup(0, len, 1, vd, vn, vm); }
void CodeGenerator::tbx(const VReg16B &vd, const VReg16B &vn, const uint32_t len, const VReg16B &vm) { AdvSimdTblLkup(0, len, 1, vd, vn, vm); }
void CodeGenerator::tbl(const VReg8B &vd, const VReg16BList &vn, const VReg8B &vm) { AdvSimdTblLkup(0, 0, vd, vn, vm); }
void CodeGenerator::tbl(const VReg16B &vd, const VReg16BList &vn, const VReg16B &vm) { AdvSimdTblLkup(0, 0, vd, vn, vm); }
void CodeGenerator::tbx(const VReg8B &vd, const VReg16BList &vn, const VReg8B &vm) { AdvSimdTblLkup(0, 1, vd, vn, vm); }
void CodeGenerator::tbx(const VReg16B &vd, const VReg16BList &vn, const VReg16B &vm) { AdvSimdTblLkup(0, 1, vd, vn, vm); }
void CodeGenerator::uzp1(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimdPermute(1, vd, vn, vm); }
void CodeGenerator::uzp1(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimdPermute(1, vd, vn, vm); }
void CodeGenerator::uzp1(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimdPermute(1, vd, vn, vm); }
void CodeGenerator::uzp1(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimdPermute(1, vd, vn, vm); }
void CodeGenerator::uzp1(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimdPermute(1, vd, vn, vm); }
void CodeGenerator::uzp1(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimdPermute(1, vd, vn, vm); }
void CodeGenerator::uzp1(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimdPermute(1, vd, vn, vm); }
void CodeGenerator::trn1(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimdPermute(2, vd, vn, vm); }
void CodeGenerator::trn1(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimdPermute(2, vd, vn, vm); }
void CodeGenerator::trn1(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimdPermute(2, vd, vn, vm); }
void CodeGenerator::trn1(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimdPermute(2, vd, vn, vm); }
void CodeGenerator::trn1(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimdPermute(2, vd, vn, vm); }
void CodeGenerator::trn1(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimdPermute(2, vd, vn, vm); }
void CodeGenerator::trn1(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimdPermute(2, vd, vn, vm); }
void CodeGenerator::zip1(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimdPermute(3, vd, vn, vm); }
void CodeGenerator::zip1(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimdPermute(3, vd, vn, vm); }
void CodeGenerator::zip1(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimdPermute(3, vd, vn, vm); }
void CodeGenerator::zip1(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimdPermute(3, vd, vn, vm); }
void CodeGenerator::zip1(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimdPermute(3, vd, vn, vm); }
void CodeGenerator::zip1(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimdPermute(3, vd, vn, vm); }
void CodeGenerator::zip1(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimdPermute(3, vd, vn, vm); }
void CodeGenerator::uzp2(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimdPermute(5, vd, vn, vm); }
void CodeGenerator::uzp2(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimdPermute(5, vd, vn, vm); }
void CodeGenerator::uzp2(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimdPermute(5, vd, vn, vm); }
void CodeGenerator::uzp2(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimdPermute(5, vd, vn, vm); }
void CodeGenerator::uzp2(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimdPermute(5, vd, vn, vm); }
void CodeGenerator::uzp2(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimdPermute(5, vd, vn, vm); }
void CodeGenerator::uzp2(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimdPermute(5, vd, vn, vm); }
void CodeGenerator::trn2(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimdPermute(6, vd, vn, vm); }
void CodeGenerator::trn2(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimdPermute(6, vd, vn, vm); }
void CodeGenerator::trn2(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimdPermute(6, vd, vn, vm); }
void CodeGenerator::trn2(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimdPermute(6, vd, vn, vm); }
void CodeGenerator::trn2(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimdPermute(6, vd, vn, vm); }
void CodeGenerator::trn2(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimdPermute(6, vd, vn, vm); }
void CodeGenerator::trn2(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimdPermute(6, vd, vn, vm); }
void CodeGenerator::zip2(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimdPermute(7, vd, vn, vm); }
void CodeGenerator::zip2(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimdPermute(7, vd, vn, vm); }
void CodeGenerator::zip2(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimdPermute(7, vd, vn, vm); }
void CodeGenerator::zip2(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimdPermute(7, vd, vn, vm); }
void CodeGenerator::zip2(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimdPermute(7, vd, vn, vm); }
void CodeGenerator::zip2(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimdPermute(7, vd, vn, vm); }
void CodeGenerator::zip2(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimdPermute(7, vd, vn, vm); }
void CodeGenerator::ext(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm, const uint32_t index) { AdvSimdExtract(0, vd, vn, vm, index); }
void CodeGenerator::ext(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm, const uint32_t index) { AdvSimdExtract(0, vd, vn, vm, index); }
void CodeGenerator::dup(const VReg8B &vd, const VRegBElem &vn) { AdvSimdCopyDupElem(0, 0, vd, vn); }
void CodeGenerator::dup(const VReg16B &vd, const VRegBElem &vn) { AdvSimdCopyDupElem(0, 0, vd, vn); }
void CodeGenerator::dup(const VReg4H &vd, const VRegHElem &vn) { AdvSimdCopyDupElem(0, 0, vd, vn); }
void CodeGenerator::dup(const VReg8H &vd, const VRegHElem &vn) { AdvSimdCopyDupElem(0, 0, vd, vn); }
void CodeGenerator::dup(const VReg2S &vd, const VRegSElem &vn) { AdvSimdCopyDupElem(0, 0, vd, vn); }
void CodeGenerator::dup(const VReg4S &vd, const VRegSElem &vn) { AdvSimdCopyDupElem(0, 0, vd, vn); }
void CodeGenerator::dup(const VReg2D &vd, const VRegDElem &vn) { AdvSimdCopyDupElem(0, 0, vd, vn); }
void CodeGenerator::dup(const VReg8B &vd, const WReg &rn) { AdvSimdCopyDupGen(0, 0, vd, rn); }
void CodeGenerator::dup(const VReg16B &vd, const WReg &rn) { AdvSimdCopyDupGen(0, 0, vd, rn); }
void CodeGenerator::dup(const VReg4H &vd, const WReg &rn) { AdvSimdCopyDupGen(0, 0, vd, rn); }
void CodeGenerator::dup(const VReg8H &vd, const WReg &rn) { AdvSimdCopyDupGen(0, 0, vd, rn); }
void CodeGenerator::dup(const VReg2S &vd, const WReg &rn) { AdvSimdCopyDupGen(0, 0, vd, rn); }
void CodeGenerator::dup(const VReg4S &vd, const WReg &rn) { AdvSimdCopyDupGen(0, 0, vd, rn); }
void CodeGenerator::dup(const VReg2D &vd, const XReg &rn) { AdvSimdCopyDupGen(0, 0, vd, rn); }
void CodeGenerator::smov(const WReg &rd, const VRegBElem &vn) { AdvSimdCopyMov(0, 5, rd, vn); }
void CodeGenerator::smov(const WReg &rd, const VRegHElem &vn) { AdvSimdCopyMov(0, 5, rd, vn); }
void CodeGenerator::smov(const XReg &rd, const VRegBElem &vn) { AdvSimdCopyMov(0, 5, rd, vn); }
void CodeGenerator::smov(const XReg &rd, const VRegHElem &vn) { AdvSimdCopyMov(0, 5, rd, vn); }
void CodeGenerator::smov(const XReg &rd, const VRegSElem &vn) { AdvSimdCopyMov(0, 5, rd, vn); }
void CodeGenerator::umov(const WReg &rd, const VRegBElem &vn) { AdvSimdCopyMov(0, 7, rd, vn); }
void CodeGenerator::umov(const WReg &rd, const VRegHElem &vn) { AdvSimdCopyMov(0, 7, rd, vn); }
void CodeGenerator::umov(const WReg &rd, const VRegSElem &vn) { AdvSimdCopyMov(0, 7, rd, vn); }
void CodeGenerator::umov(const XReg &rd, const VRegDElem &vn) { AdvSimdCopyMov(0, 7, rd, vn); }
void CodeGenerator::mov(const WReg &rd, const VRegSElem &vn) { AdvSimdCopyMov(0, 7, rd, vn); }
void CodeGenerator::mov(const XReg &rd, const VRegDElem &vn) { AdvSimdCopyMov(0, 7, rd, vn); }
#ifdef XBYAK_AARCH64_FOR_DNNL
void CodeGenerator::ins_(const VRegBElem &vd, const WReg &rn) {
#else
void CodeGenerator::ins(const VRegBElem &vd, const WReg &rn) {
#endif
  AdvSimdCopyInsGen(0, 3, vd, rn);
}
#ifdef XBYAK_AARCH64_FOR_DNNL
void CodeGenerator::ins_(const VRegHElem &vd, const WReg &rn) {
#else
void CodeGenerator::ins(const VRegHElem &vd, const WReg &rn) {
#endif
  AdvSimdCopyInsGen(0, 3, vd, rn);
}
#ifdef XBYAK_AARCH64_FOR_DNNL
void CodeGenerator::ins_(const VRegSElem &vd, const WReg &rn) {
#else
void CodeGenerator::ins(const VRegSElem &vd, const WReg &rn) {
#endif
  AdvSimdCopyInsGen(0, 3, vd, rn);
}
#ifdef XBYAK_AARCH64_FOR_DNNL
void CodeGenerator::ins_(const VRegDElem &vd, const XReg &rn) {
#else
void CodeGenerator::ins(const VRegDElem &vd, const XReg &rn) {
#endif
  AdvSimdCopyInsGen(0, 3, vd, rn);
}
void CodeGenerator::mov(const VRegBElem &vd, const WReg &rn) { AdvSimdCopyInsGen(0, 3, vd, rn); }
void CodeGenerator::mov(const VRegHElem &vd, const WReg &rn) { AdvSimdCopyInsGen(0, 3, vd, rn); }
void CodeGenerator::mov(const VRegSElem &vd, const WReg &rn) { AdvSimdCopyInsGen(0, 3, vd, rn); }
void CodeGenerator::mov(const VRegDElem &vd, const XReg &rn) { AdvSimdCopyInsGen(0, 3, vd, rn); }
#ifdef XBYAK_AARCH64_FOR_DNNL
void CodeGenerator::ins_(const VRegBElem &vd, const VRegBElem &vn) {
#else
void CodeGenerator::ins(const VRegBElem &vd, const VRegBElem &vn) {
#endif
  AdvSimdCopyElemIns(1, vd, vn);
}
#ifdef XBYAK_AARCH64_FOR_DNNL
void CodeGenerator::ins_(const VRegHElem &vd, const VRegHElem &vn) {
#else
void CodeGenerator::ins(const VRegHElem &vd, const VRegHElem &vn) {
#endif
  AdvSimdCopyElemIns(1, vd, vn);
}
#ifdef XBYAK_AARCH64_FOR_DNNL
void CodeGenerator::ins_(const VRegSElem &vd, const VRegSElem &vn) {
#else
void CodeGenerator::ins(const VRegSElem &vd, const VRegSElem &vn) {
#endif
  AdvSimdCopyElemIns(1, vd, vn);
}
#ifdef XBYAK_AARCH64_FOR_DNNL
void CodeGenerator::ins_(const VRegDElem &vd, const VRegDElem &vn) {
#else
void CodeGenerator::ins(const VRegDElem &vd, const VRegDElem &vn) {
#endif
  AdvSimdCopyElemIns(1, vd, vn);
}
void CodeGenerator::mov(const VRegBElem &vd, const VRegBElem &vn) { AdvSimdCopyElemIns(1, vd, vn); }
void CodeGenerator::mov(const VRegHElem &vd, const VRegHElem &vn) { AdvSimdCopyElemIns(1, vd, vn); }
void CodeGenerator::mov(const VRegSElem &vd, const VRegSElem &vn) { AdvSimdCopyElemIns(1, vd, vn); }
void CodeGenerator::mov(const VRegDElem &vd, const VRegDElem &vn) { AdvSimdCopyElemIns(1, vd, vn); }
void CodeGenerator::fmaxnm(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(0, 0, 0, vd, vn, vm); }
void CodeGenerator::fmaxnm(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(0, 0, 0, vd, vn, vm); }
void CodeGenerator::fmla(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(0, 0, 1, vd, vn, vm); }
void CodeGenerator::fmla(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(0, 0, 1, vd, vn, vm); }
void CodeGenerator::fadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(0, 0, 2, vd, vn, vm); }
void CodeGenerator::fadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(0, 0, 2, vd, vn, vm); }
void CodeGenerator::fmulx(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(0, 0, 3, vd, vn, vm); }
void CodeGenerator::fmulx(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(0, 0, 3, vd, vn, vm); }
void CodeGenerator::fcmeq(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(0, 0, 4, vd, vn, vm); }
void CodeGenerator::fcmeq(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(0, 0, 4, vd, vn, vm); }
void CodeGenerator::fmax(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(0, 0, 6, vd, vn, vm); }
void CodeGenerator::fmax(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(0, 0, 6, vd, vn, vm); }
void CodeGenerator::frecps(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(0, 0, 7, vd, vn, vm); }
void CodeGenerator::frecps(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(0, 0, 7, vd, vn, vm); }
void CodeGenerator::fminnm(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(0, 1, 0, vd, vn, vm); }
void CodeGenerator::fminnm(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(0, 1, 0, vd, vn, vm); }
void CodeGenerator::fmls(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(0, 1, 1, vd, vn, vm); }
void CodeGenerator::fmls(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(0, 1, 1, vd, vn, vm); }
void CodeGenerator::fsub(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(0, 1, 2, vd, vn, vm); }
void CodeGenerator::fsub(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(0, 1, 2, vd, vn, vm); }
void CodeGenerator::fmin(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(0, 1, 6, vd, vn, vm); }
void CodeGenerator::fmin(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(0, 1, 6, vd, vn, vm); }
void CodeGenerator::frsqrts(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(0, 1, 7, vd, vn, vm); }
void CodeGenerator::frsqrts(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(0, 1, 7, vd, vn, vm); }
void CodeGenerator::fmaxnmp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(1, 0, 0, vd, vn, vm); }
void CodeGenerator::fmaxnmp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(1, 0, 0, vd, vn, vm); }
void CodeGenerator::faddp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(1, 0, 2, vd, vn, vm); }
void CodeGenerator::faddp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(1, 0, 2, vd, vn, vm); }
void CodeGenerator::fmul(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(1, 0, 3, vd, vn, vm); }
void CodeGenerator::fmul(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(1, 0, 3, vd, vn, vm); }
void CodeGenerator::fcmge(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(1, 0, 4, vd, vn, vm); }
void CodeGenerator::fcmge(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(1, 0, 4, vd, vn, vm); }
void CodeGenerator::facge(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(1, 0, 5, vd, vn, vm); }
void CodeGenerator::facge(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(1, 0, 5, vd, vn, vm); }
void CodeGenerator::fmaxp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(1, 0, 6, vd, vn, vm); }
void CodeGenerator::fmaxp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(1, 0, 6, vd, vn, vm); }
void CodeGenerator::fdiv(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(1, 0, 7, vd, vn, vm); }
void CodeGenerator::fdiv(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(1, 0, 7, vd, vn, vm); }
void CodeGenerator::fminnmp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(1, 1, 0, vd, vn, vm); }
void CodeGenerator::fminnmp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(1, 1, 0, vd, vn, vm); }
void CodeGenerator::fabd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(1, 1, 2, vd, vn, vm); }
void CodeGenerator::fabd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(1, 1, 2, vd, vn, vm); }
void CodeGenerator::fcmgt(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(1, 1, 4, vd, vn, vm); }
void CodeGenerator::fcmgt(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(1, 1, 4, vd, vn, vm); }
void CodeGenerator::facgt(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(1, 1, 5, vd, vn, vm); }
void CodeGenerator::facgt(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(1, 1, 5, vd, vn, vm); }
void CodeGenerator::fminp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameFp16(1, 1, 6, vd, vn, vm); }
void CodeGenerator::fminp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameFp16(1, 1, 6, vd, vn, vm); }
void CodeGenerator::frintn(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(0, 0, 24, vd, vn); }
void CodeGenerator::frintn(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(0, 0, 24, vd, vn); }
void CodeGenerator::frintm(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(0, 0, 25, vd, vn); }
void CodeGenerator::frintm(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(0, 0, 25, vd, vn); }
void CodeGenerator::fcvtns(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(0, 0, 26, vd, vn); }
void CodeGenerator::fcvtns(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(0, 0, 26, vd, vn); }
void CodeGenerator::fcvtms(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(0, 0, 27, vd, vn); }
void CodeGenerator::fcvtms(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(0, 0, 27, vd, vn); }
void CodeGenerator::fcvtas(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(0, 0, 28, vd, vn); }
void CodeGenerator::fcvtas(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(0, 0, 28, vd, vn); }
void CodeGenerator::scvtf(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(0, 0, 29, vd, vn); }
void CodeGenerator::scvtf(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(0, 0, 29, vd, vn); }
void CodeGenerator::fcmgt(const VReg4H &vd, const VReg4H &vn, const double zero) { AdvSimd2RegMiscFp16(0, 1, 12, vd, vn, zero); }
void CodeGenerator::fcmgt(const VReg8H &vd, const VReg8H &vn, const double zero) { AdvSimd2RegMiscFp16(0, 1, 12, vd, vn, zero); }
void CodeGenerator::fcmeq(const VReg4H &vd, const VReg4H &vn, const double zero) { AdvSimd2RegMiscFp16(0, 1, 13, vd, vn, zero); }
void CodeGenerator::fcmeq(const VReg8H &vd, const VReg8H &vn, const double zero) { AdvSimd2RegMiscFp16(0, 1, 13, vd, vn, zero); }
void CodeGenerator::fcmlt(const VReg4H &vd, const VReg4H &vn, const double zero) { AdvSimd2RegMiscFp16(0, 1, 14, vd, vn, zero); }
void CodeGenerator::fcmlt(const VReg8H &vd, const VReg8H &vn, const double zero) { AdvSimd2RegMiscFp16(0, 1, 14, vd, vn, zero); }
void CodeGenerator::fabs(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(0, 1, 15, vd, vn); }
void CodeGenerator::fabs(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(0, 1, 15, vd, vn); }
void CodeGenerator::frintp(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(0, 1, 24, vd, vn); }
void CodeGenerator::frintp(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(0, 1, 24, vd, vn); }
void CodeGenerator::frintz(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(0, 1, 25, vd, vn); }
void CodeGenerator::frintz(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(0, 1, 25, vd, vn); }
void CodeGenerator::fcvtps(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(0, 1, 26, vd, vn); }
void CodeGenerator::fcvtps(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(0, 1, 26, vd, vn); }
void CodeGenerator::fcvtzs(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(0, 1, 27, vd, vn); }
void CodeGenerator::fcvtzs(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(0, 1, 27, vd, vn); }
void CodeGenerator::frecpe(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(0, 1, 29, vd, vn); }
void CodeGenerator::frecpe(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(0, 1, 29, vd, vn); }
void CodeGenerator::frinta(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(1, 0, 24, vd, vn); }
void CodeGenerator::frinta(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(1, 0, 24, vd, vn); }
void CodeGenerator::frintx(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(1, 0, 25, vd, vn); }
void CodeGenerator::frintx(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(1, 0, 25, vd, vn); }
void CodeGenerator::fcvtnu(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(1, 0, 26, vd, vn); }
void CodeGenerator::fcvtnu(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(1, 0, 26, vd, vn); }
void CodeGenerator::fcvtmu(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(1, 0, 27, vd, vn); }
void CodeGenerator::fcvtmu(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(1, 0, 27, vd, vn); }
void CodeGenerator::fcvtau(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(1, 0, 28, vd, vn); }
void CodeGenerator::fcvtau(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(1, 0, 28, vd, vn); }
void CodeGenerator::ucvtf(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(1, 0, 29, vd, vn); }
void CodeGenerator::ucvtf(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(1, 0, 29, vd, vn); }
void CodeGenerator::fcmge(const VReg4H &vd, const VReg4H &vn, const double zero) { AdvSimd2RegMiscFp16(1, 1, 12, vd, vn, zero); }
void CodeGenerator::fcmge(const VReg8H &vd, const VReg8H &vn, const double zero) { AdvSimd2RegMiscFp16(1, 1, 12, vd, vn, zero); }
void CodeGenerator::fcmle(const VReg4H &vd, const VReg4H &vn, const double zero) { AdvSimd2RegMiscFp16(1, 1, 13, vd, vn, zero); }
void CodeGenerator::fcmle(const VReg8H &vd, const VReg8H &vn, const double zero) { AdvSimd2RegMiscFp16(1, 1, 13, vd, vn, zero); }
void CodeGenerator::fneg(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(1, 1, 15, vd, vn); }
void CodeGenerator::fneg(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(1, 1, 15, vd, vn); }
void CodeGenerator::frinti(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(1, 1, 25, vd, vn); }
void CodeGenerator::frinti(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(1, 1, 25, vd, vn); }
void CodeGenerator::fcvtpu(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(1, 1, 26, vd, vn); }
void CodeGenerator::fcvtpu(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(1, 1, 26, vd, vn); }
void CodeGenerator::fcvtzu(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(1, 1, 27, vd, vn); }
void CodeGenerator::fcvtzu(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(1, 1, 27, vd, vn); }
void CodeGenerator::frsqrte(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(1, 1, 29, vd, vn); }
void CodeGenerator::frsqrte(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(1, 1, 29, vd, vn); }
void CodeGenerator::fsqrt(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMiscFp16(1, 1, 31, vd, vn); }
void CodeGenerator::fsqrt(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMiscFp16(1, 1, 31, vd, vn); }
void CodeGenerator::sdot(const VReg2S &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3SameExtra(0, 2, vd, vn, vm); }
void CodeGenerator::sdot(const VReg4S &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3SameExtra(0, 2, vd, vn, vm); }
void CodeGenerator::sqrdmlah(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameExtra(1, 0, vd, vn, vm); }
void CodeGenerator::sqrdmlah(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameExtra(1, 0, vd, vn, vm); }
void CodeGenerator::sqrdmlah(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameExtra(1, 0, vd, vn, vm); }
void CodeGenerator::sqrdmlah(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameExtra(1, 0, vd, vn, vm); }
void CodeGenerator::sqrdmlsh(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameExtra(1, 1, vd, vn, vm); }
void CodeGenerator::sqrdmlsh(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3SameExtra(1, 1, vd, vn, vm); }
void CodeGenerator::sqrdmlsh(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameExtra(1, 1, vd, vn, vm); }
void CodeGenerator::sqrdmlsh(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameExtra(1, 1, vd, vn, vm); }
void CodeGenerator::udot(const VReg2S &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3SameExtra(1, 2, vd, vn, vm); }
void CodeGenerator::udot(const VReg4S &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3SameExtra(1, 2, vd, vn, vm); }
void CodeGenerator::fcmla(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm, const uint32_t rotate) { AdvSimd3SameExtraRotate(1, 2, vd, vn, vm, rotate); }
void CodeGenerator::fcmla(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm, const uint32_t rotate) { AdvSimd3SameExtraRotate(1, 2, vd, vn, vm, rotate); }
void CodeGenerator::fcmla(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm, const uint32_t rotate) { AdvSimd3SameExtraRotate(1, 2, vd, vn, vm, rotate); }
void CodeGenerator::fcmla(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm, const uint32_t rotate) { AdvSimd3SameExtraRotate(1, 2, vd, vn, vm, rotate); }
void CodeGenerator::fcmla(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm, const uint32_t rotate) { AdvSimd3SameExtraRotate(1, 2, vd, vn, vm, rotate); }
void CodeGenerator::fcadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm, const uint32_t rotate) { AdvSimd3SameExtraRotate(1, 3, vd, vn, vm, rotate); }
void CodeGenerator::fcadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm, const uint32_t rotate) { AdvSimd3SameExtraRotate(1, 3, vd, vn, vm, rotate); }
void CodeGenerator::fcadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm, const uint32_t rotate) { AdvSimd3SameExtraRotate(1, 3, vd, vn, vm, rotate); }
void CodeGenerator::fcadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm, const uint32_t rotate) { AdvSimd3SameExtraRotate(1, 3, vd, vn, vm, rotate); }
void CodeGenerator::fcadd(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm, const uint32_t rotate) { AdvSimd3SameExtraRotate(1, 3, vd, vn, vm, rotate); }
void CodeGenerator::rev64(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMisc(0, 0, vd, vn); }
void CodeGenerator::rev64(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMisc(0, 0, vd, vn); }
void CodeGenerator::rev64(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMisc(0, 0, vd, vn); }
void CodeGenerator::rev64(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMisc(0, 0, vd, vn); }
void CodeGenerator::rev64(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMisc(0, 0, vd, vn); }
void CodeGenerator::rev64(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMisc(0, 0, vd, vn); }
void CodeGenerator::rev16(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMisc(0, 1, vd, vn); }
void CodeGenerator::rev16(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMisc(0, 1, vd, vn); }
void CodeGenerator::saddlp(const VReg4H &vd, const VReg8B &vn) { AdvSimd2RegMisc(0, 2, vd, vn); }
void CodeGenerator::saddlp(const VReg8H &vd, const VReg16B &vn) { AdvSimd2RegMisc(0, 2, vd, vn); }
void CodeGenerator::saddlp(const VReg2S &vd, const VReg4H &vn) { AdvSimd2RegMisc(0, 2, vd, vn); }
void CodeGenerator::saddlp(const VReg4S &vd, const VReg8H &vn) { AdvSimd2RegMisc(0, 2, vd, vn); }
void CodeGenerator::saddlp(const VReg1D &vd, const VReg2S &vn) { AdvSimd2RegMisc(0, 2, vd, vn); }
void CodeGenerator::saddlp(const VReg2D &vd, const VReg4S &vn) { AdvSimd2RegMisc(0, 2, vd, vn); }
void CodeGenerator::suqadd(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMisc(0, 3, vd, vn); }
void CodeGenerator::suqadd(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMisc(0, 3, vd, vn); }
void CodeGenerator::suqadd(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMisc(0, 3, vd, vn); }
void CodeGenerator::suqadd(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMisc(0, 3, vd, vn); }
void CodeGenerator::suqadd(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMisc(0, 3, vd, vn); }
void CodeGenerator::suqadd(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMisc(0, 3, vd, vn); }
void CodeGenerator::suqadd(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMisc(0, 3, vd, vn); }
void CodeGenerator::cls(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMisc(0, 4, vd, vn); }
void CodeGenerator::cls(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMisc(0, 4, vd, vn); }
void CodeGenerator::cls(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMisc(0, 4, vd, vn); }
void CodeGenerator::cls(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMisc(0, 4, vd, vn); }
void CodeGenerator::cls(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMisc(0, 4, vd, vn); }
void CodeGenerator::cls(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMisc(0, 4, vd, vn); }
void CodeGenerator::cnt(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMisc(0, 5, vd, vn); }
void CodeGenerator::cnt(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMisc(0, 5, vd, vn); }
void CodeGenerator::sadalp(const VReg4H &vd, const VReg8B &vn) { AdvSimd2RegMisc(0, 6, vd, vn); }
void CodeGenerator::sadalp(const VReg8H &vd, const VReg16B &vn) { AdvSimd2RegMisc(0, 6, vd, vn); }
void CodeGenerator::sadalp(const VReg2S &vd, const VReg4H &vn) { AdvSimd2RegMisc(0, 6, vd, vn); }
void CodeGenerator::sadalp(const VReg4S &vd, const VReg8H &vn) { AdvSimd2RegMisc(0, 6, vd, vn); }
void CodeGenerator::sadalp(const VReg1D &vd, const VReg2S &vn) { AdvSimd2RegMisc(0, 6, vd, vn); }
void CodeGenerator::sadalp(const VReg2D &vd, const VReg4S &vn) { AdvSimd2RegMisc(0, 6, vd, vn); }
void CodeGenerator::sqabs(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMisc(0, 7, vd, vn); }
void CodeGenerator::sqabs(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMisc(0, 7, vd, vn); }
void CodeGenerator::sqabs(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMisc(0, 7, vd, vn); }
void CodeGenerator::sqabs(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMisc(0, 7, vd, vn); }
void CodeGenerator::sqabs(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMisc(0, 7, vd, vn); }
void CodeGenerator::sqabs(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMisc(0, 7, vd, vn); }
void CodeGenerator::sqabs(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMisc(0, 7, vd, vn); }
void CodeGenerator::abs(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMisc(0, 11, vd, vn); }
void CodeGenerator::abs(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMisc(0, 11, vd, vn); }
void CodeGenerator::abs(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMisc(0, 11, vd, vn); }
void CodeGenerator::abs(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMisc(0, 11, vd, vn); }
void CodeGenerator::abs(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMisc(0, 11, vd, vn); }
void CodeGenerator::abs(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMisc(0, 11, vd, vn); }
void CodeGenerator::abs(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMisc(0, 11, vd, vn); }
void CodeGenerator::xtn(const VReg8B &vd, const VReg8H &vn) { AdvSimd2RegMisc(0, 18, vd, vn); }
void CodeGenerator::xtn(const VReg4H &vd, const VReg4S &vn) { AdvSimd2RegMisc(0, 18, vd, vn); }
void CodeGenerator::xtn(const VReg2S &vd, const VReg2D &vn) { AdvSimd2RegMisc(0, 18, vd, vn); }
void CodeGenerator::xtn2(const VReg16B &vd, const VReg8H &vn) { AdvSimd2RegMisc(0, 18, vd, vn); }
void CodeGenerator::xtn2(const VReg8H &vd, const VReg4S &vn) { AdvSimd2RegMisc(0, 18, vd, vn); }
void CodeGenerator::xtn2(const VReg4S &vd, const VReg2D &vn) { AdvSimd2RegMisc(0, 18, vd, vn); }
void CodeGenerator::sqxtn(const VReg8B &vd, const VReg8H &vn) { AdvSimd2RegMisc(0, 20, vd, vn); }
void CodeGenerator::sqxtn(const VReg4H &vd, const VReg4S &vn) { AdvSimd2RegMisc(0, 20, vd, vn); }
void CodeGenerator::sqxtn(const VReg2S &vd, const VReg2D &vn) { AdvSimd2RegMisc(0, 20, vd, vn); }
void CodeGenerator::sqxtn2(const VReg16B &vd, const VReg8H &vn) { AdvSimd2RegMisc(0, 20, vd, vn); }
void CodeGenerator::sqxtn2(const VReg8H &vd, const VReg4S &vn) { AdvSimd2RegMisc(0, 20, vd, vn); }
void CodeGenerator::sqxtn2(const VReg4S &vd, const VReg2D &vn) { AdvSimd2RegMisc(0, 20, vd, vn); }
void CodeGenerator::rev32(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMisc(1, 0, vd, vn); }
void CodeGenerator::rev32(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMisc(1, 0, vd, vn); }
void CodeGenerator::rev32(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMisc(1, 0, vd, vn); }
void CodeGenerator::rev32(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMisc(1, 0, vd, vn); }
void CodeGenerator::uaddlp(const VReg4H &vd, const VReg8B &vn) { AdvSimd2RegMisc(1, 2, vd, vn); }
void CodeGenerator::uaddlp(const VReg8H &vd, const VReg16B &vn) { AdvSimd2RegMisc(1, 2, vd, vn); }
void CodeGenerator::uaddlp(const VReg2S &vd, const VReg4H &vn) { AdvSimd2RegMisc(1, 2, vd, vn); }
void CodeGenerator::uaddlp(const VReg4S &vd, const VReg8H &vn) { AdvSimd2RegMisc(1, 2, vd, vn); }
void CodeGenerator::uaddlp(const VReg1D &vd, const VReg2S &vn) { AdvSimd2RegMisc(1, 2, vd, vn); }
void CodeGenerator::uaddlp(const VReg2D &vd, const VReg4S &vn) { AdvSimd2RegMisc(1, 2, vd, vn); }
void CodeGenerator::usqadd(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMisc(1, 3, vd, vn); }
void CodeGenerator::usqadd(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMisc(1, 3, vd, vn); }
void CodeGenerator::usqadd(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMisc(1, 3, vd, vn); }
void CodeGenerator::usqadd(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMisc(1, 3, vd, vn); }
void CodeGenerator::usqadd(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMisc(1, 3, vd, vn); }
void CodeGenerator::usqadd(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMisc(1, 3, vd, vn); }
void CodeGenerator::usqadd(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMisc(1, 3, vd, vn); }
void CodeGenerator::clz(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMisc(1, 4, vd, vn); }
void CodeGenerator::clz(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMisc(1, 4, vd, vn); }
void CodeGenerator::clz(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMisc(1, 4, vd, vn); }
void CodeGenerator::clz(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMisc(1, 4, vd, vn); }
void CodeGenerator::clz(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMisc(1, 4, vd, vn); }
void CodeGenerator::clz(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMisc(1, 4, vd, vn); }
void CodeGenerator::uadalp(const VReg4H &vd, const VReg8B &vn) { AdvSimd2RegMisc(1, 6, vd, vn); }
void CodeGenerator::uadalp(const VReg8H &vd, const VReg16B &vn) { AdvSimd2RegMisc(1, 6, vd, vn); }
void CodeGenerator::uadalp(const VReg2S &vd, const VReg4H &vn) { AdvSimd2RegMisc(1, 6, vd, vn); }
void CodeGenerator::uadalp(const VReg4S &vd, const VReg8H &vn) { AdvSimd2RegMisc(1, 6, vd, vn); }
void CodeGenerator::uadalp(const VReg1D &vd, const VReg2S &vn) { AdvSimd2RegMisc(1, 6, vd, vn); }
void CodeGenerator::uadalp(const VReg2D &vd, const VReg4S &vn) { AdvSimd2RegMisc(1, 6, vd, vn); }
void CodeGenerator::sqneg(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMisc(1, 7, vd, vn); }
void CodeGenerator::sqneg(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMisc(1, 7, vd, vn); }
void CodeGenerator::sqneg(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMisc(1, 7, vd, vn); }
void CodeGenerator::sqneg(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMisc(1, 7, vd, vn); }
void CodeGenerator::sqneg(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMisc(1, 7, vd, vn); }
void CodeGenerator::sqneg(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMisc(1, 7, vd, vn); }
void CodeGenerator::sqneg(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMisc(1, 7, vd, vn); }
void CodeGenerator::neg(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMisc(1, 11, vd, vn); }
void CodeGenerator::neg(const VReg4H &vd, const VReg4H &vn) { AdvSimd2RegMisc(1, 11, vd, vn); }
void CodeGenerator::neg(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMisc(1, 11, vd, vn); }
void CodeGenerator::neg(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMisc(1, 11, vd, vn); }
void CodeGenerator::neg(const VReg8H &vd, const VReg8H &vn) { AdvSimd2RegMisc(1, 11, vd, vn); }
void CodeGenerator::neg(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMisc(1, 11, vd, vn); }
void CodeGenerator::neg(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMisc(1, 11, vd, vn); }
void CodeGenerator::sqxtun(const VReg8B &vd, const VReg8H &vn) { AdvSimd2RegMisc(1, 18, vd, vn); }
void CodeGenerator::sqxtun(const VReg4H &vd, const VReg4S &vn) { AdvSimd2RegMisc(1, 18, vd, vn); }
void CodeGenerator::sqxtun(const VReg2S &vd, const VReg2D &vn) { AdvSimd2RegMisc(1, 18, vd, vn); }
void CodeGenerator::sqxtun2(const VReg16B &vd, const VReg8H &vn) { AdvSimd2RegMisc(1, 18, vd, vn); }
void CodeGenerator::sqxtun2(const VReg8H &vd, const VReg4S &vn) { AdvSimd2RegMisc(1, 18, vd, vn); }
void CodeGenerator::sqxtun2(const VReg4S &vd, const VReg2D &vn) { AdvSimd2RegMisc(1, 18, vd, vn); }
void CodeGenerator::shll(const VReg8H &vd, const VReg8B &vn, const uint32_t sh) { AdvSimd2RegMisc(1, 19, vd, vn, sh); }
void CodeGenerator::shll(const VReg4S &vd, const VReg4H &vn, const uint32_t sh) { AdvSimd2RegMisc(1, 19, vd, vn, sh); }
void CodeGenerator::shll(const VReg2D &vd, const VReg2S &vn, const uint32_t sh) { AdvSimd2RegMisc(1, 19, vd, vn, sh); }
void CodeGenerator::shll2(const VReg8H &vd, const VReg16B &vn, const uint32_t sh) { AdvSimd2RegMisc(1, 19, vd, vn, sh); }
void CodeGenerator::shll2(const VReg4S &vd, const VReg8H &vn, const uint32_t sh) { AdvSimd2RegMisc(1, 19, vd, vn, sh); }
void CodeGenerator::shll2(const VReg2D &vd, const VReg4S &vn, const uint32_t sh) { AdvSimd2RegMisc(1, 19, vd, vn, sh); }
void CodeGenerator::uqxtn(const VReg8B &vd, const VReg8H &vn) { AdvSimd2RegMisc(1, 20, vd, vn); }
void CodeGenerator::uqxtn(const VReg4H &vd, const VReg4S &vn) { AdvSimd2RegMisc(1, 20, vd, vn); }
void CodeGenerator::uqxtn(const VReg2S &vd, const VReg2D &vn) { AdvSimd2RegMisc(1, 20, vd, vn); }
void CodeGenerator::uqxtn2(const VReg16B &vd, const VReg8H &vn) { AdvSimd2RegMisc(1, 20, vd, vn); }
void CodeGenerator::uqxtn2(const VReg8H &vd, const VReg4S &vn) { AdvSimd2RegMisc(1, 20, vd, vn); }
void CodeGenerator::uqxtn2(const VReg4S &vd, const VReg2D &vn) { AdvSimd2RegMisc(1, 20, vd, vn); }
void CodeGenerator::cmgt(const VReg8B &vd, const VReg8B &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 8, vd, vn, zero); }
void CodeGenerator::cmgt(const VReg4H &vd, const VReg4H &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 8, vd, vn, zero); }
void CodeGenerator::cmgt(const VReg2S &vd, const VReg2S &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 8, vd, vn, zero); }
void CodeGenerator::cmgt(const VReg16B &vd, const VReg16B &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 8, vd, vn, zero); }
void CodeGenerator::cmgt(const VReg8H &vd, const VReg8H &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 8, vd, vn, zero); }
void CodeGenerator::cmgt(const VReg4S &vd, const VReg4S &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 8, vd, vn, zero); }
void CodeGenerator::cmgt(const VReg2D &vd, const VReg2D &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 8, vd, vn, zero); }
void CodeGenerator::cmeq(const VReg8B &vd, const VReg8B &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 9, vd, vn, zero); }
void CodeGenerator::cmeq(const VReg4H &vd, const VReg4H &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 9, vd, vn, zero); }
void CodeGenerator::cmeq(const VReg2S &vd, const VReg2S &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 9, vd, vn, zero); }
void CodeGenerator::cmeq(const VReg16B &vd, const VReg16B &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 9, vd, vn, zero); }
void CodeGenerator::cmeq(const VReg8H &vd, const VReg8H &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 9, vd, vn, zero); }
void CodeGenerator::cmeq(const VReg4S &vd, const VReg4S &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 9, vd, vn, zero); }
void CodeGenerator::cmeq(const VReg2D &vd, const VReg2D &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 9, vd, vn, zero); }
void CodeGenerator::cmlt(const VReg8B &vd, const VReg8B &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 10, vd, vn, zero); }
void CodeGenerator::cmlt(const VReg4H &vd, const VReg4H &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 10, vd, vn, zero); }
void CodeGenerator::cmlt(const VReg2S &vd, const VReg2S &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 10, vd, vn, zero); }
void CodeGenerator::cmlt(const VReg16B &vd, const VReg16B &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 10, vd, vn, zero); }
void CodeGenerator::cmlt(const VReg8H &vd, const VReg8H &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 10, vd, vn, zero); }
void CodeGenerator::cmlt(const VReg4S &vd, const VReg4S &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 10, vd, vn, zero); }
void CodeGenerator::cmlt(const VReg2D &vd, const VReg2D &vn, const uint32_t zero) { AdvSimd2RegMiscZero(0, 10, vd, vn, zero); }
void CodeGenerator::cmge(const VReg8B &vd, const VReg8B &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 8, vd, vn, zero); }
void CodeGenerator::cmge(const VReg4H &vd, const VReg4H &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 8, vd, vn, zero); }
void CodeGenerator::cmge(const VReg2S &vd, const VReg2S &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 8, vd, vn, zero); }
void CodeGenerator::cmge(const VReg16B &vd, const VReg16B &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 8, vd, vn, zero); }
void CodeGenerator::cmge(const VReg8H &vd, const VReg8H &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 8, vd, vn, zero); }
void CodeGenerator::cmge(const VReg4S &vd, const VReg4S &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 8, vd, vn, zero); }
void CodeGenerator::cmge(const VReg2D &vd, const VReg2D &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 8, vd, vn, zero); }
void CodeGenerator::cmle(const VReg8B &vd, const VReg8B &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 9, vd, vn, zero); }
void CodeGenerator::cmle(const VReg4H &vd, const VReg4H &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 9, vd, vn, zero); }
void CodeGenerator::cmle(const VReg2S &vd, const VReg2S &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 9, vd, vn, zero); }
void CodeGenerator::cmle(const VReg16B &vd, const VReg16B &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 9, vd, vn, zero); }
void CodeGenerator::cmle(const VReg8H &vd, const VReg8H &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 9, vd, vn, zero); }
void CodeGenerator::cmle(const VReg4S &vd, const VReg4S &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 9, vd, vn, zero); }
void CodeGenerator::cmle(const VReg2D &vd, const VReg2D &vn, const uint32_t zero) { AdvSimd2RegMiscZero(1, 9, vd, vn, zero); }
void CodeGenerator::not_(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMiscSz(1, 0, 5, vd, vn); }
void CodeGenerator::not_(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMiscSz(1, 0, 5, vd, vn); }
void CodeGenerator::mvn(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMiscSz(1, 0, 5, vd, vn); }
void CodeGenerator::mvn(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMiscSz(1, 0, 5, vd, vn); }
void CodeGenerator::rbit(const VReg8B &vd, const VReg8B &vn) { AdvSimd2RegMiscSz(1, 1, 5, vd, vn); }
void CodeGenerator::rbit(const VReg16B &vd, const VReg16B &vn) { AdvSimd2RegMiscSz(1, 1, 5, vd, vn); }
void CodeGenerator::fcvtn(const VReg4H &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(0, 22, vd, vn); }
void CodeGenerator::fcvtn(const VReg2S &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(0, 22, vd, vn); }
void CodeGenerator::fcvtn2(const VReg8H &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(0, 22, vd, vn); }
void CodeGenerator::fcvtn2(const VReg4S &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(0, 22, vd, vn); }
void CodeGenerator::fcvtl(const VReg4S &vd, const VReg4H &vn) { AdvSimd2RegMiscSz0x(0, 23, vd, vn); }
void CodeGenerator::fcvtl(const VReg2D &vd, const VReg2S &vn) { AdvSimd2RegMiscSz0x(0, 23, vd, vn); }
void CodeGenerator::fcvtl2(const VReg4S &vd, const VReg8H &vn) { AdvSimd2RegMiscSz0x(0, 23, vd, vn); }
void CodeGenerator::fcvtl2(const VReg2D &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(0, 23, vd, vn); }
void CodeGenerator::frintn(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz0x(0, 24, vd, vn); }
void CodeGenerator::frintn(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(0, 24, vd, vn); }
void CodeGenerator::frintn(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(0, 24, vd, vn); }
void CodeGenerator::frintm(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz0x(0, 25, vd, vn); }
void CodeGenerator::frintm(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(0, 25, vd, vn); }
void CodeGenerator::frintm(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(0, 25, vd, vn); }
void CodeGenerator::fcvtns(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz0x(0, 26, vd, vn); }
void CodeGenerator::fcvtns(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(0, 26, vd, vn); }
void CodeGenerator::fcvtns(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(0, 26, vd, vn); }
void CodeGenerator::fcvtms(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz0x(0, 27, vd, vn); }
void CodeGenerator::fcvtms(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(0, 27, vd, vn); }
void CodeGenerator::fcvtms(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(0, 27, vd, vn); }
void CodeGenerator::fcvtas(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz0x(0, 28, vd, vn); }
void CodeGenerator::fcvtas(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(0, 28, vd, vn); }
void CodeGenerator::fcvtas(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(0, 28, vd, vn); }
void CodeGenerator::scvtf(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz0x(0, 29, vd, vn); }
void CodeGenerator::scvtf(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(0, 29, vd, vn); }
void CodeGenerator::scvtf(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(0, 29, vd, vn); }
void CodeGenerator::fcvtxn(const VReg2S &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(1, 22, vd, vn); }
void CodeGenerator::fcvtxn(const VReg4S &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(1, 22, vd, vn); }
void CodeGenerator::fcvtxn2(const VReg2S &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(1, 22, vd, vn); }
void CodeGenerator::fcvtxn2(const VReg4S &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(1, 22, vd, vn); }
void CodeGenerator::frinta(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz0x(1, 24, vd, vn); }
void CodeGenerator::frinta(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(1, 24, vd, vn); }
void CodeGenerator::frinta(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(1, 24, vd, vn); }
void CodeGenerator::frintx(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz0x(1, 25, vd, vn); }
void CodeGenerator::frintx(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(1, 25, vd, vn); }
void CodeGenerator::frintx(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(1, 25, vd, vn); }
void CodeGenerator::fcvtnu(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz0x(1, 26, vd, vn); }
void CodeGenerator::fcvtnu(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(1, 26, vd, vn); }
void CodeGenerator::fcvtnu(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(1, 26, vd, vn); }
void CodeGenerator::fcvtmu(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz0x(1, 27, vd, vn); }
void CodeGenerator::fcvtmu(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(1, 27, vd, vn); }
void CodeGenerator::fcvtmu(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(1, 27, vd, vn); }
void CodeGenerator::fcvtau(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz0x(1, 28, vd, vn); }
void CodeGenerator::fcvtau(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(1, 28, vd, vn); }
void CodeGenerator::fcvtau(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(1, 28, vd, vn); }
void CodeGenerator::ucvtf(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz0x(1, 29, vd, vn); }
void CodeGenerator::ucvtf(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz0x(1, 29, vd, vn); }
void CodeGenerator::ucvtf(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz0x(1, 29, vd, vn); }
void CodeGenerator::fcmgt(const VReg2S &vd, const VReg2S &vn, const double zero) { AdvSimd2RegMiscSz1x(0, 12, vd, vn, zero); }
void CodeGenerator::fcmgt(const VReg4S &vd, const VReg4S &vn, const double zero) { AdvSimd2RegMiscSz1x(0, 12, vd, vn, zero); }
void CodeGenerator::fcmgt(const VReg2D &vd, const VReg2D &vn, const double zero) { AdvSimd2RegMiscSz1x(0, 12, vd, vn, zero); }
void CodeGenerator::fcmeq(const VReg2S &vd, const VReg2S &vn, const double zero) { AdvSimd2RegMiscSz1x(0, 13, vd, vn, zero); }
void CodeGenerator::fcmeq(const VReg4S &vd, const VReg4S &vn, const double zero) { AdvSimd2RegMiscSz1x(0, 13, vd, vn, zero); }
void CodeGenerator::fcmeq(const VReg2D &vd, const VReg2D &vn, const double zero) { AdvSimd2RegMiscSz1x(0, 13, vd, vn, zero); }
void CodeGenerator::fcmlt(const VReg2S &vd, const VReg2S &vn, const double zero) { AdvSimd2RegMiscSz1x(0, 14, vd, vn, zero); }
void CodeGenerator::fcmlt(const VReg4S &vd, const VReg4S &vn, const double zero) { AdvSimd2RegMiscSz1x(0, 14, vd, vn, zero); }
void CodeGenerator::fcmlt(const VReg2D &vd, const VReg2D &vn, const double zero) { AdvSimd2RegMiscSz1x(0, 14, vd, vn, zero); }
void CodeGenerator::fabs(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(0, 15, vd, vn); }
void CodeGenerator::fabs(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(0, 15, vd, vn); }
void CodeGenerator::fabs(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz1x(0, 15, vd, vn); }
void CodeGenerator::frintp(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(0, 24, vd, vn); }
void CodeGenerator::frintp(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(0, 24, vd, vn); }
void CodeGenerator::frintp(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz1x(0, 24, vd, vn); }
void CodeGenerator::frintz(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(0, 25, vd, vn); }
void CodeGenerator::frintz(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(0, 25, vd, vn); }
void CodeGenerator::frintz(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz1x(0, 25, vd, vn); }
void CodeGenerator::fcvtps(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(0, 26, vd, vn); }
void CodeGenerator::fcvtps(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(0, 26, vd, vn); }
void CodeGenerator::fcvtps(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz1x(0, 26, vd, vn); }
void CodeGenerator::fcvtzs(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(0, 27, vd, vn); }
void CodeGenerator::fcvtzs(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(0, 27, vd, vn); }
void CodeGenerator::fcvtzs(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz1x(0, 27, vd, vn); }
void CodeGenerator::urecpe(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(0, 28, vd, vn); }
void CodeGenerator::urecpe(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(0, 28, vd, vn); }
void CodeGenerator::frecpe(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(0, 29, vd, vn); }
void CodeGenerator::frecpe(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(0, 29, vd, vn); }
void CodeGenerator::frecpe(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz1x(0, 29, vd, vn); }
void CodeGenerator::fcmge(const VReg2S &vd, const VReg2S &vn, const double zero) { AdvSimd2RegMiscSz1x(1, 12, vd, vn, zero); }
void CodeGenerator::fcmge(const VReg4S &vd, const VReg4S &vn, const double zero) { AdvSimd2RegMiscSz1x(1, 12, vd, vn, zero); }
void CodeGenerator::fcmge(const VReg2D &vd, const VReg2D &vn, const double zero) { AdvSimd2RegMiscSz1x(1, 12, vd, vn, zero); }
void CodeGenerator::fcmle(const VReg2S &vd, const VReg2S &vn, const double zero) { AdvSimd2RegMiscSz1x(1, 13, vd, vn, zero); }
void CodeGenerator::fcmle(const VReg4S &vd, const VReg4S &vn, const double zero) { AdvSimd2RegMiscSz1x(1, 13, vd, vn, zero); }
void CodeGenerator::fcmle(const VReg2D &vd, const VReg2D &vn, const double zero) { AdvSimd2RegMiscSz1x(1, 13, vd, vn, zero); }
void CodeGenerator::fneg(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(1, 15, vd, vn); }
void CodeGenerator::fneg(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(1, 15, vd, vn); }
void CodeGenerator::fneg(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz1x(1, 15, vd, vn); }
void CodeGenerator::frinti(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(1, 25, vd, vn); }
void CodeGenerator::frinti(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(1, 25, vd, vn); }
void CodeGenerator::frinti(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz1x(1, 25, vd, vn); }
void CodeGenerator::fcvtpu(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(1, 26, vd, vn); }
void CodeGenerator::fcvtpu(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(1, 26, vd, vn); }
void CodeGenerator::fcvtpu(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz1x(1, 26, vd, vn); }
void CodeGenerator::fcvtzu(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(1, 27, vd, vn); }
void CodeGenerator::fcvtzu(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(1, 27, vd, vn); }
void CodeGenerator::fcvtzu(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz1x(1, 27, vd, vn); }
void CodeGenerator::ursqrte(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(1, 28, vd, vn); }
void CodeGenerator::ursqrte(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(1, 28, vd, vn); }
void CodeGenerator::frsqrte(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(1, 29, vd, vn); }
void CodeGenerator::frsqrte(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(1, 29, vd, vn); }
void CodeGenerator::frsqrte(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz1x(1, 29, vd, vn); }
void CodeGenerator::fsqrt(const VReg2S &vd, const VReg2S &vn) { AdvSimd2RegMiscSz1x(1, 31, vd, vn); }
void CodeGenerator::fsqrt(const VReg4S &vd, const VReg4S &vn) { AdvSimd2RegMiscSz1x(1, 31, vd, vn); }
void CodeGenerator::fsqrt(const VReg2D &vd, const VReg2D &vn) { AdvSimd2RegMiscSz1x(1, 31, vd, vn); }
void CodeGenerator::saddlv(const HReg &vd, const VReg8B &vn) { AdvSimdAcrossLanes(0, 3, vd, vn); }
void CodeGenerator::saddlv(const HReg &vd, const VReg16B &vn) { AdvSimdAcrossLanes(0, 3, vd, vn); }
void CodeGenerator::saddlv(const SReg &vd, const VReg4H &vn) { AdvSimdAcrossLanes(0, 3, vd, vn); }
void CodeGenerator::saddlv(const SReg &vd, const VReg8H &vn) { AdvSimdAcrossLanes(0, 3, vd, vn); }
void CodeGenerator::saddlv(const DReg &vd, const VReg4S &vn) { AdvSimdAcrossLanes(0, 3, vd, vn); }
void CodeGenerator::smaxv(const BReg &vd, const VReg8B &vn) { AdvSimdAcrossLanes(0, 10, vd, vn); }
void CodeGenerator::smaxv(const BReg &vd, const VReg16B &vn) { AdvSimdAcrossLanes(0, 10, vd, vn); }
void CodeGenerator::smaxv(const HReg &vd, const VReg4H &vn) { AdvSimdAcrossLanes(0, 10, vd, vn); }
void CodeGenerator::smaxv(const HReg &vd, const VReg8H &vn) { AdvSimdAcrossLanes(0, 10, vd, vn); }
void CodeGenerator::smaxv(const SReg &vd, const VReg4S &vn) { AdvSimdAcrossLanes(0, 10, vd, vn); }
void CodeGenerator::sminv(const BReg &vd, const VReg8B &vn) { AdvSimdAcrossLanes(0, 26, vd, vn); }
void CodeGenerator::sminv(const BReg &vd, const VReg16B &vn) { AdvSimdAcrossLanes(0, 26, vd, vn); }
void CodeGenerator::sminv(const HReg &vd, const VReg4H &vn) { AdvSimdAcrossLanes(0, 26, vd, vn); }
void CodeGenerator::sminv(const HReg &vd, const VReg8H &vn) { AdvSimdAcrossLanes(0, 26, vd, vn); }
void CodeGenerator::sminv(const SReg &vd, const VReg4S &vn) { AdvSimdAcrossLanes(0, 26, vd, vn); }
void CodeGenerator::addv(const BReg &vd, const VReg8B &vn) { AdvSimdAcrossLanes(0, 27, vd, vn); }
void CodeGenerator::addv(const BReg &vd, const VReg16B &vn) { AdvSimdAcrossLanes(0, 27, vd, vn); }
void CodeGenerator::addv(const HReg &vd, const VReg4H &vn) { AdvSimdAcrossLanes(0, 27, vd, vn); }
void CodeGenerator::addv(const HReg &vd, const VReg8H &vn) { AdvSimdAcrossLanes(0, 27, vd, vn); }
void CodeGenerator::addv(const SReg &vd, const VReg4S &vn) { AdvSimdAcrossLanes(0, 27, vd, vn); }
void CodeGenerator::uaddlv(const HReg &vd, const VReg8B &vn) { AdvSimdAcrossLanes(1, 3, vd, vn); }
void CodeGenerator::uaddlv(const HReg &vd, const VReg16B &vn) { AdvSimdAcrossLanes(1, 3, vd, vn); }
void CodeGenerator::uaddlv(const SReg &vd, const VReg4H &vn) { AdvSimdAcrossLanes(1, 3, vd, vn); }
void CodeGenerator::uaddlv(const SReg &vd, const VReg8H &vn) { AdvSimdAcrossLanes(1, 3, vd, vn); }
void CodeGenerator::uaddlv(const DReg &vd, const VReg4S &vn) { AdvSimdAcrossLanes(1, 3, vd, vn); }
void CodeGenerator::umaxv(const BReg &vd, const VReg8B &vn) { AdvSimdAcrossLanes(1, 10, vd, vn); }
void CodeGenerator::umaxv(const BReg &vd, const VReg16B &vn) { AdvSimdAcrossLanes(1, 10, vd, vn); }
void CodeGenerator::umaxv(const HReg &vd, const VReg4H &vn) { AdvSimdAcrossLanes(1, 10, vd, vn); }
void CodeGenerator::umaxv(const HReg &vd, const VReg8H &vn) { AdvSimdAcrossLanes(1, 10, vd, vn); }
void CodeGenerator::umaxv(const SReg &vd, const VReg4S &vn) { AdvSimdAcrossLanes(1, 10, vd, vn); }
void CodeGenerator::uminv(const BReg &vd, const VReg8B &vn) { AdvSimdAcrossLanes(1, 26, vd, vn); }
void CodeGenerator::uminv(const BReg &vd, const VReg16B &vn) { AdvSimdAcrossLanes(1, 26, vd, vn); }
void CodeGenerator::uminv(const HReg &vd, const VReg4H &vn) { AdvSimdAcrossLanes(1, 26, vd, vn); }
void CodeGenerator::uminv(const HReg &vd, const VReg8H &vn) { AdvSimdAcrossLanes(1, 26, vd, vn); }
void CodeGenerator::uminv(const SReg &vd, const VReg4S &vn) { AdvSimdAcrossLanes(1, 26, vd, vn); }
void CodeGenerator::fmaxnmv(const HReg &vd, const VReg4H &vn) { AdvSimdAcrossLanesSz0x(0, 12, vd, vn); }
void CodeGenerator::fmaxnmv(const HReg &vd, const VReg8H &vn) { AdvSimdAcrossLanesSz0x(0, 12, vd, vn); }
void CodeGenerator::fmaxv(const HReg &vd, const VReg4H &vn) { AdvSimdAcrossLanesSz0x(0, 15, vd, vn); }
void CodeGenerator::fmaxv(const HReg &vd, const VReg8H &vn) { AdvSimdAcrossLanesSz0x(0, 15, vd, vn); }
void CodeGenerator::fmaxnmv(const SReg &vd, const VReg4S &vn) { AdvSimdAcrossLanesSz0x(1, 12, vd, vn); }
void CodeGenerator::fmaxv(const SReg &vd, const VReg4S &vn) { AdvSimdAcrossLanesSz0x(1, 15, vd, vn); }
void CodeGenerator::fminnmv(const HReg &vd, const VReg4H &vn) { AdvSimdAcrossLanesSz1x(0, 12, vd, vn); }
void CodeGenerator::fminnmv(const HReg &vd, const VReg8H &vn) { AdvSimdAcrossLanesSz1x(0, 12, vd, vn); }
void CodeGenerator::fminv(const HReg &vd, const VReg4H &vn) { AdvSimdAcrossLanesSz1x(0, 15, vd, vn); }
void CodeGenerator::fminv(const HReg &vd, const VReg8H &vn) { AdvSimdAcrossLanesSz1x(0, 15, vd, vn); }
void CodeGenerator::fminnmv(const SReg &vd, const VReg4S &vn) { AdvSimdAcrossLanesSz1x(1, 12, vd, vn); }
void CodeGenerator::fminv(const SReg &vd, const VReg4S &vn) { AdvSimdAcrossLanesSz1x(1, 15, vd, vn); }
void CodeGenerator::saddl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(0, 0, vd, vn, vm); }
void CodeGenerator::saddl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(0, 0, vd, vn, vm); }
void CodeGenerator::saddl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(0, 0, vd, vn, vm); }
void CodeGenerator::saddl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(0, 0, vd, vn, vm); }
void CodeGenerator::saddl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 0, vd, vn, vm); }
void CodeGenerator::saddl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 0, vd, vn, vm); }
void CodeGenerator::saddw(const VReg8H &vd, const VReg8H &vn, const VReg8B &vm) { AdvSimd3Diff(0, 1, vd, vn, vm); }
void CodeGenerator::saddw(const VReg4S &vd, const VReg4S &vn, const VReg4H &vm) { AdvSimd3Diff(0, 1, vd, vn, vm); }
void CodeGenerator::saddw(const VReg2D &vd, const VReg2D &vn, const VReg2S &vm) { AdvSimd3Diff(0, 1, vd, vn, vm); }
void CodeGenerator::saddw2(const VReg8H &vd, const VReg8H &vn, const VReg16B &vm) { AdvSimd3Diff(0, 1, vd, vn, vm); }
void CodeGenerator::saddw2(const VReg4S &vd, const VReg4S &vn, const VReg8H &vm) { AdvSimd3Diff(0, 1, vd, vn, vm); }
void CodeGenerator::saddw2(const VReg2D &vd, const VReg2D &vn, const VReg4S &vm) { AdvSimd3Diff(0, 1, vd, vn, vm); }
void CodeGenerator::ssubl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(0, 2, vd, vn, vm); }
void CodeGenerator::ssubl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(0, 2, vd, vn, vm); }
void CodeGenerator::ssubl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(0, 2, vd, vn, vm); }
void CodeGenerator::ssubl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(0, 2, vd, vn, vm); }
void CodeGenerator::ssubl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 2, vd, vn, vm); }
void CodeGenerator::ssubl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 2, vd, vn, vm); }
void CodeGenerator::ssubw(const VReg8H &vd, const VReg8H &vn, const VReg8B &vm) { AdvSimd3Diff(0, 3, vd, vn, vm); }
void CodeGenerator::ssubw(const VReg4S &vd, const VReg4S &vn, const VReg4H &vm) { AdvSimd3Diff(0, 3, vd, vn, vm); }
void CodeGenerator::ssubw(const VReg2D &vd, const VReg2D &vn, const VReg2S &vm) { AdvSimd3Diff(0, 3, vd, vn, vm); }
void CodeGenerator::ssubw2(const VReg8H &vd, const VReg8H &vn, const VReg16B &vm) { AdvSimd3Diff(0, 3, vd, vn, vm); }
void CodeGenerator::ssubw2(const VReg4S &vd, const VReg4S &vn, const VReg8H &vm) { AdvSimd3Diff(0, 3, vd, vn, vm); }
void CodeGenerator::ssubw2(const VReg2D &vd, const VReg2D &vn, const VReg4S &vm) { AdvSimd3Diff(0, 3, vd, vn, vm); }
void CodeGenerator::addhn(const VReg8B &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 4, vd, vn, vm); }
void CodeGenerator::addhn(const VReg4H &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 4, vd, vn, vm); }
void CodeGenerator::addhn(const VReg2S &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Diff(0, 4, vd, vn, vm); }
void CodeGenerator::addhn2(const VReg16B &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 4, vd, vn, vm); }
void CodeGenerator::addhn2(const VReg8H &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 4, vd, vn, vm); }
void CodeGenerator::addhn2(const VReg4S &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Diff(0, 4, vd, vn, vm); }
void CodeGenerator::sabal(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(0, 5, vd, vn, vm); }
void CodeGenerator::sabal(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(0, 5, vd, vn, vm); }
void CodeGenerator::sabal(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(0, 5, vd, vn, vm); }
void CodeGenerator::sabal2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(0, 5, vd, vn, vm); }
void CodeGenerator::sabal2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 5, vd, vn, vm); }
void CodeGenerator::sabal2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 5, vd, vn, vm); }
void CodeGenerator::subhn(const VReg8B &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 6, vd, vn, vm); }
void CodeGenerator::subhn(const VReg4H &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 6, vd, vn, vm); }
void CodeGenerator::subhn(const VReg2S &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Diff(0, 6, vd, vn, vm); }
void CodeGenerator::subhn2(const VReg16B &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 6, vd, vn, vm); }
void CodeGenerator::subhn2(const VReg8H &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 6, vd, vn, vm); }
void CodeGenerator::subhn2(const VReg4S &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Diff(0, 6, vd, vn, vm); }
void CodeGenerator::sabdl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(0, 7, vd, vn, vm); }
void CodeGenerator::sabdl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(0, 7, vd, vn, vm); }
void CodeGenerator::sabdl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(0, 7, vd, vn, vm); }
void CodeGenerator::sabdl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(0, 7, vd, vn, vm); }
void CodeGenerator::sabdl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 7, vd, vn, vm); }
void CodeGenerator::sabdl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 7, vd, vn, vm); }
void CodeGenerator::smlal(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(0, 8, vd, vn, vm); }
void CodeGenerator::smlal(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(0, 8, vd, vn, vm); }
void CodeGenerator::smlal(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(0, 8, vd, vn, vm); }
void CodeGenerator::smlal2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(0, 8, vd, vn, vm); }
void CodeGenerator::smlal2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 8, vd, vn, vm); }
void CodeGenerator::smlal2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 8, vd, vn, vm); }
void CodeGenerator::sqdmlal(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(0, 9, vd, vn, vm); }
void CodeGenerator::sqdmlal(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(0, 9, vd, vn, vm); }
void CodeGenerator::sqdmlal2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 9, vd, vn, vm); }
void CodeGenerator::sqdmlal2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 9, vd, vn, vm); }
void CodeGenerator::smlsl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(0, 10, vd, vn, vm); }
void CodeGenerator::smlsl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(0, 10, vd, vn, vm); }
void CodeGenerator::smlsl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(0, 10, vd, vn, vm); }
void CodeGenerator::smlsl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(0, 10, vd, vn, vm); }
void CodeGenerator::smlsl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 10, vd, vn, vm); }
void CodeGenerator::smlsl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 10, vd, vn, vm); }
void CodeGenerator::sqdmlsl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(0, 11, vd, vn, vm); }
void CodeGenerator::sqdmlsl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(0, 11, vd, vn, vm); }
void CodeGenerator::sqdmlsl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 11, vd, vn, vm); }
void CodeGenerator::sqdmlsl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 11, vd, vn, vm); }
void CodeGenerator::smull(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(0, 12, vd, vn, vm); }
void CodeGenerator::smull(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(0, 12, vd, vn, vm); }
void CodeGenerator::smull(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(0, 12, vd, vn, vm); }
void CodeGenerator::smull2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(0, 12, vd, vn, vm); }
void CodeGenerator::smull2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 12, vd, vn, vm); }
void CodeGenerator::smull2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 12, vd, vn, vm); }
void CodeGenerator::sqdmull(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(0, 13, vd, vn, vm); }
void CodeGenerator::sqdmull(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(0, 13, vd, vn, vm); }
void CodeGenerator::sqdmull2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(0, 13, vd, vn, vm); }
void CodeGenerator::sqdmull2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(0, 13, vd, vn, vm); }
void CodeGenerator::pmull(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(0, 14, vd, vn, vm); }
void CodeGenerator::pmull(const VReg1Q &vd, const VReg1D &vn, const VReg1D &vm) { AdvSimd3Diff(0, 14, vd, vn, vm); }
void CodeGenerator::pmull2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(0, 14, vd, vn, vm); }
void CodeGenerator::pmull2(const VReg1Q &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Diff(0, 14, vd, vn, vm); }
void CodeGenerator::uaddl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(1, 0, vd, vn, vm); }
void CodeGenerator::uaddl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(1, 0, vd, vn, vm); }
void CodeGenerator::uaddl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(1, 0, vd, vn, vm); }
void CodeGenerator::uaddl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(1, 0, vd, vn, vm); }
void CodeGenerator::uaddl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(1, 0, vd, vn, vm); }
void CodeGenerator::uaddl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(1, 0, vd, vn, vm); }
void CodeGenerator::uaddw(const VReg8H &vd, const VReg8H &vn, const VReg8B &vm) { AdvSimd3Diff(1, 1, vd, vn, vm); }
void CodeGenerator::uaddw(const VReg4S &vd, const VReg4S &vn, const VReg4H &vm) { AdvSimd3Diff(1, 1, vd, vn, vm); }
void CodeGenerator::uaddw(const VReg2D &vd, const VReg2D &vn, const VReg2S &vm) { AdvSimd3Diff(1, 1, vd, vn, vm); }
void CodeGenerator::uaddw2(const VReg8H &vd, const VReg8H &vn, const VReg16B &vm) { AdvSimd3Diff(1, 1, vd, vn, vm); }
void CodeGenerator::uaddw2(const VReg4S &vd, const VReg4S &vn, const VReg8H &vm) { AdvSimd3Diff(1, 1, vd, vn, vm); }
void CodeGenerator::uaddw2(const VReg2D &vd, const VReg2D &vn, const VReg4S &vm) { AdvSimd3Diff(1, 1, vd, vn, vm); }
void CodeGenerator::usubl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(1, 2, vd, vn, vm); }
void CodeGenerator::usubl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(1, 2, vd, vn, vm); }
void CodeGenerator::usubl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(1, 2, vd, vn, vm); }
void CodeGenerator::usubl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(1, 2, vd, vn, vm); }
void CodeGenerator::usubl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(1, 2, vd, vn, vm); }
void CodeGenerator::usubl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(1, 2, vd, vn, vm); }
void CodeGenerator::usubw(const VReg8H &vd, const VReg8H &vn, const VReg8B &vm) { AdvSimd3Diff(1, 3, vd, vn, vm); }
void CodeGenerator::usubw(const VReg4S &vd, const VReg4S &vn, const VReg4H &vm) { AdvSimd3Diff(1, 3, vd, vn, vm); }
void CodeGenerator::usubw(const VReg2D &vd, const VReg2D &vn, const VReg2S &vm) { AdvSimd3Diff(1, 3, vd, vn, vm); }
void CodeGenerator::usubw2(const VReg8H &vd, const VReg8H &vn, const VReg16B &vm) { AdvSimd3Diff(1, 3, vd, vn, vm); }
void CodeGenerator::usubw2(const VReg4S &vd, const VReg4S &vn, const VReg8H &vm) { AdvSimd3Diff(1, 3, vd, vn, vm); }
void CodeGenerator::usubw2(const VReg2D &vd, const VReg2D &vn, const VReg4S &vm) { AdvSimd3Diff(1, 3, vd, vn, vm); }
void CodeGenerator::raddhn(const VReg8B &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(1, 4, vd, vn, vm); }
void CodeGenerator::raddhn(const VReg4H &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(1, 4, vd, vn, vm); }
void CodeGenerator::raddhn(const VReg2S &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Diff(1, 4, vd, vn, vm); }
void CodeGenerator::raddhn2(const VReg16B &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(1, 4, vd, vn, vm); }
void CodeGenerator::raddhn2(const VReg8H &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(1, 4, vd, vn, vm); }
void CodeGenerator::raddhn2(const VReg4S &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Diff(1, 4, vd, vn, vm); }
void CodeGenerator::uabal(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(1, 5, vd, vn, vm); }
void CodeGenerator::uabal(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(1, 5, vd, vn, vm); }
void CodeGenerator::uabal(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(1, 5, vd, vn, vm); }
void CodeGenerator::uabal2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(1, 5, vd, vn, vm); }
void CodeGenerator::uabal2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(1, 5, vd, vn, vm); }
void CodeGenerator::uabal2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(1, 5, vd, vn, vm); }
void CodeGenerator::rsubhn(const VReg8B &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(1, 6, vd, vn, vm); }
void CodeGenerator::rsubhn(const VReg4H &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(1, 6, vd, vn, vm); }
void CodeGenerator::rsubhn(const VReg2S &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Diff(1, 6, vd, vn, vm); }
void CodeGenerator::rsubhn2(const VReg16B &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(1, 6, vd, vn, vm); }
void CodeGenerator::rsubhn2(const VReg8H &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(1, 6, vd, vn, vm); }
void CodeGenerator::rsubhn2(const VReg4S &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Diff(1, 6, vd, vn, vm); }
void CodeGenerator::uabdl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(1, 7, vd, vn, vm); }
void CodeGenerator::uabdl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(1, 7, vd, vn, vm); }
void CodeGenerator::uabdl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(1, 7, vd, vn, vm); }
void CodeGenerator::uabdl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(1, 7, vd, vn, vm); }
void CodeGenerator::uabdl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(1, 7, vd, vn, vm); }
void CodeGenerator::uabdl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(1, 7, vd, vn, vm); }
void CodeGenerator::umlal(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(1, 8, vd, vn, vm); }
void CodeGenerator::umlal(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(1, 8, vd, vn, vm); }
void CodeGenerator::umlal(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(1, 8, vd, vn, vm); }
void CodeGenerator::umlal2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(1, 8, vd, vn, vm); }
void CodeGenerator::umlal2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(1, 8, vd, vn, vm); }
void CodeGenerator::umlal2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(1, 8, vd, vn, vm); }
void CodeGenerator::umlsl(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(1, 10, vd, vn, vm); }
void CodeGenerator::umlsl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(1, 10, vd, vn, vm); }
void CodeGenerator::umlsl(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(1, 10, vd, vn, vm); }
void CodeGenerator::umlsl2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(1, 10, vd, vn, vm); }
void CodeGenerator::umlsl2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(1, 10, vd, vn, vm); }
void CodeGenerator::umlsl2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(1, 10, vd, vn, vm); }
void CodeGenerator::umull(const VReg8H &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Diff(1, 12, vd, vn, vm); }
void CodeGenerator::umull(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Diff(1, 12, vd, vn, vm); }
void CodeGenerator::umull(const VReg2D &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Diff(1, 12, vd, vn, vm); }
void CodeGenerator::umull2(const VReg8H &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Diff(1, 12, vd, vn, vm); }
void CodeGenerator::umull2(const VReg4S &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Diff(1, 12, vd, vn, vm); }
void CodeGenerator::umull2(const VReg2D &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Diff(1, 12, vd, vn, vm); }
void CodeGenerator::shadd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 0, vd, vn, vm); }
void CodeGenerator::shadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 0, vd, vn, vm); }
void CodeGenerator::shadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 0, vd, vn, vm); }
void CodeGenerator::shadd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 0, vd, vn, vm); }
void CodeGenerator::shadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 0, vd, vn, vm); }
void CodeGenerator::shadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 0, vd, vn, vm); }
void CodeGenerator::sqadd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 1, vd, vn, vm); }
void CodeGenerator::sqadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 1, vd, vn, vm); }
void CodeGenerator::sqadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 1, vd, vn, vm); }
void CodeGenerator::sqadd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 1, vd, vn, vm); }
void CodeGenerator::sqadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 1, vd, vn, vm); }
void CodeGenerator::sqadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 1, vd, vn, vm); }
void CodeGenerator::sqadd(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(0, 1, vd, vn, vm); }
void CodeGenerator::srhadd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 2, vd, vn, vm); }
void CodeGenerator::srhadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 2, vd, vn, vm); }
void CodeGenerator::srhadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 2, vd, vn, vm); }
void CodeGenerator::srhadd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 2, vd, vn, vm); }
void CodeGenerator::srhadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 2, vd, vn, vm); }
void CodeGenerator::srhadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 2, vd, vn, vm); }
void CodeGenerator::shsub(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 4, vd, vn, vm); }
void CodeGenerator::shsub(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 4, vd, vn, vm); }
void CodeGenerator::shsub(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 4, vd, vn, vm); }
void CodeGenerator::shsub(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 4, vd, vn, vm); }
void CodeGenerator::shsub(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 4, vd, vn, vm); }
void CodeGenerator::shsub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 4, vd, vn, vm); }
void CodeGenerator::sqsub(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 5, vd, vn, vm); }
void CodeGenerator::sqsub(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 5, vd, vn, vm); }
void CodeGenerator::sqsub(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 5, vd, vn, vm); }
void CodeGenerator::sqsub(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 5, vd, vn, vm); }
void CodeGenerator::sqsub(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 5, vd, vn, vm); }
void CodeGenerator::sqsub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 5, vd, vn, vm); }
void CodeGenerator::sqsub(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(0, 5, vd, vn, vm); }
void CodeGenerator::cmgt(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 6, vd, vn, vm); }
void CodeGenerator::cmgt(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 6, vd, vn, vm); }
void CodeGenerator::cmgt(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 6, vd, vn, vm); }
void CodeGenerator::cmgt(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 6, vd, vn, vm); }
void CodeGenerator::cmgt(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 6, vd, vn, vm); }
void CodeGenerator::cmgt(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 6, vd, vn, vm); }
void CodeGenerator::cmgt(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(0, 6, vd, vn, vm); }
void CodeGenerator::cmge(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 7, vd, vn, vm); }
void CodeGenerator::cmge(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 7, vd, vn, vm); }
void CodeGenerator::cmge(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 7, vd, vn, vm); }
void CodeGenerator::cmge(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 7, vd, vn, vm); }
void CodeGenerator::cmge(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 7, vd, vn, vm); }
void CodeGenerator::cmge(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 7, vd, vn, vm); }
void CodeGenerator::cmge(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(0, 7, vd, vn, vm); }
void CodeGenerator::sshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 8, vd, vn, vm); }
void CodeGenerator::sshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 8, vd, vn, vm); }
void CodeGenerator::sshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 8, vd, vn, vm); }
void CodeGenerator::sshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 8, vd, vn, vm); }
void CodeGenerator::sshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 8, vd, vn, vm); }
void CodeGenerator::sshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 8, vd, vn, vm); }
void CodeGenerator::sshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(0, 8, vd, vn, vm); }
void CodeGenerator::sqshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 9, vd, vn, vm); }
void CodeGenerator::sqshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 9, vd, vn, vm); }
void CodeGenerator::sqshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 9, vd, vn, vm); }
void CodeGenerator::sqshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 9, vd, vn, vm); }
void CodeGenerator::sqshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 9, vd, vn, vm); }
void CodeGenerator::sqshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 9, vd, vn, vm); }
void CodeGenerator::sqshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(0, 9, vd, vn, vm); }
void CodeGenerator::srshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 10, vd, vn, vm); }
void CodeGenerator::srshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 10, vd, vn, vm); }
void CodeGenerator::srshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 10, vd, vn, vm); }
void CodeGenerator::srshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 10, vd, vn, vm); }
void CodeGenerator::srshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 10, vd, vn, vm); }
void CodeGenerator::srshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 10, vd, vn, vm); }
void CodeGenerator::srshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(0, 10, vd, vn, vm); }
void CodeGenerator::sqrshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 11, vd, vn, vm); }
void CodeGenerator::sqrshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 11, vd, vn, vm); }
void CodeGenerator::sqrshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 11, vd, vn, vm); }
void CodeGenerator::sqrshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 11, vd, vn, vm); }
void CodeGenerator::sqrshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 11, vd, vn, vm); }
void CodeGenerator::sqrshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 11, vd, vn, vm); }
void CodeGenerator::sqrshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(0, 11, vd, vn, vm); }
void CodeGenerator::smax(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 12, vd, vn, vm); }
void CodeGenerator::smax(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 12, vd, vn, vm); }
void CodeGenerator::smax(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 12, vd, vn, vm); }
void CodeGenerator::smax(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 12, vd, vn, vm); }
void CodeGenerator::smax(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 12, vd, vn, vm); }
void CodeGenerator::smax(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 12, vd, vn, vm); }
void CodeGenerator::smin(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 13, vd, vn, vm); }
void CodeGenerator::smin(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 13, vd, vn, vm); }
void CodeGenerator::smin(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 13, vd, vn, vm); }
void CodeGenerator::smin(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 13, vd, vn, vm); }
void CodeGenerator::smin(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 13, vd, vn, vm); }
void CodeGenerator::smin(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 13, vd, vn, vm); }
void CodeGenerator::sabd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 14, vd, vn, vm); }
void CodeGenerator::sabd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 14, vd, vn, vm); }
void CodeGenerator::sabd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 14, vd, vn, vm); }
void CodeGenerator::sabd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 14, vd, vn, vm); }
void CodeGenerator::sabd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 14, vd, vn, vm); }
void CodeGenerator::sabd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 14, vd, vn, vm); }
void CodeGenerator::saba(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 15, vd, vn, vm); }
void CodeGenerator::saba(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 15, vd, vn, vm); }
void CodeGenerator::saba(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 15, vd, vn, vm); }
void CodeGenerator::saba(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 15, vd, vn, vm); }
void CodeGenerator::saba(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 15, vd, vn, vm); }
void CodeGenerator::saba(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 15, vd, vn, vm); }
void CodeGenerator::add(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 16, vd, vn, vm); }
void CodeGenerator::add(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 16, vd, vn, vm); }
void CodeGenerator::add(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 16, vd, vn, vm); }
void CodeGenerator::add(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 16, vd, vn, vm); }
void CodeGenerator::add(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 16, vd, vn, vm); }
void CodeGenerator::add(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 16, vd, vn, vm); }
void CodeGenerator::add(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(0, 16, vd, vn, vm); }
void CodeGenerator::cmtst(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 17, vd, vn, vm); }
void CodeGenerator::cmtst(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 17, vd, vn, vm); }
void CodeGenerator::cmtst(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 17, vd, vn, vm); }
void CodeGenerator::cmtst(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 17, vd, vn, vm); }
void CodeGenerator::cmtst(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 17, vd, vn, vm); }
void CodeGenerator::cmtst(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 17, vd, vn, vm); }
void CodeGenerator::cmtst(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(0, 17, vd, vn, vm); }
void CodeGenerator::mla(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 18, vd, vn, vm); }
void CodeGenerator::mla(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 18, vd, vn, vm); }
void CodeGenerator::mla(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 18, vd, vn, vm); }
void CodeGenerator::mla(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 18, vd, vn, vm); }
void CodeGenerator::mla(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 18, vd, vn, vm); }
void CodeGenerator::mla(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 18, vd, vn, vm); }
void CodeGenerator::mul(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 19, vd, vn, vm); }
void CodeGenerator::mul(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 19, vd, vn, vm); }
void CodeGenerator::mul(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 19, vd, vn, vm); }
void CodeGenerator::mul(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 19, vd, vn, vm); }
void CodeGenerator::mul(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 19, vd, vn, vm); }
void CodeGenerator::mul(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 19, vd, vn, vm); }
void CodeGenerator::smaxp(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 20, vd, vn, vm); }
void CodeGenerator::smaxp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 20, vd, vn, vm); }
void CodeGenerator::smaxp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 20, vd, vn, vm); }
void CodeGenerator::smaxp(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 20, vd, vn, vm); }
void CodeGenerator::smaxp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 20, vd, vn, vm); }
void CodeGenerator::smaxp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 20, vd, vn, vm); }
void CodeGenerator::sminp(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 21, vd, vn, vm); }
void CodeGenerator::sminp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 21, vd, vn, vm); }
void CodeGenerator::sminp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 21, vd, vn, vm); }
void CodeGenerator::sminp(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 21, vd, vn, vm); }
void CodeGenerator::sminp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 21, vd, vn, vm); }
void CodeGenerator::sminp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 21, vd, vn, vm); }
void CodeGenerator::sqdmulh(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 22, vd, vn, vm); }
void CodeGenerator::sqdmulh(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 22, vd, vn, vm); }
void CodeGenerator::sqdmulh(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 22, vd, vn, vm); }
void CodeGenerator::sqdmulh(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 22, vd, vn, vm); }
void CodeGenerator::addp(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(0, 23, vd, vn, vm); }
void CodeGenerator::addp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(0, 23, vd, vn, vm); }
void CodeGenerator::addp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(0, 23, vd, vn, vm); }
void CodeGenerator::addp(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(0, 23, vd, vn, vm); }
void CodeGenerator::addp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(0, 23, vd, vn, vm); }
void CodeGenerator::addp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(0, 23, vd, vn, vm); }
void CodeGenerator::addp(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(0, 23, vd, vn, vm); }
void CodeGenerator::uhadd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 0, vd, vn, vm); }
void CodeGenerator::uhadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 0, vd, vn, vm); }
void CodeGenerator::uhadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 0, vd, vn, vm); }
void CodeGenerator::uhadd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 0, vd, vn, vm); }
void CodeGenerator::uhadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 0, vd, vn, vm); }
void CodeGenerator::uhadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 0, vd, vn, vm); }
void CodeGenerator::uqadd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 1, vd, vn, vm); }
void CodeGenerator::uqadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 1, vd, vn, vm); }
void CodeGenerator::uqadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 1, vd, vn, vm); }
void CodeGenerator::uqadd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 1, vd, vn, vm); }
void CodeGenerator::uqadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 1, vd, vn, vm); }
void CodeGenerator::uqadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 1, vd, vn, vm); }
void CodeGenerator::uqadd(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(1, 1, vd, vn, vm); }
void CodeGenerator::urhadd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 2, vd, vn, vm); }
void CodeGenerator::urhadd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 2, vd, vn, vm); }
void CodeGenerator::urhadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 2, vd, vn, vm); }
void CodeGenerator::urhadd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 2, vd, vn, vm); }
void CodeGenerator::urhadd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 2, vd, vn, vm); }
void CodeGenerator::urhadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 2, vd, vn, vm); }
void CodeGenerator::uhsub(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 4, vd, vn, vm); }
void CodeGenerator::uhsub(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 4, vd, vn, vm); }
void CodeGenerator::uhsub(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 4, vd, vn, vm); }
void CodeGenerator::uhsub(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 4, vd, vn, vm); }
void CodeGenerator::uhsub(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 4, vd, vn, vm); }
void CodeGenerator::uhsub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 4, vd, vn, vm); }
void CodeGenerator::uqsub(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 5, vd, vn, vm); }
void CodeGenerator::uqsub(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 5, vd, vn, vm); }
void CodeGenerator::uqsub(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 5, vd, vn, vm); }
void CodeGenerator::uqsub(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 5, vd, vn, vm); }
void CodeGenerator::uqsub(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 5, vd, vn, vm); }
void CodeGenerator::uqsub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 5, vd, vn, vm); }
void CodeGenerator::uqsub(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(1, 5, vd, vn, vm); }
void CodeGenerator::cmhi(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 6, vd, vn, vm); }
void CodeGenerator::cmhi(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 6, vd, vn, vm); }
void CodeGenerator::cmhi(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 6, vd, vn, vm); }
void CodeGenerator::cmhi(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 6, vd, vn, vm); }
void CodeGenerator::cmhi(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 6, vd, vn, vm); }
void CodeGenerator::cmhi(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 6, vd, vn, vm); }
void CodeGenerator::cmhi(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(1, 6, vd, vn, vm); }
void CodeGenerator::cmhs(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 7, vd, vn, vm); }
void CodeGenerator::cmhs(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 7, vd, vn, vm); }
void CodeGenerator::cmhs(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 7, vd, vn, vm); }
void CodeGenerator::cmhs(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 7, vd, vn, vm); }
void CodeGenerator::cmhs(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 7, vd, vn, vm); }
void CodeGenerator::cmhs(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 7, vd, vn, vm); }
void CodeGenerator::cmhs(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(1, 7, vd, vn, vm); }
void CodeGenerator::ushl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 8, vd, vn, vm); }
void CodeGenerator::ushl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 8, vd, vn, vm); }
void CodeGenerator::ushl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 8, vd, vn, vm); }
void CodeGenerator::ushl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 8, vd, vn, vm); }
void CodeGenerator::ushl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 8, vd, vn, vm); }
void CodeGenerator::ushl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 8, vd, vn, vm); }
void CodeGenerator::ushl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(1, 8, vd, vn, vm); }
void CodeGenerator::uqshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 9, vd, vn, vm); }
void CodeGenerator::uqshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 9, vd, vn, vm); }
void CodeGenerator::uqshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 9, vd, vn, vm); }
void CodeGenerator::uqshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 9, vd, vn, vm); }
void CodeGenerator::uqshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 9, vd, vn, vm); }
void CodeGenerator::uqshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 9, vd, vn, vm); }
void CodeGenerator::uqshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(1, 9, vd, vn, vm); }
void CodeGenerator::urshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 10, vd, vn, vm); }
void CodeGenerator::urshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 10, vd, vn, vm); }
void CodeGenerator::urshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 10, vd, vn, vm); }
void CodeGenerator::urshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 10, vd, vn, vm); }
void CodeGenerator::urshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 10, vd, vn, vm); }
void CodeGenerator::urshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 10, vd, vn, vm); }
void CodeGenerator::urshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(1, 10, vd, vn, vm); }
void CodeGenerator::uqrshl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 11, vd, vn, vm); }
void CodeGenerator::uqrshl(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 11, vd, vn, vm); }
void CodeGenerator::uqrshl(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 11, vd, vn, vm); }
void CodeGenerator::uqrshl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 11, vd, vn, vm); }
void CodeGenerator::uqrshl(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 11, vd, vn, vm); }
void CodeGenerator::uqrshl(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 11, vd, vn, vm); }
void CodeGenerator::uqrshl(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(1, 11, vd, vn, vm); }
void CodeGenerator::umax(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 12, vd, vn, vm); }
void CodeGenerator::umax(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 12, vd, vn, vm); }
void CodeGenerator::umax(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 12, vd, vn, vm); }
void CodeGenerator::umax(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 12, vd, vn, vm); }
void CodeGenerator::umax(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 12, vd, vn, vm); }
void CodeGenerator::umax(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 12, vd, vn, vm); }
void CodeGenerator::umin(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 13, vd, vn, vm); }
void CodeGenerator::umin(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 13, vd, vn, vm); }
void CodeGenerator::umin(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 13, vd, vn, vm); }
void CodeGenerator::umin(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 13, vd, vn, vm); }
void CodeGenerator::umin(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 13, vd, vn, vm); }
void CodeGenerator::umin(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 13, vd, vn, vm); }
void CodeGenerator::uabd(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 14, vd, vn, vm); }
void CodeGenerator::uabd(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 14, vd, vn, vm); }
void CodeGenerator::uabd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 14, vd, vn, vm); }
void CodeGenerator::uabd(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 14, vd, vn, vm); }
void CodeGenerator::uabd(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 14, vd, vn, vm); }
void CodeGenerator::uabd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 14, vd, vn, vm); }
void CodeGenerator::uaba(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 15, vd, vn, vm); }
void CodeGenerator::uaba(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 15, vd, vn, vm); }
void CodeGenerator::uaba(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 15, vd, vn, vm); }
void CodeGenerator::uaba(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 15, vd, vn, vm); }
void CodeGenerator::uaba(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 15, vd, vn, vm); }
void CodeGenerator::uaba(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 15, vd, vn, vm); }
void CodeGenerator::sub(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 16, vd, vn, vm); }
void CodeGenerator::sub(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 16, vd, vn, vm); }
void CodeGenerator::sub(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 16, vd, vn, vm); }
void CodeGenerator::sub(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 16, vd, vn, vm); }
void CodeGenerator::sub(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 16, vd, vn, vm); }
void CodeGenerator::sub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 16, vd, vn, vm); }
void CodeGenerator::sub(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(1, 16, vd, vn, vm); }
void CodeGenerator::cmeq(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 17, vd, vn, vm); }
void CodeGenerator::cmeq(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 17, vd, vn, vm); }
void CodeGenerator::cmeq(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 17, vd, vn, vm); }
void CodeGenerator::cmeq(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 17, vd, vn, vm); }
void CodeGenerator::cmeq(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 17, vd, vn, vm); }
void CodeGenerator::cmeq(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 17, vd, vn, vm); }
void CodeGenerator::cmeq(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3Same(1, 17, vd, vn, vm); }
void CodeGenerator::mls(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 18, vd, vn, vm); }
void CodeGenerator::mls(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 18, vd, vn, vm); }
void CodeGenerator::mls(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 18, vd, vn, vm); }
void CodeGenerator::mls(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 18, vd, vn, vm); }
void CodeGenerator::mls(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 18, vd, vn, vm); }
void CodeGenerator::mls(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 18, vd, vn, vm); }
void CodeGenerator::pmul(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 19, vd, vn, vm); }
void CodeGenerator::pmul(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 19, vd, vn, vm); }
void CodeGenerator::umaxp(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 20, vd, vn, vm); }
void CodeGenerator::umaxp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 20, vd, vn, vm); }
void CodeGenerator::umaxp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 20, vd, vn, vm); }
void CodeGenerator::umaxp(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 20, vd, vn, vm); }
void CodeGenerator::umaxp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 20, vd, vn, vm); }
void CodeGenerator::umaxp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 20, vd, vn, vm); }
void CodeGenerator::uminp(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3Same(1, 21, vd, vn, vm); }
void CodeGenerator::uminp(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 21, vd, vn, vm); }
void CodeGenerator::uminp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 21, vd, vn, vm); }
void CodeGenerator::uminp(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3Same(1, 21, vd, vn, vm); }
void CodeGenerator::uminp(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 21, vd, vn, vm); }
void CodeGenerator::uminp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 21, vd, vn, vm); }
void CodeGenerator::sqrdmulh(const VReg4H &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3Same(1, 22, vd, vn, vm); }
void CodeGenerator::sqrdmulh(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3Same(1, 22, vd, vn, vm); }
void CodeGenerator::sqrdmulh(const VReg8H &vd, const VReg8H &vn, const VReg8H &vm) { AdvSimd3Same(1, 22, vd, vn, vm); }
void CodeGenerator::sqrdmulh(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3Same(1, 22, vd, vn, vm); }
void CodeGenerator::fmaxnm(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(0, 24, vd, vn, vm); }
void CodeGenerator::fmaxnm(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(0, 24, vd, vn, vm); }
void CodeGenerator::fmaxnm(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(0, 24, vd, vn, vm); }
void CodeGenerator::fmla(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(0, 25, vd, vn, vm); }
void CodeGenerator::fmla(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(0, 25, vd, vn, vm); }
void CodeGenerator::fmla(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(0, 25, vd, vn, vm); }
void CodeGenerator::fadd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(0, 26, vd, vn, vm); }
void CodeGenerator::fadd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(0, 26, vd, vn, vm); }
void CodeGenerator::fadd(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(0, 26, vd, vn, vm); }
void CodeGenerator::fmulx(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(0, 27, vd, vn, vm); }
void CodeGenerator::fmulx(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(0, 27, vd, vn, vm); }
void CodeGenerator::fmulx(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(0, 27, vd, vn, vm); }
void CodeGenerator::fcmeq(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(0, 28, vd, vn, vm); }
void CodeGenerator::fcmeq(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(0, 28, vd, vn, vm); }
void CodeGenerator::fcmeq(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(0, 28, vd, vn, vm); }
void CodeGenerator::fmax(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(0, 30, vd, vn, vm); }
void CodeGenerator::fmax(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(0, 30, vd, vn, vm); }
void CodeGenerator::fmax(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(0, 30, vd, vn, vm); }
void CodeGenerator::frecps(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(0, 31, vd, vn, vm); }
void CodeGenerator::frecps(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(0, 31, vd, vn, vm); }
void CodeGenerator::frecps(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(0, 31, vd, vn, vm); }
void CodeGenerator::fmaxnmp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(1, 24, vd, vn, vm); }
void CodeGenerator::fmaxnmp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(1, 24, vd, vn, vm); }
void CodeGenerator::fmaxnmp(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(1, 24, vd, vn, vm); }
void CodeGenerator::faddp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(1, 26, vd, vn, vm); }
void CodeGenerator::faddp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(1, 26, vd, vn, vm); }
void CodeGenerator::faddp(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(1, 26, vd, vn, vm); }
void CodeGenerator::fmul(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(1, 27, vd, vn, vm); }
void CodeGenerator::fmul(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(1, 27, vd, vn, vm); }
void CodeGenerator::fmul(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(1, 27, vd, vn, vm); }
void CodeGenerator::fcmge(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(1, 28, vd, vn, vm); }
void CodeGenerator::fcmge(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(1, 28, vd, vn, vm); }
void CodeGenerator::fcmge(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(1, 28, vd, vn, vm); }
void CodeGenerator::facge(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(1, 29, vd, vn, vm); }
void CodeGenerator::facge(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(1, 29, vd, vn, vm); }
void CodeGenerator::facge(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(1, 29, vd, vn, vm); }
void CodeGenerator::fmaxp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(1, 30, vd, vn, vm); }
void CodeGenerator::fmaxp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(1, 30, vd, vn, vm); }
void CodeGenerator::fmaxp(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(1, 30, vd, vn, vm); }
void CodeGenerator::fdiv(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz0x(1, 31, vd, vn, vm); }
void CodeGenerator::fdiv(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz0x(1, 31, vd, vn, vm); }
void CodeGenerator::fdiv(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz0x(1, 31, vd, vn, vm); }
void CodeGenerator::fminnm(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz1x(0, 24, vd, vn, vm); }
void CodeGenerator::fminnm(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz1x(0, 24, vd, vn, vm); }
void CodeGenerator::fminnm(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz1x(0, 24, vd, vn, vm); }
void CodeGenerator::fmls(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz1x(0, 25, vd, vn, vm); }
void CodeGenerator::fmls(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz1x(0, 25, vd, vn, vm); }
void CodeGenerator::fmls(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz1x(0, 25, vd, vn, vm); }
void CodeGenerator::fsub(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz1x(0, 26, vd, vn, vm); }
void CodeGenerator::fsub(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz1x(0, 26, vd, vn, vm); }
void CodeGenerator::fsub(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz1x(0, 26, vd, vn, vm); }
void CodeGenerator::fmin(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz1x(0, 30, vd, vn, vm); }
void CodeGenerator::fmin(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz1x(0, 30, vd, vn, vm); }
void CodeGenerator::fmin(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz1x(0, 30, vd, vn, vm); }
void CodeGenerator::frsqrts(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz1x(0, 31, vd, vn, vm); }
void CodeGenerator::frsqrts(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz1x(0, 31, vd, vn, vm); }
void CodeGenerator::frsqrts(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz1x(0, 31, vd, vn, vm); }
void CodeGenerator::fminnmp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz1x(1, 24, vd, vn, vm); }
void CodeGenerator::fminnmp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz1x(1, 24, vd, vn, vm); }
void CodeGenerator::fminnmp(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz1x(1, 24, vd, vn, vm); }
void CodeGenerator::fabd(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz1x(1, 26, vd, vn, vm); }
void CodeGenerator::fabd(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz1x(1, 26, vd, vn, vm); }
void CodeGenerator::fabd(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz1x(1, 26, vd, vn, vm); }
void CodeGenerator::fcmgt(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz1x(1, 28, vd, vn, vm); }
void CodeGenerator::fcmgt(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz1x(1, 28, vd, vn, vm); }
void CodeGenerator::fcmgt(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz1x(1, 28, vd, vn, vm); }
void CodeGenerator::facgt(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz1x(1, 29, vd, vn, vm); }
void CodeGenerator::facgt(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz1x(1, 29, vd, vn, vm); }
void CodeGenerator::facgt(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz1x(1, 29, vd, vn, vm); }
void CodeGenerator::fminp(const VReg2S &vd, const VReg2S &vn, const VReg2S &vm) { AdvSimd3SameSz1x(1, 30, vd, vn, vm); }
void CodeGenerator::fminp(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { AdvSimd3SameSz1x(1, 30, vd, vn, vm); }
void CodeGenerator::fminp(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { AdvSimd3SameSz1x(1, 30, vd, vn, vm); }
void CodeGenerator::and_(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3SameSz(0, 0, 3, vd, vn, vm); }
void CodeGenerator::and_(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3SameSz(0, 0, 3, vd, vn, vm); }
void CodeGenerator::fmlal(const VReg2S &vd, const VReg2H &vn, const VReg2H &vm) { AdvSimd3SameSz(0, 0, 29, vd, vn, vm); }
void CodeGenerator::fmlal(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameSz(0, 0, 29, vd, vn, vm); }
void CodeGenerator::bic(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3SameSz(0, 1, 3, vd, vn, vm); }
void CodeGenerator::bic(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3SameSz(0, 1, 3, vd, vn, vm); }
void CodeGenerator::orr(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3SameSz(0, 2, 3, vd, vn, vm); }
void CodeGenerator::orr(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3SameSz(0, 2, 3, vd, vn, vm); }
void CodeGenerator::mov(const VReg8B &vd, const VReg8B &vn) { AdvSimd3SameSz(0, 2, 3, vd, vn, vn); }
void CodeGenerator::mov(const VReg16B &vd, const VReg16B &vn) { AdvSimd3SameSz(0, 2, 3, vd, vn, vn); }
void CodeGenerator::fmlsl(const VReg2S &vd, const VReg2H &vn, const VReg2H &vm) { AdvSimd3SameSz(0, 2, 29, vd, vn, vm); }
void CodeGenerator::fmlsl(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameSz(0, 2, 29, vd, vn, vm); }
void CodeGenerator::orn(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3SameSz(0, 3, 3, vd, vn, vm); }
void CodeGenerator::orn(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3SameSz(0, 3, 3, vd, vn, vm); }
void CodeGenerator::eor(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3SameSz(1, 0, 3, vd, vn, vm); }
void CodeGenerator::eor(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3SameSz(1, 0, 3, vd, vn, vm); }
void CodeGenerator::fmlal2(const VReg2S &vd, const VReg2H &vn, const VReg2H &vm) { AdvSimd3SameSz(1, 0, 25, vd, vn, vm); }
void CodeGenerator::fmlal2(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameSz(1, 0, 25, vd, vn, vm); }
void CodeGenerator::bsl(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3SameSz(1, 1, 3, vd, vn, vm); }
void CodeGenerator::bsl(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3SameSz(1, 1, 3, vd, vn, vm); }
void CodeGenerator::bit(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3SameSz(1, 2, 3, vd, vn, vm); }
void CodeGenerator::bit(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3SameSz(1, 2, 3, vd, vn, vm); }
void CodeGenerator::fmlsl2(const VReg2S &vd, const VReg2H &vn, const VReg2H &vm) { AdvSimd3SameSz(1, 2, 25, vd, vn, vm); }
void CodeGenerator::fmlsl2(const VReg4S &vd, const VReg4H &vn, const VReg4H &vm) { AdvSimd3SameSz(1, 2, 25, vd, vn, vm); }
void CodeGenerator::bif(const VReg8B &vd, const VReg8B &vn, const VReg8B &vm) { AdvSimd3SameSz(1, 3, 3, vd, vn, vm); }
void CodeGenerator::bif(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm) { AdvSimd3SameSz(1, 3, 3, vd, vn, vm); }
void CodeGenerator::movi(const VReg2S &vd, const uint32_t imm8, const ShMod mod, const uint32_t sh) { AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh); }
void CodeGenerator::movi(const VReg4S &vd, const uint32_t imm8, const ShMod mod, const uint32_t sh) { AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh); }
void CodeGenerator::movi(const VReg8B &vd, const uint32_t imm8, const ShMod mod, const uint32_t sh) { AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh); }
void CodeGenerator::movi(const VReg16B &vd, const uint32_t imm8, const ShMod mod, const uint32_t sh) { AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh); }
void CodeGenerator::movi(const VReg4H &vd, const uint32_t imm8, const ShMod mod, const uint32_t sh) { AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh); }
void CodeGenerator::movi(const VReg8H &vd, const uint32_t imm8, const ShMod mod, const uint32_t sh) { AdvSimdModiImmMoviMvni(0, 0, vd, imm8, mod, sh); }
void CodeGenerator::movi(const DReg &vd, const uint64_t imm) { AdvSimdModiImmMoviMvni(1, 0, vd, imm); }
void CodeGenerator::movi(const VReg2D &vd, const uint64_t imm) { AdvSimdModiImmMoviMvni(1, 0, vd, imm); }
void CodeGenerator::mvni(const VReg2S &vd, const uint32_t imm8, const ShMod mod, const uint32_t sh) { AdvSimdModiImmMoviMvni(1, 0, vd, imm8, mod, sh); }
void CodeGenerator::mvni(const VReg4S &vd, const uint32_t imm8, const ShMod mod, const uint32_t sh) { AdvSimdModiImmMoviMvni(1, 0, vd, imm8, mod, sh); }
void CodeGenerator::mvni(const VReg4H &vd, const uint32_t imm8, const ShMod mod, const uint32_t sh) { AdvSimdModiImmMoviMvni(1, 0, vd, imm8, mod, sh); }
void CodeGenerator::mvni(const VReg8H &vd, const uint32_t imm8, const ShMod mod, const uint32_t sh) { AdvSimdModiImmMoviMvni(1, 0, vd, imm8, mod, sh); }
void CodeGenerator::orr(const VReg4H &vd, const uint32_t imm, const ShMod mod, const uint32_t sh) { AdvSimdModiImmOrrBic(0, 0, vd, imm, mod, sh); }
void CodeGenerator::orr(const VReg8H &vd, const uint32_t imm, const ShMod mod, const uint32_t sh) { AdvSimdModiImmOrrBic(0, 0, vd, imm, mod, sh); }
void CodeGenerator::orr(const VReg2S &vd, const uint32_t imm, const ShMod mod, const uint32_t sh) { AdvSimdModiImmOrrBic(0, 0, vd, imm, mod, sh); }
void CodeGenerator::orr(const VReg4S &vd, const uint32_t imm, const ShMod mod, const uint32_t sh) { AdvSimdModiImmOrrBic(0, 0, vd, imm, mod, sh); }
void CodeGenerator::bic(const VReg4H &vd, const uint32_t imm, const ShMod mod, const uint32_t sh) { AdvSimdModiImmOrrBic(1, 0, vd, imm, mod, sh); }
void CodeGenerator::bic(const VReg8H &vd, const uint32_t imm, const ShMod mod, const uint32_t sh) { AdvSimdModiImmOrrBic(1, 0, vd, imm, mod, sh); }
void CodeGenerator::bic(const VReg2S &vd, const uint32_t imm, const ShMod mod, const uint32_t sh) { AdvSimdModiImmOrrBic(1, 0, vd, imm, mod, sh); }
void CodeGenerator::bic(const VReg4S &vd, const uint32_t imm, const ShMod mod, const uint32_t sh) { AdvSimdModiImmOrrBic(1, 0, vd, imm, mod, sh); }
void CodeGenerator::fmov(const VReg2S &vd, const double imm) { AdvSimdModiImmFmov(0, 0, vd, imm); }
void CodeGenerator::fmov(const VReg4S &vd, const double imm) { AdvSimdModiImmFmov(0, 0, vd, imm); }
void CodeGenerator::fmov(const VReg4H &vd, const double imm) { AdvSimdModiImmFmov(0, 1, vd, imm); }
void CodeGenerator::fmov(const VReg8H &vd, const double imm) { AdvSimdModiImmFmov(0, 1, vd, imm); }
void CodeGenerator::fmov(const VReg2D &vd, const double imm) { AdvSimdModiImmFmov(1, 0, vd, imm); }
void CodeGenerator::sshr(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(0, 0, vd, vn, sh); }
void CodeGenerator::sshr(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(0, 0, vd, vn, sh); }
void CodeGenerator::sshr(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(0, 0, vd, vn, sh); }
void CodeGenerator::sshr(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(0, 0, vd, vn, sh); }
void CodeGenerator::sshr(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 0, vd, vn, sh); }
void CodeGenerator::sshr(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 0, vd, vn, sh); }
void CodeGenerator::sshr(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 0, vd, vn, sh); }
void CodeGenerator::ssra(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(0, 2, vd, vn, sh); }
void CodeGenerator::ssra(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(0, 2, vd, vn, sh); }
void CodeGenerator::ssra(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(0, 2, vd, vn, sh); }
void CodeGenerator::ssra(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(0, 2, vd, vn, sh); }
void CodeGenerator::ssra(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 2, vd, vn, sh); }
void CodeGenerator::ssra(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 2, vd, vn, sh); }
void CodeGenerator::ssra(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 2, vd, vn, sh); }
void CodeGenerator::srshr(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(0, 4, vd, vn, sh); }
void CodeGenerator::srshr(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(0, 4, vd, vn, sh); }
void CodeGenerator::srshr(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(0, 4, vd, vn, sh); }
void CodeGenerator::srshr(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(0, 4, vd, vn, sh); }
void CodeGenerator::srshr(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 4, vd, vn, sh); }
void CodeGenerator::srshr(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 4, vd, vn, sh); }
void CodeGenerator::srshr(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 4, vd, vn, sh); }
void CodeGenerator::srsra(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(0, 6, vd, vn, sh); }
void CodeGenerator::srsra(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(0, 6, vd, vn, sh); }
void CodeGenerator::srsra(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(0, 6, vd, vn, sh); }
void CodeGenerator::srsra(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(0, 6, vd, vn, sh); }
void CodeGenerator::srsra(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 6, vd, vn, sh); }
void CodeGenerator::srsra(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 6, vd, vn, sh); }
void CodeGenerator::srsra(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 6, vd, vn, sh); }
void CodeGenerator::shl(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(0, 10, vd, vn, sh); }
void CodeGenerator::shl(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(0, 10, vd, vn, sh); }
void CodeGenerator::shl(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(0, 10, vd, vn, sh); }
void CodeGenerator::shl(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(0, 10, vd, vn, sh); }
void CodeGenerator::shl(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 10, vd, vn, sh); }
void CodeGenerator::shl(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 10, vd, vn, sh); }
void CodeGenerator::shl(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 10, vd, vn, sh); }
void CodeGenerator::sqshl(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(0, 14, vd, vn, sh); }
void CodeGenerator::sqshl(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(0, 14, vd, vn, sh); }
void CodeGenerator::sqshl(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(0, 14, vd, vn, sh); }
void CodeGenerator::sqshl(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(0, 14, vd, vn, sh); }
void CodeGenerator::sqshl(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 14, vd, vn, sh); }
void CodeGenerator::sqshl(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 14, vd, vn, sh); }
void CodeGenerator::sqshl(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 14, vd, vn, sh); }
void CodeGenerator::shrn(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 16, vd, vn, sh); }
void CodeGenerator::shrn(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 16, vd, vn, sh); }
void CodeGenerator::shrn(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 16, vd, vn, sh); }
void CodeGenerator::shrn2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 16, vd, vn, sh); }
void CodeGenerator::shrn2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 16, vd, vn, sh); }
void CodeGenerator::shrn2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 16, vd, vn, sh); }
void CodeGenerator::rshrn(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 17, vd, vn, sh); }
void CodeGenerator::rshrn(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 17, vd, vn, sh); }
void CodeGenerator::rshrn(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 17, vd, vn, sh); }
void CodeGenerator::rshrn2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 17, vd, vn, sh); }
void CodeGenerator::rshrn2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 17, vd, vn, sh); }
void CodeGenerator::rshrn2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 17, vd, vn, sh); }
void CodeGenerator::sqshrn(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 18, vd, vn, sh); }
void CodeGenerator::sqshrn(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 18, vd, vn, sh); }
void CodeGenerator::sqshrn(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 18, vd, vn, sh); }
void CodeGenerator::sqshrn2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 18, vd, vn, sh); }
void CodeGenerator::sqshrn2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 18, vd, vn, sh); }
void CodeGenerator::sqshrn2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 18, vd, vn, sh); }
void CodeGenerator::sqrshrn(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 19, vd, vn, sh); }
void CodeGenerator::sqrshrn(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 19, vd, vn, sh); }
void CodeGenerator::sqrshrn(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 19, vd, vn, sh); }
void CodeGenerator::sqrshrn2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 19, vd, vn, sh); }
void CodeGenerator::sqrshrn2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 19, vd, vn, sh); }
void CodeGenerator::sqrshrn2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 19, vd, vn, sh); }
void CodeGenerator::sshll(const VReg8H &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(0, 20, vd, vn, sh); }
void CodeGenerator::sshll(const VReg4S &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(0, 20, vd, vn, sh); }
void CodeGenerator::sshll(const VReg2D &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(0, 20, vd, vn, sh); }
void CodeGenerator::sshll2(const VReg8H &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(0, 20, vd, vn, sh); }
void CodeGenerator::sshll2(const VReg4S &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 20, vd, vn, sh); }
void CodeGenerator::sshll2(const VReg2D &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 20, vd, vn, sh); }
void CodeGenerator::sxtl(const VReg8H &vd, const VReg8B &vn) { AdvSimdShImm(0, 20, vd, vn, 0); }
void CodeGenerator::sxtl(const VReg4S &vd, const VReg4H &vn) { AdvSimdShImm(0, 20, vd, vn, 0); }
void CodeGenerator::sxtl(const VReg2D &vd, const VReg2S &vn) { AdvSimdShImm(0, 20, vd, vn, 0); }
void CodeGenerator::sxtl2(const VReg8H &vd, const VReg16B &vn) { AdvSimdShImm(0, 20, vd, vn, 0); }
void CodeGenerator::sxtl2(const VReg4S &vd, const VReg8H &vn) { AdvSimdShImm(0, 20, vd, vn, 0); }
void CodeGenerator::sxtl2(const VReg2D &vd, const VReg4S &vn) { AdvSimdShImm(0, 20, vd, vn, 0); }
void CodeGenerator::scvtf(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(0, 28, vd, vn, sh); }
void CodeGenerator::scvtf(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(0, 28, vd, vn, sh); }
void CodeGenerator::scvtf(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 28, vd, vn, sh); }
void CodeGenerator::scvtf(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 28, vd, vn, sh); }
void CodeGenerator::scvtf(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 28, vd, vn, sh); }
void CodeGenerator::fcvtzs(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(0, 31, vd, vn, sh); }
void CodeGenerator::fcvtzs(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(0, 31, vd, vn, sh); }
void CodeGenerator::fcvtzs(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(0, 31, vd, vn, sh); }
void CodeGenerator::fcvtzs(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(0, 31, vd, vn, sh); }
void CodeGenerator::fcvtzs(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(0, 31, vd, vn, sh); }
void CodeGenerator::ushr(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(1, 0, vd, vn, sh); }
void CodeGenerator::ushr(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(1, 0, vd, vn, sh); }
void CodeGenerator::ushr(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(1, 0, vd, vn, sh); }
void CodeGenerator::ushr(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(1, 0, vd, vn, sh); }
void CodeGenerator::ushr(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 0, vd, vn, sh); }
void CodeGenerator::ushr(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 0, vd, vn, sh); }
void CodeGenerator::ushr(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 0, vd, vn, sh); }
void CodeGenerator::usra(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(1, 2, vd, vn, sh); }
void CodeGenerator::usra(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(1, 2, vd, vn, sh); }
void CodeGenerator::usra(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(1, 2, vd, vn, sh); }
void CodeGenerator::usra(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(1, 2, vd, vn, sh); }
void CodeGenerator::usra(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 2, vd, vn, sh); }
void CodeGenerator::usra(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 2, vd, vn, sh); }
void CodeGenerator::usra(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 2, vd, vn, sh); }
void CodeGenerator::urshr(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(1, 4, vd, vn, sh); }
void CodeGenerator::urshr(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(1, 4, vd, vn, sh); }
void CodeGenerator::urshr(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(1, 4, vd, vn, sh); }
void CodeGenerator::urshr(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(1, 4, vd, vn, sh); }
void CodeGenerator::urshr(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 4, vd, vn, sh); }
void CodeGenerator::urshr(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 4, vd, vn, sh); }
void CodeGenerator::urshr(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 4, vd, vn, sh); }
void CodeGenerator::ursra(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(1, 6, vd, vn, sh); }
void CodeGenerator::ursra(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(1, 6, vd, vn, sh); }
void CodeGenerator::ursra(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(1, 6, vd, vn, sh); }
void CodeGenerator::ursra(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(1, 6, vd, vn, sh); }
void CodeGenerator::ursra(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 6, vd, vn, sh); }
void CodeGenerator::ursra(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 6, vd, vn, sh); }
void CodeGenerator::ursra(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 6, vd, vn, sh); }
void CodeGenerator::sri(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(1, 8, vd, vn, sh); }
void CodeGenerator::sri(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(1, 8, vd, vn, sh); }
void CodeGenerator::sri(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(1, 8, vd, vn, sh); }
void CodeGenerator::sri(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(1, 8, vd, vn, sh); }
void CodeGenerator::sri(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 8, vd, vn, sh); }
void CodeGenerator::sri(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 8, vd, vn, sh); }
void CodeGenerator::sri(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 8, vd, vn, sh); }
void CodeGenerator::sli(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(1, 10, vd, vn, sh); }
void CodeGenerator::sli(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(1, 10, vd, vn, sh); }
void CodeGenerator::sli(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(1, 10, vd, vn, sh); }
void CodeGenerator::sli(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(1, 10, vd, vn, sh); }
void CodeGenerator::sli(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 10, vd, vn, sh); }
void CodeGenerator::sli(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 10, vd, vn, sh); }
void CodeGenerator::sli(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 10, vd, vn, sh); }
void CodeGenerator::sqshlu(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(1, 12, vd, vn, sh); }
void CodeGenerator::sqshlu(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(1, 12, vd, vn, sh); }
void CodeGenerator::sqshlu(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(1, 12, vd, vn, sh); }
void CodeGenerator::sqshlu(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(1, 12, vd, vn, sh); }
void CodeGenerator::sqshlu(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 12, vd, vn, sh); }
void CodeGenerator::sqshlu(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 12, vd, vn, sh); }
void CodeGenerator::sqshlu(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 12, vd, vn, sh); }
void CodeGenerator::uqshl(const VReg8B &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(1, 14, vd, vn, sh); }
void CodeGenerator::uqshl(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(1, 14, vd, vn, sh); }
void CodeGenerator::uqshl(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(1, 14, vd, vn, sh); }
void CodeGenerator::uqshl(const VReg16B &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(1, 14, vd, vn, sh); }
void CodeGenerator::uqshl(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 14, vd, vn, sh); }
void CodeGenerator::uqshl(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 14, vd, vn, sh); }
void CodeGenerator::uqshl(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 14, vd, vn, sh); }
void CodeGenerator::sqshrun(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 16, vd, vn, sh); }
void CodeGenerator::sqshrun(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 16, vd, vn, sh); }
void CodeGenerator::sqshrun(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 16, vd, vn, sh); }
void CodeGenerator::sqshrun2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 16, vd, vn, sh); }
void CodeGenerator::sqshrun2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 16, vd, vn, sh); }
void CodeGenerator::sqshrun2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 16, vd, vn, sh); }
void CodeGenerator::sqrshrun(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 17, vd, vn, sh); }
void CodeGenerator::sqrshrun(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 17, vd, vn, sh); }
void CodeGenerator::sqrshrun(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 17, vd, vn, sh); }
void CodeGenerator::sqrshrun2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 17, vd, vn, sh); }
void CodeGenerator::sqrshrun2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 17, vd, vn, sh); }
void CodeGenerator::sqrshrun2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 17, vd, vn, sh); }
void CodeGenerator::uqshrn(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 18, vd, vn, sh); }
void CodeGenerator::uqshrn(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 18, vd, vn, sh); }
void CodeGenerator::uqshrn(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 18, vd, vn, sh); }
void CodeGenerator::uqshrn2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 18, vd, vn, sh); }
void CodeGenerator::uqshrn2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 18, vd, vn, sh); }
void CodeGenerator::uqshrn2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 18, vd, vn, sh); }
void CodeGenerator::uqrshrn(const VReg8B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 19, vd, vn, sh); }
void CodeGenerator::uqrshrn(const VReg4H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 19, vd, vn, sh); }
void CodeGenerator::uqrshrn(const VReg2S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 19, vd, vn, sh); }
void CodeGenerator::uqrshrn2(const VReg16B &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 19, vd, vn, sh); }
void CodeGenerator::uqrshrn2(const VReg8H &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 19, vd, vn, sh); }
void CodeGenerator::uqrshrn2(const VReg4S &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 19, vd, vn, sh); }
void CodeGenerator::ushll(const VReg8H &vd, const VReg8B &vn, const uint32_t sh) { AdvSimdShImm(1, 20, vd, vn, sh); }
void CodeGenerator::ushll(const VReg4S &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(1, 20, vd, vn, sh); }
void CodeGenerator::ushll(const VReg2D &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(1, 20, vd, vn, sh); }
void CodeGenerator::ushll2(const VReg8H &vd, const VReg16B &vn, const uint32_t sh) { AdvSimdShImm(1, 20, vd, vn, sh); }
void CodeGenerator::ushll2(const VReg4S &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 20, vd, vn, sh); }
void CodeGenerator::ushll2(const VReg2D &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 20, vd, vn, sh); }
void CodeGenerator::uxtl(const VReg8H &vd, const VReg8B &vn) { AdvSimdShImm(1, 20, vd, vn, 0); }
void CodeGenerator::uxtl(const VReg4S &vd, const VReg4H &vn) { AdvSimdShImm(1, 20, vd, vn, 0); }
void CodeGenerator::uxtl(const VReg2D &vd, const VReg2S &vn) { AdvSimdShImm(1, 20, vd, vn, 0); }
void CodeGenerator::uxtl2(const VReg8H &vd, const VReg16B &vn) { AdvSimdShImm(1, 20, vd, vn, 0); }
void CodeGenerator::uxtl2(const VReg4S &vd, const VReg8H &vn) { AdvSimdShImm(1, 20, vd, vn, 0); }
void CodeGenerator::uxtl2(const VReg2D &vd, const VReg4S &vn) { AdvSimdShImm(1, 20, vd, vn, 0); }
void CodeGenerator::ucvtf(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(1, 28, vd, vn, sh); }
void CodeGenerator::ucvtf(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(1, 28, vd, vn, sh); }
void CodeGenerator::ucvtf(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 28, vd, vn, sh); }
void CodeGenerator::ucvtf(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 28, vd, vn, sh); }
void CodeGenerator::ucvtf(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 28, vd, vn, sh); }
void CodeGenerator::fcvtzu(const VReg4H &vd, const VReg4H &vn, const uint32_t sh) { AdvSimdShImm(1, 31, vd, vn, sh); }
void CodeGenerator::fcvtzu(const VReg2S &vd, const VReg2S &vn, const uint32_t sh) { AdvSimdShImm(1, 31, vd, vn, sh); }
void CodeGenerator::fcvtzu(const VReg8H &vd, const VReg8H &vn, const uint32_t sh) { AdvSimdShImm(1, 31, vd, vn, sh); }
void CodeGenerator::fcvtzu(const VReg4S &vd, const VReg4S &vn, const uint32_t sh) { AdvSimdShImm(1, 31, vd, vn, sh); }
void CodeGenerator::fcvtzu(const VReg2D &vd, const VReg2D &vn, const uint32_t sh) { AdvSimdShImm(1, 31, vd, vn, sh); }
void CodeGenerator::smlal(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 2, vd, vn, vm); }
void CodeGenerator::smlal(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 2, vd, vn, vm); }
void CodeGenerator::smlal2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 2, vd, vn, vm); }
void CodeGenerator::smlal2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 2, vd, vn, vm); }
void CodeGenerator::sqdmlal(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 3, vd, vn, vm); }
void CodeGenerator::sqdmlal(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 3, vd, vn, vm); }
void CodeGenerator::sqdmlal2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 3, vd, vn, vm); }
void CodeGenerator::sqdmlal2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 3, vd, vn, vm); }
void CodeGenerator::smlsl(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 6, vd, vn, vm); }
void CodeGenerator::smlsl(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 6, vd, vn, vm); }
void CodeGenerator::smlsl2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 6, vd, vn, vm); }
void CodeGenerator::smlsl2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 6, vd, vn, vm); }
void CodeGenerator::sqdmlsl(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 7, vd, vn, vm); }
void CodeGenerator::sqdmlsl(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 7, vd, vn, vm); }
void CodeGenerator::sqdmlsl2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 7, vd, vn, vm); }
void CodeGenerator::sqdmlsl2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 7, vd, vn, vm); }
void CodeGenerator::mul(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 8, vd, vn, vm); }
void CodeGenerator::mul(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 8, vd, vn, vm); }
void CodeGenerator::mul(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 8, vd, vn, vm); }
void CodeGenerator::mul(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 8, vd, vn, vm); }
void CodeGenerator::smull(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 10, vd, vn, vm); }
void CodeGenerator::smull(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 10, vd, vn, vm); }
void CodeGenerator::smull2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 10, vd, vn, vm); }
void CodeGenerator::smull2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 10, vd, vn, vm); }
void CodeGenerator::sqdmull(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 11, vd, vn, vm); }
void CodeGenerator::sqdmull(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 11, vd, vn, vm); }
void CodeGenerator::sqdmull2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 11, vd, vn, vm); }
void CodeGenerator::sqdmull2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 11, vd, vn, vm); }
void CodeGenerator::sqdmulh(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 12, vd, vn, vm); }
void CodeGenerator::sqdmulh(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 12, vd, vn, vm); }
void CodeGenerator::sqdmulh(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 12, vd, vn, vm); }
void CodeGenerator::sqdmulh(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 12, vd, vn, vm); }
void CodeGenerator::sqrdmulh(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 13, vd, vn, vm); }
void CodeGenerator::sqrdmulh(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 13, vd, vn, vm); }
void CodeGenerator::sqrdmulh(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(0, 13, vd, vn, vm); }
void CodeGenerator::sqrdmulh(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(0, 13, vd, vn, vm); }
void CodeGenerator::sdot(const VReg2S &vd, const VReg8B &vn, const VRegBElem &vm) { AdvSimdVecXindElem(0, 14, vd, vn, vm); }
void CodeGenerator::sdot(const VReg4S &vd, const VReg16B &vn, const VRegBElem &vm) { AdvSimdVecXindElem(0, 14, vd, vn, vm); }
void CodeGenerator::mla(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 0, vd, vn, vm); }
void CodeGenerator::mla(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 0, vd, vn, vm); }
void CodeGenerator::mla(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 0, vd, vn, vm); }
void CodeGenerator::mla(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 0, vd, vn, vm); }
void CodeGenerator::umlal(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 2, vd, vn, vm); }
void CodeGenerator::umlal(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 2, vd, vn, vm); }
void CodeGenerator::umlal2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 2, vd, vn, vm); }
void CodeGenerator::umlal2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 2, vd, vn, vm); }
void CodeGenerator::mls(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 4, vd, vn, vm); }
void CodeGenerator::mls(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 4, vd, vn, vm); }
void CodeGenerator::mls(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 4, vd, vn, vm); }
void CodeGenerator::mls(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 4, vd, vn, vm); }
void CodeGenerator::umlsl(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 6, vd, vn, vm); }
void CodeGenerator::umlsl(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 6, vd, vn, vm); }
void CodeGenerator::umlsl2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 6, vd, vn, vm); }
void CodeGenerator::umlsl2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 6, vd, vn, vm); }
void CodeGenerator::umull(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 10, vd, vn, vm); }
void CodeGenerator::umull(const VReg2D &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 10, vd, vn, vm); }
void CodeGenerator::umull2(const VReg4S &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 10, vd, vn, vm); }
void CodeGenerator::umull2(const VReg2D &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 10, vd, vn, vm); }
void CodeGenerator::sqrdmlah(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 13, vd, vn, vm); }
void CodeGenerator::sqrdmlah(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 13, vd, vn, vm); }
void CodeGenerator::sqrdmlah(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 13, vd, vn, vm); }
void CodeGenerator::sqrdmlah(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 13, vd, vn, vm); }
void CodeGenerator::udot(const VReg2S &vd, const VReg8B &vn, const VRegBElem &vm) { AdvSimdVecXindElem(1, 14, vd, vn, vm); }
void CodeGenerator::udot(const VReg4S &vd, const VReg16B &vn, const VRegBElem &vm) { AdvSimdVecXindElem(1, 14, vd, vn, vm); }
void CodeGenerator::sqrdmlsh(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 15, vd, vn, vm); }
void CodeGenerator::sqrdmlsh(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 15, vd, vn, vm); }
void CodeGenerator::sqrdmlsh(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElem(1, 15, vd, vn, vm); }
void CodeGenerator::sqrdmlsh(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElem(1, 15, vd, vn, vm); }
void CodeGenerator::fcmla(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm, const uint32_t rotate) { AdvSimdVecXindElem(1, 1, vd, vn, vm, rotate); }
void CodeGenerator::fcmla(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm, const uint32_t rotate) { AdvSimdVecXindElem(1, 1, vd, vn, vm, rotate); }
void CodeGenerator::fcmla(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm, const uint32_t rotate) { AdvSimdVecXindElem(1, 1, vd, vn, vm, rotate); }
void CodeGenerator::fmla(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(0, 0, 1, vd, vn, vm); }
void CodeGenerator::fmla(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(0, 0, 1, vd, vn, vm); }
void CodeGenerator::fmls(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(0, 0, 5, vd, vn, vm); }
void CodeGenerator::fmls(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(0, 0, 5, vd, vn, vm); }
void CodeGenerator::fmul(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(0, 0, 9, vd, vn, vm); }
void CodeGenerator::fmul(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(0, 0, 9, vd, vn, vm); }
void CodeGenerator::fmla(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElemSz(0, 2, 1, vd, vn, vm); }
void CodeGenerator::fmla(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElemSz(0, 2, 1, vd, vn, vm); }
void CodeGenerator::fmls(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElemSz(0, 2, 5, vd, vn, vm); }
void CodeGenerator::fmls(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElemSz(0, 2, 5, vd, vn, vm); }
void CodeGenerator::fmul(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElemSz(0, 2, 9, vd, vn, vm); }
void CodeGenerator::fmul(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElemSz(0, 2, 9, vd, vn, vm); }
void CodeGenerator::fmla(const VReg2D &vd, const VReg2D &vn, const VRegDElem &vm) { AdvSimdVecXindElemSz(0, 3, 1, vd, vn, vm); }
void CodeGenerator::fmls(const VReg2D &vd, const VReg2D &vn, const VRegDElem &vm) { AdvSimdVecXindElemSz(0, 3, 5, vd, vn, vm); }
void CodeGenerator::fmul(const VReg2D &vd, const VReg2D &vn, const VRegDElem &vm) { AdvSimdVecXindElemSz(0, 3, 9, vd, vn, vm); }
void CodeGenerator::fmlal(const VReg2S &vd, const VReg2H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(0, 2, 0, vd, vn, vm); }
void CodeGenerator::fmlal(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(0, 2, 0, vd, vn, vm); }
void CodeGenerator::fmlsl(const VReg2S &vd, const VReg2H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(0, 2, 4, vd, vn, vm); }
void CodeGenerator::fmlsl(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(0, 2, 4, vd, vn, vm); }
void CodeGenerator::fmulx(const VReg4H &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(1, 0, 9, vd, vn, vm); }
void CodeGenerator::fmulx(const VReg8H &vd, const VReg8H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(1, 0, 9, vd, vn, vm); }
void CodeGenerator::fmulx(const VReg2S &vd, const VReg2S &vn, const VRegSElem &vm) { AdvSimdVecXindElemSz(1, 2, 9, vd, vn, vm); }
void CodeGenerator::fmulx(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { AdvSimdVecXindElemSz(1, 2, 9, vd, vn, vm); }
void CodeGenerator::fmulx(const VReg2D &vd, const VReg2D &vn, const VRegDElem &vm) { AdvSimdVecXindElemSz(1, 3, 9, vd, vn, vm); }
void CodeGenerator::fmlal2(const VReg2S &vd, const VReg2H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(1, 2, 8, vd, vn, vm); }
void CodeGenerator::fmlal2(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(1, 2, 8, vd, vn, vm); }
void CodeGenerator::fmlsl2(const VReg2S &vd, const VReg2H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(1, 2, 12, vd, vn, vm); }
void CodeGenerator::fmlsl2(const VReg4S &vd, const VReg4H &vn, const VRegHElem &vm) { AdvSimdVecXindElemSz(1, 2, 12, vd, vn, vm); }
void CodeGenerator::sm3tt1a(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { Crypto3RegImm2(0, vd, vn, vm); }
void CodeGenerator::sm3tt1b(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { Crypto3RegImm2(1, vd, vn, vm); }
void CodeGenerator::sm3tt2a(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { Crypto3RegImm2(2, vd, vn, vm); }
void CodeGenerator::sm3tt2b(const VReg4S &vd, const VReg4S &vn, const VRegSElem &vm) { Crypto3RegImm2(3, vd, vn, vm); }
void CodeGenerator::sha512h(const QReg &vd, const QReg &vn, const VReg2D &vm) { Crypto3RegSHA512(0, 0, vd, vn, vm); }
void CodeGenerator::sha512h2(const QReg &vd, const QReg &vn, const VReg2D &vm) { Crypto3RegSHA512(0, 1, vd, vn, vm); }
void CodeGenerator::sha512su1(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { Crypto3RegSHA512(0, 2, vd, vn, vm); }
void CodeGenerator::rax1(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm) { Crypto3RegSHA512(0, 3, vd, vn, vm); }
void CodeGenerator::sm3partw1(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { Crypto3RegSHA512(1, 0, vd, vn, vm); }
void CodeGenerator::sm3partw2(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { Crypto3RegSHA512(1, 1, vd, vn, vm); }
void CodeGenerator::sm4ekey(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm) { Crypto3RegSHA512(1, 2, vd, vn, vm); }
void CodeGenerator::xar(const VReg2D &vd, const VReg2D &vn, const VReg2D &vm, const uint32_t imm6) { CryptoSHA(vd, vn, vm, imm6); }
void CodeGenerator::eor3(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm, const VReg16B &va) { Crypto4Reg(0, vd, vn, vm, va); }
void CodeGenerator::bcax(const VReg16B &vd, const VReg16B &vn, const VReg16B &vm, const VReg16B &va) { Crypto4Reg(1, vd, vn, vm, va); }
void CodeGenerator::sm3ss1(const VReg4S &vd, const VReg4S &vn, const VReg4S &vm, const VReg4S &va) { Crypto4Reg(2, vd, vn, vm, va); }
void CodeGenerator::sha512su0(const VReg2D &vd, const VReg2D &vn) { Crypto2RegSHA512(0, vd, vn); }
void CodeGenerator::sm4e(const VReg4S &vd, const VReg4S &vn) { Crypto2RegSHA512(1, vd, vn); }
void CodeGenerator::scvtf(const SReg &d, const WReg &n, const uint32_t fbits) { ConversionFpFix(0, 0, 0, 2, d, n, fbits); }
void CodeGenerator::scvtf(const SReg &d, const XReg &n, const uint32_t fbits) { ConversionFpFix(0, 0, 0, 2, d, n, fbits); }
void CodeGenerator::ucvtf(const SReg &d, const WReg &n, const uint32_t fbits) { ConversionFpFix(0, 0, 0, 3, d, n, fbits); }
void CodeGenerator::ucvtf(const SReg &d, const XReg &n, const uint32_t fbits) { ConversionFpFix(0, 0, 0, 3, d, n, fbits); }
void CodeGenerator::fcvtzs(const WReg &d, const SReg &n, const uint32_t fbits) { ConversionFpFix(0, 0, 3, 0, d, n, fbits); }
void CodeGenerator::fcvtzs(const XReg &d, const SReg &n, const uint32_t fbits) { ConversionFpFix(0, 0, 3, 0, d, n, fbits); }
void CodeGenerator::fcvtzu(const WReg &d, const SReg &n, const uint32_t fbits) { ConversionFpFix(0, 0, 3, 1, d, n, fbits); }
void CodeGenerator::fcvtzu(const XReg &d, const SReg &n, const uint32_t fbits) { ConversionFpFix(0, 0, 3, 1, d, n, fbits); }
void CodeGenerator::scvtf(const DReg &d, const WReg &n, const uint32_t fbits) { ConversionFpFix(0, 1, 0, 2, d, n, fbits); }
void CodeGenerator::scvtf(const DReg &d, const XReg &n, const uint32_t fbits) { ConversionFpFix(0, 1, 0, 2, d, n, fbits); }
void CodeGenerator::ucvtf(const DReg &d, const WReg &n, const uint32_t fbits) { ConversionFpFix(0, 1, 0, 3, d, n, fbits); }
void CodeGenerator::ucvtf(const DReg &d, const XReg &n, const uint32_t fbits) { ConversionFpFix(0, 1, 0, 3, d, n, fbits); }
void CodeGenerator::fcvtzs(const WReg &d, const DReg &n, const uint32_t fbits) { ConversionFpFix(0, 1, 3, 0, d, n, fbits); }
void CodeGenerator::fcvtzs(const XReg &d, const DReg &n, const uint32_t fbits) { ConversionFpFix(0, 1, 3, 0, d, n, fbits); }
void CodeGenerator::fcvtzu(const WReg &d, const DReg &n, const uint32_t fbits) { ConversionFpFix(0, 1, 3, 1, d, n, fbits); }
void CodeGenerator::fcvtzu(const XReg &d, const DReg &n, const uint32_t fbits) { ConversionFpFix(0, 1, 3, 1, d, n, fbits); }
void CodeGenerator::scvtf(const HReg &d, const WReg &n, const uint32_t fbits) { ConversionFpFix(0, 3, 0, 2, d, n, fbits); }
void CodeGenerator::scvtf(const HReg &d, const XReg &n, const uint32_t fbits) { ConversionFpFix(0, 3, 0, 2, d, n, fbits); }
void CodeGenerator::ucvtf(const HReg &d, const WReg &n, const uint32_t fbits) { ConversionFpFix(0, 3, 0, 3, d, n, fbits); }
void CodeGenerator::ucvtf(const HReg &d, const XReg &n, const uint32_t fbits) { ConversionFpFix(0, 3, 0, 3, d, n, fbits); }
void CodeGenerator::fcvtzs(const WReg &d, const HReg &n, const uint32_t fbits) { ConversionFpFix(0, 3, 3, 0, d, n, fbits); }
void CodeGenerator::fcvtzs(const XReg &d, const HReg &n, const uint32_t fbits) { ConversionFpFix(0, 3, 3, 0, d, n, fbits); }
void CodeGenerator::fcvtzu(const WReg &d, const HReg &n, const uint32_t fbits) { ConversionFpFix(0, 3, 3, 1, d, n, fbits); }
void CodeGenerator::fcvtzu(const XReg &d, const HReg &n, const uint32_t fbits) { ConversionFpFix(0, 3, 3, 1, d, n, fbits); }
void CodeGenerator::fcvtns(const WReg &d, const SReg &n) { ConversionFpInt(0, 0, 0, 0, 0, d, n); }
void CodeGenerator::fcvtnu(const WReg &d, const SReg &n) { ConversionFpInt(0, 0, 0, 0, 1, d, n); }
void CodeGenerator::fcvtas(const WReg &d, const SReg &n) { ConversionFpInt(0, 0, 0, 0, 4, d, n); }
void CodeGenerator::fcvtau(const WReg &d, const SReg &n) { ConversionFpInt(0, 0, 0, 0, 5, d, n); }
void CodeGenerator::fmov(const WReg &d, const SReg &n) { ConversionFpInt(0, 0, 0, 0, 6, d, n); }
void CodeGenerator::fcvtps(const WReg &d, const SReg &n) { ConversionFpInt(0, 0, 0, 1, 0, d, n); }
void CodeGenerator::fcvtpu(const WReg &d, const SReg &n) { ConversionFpInt(0, 0, 0, 1, 1, d, n); }
void CodeGenerator::fcvtms(const WReg &d, const SReg &n) { ConversionFpInt(0, 0, 0, 2, 0, d, n); }
void CodeGenerator::fcvtmu(const WReg &d, const SReg &n) { ConversionFpInt(0, 0, 0, 2, 1, d, n); }
void CodeGenerator::fcvtzs(const WReg &d, const SReg &n) { ConversionFpInt(0, 0, 0, 3, 0, d, n); }
void CodeGenerator::fcvtzu(const WReg &d, const SReg &n) { ConversionFpInt(0, 0, 0, 3, 1, d, n); }
void CodeGenerator::scvtf(const SReg &d, const WReg &n) { ConversionFpInt(0, 0, 0, 0, 2, d, n); }
void CodeGenerator::ucvtf(const SReg &d, const WReg &n) { ConversionFpInt(0, 0, 0, 0, 3, d, n); }
void CodeGenerator::fmov(const SReg &d, const WReg &n) { ConversionFpInt(0, 0, 0, 0, 7, d, n); }
void CodeGenerator::fcvtns(const WReg &d, const DReg &n) { ConversionFpInt(0, 0, 1, 0, 0, d, n); }
void CodeGenerator::fcvtnu(const WReg &d, const DReg &n) { ConversionFpInt(0, 0, 1, 0, 1, d, n); }
void CodeGenerator::fcvtas(const WReg &d, const DReg &n) { ConversionFpInt(0, 0, 1, 0, 4, d, n); }
void CodeGenerator::fcvtau(const WReg &d, const DReg &n) { ConversionFpInt(0, 0, 1, 0, 5, d, n); }
void CodeGenerator::fcvtps(const WReg &d, const DReg &n) { ConversionFpInt(0, 0, 1, 1, 0, d, n); }
void CodeGenerator::fcvtpu(const WReg &d, const DReg &n) { ConversionFpInt(0, 0, 1, 1, 1, d, n); }
void CodeGenerator::fcvtms(const WReg &d, const DReg &n) { ConversionFpInt(0, 0, 1, 2, 0, d, n); }
void CodeGenerator::fcvtmu(const WReg &d, const DReg &n) { ConversionFpInt(0, 0, 1, 2, 1, d, n); }
void CodeGenerator::fcvtzs(const WReg &d, const DReg &n) { ConversionFpInt(0, 0, 1, 3, 0, d, n); }
void CodeGenerator::fcvtzu(const WReg &d, const DReg &n) { ConversionFpInt(0, 0, 1, 3, 1, d, n); }
void CodeGenerator::fjcvtzs(const WReg &d, const DReg &n) { ConversionFpInt(0, 0, 1, 3, 6, d, n); }
void CodeGenerator::scvtf(const DReg &d, const WReg &n) { ConversionFpInt(0, 0, 1, 0, 2, d, n); }
void CodeGenerator::ucvtf(const DReg &d, const WReg &n) { ConversionFpInt(0, 0, 1, 0, 3, d, n); }
void CodeGenerator::fcvtns(const WReg &d, const HReg &n) { ConversionFpInt(0, 0, 3, 0, 0, d, n); }
void CodeGenerator::fcvtnu(const WReg &d, const HReg &n) { ConversionFpInt(0, 0, 3, 0, 1, d, n); }
void CodeGenerator::fcvtas(const WReg &d, const HReg &n) { ConversionFpInt(0, 0, 3, 0, 4, d, n); }
void CodeGenerator::fcvtau(const WReg &d, const HReg &n) { ConversionFpInt(0, 0, 3, 0, 5, d, n); }
void CodeGenerator::fmov(const WReg &d, const HReg &n) { ConversionFpInt(0, 0, 3, 0, 6, d, n); }
void CodeGenerator::fcvtps(const WReg &d, const HReg &n) { ConversionFpInt(0, 0, 3, 1, 0, d, n); }
void CodeGenerator::fcvtpu(const WReg &d, const HReg &n) { ConversionFpInt(0, 0, 3, 1, 1, d, n); }
void CodeGenerator::fcvtms(const WReg &d, const HReg &n) { ConversionFpInt(0, 0, 3, 2, 0, d, n); }
void CodeGenerator::fcvtmu(const WReg &d, const HReg &n) { ConversionFpInt(0, 0, 3, 2, 1, d, n); }
void CodeGenerator::fcvtzs(const WReg &d, const HReg &n) { ConversionFpInt(0, 0, 3, 3, 0, d, n); }
void CodeGenerator::fcvtzu(const WReg &d, const HReg &n) { ConversionFpInt(0, 0, 3, 3, 1, d, n); }
void CodeGenerator::scvtf(const HReg &d, const WReg &n) { ConversionFpInt(0, 0, 3, 0, 2, d, n); }
void CodeGenerator::ucvtf(const HReg &d, const WReg &n) { ConversionFpInt(0, 0, 3, 0, 3, d, n); }
void CodeGenerator::fmov(const HReg &d, const WReg &n) { ConversionFpInt(0, 0, 3, 0, 7, d, n); }
void CodeGenerator::fcvtns(const XReg &d, const SReg &n) { ConversionFpInt(1, 0, 0, 0, 0, d, n); }
void CodeGenerator::fcvtnu(const XReg &d, const SReg &n) { ConversionFpInt(1, 0, 0, 0, 1, d, n); }
void CodeGenerator::fcvtas(const XReg &d, const SReg &n) { ConversionFpInt(1, 0, 0, 0, 4, d, n); }
void CodeGenerator::fcvtau(const XReg &d, const SReg &n) { ConversionFpInt(1, 0, 0, 0, 5, d, n); }
void CodeGenerator::fcvtps(const XReg &d, const SReg &n) { ConversionFpInt(1, 0, 0, 1, 0, d, n); }
void CodeGenerator::fcvtpu(const XReg &d, const SReg &n) { ConversionFpInt(1, 0, 0, 1, 1, d, n); }
void CodeGenerator::fcvtms(const XReg &d, const SReg &n) { ConversionFpInt(1, 0, 0, 2, 0, d, n); }
void CodeGenerator::fcvtmu(const XReg &d, const SReg &n) { ConversionFpInt(1, 0, 0, 2, 1, d, n); }
void CodeGenerator::fcvtzs(const XReg &d, const SReg &n) { ConversionFpInt(1, 0, 0, 3, 0, d, n); }
void CodeGenerator::fcvtzu(const XReg &d, const SReg &n) { ConversionFpInt(1, 0, 0, 3, 1, d, n); }
void CodeGenerator::scvtf(const SReg &d, const XReg &n) { ConversionFpInt(1, 0, 0, 0, 2, d, n); }
void CodeGenerator::ucvtf(const SReg &d, const XReg &n) { ConversionFpInt(1, 0, 0, 0, 3, d, n); }
void CodeGenerator::fcvtns(const XReg &d, const DReg &n) { ConversionFpInt(1, 0, 1, 0, 0, d, n); }
void CodeGenerator::fcvtnu(const XReg &d, const DReg &n) { ConversionFpInt(1, 0, 1, 0, 1, d, n); }
void CodeGenerator::fcvtas(const XReg &d, const DReg &n) { ConversionFpInt(1, 0, 1, 0, 4, d, n); }
void CodeGenerator::fcvtau(const XReg &d, const DReg &n) { ConversionFpInt(1, 0, 1, 0, 5, d, n); }
void CodeGenerator::fmov(const XReg &d, const DReg &n) { ConversionFpInt(1, 0, 1, 0, 6, d, n); }
void CodeGenerator::fcvtps(const XReg &d, const DReg &n) { ConversionFpInt(1, 0, 1, 1, 0, d, n); }
void CodeGenerator::fcvtpu(const XReg &d, const DReg &n) { ConversionFpInt(1, 0, 1, 1, 1, d, n); }
void CodeGenerator::fcvtms(const XReg &d, const DReg &n) { ConversionFpInt(1, 0, 1, 2, 0, d, n); }
void CodeGenerator::fcvtmu(const XReg &d, const DReg &n) { ConversionFpInt(1, 0, 1, 2, 1, d, n); }
void CodeGenerator::fcvtzs(const XReg &d, const DReg &n) { ConversionFpInt(1, 0, 1, 3, 0, d, n); }
void CodeGenerator::fcvtzu(const XReg &d, const DReg &n) { ConversionFpInt(1, 0, 1, 3, 1, d, n); }
void CodeGenerator::scvtf(const DReg &d, const XReg &n) { ConversionFpInt(1, 0, 1, 0, 2, d, n); }
void CodeGenerator::ucvtf(const DReg &d, const XReg &n) { ConversionFpInt(1, 0, 1, 0, 3, d, n); }
void CodeGenerator::fmov(const DReg &d, const XReg &n) { ConversionFpInt(1, 0, 1, 0, 7, d, n); }
void CodeGenerator::fmov(const XReg &d, const VRegDElem &n) { ConversionFpInt(1, 0, 2, 1, 6, d, n); }
void CodeGenerator::fmov(const VRegDElem &d, const XReg &n) { ConversionFpInt(1, 0, 2, 1, 7, d, n); }
void CodeGenerator::fcvtns(const XReg &d, const HReg &n) { ConversionFpInt(1, 0, 3, 0, 0, d, n); }
void CodeGenerator::fcvtnu(const XReg &d, const HReg &n) { ConversionFpInt(1, 0, 3, 0, 1, d, n); }
void CodeGenerator::fcvtas(const XReg &d, const HReg &n) { ConversionFpInt(1, 0, 3, 0, 4, d, n); }
void CodeGenerator::fcvtau(const XReg &d, const HReg &n) { ConversionFpInt(1, 0, 3, 0, 5, d, n); }
void CodeGenerator::fmov(const XReg &d, const HReg &n) { ConversionFpInt(1, 0, 3, 0, 6, d, n); }
void CodeGenerator::fcvtps(const XReg &d, const HReg &n) { ConversionFpInt(1, 0, 3, 1, 0, d, n); }
void CodeGenerator::fcvtpu(const XReg &d, const HReg &n) { ConversionFpInt(1, 0, 3, 1, 1, d, n); }
void CodeGenerator::fcvtms(const XReg &d, const HReg &n) { ConversionFpInt(1, 0, 3, 2, 0, d, n); }
void CodeGenerator::fcvtmu(const XReg &d, const HReg &n) { ConversionFpInt(1, 0, 3, 2, 1, d, n); }
void CodeGenerator::fcvtzs(const XReg &d, const HReg &n) { ConversionFpInt(1, 0, 3, 3, 0, d, n); }
void CodeGenerator::fcvtzu(const XReg &d, const HReg &n) { ConversionFpInt(1, 0, 3, 3, 1, d, n); }
void CodeGenerator::scvtf(const HReg &d, const XReg &n) { ConversionFpInt(1, 0, 3, 0, 2, d, n); }
void CodeGenerator::ucvtf(const HReg &d, const XReg &n) { ConversionFpInt(1, 0, 3, 0, 3, d, n); }
void CodeGenerator::fmov(const HReg &d, const XReg &n) { ConversionFpInt(1, 0, 3, 0, 7, d, n); }
void CodeGenerator::fmov(const SReg &vd, const SReg &vn) { FpDataProc1Reg(0, 0, 0, 0, vd, vn); }
void CodeGenerator::fabs(const SReg &vd, const SReg &vn) { FpDataProc1Reg(0, 0, 0, 1, vd, vn); }
void CodeGenerator::fneg(const SReg &vd, const SReg &vn) { FpDataProc1Reg(0, 0, 0, 2, vd, vn); }
void CodeGenerator::fsqrt(const SReg &vd, const SReg &vn) { FpDataProc1Reg(0, 0, 0, 3, vd, vn); }
void CodeGenerator::frintn(const SReg &vd, const SReg &vn) { FpDataProc1Reg(0, 0, 0, 8, vd, vn); }
void CodeGenerator::frintp(const SReg &vd, const SReg &vn) { FpDataProc1Reg(0, 0, 0, 9, vd, vn); }
void CodeGenerator::frintm(const SReg &vd, const SReg &vn) { FpDataProc1Reg(0, 0, 0, 10, vd, vn); }
void CodeGenerator::frintz(const SReg &vd, const SReg &vn) { FpDataProc1Reg(0, 0, 0, 11, vd, vn); }
void CodeGenerator::frinta(const SReg &vd, const SReg &vn) { FpDataProc1Reg(0, 0, 0, 12, vd, vn); }
void CodeGenerator::frintx(const SReg &vd, const SReg &vn) { FpDataProc1Reg(0, 0, 0, 14, vd, vn); }
void CodeGenerator::frinti(const SReg &vd, const SReg &vn) { FpDataProc1Reg(0, 0, 0, 15, vd, vn); }
void CodeGenerator::fcvt(const DReg &vd, const SReg &vn) { FpDataProc1Reg(0, 0, 0, 5, vd, vn); }
void CodeGenerator::fcvt(const HReg &vd, const SReg &vn) { FpDataProc1Reg(0, 0, 0, 7, vd, vn); }
void CodeGenerator::fmov(const DReg &vd, const DReg &vn) { FpDataProc1Reg(0, 0, 1, 0, vd, vn); }
void CodeGenerator::fabs(const DReg &vd, const DReg &vn) { FpDataProc1Reg(0, 0, 1, 1, vd, vn); }
void CodeGenerator::fneg(const DReg &vd, const DReg &vn) { FpDataProc1Reg(0, 0, 1, 2, vd, vn); }
void CodeGenerator::fsqrt(const DReg &vd, const DReg &vn) { FpDataProc1Reg(0, 0, 1, 3, vd, vn); }
void CodeGenerator::frintn(const DReg &vd, const DReg &vn) { FpDataProc1Reg(0, 0, 1, 8, vd, vn); }
void CodeGenerator::frintp(const DReg &vd, const DReg &vn) { FpDataProc1Reg(0, 0, 1, 9, vd, vn); }
void CodeGenerator::frintm(const DReg &vd, const DReg &vn) { FpDataProc1Reg(0, 0, 1, 10, vd, vn); }
void CodeGenerator::frintz(const DReg &vd, const DReg &vn) { FpDataProc1Reg(0, 0, 1, 11, vd, vn); }
void CodeGenerator::frinta(const DReg &vd, const DReg &vn) { FpDataProc1Reg(0, 0, 1, 12, vd, vn); }
void CodeGenerator::frintx(const DReg &vd, const DReg &vn) { FpDataProc1Reg(0, 0, 1, 14, vd, vn); }
void CodeGenerator::frinti(const DReg &vd, const DReg &vn) { FpDataProc1Reg(0, 0, 1, 15, vd, vn); }
void CodeGenerator::fcvt(const SReg &vd, const DReg &vn) { FpDataProc1Reg(0, 0, 1, 4, vd, vn); }
void CodeGenerator::fcvt(const HReg &vd, const DReg &vn) { FpDataProc1Reg(0, 0, 1, 7, vd, vn); }
void CodeGenerator::fmov(const HReg &vd, const HReg &vn) { FpDataProc1Reg(0, 0, 3, 0, vd, vn); }
void CodeGenerator::fabs(const HReg &vd, const HReg &vn) { FpDataProc1Reg(0, 0, 3, 1, vd, vn); }
void CodeGenerator::fneg(const HReg &vd, const HReg &vn) { FpDataProc1Reg(0, 0, 3, 2, vd, vn); }
void CodeGenerator::fsqrt(const HReg &vd, const HReg &vn) { FpDataProc1Reg(0, 0, 3, 3, vd, vn); }
void CodeGenerator::frintn(const HReg &vd, const HReg &vn) { FpDataProc1Reg(0, 0, 3, 8, vd, vn); }
void CodeGenerator::frintp(const HReg &vd, const HReg &vn) { FpDataProc1Reg(0, 0, 3, 9, vd, vn); }
void CodeGenerator::frintm(const HReg &vd, const HReg &vn) { FpDataProc1Reg(0, 0, 3, 10, vd, vn); }
void CodeGenerator::frintz(const HReg &vd, const HReg &vn) { FpDataProc1Reg(0, 0, 3, 11, vd, vn); }
void CodeGenerator::frinta(const HReg &vd, const HReg &vn) { FpDataProc1Reg(0, 0, 3, 12, vd, vn); }
void CodeGenerator::frintx(const HReg &vd, const HReg &vn) { FpDataProc1Reg(0, 0, 3, 14, vd, vn); }
void CodeGenerator::frinti(const HReg &vd, const HReg &vn) { FpDataProc1Reg(0, 0, 3, 15, vd, vn); }
void CodeGenerator::fcvt(const SReg &vd, const HReg &vn) { FpDataProc1Reg(0, 0, 3, 4, vd, vn); }
void CodeGenerator::fcvt(const DReg &vd, const HReg &vn) { FpDataProc1Reg(0, 0, 3, 5, vd, vn); }
void CodeGenerator::fcmp(const SReg &vn, const SReg &vm) { FpComp(0, 0, 0, 0, 0, vn, vm); }
void CodeGenerator::fcmpe(const SReg &vn, const SReg &vm) { FpComp(0, 0, 0, 0, 16, vn, vm); }
void CodeGenerator::fcmp(const SReg &vn, const double imm) { FpComp(0, 0, 0, 0, 8, vn, imm); }
void CodeGenerator::fcmpe(const SReg &vn, const double imm) { FpComp(0, 0, 0, 0, 24, vn, imm); }
void CodeGenerator::fcmp(const DReg &vn, const DReg &vm) { FpComp(0, 0, 1, 0, 0, vn, vm); }
void CodeGenerator::fcmpe(const DReg &vn, const DReg &vm) { FpComp(0, 0, 1, 0, 16, vn, vm); }
void CodeGenerator::fcmp(const DReg &vn, const double imm) { FpComp(0, 0, 1, 0, 8, vn, imm); }
void CodeGenerator::fcmpe(const DReg &vn, const double imm) { FpComp(0, 0, 1, 0, 24, vn, imm); }
void CodeGenerator::fcmp(const HReg &vn, const HReg &vm) { FpComp(0, 0, 3, 0, 0, vn, vm); }
void CodeGenerator::fcmpe(const HReg &vn, const HReg &vm) { FpComp(0, 0, 3, 0, 16, vn, vm); }
void CodeGenerator::fcmp(const HReg &vn, const double imm) { FpComp(0, 0, 3, 0, 8, vn, imm); }
void CodeGenerator::fcmpe(const HReg &vn, const double imm) { FpComp(0, 0, 3, 0, 24, vn, imm); }
void CodeGenerator::fmov(const SReg &vd, const double imm) { FpImm(0, 0, 0, vd, imm); }
void CodeGenerator::fmov(const DReg &vd, const double imm) { FpImm(0, 0, 1, vd, imm); }
void CodeGenerator::fmov(const HReg &vd, const double imm) { FpImm(0, 0, 3, vd, imm); }
void CodeGenerator::fccmp(const SReg &vn, const SReg &vm, const uint32_t nzcv, const Cond cond) { FpCondComp(0, 0, 0, 0, vn, vm, nzcv, cond); }
void CodeGenerator::fccmpe(const SReg &vn, const SReg &vm, const uint32_t nzcv, const Cond cond) { FpCondComp(0, 0, 0, 1, vn, vm, nzcv, cond); }
void CodeGenerator::fccmp(const DReg &vn, const DReg &vm, const uint32_t nzcv, const Cond cond) { FpCondComp(0, 0, 1, 0, vn, vm, nzcv, cond); }
void CodeGenerator::fccmpe(const DReg &vn, const DReg &vm, const uint32_t nzcv, const Cond cond) { FpCondComp(0, 0, 1, 1, vn, vm, nzcv, cond); }
void CodeGenerator::fccmp(const HReg &vn, const HReg &vm, const uint32_t nzcv, const Cond cond) { FpCondComp(0, 0, 3, 0, vn, vm, nzcv, cond); }
void CodeGenerator::fccmpe(const HReg &vn, const HReg &vm, const uint32_t nzcv, const Cond cond) { FpCondComp(0, 0, 3, 1, vn, vm, nzcv, cond); }
void CodeGenerator::fmul(const SReg &vd, const SReg &vn, const SReg &vm) { FpDataProc2Reg(0, 0, 0, 0, vd, vn, vm); }
void CodeGenerator::fdiv(const SReg &vd, const SReg &vn, const SReg &vm) { FpDataProc2Reg(0, 0, 0, 1, vd, vn, vm); }
void CodeGenerator::fadd(const SReg &vd, const SReg &vn, const SReg &vm) { FpDataProc2Reg(0, 0, 0, 2, vd, vn, vm); }
void CodeGenerator::fsub(const SReg &vd, const SReg &vn, const SReg &vm) { FpDataProc2Reg(0, 0, 0, 3, vd, vn, vm); }
void CodeGenerator::fmax(const SReg &vd, const SReg &vn, const SReg &vm) { FpDataProc2Reg(0, 0, 0, 4, vd, vn, vm); }
void CodeGenerator::fmin(const SReg &vd, const SReg &vn, const SReg &vm) { FpDataProc2Reg(0, 0, 0, 5, vd, vn, vm); }
void CodeGenerator::fmaxnm(const SReg &vd, const SReg &vn, const SReg &vm) { FpDataProc2Reg(0, 0, 0, 6, vd, vn, vm); }
void CodeGenerator::fminnm(const SReg &vd, const SReg &vn, const SReg &vm) { FpDataProc2Reg(0, 0, 0, 7, vd, vn, vm); }
void CodeGenerator::fnmul(const SReg &vd, const SReg &vn, const SReg &vm) { FpDataProc2Reg(0, 0, 0, 8, vd, vn, vm); }
void CodeGenerator::fmul(const DReg &vd, const DReg &vn, const DReg &vm) { FpDataProc2Reg(0, 0, 1, 0, vd, vn, vm); }
void CodeGenerator::fdiv(const DReg &vd, const DReg &vn, const DReg &vm) { FpDataProc2Reg(0, 0, 1, 1, vd, vn, vm); }
void CodeGenerator::fadd(const DReg &vd, const DReg &vn, const DReg &vm) { FpDataProc2Reg(0, 0, 1, 2, vd, vn, vm); }
void CodeGenerator::fsub(const DReg &vd, const DReg &vn, const DReg &vm) { FpDataProc2Reg(0, 0, 1, 3, vd, vn, vm); }
void CodeGenerator::fmax(const DReg &vd, const DReg &vn, const DReg &vm) { FpDataProc2Reg(0, 0, 1, 4, vd, vn, vm); }
void CodeGenerator::fmin(const DReg &vd, const DReg &vn, const DReg &vm) { FpDataProc2Reg(0, 0, 1, 5, vd, vn, vm); }
void CodeGenerator::fmaxnm(const DReg &vd, const DReg &vn, const DReg &vm) { FpDataProc2Reg(0, 0, 1, 6, vd, vn, vm); }
void CodeGenerator::fminnm(const DReg &vd, const DReg &vn, const DReg &vm) { FpDataProc2Reg(0, 0, 1, 7, vd, vn, vm); }
void CodeGenerator::fnmul(const DReg &vd, const DReg &vn, const DReg &vm) { FpDataProc2Reg(0, 0, 1, 8, vd, vn, vm); }
void CodeGenerator::fmul(const HReg &vd, const HReg &vn, const HReg &vm) { FpDataProc2Reg(0, 0, 3, 0, vd, vn, vm); }
void CodeGenerator::fdiv(const HReg &vd, const HReg &vn, const HReg &vm) { FpDataProc2Reg(0, 0, 3, 1, vd, vn, vm); }
void CodeGenerator::fadd(const HReg &vd, const HReg &vn, const HReg &vm) { FpDataProc2Reg(0, 0, 3, 2, vd, vn, vm); }
void CodeGenerator::fsub(const HReg &vd, const HReg &vn, const HReg &vm) { FpDataProc2Reg(0, 0, 3, 3, vd, vn, vm); }
void CodeGenerator::fmax(const HReg &vd, const HReg &vn, const HReg &vm) { FpDataProc2Reg(0, 0, 3, 4, vd, vn, vm); }
void CodeGenerator::fmin(const HReg &vd, const HReg &vn, const HReg &vm) { FpDataProc2Reg(0, 0, 3, 5, vd, vn, vm); }
void CodeGenerator::fmaxnm(const HReg &vd, const HReg &vn, const HReg &vm) { FpDataProc2Reg(0, 0, 3, 6, vd, vn, vm); }
void CodeGenerator::fminnm(const HReg &vd, const HReg &vn, const HReg &vm) { FpDataProc2Reg(0, 0, 3, 7, vd, vn, vm); }
void CodeGenerator::fnmul(const HReg &vd, const HReg &vn, const HReg &vm) { FpDataProc2Reg(0, 0, 3, 8, vd, vn, vm); }
void CodeGenerator::fcsel(const SReg &vd, const SReg &vn, const SReg &vm, const Cond cond) { FpCondSel(0, 0, 0, vd, vn, vm, cond); }
void CodeGenerator::fcsel(const DReg &vd, const DReg &vn, const DReg &vm, const Cond cond) { FpCondSel(0, 0, 1, vd, vn, vm, cond); }
void CodeGenerator::fcsel(const HReg &vd, const HReg &vn, const HReg &vm, const Cond cond) { FpCondSel(0, 0, 3, vd, vn, vm, cond); }
void CodeGenerator::fmadd(const SReg &vd, const SReg &vn, const SReg &vm, const SReg &va) { FpDataProc3Reg(0, 0, 0, 0, 0, vd, vn, vm, va); }
void CodeGenerator::fmsub(const SReg &vd, const SReg &vn, const SReg &vm, const SReg &va) { FpDataProc3Reg(0, 0, 0, 0, 1, vd, vn, vm, va); }
void CodeGenerator::fnmadd(const SReg &vd, const SReg &vn, const SReg &vm, const SReg &va) { FpDataProc3Reg(0, 0, 0, 1, 0, vd, vn, vm, va); }
void CodeGenerator::fnmsub(const SReg &vd, const SReg &vn, const SReg &vm, const SReg &va) { FpDataProc3Reg(0, 0, 0, 1, 1, vd, vn, vm, va); }
void CodeGenerator::fmadd(const DReg &vd, const DReg &vn, const DReg &vm, const DReg &va) { FpDataProc3Reg(0, 0, 1, 0, 0, vd, vn, vm, va); }
void CodeGenerator::fmsub(const DReg &vd, const DReg &vn, const DReg &vm, const DReg &va) { FpDataProc3Reg(0, 0, 1, 0, 1, vd, vn, vm, va); }
void CodeGenerator::fnmadd(const DReg &vd, const DReg &vn, const DReg &vm, const DReg &va) { FpDataProc3Reg(0, 0, 1, 1, 0, vd, vn, vm, va); }
void CodeGenerator::fnmsub(const DReg &vd, const DReg &vn, const DReg &vm, const DReg &va) { FpDataProc3Reg(0, 0, 1, 1, 1, vd, vn, vm, va); }
void CodeGenerator::fmadd(const HReg &vd, const HReg &vn, const HReg &vm, const HReg &va) { FpDataProc3Reg(0, 0, 3, 0, 0, vd, vn, vm, va); }
void CodeGenerator::fmsub(const HReg &vd, const HReg &vn, const HReg &vm, const HReg &va) { FpDataProc3Reg(0, 0, 3, 0, 1, vd, vn, vm, va); }
void CodeGenerator::fnmadd(const HReg &vd, const HReg &vn, const HReg &vm, const HReg &va) { FpDataProc3Reg(0, 0, 3, 1, 0, vd, vn, vm, va); }
void CodeGenerator::fnmsub(const HReg &vd, const HReg &vn, const HReg &vm, const HReg &va) { FpDataProc3Reg(0, 0, 3, 1, 1, vd, vn, vm, va); }
void CodeGenerator::orr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveBitwiseLOpPred(0, zdn, pg, zm); }
void CodeGenerator::orr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveBitwiseLOpPred(0, zdn, pg, zm); }
void CodeGenerator::orr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveBitwiseLOpPred(0, zdn, pg, zm); }
void CodeGenerator::orr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseLOpPred(0, zdn, pg, zm); }
void CodeGenerator::eor(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveBitwiseLOpPred(1, zdn, pg, zm); }
void CodeGenerator::eor(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveBitwiseLOpPred(1, zdn, pg, zm); }
void CodeGenerator::eor(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveBitwiseLOpPred(1, zdn, pg, zm); }
void CodeGenerator::eor(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseLOpPred(1, zdn, pg, zm); }
void CodeGenerator::and_(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveBitwiseLOpPred(2, zdn, pg, zm); }
void CodeGenerator::and_(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveBitwiseLOpPred(2, zdn, pg, zm); }
void CodeGenerator::and_(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveBitwiseLOpPred(2, zdn, pg, zm); }
void CodeGenerator::and_(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseLOpPred(2, zdn, pg, zm); }
void CodeGenerator::bic(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveBitwiseLOpPred(3, zdn, pg, zm); }
void CodeGenerator::bic(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveBitwiseLOpPred(3, zdn, pg, zm); }
void CodeGenerator::bic(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveBitwiseLOpPred(3, zdn, pg, zm); }
void CodeGenerator::bic(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseLOpPred(3, zdn, pg, zm); }
void CodeGenerator::add(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveIntAddSubVecPred(0, zdn, pg, zm); }
void CodeGenerator::add(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveIntAddSubVecPred(0, zdn, pg, zm); }
void CodeGenerator::add(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntAddSubVecPred(0, zdn, pg, zm); }
void CodeGenerator::add(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntAddSubVecPred(0, zdn, pg, zm); }
void CodeGenerator::sub(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveIntAddSubVecPred(1, zdn, pg, zm); }
void CodeGenerator::sub(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveIntAddSubVecPred(1, zdn, pg, zm); }
void CodeGenerator::sub(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntAddSubVecPred(1, zdn, pg, zm); }
void CodeGenerator::sub(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntAddSubVecPred(1, zdn, pg, zm); }
void CodeGenerator::subr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveIntAddSubVecPred(3, zdn, pg, zm); }
void CodeGenerator::subr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveIntAddSubVecPred(3, zdn, pg, zm); }
void CodeGenerator::subr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntAddSubVecPred(3, zdn, pg, zm); }
void CodeGenerator::subr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntAddSubVecPred(3, zdn, pg, zm); }
void CodeGenerator::smax(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveIntMinMaxDiffPred(0, 0, zdn, pg, zm); }
void CodeGenerator::smax(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveIntMinMaxDiffPred(0, 0, zdn, pg, zm); }
void CodeGenerator::smax(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntMinMaxDiffPred(0, 0, zdn, pg, zm); }
void CodeGenerator::smax(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntMinMaxDiffPred(0, 0, zdn, pg, zm); }
void CodeGenerator::umax(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveIntMinMaxDiffPred(0, 1, zdn, pg, zm); }
void CodeGenerator::umax(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveIntMinMaxDiffPred(0, 1, zdn, pg, zm); }
void CodeGenerator::umax(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntMinMaxDiffPred(0, 1, zdn, pg, zm); }
void CodeGenerator::umax(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntMinMaxDiffPred(0, 1, zdn, pg, zm); }
void CodeGenerator::smin(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveIntMinMaxDiffPred(1, 0, zdn, pg, zm); }
void CodeGenerator::smin(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveIntMinMaxDiffPred(1, 0, zdn, pg, zm); }
void CodeGenerator::smin(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntMinMaxDiffPred(1, 0, zdn, pg, zm); }
void CodeGenerator::smin(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntMinMaxDiffPred(1, 0, zdn, pg, zm); }
void CodeGenerator::umin(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveIntMinMaxDiffPred(1, 1, zdn, pg, zm); }
void CodeGenerator::umin(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveIntMinMaxDiffPred(1, 1, zdn, pg, zm); }
void CodeGenerator::umin(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntMinMaxDiffPred(1, 1, zdn, pg, zm); }
void CodeGenerator::umin(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntMinMaxDiffPred(1, 1, zdn, pg, zm); }
void CodeGenerator::sabd(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveIntMinMaxDiffPred(2, 0, zdn, pg, zm); }
void CodeGenerator::sabd(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveIntMinMaxDiffPred(2, 0, zdn, pg, zm); }
void CodeGenerator::sabd(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntMinMaxDiffPred(2, 0, zdn, pg, zm); }
void CodeGenerator::sabd(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntMinMaxDiffPred(2, 0, zdn, pg, zm); }
void CodeGenerator::uabd(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveIntMinMaxDiffPred(2, 1, zdn, pg, zm); }
void CodeGenerator::uabd(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveIntMinMaxDiffPred(2, 1, zdn, pg, zm); }
void CodeGenerator::uabd(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntMinMaxDiffPred(2, 1, zdn, pg, zm); }
void CodeGenerator::uabd(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntMinMaxDiffPred(2, 1, zdn, pg, zm); }
void CodeGenerator::mul(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveIntMultDivVecPred(0, 0, zdn, pg, zm); }
void CodeGenerator::mul(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveIntMultDivVecPred(0, 0, zdn, pg, zm); }
void CodeGenerator::mul(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntMultDivVecPred(0, 0, zdn, pg, zm); }
void CodeGenerator::mul(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntMultDivVecPred(0, 0, zdn, pg, zm); }
void CodeGenerator::smulh(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveIntMultDivVecPred(1, 0, zdn, pg, zm); }
void CodeGenerator::smulh(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveIntMultDivVecPred(1, 0, zdn, pg, zm); }
void CodeGenerator::smulh(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntMultDivVecPred(1, 0, zdn, pg, zm); }
void CodeGenerator::smulh(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntMultDivVecPred(1, 0, zdn, pg, zm); }
void CodeGenerator::umulh(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveIntMultDivVecPred(1, 1, zdn, pg, zm); }
void CodeGenerator::umulh(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveIntMultDivVecPred(1, 1, zdn, pg, zm); }
void CodeGenerator::umulh(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntMultDivVecPred(1, 1, zdn, pg, zm); }
void CodeGenerator::umulh(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntMultDivVecPred(1, 1, zdn, pg, zm); }
void CodeGenerator::sdiv(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntMultDivVecPred(2, 0, zdn, pg, zm); }
void CodeGenerator::sdiv(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntMultDivVecPred(2, 0, zdn, pg, zm); }
void CodeGenerator::udiv(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntMultDivVecPred(2, 1, zdn, pg, zm); }
void CodeGenerator::udiv(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntMultDivVecPred(2, 1, zdn, pg, zm); }
void CodeGenerator::sdivr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntMultDivVecPred(3, 0, zdn, pg, zm); }
void CodeGenerator::sdivr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntMultDivVecPred(3, 0, zdn, pg, zm); }
void CodeGenerator::udivr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveIntMultDivVecPred(3, 1, zdn, pg, zm); }
void CodeGenerator::udivr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveIntMultDivVecPred(3, 1, zdn, pg, zm); }
void CodeGenerator::orv(const BReg &vd, const _PReg &pg, const ZRegB &zn) { SveBitwiseLReductPred(0, vd, pg, zn); }
void CodeGenerator::orv(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveBitwiseLReductPred(0, vd, pg, zn); }
void CodeGenerator::orv(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveBitwiseLReductPred(0, vd, pg, zn); }
void CodeGenerator::orv(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveBitwiseLReductPred(0, vd, pg, zn); }
void CodeGenerator::eorv(const BReg &vd, const _PReg &pg, const ZRegB &zn) { SveBitwiseLReductPred(1, vd, pg, zn); }
void CodeGenerator::eorv(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveBitwiseLReductPred(1, vd, pg, zn); }
void CodeGenerator::eorv(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveBitwiseLReductPred(1, vd, pg, zn); }
void CodeGenerator::eorv(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveBitwiseLReductPred(1, vd, pg, zn); }
void CodeGenerator::andv(const BReg &vd, const _PReg &pg, const ZRegB &zn) { SveBitwiseLReductPred(2, vd, pg, zn); }
void CodeGenerator::andv(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveBitwiseLReductPred(2, vd, pg, zn); }
void CodeGenerator::andv(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveBitwiseLReductPred(2, vd, pg, zn); }
void CodeGenerator::andv(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveBitwiseLReductPred(2, vd, pg, zn); }
void CodeGenerator::movprfx(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) { SveConstPrefPred(0, zd, pg, zn); }
void CodeGenerator::movprfx(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveConstPrefPred(0, zd, pg, zn); }
void CodeGenerator::movprfx(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveConstPrefPred(0, zd, pg, zn); }
void CodeGenerator::movprfx(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveConstPrefPred(0, zd, pg, zn); }
void CodeGenerator::saddv(const DReg &vd, const _PReg &pg, const ZRegB &zn) { SveIntAddReductPred(0, 0, vd, pg, zn); }
void CodeGenerator::saddv(const DReg &vd, const _PReg &pg, const ZRegH &zn) { SveIntAddReductPred(0, 0, vd, pg, zn); }
void CodeGenerator::saddv(const DReg &vd, const _PReg &pg, const ZRegS &zn) { SveIntAddReductPred(0, 0, vd, pg, zn); }
void CodeGenerator::uaddv(const DReg &vd, const _PReg &pg, const ZRegB &zn) { SveIntAddReductPred(0, 1, vd, pg, zn); }
void CodeGenerator::uaddv(const DReg &vd, const _PReg &pg, const ZRegH &zn) { SveIntAddReductPred(0, 1, vd, pg, zn); }
void CodeGenerator::uaddv(const DReg &vd, const _PReg &pg, const ZRegS &zn) { SveIntAddReductPred(0, 1, vd, pg, zn); }
void CodeGenerator::uaddv(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveIntAddReductPred(0, 1, vd, pg, zn); }
void CodeGenerator::smaxv(const BReg &vd, const _PReg &pg, const ZRegB &zn) { SveIntMinMaxReductPred(0, 0, vd, pg, zn); }
void CodeGenerator::smaxv(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveIntMinMaxReductPred(0, 0, vd, pg, zn); }
void CodeGenerator::smaxv(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveIntMinMaxReductPred(0, 0, vd, pg, zn); }
void CodeGenerator::smaxv(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveIntMinMaxReductPred(0, 0, vd, pg, zn); }
void CodeGenerator::umaxv(const BReg &vd, const _PReg &pg, const ZRegB &zn) { SveIntMinMaxReductPred(0, 1, vd, pg, zn); }
void CodeGenerator::umaxv(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveIntMinMaxReductPred(0, 1, vd, pg, zn); }
void CodeGenerator::umaxv(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveIntMinMaxReductPred(0, 1, vd, pg, zn); }
void CodeGenerator::umaxv(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveIntMinMaxReductPred(0, 1, vd, pg, zn); }
void CodeGenerator::sminv(const BReg &vd, const _PReg &pg, const ZRegB &zn) { SveIntMinMaxReductPred(1, 0, vd, pg, zn); }
void CodeGenerator::sminv(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveIntMinMaxReductPred(1, 0, vd, pg, zn); }
void CodeGenerator::sminv(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveIntMinMaxReductPred(1, 0, vd, pg, zn); }
void CodeGenerator::sminv(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveIntMinMaxReductPred(1, 0, vd, pg, zn); }
void CodeGenerator::uminv(const BReg &vd, const _PReg &pg, const ZRegB &zn) { SveIntMinMaxReductPred(1, 1, vd, pg, zn); }
void CodeGenerator::uminv(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveIntMinMaxReductPred(1, 1, vd, pg, zn); }
void CodeGenerator::uminv(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveIntMinMaxReductPred(1, 1, vd, pg, zn); }
void CodeGenerator::uminv(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveIntMinMaxReductPred(1, 1, vd, pg, zn); }
void CodeGenerator::asr(const ZRegB &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(0, zdn, pg, amount); }
void CodeGenerator::asr(const ZRegH &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(0, zdn, pg, amount); }
void CodeGenerator::asr(const ZRegS &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(0, zdn, pg, amount); }
void CodeGenerator::asr(const ZRegD &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(0, zdn, pg, amount); }
void CodeGenerator::lsr(const ZRegB &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(1, zdn, pg, amount); }
void CodeGenerator::lsr(const ZRegH &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(1, zdn, pg, amount); }
void CodeGenerator::lsr(const ZRegS &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(1, zdn, pg, amount); }
void CodeGenerator::lsr(const ZRegD &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(1, zdn, pg, amount); }
void CodeGenerator::lsl(const ZRegB &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(3, zdn, pg, amount); }
void CodeGenerator::lsl(const ZRegH &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(3, zdn, pg, amount); }
void CodeGenerator::lsl(const ZRegS &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(3, zdn, pg, amount); }
void CodeGenerator::lsl(const ZRegD &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(3, zdn, pg, amount); }
void CodeGenerator::asrd(const ZRegB &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(4, zdn, pg, amount); }
void CodeGenerator::asrd(const ZRegH &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(4, zdn, pg, amount); }
void CodeGenerator::asrd(const ZRegS &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(4, zdn, pg, amount); }
void CodeGenerator::asrd(const ZRegD &zdn, const _PReg &pg, const uint32_t amount) { SveBitwiseShByImmPred(4, zdn, pg, amount); }
void CodeGenerator::asr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveBitwiseShVecPred(0, zdn, pg, zm); }
void CodeGenerator::asr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveBitwiseShVecPred(0, zdn, pg, zm); }
void CodeGenerator::asr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveBitwiseShVecPred(0, zdn, pg, zm); }
void CodeGenerator::asr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShVecPred(0, zdn, pg, zm); }
void CodeGenerator::lsr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveBitwiseShVecPred(1, zdn, pg, zm); }
void CodeGenerator::lsr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveBitwiseShVecPred(1, zdn, pg, zm); }
void CodeGenerator::lsr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveBitwiseShVecPred(1, zdn, pg, zm); }
void CodeGenerator::lsr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShVecPred(1, zdn, pg, zm); }
void CodeGenerator::lsl(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveBitwiseShVecPred(3, zdn, pg, zm); }
void CodeGenerator::lsl(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveBitwiseShVecPred(3, zdn, pg, zm); }
void CodeGenerator::lsl(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveBitwiseShVecPred(3, zdn, pg, zm); }
void CodeGenerator::lsl(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShVecPred(3, zdn, pg, zm); }
void CodeGenerator::asrr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveBitwiseShVecPred(4, zdn, pg, zm); }
void CodeGenerator::asrr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveBitwiseShVecPred(4, zdn, pg, zm); }
void CodeGenerator::asrr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveBitwiseShVecPred(4, zdn, pg, zm); }
void CodeGenerator::asrr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShVecPred(4, zdn, pg, zm); }
void CodeGenerator::lsrr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveBitwiseShVecPred(5, zdn, pg, zm); }
void CodeGenerator::lsrr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveBitwiseShVecPred(5, zdn, pg, zm); }
void CodeGenerator::lsrr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveBitwiseShVecPred(5, zdn, pg, zm); }
void CodeGenerator::lsrr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShVecPred(5, zdn, pg, zm); }
void CodeGenerator::lslr(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveBitwiseShVecPred(7, zdn, pg, zm); }
void CodeGenerator::lslr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveBitwiseShVecPred(7, zdn, pg, zm); }
void CodeGenerator::lslr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveBitwiseShVecPred(7, zdn, pg, zm); }
void CodeGenerator::lslr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShVecPred(7, zdn, pg, zm); }
void CodeGenerator::asr(const ZRegB &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShWElemPred(0, zdn, pg, zm); }
void CodeGenerator::asr(const ZRegH &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShWElemPred(0, zdn, pg, zm); }
void CodeGenerator::asr(const ZRegS &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShWElemPred(0, zdn, pg, zm); }
void CodeGenerator::lsr(const ZRegB &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShWElemPred(1, zdn, pg, zm); }
void CodeGenerator::lsr(const ZRegH &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShWElemPred(1, zdn, pg, zm); }
void CodeGenerator::lsr(const ZRegS &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShWElemPred(1, zdn, pg, zm); }
void CodeGenerator::lsl(const ZRegB &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShWElemPred(3, zdn, pg, zm); }
void CodeGenerator::lsl(const ZRegH &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShWElemPred(3, zdn, pg, zm); }
void CodeGenerator::lsl(const ZRegS &zdn, const _PReg &pg, const ZRegD &zm) { SveBitwiseShWElemPred(3, zdn, pg, zm); }
void CodeGenerator::cls(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) { SveBitwiseUnaryOpPred(0, zd, pg, zn); }
void CodeGenerator::cls(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveBitwiseUnaryOpPred(0, zd, pg, zn); }
void CodeGenerator::cls(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveBitwiseUnaryOpPred(0, zd, pg, zn); }
void CodeGenerator::cls(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveBitwiseUnaryOpPred(0, zd, pg, zn); }
void CodeGenerator::clz(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) { SveBitwiseUnaryOpPred(1, zd, pg, zn); }
void CodeGenerator::clz(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveBitwiseUnaryOpPred(1, zd, pg, zn); }
void CodeGenerator::clz(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveBitwiseUnaryOpPred(1, zd, pg, zn); }
void CodeGenerator::clz(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveBitwiseUnaryOpPred(1, zd, pg, zn); }
void CodeGenerator::cnt(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) { SveBitwiseUnaryOpPred(2, zd, pg, zn); }
void CodeGenerator::cnt(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveBitwiseUnaryOpPred(2, zd, pg, zn); }
void CodeGenerator::cnt(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveBitwiseUnaryOpPred(2, zd, pg, zn); }
void CodeGenerator::cnt(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveBitwiseUnaryOpPred(2, zd, pg, zn); }
void CodeGenerator::cnot(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) { SveBitwiseUnaryOpPred(3, zd, pg, zn); }
void CodeGenerator::cnot(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveBitwiseUnaryOpPred(3, zd, pg, zn); }
void CodeGenerator::cnot(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveBitwiseUnaryOpPred(3, zd, pg, zn); }
void CodeGenerator::cnot(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveBitwiseUnaryOpPred(3, zd, pg, zn); }
void CodeGenerator::fabs(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveBitwiseUnaryOpPred(4, zd, pg, zn); }
void CodeGenerator::fabs(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveBitwiseUnaryOpPred(4, zd, pg, zn); }
void CodeGenerator::fabs(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveBitwiseUnaryOpPred(4, zd, pg, zn); }
void CodeGenerator::fneg(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveBitwiseUnaryOpPred(5, zd, pg, zn); }
void CodeGenerator::fneg(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveBitwiseUnaryOpPred(5, zd, pg, zn); }
void CodeGenerator::fneg(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveBitwiseUnaryOpPred(5, zd, pg, zn); }
void CodeGenerator::not_(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) { SveBitwiseUnaryOpPred(6, zd, pg, zn); }
void CodeGenerator::not_(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveBitwiseUnaryOpPred(6, zd, pg, zn); }
void CodeGenerator::not_(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveBitwiseUnaryOpPred(6, zd, pg, zn); }
void CodeGenerator::not_(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveBitwiseUnaryOpPred(6, zd, pg, zn); }
void CodeGenerator::sxtb(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveIntUnaryOpPred(0, zd, pg, zn); }
void CodeGenerator::sxtb(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveIntUnaryOpPred(0, zd, pg, zn); }
void CodeGenerator::sxtb(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveIntUnaryOpPred(0, zd, pg, zn); }
void CodeGenerator::uxtb(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveIntUnaryOpPred(1, zd, pg, zn); }
void CodeGenerator::uxtb(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveIntUnaryOpPred(1, zd, pg, zn); }
void CodeGenerator::uxtb(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveIntUnaryOpPred(1, zd, pg, zn); }
void CodeGenerator::sxth(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveIntUnaryOpPred(2, zd, pg, zn); }
void CodeGenerator::sxth(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveIntUnaryOpPred(2, zd, pg, zn); }
void CodeGenerator::uxth(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveIntUnaryOpPred(3, zd, pg, zn); }
void CodeGenerator::uxth(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveIntUnaryOpPred(3, zd, pg, zn); }
void CodeGenerator::sxtw(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveIntUnaryOpPred(4, zd, pg, zn); }
void CodeGenerator::uxtw(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveIntUnaryOpPred(5, zd, pg, zn); }
void CodeGenerator::abs(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) { SveIntUnaryOpPred(6, zd, pg, zn); }
void CodeGenerator::abs(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveIntUnaryOpPred(6, zd, pg, zn); }
void CodeGenerator::abs(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveIntUnaryOpPred(6, zd, pg, zn); }
void CodeGenerator::abs(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveIntUnaryOpPred(6, zd, pg, zn); }
void CodeGenerator::neg(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) { SveIntUnaryOpPred(7, zd, pg, zn); }
void CodeGenerator::neg(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveIntUnaryOpPred(7, zd, pg, zn); }
void CodeGenerator::neg(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveIntUnaryOpPred(7, zd, pg, zn); }
void CodeGenerator::neg(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveIntUnaryOpPred(7, zd, pg, zn); }
void CodeGenerator::mla(const ZRegB &zda, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) { SveIntMultAccumPred(0, zda, pg, zn, zm); }
void CodeGenerator::mla(const ZRegH &zda, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveIntMultAccumPred(0, zda, pg, zn, zm); }
void CodeGenerator::mla(const ZRegS &zda, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveIntMultAccumPred(0, zda, pg, zn, zm); }
void CodeGenerator::mla(const ZRegD &zda, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveIntMultAccumPred(0, zda, pg, zn, zm); }
void CodeGenerator::mls(const ZRegB &zda, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) { SveIntMultAccumPred(1, zda, pg, zn, zm); }
void CodeGenerator::mls(const ZRegH &zda, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveIntMultAccumPred(1, zda, pg, zn, zm); }
void CodeGenerator::mls(const ZRegS &zda, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveIntMultAccumPred(1, zda, pg, zn, zm); }
void CodeGenerator::mls(const ZRegD &zda, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveIntMultAccumPred(1, zda, pg, zn, zm); }
void CodeGenerator::mad(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm, const ZRegB &za) { SveIntMultAddPred(0, zdn, pg, zm, za); }
void CodeGenerator::mad(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm, const ZRegH &za) { SveIntMultAddPred(0, zdn, pg, zm, za); }
void CodeGenerator::mad(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm, const ZRegS &za) { SveIntMultAddPred(0, zdn, pg, zm, za); }
void CodeGenerator::mad(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm, const ZRegD &za) { SveIntMultAddPred(0, zdn, pg, zm, za); }
void CodeGenerator::msb(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm, const ZRegB &za) { SveIntMultAddPred(1, zdn, pg, zm, za); }
void CodeGenerator::msb(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm, const ZRegH &za) { SveIntMultAddPred(1, zdn, pg, zm, za); }
void CodeGenerator::msb(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm, const ZRegS &za) { SveIntMultAddPred(1, zdn, pg, zm, za); }
void CodeGenerator::msb(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm, const ZRegD &za) { SveIntMultAddPred(1, zdn, pg, zm, za); }
void CodeGenerator::add(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) { SveIntAddSubUnpred(0, zd, zn, zm); }
void CodeGenerator::add(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveIntAddSubUnpred(0, zd, zn, zm); }
void CodeGenerator::add(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveIntAddSubUnpred(0, zd, zn, zm); }
void CodeGenerator::add(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveIntAddSubUnpred(0, zd, zn, zm); }
void CodeGenerator::sub(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) { SveIntAddSubUnpred(1, zd, zn, zm); }
void CodeGenerator::sub(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveIntAddSubUnpred(1, zd, zn, zm); }
void CodeGenerator::sub(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveIntAddSubUnpred(1, zd, zn, zm); }
void CodeGenerator::sub(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveIntAddSubUnpred(1, zd, zn, zm); }
void CodeGenerator::sqadd(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) { SveIntAddSubUnpred(4, zd, zn, zm); }
void CodeGenerator::sqadd(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveIntAddSubUnpred(4, zd, zn, zm); }
void CodeGenerator::sqadd(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveIntAddSubUnpred(4, zd, zn, zm); }
void CodeGenerator::sqadd(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveIntAddSubUnpred(4, zd, zn, zm); }
void CodeGenerator::uqadd(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) { SveIntAddSubUnpred(5, zd, zn, zm); }
void CodeGenerator::uqadd(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveIntAddSubUnpred(5, zd, zn, zm); }
void CodeGenerator::uqadd(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveIntAddSubUnpred(5, zd, zn, zm); }
void CodeGenerator::uqadd(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveIntAddSubUnpred(5, zd, zn, zm); }
void CodeGenerator::sqsub(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) { SveIntAddSubUnpred(6, zd, zn, zm); }
void CodeGenerator::sqsub(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveIntAddSubUnpred(6, zd, zn, zm); }
void CodeGenerator::sqsub(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveIntAddSubUnpred(6, zd, zn, zm); }
void CodeGenerator::sqsub(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveIntAddSubUnpred(6, zd, zn, zm); }
void CodeGenerator::uqsub(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) { SveIntAddSubUnpred(7, zd, zn, zm); }
void CodeGenerator::uqsub(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveIntAddSubUnpred(7, zd, zn, zm); }
void CodeGenerator::uqsub(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveIntAddSubUnpred(7, zd, zn, zm); }
void CodeGenerator::uqsub(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveIntAddSubUnpred(7, zd, zn, zm); }
void CodeGenerator::and_(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveBitwiseLOpUnpred(0, zd, zn, zm); }
void CodeGenerator::orr(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveBitwiseLOpUnpred(1, zd, zn, zm); }
void CodeGenerator::mov(const ZRegD &zd, const ZRegD &zn) { SveBitwiseLOpUnpred(1, zd, zn, zn); }
void CodeGenerator::eor(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveBitwiseLOpUnpred(2, zd, zn, zm); }
void CodeGenerator::bic(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveBitwiseLOpUnpred(3, zd, zn, zm); }
void CodeGenerator::index(const ZRegB &zd, const int32_t imm1, const int32_t imm2) { SveIndexGenImmImmInc(zd, imm1, imm2); }
void CodeGenerator::index(const ZRegH &zd, const int32_t imm1, const int32_t imm2) { SveIndexGenImmImmInc(zd, imm1, imm2); }
void CodeGenerator::index(const ZRegS &zd, const int32_t imm1, const int32_t imm2) { SveIndexGenImmImmInc(zd, imm1, imm2); }
void CodeGenerator::index(const ZRegD &zd, const int32_t imm1, const int32_t imm2) { SveIndexGenImmImmInc(zd, imm1, imm2); }
void CodeGenerator::index(const ZRegB &zd, const int32_t imm, const WReg &rm) { SveIndexGenImmRegInc(zd, imm, rm); }
void CodeGenerator::index(const ZRegH &zd, const int32_t imm, const WReg &rm) { SveIndexGenImmRegInc(zd, imm, rm); }
void CodeGenerator::index(const ZRegS &zd, const int32_t imm, const WReg &rm) { SveIndexGenImmRegInc(zd, imm, rm); }
void CodeGenerator::index(const ZRegD &zd, const int32_t imm, const XReg &rm) { SveIndexGenImmRegInc(zd, imm, rm); }
void CodeGenerator::index(const ZRegB &zd, const WReg &rn, const int32_t imm) { SveIndexGenRegImmInc(zd, rn, imm); }
void CodeGenerator::index(const ZRegH &zd, const WReg &rn, const int32_t imm) { SveIndexGenRegImmInc(zd, rn, imm); }
void CodeGenerator::index(const ZRegS &zd, const WReg &rn, const int32_t imm) { SveIndexGenRegImmInc(zd, rn, imm); }
void CodeGenerator::index(const ZRegD &zd, const XReg &rn, const int32_t imm) { SveIndexGenRegImmInc(zd, rn, imm); }
void CodeGenerator::index(const ZRegB &zd, const WReg &rn, const WReg &rm) { SveIndexGenRegRegInc(zd, rn, rm); }
void CodeGenerator::index(const ZRegH &zd, const WReg &rn, const WReg &rm) { SveIndexGenRegRegInc(zd, rn, rm); }
void CodeGenerator::index(const ZRegS &zd, const WReg &rn, const WReg &rm) { SveIndexGenRegRegInc(zd, rn, rm); }
void CodeGenerator::index(const ZRegD &zd, const XReg &rn, const XReg &rm) { SveIndexGenRegRegInc(zd, rn, rm); }
void CodeGenerator::addvl(const XReg &xd, const XReg &xn, const int32_t imm) { SveStackFrameAdjust(0, xd, xn, imm); }
void CodeGenerator::addpl(const XReg &xd, const XReg &xn, const int32_t imm) { SveStackFrameAdjust(1, xd, xn, imm); }
void CodeGenerator::rdvl(const XReg &xd, const int32_t imm) { SveStackFrameSize(0, 31, xd, imm); }
void CodeGenerator::asr(const ZRegB &zd, const ZRegB &zn, const uint32_t amount) { SveBitwiseShByImmUnpred(0, zd, zn, amount); }
void CodeGenerator::asr(const ZRegH &zd, const ZRegH &zn, const uint32_t amount) { SveBitwiseShByImmUnpred(0, zd, zn, amount); }
void CodeGenerator::asr(const ZRegS &zd, const ZRegS &zn, const uint32_t amount) { SveBitwiseShByImmUnpred(0, zd, zn, amount); }
void CodeGenerator::asr(const ZRegD &zd, const ZRegD &zn, const uint32_t amount) { SveBitwiseShByImmUnpred(0, zd, zn, amount); }
void CodeGenerator::lsr(const ZRegB &zd, const ZRegB &zn, const uint32_t amount) { SveBitwiseShByImmUnpred(1, zd, zn, amount); }
void CodeGenerator::lsr(const ZRegH &zd, const ZRegH &zn, const uint32_t amount) { SveBitwiseShByImmUnpred(1, zd, zn, amount); }
void CodeGenerator::lsr(const ZRegS &zd, const ZRegS &zn, const uint32_t amount) { SveBitwiseShByImmUnpred(1, zd, zn, amount); }
void CodeGenerator::lsr(const ZRegD &zd, const ZRegD &zn, const uint32_t amount) { SveBitwiseShByImmUnpred(1, zd, zn, amount); }
void CodeGenerator::lsl(const ZRegB &zd, const ZRegB &zn, const uint32_t amount) { SveBitwiseShByImmUnpred(3, zd, zn, amount); }
void CodeGenerator::lsl(const ZRegH &zd, const ZRegH &zn, const uint32_t amount) { SveBitwiseShByImmUnpred(3, zd, zn, amount); }
void CodeGenerator::lsl(const ZRegS &zd, const ZRegS &zn, const uint32_t amount) { SveBitwiseShByImmUnpred(3, zd, zn, amount); }
void CodeGenerator::lsl(const ZRegD &zd, const ZRegD &zn, const uint32_t amount) { SveBitwiseShByImmUnpred(3, zd, zn, amount); }
void CodeGenerator::asr(const ZRegB &zd, const ZRegB &zn, const ZRegD &zm) { SveBitwiseShByWideElemUnPred(0, zd, zn, zm); }
void CodeGenerator::asr(const ZRegH &zd, const ZRegH &zn, const ZRegD &zm) { SveBitwiseShByWideElemUnPred(0, zd, zn, zm); }
void CodeGenerator::asr(const ZRegS &zd, const ZRegS &zn, const ZRegD &zm) { SveBitwiseShByWideElemUnPred(0, zd, zn, zm); }
void CodeGenerator::lsr(const ZRegB &zd, const ZRegB &zn, const ZRegD &zm) { SveBitwiseShByWideElemUnPred(1, zd, zn, zm); }
void CodeGenerator::lsr(const ZRegH &zd, const ZRegH &zn, const ZRegD &zm) { SveBitwiseShByWideElemUnPred(1, zd, zn, zm); }
void CodeGenerator::lsr(const ZRegS &zd, const ZRegS &zn, const ZRegD &zm) { SveBitwiseShByWideElemUnPred(1, zd, zn, zm); }
void CodeGenerator::lsl(const ZRegB &zd, const ZRegB &zn, const ZRegD &zm) { SveBitwiseShByWideElemUnPred(3, zd, zn, zm); }
void CodeGenerator::lsl(const ZRegH &zd, const ZRegH &zn, const ZRegD &zm) { SveBitwiseShByWideElemUnPred(3, zd, zn, zm); }
void CodeGenerator::lsl(const ZRegS &zd, const ZRegS &zn, const ZRegD &zm) { SveBitwiseShByWideElemUnPred(3, zd, zn, zm); }
void CodeGenerator::adr(const ZRegS &zd, const AdrVec &adr) { SveAddressGen(zd, adr); }
void CodeGenerator::adr(const ZRegD &zd, const AdrVec &adr) { SveAddressGen(zd, adr); }
void CodeGenerator::adr(const ZRegD &zd, const AdrVecU &adr) { SveAddressGen(zd, adr); }
void CodeGenerator::movprfx(const ZReg &zd, const ZReg &zn) { SveConstPrefUnpred(0, 0, zd, zn); }
void CodeGenerator::fexpa(const ZRegH &zd, const ZRegH &zn) { SveFpExpAccel(0, zd, zn); }
void CodeGenerator::fexpa(const ZRegS &zd, const ZRegS &zn) { SveFpExpAccel(0, zd, zn); }
void CodeGenerator::fexpa(const ZRegD &zd, const ZRegD &zn) { SveFpExpAccel(0, zd, zn); }
void CodeGenerator::ftssel(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveFpTrigSelCoef(0, zd, zn, zm); }
void CodeGenerator::ftssel(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveFpTrigSelCoef(0, zd, zn, zm); }
void CodeGenerator::ftssel(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveFpTrigSelCoef(0, zd, zn, zm); }
void CodeGenerator::cntb(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveElemCount(0, 0, xd, pat, mod, imm); }
void CodeGenerator::cnth(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveElemCount(1, 0, xd, pat, mod, imm); }
void CodeGenerator::cntw(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveElemCount(2, 0, xd, pat, mod, imm); }
void CodeGenerator::cntd(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveElemCount(3, 0, xd, pat, mod, imm); }
void CodeGenerator::incb(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecRegByElemCount(0, 0, xd, pat, mod, imm); }
void CodeGenerator::decb(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecRegByElemCount(0, 1, xd, pat, mod, imm); }
void CodeGenerator::inch(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecRegByElemCount(1, 0, xd, pat, mod, imm); }
void CodeGenerator::dech(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecRegByElemCount(1, 1, xd, pat, mod, imm); }
void CodeGenerator::incw(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecRegByElemCount(2, 0, xd, pat, mod, imm); }
void CodeGenerator::decw(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecRegByElemCount(2, 1, xd, pat, mod, imm); }
void CodeGenerator::incd(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecRegByElemCount(3, 0, xd, pat, mod, imm); }
void CodeGenerator::decd(const XReg &xd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecRegByElemCount(3, 1, xd, pat, mod, imm); }
void CodeGenerator::inch(const ZRegH &zd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecVecByElemCount(1, 0, zd, pat, mod, imm); }
void CodeGenerator::dech(const ZRegH &zd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecVecByElemCount(1, 1, zd, pat, mod, imm); }
void CodeGenerator::incw(const ZRegS &zd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecVecByElemCount(2, 0, zd, pat, mod, imm); }
void CodeGenerator::decw(const ZRegS &zd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecVecByElemCount(2, 1, zd, pat, mod, imm); }
void CodeGenerator::incd(const ZRegD &zd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecVecByElemCount(3, 0, zd, pat, mod, imm); }
void CodeGenerator::decd(const ZRegD &zd, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveIncDecVecByElemCount(3, 1, zd, pat, mod, imm); }
void CodeGenerator::sqincb(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(0, 0, 0, rdn, pat, mod, imm); }
void CodeGenerator::sqincb(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(0, 0, 0, rdn, pat, mod, imm); }
void CodeGenerator::uqincb(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(0, 0, 1, rdn, pat, mod, imm); }
void CodeGenerator::uqincb(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(0, 0, 1, rdn, pat, mod, imm); }
void CodeGenerator::sqdecb(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(0, 1, 0, rdn, pat, mod, imm); }
void CodeGenerator::sqdecb(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(0, 1, 0, rdn, pat, mod, imm); }
void CodeGenerator::uqdecb(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(0, 1, 1, rdn, pat, mod, imm); }
void CodeGenerator::uqdecb(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(0, 1, 1, rdn, pat, mod, imm); }
void CodeGenerator::sqinch(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(1, 0, 0, rdn, pat, mod, imm); }
void CodeGenerator::sqinch(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(1, 0, 0, rdn, pat, mod, imm); }
void CodeGenerator::uqinch(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(1, 0, 1, rdn, pat, mod, imm); }
void CodeGenerator::uqinch(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(1, 0, 1, rdn, pat, mod, imm); }
void CodeGenerator::sqdech(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(1, 1, 0, rdn, pat, mod, imm); }
void CodeGenerator::sqdech(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(1, 1, 0, rdn, pat, mod, imm); }
void CodeGenerator::uqdech(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(1, 1, 1, rdn, pat, mod, imm); }
void CodeGenerator::uqdech(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(1, 1, 1, rdn, pat, mod, imm); }
void CodeGenerator::sqincw(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(2, 0, 0, rdn, pat, mod, imm); }
void CodeGenerator::sqincw(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(2, 0, 0, rdn, pat, mod, imm); }
void CodeGenerator::uqincw(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(2, 0, 1, rdn, pat, mod, imm); }
void CodeGenerator::uqincw(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(2, 0, 1, rdn, pat, mod, imm); }
void CodeGenerator::sqdecw(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(2, 1, 0, rdn, pat, mod, imm); }
void CodeGenerator::sqdecw(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(2, 1, 0, rdn, pat, mod, imm); }
void CodeGenerator::uqdecw(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(2, 1, 1, rdn, pat, mod, imm); }
void CodeGenerator::uqdecw(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(2, 1, 1, rdn, pat, mod, imm); }
void CodeGenerator::sqincd(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(3, 0, 0, rdn, pat, mod, imm); }
void CodeGenerator::sqincd(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(3, 0, 0, rdn, pat, mod, imm); }
void CodeGenerator::uqincd(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(3, 0, 1, rdn, pat, mod, imm); }
void CodeGenerator::uqincd(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(3, 0, 1, rdn, pat, mod, imm); }
void CodeGenerator::sqdecd(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(3, 1, 0, rdn, pat, mod, imm); }
void CodeGenerator::sqdecd(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(3, 1, 0, rdn, pat, mod, imm); }
void CodeGenerator::uqdecd(const WReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(3, 1, 1, rdn, pat, mod, imm); }
void CodeGenerator::uqdecd(const XReg &rdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecRegByElemCount(3, 1, 1, rdn, pat, mod, imm); }
void CodeGenerator::sqinch(const ZRegH &zdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecVecByElemCount(1, 0, 0, zdn, pat, mod, imm); }
void CodeGenerator::uqinch(const ZRegH &zdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecVecByElemCount(1, 0, 1, zdn, pat, mod, imm); }
void CodeGenerator::sqdech(const ZRegH &zdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecVecByElemCount(1, 1, 0, zdn, pat, mod, imm); }
void CodeGenerator::uqdech(const ZRegH &zdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecVecByElemCount(1, 1, 1, zdn, pat, mod, imm); }
void CodeGenerator::sqincw(const ZRegS &zdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecVecByElemCount(2, 0, 0, zdn, pat, mod, imm); }
void CodeGenerator::uqincw(const ZRegS &zdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecVecByElemCount(2, 0, 1, zdn, pat, mod, imm); }
void CodeGenerator::sqdecw(const ZRegS &zdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecVecByElemCount(2, 1, 0, zdn, pat, mod, imm); }
void CodeGenerator::uqdecw(const ZRegS &zdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecVecByElemCount(2, 1, 1, zdn, pat, mod, imm); }
void CodeGenerator::sqincd(const ZRegD &zdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecVecByElemCount(3, 0, 0, zdn, pat, mod, imm); }
void CodeGenerator::uqincd(const ZRegD &zdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecVecByElemCount(3, 0, 1, zdn, pat, mod, imm); }
void CodeGenerator::sqdecd(const ZRegD &zdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecVecByElemCount(3, 1, 0, zdn, pat, mod, imm); }
void CodeGenerator::uqdecd(const ZRegD &zdn, const Pattern pat, const ExtMod mod, const uint32_t imm) { SveSatuIncDecVecByElemCount(3, 1, 1, zdn, pat, mod, imm); }
void CodeGenerator::orr(const ZRegB &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(0, zdn, imm); }
void CodeGenerator::orr(const ZRegH &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(0, zdn, imm); }
void CodeGenerator::orr(const ZRegS &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(0, zdn, imm); }
void CodeGenerator::orr(const ZRegD &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(0, zdn, imm); }
void CodeGenerator::orn(const ZRegB &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(0, zdn, ((-1) * imm - 1)); }
void CodeGenerator::orn(const ZRegH &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(0, zdn, ((-1) * imm - 1)); }
void CodeGenerator::orn(const ZRegS &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(0, zdn, ((-1) * imm - 1)); }
void CodeGenerator::orn(const ZRegD &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(0, zdn, ((-1) * imm - 1)); }
void CodeGenerator::eor(const ZRegB &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(1, zdn, imm); }
void CodeGenerator::eor(const ZRegH &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(1, zdn, imm); }
void CodeGenerator::eor(const ZRegS &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(1, zdn, imm); }
void CodeGenerator::eor(const ZRegD &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(1, zdn, imm); }
void CodeGenerator::eon(const ZRegB &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(1, zdn, ((-1) * imm - 1)); }
void CodeGenerator::eon(const ZRegH &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(1, zdn, ((-1) * imm - 1)); }
void CodeGenerator::eon(const ZRegS &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(1, zdn, ((-1) * imm - 1)); }
void CodeGenerator::eon(const ZRegD &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(1, zdn, ((-1) * imm - 1)); }
void CodeGenerator::and_(const ZRegB &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(2, zdn, imm); }
void CodeGenerator::and_(const ZRegH &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(2, zdn, imm); }
void CodeGenerator::and_(const ZRegS &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(2, zdn, imm); }
void CodeGenerator::and_(const ZRegD &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(2, zdn, imm); }
void CodeGenerator::bic(const ZRegB &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(2, zdn, ((-1) * imm - 1)); }
void CodeGenerator::bic(const ZRegH &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(2, zdn, ((-1) * imm - 1)); }
void CodeGenerator::bic(const ZRegS &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(2, zdn, ((-1) * imm - 1)); }
void CodeGenerator::bic(const ZRegD &zdn, const uint64_t imm) { SveBitwiseLogicalImmUnpred(2, zdn, ((-1) * imm - 1)); }
void CodeGenerator::dupm(const ZRegB &zd, const uint64_t imm) { SveBcBitmaskImm(zd, imm); }
void CodeGenerator::dupm(const ZRegH &zd, const uint64_t imm) { SveBcBitmaskImm(zd, imm); }
void CodeGenerator::dupm(const ZRegS &zd, const uint64_t imm) { SveBcBitmaskImm(zd, imm); }
void CodeGenerator::dupm(const ZRegD &zd, const uint64_t imm) { SveBcBitmaskImm(zd, imm); }
void CodeGenerator::mov(const ZRegB &zd, const uint64_t imm) { SveBcBitmaskImm(zd, genMoveMaskPrefferd(imm)); }
void CodeGenerator::mov(const ZRegH &zd, const uint64_t imm) { SveBcBitmaskImm(zd, genMoveMaskPrefferd(imm)); }
void CodeGenerator::mov(const ZRegS &zd, const uint64_t imm) { SveBcBitmaskImm(zd, genMoveMaskPrefferd(imm)); }
void CodeGenerator::mov(const ZRegD &zd, const uint64_t imm) { SveBcBitmaskImm(zd, genMoveMaskPrefferd(imm)); }
void CodeGenerator::fcpy(const ZRegH &zd, const _PReg &pg, const double imm) { SveCopyFpImmPred(zd, pg, imm); }
void CodeGenerator::fcpy(const ZRegS &zd, const _PReg &pg, const double imm) { SveCopyFpImmPred(zd, pg, imm); }
void CodeGenerator::fcpy(const ZRegD &zd, const _PReg &pg, const double imm) { SveCopyFpImmPred(zd, pg, imm); }
void CodeGenerator::fmov(const ZRegH &zd, const _PReg &pg, const double imm) { SveCopyFpImmPred(zd, pg, imm); }
void CodeGenerator::fmov(const ZRegS &zd, const _PReg &pg, const double imm) { SveCopyFpImmPred(zd, pg, imm); }
void CodeGenerator::fmov(const ZRegD &zd, const _PReg &pg, const double imm) { SveCopyFpImmPred(zd, pg, imm); }
void CodeGenerator::cpy(const ZRegB &zd, const _PReg &pg, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveCopyIntImmPred(zd, pg, imm, mod, sh); }
void CodeGenerator::cpy(const ZRegH &zd, const _PReg &pg, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveCopyIntImmPred(zd, pg, imm, mod, sh); }
void CodeGenerator::cpy(const ZRegS &zd, const _PReg &pg, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveCopyIntImmPred(zd, pg, imm, mod, sh); }
void CodeGenerator::cpy(const ZRegD &zd, const _PReg &pg, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveCopyIntImmPred(zd, pg, imm, mod, sh); }
void CodeGenerator::mov(const ZRegB &zd, const _PReg &pg, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveCopyIntImmPred(zd, pg, imm, mod, sh); }
void CodeGenerator::mov(const ZRegH &zd, const _PReg &pg, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveCopyIntImmPred(zd, pg, imm, mod, sh); }
void CodeGenerator::mov(const ZRegS &zd, const _PReg &pg, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveCopyIntImmPred(zd, pg, imm, mod, sh); }
void CodeGenerator::mov(const ZRegD &zd, const _PReg &pg, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveCopyIntImmPred(zd, pg, imm, mod, sh); }
void CodeGenerator::fmov(const ZRegH &zd, const _PReg &pg, const uint32_t imm) { SveCopyIntImmPred(zd, pg, imm, LSL, 0); }
void CodeGenerator::fmov(const ZRegS &zd, const _PReg &pg, const uint32_t imm) { SveCopyIntImmPred(zd, pg, imm, LSL, 0); }
void CodeGenerator::fmov(const ZRegD &zd, const _PReg &pg, const uint32_t imm) { SveCopyIntImmPred(zd, pg, imm, LSL, 0); }
void CodeGenerator::ext(const ZRegB &zdn, const ZRegB &zm, const uint32_t imm) { SveExtVec(zdn, zm, imm); }
void CodeGenerator::dup(const ZRegB &zd, const WReg &rn) { SveBcGeneralReg(zd, rn); }
void CodeGenerator::dup(const ZRegH &zd, const WReg &rn) { SveBcGeneralReg(zd, rn); }
void CodeGenerator::dup(const ZRegS &zd, const WReg &rn) { SveBcGeneralReg(zd, rn); }
void CodeGenerator::dup(const ZRegD &zd, const XReg &rn) { SveBcGeneralReg(zd, rn); }
void CodeGenerator::mov(const ZRegB &zd, const WReg &rn) { SveBcGeneralReg(zd, rn); }
void CodeGenerator::mov(const ZRegH &zd, const WReg &rn) { SveBcGeneralReg(zd, rn); }
void CodeGenerator::mov(const ZRegS &zd, const WReg &rn) { SveBcGeneralReg(zd, rn); }
void CodeGenerator::mov(const ZRegD &zd, const XReg &rn) { SveBcGeneralReg(zd, rn); }
void CodeGenerator::dup(const ZRegB &zd, const ZRegBElem &zn) { SveBcIndexedElem(zd, zn); }
void CodeGenerator::dup(const ZRegH &zd, const ZRegHElem &zn) { SveBcIndexedElem(zd, zn); }
void CodeGenerator::dup(const ZRegS &zd, const ZRegSElem &zn) { SveBcIndexedElem(zd, zn); }
void CodeGenerator::dup(const ZRegD &zd, const ZRegDElem &zn) { SveBcIndexedElem(zd, zn); }
void CodeGenerator::dup(const ZRegQ &zd, const ZRegQElem &zn) { SveBcIndexedElem(zd, zn); }
void CodeGenerator::mov(const ZRegB &zd, const ZRegBElem &zn) { SveBcIndexedElem(zd, zn); }
void CodeGenerator::mov(const ZRegH &zd, const ZRegHElem &zn) { SveBcIndexedElem(zd, zn); }
void CodeGenerator::mov(const ZRegS &zd, const ZRegSElem &zn) { SveBcIndexedElem(zd, zn); }
void CodeGenerator::mov(const ZRegD &zd, const ZRegDElem &zn) { SveBcIndexedElem(zd, zn); }
void CodeGenerator::mov(const ZRegQ &zd, const ZRegQElem &zn) { SveBcIndexedElem(zd, zn); }
void CodeGenerator::mov(const ZRegB &zd, const BReg &vn) { SveBcIndexedElem(zd, ZRegElem(vn.getIdx(), 0, vn.getBit())); }
void CodeGenerator::mov(const ZRegH &zd, const HReg &vn) { SveBcIndexedElem(zd, ZRegElem(vn.getIdx(), 0, vn.getBit())); }
void CodeGenerator::mov(const ZRegS &zd, const SReg &vn) { SveBcIndexedElem(zd, ZRegElem(vn.getIdx(), 0, vn.getBit())); }
void CodeGenerator::mov(const ZRegD &zd, const DReg &vn) { SveBcIndexedElem(zd, ZRegElem(vn.getIdx(), 0, vn.getBit())); }
void CodeGenerator::mov(const ZRegQ &zd, const QReg &vn) { SveBcIndexedElem(zd, ZRegElem(vn.getIdx(), 0, vn.getBit())); }
void CodeGenerator::insr(const ZRegB &zdn, const BReg &vm) { SveInsSimdFpSclarReg(zdn, vm); }
void CodeGenerator::insr(const ZRegH &zdn, const HReg &vm) { SveInsSimdFpSclarReg(zdn, vm); }
void CodeGenerator::insr(const ZRegS &zdn, const SReg &vm) { SveInsSimdFpSclarReg(zdn, vm); }
void CodeGenerator::insr(const ZRegD &zdn, const DReg &vm) { SveInsSimdFpSclarReg(zdn, vm); }
void CodeGenerator::insr(const ZRegB &zdn, const WReg &rm) { SveInsGeneralReg(zdn, rm); }
void CodeGenerator::insr(const ZRegH &zdn, const WReg &rm) { SveInsGeneralReg(zdn, rm); }
void CodeGenerator::insr(const ZRegS &zdn, const WReg &rm) { SveInsGeneralReg(zdn, rm); }
void CodeGenerator::insr(const ZRegD &zdn, const XReg &rm) { SveInsGeneralReg(zdn, rm); }
void CodeGenerator::rev(const ZRegB &zd, const ZRegB &zn) { SveRevVecElem(zd, zn); }
void CodeGenerator::rev(const ZRegH &zd, const ZRegH &zn) { SveRevVecElem(zd, zn); }
void CodeGenerator::rev(const ZRegS &zd, const ZRegS &zn) { SveRevVecElem(zd, zn); }
void CodeGenerator::rev(const ZRegD &zd, const ZRegD &zn) { SveRevVecElem(zd, zn); }
void CodeGenerator::tbl(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) { SveTableLookup(zd, zn, zm); }
void CodeGenerator::tbl(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveTableLookup(zd, zn, zm); }
void CodeGenerator::tbl(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveTableLookup(zd, zn, zm); }
void CodeGenerator::tbl(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveTableLookup(zd, zn, zm); }
void CodeGenerator::sunpklo(const ZRegH &zd, const ZRegB &zn) { SveUnpackVecElem(0, 0, zd, zn); }
void CodeGenerator::sunpklo(const ZRegS &zd, const ZRegH &zn) { SveUnpackVecElem(0, 0, zd, zn); }
void CodeGenerator::sunpklo(const ZRegD &zd, const ZRegS &zn) { SveUnpackVecElem(0, 0, zd, zn); }
void CodeGenerator::sunpkhi(const ZRegH &zd, const ZRegB &zn) { SveUnpackVecElem(0, 1, zd, zn); }
void CodeGenerator::sunpkhi(const ZRegS &zd, const ZRegH &zn) { SveUnpackVecElem(0, 1, zd, zn); }
void CodeGenerator::sunpkhi(const ZRegD &zd, const ZRegS &zn) { SveUnpackVecElem(0, 1, zd, zn); }
void CodeGenerator::uunpklo(const ZRegH &zd, const ZRegB &zn) { SveUnpackVecElem(1, 0, zd, zn); }
void CodeGenerator::uunpklo(const ZRegS &zd, const ZRegH &zn) { SveUnpackVecElem(1, 0, zd, zn); }
void CodeGenerator::uunpklo(const ZRegD &zd, const ZRegS &zn) { SveUnpackVecElem(1, 0, zd, zn); }
void CodeGenerator::uunpkhi(const ZRegH &zd, const ZRegB &zn) { SveUnpackVecElem(1, 1, zd, zn); }
void CodeGenerator::uunpkhi(const ZRegS &zd, const ZRegH &zn) { SveUnpackVecElem(1, 1, zd, zn); }
void CodeGenerator::uunpkhi(const ZRegD &zd, const ZRegS &zn) { SveUnpackVecElem(1, 1, zd, zn); }
void CodeGenerator::zip1(const PRegB &pd, const PRegB &pn, const PRegB &pm) { SvePermutePredElem(0, 0, pd, pn, pm); }
void CodeGenerator::zip1(const PRegH &pd, const PRegH &pn, const PRegH &pm) { SvePermutePredElem(0, 0, pd, pn, pm); }
void CodeGenerator::zip1(const PRegS &pd, const PRegS &pn, const PRegS &pm) { SvePermutePredElem(0, 0, pd, pn, pm); }
void CodeGenerator::zip1(const PRegD &pd, const PRegD &pn, const PRegD &pm) { SvePermutePredElem(0, 0, pd, pn, pm); }
void CodeGenerator::zip2(const PRegB &pd, const PRegB &pn, const PRegB &pm) { SvePermutePredElem(0, 1, pd, pn, pm); }
void CodeGenerator::zip2(const PRegH &pd, const PRegH &pn, const PRegH &pm) { SvePermutePredElem(0, 1, pd, pn, pm); }
void CodeGenerator::zip2(const PRegS &pd, const PRegS &pn, const PRegS &pm) { SvePermutePredElem(0, 1, pd, pn, pm); }
void CodeGenerator::zip2(const PRegD &pd, const PRegD &pn, const PRegD &pm) { SvePermutePredElem(0, 1, pd, pn, pm); }
void CodeGenerator::uzp1(const PRegB &pd, const PRegB &pn, const PRegB &pm) { SvePermutePredElem(1, 0, pd, pn, pm); }
void CodeGenerator::uzp1(const PRegH &pd, const PRegH &pn, const PRegH &pm) { SvePermutePredElem(1, 0, pd, pn, pm); }
void CodeGenerator::uzp1(const PRegS &pd, const PRegS &pn, const PRegS &pm) { SvePermutePredElem(1, 0, pd, pn, pm); }
void CodeGenerator::uzp1(const PRegD &pd, const PRegD &pn, const PRegD &pm) { SvePermutePredElem(1, 0, pd, pn, pm); }
void CodeGenerator::uzp2(const PRegB &pd, const PRegB &pn, const PRegB &pm) { SvePermutePredElem(1, 1, pd, pn, pm); }
void CodeGenerator::uzp2(const PRegH &pd, const PRegH &pn, const PRegH &pm) { SvePermutePredElem(1, 1, pd, pn, pm); }
void CodeGenerator::uzp2(const PRegS &pd, const PRegS &pn, const PRegS &pm) { SvePermutePredElem(1, 1, pd, pn, pm); }
void CodeGenerator::uzp2(const PRegD &pd, const PRegD &pn, const PRegD &pm) { SvePermutePredElem(1, 1, pd, pn, pm); }
void CodeGenerator::trn1(const PRegB &pd, const PRegB &pn, const PRegB &pm) { SvePermutePredElem(2, 0, pd, pn, pm); }
void CodeGenerator::trn1(const PRegH &pd, const PRegH &pn, const PRegH &pm) { SvePermutePredElem(2, 0, pd, pn, pm); }
void CodeGenerator::trn1(const PRegS &pd, const PRegS &pn, const PRegS &pm) { SvePermutePredElem(2, 0, pd, pn, pm); }
void CodeGenerator::trn1(const PRegD &pd, const PRegD &pn, const PRegD &pm) { SvePermutePredElem(2, 0, pd, pn, pm); }
void CodeGenerator::trn2(const PRegB &pd, const PRegB &pn, const PRegB &pm) { SvePermutePredElem(2, 1, pd, pn, pm); }
void CodeGenerator::trn2(const PRegH &pd, const PRegH &pn, const PRegH &pm) { SvePermutePredElem(2, 1, pd, pn, pm); }
void CodeGenerator::trn2(const PRegS &pd, const PRegS &pn, const PRegS &pm) { SvePermutePredElem(2, 1, pd, pn, pm); }
void CodeGenerator::trn2(const PRegD &pd, const PRegD &pn, const PRegD &pm) { SvePermutePredElem(2, 1, pd, pn, pm); }
void CodeGenerator::rev(const PRegB &pd, const PRegB &pn) { SveRevPredElem(pd, pn); }
void CodeGenerator::rev(const PRegH &pd, const PRegH &pn) { SveRevPredElem(pd, pn); }
void CodeGenerator::rev(const PRegS &pd, const PRegS &pn) { SveRevPredElem(pd, pn); }
void CodeGenerator::rev(const PRegD &pd, const PRegD &pn) { SveRevPredElem(pd, pn); }
void CodeGenerator::punpklo(const PRegH &pd, const PRegB &pn) { SveUnpackPredElem(0, pd, pn); }
void CodeGenerator::punpkhi(const PRegH &pd, const PRegB &pn) { SveUnpackPredElem(1, pd, pn); }
void CodeGenerator::zip1(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) { SvePermuteVecElem(0, zd, zn, zm); }
void CodeGenerator::zip1(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SvePermuteVecElem(0, zd, zn, zm); }
void CodeGenerator::zip1(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SvePermuteVecElem(0, zd, zn, zm); }
void CodeGenerator::zip1(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SvePermuteVecElem(0, zd, zn, zm); }
void CodeGenerator::zip2(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) { SvePermuteVecElem(1, zd, zn, zm); }
void CodeGenerator::zip2(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SvePermuteVecElem(1, zd, zn, zm); }
void CodeGenerator::zip2(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SvePermuteVecElem(1, zd, zn, zm); }
void CodeGenerator::zip2(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SvePermuteVecElem(1, zd, zn, zm); }
void CodeGenerator::uzp1(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) { SvePermuteVecElem(2, zd, zn, zm); }
void CodeGenerator::uzp1(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SvePermuteVecElem(2, zd, zn, zm); }
void CodeGenerator::uzp1(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SvePermuteVecElem(2, zd, zn, zm); }
void CodeGenerator::uzp1(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SvePermuteVecElem(2, zd, zn, zm); }
void CodeGenerator::uzp2(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) { SvePermuteVecElem(3, zd, zn, zm); }
void CodeGenerator::uzp2(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SvePermuteVecElem(3, zd, zn, zm); }
void CodeGenerator::uzp2(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SvePermuteVecElem(3, zd, zn, zm); }
void CodeGenerator::uzp2(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SvePermuteVecElem(3, zd, zn, zm); }
void CodeGenerator::trn1(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) { SvePermuteVecElem(4, zd, zn, zm); }
void CodeGenerator::trn1(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SvePermuteVecElem(4, zd, zn, zm); }
void CodeGenerator::trn1(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SvePermuteVecElem(4, zd, zn, zm); }
void CodeGenerator::trn1(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SvePermuteVecElem(4, zd, zn, zm); }
void CodeGenerator::trn2(const ZRegB &zd, const ZRegB &zn, const ZRegB &zm) { SvePermuteVecElem(5, zd, zn, zm); }
void CodeGenerator::trn2(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SvePermuteVecElem(5, zd, zn, zm); }
void CodeGenerator::trn2(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SvePermuteVecElem(5, zd, zn, zm); }
void CodeGenerator::trn2(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SvePermuteVecElem(5, zd, zn, zm); }
void CodeGenerator::compact(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveCompressActElem(zd, pg, zn); }
void CodeGenerator::compact(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveCompressActElem(zd, pg, zn); }
void CodeGenerator::clasta(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveCondBcElemToVec(0, zdn, pg, zm); }
void CodeGenerator::clasta(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveCondBcElemToVec(0, zdn, pg, zm); }
void CodeGenerator::clasta(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveCondBcElemToVec(0, zdn, pg, zm); }
void CodeGenerator::clasta(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveCondBcElemToVec(0, zdn, pg, zm); }
void CodeGenerator::clastb(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveCondBcElemToVec(1, zdn, pg, zm); }
void CodeGenerator::clastb(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveCondBcElemToVec(1, zdn, pg, zm); }
void CodeGenerator::clastb(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveCondBcElemToVec(1, zdn, pg, zm); }
void CodeGenerator::clastb(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveCondBcElemToVec(1, zdn, pg, zm); }
void CodeGenerator::clasta(const BReg &vdn, const _PReg &pg, const ZRegB &zm) { SveCondExtElemToSimdFpScalar(0, vdn, pg, zm); }
void CodeGenerator::clasta(const HReg &vdn, const _PReg &pg, const ZRegH &zm) { SveCondExtElemToSimdFpScalar(0, vdn, pg, zm); }
void CodeGenerator::clasta(const SReg &vdn, const _PReg &pg, const ZRegS &zm) { SveCondExtElemToSimdFpScalar(0, vdn, pg, zm); }
void CodeGenerator::clasta(const DReg &vdn, const _PReg &pg, const ZRegD &zm) { SveCondExtElemToSimdFpScalar(0, vdn, pg, zm); }
void CodeGenerator::clastb(const BReg &vdn, const _PReg &pg, const ZRegB &zm) { SveCondExtElemToSimdFpScalar(1, vdn, pg, zm); }
void CodeGenerator::clastb(const HReg &vdn, const _PReg &pg, const ZRegH &zm) { SveCondExtElemToSimdFpScalar(1, vdn, pg, zm); }
void CodeGenerator::clastb(const SReg &vdn, const _PReg &pg, const ZRegS &zm) { SveCondExtElemToSimdFpScalar(1, vdn, pg, zm); }
void CodeGenerator::clastb(const DReg &vdn, const _PReg &pg, const ZRegD &zm) { SveCondExtElemToSimdFpScalar(1, vdn, pg, zm); }
void CodeGenerator::clasta(const WReg &rdn, const _PReg &pg, const ZRegB &zm) { SveCondExtElemToGeneralReg(0, rdn, pg, zm); }
void CodeGenerator::clasta(const WReg &rdn, const _PReg &pg, const ZRegH &zm) { SveCondExtElemToGeneralReg(0, rdn, pg, zm); }
void CodeGenerator::clasta(const WReg &rdn, const _PReg &pg, const ZRegS &zm) { SveCondExtElemToGeneralReg(0, rdn, pg, zm); }
void CodeGenerator::clasta(const XReg &rdn, const _PReg &pg, const ZRegD &zm) { SveCondExtElemToGeneralReg(0, rdn, pg, zm); }
void CodeGenerator::clastb(const WReg &rdn, const _PReg &pg, const ZRegB &zm) { SveCondExtElemToGeneralReg(1, rdn, pg, zm); }
void CodeGenerator::clastb(const WReg &rdn, const _PReg &pg, const ZRegH &zm) { SveCondExtElemToGeneralReg(1, rdn, pg, zm); }
void CodeGenerator::clastb(const WReg &rdn, const _PReg &pg, const ZRegS &zm) { SveCondExtElemToGeneralReg(1, rdn, pg, zm); }
void CodeGenerator::clastb(const XReg &rdn, const _PReg &pg, const ZRegD &zm) { SveCondExtElemToGeneralReg(1, rdn, pg, zm); }
void CodeGenerator::cpy(const ZRegB &zd, const _PReg &pg, const BReg &vn) { SveCopySimdFpScalarToVecPred(zd, pg, vn); }
void CodeGenerator::cpy(const ZRegH &zd, const _PReg &pg, const HReg &vn) { SveCopySimdFpScalarToVecPred(zd, pg, vn); }
void CodeGenerator::cpy(const ZRegS &zd, const _PReg &pg, const SReg &vn) { SveCopySimdFpScalarToVecPred(zd, pg, vn); }
void CodeGenerator::cpy(const ZRegD &zd, const _PReg &pg, const DReg &vn) { SveCopySimdFpScalarToVecPred(zd, pg, vn); }
void CodeGenerator::mov(const ZRegB &zd, const _PReg &pg, const BReg &vn) { SveCopySimdFpScalarToVecPred(zd, pg, vn); }
void CodeGenerator::mov(const ZRegH &zd, const _PReg &pg, const HReg &vn) { SveCopySimdFpScalarToVecPred(zd, pg, vn); }
void CodeGenerator::mov(const ZRegS &zd, const _PReg &pg, const SReg &vn) { SveCopySimdFpScalarToVecPred(zd, pg, vn); }
void CodeGenerator::mov(const ZRegD &zd, const _PReg &pg, const DReg &vn) { SveCopySimdFpScalarToVecPred(zd, pg, vn); }
void CodeGenerator::cpy(const ZRegB &zd, const _PReg &pg, const WReg &rn) { SveCopyGeneralRegToVecPred(zd, pg, rn); }
void CodeGenerator::cpy(const ZRegH &zd, const _PReg &pg, const WReg &rn) { SveCopyGeneralRegToVecPred(zd, pg, rn); }
void CodeGenerator::cpy(const ZRegS &zd, const _PReg &pg, const WReg &rn) { SveCopyGeneralRegToVecPred(zd, pg, rn); }
void CodeGenerator::cpy(const ZRegD &zd, const _PReg &pg, const XReg &rn) { SveCopyGeneralRegToVecPred(zd, pg, rn); }
void CodeGenerator::mov(const ZRegB &zd, const _PReg &pg, const WReg &rn) { SveCopyGeneralRegToVecPred(zd, pg, rn); }
void CodeGenerator::mov(const ZRegH &zd, const _PReg &pg, const WReg &rn) { SveCopyGeneralRegToVecPred(zd, pg, rn); }
void CodeGenerator::mov(const ZRegS &zd, const _PReg &pg, const WReg &rn) { SveCopyGeneralRegToVecPred(zd, pg, rn); }
void CodeGenerator::mov(const ZRegD &zd, const _PReg &pg, const XReg &rn) { SveCopyGeneralRegToVecPred(zd, pg, rn); }
void CodeGenerator::lasta(const BReg &vd, const _PReg &pg, const ZRegB &zn) { SveExtElemToSimdFpScalar(0, vd, pg, zn); }
void CodeGenerator::lasta(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveExtElemToSimdFpScalar(0, vd, pg, zn); }
void CodeGenerator::lasta(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveExtElemToSimdFpScalar(0, vd, pg, zn); }
void CodeGenerator::lasta(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveExtElemToSimdFpScalar(0, vd, pg, zn); }
void CodeGenerator::lastb(const BReg &vd, const _PReg &pg, const ZRegB &zn) { SveExtElemToSimdFpScalar(1, vd, pg, zn); }
void CodeGenerator::lastb(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveExtElemToSimdFpScalar(1, vd, pg, zn); }
void CodeGenerator::lastb(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveExtElemToSimdFpScalar(1, vd, pg, zn); }
void CodeGenerator::lastb(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveExtElemToSimdFpScalar(1, vd, pg, zn); }
void CodeGenerator::lasta(const WReg &rd, const _PReg &pg, const ZRegB &zn) { SveExtElemToGeneralReg(0, rd, pg, zn); }
void CodeGenerator::lasta(const WReg &rd, const _PReg &pg, const ZRegH &zn) { SveExtElemToGeneralReg(0, rd, pg, zn); }
void CodeGenerator::lasta(const WReg &rd, const _PReg &pg, const ZRegS &zn) { SveExtElemToGeneralReg(0, rd, pg, zn); }
void CodeGenerator::lasta(const XReg &rd, const _PReg &pg, const ZRegD &zn) { SveExtElemToGeneralReg(0, rd, pg, zn); }
void CodeGenerator::lastb(const WReg &rd, const _PReg &pg, const ZRegB &zn) { SveExtElemToGeneralReg(1, rd, pg, zn); }
void CodeGenerator::lastb(const WReg &rd, const _PReg &pg, const ZRegH &zn) { SveExtElemToGeneralReg(1, rd, pg, zn); }
void CodeGenerator::lastb(const WReg &rd, const _PReg &pg, const ZRegS &zn) { SveExtElemToGeneralReg(1, rd, pg, zn); }
void CodeGenerator::lastb(const XReg &rd, const _PReg &pg, const ZRegD &zn) { SveExtElemToGeneralReg(1, rd, pg, zn); }
void CodeGenerator::revb(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveRevWithinElem(0, zd, pg, zn); }
void CodeGenerator::revb(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveRevWithinElem(0, zd, pg, zn); }
void CodeGenerator::revb(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveRevWithinElem(0, zd, pg, zn); }
void CodeGenerator::revh(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveRevWithinElem(1, zd, pg, zn); }
void CodeGenerator::revh(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveRevWithinElem(1, zd, pg, zn); }
void CodeGenerator::revw(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveRevWithinElem(2, zd, pg, zn); }
void CodeGenerator::rbit(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) { SveRevWithinElem(3, zd, pg, zn); }
void CodeGenerator::rbit(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveRevWithinElem(3, zd, pg, zn); }
void CodeGenerator::rbit(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveRevWithinElem(3, zd, pg, zn); }
void CodeGenerator::rbit(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveRevWithinElem(3, zd, pg, zn); }
void CodeGenerator::splice(const ZRegB &zdn, const _PReg &pg, const ZRegB &zm) { SveSelVecSplice(zdn, pg, zm); }
void CodeGenerator::splice(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveSelVecSplice(zdn, pg, zm); }
void CodeGenerator::splice(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveSelVecSplice(zdn, pg, zm); }
void CodeGenerator::splice(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveSelVecSplice(zdn, pg, zm); }
void CodeGenerator::sel(const ZRegB &zd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) { SveSelVecElemPred(zd, pg, zn, zm); }
void CodeGenerator::sel(const ZRegH &zd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveSelVecElemPred(zd, pg, zn, zm); }
void CodeGenerator::sel(const ZRegS &zd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveSelVecElemPred(zd, pg, zn, zm); }
void CodeGenerator::sel(const ZRegD &zd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveSelVecElemPred(zd, pg, zn, zm); }
void CodeGenerator::mov(const ZRegB &zd, const _PReg &pg, const ZRegB &zn) { SveSelVecElemPred(zd, pg, zn, zd); }
void CodeGenerator::mov(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveSelVecElemPred(zd, pg, zn, zd); }
void CodeGenerator::mov(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveSelVecElemPred(zd, pg, zn, zd); }
void CodeGenerator::mov(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveSelVecElemPred(zd, pg, zn, zd); }
void CodeGenerator::cmphs(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) { SveIntCompVec(0, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmphs(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveIntCompVec(0, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmphs(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveIntCompVec(0, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmphs(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveIntCompVec(0, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmphi(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) { SveIntCompVec(0, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmphi(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveIntCompVec(0, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmphi(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveIntCompVec(0, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmphi(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveIntCompVec(0, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpeq(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) { SveIntCompVec(0, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpeq(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) { SveIntCompVec(0, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpeq(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) { SveIntCompVec(0, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpne(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) { SveIntCompVec(0, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpne(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) { SveIntCompVec(0, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpne(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) { SveIntCompVec(0, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpge(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) { SveIntCompVec(1, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpge(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveIntCompVec(1, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpge(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveIntCompVec(1, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpge(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveIntCompVec(1, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpgt(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) { SveIntCompVec(1, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpgt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveIntCompVec(1, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpgt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveIntCompVec(1, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpgt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveIntCompVec(1, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpeq(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) { SveIntCompVec(1, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpeq(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveIntCompVec(1, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpeq(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveIntCompVec(1, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpeq(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveIntCompVec(1, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpne(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) { SveIntCompVec(1, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpne(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveIntCompVec(1, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpne(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveIntCompVec(1, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpne(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveIntCompVec(1, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::cmple(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) { SveIntCompVec(1, 0, 0, pd, pg, zm, zn); }
void CodeGenerator::cmple(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveIntCompVec(1, 0, 0, pd, pg, zm, zn); }
void CodeGenerator::cmple(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveIntCompVec(1, 0, 0, pd, pg, zm, zn); }
void CodeGenerator::cmple(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveIntCompVec(1, 0, 0, pd, pg, zm, zn); }
void CodeGenerator::cmplo(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) { SveIntCompVec(0, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::cmplo(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveIntCompVec(0, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::cmplo(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveIntCompVec(0, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::cmplo(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveIntCompVec(0, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::cmpls(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) { SveIntCompVec(0, 0, 0, pd, pg, zm, zn); }
void CodeGenerator::cmpls(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveIntCompVec(0, 0, 0, pd, pg, zm, zn); }
void CodeGenerator::cmpls(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveIntCompVec(0, 0, 0, pd, pg, zm, zn); }
void CodeGenerator::cmpls(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveIntCompVec(0, 0, 0, pd, pg, zm, zn); }
void CodeGenerator::cmplt(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegB &zm) { SveIntCompVec(1, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::cmplt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveIntCompVec(1, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::cmplt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveIntCompVec(1, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::cmplt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveIntCompVec(1, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::cmpge(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) { SveIntCompWideElem(0, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpge(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) { SveIntCompWideElem(0, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpge(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) { SveIntCompWideElem(0, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpgt(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) { SveIntCompWideElem(0, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpgt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) { SveIntCompWideElem(0, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpgt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) { SveIntCompWideElem(0, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmplt(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) { SveIntCompWideElem(0, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::cmplt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) { SveIntCompWideElem(0, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::cmplt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) { SveIntCompWideElem(0, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::cmple(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) { SveIntCompWideElem(0, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::cmple(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) { SveIntCompWideElem(0, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::cmple(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) { SveIntCompWideElem(0, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::cmphs(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) { SveIntCompWideElem(1, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmphs(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) { SveIntCompWideElem(1, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmphs(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) { SveIntCompWideElem(1, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::cmphi(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) { SveIntCompWideElem(1, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmphi(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) { SveIntCompWideElem(1, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmphi(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) { SveIntCompWideElem(1, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::cmplo(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) { SveIntCompWideElem(1, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::cmplo(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) { SveIntCompWideElem(1, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::cmplo(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) { SveIntCompWideElem(1, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::cmpls(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const ZRegD &zm) { SveIntCompWideElem(1, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpls(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegD &zm) { SveIntCompWideElem(1, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::cmpls(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegD &zm) { SveIntCompWideElem(1, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::cmphs(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const uint32_t imm) { SveIntCompUImm(0, 0, pd, pg, zn, imm); }
void CodeGenerator::cmphs(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const uint32_t imm) { SveIntCompUImm(0, 0, pd, pg, zn, imm); }
void CodeGenerator::cmphs(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const uint32_t imm) { SveIntCompUImm(0, 0, pd, pg, zn, imm); }
void CodeGenerator::cmphs(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const uint32_t imm) { SveIntCompUImm(0, 0, pd, pg, zn, imm); }
void CodeGenerator::cmphi(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const uint32_t imm) { SveIntCompUImm(0, 1, pd, pg, zn, imm); }
void CodeGenerator::cmphi(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const uint32_t imm) { SveIntCompUImm(0, 1, pd, pg, zn, imm); }
void CodeGenerator::cmphi(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const uint32_t imm) { SveIntCompUImm(0, 1, pd, pg, zn, imm); }
void CodeGenerator::cmphi(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const uint32_t imm) { SveIntCompUImm(0, 1, pd, pg, zn, imm); }
void CodeGenerator::cmplo(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const uint32_t imm) { SveIntCompUImm(1, 0, pd, pg, zn, imm); }
void CodeGenerator::cmplo(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const uint32_t imm) { SveIntCompUImm(1, 0, pd, pg, zn, imm); }
void CodeGenerator::cmplo(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const uint32_t imm) { SveIntCompUImm(1, 0, pd, pg, zn, imm); }
void CodeGenerator::cmplo(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const uint32_t imm) { SveIntCompUImm(1, 0, pd, pg, zn, imm); }
void CodeGenerator::cmpls(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const uint32_t imm) { SveIntCompUImm(1, 1, pd, pg, zn, imm); }
void CodeGenerator::cmpls(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const uint32_t imm) { SveIntCompUImm(1, 1, pd, pg, zn, imm); }
void CodeGenerator::cmpls(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const uint32_t imm) { SveIntCompUImm(1, 1, pd, pg, zn, imm); }
void CodeGenerator::cmpls(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const uint32_t imm) { SveIntCompUImm(1, 1, pd, pg, zn, imm); }
void CodeGenerator::and_(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(0, 0, 0, 0, pd, pg, pn, pm); }
void CodeGenerator::mov(const PRegB &pd, const _PReg &pg, const PRegB &pn) { SvePredLOp(0, 0, pg.isM(), pg.isM(), pd, pg, pn, (pg.isZ()) ? pn : pd); }
void CodeGenerator::bic(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(0, 0, 0, 1, pd, pg, pn, pm); }
void CodeGenerator::eor(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(0, 0, 1, 0, pd, pg, pn, pm); }
void CodeGenerator::not_(const PRegB &pd, const _PReg &pg, const PRegB &pn) { SvePredLOp(0, 0, 1, 0, pd, pg, pn, pg); }
void CodeGenerator::sel(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(0, 0, 1, 1, pd, pg, pn, pm); }
void CodeGenerator::ands(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(0, 1, 0, 0, pd, pg, pn, pm); }
void CodeGenerator::movs(const PRegB &pd, const _PReg &pg, const PRegB &pn) { SvePredLOp(0, 1, 0, 0, pd, pg, pn, pn); }
void CodeGenerator::bics(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(0, 1, 0, 1, pd, pg, pn, pm); }
void CodeGenerator::eors(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(0, 1, 1, 0, pd, pg, pn, pm); }
void CodeGenerator::nots(const PRegB &pd, const _PReg &pg, const PRegB &pn) { SvePredLOp(0, 1, 1, 0, pd, pg, pn, pg); }
void CodeGenerator::orr(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(1, 0, 0, 0, pd, pg, pn, pm); }
void CodeGenerator::mov(const PRegB &pd, const PRegB &pn) { SvePredLOp(1, 0, 0, 0, pd, pn, pn, pn); }
void CodeGenerator::orn(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(1, 0, 0, 1, pd, pg, pn, pm); }
void CodeGenerator::nor(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(1, 0, 1, 0, pd, pg, pn, pm); }
void CodeGenerator::nand(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(1, 0, 1, 1, pd, pg, pn, pm); }
void CodeGenerator::orrs(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(1, 1, 0, 0, pd, pg, pn, pm); }
void CodeGenerator::movs(const PRegB &pd, const PRegB &pn) { SvePredLOp(1, 1, 0, 0, pd, pn, pn, pn); }
void CodeGenerator::orns(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(1, 1, 0, 1, pd, pg, pn, pm); }
void CodeGenerator::nors(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(1, 1, 1, 0, pd, pg, pn, pm); }
void CodeGenerator::nands(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePredLOp(1, 1, 1, 1, pd, pg, pn, pm); }
void CodeGenerator::brkpa(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePropagateBreakPrevPtn(0, 0, 0, pd, pg, pn, pm); }
void CodeGenerator::brkpb(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePropagateBreakPrevPtn(0, 0, 1, pd, pg, pn, pm); }
void CodeGenerator::brkpas(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePropagateBreakPrevPtn(0, 1, 0, pd, pg, pn, pm); }
void CodeGenerator::brkpbs(const PRegB &pd, const _PReg &pg, const PRegB &pn, const PRegB &pm) { SvePropagateBreakPrevPtn(0, 1, 1, pd, pg, pn, pm); }
void CodeGenerator::brka(const PRegB &pd, const _PReg &pg, const PRegB &pn) { SvePartitionBreakCond(0, 0, pd, pg, pn); }
void CodeGenerator::brkas(const PRegB &pd, const _PReg &pg, const PRegB &pn) { SvePartitionBreakCond(0, 1, pd, pg, pn); }
void CodeGenerator::brkb(const PRegB &pd, const _PReg &pg, const PRegB &pn) { SvePartitionBreakCond(1, 0, pd, pg, pn); }
void CodeGenerator::brkbs(const PRegB &pd, const _PReg &pg, const PRegB &pn) { SvePartitionBreakCond(1, 1, pd, pg, pn); }
void CodeGenerator::brkn(const PRegB &pdm, const _PReg &pg, const PRegB &pn) { SvePropagateBreakNextPart(0, pdm, pg, pn); }
void CodeGenerator::brkns(const PRegB &pdm, const _PReg &pg, const PRegB &pn) { SvePropagateBreakNextPart(1, pdm, pg, pn); }
void CodeGenerator::pfirst(const PRegB &pdn, const _PReg &pg) { SvePredFirstAct(0, 1, pdn, pg); }
void CodeGenerator::ptrue(const PRegB &pd, const Pattern pat) { SvePredInit(0, pd, pat); }
void CodeGenerator::ptrue(const PRegH &pd, const Pattern pat) { SvePredInit(0, pd, pat); }
void CodeGenerator::ptrue(const PRegS &pd, const Pattern pat) { SvePredInit(0, pd, pat); }
void CodeGenerator::ptrue(const PRegD &pd, const Pattern pat) { SvePredInit(0, pd, pat); }
void CodeGenerator::ptrues(const PRegB &pd, const Pattern pat) { SvePredInit(1, pd, pat); }
void CodeGenerator::ptrues(const PRegH &pd, const Pattern pat) { SvePredInit(1, pd, pat); }
void CodeGenerator::ptrues(const PRegS &pd, const Pattern pat) { SvePredInit(1, pd, pat); }
void CodeGenerator::ptrues(const PRegD &pd, const Pattern pat) { SvePredInit(1, pd, pat); }
void CodeGenerator::pnext(const PRegB &pdn, const _PReg &pg) { SvePredNextAct(pdn, pg); }
void CodeGenerator::pnext(const PRegH &pdn, const _PReg &pg) { SvePredNextAct(pdn, pg); }
void CodeGenerator::pnext(const PRegS &pdn, const _PReg &pg) { SvePredNextAct(pdn, pg); }
void CodeGenerator::pnext(const PRegD &pdn, const _PReg &pg) { SvePredNextAct(pdn, pg); }
void CodeGenerator::rdffr(const PRegB &pd, const _PReg &pg) { SvePredReadFFRPred(0, 0, pd, pg); }
void CodeGenerator::rdffrs(const PRegB &pd, const _PReg &pg) { SvePredReadFFRPred(0, 1, pd, pg); }
void CodeGenerator::rdffr(const PRegB &pd) { SvePredReadFFRUnpred(0, 0, pd); }
void CodeGenerator::ptest(const _PReg &pg, const PRegB &pn) { SvePredTest(0, 1, 0, pg, pn); }
void CodeGenerator::pfalse(const PRegB &pd) { SvePredZero(0, 0, pd); }
void CodeGenerator::cmpge(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const int32_t imm) { SveIntCompSImm(0, 0, 0, pd, pg, zn, imm); }
void CodeGenerator::cmpge(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const int32_t imm) { SveIntCompSImm(0, 0, 0, pd, pg, zn, imm); }
void CodeGenerator::cmpge(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const int32_t imm) { SveIntCompSImm(0, 0, 0, pd, pg, zn, imm); }
void CodeGenerator::cmpge(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const int32_t imm) { SveIntCompSImm(0, 0, 0, pd, pg, zn, imm); }
void CodeGenerator::cmpgt(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const int32_t imm) { SveIntCompSImm(0, 0, 1, pd, pg, zn, imm); }
void CodeGenerator::cmpgt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const int32_t imm) { SveIntCompSImm(0, 0, 1, pd, pg, zn, imm); }
void CodeGenerator::cmpgt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const int32_t imm) { SveIntCompSImm(0, 0, 1, pd, pg, zn, imm); }
void CodeGenerator::cmpgt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const int32_t imm) { SveIntCompSImm(0, 0, 1, pd, pg, zn, imm); }
void CodeGenerator::cmplt(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const int32_t imm) { SveIntCompSImm(0, 1, 0, pd, pg, zn, imm); }
void CodeGenerator::cmplt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const int32_t imm) { SveIntCompSImm(0, 1, 0, pd, pg, zn, imm); }
void CodeGenerator::cmplt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const int32_t imm) { SveIntCompSImm(0, 1, 0, pd, pg, zn, imm); }
void CodeGenerator::cmplt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const int32_t imm) { SveIntCompSImm(0, 1, 0, pd, pg, zn, imm); }
void CodeGenerator::cmple(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const int32_t imm) { SveIntCompSImm(0, 1, 1, pd, pg, zn, imm); }
void CodeGenerator::cmple(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const int32_t imm) { SveIntCompSImm(0, 1, 1, pd, pg, zn, imm); }
void CodeGenerator::cmple(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const int32_t imm) { SveIntCompSImm(0, 1, 1, pd, pg, zn, imm); }
void CodeGenerator::cmple(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const int32_t imm) { SveIntCompSImm(0, 1, 1, pd, pg, zn, imm); }
void CodeGenerator::cmpeq(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const int32_t imm) { SveIntCompSImm(1, 0, 0, pd, pg, zn, imm); }
void CodeGenerator::cmpeq(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const int32_t imm) { SveIntCompSImm(1, 0, 0, pd, pg, zn, imm); }
void CodeGenerator::cmpeq(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const int32_t imm) { SveIntCompSImm(1, 0, 0, pd, pg, zn, imm); }
void CodeGenerator::cmpeq(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const int32_t imm) { SveIntCompSImm(1, 0, 0, pd, pg, zn, imm); }
void CodeGenerator::cmpne(const PRegB &pd, const _PReg &pg, const ZRegB &zn, const int32_t imm) { SveIntCompSImm(1, 0, 1, pd, pg, zn, imm); }
void CodeGenerator::cmpne(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const int32_t imm) { SveIntCompSImm(1, 0, 1, pd, pg, zn, imm); }
void CodeGenerator::cmpne(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const int32_t imm) { SveIntCompSImm(1, 0, 1, pd, pg, zn, imm); }
void CodeGenerator::cmpne(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const int32_t imm) { SveIntCompSImm(1, 0, 1, pd, pg, zn, imm); }
void CodeGenerator::cntp(const XReg &rd, const _PReg &pg, const PRegB &pn) { SvePredCount(0, 0, rd, pg, pn); }
void CodeGenerator::cntp(const XReg &rd, const _PReg &pg, const PRegH &pn) { SvePredCount(0, 0, rd, pg, pn); }
void CodeGenerator::cntp(const XReg &rd, const _PReg &pg, const PRegS &pn) { SvePredCount(0, 0, rd, pg, pn); }
void CodeGenerator::cntp(const XReg &rd, const _PReg &pg, const PRegD &pn) { SvePredCount(0, 0, rd, pg, pn); }
void CodeGenerator::incp(const XReg &xdn, const PRegB &pg) { SveIncDecRegByPredCount(0, 0, 0, xdn, pg); }
void CodeGenerator::incp(const XReg &xdn, const PRegH &pg) { SveIncDecRegByPredCount(0, 0, 0, xdn, pg); }
void CodeGenerator::incp(const XReg &xdn, const PRegS &pg) { SveIncDecRegByPredCount(0, 0, 0, xdn, pg); }
void CodeGenerator::incp(const XReg &xdn, const PRegD &pg) { SveIncDecRegByPredCount(0, 0, 0, xdn, pg); }
void CodeGenerator::decp(const XReg &xdn, const PRegB &pg) { SveIncDecRegByPredCount(0, 1, 0, xdn, pg); }
void CodeGenerator::decp(const XReg &xdn, const PRegH &pg) { SveIncDecRegByPredCount(0, 1, 0, xdn, pg); }
void CodeGenerator::decp(const XReg &xdn, const PRegS &pg) { SveIncDecRegByPredCount(0, 1, 0, xdn, pg); }
void CodeGenerator::decp(const XReg &xdn, const PRegD &pg) { SveIncDecRegByPredCount(0, 1, 0, xdn, pg); }
void CodeGenerator::incp(const ZRegH &zdn, const _PReg &pg) { SveIncDecVecByPredCount(0, 0, 0, zdn, pg); }
void CodeGenerator::incp(const ZRegS &zdn, const _PReg &pg) { SveIncDecVecByPredCount(0, 0, 0, zdn, pg); }
void CodeGenerator::incp(const ZRegD &zdn, const _PReg &pg) { SveIncDecVecByPredCount(0, 0, 0, zdn, pg); }
void CodeGenerator::decp(const ZRegH &zdn, const _PReg &pg) { SveIncDecVecByPredCount(0, 1, 0, zdn, pg); }
void CodeGenerator::decp(const ZRegS &zdn, const _PReg &pg) { SveIncDecVecByPredCount(0, 1, 0, zdn, pg); }
void CodeGenerator::decp(const ZRegD &zdn, const _PReg &pg) { SveIncDecVecByPredCount(0, 1, 0, zdn, pg); }
void CodeGenerator::sqincp(const WReg &rdn, const PRegB &pg) { SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg); }
void CodeGenerator::sqincp(const WReg &rdn, const PRegH &pg) { SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg); }
void CodeGenerator::sqincp(const WReg &rdn, const PRegS &pg) { SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg); }
void CodeGenerator::sqincp(const WReg &rdn, const PRegD &pg) { SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg); }
void CodeGenerator::sqincp(const XReg &rdn, const PRegB &pg) { SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg); }
void CodeGenerator::sqincp(const XReg &rdn, const PRegH &pg) { SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg); }
void CodeGenerator::sqincp(const XReg &rdn, const PRegS &pg) { SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg); }
void CodeGenerator::sqincp(const XReg &rdn, const PRegD &pg) { SveSatuIncDecRegByPredCount(0, 0, 0, rdn, pg); }
void CodeGenerator::uqincp(const WReg &rdn, const PRegB &pg) { SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg); }
void CodeGenerator::uqincp(const WReg &rdn, const PRegH &pg) { SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg); }
void CodeGenerator::uqincp(const WReg &rdn, const PRegS &pg) { SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg); }
void CodeGenerator::uqincp(const WReg &rdn, const PRegD &pg) { SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg); }
void CodeGenerator::uqincp(const XReg &rdn, const PRegB &pg) { SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg); }
void CodeGenerator::uqincp(const XReg &rdn, const PRegH &pg) { SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg); }
void CodeGenerator::uqincp(const XReg &rdn, const PRegS &pg) { SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg); }
void CodeGenerator::uqincp(const XReg &rdn, const PRegD &pg) { SveSatuIncDecRegByPredCount(0, 1, 0, rdn, pg); }
void CodeGenerator::sqdecp(const WReg &rdn, const PRegB &pg) { SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg); }
void CodeGenerator::sqdecp(const WReg &rdn, const PRegH &pg) { SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg); }
void CodeGenerator::sqdecp(const WReg &rdn, const PRegS &pg) { SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg); }
void CodeGenerator::sqdecp(const WReg &rdn, const PRegD &pg) { SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg); }
void CodeGenerator::sqdecp(const XReg &rdn, const PRegB &pg) { SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg); }
void CodeGenerator::sqdecp(const XReg &rdn, const PRegH &pg) { SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg); }
void CodeGenerator::sqdecp(const XReg &rdn, const PRegS &pg) { SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg); }
void CodeGenerator::sqdecp(const XReg &rdn, const PRegD &pg) { SveSatuIncDecRegByPredCount(1, 0, 0, rdn, pg); }
void CodeGenerator::uqdecp(const WReg &rdn, const PRegB &pg) { SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg); }
void CodeGenerator::uqdecp(const WReg &rdn, const PRegH &pg) { SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg); }
void CodeGenerator::uqdecp(const WReg &rdn, const PRegS &pg) { SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg); }
void CodeGenerator::uqdecp(const WReg &rdn, const PRegD &pg) { SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg); }
void CodeGenerator::uqdecp(const XReg &rdn, const PRegB &pg) { SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg); }
void CodeGenerator::uqdecp(const XReg &rdn, const PRegH &pg) { SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg); }
void CodeGenerator::uqdecp(const XReg &rdn, const PRegS &pg) { SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg); }
void CodeGenerator::uqdecp(const XReg &rdn, const PRegD &pg) { SveSatuIncDecRegByPredCount(1, 1, 0, rdn, pg); }
void CodeGenerator::sqincp(const ZRegH &zdn, const _PReg &pg) { SveSatuIncDecVecByPredCount(0, 0, 0, zdn, pg); }
void CodeGenerator::sqincp(const ZRegS &zdn, const _PReg &pg) { SveSatuIncDecVecByPredCount(0, 0, 0, zdn, pg); }
void CodeGenerator::sqincp(const ZRegD &zdn, const _PReg &pg) { SveSatuIncDecVecByPredCount(0, 0, 0, zdn, pg); }
void CodeGenerator::uqincp(const ZRegH &zdn, const _PReg &pg) { SveSatuIncDecVecByPredCount(0, 1, 0, zdn, pg); }
void CodeGenerator::uqincp(const ZRegS &zdn, const _PReg &pg) { SveSatuIncDecVecByPredCount(0, 1, 0, zdn, pg); }
void CodeGenerator::uqincp(const ZRegD &zdn, const _PReg &pg) { SveSatuIncDecVecByPredCount(0, 1, 0, zdn, pg); }
void CodeGenerator::sqdecp(const ZRegH &zdn, const _PReg &pg) { SveSatuIncDecVecByPredCount(1, 0, 0, zdn, pg); }
void CodeGenerator::sqdecp(const ZRegS &zdn, const _PReg &pg) { SveSatuIncDecVecByPredCount(1, 0, 0, zdn, pg); }
void CodeGenerator::sqdecp(const ZRegD &zdn, const _PReg &pg) { SveSatuIncDecVecByPredCount(1, 0, 0, zdn, pg); }
void CodeGenerator::uqdecp(const ZRegH &zdn, const _PReg &pg) { SveSatuIncDecVecByPredCount(1, 1, 0, zdn, pg); }
void CodeGenerator::uqdecp(const ZRegS &zdn, const _PReg &pg) { SveSatuIncDecVecByPredCount(1, 1, 0, zdn, pg); }
void CodeGenerator::uqdecp(const ZRegD &zdn, const _PReg &pg) { SveSatuIncDecVecByPredCount(1, 1, 0, zdn, pg); }
void CodeGenerator::setffr() { SveFFRInit(0); }
void CodeGenerator::wrffr(const PRegB &pn) { SveFFRWritePred(0, pn); }
void CodeGenerator::ctermeq(const WReg &rn, const WReg &rm) { SveCondTermScalars(1, 0, rn, rm); }
void CodeGenerator::ctermeq(const XReg &rn, const XReg &rm) { SveCondTermScalars(1, 0, rn, rm); }
void CodeGenerator::ctermne(const WReg &rn, const WReg &rm) { SveCondTermScalars(1, 1, rn, rm); }
void CodeGenerator::ctermne(const XReg &rn, const XReg &rm) { SveCondTermScalars(1, 1, rn, rm); }
void CodeGenerator::whilelt(const PRegB &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelt(const PRegH &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelt(const PRegS &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelt(const PRegD &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelt(const PRegB &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelt(const PRegH &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelt(const PRegS &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelt(const PRegD &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 0, pd, rn, rm); }
void CodeGenerator::whilele(const PRegB &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm); }
void CodeGenerator::whilele(const PRegH &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm); }
void CodeGenerator::whilele(const PRegS &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm); }
void CodeGenerator::whilele(const PRegD &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm); }
void CodeGenerator::whilele(const PRegB &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm); }
void CodeGenerator::whilele(const PRegH &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm); }
void CodeGenerator::whilele(const PRegS &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm); }
void CodeGenerator::whilele(const PRegD &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(0, 1, 1, pd, rn, rm); }
void CodeGenerator::whilelo(const PRegB &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelo(const PRegH &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelo(const PRegS &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelo(const PRegD &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelo(const PRegB &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelo(const PRegH &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelo(const PRegS &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm); }
void CodeGenerator::whilelo(const PRegD &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 0, pd, rn, rm); }
void CodeGenerator::whilels(const PRegB &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm); }
void CodeGenerator::whilels(const PRegH &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm); }
void CodeGenerator::whilels(const PRegS &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm); }
void CodeGenerator::whilels(const PRegD &pd, const WReg &rn, const WReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm); }
void CodeGenerator::whilels(const PRegB &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm); }
void CodeGenerator::whilels(const PRegH &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm); }
void CodeGenerator::whilels(const PRegS &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm); }
void CodeGenerator::whilels(const PRegD &pd, const XReg &rn, const XReg &rm) { SveIntCompScalarCountAndLimit(1, 1, 1, pd, rn, rm); }
void CodeGenerator::fdup(const ZRegH &zd, const double imm) { SveBcFpImmUnpred(0, 0, zd, imm); }
void CodeGenerator::fdup(const ZRegS &zd, const double imm) { SveBcFpImmUnpred(0, 0, zd, imm); }
void CodeGenerator::fdup(const ZRegD &zd, const double imm) { SveBcFpImmUnpred(0, 0, zd, imm); }
void CodeGenerator::fmov(const ZRegH &zd, const double imm) { SveBcFpImmUnpred(0, 0, zd, imm); }
void CodeGenerator::fmov(const ZRegS &zd, const double imm) { SveBcFpImmUnpred(0, 0, zd, imm); }
void CodeGenerator::fmov(const ZRegD &zd, const double imm) { SveBcFpImmUnpred(0, 0, zd, imm); }
void CodeGenerator::dup(const ZRegB &zd, const int32_t imm, const ShMod mod, const uint32_t sh) { SveBcIntImmUnpred(0, zd, imm, mod, sh); }
void CodeGenerator::dup(const ZRegH &zd, const int32_t imm, const ShMod mod, const uint32_t sh) { SveBcIntImmUnpred(0, zd, imm, mod, sh); }
void CodeGenerator::dup(const ZRegS &zd, const int32_t imm, const ShMod mod, const uint32_t sh) { SveBcIntImmUnpred(0, zd, imm, mod, sh); }
void CodeGenerator::dup(const ZRegD &zd, const int32_t imm, const ShMod mod, const uint32_t sh) { SveBcIntImmUnpred(0, zd, imm, mod, sh); }
void CodeGenerator::mov(const ZRegB &zd, const int32_t imm, const ShMod mod, const uint32_t sh) { SveBcIntImmUnpred(0, zd, imm, mod, sh); }
void CodeGenerator::mov(const ZRegH &zd, const int32_t imm, const ShMod mod, const uint32_t sh) { SveBcIntImmUnpred(0, zd, imm, mod, sh); }
void CodeGenerator::mov(const ZRegS &zd, const int32_t imm, const ShMod mod, const uint32_t sh) { SveBcIntImmUnpred(0, zd, imm, mod, sh); }
void CodeGenerator::mov(const ZRegD &zd, const int32_t imm, const ShMod mod, const uint32_t sh) { SveBcIntImmUnpred(0, zd, imm, mod, sh); }
void CodeGenerator::fmov(const ZRegB &zd, const float imm) { SveBcIntImmUnpred(0, zd, static_cast<uint32_t>(imm), LSL, 0); }
void CodeGenerator::fmov(const ZRegH &zd, const float imm) { SveBcIntImmUnpred(0, zd, static_cast<uint32_t>(imm), LSL, 0); }
void CodeGenerator::fmov(const ZRegS &zd, const float imm) { SveBcIntImmUnpred(0, zd, static_cast<uint32_t>(imm), LSL, 0); }
void CodeGenerator::fmov(const ZRegD &zd, const float imm) { SveBcIntImmUnpred(0, zd, static_cast<uint32_t>(imm), LSL, 0); }
void CodeGenerator::add(const ZRegB &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(0, zdn, imm, mod, sh); }
void CodeGenerator::add(const ZRegH &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(0, zdn, imm, mod, sh); }
void CodeGenerator::add(const ZRegS &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(0, zdn, imm, mod, sh); }
void CodeGenerator::add(const ZRegD &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(0, zdn, imm, mod, sh); }
void CodeGenerator::sub(const ZRegB &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(1, zdn, imm, mod, sh); }
void CodeGenerator::sub(const ZRegH &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(1, zdn, imm, mod, sh); }
void CodeGenerator::sub(const ZRegS &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(1, zdn, imm, mod, sh); }
void CodeGenerator::sub(const ZRegD &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(1, zdn, imm, mod, sh); }
void CodeGenerator::subr(const ZRegB &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(3, zdn, imm, mod, sh); }
void CodeGenerator::subr(const ZRegH &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(3, zdn, imm, mod, sh); }
void CodeGenerator::subr(const ZRegS &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(3, zdn, imm, mod, sh); }
void CodeGenerator::subr(const ZRegD &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(3, zdn, imm, mod, sh); }
void CodeGenerator::sqadd(const ZRegB &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(4, zdn, imm, mod, sh); }
void CodeGenerator::sqadd(const ZRegH &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(4, zdn, imm, mod, sh); }
void CodeGenerator::sqadd(const ZRegS &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(4, zdn, imm, mod, sh); }
void CodeGenerator::sqadd(const ZRegD &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(4, zdn, imm, mod, sh); }
void CodeGenerator::uqadd(const ZRegB &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(5, zdn, imm, mod, sh); }
void CodeGenerator::uqadd(const ZRegH &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(5, zdn, imm, mod, sh); }
void CodeGenerator::uqadd(const ZRegS &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(5, zdn, imm, mod, sh); }
void CodeGenerator::uqadd(const ZRegD &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(5, zdn, imm, mod, sh); }
void CodeGenerator::sqsub(const ZRegB &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(6, zdn, imm, mod, sh); }
void CodeGenerator::sqsub(const ZRegH &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(6, zdn, imm, mod, sh); }
void CodeGenerator::sqsub(const ZRegS &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(6, zdn, imm, mod, sh); }
void CodeGenerator::sqsub(const ZRegD &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(6, zdn, imm, mod, sh); }
void CodeGenerator::uqsub(const ZRegB &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(7, zdn, imm, mod, sh); }
void CodeGenerator::uqsub(const ZRegH &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(7, zdn, imm, mod, sh); }
void CodeGenerator::uqsub(const ZRegS &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(7, zdn, imm, mod, sh); }
void CodeGenerator::uqsub(const ZRegD &zdn, const uint32_t imm, const ShMod mod, const uint32_t sh) { SveIntAddSubImmUnpred(7, zdn, imm, mod, sh); }
void CodeGenerator::smax(const ZRegB &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(0, 0, zdn, imm); }
void CodeGenerator::smax(const ZRegH &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(0, 0, zdn, imm); }
void CodeGenerator::smax(const ZRegS &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(0, 0, zdn, imm); }
void CodeGenerator::smax(const ZRegD &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(0, 0, zdn, imm); }
void CodeGenerator::umax(const ZRegB &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(1, 0, zdn, imm); }
void CodeGenerator::umax(const ZRegH &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(1, 0, zdn, imm); }
void CodeGenerator::umax(const ZRegS &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(1, 0, zdn, imm); }
void CodeGenerator::umax(const ZRegD &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(1, 0, zdn, imm); }
void CodeGenerator::smin(const ZRegB &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(2, 0, zdn, imm); }
void CodeGenerator::smin(const ZRegH &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(2, 0, zdn, imm); }
void CodeGenerator::smin(const ZRegS &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(2, 0, zdn, imm); }
void CodeGenerator::smin(const ZRegD &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(2, 0, zdn, imm); }
void CodeGenerator::umin(const ZRegB &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(3, 0, zdn, imm); }
void CodeGenerator::umin(const ZRegH &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(3, 0, zdn, imm); }
void CodeGenerator::umin(const ZRegS &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(3, 0, zdn, imm); }
void CodeGenerator::umin(const ZRegD &zdn, const int32_t imm) { SveIntMinMaxImmUnpred(3, 0, zdn, imm); }
void CodeGenerator::mul(const ZRegB &zdn, const int32_t imm) { SveIntMultImmUnpred(0, 0, zdn, imm); }
void CodeGenerator::mul(const ZRegH &zdn, const int32_t imm) { SveIntMultImmUnpred(0, 0, zdn, imm); }
void CodeGenerator::mul(const ZRegS &zdn, const int32_t imm) { SveIntMultImmUnpred(0, 0, zdn, imm); }
void CodeGenerator::mul(const ZRegD &zdn, const int32_t imm) { SveIntMultImmUnpred(0, 0, zdn, imm); }
void CodeGenerator::sdot(const ZRegS &zda, const ZRegB &zn, const ZRegB &zm) { SveIntDotProdcutUnpred(0, zda, zn, zm); }
void CodeGenerator::sdot(const ZRegD &zda, const ZRegH &zn, const ZRegH &zm) { SveIntDotProdcutUnpred(0, zda, zn, zm); }
void CodeGenerator::udot(const ZRegS &zda, const ZRegB &zn, const ZRegB &zm) { SveIntDotProdcutUnpred(1, zda, zn, zm); }
void CodeGenerator::udot(const ZRegD &zda, const ZRegH &zn, const ZRegH &zm) { SveIntDotProdcutUnpred(1, zda, zn, zm); }
void CodeGenerator::sdot(const ZRegS &zda, const ZRegB &zn, const ZRegBElem &zm) { SveIntDotProdcutIndexed(2, 0, zda, zn, zm); }
void CodeGenerator::udot(const ZRegS &zda, const ZRegB &zn, const ZRegBElem &zm) { SveIntDotProdcutIndexed(2, 1, zda, zn, zm); }
void CodeGenerator::sdot(const ZRegD &zda, const ZRegH &zn, const ZRegHElem &zm) { SveIntDotProdcutIndexed(3, 0, zda, zn, zm); }
void CodeGenerator::udot(const ZRegD &zda, const ZRegH &zn, const ZRegHElem &zm) { SveIntDotProdcutIndexed(3, 1, zda, zn, zm); }
void CodeGenerator::fcadd(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm, const uint32_t ct) { SveFpComplexAddPred(zdn, pg, zm, ct); }
void CodeGenerator::fcadd(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm, const uint32_t ct) { SveFpComplexAddPred(zdn, pg, zm, ct); }
void CodeGenerator::fcadd(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm, const uint32_t ct) { SveFpComplexAddPred(zdn, pg, zm, ct); }
void CodeGenerator::fcmla(const ZRegH &zda, const _PReg &pg, const ZRegH &zn, const ZRegH &zm, const uint32_t ct) { SveFpComplexMultAddPred(zda, pg, zn, zm, ct); }
void CodeGenerator::fcmla(const ZRegS &zda, const _PReg &pg, const ZRegS &zn, const ZRegS &zm, const uint32_t ct) { SveFpComplexMultAddPred(zda, pg, zn, zm, ct); }
void CodeGenerator::fcmla(const ZRegD &zda, const _PReg &pg, const ZRegD &zn, const ZRegD &zm, const uint32_t ct) { SveFpComplexMultAddPred(zda, pg, zn, zm, ct); }
void CodeGenerator::fmla(const ZRegH &zda, const ZRegH &zn, const ZRegHElem &zm) { SveFpMultAddIndexed(0, zda, zn, zm); }
void CodeGenerator::fmla(const ZRegS &zda, const ZRegS &zn, const ZRegSElem &zm) { SveFpMultAddIndexed(0, zda, zn, zm); }
void CodeGenerator::fmla(const ZRegD &zda, const ZRegD &zn, const ZRegDElem &zm) { SveFpMultAddIndexed(0, zda, zn, zm); }
void CodeGenerator::fmls(const ZRegH &zda, const ZRegH &zn, const ZRegHElem &zm) { SveFpMultAddIndexed(1, zda, zn, zm); }
void CodeGenerator::fmls(const ZRegS &zda, const ZRegS &zn, const ZRegSElem &zm) { SveFpMultAddIndexed(1, zda, zn, zm); }
void CodeGenerator::fmls(const ZRegD &zda, const ZRegD &zn, const ZRegDElem &zm) { SveFpMultAddIndexed(1, zda, zn, zm); }
void CodeGenerator::fcmla(const ZRegH &zda, const ZRegH &zn, const ZRegHElem &zm, const uint32_t ct) { SveFpComplexMultAddIndexed(zda, zn, zm, ct); }
void CodeGenerator::fcmla(const ZRegS &zda, const ZRegS &zn, const ZRegSElem &zm, const uint32_t ct) { SveFpComplexMultAddIndexed(zda, zn, zm, ct); }
void CodeGenerator::fmul(const ZRegH &zd, const ZRegH &zn, const ZRegHElem &zm) { SveFpMultIndexed(zd, zn, zm); }
void CodeGenerator::fmul(const ZRegS &zd, const ZRegS &zn, const ZRegSElem &zm) { SveFpMultIndexed(zd, zn, zm); }
void CodeGenerator::fmul(const ZRegD &zd, const ZRegD &zn, const ZRegDElem &zm) { SveFpMultIndexed(zd, zn, zm); }
void CodeGenerator::faddv(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveFpRecurReduct(0, vd, pg, zn); }
void CodeGenerator::faddv(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveFpRecurReduct(0, vd, pg, zn); }
void CodeGenerator::faddv(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveFpRecurReduct(0, vd, pg, zn); }
void CodeGenerator::fmaxnmv(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveFpRecurReduct(4, vd, pg, zn); }
void CodeGenerator::fmaxnmv(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveFpRecurReduct(4, vd, pg, zn); }
void CodeGenerator::fmaxnmv(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveFpRecurReduct(4, vd, pg, zn); }
void CodeGenerator::fminnmv(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveFpRecurReduct(5, vd, pg, zn); }
void CodeGenerator::fminnmv(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveFpRecurReduct(5, vd, pg, zn); }
void CodeGenerator::fminnmv(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveFpRecurReduct(5, vd, pg, zn); }
void CodeGenerator::fmaxv(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveFpRecurReduct(6, vd, pg, zn); }
void CodeGenerator::fmaxv(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveFpRecurReduct(6, vd, pg, zn); }
void CodeGenerator::fmaxv(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveFpRecurReduct(6, vd, pg, zn); }
void CodeGenerator::fminv(const HReg &vd, const _PReg &pg, const ZRegH &zn) { SveFpRecurReduct(7, vd, pg, zn); }
void CodeGenerator::fminv(const SReg &vd, const _PReg &pg, const ZRegS &zn) { SveFpRecurReduct(7, vd, pg, zn); }
void CodeGenerator::fminv(const DReg &vd, const _PReg &pg, const ZRegD &zn) { SveFpRecurReduct(7, vd, pg, zn); }
void CodeGenerator::frecpe(const ZRegH &zd, const ZRegH &zn) { SveFpReciproEstUnPred(6, zd, zn); }
void CodeGenerator::frecpe(const ZRegS &zd, const ZRegS &zn) { SveFpReciproEstUnPred(6, zd, zn); }
void CodeGenerator::frecpe(const ZRegD &zd, const ZRegD &zn) { SveFpReciproEstUnPred(6, zd, zn); }
void CodeGenerator::frsqrte(const ZRegH &zd, const ZRegH &zn) { SveFpReciproEstUnPred(7, zd, zn); }
void CodeGenerator::frsqrte(const ZRegS &zd, const ZRegS &zn) { SveFpReciproEstUnPred(7, zd, zn); }
void CodeGenerator::frsqrte(const ZRegD &zd, const ZRegD &zn) { SveFpReciproEstUnPred(7, zd, zn); }
void CodeGenerator::fcmge(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const double zero) { SveFpCompWithZero(0, 0, 0, pd, pg, zn, zero); }
void CodeGenerator::fcmge(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const double zero) { SveFpCompWithZero(0, 0, 0, pd, pg, zn, zero); }
void CodeGenerator::fcmge(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const double zero) { SveFpCompWithZero(0, 0, 0, pd, pg, zn, zero); }
void CodeGenerator::fcmgt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const double zero) { SveFpCompWithZero(0, 0, 1, pd, pg, zn, zero); }
void CodeGenerator::fcmgt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const double zero) { SveFpCompWithZero(0, 0, 1, pd, pg, zn, zero); }
void CodeGenerator::fcmgt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const double zero) { SveFpCompWithZero(0, 0, 1, pd, pg, zn, zero); }
void CodeGenerator::fcmlt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const double zero) { SveFpCompWithZero(0, 1, 0, pd, pg, zn, zero); }
void CodeGenerator::fcmlt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const double zero) { SveFpCompWithZero(0, 1, 0, pd, pg, zn, zero); }
void CodeGenerator::fcmlt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const double zero) { SveFpCompWithZero(0, 1, 0, pd, pg, zn, zero); }
void CodeGenerator::fcmle(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const double zero) { SveFpCompWithZero(0, 1, 1, pd, pg, zn, zero); }
void CodeGenerator::fcmle(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const double zero) { SveFpCompWithZero(0, 1, 1, pd, pg, zn, zero); }
void CodeGenerator::fcmle(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const double zero) { SveFpCompWithZero(0, 1, 1, pd, pg, zn, zero); }
void CodeGenerator::fcmeq(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const double zero) { SveFpCompWithZero(1, 0, 0, pd, pg, zn, zero); }
void CodeGenerator::fcmeq(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const double zero) { SveFpCompWithZero(1, 0, 0, pd, pg, zn, zero); }
void CodeGenerator::fcmeq(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const double zero) { SveFpCompWithZero(1, 0, 0, pd, pg, zn, zero); }
void CodeGenerator::fcmne(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const double zero) { SveFpCompWithZero(1, 1, 0, pd, pg, zn, zero); }
void CodeGenerator::fcmne(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const double zero) { SveFpCompWithZero(1, 1, 0, pd, pg, zn, zero); }
void CodeGenerator::fcmne(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const double zero) { SveFpCompWithZero(1, 1, 0, pd, pg, zn, zero); }
void CodeGenerator::fadda(const HReg &vdn, const _PReg &pg, const ZRegH &zm) { SveFpSerialReductPred(0, vdn, pg, zm); }
void CodeGenerator::fadda(const SReg &vdn, const _PReg &pg, const ZRegS &zm) { SveFpSerialReductPred(0, vdn, pg, zm); }
void CodeGenerator::fadda(const DReg &vdn, const _PReg &pg, const ZRegD &zm) { SveFpSerialReductPred(0, vdn, pg, zm); }
void CodeGenerator::fadd(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveFpArithmeticUnpred(0, zd, zn, zm); }
void CodeGenerator::fadd(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveFpArithmeticUnpred(0, zd, zn, zm); }
void CodeGenerator::fadd(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveFpArithmeticUnpred(0, zd, zn, zm); }
void CodeGenerator::fsub(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveFpArithmeticUnpred(1, zd, zn, zm); }
void CodeGenerator::fsub(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveFpArithmeticUnpred(1, zd, zn, zm); }
void CodeGenerator::fsub(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveFpArithmeticUnpred(1, zd, zn, zm); }
void CodeGenerator::fmul(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveFpArithmeticUnpred(2, zd, zn, zm); }
void CodeGenerator::fmul(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveFpArithmeticUnpred(2, zd, zn, zm); }
void CodeGenerator::fmul(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveFpArithmeticUnpred(2, zd, zn, zm); }
void CodeGenerator::ftsmul(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveFpArithmeticUnpred(3, zd, zn, zm); }
void CodeGenerator::ftsmul(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveFpArithmeticUnpred(3, zd, zn, zm); }
void CodeGenerator::ftsmul(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveFpArithmeticUnpred(3, zd, zn, zm); }
void CodeGenerator::frecps(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveFpArithmeticUnpred(6, zd, zn, zm); }
void CodeGenerator::frecps(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveFpArithmeticUnpred(6, zd, zn, zm); }
void CodeGenerator::frecps(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveFpArithmeticUnpred(6, zd, zn, zm); }
void CodeGenerator::frsqrts(const ZRegH &zd, const ZRegH &zn, const ZRegH &zm) { SveFpArithmeticUnpred(7, zd, zn, zm); }
void CodeGenerator::frsqrts(const ZRegS &zd, const ZRegS &zn, const ZRegS &zm) { SveFpArithmeticUnpred(7, zd, zn, zm); }
void CodeGenerator::frsqrts(const ZRegD &zd, const ZRegD &zn, const ZRegD &zm) { SveFpArithmeticUnpred(7, zd, zn, zm); }
void CodeGenerator::fadd(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveFpArithmeticPred(0, zdn, pg, zm); }
void CodeGenerator::fadd(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveFpArithmeticPred(0, zdn, pg, zm); }
void CodeGenerator::fadd(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveFpArithmeticPred(0, zdn, pg, zm); }
void CodeGenerator::fsub(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveFpArithmeticPred(1, zdn, pg, zm); }
void CodeGenerator::fsub(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveFpArithmeticPred(1, zdn, pg, zm); }
void CodeGenerator::fsub(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveFpArithmeticPred(1, zdn, pg, zm); }
void CodeGenerator::fmul(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveFpArithmeticPred(2, zdn, pg, zm); }
void CodeGenerator::fmul(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveFpArithmeticPred(2, zdn, pg, zm); }
void CodeGenerator::fmul(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveFpArithmeticPred(2, zdn, pg, zm); }
void CodeGenerator::fsubr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveFpArithmeticPred(3, zdn, pg, zm); }
void CodeGenerator::fsubr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveFpArithmeticPred(3, zdn, pg, zm); }
void CodeGenerator::fsubr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveFpArithmeticPred(3, zdn, pg, zm); }
void CodeGenerator::fmaxnm(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveFpArithmeticPred(4, zdn, pg, zm); }
void CodeGenerator::fmaxnm(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveFpArithmeticPred(4, zdn, pg, zm); }
void CodeGenerator::fmaxnm(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveFpArithmeticPred(4, zdn, pg, zm); }
void CodeGenerator::fminnm(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveFpArithmeticPred(5, zdn, pg, zm); }
void CodeGenerator::fminnm(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveFpArithmeticPred(5, zdn, pg, zm); }
void CodeGenerator::fminnm(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveFpArithmeticPred(5, zdn, pg, zm); }
void CodeGenerator::fmax(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveFpArithmeticPred(6, zdn, pg, zm); }
void CodeGenerator::fmax(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveFpArithmeticPred(6, zdn, pg, zm); }
void CodeGenerator::fmax(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveFpArithmeticPred(6, zdn, pg, zm); }
void CodeGenerator::fmin(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveFpArithmeticPred(7, zdn, pg, zm); }
void CodeGenerator::fmin(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveFpArithmeticPred(7, zdn, pg, zm); }
void CodeGenerator::fmin(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveFpArithmeticPred(7, zdn, pg, zm); }
void CodeGenerator::fabd(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveFpArithmeticPred(8, zdn, pg, zm); }
void CodeGenerator::fabd(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveFpArithmeticPred(8, zdn, pg, zm); }
void CodeGenerator::fabd(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveFpArithmeticPred(8, zdn, pg, zm); }
void CodeGenerator::fscale(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveFpArithmeticPred(9, zdn, pg, zm); }
void CodeGenerator::fscale(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveFpArithmeticPred(9, zdn, pg, zm); }
void CodeGenerator::fscale(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveFpArithmeticPred(9, zdn, pg, zm); }
void CodeGenerator::fmulx(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveFpArithmeticPred(10, zdn, pg, zm); }
void CodeGenerator::fmulx(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveFpArithmeticPred(10, zdn, pg, zm); }
void CodeGenerator::fmulx(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveFpArithmeticPred(10, zdn, pg, zm); }
void CodeGenerator::fdivr(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveFpArithmeticPred(12, zdn, pg, zm); }
void CodeGenerator::fdivr(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveFpArithmeticPred(12, zdn, pg, zm); }
void CodeGenerator::fdivr(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveFpArithmeticPred(12, zdn, pg, zm); }
void CodeGenerator::fdiv(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm) { SveFpArithmeticPred(13, zdn, pg, zm); }
void CodeGenerator::fdiv(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm) { SveFpArithmeticPred(13, zdn, pg, zm); }
void CodeGenerator::fdiv(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm) { SveFpArithmeticPred(13, zdn, pg, zm); }
void CodeGenerator::fadd(const ZRegH &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(0, zdn, pg, ct); }
void CodeGenerator::fadd(const ZRegS &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(0, zdn, pg, ct); }
void CodeGenerator::fadd(const ZRegD &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(0, zdn, pg, ct); }
void CodeGenerator::fsub(const ZRegH &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(1, zdn, pg, ct); }
void CodeGenerator::fsub(const ZRegS &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(1, zdn, pg, ct); }
void CodeGenerator::fsub(const ZRegD &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(1, zdn, pg, ct); }
void CodeGenerator::fmul(const ZRegH &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(2, zdn, pg, ct); }
void CodeGenerator::fmul(const ZRegS &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(2, zdn, pg, ct); }
void CodeGenerator::fmul(const ZRegD &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(2, zdn, pg, ct); }
void CodeGenerator::fsubr(const ZRegH &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(3, zdn, pg, ct); }
void CodeGenerator::fsubr(const ZRegS &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(3, zdn, pg, ct); }
void CodeGenerator::fsubr(const ZRegD &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(3, zdn, pg, ct); }
void CodeGenerator::fmaxnm(const ZRegH &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(4, zdn, pg, ct); }
void CodeGenerator::fmaxnm(const ZRegS &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(4, zdn, pg, ct); }
void CodeGenerator::fmaxnm(const ZRegD &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(4, zdn, pg, ct); }
void CodeGenerator::fminnm(const ZRegH &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(5, zdn, pg, ct); }
void CodeGenerator::fminnm(const ZRegS &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(5, zdn, pg, ct); }
void CodeGenerator::fminnm(const ZRegD &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(5, zdn, pg, ct); }
void CodeGenerator::fmax(const ZRegH &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(6, zdn, pg, ct); }
void CodeGenerator::fmax(const ZRegS &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(6, zdn, pg, ct); }
void CodeGenerator::fmax(const ZRegD &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(6, zdn, pg, ct); }
void CodeGenerator::fmin(const ZRegH &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(7, zdn, pg, ct); }
void CodeGenerator::fmin(const ZRegS &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(7, zdn, pg, ct); }
void CodeGenerator::fmin(const ZRegD &zdn, const _PReg &pg, const float ct) { SveFpArithmeticImmPred(7, zdn, pg, ct); }
void CodeGenerator::ftmad(const ZRegH &zdn, const ZRegH &zm, const uint32_t imm) { SveFpTrigMultAddCoef(zdn, zm, imm); }
void CodeGenerator::ftmad(const ZRegS &zdn, const ZRegS &zm, const uint32_t imm) { SveFpTrigMultAddCoef(zdn, zm, imm); }
void CodeGenerator::ftmad(const ZRegD &zdn, const ZRegD &zm, const uint32_t imm) { SveFpTrigMultAddCoef(zdn, zm, imm); }
void CodeGenerator::fcvt(const ZRegH &zd, const _PReg &pg, const ZRegS &zn) { SveFpCvtPrecision(2, 0, zd, pg, zn); }
void CodeGenerator::fcvt(const ZRegS &zd, const _PReg &pg, const ZRegH &zn) { SveFpCvtPrecision(2, 1, zd, pg, zn); }
void CodeGenerator::fcvt(const ZRegH &zd, const _PReg &pg, const ZRegD &zn) { SveFpCvtPrecision(3, 0, zd, pg, zn); }
void CodeGenerator::fcvt(const ZRegD &zd, const _PReg &pg, const ZRegH &zn) { SveFpCvtPrecision(3, 1, zd, pg, zn); }
void CodeGenerator::fcvt(const ZRegS &zd, const _PReg &pg, const ZRegD &zn) { SveFpCvtPrecision(3, 2, zd, pg, zn); }
void CodeGenerator::fcvt(const ZRegD &zd, const _PReg &pg, const ZRegS &zn) { SveFpCvtPrecision(3, 3, zd, pg, zn); }
void CodeGenerator::fcvtzs(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveFpCvtToInt(1, 1, 0, zd, pg, zn); }
void CodeGenerator::fcvtzu(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveFpCvtToInt(1, 1, 1, zd, pg, zn); }
void CodeGenerator::fcvtzs(const ZRegS &zd, const _PReg &pg, const ZRegH &zn) { SveFpCvtToInt(1, 2, 0, zd, pg, zn); }
void CodeGenerator::fcvtzu(const ZRegS &zd, const _PReg &pg, const ZRegH &zn) { SveFpCvtToInt(1, 2, 1, zd, pg, zn); }
void CodeGenerator::fcvtzs(const ZRegD &zd, const _PReg &pg, const ZRegH &zn) { SveFpCvtToInt(1, 3, 0, zd, pg, zn); }
void CodeGenerator::fcvtzu(const ZRegD &zd, const _PReg &pg, const ZRegH &zn) { SveFpCvtToInt(1, 3, 1, zd, pg, zn); }
void CodeGenerator::fcvtzs(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveFpCvtToInt(2, 2, 0, zd, pg, zn); }
void CodeGenerator::fcvtzu(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveFpCvtToInt(2, 2, 1, zd, pg, zn); }
void CodeGenerator::fcvtzs(const ZRegS &zd, const _PReg &pg, const ZRegD &zn) { SveFpCvtToInt(3, 0, 0, zd, pg, zn); }
void CodeGenerator::fcvtzu(const ZRegS &zd, const _PReg &pg, const ZRegD &zn) { SveFpCvtToInt(3, 0, 1, zd, pg, zn); }
void CodeGenerator::fcvtzs(const ZRegD &zd, const _PReg &pg, const ZRegS &zn) { SveFpCvtToInt(3, 2, 0, zd, pg, zn); }
void CodeGenerator::fcvtzu(const ZRegD &zd, const _PReg &pg, const ZRegS &zn) { SveFpCvtToInt(3, 2, 1, zd, pg, zn); }
void CodeGenerator::fcvtzs(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveFpCvtToInt(3, 3, 0, zd, pg, zn); }
void CodeGenerator::fcvtzu(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveFpCvtToInt(3, 3, 1, zd, pg, zn); }
void CodeGenerator::frintn(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveFpRoundToIntegral(0, zd, pg, zn); }
void CodeGenerator::frintn(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveFpRoundToIntegral(0, zd, pg, zn); }
void CodeGenerator::frintn(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveFpRoundToIntegral(0, zd, pg, zn); }
void CodeGenerator::frintp(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveFpRoundToIntegral(1, zd, pg, zn); }
void CodeGenerator::frintp(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveFpRoundToIntegral(1, zd, pg, zn); }
void CodeGenerator::frintp(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveFpRoundToIntegral(1, zd, pg, zn); }
void CodeGenerator::frintm(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveFpRoundToIntegral(2, zd, pg, zn); }
void CodeGenerator::frintm(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveFpRoundToIntegral(2, zd, pg, zn); }
void CodeGenerator::frintm(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveFpRoundToIntegral(2, zd, pg, zn); }
void CodeGenerator::frintz(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveFpRoundToIntegral(3, zd, pg, zn); }
void CodeGenerator::frintz(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveFpRoundToIntegral(3, zd, pg, zn); }
void CodeGenerator::frintz(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveFpRoundToIntegral(3, zd, pg, zn); }
void CodeGenerator::frinta(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveFpRoundToIntegral(4, zd, pg, zn); }
void CodeGenerator::frinta(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveFpRoundToIntegral(4, zd, pg, zn); }
void CodeGenerator::frinta(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveFpRoundToIntegral(4, zd, pg, zn); }
void CodeGenerator::frintx(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveFpRoundToIntegral(6, zd, pg, zn); }
void CodeGenerator::frintx(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveFpRoundToIntegral(6, zd, pg, zn); }
void CodeGenerator::frintx(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveFpRoundToIntegral(6, zd, pg, zn); }
void CodeGenerator::frinti(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveFpRoundToIntegral(7, zd, pg, zn); }
void CodeGenerator::frinti(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveFpRoundToIntegral(7, zd, pg, zn); }
void CodeGenerator::frinti(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveFpRoundToIntegral(7, zd, pg, zn); }
void CodeGenerator::frecpx(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveFpUnaryOp(0, zd, pg, zn); }
void CodeGenerator::frecpx(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveFpUnaryOp(0, zd, pg, zn); }
void CodeGenerator::frecpx(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveFpUnaryOp(0, zd, pg, zn); }
void CodeGenerator::fsqrt(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveFpUnaryOp(1, zd, pg, zn); }
void CodeGenerator::fsqrt(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveFpUnaryOp(1, zd, pg, zn); }
void CodeGenerator::fsqrt(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveFpUnaryOp(1, zd, pg, zn); }
void CodeGenerator::scvtf(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveIntCvtToFp(1, 1, 0, zd, pg, zn); }
void CodeGenerator::ucvtf(const ZRegH &zd, const _PReg &pg, const ZRegH &zn) { SveIntCvtToFp(1, 1, 1, zd, pg, zn); }
void CodeGenerator::scvtf(const ZRegH &zd, const _PReg &pg, const ZRegS &zn) { SveIntCvtToFp(1, 2, 0, zd, pg, zn); }
void CodeGenerator::ucvtf(const ZRegH &zd, const _PReg &pg, const ZRegS &zn) { SveIntCvtToFp(1, 2, 1, zd, pg, zn); }
void CodeGenerator::scvtf(const ZRegH &zd, const _PReg &pg, const ZRegD &zn) { SveIntCvtToFp(1, 3, 0, zd, pg, zn); }
void CodeGenerator::ucvtf(const ZRegH &zd, const _PReg &pg, const ZRegD &zn) { SveIntCvtToFp(1, 3, 1, zd, pg, zn); }
void CodeGenerator::scvtf(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveIntCvtToFp(2, 2, 0, zd, pg, zn); }
void CodeGenerator::ucvtf(const ZRegS &zd, const _PReg &pg, const ZRegS &zn) { SveIntCvtToFp(2, 2, 1, zd, pg, zn); }
void CodeGenerator::scvtf(const ZRegD &zd, const _PReg &pg, const ZRegS &zn) { SveIntCvtToFp(3, 0, 0, zd, pg, zn); }
void CodeGenerator::ucvtf(const ZRegD &zd, const _PReg &pg, const ZRegS &zn) { SveIntCvtToFp(3, 0, 1, zd, pg, zn); }
void CodeGenerator::scvtf(const ZRegS &zd, const _PReg &pg, const ZRegD &zn) { SveIntCvtToFp(3, 2, 0, zd, pg, zn); }
void CodeGenerator::ucvtf(const ZRegS &zd, const _PReg &pg, const ZRegD &zn) { SveIntCvtToFp(3, 2, 1, zd, pg, zn); }
void CodeGenerator::scvtf(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveIntCvtToFp(3, 3, 0, zd, pg, zn); }
void CodeGenerator::ucvtf(const ZRegD &zd, const _PReg &pg, const ZRegD &zn) { SveIntCvtToFp(3, 3, 1, zd, pg, zn); }
void CodeGenerator::fcmge(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpCompVec(0, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::fcmge(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpCompVec(0, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::fcmge(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpCompVec(0, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::fcmgt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpCompVec(0, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::fcmgt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpCompVec(0, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::fcmgt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpCompVec(0, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::fcmle(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpCompVec(0, 0, 0, pd, pg, zm, zn); }
void CodeGenerator::fcmle(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpCompVec(0, 0, 0, pd, pg, zm, zn); }
void CodeGenerator::fcmle(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpCompVec(0, 0, 0, pd, pg, zm, zn); }
void CodeGenerator::fcmlt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpCompVec(0, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::fcmlt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpCompVec(0, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::fcmlt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpCompVec(0, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::fcmeq(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpCompVec(0, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::fcmeq(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpCompVec(0, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::fcmeq(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpCompVec(0, 1, 0, pd, pg, zn, zm); }
void CodeGenerator::fcmne(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpCompVec(0, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::fcmne(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpCompVec(0, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::fcmne(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpCompVec(0, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::fcmuo(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpCompVec(1, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::fcmuo(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpCompVec(1, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::fcmuo(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpCompVec(1, 0, 0, pd, pg, zn, zm); }
void CodeGenerator::facge(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpCompVec(1, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::facge(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpCompVec(1, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::facge(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpCompVec(1, 0, 1, pd, pg, zn, zm); }
void CodeGenerator::facgt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpCompVec(1, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::facgt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpCompVec(1, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::facgt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpCompVec(1, 1, 1, pd, pg, zn, zm); }
void CodeGenerator::facle(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpCompVec(1, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::facle(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpCompVec(1, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::facle(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpCompVec(1, 0, 1, pd, pg, zm, zn); }
void CodeGenerator::faclt(const PRegH &pd, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpCompVec(1, 1, 1, pd, pg, zm, zn); }
void CodeGenerator::faclt(const PRegS &pd, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpCompVec(1, 1, 1, pd, pg, zm, zn); }
void CodeGenerator::faclt(const PRegD &pd, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpCompVec(1, 1, 1, pd, pg, zm, zn); }
void CodeGenerator::fmla(const ZRegH &zda, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpMultAccumAddend(0, zda, pg, zn, zm); }
void CodeGenerator::fmla(const ZRegS &zda, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpMultAccumAddend(0, zda, pg, zn, zm); }
void CodeGenerator::fmla(const ZRegD &zda, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpMultAccumAddend(0, zda, pg, zn, zm); }
void CodeGenerator::fmls(const ZRegH &zda, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpMultAccumAddend(1, zda, pg, zn, zm); }
void CodeGenerator::fmls(const ZRegS &zda, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpMultAccumAddend(1, zda, pg, zn, zm); }
void CodeGenerator::fmls(const ZRegD &zda, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpMultAccumAddend(1, zda, pg, zn, zm); }
void CodeGenerator::fnmla(const ZRegH &zda, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpMultAccumAddend(2, zda, pg, zn, zm); }
void CodeGenerator::fnmla(const ZRegS &zda, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpMultAccumAddend(2, zda, pg, zn, zm); }
void CodeGenerator::fnmla(const ZRegD &zda, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpMultAccumAddend(2, zda, pg, zn, zm); }
void CodeGenerator::fnmls(const ZRegH &zda, const _PReg &pg, const ZRegH &zn, const ZRegH &zm) { SveFpMultAccumAddend(3, zda, pg, zn, zm); }
void CodeGenerator::fnmls(const ZRegS &zda, const _PReg &pg, const ZRegS &zn, const ZRegS &zm) { SveFpMultAccumAddend(3, zda, pg, zn, zm); }
void CodeGenerator::fnmls(const ZRegD &zda, const _PReg &pg, const ZRegD &zn, const ZRegD &zm) { SveFpMultAccumAddend(3, zda, pg, zn, zm); }
void CodeGenerator::fmad(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm, const ZRegH &za) { SveFpMultAccumMulti(0, zdn, pg, zm, za); }
void CodeGenerator::fmad(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm, const ZRegS &za) { SveFpMultAccumMulti(0, zdn, pg, zm, za); }
void CodeGenerator::fmad(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm, const ZRegD &za) { SveFpMultAccumMulti(0, zdn, pg, zm, za); }
void CodeGenerator::fmsb(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm, const ZRegH &za) { SveFpMultAccumMulti(1, zdn, pg, zm, za); }
void CodeGenerator::fmsb(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm, const ZRegS &za) { SveFpMultAccumMulti(1, zdn, pg, zm, za); }
void CodeGenerator::fmsb(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm, const ZRegD &za) { SveFpMultAccumMulti(1, zdn, pg, zm, za); }
void CodeGenerator::fnmad(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm, const ZRegH &za) { SveFpMultAccumMulti(2, zdn, pg, zm, za); }
void CodeGenerator::fnmad(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm, const ZRegS &za) { SveFpMultAccumMulti(2, zdn, pg, zm, za); }
void CodeGenerator::fnmad(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm, const ZRegD &za) { SveFpMultAccumMulti(2, zdn, pg, zm, za); }
void CodeGenerator::fnmsb(const ZRegH &zdn, const _PReg &pg, const ZRegH &zm, const ZRegH &za) { SveFpMultAccumMulti(3, zdn, pg, zm, za); }
void CodeGenerator::fnmsb(const ZRegS &zdn, const _PReg &pg, const ZRegS &zm, const ZRegS &za) { SveFpMultAccumMulti(3, zdn, pg, zm, za); }
void CodeGenerator::fnmsb(const ZRegD &zdn, const _PReg &pg, const ZRegD &zm, const ZRegD &za) { SveFpMultAccumMulti(3, zdn, pg, zm, za); }
void CodeGenerator::ld1sb(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) { Sve32GatherLdSc32U(0, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sb(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) { Sve32GatherLdSc32U(0, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) { Sve32GatherLdSc32U(0, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1b(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) { Sve32GatherLdSc32U(0, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) { Sve32GatherLdSc32U(1, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sh(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) { Sve32GatherLdSc32U(1, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) { Sve32GatherLdSc32U(1, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) { Sve32GatherLdSc32U(1, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) { Sve32GatherLdSc32U(2, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1w(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) { Sve32GatherLdSc32U(2, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sb(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherLdVecImm(0, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sb(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherLdVecImm(0, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherLdVecImm(0, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1b(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherLdVecImm(0, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherLdVecImm(1, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sh(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherLdVecImm(1, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherLdVecImm(1, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherLdVecImm(1, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherLdVecImm(2, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1w(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherLdVecImm(2, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) { Sve32GatherLdHSc32S(0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sh(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) { Sve32GatherLdHSc32S(0, 1, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) { Sve32GatherLdHSc32S(1, 0, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) { Sve32GatherLdHSc32S(1, 1, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) { Sve32GatherLdWSc32S(1, 0, zt, pg, adr); }
void CodeGenerator::ldff1w(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) { Sve32GatherLdWSc32S(1, 1, zt, pg, adr); }
void CodeGenerator::prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32S &adr) { Sve32GatherPfSc32S(prfop_sve, 0, pg, adr); }
void CodeGenerator::prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32S &adr) { Sve32GatherPfSc32S(prfop_sve, 1, pg, adr); }
void CodeGenerator::prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32S &adr) { Sve32GatherPfSc32S(prfop_sve, 2, pg, adr); }
void CodeGenerator::prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32S &adr) { Sve32GatherPfSc32S(prfop_sve, 3, pg, adr); }
void CodeGenerator::prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherPfVecImm(prfop_sve, 0, pg, adr); }
void CodeGenerator::prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherPfVecImm(prfop_sve, 1, pg, adr); }
void CodeGenerator::prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherPfVecImm(prfop_sve, 2, pg, adr); }
void CodeGenerator::prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm32 &adr) { Sve32GatherPfVecImm(prfop_sve, 3, pg, adr); }
void CodeGenerator::prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrScImm &adr) { Sve32ContiPfScImm(prfop_sve, 0, pg, adr); }
void CodeGenerator::prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrNoOfs &adr) { Sve32ContiPfScImm(prfop_sve, 0, pg, adr); }
void CodeGenerator::prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrScImm &adr) { Sve32ContiPfScImm(prfop_sve, 1, pg, adr); }
void CodeGenerator::prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrNoOfs &adr) { Sve32ContiPfScImm(prfop_sve, 1, pg, adr); }
void CodeGenerator::prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrScImm &adr) { Sve32ContiPfScImm(prfop_sve, 2, pg, adr); }
void CodeGenerator::prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrNoOfs &adr) { Sve32ContiPfScImm(prfop_sve, 2, pg, adr); }
void CodeGenerator::prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrScImm &adr) { Sve32ContiPfScImm(prfop_sve, 3, pg, adr); }
void CodeGenerator::prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrNoOfs &adr) { Sve32ContiPfScImm(prfop_sve, 3, pg, adr); }
void CodeGenerator::prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrScSc &adr) { Sve32ContiPfScSc(prfop_sve, 0, pg, adr); }
void CodeGenerator::prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrScSc &adr) { Sve32ContiPfScSc(prfop_sve, 1, pg, adr); }
void CodeGenerator::prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrScSc &adr) { Sve32ContiPfScSc(prfop_sve, 2, pg, adr); }
void CodeGenerator::prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrScSc &adr) { Sve32ContiPfScSc(prfop_sve, 3, pg, adr); }
void CodeGenerator::ld1rb(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(0, 0, zt, pg, adr); }
void CodeGenerator::ld1rb(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(0, 0, zt, pg, adr); }
void CodeGenerator::ld1rb(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(0, 1, zt, pg, adr); }
void CodeGenerator::ld1rb(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(0, 1, zt, pg, adr); }
void CodeGenerator::ld1rb(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(0, 2, zt, pg, adr); }
void CodeGenerator::ld1rb(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(0, 2, zt, pg, adr); }
void CodeGenerator::ld1rb(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(0, 3, zt, pg, adr); }
void CodeGenerator::ld1rb(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(0, 3, zt, pg, adr); }
void CodeGenerator::ld1rsw(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(1, 0, zt, pg, adr); }
void CodeGenerator::ld1rsw(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(1, 0, zt, pg, adr); }
void CodeGenerator::ld1rh(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(1, 1, zt, pg, adr); }
void CodeGenerator::ld1rh(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(1, 1, zt, pg, adr); }
void CodeGenerator::ld1rh(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(1, 2, zt, pg, adr); }
void CodeGenerator::ld1rh(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(1, 2, zt, pg, adr); }
void CodeGenerator::ld1rh(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(1, 3, zt, pg, adr); }
void CodeGenerator::ld1rh(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(1, 3, zt, pg, adr); }
void CodeGenerator::ld1rsh(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(2, 0, zt, pg, adr); }
void CodeGenerator::ld1rsh(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(2, 0, zt, pg, adr); }
void CodeGenerator::ld1rsh(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(2, 1, zt, pg, adr); }
void CodeGenerator::ld1rsh(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(2, 1, zt, pg, adr); }
void CodeGenerator::ld1rw(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(2, 2, zt, pg, adr); }
void CodeGenerator::ld1rw(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(2, 2, zt, pg, adr); }
void CodeGenerator::ld1rw(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(2, 3, zt, pg, adr); }
void CodeGenerator::ld1rw(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(2, 3, zt, pg, adr); }
void CodeGenerator::ld1rsb(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(3, 0, zt, pg, adr); }
void CodeGenerator::ld1rsb(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(3, 0, zt, pg, adr); }
void CodeGenerator::ld1rsb(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(3, 1, zt, pg, adr); }
void CodeGenerator::ld1rsb(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(3, 1, zt, pg, adr); }
void CodeGenerator::ld1rsb(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(3, 2, zt, pg, adr); }
void CodeGenerator::ld1rsb(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(3, 2, zt, pg, adr); }
void CodeGenerator::ld1rd(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveLoadAndBcElem(3, 3, zt, pg, adr); }
void CodeGenerator::ld1rd(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLoadAndBcElem(3, 3, zt, pg, adr); }
void CodeGenerator::ldr(const _PReg &pt, const AdrScImm &adr) { SveLoadPredReg(pt, adr); }
void CodeGenerator::ldr(const _PReg &pt, const AdrNoOfs &adr) { SveLoadPredReg(pt, adr); }
void CodeGenerator::ldr(const ZReg &zt, const AdrScImm &adr) { SveLoadPredVec(zt, adr); }
void CodeGenerator::ldr(const ZReg &zt, const AdrNoOfs &adr) { SveLoadPredVec(zt, adr); }
void CodeGenerator::ldff1b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(0, zt, pg, adr); }
void CodeGenerator::ldff1b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(0, zt, pg, adr); }
void CodeGenerator::ldff1b(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(1, zt, pg, adr); }
void CodeGenerator::ldff1b(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(1, zt, pg, adr); }
void CodeGenerator::ldff1b(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(2, zt, pg, adr); }
void CodeGenerator::ldff1b(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(2, zt, pg, adr); }
void CodeGenerator::ldff1b(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(3, zt, pg, adr); }
void CodeGenerator::ldff1b(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(3, zt, pg, adr); }
void CodeGenerator::ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(4, zt, pg, adr); }
void CodeGenerator::ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(4, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(5, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(5, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(6, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(6, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(7, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(7, zt, pg, adr); }
void CodeGenerator::ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(8, zt, pg, adr); }
void CodeGenerator::ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(8, zt, pg, adr); }
void CodeGenerator::ldff1sh(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(9, zt, pg, adr); }
void CodeGenerator::ldff1sh(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(9, zt, pg, adr); }
void CodeGenerator::ldff1w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(10, zt, pg, adr); }
void CodeGenerator::ldff1w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(10, zt, pg, adr); }
void CodeGenerator::ldff1w(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(11, zt, pg, adr); }
void CodeGenerator::ldff1w(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(11, zt, pg, adr); }
void CodeGenerator::ldff1sb(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(12, zt, pg, adr); }
void CodeGenerator::ldff1sb(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(12, zt, pg, adr); }
void CodeGenerator::ldff1sb(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(13, zt, pg, adr); }
void CodeGenerator::ldff1sb(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(13, zt, pg, adr); }
void CodeGenerator::ldff1sb(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(14, zt, pg, adr); }
void CodeGenerator::ldff1sb(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(14, zt, pg, adr); }
void CodeGenerator::ldff1d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiFFLdScSc(15, zt, pg, adr); }
void CodeGenerator::ldff1d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiFFLdScSc(15, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(0, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(0, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(1, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(1, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(2, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(2, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(3, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(3, zt, pg, adr); }
void CodeGenerator::ld1sw(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(4, zt, pg, adr); }
void CodeGenerator::ld1sw(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(4, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(5, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(5, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(6, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(6, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(7, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(7, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(8, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(8, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(9, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(9, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(10, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(10, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(11, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(11, zt, pg, adr); }
void CodeGenerator::ld1sb(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(12, zt, pg, adr); }
void CodeGenerator::ld1sb(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(12, zt, pg, adr); }
void CodeGenerator::ld1sb(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(13, zt, pg, adr); }
void CodeGenerator::ld1sb(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(13, zt, pg, adr); }
void CodeGenerator::ld1sb(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(14, zt, pg, adr); }
void CodeGenerator::ld1sb(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(14, zt, pg, adr); }
void CodeGenerator::ld1d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiLdScImm(15, zt, pg, adr); }
void CodeGenerator::ld1d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiLdScImm(15, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(0, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(1, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(2, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(3, zt, pg, adr); }
void CodeGenerator::ld1sw(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(4, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(5, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(6, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(7, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(8, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(9, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(10, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(11, zt, pg, adr); }
void CodeGenerator::ld1sb(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(12, zt, pg, adr); }
void CodeGenerator::ld1sb(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(13, zt, pg, adr); }
void CodeGenerator::ld1sb(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(14, zt, pg, adr); }
void CodeGenerator::ld1d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiLdScSc(15, zt, pg, adr); }
void CodeGenerator::ldnf1b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(0, zt, pg, adr); }
void CodeGenerator::ldnf1b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(0, zt, pg, adr); }
void CodeGenerator::ldnf1b(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(1, zt, pg, adr); }
void CodeGenerator::ldnf1b(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(1, zt, pg, adr); }
void CodeGenerator::ldnf1b(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(2, zt, pg, adr); }
void CodeGenerator::ldnf1b(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(2, zt, pg, adr); }
void CodeGenerator::ldnf1b(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(3, zt, pg, adr); }
void CodeGenerator::ldnf1b(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(3, zt, pg, adr); }
void CodeGenerator::ldnf1sw(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(4, zt, pg, adr); }
void CodeGenerator::ldnf1sw(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(4, zt, pg, adr); }
void CodeGenerator::ldnf1h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(5, zt, pg, adr); }
void CodeGenerator::ldnf1h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(5, zt, pg, adr); }
void CodeGenerator::ldnf1h(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(6, zt, pg, adr); }
void CodeGenerator::ldnf1h(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(6, zt, pg, adr); }
void CodeGenerator::ldnf1h(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(7, zt, pg, adr); }
void CodeGenerator::ldnf1h(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(7, zt, pg, adr); }
void CodeGenerator::ldnf1sh(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(8, zt, pg, adr); }
void CodeGenerator::ldnf1sh(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(8, zt, pg, adr); }
void CodeGenerator::ldnf1sh(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(9, zt, pg, adr); }
void CodeGenerator::ldnf1sh(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(9, zt, pg, adr); }
void CodeGenerator::ldnf1w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(10, zt, pg, adr); }
void CodeGenerator::ldnf1w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(10, zt, pg, adr); }
void CodeGenerator::ldnf1w(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(11, zt, pg, adr); }
void CodeGenerator::ldnf1w(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(11, zt, pg, adr); }
void CodeGenerator::ldnf1sb(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(12, zt, pg, adr); }
void CodeGenerator::ldnf1sb(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(12, zt, pg, adr); }
void CodeGenerator::ldnf1sb(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(13, zt, pg, adr); }
void CodeGenerator::ldnf1sb(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(13, zt, pg, adr); }
void CodeGenerator::ldnf1sb(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(14, zt, pg, adr); }
void CodeGenerator::ldnf1sb(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(14, zt, pg, adr); }
void CodeGenerator::ldnf1d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNFLdScImm(15, zt, pg, adr); }
void CodeGenerator::ldnf1d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNFLdScImm(15, zt, pg, adr); }
void CodeGenerator::ldnt1b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNTLdScImm(0, zt, pg, adr); }
void CodeGenerator::ldnt1b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNTLdScImm(0, zt, pg, adr); }
void CodeGenerator::ldnt1h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNTLdScImm(1, zt, pg, adr); }
void CodeGenerator::ldnt1h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNTLdScImm(1, zt, pg, adr); }
void CodeGenerator::ldnt1w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNTLdScImm(2, zt, pg, adr); }
void CodeGenerator::ldnt1w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNTLdScImm(2, zt, pg, adr); }
void CodeGenerator::ldnt1d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNTLdScImm(3, zt, pg, adr); }
void CodeGenerator::ldnt1d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNTLdScImm(3, zt, pg, adr); }
void CodeGenerator::ldnt1b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) { SveContiNTLdScSc(0, zt, pg, adr); }
void CodeGenerator::ldnt1h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveContiNTLdScSc(1, zt, pg, adr); }
void CodeGenerator::ldnt1w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiNTLdScSc(2, zt, pg, adr); }
void CodeGenerator::ldnt1d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiNTLdScSc(3, zt, pg, adr); }
void CodeGenerator::ld1rqb(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) { SveLdBcQuadScImm(0, 0, zt, pg, adr); }
void CodeGenerator::ld1rqb(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdBcQuadScImm(0, 0, zt, pg, adr); }
void CodeGenerator::ld1rqh(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveLdBcQuadScImm(1, 0, zt, pg, adr); }
void CodeGenerator::ld1rqh(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdBcQuadScImm(1, 0, zt, pg, adr); }
void CodeGenerator::ld1rqw(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveLdBcQuadScImm(2, 0, zt, pg, adr); }
void CodeGenerator::ld1rqw(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdBcQuadScImm(2, 0, zt, pg, adr); }
void CodeGenerator::ld1rqd(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveLdBcQuadScImm(3, 0, zt, pg, adr); }
void CodeGenerator::ld1rqd(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdBcQuadScImm(3, 0, zt, pg, adr); }
void CodeGenerator::ld1rqb(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) { SveLdBcQuadScSc(0, 0, zt, pg, adr); }
void CodeGenerator::ld1rqh(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveLdBcQuadScSc(1, 0, zt, pg, adr); }
void CodeGenerator::ld1rqw(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveLdBcQuadScSc(2, 0, zt, pg, adr); }
void CodeGenerator::ld1rqd(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveLdBcQuadScSc(3, 0, zt, pg, adr); }
void CodeGenerator::ld2b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) { SveLdMultiStructScImm(0, 1, zt, pg, adr); }
void CodeGenerator::ld2b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdMultiStructScImm(0, 1, zt, pg, adr); }
void CodeGenerator::ld3b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) { SveLdMultiStructScImm(0, 2, zt, pg, adr); }
void CodeGenerator::ld3b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdMultiStructScImm(0, 2, zt, pg, adr); }
void CodeGenerator::ld4b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) { SveLdMultiStructScImm(0, 3, zt, pg, adr); }
void CodeGenerator::ld4b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdMultiStructScImm(0, 3, zt, pg, adr); }
void CodeGenerator::ld2h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveLdMultiStructScImm(1, 1, zt, pg, adr); }
void CodeGenerator::ld2h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdMultiStructScImm(1, 1, zt, pg, adr); }
void CodeGenerator::ld3h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveLdMultiStructScImm(1, 2, zt, pg, adr); }
void CodeGenerator::ld3h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdMultiStructScImm(1, 2, zt, pg, adr); }
void CodeGenerator::ld4h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveLdMultiStructScImm(1, 3, zt, pg, adr); }
void CodeGenerator::ld4h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdMultiStructScImm(1, 3, zt, pg, adr); }
void CodeGenerator::ld2w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveLdMultiStructScImm(2, 1, zt, pg, adr); }
void CodeGenerator::ld2w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdMultiStructScImm(2, 1, zt, pg, adr); }
void CodeGenerator::ld3w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveLdMultiStructScImm(2, 2, zt, pg, adr); }
void CodeGenerator::ld3w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdMultiStructScImm(2, 2, zt, pg, adr); }
void CodeGenerator::ld4w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveLdMultiStructScImm(2, 3, zt, pg, adr); }
void CodeGenerator::ld4w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdMultiStructScImm(2, 3, zt, pg, adr); }
void CodeGenerator::ld2d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveLdMultiStructScImm(3, 1, zt, pg, adr); }
void CodeGenerator::ld2d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdMultiStructScImm(3, 1, zt, pg, adr); }
void CodeGenerator::ld3d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveLdMultiStructScImm(3, 2, zt, pg, adr); }
void CodeGenerator::ld3d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdMultiStructScImm(3, 2, zt, pg, adr); }
void CodeGenerator::ld4d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveLdMultiStructScImm(3, 3, zt, pg, adr); }
void CodeGenerator::ld4d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveLdMultiStructScImm(3, 3, zt, pg, adr); }
void CodeGenerator::ld2b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) { SveLdMultiStructScSc(0, 1, zt, pg, adr); }
void CodeGenerator::ld3b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) { SveLdMultiStructScSc(0, 2, zt, pg, adr); }
void CodeGenerator::ld4b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) { SveLdMultiStructScSc(0, 3, zt, pg, adr); }
void CodeGenerator::ld2h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveLdMultiStructScSc(1, 1, zt, pg, adr); }
void CodeGenerator::ld3h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveLdMultiStructScSc(1, 2, zt, pg, adr); }
void CodeGenerator::ld4h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveLdMultiStructScSc(1, 3, zt, pg, adr); }
void CodeGenerator::ld2w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveLdMultiStructScSc(2, 1, zt, pg, adr); }
void CodeGenerator::ld3w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveLdMultiStructScSc(2, 2, zt, pg, adr); }
void CodeGenerator::ld4w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveLdMultiStructScSc(2, 3, zt, pg, adr); }
void CodeGenerator::ld2d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveLdMultiStructScSc(3, 1, zt, pg, adr); }
void CodeGenerator::ld3d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveLdMultiStructScSc(3, 2, zt, pg, adr); }
void CodeGenerator::ld4d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveLdMultiStructScSc(3, 3, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherLdSc32US(1, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherLdSc32US(1, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherLdSc32US(1, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherLdSc32US(1, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sw(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherLdSc32US(2, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherLdSc32US(2, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherLdSc32US(2, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1w(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherLdSc32US(2, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1d(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherLdSc32US(3, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1d(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherLdSc32US(3, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherLdSc64S(1, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherLdSc64S(1, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherLdSc64S(1, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherLdSc64S(1, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sw(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherLdSc64S(2, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherLdSc64S(2, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherLdSc64S(2, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1w(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherLdSc64S(2, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1d(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherLdSc64S(3, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1d(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherLdSc64S(3, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sb(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(0, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sb(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(0, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(0, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1b(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(0, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(1, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(1, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(1, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(1, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sw(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(2, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(2, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(2, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1w(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(2, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1d(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(3, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1d(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64GatherLdSc64U(3, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sb(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(0, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sb(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(0, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(0, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1b(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(0, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(1, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(1, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(1, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(1, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sw(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(2, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(2, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(2, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1w(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(2, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1d(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(3, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1d(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64GatherLdSc32UU(3, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sb(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(0, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sb(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(0, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1b(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(0, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1b(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(0, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sh(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(1, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sh(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(1, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1h(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(1, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1h(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(1, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1sw(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(2, 0, 0, zt, pg, adr); }
void CodeGenerator::ldff1sw(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(2, 0, 1, zt, pg, adr); }
void CodeGenerator::ld1w(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(2, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1w(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(2, 1, 1, zt, pg, adr); }
void CodeGenerator::ld1d(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(3, 1, 0, zt, pg, adr); }
void CodeGenerator::ldff1d(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherLdVecImm(3, 1, 1, zt, pg, adr); }
void CodeGenerator::prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherPfSc64S(prfop_sve, 0, pg, adr); }
void CodeGenerator::prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherPfSc64S(prfop_sve, 1, pg, adr); }
void CodeGenerator::prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherPfSc64S(prfop_sve, 2, pg, adr); }
void CodeGenerator::prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc64S &adr) { Sve64GatherPfSc64S(prfop_sve, 3, pg, adr); }
void CodeGenerator::prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherPfSc32US(prfop_sve, 0, pg, adr); }
void CodeGenerator::prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherPfSc32US(prfop_sve, 1, pg, adr); }
void CodeGenerator::prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherPfSc32US(prfop_sve, 2, pg, adr); }
void CodeGenerator::prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrSc32US &adr) { Sve64GatherPfSc32US(prfop_sve, 3, pg, adr); }
void CodeGenerator::prfb(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherPfVecImm(prfop_sve, 0, pg, adr); }
void CodeGenerator::prfh(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherPfVecImm(prfop_sve, 1, pg, adr); }
void CodeGenerator::prfw(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherPfVecImm(prfop_sve, 2, pg, adr); }
void CodeGenerator::prfd(const PrfopSve prfop_sve, const _PReg &pg, const AdrVecImm64 &adr) { Sve64GatherPfVecImm(prfop_sve, 3, pg, adr); }
void CodeGenerator::st1h(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) { Sve32ScatterStSc32S(1, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegS &zt, const _PReg &pg, const AdrSc32S &adr) { Sve32ScatterStSc32S(2, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) { Sve32ScatterStSc32U(0, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) { Sve32ScatterStSc32U(1, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegS &zt, const _PReg &pg, const AdrSc32U &adr) { Sve32ScatterStSc32U(2, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) { Sve32ScatterStVecImm(0, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) { Sve32ScatterStVecImm(1, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegS &zt, const _PReg &pg, const AdrVecImm32 &adr) { Sve32ScatterStVecImm(2, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) { Sve64ScatterStSc64S(1, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) { Sve64ScatterStSc64S(2, zt, pg, adr); }
void CodeGenerator::st1d(const ZRegD &zt, const _PReg &pg, const AdrSc64S &adr) { Sve64ScatterStSc64S(3, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64ScatterStSc64U(0, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64ScatterStSc64U(1, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64ScatterStSc64U(2, zt, pg, adr); }
void CodeGenerator::st1d(const ZRegD &zt, const _PReg &pg, const AdrSc64U &adr) { Sve64ScatterStSc64U(3, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) { Sve64ScatterStSc32US(1, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) { Sve64ScatterStSc32US(2, zt, pg, adr); }
void CodeGenerator::st1d(const ZRegD &zt, const _PReg &pg, const AdrSc32US &adr) { Sve64ScatterStSc32US(3, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64ScatterStSc32UU(0, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64ScatterStSc32UU(1, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64ScatterStSc32UU(2, zt, pg, adr); }
void CodeGenerator::st1d(const ZRegD &zt, const _PReg &pg, const AdrSc32UU &adr) { Sve64ScatterStSc32UU(3, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64ScatterStVecImm(0, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64ScatterStVecImm(1, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64ScatterStVecImm(2, zt, pg, adr); }
void CodeGenerator::st1d(const ZRegD &zt, const _PReg &pg, const AdrVecImm64 &adr) { Sve64ScatterStVecImm(3, zt, pg, adr); }
void CodeGenerator::stnt1b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNTStScImm(0, zt, pg, adr); }
void CodeGenerator::stnt1b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNTStScImm(0, zt, pg, adr); }
void CodeGenerator::stnt1h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNTStScImm(1, zt, pg, adr); }
void CodeGenerator::stnt1h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNTStScImm(1, zt, pg, adr); }
void CodeGenerator::stnt1w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNTStScImm(2, zt, pg, adr); }
void CodeGenerator::stnt1w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNTStScImm(2, zt, pg, adr); }
void CodeGenerator::stnt1d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiNTStScImm(3, zt, pg, adr); }
void CodeGenerator::stnt1d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiNTStScImm(3, zt, pg, adr); }
void CodeGenerator::stnt1b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) { SveContiNTStScSc(0, zt, pg, adr); }
void CodeGenerator::stnt1h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveContiNTStScSc(1, zt, pg, adr); }
void CodeGenerator::stnt1w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiNTStScSc(2, zt, pg, adr); }
void CodeGenerator::stnt1d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiNTStScSc(3, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) { SveContiStScImm(0, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveContiStScImm(0, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiStScImm(0, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiStScImm(0, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiStScImm(0, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiStScImm(0, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiStScImm(0, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiStScImm(0, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveContiStScImm(1, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiStScImm(1, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiStScImm(1, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiStScImm(1, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiStScImm(1, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiStScImm(1, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveContiStScImm(2, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiStScImm(2, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiStScImm(2, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiStScImm(2, zt, pg, adr); }
void CodeGenerator::st1d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveContiStScImm(3, zt, pg, adr); }
void CodeGenerator::st1d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveContiStScImm(3, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) { SveContiStScSc(0, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveContiStScSc(0, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiStScSc(0, zt, pg, adr); }
void CodeGenerator::st1b(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiStScSc(0, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveContiStScSc(1, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiStScSc(1, zt, pg, adr); }
void CodeGenerator::st1h(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiStScSc(1, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveContiStScSc(2, zt, pg, adr); }
void CodeGenerator::st1w(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiStScSc(2, zt, pg, adr); }
void CodeGenerator::st1d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveContiStScSc(3, zt, pg, adr); }
void CodeGenerator::st2b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) { SveStMultiStructScImm(0, 1, zt, pg, adr); }
void CodeGenerator::st2b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveStMultiStructScImm(0, 1, zt, pg, adr); }
void CodeGenerator::st3b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) { SveStMultiStructScImm(0, 2, zt, pg, adr); }
void CodeGenerator::st3b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveStMultiStructScImm(0, 2, zt, pg, adr); }
void CodeGenerator::st4b(const ZRegB &zt, const _PReg &pg, const AdrScImm &adr) { SveStMultiStructScImm(0, 3, zt, pg, adr); }
void CodeGenerator::st4b(const ZRegB &zt, const _PReg &pg, const AdrNoOfs &adr) { SveStMultiStructScImm(0, 3, zt, pg, adr); }
void CodeGenerator::st2h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveStMultiStructScImm(1, 1, zt, pg, adr); }
void CodeGenerator::st2h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveStMultiStructScImm(1, 1, zt, pg, adr); }
void CodeGenerator::st3h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveStMultiStructScImm(1, 2, zt, pg, adr); }
void CodeGenerator::st3h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveStMultiStructScImm(1, 2, zt, pg, adr); }
void CodeGenerator::st4h(const ZRegH &zt, const _PReg &pg, const AdrScImm &adr) { SveStMultiStructScImm(1, 3, zt, pg, adr); }
void CodeGenerator::st4h(const ZRegH &zt, const _PReg &pg, const AdrNoOfs &adr) { SveStMultiStructScImm(1, 3, zt, pg, adr); }
void CodeGenerator::st2w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveStMultiStructScImm(2, 1, zt, pg, adr); }
void CodeGenerator::st2w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveStMultiStructScImm(2, 1, zt, pg, adr); }
void CodeGenerator::st3w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveStMultiStructScImm(2, 2, zt, pg, adr); }
void CodeGenerator::st3w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveStMultiStructScImm(2, 2, zt, pg, adr); }
void CodeGenerator::st4w(const ZRegS &zt, const _PReg &pg, const AdrScImm &adr) { SveStMultiStructScImm(2, 3, zt, pg, adr); }
void CodeGenerator::st4w(const ZRegS &zt, const _PReg &pg, const AdrNoOfs &adr) { SveStMultiStructScImm(2, 3, zt, pg, adr); }
void CodeGenerator::st2d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveStMultiStructScImm(3, 1, zt, pg, adr); }
void CodeGenerator::st2d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveStMultiStructScImm(3, 1, zt, pg, adr); }
void CodeGenerator::st3d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveStMultiStructScImm(3, 2, zt, pg, adr); }
void CodeGenerator::st3d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveStMultiStructScImm(3, 2, zt, pg, adr); }
void CodeGenerator::st4d(const ZRegD &zt, const _PReg &pg, const AdrScImm &adr) { SveStMultiStructScImm(3, 3, zt, pg, adr); }
void CodeGenerator::st4d(const ZRegD &zt, const _PReg &pg, const AdrNoOfs &adr) { SveStMultiStructScImm(3, 3, zt, pg, adr); }
void CodeGenerator::st2b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) { SveStMultiStructScSc(0, 1, zt, pg, adr); }
void CodeGenerator::st3b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) { SveStMultiStructScSc(0, 2, zt, pg, adr); }
void CodeGenerator::st4b(const ZRegB &zt, const _PReg &pg, const AdrScSc &adr) { SveStMultiStructScSc(0, 3, zt, pg, adr); }
void CodeGenerator::st2h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveStMultiStructScSc(1, 1, zt, pg, adr); }
void CodeGenerator::st3h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveStMultiStructScSc(1, 2, zt, pg, adr); }
void CodeGenerator::st4h(const ZRegH &zt, const _PReg &pg, const AdrScSc &adr) { SveStMultiStructScSc(1, 3, zt, pg, adr); }
void CodeGenerator::st2w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveStMultiStructScSc(2, 1, zt, pg, adr); }
void CodeGenerator::st3w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveStMultiStructScSc(2, 2, zt, pg, adr); }
void CodeGenerator::st4w(const ZRegS &zt, const _PReg &pg, const AdrScSc &adr) { SveStMultiStructScSc(2, 3, zt, pg, adr); }
void CodeGenerator::st2d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveStMultiStructScSc(3, 1, zt, pg, adr); }
void CodeGenerator::st3d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveStMultiStructScSc(3, 2, zt, pg, adr); }
void CodeGenerator::st4d(const ZRegD &zt, const _PReg &pg, const AdrScSc &adr) { SveStMultiStructScSc(3, 3, zt, pg, adr); }
void CodeGenerator::str(const _PReg &pt, const AdrScImm &adr) { SveStorePredReg(pt, adr); }
void CodeGenerator::str(const _PReg &pt, const AdrNoOfs &adr) { SveStorePredReg(pt, adr); }
void CodeGenerator::str(const ZReg &zt, const AdrScImm &adr) { SveStorePredVec(zt, adr); }
void CodeGenerator::str(const ZReg &zt, const AdrNoOfs &adr) { SveStorePredVec(zt, adr); }

void CodeGenerator::beq(const Label &label) { b(EQ, label); }
void CodeGenerator::beq(int64_t label) { b(EQ, label); }
void CodeGenerator::bne(const Label &label) { b(NE, label); }
void CodeGenerator::bne(int64_t label) { b(NE, label); }
void CodeGenerator::bcs(const Label &label) { b(CS, label); }
void CodeGenerator::bcs(int64_t label) { b(CS, label); }
void CodeGenerator::bcc(const Label &label) { b(CC, label); }
void CodeGenerator::bcc(int64_t label) { b(CC, label); }
void CodeGenerator::bmi(const Label &label) { b(MI, label); }
void CodeGenerator::bmi(int64_t label) { b(MI, label); }
void CodeGenerator::bpl(const Label &label) { b(PL, label); }
void CodeGenerator::bpl(int64_t label) { b(PL, label); }
void CodeGenerator::bvs(const Label &label) { b(VS, label); }
void CodeGenerator::bvs(int64_t label) { b(VS, label); }
void CodeGenerator::bvc(const Label &label) { b(VC, label); }
void CodeGenerator::bvc(int64_t label) { b(VC, label); }
void CodeGenerator::bhi(const Label &label) { b(HI, label); }
void CodeGenerator::bhi(int64_t label) { b(HI, label); }
void CodeGenerator::bls(const Label &label) { b(LS, label); }
void CodeGenerator::bls(int64_t label) { b(LS, label); }
void CodeGenerator::bge(const Label &label) { b(GE, label); }
void CodeGenerator::bge(int64_t label) { b(GE, label); }
void CodeGenerator::blt(const Label &label) { b(LT, label); }
void CodeGenerator::blt(int64_t label) { b(LT, label); }
void CodeGenerator::bgt(const Label &label) { b(GT, label); }
void CodeGenerator::bgt(int64_t label) { b(GT, label); }
void CodeGenerator::ble(const Label &label) { b(LE, label); }
void CodeGenerator::ble(int64_t label) { b(LE, label); }
void CodeGenerator::b_none(const Label &label) { beq(label); }
void CodeGenerator::b_none(int64_t label) { beq(label); }
void CodeGenerator::b_any(const Label &label) { bne(label); }
void CodeGenerator::b_any(int64_t label) { bne(label); }
void CodeGenerator::b_nlast(const Label &label) { bcs(label); }
void CodeGenerator::b_nlast(int64_t label) { bcs(label); }
void CodeGenerator::b_last(const Label &label) { bcc(label); }
void CodeGenerator::b_last(int64_t label) { bcc(label); }
void CodeGenerator::b_first(const Label &label) { bmi(label); }
void CodeGenerator::b_first(int64_t label) { bmi(label); }
void CodeGenerator::b_nfrst(const Label &label) { bpl(label); }
void CodeGenerator::b_nfrst(int64_t label) { bpl(label); }
void CodeGenerator::b_pmore(const Label &label) { bhi(label); }
void CodeGenerator::b_pmore(int64_t label) { bhi(label); }
void CodeGenerator::b_plast(const Label &label) { bls(label); }
void CodeGenerator::b_plast(int64_t label) { bls(label); }
void CodeGenerator::b_tcont(const Label &label) { bge(label); }
void CodeGenerator::b_tcont(int64_t label) { bge(label); }
void CodeGenerator::b_tstop(const Label &label) { blt(label); }
void CodeGenerator::b_tstop(int64_t label) { blt(label); }

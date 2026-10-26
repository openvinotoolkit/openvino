//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bytecode_builder.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "utils/common.hpp"
#include "vm_runtime/virtual_machine.h"
#include "vm_value_helpers.hpp"

#include <llvm/Support/FormatVariadic.h>

#include <common_test_utils/test_common.hpp>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

using namespace intel_npu::vm;
using namespace utils;

namespace {

struct CmpInstructionParams {
    OpCode opcode;
    int64_t lhs;
    int64_t rhs;
    CmpPredicate predicate;
    std::optional<bool> isSigned;
    bool expected;

    CmpInstructionParams(OpCode opcode, int64_t lhs, int64_t rhs, CmpPredicate predicate, bool isSigned, bool expected)
            : opcode(opcode), lhs(lhs), rhs(rhs), predicate(predicate), isSigned(isSigned), expected(expected) {
    }
    CmpInstructionParams(OpCode opcode, int64_t lhs, int64_t rhs, CmpPredicate predicate, bool expected)
            : opcode(opcode), lhs(lhs), rhs(rhs), predicate(predicate), isSigned(std::nullopt), expected(expected) {
    }

    [[maybe_unused]] friend std::ostream& operator<<(std::ostream& os, const CmpInstructionParams& params) {
        os << llvm::formatv("{ opcode={0}, lhs={1}, rhs={2}, predicate={3}, isSigned={4}, expected={5} }",
                            static_cast<uint16_t>(params.opcode), params.lhs, params.rhs,
                            static_cast<uint16_t>(params.predicate),
                            params.isSigned.has_value() ? std::to_string(*params.isSigned) : "none", params.expected)
                        .str();
        return os;
    }
};

class VirtualMachineComparisonInstructionTest :
        public VirtualMachineInstructionTest,
        public testing::WithParamInterface<CmpInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CmpInstructionParams>& obj) {
        const auto predicateToStr = [](CmpPredicate predicate) {
            switch (predicate) {
            case CmpPredicate::EQ:
                return "EQ";
            case CmpPredicate::NE:
                return "NE";
            case CmpPredicate::GT:
                return "GT";
            case CmpPredicate::GTE:
                return "GTE";
            case CmpPredicate::LT:
                return "LT";
            case CmpPredicate::LTE:
                return "LTE";
            default:
                return "unknown";
            }
        };

        std::ostringstream result;
        result << "opcode=" << static_cast<uint16_t>(obj.param.opcode) << "_";
        result << "lhs=" << obj.param.lhs << "_";
        result << "rhs=" << obj.param.rhs << "_";
        result << "predicate=" << predicateToStr(obj.param.predicate) << "_";
        result << "isSigned=" << (obj.param.isSigned.has_value() ? std::to_string(*obj.param.isSigned) : "none") << "_";
        result << "expected=" << obj.param.expected;
        return result.str();
    }
};

}  // namespace

TEST_P(VirtualMachineComparisonInstructionTest, Execute) {
    const auto& params = GetParam();

    auto flag = static_cast<int16_t>(params.predicate);
    if (params.isSigned.value_or(false)) {
        constexpr auto numBitsCmpFn = 8;
        flag |= (1 << numBitsCmpFn);
    }

    const auto funcName = "test";
    auto bytecode =
            BytecodeBuilder{}
                    .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/3,
                                                                  {getInt64Type(), getInt64Type()}, {getInt64Type()},
                                                                  /*isEntrypoint=*/true)
                                         .instruction(params.opcode, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2, /*flag=*/flag})
                                         .retv(/*resultRegs=*/{0}))
                    .build();
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {intel_npu::vm::makeI64(params.lhs), intel_npu::vm::makeI64(params.rhs)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(funcName, args, results), NPU_VM_SUCCESS);
    EXPECT_EQ(static_cast<bool>(results.at(0).i64), params.expected);
}

INSTANTIATE_TEST_SUITE_P(
        CMP_I64, VirtualMachineComparisonInstructionTest,
        testing::Values(
                // Equal
                CmpInstructionParams{OpCode::CMP_I64, 0, 0, CmpPredicate::EQ, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 0, 5, CmpPredicate::EQ, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, -5, 0, CmpPredicate::EQ, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, 5, 6, CmpPredicate::EQ, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 5, 5, CmpPredicate::EQ, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 6, 5, CmpPredicate::EQ, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, -5, -5, CmpPredicate::EQ, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, -6, -5, CmpPredicate::EQ, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, MIN_I64, MIN_I64, CmpPredicate::EQ, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, MAX_I64, MAX_I64, CmpPredicate::EQ, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, MIN_I64, MAX_I64, CmpPredicate::EQ, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::EQ,
                                     /*isSigned=*/false, false},  // 0xFFFFFFFFFFFFFFFF is a large unsigned value
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::EQ,
                                     /*isSigned=*/true, false},  // 0xFFFFFFFFFFFFFFFF is -1
                // Not equal
                CmpInstructionParams{OpCode::CMP_I64, 0, 0, CmpPredicate::NE, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 0, 5, CmpPredicate::NE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, -5, 0, CmpPredicate::NE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, 5, 6, CmpPredicate::NE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 5, 5, CmpPredicate::NE, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 6, 5, CmpPredicate::NE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, -5, -5, CmpPredicate::NE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, -6, -5, CmpPredicate::NE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, MIN_I64, MIN_I64, CmpPredicate::NE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, MAX_I64, MAX_I64, CmpPredicate::NE, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, MIN_I64, MAX_I64, CmpPredicate::NE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::NE,
                                     /*isSigned=*/false, true},  // 0xFFFFFFFFFFFFFFFF is a large unsigned value
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::NE,
                                     /*isSigned=*/true, true},  // 0xFFFFFFFFFFFFFFFF is -1
                // Greater than
                CmpInstructionParams{OpCode::CMP_I64, 0, 0, CmpPredicate::GT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 0, 5, CmpPredicate::GT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, -5, 0, CmpPredicate::GT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, 5, 6, CmpPredicate::GT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 5, 5, CmpPredicate::GT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 6, 5, CmpPredicate::GT, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, -5, -6, CmpPredicate::GT, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, -6, -5, CmpPredicate::GT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, MIN_I64, MIN_I64, CmpPredicate::GT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, MAX_I64, MAX_I64, CmpPredicate::GT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, MIN_I64, MAX_I64, CmpPredicate::GT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, MAX_I64, MIN_I64, CmpPredicate::GT, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::GT,
                                     /*isSigned=*/false, false},  // 0xFFFFFFFFFFFFFFFF is a large unsigned value
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::GT,
                                     /*isSigned=*/true, true},  // 0xFFFFFFFFFFFFFFFF is -1
                // Greater than or equal
                CmpInstructionParams{OpCode::CMP_I64, 0, 0, CmpPredicate::GTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 0, 5, CmpPredicate::GTE, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, -5, 0, CmpPredicate::GTE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, 5, 6, CmpPredicate::GTE, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 5, 5, CmpPredicate::GTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 6, 5, CmpPredicate::GTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, -5, -6, CmpPredicate::GTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, -6, -5, CmpPredicate::GTE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, MIN_I64, MIN_I64, CmpPredicate::GTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, MAX_I64, MAX_I64, CmpPredicate::GTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, MIN_I64, MAX_I64, CmpPredicate::GTE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, MAX_I64, MIN_I64, CmpPredicate::GTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::GTE,
                                     /*isSigned=*/false, false},  // 0xFFFFFFFFFFFFFFFF is a large unsigned value
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::GTE,
                                     /*isSigned=*/true, true},  // 0xFFFFFFFFFFFFFFFF is -1
                // Less than
                CmpInstructionParams{OpCode::CMP_I64, 0, 0, CmpPredicate::LT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 0, 5, CmpPredicate::LT, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, -5, 0, CmpPredicate::LT, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, 5, 6, CmpPredicate::LT, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 5, 5, CmpPredicate::LT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, 6, 5, CmpPredicate::LT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, -5, -6, CmpPredicate::LT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, -6, -5, CmpPredicate::LT, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, MIN_I64, MIN_I64, CmpPredicate::LT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, MAX_I64, MAX_I64, CmpPredicate::LT, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, MIN_I64, MAX_I64, CmpPredicate::LT, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, MAX_I64, MIN_I64, CmpPredicate::LT, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::LT,
                                     /*isSigned=*/false, true},  // 0xFFFFFFFFFFFFFFFF is a large unsigned value
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::LT,
                                     /*isSigned=*/true, false},  // 0xFFFFFFFFFFFFFFFF is -1
                // Less than or equal
                CmpInstructionParams{OpCode::CMP_I64, 0, 0, CmpPredicate::LTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 0, 5, CmpPredicate::LTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, -5, 0, CmpPredicate::LTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, 5, 6, CmpPredicate::LTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 5, 5, CmpPredicate::LTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, 6, 5, CmpPredicate::LTE, /*isSigned=*/false, false},
                CmpInstructionParams{OpCode::CMP_I64, -5, -6, CmpPredicate::LTE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, -6, -5, CmpPredicate::LTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, MIN_I64, MIN_I64, CmpPredicate::LTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, MAX_I64, MAX_I64, CmpPredicate::LTE, /*isSigned=*/false, true},
                CmpInstructionParams{OpCode::CMP_I64, MIN_I64, MAX_I64, CmpPredicate::LTE, /*isSigned=*/true, true},
                CmpInstructionParams{OpCode::CMP_I64, MAX_I64, MIN_I64, CmpPredicate::LTE, /*isSigned=*/true, false},
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::LTE,
                                     /*isSigned=*/false, true},  // 0xFFFFFFFFFFFFFFFF is a large unsigned value
                CmpInstructionParams{OpCode::CMP_I64, 0, static_cast<int64_t>(0xFFFFFFFFFFFFFFFF), CmpPredicate::LTE,
                                     /*isSigned=*/true, false}),  // 0xFFFFFFFFFFFFFFFF is -1
        VirtualMachineComparisonInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        CMP_F64, VirtualMachineComparisonInstructionTest,
        testing::Values(
                // Equal
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(0.0), CmpPredicate::EQ, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(-0.0), CmpPredicate::EQ, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(1.0), CmpPredicate::EQ, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(-1.0), CmpPredicate::EQ, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(1.0), CmpPredicate::EQ, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(-1.0), CmpPredicate::EQ, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(INF_F64), f64Bits(INF_F64), CmpPredicate::EQ, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-INF_F64), f64Bits(-INF_F64), CmpPredicate::EQ, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(INF_F64), f64Bits(-INF_F64), CmpPredicate::EQ, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(NAN_F64), f64Bits(NAN_F64), CmpPredicate::EQ, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(NAN_F64), f64Bits(1.0), CmpPredicate::EQ, false},
                // Not equal
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(0.0), CmpPredicate::NE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(-0.0), CmpPredicate::NE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(1.0), CmpPredicate::NE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(-1.0), CmpPredicate::NE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(1.0), CmpPredicate::NE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(-1.0), CmpPredicate::NE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(INF_F64), f64Bits(INF_F64), CmpPredicate::NE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-INF_F64), f64Bits(-INF_F64), CmpPredicate::NE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(INF_F64), f64Bits(-INF_F64), CmpPredicate::NE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(NAN_F64), f64Bits(NAN_F64), CmpPredicate::NE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(NAN_F64), f64Bits(1.0), CmpPredicate::NE, true},
                // Greater than
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(0.0), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(-0.0), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(1.0), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(-1.0), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(1.0), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(-1.0), CmpPredicate::GT, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(INF_F64), f64Bits(INF_F64), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-INF_F64), f64Bits(-INF_F64), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(INF_F64), f64Bits(-INF_F64), CmpPredicate::GT, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(NAN_F64), f64Bits(NAN_F64), CmpPredicate::GT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(NAN_F64), f64Bits(1.0), CmpPredicate::GT, false},
                // Greater than or equal
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(0.0), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(-0.0), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(1.0), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(-1.0), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(1.0), CmpPredicate::GTE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(-1.0), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(INF_F64), f64Bits(INF_F64), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-INF_F64), f64Bits(-INF_F64), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(INF_F64), f64Bits(-INF_F64), CmpPredicate::GTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(NAN_F64), f64Bits(NAN_F64), CmpPredicate::GTE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(NAN_F64), f64Bits(1.0), CmpPredicate::GTE, false},
                // Less than
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(0.0), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(-0.0), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(1.0), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(-1.0), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(1.0), CmpPredicate::LT, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(-1.0), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(INF_F64), f64Bits(INF_F64), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-INF_F64), f64Bits(-INF_F64), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(INF_F64), f64Bits(-INF_F64), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(NAN_F64), f64Bits(NAN_F64), CmpPredicate::LT, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(NAN_F64), f64Bits(1.0), CmpPredicate::LT, false},
                // Less than or equal
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(0.0), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(0.0), f64Bits(-0.0), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(1.0), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(-1.0), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-1.0), f64Bits(1.0), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(1.0), f64Bits(-1.0), CmpPredicate::LTE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(INF_F64), f64Bits(INF_F64), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(-INF_F64), f64Bits(-INF_F64), CmpPredicate::LTE, true},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(INF_F64), f64Bits(-INF_F64), CmpPredicate::LTE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(NAN_F64), f64Bits(NAN_F64), CmpPredicate::LTE, false},
                CmpInstructionParams{OpCode::CMP_F64, f64Bits(NAN_F64), f64Bits(1.0), CmpPredicate::LTE, false}),
        VirtualMachineComparisonInstructionTest::getTestCaseName);

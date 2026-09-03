// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Coverage gate: every ggml op registered in op_table.cpp must be exercised by some test in this
// binary.
//
// Without this, the per-op suite silently stops keeping pace with the op table -- adding a
// translator and forgetting its test is invisible, and a wrong-but-plausible formula ships. That is
// not hypothetical: GGML_UNARY_OP_GELU_QUICK was registered with the tanh-GELU formula instead of
// ggml's x*sigmoid(1.702x) and went unnoticed because nothing converted it.
//
// The "tested" side of the comparison is collected at run time: SingleOpDecoder's constructor
// records its op type in converted_op_types(), so the record cannot drift from what the tests
// actually do (a hand-written list would just be a second thing to forget). Consequently this check
// must run AFTER all other tests, which gtest guarantees for a global test environment's TearDown --
// so the assertion lives there rather than in a TEST body.
//
// A gtest --gtest_filter that excludes op tests would leave the record incomplete and the gate would
// fire spuriously, so it only asserts when the full suite ran (no filter narrowing in effect).

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "op_table.hpp"
#include "op_test_utils.hpp"

using namespace ov_gguf_test;

namespace {

// Ops that are registered but intentionally not covered by a single-op test, each with the reason.
// Anything here needs a justification that is about the op's nature, not about effort -- an op that
// is merely awkward to test belongs in the suite, not on this list.
const std::set<std::string>& coverage_exemptions() {
    static const std::set<std::string> exemptions{
        // Aliases of ops already covered under a different ggml name, translated by the very same
        // function pointer, so a separate case would test the same code path twice.
        // (none currently -- GGML_OP_ADD1 has its own test because its broadcast shape differs)
    };
    return exemptions;
}

class OpCoverageEnvironment : public ::testing::Environment {
public:
    void TearDown() override {
        // Only meaningful when the whole suite ran; a narrowing filter makes the record partial.
        const std::string filter = ::testing::GTEST_FLAG(filter);
        if (filter != "*" && filter != "*.*") {
            GTEST_LOG_(INFO) << "op coverage gate skipped: --gtest_filter=" << filter << " is in effect";
            return;
        }

        const auto& converted = converted_op_types();
        std::vector<std::string> missing;
        for (const auto& entry : ov::frontend::gguf::get_supported_ops()) {
            const std::string& op = entry.first;
            if (converted.count(op) == 0 && coverage_exemptions().count(op) == 0) {
                missing.push_back(op);
            }
        }
        std::sort(missing.begin(), missing.end());

        if (!missing.empty()) {
            std::string list;
            for (const auto& op : missing) {
                list += "\n    " + op;
            }
            // Failures raised from an Environment's TearDown are counted separately from tests, so
            // the run reports "0 FAILED TESTS" while still exiting non-zero. Name the check
            // explicitly so the CI log is not misread as a spurious failure.
            ADD_FAILURE() << "[GGUF op coverage gate] " << missing.size()
                          << " op(s) registered in op_table.cpp have no test in ov_gguf_frontend_tests:" << list
                          << "\nAdd a case to test_ops.cpp (or test_weights.cpp for weight leaves). If the op "
                          << "genuinely cannot be tested in isolation, add it to coverage_exemptions() in "
                          << "test_op_coverage.cpp with the reason.";
        }
    }
};

// Registered at static-init time; gtest runs environment TearDown after the last test.
const auto* const op_coverage_env = ::testing::AddGlobalTestEnvironment(new OpCoverageEnvironment());

}  // namespace

// Guard the guard: the coverage record must be non-empty and must contain ops the suite obviously
// converts. If SingleOpDecoder ever stops recording, the TearDown check above would pass vacuously
// for an empty op table and fail confusingly otherwise; this makes the wiring itself testable.
TEST(GGUFOpCoverage, RecordIsPopulated) {
    // This test's own decoder construction guarantees at least one entry regardless of test order.
    SingleOpBuilder()
        .op("GGML_OP_ADD")
        .input("a", ov::element::f32, {1})
        .output("out", ov::element::f32, {1})
        .decoder();
    EXPECT_NE(converted_op_types().count("GGML_OP_ADD"), 0u);
}

// The op table itself must be non-degenerate: a build that dropped the registrations would make the
// coverage gate above pass trivially.
TEST(GGUFOpCoverage, OpTableIsNonEmpty) {
    EXPECT_GT(ov::frontend::gguf::get_supported_ops().size(), 50u);
}

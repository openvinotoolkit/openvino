// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <cstdlib>

#include "openvino/core/model.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/util/env_util.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

// RAII helper for MLIR op tests. Sets 'OV_MLIR_PATTERNS' to an empty
// string (match-all to mlir) for the lifetime of the test-class object.
struct MlirMatchAllEnv {
    MlirMatchAllEnv() {
        // Respect an explicitly provided value: only inject "match all" when the
        // variable is not already set, and only then restore (unset) it later.
        if (std::getenv("OV_MLIR_PATTERNS") == nullptr) {
            m_owned = true;
            setenv("OV_MLIR_PATTERNS", "", /*overwrite=*/1);
        }
    }

    ~MlirMatchAllEnv() {
        if (m_owned) {
            unsetenv("OV_MLIR_PATTERNS");
        }
    }

    MlirMatchAllEnv(const MlirMatchAllEnv&) = delete;
    MlirMatchAllEnv& operator=(const MlirMatchAllEnv&) = delete;

private:
    bool m_owned = false;
};

inline bool is_mlir_enabled() {
    return ov::util::getenv_bool("OV_GPU_ENABLE_MLIR");
}

// Returns true if the compiled model's runtime graph contains at least one MLIROp.
inline bool has_mlir_op(const ov::CompiledModel& compiled) {
    const auto exec_model = compiled.get_runtime_model();
    if (!exec_model) {
        return false;
    }
    for (const auto& node : exec_model->get_ordered_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto it = rt_info.find(ov::exec_model_info::LAYER_TYPE);
        if (it == rt_info.end()) {
            continue;
        }
        const auto layer_type = it->second.as<std::string>();
        if (layer_type == "mlir_primitive" || layer_type == "MLIROp") {
            return true;
        }
    }
    return false;
}

// Common base for all MLIR op tests.
//
// - Sets OV_MLIR_PATTERNS="" for the test lifetime (match-all), see MlirMatchAllEnv.
// - After run(), verifies that the MLIR path actually was actually involved: when
//   OV_GPU_ENABLE_MLIR is on, at least one MLIROp must appear in the runtime graph.
template <typename Base>
class MlirTestFixture : public Base {
protected:
    void run() override {
        Base::run();
        if (m_check_mlir_execution) {
            check_mlir_execution();
        }
    }

    virtual void check_mlir_execution() {
        if (!is_mlir_enabled()) {
            GTEST_LOG_(WARNING) << "Skipping MLIROp presence check: 'OV_GPU_ENABLE_MLIR' is not set. "
                                << "The model was compiled without the MLIR path.";
            return;
        }
        EXPECT_TRUE(ov::test::has_mlir_op(this->compiledModel)) << "Expected at least one MLIROp in the execution graph, but none was found.";
    }

    bool m_check_mlir_execution = true;

private:
    MlirMatchAllEnv m_match_all_env;
};

using MlirSubgraphTest = MlirTestFixture<ov::test::SubgraphBaseTest>;
using MlirSubgraphStaticTest = MlirTestFixture<ov::test::SubgraphBaseStaticTest>;

}  // namespace ov::test

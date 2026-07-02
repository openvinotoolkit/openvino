// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <functional>
#include <string>

#include "common_test_utils/test_constants.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/exec_model_info.hpp"

namespace ov::test {

static std::string find_impl_type(const std::shared_ptr<const ov::Model>& m, const std::string& layer) {
    for (const auto& n : m->get_ops()) {
        const auto& rt = n->get_rt_info();
        auto lt = rt.find(ov::exec_model_info::LAYER_TYPE);
        if (lt == rt.end() || lt->second.as<std::string>() != layer)
            continue;
        auto it = rt.find(ov::exec_model_info::IMPL_TYPE);
        return it != rt.end() ? it->second.as<std::string>() : "";
    }
    return "";
}


// ---------------------------------------------------------------------------
// Test scheme
// ---------------------------------------------------------------------------
//
// Compile a single-op dynamic model with 2 streams, run exactly 1 inference
// from the main thread, then call get_runtime_model() and assert the target
// node's IMPL_TYPE does not contain "unknown".
//
// Why graph[0] is reliably untouched:
//   Worker threads share one task queue, woken via notify_one().
//   Linux/glibc pthread_cond_signal is LIFO: the last thread to sleep is woken
//   first.  After compile_model init, thread(loop=0, stream_id=1) is the last
//   sleeper, so the single inference always lands on graph[1].  graph[0] is
//   never touched — prepareParams is never called on it — its nodes still hold
//   impl_desc_type::unknown.
//
//   The main thread's stream_id = 2 (workers claimed 0,1), so
//   get_runtime_model() always maps to graph[2 % 2] = graph[0].
//
//   WITHOUT fix: returns "unknown_*" → FAIL.
//   WITH fix:    m_inferenceHappened is set on graph[1]; the loop in
//                get_runtime_model() returns graph[1] → real IMPL_TYPE → PASS.

using ModelBuilder = std::function<std::shared_ptr<ov::Model>()>;
struct Param {
    std::string label;
    ModelBuilder build;
    std::string node_type;
};

class NoUnknownImplTypeTest : public ::testing::TestWithParam<Param> {
protected:
    ov::Core core;
};

TEST_P(NoUnknownImplTypeTest, smoke) {
    const auto& p = GetParam();
    auto cm = core.compile_model(p.build(), utils::DEVICE_CPU, {ov::num_streams(2)});

    auto req = cm.create_infer_request();
    req.set_input_tensor(ov::Tensor{cm.input(0).get_element_type(), {2, 4, 4, 1}});
    req.infer();

    const auto impl_type = find_impl_type(cm.get_runtime_model(), p.node_type);
    ASSERT_FALSE(impl_type.empty()) << p.node_type << " node not found in runtime model";
    EXPECT_EQ(impl_type.find("unknown"), std::string::npos)
        << p.node_type << " impl_type='" << impl_type << "' must not contain 'unknown'";
}

INSTANTIATE_TEST_SUITE_P(
    smoke_RuntimeModelImplType,
    NoUnknownImplTypeTest,
    ::testing::Values(
        Param{"Convert_2streams_graph0_untouched",
              [] {
                  auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
                  return std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Convert>(p, ov::element::f16)},
                                                    ov::ParameterVector{p});
              },
              "Convert"},
        Param{"Transpose_2streams_graph0_untouched",
              [] {
                  auto p = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
                  auto order = ov::op::v0::Constant::create(ov::element::i32, {4}, {0, 2, 3, 1});
                  return std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v1::Transpose>(p, order)},
                                                    ov::ParameterVector{p});
              },
              "Transpose"}),
    [](const ::testing::TestParamInfo<Param>& info) {
        return info.param.label;
    });

}  // namespace ov::test

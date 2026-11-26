// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef OV_GPU_WITH_OCL_RT

#    include "subgraph_tests/runtime_weights_update.hpp"

#    include <sstream>
#    include <tuple>

#    include "openvino/runtime/intel_gpu/ocl/ocl.hpp"

namespace ov::test::gpu {
namespace {
using UserQueueWeightsParams = std::tuple<bool, bool, bool, InputShape, bool>;

class UserQueueWeightsUpdateTest : public RuntimeWeightsUpdateTestBase, public testing::WithParamInterface<UserQueueWeightsParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<UserQueueWeightsParams>& obj) {
        const auto& [matmul, compressed, constant_weights, input_shape, out_of_order] = obj.param;
        std::ostringstream result;
        result << (matmul ? "MatMul" : "Convolution") << (compressed ? "_Compressed" : "_F16") << (constant_weights ? "_Constant" : "_Runtime") << "_"
               << "IS=" << ov::test::utils::partialShape2str({input_shape.first});
        for (const auto& shape : input_shape.second) {
            result << "_TS=" << ov::test::utils::vec2str(shape);
        }
        result << (out_of_order ? "_OutOfOrder" : "_InOrder");
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [matmul, compressed, constant_weights, input_shape, out_of_order] = GetParam();
        setup_model(matmul, compressed, constant_weights, input_shape);
    }

    void compile_model() override {
        configure_model();
        core_configuration(this);
        auto context = core->get_default_context(targetDevice).as<ov::intel_gpu::ocl::ClContext>();
        cl::Context cl_context(context.get(), true);
        const auto devices = cl_context.getInfo<CL_CONTEXT_DEVICES>();
        ASSERT_EQ(devices.size(), 1u);
        const auto properties = std::get<4>(GetParam()) ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0;
        queue = cl::CommandQueue(cl_context, devices.front(), properties);
        auto external_context = ov::intel_gpu::ocl::ClContext(*core, queue());
        compiledModel = core->compile_model(function, external_context, configuration);
        inference_precision = compiledModel.get_property(ov::hint::inference_precision);
    }

private:
    cl::CommandQueue queue;
};

TEST_P(UserQueueWeightsUpdateTest, uses_current_weights) {
    run();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_MatMul,
    UserQueueWeightsUpdateTest,
    testing::Combine(testing::Values(true), testing::Bool(), testing::Bool(), testing::ValuesIn(runtime_weights_input_shapes(true)), testing::Bool()),
    UserQueueWeightsUpdateTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Convolution,
    UserQueueWeightsUpdateTest,
    testing::Combine(testing::Values(false), testing::Bool(), testing::Bool(), testing::ValuesIn(runtime_weights_input_shapes(false)), testing::Bool()),
    UserQueueWeightsUpdateTest::getTestCaseName);
}  // namespace
}  // namespace ov::test::gpu
#endif

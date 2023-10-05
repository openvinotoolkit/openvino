// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/reference/multiply.hpp>
#include <thread>
#include <utility>
#include <vector>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/test_common.hpp"
#include "utils/denormals.hpp"
#include "openvino/core/parallel.hpp"

using namespace testing;

class DenormalsOptimizeTestF : public ov::test::TestsCommon {
public:
    DenormalsOptimizeTestF() {
        // Default setting
        ov::intel_cpu::flush_to_zero(false);
        ov::intel_cpu::denormals_as_zero(false);

        for (int i = 0; i < 3; i++) {
            vecConst.emplace_back(new float[shape_size(inpShape)]);
        }
        input1 = ov::Tensor(ov::element::f32, inpShape, vecConst[0]);
        input2 = ov::Tensor(ov::element::f32, inpShape, vecConst[1]);
    }
    ~DenormalsOptimizeTestF() {
        for (size_t i = 0; i < vecConst.size(); i++) {
            if (vecConst[i])
                delete[] vecConst[i];
        }

        // Recover the default setting to avoid impacting other tests.
        ov::intel_cpu::flush_to_zero(false);
        ov::intel_cpu::denormals_as_zero(false);
    }

    void SetUp() override {
        initBuf(vecConst[0], shape_size(inpShape), std::numeric_limits<float>::min());
        initBuf(vecConst[1], shape_size(inpShape), 0.5f);
        initModel();
    }

    void TearDown() override {}

    ov::Shape inpShape = ov::Shape{1, 5, 12, 64};
    ov::element::Type rtPrc = ov::element::f32;
    std::vector<float*> vecConst;
    ov::Tensor input1, input2;
    std::shared_ptr<ov::Model> model;

    ov::Core core = ov::Core();
    ov::Tensor outTensor;

    void run_reference_multiply() {
        // Subnormal optimization is not used in the current thread.
        ov::reference::multiply<float>(vecConst[0], vecConst[1], vecConst[2], shape_size(inpShape));
        for (size_t i = 0; i < shape_size(inpShape); i++) {
            EXPECT_GT(vecConst[2][i], 0);
        }
    }

    void set_denormals_optimization(bool opt = false) {
        core.set_property("CPU", {{"CPU_DENORMALS_OPTIMIZATION", opt ? "YES" : "NO"}});
    }

    void run(bool sync) {
        auto compiledModel = core.compile_model(model, "CPU");
        auto inferRequest = compiledModel.create_infer_request();

        inferRequest.set_input_tensors(0, {input1});
        inferRequest.set_input_tensors(1, {input2});

        if (sync) {
            inferRequest.infer();
        } else {
            inferRequest.start_async();
            inferRequest.wait();
        }

        outTensor = inferRequest.get_output_tensor();

        // Check output shape
        const auto& outShape = outTensor.get_shape();
        EXPECT_EQ(inpShape.size(), outShape.size());
        for (size_t i = 0; i < inpShape.size(); i++) {
            EXPECT_EQ(inpShape[i], outShape[i]);
        }
    }

    void checkOutput(bool expectEQZero) {
        const float* ouptputData = outTensor.data<float>();
        for (size_t i = 0; i < outTensor.get_size(); i++) {
            if (expectEQZero) {
                EXPECT_EQ(ouptputData[i], 0);
            } else {
                EXPECT_GT(ouptputData[i], 0);
            }
        }
    }

private:
    void initBuf(float* pConst, size_t sz, float value) {
        for (size_t i = 0; i < shape_size(inpShape); ++i) {
            pConst[i] = value;
        }
    }

    void initModel() {
        auto params1 = std::make_shared<ov::opset1::Parameter>(rtPrc, inpShape);
        auto params2 = std::make_shared<ov::opset1::Parameter>(rtPrc, inpShape);
        auto mul = std::make_shared<ov::opset1::Multiply>(params1, params2);
        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{params1, params2});
    }
};

TEST_F(DenormalsOptimizeTestF, Sync) {
    run_reference_multiply();

    run(true);

#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
    checkOutput(true);

    // To prove that there is no impact on customers' applications for the default setting.
    run_reference_multiply();
#else
    checkOutput(false);
#endif
}

TEST_F(DenormalsOptimizeTestF, Sync_Opt_YES) {
    run_reference_multiply();

    set_denormals_optimization(true);

    run(true);

    checkOutput(true);
}

TEST_F(DenormalsOptimizeTestF, Sync_Opt_NO) {
    run_reference_multiply();

    set_denormals_optimization(false);

    run(true);

    checkOutput(false);
}

TEST_F(DenormalsOptimizeTestF, Async) {
    run_reference_multiply();

    run(false);

#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
    checkOutput(true);

    // To prove that there is no impact on customers' applications for the default setting.
    run_reference_multiply();
#else
    checkOutput(false);
#endif
}

TEST_F(DenormalsOptimizeTestF, Async_Opt_YES) {
    run_reference_multiply();

    set_denormals_optimization(true);

    run(false);

    checkOutput(true);
}

TEST_F(DenormalsOptimizeTestF, Async_Opt_NO) {
    run_reference_multiply();

    set_denormals_optimization(false);

    run(false);

    checkOutput(false);
}

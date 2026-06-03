// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gated_delta_net.hpp"

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/core.hpp"

namespace {

using GatedDeltaNetParams = std::tuple<std::vector<ov::Shape>,  // Input shapes: query, key, value, state, gate, beta
                                       ov::element::Type,       // Input precision
                                       bool>;                   // fuse_qk_l2norm

class GatedDeltaNetStaticTest : public testing::WithParamInterface<GatedDeltaNetParams>, public ov::test::TestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatedDeltaNetParams>& obj) {
        const auto& [input_shapes, precision, fuse_qk_l2norm] = obj.param;

        std::ostringstream result;
        result << "IS=";
        for (size_t i = 0; i < input_shapes.size(); i++) {
            result << ov::test::utils::vec2str(input_shapes[i]);
            if (i < input_shapes.size() - 1)
                result << "_";
        }
        result << "_prec=" << precision;
        result << "_fuse_l2norm=" << fuse_qk_l2norm;
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [input_shapes, precision, fuse_qk_l2norm] = GetParam();

        auto query = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
        auto key = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[1]);
        auto value = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[2]);
        auto state = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[3]);
        auto gate = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[4]);
        auto beta = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[5]);

        auto gdn = std::make_shared<ov::op::internal::GatedDeltaNet>(query, key, value, state, gate, beta, fuse_qk_l2norm);

        auto result0 = std::make_shared<ov::op::v0::Result>(gdn->output(0));
        auto result1 = std::make_shared<ov::op::v0::Result>(gdn->output(1));

        model = std::make_shared<ov::Model>(ov::ResultVector{result0, result1}, ov::ParameterVector{query, key, value, state, gate, beta}, "GatedDeltaNetTest");
    }

    std::map<std::shared_ptr<ov::op::v0::Parameter>, ov::Tensor> generate_inputs() {
        std::map<std::shared_ptr<ov::op::v0::Parameter>, ov::Tensor> inputs;
        const auto& params = model->get_parameters();
        for (size_t i = 0; i < params.size(); ++i) {
            ov::test::utils::InputGenerateData in_data(0.0, 1, 1000, 1);
            if (i == 4) {
                in_data = ov::test::utils::InputGenerateData(-1, 1, 1000, 1);
            }
            inputs[params[i]] = ov::test::utils::create_and_fill_tensor(params[i]->get_element_type(), params[i]->get_shape(), in_data);
        }
        return inputs;
    }

    std::shared_ptr<ov::Model> model;
};

TEST_P(GatedDeltaNetStaticTest, CompareWithTemplate) {
    auto inputs = generate_inputs();

    // Build input tensor vector for infer_on_template
    ov::TensorVector input_tensors;
    for (const auto& param : model->get_parameters()) {
        input_tensors.push_back(inputs.at(param));
    }

    // Run on TEMPLATE (reference)
    auto ref_outputs = ov::test::utils::infer_on_template(model, input_tensors);

    // Run on GPU
    ov::Core core;
    auto compiled_gpu = core.compile_model(model, "GPU");
    auto req_gpu = compiled_gpu.create_infer_request();
    for (const auto& [param, tensor] : inputs) {
        req_gpu.set_tensor(param->output(0), tensor);
    }
    req_gpu.infer();

    // Compare outputs
    for (size_t i = 0; i < model->get_output_size(); i++) {
        auto out_gpu = req_gpu.get_output_tensor(i);
        ov::test::utils::compare(ref_outputs[i], out_gpu, 1e-2, 1e-2);
    }
}

// Shapes: query[B,S,qk_H,D], key[B,S,qk_H,D], value[B,S,v_H,Dv], state[B,v_H,D,Dv], gate[B,S,v_H], beta[B,S,v_H]
const std::vector<std::vector<ov::Shape>> static_shapes = {
    // B=1, S=1, qk_H=4, v_H=4, D=16, Dv=16 (minimal)
    {{1, 1, 4, 16}, {1, 1, 4, 16}, {1, 1, 4, 16}, {1, 4, 16, 16}, {1, 1, 4}, {1, 1, 4}},
    // B=1, S=1, qk_H=32, v_H=32, D=128, Dv=128 (typical LLM decode)
    {{1, 1, 32, 128}, {1, 1, 32, 128}, {1, 1, 32, 128}, {1, 32, 128, 128}, {1, 1, 32}, {1, 1, 32}},
    // B=1, S=16, qk_H=2, v_H=2, D=16, Dv=32 (seq_len > 1, different D and Dv)
    {{1, 16, 2, 16}, {1, 16, 2, 16}, {1, 16, 2, 32}, {1, 2, 16, 32}, {1, 16, 2}, {1, 16, 2}},
    // B=2, S=1, qk_H=8, v_H=8, D=64, Dv=64 (batch > 1)
    {{2, 1, 8, 64}, {2, 1, 8, 64}, {2, 1, 8, 64}, {2, 8, 64, 64}, {2, 1, 8}, {2, 1, 8}},
    // B=1, S=4, qk_H=2, v_H=8, D=16, Dv=16 (GQA: v_H is multiple of qk_H)
    {{1, 4, 2, 16}, {1, 4, 2, 16}, {1, 4, 8, 16}, {1, 8, 16, 16}, {1, 4, 8}, {1, 4, 8}},
};

INSTANTIATE_TEST_SUITE_P(smoke_GatedDeltaNetStatic,
                         GatedDeltaNetStaticTest,
                         ::testing::Combine(::testing::ValuesIn(static_shapes), ::testing::Values(ov::element::f32), ::testing::Values(false, true)),
                         GatedDeltaNetStaticTest::getTestCaseName);

}  // namespace

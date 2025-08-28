// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/engine_configuration.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"

#include "openvino/opsets/opset13.hpp"

#include "openvino/util/log.hpp"
#include "intel_gpu/runtime/execution_config.hpp"


// Transpose + SDPA + Transpose pattern fusion (TransposeSDPAMatcher)
namespace ov {
namespace test {
using TransposeSDPATestParamsTuple = std::tuple<ElementType,
                                                    ov::Dimension::value_type,
                                                    ov::Dimension::value_type,
                                                    ov::Dimension::value_type,
                                                    bool,
                                                    bool,
                                                    bool,
                                                    bool>;
struct TransposeSDPATestParams {
    TransposeSDPATestParams(const TransposeSDPATestParamsTuple &t) {
        std::tie(inType, batch_size, seq_len, emb_size, transpose_q, transpose_k, transpose_v, transpose_out) = t;
    }
ElementType inType;
ov::Dimension::value_type batch_size, seq_len, emb_size;
bool transpose_q, transpose_k, transpose_v, transpose_out;
};

class TransposeSDPATest: public testing::WithParamInterface<TransposeSDPATestParamsTuple>,
                                virtual public test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<TransposeSDPATestParamsTuple> obj) {
        TransposeSDPATestParams params{obj.param};
        std::ostringstream result;
        result << params.inType << "_";
        result << std::string(utils::DEVICE_GPU) << "_";
        result << params.batch_size << "_";
        result << params.seq_len << "_";
        result << params.emb_size << "_";
        result << params.transpose_q << "_";
        result << params.transpose_k << "_";
        result << params.transpose_v << "_";
        result << params.transpose_out << "_";
        return result.str();
    }

protected:
    void SetUp() override {
        TransposeSDPATestParams params{GetParam()};
        targetDevice = test::utils::DEVICE_GPU;

        rel_threshold = 0.02f;
        abs_threshold = 0.02f;
        if (inType == ov::element::f32)
            configuration[ov::hint::inference_precision.name()] = ov::element::f32.get_type_name();

        const auto B = static_cast<size_t>(params.batch_size);
        const auto S = static_cast<size_t>(params.seq_len);
        const auto E = static_cast<size_t>(params.emb_size);
        PartialShape s;
        const std::vector<InputShape> inputShapes{
            // q
            {PartialShape{params.batch_size, params.seq_len, params.emb_size}, {Shape{B, S, E}}},
            // k
            {PartialShape{params.batch_size, params.seq_len, params.emb_size}, {Shape{B, S, E}}},
            // v
            {PartialShape{params.batch_size, params.seq_len, params.emb_size}, {Shape{B, S, E}}}
        };
        init_input_shapes(inputShapes);

        function = get_function(params);
        functionRefs = function->clone();
    }

    std::shared_ptr<ov::Model> get_function(const TransposeSDPATestParams &params) {
        auto qkv_shape = PartialShape{params.batch_size, params.seq_len, params.emb_size};
        const std::vector<int64_t> transpose_order = {1, 0, 2};

        auto q = std::make_shared<ov::op::v0::Parameter>(params.inType, qkv_shape);
        auto k = std::make_shared<ov::op::v0::Parameter>(params.inType, qkv_shape);
        auto v = std::make_shared<ov::op::v0::Parameter>(params.inType, qkv_shape);

        auto transpose_q = std::make_shared<ov::op::v1::Transpose>(q, ov::op::v0::Constant::create(element::i64, Shape{3}, transpose_order));
        auto transpose_k = std::make_shared<ov::op::v1::Transpose>(q, ov::op::v0::Constant::create(element::i64, Shape{3}, transpose_order));
        auto transpose_v = std::make_shared<ov::op::v1::Transpose>(q, ov::op::v0::Constant::create(element::i64, Shape{3}, transpose_order));
        transpose_q->set_friendly_name("transpose_q");
        transpose_k->set_friendly_name("transpose_k");
        transpose_v->set_friendly_name("transpose_v");

        const auto casual = true;
        auto sdpa_q = params.transpose_q ? transpose_q->output(0) : q->output(0);
        auto sdpa_k = params.transpose_k ? transpose_k->output(0) : k->output(0);
        auto sdpa_v = params.transpose_v ? transpose_v->output(0) : v->output(0);
        auto sdpa = std::make_shared<opset13::ScaledDotProductAttention>(sdpa_q,
                                                                         sdpa_k,
                                                                         sdpa_v,
                                                                         casual);
        sdpa->set_friendly_name("sdpa");
        auto transpose_o = std::make_shared<ov::op::v1::Transpose>(sdpa, ov::op::v0::Constant::create(element::i64, Shape{3}, transpose_order));
        transpose_o->set_friendly_name("transpose_o");

        std::shared_ptr<Model> model;
        if (params.transpose_out) {
            model = std::make_shared<Model>(transpose_o, ParameterVector{q, k, v});
        } else {
            model = std::make_shared<Model>(sdpa, ParameterVector{q, k, v});
        }
        return model;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        inputs_ref.clear();
        const auto& funcInputs = compiledModel.inputs();
        for (size_t i = 0lu; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            if (i < 3) { // q, k, v
                auto tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                inputs.insert({std::const_pointer_cast<Node>(funcInput.get_node_shared_ptr()), tensor});
                inputs_ref.emplace_back(tensor);
            }
        }
    }

    std::vector<ov::Tensor> calculate_refs() override {
        if (is_report_stages) {
            std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is started"<< std::endl;
        }
        auto start_time = std::chrono::system_clock::now();

        auto outputs = ov::test::utils::infer_on_template(functionRefs, inputs_ref);

        if (is_report_stages) {
            auto end_time = std::chrono::system_clock::now();
            std::chrono::duration<double> duration = end_time - start_time;
            std::cout << "[ REFERENCE   ] `SubgraphBaseTest::calculate_refs()` is finished successfully. Duration is " << duration.count() << "s" << std::endl;
        }
        return outputs;
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        ASSERT_EQ(expected.size(), actual.size());
        ASSERT_EQ(expected.size(), function->get_results().size());
        const auto& results = function->get_results();
        for (size_t j = 0; j < results.size(); j++) {
            const auto result = results[j];
            for (size_t i = 0; i < result->get_input_size(); ++i) {
                utils::compare(expected[j], actual[j], abs_threshold, rel_threshold);
            }
        }
    }

private:
    ov::TensorVector inputs_ref;
};


TEST_P(TransposeSDPATest, CompareWithRefs) {
    run();
}

namespace {

INSTANTIATE_TEST_SUITE_P(TransposeSDPAFusion,
                         TransposeSDPATest,
                         ::testing::Combine(::testing::Values(ov::element::f16),
                                            ::testing::Values(1),       // batch_size
                                            ::testing::Values(32),      // seq_len
                                            ::testing::Values(64),      // emb_size
                                            ::testing::Values(true, false),       // transpose q
                                            ::testing::Values(true, false),       // transpose k
                                            ::testing::Values(true, false),       // transpose v
                                            ::testing::Values(true, false)),      // transpose out
                         TransposeSDPATest::getTestCaseName);

}  // namespace

}  // namespace test
}  // namespace ov

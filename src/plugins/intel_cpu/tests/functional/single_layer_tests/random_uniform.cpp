// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ov_models/builders.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {

typedef std::tuple<
        ov::Shape,                       // Output shapes
        std::tuple<double, double>,      // Min and Max values
        ElementType,                     // Shape precision
        ElementType,                     // Output precision
        uint64_t,                        // Global seed
        uint64_t,                        // Operational seed
        bool,                            // Is 1st input constant
        bool,                            // Is 2nd input constant
        bool,                            // Is 3rd input constant
        CPUSpecificParams,               // CPU specific params
        ov::AnyMap                       // Additional plugin configuration
> RandomUniformLayerTestCPUParamSet;

class RandomUniformLayerTestCPU : public testing::WithParamInterface<RandomUniformLayerTestCPUParamSet>,
                                  public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RandomUniformLayerTestCPUParamSet>& obj) {
        const auto& out_shape = std::get<0>(obj.param);
        const auto& min_max   = std::get<1>(obj.param);

        std::ostringstream result;

        result << "IS={"              << out_shape.size();
        result << "}_OS="             << out_shape;
        result << "_Min="             << std::get<0>(min_max);
        result << "_Max="             << std::get<1>(min_max);
        result << "_ShapePrc="        << std::get<2>(obj.param);
        result << "_OutPrc="          << std::get<3>(obj.param);
        result << "_GlobalSeed="      << std::get<4>(obj.param);
        result << "_OperationalSeed=" << std::get<5>(obj.param);
        result << "_ConstIn={"        << utils::bool2str(std::get<6>(obj.param)) << ","
                                      << utils::bool2str(std::get<7>(obj.param)) << ","
                                      << utils::bool2str(std::get<8>(obj.param)) << "}";

        result << CPUTestsBase::getTestCaseName(std::get<9>(obj.param));

        const auto& config = std::get<10>(obj.param);
        if (!config.empty()) {
            result << "_PluginConf={";
            for (const auto& conf_item : config) {
                result << "_" << conf_item.first << "=";
                conf_item.second.print(result);
            }
            result << "}";
        }

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;

        const auto& params     = this->GetParam();
        m_output_shape         = std::get<0>(params);
        const auto& min_max    = std::get<1>(params);
        const auto& shape_prc  = std::get<2>(params);
        const auto& output_prc = std::get<3>(params);
        m_global_seed          = std::get<4>(params);
        m_operational_seed     = std::get<5>(params);
        const auto& const_in_1 = std::get<6>(params);
        const auto& const_in_2 = std::get<7>(params);
        const auto& const_in_3 = std::get<8>(params);
        const auto& cpu_params = std::get<9>(params);
        configuration          = std::get<10>(params);

        m_min_val = std::get<0>(min_max);
        m_max_val = std::get<1>(min_max);
        std::tie(inFmts, outFmts, priority, selectedType) = cpu_params;

        if (output_prc == ElementType::i64) {
            updateSelectedType(getPrimitiveType(), ElementType::i32, configuration);
        } else if (output_prc == ElementType::f64) {
            updateSelectedType(getPrimitiveType(), ElementType::f32, configuration);
        } else if (output_prc == ElementType::f16) {
            if (InferenceEngine::with_cpu_x86_avx512_core_fp16()) {
                updateSelectedType(getPrimitiveType(), ElementType::f16, configuration);
            } else {
                updateSelectedType(getPrimitiveType(), ElementType::f32, configuration);
            }
        } else if (output_prc == ElementType::bf16) {
            if (InferenceEngine::with_cpu_x86_bfloat16()) {
                updateSelectedType(getPrimitiveType(), ElementType::bf16, configuration);
            } else {
                updateSelectedType("ref_any", ElementType::bf16, configuration);
            }
        } else {
            updateSelectedType(getPrimitiveType(), output_prc, configuration);
        }

        std::vector<InputShape> in_shapes;
        ov::ParameterVector in_params;
        std::vector<std::shared_ptr<ov::Node>> inputs;

        if (!const_in_1) {
            in_shapes.push_back({{}, {{m_output_shape.size()}}});
            in_params.push_back(std::make_shared<ov::op::v0::Parameter>(shape_prc, ov::PartialShape{static_cast<int64_t>(m_output_shape.size())}));
            in_params.back()->set_friendly_name("shape");
            inputs.push_back(in_params.back());
        } else {
            inputs.push_back(ngraph::builder::makeConstant(shape_prc, {m_output_shape.size()}, m_output_shape));
        }
        if (!const_in_2) {
            in_shapes.push_back({{}, {{1}}});
            in_params.push_back(std::make_shared<ov::op::v0::Parameter>(output_prc, ov::PartialShape{1}));
            in_params.back()->set_friendly_name("minval");
            inputs.push_back(in_params.back());
        } else {
            inputs.push_back(ngraph::builder::makeConstant(output_prc, {1}, std::vector<double>{m_min_val}));
        }
        if (!const_in_3) {
            in_shapes.push_back({{}, {{1}}});
            in_params.push_back(std::make_shared<ov::op::v0::Parameter>(output_prc, ov::PartialShape{1}));
            in_params.back()->set_friendly_name("maxval");
            inputs.push_back(in_params.back());
        } else {
            inputs.push_back(ngraph::builder::makeConstant(output_prc, {1}, std::vector<double>{m_max_val}));
        }

        init_input_shapes(in_shapes);

        const auto rnd_op = std::make_shared<ov::op::v8::RandomUniform>(inputs[0], inputs[1], inputs[2], output_prc, m_global_seed, m_operational_seed);
        const ov::ResultVector results{std::make_shared<ov::op::v0::Result>(rnd_op)};

        function = std::make_shared<ov::Model>(results, in_params, "RandomUniformLayerTestCPU");
    }

    template<typename TD, typename TS>
    void fill_data(TD* dst, const TS* src, size_t len) {
        for (size_t i = 0llu; i < len; i++) {
            dst[i] = static_cast<TD>(src[i]);
        }
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& func_inputs = function->inputs();

        for (size_t i = 0llu; i < func_inputs.size(); ++i) {
            const auto& func_input = func_inputs[i];
            const auto& name = func_input.get_node()->get_friendly_name();
            const auto& in_prc = func_input.get_element_type();
            auto tensor = ov::Tensor(in_prc, targetInputStaticShapes[i]);

#define CASE(P, S, L)                                                              \
case P :                                                                           \
    fill_data(tensor.data<ov::element_type_traits<P>::value_type>(), S, L); break;

            if (name == "shape") {
                switch (in_prc) {
                    CASE(ElementType::i32, m_output_shape.data(), m_output_shape.size())
                    CASE(ElementType::i64, m_output_shape.data(), m_output_shape.size())
                    default:
                        OPENVINO_THROW("RandomUniform does not support precision ", in_prc, " for the Shape input.");
                }
            } else if (name == "minval") {
                switch (in_prc) {
                    CASE(ElementType::f32,  &m_min_val, 1)
                    CASE(ElementType::f16,  &m_min_val, 1)
                    CASE(ElementType::bf16, &m_min_val, 1)
                    CASE(ElementType::i32,  &m_min_val, 1)
                    CASE(ElementType::i64,  &m_min_val, 1)
                    CASE(ElementType::f64,  &m_min_val, 1)
                    default:
                        OPENVINO_THROW("RandomUniform does not support precision ", in_prc, " for the Minval input.");
                }
            } else if (name == "maxval") {
                switch (in_prc) {
                    CASE(ElementType::f32,  &m_max_val, 1)
                    CASE(ElementType::f16,  &m_max_val, 1)
                    CASE(ElementType::bf16, &m_max_val, 1)
                    CASE(ElementType::i32,  &m_max_val, 1)
                    CASE(ElementType::i64,  &m_max_val, 1)
                    CASE(ElementType::f64,  &m_max_val, 1)
                    default:
                        OPENVINO_THROW("RandomUniform does not support precision ", in_prc, " for the Maxval input.");
                }
            }

#undef CASE

            inputs.insert({func_input.get_node_shared_ptr(), tensor});
        }
    }

    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        if (m_global_seed != 0llu || m_operational_seed != 0llu) {
            SubgraphBaseTest::compare(expected, actual);
            return;
        }

        // When both seed values are equal to zero, RandomUniform should generate non-deterministic sequence.
        // In this case will use Mean and Variance metrics.

#define CASE(X) case X : rndUCompare<ov::element_type_traits<X>::value_type>(expected[0], actual[0]); break;

        switch (expected[0].get_element_type()) {
            CASE(ElementType::f32)
            CASE(ElementType::i32)
            CASE(ElementType::f16)
            CASE(ElementType::bf16)
            CASE(ElementType::i64)
            CASE(ElementType::f64)
            default: OPENVINO_THROW("Unsupported element type: ", expected[0].get_element_type());
        }

#undef CASE
    }

    precisions_map get_ref_precisions_convert_map() override {
        precisions_map precisions;

        if (!InferenceEngine::with_cpu_x86_avx512_core()) {
            precisions.insert({ ov::element::bf16, ov::element::f32 });
        }
        if (!InferenceEngine::with_cpu_x86_avx512_core_fp16()) {
            precisions.insert({ ov::element::f16, ov::element::f32 });
        }

        return precisions;
    }

    inline double less_or_equal(double a, double b) {
        return (b - a) >= (std::fmax(std::fabs(a), std::fabs(b)) * std::numeric_limits<double>::epsilon());
    }

    template<typename T>
    void rndUCompare(const ov::Tensor& expected, const ov::Tensor& actual) {
        auto actual_data = actual.data<T>();
        size_t shape_size_cnt = ov::shape_size(expected.get_shape());
        double act_mean = 0.0;
        double act_variance = 0.0;
        const double exp_mean = (m_max_val + m_min_val) / 2.0;
        const double exp_variance = std::pow(m_max_val - m_min_val, 2) / 12.0;

        for (size_t i = 0; i < shape_size_cnt; ++i) {
            auto actual_value = static_cast<double>(actual_data[i]);
            if (std::isnan(actual_value)) {
                std::ostringstream out_stream;
                out_stream << "Actual value is NAN on coordinate: " << i;
                throw std::runtime_error(out_stream.str());
            }
            act_mean += actual_value;
            act_variance += std::pow(actual_value - exp_mean, 2);
        }
        act_mean /= shape_size_cnt;
        act_variance /= shape_size_cnt;

        auto rel_mean = (exp_mean - act_mean) / (m_max_val - m_min_val);
        auto rel_variance = (exp_variance - act_variance) / std::pow(m_max_val - m_min_val, 2);

        if (!(less_or_equal(rel_mean, m_mean_threshold) && less_or_equal(rel_variance, m_variance_threshold))) {
            std::ostringstream out_stream;
            out_stream << "rel_mean < m_mean_threshold && rel_variance < m_variance_threshold" <<
                    "\n\t rel_mean: " << rel_mean <<
                    "\n\t rel_variance: " << rel_variance;
            throw std::runtime_error(out_stream.str());
        }
    }

private:
    ov::Shape m_output_shape;
    uint64_t m_global_seed;
    uint64_t m_operational_seed;
    double m_min_val;
    double m_max_val;
    static constexpr double m_mean_threshold = 0.05;
    static constexpr double m_variance_threshold = 0.1;
};

TEST_P(RandomUniformLayerTestCPU, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "RandomUniform");
}

namespace {

const std::vector<ElementType> shape_prc = {
        ElementType::i32,
        ElementType::i64
};

const std::vector<ElementType> output_prc_nightly = {
        ElementType::f32,
        ElementType::f16,
        ElementType::bf16,
        ElementType::i32,
        ElementType::i64
};

std::vector<ov::Shape> output_shapes = {
        {500},
        {4, 3, 210}
};

// Need to validate the Kernel corner cases.
std::vector<ov::Shape> output_shapes_nightly = {
        {1}, {2}, {3}, {2, 2}, {5}, {2, 3}, {7}, {2, 2, 2}, {3, 3}, {2, 5}, {11}, {2, 3, 2}, {13}, {2, 7}, {3, 5},
        {4, 4}, {1, 17}, {2, 9}, {19}, {4, 5}, {21}, {11, 2}, {23, 1}, {4, 2, 3}, {5, 5}, {26}, {1, 27}, {14, 2},
        {29}, {10, 3}, {31}, {2, 8, 2}, {33}, {17, 2}, {5, 7}, {2, 3, 2, 3}, {37}, {2, 19}, {2, 20}, {41}, {42},
        {43}, {22, 2}, {3, 5, 3}, {5, 2, 5}, {1, 3, 1, 17, 1}, {26, 2}, {53}, {54}, {55}, {56}, {57}, {58}, {59},
        {2, 32}, {99}, {127}, {128}, {129}, {199}, {255}, {499}, {997}, {1753}, {2899}
};

const std::vector<std::tuple<double, double>> min_max = {
        {0, 50},
        {-50, 50},
        {-50, 0}
};

const std::vector<uint64_t> global_seed = {
        0, 8
};

const std::vector<uint64_t> operational_seed = {
        0, 3, 5
};

const ov::AnyMap empty_plugin_config{};

INSTANTIATE_TEST_SUITE_P(smoke_Param, RandomUniformLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(output_shapes),
                ::testing::ValuesIn(min_max),
                ::testing::ValuesIn(shape_prc),
                ::testing::Values(ElementType::f32, ElementType::i32),
                ::testing::ValuesIn(global_seed),
                ::testing::ValuesIn(operational_seed),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::Values(false),
                ::testing::Values(emptyCPUSpec),
                ::testing::Values(empty_plugin_config)),
        RandomUniformLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ParamConst, RandomUniformLayerTestCPU,
        ::testing::Combine(
                ::testing::Values(output_shapes[0]),
                ::testing::Values(min_max[0]),
                ::testing::Values(ElementType::i32),
                ::testing::Values(ElementType::f32),
                ::testing::Values(1),
                ::testing::Values(0),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(emptyCPUSpec),
                ::testing::Values(empty_plugin_config)),
        RandomUniformLayerTestCPU::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Param, RandomUniformLayerTestCPU,
        ::testing::Combine(
                ::testing::ValuesIn(output_shapes_nightly),
                ::testing::Values(min_max[0]),
                ::testing::Values(ElementType::i32),
                ::testing::ValuesIn(output_prc_nightly),
                ::testing::Values(3),
                ::testing::Values(1),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(true, false),
                ::testing::Values(emptyCPUSpec),
                ::testing::Values(empty_plugin_config)),
        RandomUniformLayerTestCPU::getTestCaseName);

} // namespace
} // namespace CPULayerTestsDefinitions

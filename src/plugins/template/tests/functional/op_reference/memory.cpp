// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/util/variable.hpp"

using namespace ov;
using namespace reference_tests;

namespace {

struct ReadValueAssignParams {
    template <class IT>
    ReadValueAssignParams(const Shape& input_shape,
                          const Shape& output_shape,
                          const element::Type& input_type,
                          const element::Type& ouput_type,
                          const std::vector<IT>& input_values,
                          const std::vector<IT>& output_values,
                          const std::string& variable_id)
        : m_input_shape(input_shape),
          m_output_shape(output_shape),
          m_input_type(input_type),
          m_output_type(ouput_type),
          m_input_data(CreateTensor(input_shape, input_type, input_values)),
          m_expected_data(CreateTensor(output_shape, ouput_type, output_values)),
          m_variable_id(variable_id) {}
    Shape m_input_shape;
    Shape m_output_shape;
    element::Type m_input_type;
    element::Type m_output_type;
    ov::Tensor m_input_data;
    ov::Tensor m_expected_data;
    std::string m_variable_id;
};

class ReferenceReadValueAssignV3LayerTest : public testing::TestWithParam<ReadValueAssignParams>,
                                            public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.m_input_shape, params.m_input_type, params.m_variable_id);
        inputData = {params.m_input_data};
        refOutData = {params.m_expected_data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ReadValueAssignParams>& obj) {
        auto params = obj.param;
        std::ostringstream result;
        result << "shape=" << params.m_input_shape << "_";
        result << "iType=" << params.m_input_type << "_";
        result << "shape=" << params.m_output_shape << "_";
        result << "oType=" << params.m_output_type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const std::string variable_id) {
        auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        auto read_value = std::make_shared<op::v3::ReadValue>(in, variable_id);
        auto assign = std::make_shared<op::v3::Assign>(read_value, variable_id);
        return std::make_shared<Model>(OutputVector{assign}, ParameterVector{in});
    }
};

class ReferenceReadValueAssignV6LayerTest : public testing::TestWithParam<ReadValueAssignParams>,
                                            public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.m_input_shape, params.m_input_type, params.m_variable_id);
        inputData = {params.m_input_data};
        refOutData = {params.m_expected_data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ReadValueAssignParams>& obj) {
        auto params = obj.param;
        std::ostringstream result;
        result << "shape=" << params.m_input_shape << "_";
        result << "iType=" << params.m_input_type << "_";
        result << "shape=" << params.m_output_shape << "_";
        result << "oType=" << params.m_output_type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type& input_type,
                                                 const std::string variable_id) {
        auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);
        auto variable =
            std::make_shared<op::util::Variable>(op::util::VariableInfo{input_shape, input_type, variable_id});
        auto assign = std::make_shared<op::v6::Assign>(in, variable);
        auto read_value = std::make_shared<op::v6::ReadValue>(assign, variable);
        return std::make_shared<Model>(OutputVector{read_value},
                                       ParameterVector{in},
                                       op::util::VariableVector{variable});
    }
};

TEST_P(ReferenceReadValueAssignV3LayerTest, ReadValueAssignWithHardcodedRefs) {
    Exec();
    if (executableNetwork) {
        const int COUNT_RUNS = 10;
        for (int i = 0; i < COUNT_RUNS; ++i) {
            Infer();
            Validate();
        }
    }
}

TEST_P(ReferenceReadValueAssignV6LayerTest, ReadValueAssignWithHardcodedRefs) {
    Exec();
    if (executableNetwork) {
        const int COUNT_RUNS = 10;
        for (int i = 0; i < COUNT_RUNS; ++i) {
            Infer();
            Validate();
        }
    }
}

template <element::Type_t IN_ET>
std::vector<ReadValueAssignParams> generateParamsForReadValueAssign() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ReadValueAssignParams> params{
        ReadValueAssignParams(ov::Shape{1}, ov::Shape{1}, IN_ET, IN_ET, std::vector<T>{1}, std::vector<T>{1}, "v0"),
        ReadValueAssignParams(ov::Shape{2, 2},
                              ov::Shape{2, 2},
                              IN_ET,
                              IN_ET,
                              std::vector<T>{1, 2, 3, 4},
                              std::vector<T>{1, 2, 3, 4},
                              "v0"),
        ReadValueAssignParams(ov::Shape{1, 2, 3},
                              ov::Shape{1, 2, 3},
                              IN_ET,
                              IN_ET,
                              std::vector<T>{1, 2, 3, 4, 5, 6},
                              std::vector<T>{1, 2, 3, 4, 5, 6},
                              "v0")};
    return params;
}

template <element::Type_t IN_ET>
std::vector<ReadValueAssignParams> generateParamsForReadValueAssignBoolean() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ReadValueAssignParams> params{
        ReadValueAssignParams(ov::Shape{1},
                              ov::Shape{1},
                              IN_ET,
                              IN_ET,
                              std::vector<T>{true},
                              std::vector<T>{true},
                              "v0"),
        ReadValueAssignParams(ov::Shape{2, 2},
                              ov::Shape{2, 2},
                              IN_ET,
                              IN_ET,
                              std::vector<T>{true, true, false, false},
                              std::vector<T>{true, true, false, false},
                              "v0"),
        ReadValueAssignParams(ov::Shape{1, 2, 3},
                              ov::Shape{1, 2, 3},
                              IN_ET,
                              IN_ET,
                              std::vector<T>{true, false, true, false, true, false},
                              std::vector<T>{true, false, true, false, true, false},
                              "v0")};
    return params;
}

std::vector<ReadValueAssignParams> generateCombinedParamsForReadValueAssign() {
    const std::vector<std::vector<ReadValueAssignParams>> allTypeParams{
        generateParamsForReadValueAssign<element::Type_t::f64>(),
        generateParamsForReadValueAssign<element::Type_t::f32>(),
        generateParamsForReadValueAssign<element::Type_t::f16>(),
        generateParamsForReadValueAssign<element::Type_t::bf16>(),
        generateParamsForReadValueAssign<element::Type_t::i64>(),
        generateParamsForReadValueAssign<element::Type_t::i32>(),
        generateParamsForReadValueAssign<element::Type_t::i16>(),
        generateParamsForReadValueAssign<element::Type_t::i8>(),
        generateParamsForReadValueAssign<element::Type_t::i4>(),
        generateParamsForReadValueAssign<element::Type_t::u64>(),
        generateParamsForReadValueAssign<element::Type_t::u32>(),
        generateParamsForReadValueAssign<element::Type_t::u16>(),
        generateParamsForReadValueAssign<element::Type_t::u8>(),
        generateParamsForReadValueAssign<element::Type_t::u4>(),
        generateParamsForReadValueAssignBoolean<element::Type_t::boolean>()};

    std::vector<ReadValueAssignParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ReadValue_Assign_With_Hardcoded_Refs,
                         ReferenceReadValueAssignV3LayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForReadValueAssign()),
                         ReferenceReadValueAssignV3LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ReadValue_Assign_With_Hardcoded_Refs,
                         ReferenceReadValueAssignV6LayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForReadValueAssign()),
                         ReferenceReadValueAssignV6LayerTest::getTestCaseName);

}  // namespace

namespace {
struct MemoryTestParams {
    template <class IT>
    MemoryTestParams(const ov::Shape& input_shape,
                     const ov::Shape& output_shape,
                     const ov::element::Type& input_type,
                     const ov::element::Type& ouput_type,
                     const std::vector<IT>& input_values,
                     const std::vector<std::vector<IT>>& output_values,
                     const size_t& count_runs,
                     const std::vector<std::string>& variable_id,
                     const size_t& reset_on_run = 0)
        : m_input_shape(input_shape),
          m_output_shape(output_shape),
          m_input_type(input_type),
          m_output_type(ouput_type),
          m_input_data(reference_tests::CreateTensor(input_shape, input_type, input_values)),
          m_expected_data(reference_tests::CreateTensor(output_shape, ouput_type, output_values[0])),
          m_variable_id(variable_id),
          m_count_runs(count_runs),
          m_reset_on_run(reset_on_run) {
        for (size_t i = 0; i < m_count_runs; i++) {
            m_expected_data_vector.push_back(reference_tests::CreateTensor(output_shape, ouput_type, output_values[i]));
        }
    }
    ov::Shape m_input_shape;
    ov::Shape m_output_shape;
    ov::element::Type m_input_type;
    ov::element::Type m_output_type;
    ov::Tensor m_input_data;
    ov::Tensor m_expected_data;
    std::vector<std::string> m_variable_id;
    size_t m_count_runs;
    size_t m_reset_on_run;
    std::vector<ov::Tensor> m_expected_data_vector;
};

class ReferenceMemoryTest : public testing::TestWithParam<MemoryTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MemoryTestParams>& obj) {
        auto params = obj.param;
        std::ostringstream result;
        result << "shape=" << params.m_input_shape << "_";
        result << "iType=" << params.m_input_type << "_";
        result << "shape=" << params.m_output_shape << "_";
        result << "oType=" << params.m_output_type;
        return result.str();
    }

protected:
    const std::string targetDevice;
    std::shared_ptr<ov::Core> core;
    std::shared_ptr<ov::Model> function;
    ov::CompiledModel executableNetwork;
    ov::InferRequest inferRequest;

    ReferenceMemoryTest() : targetDevice("TEMPLATE"), function(), executableNetwork(), inferRequest() {
        core = ov::test::utils::PluginCache::get().core(targetDevice);
    };

    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.m_input_shape, params.m_input_type, params.m_variable_id);
        executableNetwork = core->compile_model(function, targetDevice);
        inferRequest = executableNetwork.create_infer_request();
    }

    void CommonTestSteps(const std::function<void(size_t, ov::InferRequest&)>& custom_step = nullptr) {
        auto params = GetParam();

        inferRequest.set_tensor(executableNetwork.input(0), params.m_input_data);
        for (size_t i = 0; i < params.m_count_runs; ++i) {
            if (custom_step) {
                custom_step(i, inferRequest);
            }
            inferRequest.infer();
            auto actualOutData = inferRequest.get_tensor(executableNetwork.output(0));
            reference_tests::CommonReferenceTest::ValidateBlobs(params.m_expected_data_vector[i],
                                                                actualOutData,
                                                                i,
                                                                1e-2f,
                                                                -1.f,
                                                                true);
        }
    }

    virtual std::shared_ptr<ov::Model> CreateFunction(const ov::Shape& input_shape,
                                                      const ov::element::Type& input_type,
                                                      const std::vector<std::string>& variable_id) = 0;
};

std::shared_ptr<ov::Model> CreateFunction_ReadValueAssingAdd(const ov::Shape& input_shape,
                                                             const ov::element::Type& input_type,
                                                             const std::vector<std::string>& variable_id) {
    auto in = std::make_shared<ov::op::v0::Parameter>(input_type, input_shape);
    auto c = std::make_shared<ov::op::v0::Constant>(input_type, input_shape, 0);
    auto variable =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{input_shape, input_type, variable_id[0]});
    auto read_value = std::make_shared<ov::op::v6::ReadValue>(c, variable);
    auto add = std::make_shared<ov::op::v1::Add>(in, read_value);
    auto assign = std::make_shared<ov::op::v6::Assign>(add, variable);
    return std::make_shared<ov::Model>(ov::OutputVector{assign},
                                       ov::ParameterVector{in},
                                       ov::op::util::VariableVector{variable});
}

std::shared_ptr<ov::Model> CreateFunction_ReadValueAssingAddMultiVariable(const ov::Shape& input_shape,
                                                                          const ov::element::Type& input_type,
                                                                          const std::vector<std::string>& variable_id) {
    auto in = std::make_shared<ov::op::v0::Parameter>(input_type, input_shape);
    auto variable1 =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{input_shape, input_type, variable_id[0]});
    auto variable2 =
        std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{input_shape, input_type, variable_id[1]});
    auto read_value1 = std::make_shared<ov::op::v6::ReadValue>(in, variable1);
    auto read_value2 = std::make_shared<ov::op::v6::ReadValue>(in, variable2);
    auto add1 = std::make_shared<ov::op::v1::Add>(read_value1, read_value2);
    auto add2 = std::make_shared<ov::op::v1::Add>(in, add1);
    auto assign1 = std::make_shared<ov::op::v6::Assign>(add2, variable1);
    auto assign2 = std::make_shared<ov::op::v6::Assign>(read_value2, variable2);
    return std::make_shared<ov::Model>(ov::OutputVector{assign1},
                                       ov::SinkVector{assign2},
                                       ov::ParameterVector{in},
                                       ov::op::util::VariableVector{variable1, variable2});
}

class ReferenceReadValueAssignAddLayerTest : public ReferenceMemoryTest {
protected:
    std::shared_ptr<ov::Model> CreateFunction(const ov::Shape& input_shape,
                                              const ov::element::Type& input_type,
                                              const std::vector<std::string>& variable_id) override {
        return CreateFunction_ReadValueAssingAdd(input_shape, input_type, variable_id);
    }
};

TEST_P(ReferenceReadValueAssignAddLayerTest, MemoryWithHardcodedRefs) {
    CommonTestSteps();
}

template <ov::element::Type_t IN_ET>
std::vector<MemoryTestParams> generateParamsForReadValueAssignAdd() {
    using T = typename ov::element_type_traits<IN_ET>::value_type;
    size_t count_runs = 10;

    std::vector<T> first_result_shape1 = {1};
    std::vector<T> first_result_shape22 = {1, 2, 3, 4};
    std::vector<T> first_result_shape123 = {1, 2, 3, 4, 5, 6};

    std::vector<T> new_result_shape1(1, T(0));
    std::vector<T> new_result_shape22(4, T(0));
    std::vector<T> new_result_shape123(6, T(0));

    std::vector<std::vector<T>> result_shape1;
    std::vector<std::vector<T>> result_shape22;
    std::vector<std::vector<T>> result_shape123;

    for (size_t i = 0; i < count_runs; i++) {
        std::transform(new_result_shape1.begin(),
                       new_result_shape1.end(),
                       first_result_shape1.begin(),
                       new_result_shape1.begin(),
                       std::plus<T>());
        std::transform(new_result_shape22.begin(),
                       new_result_shape22.end(),
                       first_result_shape22.begin(),
                       new_result_shape22.begin(),
                       std::plus<T>());
        std::transform(new_result_shape123.begin(),
                       new_result_shape123.end(),
                       first_result_shape123.begin(),
                       new_result_shape123.begin(),
                       std::plus<T>());
        result_shape1.push_back(new_result_shape1);
        result_shape22.push_back(new_result_shape22);
        result_shape123.push_back(new_result_shape123);
    }

    std::vector<MemoryTestParams> params{MemoryTestParams(ov::Shape{1},
                                                          ov::Shape{1},
                                                          IN_ET,
                                                          IN_ET,
                                                          std::vector<T>{1},
                                                          result_shape1,
                                                          count_runs,
                                                          {"v0"}),
                                         MemoryTestParams(ov::Shape{2, 2},
                                                          ov::Shape{2, 2},
                                                          IN_ET,
                                                          IN_ET,
                                                          std::vector<T>{1, 2, 3, 4},
                                                          result_shape22,
                                                          count_runs,
                                                          {"v0"}),
                                         MemoryTestParams(ov::Shape{1, 2, 3},
                                                          ov::Shape{1, 2, 3},
                                                          IN_ET,
                                                          IN_ET,
                                                          std::vector<T>{1, 2, 3, 4, 5, 6},
                                                          result_shape123,
                                                          count_runs,
                                                          {"v0"})};
    return params;
}

std::vector<MemoryTestParams> generateCombinedParamsForReadValueAssignAdd() {
    const std::vector<std::vector<MemoryTestParams>> allTypeParams{
        generateParamsForReadValueAssignAdd<ov::element::Type_t::f32>(),
        generateParamsForReadValueAssignAdd<ov::element::Type_t::f16>(),
        generateParamsForReadValueAssignAdd<ov::element::Type_t::bf16>(),
        generateParamsForReadValueAssignAdd<ov::element::Type_t::i64>(),
        generateParamsForReadValueAssignAdd<ov::element::Type_t::i32>(),
        generateParamsForReadValueAssignAdd<ov::element::Type_t::i16>(),
        generateParamsForReadValueAssignAdd<ov::element::Type_t::i8>(),
        generateParamsForReadValueAssignAdd<ov::element::Type_t::u64>(),
        generateParamsForReadValueAssignAdd<ov::element::Type_t::u32>(),
        generateParamsForReadValueAssignAdd<ov::element::Type_t::u16>(),
        generateParamsForReadValueAssignAdd<ov::element::Type_t::u8>()};

    std::vector<MemoryTestParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Memory_With_Hardcoded_Refs,
                         ReferenceReadValueAssignAddLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForReadValueAssignAdd()),
                         ReferenceReadValueAssignAddLayerTest::getTestCaseName);

class ReferenceReadValueAssignAddMultiVariableLayerTest : public ReferenceMemoryTest {
protected:
    std::shared_ptr<ov::Model> CreateFunction(const ov::Shape& input_shape,
                                              const ov::element::Type& input_type,
                                              const std::vector<std::string>& variable_id) override {
        return CreateFunction_ReadValueAssingAddMultiVariable(input_shape, input_type, variable_id);
    }
};

TEST_P(ReferenceReadValueAssignAddMultiVariableLayerTest, MemoryWithHardcodedRefs) {
    CommonTestSteps();
}

template <ov::element::Type_t IN_ET>
std::vector<MemoryTestParams> ReadValueAssignAddMultiVariableLayer() {
    using T = typename ov::element_type_traits<IN_ET>::value_type;
    const size_t num_tests = 3;
    size_t count_runs = 10;

    std::vector<std::vector<T>> parameter_value(num_tests);
    parameter_value[0] = {1};
    parameter_value[1] = {1, 2, 3, 4};
    parameter_value[2] = {1, 2, 3, 4, 5, 6};

    std::vector<Shape> in_out_shapes = {{1}, {1, 2}, {1, 2, 3}};

    // the initial value for the buffers is equal to the params values on the 1st iteration
    auto state_buffer_value = parameter_value;

    // the result contain values after each inference request
    // number of inferences = count_runs
    std::vector<std::vector<std::vector<T>>> expected_res(num_tests);

    // the reference for ov::Model:
    //   ___________
    //  |           | -> [ReadValue 1] ->   ________
    //                                     |  Add 1 |        _______
    //  | Parameter | -> [ReadValue 2] ->  |________| ----> |       |
    //                                                      |  Add 2| -> Assign_1 -> Result
    //  | __________| ---------------------------------->   |_______|
    //  Note: Assign_2 is not shown in the graph here, it exists and connected to ReadValue2 directly,
    //  but we don't check its value.
    std::vector<std::vector<T>> add_1(num_tests);
    for (size_t i = 0; i < num_tests; ++i) {
        add_1[i].resize(parameter_value[i].size(), 0);
    }

    std::vector<MemoryTestParams> params;
    for (size_t test_i = 0; test_i < num_tests; ++test_i) {
        for (size_t i = 0; i < count_runs; i++) {
            // Add1 = ReadValue1 + ReadValue2
            std::transform(state_buffer_value[test_i].begin(),
                           state_buffer_value[test_i].end(),
                           parameter_value[test_i].begin(),
                           add_1[test_i].begin(),
                           std::plus<T>());
            // Res = Add1 + Parameter
            std::transform(add_1[test_i].begin(),
                           add_1[test_i].end(),
                           parameter_value[test_i].begin(),
                           state_buffer_value[test_i].begin(),
                           std::plus<T>());

            expected_res[test_i].push_back(state_buffer_value[test_i]);
        }
        params.push_back(MemoryTestParams(in_out_shapes[test_i],
                                          in_out_shapes[test_i],
                                          IN_ET,
                                          IN_ET,
                                          parameter_value[test_i],
                                          expected_res[test_i],
                                          count_runs,
                                          {"v0", "v1"}));
    }
    return params;
}

std::vector<MemoryTestParams> generateCombinedParamsForReadValueAssignAddMultiVariableLayer() {
    const std::vector<std::vector<MemoryTestParams>> allTypeParams{
        ReadValueAssignAddMultiVariableLayer<ov::element::Type_t::f32>(),
        ReadValueAssignAddMultiVariableLayer<ov::element::Type_t::f16>(),
        ReadValueAssignAddMultiVariableLayer<ov::element::Type_t::bf16>(),
        ReadValueAssignAddMultiVariableLayer<ov::element::Type_t::i64>(),
        ReadValueAssignAddMultiVariableLayer<ov::element::Type_t::i32>(),
        ReadValueAssignAddMultiVariableLayer<ov::element::Type_t::i16>(),
        ReadValueAssignAddMultiVariableLayer<ov::element::Type_t::i8>(),
        ReadValueAssignAddMultiVariableLayer<ov::element::Type_t::u64>(),
        ReadValueAssignAddMultiVariableLayer<ov::element::Type_t::u32>(),
        ReadValueAssignAddMultiVariableLayer<ov::element::Type_t::u16>(),
        ReadValueAssignAddMultiVariableLayer<ov::element::Type_t::u8>()};

    std::vector<MemoryTestParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Memory_With_Hardcoded_Refs,
                         ReferenceReadValueAssignAddMultiVariableLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForReadValueAssignAddMultiVariableLayer()),
                         ReferenceReadValueAssignAddMultiVariableLayerTest::getTestCaseName);

class ReferenceReadValueAssignAddResetLayerTest : public ReferenceMemoryTest {
protected:
    std::shared_ptr<ov::Model> CreateFunction(const ov::Shape& input_shape,
                                              const ov::element::Type& input_type,
                                              const std::vector<std::string>& variable_id) override {
        return CreateFunction_ReadValueAssingAdd(input_shape, input_type, variable_id);
    }
};

TEST_P(ReferenceReadValueAssignAddResetLayerTest, MemoryResetWithHardcodedRefs) {
    auto params = GetParam();

    auto reset_var = [&](size_t iter, ov::InferRequest& inferRequest) {
        if (params.m_reset_on_run == iter) {
            inferRequest.reset_state();
        }
    };
    CommonTestSteps(reset_var);
}

template <ov::element::Type_t IN_ET>
std::vector<MemoryTestParams> generateParamsForReadValueAssignAddReset() {
    using T = typename ov::element_type_traits<IN_ET>::value_type;
    size_t count_runs = 10;
    size_t reset_on_run = 5;

    std::vector<T> first_result_shape1 = {1};
    std::vector<T> first_result_shape22 = {1, 2, 3, 4};
    std::vector<T> first_result_shape123 = {1, 2, 3, 4, 5, 6};

    std::vector<T> new_result_shape1(1, T(0));
    std::vector<T> new_result_shape22(4, T(0));
    std::vector<T> new_result_shape123(6, T(0));

    std::vector<std::vector<T>> result_shape1;
    std::vector<std::vector<T>> result_shape22;
    std::vector<std::vector<T>> result_shape123;

    for (size_t i = 0; i < count_runs - reset_on_run; i++) {
        std::transform(new_result_shape1.begin(),
                       new_result_shape1.end(),
                       first_result_shape1.begin(),
                       new_result_shape1.begin(),
                       std::plus<T>());
        std::transform(new_result_shape22.begin(),
                       new_result_shape22.end(),
                       first_result_shape22.begin(),
                       new_result_shape22.begin(),
                       std::plus<T>());
        std::transform(new_result_shape123.begin(),
                       new_result_shape123.end(),
                       first_result_shape123.begin(),
                       new_result_shape123.begin(),
                       std::plus<T>());
        result_shape1.push_back(new_result_shape1);
        result_shape22.push_back(new_result_shape22);
        result_shape123.push_back(new_result_shape123);
    }

    new_result_shape1 = std::vector<T>(1, T(0));
    new_result_shape22 = std::vector<T>(4, T(0));
    new_result_shape123 = std::vector<T>(6, T(0));

    for (size_t i = count_runs - reset_on_run; i < count_runs; i++) {
        std::transform(new_result_shape1.begin(),
                       new_result_shape1.end(),
                       first_result_shape1.begin(),
                       new_result_shape1.begin(),
                       std::plus<T>());
        std::transform(new_result_shape22.begin(),
                       new_result_shape22.end(),
                       first_result_shape22.begin(),
                       new_result_shape22.begin(),
                       std::plus<T>());
        std::transform(new_result_shape123.begin(),
                       new_result_shape123.end(),
                       first_result_shape123.begin(),
                       new_result_shape123.begin(),
                       std::plus<T>());
        result_shape1.push_back(new_result_shape1);
        result_shape22.push_back(new_result_shape22);
        result_shape123.push_back(new_result_shape123);
    }

    std::vector<MemoryTestParams> params{MemoryTestParams(ov::Shape{1},
                                                          ov::Shape{1},
                                                          IN_ET,
                                                          IN_ET,
                                                          std::vector<T>{1},
                                                          result_shape1,
                                                          count_runs,
                                                          {"v0"},
                                                          reset_on_run),
                                         MemoryTestParams(ov::Shape{2, 2},
                                                          ov::Shape{2, 2},
                                                          IN_ET,
                                                          IN_ET,
                                                          std::vector<T>{1, 2, 3, 4},
                                                          result_shape22,
                                                          count_runs,
                                                          {"v0"},
                                                          reset_on_run),
                                         MemoryTestParams(ov::Shape{1, 2, 3},
                                                          ov::Shape{1, 2, 3},
                                                          IN_ET,
                                                          IN_ET,
                                                          std::vector<T>{1, 2, 3, 4, 5, 6},
                                                          result_shape123,
                                                          count_runs,
                                                          {"v0"},
                                                          reset_on_run)};
    return params;
}

std::vector<MemoryTestParams> generateCombinedParamsForReadValueAssignAddReset() {
    const std::vector<std::vector<MemoryTestParams>> allTypeParams{
        generateParamsForReadValueAssignAddReset<ov::element::Type_t::f32>(),
        generateParamsForReadValueAssignAddReset<ov::element::Type_t::f16>(),
        generateParamsForReadValueAssignAddReset<ov::element::Type_t::bf16>(),
        generateParamsForReadValueAssignAddReset<ov::element::Type_t::i64>(),
        generateParamsForReadValueAssignAddReset<ov::element::Type_t::i32>(),
        generateParamsForReadValueAssignAddReset<ov::element::Type_t::i16>(),
        generateParamsForReadValueAssignAddReset<ov::element::Type_t::i8>(),
        generateParamsForReadValueAssignAddReset<ov::element::Type_t::u64>(),
        generateParamsForReadValueAssignAddReset<ov::element::Type_t::u32>(),
        generateParamsForReadValueAssignAddReset<ov::element::Type_t::u16>(),
        generateParamsForReadValueAssignAddReset<ov::element::Type_t::u8>()};

    std::vector<MemoryTestParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Memory_With_Hardcoded_Refs,
                         ReferenceReadValueAssignAddResetLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForReadValueAssignAddReset()),
                         ReferenceReadValueAssignAddResetLayerTest::getTestCaseName);

class ReferenceReadValueAssignAddModifyLayerTest : public ReferenceMemoryTest {
protected:
    std::shared_ptr<ov::Model> CreateFunction(const ov::Shape& input_shape,
                                              const ov::element::Type& input_type,
                                              const std::vector<std::string>& variable_id) override {
        return CreateFunction_ReadValueAssingAdd(input_shape, input_type, variable_id);
    }
};

TEST_P(ReferenceReadValueAssignAddModifyLayerTest, MemoryResetWithHardcodedRefs) {
    auto params = GetParam();

    auto reset_var = [&](size_t iter, ov::InferRequest& inferRequest) {
        if (params.m_reset_on_run == iter) {
            auto vars = inferRequest.query_state();
            for (auto& var : vars) {
                var.set_state(params.m_input_data);
            }
        }
    };
    CommonTestSteps(reset_var);
}

template <ov::element::Type_t IN_ET>
std::vector<MemoryTestParams> generateParamsForReadValueAssignAddModify() {
    using T = typename ov::element_type_traits<IN_ET>::value_type;
    size_t count_runs = 10;
    size_t reset_on_run = 5;

    std::vector<T> first_result_shape1 = {1};
    std::vector<T> first_result_shape22 = {1, 2, 3, 4};
    std::vector<T> first_result_shape123 = {1, 2, 3, 4, 5, 6};

    std::vector<T> new_result_shape1(1, T(0));
    std::vector<T> new_result_shape22(4, T(0));
    std::vector<T> new_result_shape123(6, T(0));

    std::vector<std::vector<T>> result_shape1;
    std::vector<std::vector<T>> result_shape22;
    std::vector<std::vector<T>> result_shape123;

    for (size_t i = 0; i < count_runs - reset_on_run; i++) {
        std::transform(new_result_shape1.begin(),
                       new_result_shape1.end(),
                       first_result_shape1.begin(),
                       new_result_shape1.begin(),
                       std::plus<T>());
        std::transform(new_result_shape22.begin(),
                       new_result_shape22.end(),
                       first_result_shape22.begin(),
                       new_result_shape22.begin(),
                       std::plus<T>());
        std::transform(new_result_shape123.begin(),
                       new_result_shape123.end(),
                       first_result_shape123.begin(),
                       new_result_shape123.begin(),
                       std::plus<T>());
        result_shape1.push_back(new_result_shape1);
        result_shape22.push_back(new_result_shape22);
        result_shape123.push_back(new_result_shape123);
    }

    new_result_shape1 = std::vector<T>(1, T(0));
    new_result_shape22 = std::vector<T>(4, T(0));
    new_result_shape123 = std::vector<T>(6, T(0));

    std::transform(new_result_shape1.begin(),
                   new_result_shape1.end(),
                   first_result_shape1.begin(),
                   new_result_shape1.begin(),
                   std::plus<T>());
    std::transform(new_result_shape22.begin(),
                   new_result_shape22.end(),
                   first_result_shape22.begin(),
                   new_result_shape22.begin(),
                   std::plus<T>());
    std::transform(new_result_shape123.begin(),
                   new_result_shape123.end(),
                   first_result_shape123.begin(),
                   new_result_shape123.begin(),
                   std::plus<T>());

    for (size_t i = count_runs - reset_on_run; i < count_runs; i++) {
        std::transform(new_result_shape1.begin(),
                       new_result_shape1.end(),
                       first_result_shape1.begin(),
                       new_result_shape1.begin(),
                       std::plus<T>());
        std::transform(new_result_shape22.begin(),
                       new_result_shape22.end(),
                       first_result_shape22.begin(),
                       new_result_shape22.begin(),
                       std::plus<T>());
        std::transform(new_result_shape123.begin(),
                       new_result_shape123.end(),
                       first_result_shape123.begin(),
                       new_result_shape123.begin(),
                       std::plus<T>());
        result_shape1.push_back(new_result_shape1);
        result_shape22.push_back(new_result_shape22);
        result_shape123.push_back(new_result_shape123);
    }

    std::vector<MemoryTestParams> params{MemoryTestParams(ov::Shape{1},
                                                          ov::Shape{1},
                                                          IN_ET,
                                                          IN_ET,
                                                          std::vector<T>{1},
                                                          result_shape1,
                                                          count_runs,
                                                          {"v0"},
                                                          reset_on_run),
                                         MemoryTestParams(ov::Shape{2, 2},
                                                          ov::Shape{2, 2},
                                                          IN_ET,
                                                          IN_ET,
                                                          std::vector<T>{1, 2, 3, 4},
                                                          result_shape22,
                                                          count_runs,
                                                          {"v0"},
                                                          reset_on_run),
                                         MemoryTestParams(ov::Shape{1, 2, 3},
                                                          ov::Shape{1, 2, 3},
                                                          IN_ET,
                                                          IN_ET,
                                                          std::vector<T>{1, 2, 3, 4, 5, 6},
                                                          result_shape123,
                                                          count_runs,
                                                          {"v0"},
                                                          reset_on_run)};
    return params;
}

std::vector<MemoryTestParams> generateCombinedParamsForReadValueAssignAddModify() {
    const std::vector<std::vector<MemoryTestParams>> allTypeParams{
        generateParamsForReadValueAssignAddModify<ov::element::Type_t::f32>(),
        generateParamsForReadValueAssignAddModify<ov::element::Type_t::f16>(),
        generateParamsForReadValueAssignAddModify<ov::element::Type_t::bf16>(),
        generateParamsForReadValueAssignAddModify<ov::element::Type_t::i64>(),
        generateParamsForReadValueAssignAddModify<ov::element::Type_t::i32>(),
        generateParamsForReadValueAssignAddModify<ov::element::Type_t::i16>(),
        generateParamsForReadValueAssignAddModify<ov::element::Type_t::i8>(),
        generateParamsForReadValueAssignAddModify<ov::element::Type_t::u64>(),
        generateParamsForReadValueAssignAddModify<ov::element::Type_t::u32>(),
        generateParamsForReadValueAssignAddModify<ov::element::Type_t::u16>(),
        generateParamsForReadValueAssignAddModify<ov::element::Type_t::u8>()};

    std::vector<MemoryTestParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Memory_With_Hardcoded_Refs,
                         ReferenceReadValueAssignAddModifyLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForReadValueAssignAddModify()),
                         ReferenceReadValueAssignAddModifyLayerTest::getTestCaseName);

class ReferenceReadValueAssignAddMultiVariableModifyLayerTest : public ReferenceMemoryTest {
protected:
    std::shared_ptr<ov::Model> CreateFunction(const ov::Shape& input_shape,
                                              const ov::element::Type& input_type,
                                              const std::vector<std::string>& variable_id) override {
        return CreateFunction_ReadValueAssingAddMultiVariable(input_shape, input_type, variable_id);
    }
};

TEST_P(ReferenceReadValueAssignAddMultiVariableModifyLayerTest, MemoryResetWithHardcodedRefs) {
    auto params = GetParam();

    auto reset_var = [&](size_t iter, ov::InferRequest& inferRequest) {
        if (params.m_reset_on_run == iter) {
            inferRequest.reset_state();
        }
    };
    CommonTestSteps(reset_var);
}

template <ov::element::Type_t IN_ET>
std::vector<MemoryTestParams> generateParamsForReadValueAssignAddMultiVariableModify() {
    using T = typename ov::element_type_traits<IN_ET>::value_type;
    const size_t num_tests = 3;
    size_t count_runs = 10;
    size_t reset_on_run = 5;

    std::vector<std::vector<T>> parameter_value(num_tests);
    parameter_value[0] = {1};
    parameter_value[1] = {1, 2, 3, 4};
    parameter_value[2] = {1, 2, 3, 4, 5, 6};

    std::vector<Shape> in_out_shapes = {{1}, {1, 2}, {1, 2, 3}};

    // the initial value for the buffers is equal to the params values on the 1st iteration
    auto state_buffer_value = parameter_value;

    // the result contain values after each inference request
    // number of inferences = count_runs
    std::vector<std::vector<std::vector<T>>> expected_res(num_tests);

    // the reference for ov::Model:
    //   ___________
    //  |           | -> [ReadValue 1] ->   ________
    //                                     |  Add 1 |        _______
    //  | Parameter | -> [ReadValue 2] ->  |________| ----> |       |
    //                                                      |  Add 2| -> Assign_1 -> Result
    //  | __________| ---------------------------------->   |_______|
    //  Note: Assign_2 is not shown in the graph here, it exists and connected to ReadValue2 directly,
    //  but we don't check its value.
    std::vector<std::vector<T>> add_1(num_tests);
    for (size_t i = 0; i < num_tests; ++i) {
        add_1[i].resize(parameter_value[i].size(), 0);
    }

    std::vector<MemoryTestParams> params;
    for (size_t test_i = 0; test_i < num_tests; ++test_i) {
        for (size_t i = 0; i < count_runs; i++) {
            if (i == reset_on_run) {
                state_buffer_value[test_i] = parameter_value[test_i];
            }
            // Add1 = ReadValue1 + ReadValue2
            std::transform(state_buffer_value[test_i].begin(),
                           state_buffer_value[test_i].end(),
                           parameter_value[test_i].begin(),
                           add_1[test_i].begin(),
                           std::plus<T>());
            // Res = Add1 + Parameter
            std::transform(add_1[test_i].begin(),
                           add_1[test_i].end(),
                           parameter_value[test_i].begin(),
                           state_buffer_value[test_i].begin(),
                           std::plus<T>());

            expected_res[test_i].push_back(state_buffer_value[test_i]);
        }
        params.push_back(MemoryTestParams(in_out_shapes[test_i],
                                          in_out_shapes[test_i],
                                          IN_ET,
                                          IN_ET,
                                          parameter_value[test_i],
                                          expected_res[test_i],
                                          count_runs,
                                          {"v0", "v1"},
                                          reset_on_run));
    }
    return params;
}

std::vector<MemoryTestParams> generateCombinedParamsForReadValueAssignAddMultiVariableModify() {
    const std::vector<std::vector<MemoryTestParams>> allTypeParams{
        generateParamsForReadValueAssignAddMultiVariableModify<ov::element::Type_t::f32>(),
        generateParamsForReadValueAssignAddMultiVariableModify<ov::element::Type_t::f16>(),
        generateParamsForReadValueAssignAddMultiVariableModify<ov::element::Type_t::bf16>(),
        generateParamsForReadValueAssignAddMultiVariableModify<ov::element::Type_t::i64>(),
        generateParamsForReadValueAssignAddMultiVariableModify<ov::element::Type_t::i32>(),
        generateParamsForReadValueAssignAddMultiVariableModify<ov::element::Type_t::i16>(),
        generateParamsForReadValueAssignAddMultiVariableModify<ov::element::Type_t::i8>(),
        generateParamsForReadValueAssignAddMultiVariableModify<ov::element::Type_t::u64>(),
        generateParamsForReadValueAssignAddMultiVariableModify<ov::element::Type_t::u32>(),
        generateParamsForReadValueAssignAddMultiVariableModify<ov::element::Type_t::u16>(),
        generateParamsForReadValueAssignAddMultiVariableModify<ov::element::Type_t::u8>()};

    std::vector<MemoryTestParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Memory_With_Hardcoded_Refs,
                         ReferenceReadValueAssignAddMultiVariableModifyLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForReadValueAssignAddMultiVariableModify()),
                         ReferenceReadValueAssignAddMultiVariableModifyLayerTest::getTestCaseName);
}  // namespace

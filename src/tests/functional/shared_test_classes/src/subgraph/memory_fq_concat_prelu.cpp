// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/memory_fq_concat_prelu.hpp"
#include <type_traits>

namespace SubgraphTestsDefinitions {

template<typename T>
inline typename std::enable_if<std::is_integral<T>::value, void>::type
    printTupleElement(std::ostringstream& out, const T& value) {
    out << "_" << value;
}

template<typename T>
inline typename std::enable_if<std::is_same<T, std::vector<typename T::value_type>>::value, void>::type
    printTupleElement(std::ostringstream& out, const T& vector) {
    for (const auto& value : vector) {
        out << "_" << value;
    }
}

template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I == sizeof...(Tp), void>::type printTuple(std::ostringstream& out, std::tuple<Tp...>& t) {
}

template<std::size_t I = 0, typename... Tp>
inline typename std::enable_if<I < sizeof...(Tp), void>::type printTuple(std::ostringstream& out, std::tuple<Tp...>& t) {
    printTupleElement(out, std::get<I>(t));
    printTuple<I + 1, Tp...>(out, t);
}

std::string MemoryFqConcatPrelu::getTestCaseName(const testing::TestParamInfo<MemoryFqConcatPreluTuple> &obj) {
    std::vector<std::vector<size_t>> input;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::map<std::string, std::string> additional_config;
    std::tuple<
        std::vector<int64_t>,
        std::vector<int64_t>,
        std::vector<int64_t>,
        std::vector<int64_t>,
        std::vector<int64_t>> strided_slice_params;
    std::tuple<
        std::size_t,
        std::vector<size_t>,
        std::vector<float>,
        std::vector<float>,
        std::vector<float>,
        std::vector<float>> fake_quantize_params;
    std::tie(input, netPrecision, targetName, additional_config, strided_slice_params, fake_quantize_params) = obj.param;
    std::ostringstream results;

    results << "IS=" << ov::test::utils::vec2str(input[0]) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName << "_";
    for (auto const &item : additional_config) {
        results << "_additional_config=" << item.first << "_" << item.second;
    }
    results << "_strided_slice_params=";
    printTuple(results, strided_slice_params);
    results << "_fake_quantize_params=";
    printTuple(results, fake_quantize_params);
    return results.str();
}

void MemoryFqConcatPrelu::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    LoadNetwork();
    GenerateInputs();
    Infer();
}

void MemoryFqConcatPrelu::SetUp() {
    std::vector<std::vector<size_t>> inputs;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    std::tuple<
        std::vector<int64_t>,
        std::vector<int64_t>,
        std::vector<int64_t>,
        std::vector<int64_t>,
        std::vector<int64_t>> strided_slice_params;
    std::tuple<
        std::size_t,
        std::vector<size_t>,
        std::vector<float>,
        std::vector<float>,
        std::vector<float>,
        std::vector<float>> fake_quantize_params;
    std::tie(inputs, netPrecision, targetDevice, additional_config, strided_slice_params, fake_quantize_params) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ov::ParameterVector input;
    for (auto&& shape : inputs) {
        input.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape)));
    }
    auto memory_read = ngraph::builder::makeConstant<size_t>(ngPrc, {inputs[0]}, {0});
    auto read = std::make_shared<ngraph::opset3::ReadValue>(memory_read, "variable1");
    auto fake_constatnt = ngraph::builder::makeConstant<size_t>(ngPrc, {inputs[0]}, {0});
    auto fake = ngraph::builder::makeFakeQuantize(fake_constatnt, ngPrc,
        std::get<0>(fake_quantize_params),
        std::get<1>(fake_quantize_params),
        std::get<2>(fake_quantize_params),
        std::get<3>(fake_quantize_params),
        std::get<4>(fake_quantize_params),
        std::get<5>(fake_quantize_params));
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{read, fake, input[0]}, 1);
    auto prelu_constant = ngraph::op::Constant::create(ngPrc, {1}, {-2});
    auto prelu = std::make_shared<ngraph::opset1::PRelu>(concat, prelu_constant);

    auto begin = std::get<0>(strided_slice_params);
    auto end = std::get<1>(strided_slice_params);
    auto stride = std::get<2>(strided_slice_params);
    auto begin_mask = std::get<3>(strided_slice_params);
    auto end_mask = std::get<4>(strided_slice_params);
    ov::Shape constShape = {begin.size()};
    auto beginNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, begin.data());
    auto endNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, end.data());
    auto strideNode = std::make_shared<ov::op::v0::Constant>(ov::element::i64, constShape, stride.data());
    auto slice = std::make_shared<ov::op::v1::StridedSlice>(prelu,
                                                            beginNode,
                                                            endNode,
                                                            strideNode,
                                                            begin_mask,
                                                            end_mask,
                                                            std::vector<int64_t>{},
                                                            std::vector<int64_t>{},
                                                            std::vector<int64_t>{});

    auto assign = std::make_shared<ngraph::opset3::Assign>(slice, "variable1");
    auto result = std::make_shared<ngraph::opset1::Result>(prelu);
    assign->add_control_dependency(read);
    result->add_control_dependency(assign);
    function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, input, "memory_fq_concat_prelu");
}

} // namespace SubgraphTestsDefinitions

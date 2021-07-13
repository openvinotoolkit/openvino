// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <tuple>

#include "base_reference_test.hpp"

namespace {

template <typename T>
std::vector<T> create_iota_vector(size_t size, T first_value = {}) {
    std::vector<T> d(size);
    std::iota(begin(d), end(d), first_value);
    return d;
}

template <typename T>
struct Iota {
    Iota(T e = {}): first_element_value(e) {}
    T first_element_value;
};

template <typename T>
Iota<T> iota_data(T e = {}) {
    return Iota<T> {e};
}

struct Tensor {
    Tensor() = default;
    Tensor(const ngraph::Shape& shape, ngraph::element::Type type, const InferenceEngine::Blob::Ptr& data): shape {shape}, type {type}, data {data} {}

    template <typename T>
    Tensor(const ngraph::Shape& shape, ngraph::element::Type type, const Iota<T>& data_info): shape {shape}, type {type} {
        data = CreateBlob(type, create_iota_vector<T>(shape_size(shape), data_info.first_element_value));
    }

    template <typename T>
    Tensor(const ngraph::Shape& shape, ngraph::element::Type type, std::initializer_list<T> data_elements): shape {shape}, type {type} {
        data = CreateBlob(type, std::vector<T>(data_elements));
    }

    ngraph::Shape shape;
    ngraph::element::Type type;
    InferenceEngine::Blob::Ptr data;
};

struct TestParams {
    Tensor input;
    Tensor in_low, in_high, out_low, out_high;
    size_t levels;
    Tensor expected;
};

struct ParameterBuilder {
    TestParams params;

#define ADD_SET_PARAM(set_p)                            \
    ParameterBuilder& set_p(decltype(params.set_p) t) { \
        params.set_p = std::move(t);                    \
        return *this;                                   \
    }
    ADD_SET_PARAM(input);
    ADD_SET_PARAM(in_low);
    ADD_SET_PARAM(in_high);
    ADD_SET_PARAM(out_low);
    ADD_SET_PARAM(out_high);
    ADD_SET_PARAM(levels);
    ADD_SET_PARAM(expected);
#undef ADD_SET_PARAM
};

template <typename T>
std::vector<T> iota_vector(size_t size, T first_value = {}) {
    std::vector<T> d(size);
    std::iota(begin(d), end(d), first_value);
    return d;
}

}  // namespace

class ReferenceFakeQuantizeLayerTest : public testing::TestWithParam<TestParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto p = GetParam();
        function = CreateFunction(p.input, p.in_low, p.in_high, p.out_low, p.out_high, p.levels, p.expected.type);
        inputData = {p.input.data};
        refOutData = {p.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<TestParams>& obj) {
        const TestParams& p = obj.param;
        std::ostringstream result;
        result << "in_shape=" << p.input.shape;
        result << "_in_type=" << p.input.type;
        result << "_in_low_shape=" << p.in_low.shape;
        result << "_in_low_type=" << p.in_low.type;
        result << "_in_high_shape=" << p.in_high.shape;
        result << "_in_high_type=" << p.in_high.type;
        result << "_out_low_shape=" << p.out_low.shape;
        result << "_out_low_type=" << p.out_low.type;
        result << "_out_high_shape=" << p.out_high.shape;
        result << "_out_high_type=" << p.out_high.type;
        result << "_levels=" << p.levels;
        return result.str();
    }

private:
    static std::shared_ptr<ngraph::Function> CreateFunction(const Tensor& input, const Tensor& in_low, const Tensor& in_high, const Tensor& out_low,
                                                            const Tensor& out_high, size_t levels, const ngraph::element::Type& expected_output_type) {
        using namespace ngraph;

        const auto data = std::make_shared<op::Parameter>(input.type, input.shape);

        auto in_low_mb = InferenceEngine::as<InferenceEngine::MemoryBlob>(in_low.data);
        auto in_low_m = in_low_mb->wmap();

        auto in_high_mb = InferenceEngine::as<InferenceEngine::MemoryBlob>(in_high.data);
        auto in_high_m = in_high_mb->wmap();

        auto out_low_mb = InferenceEngine::as<InferenceEngine::MemoryBlob>(out_low.data);
        auto out_low_m = out_low_mb->wmap();

        auto out_high_mb = InferenceEngine::as<InferenceEngine::MemoryBlob>(out_high.data);
        auto out_high_m = out_high_mb->wmap();

        const auto input_low = op::Constant::create(in_low.type, in_low.shape, in_low_m.as<void*>());
        const auto input_high = op::Constant::create(in_high.type, in_high.shape, in_high_m.as<void*>());
        const auto output_low = op::Constant::create(out_low.type, out_low.shape, out_low_m.as<void*>());
        const auto output_high = op::Constant::create(out_high.type, out_high.shape, out_high_m.as<void*>());

        const auto quantize = std::make_shared<op::FakeQuantize>(data, input_low, input_high, output_low, output_high, levels);
        const auto function = std::make_shared<Function>(ngraph::NodeVector {quantize}, ngraph::ParameterVector {data});

        return function;
    }
};

TEST_P(ReferenceFakeQuantizeLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

INSTANTIATE_TEST_SUITE_P(smoke_Fake_Quantize_With_Hardcoded_Refs, ReferenceFakeQuantizeLayerTest,
                         ::testing::Values(ParameterBuilder {}
                                               .input(Tensor {{1, 2, 3, 4}, ngraph::element::f32, iota_data(0.f)})
                                               .in_low(Tensor {{}, ngraph::element::f32, {0.f}})
                                               .in_high(Tensor {{}, ngraph::element::f32, {23.f}})
                                               .out_low(Tensor {{}, ngraph::element::f32, {2.f}})
                                               .out_high(Tensor {{}, ngraph::element::f32, {16.f}})
                                               .levels(4)
                                               .expected(Tensor {{1, 2, 3, 4},
                                                                 ngraph::element::f32,
                                                                 {2.f,          2.f,          2.f,          2.f,          6.6666669f,   6.6666669f,
                                                                  6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,   6.6666669f,
                                                                  11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f, 11.33333301f,
                                                                  11.33333301f, 11.33333301f, 16.f,         16.f,         16.f,         16.f}})
                                               .params),
                         ReferenceFakeQuantizeLayerTest::getTestCaseName);

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ie_core.hpp>
#include <ie_ngraph_utils.hpp>
#include <ngraph/ngraph.hpp>
#include <shared_test_classes/base/layer_test_utils.hpp>
#include <vector>

#include "base_reference_test.hpp"

using namespace ngraph;

namespace {

struct Tensor {
    Tensor() = default;

    Tensor(const ngraph::Shape& shape, ngraph::element::Type type, const InferenceEngine::Blob::Ptr& data): shape {shape}, type {type}, data {data} {}

    template <typename T>
    Tensor(const ngraph::Shape& shape, ngraph::element::Type type, const std::vector<T>& data_elements)
        : Tensor {shape, type, CreateBlob(type, data_elements)} {}

    ngraph::Shape shape;
    ngraph::element::Type type;
    InferenceEngine::Blob::Ptr data;
};

struct AcoshParams {
    Tensor input;
    Tensor expected;
};

struct Builder {
    AcoshParams params;

    operator AcoshParams() const {
        return params;
    }

#define ADD_SET_PARAM(set_p)                   \
    Builder& set_p(decltype(params.set_p) t) { \
        params.set_p = std::move(t);           \
        return *this;                          \
    }
    ADD_SET_PARAM(input);
    ADD_SET_PARAM(expected);
#undef ADD_SET_PARAM
};

class ReferenceAcoshLayerTest : public testing::TestWithParam<AcoshParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.input.shape, params.input.type);
        inputData = {params.input.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<AcoshParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.input.shape << "_";
        result << "type=" << param.input.type;
        return result.str();
    }

private:
    static std::shared_ptr<Function> CreateFunction(const Shape& shape, const element::Type& type) {
        const auto in = std::make_shared<op::Parameter>(type, shape);
        const auto acosh = std::make_shared<op::Acosh>(in);
        return std::make_shared<Function>(NodeVector {acosh}, ParameterVector {in});
    }
};

TEST_P(ReferenceAcoshLayerTest, AcoshWithHardcodedRefs) {
    Exec();
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_Acosh_With_Hardcoded_Refs, ReferenceAcoshLayerTest,
    ::testing::Values(Builder {}
                          .input({{8}, element::f16, std::vector<ngraph::float16> {1.f, 2.f, 3.f, 4.f, 5.f, 10.f, 100.f, 1000.f}})
                          .expected({{8}, element::f16, std::vector<ngraph::float16> {0., 1.317, 1.763, 2.063, 2.292, 2.993, 5.298, 7.6012}}),
                      Builder {}
                          .input({{8}, element::f32, std::vector<float> {1.f, 2.f, 3.f, 4.f, 5.f, 10.f, 100.f, 1000.f}})
                          .expected({{8}, element::f32, std::vector<float> {0., 1.317, 1.763, 2.063, 2.292, 2.993, 5.298, 7.6012}}),
                      Builder {}
                          .input({{8}, element::i32, std::vector<int32_t> {1, 2, 3, 4, 5, 10, 100, 1000}})
                          .expected({{8}, element::i32, std::vector<int32_t> {0, 1, 2, 2, 2, 3, 5, 8}}),
                      Builder {}
                          .input({{8}, element::i64, std::vector<int64_t> {1, 2, 3, 4, 5, 10, 100, 1000}})
                          .expected({{8}, element::i64, std::vector<int64_t> {0, 1, 2, 2, 2, 3, 5, 8}}),
                      Builder {}
                          .input({{8}, element::u32, std::vector<uint32_t> {1, 2, 3, 4, 5, 10, 100, 1000}})
                          .expected({{8}, element::u32, std::vector<uint32_t> {0, 1, 2, 2, 2, 3, 5, 8}}),
                      Builder {}
                          .input({{8}, element::u64, std::vector<uint64_t> {1, 2, 3, 4, 5, 10, 100, 1000}})
                          .expected({{8}, element::u64, std::vector<uint64_t> {0, 1, 2, 2, 2, 3, 5, 8}})),
    ReferenceAcoshLayerTest::getTestCaseName);

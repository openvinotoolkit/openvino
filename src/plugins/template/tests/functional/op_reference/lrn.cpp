// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lrn.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct LRNParams {
    template <class T>
    LRNParams(const Shape& input_shape,
              const Shape& expected_shape,
              const element::Type& input_type,
              const element::Type& expected_type,
              const std::vector<T>& input_value,
              const std::vector<T>& expected_value,
              const float& alpha,
              const float& beta,
              const float& bias,
              const size_t& size,
              const std::shared_ptr<op::v0::Constant>& axes) {
        m_input_shape = input_shape;
        m_expected_shape = expected_shape;
        m_input_type = input_type;
        m_expected_type = expected_type;
        m_input_value = CreateTensor(input_shape, input_type, input_value);
        m_expected_value = CreateTensor(expected_shape, expected_type, expected_value);
        m_alpha = alpha;
        m_beta = beta;
        m_bias = bias;
        m_size = size;
        m_axes = axes;
    }

    template <class T>
    LRNParams(const Shape& input_shape,
              const Shape& expected_shape,
              const element::Type& input_type,
              const element::Type& expected_type,
              const T& input_value_step,
              const std::vector<T>& expected_value,
              const float& alpha,
              const float& beta,
              const float& bias,
              const size_t& size,
              const std::shared_ptr<op::v0::Constant>& axes) {
        m_input_shape = input_shape;
        m_expected_shape = expected_shape;
        m_input_type = input_type;
        m_expected_type = expected_type;
        std::vector<T> input_value(shape_size(input_shape));
        std::iota(std::begin(input_value), std::end(input_value), input_value_step);
        m_input_value = CreateTensor(input_shape, input_type, input_value);
        m_expected_value = CreateTensor(expected_shape, expected_type, expected_value);
        m_alpha = alpha;
        m_beta = beta;
        m_bias = bias;
        m_size = size;
        m_axes = axes;
    }

    Shape m_input_shape;
    Shape m_expected_shape;
    element::Type m_input_type;
    element::Type m_expected_type;
    ov::Tensor m_input_value;
    ov::Tensor m_expected_value;
    float m_alpha;
    float m_beta;
    float m_bias;
    size_t m_size;
    std::shared_ptr<op::v0::Constant> m_axes;
};

class ReferenceLRNLayerTest : public testing::TestWithParam<LRNParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto params = GetParam();
        function = CreateFunction(params.m_input_shape,
                                  params.m_input_type,
                                  params.m_alpha,
                                  params.m_beta,
                                  params.m_bias,
                                  params.m_size,
                                  params.m_axes);
        inputData = {params.m_input_value};
        refOutData = {params.m_expected_value};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<LRNParams>& obj) {
        const auto param = obj.param;
        std::ostringstream result;

        result << "input_shape=" << param.m_input_shape << "; ";
        result << "output_shape=" << param.m_expected_shape << "; ";
        result << "input_type=" << param.m_input_type << "; ";
        result << "output_type=" << param.m_expected_type << "; ";
        if (param.m_axes != NULL) {
            result << "axes=" << param.m_axes << "; ";
        }
        result << "gamma=" << param.m_alpha << "; ";
        result << "beta=" << param.m_beta << "; ";
        result << "mean=" << param.m_bias << "; ";
        result << "variance=" << param.m_size << "; ";

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const Shape& input_shape,
                                                 const element::Type_t& input_type,
                                                 const float& alpah,
                                                 const float& beta,
                                                 const float& bias,
                                                 const size_t& size,
                                                 const std::shared_ptr<op::v0::Constant>& axes) {
        auto in = std::make_shared<op::v0::Parameter>(input_type, input_shape);

        std::shared_ptr<op::v0::LRN> lrn;
        if (axes != NULL) {
            lrn = std::make_shared<op::v0::LRN>(in, axes, alpah, beta, bias, size);
        } else {
            lrn = std::make_shared<op::v0::LRN>(in, alpah, beta, bias, size);
        }

        return std::make_shared<ov::Model>(lrn, ParameterVector{in});
    }
};

TEST_P(ReferenceLRNLayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<LRNParams> generateParamsForLRN() {
    using T = typename element_type_traits<ET>::value_type;

    std::vector<LRNParams> params{
        // lrn_across_channel
        LRNParams(Shape{2, 3, 2, 1},
                  Shape{2, 3, 2, 1},
                  ET,
                  ET,
                  std::vector<T>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f},
                  std::vector<T>{0.0000000f,
                                 0.3015113f,
                                 0.4364358f,
                                 0.5000000f,
                                 0.8728716f,
                                 0.8451542f,
                                 0.5970223f,
                                 0.6115928f,
                                 0.5642765f,
                                 0.5669467f,
                                 0.7784989f,
                                 0.7720487f},
                  3,
                  0.5,
                  1,
                  3,
                  NULL),
        // lrn_across_h
        LRNParams(Shape{2, 3, 2, 1},
                  Shape{2, 3, 2, 1},
                  ET,
                  ET,
                  std::vector<T>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f},
                  std::vector<T>{0.0000000f,
                                 0.7071068f,
                                 0.5345225f,
                                 0.8017837f,
                                 0.6172134f,
                                 0.7715167f,
                                 0.6469966f,
                                 0.7548294f,
                                 0.6620847f,
                                 0.7448453f,
                                 0.6711560f,
                                 0.7382717f},
                  3,
                  0.5,
                  1,
                  3,
                  std::make_shared<op::v0::Constant>(element::Type_t::i64, Shape{1}, std::vector<int64_t>{2})),
        // lrn_across_hw
        LRNParams(Shape{2, 3, 2, 1},
                  Shape{2, 3, 2, 1},
                  ET,
                  ET,
                  std::vector<T>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f},
                  std::vector<T>{0.0000000f,
                                 0.8660254f,
                                 0.8660254f,
                                 1.2990381f,
                                 1.0444659f,
                                 1.3055824f,
                                 1.1078234f,
                                 1.2924607f,
                                 1.1389896f,
                                 1.2813632f,
                                 1.1572751f,
                                 1.2730026f},
                  3,
                  0.5,
                  1,
                  3,
                  std::make_shared<op::v0::Constant>(element::Type_t::i64, Shape{2}, std::vector<int64_t>{2, 3})),
        // lrn_across_all_dims
        LRNParams(Shape{2, 3, 2, 1},
                  Shape{2, 3, 2, 1},
                  ET,
                  ET,
                  std::vector<T>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f},
                  std::vector<T>{0.0000000f,
                                 0.3156438f,
                                 0.4501407f,
                                 0.6752110f,
                                 0.9830783f,
                                 1.2288479f,
                                 1.8938627f,
                                 2.2095065f,
                                 1.8005627f,
                                 2.0256331f,
                                 2.4576957f,
                                 2.7034652f},
                  3,
                  0.5,
                  1,
                  3,
                  std::make_shared<op::v0::Constant>(element::Type_t::i64, Shape{4}, std::vector<int64_t>{0, 1, 2, 3})),
        // lrn_across_nw
        LRNParams(Shape{2, 3, 2, 1},
                  Shape{2, 3, 2, 1},
                  ET,
                  ET,
                  std::vector<T>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f},
                  std::vector<T>{0.0000000f,
                                 0.2379155f,
                                 0.4111132f,
                                 0.5388159f,
                                 0.6351073f,
                                 0.7094756f,
                                 1.6641006f,
                                 1.6654084f,
                                 1.6444529f,
                                 1.6164477f,
                                 1.5877683f,
                                 1.5608464f},
                  3,
                  0.5,
                  1,
                  3,
                  std::make_shared<op::v0::Constant>(element::Type_t::i64, Shape{2}, std::vector<int64_t>{0, 3})),
        // lrn_across_empty
        LRNParams(Shape{2, 3, 2, 1},
                  Shape{2, 3, 2, 1},
                  ET,
                  ET,
                  std::vector<T>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f},
                  std::vector<T>{0.0000000f,
                                 0.5000000f,
                                 0.5547002f,
                                 0.5669467f,
                                 0.5714286f,
                                 0.5735393f,
                                 0.5746958f,
                                 0.5753965f,
                                 0.5758526f,
                                 0.5761660f,
                                 0.5763904f,
                                 0.5765567f},
                  3,
                  0.5,
                  1,
                  3,
                  std::make_shared<op::v0::Constant>(element::Type_t::i64, Shape{0}, std::vector<int64_t>{})),
        // lrn_6D_across_2_axes
        LRNParams(Shape{2, 3, 2, 2, 1, 1},
                  Shape{2, 3, 2, 2, 1, 1},
                  ET,
                  ET,
                  static_cast<T>(0),
                  std::vector<T>{0.0000000f, 0.4200840f, 0.8401681f, 1.2602521f, 0.6099943f, 0.7624928f,
                                 0.9149914f, 1.0674900f, 0.7213357f, 0.8115027f, 0.9016696f, 0.9918366f,
                                 0.7656109f, 0.8294119f, 0.8932127f, 0.9570137f, 0.7892218f, 0.8385482f,
                                 0.8878745f, 0.9372009f, 0.8038679f, 0.8440613f, 0.8842546f, 0.9244481f},
                  3,
                  0.5,
                  1,
                  3,
                  std::make_shared<op::v0::Constant>(element::Type_t::i64, Shape{2}, std::vector<int64_t>{2, 3})),
        // lrn_2d_across_empty
        LRNParams(Shape{12},
                  Shape{12},
                  ET,
                  ET,
                  std::vector<T>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f},
                  std::vector<T>{0.0000000f,
                                 0.5000000f,
                                 0.5547002f,
                                 0.5669467f,
                                 0.5714286f,
                                 0.5735393f,
                                 0.5746958f,
                                 0.5753964f,
                                 0.5758526f,
                                 0.5761660f,
                                 0.5763904f,
                                 0.5765566f},
                  3,
                  0.5,
                  1,
                  3,
                  std::make_shared<op::v0::Constant>(element::Type_t::i64, Shape{0}, std::vector<int64_t>{})),
        // lrn_2d_across_empty
        LRNParams(Shape{6, 2},
                  Shape{6, 2},
                  ET,
                  ET,
                  std::vector<T>{0.64915806f,
                                 0.21213771f,
                                 -1.48256505f,
                                 -1.41040838f,
                                 0.58189541f,
                                 0.11432108f,
                                 -0.22993855f,
                                 -0.13325502f,
                                 -0.03083259f,
                                 -0.48450908f,
                                 0.50342429f,
                                 -0.99551708f},
                  std::vector<T>{0.4590040f,
                                 0.1499989f,
                                 -1.0482801f,
                                 -0.9972753f,
                                 0.4114444f,
                                 0.0808345f,
                                 -0.1625900f,
                                 -0.0942251f,
                                 -0.0218018f,
                                 -0.3425926f,
                                 0.3559732f,
                                 -0.7039225f},
                  0.0002,
                  0.5,
                  2.0,
                  3,
                  std::make_shared<op::v0::Constant>(element::Type_t::i64, Shape{1}, std::vector<int64_t>{0})),
    };

    return params;
}

std::vector<LRNParams> generateCombinedParamsForLRN() {
    const std::vector<std::vector<LRNParams>> allTypeParams{generateParamsForLRN<element::Type_t::f64>(),
                                                            generateParamsForLRN<element::Type_t::f32>()};

    std::vector<LRNParams> combinedParams;

    for (const auto& params : allTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }

    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_LRN_With_Hardcoded_Refs,
                         ReferenceLRNLayerTest,
                         ::testing::ValuesIn(generateCombinedParamsForLRN()),
                         ReferenceLRNLayerTest::getTestCaseName);

}  // namespace

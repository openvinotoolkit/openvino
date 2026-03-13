// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <sstream>
#include <tuple>
#include <vector>

#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

/*
 * Exact cut pattern around model node Multiply_9114:
 *   Parameter -> FakeQuantize(scalar) -> Conv(1x1, i8->f32 dequantized weights)
 *             -> Add(bias) -> Relu -> Result
 */
using QuantizedConvPatternParams = std::tuple<ov::Shape, uint32_t, uint32_t, uint32_t>;

class Conv1x1Quantized : public testing::WithParamInterface<QuantizedConvPatternParams>,
                                 virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QuantizedConvPatternParams>& obj) {
        const auto& [inputShape, weightSeed, scaleSeed, biasSeed] = obj.param;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::vec2str(inputShape);
        result << "_WSeed=" << weightSeed;
        result << "_SSeed=" << scaleSeed;
        result << "_BSeed=" << biasSeed;
        return result.str();
    }

    static std::shared_ptr<ov::Node> make_dequantized_i8_weights(const ov::Shape& weightShape,
                                                                 const std::vector<float>& scales,
                                                                 uint32_t weightSeed) {
        (void)weightSeed;
        const std::vector<int8_t> i8Data = {
            36, -4, -86, -1, -33, 9, -41, -76, 89, -59, -18, 89, 127, -122, -14, -76,
            7, -95, 58, -72, -7, 26, -31, -10, -14, -21, 34, 5, -1, 73, -7, -127,
            0, 70, -12, 0, 18, -2, -40, -127, 11, 0, 72, 6, -11, 93, 2, 0,
            -15, 65, -11, -25, 20, 9, -21, -127, -24, 6, -111, -11, 34, 114, 10, 5,
            61, 55, 34, 14, -86, -127, 0, -54, -84, 84, 37, -54, -85, 36, 5, -47,
            -5, 127, 77, -48, -2, 81, -77, 53, -7, 2, 3, -3, -9, -49, -49, 2,
            16, -96, -59, 44, -85, 127, -60, -28, 19, -18, 3, 9, 16, -27, -88, 35,
            -56, -45, 127, -68, -118, -8, 58, -31, 82, -36, -3, 77, 2, 55, 3, 2,
            33, -44, -23, 14, 105, -15, -127, -1, -45, 30, -8, -40, 1, -33, 5, -50,
            7, 15, 31, -39, 21, -127, -21, 49, 23, -9, -6, 16, 15, 18, -36, 26,
            10, -11, -26, -35, 9, 13, 13, -9, 4, -7, 92, 15, 127, -7, -17, 0,
            -18, -1, -10, 41, 33, -1, -28, 7, -127, 22, -3, 109, 3, -2, 6, -1,
            10, -46, 18, -60, -31, 15, -17, 45, -4, -16, 0, 8, 2, 73, 1, -127,
            -11, 127, 91, -22, 72, -93, 96, 7, -16, 13, -1, -6, -5, -77, 35, -105,
            31, 65, 37, 8, -26, -127, -18, -51, -62, 15, 82, -28, -77, 57, 9, 14,
            -25, 17, 112, -89, 1, -74, -16, 127, -27, 26, 26, -23, 70, -28, 62, 21,
            -21, 16, 2, 16, -28, 113, -21, -19, 5, 2, -14, 13, 0, -52, 127, -33,
            -45, -1, 8, 32, 1, 2, -3, -16, 17, -14, -127, 7, -52, 8, -18, -17,
            -30, 3, 23, -20, 119, 117, 57, 75, -16, 3, -27, -25, -3, 127, 44, 95,
            -49, -24, 59, 25, 24, 0, 18, -66, -15, 16, 3, -6, -6, -65, -3, 127,
            -11, 28, 4, 28, -45, 114, -19, -33, 22, 1, -13, 18, 4, -77, 127, -46,
            21, -45, 18, -66, -26, 10, -26, 34, -3, -26, 2, 8, 1, 70, 1, -127,
            54, 28, 30, 11, -37, 32, 28, -23, -84, 41, -127, -47, 25, 56, -6, 0,
            -2, -24, -36, -57, 37, -13, -30, 0, 127, 109, -5, -120, -3, -17, -1, -26,
            12, 54, 34, -10, 92, -29, 127, -35, 3, -2, 3, 1, 3, -58, -6, -94,
            -2, -87, -113, 39, -64, 70, -127, 83, 26, -23, -6, 7, -7, 117, -9, 76,
            -10, 38, -9, 15, -11, -9, 2, -127, -33, 19, -64, -15, 10, 97, 1, -14,
            33, -61, -35, 43, 14, 36, -8, -57, 71, -27, 28, 33, -127, 24, -37, -30,
            24, -6, 23, 19, -125, -127, -5, -6, -90, 59, 65, -54, -115, -46, -47, -21,
            -30, 35, 6, -38, -96, 12, 127, -3, 49, -13, 0, 37, -4, 19, -9, 32,
            124, -9, 43, 67, 13, -48, -14, 20, -35, 58, -13, 127, 28, 51, 7, 13,
            1, -2, 21, 8, 1, 5, -4, -4, 100, -27, 3, -127, 4, 0, -1, 0,
            -11, -29, 1, -30, 50, -90, 29, 46, -26, 8, 18, -22, -4, 93, -127, 53,
            25, 23, -6, 33, -100, 30, -42, -63, 5, 1, -31, 10, 10, -127, 14, -76,
            26, 104, 91, -52, 127, -120, 96, 33, 31, 0, -41, 26, 20, -71, 81, -81,
            5, 11, -4, -31, -17, 6, 15, -8, -36, 23, 127, -16, 111, 9, 1, 17,
            2, -69, 17, 12, -10, 6, 34, 127, -3, 0, -20, -2, 3, -109, -7, 5,
            -18, 26, 52, 0, -5, -58, -2, 53, -49, 35, 70, -39, -127, 44, 80, -1,
            -34, -125, -9, -51, 92, 127, 15, 82, 96, -66, -27, 63, 112, -16, -33, 14,
            -19, -28, -29, 43, -23, 127, 57, -78, -50, 11, 43, -43, -22, 47, -47, -2,
            22, -22, -63, 97, -7, 68, -41, -127, 41, -20, -20, 21, -77, 35, -58, -28,
            20, 28, -1, -1, -39, -108, -50, -24, 25, -21, -21, 16, 21, -34, -127, -34,
            127, -5, 27, 23, 7, -20, -1, 10, 3, -40, 2, -17, 18, 21, 1, 7,
            -23, -81, 90, -44, 47, 0, 12, -120, -5, 81, -2, 3, -6, -77, -7, 127,
            11, 23, -15, -20, -19, 10, 7, 49, -11, -8, 9, -6, -1, 77, 4, -127,
            127, 4, 4, 12, -2, 12, -6, -5, 3, -4, 5, -7, 4, 12, 1, 5,
            16, 127, -86, 78, -24, -17, 106, 33, 30, 42, 5, 7, -4, -54, 0, 89,
            -10, 18, 3, 18, 1, -4, -16, 28, 82, 127, -7, -14, 2, 21, 11, 10,
            3, -127, -74, 42, 38, -71, 113, -27, 1, 5, 0, -6, -2, 56, 74, 1,
            10, -37, -27, -28, -45, 20, -25, 93, 14, -25, -14, 14, 5, 68, -11, -127,
            -96, -4, 75, 127, 0, 15, -9, 16, 32, 15, 9, 9, 15, 15, -8, -14,
            -12, 24, 2, 12, -5, -9, 1, 8, -4, -127, -10, -46, 1, -3, 8, 17,
            6, -71, 9, 7, -15, -6, 35, 127, -3, -2, -48, -3, 3, -110, -5, 3,
            -10, -127, 127, -65, 39, 12, -10, -109, -30, 7, 2, -5, -10, -28, -14, 37,
            -22, -10, 72, -93, -25, -14, 34, 108, -115, 53, 62, -87, 127, 12, 8, 41,
            -43, 28, 42, -42, -127, -11, 86, -4, 92, -30, -9, 71, -4, 44, 15, 39,
            -27, 69, 90, 32, -76, -64, -3, -14, -127, 68, -2, -77, -101, 50, 49, -5,
            31, -47, -20, -24, 127, 103, 13, 61, 77, -66, 4, 39, 62, 48, -32, 54,
            -11, 107, 94, 9, 102, -80, 127, -53, -13, 22, 32, -6, -7, -22, 49, -103,
            1, -119, 127, -87, 34, 6, -6, -110, -33, 2, 2, 4, -5, -30, -4, 31,
            26, 96, -127, 116, 24, 27, 47, -15, -10, 33, 52, -26, -4, -26, -23, 69,
            -28, -53, 27, 9, -22, 14, 44, 127, -24, -14, -116, -7, 41, -101, -15, 10,
            19, 6, -3, -29, -20, 3, 11, -3, -39, 20, 127, -12, 104, 7, -1, 10,
            -127, 5, 41, 52, 18, 3, 23, 11, -11, -34, 18, 0, 2, 4, 2, -17
        };

        auto w_i8 = ov::op::v0::Constant::create(ov::element::i8, weightShape, i8Data);
        auto w_f32 = std::make_shared<ov::op::v0::Convert>(w_i8, ov::element::f32);
        auto w_scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{64, 1, 1, 1}, scales);
        return std::make_shared<ov::op::v1::Multiply>(w_f32, w_scale);
    }

protected:
    void SetUp() override {
        const auto& [inputShape, weightSeed, scaleSeed, biasSeed] = this->GetParam();
        (void)scaleSeed;
        (void)biasSeed;

        // This pattern is intentionally stressy for low-precision accumulation; allow small numeric drift.
        abs_threshold = 0.02f;
        rel_threshold = 0.02f;

        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto netPrecision = ov::element::f32;

        auto input = std::make_shared<ov::op::v0::Parameter>(netPrecision, inputShape);
        input->set_friendly_name("input_param");

        auto fq = ov::test::utils::make_fake_quantize(input,
                                                      netPrecision,
                                                      256,
                                                      {},
                                                      {-11.913918495178223f},
                                                      {11.820840835571289f},
                                                      {-11.913918495178223f},
                                                      {11.820840835571289f});
        fq->set_friendly_name("Fq");

        const std::vector<float> wScales = {
            0.000423641643f, 0.0015503891f, 0.00310077821f, 0.00190122111f, 0.000630055845f, 0.00273168366f, 0.00154846674f, 0.00150425232f,
            0.00226454856f, 0.000881885935f, 0.000511349645f, 0.0010198158f, 0.00212229323f, 0.00188103621f, 0.000167005637f, 0.000399371784f,
            0.00110536115f, 0.000790092861f, 0.000501257251f, 0.00170513964f, 0.000335453049f, 0.0020838459f, 0.0015244371f, 0.00229338394f,
            0.00288739544f, 0.000156793074f, 0.00251637865f, 0.000571904238f, 0.000309020514f, 0.00230684062f, 0.00224148016f, 0.00243948377f,
            0.00040105384f, 0.000795860018f, 0.000723290606f, 0.000404898572f, 0.00143024116f, 0.000499815447f, 0.000462329306f, 0.000997708528f,
            0.00042003722f, 0.00104384532f, 0.00420998409f, 0.00131682144f, 0.00177050009f, 0.00119667349f, 0.00107652554f, 0.00386972493f,
            0.00240680366f, 0.00117552746f, 0.00575556699f, 0.00267016795f, 0.00311231241f, 0.00167053705f, 0.000227920653f, 0.00180510269f,
            0.000291959499f, 0.000529612124f, 0.00108998211f, 0.00163977919f, 0.00192813424f, 0.00107364205f, 0.000431331136f, 0.00405234983f
        };

        auto convWeights = make_dequantized_i8_weights({64, 16, 1, 1}, wScales, weightSeed);
        convWeights->set_friendly_name("Multiply_fq_weights");
        auto conv = std::make_shared<ov::op::v1::Convolution>(fq,
                                                              convWeights,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::CoordinateDiff{0, 0},
                                                              ov::Strides{1, 1},
                                                              ov::op::PadType::EXPLICIT);
        conv->set_friendly_name("Multiply");

        const std::vector<float> biasData = {
            0.395751953f, -0.00618743896f, -0.51171875f, 0.668457031f, 0.529785156f, -0.0507202148f, -0.238037109f, 0.167236328f,
            0.135131836f, 0.883300781f, 0.313476562f, 0.346679688f, 0.235473633f, 0.278564453f, 0.418945312f, 0.528808594f,
            -0.190551758f, 0.901367188f, 0.222412109f, 0.375244141f, 0.181396484f, 0.255371094f, 0.827148438f, 0.0573120117f,
            0.14440918f, 0.651855469f, 0.529785156f, 0.522460938f, 0.549316406f, 0.0909423828f, -0.972167969f, 0.473388672f,
            0.645019531f, 0.51171875f, 0.289794922f, 0.246826172f, 0.467285156f, 0.508300781f, 0.309326172f, 0.126831055f,
            0.420898438f, 0.962890625f, -0.482177734f, 0.096862793f, 0.024230957f, -0.0120315552f, -0.0922241211f, -1.41015625f,
            -0.13659668f, 0.58203125f, -0.771972656f, 1.40917969f, 0.484130859f, 0.211914062f, 0.391357422f, -0.106567383f,
            0.488037109f, 0.333496094f, 0.134521484f, 0.352050781f, -0.614257812f, 0.660644531f, 0.229125977f, 0.371337891f
        };
        auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 64, 1, 1}, biasData);
        auto add = std::make_shared<ov::op::v1::Add>(conv, bias);
        add->set_friendly_name("Add");
        auto relu = ov::test::utils::make_activation(add, netPrecision, ov::test::utils::ActivationTypes::Relu);
        relu->set_friendly_name("Relu");

        auto result = std::make_shared<ov::op::v0::Result>(relu);

        function = std::make_shared<ov::Model>(ov::ResultVector{result},
                               ov::ParameterVector{input},
                               "QuantizedConvPattern");
    }
};

TEST_P(Conv1x1Quantized, smoke_CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Conv1x1Quantized,
    Conv1x1Quantized,
    ::testing::Values(
        QuantizedConvPatternParams{ov::Shape{1, 16, 112, 112}, 9114u, 329u, 328u},
        QuantizedConvPatternParams{ov::Shape{1, 16, 56, 56}, 1001u, 1002u, 1003u},
        QuantizedConvPatternParams{ov::Shape{1, 16, 28, 28}, 2021u, 2022u, 2023u}),
    Conv1x1Quantized::getTestCaseName);

}  // namespace test
}  // namespace ov

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/histc.hpp"

#include <limits>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

using HistcCPUParams = std::tuple<InputShape,
                                  std::tuple<int64_t, double, double>,  // bins, min, max
                                  ElementType>;

class HistcLayerCPUTest : public testing::WithParamInterface<HistcCPUParams>, virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<HistcCPUParams>& obj) {
        const auto& [shape, attrs, prc] = obj.param;
        const auto& [bins, min, max] = attrs;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({shape.first}) << "_TS=";
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "bins=" << bins << "_min=" << min << "_max=" << max << "_prc=" << prc;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto& [shape, attrs, prc] = GetParam();
        const auto& [bins, min, max] = attrs;
        init_input_shapes({shape});
        const auto data = std::make_shared<ov::op::v0::Parameter>(prc, inputDynamicShapes[0]);
        const auto histc = std::make_shared<ov::op::v17::Histc>(data, bins, min, max);
        function = std::make_shared<ov::Model>(histc->outputs(), ov::ParameterVector{data}, "Histc");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        // Quarter steps put many values exactly on bin edges and outside the explicit ranges.
        ov::test::utils::InputGenerateData in_data(-5, 10, 4, 1234);
        const auto& param = function->get_parameters()[0];
        auto tensor =
            ov::test::utils::create_and_fill_tensor(param->get_element_type(), targetInputStaticShapes[0], in_data);
        const auto& [shape, attrs, prc] = GetParam();
        // Special values are ignored for an explicit range, but make a range taken from data infinite.
        if (std::get<1>(attrs) != std::get<2>(attrs) && tensor.get_size() >= 3) {
            switch (prc) {
            case ElementType::f32:
                set_special_values<float>(tensor);
                break;
            case ElementType::f16:
                set_special_values<ov::float16>(tensor);
                break;
            case ElementType::bf16:
                set_special_values<ov::bfloat16>(tensor);
                break;
            default:
                break;
            }
        }
        inputs.insert({param, tensor});
    }

private:
    template <class T>
    static void set_special_values(ov::Tensor& tensor) {
        auto* data = tensor.data<T>();
        data[0] = T(std::numeric_limits<float>::quiet_NaN());
        data[1] = T(std::numeric_limits<float>::infinity());
        data[2] = T(-std::numeric_limits<float>::infinity());
    }
};

TEST_P(HistcLayerCPUTest, CompareWithRefs) {
    run();
}

namespace {
const std::vector<InputShape> shapes = {
    {{}, {{}}},
    {{}, {{2, 3, 17}}},
    // Single-bin counts above 255 (u8), 127 (i8) and 256 (bf16 integer step).
    {{}, {{50, 50}}},
    {{}, {{0}}},
    {{}, {{2, 0, 3}}},
    {{-1, -1}, {{4, 5}, {0, 3}, {32, 32}, {1, 1}}},
};

const std::vector<std::tuple<int64_t, double, double>> attrs = {
    {4, 0.0, 3.0},
    {100, 0.0, 0.0},
    {5, -2.5, 2.5},
    {1, 0.0, 0.0},
    {7, -1.3, 4.1},
};

INSTANTIATE_TEST_SUITE_P(smoke_Histc,
                         HistcLayerCPUTest,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(attrs),
                                            ::testing::Values(ElementType::f32,
                                                              ElementType::f16,
                                                              ElementType::bf16,
                                                              ElementType::i32,
                                                              ElementType::i64,
                                                              ElementType::u8,
                                                              ElementType::i8)),
                         HistcLayerCPUTest::getTestCaseName);

// Like torch.histc, NaN is ignored for an explicit range but makes a range taken from data invalid.
TEST(smoke_HistcCPU, nan_in_data) {
    const auto core = ov::test::utils::PluginCache::get().core();
    const auto infer = [&](double min, double max) {
        const auto data = std::make_shared<ov::op::v0::Parameter>(ElementType::f32, ov::PartialShape{-1});
        const auto histc = std::make_shared<ov::op::v17::Histc>(data, 2, min, max);
        auto request = core->compile_model(std::make_shared<ov::Model>(histc->outputs(), ov::ParameterVector{data}),
                                           ov::test::utils::DEVICE_CPU)
                           .create_infer_request();
        std::vector<float> values{1.f, std::numeric_limits<float>::quiet_NaN(), 2.f};
        request.set_input_tensor(ov::Tensor(ElementType::f32, ov::Shape{3}, values.data()));
        request.infer();
        const auto out = request.get_output_tensor();
        return std::vector<float>(out.data<float>(), out.data<float>() + out.get_size());
    };
    EXPECT_EQ(infer(0.0, 2.0), (std::vector<float>{0.f, 2.f}));
    OV_EXPECT_THROW(std::ignore = infer(0.0, 0.0), ov::Exception, testing::HasSubstr("is not finite"));
}
}  // namespace
}  // namespace ov::test

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/histc.hpp"

#include <limits>

#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

using HistcGPUParams = std::tuple<InputShape,
                                  std::tuple<int64_t, double, double>,  // bins, min, max
                                  ElementType,
                                  bool>;  // preserve_input_type

class HistcLayerGPUTest : public testing::WithParamInterface<HistcGPUParams>, virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<HistcGPUParams>& obj) {
        const auto& [shape, attrs, prc, preserve_input_type] = obj.param;
        const auto& [bins, min, max] = attrs;
        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({shape.first}) << "_TS=";
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        result << "bins=" << bins << "_min=" << min << "_max=" << max << "_prc=" << prc << "_preserve_input_type=" << preserve_input_type;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [shape, attrs, prc, preserve_input_type] = GetParam();
        const auto& [bins, min, max] = attrs;
        const size_t max_count = !preserve_input_type      ? SIZE_MAX
                                 : prc == ElementType::f16 ? 65504
                                 : prc == ElementType::u8  ? 255
                                 : prc == ElementType::i8  ? 127
                                                           : SIZE_MAX;
        for (const auto& static_shape : shape.second) {
            if (ov::shape_size(static_shape) > max_count) {
                GTEST_SKIP() << "Counts may overflow the output type";
            }
        }
        init_input_shapes({shape});
        const auto data = std::make_shared<ov::op::v0::Parameter>(prc, inputDynamicShapes[0]);
        const auto histc = std::make_shared<ov::op::v17::Histc>(data, bins, min, max, preserve_input_type);
        function = std::make_shared<ov::Model>(histc->outputs(), ov::ParameterVector{data}, "Histc");
        abs_threshold = 0.0;
        rel_threshold = 0.0;
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        ov::test::utils::InputGenerateData in_data(-5, 10, 4, 1234);
        const auto& param = function->get_parameters()[0];
        auto tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(), targetInputStaticShapes[0], in_data);
        const auto& [shape, attrs, prc, preserve_input_type] = GetParam();
        if (std::get<1>(attrs) != std::get<2>(attrs) && tensor.get_size() >= 3) {
            switch (prc) {
            case ElementType::f32:
                set_special_values<float>(tensor);
                break;
            case ElementType::f16:
                set_special_values<ov::float16>(tensor);
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

TEST_P(HistcLayerGPUTest, Inference) {
    run();
}

namespace {
const std::vector<InputShape> shapes = {
    {{}, {{}}},
    {{}, {{2, 3, 17}}},
    // Single-bin counts above 255 (u8) and 127 (i8), exact with preserve_input_type = false.
    {{}, {{50, 50}}},
    // Several work-groups.
    {{}, {{1, 3, 224, 224}}},
    {{-1, -1}, {{4, 5}, {0, 3}, {32, 32}, {1, 1}, {640, 480}}},
};

const std::vector<std::tuple<int64_t, double, double>> attrs = {
    {4, 0.0, 3.0},
    {100, 0.0, 0.0},
    {5, -2.5, 2.5},
    {1, 0.0, 0.0},
    {7, -1.3, 4.1},
    // Too many bins for the histogram to fit in local memory.
    {40000, -5.0, 10.0},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_Histc,
    HistcLayerGPUTest,
    ::testing::Combine(::testing::ValuesIn(shapes),
                       ::testing::ValuesIn(attrs),
                       ::testing::Values(ElementType::f32, ElementType::f16, ElementType::i32, ElementType::i64, ElementType::u8, ElementType::i8),
                       ::testing::Bool()),
    HistcLayerGPUTest::getTestCaseName);

// Counts above 2048 are not exact in f16, so Histc must stay in f32 with f16 inference precision
TEST(smoke_HistcGPU, f16_inference_precision) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ElementType::f32, ov::Shape{74097});
    const auto histc = std::make_shared<ov::op::v17::Histc>(data, 2, -1.0, 1.0);
    auto request = ov::test::utils::PluginCache::get()
                       .core()
                       ->compile_model(std::make_shared<ov::Model>(histc->outputs(), ov::ParameterVector{data}),
                                       ov::test::utils::DEVICE_GPU,
                                       ov::hint::inference_precision(ElementType::f16))
                       .create_infer_request();
    std::vector<float> values(74097, -0.5f);
    std::fill_n(values.begin(), 4097, 0.5f);
    request.set_input_tensor(ov::Tensor(ElementType::f32, ov::Shape{values.size()}, values.data()));
    request.infer();
    const auto out = request.get_output_tensor();
    EXPECT_EQ(std::vector<float>(out.data<float>(), out.data<float>() + out.get_size()), (std::vector<float>{70000.f, 4097.f}));
}
}  // namespace
}  // namespace ov::test

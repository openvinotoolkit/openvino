// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "internal_properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

typedef std::tuple<std::vector<InputShape>, ElementType, ElementType, size_t> NgramTestParams;

static std::shared_ptr<ov::Node> getStridedSlice(const std::shared_ptr<ov::Node>& data,
                                                 const std::shared_ptr<ov::Node>& begin,
                                                 const std::shared_ptr<ov::Node>& end,
                                                 const std::vector<int64_t>& shrink_axis_mask = {}) {
    std::vector<int64_t> default_mask(begin->get_shape()[0], 0);
    return std::make_shared<ov::op::v1::StridedSlice>(data,
                                                      begin,
                                                      end,
                                                      default_mask,
                                                      default_mask,
                                                      std::vector<int64_t>{},
                                                      shrink_axis_mask);
}

static std::shared_ptr<ov::Node> getReshape(const std::shared_ptr<ov::Node>& data,
                                            const std::vector<int64_t>& requested_shape,
                                            const ov::element::Type& prc) {
    auto requested_shape_node = ov::op::v0::Constant::create(prc, {requested_shape.size()}, requested_shape);
    return std::make_shared<ov::op::v1::Reshape>(data, requested_shape_node, true);
}

static std::shared_ptr<ov::Model> initNgram(std::vector<ov::PartialShape>& input_shapes,
                                            const ov::element::Type& data_et,
                                            const ov::element::Type& idces_et,
                                            const size_t k) {
    const size_t left_pad = k % 2 == 0 ? (k - 1) / 2 : k / 2;
    const size_t right_pad = k / 2;
    const size_t mid_idx = left_pad;

    ov::element::TypeVector input_precisions{data_et, idces_et};
    ov::ParameterVector params;
    for (size_t i = 0; i < input_precisions.size(); i++) {
        auto param_node = std::make_shared<ov::op::v0::Parameter>(input_precisions[i], input_shapes[i]);
        params.push_back(param_node);
    }
    auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(params[0], idces_et);
    auto shape_ss_begin = ov::op::v0::Constant::create(idces_et, {1}, {0});
    auto shape_ss_end = ov::op::v0::Constant::create(idces_et, {1}, {1});
    auto shape_ss = getStridedSlice(shape_of, shape_ss_begin, shape_ss_end, {1});

    auto getInputsToPad = [&](const ov::Output<ov::Node> data, const int pad_value) {
        const size_t length = data.get_partial_shape()[1].get_length();
        ov::OutputVector inputs;
        if (left_pad > 0) {
            inputs.push_back(ov::op::v0::Constant::create(data.get_element_type(), {left_pad, length}, {pad_value}));
        }
        inputs.push_back(data);
        if (right_pad > 0) {
            inputs.push_back(ov::op::v0::Constant::create(data.get_element_type(), {right_pad, length}, {pad_value}));
        }
        return inputs;
    };

    auto data_padded = std::make_shared<ov::op::v0::Concat>(getInputsToPad(params[0], 0), 0);
    auto idces_padded = std::make_shared<ov::op::v0::Concat>(getInputsToPad(params[1], -1), 0);

    std::shared_ptr<ov::Node> as_is_bias = shape_ss;
    if (mid_idx != 0) {
        auto bias_const = ov::op::v0::Constant::create(idces_et, {}, {mid_idx});
        as_is_bias = std::make_shared<ov::op::v1::Add>(shape_ss, bias_const);
    }
    auto as_is_ss_begin = ov::op::v0::Constant::create(idces_et, {1}, {mid_idx});
    auto as_is_ss_end = getReshape(as_is_bias, {1}, idces_et);
    auto as_is_ss = getStridedSlice(data_padded, as_is_ss_begin, as_is_ss_end);

    auto getSelectBranch = [&](const size_t cur_idx, const size_t mid_idx) {
        std::shared_ptr<ov::Node> eq_left_bias = shape_ss;
        if (cur_idx != 0) {
            auto bias_const = ov::op::v0::Constant::create(idces_et, {}, {cur_idx});
            eq_left_bias = std::make_shared<ov::op::v1::Add>(shape_ss, bias_const);
        }
        auto eq_left_reshape = getReshape(eq_left_bias, {1}, idces_et);
        auto eq_left_concat_const = ov::op::v0::Constant::create(idces_et, {1}, {1});
        auto eq_left_concat =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{eq_left_reshape, eq_left_concat_const}, 0);
        auto eq_left_ss_begin = ov::op::v0::Constant::create(idces_et, {2}, std::vector<size_t>{cur_idx, 0ul});
        auto eq_left_ss = getStridedSlice(idces_padded, eq_left_ss_begin, eq_left_concat, {0, 1});

        std::shared_ptr<ov::Node> eq_right_bias = shape_ss;
        if (mid_idx != 0) {
            auto bias_const = ov::op::v0::Constant::create(idces_et, {}, {mid_idx});
            eq_right_bias = std::make_shared<ov::op::v1::Add>(shape_ss, bias_const);
        }
        auto eq_right_reshape = getReshape(eq_right_bias, {1}, idces_et);
        auto eq_right_concat_const = ov::op::v0::Constant::create(idces_et, {1}, {1});
        auto eq_right_concat =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{eq_right_reshape, eq_right_concat_const}, 0);
        auto eq_right_ss_begin = ov::op::v0::Constant::create(idces_et, {2}, std::vector<size_t>{mid_idx, 0ul});
        auto eq_right_ss = getStridedSlice(idces_padded, eq_right_ss_begin, eq_right_concat, {0, 1});

        auto equal = std::make_shared<ov::op::v1::Equal>(eq_left_ss, eq_right_ss);
        auto cond = getReshape(equal, {-1, 1}, idces_et);

        std::shared_ptr<ov::Node> then_bias = shape_ss;
        if (cur_idx != 0) {
            auto bias_const = ov::op::v0::Constant::create(idces_et, {}, {cur_idx});
            then_bias = std::make_shared<ov::op::v1::Add>(shape_ss, bias_const);
        }
        auto then_reshape = getReshape(then_bias, {1}, idces_et);
        auto then_ss_begin = ov::op::v0::Constant::create(idces_et, {1}, {cur_idx});
        auto then = getStridedSlice(data_padded, then_ss_begin, then_reshape);

        auto else_reshape = getReshape(shape_ss, {1}, idces_et);
        auto else_concat_const = ov::op::v0::Constant::create(idces_et, {1}, {input_shapes[0][1].get_length()});
        auto else_concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{else_reshape, else_concat_const}, 0);
        auto else_bcast_const = ov::op::v0::Constant::create(data_et, {}, {0});
        auto else_bcast = std::make_shared<ov::op::v1::Broadcast>(else_bcast_const, else_concat);

        auto select = std::make_shared<ov::op::v1::Select>(cond, then, else_bcast);
        return select;
    };

    ov::OutputVector concat_inputs(k);
    concat_inputs[mid_idx] = as_is_ss;
    for (size_t i = 0; i < k; ++i) {
        if (i == mid_idx)
            continue;
        concat_inputs[i] = getSelectBranch(i, mid_idx);
    }

    auto final_concat = std::make_shared<ov::op::v0::Concat>(concat_inputs, 1);
    return std::make_shared<ov::Model>(final_concat, params, "ngram");
}

class NgramCPUTest : public testing::WithParamInterface<NgramTestParams>,
                     virtual public SubgraphBaseTest,
                     public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<NgramTestParams>& obj) {
        std::vector<InputShape> input_shapes;
        size_t k;
        ElementType data_et;
        ElementType idces_et;
        std::tie(input_shapes, data_et, idces_et, k) = obj.param;
        std::ostringstream results;

        results << "IS=(";
        for (const auto& shape : input_shapes) {
            results << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        results << ")_TS=(";
        for (const auto& shape : input_shapes) {
            for (const auto& item : shape.second) {
                results << ov::test::utils::vec2str(item) << "_";
            }
        }
        results << ")_data_prc=" << data_et << "_idces_prc=" << idces_et << "_k=" << k;
        return results.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& model_inputs = function->inputs();

        const auto& data_et = model_inputs[0].get_element_type();
        const auto& data_shape = targetInputStaticShapes[0];
        auto embeddings_tensor = ov::test::utils::create_and_fill_tensor_consistently(data_et, data_shape, 100, 1, 1);
        inputs.insert({model_inputs[0].get_node_shared_ptr(), embeddings_tensor});

        const auto& indices_et = model_inputs[1].get_element_type();
        const auto& indices_shape = targetInputStaticShapes[1];
        const size_t batch_size = data_shape[0];
        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 0;
        in_data.range = batch_size;
        auto indices_tensor = ov::test::utils::create_and_fill_tensor(indices_et, indices_shape, in_data);

        if (indices_et == ov::element::i32) {
            auto* indices_data = indices_tensor.data<int32_t>();
            std::sort(indices_data, indices_data + indices_tensor.get_size());
        } else if (indices_et == ov::element::i64) {
            auto* indices_data = indices_tensor.data<int64_t>();
            std::sort(indices_data, indices_data + indices_tensor.get_size());
        } else {
            OPENVINO_THROW("Unexpected indices precision: ", indices_et);
        }
        inputs.insert({model_inputs[1].get_node_shared_ptr(), indices_tensor});
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        std::vector<InputShape> inputShapes;
        ElementType data_et;
        ElementType idces_et;
        size_t k;
        std::tie(inputShapes, data_et, idces_et, k) = this->GetParam();
        init_input_shapes(inputShapes);
        function = initNgram(inputDynamicShapes, data_et, idces_et, k);

        if (!configuration.count(ov::intel_cpu::snippets_mode.name())) {
            configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
        }
    }
};

TEST_P(NgramCPUTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Ngram", 1);
}

namespace {

std::vector<std::vector<InputShape>> inputShapes = {
    {InputShape{{-1, 2}, {{3, 2}, {5, 2}, {7, 2}}}, InputShape{{-1, 2}, {{3, 2}, {5, 2}, {7, 2}}}},
    {InputShape{{-1, 256}, {{12, 256}, {25, 256}}}, InputShape{{-1, 2}, {{12, 2}, {25, 2}}}},
    {InputShape{{-1, 1}, {{12, 1}}}, InputShape{{-1, 2}, {{12, 2}}}},
};

std::vector<size_t> k_values = {2, 3, 5, 7};
std::vector<ElementType> idces_precisions = {ElementType::i32, ElementType::i64};

INSTANTIATE_TEST_SUITE_P(smoke_Ngram,
                         NgramCPUTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapes),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(idces_precisions),
                                            ::testing::ValuesIn(k_values)),
                         NgramCPUTest::getTestCaseName);
}  // namespace
}  // namespace test
}  // namespace ov

// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/manager.hpp"

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/include/common_test_utils/ov_tensor_utils.hpp"

using namespace ov::test;
using namespace CPUTestUtils;
using namespace ov::op;

namespace ov {
namespace test {
/*
  This test runs a graph that is equivelent to torch.Tensor.index_add_.
  TorchFE maps it to a compilicated subgraph which could be briefed similar to this -
 *                       Indices(1D)
 *                           |
 *                           |
 *                  X    Broadcast   Updates
 *                   \       |         /
 *                    \      |        /
 *                  ScatterElementsUpdate
 *                           |
 *                         Result
*/
using InputsAndAxis = std::tuple<
        std::vector<InputShape>,           // Input, shape of data and updates
        int                                // Axis
>;
using IndexAddTestParams = std::tuple<InputsAndAxis,                   // Input shapes and axis
                                    v12::ScatterElementsUpdate::Reduction,  // Reduce mode
                                    ElementType,  // model precision
                                    ElementType,  // indices precision
                                    float,        // alpha
                                    bool          // dynamic shape test
                                    >;

class IndexAddTest : public testing::WithParamInterface<IndexAddTestParams>,
                     virtual public ov::test::SubgraphBaseTest,
                     public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<IndexAddTestParams> obj) {
        auto shapes_ss = [](const InputShape& shape) {
            std::stringstream ss;
            ss << "_IS=(" << ov::test::utils::partialShape2str({shape.first}) << ")_TS=";
            for (size_t j = 0lu; j < shape.second.size(); j++)
                ss << "{" << ov::test::utils::vec2str(shape.second[j]) << "}";
            return ss;
        };

        InputsAndAxis shapes_desc;
        std::vector<InputShape> input_shapes;
        int axis;
        v12::ScatterElementsUpdate::Reduction reduceMode;
        ov::element::Type data_type, indices_type;
        float alpha;
        bool dynamic;

        std::tie(shapes_desc, reduceMode, data_type, indices_type, alpha, dynamic) = obj.param;
        std::tie(input_shapes, axis) = shapes_desc;
        std::ostringstream result;
        result << "InputShape=" << shapes_ss(input_shapes.at(0)).str() << "_";
        result << "UpdateShape=" << ov::test::utils::vec2str(input_shapes.at(1).second) << "_";
        result << "Axis=" << axis << "_";
        result << "ReduceMode=" << as_string(reduceMode) << "_";
        result << "modelType=" << data_type.to_string() << "_";
        result << "idxType=" << indices_type.to_string() << "_";
        result << "alpha=" << alpha;
        result << "dynamic=" << dynamic;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        constexpr size_t DATA_INPUT_IDX = 0;
        constexpr size_t UPDATES_INPUT_IDX = 1;

        InputsAndAxis shapes_desc;
        std::vector<InputShape> input_shapes;
        int axis;

        v12::ScatterElementsUpdate::Reduction reduceMode;
        float alpha_value;
        bool dynamic;

        ov::element::Type data_type, indices_type;
        std::string target_device;
        std::tie(shapes_desc, reduceMode, data_type, indices_type, alpha_value, dynamic) = this->GetParam();
        std::tie(input_shapes, axis) = shapes_desc;

        if (ov::element::bf16 == data_type || ov::element::f16 == data_type) {
            configuration.insert({ov::hint::inference_precision.name(), data_type});
            abs_threshold = 0.01f;
            rel_threshold = 0.01f;
        }

        init_input_shapes(input_shapes);

        //
        normalized_axis = axis < 0 ? axis + inputDynamicShapes.at(DATA_INPUT_IDX).rank().get_length(): axis;

        if (dynamic) {
            // infer dynamic shape from axis
            inputDynamicShapes.at(DATA_INPUT_IDX)[normalized_axis] = -1;
            inputDynamicShapes.at(UPDATES_INPUT_IDX)[normalized_axis] = -1;
        }

        auto param = std::make_shared<v0::Parameter>(data_type, inputDynamicShapes.at(DATA_INPUT_IDX));
        param->set_friendly_name("data");
        auto update_param = std::make_shared<v0::Parameter>(data_type, inputDynamicShapes.at(UPDATES_INPUT_IDX));
        update_param->set_friendly_name("update");
        auto indices_param = std::make_shared<v0::Parameter>(indices_type, ov::PartialShape{-1}); // 1D
        indices_param->set_friendly_name("indices");

        auto axis_const =
            std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{}, std::vector<int>{axis});
        axis_const->set_friendly_name("axis");
        auto alpha_const =
            std::make_shared<v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{alpha_value});
        alpha_const->set_friendly_name("alpha");

        auto input = param;
        auto dim = axis_const;
        auto index = std::make_shared<v0::Convert>(indices_param, element::i32);
        auto src = update_param;
        auto alpha = alpha_const;
        auto converted_alpha = std::make_shared<v1::ConvertLike>(alpha, src);
        auto alpha_src = std::make_shared<v1::Multiply>(converted_alpha, src);
        auto input_shape_rank = get_shape_rank(input);
        auto const_one = v0::Constant::create(element::i32, Shape{1}, {1});
        auto const_one_0d = v0::Constant::create(element::i32, Shape{}, {1});
        auto inp_rank = std::get<1>(input_shape_rank);
        // ScatterElementsUpdate required that index, source and update have the same rank
        // in aten::index_add index represents as 1d-array for specific dim and update may have different size
        // from source in non-indexing axes
        // slice src for having only relevant data
        auto src_broadcast_shape = std::make_shared<v3::Broadcast>(const_one, inp_rank);
        auto src_broadcasted =
            std::make_shared<v3::Broadcast>(alpha_src, src_broadcast_shape, BroadcastType::BIDIRECTIONAL);
        auto src_shape_rank = get_shape_rank(src_broadcasted);
        auto const_zero = v0::Constant::create(element::i32, Shape{1}, {0});
        auto src_rank = std::get<1>(src_shape_rank);
        auto slice_start = std::make_shared<v3::Broadcast>(const_zero, inp_rank);
        auto axes = get_node_axes_range(src_broadcasted);
        auto const_inf =
            v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()});
        auto slice_end = std::make_shared<v3::Broadcast>(const_inf, src_rank);
        auto slice_step = std::make_shared<v3::Broadcast>(const_one, src_rank);
        auto dim_1d = std::make_shared<v3::Broadcast>(dim, const_one);
        auto slice_end2 =
            std::make_shared<v12::ScatterElementsUpdate>(slice_end,
                                                        dim_1d,
                                                        const_one,
                                                        const_zero,
                                                        v12::ScatterElementsUpdate::Reduction::NONE);
        auto new_shape_ = std::make_shared<v8::Slice>(input, slice_start, slice_end2, slice_step, axes);
        auto new_shape = std::make_shared<v3::ShapeOf>(new_shape_, element::i32);
        auto src_ =
            std::make_shared<v3::Broadcast>(src_broadcasted, new_shape, BroadcastType::BIDIRECTIONAL);
        auto src_input_dtype = std::make_shared<v1::ConvertLike>(src_, input);
        // brodcast index to input rank size
        src_rank = std::make_shared<v3::ShapeOf>(new_shape, element::i32);
        auto new_index_shape_ = std::make_shared<v3::Broadcast>(const_one, src_rank);
        auto const_minus_one = v0::Constant::create(element::i32, Shape{1}, {-1});
        auto new_index_shape =
            std::make_shared<v12::ScatterElementsUpdate>(new_index_shape_, dim_1d, const_minus_one, const_zero);
        // precerve indicies location for spicifc dim
        auto reshaped_index = std::make_shared<v1::Reshape>(index, new_index_shape, false);
        auto broadcasted_index =
            std::make_shared<v3::Broadcast>(reshaped_index, new_shape, BroadcastType::BIDIRECTIONAL);
        auto scatter_result =
            std::make_shared<v12::ScatterElementsUpdate>(input,
                                                        broadcasted_index,
                                                        src_,
                                                        dim,
                                                        reduceMode);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(scatter_result)};
        function = std::make_shared<ov::Model>(results, ov::ParameterVector{param, indices_param, update_param}, "index_add");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        auto dataShape = targetInputStaticShapes[0];
        auto updateShape = targetInputStaticShapes[1];
        // The dim-th dimension of update must have the same size as the length of index (which must be a vector)
        auto indicesShape = ov::Shape{updateShape[normalized_axis]};  // 1D

        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            ov::test::utils::InputGenerateData in_data;

            if (i == 0) {  // "data"
                in_data.start_from = 1;
                in_data.range = 1;
                in_data.resolution = 1;
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), dataShape, in_data);
            } else if (i == 1) {  // "indices"
                // All index values are expected to be within bounds [-d, d - 1] along dimension d pointed by axis.
                auto d = dataShape[normalized_axis];
                in_data.start_from = -1.0 * static_cast<int64_t>(d);
                in_data.range = static_cast<uint32_t>(d-1 - in_data.start_from);
                in_data.resolution = 1;
                tensor = shape_size(indicesShape) == 0 ? ov::Tensor(funcInput.get_element_type(), indicesShape) :
                                            ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), indicesShape, in_data);
            } else if (i == 2) {  // "updates"
                in_data.start_from = -50;
                in_data.range = 100;
                in_data.resolution = 1;
                tensor = shape_size(updateShape) == 0 ? ov::Tensor(funcInput.get_element_type(), updateShape) :
                            ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), updateShape, in_data);
            } else {
                OPENVINO_THROW("Unknown input");
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }

private:
    std::tuple<Output<Node>, Output<Node>> get_shape_rank(const Output<Node>& x,
                                                      bool as_scalar = false,
                                                      element::Type output_type = element::i32) {
        auto shape = std::make_shared<opset10::ShapeOf>(x, output_type);
        Output<Node> rank = std::make_shared<opset10::ShapeOf>(shape, output_type);
        if (as_scalar) {
            auto axis_0 = opset10::Constant::create(output_type, Shape{}, {0});
            rank = std::make_shared<opset10::Squeeze>(rank, axis_0);
        }
        return std::make_tuple(shape, rank);
    }

    std::shared_ptr<Node> get_node_axes_range(const Output<Node>& x) {
        auto start = std::make_shared<opset10::Constant>(element::i32, Shape{}, 0);
        auto step = std::make_shared<opset10::Constant>(element::i32, Shape{}, 1);
        Output<Node> reduced_rank;
        std::tie(std::ignore, reduced_rank) = get_shape_rank(x, true);
        return std::make_shared<opset10::Range>(start, reduced_rank, step, element::i32);
    }

    size_t normalized_axis;  // normalized_axis
};

TEST_P(IndexAddTest, CompareWithRefs) {
    run();
}

namespace {
// map<inputShape, map<updatesShape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape {
    {{3}, {{{2}, {0, -1}}, {{3}, {0, -1}}/*, {{0}, {0, -1}}*/}}, // TODO: empty tensor failing in template plugin
    {{4, 6}, {{{3, 6}, {0, -2}}, {{4, 6}, {0, 1, -1}}/*, {{0, 2}, {0, -2}}*/}},  // axis 0
    {{2, 4}, {{{2, 3}, {1, -1}}, {{2, 4}, {0, 1, -1}}/*, {{4, 0}, {1, -1}}*/}},  // axis 1
    {{1, 120}, {{{1, 120}, {0}}}},
    {{32, 120}, {{{16, 120}, {0}}, {{32, 120}, {0}}}},
    {{120, 32}, {{{120, 16}, {1}}, {{120, 32}, {1}}}},
};

inline std::vector<InputShape> partial_shapes_to_test_representation(
    const std::vector<ov::PartialShape>& shapes) {
    std::vector<InputShape> result;
    for (const auto& staticShape : shapes) {
        result.push_back({{staticShape}, {staticShape.get_shape()}});
    }
    return result;
}

std::vector<ov::test::InputsAndAxis> combine_shapes(
    const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& input_shapes) {
    std::vector<ov::test::InputsAndAxis> res_vec;
    for (auto& input_shape : input_shapes) {
        for (auto& item : input_shape.second) {
            for (auto& elt : item.second) {
                res_vec.push_back(ov::test::InputsAndAxis{
                    partial_shapes_to_test_representation({ov::PartialShape(input_shape.first), ov::PartialShape(item.first)}),
                    elt});
            }
        }
    }
    return res_vec;
}

INSTANTIATE_TEST_SUITE_P(smoke_IndexAddTest,
                         IndexAddTest,
                         ::testing::Combine(::testing::ValuesIn(combine_shapes(axesShapeInShape)),
                                            ::testing::Values(v12::ScatterElementsUpdate::Reduction::SUM, v12::ScatterElementsUpdate::Reduction::NONE),
                                            ::testing::Values(ElementType::f32, ElementType::i32,
                                                            //   ElementType::u8, ElementType::i8,     // cannot validate until CVS-136858 addressed
                                                              ElementType::f16, ElementType::bf16), // data precision
                                            ::testing::Values(ElementType::i32, ElementType::i64), // indices precision
                                            ::testing::Values(1.0),              // alpha
                                            ::testing::Values(true, false)),     // dynamic shape test
                         IndexAddTest::getTestCaseName);
} //  namespace

}  // namespace test
}  // namespace ov

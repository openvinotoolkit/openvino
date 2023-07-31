// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/transpose_sinking/ts_strided_slice.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "gtest/gtest.h"
#include "openvino/opsets/opset12.hpp"
#include "openvino/pass/manager.hpp"
#include "ts_test_case.hpp"
#include "ts_test_utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::element;
using namespace ov::opset12;
using namespace ov::pass::transpose_sinking;
using namespace transpose_sinking::testing::utils;

namespace transpose_sinking {
namespace testing {
namespace strided_slice {

struct StridedSliceMasks {
    std::vector<int64_t> begin;
    std::vector<int64_t> end;
    std::vector<int64_t> new_axis;
    std::vector<int64_t> shrink_axis;
    std::vector<int64_t> ellipsis;
};

class StridedSliceFactory : public IFactory {
public:
    explicit StridedSliceFactory(const std::string& type_name) : IFactory(type_name) {}
    NodePtr create(const OutputVector& parent_nodes) const override {
        if (parent_nodes.size() == 4) {
            return make_shared<StridedSlice>(parent_nodes[0],
                                             parent_nodes[1],
                                             parent_nodes[2],
                                             parent_nodes[3],
                                             m_masks.begin,
                                             m_masks.end,
                                             m_masks.new_axis,
                                             m_masks.shrink_axis,
                                             m_masks.ellipsis);
        }
        OPENVINO_ASSERT(false, "Unexpected number of inputs to StridedSlice operation.");
    }

    void set_masks(const StridedSliceMasks& masks) {
        m_masks = masks;
    }
private:
    StridedSliceMasks m_masks;
};

shared_ptr<StridedSliceFactory> CreateStridedSliceFactory(const std::string& type_name) {
    return std::make_shared<StridedSliceFactory>(type_name);
}
// ----------------------------------------------------------------------------

#undef CREATE_STRIDED_SLICE_FACTORY
#define CREATE_STRIDED_SLICE_FACTORY(type_name) CreateStridedSliceFactory(#type_name)
// ----------------------------------------------------------------------------

shared_ptr<ov::Model> create_model(size_t main_node_idx,
                                   const ModelDescription& model_desc,
                                   size_t num_ops,
                                   const OutputVector& inputs_to_main) {
    auto new_inputs = model_desc.preprocess_inputs_to_main.apply(inputs_to_main);
    auto main_node = create_main_node(new_inputs, num_ops, model_desc.main_op[main_node_idx]);
    auto outputs = model_desc.preprocess_outputs_of_main.apply(main_node->outputs());
    return make_shared<ov::Model>(outputs, filter_parameters(inputs_to_main));
}

auto wrapper = [](const TestCase& test_case) {
    OPENVINO_ASSERT(test_case.model.main_op.size() == test_case.model_ref.main_op.size(),
                    "The number of main op (testing op) creator have to be the same for the testing model and for"
                    "the reference model.");
    return ::testing::Combine(::testing::Range<size_t>(0, test_case.num_main_ops.size()),
                              ::testing::Range<size_t>(0, test_case.model.main_op.size()),
                              ::testing::Values(test_case));
};

struct StridedSliceForwardArguments {
    OutputVector inputs_to_main;
    StridedSliceMasks masks;
    StridedSliceMasks ref_masks;
    vector<int64_t> reference_transpose_order;
    vector<int64_t> reference_gather_order;
};

auto test_forward_strided_slice = [](const StridedSliceForwardArguments& test_arguments) {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSStridedSliceForward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = test_arguments.inputs_to_main;

    // Test model description:
    test_case.model.preprocess_inputs_to_main = {{set_transpose_for}, {{0}}};
    auto strided_slice_factory = CREATE_STRIDED_SLICE_FACTORY(StridedSlice);
    strided_slice_factory->set_masks(test_arguments.masks);
    test_case.model.main_op = {strided_slice_factory};
    test_case.model.model_template = create_model;

    // Reference model description
    const auto& ref_transpose_order = test_arguments.reference_transpose_order;
    const auto& ref_gather_order = test_arguments.reference_gather_order;
    auto new_transpose = [ref_transpose_order](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec(out_vec.size());
        auto order = make_shared<Constant>(element::i32,
                                           Shape{ref_transpose_order.size()},
                                           ref_transpose_order);
        new_out_vec[0] = make_shared<Transpose>(out_vec[0], order);
        return new_out_vec;
    };

    auto new_axes_cnt = 0;
    for (const auto& i : test_arguments.masks.new_axis) {
        if (i != 0) {
            new_axes_cnt++;
        }
    }


    auto update_gather_inputs = [ref_gather_order, new_axes_cnt](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec = out_vec;
        auto axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
        size_t expected_size = out_vec[0].get_partial_shape().rank().get_length() + new_axes_cnt;
        for (auto idx : idxs) {
            auto input_const = ov::as_type_ptr<ov::op::v0::Constant>(out_vec[idx].get_node_shared_ptr());
            EXPECT_NE(input_const, nullptr) << "Only constant begin, end, strided inputs are "
                                               "supported in TSStridedSlice transformation.";

            auto input_const_val = input_const->cast_vector<int64_t>();
            if (idx == 1) {
                // `begin` input have to be initialized with 0
                input_const_val.resize(expected_size, 0);
            } else if (idx == 2) {
                // 'end' input have to be initialized with the corresponding `data` input dim value
                input_const_val.resize(expected_size, std::numeric_limits<int32_t>::max());
            } else {
                // `stride` input have to be initialized with 1
                input_const_val.resize(expected_size, 1);
            }
            auto new_input = ov::op::v0::Constant::create(input_const->get_element_type(), {input_const_val.size()}, input_const_val);

            auto indices = std::make_shared<ov::op::v0::Constant>(element::i32,
                                                                  Shape{ref_gather_order.size()},
                                                                  ref_gather_order);
            new_out_vec[idx] = std::make_shared<ov::op::v8::Gather>(new_input, indices, axis);
        }
        return new_out_vec;
    };

    test_case.model_ref.preprocess_inputs_to_main = {{update_gather_inputs}, {{1, 2, 3}}};
    auto strided_slice_factory_ref = CREATE_STRIDED_SLICE_FACTORY(StridedSlice);
    strided_slice_factory_ref->set_masks(test_arguments.ref_masks);
    test_case.model_ref.main_op = {strided_slice_factory_ref};
    test_case.model_ref.preprocess_outputs_of_main = {{new_transpose}, {{0}}};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

auto fw_test_1 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 10}), // data
            constant<int>(i32, {2}, {1, 2}), // begin
            constant<int>(i32, {2}, {7, 6}), // end
            constant<int>(i32, {2}, {1, 2})  // stride
    };
    // empty masks
    args.masks.begin = {0, 0};
    args.masks.end = {0, 0};
    args.masks.new_axis = {0, 0};
    args.masks.shrink_axis = {0, 0};
    args.masks.ellipsis = {0, 0};

    // reference:
    args.reference_transpose_order = {1, 0};
    args.reference_gather_order = {1, 0};

    // ref masks
    args.ref_masks.begin = {0, 0};
    args.ref_masks.end = {0, 0};
    args.ref_masks.new_axis = {0, 0};
    args.ref_masks.shrink_axis = {0, 0};
    args.ref_masks.ellipsis = {0, 0};

    return args;
};

auto fw_test_2 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 10}), // data
            constant<int>(i32, {2}, {1, 2}), // begin
            constant<int>(i32, {2}, {7, 6}), // end
            constant<int>(i32, {2}, {1, 2})  // stride
    };
    // begin and end masks
    args.masks.begin = {1, 0};
    args.masks.end = {0, 1};
    args.masks.new_axis = {0, 0};
    args.masks.shrink_axis = {0, 0};
    args.masks.ellipsis = {0, 0};

    // reference:
    args.reference_transpose_order = {1, 0};
    args.reference_gather_order = {1, 0};

    // ref masks
    args.ref_masks.begin = {0, 1};
    args.ref_masks.end = {1, 0};
    args.ref_masks.new_axis = {0, 0};
    args.ref_masks.shrink_axis = {0, 0};
    args.ref_masks.ellipsis = {0, 0};
    return args;
};

auto fw_test_3 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 10}), // data
            constant<int>(i32, {2}, {1, 2}), // begin
            constant<int>(i32, {2}, {7, 6}), // end
            constant<int>(i32, {2}, {1, 2})  // stride
    };
    // new axis mask
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {1, 1, 0 ,0};
    args.masks.shrink_axis = {0, 0, 0, 0};
    args.masks.ellipsis = {0, 0, 0, 0};

    // reference:
    args.reference_transpose_order = {0, 1, 3, 2};
    args.reference_gather_order = {0, 1, 3, 2};

    // ref masks
    args.ref_masks.begin = {0, 0, 0, 0};
    args.ref_masks.end = {0, 0, 0, 0};
    args.ref_masks.new_axis = {1, 1, 0, 0};
    args.ref_masks.shrink_axis = {0, 0, 0, 0};
    args.ref_masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto fw_test_4 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 10}), // data
            constant<int>(i32, {2}, {1, 2}), // begin
            constant<int>(i32, {2}, {7, 6}), // end
            constant<int>(i32, {2}, {1, 2})  // stride
    };
    // shrink mask
    args.masks.begin = {0, 0};
    args.masks.end = {0, 0};
    args.masks.new_axis = {0, 0};
    args.masks.shrink_axis = {0, 1};
    args.masks.ellipsis = {0, 0};

    // reference:
    args.reference_transpose_order = {0};
    args.reference_gather_order = {1, 0};

    // ref masks
    args.ref_masks.begin = {0, 0};
    args.ref_masks.end = {0, 0};
    args.ref_masks.new_axis = {0, 0};
    args.ref_masks.shrink_axis = {1, 0};
    args.ref_masks.ellipsis = {0, 0};
    return args;
};

auto fw_test_5 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 10}), // data
            constant<int>(i32, {4}, {1, 2, 2, 1}), // begin
            constant<int>(i32, {4}, {5, 4, 4, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    // 4dims input, shrink mask
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {0, 0, 0, 0};
    args.masks.shrink_axis = {0, 1, 0, 1};
    args.masks.ellipsis = {0, 0, 0, 0};

    // reference:
    args.reference_transpose_order = {1, 0};
    args.reference_gather_order = {3, 2, 1, 0};

    // ref masks
    args.ref_masks.begin = {0, 0, 0, 0};
    args.ref_masks.end = {0, 0, 0, 0};
    args.ref_masks.new_axis = {0, 0, 0, 0};
    args.ref_masks.shrink_axis = {1, 0, 1, 0};
    args.ref_masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto fw_test_6 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 10}), // data
            constant<int>(i32, {4}, {1, 0, 2, 0}), // begin
            constant<int>(i32, {4}, {7, 1, 6, 1}), // end
            constant<int>(i32, {4}, {1, 1, 2, 1})  // stride
    };

    // mixed masks: new_axis and shrink_mask
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {0, 1, 0, 1};
    // the shrink mask will be ignored because new_axis and shrink_axis are at the same positions
    args.masks.shrink_axis = {0, 1, 0, 1};
    args.masks.ellipsis = {0, 0, 0, 0};

    // reference:
    args.reference_transpose_order = {2, 1, 0, 3};
    args.reference_gather_order = {2, 1, 0, 3};

    // ref masks
    args.ref_masks.begin = {0, 0, 0, 0};
    args.ref_masks.end = {0, 0, 0, 0};
    args.ref_masks.new_axis = {0, 1, 0, 1};
    args.ref_masks.shrink_axis = {0, 0, 0, 0};
    args.ref_masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto fw_test_7 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 10}), // data
            constant<int>(i32, {4}, {1, 0, 2, 0}), // begin
            constant<int>(i32, {4}, {7, 1, 6, 1}), // end
            constant<int>(i32, {4}, {1, 1, 2, 1})  // stride
    };

    // mixed masks: new_axis and shrink_mask
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {0, 1, 0, 1};
    // the shrink mask values will be ignored for dims which are at the same positions as in new_axis mask
    args.masks.shrink_axis = {1, 1, 0, 1};
    args.masks.ellipsis = {0, 0, 0, 0};

    // reference:
    args.reference_transpose_order = {1, 0, 2};
    args.reference_gather_order = {2, 1, 0, 3};

    // ref masks
    args.ref_masks.begin = {0, 0, 0, 0};
    args.ref_masks.end = {0, 0, 0, 0};
    args.ref_masks.new_axis = {0, 1, 0, 1};
    args.ref_masks.shrink_axis = {0, 0, 1, 0};
    args.ref_masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto fw_test_8 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 10}), // data
            constant<int>(i32, {6}, {0, 0, 1, 2, 2, 1}), // begin
            constant<int>(i32, {6}, {1, 1, 5, 4, 4, 4}), // end
            constant<int>(i32, {6}, {1, 1, 1, 2, 1, 2})  // stride
    };
    // mixed masks: begin, end, shrink, new_axis
    args.masks.begin = {0, 0, 0, 1, 0, 0};
    args.masks.end = {0, 0, 0, 1, 0, 1};
    args.masks.new_axis = {1, 1, 0, 0, 0, 0};
    args.masks.shrink_axis = {0, 1, 0, 1, 0, 0};
    args.masks.ellipsis = {0, 0, 0, 0, 0, 0};

    // reference:
    args.reference_transpose_order = {0, 1, 4, 3, 2};
    args.reference_gather_order = {0, 1, 5, 4, 3, 2};

    // ref masks
    args.ref_masks.begin = {0, 0, 0, 0, 1, 0};
    args.ref_masks.end = {0, 0, 1, 0, 1, 0};
    args.ref_masks.new_axis = {1, 1, 0, 0, 0, 0};
    args.ref_masks.shrink_axis = {0, 0, 0, 0, 1, 0};
    args.ref_masks.ellipsis = {0, 0, 0, 0, 0, 0};
    return args;
};

auto fw_test_9 = []() {
    StridedSliceForwardArguments args;
    args.inputs_to_main = {
            parameter(f32, {7, 4, 5, 10}), // data
            constant<int>(i32, {4}, {1, 2, 2, 1}), // begin
            constant<int>(i32, {4}, {5, 4, 4, 4}), // end
            constant<int>(i32, {4}, {1, 2, 1, 2})  // stride
    };
    // mixed masks: shrink, new_axis
    args.masks.begin = {0, 0, 0, 0};
    args.masks.end = {0, 0, 0, 0};
    args.masks.new_axis = {0, 1, 0, 1};
    args.masks.shrink_axis = {1, 0, 1, 0};
    args.masks.ellipsis = {0, 0, 0, 0};

    // reference:
    args.reference_transpose_order = {1, 3, 2, 0};
    args.reference_gather_order = {5, 1, 4, 3, 2, 0};

    // ref masks
    args.ref_masks.begin = {0, 0, 0, 0, 0, 0};
    args.ref_masks.end = {0, 0, 0, 0, 0, 0};
    args.ref_masks.new_axis = {0, 1, 0, 1, 0, 0};
    args.ref_masks.shrink_axis = {0, 0, 0, 0, 1, 1};
    args.ref_masks.ellipsis = {0, 0, 0, 0, 0, 0};
    return args;
};

INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_1, TSTestFixture, test_forward_strided_slice(fw_test_1()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_2, TSTestFixture, test_forward_strided_slice(fw_test_2()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_3, TSTestFixture, test_forward_strided_slice(fw_test_3()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_4, TSTestFixture, test_forward_strided_slice(fw_test_4()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_5, TSTestFixture, test_forward_strided_slice(fw_test_5()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_6, TSTestFixture, test_forward_strided_slice(fw_test_6()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_7, TSTestFixture, test_forward_strided_slice(fw_test_7()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_8, TSTestFixture, test_forward_strided_slice(fw_test_8()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceForward_9, TSTestFixture, test_forward_strided_slice(fw_test_9()));

auto test_backward_strided_slice = [](const StridedSliceForwardArguments& test_arguments) {
    TestCase test_case;

    // Initialize common attributes
    test_case.transformation = CREATE_PASS_FACTORY(TSStridedSliceBackward);
    test_case.num_main_ops = {1};
    test_case.inputs_to_main = test_arguments.inputs_to_main;

    // Test model description:
    auto strided_slice_factory = CREATE_STRIDED_SLICE_FACTORY(StridedSlice);
    strided_slice_factory->set_masks(test_arguments.masks);
    test_case.model.main_op = {strided_slice_factory};
    test_case.model.preprocess_outputs_of_main = {{set_transpose_for}, {{0}}};
    test_case.model.model_template = create_model;

    // Reference model description
    const auto& ref_transpose_order = test_arguments.reference_transpose_order;
    const auto& ref_gather_order = test_arguments.reference_gather_order;
    auto new_transpose = [ref_transpose_order](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec = out_vec;
        auto order = make_shared<Constant>(element::i32,
                                           Shape{ref_transpose_order.size()},
                                           ref_transpose_order);
        new_out_vec[0] = make_shared<Transpose>(out_vec[0], order);
        return new_out_vec;
    };

    auto new_axes_cnt = 0;
    for (const auto& i : test_arguments.masks.new_axis) {
        if (i != 0) {
            new_axes_cnt++;
        }
    }

    auto update_gather_inputs = [ref_gather_order, new_axes_cnt](const vector<size_t>& idxs, const OutputVector& out_vec) -> OutputVector {
        OutputVector new_out_vec = out_vec;
        auto axis = std::make_shared<ov::op::v0::Constant>(element::i32, Shape{}, 0);
        size_t expected_size = out_vec[0].get_partial_shape().rank().get_length() + new_axes_cnt;
        for (auto idx : idxs) {
            auto input_const = ov::as_type_ptr<ov::op::v0::Constant>(out_vec[idx].get_node_shared_ptr());
            EXPECT_NE(input_const, nullptr) << "Only constant begin, end, strided inputs are "
                                               "supported in TSStridedSlice transformation.";

            auto input_const_val = input_const->cast_vector<int64_t>();
            if (idx == 1) {
                // `begin` input have to be initialized with 0
                input_const_val.resize(expected_size, 0);
            } else if (idx == 2) {
                // 'end' input have to be initialized with the corresponding `data` input dim value
                input_const_val.resize(expected_size, std::numeric_limits<int32_t>::max());
            } else {
                // `stride` input have to be initialized with 1
                input_const_val.resize(expected_size, 1);
            }
            auto new_input = ov::op::v0::Constant::create(input_const->get_element_type(), {input_const_val.size()}, input_const_val);

            auto indices = std::make_shared<ov::op::v0::Constant>(element::i32,
                                                                  Shape{ref_gather_order.size()},
                                                                  ref_gather_order);
            new_out_vec[idx] = std::make_shared<ov::op::v8::Gather>(new_input, indices, axis);
        }
        return new_out_vec;
    };

    test_case.model_ref.preprocess_inputs_to_main = {{new_transpose, update_gather_inputs}, {{0}, {1, 2, 3}}};
    auto strided_slice_factory_ref = CREATE_STRIDED_SLICE_FACTORY(StridedSlice);
    strided_slice_factory_ref->set_masks(test_arguments.ref_masks);
    test_case.model_ref.main_op = {strided_slice_factory_ref};
    test_case.model_ref.model_template = create_model;

    return wrapper(test_case);
};

auto bw_test_1 = []() {
    auto args = fw_test_1();

    // reference:
    args.reference_transpose_order = {1, 0};
    args.reference_gather_order = {1, 0};

    // ref masks
    args.ref_masks.begin = {0, 0};
    args.ref_masks.end = {0, 0};
    args.ref_masks.new_axis = {0, 0};
    args.ref_masks.shrink_axis = {0, 0};
    args.ref_masks.ellipsis = {0, 0};

    return args;
};

auto bw_test_2 = []() {
    auto args = fw_test_2();

    // reference:
    args.reference_transpose_order = {1, 0};
    args.reference_gather_order = {1, 0};

    // ref masks
    args.ref_masks.begin = {0, 1};
    args.ref_masks.end = {1, 0};
    args.ref_masks.new_axis = {0, 0};
    args.ref_masks.shrink_axis = {0, 0};
    args.ref_masks.ellipsis = {0, 0};
    return args;
};

auto bw_test_3 = []() {
    auto args = fw_test_3();

    // reference:
    args.reference_transpose_order = {1, 0};
    args.reference_gather_order = {3, 2, 1, 0};

    // ref masks
    args.ref_masks.begin = {0, 0, 0, 0};
    args.ref_masks.end = {0, 0, 0, 0};
    args.ref_masks.new_axis = {0, 0, 1, 1};
    args.ref_masks.shrink_axis = {0, 0, 0, 0};
    args.ref_masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto bw_test_4 = []() {
    auto args = fw_test_4();

    // reference:
    args.reference_transpose_order = {0, 1};
    args.reference_gather_order = {0, 1};

    // ref masks
    args.ref_masks.begin = {0, 0};
    args.ref_masks.end = {0, 0};
    args.ref_masks.new_axis = {0, 0};
    args.ref_masks.shrink_axis = {0, 1};
    args.ref_masks.ellipsis = {0, 0};
    return args;
};

auto bw_test_5 = []() {
    auto args = fw_test_5();

    // reference:
    args.reference_transpose_order = {2, 1, 0, 3};
    args.reference_gather_order = {2, 1, 0, 3};

    // ref masks
    args.ref_masks.begin = {0, 0, 0, 0};
    args.ref_masks.end = {0, 0, 0, 0};
    args.ref_masks.new_axis = {0, 0, 0, 0};
    args.ref_masks.shrink_axis = {0, 1, 0, 1};
    args.ref_masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto bw_test_6 = []() {
    auto args = fw_test_6();

    // reference:
    args.reference_transpose_order = {1, 0};
    args.reference_gather_order = {3, 2, 1, 0};

    // ref masks
    args.ref_masks.begin = {0, 0, 0, 0};
    args.ref_masks.end = {0, 0, 0, 0};
    args.ref_masks.new_axis = {1, 0, 1, 0};
    args.ref_masks.shrink_axis = {0, 0, 0, 0};
    args.ref_masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto bw_test_7 = []() {
    auto args = fw_test_7();

    // reference:
    args.reference_transpose_order = {0, 1};
    args.reference_gather_order = {0, 3, 2, 1};

    // ref masks
    args.ref_masks.begin = {0, 0, 0, 0};
    args.ref_masks.end = {0, 0, 0, 0};
    args.ref_masks.new_axis = {0, 1, 0, 1};
    args.ref_masks.shrink_axis = {1, 0, 0, 0};
    args.ref_masks.ellipsis = {0, 0, 0, 0};
    return args;
};

auto bw_test_8 = []() {
    auto args = fw_test_8();

    // reference:
    args.reference_transpose_order = {2, 3, 1, 0};
    args.reference_gather_order = {5, 4, 2, 3, 1, 0};

    // ref masks
    args.ref_masks.begin = {0, 0, 0, 1, 0, 0};
    args.ref_masks.end = {1, 0, 0, 1, 0, 0};
    args.ref_masks.new_axis = {0, 0, 0, 0, 1, 1};
    args.ref_masks.shrink_axis = {0, 0, 0, 1, 0, 0};
    args.ref_masks.ellipsis = {0, 0, 0, 0, 0, 0};
    return args;
};

auto bw_test_9 = []() {
    auto args = fw_test_9();

    // reference:
    args.reference_transpose_order = {0, 2, 3, 1};
    args.reference_gather_order = {0, 5, 2, 4, 3, 1};

    // ref masks
    args.ref_masks.begin = {0, 0, 0, 0, 0, 0};
    args.ref_masks.end = {0, 0, 0, 0, 0, 0};
    args.ref_masks.new_axis = {0, 0, 0, 0, 1, 1};
    args.ref_masks.shrink_axis = {1, 0, 1, 0, 0, 0};
    args.ref_masks.ellipsis = {0, 0, 0, 0, 0, 0};
    return args;
};

INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceBackward_1, TSTestFixture, test_backward_strided_slice(bw_test_1()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceBackward_2, TSTestFixture, test_backward_strided_slice(bw_test_2()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceBackward_3, TSTestFixture, test_backward_strided_slice(bw_test_3()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceBackward_4, TSTestFixture, test_backward_strided_slice(bw_test_4()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceBackward_5, TSTestFixture, test_backward_strided_slice(bw_test_5()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceBackward_6, TSTestFixture, test_backward_strided_slice(bw_test_6()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceBackward_7, TSTestFixture, test_backward_strided_slice(bw_test_7()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceBackward_8, TSTestFixture, test_backward_strided_slice(bw_test_8()));
INSTANTIATE_TEST_SUITE_P(TSCommonStridedSliceBackward_9, TSTestFixture, test_backward_strided_slice(bw_test_9()));

}  // namespace gather
}  // namespace testing
}  // namespace transpose_sinking

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_models.hpp"

#include <memory>
#include <vector>

#include "openvino/opsets/opset10.hpp"

using namespace ov;
using namespace ov::opset10;
using namespace ov::element;
using std::make_shared;
using std::vector;

std::shared_ptr<Model> eltwise_add_model() {
    auto precision = Type_t::f32;
    auto input_shape = Shape({1, 10});
    vector<float> weights(10, 2.5f);

    auto input = make_shared<Parameter>(precision, input_shape);

    auto eltwise_add_constant = make_shared<Constant>(precision, input_shape, weights);
    auto eltwise_add = make_shared<Add>(input->output(0), eltwise_add_constant->output(0));

    auto function = make_shared<Model>(eltwise_add, ParameterVector({input}), "EltwiseAddModel");
    return function;
}

std::shared_ptr<Model> fc_with_padding_after_split_model() {
    auto precision = Type_t::f32;

    // input
    auto input = make_shared<Parameter>(precision, Shape{1, 20});

    // split
    auto split_axis = make_shared<Constant>(Type_t::i32, Shape{}, vector<int32_t>{1});
    auto split = make_shared<Split>(input->output(0), split_axis->output(0), 2);

    // fully connected
    auto fc_weights = make_shared<Constant>(precision, Shape{10, 10}, vector<float>(100, 1.0f));
    auto fc_biases = make_shared<Constant>(precision, Shape{1, 10}, vector<float>(10, 1.0f));
    auto fc_before_biases = make_shared<MatMul>(split->output(1), fc_weights->output(0));
    auto fc = make_shared<Add>(fc_before_biases->output(0), fc_biases->output(0));

    // eltwise add
    auto eltwise = make_shared<Add>(split->output(0), fc->output(0));
    auto function = make_shared<Model>(eltwise, ParameterVector({input}), "FCWithPaddingAfterSplitModel");
    return function;
}

std::shared_ptr<ov::Model> slice_model_with_aligned_outputs() {
    auto precision = Type_t::f32;

    // input
    auto input = make_shared<Parameter>(precision, Shape{1, 20});

    // slice to fc
    auto slice_start = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{0});
    auto slice_stop = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{16});
    auto slice_step = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{1});
    auto slice_axis = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{1});
    auto slice = make_shared<Slice>(input->output(0),
                                    slice_start->output(0),
                                    slice_stop->output(0),
                                    slice_step->output(0),
                                    slice_axis->output(0));

    // fully connected
    auto fc_weights = make_shared<Constant>(precision, Shape{16, 4}, vector<float>(64, 1.0f));
    auto fc_biases = make_shared<Constant>(precision, Shape{1, 4}, vector<float>(4, 1.0f));
    auto fc_before_biases = make_shared<MatMul>(slice->output(0), fc_weights->output(0));
    auto fc = make_shared<Add>(fc_before_biases->output(0), fc_biases->output(0));

    // slice to eltwise
    auto slice_to_eltwise_start = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{16});
    auto slice_to_eltwise_stop = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{20});
    auto slice_to_eltwise_step = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{1});
    auto slice_to_eltwise_axis = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{1});
    auto slice_to_eltwise = make_shared<Slice>(input->output(0),
                                               slice_to_eltwise_start->output(0),
                                               slice_to_eltwise_stop->output(0),
                                               slice_to_eltwise_step->output(0),
                                               slice_to_eltwise_axis->output(0));

    // eltwise add
    auto eltwise = make_shared<Add>(slice_to_eltwise->output(0), fc->output(0));
    auto function = make_shared<Model>(eltwise, ParameterVector({input}), "SliceModelWithAlignedOutputs");
    return function;
}

std::shared_ptr<ov::Model> two_fc_with_padding_after_slice_model() {
    auto precision = Type_t::f32;

    // input
    auto input = make_shared<Parameter>(precision, Shape{1, 20});

    // slice to eltwise
    auto slice_to_eltwise_start = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{0});
    auto slice_to_eltwise_stop = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{8});
    auto slice_to_eltwise_step = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{1});
    auto slice_to_eltwise_axis = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{1});
    auto slice_to_eltwise = make_shared<Slice>(input->output(0),
                                               slice_to_eltwise_start->output(0),
                                               slice_to_eltwise_stop->output(0),
                                               slice_to_eltwise_step->output(0),
                                               slice_to_eltwise_axis->output(0));

    // slice to fc
    auto slice_start = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{8});
    auto slice_stop = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{20});
    auto slice_step = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{1});
    auto slice_axis = make_shared<Constant>(Type_t::i32, Shape{1}, vector<int32_t>{1});
    auto slice = make_shared<Slice>(input->output(0),
                                    slice_start->output(0),
                                    slice_stop->output(0),
                                    slice_step->output(0),
                                    slice_axis->output(0));

    // fully connected 1
    auto fc_weights = make_shared<Constant>(precision, Shape{12, 8}, vector<float>(96, 1.0f));
    auto fc_biases = make_shared<Constant>(precision, Shape{1, 8}, vector<float>(8, 1.0f));
    auto fc_before_biases = make_shared<MatMul>(slice->output(0), fc_weights->output(0));
    auto fc = make_shared<Add>(fc_before_biases->output(0), fc_biases->output(0));

    // eltwise add 1
    auto eltwise = make_shared<Add>(slice_to_eltwise->output(0), fc->output(0));

    // fully connected 2
    auto fc_weights_2 = make_shared<Constant>(precision, Shape{12, 8}, vector<float>(96, 1.0f));
    auto fc_biases_2 = make_shared<Constant>(precision, Shape{1, 8}, vector<float>(8, 1.0f));
    auto fc_before_biases_2 = make_shared<MatMul>(slice->output(0), fc_weights_2->output(0));
    auto fc_2 = make_shared<Add>(fc_before_biases_2->output(0), fc_biases_2->output(0));

    // eltwise add 2
    auto eltwise_2 = make_shared<Add>(eltwise->output(0), fc_2->output(0));

    auto function = make_shared<Model>(eltwise_2, ParameterVector({input}), "twoFCWithPaddingAfterSliceModel");
    return function;
}

std::shared_ptr<ov::Model> scaleshift_3d_model() {
    auto precision = Type_t::f32;
    vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                             7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                             5.0f, 6.0f, 7.0f, 8.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

    // input
    auto input = make_shared<Parameter>(precision, Shape{1, 40});

    // reshape
    auto shape_pattern = make_shared<Constant>(Type_t::i32, Shape{3}, vector<int32_t>{1, 5, 8});
    auto reshape = make_shared<Reshape>(input->output(0), shape_pattern->output(0), false);

    // scaleshift
    auto scaleshift_mul_constant = make_shared<Constant>(precision, Shape{1, 5, 8}, weights);
    auto scaleshift_mul = make_shared<Multiply>(reshape->output(0), scaleshift_mul_constant->output(0));
    auto scaleshift_add_constant = make_shared<Constant>(precision, Shape{1, 5, 8}, weights);
    auto scaleshift_add = make_shared<Add>(scaleshift_mul->output(0), scaleshift_add_constant->output(0));

    auto function = make_shared<Model>(scaleshift_add, ParameterVector{input}, "ScaleShift3dModel");
    return function;
}

std::shared_ptr<ov::Model> input_split_concat_model() {
    auto precision = Type_t::f32;
    auto input_shape = Shape{1, 64};

    // input
    auto input = make_shared<Parameter>(precision, input_shape);

    // scaleshift
    auto scaleshift_mul_constant = make_shared<Constant>(precision, input_shape, vector<float>(64, 1));
    auto scaleshift_mul = make_shared<Multiply>(input->output(0), scaleshift_mul_constant->output(0));
    auto scaleshift_add_constant = make_shared<Constant>(precision, input_shape, vector<float>(64, 1));
    auto scaleshift_add = make_shared<Add>(scaleshift_mul->output(0), scaleshift_add_constant->output(0));

    // split
    auto split_axis = make_shared<Constant>(Type_t::i32, Shape{}, vector<int32_t>{1});
    auto split = make_shared<Split>(scaleshift_add->output(0), split_axis->output(0), 2);

    // concat
    auto concat = make_shared<Concat>(OutputVector{split->output(0), split->output(1)}, 1);

    // fully connected
    auto fc_weights = make_shared<Constant>(precision, Shape{64, 10}, vector<float>(640, 1.0f));
    auto fc_biases = make_shared<Constant>(precision, Shape{1, 10}, vector<float>(10, 1.0f));
    auto fc_before_biases = make_shared<MatMul>(concat->output(0), fc_weights->output(0));
    auto fc = make_shared<Add>(fc_before_biases->output(0), fc_biases->output(0));

    auto function = make_shared<Model>(fc, ParameterVector{input}, "InputSplitConcatModel");
    return function;
}

std::shared_ptr<ov::Model> input_split_concat_unaligned_model() {
    auto precision = Type_t::f32;
    auto input_shape = Shape({1, 20});

    // input
    auto input = make_shared<Parameter>(precision, input_shape);

    // scaleshift
    auto scaleshift_mul_constant = make_shared<Constant>(precision, input_shape, vector<float>(20, 1));
    auto scaleshift_mul = make_shared<Multiply>(input->output(0), scaleshift_mul_constant->output(0));
    auto scaleshift_add_constant = make_shared<Constant>(precision, input_shape, vector<float>(20, 1));
    auto scaleshift_add = make_shared<Add>(scaleshift_mul->output(0), scaleshift_add_constant->output(0));

    // split
    auto split_axis = make_shared<Constant>(Type_t::i32, Shape{}, vector<int32_t>{1});
    auto split = make_shared<Split>(scaleshift_add->output(0), split_axis->output(0), 2);

    // concat
    auto concat = make_shared<Concat>(OutputVector{split->output(0), split->output(1)}, 1);

    // fully connected
    auto fc_weights = make_shared<Constant>(precision, Shape{20, 10}, vector<float>(200, 1.0f));
    auto fc_biases = make_shared<Constant>(precision, Shape{1, 10}, vector<float>(10, 1.0f));
    auto fc_before_biases = make_shared<MatMul>(concat->output(0), fc_weights->output(0));
    auto fc = make_shared<Add>(fc_before_biases->output(0), fc_biases->output(0));

    auto function = make_shared<Model>(fc, ParameterVector{input}, "InputSplitConcatModel");
    return function;
}

std::shared_ptr<ov::Model> power_with_scale_factor_model() {
    auto precision = Type_t::f32;

    // input
    auto input = make_shared<Parameter>(precision, Shape{1, 10});

    // power
    auto exponents = make_shared<Constant>(precision, Shape{1, 10}, vector<float>(10, 1.0f));
    auto power = make_shared<Power>(input->output(0), exponents->output(0));

    // fully connected
    auto fc_weights = make_shared<Constant>(precision, Shape{10, 12}, vector<float>(120, 1.0f));
    auto fc_biases = make_shared<Constant>(precision, Shape{1, 12}, vector<float>(12, 1.0f));
    auto fc_before_biases = make_shared<MatMul>(power->output(0), fc_weights->output(0));
    auto fc = make_shared<Add>(fc_before_biases->output(0), fc_biases->output(0));

    auto function = make_shared<Model>(fc, ParameterVector{input}, "PowerWithScaleFactor1");
    return function;
}

std::shared_ptr<Model> lstm_cell_only_model() {
    auto precision = Type_t::f32;

    // input
    auto input = make_shared<Parameter>(precision, Shape{1, 96});

    // scaleshift
    auto scaleshift_mul_constant = make_shared<Constant>(precision, Shape{1, 96}, vector<float>(96, 0.10f));
    auto scaleshift_mul = make_shared<Multiply>(input->output(0), scaleshift_mul_constant->output(0));
    auto scaleshift_add_constant = make_shared<Constant>(precision, Shape{1, 96}, vector<float>(96, 0.10f));
    auto scaleshift_add = make_shared<Add>(scaleshift_mul->output(0), scaleshift_add_constant->output(0));

    // split
    auto split_axis = make_shared<Constant>(Type_t::i32, Shape{}, vector<int32_t>{1});
    auto split = make_shared<Split>(scaleshift_add->output(0), split_axis->output(0), 3);

    // LSTM cell
    auto lstm_blob_w = make_shared<Constant>(precision, Shape({128, 32}), vector<float>(4096, 0.1f));
    auto lstm_blob_r = make_shared<Constant>(precision, Shape({128, 32}), vector<float>(4096, 0.1f));
    auto lstm_blob_b = make_shared<Constant>(precision, Shape({128}), vector<float>(128, 0.1f));
    auto lstm_cell = make_shared<LSTMCell>(split->output(0),
                                           split->output(1),
                                           split->output(2),
                                           lstm_blob_w->output(0),
                                           lstm_blob_r->output(0),
                                           lstm_blob_b->output(0),
                                           32);

    // eltwise add
    auto eltwise = make_shared<Add>(lstm_cell->output(0), lstm_cell->output(1));
    auto function = make_shared<Model>(eltwise, ParameterVector({input}), "LSTMCellOnlyModel");
    return function;
}

std::shared_ptr<Model> lstm_cell_only_model_unaligned() {
    auto precision = Type_t::f32;

    // input
    auto input = make_shared<Parameter>(precision, Shape{1, 30});

    // scaleshift
    auto scaleshift_mul_constant = make_shared<Constant>(precision, Shape{1, 30}, vector<float>(30, 0.10f));
    auto scaleshift_mul = make_shared<Multiply>(input->output(0), scaleshift_mul_constant->output(0));
    auto scaleshift_add_constant = make_shared<Constant>(precision, Shape{1, 30}, vector<float>(30, 0.10f));
    auto scaleshift_add = make_shared<Add>(scaleshift_mul->output(0), scaleshift_add_constant->output(0));

    // split
    auto split_axis = make_shared<Constant>(Type_t::i32, Shape{}, vector<int32_t>{1});
    auto split = make_shared<Split>(scaleshift_add->output(0), split_axis->output(0), 3);

    // LSTM cell
    auto lstm_blob_w = make_shared<Constant>(precision, Shape({40, 10}), vector<float>(400, 0.1f));
    auto lstm_blob_r = make_shared<Constant>(precision, Shape({40, 10}), vector<float>(400, 0.1f));
    auto lstm_blob_b = make_shared<Constant>(precision, Shape({40}), vector<float>(40, 0.1f));
    auto lstm_cell = make_shared<LSTMCell>(split->output(0),
                                           split->output(1),
                                           split->output(2),
                                           lstm_blob_w->output(0),
                                           lstm_blob_r->output(0),
                                           lstm_blob_b->output(0),
                                           10);

    // eltwise add
    auto eltwise = make_shared<Add>(lstm_cell->output(0), lstm_cell->output(1));
    auto function = make_shared<Model>(eltwise, ParameterVector({input}), "LSTMCellOnlyModel");
    return function;
}

std::shared_ptr<ov::Model> two_inputs_to_affine_model() {
    auto precision = Type_t::f32;

    // input
    auto input_1 = make_shared<Parameter>(precision, Shape{1, 10});
    auto input_2 = make_shared<Parameter>(precision, Shape{1, 10});

    // fully connected 1
    auto fc_weights_1 = make_shared<Constant>(precision, Shape{10, 10}, vector<float>(100, 1.0f));
    auto fc_biases_1 = make_shared<Constant>(precision, Shape{1, 10}, vector<float>(10, 1.0f));
    auto fc_before_biases_1 = make_shared<MatMul>(input_1->output(0), fc_weights_1->output(0));
    auto fc_1 = make_shared<Add>(fc_before_biases_1->output(0), fc_biases_1->output(0));

    // fully connected 2
    auto fc_weights_2 = make_shared<Constant>(precision, Shape{10, 10}, vector<float>(100, 1.0f));
    auto fc_biases_2 = make_shared<Constant>(precision, Shape{1, 10}, vector<float>(10, 1.0f));
    auto fc_before_biases_2 = make_shared<MatMul>(input_2->output(0), fc_weights_2->output(0));
    auto fc_2 = make_shared<Add>(fc_before_biases_2->output(0), fc_biases_2->output(0));

    // eltwise add
    auto add = make_shared<Add>(fc_1->output(0), fc_2->output(0));

    auto function = make_shared<Model>(add, ParameterVector({input_1, input_2}), "two_inputs_to_affine");
    return function;
}

std::shared_ptr<ov::Model> reshape_convolution_less_than_48_filters() {
    auto precision = Type_t::f32;

    // input
    auto input = make_shared<Parameter>(precision, Shape{1, 800});

    // reshape 1
    auto shape_pattern_1 = make_shared<Constant>(Type_t::i32, Shape{4}, vector<int32_t>{1, 4, 1, 200});
    auto reshape_1 = make_shared<Reshape>(input->output(0), shape_pattern_1->output(0), false);

    // convolution
    Strides strides = Strides{1, 2};
    Strides dilations = Strides{1, 1};
    CoordinateDiff pads_begin = CoordinateDiff{0, 0};
    CoordinateDiff pads_end = CoordinateDiff{0, 0};
    auto kernel = make_shared<Constant>(precision, Shape{16, 4, 1, 2}, vector<float>(128, 1.0f));
    auto convolution = make_shared<Convolution>(reshape_1->output(0),
                                                kernel->output(0),
                                                strides,
                                                pads_begin,
                                                pads_end,
                                                dilations,
                                                op::PadType::VALID);

    // reshape 2
    auto shape_pattern_2 = make_shared<Constant>(Type_t::i32, Shape{2}, vector<int32_t>{1, 1600});
    auto reshape_2 = make_shared<Reshape>(convolution->output(0), shape_pattern_2->output(0), false);

    auto function = make_shared<Model>(reshape_2, ParameterVector({input}), "ReshapeConvolutionLessThan48Filters");
    return function;
}

std::shared_ptr<ov::Model> two_outputs_model() {
    auto precision = Type_t::f32;

    // input
    auto input = make_shared<Parameter>(precision, Shape{1, 10});

    // fully connected 1
    auto fc_weights_1 = make_shared<Constant>(precision, Shape{10, 20}, vector<float>(200, 1.0f));
    auto fc_biases_1 = make_shared<Constant>(precision, Shape{1, 20}, vector<float>(20, 1.0f));
    auto fc_before_biases_1 = make_shared<MatMul>(input->output(0), fc_weights_1->output(0));
    auto fc_1 = make_shared<Add>(fc_before_biases_1->output(0), fc_biases_1->output(0));

    // fully connected 2
    auto fc_weights_2 = make_shared<Constant>(precision, Shape{10, 10}, vector<float>(100, 1.0f));
    auto fc_biases_2 = make_shared<Constant>(precision, Shape{1, 10}, vector<float>(10, 1.0f));
    auto fc_before_biases_2 = make_shared<MatMul>(input->output(0), fc_weights_2->output(0));
    auto fc_2 = make_shared<Add>(fc_before_biases_2->output(0), fc_biases_2->output(0));

    auto function =
        make_shared<Model>(OutputVector{fc_1->output(0), fc_2->output(0)}, ParameterVector({input}), "TwoOutputs");
    return function;
}

std::shared_ptr<ov::Model> two_outputs_relu_model() {
    auto precision = Type_t::f32;

    // input
    auto input = make_shared<Parameter>(precision, Shape{1, 10});

    // fully connected 1
    auto fc_weights_1 = make_shared<Constant>(precision, Shape{10, 20}, vector<float>(200, 1.0f));
    auto fc_biases_1 = make_shared<Constant>(precision, Shape{1, 20}, vector<float>(20, 1.0f));
    auto fc_before_biases_1 = make_shared<MatMul>(input->output(0), fc_weights_1->output(0));
    auto fc_1 = make_shared<Add>(fc_before_biases_1->output(0), fc_biases_1->output(0));

    // fully connected 2
    auto fc_weights_2 = make_shared<Constant>(precision, Shape{10, 10}, vector<float>(100, 1.0f));
    auto fc_biases_2 = make_shared<Constant>(precision, Shape{1, 10}, vector<float>(10, 1.0f));
    auto fc_before_biases_2 = make_shared<MatMul>(input->output(0), fc_weights_2->output(0));
    auto fc_2 = make_shared<Add>(fc_before_biases_2->output(0), fc_biases_2->output(0));

    auto relu = make_shared<Relu>(fc_2->output(0));

    auto function =
        make_shared<Model>(OutputVector{fc_1->output(0), relu->output(0)}, ParameterVector({input}), "TwoOutputs");
    return function;
}

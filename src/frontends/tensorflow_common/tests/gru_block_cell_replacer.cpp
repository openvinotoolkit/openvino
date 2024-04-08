// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_transforms/gru_block_cell_replacer.hpp"

#include <gtest/gtest.h>

#include "conversion_with_reference.hpp"
#include "helper_ops/gru_block_cell.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;
using namespace element;
using namespace frontend::tensorflow;
using namespace frontend::tensorflow::pass;

namespace {
shared_ptr<Model> gen_model(Dimension batch_size, int64_t hidden_size, Dimension input_size) {
    auto x = make_shared<v0::Parameter>(f32, PartialShape{batch_size, input_size});
    auto h_prev = make_shared<v0::Parameter>(f32, PartialShape{batch_size, hidden_size});
    auto w_ru = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto w_c = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto b_ru = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto b_c = make_shared<v0::Parameter>(f32, PartialShape::dynamic());

    auto gru_block_cell = make_shared<GRUBlockCell>(x, h_prev, w_ru, w_c, b_ru, b_c);

    return make_shared<Model>(OutputVector{gru_block_cell->output(3)},
                              ParameterVector{x, h_prev, w_ru, w_c, b_ru, b_c});
}

shared_ptr<Model> gen_model_with_two_outputs(Dimension batch_size, int64_t hidden_size, Dimension input_size) {
    auto x = make_shared<v0::Parameter>(f32, PartialShape{batch_size, input_size});
    auto h_prev = make_shared<v0::Parameter>(f32, PartialShape{batch_size, hidden_size});
    auto w_ru = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto w_c = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto b_ru = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto b_c = make_shared<v0::Parameter>(f32, PartialShape::dynamic());

    auto gru_block_cell = make_shared<GRUBlockCell>(x, h_prev, w_ru, w_c, b_ru, b_c);

    return make_shared<Model>(OutputVector{gru_block_cell->output(0), gru_block_cell->output(3)},
                              ParameterVector{x, h_prev, w_ru, w_c, b_ru, b_c});
}

shared_ptr<Model> gen_model_ref(Dimension m_batch_size, int64_t m_hidden_size, Dimension m_input_size) {
    auto x = make_shared<v0::Parameter>(f32, PartialShape{m_batch_size, m_input_size});
    auto h_prev = make_shared<v0::Parameter>(f32, PartialShape{m_batch_size, m_hidden_size});
    auto w_ru = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto w_c = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto b_ru = make_shared<v0::Parameter>(f32, PartialShape::dynamic());
    auto b_c = make_shared<v0::Parameter>(f32, PartialShape::dynamic());

    // retrive input_size and hidden_size
    auto x_shape = make_shared<v3::ShapeOf>(x, element::i64);
    auto ss_start = make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto ss_end = make_shared<v0::Constant>(element::i64, Shape{1}, 2);
    auto ss_step = make_shared<v0::Constant>(element::i64, Shape{1}, 1);
    auto input_size = make_shared<v8::Slice>(x_shape, ss_start, ss_end, ss_step);
    auto h_prev_shape = make_shared<v3::ShapeOf>(h_prev, element::i64);
    auto hidden_size = make_shared<v8::Slice>(h_prev_shape, ss_start, ss_end, ss_step);

    // prepare weights input
    // TensorFlow provides weights in a format w_ru and w_c, where
    // z or u - update, r - reset, c or h - hidden (connection)
    // OpenVINO GRUCell accepts weights in a format w_zrh (or w_urс)
    // 1. split w_ru into w_r and w_u
    auto split_w_ru = make_shared<v1::Split>(w_ru, make_shared<v0::Constant>(element::i64, Shape{}, 1), 2);
    // 2. concatenate different parts of weights into w_zrh (or w_urс)
    auto w_urc = make_shared<v0::Concat>(OutputVector{split_w_ru->output(1), split_w_ru->output(0), w_c}, 1);

    // prepare bias in the same way
    auto split_b_ru = make_shared<v1::Split>(b_ru, make_shared<v0::Constant>(element::i64, Shape{}, 0), 2);
    auto b_urc = make_shared<v0::Concat>(OutputVector{split_b_ru->output(1), split_b_ru->output(0), b_c}, 0);

    // transpose weights
    // the current shape - [input_size + hidden_size, 3 * hidden_size]
    // we need the shape [3 * hidden_size, input_size + hidden_size]
    // in order to split WR into W and R
    auto transpose_order = make_shared<v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{1, 0});
    auto w_urc_transpose = make_shared<v1::Transpose>(w_urc, transpose_order);

    // split combined weights WR into W and R
    auto split_axis = make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto split_nums = make_shared<v0::Concat>(OutputVector{input_size, hidden_size}, 0);
    auto split_WR = make_shared<v1::VariadicSplit>(w_urc_transpose, split_axis, split_nums);

    auto gru_cell = make_shared<v3::GRUCell>(x, h_prev, split_WR->output(0), split_WR->output(1), b_urc, m_hidden_size);

    return make_shared<Model>(OutputVector{gru_cell->output(0)}, ParameterVector{x, h_prev, w_ru, w_c, b_ru, b_c});
}

}  // namespace

TEST_F(FrontEndConversionWithReferenceTestsF, GRUBlockCellReplacerOneOutput) {
    {
        model = gen_model(2, 10, 120);
        manager.register_pass<GRUBlockCellReplacer>();
    }
    { model_ref = gen_model_ref(2, 10, 120); }
}

TEST_F(FrontEndConversionWithReferenceTestsF, GRUBlockCellReplacerTwoOutputs) {
    {
        model = gen_model_with_two_outputs(2, 10, 120);
        manager.register_pass<GRUBlockCellReplacer>();
    }
    {
        // transformation is not applied due to presence of the first output
        model_ref = gen_model_with_two_outputs(2, 10, 120);
    }
}

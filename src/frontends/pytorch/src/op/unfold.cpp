// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_unfold(NodeContext& context) {
    // constants
    auto const_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
    auto const_1 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {1}));
    auto const_1_list = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {1}));
    auto const_neg_1 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {-1}));
    auto const_neg_1_list = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {-1}));

    // inputs
    auto input = context.get_input(0);
    int64_t dimension_int = context.const_input<int64_t>(1);
    auto dimension = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {dimension_int}));
    int size_int = context.const_input<int64_t>(2);
    auto size = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {size_int}));
    auto size_list = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {size_int}));
    auto step = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {context.const_input<int64_t>(3)}));

    auto sizes = context.mark_node(std::make_shared<opset8::ShapeOf>(input));
    auto sizedim_tmp = context.mark_node(std::make_shared<opset8::Gather>(sizes, dimension, const_0));
    auto sizedim = context.mark_node(std::make_shared<opset8::Reshape>(sizedim_tmp, const_1, false));
    auto sizedim_minus_size = std::make_shared<opset8::Subtract>(sizedim, size_list);
    auto fraction = std::make_shared<opset8::Divide>(sizedim_minus_size, step);
    auto slices_count = std::make_shared<opset8::Add>(fraction, const_1);
    auto slices_count_scalar = context.mark_node(std::make_shared<opset8::Reshape>(slices_count, const_1, false));

    auto start_indices_tmp = std::make_shared<opset8::Range>(const_0, slices_count_scalar, const_1, element::i64);
    auto start_indices = std::make_shared<opset8::Multiply>(start_indices_tmp, step);
    auto tmp_unsqueeze = std::make_shared<opset8::Unsqueeze>(start_indices, const_0);
    auto tmp_const = context.mark_node(opset8::Constant::create(element::i64, Shape{2}, {size_int, 1}));
    auto tile = std::make_shared<opset8::Tile>(tmp_unsqueeze, tmp_const);
    auto shape_perm =
                context.mark_node(context.mark_node(opset8::Constant::create(element::i64, Shape{2}, {1, 0})));
    auto tmp_transpose = context.mark_node(std::make_shared<opset8::Transpose>(tile, shape_perm));
    auto tmp_range = std::make_shared<opset8::Range>(const_0, size, const_1, element::i64);
    auto tmp_add = std::make_shared<opset8::Add>(tmp_transpose, tmp_range);
    auto tmp_reshape = std::make_shared<opset8::Reshape>(tmp_add, const_neg_1_list, false);
    // w tym momencie mam taką listę indeksów np [0,1,2,3,2,3,4,5,4,5,6,7] - n grup długości size

    auto ex_shape = std::make_shared<opset8::Concat>(OutputVector{const_neg_1_list, slices_count}, 0);
    auto ex_reshape = std::make_shared<opset8::Reshape>(tmp_reshape, ex_shape, false);
    auto ex_perm = context.mark_node(opset8::Constant::create(element::i64, Shape{2}, {1, 0}));
    auto ex_transpose = std::make_shared<opset8::Transpose>(ex_reshape, ex_perm);
    auto ex_res = std::make_shared<opset8::Reshape>(ex_transpose, const_neg_1_list, false);
    // teraz lista indeksów zamieniona jest na [0,2,4,1,3,5,2,4,6,3,5,7] - size grup długości n

    auto tmp_gather = std::make_shared<opset8::Gather>(input, ex_res, dimension); //wcześniej zamiast ex_res było tmp_reshape

    auto slices_count_tile = std::make_shared<opset8::Tile>(slices_count, size_list);
    auto split_indices = std::make_shared<opset8::Concat>(OutputVector{slices_count_tile, const_neg_1_list}, 0);
    auto variadic_split = std::make_shared<opset8::VariadicSplit>(tmp_gather, dimension, slices_count_tile);
    OutputVector tmp_vec;
    for (int i = 0; i < size_int; i++) {
        auto loop_unsqueeze = std::make_shared<opset8::Unsqueeze>(variadic_split->output(i), const_neg_1_list);
        tmp_vec.push_back(loop_unsqueeze);
    }
    auto new_concat = std::make_shared<opset8::Concat>(tmp_vec, -1);
    // auto new_concat = std::make_shared<opset8::Concat>(tmp_vec, dimension_int);
    return {variadic_split->output(0)};
};

// OutputVector translate_unfold(NodeContext& context) {
//     // constants
//     auto const_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
//     auto const_1 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {1}));
//     auto const_0_list = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {0}));
//     auto const_1_list = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {1}));
//     auto const_neg_1_list = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {-1}));

//     // inputs
//     auto input = context.get_input(0);
//     int64_t dimension_int = context.const_input<int64_t>(1);
//     auto dimension = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {dimension_int}));
//     auto size = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {context.const_input<int64_t>(2)}));
//     auto size_list = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {context.const_input<int64_t>(2)}));
//     auto step = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {context.const_input<int64_t>(3)}));

//     auto sizes = context.mark_node(std::make_shared<opset8::ShapeOf>(input));
//     auto dimension_plus_1 = context.mark_node(std::make_shared<opset8::Add>(dimension, const_1_list));
//     auto sizedim_tmp = context.mark_node(std::make_shared<opset8::Gather>(sizes, dimension, const_0));
//     auto sizedim = context.mark_node(std::make_shared<opset8::Reshape>(sizedim_tmp, const_1, false));
//     // auto sizedim_plus_1 = context.mark_node(std::make_shared<opset8::Add>(sizedim, const_1));
//     auto low_indices = context.mark_node(std::make_shared<opset8::Range>(const_0, sizedim, step, element::i64));
//     // auto hi_indices = context.mark_node(std::make_shared<opset8::Range>(size, sizedim_plus_1, step, element::i64));
//     // auto low_indices_count = context.mark_node(std::make_shared<opset8::ShapeOf>(low_indices));
//     // auto hi_indices_count = context.mark_node(std::make_shared<opset8::ShapeOf>(hi_indices));
//     // auto iterations_count = context.mark_node(std::make_shared<opset8::Minimum>(low_indices_count, hi_indices_count));
//     // auto iterations_count_scalar =
//     //     context.mark_node(std::make_shared<opset8::Reshape>(iterations_count, const_1, false));

//     auto neg_size = std::make_shared<opset8::Multiply>(size, const_neg_1_list);
//     // nie trzeba robić tak - mamy operator Substract
//     auto sizedim_minus_size = std::make_shared<opset8::Add>(sizedim, neg_size);
//     auto tmp_fraction = std::make_shared<opset8::Divide>(sizedim_minus_size, step);
//     auto tmp_output_dim = std::make_shared<opset8::Add>(tmp_fraction, const_1);
//     // Może zamiast takiego flatowania lepiej dać po prostu Squeeze?
//     // ta wartość jest dobrze liczona
//     auto tmp_output_dim_scalar = context.mark_node(std::make_shared<opset8::Reshape>(tmp_output_dim, const_1, false));
//     auto tmp_output_dim_scalar_minus_1 = std::make_shared<opset8::Add>(tmp_output_dim_scalar, const_neg_1_list);

//     // auto last_low_ind = std::make_shared<opset8::Gather>(low_indices, iterations_count_scalar, const_0);
//     // auto last_low_ind_scalar =
//     //     context.mark_node(std::make_shared<opset8::Reshape>(last_low_ind, const_1, false));
//     // muszę wziąć minus 1, bo tmp_output_dim_scalar to łączna liczba elementów, a ja chcę wziąć ostatni
//     auto last_low_ind = std::make_shared<opset8::Gather>(low_indices, tmp_output_dim_scalar_minus_1, const_0);
//     auto last_low_ind_scalar =
//         context.mark_node(std::make_shared<opset8::Reshape>(last_low_ind, const_1, false));

//     // mój nowy pomysł - jeszcze do zastanowienia, czy dać plus jeden, czy nie
//     auto last_low_ind_scalar_plus_1 = std::make_shared<opset8::Add>(last_low_ind_scalar, const_1);
//     auto start_indices = std::make_shared<opset8::Range>(const_0, last_low_ind_scalar_plus_1, step, element::i64);
//     auto tmp_unsqueeze = std::make_shared<opset8::Unsqueeze>(start_indices, const_0);
//     // return {context.mark_node(tmp_unsqueeze)};

//     int tmp_size = context.const_input<int64_t>(2);
//     auto tmp_const = context.mark_node(opset8::Constant::create(element::i64, Shape{2}, {tmp_size, 1}));
//     auto tile = std::make_shared<opset8::Tile>(tmp_unsqueeze, tmp_const);
//     auto shape_perm =
//                 context.mark_node(context.mark_node(opset8::Constant::create(element::i64, Shape{2}, {1, 0})));
//     auto tmp_transpose = context.mark_node(std::make_shared<opset8::Transpose>(tile, shape_perm));

//     auto tmp_range = std::make_shared<opset8::Range>(const_0, size, const_1, element::i64);
//     auto tmp_add = std::make_shared<opset8::Add>(tmp_transpose, tmp_range);
//     auto tmp_reshape = std::make_shared<opset8::Reshape>(tmp_add, const_neg_1_list, false);
//     // do tego momenty tmmp_reshape to lista tak jakby rosnąca
//     // eksperyment
//     auto ex_shape = std::make_shared<opset8::Concat>(OutputVector{const_neg_1_list, tmp_output_dim}, 0);
//     auto ex_reshape = std::make_shared<opset8::Reshape>(tmp_reshape, ex_shape, false);
//     auto ex_perm = context.mark_node(opset8::Constant::create(element::i64, Shape{2}, {1, 0}));
//     auto ex_transpose = std::make_shared<opset8::Transpose>(ex_reshape, ex_perm);
//     auto ex_res = std::make_shared<opset8::Reshape>(ex_transpose, const_neg_1_list, false);
//     // ex_res to ta lista, ktróra jest postaci [(pierwsze indeksy), (drugie indeksy), ..., (size'te indeksy)]
//     // eksperyment
//     auto tmp_gather = std::make_shared<opset8::Gather>(input, ex_res, dimension);
//     auto tmp_tmp_unsqueeze = context.mark_node(std::make_shared<opset8::Unsqueeze>(tmp_gather, dimension));
//     // auto tmp_gather = std::make_shared<opset8::Gather>(input, tmp_reshape, dimension);
//     // fajnie, wydaje się, że to może działać! - generowanie odpowiednich elementów

//     // uzyskanie odpowiedniego shape'u
//     auto ndim_tmp = context.mark_node(std::make_shared<opset8::ShapeOf>(sizes));
//     auto ndim = context.mark_node(std::make_shared<opset8::Reshape>(ndim_tmp, const_1, false));
//     auto dimension_scalar = context.mark_node(std::make_shared<opset8::Reshape>(dimension, const_1, false));
//     auto dimension_plus_1_scalar =
//         context.mark_node(std::make_shared<opset8::Reshape>(dimension_plus_1, const_1, false));
//     auto perm_begin =
//         context.mark_node(std::make_shared<opset8::Range>(const_0, dimension_scalar, const_1, element::i64));
//     auto perm_end =
//         context.mark_node(std::make_shared<opset8::Range>(dimension_plus_1_scalar, ndim, const_1, element::i64));
//     // perm to lista posortowana z wartością dimension na samym końcu
//     auto perm = context.mark_node(std::make_shared<opset8::Concat>(OutputVector{perm_begin, perm_end, dimension}, 0));
//     auto transpose = context.mark_node(std::make_shared<opset8::Transpose>(tmp_gather, perm));
//     auto unsqueeze = context.mark_node(std::make_shared<opset8::Unsqueeze>(transpose, dimension));

//     // eksperyment 2
//     auto ex_split = std::make_shared<opset8::Split>(ex_res, const_0, tmp_size);
//     OutputVector ex_vector;
//     for (int i = 0; i < tmp_size; i++){
//         auto ex_tmp_gather = std::make_shared<opset8::Gather>(input, ex_split->output(i), dimension);
//         auto ex_tmp_transpose = context.mark_node(std::make_shared<opset8::Transpose>(ex_tmp_gather, perm));
//         auto ex_tmp_unsqueeze = context.mark_node(std::make_shared<opset8::Unsqueeze>(ex_tmp_transpose, dimension));
//         ex_vector.push_back(ex_tmp_unsqueeze);
//     }
//     auto ex_tmp_concat = std::make_shared<opset8::Concat>(ex_vector, dimension_int);
//     // eksperyment 2

//     // tutaj muszę ustawić odpowiednią wartość w input_shape
//     // może zamiast tego -1 w reshape moża dać wprost odpowiedni wymiar
//     // niby final shape mam dobry, ale źle są poustawiane elementy
//     auto scatter_update = std::make_shared<opset8::ScatterUpdate>(sizes, dimension, tmp_output_dim, const_0);
//     OutputVector final_shape_vec{scatter_update, size_list};
//     auto output_shape = context.mark_node(std::make_shared<opset8::Concat>(final_shape_vec, 0));
//     auto result = std::make_shared<opset8::Reshape>(ex_tmp_concat, output_shape, false);

//     return {context.mark_node(result)};
// };

// OutputVector translate_unfold(NodeContext& context) {
//     // constants
//     auto const_0 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {0}));
//     auto const_1 = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {1}));
//     auto const_0_list = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {0}));
//     auto const_1_list = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {1}));

//     // inputs
//     auto input = context.get_input(0);
//     int64_t dimension_int = context.const_input<int64_t>(1);
//     auto dimension = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {dimension_int}));
//     auto size = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {context.const_input<int64_t>(2)}));
//     auto step = context.mark_node(opset8::Constant::create(element::i64, Shape{}, {context.const_input<int64_t>(3)}));

//     auto sizes = context.mark_node(std::make_shared<opset8::ShapeOf>(input));
//     auto dimension_plus_1 = context.mark_node(std::make_shared<opset8::Add>(dimension, const_1_list));
//     auto sizedim_tmp =
//         context.mark_node(std::make_shared<opset8::Slice>(sizes, dimension, dimension_plus_1, const_1_list));
//     auto sizedim = context.mark_node(std::make_shared<opset8::Reshape>(sizedim_tmp, const_1, false));
//     auto sizedim_plus_1 = context.mark_node(std::make_shared<opset8::Add>(sizedim, const_1));

//     auto low_indices = context.mark_node(std::make_shared<opset8::Range>(const_0, sizedim, step, element::i64));
//     auto hi_indices = context.mark_node(std::make_shared<opset8::Range>(size, sizedim_plus_1, step, element::i64));

//     auto ndim_tmp = context.mark_node(std::make_shared<opset8::ShapeOf>(sizes));
//     auto ndim = context.mark_node(std::make_shared<opset8::Reshape>(ndim_tmp, const_1, false));
//     auto dimension_scalar = context.mark_node(std::make_shared<opset8::Reshape>(dimension, const_1, false));
//     auto dimension_plus_1_scalar =
//         context.mark_node(std::make_shared<opset8::Reshape>(dimension_plus_1, const_1, false));
//     auto perm_begin =
//         context.mark_node(std::make_shared<opset8::Range>(const_0, dimension_scalar, const_1, element::i64));
//     auto perm_end =
//         context.mark_node(std::make_shared<opset8::Range>(dimension_plus_1_scalar, ndim, const_1, element::i64));
//     auto perm = context.mark_node(std::make_shared<opset8::Concat>(OutputVector{perm_begin, perm_end, dimension}, 0));

//     // body parameters
//     auto input_param = std::make_shared<opset8::Parameter>(element::f32, PartialShape::dynamic());
//     auto low_ind_param = std::make_shared<opset8::Parameter>(element::i64, PartialShape::dynamic());
//     auto hi_ind_param = std::make_shared<opset8::Parameter>(element::i64, PartialShape::dynamic());
//     auto perm_param = std::make_shared<opset8::Parameter>(element::i64, PartialShape::dynamic());
//     auto iter_param = std::make_shared<opset8::Parameter>(element::i64, PartialShape::dynamic());

//     // body
//     auto iter_plus_1 = context.mark_node(std::make_shared<opset8::Add>(iter_param, const_1_list));
//     auto low_ind_curr_iter = context.mark_node(
//         std::make_shared<opset8::Slice>(low_ind_param, iter_param, iter_plus_1, const_1_list, const_0_list));
//     auto hi_ind_curr_iter = context.mark_node(
//         std::make_shared<opset8::Slice>(hi_ind_param, iter_param, iter_plus_1, const_1_list, const_0_list));
//     auto slice = context.mark_node(
//         std::make_shared<opset8::Slice>(input_param, low_ind_curr_iter, hi_ind_curr_iter, const_1_list, dimension));
//     auto transpose = context.mark_node(std::make_shared<opset8::Transpose>(slice, perm_param));
//     auto unsqueeze = context.mark_node(std::make_shared<opset8::Unsqueeze>(transpose, dimension));
//     auto body =
//         std::make_shared<Model>(OutputVector{unsqueeze},
//                                 ParameterVector{iter_param, input_param, low_ind_param, hi_ind_param, perm_param});

//     // number of iterations
//     auto low_indices_count = context.mark_node(std::make_shared<opset8::ShapeOf>(low_indices));
//     auto hi_indices_count = context.mark_node(std::make_shared<opset8::ShapeOf>(hi_indices));
//     auto iterations_count = context.mark_node(std::make_shared<opset8::Minimum>(low_indices_count, hi_indices_count));
//     auto iterations_count_scalar =
//         context.mark_node(std::make_shared<opset8::Reshape>(iterations_count, const_1, false));
//     auto iter_values =
//         context.mark_node(std::make_shared<opset8::Range>(const_0, iterations_count_scalar, const_1, element::i64));
//     auto tensor_iterator = std::make_shared<opset8::TensorIterator>();

//     // body input preparation
//     tensor_iterator->set_function(body);
//     tensor_iterator->set_invariant_input(input_param, input);
//     tensor_iterator->set_invariant_input(perm_param, perm);
//     tensor_iterator->set_invariant_input(low_ind_param, low_indices);
//     tensor_iterator->set_invariant_input(hi_ind_param, hi_indices);
//     tensor_iterator->set_sliced_input(iter_param, iter_values, 0, 1, 1, -1, 0);

//     context.mark_nodes({tensor_iterator, input_param, low_ind_param, hi_ind_param, perm_param, iter_param});

//     auto result = tensor_iterator->get_concatenated_slices(unsqueeze, 0, 1, 1, -1, dimension_int);
//     return {context.mark_node(result.get_node_shared_ptr())};
// };

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
#include "openvino/op/logical_and.hpp"
#include "openvino/op/less_equal.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_matrix_band_part_op(const NodeContext& node) {
    default_op_checks(node, 1, {"MatrixBandPart", "MATRIX_BAND_PART"});

    auto input = node.get_input(0);
    auto input_shape = make_shared<v3::ShapeOf>(input);
    auto last_dim = make_shared<v1::StridedSlice>(
        input_shape,
        make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-2}),
        make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1}),
        make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1}),
        std::vector<int64_t>({0}),
        std::vector<int64_t>({0}));

    auto m = make_shared<v1::StridedSlice>(
        last_dim,
        make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0}),
        make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1}),
        make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1}),
        std::vector<int64_t>({0}),
        std::vector<int64_t>({0}));

    auto n = make_shared<v1::StridedSlice>(
        last_dim,
        make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1}),
        make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{2}),
        make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1}),
        std::vector<int64_t>({0}),
        std::vector<int64_t>({0}));

    auto range_m = make_shared<v0::Range>(make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                          m,
                                          make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1}));

    auto range_n = make_shared<v0::Range>(make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                          n,
                                          make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1}));

    auto unsqueeze_range_m = make_shared<v0::Unsqueeze>(range_m, make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1}));
    auto unsqueeze_range_n = make_shared<v0::Unsqueeze>(range_n, make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0}));

    auto in_band_indicator = make_shared<v1::LessEqual>(unsqueeze_range_m, unsqueeze_range_n);

    auto unsqueeze_in_band_indicator = make_shared<v0::Unsqueeze>(in_band_indicator, last_dim);

    auto zero_padding = make_shared<v0::Concat>(
        OutputVector({make_shared<v0::Constant>(input.get_element_type(), Shape{1}, std::vector<int64_t>{0})}),
        0);

    auto band_part = make_shared<v1::Select>(unsqueeze_in_band_indicator, input, zero_padding);

    auto new_shape = make_shared<v0::Concat>(OutputVector({input_shape, last_dim}), 0);

    auto reshaped_band_part = make_shared<v1::Reshape>(band_part, new_shape, false);

    set_node_name(node.get_name(), reshaped_band_part);
    return {reshaped_band_part};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov

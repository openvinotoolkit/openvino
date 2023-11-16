#include "common_op_table.hpp"
#include "common_op_table.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"


using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_atan2_op(const NodeContext& node) {
    // handle the first condition
    auto div_y_x = make_shared<Divide>(y, x);
    auto atan = make_shared<Atan>(div_y_x);
    auto const_zero = create_same_type_const_scalar<int32_t>(x, 0);
    auto is_x_positive = make_shared<Greater>(x, const_zero);
    auto result = make_shared<Select>(is_x_positive, atan, result);;

    // handle the second condition
    auto const_pi = create_same_type_const_scalar<double>(x, std::atan(1.0)*4);
    auto x_negative = make_shared<Less>(x, const_zero);
    auto y_non_negative = make_shared<GreaterEqual>(y, const_zero);
    auto cond1 = make_shared<LogicalAnd>(x_negative, y_non_negative);
    auto arctan_y_x_plus_pi = make_shared<Add>(arctan_y_x, const_pi)
    auto result = make_shared<Select>(cond1, arctan_y_x_plus_pi, result);

    // handle the third consition
    auto y_negative = make_shared<Less>(y, const_zero);
    auto cond2 = make_shared<LogicalAnd>(x_negative, y_negative);
    auto arctan_y_x_minus_pi = make_shared<Subtract>(arctan_y_x, const_pi);
    auto result = make_shared<Select>(cond2, arctan_y_x_minus_pi, result);

    // handle the fourth condition
    auto is_x_zero = make_shared<Equal>(x, const_zero);
    auto is_y_negative = make_shared<Less>(y, const_zero);
    auto cond3 = make_shared<LogicalAnd>(is_x_zero, is_y_negative);
    auto const_two = create_same_type_const_scalar<int32_t>(x, 2);
    auto pi_div_two = make_shared<Divide>(const_pi, const_two);
    auto result = make_shared<Select>(cond3, pi_div_two, result);

    // handle the fifth condition
    auto is_x_zero = make_shared<Equal>(x, const_zero);
    auto is_y_negative = make_shared<Less>(y, const_zero);
    auto cond4 = make_shared<LogicalAnd>(is_x_zero, is_y_negative);
    auto const_minus_two = create_same_type_const_scalar<int32_t>(x, -2);
    auto pi_div_minus_two = make_shared<Divide>(const_pi, const_minus_two);
    auto result = make_shared<Select>(cond4, pi_div_two, result);

       
    set_node_name(node.get_name(), result);
    return result->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov

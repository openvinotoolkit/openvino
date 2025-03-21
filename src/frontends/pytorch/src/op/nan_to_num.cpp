#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_nan_to_num(const NodeContext &context) {
  num_inputs_check(context, 1, 4);

  auto x = context.get_input(0);

  if (x.get_element_type().is_integral()) {
    // Wrapping integer tensor before returning to avoid issues with
    // OutputVector expectations
    return {context.mark_node(x)};
  }

  auto nan_value = context.input_is_none(1)
                       ? v0::Constant::create(element::f32, Shape{}, {0})
                       : context.get_input(1);
  auto posinf_value =
      context.input_is_none(2)
          ? v0::Constant::create(element::f32, Shape{},
                                 {std::numeric_limits<float>::max()})
          : context.get_input(2);
  auto neginf_value =
      context.input_is_none(3)
          ? v0::Constant::create(element::f32, Shape{},
                                 {std::numeric_limits<float>::lowest()})
          : context.get_input(3);

  auto nan_replacement =
      context.mark_node(std::make_shared<v1::ConvertLike>(nan_value, x));
  auto posinf_replacement =
      context.mark_node(std::make_shared<v1::ConvertLike>(posinf_value, x));
  auto neginf_replacement =
      context.mark_node(std::make_shared<v1::ConvertLike>(neginf_value, x));

  auto is_nan = context.mark_node(std::make_shared<v10::IsNaN>(x));
  auto is_posinf = context.mark_node(
      std::make_shared<v10::IsInf>(x, v10::IsInf::Attributes(false, true)));
  auto is_neginf = context.mark_node(
      std::make_shared<v10::IsInf>(x, v10::IsInf::Attributes(true, false)));

  auto replaced_nan = context.mark_node(
      std::make_shared<v1::Select>(is_nan, nan_replacement, x));
  auto replaced_posinf = context.mark_node(std::make_shared<v1::Select>(
      is_posinf, posinf_replacement, replaced_nan));
  auto replaced_neginf = context.mark_node(std::make_shared<v1::Select>(
      is_neginf, neginf_replacement, replaced_posinf));

  return {replaced_neginf};
}

} // namespace op
} // namespace pytorch
} // namespace frontend
} // namespace ov

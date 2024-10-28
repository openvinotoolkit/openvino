#include "openvino/op/util/squeeze_base.hpp"

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace op {

namespace validate {
namespace {

bool axes_has_and_set_bound(const Node& op) {
    return (op.get_input_size() < 2) || op.get_input_tensor(1).has_and_set_bound();
}
}  // namespace
}  // namespace validate

namespace util {
SqueezeBase::SqueezeBase(const Output<Node>& data, const Output<Node>& axes) : Op({data, axes}) {
    constructor_validate_and_infer_types();
}

SqueezeBase::SqueezeBase(const Output<Node>& data) : Op({data}) {
    constructor_validate_and_infer_types();
}

bool SqueezeBase::has_evaluate() const {
    OV_OP_SCOPE(util_SqueezeBase_has_evaluate);
    const auto validate_axes_type = [](const element::Type& et) -> bool {
        switch (et) {
        case element::i8:
        case element::i16:
        case element::i32:
        case element::i64:
        case element::u8:
        case element::u16:
        case element::u32:
        case element::u64:
            return true;
        default:
            return false;
        }
    };

    return (get_input_size() < 2) || validate_axes_type(get_input_element_type(1));
}

bool SqueezeBase::evaluate_lower(TensorVector& output_values) const {
    OV_OP_SCOPE(util_SqueezeBase_evaluate_lower);
    return validate::axes_has_and_set_bound(*this) && default_lower_bound_evaluator(this, output_values);
}

bool SqueezeBase::evaluate_upper(TensorVector& output_values) const {
    OV_OP_SCOPE(util_SqueezeBase_evaluate_upper);
    return validate::axes_has_and_set_bound(*this) && default_upper_bound_evaluator(this, output_values);
}

bool SqueezeBase::evaluate_symbol(TensorSymbolVector& output_symbols) const {
    OV_OP_SCOPE(util_SqueezeBase_evaluate_symbol);
    return validate::axes_has_and_set_bound(*this) && ov::util::default_symbol_evaluator(this, output_symbols);
}

bool SqueezeBase::can_constant_fold(const OutputVector& inputs_values) const {
    OV_OP_SCOPE(util_SqueezeBase_can_constant_fold);
    return get_output_partial_shape(0).is_static() && !is_const_fold_disabled();
}

bool SqueezeBase::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    OV_OP_SCOPE(util_SqueezeBase_constant_fold);
    if (!can_constant_fold(inputs_values)) {
        return false;
    }

    if (auto data_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(inputs_values[0].get_node_shared_ptr())) {
        const auto& shape = get_output_shape(0);
        output_values[0] = std::make_shared<ov::op::v0::Constant>(*data_const, shape);
        return true;
    }
    return false;
}

bool SqueezeBase::is_dynamic() const {
    OV_OP_SCOPE(util_SqueezeBase_is_dynamic);
    return get_output_partial_shape(0).is_dynamic();
}

}  // namespace util
}  // namespace op
}  // namespace ov

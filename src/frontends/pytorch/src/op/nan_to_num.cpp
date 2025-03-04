OutputVector translate_nan_to_num_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 4);

    auto x = context.get_input(0);

    // Checking if input is an integer type
    if (x.get_element_type().is_integral()) {
        // Wrapping integer tensor before returning to avoid issues with OutputVector expectations
        return {context.mark_node(x)};
    }

    auto x_type = x.get_element_type();

    auto nan_replacement = context.mark_node(v0::Constant::create(x_type, Shape{}, {0}));
    auto posinf_replacement = context.mark_node(v0::Constant::create(x_type, Shape{}, {std::numeric_limits<float>::max()}));
    auto neginf_replacement = context.mark_node(v0::Constant::create(x_type, Shape{}, {std::numeric_limits<float>::lowest()}));

    if (!context.input_is_none(1)) {
        nan_replacement = context.get_input(1);
    }
    if (!context.input_is_none(2)) {
        posinf_replacement = context.get_input(2);
    }
    if (!context.input_is_none(3)) {
        neginf_replacement = context.get_input(3);
    }

    auto is_nan = context.mark_node(std::make_shared<v10::IsNan>(x));
    auto is_posinf = context.mark_node(std::make_shared<v10::IsInf>(x, v10::IsInf::Attributes(false, true)));
    auto is_neginf = context.mark_node(std::make_shared<v10::IsInf>(x, v10::IsInf::Attributes(true, false)));

    auto replaced_nan = context.mark_node(std::make_shared<v1::Select>(is_nan, nan_replacement, x));
    auto replaced_posinf = context.mark_node(std::make_shared<v1::Select>(is_posinf, posinf_replacement, replaced_nan));
    auto replaced_neginf = context.mark_node(std::make_shared<v1::Select>(is_neginf, neginf_replacement, replaced_posinf));

    return {replaced_neginf};
};

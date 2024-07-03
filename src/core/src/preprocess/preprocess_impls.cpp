// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocess_impls.hpp"

#include "layout_utils.hpp"
#include "openvino/core/descriptor_tensor.hpp"

namespace ov {
namespace preprocess {

namespace {
void dump_tensor(std::ostream& str,
                 const PartialShape& shape,
                 const Layout& layout,
                 const element::Type& type,
                 const ColorFormat& color = ColorFormat::UNDEFINED) {
    str << shape << ", ";
    if (layout.empty()) {
        str << "<no layout>";
    } else {
        str << layout.to_string();
    }
    str << ", " << type;
    if (color != ColorFormat::UNDEFINED) {
        str << ", " << color_format_name(color);
    }
}
}  // namespace

InputInfo::InputInfoImpl::InputInfoData InputInfo::InputInfoImpl::create_new_params(
    std::tuple<std::unordered_set<std::string>, bool>& existing_names,
    const std::shared_ptr<Model>& model) const {
    InputInfoData res;
    res.m_param = m_resolved_param;
    auto tensor_elem_type = get_tensor_data()->is_element_type_set() ? get_tensor_data()->get_element_type()
                                                                     : res.m_param->get_element_type();
    res.m_tensor_layout = get_tensor_data()->get_layout();
    auto color_info = ColorFormatInfo::get(get_tensor_data()->get_color_format());
    if (!get_tensor_data()->is_layout_set()) {
        if (!color_info->default_layout().empty()) {
            res.m_tensor_layout = color_info->default_layout();
        }
    }

    const auto& model_shape = res.m_param->get_partial_shape();
    auto new_param_shape = model_shape;
    if (get_tensor_data()->is_shape_set()) {
        new_param_shape = get_tensor_data()->get_shape();
    }
    res.m_model_layout = get_model()->is_layout_set() ? get_model()->get_layout() : res.m_param->get_layout();
    if (res.m_model_layout.empty() && get_tensor_data()->is_layout_set()) {
        res.m_model_layout = get_preprocess()->propagate_layout(res.m_tensor_layout);
    }
    if (!res.m_tensor_layout.empty() && !res.m_model_layout.empty() && res.m_model_layout != res.m_tensor_layout) {
        auto sq_layout = Layout();
        // Find if some squeeze is needed between model and tensor
        // E.g. model=NCHW, tensor=HWC
        std::tie(new_param_shape, sq_layout) =
            layout::utils::find_squeeze(res.m_model_layout, model_shape, res.m_tensor_layout);
        // Find transpose between model and tensor layouts and update tensor shape
        auto net_to_tensor = layout::utils::find_permutation(sq_layout, new_param_shape, res.m_tensor_layout);
        if (!net_to_tensor.empty() && new_param_shape.rank().is_static()) {
            std::vector<ov::Dimension> dims(new_param_shape.size());
            std::transform(net_to_tensor.begin(),
                           net_to_tensor.end(),
                           dims.begin(),
                           [&](int64_t v) -> const Dimension& {
                               return new_param_shape[v];
                           });
            new_param_shape = PartialShape(std::move(dims));
        }
    } else {
        Layout new_layout;
        std::tie(new_param_shape, new_layout) =
            get_preprocess()->calculate_param_shape(new_param_shape, res.m_model_layout);
        if (res.m_tensor_layout.empty()) {
            // Reusing param's layout according to converted calculated layout
            res.m_tensor_layout = std::move(new_layout);
        }
    }

    if (get_tensor_data()->is_shape_set()) {
        new_param_shape = get_tensor_data()->get_shape();
    } else if (get_tensor_data()->is_spatial_shape_set()) {
        auto height_idx = get_and_check_height_idx(res.m_tensor_layout, new_param_shape);
        auto width_idx = get_and_check_width_idx(res.m_tensor_layout, new_param_shape);
        if (get_tensor_data()->is_spatial_shape_dynamic()) {
            // Use dynamic spatial dimensions
            new_param_shape[height_idx] = Dimension::dynamic();
            new_param_shape[width_idx] = Dimension::dynamic();
        } else {
            // Use static spatial dimensions
            new_param_shape[height_idx] = get_tensor_data()->get_spatial_height();
            new_param_shape[width_idx] = get_tensor_data()->get_spatial_width();
        }
    }

    // Create separate parameter for each plane. Shape is based on color format
    for (size_t plane = 0; plane < color_info->planes_count(); plane++) {
        auto plane_shape = color_info->shape(plane, new_param_shape, res.m_tensor_layout);
        auto plane_param = std::make_shared<opset8::Parameter>(tensor_elem_type, plane_shape);
        if (plane < get_tensor_data()->planes_sub_names().size()) {
            std::unordered_set<std::string> plane_tensor_names;
            std::string sub_name;
            sub_name = std::string("/") + get_tensor_data()->planes_sub_names()[plane];
            if (!std::get<1>(existing_names)) {
                existing_names = std::make_tuple(get_function_tensor_names(model), true);
            }
            for (const auto& tensor_name : res.m_param->get_default_output().get_tensor().get_names()) {
                auto new_name = tensor_name + sub_name;
                OPENVINO_ASSERT(
                    std::get<0>(existing_names).count(new_name) == 0,
                    "Error while trying to create plane input with name '",
                    new_name,
                    "' - name already exists in model. Please specify another sub-name for set_color_format");
                plane_tensor_names.insert(new_name);
            }
            plane_param->get_default_output().get_tensor().set_names(plane_tensor_names);
            plane_param->set_friendly_name(res.m_param->get_friendly_name() + sub_name);
        } else if (color_info->planes_count() == 1) {
            plane_param->get_default_output().get_tensor().set_names(
                res.m_param->get_default_output().get_tensor().get_names());
            plane_param->set_friendly_name(res.m_param->get_friendly_name());
        }
        // Fill runtime info
        plane_param->get_rt_info() = res.m_param->get_rt_info();
        plane_param->output(0).get_rt_info() = res.m_param->output(0).get_rt_info();
        if (!res.m_tensor_layout.empty()) {
            plane_param->set_layout(res.m_tensor_layout);
        }
        if (get_tensor_data()->is_memory_type_set()) {
            if (get_tensor_data()->get_memory_type().empty()) {
                plane_param->output(0).get_rt_info().erase(TensorInfoMemoryType::get_type_info_static());
            } else {
                plane_param->output(0).get_rt_info()[TensorInfoMemoryType::get_type_info_static()] =
                    TensorInfoMemoryType(get_tensor_data()->get_memory_type());
            }
        }
        res.m_new_params.push_back(plane_param);
    }
    return res;
}

PreStepsList InputInfo::InputInfoImpl::create_implicit_steps(const PreprocessingContext& context, element::Type type) {
    PreStepsList implicit_steps;
    if (type != context.target_element_type()) {
        implicit_steps.add_convert_impl(context.target_element_type());
    }
    if (!context.target_layout().empty() && context.target_layout() != context.layout()) {
        implicit_steps.add_convert_layout_impl(context.target_layout());
    }
    return implicit_steps;
}

bool InputInfo::InputInfoImpl::build(const std::shared_ptr<Model>& model,
                                     std::tuple<std::unordered_set<std::string>, bool>& existing_names,
                                     std::list<std::shared_ptr<opset8::Parameter>>& parameters_list) {
    auto data = create_new_params(existing_names, model);
    auto consumers = data.m_param->output(0).get_target_inputs();
    bool need_validate = false;

    PreprocessingContext context(data.m_tensor_layout);
    context.color_format() = get_tensor_data()->get_color_format();
    context.target_layout() = data.m_model_layout;
    context.model_shape() = data.m_param->get_partial_shape();
    context.target_element_type() = data.m_param->get_element_type();

    // Apply preprocessing
    auto nodes = data.as_nodes();
    for (const auto& action : get_preprocess()->actions()) {
        auto action_result = action.m_op(nodes, model, context);
        nodes = std::get<0>(action_result);
        need_validate = need_validate || std::get<1>(action_result);
    }

    OPENVINO_ASSERT(nodes.size() == 1,
                    "Multiple plane input is not allowed as model input. Consider using of convert_color "
                    "preprocessing operation. Current format is '",
                    color_format_name(context.color_format()),
                    "'");
    OPENVINO_ASSERT(is_rgb_family(context.color_format()) || context.color_format() == ColorFormat::GRAY ||
                        context.color_format() == ColorFormat::UNDEFINED,
                    "model shall have RGB/BGR/GRAY color format. Consider add 'convert_color' preprocessing operation "
                    "to convert current color format '",
                    color_format_name(context.color_format()),
                    "'to RGB/BGR/GRAY");

    // Implicit: Convert element type + layout to user's tensor implicitly
    auto implicit_steps = create_implicit_steps(context, nodes[0].get_element_type());
    for (const auto& action : implicit_steps.actions()) {
        auto action_result = action.m_op(nodes, model, context);
        nodes = std::get<0>(action_result);
    }

    const auto& node = nodes[0];
    if (node.get_partial_shape() != context.model_shape()) {
        need_validate = true;  // Trigger revalidation if input parameter shape is changed
    }

    // Check final shape
    OPENVINO_ASSERT(node.get_partial_shape().compatible(context.model_shape()),
                    "Resulting shape '",
                    node.get_partial_shape(),
                    "' after preprocessing is not aligned with original parameter's shape: ",
                    context.model_shape(),
                    ", input parameter: ",
                    data.m_param->get_friendly_name());

    // Replace parameter
    for (auto consumer : consumers) {
        if (dynamic_cast<ov::opset8::Result*>(consumer.get_node())) {
            // Some result points to old parameter (Param->Result case), need to trigger revalidation
            need_validate = true;
        }
        consumer.replace_source_output(node);
    }
    {
        auto param_it = std::find(parameters_list.begin(), parameters_list.end(), data.m_param);
        OPENVINO_ASSERT(param_it != parameters_list.end(),
                        "Parameter to replace has been replaced by previous steps of preprocessing. Use only one "
                        "InputInfo for one input parameter");
        // Insert list of new parameters to the place of original parameter
        param_it = parameters_list.erase(param_it);
        parameters_list.insert(param_it, data.m_new_params.begin(), data.m_new_params.end());
    }
    return need_validate;
}

void InputInfo::InputInfoImpl::dump(std::ostream& str,
                                    const std::shared_ptr<Model>& model,
                                    std::tuple<std::unordered_set<std::string>, bool>& existing_names) const {
    auto data = create_new_params(existing_names, model);
    auto nodes = data.as_nodes();

    PreprocessingContext context(data.m_tensor_layout);
    context.color_format() = get_tensor_data()->get_color_format();
    context.target_layout() = data.m_model_layout;
    context.model_shape() = data.m_param->get_partial_shape();
    context.target_element_type() = data.m_param->get_element_type();
    bool need_dump = nodes.size() > 1 || nodes[0].get_partial_shape() != context.model_shape() ||
                     data.m_param->get_layout() != context.target_layout() ||
                     nodes[0].get_element_type() != context.target_element_type() ||
                     get_tensor_data()->is_memory_type_set() || !get_preprocess()->actions().empty();
    if (!need_dump) {
        return;
    }
    // Dump tensor and model shapes if any preprocessing is needed
    str << "Input ";
    if (!data.m_param->output(0).get_names().empty()) {
        str << "\"" << data.m_param->output(0).get_any_name() << "\"";
    }
    if (context.color_format() != ColorFormat::UNDEFINED) {
        str << " (color " << color_format_name(context.color_format()) << ")";
    }
    if (get_tensor_data()->is_memory_type_set()) {
        str << " memory type=" << get_tensor_data()->get_memory_type();
    }
    str << ":" << std::endl;
    if (nodes.size() == 1) {
        str << "    User's input tensor: ";
        dump_tensor(str, nodes[0].get_partial_shape(), context.layout(), nodes[0].get_element_type());
        str << std::endl;
    } else {
        str << "    " << nodes.size() << " user's tensors expected for each plane:" << std::endl;
        for (size_t i = 0; i < nodes.size(); i++) {
            str << "       " << i << ": ";
            if (!nodes[i].get_names().empty()) {
                str << nodes[i].get_any_name() << " ";
            }
            dump_tensor(str, nodes[i].get_partial_shape(), context.layout(), nodes[i].get_element_type());
            str << std::endl;
        }
    }
    str << "    Model's expected tensor: ";
    dump_tensor(str, context.model_shape(), context.target_layout(), context.target_element_type());
    str << std::endl;

    // Apply and dump preprocessing operations
    if (!get_preprocess()->actions().empty()) {
        str << "    Pre-processing steps (" << get_preprocess()->actions().size() << "):" << std::endl;
    }
    for (const auto& action : get_preprocess()->actions()) {
        str << "      " << action.m_name << ": (";
        dump_tensor(str,
                    nodes[0].get_partial_shape(),
                    context.layout(),
                    nodes[0].get_element_type(),
                    context.color_format());
        auto action_result = action.m_op(nodes, model, context);
        nodes = std::get<0>(action_result);
        str << ") -> (";
        dump_tensor(str,
                    nodes[0].get_partial_shape(),
                    context.layout(),
                    nodes[0].get_element_type(),
                    context.color_format());
        str << ")" << std::endl;
    }

    // Implicit: Convert element type + layout to user's tensor implicitly
    auto implicit_steps = create_implicit_steps(context, nodes[0].get_element_type());
    if (!implicit_steps.actions().empty()) {
        str << "    Implicit pre-processing steps (" << implicit_steps.actions().size() << "):" << std::endl;
    }
    for (const auto& action : implicit_steps.actions()) {
        str << "      " << action.m_name << ": (";
        dump_tensor(str,
                    nodes[0].get_partial_shape(),
                    context.layout(),
                    nodes[0].get_element_type(),
                    context.color_format());
        auto action_result = action.m_op(nodes, model, context);
        nodes = std::get<0>(action_result);
        str << ") -> (";
        dump_tensor(str,
                    nodes[0].get_partial_shape(),
                    context.layout(),
                    nodes[0].get_element_type(),
                    context.color_format());
        str << ")" << std::endl;
    }
}

//----------- OutputInfoImpl ----------
void OutputInfo::OutputInfoImpl::build(ov::ResultVector& results) {
    std::shared_ptr<opset8::Result> result;
    auto node = m_output_node;
    const auto start_out_node_names = node.get_tensor().get_names();
    node.get_tensor().set_names({});
    result = std::dynamic_pointer_cast<opset8::Result>(node.get_node_shared_ptr());
    // Set result layout from 'model' information
    if (get_model_data()->is_layout_set()) {
        // Overwrite existing model's layout here (fix 74065)
        result->set_layout(get_model_data()->get_layout());
    }
    PostprocessingContext context(result->get_layout());
    if (get_tensor_data()->is_layout_set()) {
        context.target_layout() = get_tensor_data()->get_layout();
    }
    if (get_tensor_data()->is_element_type_set()) {
        context.target_element_type() = get_tensor_data()->get_element_type();
    }
    if (get_model_data()->is_color_format_set()) {
        context.color_format() = get_model_data()->get_color_format();
    }

    // Apply post-processing
    node = result->get_input_source_output(0);
    bool post_processing_applied = false;
    for (const auto& action : get_postprocess()->actions()) {
        auto action_result = action.m_op({node}, context);
        node = std::get<0>(action_result);
        post_processing_applied = true;
    }
    // Implicit: Convert element type + layout to user's tensor implicitly
    PostStepsList implicit_steps;
    if (node.get_element_type() != get_tensor_data()->get_element_type() && get_tensor_data()->is_element_type_set() &&
        node.get_element_type() != element::dynamic) {
        implicit_steps.add_convert_impl(get_tensor_data()->get_element_type());
    }

    if (!context.target_layout().empty() && context.target_layout() != context.layout()) {
        implicit_steps.add_convert_layout_impl(context.target_layout());
    }
    for (const auto& action : implicit_steps.actions()) {
        auto action_result = action.m_op({node}, context);
        node = std::get<0>(action_result);
        post_processing_applied = true;
    }
    // Restore tensor names
    node.get_tensor().set_names(start_out_node_names);
    auto orig_parent = result->get_input_source_output(0).get_node_shared_ptr();
    bool reset_orig_friendly_name = false;
    if (!post_processing_applied) {
        return;
    }
    if (orig_parent->get_output_size() == 1) {
        node.get_node_shared_ptr()->set_friendly_name(orig_parent->get_friendly_name());
        reset_orig_friendly_name = true;
    } else if (node.get_node_shared_ptr() != orig_parent) {
        // Result node is changed - add ".<idx>" suffix
        node.get_node_shared_ptr()->set_friendly_name(orig_parent->get_friendly_name() + "." +
                                                      std::to_string(result->get_input_source_output(0).get_index()));
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto tensor_name = ov::descriptor::get_ov_tensor_legacy_name(result->get_input_tensor(0));
    if (!tensor_name.empty()) {
        ov::descriptor::set_ov_tensor_legacy_name(node.get_tensor(), tensor_name);
    }
    OPENVINO_SUPPRESS_DEPRECATED_END

    // Reset friendly name of input node to avoid names collision
    // when there is at a new node inserted by post-processing steps
    // If no new nodes are inserted by post-processing, then we need to preserve friendly name of input
    // as it's required for old API correct work
    if (reset_orig_friendly_name) {
        result->get_input_source_output(0).get_node_shared_ptr()->set_friendly_name("");
    }

    // Create result
    auto new_result = std::make_shared<opset8::Result>(node);
    new_result->set_friendly_name(result->get_friendly_name());

    // Preserve runtime info of original result
    new_result->get_rt_info() = result->get_rt_info();
    new_result->input(0).get_rt_info() = result->input(0).get_rt_info();
    new_result->output(0).get_rt_info() = result->output(0).get_rt_info();

    // Update layout
    if (!context.layout().empty()) {
        new_result->set_layout(context.layout());
    }

    for (auto& old_result : results) {
        if (result == old_result) {
            old_result = new_result;
            break;
        }
    }
}

void OutputInfo::OutputInfoImpl::dump(std::ostream& str) const {
    std::shared_ptr<opset8::Result> result;
    auto node = m_output_node;
    const auto& start_out_node_names = node.get_tensor().get_names();
    result = std::dynamic_pointer_cast<opset8::Result>(node.get_node_shared_ptr());
    auto model_layout = get_model_data()->is_layout_set() ? get_model_data()->get_layout() : result->get_layout();
    PostprocessingContext context(model_layout);
    if (get_tensor_data()->is_layout_set()) {
        context.target_layout() = get_tensor_data()->get_layout();
    }
    if (get_tensor_data()->is_element_type_set()) {
        context.target_element_type() = get_tensor_data()->get_element_type();
    }

    bool need_dump =
        (model_layout != context.target_layout() && get_tensor_data()->is_layout_set()) ||
        (node.get_element_type() != context.target_element_type() && get_tensor_data()->is_element_type_set()) ||
        !get_postprocess()->actions().empty();
    if (!need_dump) {
        return;
    }

    str << "Output ";
    if (!start_out_node_names.empty()) {
        str << "\"" << *start_out_node_names.begin() << "\"";
    }
    str << ":" << std::endl;
    str << "    Model's data tensor: ";
    dump_tensor(str, node.get_partial_shape(), model_layout, node.get_element_type());
    str << std::endl;

    if (!get_postprocess()->actions().empty()) {
        str << "    Post-processing steps (" << get_postprocess()->actions().size() << "):" << std::endl;
    }
    // Apply post-processing
    node = result->get_input_source_output(0);
    for (const auto& action : get_postprocess()->actions()) {
        str << "      " << action.m_name << ": (";
        dump_tensor(str, node.get_partial_shape(), context.layout(), node.get_element_type());
        auto action_result = action.m_op({node}, context);
        node = std::get<0>(action_result);
        str << ") -> (";
        dump_tensor(str, node.get_partial_shape(), context.layout(), node.get_element_type());
        str << ")" << std::endl;
    }
    // Implicit: Convert element type + layout to user's tensor implicitly
    PostStepsList implicit_steps;
    if (node.get_element_type() != get_tensor_data()->get_element_type() && get_tensor_data()->is_element_type_set() &&
        node.get_element_type() != element::dynamic) {
        implicit_steps.add_convert_impl(get_tensor_data()->get_element_type());
    }

    if (!context.target_layout().empty() && context.target_layout() != context.layout()) {
        implicit_steps.add_convert_layout_impl(context.target_layout());
    }
    if (!implicit_steps.actions().empty()) {
        str << "    Post-processing implicit steps (" << implicit_steps.actions().size() << "):" << std::endl;
    }
    for (const auto& action : implicit_steps.actions()) {
        str << "      " << action.m_name << ": (";
        dump_tensor(str, node.get_partial_shape(), context.layout(), node.get_element_type());
        auto action_result = action.m_op({node}, context);
        node = std::get<0>(action_result);
        str << ") -> (";
        dump_tensor(str, node.get_partial_shape(), context.layout(), node.get_element_type());
        str << ")" << std::endl;
    }

    str << "    User's output tensor: ";
    dump_tensor(str, node.get_partial_shape(), context.layout(), node.get_element_type());
    str << std::endl;
}
}  // namespace preprocess
}  // namespace ov

//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "int_executable.hpp"
#include "backend_manager.hpp"
#include "ngraph/chrome_trace.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"
#include "ngraph/util.hpp"
#include "pass/fused_op_decomposition.hpp"
#include "pass/like_replacement.hpp"
#include "pass/liveness.hpp"
#include "pass/opset0_downgrade.hpp"
#include "pass/opset1_downgrade.hpp"
#include <algorithm>
#include <cstring>

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

runtime::interpreter::OP_TYPEID runtime::interpreter::INTExecutable::get_typeid(const Node& node)
{
    const NodeTypeInfo& type_info = node.get_type_info();
    // This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
    // {Abs::type_info, OP_TYPEID::Abs},
    // {Acos::type_info, OP_TYPEID::Acos},
    // ...
    static const map<NodeTypeInfo, OP_TYPEID> type_info_map{
#define NGRAPH_OP(NAME, NAMESPACE) {NAMESPACE::NAME::type_info, OP_TYPEID::ID_SUFFIX(NAME)},
#include "opset_int_tbl.hpp"
#undef NGRAPH_OP
    };
    OP_TYPEID rc = OP_TYPEID::UnknownOp;

    auto it = type_info_map.find(type_info);
    if (it != type_info_map.end())
    {
        rc = it->second;
    }
    return rc;
}

using V5BoxEncoding = op::v5::NonMaxSuppression::BoxEncodingType;

namespace
{
    constexpr size_t boxes_port = 0;
    constexpr size_t scores_port = 1;
    constexpr size_t max_output_boxes_port = 2;
    constexpr size_t iou_threshold_port = 3;
    constexpr size_t score_threshold_port = 4;
    constexpr size_t soft_nms_sigma_port = 5;

    PartialShape infer_selected_indices_shape(
        const std::vector<std::shared_ptr<HostTensor>>& inputs,
        int64_t max_output_boxes_per_class)
    {
        const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
        const auto scores_ps = inputs[scores_port]->get_partial_shape();

        // NonMaxSuppression produces triplets
        // that have the following format: [batch_index, class_index, box_index]
        PartialShape result = {Dimension::dynamic(), 3};

        if (boxes_ps.rank().is_static() && scores_ps.rank().is_static())
        {
            const auto num_boxes_boxes = boxes_ps[1];
            if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static())
            {
                const auto num_boxes = num_boxes_boxes.get_length();
                const auto num_classes = scores_ps[1].get_length();

                result[0] = std::min(num_boxes, max_output_boxes_per_class) * num_classes *
                            scores_ps[0].get_length();
            }
        }

        return result;
    }

    void normalize_corner(float* boxes, const Shape& boxes_shape)
    {
        size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
        for (size_t i = 0; i < total_num_of_boxes; ++i)
        {
            float* current_box = boxes + 4 * i;

            float y1 = current_box[0];
            float x1 = current_box[1];
            float y2 = current_box[2];
            float x2 = current_box[3];

            float ymin = std::min(y1, y2);
            float ymax = std::max(y1, y2);
            float xmin = std::min(x1, x2);
            float xmax = std::max(x1, x2);

            current_box[0] = ymin;
            current_box[1] = xmin;
            current_box[2] = ymax;
            current_box[3] = xmax;
        }
    }

    void normalize_center(float* boxes, const Shape& boxes_shape)
    {
        size_t total_num_of_boxes = shape_size(boxes_shape) / 4;
        for (size_t i = 0; i < total_num_of_boxes; ++i)
        {
            float* current_box = boxes + 4 * i;

            float x_center = current_box[0];
            float y_center = current_box[1];
            float width = current_box[2];
            float height = current_box[3];

            float y1 = y_center - height / 2.0;
            float x1 = x_center - width / 2.0;
            float y2 = y_center + height / 2.0;
            float x2 = x_center + width / 2.0;

            current_box[0] = y1;
            current_box[1] = x1;
            current_box[2] = y2;
            current_box[3] = x2;
        }
    }

    void normalize_box_encoding(float* boxes,
                                const Shape& boxes_shape,
                                const V5BoxEncoding box_encoding)
    {
        if (box_encoding == V5BoxEncoding::CORNER)
        {
            normalize_corner(boxes, boxes_shape);
        }
        else
        {
            normalize_center(boxes, boxes_shape);
        }
    }

    std::vector<float> get_floats(const std::shared_ptr<HostTensor>& input, const Shape& shape)
    {
        size_t input_size = shape_size(shape);
        std::vector<float> result(input_size);

        switch (input->get_element_type())
        {
        case element::Type_t::bf16:
        {
            bfloat16* p = input->get_data_ptr<bfloat16>();
            for (size_t i = 0; i < input_size; ++i)
            {
                result[i] = float(p[i]);
            }
        }
        break;
        case element::Type_t::f16:
        {
            float16* p = input->get_data_ptr<float16>();
            for (size_t i = 0; i < input_size; ++i)
            {
                result[i] = float(p[i]);
            }
        }
        break;
        case element::Type_t::f32:
        {
            float* p = input->get_data_ptr<float>();
            memcpy(result.data(), p, input_size * sizeof(float));
        }
        break;
        default: throw std::runtime_error("Unsupported data type in op NonMaxSuppression-5"); break;
        }

        return result;
    }

    std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                          const Shape& boxes_shape,
                                          const V5BoxEncoding box_encoding)
    {
        auto result = get_floats(boxes, boxes_shape);
        normalize_box_encoding(result.data(), boxes_shape, box_encoding);
        return result;
    }

    std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores,
                                           const Shape& scores_shape)
    {
        auto result = get_floats(scores, scores_shape);
        return result;
    }

    void nms_postprocessing(const std::vector<std::shared_ptr<HostTensor>>& outputs,
                            const ngraph::element::Type output_type,
                            const std::vector<int64_t>& selected_indices,
                            const std::vector<float>& selected_scores,
                            int64_t valid_outputs,
                            const ngraph::element::Type selected_scores_type)
    {
        size_t num_of_outputs = outputs.size();
        size_t selected_size = valid_outputs * 3;

        if (output_type == ngraph::element::i64)
        {
            int64_t* indices_ptr = outputs[0]->get_data_ptr<int64_t>();
            memcpy(indices_ptr, selected_indices.data(), selected_size * sizeof(int64_t));
        }
        else
        {
            int32_t* indices_ptr = outputs[0]->get_data_ptr<int32_t>();
            for (size_t i = 0; i < selected_size; ++i)
            {
                indices_ptr[i] = static_cast<int32_t>(selected_indices[i]);
            }
        }

        if (num_of_outputs < 2)
        {
            return;
        }

        size_t selected_scores_size = selected_scores.size();

        switch (selected_scores_type)
        {
        case element::Type_t::bf16:
        {
            bfloat16* scores_ptr = outputs[1]->get_data_ptr<bfloat16>();
            for (size_t i = 0; i < selected_scores_size; ++i)
            {
                scores_ptr[i] = bfloat16(selected_scores[i]);
            }
        }
        break;
        case element::Type_t::f16:
        {
            float16* scores_ptr = outputs[1]->get_data_ptr<float16>();
            for (size_t i = 0; i < selected_scores_size; ++i)
            {
                scores_ptr[i] = float16(selected_scores[i]);
            }
        }
        break;
        case element::Type_t::f32:
        {
            float* scores_ptr = outputs[1]->get_data_ptr<float>();
            memcpy(scores_ptr, selected_scores.data(), selected_size * sizeof(float));
        }
        break;
        default:;
        }

        if (num_of_outputs < 3)
        {
            return;
        }

        if (output_type == ngraph::element::i64)
        {
            int64_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int64_t>();
            *valid_outputs_ptr = valid_outputs;
        }
        else
        {
            int32_t* valid_outputs_ptr = outputs[2]->get_data_ptr<int32_t>();
            *valid_outputs_ptr = static_cast<int32_t>(valid_outputs);
        }
    }
}

void runtime::interpreter::run_nms5(
    const op::v5::NonMaxSuppression* nms5,
    const std::vector<std::shared_ptr<HostTensor>>& outputs,
    const std::vector<std::shared_ptr<HostTensor>>& inputs)
{
    int64_t max_output_boxes_per_class = nms5->max_boxes_output_from_input();
    float iou_threshold = nms5->iou_threshold_from_input();
    float score_threshold = nms5->score_threshold_from_input();
    float soft_nms_sigma = nms5->soft_nms_sigma_from_input();

    auto selected_indices_shape = infer_selected_indices_shape(inputs,
                                                               max_output_boxes_per_class);
    Shape out_shape = selected_indices_shape.to_shape();

    Shape boxes_shape = inputs[boxes_port]->get_shape();
    Shape scores_shape = inputs[scores_port]->get_shape();

    auto boxes_data = prepare_boxes_data(inputs[boxes_port], boxes_shape, nms5->get_box_encoding());
    auto scores_data = prepare_scores_data(inputs[scores_port], scores_shape);

    size_t out_shape_size = shape_size(out_shape);

    std::vector<int64_t> selected_indices(out_shape_size);
    std::vector<float> selected_scores(out_shape_size);
    int64_t valid_outputs = 0;

    runtime::reference::non_max_suppression(boxes_data.data(),
                                            boxes_shape,
                                            scores_data.data(),
                                            scores_shape,
                                            max_output_boxes_per_class,
                                            iou_threshold,
                                            score_threshold,
                                            soft_nms_sigma,
                                            selected_indices.data(),
                                            out_shape,
                                            selected_scores.data(),
                                            out_shape,
                                            &valid_outputs,
                                            nms5->get_sort_result_descending());

    auto output_type = nms5->get_output_type();

    outputs[0]->set_element_type(output_type);
    outputs[0]->set_shape(Shape{static_cast<size_t>(valid_outputs), 3});

    size_t num_of_outputs = outputs.size();

    auto selected_scores_type = (inputs.size() < 4) ? element::f32 : inputs[3]->get_element_type();

    if (num_of_outputs >= 2)
    {
        outputs[1]->set_element_type(selected_scores_type);
        outputs[1]->set_shape(Shape{static_cast<size_t>(valid_outputs), 3});
    }

    if (num_of_outputs >= 3)
    {
        outputs[2]->set_element_type(output_type);
        outputs[2]->set_shape(Shape{1});
    }

    nms_postprocessing(outputs,
                       output_type,
                       selected_indices,
                       selected_scores,
                       valid_outputs,
                       selected_scores_type);
}

runtime::interpreter::INTExecutable::INTExecutable(const shared_ptr<Function>& function,
                                                   bool enable_performance_collection)
    : m_is_compiled{true}
    , m_performance_counters_enabled{enable_performance_collection}
{
    m_function = clone_function(*function);
    auto is_supported = [](const Node& node) {
        bool retval = false;
        switch (INTExecutable::get_typeid(node))
        {
        case OP_TYPEID::Clamp:
        case OP_TYPEID::MatMul:
        case OP_TYPEID::NormalizeL2:
        case OP_TYPEID::PRelu:
        case OP_TYPEID::Squeeze:
        case OP_TYPEID::Unsqueeze: retval = true; break;
        default: break;
        }
        return retval;
    };
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::LikeReplacement>();
    pass_manager.register_pass<pass::FusedOpDecomposition>(is_supported);
    pass_manager.register_pass<pass::Opset1Downgrade>();
    pass_manager.register_pass<pass::Opset0Downgrade>();
    // Need to decompose any v0 fused ops, which were produced by the downgrade pass
    pass_manager.register_pass<pass::FusedOpDecomposition>(is_supported);
    pass_manager.run_passes(m_function);
    for (auto node : m_function->get_ordered_ops())
    {
        m_nodes.push_back(node);
    }
    set_parameters_and_results(*m_function);
}

bool runtime::interpreter::INTExecutable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                               const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    event::Duration d1("call", "Interpreter");

    // convert inputs to HostTensor
    vector<shared_ptr<HostTensor>> func_inputs;
    for (auto tensor : inputs)
    {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_inputs.push_back(host_tensor);
    }
    if (m_nan_check_enabled)
    {
        perform_nan_check(func_inputs);
    }

    // convert outputs to HostTensor
    vector<shared_ptr<HostTensor>> func_outputs;
    for (auto tensor : outputs)
    {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_outputs.push_back(host_tensor);
    }

    // map function params -> HostTensor
    unordered_map<descriptor::Tensor*, shared_ptr<HostTensor>> tensor_map;
    size_t input_count = 0;
    for (auto param : get_parameters())
    {
        for (size_t i = 0; i < param->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = &param->output(i).get_tensor();
            tensor_map.insert({tensor, func_inputs[input_count++]});
        }
    }

    // map function outputs -> HostTensor
    for (size_t output_count = 0; output_count < get_results().size(); ++output_count)
    {
        auto output = get_results()[output_count];
        if (!is_type<op::Result>(output))
        {
            throw ngraph_error("One of function's outputs isn't op::Result");
        }
        descriptor::Tensor* tensor = &output->get_output_tensor(0);
        tensor_map.insert({tensor, func_outputs[output_count]});
    }

    // for each ordered op in the graph
    for (auto op : m_nodes)
    {
        event::Duration d2(op->description(), "Interpreter");
        if (op::is_parameter(op))
        {
            continue;
        }

        // get op inputs from map
        vector<shared_ptr<HostTensor>> op_inputs;
        for (auto input : op->inputs())
        {
            descriptor::Tensor* tensor = &input.get_tensor();
            op_inputs.push_back(tensor_map.at(tensor));
        }

        // get op outputs from map or create
        vector<shared_ptr<HostTensor>> op_outputs;
        for (size_t i = 0; i < op->get_output_size(); ++i)
        {
            descriptor::Tensor* tensor = &op->output(i).get_tensor();
            shared_ptr<HostTensor> host_tensor;
            auto it = tensor_map.find(tensor);
            if (it == tensor_map.end())
            {
                host_tensor = make_shared<HostTensor>(op->output(i));
                tensor_map.insert({tensor, host_tensor});
            }
            else
            {
                host_tensor = it->second;
            }
            op_outputs.push_back(host_tensor);
        }

        // get op type
        element::Type type;
        if (is_type<op::Convert>(op) || is_type<op::Quantize>(op) || is_type<op::Dequantize>(op) ||
            is_type<op::PriorBox>(op))
        {
            type = op->get_input_element_type(0);
        }
        else if (is_type<op::Equal>(op) || is_type<op::Greater>(op) || is_type<op::GreaterEq>(op) ||
                 is_type<op::Less>(op) || is_type<op::LessEq>(op) || is_type<op::NotEqual>(op))
        {
            // Get the type of the second input, not the first
            // All BinaryElementwiseComparision ops have the same type for inputs
            // Select has bool for first input and the type we are interested in for the second
            type = op->get_input_element_type(1);
        }
        else if (is_type<op::TopK>(op))
        {
            type = op->get_output_element_type(1);
        }
        else
        {
            type = op->get_output_element_type(0);
        }

        if (m_performance_counters_enabled)
        {
            m_timer_map[op].start();
        }
        if (!op->evaluate(op_outputs, op_inputs))
        {
            generate_calls(type, *op.get(), op_outputs, op_inputs);
        }
        if (m_performance_counters_enabled)
        {
            m_timer_map[op].stop();
        }
        if (m_nan_check_enabled)
        {
            perform_nan_check(op_outputs, op.get());
        }
    }

    return true;
}

void runtime::interpreter::INTExecutable::generate_calls(const element::Type& type,
                                                         const Node& op,
                                                         const vector<shared_ptr<HostTensor>>& out,
                                                         const vector<shared_ptr<HostTensor>>& in)
{
    stringstream ss;
    switch (type)
    {
    case element::Type_t::boolean: op_engine<char>(op, out, in); break;
    case element::Type_t::f32: op_engine<float>(op, out, in); break;
    case element::Type_t::f64: op_engine<double>(op, out, in); break;
    case element::Type_t::i8: op_engine<int8_t>(op, out, in); break;
    case element::Type_t::i16: op_engine<int16_t>(op, out, in); break;
    case element::Type_t::i32: op_engine<int32_t>(op, out, in); break;
    case element::Type_t::i64: op_engine<int64_t>(op, out, in); break;
    case element::Type_t::u8: op_engine<uint8_t>(op, out, in); break;
    case element::Type_t::u16: op_engine<uint16_t>(op, out, in); break;
    case element::Type_t::u32: op_engine<uint32_t>(op, out, in); break;
    case element::Type_t::u64: op_engine<uint64_t>(op, out, in); break;
    case element::Type_t::undefined:
    case element::Type_t::dynamic:
    case element::Type_t::u1:
    case element::Type_t::bf16:
    case element::Type_t::f16:
        ss << "unsupported element type " << type << " op " << op.get_name();
        throw ngraph_error(ss.str());
    }
}

void runtime::interpreter::INTExecutable::set_nan_check(bool enable)
{
    m_nan_check_enabled = enable;
}

vector<runtime::PerformanceCounter>
    runtime::interpreter::INTExecutable::get_performance_data() const
{
    vector<runtime::PerformanceCounter> rc;
    for (const pair<shared_ptr<const Node>, stopwatch> p : m_timer_map)
    {
        rc.emplace_back(p.first, p.second.get_total_microseconds(), p.second.get_call_count());
    }
    return rc;
}

void runtime::interpreter::INTExecutable::perform_nan_check(
    const vector<shared_ptr<HostTensor>>& tensors, const Node* op)
{
    size_t arg_number = 1;
    for (const shared_ptr<HostTensor>& tensor : tensors)
    {
        const element::Type& type = tensor->get_element_type();
        if (type == element::f32)
        {
            const float* data = tensor->get_data_ptr<float>();
            for (size_t i = 0; i < tensor->get_element_count(); i++)
            {
                if (std::isnan(data[i]))
                {
                    if (op)
                    {
                        throw runtime_error("nan found in op '" + op->get_name() + "' output");
                    }
                    else
                    {
                        throw runtime_error("nan found in function's input tensor number " +
                                            to_string(arg_number));
                    }
                }
            }
        }
        else if (type == element::f64)
        {
            const double* data = tensor->get_data_ptr<double>();
            for (size_t i = 0; i < tensor->get_element_count(); i++)
            {
                if (std::isnan(data[i]))
                {
                    if (op)
                    {
                        throw runtime_error("nan found in op '" + op->get_name() + "' output");
                    }
                    else
                    {
                        throw runtime_error("nan found in function's input tensor number " +
                                            to_string(arg_number));
                    }
                }
            }
        }
        arg_number++;
    }
}

shared_ptr<ngraph::op::Parameter>
    runtime::interpreter::INTExecutable::get_parameter(size_t index) const
{
    const ParameterVector& parameters = get_parameters();
    NGRAPH_CHECK(index < parameters.size(), "create_tensor for input out of bounds");
    return parameters[index];
}

shared_ptr<ngraph::op::Result> runtime::interpreter::INTExecutable::get_result(size_t index) const
{
    const ResultVector& results = get_results();
    NGRAPH_CHECK(index < results.size(), "create_tensor for input out of bounds");
    return results[index];
}
shared_ptr<runtime::Tensor>
    runtime::interpreter::INTExecutable::create_input_tensor(size_t input_index)
{
    shared_ptr<op::Parameter> parameter = get_parameter(input_index);
    return make_shared<runtime::HostTensor>(parameter->get_element_type(), parameter->get_shape());
}

shared_ptr<runtime::Tensor>
    runtime::interpreter::INTExecutable::create_output_tensor(size_t output_index)
{
    shared_ptr<op::Result> result = get_result(output_index);
    return make_shared<runtime::HostTensor>(result->get_element_type(), result->get_shape());
}

vector<shared_ptr<runtime::Tensor>>
    runtime::interpreter::INTExecutable::create_input_tensor(size_t input_index,
                                                             size_t pipeline_depth)
{
    vector<shared_ptr<runtime::HostTensor>> tensors;
    shared_ptr<op::Parameter> parameter = get_parameter(input_index);
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        shared_ptr<runtime::HostTensor> tensor;
        auto t =
            make_shared<runtime::HostTensor>(parameter->get_element_type(), parameter->get_shape());
        tensor = static_pointer_cast<runtime::HostTensor>(t);
        tensors.push_back(tensor);
    }
    vector<shared_ptr<runtime::Tensor>> result_tensors;
    for (const shared_ptr<runtime::HostTensor>& tensor : tensors)
    {
        result_tensors.push_back(tensor);
    }
    return result_tensors;
}

vector<shared_ptr<runtime::Tensor>>
    runtime::interpreter::INTExecutable::create_output_tensor(size_t output_index,
                                                              size_t pipeline_depth)
{
    vector<shared_ptr<runtime::HostTensor>> tensors;
    shared_ptr<op::Result> result = get_result(output_index);
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        shared_ptr<runtime::HostTensor> tensor;
        auto t = make_shared<runtime::HostTensor>(result->get_element_type(), result->get_shape());
        tensor = static_pointer_cast<runtime::HostTensor>(t);
        tensors.push_back(tensor);
    }
    vector<shared_ptr<runtime::Tensor>> result_tensors;
    for (const shared_ptr<runtime::HostTensor>& tensor : tensors)
    {
        result_tensors.push_back(tensor);
    }
    return result_tensors;
}

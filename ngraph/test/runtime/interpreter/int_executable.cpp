//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include <cstring>
#include "backend_manager.hpp"
#include "evaluates_map.hpp"
#include "ngraph/chrome_trace.hpp"
#include "ngraph/except.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_SUPPRESS_DEPRECATED_START

runtime::interpreter::INTExecutable::INTExecutable(const shared_ptr<Function>& function,
                                                   bool enable_performance_collection)
    : m_is_compiled{true}
    , m_performance_counters_enabled{enable_performance_collection}
{
    m_function = clone_function(*function);
    for (const auto& node : m_function->get_ordered_ops())
    {
        // TODO: WA because of references mismatch for the operation
        if (is_type<op::v1::GroupConvolutionBackpropData>(node))
        {
            auto gr_conv_bp_data = dynamic_pointer_cast<op::v1::GroupConvolutionBackpropData>(node);
            auto num_groups = gr_conv_bp_data->input_value(1).get_shape()[0];
            auto split_filter_axis = std::make_shared<op::Constant>(
                ngraph::element::Type_t::i64, ngraph::Shape{}, std::vector<uint64_t>{0});
            auto sliced_filter = std::make_shared<op::v1::Split>(
                gr_conv_bp_data->input_value(1), split_filter_axis, num_groups);
            auto split_data_axis = std::make_shared<op::Constant>(
                ngraph::element::Type_t::i64, ngraph::Shape{}, std::vector<uint64_t>{1});
            auto sliced_data = std::make_shared<op::v1::Split>(
                gr_conv_bp_data->input_value(0), split_data_axis, num_groups);

            NodeVector convs;
            auto squeeze_filter_axis = std::make_shared<op::Constant>(
                ngraph::element::Type_t::i64, ngraph::Shape{}, std::vector<uint64_t>{0});
            for (size_t i = 0; i < num_groups; ++i)
            {
                auto squeezed_filter = std::make_shared<op::v0::Squeeze>(sliced_filter->output(i),
                                                                         squeeze_filter_axis);
                auto conv = std::make_shared<op::v1::ConvolutionBackpropData>(
                    sliced_data->output(i),
                    squeezed_filter,
                    gr_conv_bp_data->get_strides(),
                    gr_conv_bp_data->get_pads_begin(),
                    gr_conv_bp_data->get_pads_end(),
                    gr_conv_bp_data->get_dilations(),
                    gr_conv_bp_data->get_auto_pad(),
                    gr_conv_bp_data->get_output_padding());
                convs.push_back(conv);
            }
            auto concat = std::make_shared<op::Concat>(convs, 1);
            replace_node(node, concat);
        }
        else if (is_type<op::v1::GroupConvolution>(node))
        {
            auto gr_conv = dynamic_pointer_cast<op::v1::GroupConvolution>(node);
            auto num_groups = gr_conv->input_value(1).get_shape()[0];
            auto split_filter_axis = std::make_shared<op::Constant>(
                ngraph::element::Type_t::i64, ngraph::Shape{}, std::vector<uint64_t>{0});
            auto sliced_filter = std::make_shared<op::v1::Split>(
                gr_conv->input_value(1), split_filter_axis, num_groups);
            auto split_data_axis = std::make_shared<op::Constant>(
                ngraph::element::Type_t::i64, ngraph::Shape{}, std::vector<uint64_t>{1});
            auto sliced_data = std::make_shared<op::v1::Split>(
                gr_conv->input_value(0), split_data_axis, num_groups);

            NodeVector convs;
            auto squeeze_filter_axis = std::make_shared<op::Constant>(
                ngraph::element::Type_t::i64, ngraph::Shape{}, std::vector<uint64_t>{0});
            for (size_t i = 0; i < num_groups; ++i)
            {
                auto squeezed_filter = std::make_shared<op::v0::Squeeze>(sliced_filter->output(i),
                                                                         squeeze_filter_axis);
                auto conv = std::make_shared<op::v1::Convolution>(sliced_data->output(i),
                                                                  squeezed_filter,
                                                                  gr_conv->get_strides(),
                                                                  gr_conv->get_pads_begin(),
                                                                  gr_conv->get_pads_end(),
                                                                  gr_conv->get_dilations(),
                                                                  gr_conv->get_auto_pad());
                convs.push_back(conv);
            }
            auto concat = std::make_shared<op::Concat>(convs, 1);
            replace_node(node, concat);
        }
    }
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
    for (const auto& tensor : inputs)
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
    for (const auto& tensor : outputs)
    {
        auto host_tensor = static_pointer_cast<runtime::HostTensor>(tensor);
        func_outputs.push_back(host_tensor);
    }

    // map function params -> HostTensor
    unordered_map<descriptor::Tensor*, shared_ptr<HostTensor>> tensor_map;
    size_t input_count = 0;
    for (const auto& param : get_parameters())
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
    for (const auto& op : m_nodes)
    {
        event::Duration d2(op->description(), "Interpreter");
        if (dynamic_pointer_cast<op::Parameter>(op) != nullptr)
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
        if (is_type<op::Convert>(op) || is_type<op::Quantize>(op) || is_type<op::PriorBox>(op))
        {
            type = op->get_input_element_type(0);
        }
        else if (is_type<op::v1::Equal>(op) || is_type<op::v1::Greater>(op) ||
                 is_type<op::v1::GreaterEqual>(op) || is_type<op::v1::Less>(op) ||
                 is_type<op::v1::LessEqual>(op) || is_type<op::v1::NotEqual>(op))
        {
            // Get the type of the second input, not the first
            // All BinaryElementwiseComparision ops have the same type for inputs
            // Select has bool for first input and the type we are interested in for the second
            type = op->get_input_element_type(1);
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
            evaluate_node(op, op_outputs, op_inputs);
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

bool runtime::interpreter::INTExecutable::evaluate_node(const std::shared_ptr<Node>& node,
                                                        const HostTensorVector& outputs,
                                                        const HostTensorVector& inputs) const
{
    auto& map = runtime::interpreter::get_evaluators_map();
    auto it = map.find(node->get_type_info());
    bool res = false;
    if (it != map.end())
    {
        res = it->second(node, outputs, inputs);
        if (!res)
        {
            throw ngraph_error(std::string("Running evaluate method for OP ") +
                               node->get_type_info().name + std::string(" failed!"));
        }
    }
    else
    {
        throw unsupported_op(
            std::string("Interpreter backend doesn't implement evaluate method for OP ") +
            node->get_type_info().name);
    }
    return res;
}

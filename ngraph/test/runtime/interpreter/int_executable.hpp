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

#pragma once

#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <ngraph/runtime/host_tensor.hpp>
#include "backend.hpp"
#include "int_backend_visibility.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/reference/hard_sigmoid.hpp"
#include "ngraph/runtime/reference/non_max_suppression.hpp"
#include "ngraph/runtime/reference/reorg_yolo.hpp"
#include "ngraph/runtime/reference/tensor_iterator.hpp"
#include "ngraph/runtime/tensor.hpp"
#include "op/avg_pool.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            class INTBackend;
            class INTExecutable;
        } // namespace interpreter
    }     // namespace runtime
} // namespace ngraph

class INTERPRETER_BACKEND_API ngraph::runtime::interpreter::INTExecutable : public Executable
{
    friend class INTBackend;

public:
    INTExecutable(const std::shared_ptr<Function>& function,
                  bool enable_performance_collection = false);

    bool call(const std::vector<std::shared_ptr<Tensor>>& outputs,
              const std::vector<std::shared_ptr<Tensor>>& inputs) override;

    void set_nan_check(bool enable);

    std::vector<PerformanceCounter> get_performance_data() const override;

    std::shared_ptr<runtime::Tensor> create_input_tensor(size_t input_index) override;

    std::shared_ptr<runtime::Tensor> create_output_tensor(size_t output_index) override;

    std::vector<std::shared_ptr<runtime::Tensor>>
        create_input_tensor(size_t input_index, size_t pipeline_depth) override;

    std::vector<std::shared_ptr<runtime::Tensor>>
        create_output_tensor(size_t output_index, size_t pipeline_depth) override;

protected:
    std::shared_ptr<ngraph::op::Parameter> get_parameter(size_t index) const;
    std::shared_ptr<ngraph::op::Result> get_result(size_t index) const;
    bool evaluate_node(const std::shared_ptr<Node>& node,
                       const HostTensorVector& outputs,
                       const HostTensorVector& inputs) const;
    bool m_is_compiled = false;
    bool m_nan_check_enabled = false;
    bool m_performance_counters_enabled = false;
    std::shared_ptr<Function> m_function;
    std::unordered_map<std::shared_ptr<const Node>, stopwatch> m_timer_map;
    std::vector<std::shared_ptr<Node>> m_nodes;

    static void perform_nan_check(const std::vector<std::shared_ptr<HostTensor>>&,
                                  const Node* op = nullptr);
    struct InfoForNMS5
    {
        int64_t max_output_boxes_per_class;
        float iou_threshold;
        float score_threshold;
        float soft_nms_sigma;
        Shape out_shape;
        Shape boxes_shape;
        Shape scores_shape;
        std::vector<float> boxes_data;
        std::vector<float> scores_data;
        size_t out_shape_size;
        bool sort_result_descending;
        ngraph::element::Type output_type;
    };

    InfoForNMS5 get_info_for_nms5_eval(const op::v5::NonMaxSuppression* nms5,
                                       const std::vector<std::shared_ptr<HostTensor>>& inputs);

        case OP_TYPEID::LSTMCell_v0:
            if (type == element::i64 || type == element::u64)
            {
                runtime::reference::lstm_sequence<T, int64_t>(args[0]->get_data_ptr<char>(),
                                                              args[0]->get_shape(),
                                                              args[1]->get_data_ptr<char>(),
                                                              args[1]->get_shape(),
                                                              args[2]->get_data_ptr<char>(),
                                                              args[2]->get_shape(),
                                                              args[3]->get_data_ptr<char>(),
                                                              args[3]->get_shape(),
                                                              args[4]->get_data_ptr<char>(),
                                                              args[4]->get_shape(),
                                                              args[5]->get_data_ptr<char>(),
                                                              args[5]->get_shape(),
                                                              args[6]->get_data_ptr<char>(),
                                                              args[6]->get_shape(),
                                                              out[0]->get_data_ptr<char>(),
                                                              out[1]->get_data_ptr<char>(),
                                                              out[2]->get_data_ptr<char>(),
                                                              lstm_seq->get_activations()[0],
                                                              lstm_seq->get_activations()[1],
                                                              lstm_seq->get_activations()[2],
                                                              lstm_seq->get_clip(),
                                                              lstm_seq->get_direction());
            }
            else if (type == element::i32 || type == element::u32)
            {
                runtime::reference::lstm_sequence<T, int32_t>(args[0]->get_data_ptr<char>(),
            }
            else
            {
                std::stringstream ss;
                ss << "unsupported element type " << type << " op LSTMSequence";
                throw std::runtime_error(ss.str());
            }
            if (type == element::i64 || type == element::u64)
            {
                runtime::reference::gru_sequence<T, int64_t>(args[0]->get_data_ptr<char>(),
                                                             args[0]->get_shape(),
                                                             args[1]->get_data_ptr<char>(),
                                                             args[1]->get_shape(),
                                                             args[2]->get_data_ptr<char>(),
                                                             args[2]->get_shape(),
                                                             args[3]->get_data_ptr<char>(),
                                                             args[3]->get_shape(),
                                                             args[4]->get_data_ptr<char>(),
                                                             args[4]->get_shape(),
                                                             args[5]->get_data_ptr<char>(),
                                                             args[5]->get_shape(),
                                                             out[0]->get_data_ptr<char>(),
                                                             out[1]->get_data_ptr<char>(),
                                                             gru_seq->get_activations()[0],
                                                             gru_seq->get_activations()[1],
                                                             gru_seq->get_clip(),
                                                             gru_seq->get_direction(),
                                                             gru_seq->get_linear_before_reset());
            }
            else if (type == element::i32 || type == element::u32)
            {
                runtime::reference::gru_sequence<T, int32_t>(args[0]->get_data_ptr<char>(),
            }
            else
            {
                std::stringstream ss;
                ss << "unsupported element type " << type << " op GRUSequence";
                throw std::runtime_error(ss.str());
            }
            break;
        }
        case OP_TYPEID::HardSigmoid:
        {
            size_t element_cout = shape_size(node.get_output_shape(0));
            const T alpha = args[1]->get_data_ptr<const T>()[0];
            const T beta = args[2]->get_data_ptr<const T>()[0];
            runtime::reference::hard_sigmoid<T>(args[0]->get_data_ptr<const T>(),
                                                alpha,
                                                beta,
                                                out[0]->get_data_ptr<T>(),
                                                element_cout);

            if (type == element::i64 || type == element::u64)
            {
                runtime::reference::rnn_sequence<T, int64_t>(args[0]->get_data_ptr<char>(),
                                                             args[0]->get_shape(),
                                                             args[1]->get_data_ptr<char>(),
                                                             args[1]->get_shape(),
                                                             args[2]->get_data_ptr<char>(),
                                                             args[2]->get_shape(),
                                                             args[3]->get_data_ptr<char>(),
                                                             args[3]->get_shape(),
                                                             args[4]->get_data_ptr<char>(),
                                                             args[4]->get_shape(),
                                                             args[5]->get_data_ptr<char>(),
                                                             args[5]->get_shape(),
                                                             out[0]->get_data_ptr<char>(),
                                                             out[1]->get_data_ptr<char>(),
                                                             rnn_seq->get_activations()[0],
                                                             rnn_seq->get_clip(),
                                                             rnn_seq->get_direction());
            }
            else if (type == element::i32 || type == element::u32)
            {
                runtime::reference::rnn_sequence<T, int32_t>(args[0]->get_data_ptr<char>(),
            }
            else
            {
                std::stringstream ss;
                ss << "unsupported element type " << type << " op RNNSequence";
                throw std::runtime_error(ss.str());
            }
            else if (node.get_input_element_type(1) == element::i64)
            {
                reference::reverse_sequence<T, int64_t>(args[0]->get_data_ptr<const T>(),
                                                        out[0]->get_data_ptr<T>(),
                                                        node.get_input_shape(0),
                                                        reverse->get_batch_axis(),
                                                        reverse->get_sequence_axis(),
                                                        args[1]->get_data_ptr<const int64_t>());
            }
            reference::custom_evaluate_function evaluate =
                [](const std::shared_ptr<ngraph::Function>& function,
                   const HostTensorVector& inputs,
                   HostTensorVector& outputs) -> void {
                const auto& parameters = function->get_parameters();
                const auto& parametersNumber = parameters.size();
                const auto& inputsNumber = inputs.size();
                NGRAPH_CHECK(parametersNumber == inputsNumber,
                             "Got function (",
                             function->get_friendly_name(),
                             ") with ",
                             parametersNumber,
                             " parameters, but ",
                             inputsNumber,
                             " input blobs");

                auto inputTensors = std::vector<std::shared_ptr<runtime::Tensor>>{};
                for (const auto& parameter : parameters)
                                 parameter->get_friendly_name(),
                                 ") of size ",
                                 parameterSize,
                                 " bytes, but corresponding input with index ",
                                 parameterIndex,
                                 " has ",
                                 inputSize,
                                 " bytes");

                    auto tensor =
                        std::make_shared<runtime::HostTensor>(parameterType, parameterShape);
                    tensor->write(input->get_data_ptr(), parameterSize);
                    inputTensors.push_back(tensor);
                const auto& results = function->get_results();
                std::vector<std::shared_ptr<ngraph::runtime::Tensor>> outputTensors;
                outputTensors.reserve(results.size());
                for (size_t i = 0; i < results.size(); ++i)
                auto backend = runtime::Backend::create("INTERPRETER");
                auto handle = backend->compile(function);
                handle->call_with_validate(outputTensors, inputTensors);

                outputs.reserve(outputTensors.size());
                for (const auto& tensor : outputTensors)
                    outputs.push_back(host_tensor);
            };
            reference::tensor_iterator(ti.get_num_iterations(),
                                       ti.get_function(),
                                       ti.get_output_descriptions(),
                                       ti.get_input_descriptions(),
                                       out,
                                       args,
                                       evaluate);
        case OP_TYPEID::NonMaxSuppression_v5:
        {
            const op::v5::NonMaxSuppression* nms =
                static_cast<const op::v5::NonMaxSuppression*>(&node);

            auto info = get_info_for_nms5_eval(nms, args);

            std::vector<int64_t> selected_indices(info.out_shape_size);
            std::vector<float> selected_scores(info.out_shape_size);
            int64_t valid_outputs = 0;

            reference::non_max_suppression(info.boxes_data.data(),
                                           info.boxes_shape,
                                           info.scores_data.data(),
                                           info.scores_shape,
                                           info.max_output_boxes_per_class,
                                           info.iou_threshold,
                                           info.score_threshold,
                                           info.soft_nms_sigma,
                                           selected_indices.data(),
                                           info.out_shape,
                                           selected_scores.data(),
                                           info.out_shape,
                                           &valid_outputs,
                                           info.sort_result_descending);

            auto selected_scores_type =
                (args.size() < 4) ? element::f32 : args[3]->get_element_type();

            reference::nms5_postprocessing(out,
                                           info.output_type,
                                           selected_indices,
                                           selected_scores,
                                           valid_outputs,
                                           selected_scores_type);
            break;
        }
        case OP_TYPEID::Loop_v5:
};

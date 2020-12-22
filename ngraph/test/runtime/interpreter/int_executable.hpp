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
};

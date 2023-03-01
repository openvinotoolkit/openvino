// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <initializer_list>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "backend.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace runtime {
namespace interpreter {
class INTBackend;

class INTExecutable : public Executable {
    friend class INTBackend;

public:
    INTExecutable(const std::shared_ptr<ov::Model>& model);

    bool call(std::vector<ov::Tensor>& outputs, const std::vector<ov::Tensor>& inputs) override;

    ov::Tensor create_input_tensor(size_t input_index) override;

    ov::Tensor create_output_tensor(size_t output_index) override;

    std::vector<ov::Tensor> create_input_tensor(size_t input_index, size_t pipeline_depth) override;

    std::vector<ov::Tensor> create_output_tensor(size_t output_index, size_t pipeline_depth) override;

protected:
    std::shared_ptr<ov::op::v0::Parameter> get_parameter(size_t index) const;
    std::shared_ptr<ov::op::v0::Result> get_result(size_t index) const;
    bool evaluate_node(const std::shared_ptr<Node>& node,
                       ov::TensorVector& outputs,
                       const ov::TensorVector& inputs) const;
    bool m_is_compiled = false;
    std::shared_ptr<ov::Model> m_model;
    std::vector<std::shared_ptr<Node>> m_nodes;

    struct InfoForNMS5 {
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

    InfoForNMS5 get_info_for_nms5_eval(const ov::op::v5::NonMaxSuppression* nms5,
                                       const std::vector<std::shared_ptr<HostTensor>>& inputs);
};

}  // namespace interpreter
}  // namespace runtime
}  // namespace ov

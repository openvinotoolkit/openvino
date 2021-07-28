// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/preprocessing/scale_inputs.hpp"

using namespace ngraph;
using namespace ngraph::pass;

NGRAPH_RTTI_DEFINITION(ngraph::pass::MeanScalePassBase, "MeanScalePassBase", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ScalePassBase, "ScalePassBase", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ScaleInputsSingle, "ScaleInputsSingle", 0);
NGRAPH_RTTI_DEFINITION(ngraph::pass::ScaleInputsVector, "ScaleInputsVector", 0);

using ConstantCreator = std::function<std::shared_ptr<opset1::Constant>(std::shared_ptr<Node>)>;

MeanScalePassBase::MeanScalePassBase() = default;

bool MeanScalePassBase::run_on_function(std::shared_ptr<ngraph::Function> function) {
    bool updated = false;
    for (auto param : function->get_parameters()) {
        updated = process_parameter(param) || updated;
    }
    return updated;
}

bool MeanScalePassBase::process_parameter(const std::shared_ptr<Node>& node) const {
    auto consumers = node->output(0).get_target_inputs();
    auto constant = createConstant(node);
    constant->set_friendly_name(constantFriendlyName(node->get_friendly_name()));
    auto new_op = createOp(node, constant);
    for (auto consumer : consumers) {
        consumer.replace_source_output(new_op);
    }
    return true;
}

/// Calculate features dimension index based on input info
/// If initial_idx is not -1, use it if dimension equals to values_size
/// E.g. node_shape = {1,3,224,224}, scale_size=3 ==> Result will be "1" - dimension #1 is a 'features index'
int MeanScalePassBase::guess_features_dim_idx(const std::shared_ptr<Node>& matched,
                                              size_t values_size,
                                              int initial_idx) const {
    auto param_shape = matched->get_output_partial_shape(0);
    if (values_size == 1) {
        // Single scale value is always fine for any shape
        return 0;
    } else if (param_shape.rank().is_dynamic()) {
        throw ngraph_error("Scale of full dynamic input is not supported: " + matched->get_friendly_name());
    } else if (initial_idx >= 0) {
        if (initial_idx < param_shape.rank().get_length() &&
                param_shape[initial_idx].is_static() &&
                values_size == static_cast<size_t>(param_shape[initial_idx].get_length())) {
            return initial_idx;
        } else {
            throw ngraph_error("Feature dimension index " + std::to_string(initial_idx) +
                               " is invalid for node " + matched->get_friendly_name());
        }
    } else { // initial_idx is not specified
        auto rank_length = param_shape.rank().get_length();
        int found = 0;
        int foundIndex = -1;
        for (auto i = 0; i < rank_length; i++) {
            if (param_shape[i].is_static() &&
                    values_size == static_cast<size_t>(param_shape[i].get_length())) {
                found++;
                foundIndex = i;
            }
        }
        if (found == 1) {
            return foundIndex;
        } else {
            // Raise an exception, not clear how to calculate constant shape for inputs like {1,3,3,3}
            throw ngraph_error(
                    "Not clear how to apply scale vector to input " + matched->get_friendly_name());
        }
    }
}

//------------------------------------------

ScalePassBase::ScalePassBase() = default;

std::shared_ptr<ngraph::Node> ScalePassBase::createOp(const std::shared_ptr<ngraph::Node> &param,
                             const std::shared_ptr<ngraph::opset1::Constant> &constant) const {
    auto new_op = std::make_shared<ngraph::opset1::Multiply>(param, constant);
    new_op->set_friendly_name(param->get_friendly_name() + "/scale/Fused_Mul");
    return new_op;
}

std::string ScalePassBase::constantFriendlyName(const std::string& paramName) const {
    return paramName + "/scale/Fused_Mul_Factor";
}

//----------------------------------------
ScaleInputsSingle::ScaleInputsSingle(float scale_factor):
    ScalePassBase(),
    m_scale_factor(scale_factor) {
}

std::shared_ptr<ngraph::opset1::Constant> ScaleInputsSingle::createConstant(
        const std::shared_ptr<ngraph::Node>&) const {
    return opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.f / m_scale_factor});
}

//----------------------------------------

ScaleInputsVector::ScaleInputsVector(const std::map<std::string, std::vector<float>> &scale_map, int features_dim_idx):
        ScalePassBase(),
        m_scale_map(scale_map),
        m_features_dim_idx(features_dim_idx) {
}

bool ScaleInputsVector::process_parameter(const std::shared_ptr<Node>& param) const {
    if (m_scale_map.count(param->get_friendly_name())) {
        return ScalePassBase::process_parameter(param);
    }
    return false; // param is not specified in scale map, don't update anything
}

std::shared_ptr<ngraph::opset1::Constant> ScaleInputsVector::createConstant(
        const std::shared_ptr<ngraph::Node>& param) const {
    auto values = m_scale_map.at(param->get_friendly_name());
    auto features_dim_idx = guess_features_dim_idx(param, values.size(), m_features_dim_idx);
    auto param_shape = param->get_output_partial_shape(0);
    if (param_shape.rank().is_dynamic()) {
        // Here we ensure that values.size()=1;
        return opset1::Constant::create(ngraph::element::f32, {1}, {1.f / values[0]});
    }
    std::vector<size_t> v(param_shape.rank().get_length(), 1);
    // Calculate shape of 'constant' based on node's partial shape and 'features dimension index'
    // E.g. node_shape = {1,3,224,224}, scale_size=3 ==> constant shape will be {1,3,1,1}
    ngraph::Shape constShape(v);
    constShape[features_dim_idx] = values.size();
    std::transform(values.begin(), values.end(), values.begin(), [](float val) -> float {
        return 1.f / val;
    });
    return opset1::Constant::create(ngraph::element::f32, constShape, values);
}

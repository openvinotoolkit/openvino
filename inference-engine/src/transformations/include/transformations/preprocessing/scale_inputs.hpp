// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/pass.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API MeanScalePassBase;
class TRANSFORMATIONS_API ScalePassBase;
class TRANSFORMATIONS_API ScaleInputsSingle;
class TRANSFORMATIONS_API ScaleInputsVector;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief Base class for mean/scale passes
 */
class ngraph::pass::MeanScalePassBase : public ngraph::pass::FunctionPass {
public:
    enum class Version {
        IR_V10
    };
    NGRAPH_RTTI_DECLARATION;

    MeanScalePassBase();

    bool run_on_function(std::shared_ptr<ngraph::Function> function) override;
protected:
    int guess_features_dim_idx(const std::shared_ptr<Node>& param, size_t values_size, int initial_idx = -1) const;
    virtual bool process_parameter(const std::shared_ptr<Node>& param) const;
    virtual std::shared_ptr<ngraph::opset1::Constant> createConstant(
            const std::shared_ptr<ngraph::Node>& param) const = 0;
    virtual std::string constantFriendlyName(const std::string& paramName) const = 0;
    virtual std::shared_ptr<ngraph::Node> createOp(
            const std::shared_ptr<ngraph::Node>& param,
            const std::shared_ptr<ngraph::opset1::Constant>& constant) const = 0;
};

class ngraph::pass::ScalePassBase : public ngraph::pass::MeanScalePassBase {
public:
    enum class Version {
        IR_V10
    };
    NGRAPH_RTTI_DECLARATION;

    ScalePassBase();
protected:
    std::shared_ptr<ngraph::Node> createOp(
            const std::shared_ptr<ngraph::Node>& param,
            const std::shared_ptr<ngraph::opset1::Constant>& constant) const override;

    std::string constantFriendlyName(const std::string& paramName) const override;
};

class ngraph::pass::ScaleInputsSingle : public ngraph::pass::ScalePassBase {
public:
    enum class Version {
        IR_V10
    };
    NGRAPH_RTTI_DECLARATION;

    ScaleInputsSingle(float scale_factor = 1.f);

private:
    std::shared_ptr<ngraph::opset1::Constant> createConstant(const std::shared_ptr<ngraph::Node>&) const override;

private:
    float m_scale_factor;
};

class ngraph::pass::ScaleInputsVector : public ngraph::pass::ScalePassBase {
public:
    enum class Version {
        IR_V10
    };
    NGRAPH_RTTI_DECLARATION;

    ScaleInputsVector(const std::map<std::string, std::vector<float>> &scale_map, int features_dim_idx = -1);

private:
    bool process_parameter(const std::shared_ptr<Node>& param) const override;
    std::shared_ptr<ngraph::opset1::Constant> createConstant(const std::shared_ptr<ngraph::Node>&) const override;

private:
    std::map<std::string, std::vector<float>> m_scale_map;
    int m_features_dim_idx;
};
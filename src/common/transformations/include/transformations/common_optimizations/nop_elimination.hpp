// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pass/pass.hpp>
#include <transformations_visibility.hpp>
#include <vector>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API EliminatePad;
class TRANSFORMATIONS_API EliminateConvert;
class TRANSFORMATIONS_API EliminateConvertNonZero;
class TRANSFORMATIONS_API EliminateConcat;
class TRANSFORMATIONS_API EliminateSplit;
class TRANSFORMATIONS_API EliminateSqueeze;
class TRANSFORMATIONS_API EliminateTranspose;
class TRANSFORMATIONS_API EliminateEltwise;
class TRANSFORMATIONS_API NopElimination;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminatePad eliminates pad that does nothing
 */
class ngraph::pass::EliminatePad : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminatePad", "0");
    EliminatePad();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateConvert eliminates convert that does nothing
 */
class ngraph::pass::EliminateConvert : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateConvert", "0");
    EliminateConvert();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateConvertNonZero eliminates convert before NonZero
 */
class ngraph::pass::EliminateConvertNonZero : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateConvertNonZero", "0");
    EliminateConvertNonZero();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateConcat eliminates concat that does nothing
 */
class ngraph::pass::EliminateConcat : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateConcat", "0");
    EliminateConcat();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateSplit eliminates split that does nothing
 */
class ngraph::pass::EliminateSplit : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateSplit", "0");
    EliminateSplit();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateSqueeze eliminates squeeze that does nothing
 */
class ngraph::pass::EliminateSqueeze : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateSqueeze", "0");
    EliminateSqueeze();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateTranspose eliminates transpose that does nothing
 */
class ngraph::pass::EliminateTranspose : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateTranspose", "0");
    EliminateTranspose();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateEltwise eliminates eltwise ops that do nothing
 */
class ngraph::pass::EliminateEltwise : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateEltwise", "0");
    EliminateEltwise();
};

class ngraph::pass::NopElimination : public GraphRewrite {
public:
    OPENVINO_RTTI("NopElimination", "0");
    NopElimination(bool use_shape_for_elimination = true);
};

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <openvino/core/ov_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/pass.hpp>

namespace ngraph {
namespace pass {

class OPENVINO_API EliminatePad;
class OPENVINO_API EliminateConvert;
class OPENVINO_API EliminateConvertNonZero;
class OPENVINO_API EliminateConcat;
class OPENVINO_API EliminateSplit;
class OPENVINO_API EliminateTranspose;
class OPENVINO_API NopElimination;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminatePad eliminates pad that does nothing
 */
class ngraph::pass::EliminatePad: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminatePad();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateConvert eliminates convert that does nothing
 */
class ngraph::pass::EliminateConvert: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateConvert();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateConvertNonZero eliminates convert before NonZero
 */
class ngraph::pass::EliminateConvertNonZero: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateConvertNonZero();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateConcat eliminates concat that does nothing
 */
class ngraph::pass::EliminateConcat: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateConcat();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateSplit eliminates split that does nothing
 */
class ngraph::pass::EliminateSplit: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateSplit();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateTranspose eliminates transpose that does nothing
 */
class ngraph::pass::EliminateTranspose: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateTranspose();
};


class ngraph::pass::NopElimination: public GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    NopElimination(bool use_shape_for_elimination = true);
};

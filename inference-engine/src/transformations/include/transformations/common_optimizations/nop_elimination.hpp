// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>

#include <ngraph/ngraph.hpp>
#include <ngraph/pass/pass.hpp>

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EliminatePad;
class TRANSFORMATIONS_API EliminateConvert;
class TRANSFORMATIONS_API EliminateConvertNonZero;
class TRANSFORMATIONS_API EliminateConcat;
class TRANSFORMATIONS_API EliminateSplit;
class TRANSFORMATIONS_API EliminateTranspose;
class TRANSFORMATIONS_API NopElimination;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminatePad eliminates pad that does nothing
 */
class ov::pass::EliminatePad: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminatePad();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateConvert eliminates convert that does nothing
 */
class ov::pass::EliminateConvert: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateConvert();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateConvertNonZero eliminates convert before NonZero
 */
class ov::pass::EliminateConvertNonZero: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateConvertNonZero();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateConcat eliminates concat that does nothing
 */
class ov::pass::EliminateConcat: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateConcat();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateSplit eliminates split that does nothing
 */
class ov::pass::EliminateSplit: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateSplit();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief EliminateTranspose eliminates transpose that does nothing
 */
class ov::pass::EliminateTranspose: public ov::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    EliminateTranspose();
};


class ov::pass::NopElimination: public GraphRewrite {
public:
    NGRAPH_RTTI_DECLARATION;
    NopElimination(bool use_shape_for_elimination = true);
};

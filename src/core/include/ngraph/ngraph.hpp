// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// The public API for ngraph++
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <string>

#include "ngraph/deprecated.hpp"
#include "ngraph/version.hpp"

#ifdef IN_OV_CORE_LIBRARY
#    error("ngraph.hpp is for external use only")
#endif

#include <ngraph/ngraph_visibility.hpp>

/// \namespace ngraph
/// \brief The Intel nGraph C++ API.

/// \namespace ngraph::descriptor
/// \brief Descriptors are compile-time representations of objects that will appear at run-time.

/// \namespace ngraph::descriptor::layout
/// \brief Layout descriptors describe how tensor views are implemented.

/// \namespace ngraph::op
/// \brief Ops used in graph-building.

/// \namespace ngraph::runtime
/// \brief The objects used for executing the graph.

/// \namespace ngraph::builder
/// \brief Convenience functions that create addional graph nodes to implement commonly-used
///        recipes, for example auto-broadcast.

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/dimension.hpp"
#include "ngraph/evaluator.hpp"
#include "ngraph/except.hpp"
#include "ngraph/factory.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/rt_info.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/specialize_function.hpp"
#include "ngraph/type.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/validation_util.hpp"

// nGraph opsets
#include "ngraph/opsets/opset.hpp"

// nGraph passes
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"

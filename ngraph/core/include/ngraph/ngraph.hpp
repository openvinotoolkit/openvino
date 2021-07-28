// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//
// The public API for ngraph++
//

#pragma once

#include <string>

#ifdef IN_NGRAPH_LIBRARY
#error("ngraph.hpp is for external use only")
#endif

#include <ngraph/ngraph_visibility.hpp>

extern "C" NGRAPH_API const char* get_ngraph_version_string();

namespace ngraph
{
    /// \brief Function to query parsed version information of the version of ngraph which
    /// contains this function. Version information strictly follows Semantic Versioning
    /// http://semver.org
    /// \param major Returns the major part of the version
    /// \param minor Returns the minor part of the version
    /// \param patch Returns the patch part of the version
    /// \param extra Returns the extra part of the version. This includes everything following
    /// the patch version number.
    ///
    /// \note Throws a runtime_error if there is an error during parsing
    NGRAPH_API
    void get_version(size_t& major, size_t& minor, size_t& patch, std::string& extra);
} // namespace ngraph

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
#include "ngraph/variant.hpp"

// nGraph opsets
#include "ngraph/opsets/opset.hpp"

// nGraph passes
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"

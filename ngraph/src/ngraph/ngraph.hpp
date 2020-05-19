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
}

/// \namespace ngraph
/// \brief The Intel Nervana Graph C++ API.

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
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/dequantize_builder.hpp"
#include "ngraph/builder/numpy_transpose.hpp"
#include "ngraph/builder/quantize_builder.hpp"
#include "ngraph/builder/quantized_concat_builder.hpp"
#include "ngraph/builder/quantized_conv_builder.hpp"
#include "ngraph/builder/quantized_dot_builder.hpp"
#include "ngraph/builder/reduce_ops.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/builder/tensor_mask.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/descriptor/layout/tensor_layout.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/dimension.hpp"
#include "ngraph/except.hpp"
#include "ngraph/factory.hpp"
#include "ngraph/function.hpp"
#include "ngraph/lambda.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/specialize_function.hpp"
#include "ngraph/type.hpp"
#include "ngraph/type/element_type.hpp"

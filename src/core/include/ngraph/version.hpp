// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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
#include "ngraph/ngraph_visibility.hpp"

NGRAPH_EXTERN_C NGRAPH_API const char* NGRAPH_VERSION_NUMBER;

// clang-format off
extern "C" NGRAPH_API
NGRAPH_API_DEPRECATED
const char* get_ngraph_version_string();
// clang-format on

namespace ngraph {
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
NGRAPH_API_DEPRECATED
void get_version(size_t& major, size_t& minor, size_t& patch, std::string& extra);
}  // namespace ngraph

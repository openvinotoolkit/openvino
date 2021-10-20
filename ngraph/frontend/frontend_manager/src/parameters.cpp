// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend_manager/parameters.hpp"

using namespace ngraph;

BWDCMP_RTTI_DEFINITION(AttributeAdapter<std::istream*>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<std::istringstream*>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<Weights>);
BWDCMP_RTTI_DEFINITION(AttributeAdapter<Extensions>);

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
BWDCMP_RTTI_DEFINITION(AttributeAdapter<std::wstring>);
#endif

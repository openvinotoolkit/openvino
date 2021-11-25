// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common/parameters.hpp"

BWDCMP_RTTI_DEFINITION(ov::VariantWrapper<ov::Weights>);
BWDCMP_RTTI_DEFINITION(ov::VariantWrapper<ov::Extensions>);
BWDCMP_RTTI_DEFINITION(ov::VariantWrapper<std::istream*>);
BWDCMP_RTTI_DEFINITION(ov::VariantWrapper<std::istringstream*>);

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
BWDCMP_RTTI_DEFINITION(ov::VariantWrapper<std::wstring>);
#endif

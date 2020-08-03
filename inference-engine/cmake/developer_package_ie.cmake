# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(cpplint)
include(clang_format)

set(TBB_FIND_RELEASE_ONLY ${ENABLE_TBB_RELEASE_ONLY})

include(plugins/plugins)
include(add_ie_target)

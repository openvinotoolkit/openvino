// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>

#include "ngraph/provenance.hpp"

namespace ngraph
{
    static bool s_provenance_enabled = getenv_bool("NGRAPH_PROVENANCE_ENABLE");

    void set_provenance_enabled(bool enabled) { s_provenance_enabled = enabled; }
    bool get_provenance_enabled() { return s_provenance_enabled; }
} // namespace ngraph

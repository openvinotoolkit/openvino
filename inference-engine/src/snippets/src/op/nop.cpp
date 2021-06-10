// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/nop.hpp"
#include "snippets/generator.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(snippets::op::Nop, "Nop", 0);

snippets::op::Nop::Nop(const OutputVector& arguments, const OutputVector& results) : Op([arguments, results]() -> OutputVector {
    OutputVector x;
    x.insert(x.end(), arguments.begin(), arguments.end());
    x.insert(x.end(), results.begin(), results.end());
    return x;
    }()) {
}

NGRAPH_RTTI_DEFINITION(snippets::op::Tile, "Tile", 0);

snippets::op::Tile::Tile(const std::vector<std::pair<std::shared_ptr<snippets::Emitter>, snippets::RegInfo>>& nested) : Op(), region(nested) {
}

NGRAPH_RTTI_DEFINITION(snippets::op::Kernel, "Kernel", 0);

snippets::op::Kernel::Kernel(const std::vector<std::pair<std::shared_ptr<snippets::Emitter>, snippets::RegInfo>>& nested) : Op(), region(nested) {
}

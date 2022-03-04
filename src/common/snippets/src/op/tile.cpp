// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/op/tile.hpp"
#include "snippets/generator.hpp"

using namespace std;
using namespace ngraph;

snippets::op::Tile::Tile(const std::vector<std::pair<std::shared_ptr<snippets::Emitter>, snippets::RegInfo>>& nested) : Op(), region(nested) {
}

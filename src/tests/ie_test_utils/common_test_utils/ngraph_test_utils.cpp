// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_test_utils.hpp"

void init_unique_names(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::pass::UniqueNamesHolder>& unh) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitUniqueNames>(unh);
    manager.run_passes(f);
}

void check_unique_names(std::shared_ptr<ngraph::Function> f, const std::shared_ptr<ngraph::pass::UniqueNamesHolder>& unh) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::CheckUniqueNames>(unh, true);
    manager.run_passes(f);
}

// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "debug_new_pass.hpp"

#ifdef DEBUG_VISUALIZE
#include "ngraph/pass/visualize_tree.hpp" // DEBUG
//#include "openvino/pass/serialize.hpp" // DEBUG
#include <sstream>
#endif


namespace intel_gna_debug {

void DebugVisualize(ov::pass::Manager& manager, const std::string& name) {
#ifdef DEBUG_VISUALIZE
    static unsigned counter = 0;
    std::stringstream ss;
#ifdef DEBUG_VISUALIZETREE
    ss << counter << "_" << name << ".png";
    manager.register_pass<ov::pass::VisualizeTree>(ss.str());
#else
    ss << counter << "_" << name;
    manager.register_pass<ov::pass::Serialize>(ss.str() + ".xml", ss.str() + ".bin");
#endif
    ++counter;
#endif
}

} // namespace intel_gna_debug

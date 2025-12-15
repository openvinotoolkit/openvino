// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <assert.h>

#include <iostream>
#include <ostream>
#include <istream>
#include <sstream>
#include <vector>

#include "test_single_sections.hpp"

int main() {

    test_simple_cre_section();
    test_simple_cre_section_non_owning();

    test_simple_elf_section();
    test_simple_elf_section_non_owning();

    test_simple_ws_section();
    test_simple_ws_section_non_owning();

    test_simple_bs_section();
    test_simple_bs_section_non_owning();

    test_simple_unknown_section();
    test_simple_layouts_section();
    test_simple_layouts_section_non_owning();

    // std::stringstream stream;

    // write_metadata(stream);

    // std::vector<uint64_t> numInits = {1, 2, 3, 4};
    // std::shared_ptr<CapabilityWeightsSeparation> ws_write = std::make_shared<CapabilityWeightsSeparation>(1, numInits);

    // std::shared_ptr<CapabilityBatchSize> bs_write = std::make_shared<CapabilityBatchSize>(10);

    // // TODO: with some prints, extract some data from a blob for CapabilityInputOutputLayouts

    // caps.push_back(ws_write);
    // caps.push_back(bs_write);

    // write_capabilities(stream);

    // // thought: is it worth having it written to a file? for testing purposes

    // std::cout << stream.rdbuf() << "\n\n";
    
    // read_and_validate_metadata(stream);

    // caps.clear();

    // readCapabilities(stream, caps);

    return 0;
}

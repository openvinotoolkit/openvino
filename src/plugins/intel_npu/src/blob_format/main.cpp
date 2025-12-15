// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "test_single_sections.hpp"
#include "test_blob_e2e.hpp"

int main() {

    // Standalone section tests
    // Check that content is preserved upon serialize + read 
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

    // Tests that write/read multiple sections + blob header
    test_blob_with_header_but_no_sections();

    test_blob_cre_elf();

    test_blob_cre_unknown_ws_bs_layouts();
    test_blob_cre_unknown_ws_bs_layouts_non_owning();

    return 0;
}

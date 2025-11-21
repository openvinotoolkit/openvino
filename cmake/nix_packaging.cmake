# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Nix-specific packaging configuration for OpenVINO
# This file provides better control over test building in Nix environments

# Detect Nix build environment
if(DEFINED ENV{IN_NIX_SHELL} OR DEFINED ENV{NIX_BUILD_TOP})
    set(OPENVINO_NIX_BUILD TRUE)
    message(STATUS "Detected Nix build environment")
endif()

# Nix-specific test configuration
if(OPENVINO_NIX_BUILD)
    # In Nix builds, we want more granular control over tests
    # Allow building test binaries without running them
    option(ENABLE_NIX_TEST_BUILD "Build test binaries in Nix environment (without running)" ON)
    option(ENABLE_NIX_TEMPLATE_TESTS "Build template plugin tests in Nix environment" OFF)
    
    # Override functional tests behavior for Nix
    if(ENABLE_NIX_TEST_BUILD AND NOT ENABLE_NIX_TEMPLATE_TESTS)
        # Disable template functional tests specifically for Nix
        set(ENABLE_TEMPLATE_FUNCTIONAL_TESTS_OVERRIDE OFF)
        message(STATUS "Nix build: Template functional tests disabled")
    endif()
endif()

# Function to conditionally enable template tests based on environment
function(ov_nix_conditional_template_tests)
    if(OPENVINO_NIX_BUILD AND NOT ENABLE_NIX_TEMPLATE_TESTS)
        set(ENABLE_TEMPLATE_FUNCTIONAL_TESTS_OVERRIDE OFF PARENT_SCOPE)
        message(STATUS "Nix build: Template functional tests conditionally disabled")
    endif()
endfunction()

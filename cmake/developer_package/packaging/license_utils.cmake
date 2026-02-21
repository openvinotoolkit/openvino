# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Helper function to convert plain text LICENSE to RTF format for WiX
function(ov_generate_rtf_license source_license output_rtf)
    if(NOT EXISTS "${source_license}")
        message(WARNING "License file not found: ${source_license}")
        return()
    endif()
    
    # Read the plain text license file
    file(READ "${source_license}" license_content)
    
    # Escape special RTF characters: backslash, braces
    string(REPLACE "\\" "\\\\" license_content "${license_content}")
    string(REPLACE "{" "\\{" license_content "${license_content}")
    string(REPLACE "}" "\\}" license_content "${license_content}")
    
    # Convert newlines to RTF paragraph breaks
    string(REPLACE "\n" "\\par\n" license_content "${license_content}")
    
    # Create RTF document with minimal formatting
    set(rtf_header "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil\\fcharset0 Courier New;}}")
    set(rtf_content "\\f0\\fs20\n${license_content}")
    set(rtf_footer "}")
    
    # Write the RTF file
    file(WRITE "${output_rtf}" "${rtf_header}\n${rtf_content}\n${rtf_footer}")
    message(STATUS "Generated RTF license: ${output_rtf}")
endfunction()

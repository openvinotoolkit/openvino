# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(UNIX AND ENABLE_ERROR_HIGHLIGHT)
    function(message)
        string(ASCII 27 ESC)
        set(RESET  "${ESC}[m")
        set(RED    "${ESC}[31;1m")
        set(YELLOW "${ESC}[33;1m")

        list(GET ARGV 0 MessageType)
        if(MessageType STREQUAL FATAL_ERROR OR MessageType STREQUAL SEND_ERROR)
            list(REMOVE_AT ARGV 0)
            _message(${MessageType} "${RED}${ARGV}${RESET}")
        elseif(MessageType STREQUAL WARNING)
            list(REMOVE_AT ARGV 0)
            _message(${MessageType} "${YELLOW}${ARGV}${RESET}")
        else()
            _message("${ARGV}")
        endif()
    endfunction()
endif()

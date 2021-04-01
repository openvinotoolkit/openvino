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
        list(REMOVE_AT ARGV 0)

        foreach(arg IN LISTS ARGV)
            set(_msg "${_msg}${arg}")
        endforeach()

        if(MessageType STREQUAL FATAL_ERROR OR MessageType STREQUAL SEND_ERROR)
            _message(${MessageType} "${RED}${_msg}${RESET}")
        elseif(MessageType STREQUAL WARNING)
            _message(${MessageType} "${YELLOW}${_msg}${RESET}")
        else()
            _message(${MessageType} "${_msg}")
        endif()
    endfunction()
endif()

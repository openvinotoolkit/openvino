# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(COMMAND init_rh_decoder)
    init_rh_decoder()

    ie_add_sample(NAME speech_recognition_offline_demo
                  SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
                  HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/speech_sample.hpp"
                  INCLUDE_DIRECTORIES "${libRH_Decoder_INCLUDE_DIRS}"
                  DEPENDENCIES "${libRH_Decoder_LIBRARIES}")
endif()
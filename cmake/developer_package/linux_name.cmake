# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(target_flags)

if (LINUX)
    function(get_linux_name res_var)
        if (NOT EXISTS "/etc/lsb-release")
            execute_process(COMMAND find -L /etc/ -maxdepth 1 -type f -name *-release -exec cat {} \;
                    OUTPUT_VARIABLE release_data RESULT_VARIABLE result)
            string(REPLACE "Red Hat" "CentOS" release_data "${release_data}")
            set(name_regex "NAME=\"([^ \"\n]*).*\"\n")
            set(version_regex "VERSION=\"([0-9]+(\\.[0-9]+)?)[^\n]*\"")
        else ()
            # linux version detection using cat /etc/lsb-release
            file(READ "/etc/lsb-release" release_data)
            set(name_regex "DISTRIB_ID=([^ \n]*)\n")
            set(version_regex "DISTRIB_RELEASE=([0-9]+(\\.[0-9]+)?)")
        endif ()

        string(REGEX MATCH ${name_regex} name ${release_data})
        set(os_name ${CMAKE_MATCH_1})

        string(REGEX MATCH ${version_regex} version ${release_data})
        set(os_name "${os_name} ${CMAKE_MATCH_1}")

        if (os_name)
            set(${res_var} ${os_name} PARENT_SCOPE)
        else ()
            set(${res_var} NOTFOUND PARENT_SCOPE)
        endif ()
    endfunction()
else()
    function(get_linux_name res_var)
        set(${res_var} NOTFOUND PARENT_SCOPE)
    endfunction()
endif ()

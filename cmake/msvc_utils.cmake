macro(configure_msvc_runtime)
    if(MSVC)
        set(MSVC_RUNTIME_STATIC ON CACHE BOOL "Static linking of MSVC runtime libraries.")
        if(${MSVC_RUNTIME_STATIC})
            message(STATUS "MSVC -> Forcing use of statically-linked runtime.")
            foreach(flag_var
                    CMAKE_C_FLAGS
                    CMAKE_C_FLAGS_DEBUG
                    CMAKE_C_FLAGS_MINSIZEREL
                    CMAKE_C_FLAGS_RELEASE
                    CMAKE_C_FLAGS_RELWITHDEBINFO
                    CMAKE_CXX_FLAGS
                    CMAKE_CXX_FLAGS_DEBUG
                    CMAKE_CXX_FLAGS_MINSIZEREL
                    CMAKE_CXX_FLAGS_RELEASE
                    CMAKE_CXX_FLAGS_RELWITHDEBINFO)
                    string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
            endforeach()
        else()
            message(STATUS "MSVC -> Use of dynamically-linked runtime.")
        endif()
    endif()
endmacro()

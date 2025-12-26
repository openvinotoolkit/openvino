# Load the debug and release variables
file(GLOB DATA_FILES "${CMAKE_CURRENT_LIST_DIR}/xbyak-*-data.cmake")

foreach(f ${DATA_FILES})
    include(${f})
endforeach()

# Create the targets for all the components
foreach(_COMPONENT ${xbyak_COMPONENT_NAMES} )
    if(NOT TARGET ${_COMPONENT})
        add_library(${_COMPONENT} INTERFACE IMPORTED)
        message(${xbyak_MESSAGE_MODE} "Conan: Component target declared '${_COMPONENT}'")
    endif()
endforeach()

if(NOT TARGET xbyak::xbyak)
    add_library(xbyak::xbyak INTERFACE IMPORTED)
    message(${xbyak_MESSAGE_MODE} "Conan: Target declared 'xbyak::xbyak'")
endif()
# Load the debug and release library finders
file(GLOB CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/xbyak-Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()
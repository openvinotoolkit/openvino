# Load the debug and release variables
file(GLOB DATA_FILES "${CMAKE_CURRENT_LIST_DIR}/TBB-*-data.cmake")

foreach(f ${DATA_FILES})
    include(${f})
endforeach()

# Create the targets for all the components
foreach(_COMPONENT ${onetbb_COMPONENT_NAMES} )
    if(NOT TARGET ${_COMPONENT})
        add_library(${_COMPONENT} INTERFACE IMPORTED)
        message(${TBB_MESSAGE_MODE} "Conan: Component target declared '${_COMPONENT}'")
    endif()
endforeach()

if(NOT TARGET onetbb::onetbb)
    add_library(onetbb::onetbb INTERFACE IMPORTED)
    message(${TBB_MESSAGE_MODE} "Conan: Target declared 'onetbb::onetbb'")
endif()
# Load the debug and release library finders
file(GLOB CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/TBB-Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()
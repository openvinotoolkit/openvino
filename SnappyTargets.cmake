# Load the debug and release variables
file(GLOB DATA_FILES "${CMAKE_CURRENT_LIST_DIR}/Snappy-*-data.cmake")

foreach(f ${DATA_FILES})
    include(${f})
endforeach()

# Create the targets for all the components
foreach(_COMPONENT ${snappy_COMPONENT_NAMES} )
    if(NOT TARGET ${_COMPONENT})
        add_library(${_COMPONENT} INTERFACE IMPORTED)
        message(${Snappy_MESSAGE_MODE} "Conan: Component target declared '${_COMPONENT}'")
    endif()
endforeach()

if(NOT TARGET Snappy::snappy)
    add_library(Snappy::snappy INTERFACE IMPORTED)
    message(${Snappy_MESSAGE_MODE} "Conan: Target declared 'Snappy::snappy'")
endif()
# Load the debug and release library finders
file(GLOB CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/Snappy-Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()
# Load the debug and release variables
file(GLOB DATA_FILES "${CMAKE_CURRENT_LIST_DIR}/pybind11-*-data.cmake")

foreach(f ${DATA_FILES})
    include(${f})
endforeach()

# Create the targets for all the components
foreach(_COMPONENT ${pybind11_COMPONENT_NAMES} )
    if(NOT TARGET ${_COMPONENT})
        add_library(${_COMPONENT} INTERFACE IMPORTED)
        message(${pybind11_MESSAGE_MODE} "Conan: Component target declared '${_COMPONENT}'")
    endif()
endforeach()

if(NOT TARGET pybind11_all_do_not_use)
    add_library(pybind11_all_do_not_use INTERFACE IMPORTED)
    message(${pybind11_MESSAGE_MODE} "Conan: Target declared 'pybind11_all_do_not_use'")
endif()
# Load the debug and release library finders
file(GLOB CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/pybind11-Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()
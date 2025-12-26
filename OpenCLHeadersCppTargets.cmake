# Load the debug and release variables
file(GLOB DATA_FILES "${CMAKE_CURRENT_LIST_DIR}/OpenCLHeadersCpp-*-data.cmake")

foreach(f ${DATA_FILES})
    include(${f})
endforeach()

# Create the targets for all the components
foreach(_COMPONENT ${opencl-clhpp-headers_COMPONENT_NAMES} )
    if(NOT TARGET ${_COMPONENT})
        add_library(${_COMPONENT} INTERFACE IMPORTED)
        message(${OpenCLHeadersCpp_MESSAGE_MODE} "Conan: Component target declared '${_COMPONENT}'")
    endif()
endforeach()

if(NOT TARGET OpenCL::HeadersCpp)
    add_library(OpenCL::HeadersCpp INTERFACE IMPORTED)
    message(${OpenCLHeadersCpp_MESSAGE_MODE} "Conan: Target declared 'OpenCL::HeadersCpp'")
endif()
# Load the debug and release library finders
file(GLOB CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/OpenCLHeadersCpp-Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()
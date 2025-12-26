# Load the debug and release variables
file(GLOB DATA_FILES "${CMAKE_CURRENT_LIST_DIR}/ONNX-*-data.cmake")

foreach(f ${DATA_FILES})
    include(${f})
endforeach()

# Create the targets for all the components
foreach(_COMPONENT ${onnx_COMPONENT_NAMES} )
    if(NOT TARGET ${_COMPONENT})
        add_library(${_COMPONENT} INTERFACE IMPORTED)
        message(${ONNX_MESSAGE_MODE} "Conan: Component target declared '${_COMPONENT}'")
    endif()
endforeach()

if(NOT TARGET onnx::onnx)
    add_library(onnx::onnx INTERFACE IMPORTED)
    message(${ONNX_MESSAGE_MODE} "Conan: Target declared 'onnx::onnx'")
endif()
# Load the debug and release library finders
file(GLOB CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/ONNX-Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()
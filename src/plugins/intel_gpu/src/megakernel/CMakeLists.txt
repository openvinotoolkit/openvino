set(MEGAKERNEL_IMPLEMENTATION "" CACHE STRING
    "Megakernel implementation to link from MEGAKERNEL_POC/megakernels (empty disables integration)")

set(MEGAKERNEL_TRANSFORMATION_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/dummy_transformation.cpp")
set(MEGAKERNEL_BACKEND_SOURCES "")
set(MEGAKERNEL_RUNTIME_TARGET "")
set(MEGAKERNEL_GRAPH_COMPILE_DEFINITIONS "")

if(NOT MEGAKERNEL_IMPLEMENTATION STREQUAL "")
    if(NOT GPU_RT_TYPE MATCHES "^(OCL|COMBINED)$")
        message(FATAL_ERROR "MEGAKERNEL_IMPLEMENTATION requires an OpenCL GPU stack")
    endif()

    if(OV_GPU_RT STREQUAL "OCL")
        set(MEGAKERNEL_ROOT "${CMAKE_SOURCE_DIR}/MEGAKERNEL_POC/megakernels" CACHE PATH
            "Directory containing the megakernel runtime and OpenVINO integration sources")
        set(MEGAKERNEL_IMPLEMENTATION_DIR "${MEGAKERNEL_ROOT}/${MEGAKERNEL_IMPLEMENTATION}")
        set(MEGAKERNEL_RUNTIME_TARGET "${MEGAKERNEL_IMPLEMENTATION}")

        if(NOT EXISTS "${MEGAKERNEL_IMPLEMENTATION_DIR}/CMakeLists.txt")
            message(FATAL_ERROR
                "Megakernel implementation '${MEGAKERNEL_IMPLEMENTATION}' was not found in ${MEGAKERNEL_IMPLEMENTATION_DIR}")
        endif()

        file(GLOB MEGAKERNEL_TRANSFORMATION_SOURCES CONFIGURE_DEPENDS
            "${MEGAKERNEL_IMPLEMENTATION_DIR}/ovIntegration/*.cpp")
        if(NOT MEGAKERNEL_TRANSFORMATION_SOURCES)
            message(FATAL_ERROR
                "Megakernel implementation '${MEGAKERNEL_IMPLEMENTATION}' must provide "
                "ovIntegration/*.cpp defining mk::InsertMegakernelTransformation")
        endif()

        add_subdirectory("${MEGAKERNEL_IMPLEMENTATION_DIR}"
            "${CMAKE_CURRENT_BINARY_DIR}/${MEGAKERNEL_IMPLEMENTATION}" EXCLUDE_FROM_ALL)

        if(NOT TARGET ${MEGAKERNEL_RUNTIME_TARGET})
            message(FATAL_ERROR
                "Megakernel implementation '${MEGAKERNEL_IMPLEMENTATION}' must define target ${MEGAKERNEL_RUNTIME_TARGET}")
        endif()

        target_include_directories(${MEGAKERNEL_RUNTIME_TARGET} INTERFACE
            $<BUILD_INTERFACE:${MEGAKERNEL_ROOT}>)
        install(TARGETS ${MEGAKERNEL_RUNTIME_TARGET}
            RUNTIME DESTINATION ${OV_CPACK_PLUGINSDIR}
            LIBRARY DESTINATION ${OV_CPACK_PLUGINSDIR}
            COMPONENT gpu)

        set(MEGAKERNEL_BACKEND_SOURCES "${MAIN_DIR}/src/graph/impls/ocl_v2/megakernel/megakernel.cpp")
        set(MEGAKERNEL_GRAPH_COMPILE_DEFINITIONS OV_GPU_MEGAKERNEL_IMPL_LINKED)
    endif()
endif()

foreach(variable IN ITEMS MEGAKERNEL_TRANSFORMATION_SOURCES MEGAKERNEL_BACKEND_SOURCES
                          MEGAKERNEL_RUNTIME_TARGET MEGAKERNEL_GRAPH_COMPILE_DEFINITIONS)
    set(${variable} "${${variable}}" PARENT_SCOPE)
endforeach()
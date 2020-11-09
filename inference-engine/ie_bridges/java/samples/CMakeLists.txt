# # Copyright (C) 2020 Intel Corporation

# # Find OpenCV components if exist

# find_package(OpenCV QUIET)

# if(EXISTS "${OpenCV_INSTALL_PATH}/share/java")
#     file(GLOB_RECURSE JAVA_OPENCV_SRC ${OpenCV_INSTALL_PATH}/share/java/*.jar)
# elseif(EXISTS "${OpenCV_INSTALL_PATH}/bin")
#     file(GLOB JAVA_OPENCV_SRC ${OpenCV_INSTALL_PATH}/bin/*.jar)
# endif()

# #
# # ie_add_java_sample(NAME <target name>
# #               SOURCES <source files>
# #               [DEPENDENCIES <dependencies>]
# #               [OPENCV_DEPENDENCIES]
# #
# macro(ie_add_java_sample)
#     set(options OPENCV_DEPENDENCIES)
#     set(oneValueArgs NAME)
#     cmake_parse_arguments(IE_SAMPLE "${options}" "${oneValueArgs}"
#                           "" ${ARGN} )

#     # Collect sample sources

#     file(GLOB_RECURSE SAMPLE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.java)
#     list(APPEND SAMPLE_SOURCES ${PROJECT_SOURCE_DIR}/samples/ArgumentParser.java)

#     # Set InferenceEngine component

#     set(IE_JAVA_SRC ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/inference_engine_java_api.jar)

#     # Add OpenCV components if required

#     if(IE_SAMPLE_OPENCV_DEPENDENCIES) 
#         if(EXISTS "${JAVA_OPENCV_SRC}")
#             list(APPEND IE_JAVA_SRC ${JAVA_OPENCV_SRC})
#         else()
#             message(WARNING "OPENCV .jar file is disabled or not found, " ${IE_SAMPLE_NAME} " skipped")
#             return()
#         endif()
#     endif()

#     # Create .jar file from sources
    
#     add_jar("${IE_SAMPLE_NAME}_jar"
#                     SOURCES ${SAMPLE_SOURCES}
#                     OUTPUT_NAME ${IE_SAMPLE_NAME}  
#                     OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} 
#                     INCLUDE_JARS ${IE_JAVA_SRC})          
#     add_dependencies(${IE_SAMPLE_NAME}_jar inference_engine_jar)
# endmacro()

# # Collect all samples subdirectories

# file(GLOB samples_dirs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)

# foreach(dir ${samples_dirs})
#     if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
#             # Include subdirectory to the project.
#             add_subdirectory(${dir})
#     endif() 
# endforeach()

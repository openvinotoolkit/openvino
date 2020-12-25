
cmake_minimum_required(VERSION 3.11)

project(omz-populate)

include(FetchContent)

FetchContent_Declare(omz
  GIT_REPOSITORY    https://github.com/openvinotoolkit/open_model_zoo.git
)

FetchContent_GetProperties(omz)
if(NOT omz_POPULATED)
  FetchContent_Populate(omz) 
endif()

ie_cpack_add_component(omz)

if(UNIX)
    install(DIRECTORY ${omz_SOURCE_DIR}/
            DESTINATION deployment_tools/open_model_zoo
            COMPONENT omz
            USE_SOURCE_PERMISSIONS
            PATTERN .* EXCLUDE
            PATTERN ci EXCLUDE
            PATTERN *.bat EXCLUDE
            PATTERN demos/tests EXCLUDE
            PATTERN tools/downloader/tests EXCLUDE)
elseif(WIN32)
    install(DIRECTORY ${omz_SOURCE_DIR}/
            DESTINATION deployment_tools/open_model_zoo
            COMPONENT omz
            USE_SOURCE_PERMISSIONS
            PATTERN ci EXCLUDE
            PATTERN .* EXCLUDE
            PATTERN *.sh EXCLUDE
            PATTERN demos/tests EXCLUDE
            PATTERN tools/downloader/tests EXCLUDE)
endif()

# Load the debug and release variables
file(GLOB DATA_FILES "${CMAKE_CURRENT_LIST_DIR}/pugixml-*-data.cmake")

foreach(f ${DATA_FILES})
    include(${f})
endforeach()

# Create the targets for all the components
foreach(_COMPONENT ${pugixml_COMPONENT_NAMES} )
    if(NOT TARGET ${_COMPONENT})
        add_library(${_COMPONENT} INTERFACE IMPORTED)
        message(${pugixml_MESSAGE_MODE} "Conan: Component target declared '${_COMPONENT}'")
    endif()
endforeach()

if(NOT TARGET pugixml::pugixml)
    add_library(pugixml::pugixml INTERFACE IMPORTED)
    message(${pugixml_MESSAGE_MODE} "Conan: Target declared 'pugixml::pugixml'")
endif()
if(NOT TARGET pugixml::static)
    add_library(pugixml::static INTERFACE IMPORTED)
    set_property(TARGET pugixml::static PROPERTY INTERFACE_LINK_LIBRARIES pugixml::pugixml)
endif()
# Load the debug and release library finders
file(GLOB CONFIG_FILES "${CMAKE_CURRENT_LIST_DIR}/pugixml-Target-*.cmake")

foreach(f ${CONFIG_FILES})
    include(${f})
endforeach()
aux_source_directory(. OSL_LIST)
add_library(bsl_osl STATIC ${OSL_LIST})
target_link_libraries(bsl_osl udev)
target_link_libraries(bsl_osl pthread)

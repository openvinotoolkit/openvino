find_program(GRAPHVIZ_EXECUTABLE dot)

# Handle REQUIRED and QUIET arguments
# this will also set GRAPHVIZ_FOUND to true if GRAPHVIZ_EXECUTABLE exists
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Graphviz
                     "Failed to locate graphviz executable"
                     GRAPHVIZ_EXECUTABLE)

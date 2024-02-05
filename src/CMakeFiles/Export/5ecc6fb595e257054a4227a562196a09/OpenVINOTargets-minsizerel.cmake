#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "openvino::frontend::onnx" for configuration "MinSizeRel"
set_property(TARGET openvino::frontend::onnx APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(openvino::frontend::onnx PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino_onnx_frontend.lib"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino_onnx_frontend.dll"
  )

list(APPEND _cmake_import_check_targets openvino::frontend::onnx )
list(APPEND _cmake_import_check_files_for_openvino::frontend::onnx "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino_onnx_frontend.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino_onnx_frontend.dll" )

# Import target "openvino::frontend::pytorch" for configuration "MinSizeRel"
set_property(TARGET openvino::frontend::pytorch APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(openvino::frontend::pytorch PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino_pytorch_frontend.lib"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino_pytorch_frontend.dll"
  )

list(APPEND _cmake_import_check_targets openvino::frontend::pytorch )
list(APPEND _cmake_import_check_files_for_openvino::frontend::pytorch "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino_pytorch_frontend.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino_pytorch_frontend.dll" )

# Import target "openvino::frontend::tensorflow" for configuration "MinSizeRel"
set_property(TARGET openvino::frontend::tensorflow APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(openvino::frontend::tensorflow PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino_tensorflow_frontend.lib"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino_tensorflow_frontend.dll"
  )

list(APPEND _cmake_import_check_targets openvino::frontend::tensorflow )
list(APPEND _cmake_import_check_files_for_openvino::frontend::tensorflow "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino_tensorflow_frontend.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino_tensorflow_frontend.dll" )

# Import target "openvino::frontend::tensorflow_lite" for configuration "MinSizeRel"
set_property(TARGET openvino::frontend::tensorflow_lite APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(openvino::frontend::tensorflow_lite PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino_tensorflow_lite_frontend.lib"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino_tensorflow_lite_frontend.dll"
  )

list(APPEND _cmake_import_check_targets openvino::frontend::tensorflow_lite )
list(APPEND _cmake_import_check_files_for_openvino::frontend::tensorflow_lite "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino_tensorflow_lite_frontend.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino_tensorflow_lite_frontend.dll" )

# Import target "openvino::frontend::paddle" for configuration "MinSizeRel"
set_property(TARGET openvino::frontend::paddle APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(openvino::frontend::paddle PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino_paddle_frontend.lib"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino_paddle_frontend.dll"
  )

list(APPEND _cmake_import_check_targets openvino::frontend::paddle )
list(APPEND _cmake_import_check_files_for_openvino::frontend::paddle "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino_paddle_frontend.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino_paddle_frontend.dll" )

# Import target "openvino::runtime" for configuration "MinSizeRel"
set_property(TARGET openvino::runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(openvino::runtime PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "TBB::tbb"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino.dll"
  )

list(APPEND _cmake_import_check_targets openvino::runtime )
list(APPEND _cmake_import_check_files_for_openvino::runtime "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino.dll" )

# Import target "openvino::runtime::c" for configuration "MinSizeRel"
set_property(TARGET openvino::runtime::c APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(openvino::runtime::c PROPERTIES
  IMPORTED_IMPLIB_MINSIZEREL "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino_c.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_MINSIZEREL "openvino::runtime"
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino_c.dll"
  )

list(APPEND _cmake_import_check_targets openvino::runtime::c )
list(APPEND _cmake_import_check_files_for_openvino::runtime::c "${_IMPORT_PREFIX}/runtime/lib/intel64/MinSizeRel/openvino_c.lib" "${_IMPORT_PREFIX}/runtime/bin/intel64/MinSizeRel/openvino_c.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)

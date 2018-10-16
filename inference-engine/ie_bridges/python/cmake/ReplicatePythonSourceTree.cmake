# Note: when executed in the build dir, then CMAKE_CURRENT_SOURCE_DIR is the
# build dir.

file( COPY setup.py inference_engine tests DESTINATION "${CMAKE_ARGV3}"
  FILES_MATCHING PATTERN "*.py" )

file( COPY requirements.txt DESTINATION "${CMAKE_ARGV3}" )

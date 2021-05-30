PROJECT(zlib)


if(NOT WIN32)
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-all")
            set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-all")
endif()





if (MSVC)
  # Build with multiple processes
  add_definitions(/MP)
  # MSVC warning suppressions
  add_definitions(
    /wd4996 # The compiler encountered a deprecated declaration.
  )

endif (MSVC)





set(BUILD_SHARED_LIBS OFF)

set(TARGET_NAME "zlib")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/zlib")

#file(GLOB lib_srcs *.c)

set(lib_srcs 
    zlib/adler32.c
    zlib/compress.c
    zlib/crc32.c
    zlib/deflate.c
    zlib/gzclose.c
    zlib/gzlib.c
    zlib/gzread.c
    zlib/gzwrite.c
    zlib/inflate.c
    zlib/infback.c
    zlib/inftrees.c
    zlib/inffast.c
    zlib/trees.c
    zlib/uncompr.c
    zlib/zutil.c
)

set(lib_hdrs 
    zlib/crc32.h
    zlib/deflate.h
    zlib/gzguts.h
    zlib/inffast.h
    zlib/inffixed.h
    zlib/inflate.h
    zlib/inftrees.h
    zlib/trees.h
    zlib/zutil.h
)


#file(GLOB lib_hdrs *.h)
set(lib_ext_hdrs "zlib/zlib.h" "zlib/zconf.h")

add_library(${TARGET_NAME} STATIC ${lib_srcs} ${lib_hdrs} ${lib_ext_hdrs})

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/zlib"
                                                 "${CMAKE_CURRENT_SOURCE_DIR}/zlib/..")

if(MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3")
endif()

if(UNIX)
  if(CMAKE_COMPILER_IS_GNUCXX OR CV_ICC)
     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
  endif()
endif()


set_target_properties(zlib PROPERTIES FOLDER thirdparty)
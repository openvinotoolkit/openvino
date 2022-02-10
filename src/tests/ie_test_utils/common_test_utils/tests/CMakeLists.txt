set(TARGET_NAME CommonTestUtilsFuncTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
        commonTestUtils
        LINK_LIBRARIES
        commonTestUtils
        ADD_CPPLINT
)

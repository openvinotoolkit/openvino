ROOT_PATH:= $(call my-dir)
include $(call all-subdir-makefiles, $(ROOT_PATH))

LOCAL_PATH:= $(ROOT_PATH)
include $(ROOT_PATH)/android/ie.mk
include $(ROOT_PATH)/android/mkldnn_plugin.mk
include $(ROOT_PATH)/android/format_reader.mk
include $(ROOT_PATH)/android/sample.mk

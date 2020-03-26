ifneq ($(KERNELRELEASE),)
# kbuild part of makefile
obj-m  := myd_vsc.o
8123-y := myd_vsc.o

else

ifeq ($(shell uname -r | cut -d- -f2),yocto)
	YOCTO :=1
else ifeq ($(shell cat /etc/redhat-release 2>/dev/null | cut -d' ' -f1),CentOS)
	CentOS :=1
else ifeq ($(shell lsb_release -si),Ubuntu)
	Ubuntu :=1
else
	$(error Unrecognized OS)
endif


# normal makefile
ifeq ($(YOCTO),1)
    KDIR :=/usr/src/kernel
else
    KDIR ?= /lib/modules/`uname -r`/build
endif

default:
	$(MAKE) -C $(KDIR) M=$$PWD

# Module specific targets
genbin:
	echo "X" > myriad_vsc_bin.o_shipped

clean:
	rm -f *.o *.ko *.order *.mod.c *.symvers .*.cmd
	rm -rf .tmp_versions

TARGET := myd_vsc.ko
TARGET_BASENAME := $(basename $(TARGET))
$(TARGET): default

.PHONY: preuninstall
preuninstall:
	if [[ "$(shell lsmod | grep $(TARGET_BASENAME))" != "" ]]; then sudo rmmod $(TARGET_BASENAME); fi;
	@make clean

.PHONY: preinstall
preinstall: preuninstall
	@make

.PHONY: install
install: SHELL=/bin/bash

.PHONY: uninstall
uninstall: SHELL=/bin/bash

ifneq ($(YOCTO),1)
CONF_PATH=/etc/modules-load.d/$(basename $(TARGET)).conf
DRV_PATH=/lib/modules/$(shell uname -r)/kernel/drivers/myd/$(TARGET)
endif

install: preinstall
ifeq ($(YOCTO),1)
	$(error No defined install procedure for YOCTO)
endif

	mkdir -p $(dir $(DRV_PATH))
	cp $(TARGET) $(DRV_PATH)
	@echo -e " \e[1m" "\e[96m" "Driver $(TARGET) is installed to $(DRV_PATH)" "\e[0m"
	sudo depmod
	@modprobe $(TARGET_BASENAME)
	@modinfo $(TARGET_BASENAME)

	mkdir -p $(dir $(CONF_PATH))
	@echo -e $(basename $(TARGET)) > $(CONF_PATH)
	@echo -e " \e[1m" "\e[96m" "$(CONF_PATH) is created for auto-load at boot time" "\e[0m"

uninstall: preuninstall
ifeq ($(YOCTO),1)
	$(error No defined uninstall procedure for YOCTO)
endif

	[ ! -e $(CONF_PATH) ] || rm $(CONF_PATH)
	[ ! -e $(DRV_PATH) ] || rm $(DRV_PATH)
	if [[ `ls -A $(dir $(DRV_PATH))` == "" ]]; then sudo rm -rf $(dir $(DRV_PATH)); fi;
	sudo depmod
	@echo -e " \e[1m" "\e[96m" "$(DRV_PATH) is removed" "\e[0m"
	@echo -e " \e[1m" "\e[96m" "$(CONF_PATH) is removed" "\e[0m"

endif

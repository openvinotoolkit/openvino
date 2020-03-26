# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

INSTALLDIR := ${DESTDIR}/usr/local
LIBUSB_CFLAGS := $(shell pkg-config --cflags --libs libusb-1.0)
LDFLAGS += $(LIBUSB_CFLAGS) -ldl -ljson-c -lpthread -ludev -fsanitize=address
LDFLAGS += -ldl -z noexecstack -z relro -z now

INCLUDE_FLAGS = -I../include -I. -I./osl -I./osl/linux

CFLAGS += $(INCLUDE_FLAGS)
CFLAGS += -O2 -Wall $(LIBUSB_CFLAGS) -fPIC -D_FORTIFY_SOURCE=2
CFLAGS += -fPIC -MMD -MP -Wformat -Wformat-security -fstack-protector-strong
#CFLAGS += -DDEBUG

OBJS =		osl/linux/hidapi_linux.c 	\
			osl/linux/i2cbusses.c 		\
			osl/linux/osl_linux.c 		\
			osl/linux/smbus_linux.c 	\
			osl/linux/thread_linux.c 	\
			bsl_cfg.c					\
			bsl_error_code.c 			\
			hid_f75114.c 				\
			ioexpander.c 				\
			main.c 						\
			mcu.c						\
			utils.c

HEADERS = 	../include/hddl-bsl.h		\
			bsl_cfg.h					\
 			hddl_bsl_priv.h				\
 			hddl_bsl_thread.h 			\
 			hid-dev.h 					\
 			i2c-dev.h 					\
 			osl/osl.h 					\
 			osl/linux/i2c_linux.h 		\
 			osl/linux/i2cbusses.h

ifeq "$(wildcard $(INSTALLDIR) )" " "
	INSTALLDIR = ${DESTDIR}/usr
endif

all: libbsl.so.0 tool
tool: bsl_reset

*.o: $(HEADERS)

.c.o:
	$(CC) -c  $(CFLAGS) $<

libbsl.so.0: $(OBJS)
	$(CC) -shared -o $@ -Wl,-h$@ $(CFLAGS) $(OBJS) $(LDFLAGS)
	ln -fs libbsl.so.0 libbsl.so

clean:
	rm -f *.o *.d libbsl.so.0 libbsl.so bsl_reset

bsl_reset: bsl_reset.o
	$(CC) $(LDFLAGS) -o $@ bsl_reset.o  -L./ -lbsl

basicinstall: libbsl.so.0
	mkdir -p $(INSTALLDIR)/include/
	mkdir -p $(INSTALLDIR)/lib/
	mkdir -p $(INSTALLDIR)/bin/
	cp libbsl.so.0 $(INSTALLDIR)/lib/
	ln -fs libbsl.so.0 $(INSTALLDIR)/lib/libbsl.so
	cp ../include/* $(INSTALLDIR)/include/
	cp bsl_reset $(INSTALLDIR)/bin/
	bash generate_udev_rules.sh ${DESTDIR}/etc/udev/rules.d/

postinstall:
	modprobe i2c-i801
	udevadm control --reload-rules
	udevadm trigger
	ldconfig

install: basicinstall postinstall

uninstall:
	rm $(INSTALLDIR)/lib/libbsl.so.0
	rm $(INSTALLDIR)/lib/libbsl.so
	rm $(INSTALLDIR)/include/hddl-bsl.h

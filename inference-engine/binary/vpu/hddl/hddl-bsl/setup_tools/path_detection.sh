#!/usr/bin/env bash

searching_for="2C42:5114"

get_hid_match() {
    list=$1
    results=()
    for a_file in ${list}; do
        path=`udevadm info ${a_file} -q path`
        filtered=`echo ${path}| grep ${searching_for}`
        if (( $? == 0 )) ; then
            echo ${path}
        fi
    done
}

get_recommened_path() {
    path=$1
    hid_path=`dirname ${path}`
    device_path=`dirname ${hid_path}`
    usb_path=`dirname ${device_path}`
    echo "\"/sys${usb_path}\","
}

hidraw_devices=`find /dev/ -name hidraw*`
matched_list=$(get_hid_match "${hidraw_devices}")
echo "HID found at:"
echo "${matched_list}"
echo
echo "Suggested path(s):"
for path in ${matched_list}; do
    get_recommened_path ${path}
done

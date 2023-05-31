FILTERS = {
    TEST_TYPE: "_test_type_",
    DEVICE: "_DEVICE_"
}

FILTER_ATTRIBUTE = "data-filter"

allDevices = []
swPlugins = []

$(document).ready(function () {
    $("#devices").chosen({max_selected_options: 6});
    Array.from($("#devices")[0].options).map((element) => allDevices.push(element.value))

    $("#report #statistic td").filter(function () {
        sw_plugin = $(this).attr('data-sw_plugin')
        if (sw_plugin && !swPlugins.includes(sw_plugin)) {
            swPlugins.push(sw_plugin)
        }
    });

    update_total()
});

function apply_filters_on_report_table() {
    $("#report td").filter(function () {
        let data_filter = $(this).attr(FILTER_ATTRIBUTE)

        if (data_filter && data_filter.length > 0) {
            $(this).hide();
        } else {
            $(this).show()
        }
    });

    update_total()
}

function get_background_color(float_passrate) {
    background_color = 'white'
    if (float_passrate > 97) {
        let opacity = 1 - (100 - float_passrate) * 0.5 / (100 - 97)
        background_color = "rgba(127, 209, 127, " + opacity + ")"
    } else if (float_passrate < 70) {
        let opacity = 1 - float_passrate * 0.5 / 70
        background_color = "rgba(233, 108, 108, " + opacity + ")"
    }
    return background_color
}

function update_total() {
    for (device of allDevices) {
        for (sw_plugin of swPlugins) {
            let passed = 0
            let amount = 0
            let rel_passed = 0
            let rel_amount = 0

            $("#report #data td:not(:hidden)").filter(function () {
                if ($(this).attr('data-device') === device && $(this).attr('data-sw_plugin') === sw_plugin) {
                    if ($(this).attr('data-passed_tests') && $(this).attr('data-all_tests') &&
                        $(this).attr('data-rel_passed_tests') && $(this).attr('data-rel_all_tests')) {
                        passed += parseInt($(this).attr('data-passed_tests'))
                        amount += parseInt($(this).attr('data-all_tests'))
                        rel_passed += parseInt($(this).attr('data-rel_passed_tests'))
                        rel_amount += parseInt($(this).attr('data-rel_all_tests'))
                    }
                }
            });

            let passrate = ''
            let background_color = 'white'
            if (!amount) {
                passrate = "---";
            } else {
                let float_passrate = passed * 100 / amount
                passrate = (float_passrate).toFixed(2) + ' %';
                background_color = get_background_color(float_passrate)
            }
            let id_general = '#' + device + '_' + sw_plugin + '_general'
            if ($(id_general).length) {
                $(id_general).text(passrate)
                $(id_general).css({"background-color": background_color})
            }

            passrate = ''
            background_color = 'white'
            if (!rel_amount) {
                passrate = "---";
            } else {
                let float_passrate = rel_passed * 100 / rel_amount
                passrate = (float_passrate).toFixed(2) + ' %';
                background_color = get_background_color(float_passrate)
            }
            let id_rel = '#' + device + '_' + sw_plugin + '_rel'
            if ($(id_rel).length) {
                $(id_rel).text(passrate)
                $(id_rel).css({"background-color": background_color})
            }
        }
    }
}

function filter_by_test_type() {
    testTypeName = $('#testTypeName').val().trim();

    $("#report #data td").filter(function () {
        let test_type =  $(this).attr('data-test-type')
        let data_filter = $(this).attr(FILTER_ATTRIBUTE) || "";
        if (!testTypeName || test_type.toLowerCase().indexOf(testTypeName.toLowerCase()) > -1) {
            data_filter = data_filter.replace(FILTERS.TEST_TYPE, "")
            $(this).attr(FILTER_ATTRIBUTE, data_filter)
        } else {
            if (data_filter.indexOf(FILTERS.TEST_TYPE) == -1) {
                data_filter += FILTERS.TEST_TYPE
                $(this).attr(FILTER_ATTRIBUTE, data_filter)
            }
        }
    });

    apply_filters_on_report_table()
}

function change_device(value) {
    let selected_devices = $(value).val()

    $("#report td").filter(function () {
        let device =  $(this).attr('data-device')
        let data_filter = $(this).attr(FILTER_ATTRIBUTE) || "";
        if (!selected_devices || selected_devices.length == 0 || !device || selected_devices.includes(device)) {
            data_filter = data_filter.replace(FILTERS.DEVICE, "")
            $(this).attr(FILTER_ATTRIBUTE, data_filter)
        } else {
            if (data_filter.indexOf(FILTERS.DEVICE) == -1) {
                data_filter += FILTERS.DEVICE
                $(this).attr(FILTER_ATTRIBUTE, data_filter)
            }
        }
    });

    apply_filters_on_report_table()
}

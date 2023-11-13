FILTERS = {
    TEST_TYPE: "_test_type_",
    DEVICE: "_DEVICE_"
}

DATA_ATTRIBUTES = {
    FILTER: "data-filter",
    SW_PLUGIN: 'data-sw_plugin',
    DEVICE: 'data-device',
    PASSED: 'data-passed_tests',
    FAILED: 'data-failed',
    SKIPPED: 'data-skipped',
    CRASHED: 'data-crashed',
    HANGED: 'data-hanged',
    TESTS_AMOUNT: 'data-all_tests',
    PASSRATE: 'data-passrate',
    REL_PASSED: 'data-rel_passed_tests',
    REL_TESTS_AMOUNT: 'data-rel_all_tests',
    REL_PASSRATE: 'data-relative_passrate'
}

SCOPE_TYPE = {
    MANDATORY: 'Mandatory',
    OPTIONAL: 'Optional'
}

allDevices = []
swPlugins = []

$(document).ready(function () {
    $("#devices").chosen({max_selected_options: 6});
    Array.from($("#devices")[0].options).map((element) => allDevices.push(element.value))

    $("#report #statistic td").filter(function () {
        sw_plugin = $(this).attr(DATA_ATTRIBUTES.SW_PLUGIN)
        if (sw_plugin && !swPlugins.includes(sw_plugin)) {
            swPlugins.push(sw_plugin)
        }
    });

    update_statistic();
});

function apply_filters_on_report_table() {
    $("#report td").filter(function () {
        let data_filter = $(this).attr(DATA_ATTRIBUTES.FILTER)

        if (data_filter && data_filter.length > 0) {
            $(this).hide();
        } else {
            $(this).show()
        }
    });

    update_statistic()
}

function get_background_color(float_passrate) {
    background_color = "hsl(" + Math.floor(float_passrate) +", 80%, 60%)"
    return background_color
}

function update_statistic() {
    for (device of allDevices) {
        for (sw_plugin of swPlugins) {
            let passed = 0
            let amount = 0
            let scope = $('#scope_toggle').text()

            $("#report #data td:not(:hidden)").filter(function () {
                if ($(this).attr(DATA_ATTRIBUTES.DEVICE) === device && $(this).attr(DATA_ATTRIBUTES.SW_PLUGIN) === sw_plugin) {
                    if ($(this).attr(DATA_ATTRIBUTES.PASSED) && $(this).attr(DATA_ATTRIBUTES.TESTS_AMOUNT) &&
                        scope == SCOPE_TYPE.OPTIONAL) {
                        passed += parseInt($(this).attr(DATA_ATTRIBUTES.PASSED))
                        amount += parseInt($(this).attr(DATA_ATTRIBUTES.TESTS_AMOUNT))
                    }
                    if ($(this).attr(DATA_ATTRIBUTES.REL_PASSED) && $(this).attr(DATA_ATTRIBUTES.REL_TESTS_AMOUNT) &&
                        scope == SCOPE_TYPE.MANDATORY) {
                        passed += parseInt($(this).attr(DATA_ATTRIBUTES.REL_PASSED))
                        amount += parseInt($(this).attr(DATA_ATTRIBUTES.REL_TESTS_AMOUNT))
                    }
                }
            });

            let passrate = ''
            let background_color = "rgba(255, 255, 255, 0.2)"
            if (!amount) {
                passrate = "---";
            } else {
                let float_passrate = passed * 100 / amount
                passrate = (float_passrate).toFixed(2) + ' %';
                background_color = get_background_color(float_passrate)
            }
            let id_general = '#' + device + '_' + sw_plugin + '_statistic'
            if ($(id_general).length) {
                $(id_general).text(passrate)
                $(id_general).css("background-color", background_color)
            }
        }
    }
}

function update_data() {
    if ($('#scope_toggle').text() == SCOPE_TYPE.MANDATORY) {
        $('#scope_toggle').text(SCOPE_TYPE.OPTIONAL)
    } else {
        $('#scope_toggle').text(SCOPE_TYPE.MANDATORY)
    }

    let scope_toggle = $('#scope_toggle').text()

    if (scope_toggle == SCOPE_TYPE.OPTIONAL) {
        $('#scope_legend').text('Optinal - includes information and passrate about all apiConformance tests.')
    } else {
        $('#scope_legend').text('Mandatory - includes information about tests from mandatory tests.\
                                 This tests should be passed to determine the plugin is conformance by API.\
                                 For mandatory scope only passed and failed statuses are applicable.')
    }

    $("#report #data td").filter(function () {
        if (scope_toggle == SCOPE_TYPE.OPTIONAL && $(this).attr(DATA_ATTRIBUTES.PASSRATE)) {
            $(this).children('span').text($(this).attr(DATA_ATTRIBUTES.PASSRATE) + ' %')

            $($(this).find('.green')[0]).text('P:' + $(this).attr(DATA_ATTRIBUTES.PASSED))
            $($(this).find('.red')[0]).text('F:' + $(this).attr(DATA_ATTRIBUTES.FAILED))
            $($(this).find('.grey')[0]).show()
            $($(this).find('.grey')[0]).text('S:' + $(this).attr(DATA_ATTRIBUTES.SKIPPED))
            $($(this).find('.dark')[0]).show()
            $($(this).find('.dark')[0]).text('C:' + $(this).attr(DATA_ATTRIBUTES.CRASHED))
            $($(this).find('.grey-red')[0]).show()
            $($(this).find('.grey-red')[0]).text('H:' + $(this).attr(DATA_ATTRIBUTES.HANGED))
        } else if (scope_toggle == SCOPE_TYPE.MANDATORY && $(this).attr(DATA_ATTRIBUTES.REL_PASSRATE)) {
            if (parseInt($(this).attr(DATA_ATTRIBUTES.REL_TESTS_AMOUNT)) > 0) {
                $(this).children('span').text($(this).attr(DATA_ATTRIBUTES.REL_PASSRATE) + ' %')
            } else {
                $(this).children('span').text('-- %')
            }

            $($(this).find('.green')[0]).text('P:' + parseInt($(this).attr(DATA_ATTRIBUTES.REL_PASSED)))
            $($(this).find('.red')[0]).text('F:' + (parseInt($(this).attr(DATA_ATTRIBUTES.REL_TESTS_AMOUNT)) - parseInt($(this).attr(DATA_ATTRIBUTES.REL_PASSED))))
            $($(this).find('.grey')[0]).hide()
            $($(this).find('.dark')[0]).hide()
            $($(this).find('.grey-red')[0]).hide()
        }
    });

    update_statistic()
}

function filter_by_test_type() {
    testTypeName = $('#testTypeName').val().trim();

    $("#report #data td").filter(function () {
        let test_type =  $(this).attr('data-test-type')
        let data_filter = $(this).attr(DATA_ATTRIBUTES.FILTER) || "";
        if (!testTypeName || test_type.toLowerCase().indexOf(testTypeName.toLowerCase()) > -1) {
            data_filter = data_filter.replace(FILTERS.TEST_TYPE, "")
            $(this).attr(DATA_ATTRIBUTES.FILTER, data_filter)
        } else {
            if (data_filter.indexOf(FILTERS.TEST_TYPE) == -1) {
                data_filter += FILTERS.TEST_TYPE
                $(this).attr(DATA_ATTRIBUTES.FILTER, data_filter)
            }
        }
    });

    apply_filters_on_report_table()
}

function change_device(value) {
    let selected_devices = $(value).val()
    $("#report td").filter(function () {
        let device =  $(this).attr(DATA_ATTRIBUTES.DEVICE)
        let data_filter = $(this).attr(DATA_ATTRIBUTES.FILTER) || "";

        if (!selected_devices || selected_devices.length == 0 || !device || selected_devices.includes(device)) {
            data_filter = data_filter.replace(FILTERS.DEVICE, "")
            $(this).attr(DATA_ATTRIBUTES.FILTER, data_filter)
        } else {
            if (data_filter.indexOf(FILTERS.DEVICE) == -1) {
                data_filter += FILTERS.DEVICE
                $(this).attr(DATA_ATTRIBUTES.FILTER, data_filter)
            }
        }
    });

    apply_filters_on_report_table()
}

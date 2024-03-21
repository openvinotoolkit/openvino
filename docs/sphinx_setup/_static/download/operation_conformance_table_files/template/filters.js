deviceList = [];
$(document).ready(function () {
    LoadOpsetNumbers();
    LoadDevices();
    $('#status').prop("disabled", true);
    $("#status").chosen({max_selected_options: 6});

    $("#filters").submit(function (event) {
        event.preventDefault();
        filterTable();
    });
    $('#reset').click(function () {
        $('#opsetNumber').val(0);
        $('#operationName').val('');
        $('#status').prop("disabled", true).val('');
        $('#devices').val(0);
        $('#implementation').val(0);
        $("#status").chosen("destroy");
        $("#status").chosen({max_selected_options: 6});
        filterTable();
    });
    $('#devices').on('change', function () {
        if (this.value == 0) {
            $('#status').prop("disabled", true).val('');
            $("#status").chosen("destroy");
            $("#status").chosen({max_selected_options: 6});
        } else {
            $('#status').prop("disabled", false);
            $("#status").chosen("destroy");
            $("#status").chosen({max_selected_options: 6});
        };
    });
});

function LoadOpsetNumbers() {
    var data = [];

    $('#data th[scope="row"]').each(function () {

        num = $(this).text().split("-")[1];
        if (data.indexOf(num) < 0) {
            data.push(num);
        }
    });
    data.sort();
    data = $.map(data, function (item) {
        return "<option value=" + item + ">" + item + "</option>";
    });
    $("#opsetNumber").html('<option value="0">All</option>');
    $("#opsetNumber").append(data.join(""));
}

function LoadDevices() {
    var data = [];

    $('.table-dark.device').each(function () {
        if (data.indexOf($(this).text()) < 0) {
            data.push($(this).text());
        }
    });
    data.sort();
    deviceList = data;
    data = $.map(data, function (item) {
        return "<option value=" + item + ">" + item + "</option>";
    });
    $("#devices").html('<option value="0">All</option>');
    $("#devices").append(data.join(""));
}

function filterTable() {
    device = $("#devices").val();
    if (device == 0) {
        $("#report td.value, #report td.nr_value, #report td.table-secondary, #report td.table-primary, #report th.table-dark.device").show();
    } else {
        $("#report td.value, #report td.nr_value, #report td.table-secondary, #report td.table-primary, #report th.table-dark.device").filter(function () {
            $(this).toggle($(this).hasClass(device))
        });
    }
    opsetNumber = $("#opsetNumber").val();
    operationName = $('#operationName').val().trim();
    status = $('#status').val();
    implementation = $('#implementation').val();

    $("#report #data tr").show();
    $('#report').show();
    $('#message').hide();
    if (opsetNumber != 0) {
        $("#report #data tr").filter(function () {
            $(this).toggle(checkVersion($(this), opsetNumber));
        });
    }

    if (operationName) {
        $("#report #data tr:not(:hidden)").filter(function () {
            $(this).toggle($(this).find('th').text().toLowerCase().indexOf(operationName.toLowerCase()) > -1);
        });
    }

    if (implementation != 0) {
        if (implementation == 'ni') {
            $("#report #data tr:not(:hidden)").filter(function () {
                $(this).toggle($(this).find('td').hasClass("value " + device + " not_impl"))
            });
        } else if (implementation == 'i') {
            $("#report #data tr:not(:hidden)").filter(function () {
                $(this).toggle($(this).find('td').hasClass("value " + device + " impl"));
            });
        } else {
            $("#report #data tr:not(:hidden)").filter(function () {
                $(this).toggle(!$(this).find('td').hasClass("value"));
            });
        }
    }
    if (status) {
        select = status.split(',');
        selector = [];
        select.forEach(item => {
            if (item == '100p') {
               selector.push('.value:visible[crashed="0"][failed="0"][skipped="0"][hanged="0"][value!="---"]');
            }
            if (item == '100f') {
               selector.push('.value:visible[passed="0"][value!="---"]');
            }
            if (item == 'p') {
                selector.push('.value:visible[passed!="0"][value!="---"]');
            }
            if (item == 'f') {
                selector.push('.value:visible[failed!="0"][value!="---"]');
            }
            if (item == 'c') {
                selector.push('.value:visible[crashed!="0"][value!="---"]');
            }
            if (item == 'h') {
                selector.push('.value:visible[hanged!="0"][value!="---"]');
            }
            if (item == 's') {
                selector.push('.value:visible[value!="---"][skipped!="0"]');
            }
            if (item == 'ex') {
                selector.push('.value:visible[value!="---"]');
            }
            if (item == 'na') {
                selector.push('.table-secondary:visible');
            }
            if (item == 'ns') {
                selector.push('.value:visible[value="---"]');
            }
        });
        elements = selector.join(',');
        $("#report #data tr:not(:hidden)").filter(function () {
            $(this).toggle($(this).find(elements).length > 0)
        });
    }

    if ($("#report #data tr").length == $("#report #data tr:hidden").length) {
        $('#report').hide();
        $('#message').show();
    } else {
        calculateStatistics(device);
    }
}

function checkVersion(element, opsetNumber) {
    var name = element.find('th')[0].getAttribute('name');
    var opsets = JSON.parse(name)[element.find('th').text()];
    var realOpsetNumber = Number(opsetNumber);
    return opsets.indexOf(realOpsetNumber) != -1;
}

function calculateStatistics() {
    if (device != 0) {
        calculateColumnStatistics(device);
    } else {
        deviceList.map((el) => calculateColumnStatistics(el))
    }
}

function calculateColumnStatistics(device) {
    // total
    total = $("#report #data tr:not(:hidden)").length;
    $('#statistic .table-primary[scope="row"] i').text(total);
    // trusted op
    count_trusted_op = $("#report #data tr:not(:hidden) ." + device + ".value[value^='100'][crashed='0'][failed='0'][skipped='0']").length;
    all_operations = $("#report #data tr:not(:hidden) .value[value!='N/A'][value!='---'][value!='NOT RUN']." + device).length;
    if (!all_operations) {
        trusted_op = "---";
    } else {
        trusted_op = (count_trusted_op * 100 / all_operations).toFixed(2) + ' %';
    }
    $('#statistic .table-primary.' + device + '.trusted-ops').text(trusted_op);
    // $('#statistic .table-primary.' + device + '.test_total').text(all_operations || 0);

    // tested op_counter
    tested_op_count = 0;
    passed_tested_op_count = 0;
    $("#report #data tr:not(:hidden) ." + device + ".value span").each(function () {
        text = $(this).text().split(':')[1];
        if (text) {
            if ($(this).hasClass('green')) {
                passed_tested_op_count += +text;
            }
            tested_op_count += +text;
        }
    });

    // General Pass Rate
    if (tested_op_count == 0) {
        $('#statistic .table-primary.' + device + '.general_pass_rate').text('---');

    } else {
        general_pass_rate = (passed_tested_op_count * 100 / tested_op_count).toFixed(2) + ' %';
        $('#statistic .table-primary.' + device + '.general_pass_rate').text(general_pass_rate);
    }
    $('#statistic .table-primary.' + device + '.tested-ops_count').text(tested_op_count);

    // AVG Pass Rate
    sum_pass_rate = 0;
    $("#report #data tr:not(:hidden) ." + device + ".value").each(function () {
        if ($(this).attr('value') != 'N/A' && $(this).attr('value') != 'NOT RUN' && $(this).attr('value') != '---') {
            sum_pass_rate += +$(this).attr('value');
        }
    });
    if (all_operations == 0) {
        $('#statistic .table-primary.' + device + '.rel_pass_rate').text('---');
    } else {
        rel_pass_rate = (sum_pass_rate / all_operations).toFixed(2) + ' %';
        $('#statistic .table-primary.' + device + '.rel_pass_rate').text(rel_pass_rate);
    }
}

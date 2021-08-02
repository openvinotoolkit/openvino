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
        $('#references').val(0);
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
        $("#report td.value, #report td.table-secondary, #report td.table-primary, #report th.table-dark.device").show();
    } else {
        $("#report td.value, #report td.table-secondary, #report td.table-primary, #report th.table-dark.device").filter(function () {
            $(this).toggle($(this).hasClass(device))
        });
    }
    opsetNumber = $("#opsetNumber").val();
    operationName = $('#operationName').val().trim();
    status = $('#status').val();
    references = $('#references').val();

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

    if (references != 0) {
        if (references == 'nv') {
            $("#report #data tr:not(:hidden)").filter(function () {
                $(this).toggle($(this).find('th').hasClass("colorRed"))
            });
        } else {
            $("#report #data tr:not(:hidden)").filter(function () {
                $(this).toggle(!$(this).find('th').hasClass("colorRed"));
            });
        }
    }
    if (status) {
        select = status.split(',');
        selector = [];
        select.forEach(item => {
            if (item == 'p') {
               selector.push('.value:visible[crashed="0"][failed="0"][skipped="0"]');
            }
            if (item == 'f') {
                selector.push('.value:visible[failed!="0"]');
            }
            if (item == 'c') {
                selector.push('.value:visible[crashed!="0"]');
            }
            if (item == 's') {
                selector.push('.value:visible[skipped!="0"]');
            }
            if (item == 'ex') {
                selector.push('.value:visible');
            }
            if (item == 'na') {
                selector.push('.table-secondary:visible');
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
    var name = element.find('th').text().split("-")[0];
    var version = element.find('th').text().split("-")[1];
    if (version > opsetNumber) {
        return false;
    } else {
        var versions = [];
        $('#report #data tr th[name^="' + name + '-"]').each(function () {
            if ($(this).text().split('-')[1] <= opsetNumber) {
                versions.push(+$(this).text().split('-')[1]);
            }
        });
        return version == Math.max.apply(null, versions);
    }
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
    count_trasted_op = $("#report #data tr:not(:hidden) ." + device + ".value[value^='100'][crashed='0'][failed='0'][skipped='0']").length;
    all_operations = $("#report #data tr:not(:hidden) .value." + device).length;
    if (!all_operations) {
        trasted_op = "---";
    } else {
        trasted_op = (count_trasted_op * 100 / all_operations).toFixed(1) + ' %';
    }
    $('#statistic .table-primary.' + device + '.trusted-ops').text(trasted_op);
    $('#statistic .table-primary.' + device + '.test_total').text(all_operations || 0);

    // tested op_counter
    tested_op_count = 0;
    passed_tested_op_count = 0;
    $("#report #data tr:not(:hidden) ." + device + ".value span").each(function () {
        text = $(this).text().split(':')[1];
        if ($(this).hasClass('green')) {
            passed_tested_op_count += +text;
        }
        tested_op_count += +text;
    });

    // General Pass Rate
    if (tested_op_count == 0) {
        $('#statistic .table-primary.' + device + '.general_pass_rate').text('---');

    } else {
        general_pass_rate = (passed_tested_op_count * 100 / tested_op_count).toFixed(1) + ' %';
        $('#statistic .table-primary.' + device + '.general_pass_rate').text(general_pass_rate);
    }
    $('#statistic .table-primary.' + device + '.tested-ops_count').text(tested_op_count);

    // AVG Pass Rate
    sum_pass_rate = 0;
    $("#report #data tr:not(:hidden) ." + device + ".value").each(function () {
        sum_pass_rate += +$(this).attr('value');
    });
    if (all_operations == 0) {
        $('#statistic .table-primary.' + device + '.avg_pass_rate').text('---');
    } else {
        avg_pass_rate = (sum_pass_rate / all_operations).toFixed(1) + ' %';
        $('#statistic .table-primary.' + device + '.avg_pass_rate').text(avg_pass_rate);
    }
}

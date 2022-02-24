$(document).ready(function () {
    var chartBlock = $('.chart-block-tf-ov');
    chartBlock.each(function () {
        var url = $(this).data('loadcsv');
        Papa.parse(url, {
            download: true,
            complete: renderData($(this))
        })
    });

    function getLabels(data) {
        return data
            .map((item) => item[1]);
    }

    function getChartOptions(title, displayLabels) {
        return {
            responsive: false,
            maintainAspectRatio: false,
            legend: { display: true, position: 'bottom' },
            title: {
                display: true,
                text: title
            },
            scales: {
                xAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }],
                yAxes: [{
                    ticks: {
                        display: displayLabels, //this will remove only the label
                        beginAtZero: true
                    }
                }]
            },
            plugins: {
                datalabels: {
                    color: "#4A4A4A",
                    anchor: "end",
                    align: "end",
                    clamp: false,
                    offset: 0,
                    display: true,
                    font: {
                        size: 8,
                        family: 'Roboto'
                    }
                }
            }
        }
    }

    function getChartData(data) {
        function getDataset(data, col, label, color) {
            return {
                label: label,
                data: data.map(function (item) {
                    return item[col]
                }),
                backgroundColor: color,
                borderColor: 'rgba(170,170,170,0)',
                barThickness: 12
            }
        }
        return {
            labels: getLabels(data),
            datasets: [getDataset(data, 2, 'openvino', '#00C7FD'), getDataset(data, 3, 'TF', '#8F5DA2')]
        };
    }

    function renderData(currentChart) {
        return function (result) {
            var data = result.data;
            // remove col names
            data.shift(0);
            var chartName = data[1][0];
            var chartSlug = chartName.replace(')', '').replace(' (', '-');
            var graphContainer = $('<div>');
            var chartContainer = $('<div>');
            graphContainer.attr('id', 'ov-graph-container-' + chartSlug);
            chartContainer.addClass('chart-container');
            chartContainer.addClass('container');
            var chartWrap = $('<div>');
            chartWrap.addClass('chart-wrap');
            chartWrap.addClass('container');
            chartContainer.append(chartWrap);
            var chart = $('<div>');
            chart.addClass('chart');
            chart.addClass('col-md-12');
            var canvas = $('<canvas>');
            chart.append(canvas);
            var container = $('<div>');
            container.addClass('row');
            container.append(chart);
            var context = canvas.get(0).getContext('2d');
            context.canvas.width = context.canvas.width * 2.5;
            var chartTitle = chartName + ', Throughput (FPS) Precision: FP32 (Higher is better)';
            new Chart(context, {
                type: 'horizontalBar',
                data: getChartData(data),
                options: getChartOptions(chartTitle, true)
            });
            chartContainer.append(container);
            currentChart.append(chartContainer);
        }
    }
});

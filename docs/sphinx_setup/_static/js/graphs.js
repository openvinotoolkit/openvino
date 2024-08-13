// =================== GENERAL OUTPUT CONFIG =========================

const chartDisclaimers = {
    Value: 'Value: Performance/(No_of_sockets * Price_of_CPU_dGPU), where prices are in USD as of November 2023.',
    Efficiency: 'Efficiency: Performance/(No_of_sockets * TDP_of_CPU_dGPU), where total power dissipation (TDP) is in Watt as of November 2023.'
}

const defaultSelections = {
    platformTypes: { name: 'ietype', data: ['atom'] },
    // platforms: {
    //     name: 'platform',
    //     data: [
    //         'Intel 1 CPU-only'
    //     ]
    // },
    platformFilters: { name: 'coretype', data: ['CPU'] },
    models: {
        name: 'networkmodel',
        data: [
            'bert-base-cased'
        ]
    },
    // parameters: { name: 'parameters', data: ['Throughput', 'Latency'] },
    pracision: { name: 'precision', data: ['INT4', 'INT8'] }
}

// ====================================================


class Filter {

    // param: GraphData[], networkModels[]
    static FilterByNetworkModel(graphDataArr, networkModels) {
        // This is a bit obtuse, collect all options from all models
        // Some of them might return dupes, so convert them to a map, and get unique objects based on names
        const optionMap = new Map();
        networkModels.map((model) => graphDataArr.filter((graphData => graphData.Model === model)))
            .flat(1)
            .forEach(item => optionMap.set(item.Platform, item));
        return Array.from(optionMap.values());
    }

    // param: GraphData[], ieType
    static ByIeType(graphDataArr, value) {
        return graphDataArr.filter((data) => data.PlatformType.includes(value));
    }

    // param: GraphData[], clientPlatforms
    static ByIeKpis(graphDataArr, clientPlatforms) {
        var kpis = []
        clientPlatforms.forEach((platformName) => {
            graphDataArr.filter((data) => {
                if (data.Platform.includes(platformName)) {
                    for (var key in data.Parameters) {
                        if (!kpis.includes(key)) kpis.push(key)
                    }
                }
            })
        })
        return kpis;
    }

    // param: GraphData[]
    static getKpis(graphDataArr) {
        var kpis = []
        graphDataArr.filter((data) => {
            for (var key in data.Parameters) {
                if (!kpis.includes(key.charAt(0).toUpperCase() + key.slice(1))) kpis.push(key.charAt(0).toUpperCase() + key.slice(1))
            }
        })
        return kpis;
    }

    // param: GraphData[], clientPlatforms[]
    static ByClientPlatforms(graphDataArr, platformsArr) {
        return graphDataArr.filter((data) => {
            return platformsArr.includes(data.Platform)
        });
    }

    // param: GraphData[], coreTypes[]
    static FilterByCoreTypes(graphDataArr, coreTypes) {
        if (coreTypes) {
            return graphDataArr.filter((data) => coreTypes.includes(data.PlatformType));
        }
        return graphDataArr;
    }
}

class GraphData {
    constructor(excelData) {
        if (!excelData) {
            return;
        }
    }
}

class Modal {
    static getIeTypeLabel(ietype) {
        switch (ietype) {
            case 'core':
                return 'Client Platforms (Intel® Core™)';
            case 'xeon':
                return 'Server Platforms (Intel® Xeon®)';
            case 'atom':
                return 'Mobile Platforms (Intel® Atom™)';
            case 'accel':
                return 'Accelerator Platforms';
            default:
                return '';
        }
    }
    static getCoreTypesLabels() {
        return ['CPU', 'iGPU\\NPU', 'CPU+iGPU'];
    }

    static getPrecisionsLabels(graphDataArr) {
        var kpis = []
        graphDataArr.filter((data) => {
            for (var key in data.Parameters) {
                data.Parameters[key].Precisions.forEach((key) => {
                    Object.keys(key).forEach((key) => {
                        if (!kpis.includes(key.toUpperCase())) kpis.push(key.toUpperCase())
                    });
                })
            }
        })
        return kpis;
    }

    static getCoreTypes(labels) {
        return labels.map((label) => {
            switch (label) {
                case 'CPU':
                    return 'core';
                case 'iGPU\\NPU':
                    return 'core-iGPU';
                case 'CPU+iGPU':
                    return 'core-CPU+iGPU';
                default:
                    return '';
            }
        });
    }
    static getPrecisions(labels) {
        return labels.map((label) => {
            switch (label) {
                case 'OV-INT8 (reference)':
                    return 'ovmsint8';
                case 'OV-FP32 (reference)':
                    return 'ovmsfp32';
                case 'INT4':
                    return 'int4';
                case 'INT8':
                    return 'int8';
                case 'FP16':
                    return 'fp16';
                case 'FP32':
                    return 'fp32';
                case 'BF16':
                    return 'bf16';
                default:
                    return '';
            }
        });
    }
    static getUnitDescription(unit) {
        switch (unit) {
            case 'msec.':
                return '(lower is better)';
            case 'msec/token':
                return '(lower is better)';
            case 'Generating time, sec.':
                return '(lower is better)';
            case 'msec/token/TDP':
                return '(lower is better)';
            case 'FPS':
                return '(higher is better)';
            case 'FPS/$':
                return '(higher is better)';
            case 'FPS/TDP':
                return '(higher is better)';
            default:
                return '';
        }
    }
}


class Graph {
    constructor(data) {
        this.data = data;
    }
    data = new GraphData();

    // functions to get unique keys 
    static getNetworkModels(graphDataArr) {
        return Array.from(new Set(graphDataArr.map((obj) => obj.Model)));
    }
    static getIeTypes(graphDataArr) {
        return Array.from(new Set(graphDataArr.map((obj) => obj.PlatformType)));
    }
    static getCoreTypes(graphDataArr) {
        return Array.from(new Set(graphDataArr.map((obj) => obj.ieType)));
    }

    // param: GraphData[]
    static getPlatformNames(graphDataArr) {
        return graphDataArr.map((data) => data.Platform);
    }

    // param: GraphData[], kpi: string
    static getDatabyKPI(graphDataArr, kpi, precisions) {
        // return graphDataArr.map((item) => item.Parameters[kpi]?.Precisions);

        if (graphDataArr[0].Parameters[kpi] !== undefined) {
            return graphDataArr.map((item) => item.Parameters[kpi].Precisions);
        }
        else {
            var array = [];
            precisions.forEach((label) => {
                var data = {};
                data[label] = "1000";
                array.push(data)
            })
            console.log([[{ int4: '34', int8: '155', fp16: '30'}],[{ int4: '134', int8: '55', fp16: '134'}],[{ int4: '12', int8: '45', fp16: '234'}]])
            return [[[{ int4: '34', int8: '155', fp16: '30'}],[{ int4: '134', int8: '55', fp16: '134'}],[{ int4: '12', int8: '45', fp16: '234'}]]];
        }
        
    }


    // this returns an object that is used to ender the chart
    static getGraphConfig(kpi, item, precisions) {
        switch (kpi) {
            case 'throughput':
                return {
                    chartTitle: 'Throughput',
                    iconClass: 'throughput-icon',
                    unit: item.Parameters[kpi].Unit,
                    datasets: precisions.map((precision) => this.getPrecisionThroughputConfig(precision)),
                };
            case 'latency':
                return {
                    chartTitle: 'Latency',
                    iconClass: 'latency-icon',
                    unit: item.Parameters[kpi].Unit,
                    datasets: precisions.map((precision) => this.getPrecisionThroughputConfig(precision)),
                };
            case 'value':
                return {
                    chartTitle: 'Value',
                    iconClass: 'value-icon',
                    unit: item.Parameters[kpi]?.Unit,
                    datasets: precisions.map((precision) => this.getPrecisionThroughputConfig(precision))
                };
            case 'efficiency':
                return {
                    chartTitle: 'Efficiency',
                    iconClass: 'efficiency-icon',
                    unit: item.Parameters?.kpi?.Unit,
                    datasets: precisions.map((precision) => this.getPrecisionThroughputConfig(precision))
                };
            default:
                return {};
        }
    }

    static getPrecisionThroughputConfig(precision) {
        switch (precision) {
            case 'int4':
                return { data: null, color: '#5bd0f0', label: `INT4` };
            case 'int8':
                return { data: null, color: '#00C7FD', label: `INT8` };
            case 'fp16':
                return { data: null, color: '#009fca', label: `FP16` };
            case 'fp32':
                return { data: null, color: '#007797', label: `FP32` };
            case 'bf16':
                return { data: null, color: '#00536a', label: `BF16` };
            default:
                return {};
        }
    }

    static getGraphPlatformText(platform) {
        switch (platform) {
            case 'atom':
                return 'Mobile Platforms';
            case 'core':
                return 'Client Platforms';
            case 'xeon':
                return 'Server Platforms';
            case 'accel':
                return 'Accelerated Platforms';
            default:
                return '';
        }
    }
}


class ChartDisplay {
    constructor(mode, numberOfCharts) {
        this.mode = mode;
        this.numberOfChartsInRow = numberOfCharts;
    }
}

$(document).ready(function () {

    $('.ov-toolkit-benchmark-results').on('click', () => showModal('ov'));
    $('.ovms-toolkit-benchmark-results').on('click', () => showModal('ovms'));

    function clickBuildGraphs(graph, networkModels, ietype, platforms, kpis, precisions) {
        renderData(graph, networkModels, ietype, platforms, kpis, precisions);
        $('.modal-footer').show();
        $('#modal-display-graphs').show();
        $('.edit-settings-btn').off('click').on('click', (event) => {
            $('#modal-configure-graphs').show();
            $('#modal-display-graphs').hide();
            $('.modal-footer').hide();
            $('.chart-placeholder').empty();
        });

        $('.graph-chart-title-header').off('click').on('click', (event) => {
            var parent = event.target.parentElement;

            if ($(parent).children('.chart-wrap,.empty-chart-container').is(":visible")) {
                $(parent).children('.chart-wrap,.empty-chart-container').hide();
                $(parent).children('.chevron-right-btn').show();
                $(parent).children('.chevron-down-btn').hide();
                $
            } else {
                $(parent).children('.chart-wrap,.empty-chart-container').show();
                $(parent).children('.chevron-down-btn').show();
                $(parent).children('.chevron-right-btn').hide();
            }
        });
    }

    function hideModal() {
        $('#graphModal').remove();
        $('body').css('overflow', 'auto');
    }

    function showModal() {
        $('body').css('overflow', 'hidden');

        fetch('../_static/benchmarks_files/graph-data.json')
            .then((response) => response.json())
            .then((json) => renderModal(json));
        ;
    }

    function getSelectedNetworkModels() {
        return $('.models-column-one input:checked, .models-column-two input:checked').not('[data-networkmodel="Select All"]').map(function () {
            return $(this).data('networkmodel');
        }).get();
    }
    function getSelectedIeType() {
        return $('.ietype-column input:checked').map(function () {
            return $(this).data('ietype');
        }).get().pop();
    }
    function getSelectedClientPlatforms() {
        return $('.client-platform-column input:checked').map(function () {

            return $(this).data('platform');
        }).get();
    }
    function getSelectedKpis() {
        return $('.kpi-column input:checked').map(function () {
            return $(this).data('kpi');
        }).get();
    }
    function getSelectedPrecisions() {
        return $('.precisions-column input:checked').map(function () {
            return $(this).data('precision');
        }).get();
    }

    function validateSelections() {
        if (getSelectedNetworkModels().length > 0
            && getSelectedIeType()
            && getSelectedClientPlatforms().length > 0
            && getSelectedKpis().length > 0) {
            if (getSelectedKpis().includes('Throughput')) {
                if (getSelectedPrecisions().length > 0) {
                    $('#build-graphs-btn').prop('disabled', false);
                    return;
                }
                $('#build-graphs-btn').prop('disabled', true);
                return;
            }
            $('#build-graphs-btn').prop('disabled', false);
            return;
        }
        $('#build-graphs-btn').prop('disabled', true);
    }

    function renderModal(graph) {
        new Graph(graph);
        var networkModels = Graph.getNetworkModels(graph);
        var ieTypes = Graph.getIeTypes(graph);
        fetch('../_static/html/modal.html').then((response) => response.text()).then((text) => {

            // generate and configure modal container
            var modal = $('<div>');
            modal.attr('id', 'graphModal');
            modal.addClass('modal');
            var modalContent = $(text);
            modalContent.attr('id', 'graphModalContent');
            modalContent.addClass('modal-content');
            modal.append(modalContent);

            const models = networkModels.map((networkModel) => createCheckMark(networkModel, 'networkmodel'));
            const selectAllModelsButton = createCheckMark('Select All', 'networkmodel')
            modal.find('.models-column-one').append(selectAllModelsButton).append(models.slice(0, models.length / 2));
            modal.find('.models-column-two').append(models.slice(models.length / 2));

            const precisions = Modal.getPrecisionsLabels(graph).map((precision) => createCheckMark(precision, 'precision', false));
            modal.find('.precisions-column').append(precisions);
            selectAllCheckboxes(precisions);
            disableAllCheckboxes(precisions);

            const types = ieTypes.map((ieType) => {
                var labelText = Modal.getIeTypeLabel(ieType);
                if (labelText) {
                    const item = $('<label class="checkmark-container">');
                    const checkboxSpan = $('<span class="checkmark radiobutton">');
                    item.text(labelText);
                    const radio = $('<input type="radio" name="ietype"/>');
                    item.append(radio);
                    item.append(checkboxSpan);
                    radio.attr('data-ietype', ieType);
                    return item;
                }
            });
            modal.find('#modal-display-graphs').hide();
            modal.find('.ietype-column').append(types);
            modal.find('.ietype-column input').first().prop('checked', true);

            const kpiLabels = Filter.getKpis(graph).map((kpi) => createCheckMark(kpi, 'kpi', true));
            modal.find('.kpi-column').append(kpiLabels);

            $('body').prepend(modal);
            renderClientPlatforms(graph, modal, true);
            preselectDefaultSettings(graph, modal);

            $('.clear-all-btn').on('click', clearAll);
            $('#build-graphs-btn').on('click', () => {
                $('#modal-configure-graphs').hide();
                clickBuildGraphs(graph, getSelectedNetworkModels(), getSelectedIeType(), getSelectedClientPlatforms(), getSelectedKpis(), Modal.getPrecisions(getSelectedPrecisions()));
            });
            $('.modal-close').on('click', hideModal);
            $('.close-btn').on('click', hideModal);
            modal.find('.models-column-one input[data-networkmodel="Select All"]').on('click', function () {
                if ($(this).prop('checked'))
                    selectAllCheckboxes(models);
                else deSelectAllCheckboxes(models);
            });
            modal.find('.ietype-column input').on('click', () => renderClientPlatforms(graph, modal));
            modal.find('.client-platform-column input').on('click', () => enabledKpis(graph, getSelectedClientPlatforms(), modal));

            modal.find('.kpi-column input').on('click', validateThroughputSelection);
            modal.find('input').on('click', validateSelections);
        });
    }

    function validateThroughputSelection() {
        const precisions = $('.precisions-column').find('input')
        if (getSelectedKpis().includes('Throughput') || getSelectedKpis().includes('Latency')) {
            precisions.prop('disabled', false);
        }
        else {
            precisions.prop('disabled', true);
        }
    }

    function clearAll() {
        $('.modal-content-grid-container input:checkbox').each((index, object) => $(object).prop('checked', false));
        validateThroughputSelection();
        validateSelections();
    }

    //do not change this
    function preselectDefaultSettings(graph, modal) {

        if (defaultSelections.platformTypes) {
            const type = defaultSelections.platformTypes.data[0];
            $(`input[data-ietype="${type}"]`).prop('checked', true);
            renderClientPlatforms(graph, modal);
        }
        if (defaultSelections.platformFilters) {
            const filters = modal.find('.selectable-box-container').children('.selectable-box');
            filters.removeClass('selected');
            defaultSelections.platformFilters.data.forEach(selection => {
                filters.filter(`[data-${defaultSelections.platformFilters.name}="${selection}"]`).addClass('selected');
            });
            renderClientPlatforms(graph, modal);
        }
        clearAll();
        for (setting in defaultSelections) {
            let name = defaultSelections[setting].name;
            defaultSelections[setting].data.forEach(selection => {
                $(`input[data-${name}="${selection}"]`).prop('checked', true);
            });
        }
        validateThroughputSelection();
        validateSelections();
    }

    function showCoreSelectorTypes(coreTypes, graphDataArr, modal) {
        if ($('.client-platform-column').find('.selectable-box-container').length) {
            $('.client-platform-column').find('.selectable-box-container').show();
            return;
        }
        var container = $('<div>');
        container.addClass('selectable-box-container');
        coreTypes.forEach((type) => {
            var box = $('<div>' + type + '</div>');
            box.attr('data-coretype', type);
            box.addClass('selectable-box selected');
            container.append(box);
        });
        $('.client-platform-column').prepend(container);
        $('.client-platform-column .selectable-box').on('click', function () {
            if ($(this).hasClass('selected')) {
                $(this).removeClass('selected');
            } else {
                $(this).addClass('selected');
            }
            var fPlatforms = filterClientPlatforms(graphDataArr, getSelectedIeType());
            renderClientPlatformsItems(modal, Graph.getPlatformNames(fPlatforms), true);
            validateSelections();
        });
    }

    function hideCoreSelectorTypes() {
        $('.client-platform-column').find('.selectable-box-container').hide();
    }

    function filterClientPlatforms(graph, ietype) {
        return Filter.ByIeType(graph, ietype);
    }

    function renderClientPlatforms(graph, modal) {
        if (getSelectedIeType() === 'core') {
            showCoreSelectorTypes(Modal.getCoreTypesLabels(), data, modal);
        }
        else {
            hideCoreSelectorTypes();
        }
        var fPlatforms = filterClientPlatforms(graph, getSelectedIeType());
        renderClientPlatformsItems(modal, Graph.getPlatformNames(fPlatforms));
    }

    function enabledKpis(graph, clientPlatforms) {
        var allKpis = Filter.getKpis(graph);

        allKpis.forEach((kpi) => {
            $(`input[data-kpi="${kpi.charAt(0).toUpperCase() + kpi.slice(1)}"]`).prop('disabled', true);
        }
        )

        var kpis = Filter.ByIeKpis(graph, clientPlatforms);
        kpis.forEach((kpi) => {
            $(`input[data-kpi="${kpi.charAt(0).toUpperCase() + kpi.slice(1)}"]`).prop('disabled', false);
        })

        return null;
    }

    function renderClientPlatformsItems(modal, platformNames) {
        $('.client-platform-column .checkmark-container').remove();
        const clientPlatforms = platformNames.map((platform) => createCheckMark(platform, 'platform', false));
        selectAllCheckboxes(clientPlatforms);
        modal.find('.client-platform-column').append(clientPlatforms);
        modal.find('.client-platform-column input').on('click', validateSelections);
    }

    function createCheckMark(itemLabel, modelLabel, disabled) {
        const item = $('<label class="checkmark-container">');
        item.text(itemLabel);
        const checkbox = $('<input type="checkbox"/>');
        checkbox.prop('disabled', disabled);
        const checkboxSpan = $('<span class="checkmark">');
        item.append(checkbox);
        item.append(checkboxSpan);
        checkbox.attr('data-' + modelLabel, itemLabel);
        return item;
    }

    // receives a jquery list of items and selects all input checkboxes
    function selectAllCheckboxes(items) {
        items.forEach((item) => {
            item.find(':input').prop('checked', true);
        });
    }

    function disableAllCheckboxes(items) {
        items.forEach((item) => {
            item.find(':input').prop('disabled', true);
        })
    }

    function deSelectAllCheckboxes(items) {
        items.forEach((item) => {
            item.find(':input').prop('checked', false);
        });
    }

    // =================== HTMLLEGEND =========================

    const getOrCreateLegendList = (chart, id) => {
        const legendContainer = document.getElementById(id);
        let listContainer = legendContainer.querySelector('ul');

        if (!listContainer) {
            listContainer = document.createElement('ul');
            listContainer.style.display = 'flex';
            listContainer.style.flexDirection = 'row';
            listContainer.style.margin = 0;
            listContainer.style.padding = 0;
            listContainer.style.paddingLeft = '0px';

            legendContainer.appendChild(listContainer);
        }

        return listContainer;
    };

    const htmlLegendPlugin = {
        id: 'htmlLegend',
        afterUpdate(chart, args, options) {

            const ul = getOrCreateLegendList(chart, chart.options.plugins.htmlLegend.containerID);

            // Remove old legend items
            while (ul.firstChild) {
                ul.firstChild.remove();
            }

            // Reuse the built-in legendItems generator
            const items = chart.legend.legendItems;
            items.forEach(item => {
                const li = document.createElement('li');
                li.style.alignItems = 'center';
                li.style.display = 'block';
                li.style.flexDirection = 'column';
                li.style.marginLeft = '4px';

                li.onclick = () => {
                    const { type } = chart.config;
                    if (type === 'pie' || type === 'doughnut') {
                        // Pie and doughnut charts only have a single dataset and visibility is per item
                        chart.toggleDataVisibility(item.index);
                    } else {
                        chart.setDatasetVisibility(item.datasetIndex, !chart.isDatasetVisible(item.datasetIndex));
                    }
                    chart.update();
                };

                // Color box
                const boxSpan = document.createElement('span');
                boxSpan.style.background = item.fillStyle;
                boxSpan.style.borderColor = item.strokeStyle;
                boxSpan.style.borderWidth = item.lineWidth + 'px';
                boxSpan.style.display = 'inline-block';
                boxSpan.style.height = '10px';
                boxSpan.style.marginRight = '4px';
                boxSpan.style.width = '30px';

                // Text
                const textContainer = document.createElement('p');
                textContainer.style.color = '#666';
                textContainer.style.margin = 0;
                textContainer.style.padding = 0;
                textContainer.style.fontSize = '0.6rem';
                textContainer.style.marginLeft = '3px';
                textContainer.style.textDecoration = item.hidden ? 'line-through' : '';

                const text = document.createTextNode(item.text);
                textContainer.appendChild(text);

                li.appendChild(boxSpan);
                li.appendChild(textContainer);
                ul.appendChild(li);
            });
        }
    };

    function getChartOptions(title, containerId) {
        return {
            responsive: true,
            indexAxis: 'y',
            maintainAspectRatio: false,
            title: {
                display: false,
                text: title
            },
            scales: {
                x: {
                    ticks: {
                        beginAtZero: true
                    }
                },
                y: {
                    ticks: {
                        display: false,
                        beginAtZero: true
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                htmlLegend: {
                    containerID: containerId,
                }
            }
        }
    } Throughput

    function getChartDataNew(labels, datasets) {
        return {
            labels: labels,
            datasets: datasets.map((item) => {
                return {
                    label: item.label,
                    data: item.data,
                    backgroundColor: item.color,
                    borderColor: 'rgba(170,170,170,0)',
                    barThickness: 10
                }
            })
        }
    }

    function renderData(graph, networkModels, ietype, platforms, kpis, precisions) {
        $('.chart-placeholder').empty();
        $('.modal-footer').empty();
        const display = new ChartDisplay(getChartsDisplayMode(kpis.length), kpis.length);

        networkModels.forEach((networkModel) => {
            var chartName = networkModel;
            var chartSlug = chartName.replace(')', '').replace(' (', '-');
            var chartContainer = $('<div>');
            var chevronDown = '<span class="chevron-down-btn"></span>';
            var chevronRight = '<span style="display:none" class="chevron-right-btn"></span>';
            $(chevronRight).hide();

            var chartContainerHeader = $(chevronDown + chevronRight + '<span class="graph-chart-title">' + networkModel + '</span>');
            chartContainerHeader.addClass('graph-chart-title-header');
            chartContainer.prepend(chartContainerHeader);
            chartContainer.attr('id', 'ov-chart-container-' + chartSlug);

            chartContainer.addClass('chart-container');

            var filteredNetworkModels = Filter.FilterByNetworkModel(graph, [networkModel]);
            var filteredIeTypes = Filter.ByIeType(filteredNetworkModels, ietype);
            var filteredGraphData = Filter.ByClientPlatforms(filteredIeTypes, platforms);
            $('.chart-placeholder').append(chartContainer);
            if (filteredGraphData.length > 0) {
                createChartWithNewData(filteredGraphData, chartContainer, kpis, ietype, precisions, display);
            } else {
                createEmptyChartContainer(chartContainer);
            }
        })

        if (kpis.includes('Value') || kpis.includes('Efficiency')) {
            $('.modal-footer').append($('<div class="modal-line-divider"></div>'))
        }
        $('.modal-footer').append($('<div class="modal-footer-content"><div class="modal-disclaimer-box"></div></div>'))

        for (let kpi of kpis) {
            if (chartDisclaimers[kpi])
                $('.modal-disclaimer-box').append($('<p>').text(chartDisclaimers[kpi]))
        }

        $(window).off('resize');
        $(window).resize(() => resetChartsDisplay(display));
    };

    function createEmptyChartContainer(chartContainer) {
        chartContainer.append($('<div>').addClass('empty-chart-container').text('No data for this configuration.'));
    }

    // this function should take the final data set and turn it into graphs
    // params: GraphData, unused, chartContainer
    function createChartWithNewData(model, chartContainer, kpis, ietype, precisions, display) {
        var chartWrap = $('<div>');
        chartWrap.addClass('chart-wrap');
        chartContainer.append(chartWrap);
        var labels = Graph.getPlatformNames(model);
        var graphConfigs = kpis.map((str) => {
            var kpi = str.toLowerCase();
            var groupUnit = model[0];
            var kpiData = Graph.getDatabyKPI(model, kpi, precisions);
            var config = Graph.getGraphConfig(kpi, groupUnit, precisions);
            precisions.forEach((precision, index) => {
                config.datasets[index].data = kpiData.map(tData => tData[0][precision]);
            });
            console.log(config);
            return config;
        });


        // get the client platform labels and create labels for all the graphs
        var labelsContainer = $('<div>');
        labelsContainer.addClass('chart-labels-container');
        chartWrap.append(labelsContainer);

        // get the kpi title's and create headers for the graphs 
        var chartGraphsContainer = $('<div>');
        chartGraphsContainer.addClass('chart-graphs-container');
        chartWrap.append(chartGraphsContainer);

        graphConfigs.forEach((graphConfig, index) => {
            const id = getRandomNumber();
            var graphItem = $(`<div id=${id}>`);
            graphItem.addClass('graph-item');
            var columnHeaderContainer = $('<div>');
            columnHeaderContainer.addClass('chart-column-title');
            var columnIcon = $('<div class="icon">');
            columnIcon.addClass(graphConfig.iconClass);
            columnHeaderContainer.append(columnIcon);
            var columnHeader = $('<div class="chart-header">');
            columnHeader.append($('<div class="title">' + graphConfig.chartTitle + '</div>'));
            columnHeader.append($('<div class="title">' + Graph.getGraphPlatformText(ietype) + '</div>'));
            columnHeader.append($('<div class="subtitle">' + graphConfig.unit + ' ' + Modal.getUnitDescription(graphConfig.unit) + '</div>'));

            columnHeaderContainer.append(columnHeader);
            chartGraphsContainer.append(graphItem);
            var graphClass = $('<div>');
            graphClass.addClass('graph-row');

            graphItem.append(columnHeaderContainer);
            graphItem.append(graphClass);
            processMetricNew(labels, graphConfig.datasets, graphConfig.chartTitle, graphClass, 'graph-row-column', id);

            window.setTimeout(() => {
                const topPadding = getLabelsTopPadding(display.mode);
                const labelsHeight = (labels.length * 55);
                const chartHeight = $(graphItem).outerHeight();
                const bottomPadding = (chartHeight - (topPadding + labelsHeight));

                var labelsItem = $('<div>');
                labelsItem.addClass('chart-labels-item');

                labels.forEach((label) => {
                    labelsItem.append($('<div class="title">' + label + '</div>'));
                });

                labelsItem.css('padding-top', topPadding + 'px');
                labelsItem.css('padding-bottom', bottomPadding + 'px');
                setInitialItemsVisibility(labelsItem, index, display.mode);
                labelsContainer.append(labelsItem);
            });
        });
        setChartsDisplayDirection(display.mode);
        adjustHeaderIcons(display.mode);
    }

    function processMetricNew(labels, datasets, chartTitle, container, widthClass, id) {
        // ratio for consistent chart label height
        var heightRatio = (30 + (labels.length * 55));
        var chart = $('<div>');
        const containerId = `legend-container-${id}`;
        const legend = $(`<div id="${containerId}">`);
        legend.addClass('graph-legend-container');
        chart.addClass('chart');
        chart.addClass(widthClass);
        chart.height(heightRatio);
        var canvas = $('<canvas>');
        chart.append(canvas);
        container.append(chart);
        container.append(legend);
        var context = canvas.get(0).getContext('2d');
        context.canvas.height = heightRatio;
        window.setTimeout(() => {
            new Chart(context, {
                type: 'bar',
                data: getChartDataNew(labels, datasets),
                options: getChartOptions(chartTitle, containerId),
                plugins: [htmlLegendPlugin]
            });
        });
    }

    function getRandomNumber() {
        return Math.floor(Math.random() * 100000);
    }

    function resetChartsDisplay(currentDisplay) {
        const newDisplayMode = getChartsDisplayMode(currentDisplay.numberOfChartsInRow);
        if (currentDisplay.mode != newDisplayMode) {
            currentDisplay.mode = newDisplayMode;
            setChartsDisplayDirection(currentDisplay.mode);
            adjustLabels(currentDisplay.mode);
            adjustHeaderIcons(currentDisplay.mode);
        }
    }

    function adjustLabels(displayMode) {
        const firstLabels = $('.chart-labels-container').find('.chart-labels-item:first-child');
        const labels = $('.chart-labels-container').find('.chart-labels-item');
        labels.css('padding-top', getLabelsTopPadding(displayMode));
        if (displayMode == 'column') {
            labels.show();
        }
        else {
            labels.hide()
            firstLabels.show();
        }
    }

    function adjustHeaderIcons(displayMode) {
        const icons = $('.graph-item').find('.chart-column-title');
        if (displayMode == 'rowCompact')
            icons.css('flex-direction', 'column')
        else
            icons.css('flex-direction', 'row')
    }

    function getLabelsTopPadding(displayMode) {
        return (displayMode == 'rowCompact') ? 105.91 : 83.912;
    }

    function setChartsDisplayDirection(displayMode) {
        const container = $('.chart-placeholder').find('.chart-graphs-container');
        if (displayMode == 'column') {
            container.css('flex-direction', 'column');
        }
        else {
            container.css('flex-direction', 'row');
        }
    }

    function setInitialItemsVisibility(item, count, displayMode) {
        if (count == 0 || displayMode == 'column') item.show();
        else item.hide();
    }

    function getChartsDisplayMode(numberOfCharts) {
        switch (numberOfCharts) {
            case 4:
                return window.matchMedia('(max-width: 721px)').matches ? 'column'
                    : window.matchMedia('(max-width: 830px)').matches ? 'rowCompact'
                        : 'row';
            case 3:
                return window.matchMedia('(max-width: 569px)').matches ? 'column'
                    : window.matchMedia('(max-width: 649px)').matches ? 'rowCompact'
                        : 'row';
            case 2:
                return window.matchMedia('(max-width: 500px)').matches ? 'column'
                    : 'row';
            default:
                return 'row';
        }
    }
});

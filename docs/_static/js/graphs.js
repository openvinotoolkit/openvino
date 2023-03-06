// general output config
const chartDisclaimers = {
    Value: 'Value: Performance/(No_of_sockets * Price_of_CPU_dGPU), where prices are in USD as of December 2022.',
    Efficiency: 'Efficiency: Performance/(No_of_sockets * TDP_of_CPU_dGPU), where total power dissipation (TDP) is in Watt as of December 2022.'
}

const defaultSelections = {
    platforms: {name: 'platform',
        data: [
            'Intel® Core™  i9-12900K CPU-only',
            'Intel® Core™  i5-10500TE CPU-only',
            'Intel® Core™  i5-8500 CPU-only',
            'Intel® Core™  i7-8700T CPU-only',
            'Intel® Core™  i9-10900TE CPU-only',
            'Intel® Core™  i7-1165G7 CPU-only'
        ]
    },
    platformFilters: {name: 'coretype', data: ['CPU']},
    models: {name: 'networkmodel',
        data: [
            'bert-large-uncased-whole-word-masking-squad-0001 ',
            'mobilenet-ssd ',
            'resnet-50',
            'yolo_v3_tiny'
        ]
    },
    parameters: {name: 'kpi', data: ['Throughput']},
    pracision: {name: 'precision', data: ['INT8', 'FP32']}
}

class Filter {

    // param: GraphData[], networkModels[]
    static FilterByNetworkModel(graphDataArr, networkModels) {
        // This is a bit obtuse, collect all options from all models
        // Some of them might return dupes, so convert them to a map, and get unique objects based on names
        const optionMap = new Map();
        networkModels.map((model) => graphDataArr.filter((graphData => graphData.networkModel === model)))
          .flat(1)
          .forEach(item => optionMap.set(item.platformName, item));
        // convert the option map back to an array with just the values
        return Array.from(optionMap.values());
    }

    // param: GraphData[], ieType
    static FilterByIeType(graphDataArr, value) {
        return graphDataArr.filter((data) => data.ieType.includes(value));
    }

    // param: GraphData[], clientPlatforms[]
    static FilterByClientPlatforms(graphDataArr, platformsArr) {
        return graphDataArr.filter((data) => platformsArr.includes(data.platformName));
    }

    // param: GraphData[], coreTypes[]
    static FilterByCoreTypes(graphDataArr, coreTypes) {
        if (coreTypes) {
            return graphDataArr.filter((data) => coreTypes.includes(data.ieType));
        }
        return graphDataArr;
    }

    // param: GraphData[] (of one networkModel), key (throughput, latency, efficiency, value)
    static getKpiData(graphDataArr, key) {
        return graphDataArr.map((data) => {
            return data[key];
        });
    }
}
class ExcelDataTransformer {

    static transform(csvdata) {
        const entries = csvdata.filter((entry) => {
            return !entry.includes('begin_rec') && !entry.includes('end_rec');
        });
        // do other purging and data massaging here

        // else generate
        return entries.map((entry) => {
            return new GraphData(new ExcelData(entry));
        });
    }
}

class ExcelData {
    constructor(csvdataline) {
        if (!csvdataline) {
            return;
        }
        this.networkModel = csvdataline[0];
        this.release = csvdataline[1];
        this.ieType = csvdataline[2];
        this.platformName = csvdataline[3];
        this.throughputInt8 = csvdataline[4];
        this.throughputFP16 = csvdataline[5];
        this.throughputFP32 = csvdataline[6];
        this.value = csvdataline[7];
        this.efficiency = csvdataline[8];
        this.price = csvdataline[9];
        this.tdp = csvdataline[10];
        this.sockets = csvdataline[11];
        this.pricePerSocket = csvdataline[12];
        this.tdpPerSocket = csvdataline[13];
        this.latency = csvdataline[14];
    }
    networkModel = '';
    release = '';
    ieType = '';
    platformName = '';
    throughputInt8 = '';
    throughputFP16 = '';
    throughputFP32 = '';
    value = '';
    efficiency = '';
    price = '';
    tdp = '';
    sockets = '';
    pricePerSocket = '';
    tdpPerSocket = '';
    latency = '';
}


class GraphData {
    constructor(excelData) {
        if (!excelData) {
            return;
        }
        this.networkModel = excelData.networkModel;
        this.release = excelData.release;
        this.ieType = excelData.ieType;
        this.platformName = excelData.platformName;
        this.kpi = new KPI(
            new Precision(excelData.throughputInt8, excelData.throughputFP16, excelData.throughputFP32),
            excelData.value,
            excelData.efficiency,
            excelData.latency);
        this.price = excelData.price;
        this.tdp = excelData.tdp;
        this.sockets = excelData.sockets;
        this.pricePerSocket = excelData.pricePerSocket;
        this.tdpPerSocket = excelData.tdpPerSocket;
        this.latency = excelData.latency;
    }
    networkModel = '';
    platformName = '';
    release = '';
    ieType = '';
    kpi = new KPI();
    price = '';
    tdp = '';
    sockets = '';
    pricePerSocket = '';
    tdpPerSocket = '';
}

class KPI {
    constructor(precisions, value, efficiency, latency) {
        this.throughput = precisions;
        this.value = value;
        this.efficiency = efficiency;
        this.latency = latency;
    }
    throughput = new Precision();
    value = '';
    efficiency = '';
    latency = '';
}

class Precision {
    constructor(int8, fp16, fp32) {
        this.int8 = int8;
        this.fp16 = fp16;
        this.fp32 = fp32;
    }
    int8 = '';
    fp16 = '';
    fp32 = '';
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
        return ['CPU', 'iGPU', 'CPU+iGPU'];
    }
    static getKpisLabels() {
        return ['Throughput', 'Value', 'Efficiency', 'Latency'];
    }
    static getPrecisionsLabels() {
        return ['INT8', 'FP16', 'FP32'];
    }
    static getCoreTypes(labels) {
        return labels.map((label) => {
            switch (label) {
                case 'CPU':
                    return 'core';
                case 'iGPU':
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
                case 'INT8':
                    return 'int8';
                case 'FP16':
                    return 'fp16';
                case 'FP32':
                    return 'fp32';
                default:
                    return '';
            }
        });
    }
}

class Graph {
    constructor(data) {
        this.data = data;
    }
    data = new GraphData();

    // functions to get unique keys 
    static getNetworkModels(graphDataArr) {
        return Array.from(new Set(graphDataArr.map((obj) => obj.networkModel)));
    }
    static getIeTypes(graphDataArr) {
        return Array.from(new Set(graphDataArr.map((obj) => obj.ieType)));
    }
    static getPlatforms(graphDataArr) {
        return Array.from(new Set(graphDataArr.map((obj) => obj.platformName)));
    }
    static getCoreTypes(graphDataArr) {
        return Array.from(new Set(graphDataArr.map((obj) => obj.ieType)));
    }

    // param: GraphData[]
    static getPlatformNames(graphDataArr) {
        return graphDataArr.map((data) => data.platformName);
    }

    // param: GraphData[], kpi: string
    static getDatabyKPI(graphDataArr, kpi) {
        switch (kpi) {
            case 'throughput':
                return graphDataArr.map((data) => data.kpi.throughput);
            case 'latency':
                return graphDataArr.map((data) => data.kpi.latency);
            case 'efficiency':
                return graphDataArr.map((data) => data.kpi.efficiency);
            case 'value':
                return graphDataArr.map((data) => data.kpi.value);
            default:
                return [];
        }
    }

    // this returns an object that is used to ender the chart
    static getGraphConfig(kpi, precisions) {
        switch (kpi) {
            case 'throughput':
                return {
                    chartTitle: 'Throughput',
                    chartSubtitle: '(higher is better)',
                    iconClass: 'throughput-icon',
                    datasets: precisions.map((precision) => this.getPrecisionConfig(precision)),
                };
            case 'latency':
                return {
                    chartTitle: 'Latency',
                    chartSubtitle: '(lower is better)',
                    iconClass: 'latency-icon',
                    datasets: [{ data: null, color: '#8F5DA2', label: 'Milliseconds' }],
                };
            case 'value':
                return {
                    chartTitle: 'Value',
                    chartSubtitle: '(higher is better)',
                    iconClass: 'value-icon',
                    datasets: [{ data: null, color: '#8BAE46', label: 'FPS/$ (INT8)' }],
                };
            case 'efficiency':
                return {
                    chartTitle: 'Efficiency',
                    chartSubtitle: '(higher is better)',
                    iconClass: 'efficiency-icon',
                    datasets: [{ data: null, color: '#E96115', label: 'FPS/TDP (INT8)' }],
                };
            default:
                return {};
        }
    }

    static getPrecisionConfig(precision) {
        switch (precision) {
            case 'int8':
                return { data: null, color: '#00C7FD', label: 'FPS (INT8)' };
            case 'fp16':
                return { data: null, color: '#009fca', label: 'FPS (FP16)' };
            case 'fp32':
                return { data: null, color: '#007797', label: 'FPS (FP32)' };
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

$(document).ready(function () {

    $('.ov-toolkit-benchmark-results').on('click', showModal);

    function clickBuildGraphs(graph, networkModels, ietype, platforms, kpis, precisions) {
        renderData(graph, networkModels, ietype, platforms, kpis, precisions);

        $('.modal-footer').show();
        $('#modal-display-graphs').show();
        $('.edit-settings-btn').on('click', (event) => {
            $('#modal-configure-graphs').show();
            $('#modal-display-graphs').hide();
            $('.modal-footer').hide();
            $('.chart-placeholder').empty();
        });

        $('.graph-chart-title-header').on('click', (event) => {
            var parent = event.target.parentElement;

            if ($(parent).children('.chart-wrap.container,.empty-chart-container').is(":visible")) {
                $(parent).children('.chart-wrap.container,.empty-chart-container').hide();
                $(parent).children('.chevron-right-btn').show();
                $(parent).children('.chevron-down-btn').hide();
                $
            } else {
                $(parent).children('.chart-wrap.container,.empty-chart-container').show();
                $(parent).children('.chevron-down-btn').show();
                $(parent).children('.chevron-right-btn').hide();
            }
        });
    }

    function hideModal() {
        $('#graphModal').hide();
        $('body').css('overflow', 'auto');
    }
    
    function showModal() {
        $('body').css('overflow', 'hidden');
        if ($('#graphModal').length) {
            $('#graphModal').show();
            return;
        }

        const dataPath = '_static/benchmarks_files/benchmark-data.csv';
        Papa.parse(dataPath, {
            download: true,
            complete: renderModal
        });
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
    function getSelectedCoreTypes() {
        return $('.client-platform-column .selected').map(function () {
            return $(this).data('coretype');
        }).get();
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

    function renderModal(result) {
        // remove header from csv line
        result.data.shift();
        var graph = new Graph(ExcelDataTransformer.transform(result.data));

        var networkModels = Graph.getNetworkModels(graph.data);
        var ieTypes = Graph.getIeTypes(graph.data);

        fetch('_static/html/modal.html').then((response) => response.text()).then((text) => {

            // generate and configure modal container
            var modal = $('<div>');
            modal.attr('id', 'graphModal');
            modal.addClass('modal');
            // generate and configure modal content from html import
            var modalContent = $(text);
            modalContent.attr('id', 'graphModalContent');
            modalContent.addClass('modal-content');
            modal.append(modalContent);

            const models = networkModels.map((networkModel) => createCheckMark(networkModel, 'networkmodel'));
            const selectAllModelsButton = createCheckMark('Select All', 'networkmodel')
            modal.find('.models-column-one').append(selectAllModelsButton).append(models.slice(0, models.length / 2));
            modal.find('.models-column-two').append(models.slice(models.length / 2));

            const precisions = Modal.getPrecisionsLabels().map((precision) => createCheckMark(precision, 'precision'));
            modal.find('.precisions-column').append(precisions);
            selectAllCheckboxes(precisions);
            disableAllCheckboxes(precisions);

            const types = ieTypes.map((ieType) => {
                var labelText = Modal.getIeTypeLabel(ieType);
                if (labelText) {
                    const item = $('<label class="checkmark-container">');
                    const checkboxSpan = $('<span class="checkmark radiobutton">');
                    item.text(Modal.getIeTypeLabel(ieType));
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

            const kpiLabels = Modal.getKpisLabels().map((kpi) => createCheckMark(kpi, 'kpi'));
            modal.find('.kpi-column').append(kpiLabels);

            $('body').prepend(modal);

            renderClientPlatforms(graph.data, modal, true);
            preselectDefaultSettings(graph.data, modal);

            $('.clear-all-btn').on('click', clearAll);
            $('#build-graphs-btn').on('click', () => {
                $('#modal-configure-graphs').hide();
                clickBuildGraphs(graph, getSelectedNetworkModels(), getSelectedIeType(), getSelectedClientPlatforms(), getSelectedKpis(), Modal.getPrecisions(getSelectedPrecisions()));
            });
            $('.modal-close').on('click', hideModal);
            $('.close-btn').on('click', hideModal);
            modal.find('.models-column-one input[data-networkmodel="Select All"]').on('click', function() {
                if ($(this).prop('checked'))
                    selectAllCheckboxes(models);
                else deSelectAllCheckboxes(models);
            });
            modal.find('.ietype-column input').on('click', () => renderClientPlatforms(graph.data, modal, true));
            modal.find('.kpi-column input').on('click', validateThroughputSelection);
            modal.find('input').on('click', validateSelections);
        });
    }
    
    function validateThroughputSelection() {
        const precisions = $('.precisions-column').find('input')
        if (getSelectedKpis().includes('Throughput')) {
            precisions.prop('disabled', false);
        }
        else {
            precisions.prop('disabled', true);
        }
    }

    function clearAll() {
        $('.modal-content-grid-container input:checkbox').each((index, object) => $(object).prop('checked', false));
        // Uncomment if you want the Clear All button to reset the Platform Type column as well
        // modal.find('.ietype-column input').first().prop('checked', true);
        validateThroughputSelection();
        validateSelections();
    }

    function preselectDefaultSettings(data, modal) {
        if (defaultSelections.platformFilters) {
            const filters = modal.find('.selectable-box-container').children('.selectable-box');
            filters.removeClass('selected');
            defaultSelections.platformFilters.data.forEach(selection => {
                filters.filter(`[data-${defaultSelections.platformFilters.name}="${selection}"]`).addClass('selected');
            });
            renderClientPlatforms(data, modal);
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

    function showCoreSelectorTypes(coreTypes, graphDataArr,  modal) {
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
            var fPlatforms = filterClientPlatforms(graphDataArr, getSelectedNetworkModels(), getSelectedIeType(), Modal.getCoreTypes(getSelectedCoreTypes()));
            renderClientPlatformsItems(modal, Graph.getPlatformNames(fPlatforms), true);
            validateSelections();
        });
    }

    function hideCoreSelectorTypes() {
        $('.client-platform-column').find('.selectable-box-container').hide();
    }

    function filterClientPlatforms(data, networkModels, ietype, coreTypes) {
        // No longer filtering on the network type, if at some point we want the network type as a filter, uncomment this
        // var first = Filter.FilterByNetworkModel(data, networkModels);
        var second = Filter.FilterByIeType(data, ietype);
        if (ietype === 'core') {
          second = Filter.FilterByCoreTypes(second, coreTypes);
        }
        const optionMap = new Map();
        second.forEach(item => optionMap.set(item.platformName, item));
        return Array.from(optionMap.values());
    }

    function renderClientPlatforms(data, modal, preselectEveryItem) {
        if (getSelectedIeType() === 'core') {
            showCoreSelectorTypes(Modal.getCoreTypesLabels(), data, modal);
        }
        else {
            hideCoreSelectorTypes();
        }
        var fPlatforms = filterClientPlatforms(data, getSelectedNetworkModels(), getSelectedIeType(), Modal.getCoreTypes(getSelectedCoreTypes()));
        renderClientPlatformsItems(modal, Graph.getPlatformNames(fPlatforms), preselectEveryItem);
    }

    function renderClientPlatformsItems(modal, platformNames, preselectEveryItem) {
        $('.client-platform-column .checkmark-container').remove();
        const clientPlatforms = platformNames.map((platform) => createCheckMark(platform, 'platform'));
        if (preselectEveryItem)
            selectAllCheckboxes(clientPlatforms);
        modal.find('.client-platform-column').append(clientPlatforms);
        modal.find('.client-platform-column input').on('click', validateSelections);
    }

    function createCheckMark(itemLabel, modelLabel) {
        const item = $('<label class="checkmark-container">');
        item.text(itemLabel);
        const checkbox = $('<input type="checkbox"/>');
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

    function enableAllCheckboxes(items) {
        items.forEach((item) => {
            item.find(':input').prop('disabled', false);
        })
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

    function getChartOptions(title) {
        return {
            responsive: true,
            maintainAspectRatio: false,
            legend: { display: true, position: 'bottom' },
            title: {
                display: false,
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
                        display: false, //this will remove only the label
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

    // params: string[], Datasets[]
    function getChartDataNew(labels, datasets) {
        return {
            labels: labels,
            datasets: datasets.map((item) => {
                return {
                    label: item.label,
                    data: item.data,
                    backgroundColor: item.color,
                    borderColor: 'rgba(170,170,170,0)',
                    barThickness: 12
                }
            })
        }
    }

    function renderData(graph, networkModels, ietype, platforms, kpis, precisions) {

        $('.chart-placeholder').empty();
        $('.modal-disclaimer-box').empty();
        networkModels.forEach((networkModel) => {
            var chartName = networkModel;
            var chartSlug = chartName.replace(')', '').replace(' (', '-');
            var chartContainer = $('<div>');

            var chevronDown = '<span class="chevron-down-btn"></span>';
            var chevronRight = '<span style="display:none" class="chevron-right-btn"></span>';
            $(chevronRight).hide();
            var chartContainerHeader = $('<span class="graph-chart-title">' + networkModel + '</span>' + chevronDown + chevronRight);
            chartContainerHeader.addClass('graph-chart-title-header');
            chartContainer.prepend(chartContainerHeader);
            chartContainer.attr('id', 'ov-chart-container-' + chartSlug);

            chartContainer.addClass('chart-container');
            chartContainer.addClass('container');

            var filteredNetworkModels = Filter.FilterByNetworkModel(graph.data, [networkModel]);
            var filteredIeTypes = Filter.FilterByIeType(filteredNetworkModels, ietype);
            var filteredGraphData = Filter.FilterByClientPlatforms(filteredIeTypes, platforms);

            $('.chart-placeholder').append(chartContainer);
            if (filteredGraphData.length > 0) {
                createChartWithNewData(filteredGraphData, chartContainer, kpis, ietype, precisions);
            } else {
              createEmptyChartContainer(chartContainer);
            }
        })
       
        for (let kpi of kpis) {
            if (chartDisclaimers[kpi])
                $('.modal-disclaimer-box').append($('<p>').text(chartDisclaimers[kpi]))
        }
    };

    function createEmptyChartContainer(chartContainer) {
      chartContainer.append($('<div>').addClass('empty-chart-container').text('No data for this configuration.'));
    }


    // this function should take the final data set and turn it into graphs
    // params: GraphData, unused, chartContainer
    function createChartWithNewData(model, chartContainer, kpis, ietype, precisions) {
        var chartWrap = $('<div>');
        chartWrap.addClass('chart-wrap');
        chartWrap.addClass('container');
        chartContainer.append(chartWrap);
        var labels = Graph.getPlatformNames(model);

        var graphConfigs = kpis.map((str) => {
            var kpi = str.toLowerCase();
            if (kpi === 'throughput') {
                var throughputData = Graph.getDatabyKPI(model, kpi);
                var config = Graph.getGraphConfig(kpi, precisions);
                precisions.forEach((prec, index) => {
                    config.datasets[index].data = throughputData.map(tData => tData[prec]);
                });
                return config;
            }
            var config = Graph.getGraphConfig(kpi);
            config.datasets[0].data = Graph.getDatabyKPI(model, kpi);
            return config;
        });


        // get the kpi title's and create headers for the graphs 
        var chartColumnHeaderContainer = $('<div>');
        chartColumnHeaderContainer.addClass('chart-column-header-container');
        chartColumnHeaderContainer.append($('<div class="chart-column-title"></div>'));
        graphConfigs.forEach((graphConfig) => {
            var columnHeaderContainer = $('<div>');
            columnHeaderContainer.addClass('chart-column-title');
            var columnIcon = $('<div class="icon">');
            columnIcon.addClass(graphConfig.iconClass);
            columnHeaderContainer.append(columnIcon);
            var columnHeader = $('<div class="chart-header">');
            columnHeader.append($('<div class="title">' + graphConfig.chartTitle + '</div>'));
            columnHeader.append($('<div class="title">' + Graph.getGraphPlatformText(ietype) + '</div>'));
            columnHeader.append($('<div class="subtitle">' + graphConfig.chartSubtitle + '</div>'));
            columnHeaderContainer.append(columnHeader);
            chartColumnHeaderContainer.append(columnHeaderContainer);
        });

        // get the client platform labels and create labels for all the graphs

        var labelsContainer = $('<div>');
        labelsContainer.addClass('chart-labels-container');

        labels.forEach((label) => {
            labelsContainer.append($('<div class="title">' + label + '</div>'));
        });

        // get the legend and create legends for each graph

        var graphClass = $('<div>');
        graphClass.addClass('graph-row');
        chartWrap.append(chartColumnHeaderContainer);
        graphClass.append(labelsContainer);
        chartWrap.append(graphClass);

        graphConfigs.forEach((graphConfig) => {
            processMetricNew(labels, graphConfig.datasets, graphConfig.chartTitle, graphClass, 'graph-row-column');
        });

        // might need this line for multiple graphs on a page
        // var displayWidth = $(window).width();

    }

    function processMetricNew(labels, datasets, chartTitle, container, widthClass, displayLabels) {
        // ratio for consistent chart label height
        var heightRatio = ((labels.length * 55 + 20) / labels.length) + (labels.length * 55);
        var chart = $('<div>');
        chart.addClass('chart');
        chart.addClass(widthClass);
        chart.height(heightRatio);
        var canvas = $('<canvas>');
        chart.append(canvas);
        container.append(chart);
        var context = canvas.get(0).getContext('2d');
        context.canvas.height = heightRatio;
        new Chart(context, {
            type: 'horizontalBar',
            data: getChartDataNew(labels, datasets),
            options: getChartOptions(chartTitle, displayLabels)
        });
    }

});
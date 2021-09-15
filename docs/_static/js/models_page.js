(function() {

    function init() {
        var omzUI = $('<div class="omzUI"></div>');
        var search = $('<form><div class="form-group"><input class="form-control" \
         type="text" placeholder="Search" id="model-search"></div></form>');
        var omzDropdowns = $('<div class="omz-selectors"></div>');
        var modelTypeDropdown = createDropdown('modeltype-dropdown', 'Model Type');
        var frameworkDropdown = createDropdown('framework-dropdown', 'Frameworks');
        omzDropdowns.append(modelTypeDropdown);
        omzDropdowns.append(frameworkDropdown);
        omzUI.append(search);
        omzUI.append(omzDropdowns);
        omzUI.insertAfter('h1');
        $('#model-search').on('keyup', function() {
            var value = $(this).val().toLowerCase();
            filterByValue(value);
            updateModelTypeDropdown();
            updateFrameworksDropdown();
        });

        modelTypeDropdown.click();
        frameworkDropdown.click();

        updateModelTypeDropdown();
        updateFrameworksDropdown();
    }



    function createDropdown(id, buttonText) {
        // create dropdowns for `model type` and `frameworks`
        var dropdown = $('<div class="dropdown sst-dropdown"></div>');
        var btn = $('<button class="btn sst-btn" type="button" \
        id="' + id + '" data-toggle="dropdown" \
        aria-haspopup="true" aria-expanded="false">' + buttonText + '</button>');
        var menu = $('<div id="' + id + '-menu" class="dropdown-menu"  \
         aria-labelledby="' + id + '"></div>');
        dropdown.append(btn);
        dropdown.append(menu);
        return dropdown;
    }

    function filterByValue(value) {
        var sections = getSections();
        $(sections).find('table tbody tr').filter(function() {
            if ($(this).text().toLowerCase().indexOf(value) > -1) {
                $(this).removeClass('d-none');
            }
            else {
                $(this).addClass('d-none');
            }
        });
        syncSections();
    }

    function syncSections() {
        var sections = getSections();
        $(sections).each(function() {
            var id = $(this).attr('id');
            var tables = $(this).find('table');
            tables.each(function(){
                if ($(this).find('tbody tr.d-none').length === $(this).find('tbody tr').length) {
                    $(this).addClass('d-none');
                }
                else {
                    $(this).removeClass('d-none');
                }
            });
            if (tables.filter(function() {return !$(this).hasClass('d-none') }).length > 0) {
                $(this).removeClass('d-none');
                $('a.reference[href="#' + id + '"]').closest('li').removeClass('d-none');
            }
            else {
                $(this).addClass('d-none');
                $('a.reference[href="#' + id + '"]').closest('li').addClass('d-none');
            }
        });
    }

    function updateModelTypeDropdown() {
        var menu = $('#modeltype-dropdown-menu');
        var options = {};
        var sections = getActiveSections();
        var id;
        sections.each(function() {
            id = $(this).attr('id');
            options[id] = $(this).find('>:first-child').contents().get(0).nodeValue;
        });
        sections.each(function() {
            id = $(this).attr('id');
            options[id] = $(this).find('>:first-child').contents().get(0).nodeValue;
        });

        var link;
        var links = [];
        for (var item of Object.entries(options)) {
            let key = item[0];
            let value = item[1];
            link = $('<a class="dropdown-item" href="#">' + value + '</a>');
            link.click(function() {
                var sections = getSections();
                filterByValue($('#model-search').val().toLowerCase());
                sections.addClass('d-none');
                $('a.reference').closest('li').addClass('d-none');
                $('section#' + key).removeClass('d-none');
                $('a.reference[href="#' + key + '"]').closest('li').removeClass('d-none');
                updateFrameworksDropdown();
                $('#modeltype-dropdown').text(value);
                $('#framework-dropdown').text('Frameworks');
            });
            links.push(link);
        }
        menu.html(links);
    }

    function updateFrameworksDropdown() {
        var menu = $('#framework-dropdown-menu');
        var sections = getActiveSections();
        var options = new Set();
        sections.each(function() {
            id = $(this).attr('id');
            var index = $(this).find('thead th:contains("Implementation")').index();
            $(this).find('> table tbody tr').each(function() {
               options.add($(this).find('td').eq(index).find('p').contents().get(0).nodeValue); 
            });
        });
        var options = getFrameworksOptions();
        var links = Array.from(options.values()).map(function(item) {
            var link = $('<a class="dropdown-item" href="#">' + item + '</a>');
            link.click(function() {
                // filterByValue($('#model-search').val().toLowerCase());
                var sections = getActiveSections();
                sections.each(function() {
                    id = $(this).attr('id');
                    var index = $(this).find('thead th:contains("Implementation")').index();
                    $(this).find('> table tbody tr').each(function() {
                        $(this).removeClass('d-none');
                    }).filter(function() {
                        return $(this).find('td').eq(index).find('p').contents().get(0).nodeValue !== item;
                    }).each(function() {
                        $(this).addClass('d-none');
                    });
                });
                $('#framework-dropdown').text(item);
                // updateModelTypeDropdown();
                // syncSections();
            });
            return link;
        });
        menu.html(links);
    }

    function getFrameworksOptions() {
        var sections = getActiveSections();
        var options = new Set();
        sections.each(function() {
            id = $(this).attr('id');
            var index = $(this).find('thead th:contains("Implementation")').index();
            $(this).find('> table tbody tr:not(.d-none)').each(function() {
               options.add($(this).find('td').eq(index).find('p').contents().get(0).nodeValue); 
            });
        });
        return options;
    }

    function getActiveSections() {
        return getSections().filter(function() {
            return !$(this).hasClass('d-none');
        });
    }

    function getSections() {
        return $('section').filter(function() {
            return isModelSection($(this)) && $(this).children('h1').length === 0;
        });
    }

    function isModelSection(section) {
        // section is valid if it has table containing model name, implementation
        var requiredColumns = ['model name', 'implementation', 'mparams'];
        var th = [];
        $(section).find('table th p').each(function(){
            th.push($(this).text().toLowerCase());
        });
        if (th.length === 0) {
            return false;
        }
        for (reqc of requiredColumns) {
            if (th.indexOf(reqc) === -1) {
                return false;
            }
         }
         return true;
    }

    $(document).ready(function() {
        init();
    });
})();
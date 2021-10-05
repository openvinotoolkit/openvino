var sw_anchors = {};
var urlSearchParams = new URLSearchParams(window.location.search);
var sw_timer;

$(document).ready(function() {
    init_col_sections();
    init_switchers();
    handleSwitcherParam();
    initViewerJS();
});

function initViewerJS() {
    try {
        var images =$('main img[src*="_images"]');
        images.each(function() {
            try{
                new Viewer($(this).get(0));
            }
            catch (err) {
                console.log(err);
            }
        });
    }
    catch(err) {
        console.log(err);
    }
}

function init_col_sections() {
    var collapsible_sections = $('div.collapsible-section');
    collapsible_sections.each(function() {
        var title = $(this).data('title') || 'Click to expand';
        var summary = $('<summary>' + title + '</summary>');
        // summary.html(title);
        var details = $('<details class="col-sect-details"></details>');
        $(this).wrap(details);
        summary.insertBefore($(this));
    });
}

function handleSwitcherParam() {
    var sw_type = urlSearchParams.get('sw_type');
    var section_id;
    if (sw_type && sw_type in sw_anchors) {
        section_id = sw_anchors[sw_type];
    }
    else {
        section_id = sw_anchors['default'];
    }
    $('.reference.internal.nav-link[href="#' + section_id + '"]').parent('li').css('display', 'block');
    $('#' + section_id).css('display', 'block');
    $('#button-' + section_id).addClass('switcher-active');
}

function init_switchers() {
    var switcherAnchors = $('.switcher-anchor');
    var switcherPanel = $('<div></div>');
    switcherPanel.addClass('switcher-set');
    switcherPanel.css('top', $('#navbar-main').height());
    switcherPanel.css('left', $('main').offset().left);
    switcherPanel.width($('main').width() +
                        parseFloat($('main').css('padding-left')) +
                        parseFloat($('main').css('padding-right')) +
                        parseFloat($('.bd-toc').css('padding-left')));
    for (var i = 0; i < switcherAnchors.length; i++) {
        var anchor = $(switcherAnchors[i]);
        var option = $(anchor).text();
        var id = $(anchor).attr('id');
        var link = $('<a></a>');
        link.text(option);
        link.attr('href', '?sw_type=' + id);
        link.addClass('switcher-button');
        switcherPanel.append(link);
        var section = $(anchor).parent('div.section');
        section.css('display', 'none');
        var section_id = section.attr('id');
        link.attr('id', 'button-' + section_id);
        $('.reference.internal.nav-link[href="#' + section_id + '"]').parent('li').css('display', 'none');
        section.addClass('switcher-content');
        sw_anchors[id] = section_id;
        if (i === 0) {
            sw_anchors['default'] = section_id;
        }
    }

    $('main').prepend(switcherPanel);
    switcherAnchors.remove();
}

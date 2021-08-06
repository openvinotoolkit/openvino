$(document).ready(function() {
    var collapsible_sections = $('div.collapsible-section');
    $(collapsible_sections).each(function(item){
        item.prepend('<summary>Click to expand</summary>');
        item.wrap('<details></details>');
    });
});
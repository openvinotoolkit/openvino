document.addEventListener('DOMContentLoaded', function () {
   sortableTables();
});

function sortableTables() {

   var tablestosort = $('div.sort-table').find('table.table').addClass('table-sortable');
   var tablestosortnext = $('div.sort-table').next('table.table').addClass('table-sortable');
   var sortables = $('table.table-sortable').append(tablestosortnext);
   var tables = sortables.append(tablestosort);
   tables.each(function () {
       var table = $(this);
       var headings = table.find('th');
       headings.each(function () {
           var th = $(this);
           var index = th.index();
           var sortBtn = $('<span class="sort-btn"></span>');
           th.addClass('sort-header');
           th.click(function () {
               var counter = 0;
               sortBtn.addClass('sort-active');
               sortBy = sortBtn.data('sortby');
               var trs = table.find('tbody tr');
               sortBtn.toggleClass('ascending');
               trs.sort(function (item1, item2) {

                   if (sortBtn.hasClass('ascending')) {
                       var text1 = $(item1).find('td').eq(index).text();
                       var text2 = $(item2).find('td').eq(index).text();
                   }
                   else {
                       var text1 = $(item2).find('td').eq(index).text();
                       var text2 = $(item1).find('td').eq(index).text();
                   }
                   // try converting to num
                   var _text1 = parseFloat(text1);
                   var _text2 = parseFloat(text2);

                   if (!isNaN(_text1) && !isNaN(_text2)) {
                       text1 = _text1;
                       text2 = _text2;
                   }
                   if (text1 > text2) {
                       return 1;
                   }
                   else if (text1 < text2) {
                       return -1;
                   }
                   else {
                       return 0;
                   }
               }).map(function () {
                   var row = $(this);
                   if (counter % 2 === 0) {
                       row.removeClass('row-odd');
                       row.addClass('row-even');
                   }
                   else {
                       row.removeClass('row-even');
                       row.addClass('row-odd');
                   }
                   counter++;
                   table.find('tbody').append(row);
               });

               headings.each(function () {
                   if ($(this).index() !== index) {
                       $(this).find('.sort-btn').removeClass('ascending');
                       $(this).find('.sort-btn').removeClass('sort-active');
                   }
               });
           });
           th.find('p').append(sortBtn);
       });
   });
}
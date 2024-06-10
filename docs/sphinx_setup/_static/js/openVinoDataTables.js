$(document).ready(function () {
  var table = $('table.modeldata').DataTable({
    "autoWidth": false,
    // "aoColumnDefs": [{
    //   "aTargets": [1],
    //   type: 'sort'
    //   // render: $.fn.dataTable.render.number(',', '.', 3, '')
    //   // "mRender": function (data, type, full) {
    //   //   var TextInsideLi = data.getElementsByTagName('p')[0].innerHTML;
    //   //   return parseFloat(TextInsideLi);
    //   // }
    // }],
    // columns: [
    //   {
    //     "render": function (data, type, row, meta) {

    //       if (type === 'sort' || type === 'type') {
    //         return parseFloat(data);
    //       }

    //       return data;

    //     }
    //   }
    // ]
  });

  document.querySelectorAll('input.toggle-vis').forEach((el) => {
    if (el.checked) {
      table.columns([el.getAttribute('data-column')]).visible(false, true);
    }

    el.addEventListener('click', function (e) {
      let columnIdx = e.target.getAttribute('data-column');
      let column = table.column(columnIdx);
      column.visible(!column.visible());
    });
  });
});
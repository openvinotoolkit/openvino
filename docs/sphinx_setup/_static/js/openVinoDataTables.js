$(document).ready(function () {
  var table = $('table.modeldata').DataTable({
    "autoWidth": false,
    stateSave: true,
    lengthMenu: [
      [10, 25, 50, -1],
      ['10 rows', '25 rows', '50 rows', 'Show all rows']
    ],
    layout: {
      topStart: {
        buttons: [
          'pageLength',
          'colvis',
          {
            extend: 'colvisGroup',
            text: 'Show all columns',
            show: ':hidden'
          },
          {
            extend: 'print',
            text: 'Print pdf',
            exportOptions: {
              columns: ':visible'
            }
          }
        ]
      }
    }
  });
});
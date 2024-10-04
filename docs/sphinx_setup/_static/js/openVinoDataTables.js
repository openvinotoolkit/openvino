$(document).ready(function () {
  var table = $('table.modeldata').DataTable({
    responsive: true,
    "autoWidth": false,
    stateSave: true,
    language: {
      buttons: {
        colvisRestore: "Show all columns"
      }
    },
    lengthMenu: [
      [10, 25, 50, -1],
      ['10 rows', '25 rows', '50 rows', 'Show all rows']
    ],
    layout: {
      topStart: {
        buttons: [
          'pageLength',
          {
            extend: 'colvis',
            postfixButtons: ['colvisRestore'],
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
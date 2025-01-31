$(document).ready(function () {
  var columnDefs = [];
  var tables = $('table.modeldata');
  for (let table of tables) {
    var hidden = table.getAttribute('data-column-hidden');
    columnDefs = [{ "visible": false, "targets": JSON.parse(hidden) }]
    $(table).DataTable({
      responsive: true,
      "autoWidth": false,
      language: {
        buttons: {
          colvisRestore: "Restore default selection"
        }
      },
      lengthMenu: [
        [10, 25, 50, -1],
        ['10 rows', '25 rows', '50 rows', 'Show all records']
      ],
      "columnDefs": columnDefs,
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
  }
});
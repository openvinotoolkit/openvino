$(document).ready(function () {
  var table = $('table.modeldata').DataTable({
    "autoWidth": false,
    stateSave: true,
    layout: {
      topStart: {
        buttons: [
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
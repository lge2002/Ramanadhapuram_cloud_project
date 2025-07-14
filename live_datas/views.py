# live_datas/views.py
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from weather.models import Cloud_ramanathapuram
from django.db.models import Max
import pytz
from django.utils import timezone # Import timezone for working with datetimes

def data(request):
    """
    Django view to retrieve and display the most recent cloud analysis data.

    This view is specifically designed to show data only from the single,
    absolute latest "update loop" or "batch" that exists in the database.

    It works as follows:
    1. Identifies if the request is an AJAX call (for JSON response) or a
       standard browser request (for HTML rendering).
    2. Queries the database to find the highest (latest) timestamp across
       *all* records in the Cloud_ramanathapuram model. This timestamp
       represents the time of the most recent data update.
    3. If a latest timestamp is found, it then filters the Cloud_ramanathapuram
       records to include only those entries that exactly match this
       absolute latest timestamp. This ensures that only data belonging
       to the most recent "loop" of updates is retrieved.
    4. Orders the retrieved data by city name for consistent display.
    5. Based on the request type (AJAX or standard), it either:
       - Returns the filtered data as a JSON object (for dynamic updates).
       - Renders an HTML template ('live_datas/live.html') with the data
         passed in the context (for initial page load).
    """
    # Determine if the request is an AJAX call (e.g., from JavaScript's XMLHttpRequest)
    # or if the client specifically requested JSON format via a GET parameter.
    is_ajax = request.headers.get('x-requested-with') == 'XMLHttpRequest' or request.GET.get('format') == 'json'

    # Step 1: Find the single, absolute latest timestamp across ALL records in the table.
    # This is done by aggregating the maximum 'timestamp' value from the entire model.
    # The result will be a dictionary, e.g., {'timestamp__max': datetime.datetime(...)}.
    absolute_latest_timestamp_obj = Cloud_ramanathapuram.objects.aggregate(Max('timestamp'))
    absolute_latest_timestamp = absolute_latest_timestamp_obj['timestamp__max']

    # Initialize an empty list to hold the cloud data. This list will be populated
    # only if an absolute latest timestamp is found, ensuring graceful handling
    # when the database table is empty.
    latest_cloud_data = []

    # Step 2: If an absolute latest timestamp was found (meaning the database is not empty),
    # proceed to filter the records.
    if absolute_latest_timestamp:
        # Filter the Cloud_ramanathapuram records to include only those whose 'timestamp'
        # exactly matches the 'absolute_latest_timestamp' found in the previous step.
        # This effectively selects all records that were part of the most recent data update.
        # The results are then ordered by 'city' for consistent presentation.
        latest_cloud_data = Cloud_ramanathapuram.objects.filter(
            timestamp=absolute_latest_timestamp
        ).order_by('city')

    # Step 3: Handle the request based on whether it's an AJAX call or a standard browser request.
    if is_ajax:
        # For AJAX requests, prepare the data as a list of dictionaries for JSON serialization.
        data_for_json = []
        for record in latest_cloud_data:
            # Convert the datetime object to a string for JSON compatibility.
            # Using strftime provides a consistent format. If timezone conversion
            # to a specific local timezone is needed for the client,
            # timezone.localtime(record.timestamp) could be used before strftime.
            timestamp_str = record.timestamp.strftime('%Y-%m-%d %H:%M:%S')
            data_for_json.append({
                'city': record.city,
                'values': record.values,
                'type': record.type,
                'timestamp': timestamp_str,
            })
        # Return the data as a JSON response.
        return JsonResponse({'cloud_analysis_data': data_for_json})
    else:
        # For standard browser requests, prepare the context dictionary to pass data
        # to the HTML template.
        context = {
            'cloud_analysis_data': latest_cloud_data # The filtered latest data
        }
        # Render the 'live_datas/live.html' template, passing the context data to it.
        return render(request, 'live_datas/live.html', context)

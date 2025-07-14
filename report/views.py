from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template.loader import render_to_string
import os
from django.conf import settings
from datetime import datetime, timedelta, date, time
import pytz
from playwright.sync_api import sync_playwright
from weather.models import Cloud_ramanathapuram
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from sklearn.cluster import KMeans
from shapely.ops import unary_union
import warnings
import io
import base64
import shutil
warnings.filterwarnings("ignore")

SHAPEFILE_PATH = os.path.join(settings.BASE_DIR, 'weather', 'management', 'commands', 'gadm41_IND_3.json')
FINAL_MIN_LON = 74.80
FINAL_MAX_LON = 80.37
FINAL_MIN_LAT = 7.98
FINAL_MAX_LAT = 13.53

# --- NEW: Define the base directory for Ramanathapuram automation images ---
BASE_AUTOMATION_IMAGES_DIR = os.path.join(settings.BASE_DIR, "images", "ramanathapuram")

def save_image_and_get_filepath(image_pil, report_base_dir, timestamp_dt, image_type_name):
    image_timestamp_folder = timestamp_dt.strftime('%H-%M-%S')
    save_dir = os.path.join(report_base_dir, image_timestamp_folder)
    os.makedirs(save_dir, exist_ok=True)
    file_name = f"{image_type_name}.png"
    full_path = os.path.join(save_dir, file_name)

    try:
        image_pil.save(full_path, format="PNG")
        return full_path.replace(os.sep, '/')
    except Exception as e:
        print(f"Error saving image {full_path}: {e}")
        return None

def _generate_image_data_for_timestamp(
    base_image_path_for_this_timestamp, timestamp_dt, selected_district, gdf_tn, is_pdf_generation=False):
    try:
        if not os.path.exists(base_image_path_for_this_timestamp):
            print(f"Warning: Base image '{base_image_path_for_this_timestamp}' not found. Skipping image processing for this timestamp.")
            return None
        img_pil = Image.open(base_image_path_for_this_timestamp).convert("RGB")
        img_np = np.array(img_pil) # Convert PIL image to NumPy array for numerical operations
        height, width, _ = img_np.shape # Get dimensions of the image (height, width, channels)
        transform = from_bounds(FINAL_MIN_LON, FINAL_MIN_LAT, FINAL_MAX_LON, FINAL_MAX_LAT, width, height)
        output_images = {
            'cropped_tn': img_pil, # The original radar image of Tamil Nadu
            'masked_district': None, # Will store the masked district image
            'aligned_overlay_tn': None, # Will store the radar image with district outlines
        }

        if selected_district == 'All Districts' or gdf_tn.empty:
            output_images['masked_district'] = img_pil.copy() # Use a copy to avoid unintended modifications
        else:
            district_rows_for_name = gdf_tn[gdf_tn['NAME_2'].str.lower() == selected_district.lower()]
            if not district_rows_for_name.empty:
                all_district_geometries = district_rows_for_name.geometry.to_list()
                district_polygon_for_mask = unary_union(all_district_geometries)
                mask = rasterize(
                    [district_polygon_for_mask], # List of geometries to rasterize
                    out_shape=(height, width), # The desired shape (dimensions) of the output mask (same as image)
                    transform=transform, # The geographic transform for mapping coordinates to pixels
                    fill=0, # Value for pixels outside the given geometries
                    all_touched=True, # Include all pixels that touch the polygon boundary, not just centroids
                    dtype=np.uint8 # Data type for the mask array (unsigned 8-bit integer)
                    )

                if not is_pdf_generation: # Only print for browser view, not for every PDF image
                    print(f"Mask for '{selected_district}': unique values={np.unique(mask)}, sum={mask.sum()}")
                mask_boolean = mask.astype(bool) # Convert the 0/1 mask to a boolean mask
                original_rgba_img = Image.fromarray(img_np).convert("RGBA")
                original_rgba_np = np.array(original_rgba_img)
                transparent_image_np = np.zeros_like(original_rgba_np)
                transparent_image_np[mask_boolean] = original_rgba_np[mask_boolean]
                output_images['masked_district'] = Image.fromarray(transparent_image_np)
            else:
                print(f"Warning: District '{selected_district}' not found in shapefile for masked image generation at {timestamp_dt}.")
                output_images['masked_district'] = None # Set to None if district geometry isn't found
        fig, ax = plt.subplots(figsize=(10, 10)) # Adjust figsize as needed for image quality
        ax.imshow(img_np, extent=[FINAL_MIN_LON, FINAL_MAX_LON, FINAL_MIN_LAT, FINAL_MAX_LAT])
        gdf_tn.boundary.plot(ax=ax, edgecolor='black', linewidth=0.5)
        if selected_district != 'All Districts':
            district_rows_for_name_for_highlight = gdf_tn[gdf_tn['NAME_2'].str.lower() == selected_district.lower()]
            if not district_rows_for_name_for_highlight.empty:
                district_rows_for_name_for_highlight.boundary.plot(ax=ax, edgecolor='cyan', linewidth=2, linestyle='--', label=selected_district)
                ax.set_title(f"Aligned Screenshot with {selected_district} Highlighted ({timestamp_dt.strftime('%H:%M')})")
                ax.legend() # Display legend for highlighted district
            else:
                ax.set_title(f"Aligned Screenshot (District '{selected_district}' not found for highlight) ({timestamp_dt.strftime('%H:%M')})")
        else:
            ax.set_title(f"Aligned Screenshot with All TN District Outlines ({timestamp_dt.strftime('%H:%M')})")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect('equal')
        plt.tight_layout() # Adjust layout to prevent labels/titles from overlapping

        buffer = io.BytesIO()
        plt.savefig(buffer, format="PNG", bbox_inches='tight', pad_inches=0.1)
        buffer.seek(0) # Rewind the buffer to the beginning
        # Open the saved image from the buffer with PIL and convert to RGB
        output_images['aligned_overlay_tn'] = Image.open(buffer).convert("RGB")
        buffer.close() # Close the buffer
        plt.close(fig) # IMPORTANT: Close the Matplotlib figure to free up memory
        return output_images # Return the dictionary of generated PIL images

    except FileNotFoundError:
        # Handle cases where the base radar image file is missing
        print(f"Warning: Base map image '{base_image_path_for_this_timestamp}' not found for {timestamp_dt}. Skipping generation for this timestamp.")
        return None
    except Exception as e:
        # Catch any other unexpected errors during image generation
        print(f"An unexpected error occurred during image generation for {timestamp_dt}: {e}")
        return None
    finally:
        if 'fig' in locals() and fig: # Check if 'fig' variable was created
            plt.close(fig)

def report_view(request):
    selected_date_str = request.GET.get('date')
    selected_district = request.GET.get('district', 'All Districts')
    if selected_district == "" or selected_district is None:
        selected_district = "All Districts"

    start_time_hour_str = request.GET.get('start_time_hour')
    start_time_minute_str = request.GET.get('start_time_minute')
    end_time_hour_str = request.GET.get('end_time_hour')
    end_time_minute_str = request.GET.get('end_time_minute')
    selected_image_view = request.GET.get('image_view_type', '')

    filter_date = None
    if selected_date_str:
        try:
            filter_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
        except ValueError:
            print(f"Warning: Invalid date format '{selected_date_str}'. Defaulting to today's date.")
            filter_date = date.today()
    else:
        filter_date = date.today()
    
    target_date_display_str = filter_date.strftime('%Y-%m-%d') # Formatted date string for display
    selected_start_time_for_template = ''
    selected_end_time_for_template = ''
    filter_start_datetime = None
    filter_end_datetime = None
    current_timezone = pytz.timezone(settings.TIME_ZONE) if settings.USE_TZ else None

    if not start_time_hour_str or not start_time_minute_str:
        now = datetime.now()
        current_minute = now.minute
        current_hour = now.hour

        from_minute = (current_minute // 15) * 15
        if current_minute % 15 != 0:
            from_minute = ((current_minute // 15) + 1) * 15
            if from_minute == 60:
                from_minute = 0
                current_hour = (current_hour + 1) % 24
        from_hour = current_hour     
        start_time_obj = time(from_hour, from_minute)
        selected_start_time_for_template = start_time_obj.strftime("%H:%M")
        filter_start_datetime = datetime.combine(filter_date, start_time_obj)
        to_minute = from_minute + 15
        to_hour = from_hour
        if to_minute == 60:
            to_minute = 0
            to_hour = (to_hour + 1) % 24
        
        end_time_obj = time(to_hour, to_minute)
        selected_end_time_for_template = end_time_obj.strftime("%H:%M")
        filter_end_datetime = datetime.combine(filter_date, end_time_obj)

    else:
        try:
            start_hour = int(start_time_hour_str)
            start_minute = int(start_time_minute_str)
            end_hour = int(end_time_hour_str)
            end_minute = int(end_time_minute_str)

            start_time_obj = time(start_hour, start_minute)
            end_time_obj = time(end_hour, end_minute)

            selected_start_time_for_template = start_time_obj.strftime("%H:%M")
            selected_end_time_for_template = end_time_obj.strftime("%H:%M")

            filter_start_datetime = datetime.combine(filter_date, start_time_obj)
            filter_end_datetime = datetime.combine(filter_date, end_time_obj)

        except ValueError as e:
            print(f"Error parsing time parameters: {e}. Defaulting to full day.")
            filter_start_datetime = datetime.combine(filter_date, time(0, 0, 0))
            filter_end_datetime = datetime.combine(filter_date, time(23, 59, 59, 999999))
            selected_start_time_for_template = "00:00"
            selected_end_time_for_template = "23:59"

    if current_timezone:
        if filter_start_datetime:
            filter_start_datetime = current_timezone.localize(filter_start_datetime)
        if filter_end_datetime:
            filter_end_datetime = current_timezone.localize(filter_end_datetime)
    print(f"\n--- Report Parameters: Date={target_date_display_str}, District={selected_district}, View={selected_image_view} ---")

    generated_images_for_display = [] # List to hold image data (Base64 strings) for the HTML template
    # --- NEW: List to hold Ramanathapuram automation image data ---
    ramanathapuram_automation_images = []
    # DEBUG: Print the selected parameters
    print(f"[DEBUG] Selected date: {filter_date}, district: {selected_district}, start: {selected_start_time_for_template}, end: {selected_end_time_for_template}")

    gdf = None # GeoDataFrame for the entire shapefile
    gdf_tn = None # GeoDataFrame specifically for Tamil Nadu districts
    try:
        if not os.path.exists(SHAPEFILE_PATH):
            raise FileNotFoundError(f"Shapefile not found at {SHAPEFILE_PATH}.")
        gdf = gpd.read_file(SHAPEFILE_PATH)
        gdf_tn = gdf[
            (gdf['NAME_1'].str.strip().str.lower() == 'tamilnadu') |
            (gdf['NAME_1'].str.strip().str.lower() == 'tamil nadu')
        ].to_crs("EPSG:4326")
    except FileNotFoundError as fnfe:
        print(f"CRITICAL ERROR: Shapefile (for generation) not found: {fnfe}")
    except Exception as e:
        print(f"ERROR loading shapefile: {e}")

    if os.path.exists(settings.MEDIA_ROOT) and os.path.isdir(settings.MEDIA_ROOT) and gdf_tn is not None:
        available_image_timestamps_and_folders = []
        for d in os.listdir(settings.MEDIA_ROOT):
            full_path_to_folder = os.path.join(settings.MEDIA_ROOT, d)
            if os.path.isdir(full_path_to_folder):
                try:
                    folder_datetime_obj = datetime.strptime(d, '%Y-%m-%d_%H-%M-%S')

                    if current_timezone:
                        folder_datetime_obj = current_timezone.localize(folder_datetime_obj)
                    if folder_datetime_obj.date() == filter_date and \
                        filter_start_datetime <= folder_datetime_obj < filter_end_datetime:
                        available_image_timestamps_and_folders.append((folder_datetime_obj, full_path_to_folder))
                except (ValueError, IndexError):
                    try:
                        folder_date_part = datetime.strptime(d, '%Y-%m-%d').date()
                        if folder_date_part == filter_date:
                            potential_image_file = os.path.join(full_path_to_folder, 'cropped', 'tamil_nadu_cropped.png')
                            if os.path.exists(potential_image_file):
                                dummy_timestamp_for_day = datetime.combine(filter_date, time(12,0,0))
                                if current_timezone:
                                    dummy_timestamp_for_day = current_timezone.localize(dummy_timestamp_for_day)
                                available_image_timestamps_and_folders.append((dummy_timestamp_for_day, full_path_to_folder))
                    except (ValueError, IndexError):
                        pass # Ignore folders that don't match expected timestamp or date formats

        available_image_timestamps_and_folders.sort(key=lambda x: x[0]) # Sort images by timestamp
        print(f"[DEBUG] Found {len(available_image_timestamps_and_folders)} folders with images in MEDIA_ROOT for this date/time range.")
        for timestamp_dt, folder_path in available_image_timestamps_and_folders:
            base_image_path_for_this_timestamp = os.path.join(folder_path, 'cropped', 'tamil_nadu_cropped.png')
            pil_images = _generate_image_data_for_timestamp(
                base_image_path_for_this_timestamp, timestamp_dt, selected_district, gdf_tn, is_pdf_generation=False
            )

            if pil_images: # If image generation was successful
                current_image_set = {
                    'timestamp': timestamp_dt,
                    'cropped_tn': None,
                    'masked_district': None,
                    'aligned_overlay_tn': None,
                }

                for img_type, pil_img in pil_images.items():
                    if pil_img:
                        buffer = io.BytesIO() # Create an in-memory binary stream
                        pil_img.save(buffer, format="PNG") # Save PIL image to the buffer
                        # Encode the buffer content to Base64 and prepend data URI scheme
                        current_image_set[img_type] = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode('utf-8')
                        buffer.close()
                    # Special handling for "All Districts" in masked view
                    elif img_type == 'masked_district' and selected_district == 'All Districts':
                        current_image_set['masked_district'] = current_image_set['cropped_tn']
                    else:
                        current_image_set[img_type] = None # Set to None if image could not be generated

                generated_images_for_display.append(current_image_set) # Add to list for template
        print(f"[DEBUG] Total generated_images_for_display: {len(generated_images_for_display)}")
    else:
        print(f"Image generation skipped for browser display: Base media directory or shapefile not loaded successfully.")

    # --- NEW: Logic to fetch Ramanathapuram automation images ---
    # Show automation images for both 'ramanathapuram' and 'All Ramanathapuram Taluks'
    if selected_district.lower() == 'ramanathapuram' or selected_district == 'All Ramanathapuram Taluks':
        if os.path.exists(BASE_AUTOMATION_IMAGES_DIR) and os.path.isdir(BASE_AUTOMATION_IMAGES_DIR):
            automation_image_folders = []
            for d in os.listdir(BASE_AUTOMATION_IMAGES_DIR):
                full_path_to_automation_folder = os.path.join(BASE_AUTOMATION_IMAGES_DIR, d)
                if os.path.isdir(full_path_to_automation_folder):
                    try:
                        folder_datetime_obj = datetime.strptime(d, '%Y-%m-%d_%H-%M-%S')
                        if current_timezone:
                            folder_datetime_obj = current_timezone.localize(folder_datetime_obj)

                        if folder_datetime_obj.date() == filter_date and \
                            filter_start_datetime <= folder_datetime_obj < filter_end_datetime:
                            automation_image_folders.append((folder_datetime_obj, full_path_to_automation_folder))
                    except (ValueError, IndexError):
                        pass # Ignore folders that don't match expected timestamp format

            automation_image_folders.sort(key=lambda x: x[0])
            print(f"[DEBUG] Found {len(automation_image_folders)} automation folders for Ramanathapuram in date/time range.")

            for timestamp_dt, folder_path in automation_image_folders:
                current_auto_image_set = {
                    'timestamp': timestamp_dt,
                    'full_screenshot': None,
                    'cropped_screenshot': None,
                    'masked_taluk_images': {} # Dictionary to hold taluk-wise masked images
                }

                # Path to full screenshot (try both direct and subfolder search)
                full_screenshot_path = os.path.join(folder_path, 'windy_full.png')
                found_full_screenshot = False
                if os.path.exists(full_screenshot_path):
                    with open(full_screenshot_path, "rb") as image_file:
                        current_auto_image_set['full_screenshot'] = "data:image/png;base64," + base64.b64encode(image_file.read()).decode('utf-8')
                    found_full_screenshot = True
                else:
                    # Try to find any .png file in the folder or subfolders that could be a full screenshot
                    for root, dirs, files in os.walk(folder_path):
                        for file in files:
                            if file.lower().startswith('windy_full') and file.lower().endswith('.png'):
                                alt_full_path = os.path.join(root, file)
                                with open(alt_full_path, "rb") as image_file:
                                    current_auto_image_set['full_screenshot'] = "data:image/png;base64," + base64.b64encode(image_file.read()).decode('utf-8')
                                found_full_screenshot = True
                                break
                        if found_full_screenshot:
                            break
                if not found_full_screenshot:
                    print(f"[DEBUG] No full screenshot found for {timestamp_dt} in {folder_path}")
                
                # Path to cropped screenshot
                cropped_screenshot_path = os.path.join(folder_path, 'windy_cropped.png')
                if os.path.exists(cropped_screenshot_path):
                    with open(cropped_screenshot_path, "rb") as image_file:
                        current_auto_image_set['cropped_screenshot'] = "data:image/png;base64," + base64.b64encode(image_file.read()).decode('utf-8')

                # Paths to masked taluk images (recursive search for .png files in all subfolders)
                masked_taluks_dir = os.path.join(folder_path, 'masked_taluks')
                taluk_img_count = 0
                if os.path.exists(masked_taluks_dir) and os.path.isdir(masked_taluks_dir):
                    for root, dirs, files in os.walk(masked_taluks_dir):
                        for taluk_img_file in files:
                            if taluk_img_file.endswith('.png'):
                                taluk_image_path = os.path.join(root, taluk_img_file)
                                # Taluk name: use the immediate subfolder name if present, else fallback to filename
                                rel_path = os.path.relpath(taluk_image_path, masked_taluks_dir)
                                parts = rel_path.split(os.sep)
                                if len(parts) > 1:
                                    taluk_name_from_folder = parts[0].replace('_', ' ').title()
                                else:
                                    taluk_name_from_folder = taluk_img_file.replace('_masked.png', '').replace('_', ' ').title()
                                with open(taluk_image_path, "rb") as image_file:
                                    current_auto_image_set['masked_taluk_images'][taluk_name_from_folder] = "data:image/png;base64," + base64.b64encode(image_file.read()).decode('utf-8')
                                taluk_img_count += 1
                    print(f"[DEBUG] {timestamp_dt}: Found {taluk_img_count} masked taluk images (recursive) in {masked_taluks_dir}")

                ramanathapuram_automation_images.append(current_auto_image_set)
            print(f"[DEBUG] Total ramanathapuram_automation_images: {len(ramanathapuram_automation_images)}")
        else:
            print(f"Ramanathapuram automation images directory not found: {BASE_AUTOMATION_IMAGES_DIR}")

    # --- Filtering Cloud_ramanathapuram data for display ---
    cloud_analysis_query = Cloud_ramanathapuram.objects.filter(
        timestamp__date=filter_date, # Filter by selected date
        **{'type': 'Cloud Coverage'} # <--- ADDED THIS LINE!
    )

    if selected_district != 'All Ramanathapuram Taluks':
        cloud_analysis_query = cloud_analysis_query.filter(city__iexact=selected_district) # Filter by district

    if filter_start_datetime and filter_end_datetime:
        cloud_analysis_query = cloud_analysis_query.filter(
            timestamp__gte=filter_start_datetime,
            timestamp__lt=filter_end_datetime # Filter by time range
        )

    EXCLUDE_NO_PRECIP_MESSAGE = "No significant cloud levels found for precipitation"
    cloud_analysis_query = cloud_analysis_query.exclude(values__iexact=EXCLUDE_NO_PRECIP_MESSAGE) # Exclude no precipitation data

    filtered_cloud_analysis_data = list(cloud_analysis_query.order_by('city', 'timestamp'))
    print(f"Fetched {len(filtered_cloud_analysis_data)} weather data points for {target_date_display_str} and {selected_district}.")

    # --- Dynamically load available districts for the dropdown filter ---
    full_available_districts = []
    try:
        if not os.path.exists(SHAPEFILE_PATH):
            print(f"ERROR: Shapefile for district list not found at {SHAPEFILE_PATH}. Falling back to default list.")
            full_available_districts = ['Coimbatore', 'Chennai', 'Madurai', 'Trichy', 'Salem', 'Ariyalur']
        else:
            temp_gdf = gpd.read_file(SHAPEFILE_PATH)
            gdf_tn_districts_for_list = temp_gdf[
                (temp_gdf['NAME_1'].str.strip().str.lower() == 'tamilnadu')
            ]

            if 'NAME_2' in gdf_tn_districts_for_list.columns:
                unique_districts = gdf_tn_districts_for_list['NAME_2'].dropna().unique().tolist()
                full_available_districts = sorted(unique_districts)
            else:
                print("Warning: 'NAME_2' column not found in shapefile for district extraction. Falling back to default list.")
                full_available_districts = ['Coimbatore', 'Chennai', 'Madurai', 'Trichy', 'Salem', 'Ariyalur']

    except Exception as e:
        print(f"Error loading districts from shapefile for dropdown: {e}. Falling back to default list.")
        full_available_districts = ['Coimbatore', 'Chennai', 'Madurai', 'Trichy', 'Salem', 'Ariyalur']
    
    ramanathapuram_taluks = []
    try:
        if not os.path.exists(SHAPEFILE_PATH):
            print(f"ERROR: Shapefile not found at {SHAPEFILE_PATH}.")
            print("Cannot extract taluks without the shapefile.")
            # You might want to provide a default list here if the shapefile is crucial
            # ramanathapuram_taluks = ['Default Taluk 1', 'Default Taluk 2'] 
        else:
            temp_gdf = gpd.read_file(SHAPEFILE_PATH)

            # Step 1: Filter for 'Tamil Nadu' (assuming your shapefile contains data for multiple states)
            # Adjust 'NAME_1' if your state column has a different name
            gdf_tamilnadu = temp_gdf[
                (temp_gdf['NAME_1'].str.strip().str.lower() == 'tamilnadu')
            ]

            # Step 2: Filter for 'Ramanathapuram' district
            # Adjust 'NAME_2' if your district column has a different name
            if 'NAME_2' in gdf_tamilnadu.columns:
                gdf_ramanathapuram = gdf_tamilnadu[
                    (gdf_tamilnadu['NAME_2'].str.strip().str.lower() == 'ramanathapuram')
                ]

                # Step 3: Extract unique taluk names
                # Adjust 'NAME_3' if your taluk column has a different name
                if 'NAME_3' in gdf_ramanathapuram.columns:
                    unique_taluks = gdf_ramanathapuram['NAME_3'].dropna().unique().tolist()
                    ramanathapuram_taluks = sorted(unique_taluks)
                else:
                    print("Warning: 'NAME_3' column (for taluks) not found in shapefile.")
                    print("Cannot extract taluks for Ramanathapuram.")
            else:
                print("Warning: 'NAME_2' column (for districts) not found in shapefile.")
                print("Cannot filter for Ramanathapuram district.")

    except Exception as e:
        print(f"Error loading or processing shapefile: {e}.")
        print("Cannot extract taluks due to the error.")

    # Prepare context dictionary to pass data to the HTML template
    context = {
        'generated_images_for_display': generated_images_for_display,
        'ramanathapuram_automation_images': ramanathapuram_automation_images, # NEW: Add this to context
        'selected_date': filter_date.strftime('%Y-%m-%d'),
        'selected_district': selected_district,
        'available_districts': ramanathapuram_taluks,  # Only show taluks of Ramanathapuram
        'ramanathapuram_taluks': ramanathapuram_taluks,
        'cloud_analysis_data': filtered_cloud_analysis_data,
        'selected_image_view': selected_image_view,
        'selected_start_time': selected_start_time_for_template,
        'selected_end_time': selected_end_time_for_template,
    }
    return render(request, 'report/report.html', context) # Render the report HTML template

# --- View for generating and downloading the PDF report ---
def download_report_pdf(request):
    selected_date_str = request.GET.get('date')
    selected_district = request.GET.get('district', 'All Districts')
    if selected_district == "" or selected_district is None:
        selected_district = "All Districts"
    start_time_hour_str = request.GET.get('start_time_hour')
    start_time_minute_str = request.GET.get('start_time_minute')
    end_time_hour_str = request.GET.get('end_time_hour')
    end_time_minute_str = request.GET.get('end_time_minute')
    selected_image_view = request.GET.get('image_view_type', '')
    filter_date = None
    if selected_date_str:
        try:
            filter_date = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
        except ValueError:
            print(f"PDF Gen: Warning: Invalid date format '{selected_date_str}'. Defaulting to today's date.")
            filter_date = date.today()
    else:
        filter_date = date.today()
    target_date_display_str = filter_date.strftime('%Y-%m-%d')
    selected_start_time_for_template = ''
    selected_end_time_for_template = ''
    filter_start_datetime = None
    filter_end_datetime = None
    current_timezone = pytz.timezone(settings.TIME_ZONE) if settings.USE_TZ else None
    if not start_time_hour_str or not start_time_minute_str:
        now = datetime.now()
        current_minute = now.minute
        current_hour = now.hour
        from_minute = (current_minute // 15) * 15
        if current_minute % 15 != 0:
            from_minute = ((current_minute // 15) + 1) * 15
            if from_minute == 60:
                from_minute = 0
                current_hour = (current_hour + 1) % 24

        from_hour = current_hour
        start_time_obj = time(from_hour, from_minute)
        selected_start_time_for_template = start_time_obj.strftime("%H:%M")
        filter_start_datetime = datetime.combine(filter_date, start_time_obj)
        to_minute = from_minute + 15
        to_hour = from_hour
        if to_minute == 60:
            to_minute = 0
            to_hour = (to_hour + 1) % 24
        end_time_obj = time(to_hour, to_minute)
        selected_end_time_for_template = end_time_obj.strftime("%H:%M")
        filter_end_datetime = datetime.combine(filter_date, end_time_obj)
    else:
        try:
            start_hour = int(start_time_hour_str)
            start_minute = int(start_time_minute_str)
            end_hour = int(end_time_hour_str)
            end_minute = int(end_time_minute_str)
            start_time_obj = time(start_hour, start_minute)
            end_time_obj = time(end_hour, end_minute)
            selected_start_time_for_template = start_time_obj.strftime("%H:%M")
            selected_end_time_for_template = end_time_obj.strftime("%H:%M")
            filter_start_datetime = datetime.combine(filter_date, start_time_obj)
            filter_end_datetime = datetime.combine(filter_date, end_time_obj)

        except ValueError as e:
            print(f"PDF Gen: Error parsing time parameters: {e}. Defaulting to full day.")
            filter_start_datetime = datetime.combine(filter_date, time(0, 0, 0))
            filter_end_datetime = datetime.combine(filter_date, time(23, 59, 59, 999999))
            selected_start_time_for_template = "00:00"
            selected_end_time_for_template = "23:59"

    if current_timezone:
        if filter_start_datetime:
            filter_start_datetime = current_timezone.localize(filter_start_datetime)
        if filter_end_datetime:
            filter_end_datetime = current_timezone.localize(filter_end_datetime)
    filename_date = selected_date_str if selected_date_str else datetime.now().strftime('%Y-%m-%d')
    filename_district = selected_district.replace(' ', '_') if selected_district else 'All_Districts'
    filename_start_time = selected_start_time_for_template.replace(':', '-') if selected_start_time_for_template else '00-00'
    filename_end_time = selected_end_time_for_template.replace(':', '-') if selected_end_time_for_template else '23-59'

    report_folder_name = f"{filename_date}_{filename_district}_{filename_start_time}-{filename_end_time}"
    
    report_specific_media_dir = os.path.join(settings.MEDIA_ROOT, 'report_pdf_temp_images', report_folder_name)
    if os.path.exists(report_specific_media_dir):
        print(f"PDF Gen: Cleaning up existing report image directory: {report_specific_media_dir}")
        try:
            shutil.rmtree(report_specific_media_dir) # Remove the directory and its contents
        except OSError as e:
            print(f"PDF Gen: Error removing directory {report_specific_media_dir}: {e}")
    
    os.makedirs(report_specific_media_dir, exist_ok=True) # Create the (new) temporary directory
    generated_images_for_pdf_template = [] # List to store file paths of images for the PDF template
    
    # --- NEW: List to hold Ramanathapuram automation image data for PDF ---
    ramanathapuram_automation_images_pdf = []

    gdf = None
    gdf_tn = None
    try:
        if not os.path.exists(SHAPEFILE_PATH):
            raise FileNotFoundError(f"Shapefile not found at {SHAPEFILE_PATH}.")
        gdf = gpd.read_file(SHAPEFILE_PATH)
        gdf_tn = gdf[
            (gdf['NAME_1'].str.strip().str.lower() == 'tamilnadu') |
            (gdf['NAME_1'].str.strip().str.lower() == 'tamil nadu')
        ].to_crs("EPSG:4326")
    except FileNotFoundError as fnfe:
        print(f"PDF Gen: CRITICAL ERROR: Shapefile not found: {fnfe}")
    except Exception as e:
        print(f"PDF Gen: ERROR loading shapefile: {e}")
    if os.path.exists(settings.MEDIA_ROOT) and os.path.isdir(settings.MEDIA_ROOT) and gdf_tn is not None:
        available_image_timestamps_and_folders = []
        for d in os.listdir(settings.MEDIA_ROOT):
            full_path_to_folder = os.path.join(settings.MEDIA_ROOT, d)
            if os.path.isdir(full_path_to_folder):
                try:
                    folder_datetime_obj = datetime.strptime(d, '%Y-%m-%d_%H-%M-%S')
                    if current_timezone:
                        folder_datetime_obj = current_timezone.localize(folder_datetime_obj)
                    if folder_datetime_obj.date() == filter_date and \
                        filter_start_datetime <= folder_datetime_obj < filter_end_datetime:
                        available_image_timestamps_and_folders.append((folder_datetime_obj, full_path_to_folder))
                except (ValueError, IndexError):
                    try:
                        folder_date_part = datetime.strptime(d, '%Y-%m-%d').date()
                        if folder_date_part == filter_date:
                            potential_image_file = os.path.join(full_path_to_folder, 'cropped', 'tamil_nadu_cropped.png')
                            if os.path.exists(potential_image_file):
                                dummy_timestamp_for_day = datetime.combine(filter_date, time(12,0,0))
                                if current_timezone:
                                    dummy_timestamp_for_day = current_timezone.localize(dummy_timestamp_for_day)
                                available_image_timestamps_and_folders.append((dummy_timestamp_for_day, full_path_to_folder))
                    except (ValueError, IndexError):
                        pass

        available_image_timestamps_and_folders.sort(key=lambda x: x[0])
        for timestamp_dt, folder_path in available_image_timestamps_and_folders:
            base_image_path_for_this_timestamp = os.path.join(folder_path, 'cropped', 'tamil_nadu_cropped.png')
            pil_images = _generate_image_data_for_timestamp(
                base_image_path_for_this_timestamp, timestamp_dt, selected_district, gdf_tn, is_pdf_generation=True
            )

            if pil_images:
                current_image_set_for_pdf = {
                    'timestamp': timestamp_dt,
                    'cropped_tn_path_abs': None,
                    'masked_district_path_abs': None,
                    'aligned_overlay_tn_path_abs': None,
                }
                for img_type, pil_img in pil_images.items():
                    if pil_img:
                        filepath_abs = save_image_and_get_filepath(
                            pil_img, report_specific_media_dir, timestamp_dt, img_type
                        )
                        current_image_set_for_pdf[f'{img_type}_path_abs'] = filepath_abs
                    elif img_type == 'masked_district' and selected_district == 'All Districts':
                        current_image_set_for_pdf['masked_district_path_abs'] = current_image_set_for_pdf['cropped_tn_path_abs']
                    else:
                        current_image_set_for_pdf[f'{img_type}_path_abs'] = None

                generated_images_for_pdf_template.append(current_image_set_for_pdf)
    else:
        print(f"PDF Gen: Image generation skipped for PDF: Base media directory or shapefile not loaded successfully.")

    # --- NEW: Logic to fetch Ramanathapuram automation images for PDF ---
    if selected_district.lower() == 'ramanathapuram':
        if os.path.exists(BASE_AUTOMATION_IMAGES_DIR) and os.path.isdir(BASE_AUTOMATION_IMAGES_DIR):
            automation_image_folders_pdf = []
            for d in os.listdir(BASE_AUTOMATION_IMAGES_DIR):
                full_path_to_automation_folder = os.path.join(BASE_AUTOMATION_IMAGES_DIR, d)
                if os.path.isdir(full_path_to_automation_folder):
                    try:
                        folder_datetime_obj = datetime.strptime(d, '%Y-%m-%d_%H-%M-%S')
                        if current_timezone:
                            folder_datetime_obj = current_timezone.localize(folder_datetime_obj)

                        if folder_datetime_obj.date() == filter_date and \
                            filter_start_datetime <= folder_datetime_obj < filter_end_datetime:
                            automation_image_folders_pdf.append((folder_datetime_obj, full_path_to_automation_folder))
                    except (ValueError, IndexError):
                        pass

            automation_image_folders_pdf.sort(key=lambda x: x[0])

            for timestamp_dt, folder_path in automation_image_folders_pdf:
                current_auto_image_set_pdf = {
                    'timestamp': timestamp_dt,
                    'full_screenshot_path_abs': None,
                    'cropped_screenshot_path_abs': None,
                    'masked_taluk_images_paths_abs': {}
                }

                # Path to full screenshot
                full_screenshot_path = os.path.join(folder_path, 'windy_full.png')
                if os.path.exists(full_screenshot_path):
                    # Save a temporary copy for PDF rendering
                    temp_full_screenshot_name = f"full_windy_{timestamp_dt.strftime('%H%M%S')}.png"
                    temp_full_screenshot_path = os.path.join(report_specific_media_dir, timestamp_dt.strftime('%H-%M-%S'), temp_full_screenshot_name)
                    os.makedirs(os.path.dirname(temp_full_screenshot_path), exist_ok=True)
                    shutil.copy(full_screenshot_path, temp_full_screenshot_path)
                    current_auto_image_set_pdf['full_screenshot_path_abs'] = temp_full_screenshot_path.replace(os.sep, '/')
                
                # Path to cropped screenshot
                cropped_screenshot_path = os.path.join(folder_path, 'windy_cropped.png')
                if os.path.exists(cropped_screenshot_path):
                    # Save a temporary copy for PDF rendering
                    temp_cropped_screenshot_name = f"cropped_windy_{timestamp_dt.strftime('%H%M%S')}.png"
                    temp_cropped_screenshot_path = os.path.join(report_specific_media_dir, timestamp_dt.strftime('%H-%M-%S'), temp_cropped_screenshot_name)
                    os.makedirs(os.path.dirname(temp_cropped_screenshot_path), exist_ok=True)
                    shutil.copy(cropped_screenshot_path, temp_cropped_screenshot_path)
                    current_auto_image_set_pdf['cropped_screenshot_path_abs'] = temp_cropped_screenshot_path.replace(os.sep, '/')

                # Paths to masked taluk images
                masked_taluks_dir = os.path.join(folder_path, 'masked_taluks')
                if os.path.exists(masked_taluks_dir) and os.path.isdir(masked_taluks_dir):
                    for taluk_img_file in os.listdir(masked_taluks_dir):
                        if taluk_img_file.endswith('.png'):
                            taluk_name_from_filename = taluk_img_file.replace('_masked.png', '').replace('_', ' ').title()
                            taluk_image_original_path = os.path.join(masked_taluks_dir, taluk_img_file)
                            if os.path.exists(taluk_image_original_path):
                                # Save a temporary copy for PDF rendering
                                temp_taluk_image_name = f"{taluk_name_from_filename.lower().replace(' ', '_')}_masked_{timestamp_dt.strftime('%H%M%S')}.png"
                                temp_taluk_image_path = os.path.join(report_specific_media_dir, timestamp_dt.strftime('%H-%M-%S'), temp_taluk_image_name)
                                os.makedirs(os.path.dirname(temp_taluk_image_path), exist_ok=True)
                                shutil.copy(taluk_image_original_path, temp_taluk_image_path)
                                current_auto_image_set_pdf['masked_taluk_images_paths_abs'][taluk_name_from_filename] = temp_taluk_image_path.replace(os.sep, '/')

                ramanathapuram_automation_images_pdf.append(current_auto_image_set_pdf)
        else:
            print(f"PDF Gen: Ramanathapuram automation images directory not found: {BASE_AUTOMATION_IMAGES_DIR}")

    cloud_analysis_query_pdf = Cloud_ramanathapuram.objects.filter(
        timestamp__date=filter_date,
        **{'type': 'Cloud Coverage'} # <--- ADDED THIS LINE!
    )

    if selected_district != 'All Districts':
        cloud_analysis_query_pdf = cloud_analysis_query_pdf.filter(city__iexact=selected_district)

    if filter_start_datetime and filter_end_datetime:
        cloud_analysis_query_pdf = cloud_analysis_query_pdf.filter(
            timestamp__gte=filter_start_datetime,
            timestamp__lt=filter_end_datetime
        )

    EXCLUDE_NO_PRECIP_MESSAGE = "No significant cloud levels found for precipitation"
    cloud_analysis_query_pdf = cloud_analysis_query_pdf.exclude(values__iexact=EXCLUDE_NO_PRECIP_MESSAGE)

    filtered_cloud_analysis_data_for_pdf = list(cloud_analysis_query_pdf.order_by('city', 'timestamp'))
    print(f"PDF Gen: Fetched {len(filtered_cloud_analysis_data_for_pdf)} weather data points for PDF.")

    # --- Prepare context for the PDF HTML template ---
    context_for_pdf = {
        'report_generation_time': datetime.now(),
        'selected_date': filter_date.strftime('%Y-%m-%d'),
        'selected_district': selected_district,
        'selected_start_time': selected_start_time_for_template,
        'selected_end_time': selected_end_time_for_template,
        'selected_image_view': selected_image_view, # Used to conditionally display image types in the PDF
        'selected_image_view_label': { # Map internal codes to human-readable labels for the PDF
            'cropped_tn': 'Cropped Tamil Nadu (Radar Only)',
            'masked_district': f'Shape-Masked {selected_district}',
            'tn_overlay': 'Overall TN Map with District Outlines', # Assuming 'aligned_overlay_tn' maps to this
            'combined_full_tn': 'Combined Full TN View (Radar + Outlines)',
        }.get(selected_image_view, 'No Specific View Selected'),
        'cloud_analysis_data': filtered_cloud_analysis_data_for_pdf,
        'generated_images_for_display': generated_images_for_pdf_template, # Contains file paths for PDF
        'ramanathapuram_automation_images': ramanathapuram_automation_images_pdf, # NEW: Add for PDF
    }

    # --- NEW PDF SAVE PATH LOGIC ---
    # Define the directory for PDF reports within MEDIA_ROOT
    pdf_report_save_dir = os.path.join(settings.MEDIA_ROOT, 'reports_pdf')
    
    # Create the directory if it doesn't exist
    os.makedirs(pdf_report_save_dir, exist_ok=True)

    # Define the full PDF file path inside the new directory
    pdf_filename = f"report_{report_folder_name}.pdf"
    pdf_path = os.path.join(pdf_report_save_dir, pdf_filename) # Updated path

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch() # Launch a headless Chromium browser
            page = browser.new_page() # Open a new browser page
            html_content = render_to_string('report/report_pdf.html', context_for_pdf)
            page.set_content(html_content)  
            page.wait_for_load_state('networkidle')
            page.pdf(path=pdf_path, format='A4', print_background=True) # Save PDF to the specified path
            browser.close() # Close the browser
        print(f"PDF Gen: PDF generated successfully and saved to {pdf_path}") # Updated message to reflect new path

        with open(pdf_path, 'rb') as f: # Open the PDF file in binary read mode
            response = HttpResponse(f.read(), content_type='application/pdf') # Create HTTP response with PDF content
            response['Content-Disposition'] = f'attachment; filename="{pdf_filename}"' # Force download with specified filename
            return response

    except Exception as e:
        print(f"PDF Gen: Error generating PDF: {e}")
        return JsonResponse({"status": "error", "Message": f"Failed to generate PDF: {e}"}, status=500)
    finally:
        # --- Cleanup: Ensure the temporary image directory is removed ---
        if os.path.exists(report_specific_media_dir):
            try:
                shutil.rmtree(report_specific_media_dir)
                print(f"PDF Gen: Cleaned up temporary report images directory: {report_specific_media_dir}")
            except OSError as cleanup_error:
                print(f"PDF Gen: Error cleaning up temporary directory {report_specific_media_dir}: {cleanup_error}")
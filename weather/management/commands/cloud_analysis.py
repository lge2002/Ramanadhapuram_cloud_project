import os
import time
import json
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
import cv2
import geopandas as gpd
from shapely.affinity import scale, translate
from shapely.ops import unary_union
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import warnings
import requests
import pytz
from django.core.management.base import BaseCommand
from django.conf import settings
from weather.models import Cloud_ramanathapuram # Ensure this model is correctly defined
from django.template.loader import render_to_string
from xhtml2pdf import pisa

warnings.filterwarnings("ignore")   
 
class Command(BaseCommand):
    help = 'Automates screenshot capture from Windy.com, crops, masks, and analyzes cloud levels for Ramanathapuram district (taluk-wise), then saves to DB, JSON, PDF, and pushes to API.'

    API_ENDPOINT_URL = "http://172.16.7.118:8003/api/tamilnadu/satellite/push.windy_radar_data.php?type"

    # --- Geo-alignment parameters for Ramanathapuram on Windy.com ---
    CROP_BOX = (575, 135, 1060, 610) # Pixel coordinates (left, upper, right, lower) for the region of interest
    WINDY_URL = "https://www.windy.com/-Satellite-satellite?satellite,9.466,78.742,9,p:favs" # URL centered on Ramanathapuram
    DISTRICT_NAME = "Ramanathapuram"
    MIN_LON = 78.80
    MAX_LON = 80.35
    MIN_LAT = 8.95
    MAX_LAT = 10.53

    # Transformation parameters for the shapefile to align with the cropped image.
    ZOOM = 1.18
    MOVE_LEFT = 0.20
    MOVE_RIGHT = 1.13
    MOVE_UP = 0.52
    MOVE_DOWN = 0.32
    
    SHAPEFILE_PATH = R"C:\Users\tamilarasans\Downloads\gadm41_IND_3.json (2)\gadm41_IND_3.json"

    DISTRICT_FIELD = "NAME_2"
    TALUK_NAME_FIELD = "NAME_3"

    def _link_callback(self, uri, rel):
        """
        Callback to handle links in the HTML for xhtml2pdf.
        Ensures local file paths are correctly resolved for images embedded in PDF.
        """
        if uri.startswith('file:///'):
            return os.path.normpath(uri[len('file:///'):]).replace(os.path.sep, '/')
        
        sUrl = settings.STATIC_URL if hasattr(settings, 'STATIC_URL') else '/static/'
        sRoot = settings.STATIC_ROOT if hasattr(settings, 'STATIC_ROOT') else None
        mUrl = settings.MEDIA_URL if hasattr(settings, 'MEDIA_URL') else '/media/'
        mRoot = settings.MEDIA_ROOT if hasattr(settings, 'MEDIA_ROOT') else None

        if uri.startswith(mUrl) and mRoot:
            path = os.path.join(mRoot, uri.replace(mUrl, ""))
        elif uri.startswith(sUrl) and sRoot:
            path = os.path.join(sRoot, uri.replace(sUrl, ""))
        else:
            return uri

        if os.path.exists(path):
            return path.replace(os.path.sep, '/')
        else:
            self.stderr.write(self.style.WARNING(f"Warning: Linked file not found for PDF: {path} (Original URI: {uri})"))
            return uri

    def _generate_and_save_automation_pdf(self, results_data, current_time, base_folder, # current_time here will be rounded_time
                                         full_screenshot_path_abs, cropped_screenshot_path_abs,
                                         masked_image_paths,
                                         json_output_content):
        """
        Generates a PDF report using xhtml2pdf from a Django template.
        """
        pdf_filename = f"automation_report_{current_time.strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_output_path = os.path.join(base_folder, pdf_filename)

        full_screenshot_path_for_html = os.path.abspath(full_screenshot_path_abs).replace(os.path.sep, '/')
        cropped_screenshot_path_for_html = os.path.abspath(cropped_screenshot_path_abs).replace(os.path.sep, '/')
        
        masked_image_paths_for_html = {
            taluk_name: f'file:///{os.path.abspath(path).replace(os.path.sep, "/")}'
            for taluk_name, path in masked_image_paths.items()
        }

        context = {
            'current_time': current_time, # This now correctly uses the rounded_time passed in
            'district_name': self.DISTRICT_NAME,
            'current_run_results': results_data,
            'full_screenshot_path_abs': f'file:///{full_screenshot_path_for_html}',
            'cropped_screenshot_path_abs': f'file:///{cropped_screenshot_path_for_html}',
            'masked_image_paths': masked_image_paths_for_html,
            'json_output_content': json_output_content,
        }
        
        html_string = render_to_string('weather/ramanathapuram_automation_report_pdf.html', context)

        with open(pdf_output_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(
                html_string,
                dest=pdf_file,
                link_callback=self._link_callback
            )
        if pisa_status.err:
            raise Exception(f"PDF generation error with xhtml2pdf: {pisa_status.err}")

        self.stdout.write(self.style.SUCCESS(f"PDF report generated and saved successfully to: {pdf_output_path}"))

    def _round_to_nearest_minutes(self, dt_object, minutes=15):
        # Remove seconds and microseconds for accurate rounding
        dt_object = dt_object.replace(second=0, microsecond=0)
        discard = timedelta(minutes=dt_object.minute % minutes)
        dt_floor = dt_object - discard
        dt_ceil = dt_floor + timedelta(minutes=minutes)
        # Choose the closest rounded time
        if (dt_object - dt_floor) < (dt_ceil - dt_object):
            return dt_floor
        else:
            return dt_ceil

    def _print_time_rounding(self, dt_object, minutes=15):
        rounded_dt = self._round_to_nearest_minutes(dt_object, minutes)
        self.stdout.write(self.style.SUCCESS(f"Raw time: {dt_object.strftime('%Y-%m-%d %H:%M:%S')} | Rounded time: {rounded_dt.strftime('%Y-%m-%d %H:%M:%S')} (to nearest {minutes} min)"))
        return rounded_dt

    def _is_cloud_pixel(self, rgb_pixel):
        r, g, b = rgb_pixel
        hsv_pixel = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0][0]
        h, s, v = hsv_pixel

        # Adjusted cloud detection logic
        # High value (brightness) and low saturation often indicates white/grey clouds
        if v > 190 and s < 60:
            return True
        # Very high value (very bright)
        if v > 220:
            return True
        # Check for specific pale/white RGB ranges with low saturation
        if (180 <= r <= 255 and 180 <= g <= 255 and 190 <= b <= 255) and (s < 90):
            return True
        return False

    def handle(self, *args, **kwargs):
        self.stdout.write(self.style.SUCCESS(f'Starting Windy.com cloud analysis automation for {self.DISTRICT_NAME} district (taluk-wise)...'))
        
        # --- Initial Setup (Load Shapefile once, outside the loop) ---
        if not os.path.exists(self.SHAPEFILE_PATH):
            self.stderr.write(self.style.ERROR(f"Critical Error: Taluk shapefile not found at {self.SHAPEFILE_PATH}. Exiting."))
            return

        try:
            gdf_original = gpd.read_file(self.SHAPEFILE_PATH)
            if self.DISTRICT_FIELD not in gdf_original.columns or self.TALUK_NAME_FIELD not in gdf_original.columns:
                self.stderr.write(self.style.ERROR(f"Missing expected fields in shapefile. Found: {list(gdf_original.columns)}"))
                return

            ramanathapuram_gdf = gdf_original[gdf_original[self.DISTRICT_FIELD] == self.DISTRICT_NAME].copy().to_crs("EPSG:4326")
            if ramanathapuram_gdf.empty:
                self.stderr.write(self.style.ERROR(f"No taluks found for district: {self.DISTRICT_NAME} in the shapefile. Exiting."))
                return
            self.stdout.write(self.style.SUCCESS(f"Loaded shapefile with {len(ramanathapuram_gdf)} taluks for {self.DISTRICT_NAME}."))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Failed to load or parse shapefile: {e}. Exiting."))
            return

        # --- Main Automation Loop ---
        while True: # Loop the entire process indefinitely
            self.stdout.write(f"\n--- Starting new automation run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

            # --- 2. Setup Output Paths and Timestamps (inside the loop for new folders each run) ---
            current_raw_time = datetime.now() # Get the raw current time
            # Show raw and rounded time in terminal
            rounded_time = self._print_time_rounding(current_raw_time, 15) # Pass raw time for rounding
            # Remove seconds and microseconds for folder/file names
            rounded_time_no_sec = rounded_time.replace(second=0, microsecond=0)
            timestamp_str_for_filenames = rounded_time_no_sec.strftime('%Y-%m-%d_%H-%M-%S')
            
            base_folder = os.path.join(settings.BASE_DIR, "images", self.DISTRICT_NAME.lower().replace(" ", "_"), timestamp_str_for_filenames)
            os.makedirs(base_folder, exist_ok=True)
            full_image_folder = os.path.join(base_folder, "full")
            cropped_image_folder = os.path.join(base_folder, "cropped")
            os.makedirs(full_image_folder, exist_ok=True)
            os.makedirs(cropped_image_folder, exist_ok=True)
            full_screenshot_path = os.path.join(full_image_folder, "windy_full.png") 

            cropped_screenshot_path = os.path.join(cropped_image_folder, "ramanathapuram_cropped.png") 
            json_output_path = os.path.join(base_folder, "cloud_analysis_results.json")
            masked_image_base_folder = os.path.join(base_folder, "masked_taluks")
            os.makedirs(masked_image_base_folder, exist_ok=True)

            self.stdout.write(f"Saving output for this run to: {base_folder}")

            # --- 3. Selenium Automation: Capture Screenshot ---
            self.stdout.write("Launching Chrome and automating Windy.com...")
            options = webdriver.ChromeOptions()
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--start-maximized") 
            # options.add_argument("--headless") # Uncomment for headless operation
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-infobars")

            driver = None
            try:
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=options)
                wait = WebDriverWait(driver, 20) 

                driver.get(self.WINDY_URL)
                self.stdout.write(f"Navigated to {self.WINDY_URL}. Waiting for page to load...")
                time.sleep(7) # Give extra time for elements to load, especially maps

                try:
                    # Look for multiple common cookie consent selectors
                    cookie_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button.accept-all, .cookie-consent-button[data-action='accept'], #onetrust-accept-btn-handler")))
                    cookie_button.click()
                    self.stdout.write("Accepted cookies.")
                    time.sleep(1)
                except Exception:
                    self.stdout.write("No cookie consent banner found or couldn't click it. Continuing...")

                try:
                    # Select Visible layer within Satellite+ (if applicable)
                    # This XPath is quite specific and might change. Adapt if needed.
                    visible_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@id='plugin-radar-plus']/div[1]/div[6]/div[1]/div[2]")))
                    visible_btn.click()
                    time.sleep(3) # Give time for the layer to change
                    self.stdout.write("Selected Visible layer within Satellite+.")
                except Exception:
                    self.stdout.write("No 'Visible' layer button found or couldn't click it. Assuming current sub-layer is sufficient.")
                
                # --- Crucial: Wait for the radar/satellite animation to finish or for the map to stabilize ---
                # This is a generic wait. If the map content changes during animation, your screenshot might be inconsistent.
                # Consider using a more robust wait condition if map elements or data specifically appear/disappear.
                self.stdout.write("Waiting for map animation/data to stabilize (additional 5 seconds)...")
                time.sleep(5) 

                driver.save_screenshot(full_screenshot_path)
                self.stdout.write(self.style.SUCCESS(f"Full screenshot saved at {full_screenshot_path}"))

            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error during Selenium automation: {e}"))
                if driver:
                    driver.save_screenshot(os.path.join(base_folder, "error_screenshot.png"))
                    self.stderr.write(self.style.WARNING(f"Error screenshot saved to {os.path.join(base_folder, 'error_screenshot.png')}"))
                # If Selenium fails, we can't proceed with image processing for this run.
                # Close browser and then continue to the next loop iteration after sleep.
                if driver: driver.quit() 
                self.stdout.write(f"Waiting {120 / 60} minutes before next run due to Selenium error...\n")
                time.sleep(120) # Wait 2 minutes before retrying the full run
                continue # Skip the rest of the current loop iteration

            finally:
                if driver:
                    driver.quit()
                    self.stdout.write("Chrome driver quit successfully.")

            # --- 4. Image Processing: Crop Screenshot ---
            try:
                full_img = Image.open(full_screenshot_path).convert("RGB")
                
                # Validate CROP_BOX before cropping
                if not (0 <= self.CROP_BOX[0] < self.CROP_BOX[2] <= full_img.width and
                                0 <= self.CROP_BOX[1] < self.CROP_BOX[3] <= full_img.height):
                    self.stderr.write(self.style.ERROR(f"ERROR: CROP_BOX coordinates ({self.CROP_BOX}) are invalid or out of bounds for the screenshot dimensions ({full_img.width}x{full_img.height})."))
                    self.stderr.write(self.style.ERROR("Please re-calibrate CROP_BOX based on a full screenshot taken with current browser settings."))
                    # Skip image processing and analysis for this run if crop box is invalid
                    self.stdout.write(f"Waiting {120 / 60} minutes before next run due to CROP_BOX error...\n")
                    time.sleep(120)
                    continue # Skip to next loop iteration

                cropped_img = full_img.crop(self.CROP_BOX)
                cropped_img.save(cropped_screenshot_path)
                self.stdout.write(self.style.SUCCESS(f"Cropped screenshot saved at {cropped_screenshot_path}"))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error cropping screenshot: {e}. Skipping image analysis for this run."))
                self.stdout.write(f"Waiting {120 / 60} minutes before next run due to cropping error...\n")
                time.sleep(120)
                continue # Skip to next loop iteration

            # --- 5. Prepare Image for Analysis and Transform Shapefile ---
            self.stdout.write(f"Image extent used for plotting: LON({self.MIN_LON}-{self.MAX_LON}), LAT({self.MIN_LAT}-{self.MAX_LAT})")
            center_x = (self.MIN_LON + self.MAX_LON) / 2
            center_y = (self.MIN_LAT + self.MAX_LAT) / 2
            x_offset = self.MOVE_RIGHT - self.MOVE_LEFT
            y_offset = self.MOVE_UP - self.MOVE_DOWN
            self.stdout.write(f"Shapefile transformation parameters: ZOOM={self.ZOOM}, X_OFFSET={x_offset}, Y_OFFSET={y_offset}")

            # Prepare image for rasterization and pixel analysis
            img_pil = cropped_img.convert("RGB")
            img_np = np.array(img_pil)
            height, width, _ = img_np.shape

            # Define transform for rasterization (mapping geo-coordinates to pixel space)
            transform_for_rasterize = from_bounds(self.MIN_LON, self.MIN_LAT, self.MAX_LON, self.MAX_LAT, width, height)

            original_rgba_image = cropped_img.convert("RGBA")
            original_rgba_np = np.array(original_rgba_image)

            # --- 6. Taluk-wise Cloud Analysis ---
            self.stdout.write(f"\nPerforming taluk-wise cloud analysis for {self.DISTRICT_NAME}...")

            current_run_results = [] # Initialize for each run

            # Ensure masked_image_paths is reset for each run if used in PDF generation
            masked_image_paths = {} 

            for index, row in ramanathapuram_gdf.iterrows():
                taluk_name = row[self.TALUK_NAME_FIELD]
                taluk_geometry = row.geometry

                self.stdout.write(f"     Analyzing cloud coverage for taluk: {taluk_name}...")

                try:
                    # Apply the same transformations to individual taluk geometry for consistent alignment
                    transformed_taluk_geometry = scale(taluk_geometry, xfact=self.ZOOM, yfact=self.ZOOM, origin=(center_x, center_y))
                    transformed_taluk_geometry = translate(transformed_taluk_geometry, xoff=x_offset, yoff=y_offset)

                    taluk_mask = rasterize(
                        [transformed_taluk_geometry],
                        out_shape=(height, width),
                        transform=transform_for_rasterize,
                        fill=0,
                        all_touched=True,
                        dtype=np.uint8
                    )

                    taluk_transparent_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                    total_taluk_pixels = 0
                    cloudy_taluk_pixels = 0

                    for y in range(height):
                        for x in range(width):
                            if taluk_mask[y, x]:
                                r, g, b, a = original_rgba_np[y, x]
                                if a > 0: # Only count pixels that are not fully transparent in the original cropped image
                                    total_taluk_pixels += 1
                                    taluk_transparent_image.putpixel((x, y), (r, g, b, 255)) # Make visible in masked image
                                    if self._is_cloud_pixel((r, g, b)):
                                        cloudy_taluk_pixels += 1
                    if total_taluk_pixels == 0:
                        self.stderr.write(self.style.WARNING(f"     Warning: Taluk '{taluk_name}' has 0 total pixels after masking. This may indicate a geometry or alignment issue."))
                        cloud_percentage_taluk = 0.0
                    else:
                        cloud_percentage_taluk = (cloudy_taluk_pixels / total_taluk_pixels) * 100

                    cloud_percentage_taluk_str = f"{cloud_percentage_taluk:.2f}%"
                    taluk_safe_name = taluk_name.lower().replace(' ', '_').replace('.', '')
                    taluk_specific_output_folder = os.path.join(masked_image_base_folder, taluk_safe_name)
                    os.makedirs(taluk_specific_output_folder, exist_ok=True)
                    taluk_filename = f"{taluk_safe_name}_masked.png"
                    taluk_masked_path = os.path.join(taluk_specific_output_folder, taluk_filename)
                    taluk_transparent_image.save(taluk_masked_path)
                    masked_image_paths[taluk_name] = taluk_masked_path

                    self.stdout.write(f"     Cloud coverage for {taluk_name}: {cloud_percentage_taluk_str} (Total pixels: {total_taluk_pixels}, Cloudy pixels: {cloudy_taluk_pixels})")

                    # --- Use rounded_time for database save ---
                    # Always use rounded_time with seconds and microseconds set to zero for DB
                    timestamp_for_db = rounded_time.replace(second=0, microsecond=0)
                    if settings.USE_TZ:
                        target_tz = pytz.timezone(settings.TIME_ZONE)
                        if timestamp_for_db.tzinfo is None or timestamp_for_db.tzinfo.utcoffset(timestamp_for_db) is None:
                            timestamp_for_db = target_tz.localize(timestamp_for_db)
                        else:
                            timestamp_for_db = timestamp_for_db.astimezone(target_tz)

                    # --- 7. Save to Django Model ---
                    try:
                        Cloud_ramanathapuram.objects.update_or_create(
                            city=taluk_name,
                            timestamp=timestamp_for_db, # Use the rounded and localized time here
                            defaults={
                                "values": cloud_percentage_taluk_str,
                                "type": "Cloud Coverage"
                            }
                        )
                        self.stdout.write(self.style.SUCCESS(f"     Cloud analysis for {taluk_name} saved/updated in database (rounded to 15 min)."))
                    except Exception as e:
                        self.stderr.write(self.style.ERROR(f"     Error saving {taluk_name} to Django model: {e}"))

                    taluk_data = {
                        "district": self.DISTRICT_NAME,
                        "taluk": taluk_name,
                        "values": cloud_percentage_taluk_str,
                        "type": "Cloud Coverage",
                        "timestamp": timestamp_for_db.strftime('%Y-%m-%d %H:%M:%S') # Use rounded_time for JSON payload as well
                    }
                    current_run_results.append(taluk_data)

                except Exception as e:
                    self.stderr.write(self.style.ERROR(f"Error during image processing, masking, or analysis for taluk {taluk_name}: {e}"))
            
            json_output_content = json.dumps(current_run_results, indent=4)
            try:
                with open(json_output_path, "w") as json_file:
                    json_file.write(json_output_content)
                self.stdout.write(self.style.SUCCESS(f"Analysis results for {self.DISTRICT_NAME} taluks saved to JSON at: {json_output_path}"))
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"Error saving JSON file: {e}"))

            self.stdout.write("\nGenerating PDF report for this run...")
            try:
                full_screenshot_path_abs = os.path.abspath(full_screenshot_path)
                cropped_screenshot_path_abs = os.path.abspath(cropped_screenshot_path)

                self._generate_and_save_automation_pdf(
                    current_run_results,
                    rounded_time, # Corrected: Pass the rounded time to the PDF generation function
                    base_folder,
                    full_screenshot_path_abs,
                    cropped_screenshot_path_abs,
                    masked_image_paths,
                    json_output_content
                )
            except Exception as e:
                self.stderr.write(self.style.ERROR(f"CRITICAL: Error generating PDF report for this run: {e}"))

            # --- 10. API PUSHING ---
            num_post_attempts = 3
            for i in range(num_post_attempts):
                self.stdout.write(f"\n--- URL PUSHING ATTEMPT {i + 1} of {num_post_attempts} ---")

                if self.API_ENDPOINT_URL and current_run_results:
                    self.stdout.write(f"Attempting to send analysis data to {self.API_ENDPOINT_URL} via POST (Attempt {i+1})...")

                    headers = {
                        'Content-Type': 'application/json',
                    }

                    try:
                        self.stdout.write(f"Sending JSON payload (first 500 chars): {json.dumps(current_run_results, indent=4)[:500]}...")
                        response = requests.post(self.API_ENDPOINT_URL, json=current_run_results, headers=headers, timeout=120)
                        response.raise_for_status()

                        self.stdout.write(self.style.SUCCESS(f"Data successfully POSTed to {self.API_ENDPOINT_URL} (Attempt {i+1})."))
                        self.stdout.write(f"API Response Status Code: {response.status_code}")
                        try:
                            self.stdout.write(f"API Response JSON: {json.dumps(response.json(), indent=2)}")
                        except json.JSONDecodeError:
                            self.stdout.write(f"API Response Text (not JSON): {response.text}")
                        break # Break out of retry loop on success
                    except requests.exceptions.HTTPError as http_err:
                        self.stderr.write(self.style.ERROR(f"HTTP error during POST request (Attempt {i+1}): {http_err}"))
                        if http_err.response is not None:
                            self.stderr.write(self.style.ERROR(f"Response from API (Attempt {i+1}): {http_err.response.text}"))
                    except requests.exceptions.ConnectionError as conn_err:
                        self.stderr.write(self.style.ERROR(f"Connection error during POST request (Attempt {i+1}). Is the server at {self.API_ENDPOINT_URL} reachable and port open? Error: {conn_err}"))
                    except requests.exceptions.Timeout as timeout_err:
                        self.stderr.write(self.style.ERROR(f"Timeout error during POST request (Attempt {i+1}). API took too long to respond: {timeout_err}"))
                    except requests.exceptions.RequestException as req_err:
                        self.stderr.write(self.style.ERROR(f"An unexpected error occurred during POST request (Attempt {i+1}): {req_err}"))

                    if i < num_post_attempts - 1:
                        self.stdout.write("Waiting 10 seconds before next POST attempt...")
                        time.sleep(10)
                else:
                    if not self.API_ENDPOINT_URL:
                        self.stdout.write(self.style.WARNING(f"API_ENDPOINT_URL is not set. Skipping POST request (Attempt {i+1})."))
                    if not current_run_results:
                        self.stdout.write(self.style.WARNING(f"No analysis results to send. Skipping POST request (Attempt {i+1})."))
                    break # Break out of retry loop if no URL or no results

            self.stdout.write("\nFinished all URL pushing cycles for this data set.")
            self.stdout.write(self.style.SUCCESS("Windy.com cloud analysis automation completed for this run."))
            
            # --- Delay before the next loop iteration ---
            self.stdout.write(f"Waiting 5 minutes before next full run...\n")
            time.sleep(300) # This is the delay between full runs why the data was not roundoff for 15 minutes?
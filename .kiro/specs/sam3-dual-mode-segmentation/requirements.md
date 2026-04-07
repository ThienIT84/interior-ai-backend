# SAM3 Dual-Mode Segmentation - Requirements

## Requirement 1: Backend SAM3 Integration

**User Story**: As a backend developer, I want to integrate SAM3 via Replicate API, so that the system can perform advanced segmentation without local GPU constraints.

### Acceptance Criteria

1.1. WHEN the SAM3Segmentation class is initialized THEN it SHALL configure the Replicate client with the correct model ID

1.2. WHEN segment_by_text is called with a valid image and text prompt THEN it SHALL return a binary mask matching the described objects

1.3. WHEN segment_by_points is called with valid coordinates THEN it SHALL return a binary mask for the selected objects

1.4. WHEN the Replicate API fails THEN the system SHALL retry up to 3 times with exponential backoff

1.5. WHEN a segmentation operation completes THEN the system SHALL log the cost ($0.00098) and processing time

## Requirement 2: Text-Based Segmentation API

**User Story**: As a mobile app developer, I want a REST API endpoint for text-based segmentation, so that users can describe objects naturally.

### Acceptance Criteria

2.1. WHEN POST /api/v1/segmentation/sam3/text is called with valid image_id and text_prompt THEN it SHALL return a mask_id and mask_url

2.2. WHEN the text_prompt is empty or whitespace-only THEN the API SHALL return 400 Bad Request

2.3. WHEN the image_id does not exist THEN the API SHALL return 404 Not Found

2.4. WHEN segmentation succeeds THEN the response SHALL include processing_time_ms and cost_usd

2.5. WHEN the Replicate API is unavailable THEN the endpoint SHALL return 503 Service Unavailable

## Requirement 3: Point-Based Segmentation API

**User Story**: As a mobile app developer, I want a REST API endpoint for point-based segmentation using SAM3, so that users can tap to select objects.

### Acceptance Criteria

3.1. WHEN POST /api/v1/segmentation/sam3/points is called with valid points THEN it SHALL return a mask_id

3.2. WHEN points array is empty THEN the API SHALL return 400 Bad Request

3.3. WHEN point coordinates exceed image dimensions THEN the API SHALL return 400 Bad Request with validation error

3.4. WHEN multiple points are provided THEN the system SHALL segment all indicated objects

3.5. WHEN segmentation completes THEN the mask SHALL be saved to storage with a unique mask_id

## Requirement 4: Dual-Mode UI Implementation

**User Story**: As a user, I want to switch between text and tap modes, so that I can choose the most convenient segmentation method.

### Acceptance Criteria

4.1. WHEN the segmentation screen loads THEN it SHALL display a mode selector with "Text" and "Tap" options

4.2. WHEN I select "Text" mode THEN the UI SHALL show a text input field and hide tap instructions

4.3. WHEN I select "Tap" mode THEN the UI SHALL show tap instructions and hide the text input field

4.4. WHEN I switch modes THEN any current selection (text or points) SHALL be cleared

4.5. WHEN I switch modes THEN the displayed mask SHALL be cleared

## Requirement 5: Text Input Validation

**User Story**: As a user, I want immediate feedback on my text input, so that I know when my description is valid.

### Acceptance Criteria

5.1. WHEN I type in the text field THEN the "Segment" button SHALL be enabled only if text is non-empty

5.2. WHEN the text field is empty THEN the "Segment" button SHALL be disabled

5.3. WHEN I enter whitespace-only text THEN the system SHALL treat it as invalid

5.4. WHEN I submit valid text THEN the system SHALL trim whitespace before sending to API

5.5. WHEN text exceeds 500 characters THEN the system SHALL show a character count warning

## Requirement 6: Text-Based Segmentation Workflow

**User Story**: As a user, I want to describe objects in natural language, so that I can quickly segment multiple items without precise clicking.

### Acceptance Criteria

6.1. WHEN I enter "sofa and table" and click Segment THEN the system SHALL segment both objects

6.2. WHEN segmentation is processing THEN the UI SHALL show a loading indicator

6.3. WHEN segmentation completes THEN the mask SHALL overlay the image with adjustable opacity

6.4. WHEN segmentation fails THEN the UI SHALL display an error message with retry option

6.5. WHEN I clear the text THEN the mask SHALL be removed from display

## Requirement 7: Point-Based Segmentation Workflow

**User Story**: As a user, I want to tap on objects to segment them, so that I have precise control over selection.

### Acceptance Criteria

7.1. WHEN I tap on the image THEN a numbered marker SHALL appear at the tap location

7.2. WHEN I add a point THEN segmentation SHALL trigger automatically

7.3. WHEN I add multiple points THEN each point SHALL be numbered sequentially

7.4. WHEN segmentation completes THEN the mask SHALL overlay the image

7.5. WHEN I tap "Undo" THEN the last point SHALL be removed and segmentation re-triggered

## Requirement 8: Mask Visualization

**User Story**: As a user, I want to see the segmentation mask overlaid on my image, so that I can verify the selection before proceeding.

### Acceptance Criteria

8.1. WHEN a mask is generated THEN it SHALL be displayed as a semi-transparent red overlay

8.2. WHEN I adjust the opacity slider THEN the mask transparency SHALL update in real-time

8.3. WHEN I toggle mask visibility THEN the mask SHALL show/hide without losing the selection

8.4. WHEN no mask exists THEN the opacity slider and visibility toggle SHALL be disabled

8.5. WHEN the mask loads THEN it SHALL maintain the correct aspect ratio and alignment with the image

## Requirement 9: Error Handling and User Feedback

**User Story**: As a user, I want clear feedback when operations fail, so that I understand what went wrong and how to proceed.

### Acceptance Criteria

9.1. WHEN the API returns an error THEN the UI SHALL display a user-friendly error message

9.2. WHEN network timeout occurs THEN the UI SHALL show "Request timeout - please try again"

9.3. WHEN the backend is unavailable THEN the UI SHALL show "Service unavailable - please check connection"

9.4. WHEN segmentation fails THEN the UI SHALL offer a "Retry" button

9.5. WHEN an error is displayed THEN the loading indicator SHALL be hidden

## Requirement 10: Performance and Cost Tracking

**User Story**: As a system administrator, I want to track API usage and costs, so that I can monitor budget and optimize performance.

### Acceptance Criteria

10.1. WHEN a SAM3 API call is made THEN the system SHALL log the timestamp and cost

10.2. WHEN segmentation completes THEN the system SHALL log the processing time in milliseconds

10.3. WHEN the API response is received THEN the system SHALL include cost_usd in the response

10.4. WHEN multiple segmentations occur THEN the system SHALL maintain a running total of costs

10.5. WHEN processing time exceeds 15 seconds THEN the system SHALL log a performance warning

## Requirement 11: Backward Compatibility

**User Story**: As a developer, I want SAM3 to coexist with SAM v1, so that existing functionality remains available.

### Acceptance Criteria

11.1. WHEN SAM3 is enabled THEN existing SAM v1 endpoints SHALL continue to function

11.2. WHEN SAM3 is disabled via config THEN the system SHALL fall back to SAM v1

11.3. WHEN both SAM versions are available THEN the system SHALL use SAM3 by default

11.4. WHEN SAM3 fails THEN the system SHALL NOT automatically fall back to SAM v1 (explicit user choice)

11.5. WHEN the API is called THEN the response SHALL indicate which SAM version was used

## Requirement 12: Configuration Management

**User Story**: As a system administrator, I want to configure SAM3 settings via environment variables, so that I can adjust behavior without code changes.

### Acceptance Criteria

12.1. WHEN the application starts THEN it SHALL load SAM3 settings from environment variables

12.2. WHEN ENABLE_SAM3 is false THEN SAM3 endpoints SHALL return 503 Service Unavailable

12.3. WHEN SAM3_TIMEOUT_SECONDS is set THEN API calls SHALL timeout after the specified duration

12.4. WHEN SAM3_MAX_RETRIES is configured THEN the system SHALL retry failed calls up to that limit

12.5. WHEN configuration is invalid THEN the system SHALL log an error and use default values

## Requirement 13: Data Persistence

**User Story**: As a user, I want my segmentation masks to be saved, so that I can use them for inpainting later.

### Acceptance Criteria

13.1. WHEN a mask is generated THEN it SHALL be saved to the filesystem with a unique mask_id

13.2. WHEN a mask is saved THEN it SHALL be stored as a PNG file with binary values (0 or 255)

13.3. WHEN GET /api/v1/segmentation/mask-image/{mask_id} is called THEN it SHALL return the mask image

13.4. WHEN a mask_id does not exist THEN the endpoint SHALL return 404 Not Found

13.5. WHEN masks are no longer needed THEN they SHALL be eligible for cleanup after 24 hours

## Requirement 14: Integration with Inpainting Pipeline

**User Story**: As a user, I want to proceed from segmentation to inpainting seamlessly, so that I can remove selected objects.

### Acceptance Criteria

14.1. WHEN I click "Remove Object" THEN the app SHALL navigate to the inpainting screen with image_id and mask_id

14.2. WHEN the inpainting screen loads THEN it SHALL display the original image with mask overlay

14.3. WHEN inpainting starts THEN it SHALL use the mask generated by SAM3

14.4. WHEN I return from inpainting THEN I SHALL be able to create a new segmentation

14.5. WHEN multiple masks exist for an image THEN the system SHALL use the most recent mask_id

## Requirement 15: Testing and Validation

**User Story**: As a developer, I want comprehensive tests for SAM3 integration, so that I can ensure reliability and catch regressions.

### Acceptance Criteria

15.1. WHEN unit tests run THEN they SHALL cover both text and point segmentation methods

15.2. WHEN integration tests run THEN they SHALL test the full API endpoint flow

15.3. WHEN property-based tests run THEN they SHALL validate mask format and dimensions

15.4. WHEN tests use the Replicate API THEN they SHALL use mocked responses to avoid costs

15.5. WHEN all tests pass THEN the system SHALL be ready for deployment

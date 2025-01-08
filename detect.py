import cv2
import numpy as np

def area(contour):
    # Get the bounding box of the contour (top-left corner, width, height)
    _, _, w, h = cv2.boundingRect(contour)
    # Calculate the area of the bounding rectangle
    bounding_rect_area = w * h
    return bounding_rect_area

def is_table(contour, image_shape):
    _, _, w, h = cv2.boundingRect(contour)
    ratio = float(w) / h
    return ratio < 2 and ratio > 1.5 and w > image_shape[1] * 0.5 and h > image_shape[0] * 0.5

def annotate_angle(image, end_point1, end_point2, vertex):
    # Draw lines between the centers
    cv2.line(image, end_point1, end_point2, (0, 255, 0), 2)
    cv2.line(image, end_point2, vertex, (0, 255, 0), 2)
    cv2.line(image, vertex, end_point1, (0, 255, 0), 2)

    # Calculate vectors
    v1 = np.array(vertex) - np.array(end_point1)
    v2 = np.array(end_point2) - np.array(vertex)

    # Calculate the angle in radians and convert to degrees
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Clip for numerical stability
    angle_deg = 180 - np.degrees(angle)

    # Annotate the angle near the yellow ball center
    angle_position = (vertex[0] + 50, vertex[1] - 10)  # Adjust position as needed
    cv2.putText(
        image,
        f"{angle_deg:.1f}",
        angle_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
    )

    return angle_deg

def resize_image(image):
    # resize the image to width = 1200, height = auto
    return cv2.resize(image, (1500, int(1500/image.shape[1]*image.shape[0])))

def detect_table(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 1)

    # Step 0: Compute dynamic thresholds
    median_intensity = np.median(blurred_img)
    lower_threshold = int(max(0, 0.1 * median_intensity))
    upper_threshold = int(min(255, 1.1 * median_intensity))
    # print(f"Dynamic Thresholds: {lower_threshold}, {upper_threshold}")

    # Step 1: Detect the table (using edge detection and contour finding)
    edges = cv2.Canny(blurred_img, lower_threshold, upper_threshold)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # print("Number of contours:", len(contours))

    filtered_contours = [contour for contour in contours if is_table(contour, image.shape)]
    # print("Number of filtered contours:", len(filtered_contours))

    table_contour = min(filtered_contours, key=area)
    return table_contour

# Detect circles for each mask using Hough Circle Transform
def detect_circles(mask, table_rect):
    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=5,
        param2=10,
        minRadius=5,
        maxRadius=50,
    )
    # print(f"Number of {color_name} balls detected:", 0 if circles is None else len(circles[0]))
    balls = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if table_rect[0] <= x <= table_rect[0] + table_rect[2] and table_rect[1] <= y <= table_rect[1] + table_rect[3]:
                balls.append((x, y, r))
                
    return balls

def detect_circle(mask):
    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=10,
        param1=5,
        param2=10,
        minRadius=5,
        maxRadius=50,
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
                
    return circles[0]

def annotate_ball(image, ball, color_name):
    # Draw detected circles on the original image
    x, y, r = ball
    cv2.circle(image, (x, y), r, (0, 255, 0), 2)
    cv2.putText(
        image, color_name, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
    )

def detect_balls(image, table_rect):
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Define HSV ranges for the colors
  # Adjust these ranges as needed for your image
  # White ball
  lower_white = np.array([0, 0, 200])  # Low saturation, high brightness
  upper_white = np.array([180, 55, 255])

  # Yellow ball
  lower_yellow = np.array([20, 100, 100])
  upper_yellow = np.array([30, 255, 255])

  # Red ball
  lower_red1 = np.array([0, 120, 70])  # Red has two ranges in HSV
  upper_red1 = np.array([10, 255, 255])
  lower_red2 = np.array([170, 120, 70])
  upper_red2 = np.array([180, 255, 255])

  # Blue table (to mask out the table)
  lower_blue = np.array([100, 150, 50])
  upper_blue = np.array([140, 255, 255])

  # Create masks for each ball
  mask_white = cv2.inRange(hsv, lower_white, upper_white)
  mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
  mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
  mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
  mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # Combine the two red ranges
  mask_table = cv2.inRange(hsv, lower_blue, upper_blue)

  # Remove table from the ball masks
  mask_white = cv2.subtract(mask_white, mask_table)
  mask_yellow = cv2.subtract(mask_yellow, mask_table)
  mask_red = cv2.subtract(mask_red, mask_table)

  # Clean up masks with morphological operations
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
  mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
  mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

  # Display the masks
  # show_image("White Mask", mask_white)
  # show_image("Yellow Mask", mask_yellow)
  # show_image("Red Mask", mask_red)

  # Detect and annotate each ball
  white_ball = detect_circles(mask_white, table_rect)[0]
  yellow_ball = detect_circles(mask_yellow, table_rect)[0]
  red_ball = detect_circles(mask_red, table_rect)[0]

  return white_ball, yellow_ball, red_ball

def get_ball_near_rect(image, ball):
    x, y, _ = ball
    d = 100
    return image[y-d//2:y+d//2, x-d//2:x+d//2]

def convert_ball_to_absolute_coords(ball, old_ball):
    x, y, r = ball
    old_x, old_y, _ = old_ball
    d = 100
    return x + old_x - d//2, y + old_y - d//2, r

def detect_balls_relative(image, old_white_ball, old_yellow_ball, old_red_ball):
  # cut a region of interest around the white ball
  hsv_white = cv2.cvtColor(get_ball_near_rect(image, old_white_ball), cv2.COLOR_BGR2HSV)
  hsv_yellow = cv2.cvtColor(get_ball_near_rect(image, old_yellow_ball), cv2.COLOR_BGR2HSV)
  hsv_red = cv2.cvtColor(get_ball_near_rect(image, old_red_ball), cv2.COLOR_BGR2HSV)

  # Define HSV ranges for the colors
  # Adjust these ranges as needed for your image
  # White ball
  lower_white = np.array([0, 0, 200])  # Low saturation, high brightness
  upper_white = np.array([180, 55, 255])

  # Yellow ball
  lower_yellow = np.array([20, 100, 100])
  upper_yellow = np.array([30, 255, 255])

  # Red ball
  lower_red1 = np.array([0, 120, 70])  # Red has two ranges in HSV
  upper_red1 = np.array([10, 255, 255])
  lower_red2 = np.array([170, 120, 70])
  upper_red2 = np.array([180, 255, 255])

  # Blue table (to mask out the table)
  lower_blue = np.array([100, 150, 50])
  upper_blue = np.array([140, 255, 255])

  # Create masks for each ball
  mask_white = cv2.inRange(hsv_white, lower_white, upper_white)
  mask_yellow = cv2.inRange(hsv_yellow, lower_yellow, upper_yellow)
  mask_red1 = cv2.inRange(hsv_red, lower_red1, upper_red1)
  mask_red2 = cv2.inRange(hsv_red, lower_red2, upper_red2)
  mask_red = cv2.bitwise_or(mask_red1, mask_red2)  # Combine the two red ranges
  mask_table_for_white = cv2.inRange(hsv_white, lower_blue, upper_blue)
  mask_table_for_yellow = cv2.inRange(hsv_yellow, lower_blue, upper_blue)
  mask_table_for_red = cv2.inRange(hsv_red, lower_blue, upper_blue)

  # Remove table from the ball masks
  mask_white = cv2.subtract(mask_white, mask_table_for_white)
  mask_yellow = cv2.subtract(mask_yellow, mask_table_for_yellow)
  mask_red = cv2.subtract(mask_red, mask_table_for_red)

  # Clean up masks with morphological operations
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_CLOSE, kernel)
  mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
  mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)

  # Display the masks
  # show_image("White Mask", mask_white)
  # show_image("Yellow Mask", mask_yellow)
  # show_image("Red Mask", mask_red)

  # Detect and annotate each ball
  white_ball = detect_circle(mask_white)
  yellow_ball = detect_circle(mask_yellow)
  red_ball = detect_circle(mask_red)
  
  white_ball = convert_ball_to_absolute_coords(white_ball, old_white_ball)
  yellow_ball = convert_ball_to_absolute_coords(yellow_ball, old_yellow_ball)
  red_ball = convert_ball_to_absolute_coords(red_ball, old_red_ball)

  return white_ball, yellow_ball, red_ball

def annotate_table(image, table_contour):
    # Draw the table contour on the original image
    cv2.drawContours(image, [table_contour], -1, (0, 255, 0), 2)

    # Annotate the table
    table_position = (table_contour[0][0][0] + 10, table_contour[0][0][1] + 10)
    cv2.putText(
        image, "Table", table_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
    )

def annotate_balls_angles(image, white_ball, yellow_ball, red_ball):
    white_center = (white_ball[0], white_ball[1])
    yellow_center = (yellow_ball[0], yellow_ball[1])
    red_center = (red_ball[0], red_ball[1])
    annotate_angle(image, white_center, yellow_center, red_center)
    annotate_angle(image, yellow_center, red_center, white_center)
    annotate_angle(image, red_center, white_center, yellow_center)

    return image

def detect_and_annotate(image):
    result_image = resize_image(image)

    table_contour = detect_table(result_image)
    # annotate_table(result_image, table_contour)

    white_ball, yellow_ball, red_ball = detect_balls(result_image, cv2.boundingRect(table_contour))
    print(f"white_ball: {white_ball}")
    # annotate_ball(result_image, white_ball, "White")
    # annotate_ball(result_image, yellow_ball, "Yellow")
    # annotate_ball(result_image, red_ball, "Red")
    annotate_balls_angles(result_image, white_ball, yellow_ball, red_ball)

    return result_image

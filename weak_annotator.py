import cv2
import numpy as np
light_blue = (173, 216, 230)
pink = (255, 153, 153)
image_file = "C:\\Users\\behnam\\py\\Capturelv.png"
center = (0, 0)
radius = 50
fixed_radius = True
drawing = False   
# Variables to store the center and radius of the circle

def fill_circle(index,img,center, radius):
    if index ==0:
        radius1= radius
        center1 = center
        cv2.circle(img, center, radius, (0, 255, 0), -1)
        # cv2.imshow("Final Image", img)
        
    if index ==1:
        color=pink
        print(img.shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, (255, 255, 255), -1)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        bg = np.full(img.shape, color, dtype=np.uint8)
        bg = cv2.bitwise_and(bg, bg, mask=cv2.bitwise_not(mask))
        result= cv2.add(masked_img, bg)
        cv2.namedWindow("Image2", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image2", 800, 800)
        cv2.imshow("Image2", result)
        
        cv2.waitKey(0)
        # cv2.destroyAllWindows()        
def draw_circle(event, x, y, flags, param):
    global center, fixed_radius, drawing, radius
    scaling_factor = 1/4
    x = int(x * scaling_factor)
    y = int(y * scaling_factor)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        center = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if fixed_radius:
            center = (x, y)
        else:
            center = (x, y)
            radius = param[0]
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            radius += 3
        else:
            radius = max(0, radius - 3)
    return center, radius

# Load the image

# Create a window
cv2.namedWindow("Image")
cv2.resizeWindow("Image", 600, 600)


def annotation(img):
    # img = cv2.imread(img_file)
    centers = []
    radiuses = []
    exit_flag = False
    global center, fixed_radius, drawing, radius
    for i in range(2):
        while True:
            flag_next = False
            img_copy = img.copy()
            if drawing:
                cv2.circle(img_copy, center, radius, (0, 255, 0), 1)
            cv2.namedWindow('hi', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('hi', 800, 800)    
            cv2.imshow('hi', img_copy)

            # Wait for user input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("i"):
                radius =  radius+1
            elif key == ord("r"):
                radius =  max(0, radius - 1)
            elif key == ord("f"):
                fixed_radius = not fixed_radius
            elif key == ord("n"): #next image
                exit_flag = True
                flag_next = True
                break
        if (exit_flag):
            break

        # Draw the final circle on the original image
        centers.append(center)
        radiuses.append(radius)
        fill_circle(i,img,center, radius)
        cv2.waitKey(0)
        # Destroy the window and release resources
    # cv2.destroyAllWindows()
    return radiuses, centers, flag_next
    
       

    # Return the final circle
if __name__== '__main__':
    
    cv2.setMouseCallback("Image", draw_circle)
    radiuses, centers = annotation(img)
    print(radiuses)
    print(centers)

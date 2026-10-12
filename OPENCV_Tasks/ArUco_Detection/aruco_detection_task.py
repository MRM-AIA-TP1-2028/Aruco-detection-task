#The object in focus is an arcuno marker which is a perfectly flat 2D shape which acts like a mini QR code for the webcam.

import numpy as np # Numpy is a library for math (mainly matrices or linear algebra) and arrays.
import cv2 # Its a module to load open CV which gives access to camera & vision tools.


# Its a dictionary of markers with varying sizes. nxn represents the grid markers which describes the size of the marker. 100 just reprsents the no. of unique digits from  (0-99)IDs.
DICTS={
    "4x4": cv2.aruco.DICT_4X4_100,
    "5x5": cv2.aruco.DICT_5X5_100,
    "6x6": cv2.aruco.DICT_6X6_100,
}

# Note the units are in meters
MARKER_SIZE = 0.10
h=MARKER_SIZE/2

# Visualize it as a matrix and label the signs according to quadrants in a flat 2D cartesian plane cuz an aruco marker is a 2D, perfectly flat square.

obj_points=np.array([
    [-h,h,0], #2nd quadrant (top-left corner )
    [h,h,0], #1st quadrant (top-right corner)
    [h,-h,0], #4th quadrant (bottom-left corner)
    [-h,-h,0] #3rd quadrant (bottom-right corner)
],dtype=np.float32
)

params=cv2.aruco.DetectorParameters() # this is the default detection settings
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX # Used for subpixel corner accuracy to significantly improve the position and rotation of the object.

detectors={ #Dictionary created for detectors.
    name: cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(d), params) #one detector per family
    for name , d in DICTS.items()  # creating even more detectors for other families too
}

cap=cv2.VideoCapture(0) # Opens the default camera which for my laptop is the webcam.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #(Description of these 2 lines) setting the dimensions of the camera to 1280x720 though if not supported, it would kindly default to the closest resolution to this.
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened(): # Immediately exits the program if the cam fails to open.
    raise SystemExit("Could not open the cam")

ok,frame=cap.read() # Reads the initial frame from the camera and if the frame isnt ok then the program will immediately exit.
if not ok:
    raise SystemExit("Cam is not legible")

#Gets the real resolutioin of the camera and since i want only the width and height, i can just code in :2 to pick the first 2 values.
f_h,f_w=frame.shape[:2]
cam_matrix=np.array([
    [f_w, 0, f_w/2],
    [0, f_w, f_h/2],
    [0, 0, 1] # This row assumes that the sensor's pixel grid is a perfect rectangle.
], dtype=np.float64 # Stores the numbers in 64 bit decimals.
)

dist_coeffs=np.zeros((5,1)) #ignoring distortion

while True:

    ok,frame=cap.read()
    if not ok:
        break

    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for name,detector in detectors.items():
        corners, ids, _=detector.detectMarkers(gray) # Looks for markers of this family  in the gray scale. Colors will only slow down the process so a simple grayscale is used.
        if ids is None: # When nothing is detected, skips to the next detector.
            continue

        cv2.aruco.drawDetectedMarkers(frame,corners,ids) # Draws the detected markers on the frame with their IDs.

        for c,marker_id in zip(corners,ids.flatten()): # Iterate through each marker with its respective ID.
            success,rvec,tvec=cv2.solvePnP( # works out 3D position and rotation of the marker. Here the rvec and tvec are the rotation and translation vectors respectively.
                obj_points, #real 3D corners of a marker.
                c.reshape(4,2), # Reshapes any detected2D pixels into a 4 rows of (x,y).
                cam_matrix, #focal length and center.
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE # Using the IPPE_SQUARE method to solve for the pose of flat square markers.
            )
            if not success: # two lines to skip to the next marker if it couldnt be solved.
                continue
            cv2.drawFrameAxes(frame,cam_matrix,dist_coeffs,rvec,tvec,MARKER_SIZE*0.5) # Draws those red green blue lines which are the axes of the marker in 3D space.
            
            R, _=cv2.Rodrigues(rvec) # Converts the rotation vetctor into a 3x3 rotation matrix.
            angles=cv2.RQDecomp3x3(R)[0] # Breaks the matrix into Euler angles degrees (X,Y,Z).
            x,y,z=tvec.ravel() # flattens the position column and unpacks its x,y,z coordinates.

            # This part is just a bunch of formatting and 7 decimals are there for even better accuracy. The angles are in degrees and the position is in meters.
            descreptive_lines=[
                f"{name} & ID: {marker_id}",
                f"Position: x={x:.7f}meters, y={y:.7f}meters, z={z:.7f}meters",
                f"Degree of rotation: {angles[0]:.3f}°, {angles[1]:.3f}°, {angles[2]:.3f}°"
            ]
            ox, oy = c[0][0].astype(int) # picks the first corner of the marker and turns into whole numbers.
            for k, text in enumerate(descreptive_lines): # Loops through 3 lines with enumerate to get the index along with my text.
                ty=max(20,oy-10-22*(len(descreptive_lines)-1-k)) # Calculates the y position for each line and max is there to ensure that ty doesnt fly beyond the screen.
                cv2.putText(
                    frame, text, (ox, ty), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2 # Settinng a simple font for legibility.
                    )

    cv2.imshow("ArUco Pose Estimate", frame) # Displaying the frame with the detected markers and their poses
    if cv2.waitKey(1) & 0xFF in  (ord('e'),27): # Deciding the delay for a key stroke to be registered. 27 is the ESC key and 'e' is for exit.
        break

cap.release() # Releasing the camera so it can be available for other apps.
cv2.destroyAllWindows() # Desroying all the windows created by OpenCV to free up memory.
# Approach: 

"""

* Both the objects are identified by color so i used HSV instead of BGR as HSV seperates color and intensity which is just there to find the blue object in either
brighter or darker rooms/places.

*Since i need the 2 objects to stay seperate from each other, each objects gets its own mask and contour pipeline. Thus, the blue object and the tennis ball
can never be mistaken for each other.

*The blue objects shape is determined by using area, bounding rectangle and centroid. Even the ball has to undergo a further circularity test.

"""

# Method used to solve the problem statement:

"""

*Gauss blue to reduce sensor noise.

*cv2.inRange for the blue range and the yellow-green range ---> 2 unique masks.

*Morphological operations to clean the masks.(opening removes specks while closing fills in holes)

*cv2.findContours to find the contours of the objects in the masks.

*Blue: keep contours above a minimum area & computes area with contourArea, boundingRect and centre with moments.

*Ball: keep contours above a minimum area & with circularity above the minimum threshold, takes the largest, gets the centre and radius from minEnclosingCircle.

*A simple nearest-centroid tracker gives each object a persistent ID & a short history of centres, drawn as the tracking line.

*If the ball is detected, draw a line from each blue centre to the ball centre and print the offset and distance.

"""
# Other possible methods:

"""

$ Kalman filter on top the tracker can be another nice method to smoothen tracking and predictions when something suddenly blocks the object from the camera's view for a 
moment or two.

$ Open CV trackers like CSRT or KCF are also a good way to track selected objects frame to frame.

$ Adaptive HSV thresholds handles different lighting muhc more efficiently than me typing a different threshold each and every single time when needed.

"""


import math
import argparse 
import collections as collect
import cv2
import numpy as np

low_blue, high_blue = np.array([100,120,60]), np.array([130,255,255])
low_ball, high_ball = np.array([25,70,80]), np.array([45,255,255])

min_blue_area=800
min_ball_area=300
min_circularity=0.70
trail_len=40
max_match_distance=80
max_missed=10

def clean(m):

    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask=cv2.morphologyEx(m,cv2.MORPH_OPEN,k)

    return cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,iterations=2)

def centroid(c):

    moment= cv2.moments(c)

    if not moment['m00']:
        return None

    return int(( moment["m10"] / moment['m00'])), int(( moment['m01'] / moment['m00']))

class CentroidTracker9000:

    def __init__(self):
        self.next_id=0
        self.last={}
        self.trails={}
        self.missed={}

    def update(self,centers):
        ids=[]
        unused=set(self.last)

        for c in centers:
            good, good_d = None, max_match_distance

            for oid in unused:
                d= math.dist(c, self.last[oid])
                if d < good_d:
                    good,good_d = oid,d

            if good is None:
                good=self.next_id
                self.next_id += 1
                self.trails[good]=collect.deque(maxlen=trail_len)\

            else:
                unused.discard(good)

            self.last[good]=c
            self.missed[good]=0
            self.trails[good].append(c)
            ids.append(good)

        for oid in unused:
            
            self.missed[oid] += 1
            if self.missed[oid] > max_missed:

                for d in (self.last,self.trails,self.missed):
                    d.pop(oid,None)
                
        return ids


def detect_blue(hsv):

    mask=clean(cv2.inRange(hsv,low_blue,high_blue))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    objs=[]

    for c in cnts:
        area=cv2.contourArea(c)
        if area < min_blue_area:
            continue

        ctr= centroid(c)
        if ctr:
            objs.append({"area": area, "bbox": cv2.boundingRect(c), "center": ctr})
    
    return objs,mask

def detect_ball(hsv):
    
    mask=clean(cv2.inRange(hsv,low_ball,high_ball))
    cnts, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    good=None

    for c in cnts:
        area= cv2.contourArea(c)
        perimeter=cv2.arcLength(c,True)
        
        if area < min_ball_area or perimeter ==0 :
            continue

        circularity= (4*math.pi*area)/(perimeter**2)

        if circularity < min_circularity:
            continue

        if good is None or area > good['area']:
            (x,y), r = cv2.minEnclosingCircle(c)
            good={"center": (int(x),int(y)), "radius": int(r), "area": area }

    return good, mask

def put(img,txt,org, color=(255,255,255), scale =0.5):
    cv2.putText(img,txt,org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0),3, cv2.LINE_AA)
    cv2.putText(img,txt,org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,1, cv2.LINE_AA)

def process(frame, tracker):

    blurred=cv2.GaussianBlur(frame,(7,7),0)
    hsv=cv2.cvtColor(blurred,cv2.COLOR_BGR2HSV)

    blue_objs, blue_mask= detect_blue(hsv)
    ball_obj, ball_mask= detect_ball(hsv)
    ids=tracker.update([o['center'] for o in blue_objs])

    out=frame.copy()

    if ball_obj:
        cx, cy = ball_obj['center']
        cv2.circle(out, (cx,cy), ball_obj['radius'], (0,255,255), 2)
        cv2.circle(out, (cx,cy), 4, (0,0,255), -1)
        put(
            out, 
            f" Ball Coordinates = ({cx},{cy}) R = {ball_obj['radius']}px A={ball_obj['area']:.0f}px^2",
            (min(max(cx - ball_obj["radius"], 5), out.shape[1] - 290), max( cy - ball_obj['radius'] - 10,15)),
            (0,255,255)
            )

    for obj,oid in zip(blue_objs, ids):
        x,y,w,h = obj['bbox']
        cx, cy = obj['center']
        cv2.rectangle(out, (x,y), (x + w, y + h), (255,0,0), 2)
        cv2.circle(out, (cx,cy), 4 , (0,0,255), -1)

        trail= list(tracker.trails[oid])
        for p1,p2 in zip(trail, trail[1:]):
            cv2.line(out, p1, p2, (0,255,0), 2)

        put(out, f"Blue#{oid} A= {obj['area']:.0f}px^2", (x, max(y - 8, 15)), (255,200,0))

        if ball_obj:
            bx, by = ball_obj['center']
            dx,dy = bx - cx, by - cy
            cv2.line(out, (cx,cy), (bx,by), (255,255,255), 1, cv2.LINE_AA)
            horiz="right" if dx > 0 else "left"
            vert= "below" if dy > 0 else "above"
            put(
                out,
                f"Ball: {abs(dx)}px {horiz}, {abs(dy)}px {vert} (d={math.hypot(dx,dy):.0f}px)",
                (x, y + h + 18), (255,255,255)
            )

    return out, blue_mask, ball_mask

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type= int , default = 0)
    ap.add_argument("--image", type= str , default = None)

    args= ap.parse_args()
    tracker= CentroidTracker9000()

    if args.image:
        frame= cv2.imread(args.image)
        out, _ , _ = process(frame, tracker)
        cv2.imshow("Result", out)
        cv2.waitKey(0)
        return

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise SystemExit("Error: Could not open camera.")
    show_mask=False

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        out, blue_mask, ball_mask = process(frame, tracker)
        cv2.imshow("BLUE OBJECT + TENNIS BALL TRACKER", out)

        if show_mask:
            cv2.imshow("Blue Mask", blue_mask)
            cv2.imshow("Ball Mask", ball_mask)
        key= cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("m"):
            show_mask= not show_mask

            if not show_mask:
                cv2.destroyWindow("Blue Mask")
                cv2.destroyWindow("Ball Mask")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
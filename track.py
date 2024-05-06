from ultralytics import YOLO
import dxcam
import cv2
import pygetwindow as gw
import time
import os

def main():
    # video capture parameters
    application_name = "Destiny 2"
    displayWindow_name = "Vow Detection"
    target_fps = 30
    
    # screenshot variables
    screenshot_count = len(os.listdir("screenshots"))

    # loading model
    # model = YOLO('../runs/detect/train/weights/best.pt')
    model = YOLO('../runs/detect/train2/weights/best.pt')

    # capture postition of game window
    window = gw.getWindowsWithTitle(application_name)
    window = window[0]
    win_left, win_top, w, h = window.left, window.top, window.width, window.height

    border_pixels = 8
    title_pixels = 30
    x1, y1, x2, y2 = win_left + border_pixels, win_top + border_pixels + \
        title_pixels, win_left + w - border_pixels, win_top + h - border_pixels

    # create camera to capture game footage
    camera = dxcam.create(device_idx=0, output_idx=0,
                        region=(x1, y1, x2, y2), output_color="BGR")

    # create window to display predictions
    cv2.namedWindow(displayWindow_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(displayWindow_name, 1280, 720)  # 640, 360 : 1280, 720 : 2560, 1080
    cv2.setWindowProperty(displayWindow_name, cv2.WND_PROP_TOPMOST, 1)

    # framerate calculation variables
    start_time = time.time()
    frame_count = 0
    fps = 0

    camera.start(target_fps=target_fps)
    while camera.is_capturing:
        frame = camera.get_latest_frame()
        # results = model.predict(source=frame, conf=0.6)
        results = model.track(source=frame, conf=0.6, persist=False, tracker="bytetrack.yaml")
        image = results[0].plot(conf=False, line_width=3)

        frame_count += 1
        if frame_count == 60:
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            start_time = time.time()
            frame_count = 0

        cv2.putText(image, f"FPS: {fps:.0f}", (10, 40),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
        cv2.imshow(displayWindow_name, image)

        # Check for key press to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            camera.stop()
            
        elif key == ord('s'):
            screenshot_count += 1
            cv2.imwrite(f"screenshots/screenshot-{screenshot_count}.png", image)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
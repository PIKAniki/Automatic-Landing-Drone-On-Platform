#!/usr/bin/env python3
import roslibpy
import cv2
import cv2.aruco as aruco
import numpy as np
import base64
import time


class RoslibpyArucoProcessor:
    def __init__(self):
        self.client = roslibpy.Ros(host='172.32.0.98', port=9091)
        self.target_marker_id = 100
        self.marker_size = 0.2

        self.camera_matrix = None
        self.dist_coeffs = None
        self.fx = self.fy = 160.0
        self.cx = self.cy = 160.0

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        self.aruco_params = aruco.DetectorParameters_create()

        self.image_listener = None
        self.target_publisher = None

    def connect(self):
        self.client.run()
        print("Connected to ROS bridge")

        self.image_listener = roslibpy.Topic(
            self.client,
            '/main_camera/image_raw',
            'sensor_msgs/Image'
        )

        self.target_publisher = roslibpy.Topic(
            self.client,
            '/landing_target',
            'mavros_msgs/LandingTarget'
        )

        self.image_listener.subscribe(self.image_callback)
        self.target_publisher.advertise()
        print("Subscribed to /main_camera/image_raw, advertising /landing_target")

    def decode_image_data(self, img_msg):
        encoding = img_msg['encoding']
        height = img_msg['height']
        width = img_msg['width']
        data = img_msg['data']

        if encoding == 'bgr8' or encoding == 'rgb8':
            img_array = np.frombuffer(data, dtype=np.uint8)
            if len(img_array) == height * width * 3:
                img_array = img_array.reshape((height, width, 3))
                if encoding == 'rgb8':
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_array

        elif encoding == 'mono8':
            img_array = np.frombuffer(data, dtype=np.uint8)
            if len(img_array) == height * width:
                return img_array.reshape((height, width))

        return None

    def image_callback(self, img_msg):
        try:
            cv_image = self.decode_image_data(img_msg)
            if cv_image is None:
                return

            if len(cv_image.shape) == 3:
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv_image

            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                for i, marker_id in enumerate(ids):
                    if marker_id[0] == self.target_marker_id:
                        self.process_marker(corners[i], marker_id[0], img_msg['header']['stamp'])
                        break
        except Exception as e:
            print(f"Image processing error: {e}")

    def process_marker(self, corners, marker_id, timestamp):
        marker_corners = corners[0]
        center_x = np.mean(marker_corners[:, 0])
        center_y = np.mean(marker_corners[:, 1])

        if self.camera_matrix is None:
            self.fx = self.fy = 160.0
            self.cx = center_x
            self.cy = center_y

        angle_x = np.arctan2(center_x - self.cx, self.fx)
        angle_y = np.arctan2(center_y - self.cy, self.fy)

        min_x = np.min(marker_corners[:, 0])
        max_x = np.max(marker_corners[:, 0])
        min_y = np.min(marker_corners[:, 1])
        max_y = np.max(marker_corners[:, 1])
        size_x = np.arctan2(max_x - min_x, self.fx)
        size_y = np.arctan2(max_y - min_y, self.fy)

        half_size = self.marker_size / 2.0
        object_points = np.array([
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

        image_points = marker_corners.astype(np.float32)
        tvec = None

        if self.camera_matrix is not None and self.dist_coeffs is not None:
            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

        distance = 0.0
        if tvec is not None:
            distance = float(np.linalg.norm(tvec))

        landing_target_msg = {
            'header': {
                'stamp': timestamp,
                'frame_id': 'camera'
            },
            'target_num': int(marker_id),
            'frame': 8,
            'type': 2,
            'angle': [float(angle_x), float(angle_y)],
            'size': [float(size_x), float(size_y)],
            'distance': float(distance),
            'pose': {
                'position': {'x': 0.0, 'y': 0.0, 'z': 0.0},
                'orientation': {'x': 0.0, 'y': 0.0, 'z': 0.0, 'w': 1.0}
            }
        }

        if tvec is not None:
            landing_target_msg['pose']['position']['x'] = float(tvec[2][0])
            landing_target_msg['pose']['position']['y'] = float(tvec[0][0])
            landing_target_msg['pose']['position']['z'] = float(tvec[1][0])

        self.target_publisher.publish(roslibpy.Message(landing_target_msg))

        current_time = time.time()
        if not hasattr(self, 'last_log_time'):
            self.last_log_time = 0

        if current_time - self.last_log_time >= 1.0:
            print(f"Published marker {marker_id}")
            self.last_log_time = current_time

    def run(self):
        self.connect()

        try:
            while self.client.is_connected:
                time.sleep(0.01)
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            if self.image_listener:
                self.image_listener.unsubscribe()
            if self.target_publisher:
                self.target_publisher.unadvertise()
            self.client.terminate()


if __name__ == '__main__':
    processor = RoslibpyArucoProcessor()
    processor.run()
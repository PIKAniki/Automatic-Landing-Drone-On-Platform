#!/usr/bin/env python3
import sys
import os
sys.path.append('/root/ros_ws/ros_modules')
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from sensor_msgs.msg import Image
from mavros_msgs.msg import LandingTarget
from cv_bridge import CvBridge


class LuckfoxArucoProcessor:
    def __init__(self):
        rospy.init_node('luckfox_aruco_ros', anonymous=True)
        os.environ['ROS_MASTER_URI'] = 'http://192.168.38.130:11311'
        os.environ['ROS_IP'] = '172.32.0.70'

        self.bridge = CvBridge()
        self.target_marker_id = rospy.get_param('~marker_id', 100)
        self.marker_size = rospy.get_param('~marker_size', 0.2)

        self.camera_matrix = None
        self.dist_coeffs = None
        self.fx = self.fy = 160.0
        self.cx = self.cy = 160.0

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        self.aruco_params = aruco.DetectorParameters_create()

        self.image_sub = rospy.Subscriber('/main_camera/image_raw', Image, self.image_callback)
        self.target_pub = rospy.Publisher('/landing_target', LandingTarget, queue_size=10)

        rospy.loginfo("Luckfox ArUco ROS node started")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                for i, marker_id in enumerate(ids):
                    if marker_id[0] == self.target_marker_id:
                        self.process_marker(corners[i], marker_id[0], msg.header.stamp)
                        break
        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")

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

        landing_target = LandingTarget()
        landing_target.header.stamp = timestamp
        landing_target.header.frame_id = "camera"
        landing_target.target_num = marker_id
        landing_target.frame = 8
        landing_target.type = 2
        landing_target.angle = [float(angle_x), float(angle_y)]
        landing_target.size = [float(size_x), float(size_y)]
        landing_target.distance = distance
        landing_target.pose.orientation.w = 1.0

        if tvec is not None:
            landing_target.pose.position.x = float(tvec[2][0])
            landing_target.pose.position.y = float(tvec[0][0])
            landing_target.pose.position.z = float(tvec[1][0])

        self.target_pub.publish(landing_target)
        rospy.loginfo_throttle(1.0, f"Published marker {marker_id}")


if __name__ == '__main__':
    processor = LuckfoxArucoProcessor()
    rospy.spin()
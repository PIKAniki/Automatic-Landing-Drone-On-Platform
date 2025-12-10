#!/usr/bin/env python3
""" Программа для запуска алгоритма без участия платы Luckfox. Все действия происходят на одном хосте"""
import rospy
from clover import srv
from std_srvs.srv import Trigger
from sensor_msgs.msg import Image, Range, CameraInfo
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from cv_bridge import CvBridge
from mavros_msgs.msg import LandingTarget, HomePosition
import cv2
import cv2.aruco as aruco
import numpy as np


class ArucoLanding:
    def __init__(self):
        rospy.init_node('aruco_landing')

        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.land = rospy.ServiceProxy('land', Trigger)

        self.mode = rospy.get_param('~mode', 'custom')
        self.target_marker_id = rospy.get_param('~marker_id', 100)
        self.marker_size = rospy.get_param('~marker_size', 0.2)

        rospy.loginfo("ArUco Landing Node")
        rospy.loginfo("Mode: %s", self.mode)
        rospy.loginfo("Target Marker ID: %d", self.target_marker_id)

        self.bridge = CvBridge()
        self.marker_detected = False
        self.range_distance = None
        self.marker_corners = None
        self.landed = False

        self.camera_matrix = None
        self.dist_coeffs = None
        self.image_width = 320
        self.image_height = 240
        self.fx = 160.0
        self.fy = 160.0
        self.cx = 160.0
        self.cy = 120.0

        self.current_pose = None
        self.home_position = None

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        self.aruco_params = aruco.DetectorParameters_create()

        self.image_sub = rospy.Subscriber('/main_camera/image_raw', Image, self.image_callback)
        self.range_sub = rospy.Subscriber('/rangefinder/range', Range, self.range_callback)
        self.camera_info_sub = rospy.Subscriber('/main_camera/camera_info', CameraInfo, self.camera_info_callback,
                                                queue_size=1)

        if self.mode == 'px4_precland':
            self.landing_target_pub = rospy.Publisher('/mavros/landing_target/raw', LandingTarget, queue_size=10)
            self.local_pose_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.local_pose_callback)
            self.home_pos_sub = rospy.Subscriber('/mavros/home_position/home', HomePosition,
                                                 self.home_position_callback)
            rospy.loginfo("PX4 Precision Landing mode")
            rospy.loginfo("Publishing to /mavros/landing_target/raw")

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.K).reshape(3, 3)
            self.dist_coeffs = np.array(msg.D) if len(msg.D) > 0 else np.zeros(5)
            self.image_width = msg.width
            self.image_height = msg.height
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
            rospy.loginfo("Camera initialized: %dx%d", self.image_width, self.image_height)

    def range_callback(self, msg):
        self.range_distance = msg.range

    def local_pose_callback(self, msg):
        self.current_pose = msg.pose

    def home_position_callback(self, msg):
        self.home_position = msg

    def estimate_marker_position_camera_frame(self, corners):
        if self.camera_matrix is None:
            return None

        half_size = self.marker_size / 2.0
        object_points = np.array([
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

        image_points = corners[0].astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if success:
            return tvec.flatten()
        return None

    def publish_landing_target(self, corners, ids, timestamp):
        if self.camera_matrix is None or self.current_pose is None:
            rospy.logwarn_throttle(5.0, "Waiting for camera_info and local_pose")
            return

        for i, marker_id in enumerate(ids):
            if marker_id[0] == self.target_marker_id:
                marker_corners = corners[i][0]
                center_x = np.mean(marker_corners[:, 0])
                center_y = np.mean(marker_corners[:, 1])

                pixel_error_x = center_x - self.cx
                pixel_error_y = center_y - self.cy

                angle_x = np.arctan2(pixel_error_x, self.fx)
                angle_y = np.arctan2(pixel_error_y, self.fy)

                min_x = np.min(marker_corners[:, 0])
                max_x = np.max(marker_corners[:, 0])
                min_y = np.min(marker_corners[:, 1])
                max_y = np.max(marker_corners[:, 1])
                size_x = np.arctan2(max_x - min_x, self.fx)
                size_y = np.arctan2(max_y - min_y, self.fy)

                if self.range_distance is not None and self.range_distance > 0:
                    distance = float(self.range_distance)

                    offset_x_camera = distance * np.tan(angle_x)
                    offset_y_camera = distance * np.tan(angle_y)
                else:
                    distance = 0.0
                    offset_x_camera = 0.0
                    offset_y_camera = 0.0

                drone_x = self.current_pose.position.x
                drone_y = self.current_pose.position.y
                drone_z = self.current_pose.position.z

                q = self.current_pose.orientation
                yaw = np.arctan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

                cos_yaw = np.cos(yaw)
                sin_yaw = np.sin(yaw)

                offset_north = offset_y_camera * cos_yaw - offset_x_camera * sin_yaw
                offset_east = offset_y_camera * sin_yaw + offset_x_camera * cos_yaw

                marker_x_ned = drone_x + offset_north
                marker_y_ned = drone_y + offset_east
                marker_z_ned = 0.0

                landing_target = LandingTarget()
                landing_target.header.stamp = timestamp
                landing_target.header.frame_id = "map"

                landing_target.target_num = marker_id[0]
                landing_target.frame = 2
                landing_target.type = 2

                landing_target.angle = [float(angle_x), float(angle_y)]
                landing_target.size = [float(size_x), float(size_y)]
                landing_target.distance = distance

                landing_target.pose.position.x = marker_x_ned
                landing_target.pose.position.y = marker_y_ned
                landing_target.pose.position.z = marker_z_ned
                landing_target.pose.orientation.w = 1.0

                self.landing_target_pub.publish(landing_target)

                rospy.loginfo_throttle(1.0,
                                       "LT | px_err:[%.0f,%.0f] ang:[%.1f,%.1f]deg | d:%.2fm | drone:[%.2f,%.2f,%.2f] mark:[%.2f,%.2f]",
                                       pixel_error_x, pixel_error_y,
                                       np.degrees(angle_x), np.degrees(angle_y), distance,
                                       drone_x, drone_y, -drone_z,
                                       marker_x_ned, marker_y_ned)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            corners, ids, rejected = aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

            self.marker_detected = False

            if ids is not None:
                aruco.drawDetectedMarkers(cv_image, corners, ids)

                for i, marker_id in enumerate(ids):
                    cv2.putText(cv_image, "ID:%d" % marker_id[0],
                                (int(corners[i][0][0][0]), int(corners[i][0][0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if marker_id[0] == self.target_marker_id:
                        self.marker_detected = True
                        self.marker_corners = corners[i]

                        center = corners[i][0].mean(axis=0).astype(int)
                        cv2.circle(cv_image, tuple(center), 5, (0, 0, 255), -1)
                        cv2.putText(cv_image, "TARGET",
                                    (center[0] + 10, center[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if self.mode == 'px4_precland':
                    self.publish_landing_target(corners, ids, msg.header.stamp)

            y_offset = 30
            cv2.putText(cv_image, "Mode: %s" % self.mode, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 30

            if self.range_distance:
                cv2.putText(cv_image, "Range: %.2fm" % self.range_distance, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(cv_image, "Range: N/A", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30

            marker_text = "Marker %d: FOUND" % self.target_marker_id if self.marker_detected else "Marker %d: LOST" % self.target_marker_id
            color = (0, 255, 0) if self.marker_detected else (0, 0, 255)
            cv2.putText(cv_image, marker_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow("ArUco Detection", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            rospy.logerr("Image callback error: %s", str(e))

    def simple_track_and_land(self):
        rospy.loginfo("Starting custom tracking and landing")
        rate = rospy.Rate(10)
        land_count = 0

        while not rospy.is_shutdown():
            if self.landed:
                rate.sleep()
                continue

            if not self.marker_detected:
                rospy.loginfo_throttle(2.0, "Marker not visible, holding position")
                if self.range_distance and self.range_distance < 2:
                    self.navigate(x=0, y=0, z=0.1, speed=0.2, frame_id='body', auto_arm=True)
                else:
                    self.navigate(x=0, y=0, z=0, speed=0.2, frame_id='body', auto_arm=True)
                land_count = 0
                rate.sleep()
                continue

            if self.range_distance is not None and self.range_distance < 0.25 and self.marker_detected:
                land_count += 1
                if land_count > 5:
                    rospy.loginfo("Landing threshold reached at %.2fm", self.range_distance)
                    self.landed = True
                    self.land()
                    rospy.sleep(5)
                    return True
            else:
                land_count = 0

            corners = self.marker_corners[0]
            center_x = np.mean(corners[:, 0])
            center_y = np.mean(corners[:, 1])

            error_x = center_x - (self.image_width / 2)
            error_y = center_y - (self.image_height / 2)

            correction_x = -error_y * 0.004
            correction_y = -error_x * 0.004

            if abs(error_x) < 20 and abs(error_y) < 20:
                descent_speed = 0.15
            else:
                descent_speed = 0.1

            correction_x = np.clip(correction_x, -0.35, 0.35)
            correction_y = np.clip(correction_y, -0.35, 0.35)

            rospy.loginfo_throttle(1.0,
                                   "Tracking: err_x=%.1f err_y=%.1f descent=%.2f alt=%.2f",
                                   error_x, error_y, descent_speed, self.range_distance)

            self.navigate(
                x=correction_x,
                y=correction_y,
                z=-descent_speed,
                speed=0.45,
                frame_id='body',
                auto_arm=True
            )

            rate.sleep()

        return False

    def px4_precland_mode(self):
        rospy.loginfo("")
        rospy.loginfo("PX4 Precision Landing Mode")
        rospy.loginfo("Publishing landing_target messages")
        rospy.loginfo("")
        rospy.loginfo("Control steps:")
        rospy.loginfo("1. rosservice call /mavros/cmd/arming \"value: true\"")
        rospy.loginfo("2. rosservice call /mavros/cmd/takeoff \"{altitude: 2.0}\"")
        rospy.loginfo("3. Wait for marker to be visible")
        rospy.loginfo("4. rosservice call /mavros/set_mode \"custom_mode: 'AUTO.PRECLAND'\"")
        rospy.loginfo("")

        rate = rospy.Rate(10)
        last_status = False

        while not rospy.is_shutdown():
            if self.marker_detected != last_status:
                if self.marker_detected:
                    rospy.loginfo("Marker %d VISIBLE - safe to activate PRECLAND", self.target_marker_id)
                else:
                    rospy.logwarn("Marker %d LOST", self.target_marker_id)
                last_status = self.marker_detected
            rate.sleep()

    def run(self):
        rospy.loginfo("ArUco Landing System starting")
        rospy.sleep(2)

        if self.mode == 'custom':
            rospy.loginfo("Custom mode: autonomous landing")
            rospy.loginfo("Taking off to 2m")
            self.navigate(x=0, y=0, z=2, speed=0.5, frame_id='body', auto_arm=True)
            rospy.sleep(5)

            rospy.loginfo("Starting tracking")
            if self.simple_track_and_land():
                rospy.loginfo("Landing successful")
            else:
                rospy.logwarn("Landing failed, emergency land")
                self.land()

            rospy.sleep(3)
            rospy.loginfo("Mission complete")

        elif self.mode == 'px4_precland':
            self.px4_precland_mode()

        else:
            rospy.logerr("Unknown mode: %s", self.mode)


if __name__ == '__main__':
    try:
        mission = ArucoLanding()
        mission.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted")
    except Exception as e:
        rospy.logerr("Fatal error: %s", str(e))

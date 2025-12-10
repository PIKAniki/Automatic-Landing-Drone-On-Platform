#!/usr/bin/env python3
import rospy
from clover import srv
from std_srvs.srv import Trigger
from sensor_msgs.msg import Range, CameraInfo
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import LandingTarget, HomePosition


class ArucoBridgeROS:
    def __init__(self):
        rospy.init_node('aruco_bridge_ros')

        self.get_telemetry = rospy.ServiceProxy('get_telemetry', srv.GetTelemetry)
        self.navigate = rospy.ServiceProxy('navigate', srv.Navigate)
        self.land = rospy.ServiceProxy('land', Trigger)

        self.mode = rospy.get_param('~mode', 'custom')
        self.target_marker_id = rospy.get_param('~marker_id', 100)

        self.range_distance = None
        self.current_pose = None
        self.home_position = None

        self.luckfox_sub = rospy.Subscriber('/landing_target', LandingTarget, self.luckfox_callback)
        self.range_sub = rospy.Subscriber('/rangefinder/range', Range, self.range_callback)
        self.camera_info_sub = rospy.Subscriber('/main_camera/camera_info', CameraInfo, self.camera_info_callback,
                                                queue_size=1)

        if self.mode == 'px4_precland':
            self.px4_pub = rospy.Publisher('/mavros/landing_target/raw', LandingTarget, queue_size=10)
            self.local_pose_sub = rospy.Subscriber('/mavros/local_position/pose', PoseStamped, self.local_pose_callback)
            self.home_pos_sub = rospy.Subscriber('/mavros/home_position/home', HomePosition,
                                                 self.home_position_callback)
            rospy.loginfo("PX4 Precision Landing mode - bridging to PX4")

        rospy.loginfo("Aruco Bridge ROS node started")

    def camera_info_callback(self, msg):
        pass

    def range_callback(self, msg):
        self.range_distance = msg.range

    def local_pose_callback(self, msg):
        self.current_pose = msg.pose

    def home_position_callback(self, msg):
        self.home_position = msg

    def luckfox_callback(self, luckfox_msg):
        if self.mode == 'custom':
            self.custom_logic(luckfox_msg)
        elif self.mode == 'px4_precland' and self.current_pose and self.range_distance:
            self.publish_to_px4(luckfox_msg)

    def publish_to_px4(self, luckfox_msg):
        px4_msg = LandingTarget()
        px4_msg.header.stamp = rospy.Time.now()
        px4_msg.header.frame_id = "local_origin"
        px4_msg.target_num = luckfox_msg.target_num
        px4_msg.frame = 1
        px4_msg.type = luckfox_msg.type

        angle_x = luckfox_msg.angle[0] if len(luckfox_msg.angle) > 0 else 0.0
        angle_y = luckfox_msg.angle[1] if len(luckfox_msg.angle) > 1 else 0.0

        px4_msg.angle = [float(angle_x), float(angle_y)]
        px4_msg.distance = float(self.range_distance) if self.range_distance else 0.0

        if self.current_pose and self.range_distance:
            offset_x_camera = self.range_distance * np.tan(angle_x)
            offset_y_camera = self.range_distance * np.tan(angle_y)

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

            px4_msg.pose.position.x = marker_x_ned
            px4_msg.pose.position.y = marker_y_ned
            px4_msg.pose.position.z = marker_z_ned
        else:
            px4_msg.pose.position.x = 0.0
            px4_msg.pose.position.y = 0.0
            px4_msg.pose.position.z = -self.range_distance if self.range_distance else 0.0

        px4_msg.pose.orientation.w = 1.0
        px4_msg.size = luckfox_msg.size if hasattr(luckfox_msg, 'size') else [0.1, 0.1]

        self.px4_pub.publish(px4_msg)
        rospy.loginfo_throttle(1.0, f"Bridged to PX4: marker {px4_msg.target_num}, dist={self.range_distance:.2f}m")

    def custom_logic(self, msg):
        rospy.loginfo_throttle(1.0, f"Custom mode: received marker {msg.target_num}")

    def simple_track_and_land(self):
        rospy.loginfo("Starting custom tracking and landing")
        rate = rospy.Rate(10)
        land_count = 0

        while not rospy.is_shutdown():
            rate.sleep()

        return False

    def px4_precland_mode(self):
        rospy.loginfo("PX4 Precision Landing Mode")
        rospy.loginfo("Control steps:")
        rospy.loginfo("1. rosservice call /mavros/cmd/arming \"value: true\"")
        rospy.loginfo("2. rosservice call /mavros/cmd/takeoff \"{altitude: 2.0}\"")
        rospy.loginfo("3. Wait for marker to be visible")
        rospy.loginfo("4. rosservice call /mavros/set_mode \"custom_mode: 'AUTO.PRECLAND'\"")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

    def run(self):
        rospy.loginfo("Aruco Bridge System starting")
        rospy.sleep(2)

        if self.mode == 'custom':
            rospy.loginfo("Custom mode: autonomous landing")
            rospy.loginfo("Taking off to 2m")
            self.navigate(x=0, y=0, z=2, speed=0.5, frame_id='body', auto_arm=True)
            rospy.sleep(5)
            self.simple_track_and_land()

        elif self.mode == 'px4_precland':
            self.px4_precland_mode()

        else:
            rospy.logerr("Unknown mode: %s", self.mode)


if __name__ == '__main__':
    try:
        bridge = ArucoBridgeROS()
        bridge.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node interrupted")
    except Exception as e:
        rospy.logerr("Fatal error: %s", str(e))
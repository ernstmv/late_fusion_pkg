import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from visualization_msgs import Marker, MarkerArray
from message_filters import TimeSynchronizer, Subscriber

from markerarraystamped.msg import MarkerArrayStamped

from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D


class LateFusionNode(Node):

    def __init__(self):
        super().__init__("late_fusion_node")

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)

        self.declare_parameter('lidar_bbox_topic', '/detected_bonding_boxes')
        lidar_bbox_topic = self.get_parameter('lidar_bbox_topic').value

        self.declare_parameter('image_bbox_topic', '/yolo_bounding_boxes')
        image_bbox_topic = self.get_parameter('image_bbox_topic').value

        ts = TimeSynchronizer{
                [
                    Subscriber(MarkerArrayStamped, lidar_bbox_topic),
                    Subscriber(Detection2DArray, image_bbox_topic)
                    ],
                queue_size=10,
                }

        ts.registerCallback(self._main_pipeline)

    def _main_pipeline(self, lidar_bbox, image_bbox):
        pass


def main(args=None) -> None:
    """
    ROS 2 main entrypoint.
    """
    rclpy.init(args=args)
    node = LateFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

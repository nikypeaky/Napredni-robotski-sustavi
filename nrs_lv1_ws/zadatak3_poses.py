#!/usr/bin/env python3
import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from tf_transformations import quaternion_from_matrix

np.set_printoptions(precision=4, suppress=True)

# ---------------- PARAMETRI KOJE BIRAS ----------------
D_T_TCP = np.eye(4)                      # poza finger_tip u sustavu S_D pri kontaktu
# primjer druge orijentacije: D_T_TCP[:3, :3] = <rotacijska matrica 3x3>
PRE_OFFSET_D = np.array([0.0, 0.0, 0.05])  # pomak pre-contact poze u okviru S_D [m]
PHIS_DEG = [20, 30, 40, 50, 60, 70]        # kutovi otvorenosti [deg]
# ------------------------------------------------------


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def trans(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


def tf_to_matrix(t):
    x, y, z, w = t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
    return T


def matrix_to_msg(name, T, stamp):
    m = TransformStamped()
    m.header.frame_id = 'world'
    m.header.stamp = stamp
    m.child_frame_id = name
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = map(float, T[:3, 3])
    q = quaternion_from_matrix(T)
    (m.transform.rotation.x, m.transform.rotation.y,
     m.transform.rotation.z, m.transform.rotation.w) = map(float, q)
    return m


class Task3Poses(Node):
    def __init__(self):
        super().__init__('task3_poses')
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, '/cabinet/config', self.on_cfg, qos)
        self.buf = Buffer()
        TransformListener(self.buf, self)
        self.br = StaticTransformBroadcaster(self)
        self.cfg = None
        self.done = False
        self.create_timer(0.5, self.step)

    def on_cfg(self, msg):
        self.cfg = json.loads(msg.data)

    def step(self):
        if self.done or self.cfg is None:
            return
        try:
            t = self.buf.lookup_transform('tool0', 'finger_tip', rclpy.time.Time())
        except Exception:
            return
        tool0_T_tip = tf_to_matrix(t.transform)

        c = self.cfg
        mu = c['axis_pos']
        x, y, z, yaw = c['pose']
        W_T_O = rot_z(yaw)
        W_T_O[:3, 3] = [x, y, z]
        W_T_A = W_T_O @ np.array(c['T_A_cabinet_closed'])
        A_T_D0 = np.array(c['T_D_hinge_closed'])

        def W_T_D(phi_deg):
            return W_T_A @ rot_z(mu * np.radians(phi_deg)) @ A_T_D0

        stamp = self.get_clock().now().to_msg()
        msgs = []

        W_T_TCP_pre = W_T_D(PHIS_DEG[0]) @ trans(PRE_OFFSET_D) @ D_T_TCP
        W_T_tool0_pre = W_T_TCP_pre @ np.linalg.inv(tool0_T_tip)
        msgs += [matrix_to_msg('tcp_goal_pre', W_T_TCP_pre, stamp),
                 matrix_to_msg('tool0_goal_pre', W_T_tool0_pre, stamp)]

        for phi in PHIS_DEG:
            W_T_TCP = W_T_D(phi) @ D_T_TCP                  # formula (3)
            W_T_tool0 = W_T_TCP @ np.linalg.inv(tool0_T_tip)
            msgs += [matrix_to_msg(f'tcp_goal_{phi}', W_T_TCP, stamp),
                     matrix_to_msg(f'tool0_goal_{phi}', W_T_tool0, stamp)]
            print(f'phi={phi:3d}  tool0 u svijetu: {W_T_tool0[:3, 3]}')

        self.br.sendTransform(msgs)
        print('Objavljeno. U RViz-u (TF) prikazi tcp_goal_* i tool0_goal_*.')
        self.done = True


def main():
    rclpy.init()
    node = Task3Poses()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
        rclpy.spin_once(node, timeout_sec=1.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
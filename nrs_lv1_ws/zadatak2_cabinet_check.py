#!/usr/bin/env python3
import json
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

np.set_printoptions(precision=4, suppress=True)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def tf_to_matrix(t):
    x, y, z, w = (t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w)
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
    return T


def diff(Ta, Tb):
    """Razlika translacija [m] i kut rotacije [deg] izmedu dvije matrice."""
    dp = np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])
    R = Ta[:3, :3].T @ Tb[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return dp, ang


class CabinetCheck(Node):
    def __init__(self):
        super().__init__('cabinet_check')
        qos = QoSProfile(depth=1,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(String, '/cabinet/config', self.on_config, qos)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.W_T_D = None
        self.tries = 0
        self.create_timer(0.5, self.check_tf)

    def on_config(self, msg):
        if self.W_T_D is not None:
            return
        cfg = json.loads(msg.data)

        mu = cfg['axis_pos']
        w, h = cfg['door_width'], cfg['door_height']
        d = cfg['panel_thickness']
        depth = cfg['cabinet_depth']
        phi = np.radians(cfg['initial_angle_deg'])
        x, y, z, yaw = cfg['pose']

        # W_T_O: ishodiste ormarica u svijetu
        W_T_O = rot_z(yaw)
        W_T_O[:3, 3] = [x, y, z]

        # O_T_A: os zgloba u odnosu na ormaric
        O_T_A = rot_z(np.pi if mu < 0 else 0.0)
        O_T_A[:3, 3] = [depth / 2 - d / 2, mu * w / 2, 0.0]

        # A_T_D(0): formula (1), l = [d, w, h], r = [0, -w/2]
        lx, ly, lz = d, w, h
        rx, ry = 0.0, -w / 2 # isto kao r=np.array([0.0, -door_width/2]) u cabinet_spawner.py
        A_T_D0 = np.array([[0, 0, -mu, rx - mu * lx / 2],
                           [mu, 0, 0, ry - ly / 2],
                           [0, -1, 0, lz / 2],
                           [0, 0, 0, 1.0]])

        W_T_A = W_T_O @ O_T_A
        self.W_T_D = W_T_A @ rot_z(mu * phi) @ A_T_D0   # formula (2)

        print('\n=== OS ROTACIJE VRATA (u svijetu) ===')
        print('tocka na osi :', W_T_A[:3, 3])
        print('smjer osi    :', W_T_A[:3, 2])
        print('\n=== W_T_D(phi) izracunato ===')
        print(self.W_T_D)

        print('\n=== Usporedba s porukom /cabinet/config ===')
        for name, mine, ref in [
                ('O_T_A  vs T_A_cabinet_closed', O_T_A, cfg['T_A_cabinet_closed']),
                ('A_T_D0 vs T_D_hinge_closed  ', A_T_D0, cfg['T_D_hinge_closed']),
                ('W_T_D  vs T_D_world_initial ', self.W_T_D, cfg['T_D_world_initial'])]:
            dp, ang = diff(mine, np.array(ref))
            print(f'{name}: d_poz = {dp:.2e} m, d_rot = {ang:.2e} deg')

    def check_tf(self):
        if self.W_T_D is None:
            return
        self.tries += 1
        try:
            t = self.tf_buffer.lookup_transform(
                'world', 'cabinet_D_initial', rclpy.time.Time())
        except Exception as e:
            if self.tries > 10:
                print('TF nije dostupan:', e)
                raise SystemExit
            return
        T_tf = tf_to_matrix(t.transform)
        dp, ang = diff(self.W_T_D, T_tf)
        print('\n=== Usporedba s TF world -> cabinet_D_initial ===')
        print(T_tf)
        print(f'd_poz = {dp:.2e} m, d_rot = {ang:.2e} deg')
        raise SystemExit


def main():
    rclpy.init()
    node = CabinetCheck()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
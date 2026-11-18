#!/usr/bin/env python3
import json, time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from builtin_interfaces.msg import Duration
from std_msgs.msg import String
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.srv import GetMotionPlan, GetCartesianPath, GetPositionIK
from moveit_msgs.msg import Constraints, JointConstraint, RobotState, MoveItErrorCodes
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_from_matrix

from zadatak3_poses import D_T_TCP, PRE_OFFSET_D, rot_z, trans, tf_to_matrix

GROUP, LINK, BASE = 'ur_manipulator', 'tool0', 'base_link'
ARM = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
PHIS = list(range(20, 71, 5))     # kutovi otvorenosti [deg]
N_SEEDS = 40                      # broj nasumicnih pocetnih konfiguracija
MAX_JUMP = 0.8                    # najveci dopusten skok zgloba izmedu dvije poze [rad]

ERR = {getattr(MoveItErrorCodes, n): n for n in dir(MoveItErrorCodes)
       if n.isupper() and isinstance(getattr(MoveItErrorCodes, n), int)}


def mat_to_pose(T):
    p = Pose()
    p.position.x, p.position.y, p.position.z = map(float, T[:3, 3])
    q = quaternion_from_matrix(T)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, q)
    return p


def state_from_q(q):
    s = RobotState()
    s.joint_state.name = list(ARM)
    s.joint_state.position = [float(v) for v in q]
    s.is_diff = True
    return s


class Search(Node):
    def __init__(self):
        super().__init__('task3_search')
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.cfg = None
        self.create_subscription(String, '/cabinet/config', self.on_cfg, qos)
        self.buf = Buffer()
        TransformListener(self.buf, self)
        self.ik_cli = self.create_client(GetPositionIK, '/compute_ik')
        self.plan_cli = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.cart_cli = self.create_client(GetCartesianPath, '/compute_cartesian_path')

    def on_cfg(self, msg):
        self.cfg = json.loads(msg.data)

    def wait_ready(self, timeout=30.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.cfg is None:
                continue
            try:
                a = self.buf.lookup_transform('tool0', 'finger_tip', rclpy.time.Time())
                b = self.buf.lookup_transform('world', BASE, rclpy.time.Time())
            except Exception:
                continue
            return tf_to_matrix(a.transform), tf_to_matrix(b.transform)
        raise RuntimeError('Nema /cabinet/config ili TF-a (pokrenuta simulacija?)')

    def call(self, cli, req, timeout=30.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()

    def ik(self, T, q_seed, avoid=True):
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name = GROUP
        r.ik_link_name = LINK
        r.avoid_collisions = avoid
        r.timeout = Duration(sec=0, nanosec=100_000_000)
        ps = PoseStamped()
        ps.header.frame_id = BASE
        ps.pose = mat_to_pose(T)
        r.pose_stamped = ps
        r.robot_state = state_from_q(q_seed)
        res = self.call(self.ik_cli, req)
        if res is None or res.error_code.val != 1:
            return None
        d = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        return np.array([d[j] for j in ARM])

    def chain(self, Ts, q0):
        """IK za niz poza, svaka krece od rjesenja prethodne. Vraca (liste rjesenja, uspjeh)."""
        qs, prev = [], q0
        for T in Ts:
            q = self.ik(T, prev)
            if q is None:
                return qs, False
            qs.append(q)
            prev = q
        return qs, True

    def cartesian(self, q_start, poses, avoid):
        req = GetCartesianPath.Request()
        req.header.frame_id = BASE
        req.start_state = state_from_q(q_start)
        req.group_name, req.link_name = GROUP, LINK
        req.waypoints = [mat_to_pose(T) for T in poses]
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = avoid
        res = self.call(self.cart_cli, req)
        return (res.fraction, ERR.get(res.error_code.val, res.error_code.val)) if res else (0.0, 'NO_RESPONSE')

    def plan_joints(self, q_goal):
        req = GetMotionPlan.Request()
        r = req.motion_plan_request
        r.group_name = GROUP
        r.num_planning_attempts = 10
        r.allowed_planning_time = 5.0
        r.max_velocity_scaling_factor = 0.1
        r.max_acceleration_scaling_factor = 0.1
        c = Constraints()
        for n, v in zip(ARM, q_goal):
            jc = JointConstraint()
            jc.joint_name, jc.position = n, float(v)
            jc.tolerance_above = jc.tolerance_below = 0.001
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        r.goal_constraints.append(c)
        res = self.call(self.plan_cli, req)
        return ERR.get(res.motion_plan_response.error_code.val, '?') if res else 'NO_RESPONSE'


def main():
    rclpy.init()
    node = Search()
    try:
        for cli in (node.ik_cli, node.plan_cli, node.cart_cli):
            if not cli.wait_for_service(timeout_sec=10.0):
                raise RuntimeError(f'Servis {cli.srv_name} nije dostupan')
        tool0_T_tip, W_T_B = node.wait_ready()
        B_T_W = np.linalg.inv(W_T_B)
        inv_tip = np.linalg.inv(tool0_T_tip)

        c = node.cfg
        mu = c['axis_pos']
        x, y, z, yaw = c['pose']
        W_T_O = rot_z(yaw)
        W_T_O[:3, 3] = [x, y, z]
        W_T_A = W_T_O @ np.array(c['T_A_cabinet_closed'])
        A_T_D0 = np.array(c['T_D_hinge_closed'])
        W_T_D = lambda phi: W_T_A @ rot_z(mu * np.radians(phi)) @ A_T_D0

        rng = np.random.default_rng(0)
        seeds = [rng.uniform(-np.pi, np.pi, 6) for _ in range(N_SEEDS)]
        best = None   # (skok, k, q_pre, qs, B_pre, B_contacts)

        print(f'\nTrazim: {N_SEEDS} pocetnih konfiguracija x 4 zakreta prsta, '
              f'kutovi {PHIS[0]}..{PHIS[-1]} deg')
        for k in range(4):
            Dk = D_T_TCP @ rot_z(k * np.pi / 2)        # zakret prsta oko vlastite z-osi
            B_pre = B_T_W @ W_T_D(PHIS[0]) @ trans(PRE_OFFSET_D) @ Dk @ inv_tip
            B_cont = [B_T_W @ W_T_D(p) @ Dk @ inv_tip for p in PHIS]
            n_pre = n_full = 0
            best_k = None
            for q0 in seeds:
                q_pre = node.ik(B_pre, q0)
                if q_pre is None:
                    continue
                n_pre += 1
                qs, ok = node.chain(B_cont, q_pre)
                if not ok:
                    continue
                allq = [q_pre] + qs
                jump = max(np.max(np.abs(allq[i + 1] - allq[i])) for i in range(len(allq) - 1))
                if jump > MAX_JUMP:
                    continue
                n_full += 1
                if best_k is None or jump < best_k[0]:
                    best_k = (jump, k, q_pre, qs, B_pre, B_cont)
            print(f'zakret {k*90:3d} deg: pre-contact IK uspio {n_pre}/{N_SEEDS}, '
                  f'cijeli luk {n_full}/{N_SEEDS}'
                  + (f', najmanji skok zgloba {best_k[0]:.2f} rad' if best_k else ''))
            if best_k and (best is None or best_k[0] < best[0]):
                best = best_k

        if best is None:
            print('\nNiti jedna kombinacija ne daje cijeli luk bez skoka zgloba.')
            print('Mogucnosti: promijeniti D_T_TCP / PRE_OFFSET_D ili suziti raspon kutova.')
            return

        jump, k, q_pre, qs, B_pre, B_cont = best
        print(f'\n=== NAJBOLJE: zakret prsta {k*90} deg, najveci skok zgloba {jump:.2f} rad ===')
        print('pre-contact zglobovi [rad]:', np.round(q_pre, 3).tolist())
        print('D_T_TCP rotacija (zalijepi u zadatak3_poses.py):')
        print(np.round((D_T_TCP @ rot_z(k * np.pi / 2))[:3, :3], 3))

        print('\n=== Provjera kartezijanskim lukom iz pre-contact konfiguracije ===')
        for avoid in (True, False):
            f, code = node.cartesian(q_pre, B_cont, avoid)
            print(f'avoid_collisions={avoid!s:5}: fraction = {f:.2f}  {code}')
        print('Ako je fraction 1.00 samo bez kolizija, luk puca zbog kolizije, ne kinematike.')

        print('\n=== Plan od trenutnog stanja do pre-contact konfiguracije ===')
        print(node.plan_joints(q_pre))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
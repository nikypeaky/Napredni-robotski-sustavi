#!/usr/bin/env python3
import json
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import GetMotionPlan, GetCartesianPath
from moveit_msgs.msg import (Constraints, PositionConstraint, OrientationConstraint,
                             RobotState, MoveItErrorCodes)
from tf2_ros import Buffer, TransformListener
from tf_transformations import quaternion_from_matrix

from zadatak3_poses import (D_T_TCP, PRE_OFFSET_D, PHIS_DEG,
                            rot_z, trans, tf_to_matrix)

GROUP = 'ur_manipulator'
LINK = 'tool0'
BASE = 'base_link'

ERR = {getattr(MoveItErrorCodes, n): n for n in dir(MoveItErrorCodes)
       if n.isupper() and isinstance(getattr(MoveItErrorCodes, n), int)}


def mat_to_pose(T):
    p = Pose()
    p.position.x, p.position.y, p.position.z = map(float, T[:3, 3])
    q = quaternion_from_matrix(T)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, q)
    return p


def pose_constraints(T):
    pose = mat_to_pose(T)
    pc = PositionConstraint()
    pc.header.frame_id = BASE
    pc.link_name = LINK
    sp = SolidPrimitive()
    sp.type = SolidPrimitive.SPHERE
    sp.dimensions = [0.001]
    pc.constraint_region.primitives.append(sp)
    pc.constraint_region.primitive_poses.append(pose)
    pc.weight = 1.0
    oc = OrientationConstraint()
    oc.header.frame_id = BASE
    oc.link_name = LINK
    oc.orientation = pose.orientation
    oc.absolute_x_axis_tolerance = 0.01
    oc.absolute_y_axis_tolerance = 0.01
    oc.absolute_z_axis_tolerance = 0.01
    oc.weight = 1.0
    c = Constraints()
    c.position_constraints.append(pc)
    c.orientation_constraints.append(oc)
    return c


class Planner(Node):
    def __init__(self):
        super().__init__('task3_plan')
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.cfg = None
        self.create_subscription(String, '/cabinet/config', self.on_cfg, qos)
        self.buf = Buffer()
        TransformListener(self.buf, self)
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

    def call(self, client, req):
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=30.0)
        return fut.result()

    def plan_to(self, B_T_6, start_state=None):
        req = GetMotionPlan.Request()
        r = req.motion_plan_request
        r.group_name = GROUP
        r.num_planning_attempts = 10
        r.allowed_planning_time = 5.0
        r.max_velocity_scaling_factor = 0.1
        r.max_acceleration_scaling_factor = 0.1
        r.goal_constraints.append(pose_constraints(B_T_6))
        if start_state is not None:
            r.start_state = start_state
        res = self.call(self.plan_cli, req)
        if res is None:
            return None, 'NO_RESPONSE'
        code = res.motion_plan_response.error_code.val
        return (res.motion_plan_response.trajectory if code == 1 else None), ERR.get(code, code)

    def cartesian(self, start_state, poses):
        req = GetCartesianPath.Request()
        req.header.frame_id = BASE
        req.start_state = start_state
        req.group_name = GROUP
        req.link_name = LINK
        req.waypoints = [mat_to_pose(T) for T in poses]
        req.max_step = 0.01
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        res = self.call(self.cart_cli, req)
        if res is None:
            return 0.0, 'NO_RESPONSE'
        return res.fraction, ERR.get(res.error_code.val, res.error_code.val)


def main():
    rclpy.init()
    node = Planner()
    try:
        for cli in (node.plan_cli, node.cart_cli):
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

        def W_T_D(phi):
            return W_T_A @ rot_z(mu * np.radians(phi)) @ A_T_D0

        B_pre = B_T_W @ W_T_D(PHIS_DEG[0]) @ trans(PRE_OFFSET_D) @ D_T_TCP @ inv_tip
        B_goals = {phi: B_T_W @ W_T_D(phi) @ D_T_TCP @ inv_tip for phi in PHIS_DEG}

        print('\n=== 1) Dohvatljivost pojedinacnih poza (plan iz trenutnog stanja) ===')
        traj_pre, code = node.plan_to(B_pre)
        print(f'pre-contact : {code}')
        for phi, T in B_goals.items():
            _, code = node.plan_to(T)
            print(f'phi={phi:3d}      : {code}   doseg {np.linalg.norm(T[:3, 3]):.3f} m')

        print('\n=== 2) Kartezijanski luk kroz sve kutove, pocevsi od pre-contact poze ===')
        if traj_pre is None:
            print('Pre-contact nije planiran, luk se ne moze provjeriti.')
        else:
            jt = traj_pre.joint_trajectory
            start = RobotState()
            start.joint_state.name = list(jt.joint_names)
            start.joint_state.position = list(jt.points[-1].positions)
            start.is_diff = True
            fraction, code = node.cartesian(start, [B_goals[p] for p in PHIS_DEG])
            print(f'ostvareni dio putanje (fraction) = {fraction:.2f}   {code}')
            print('fraction = 1.00 znaci da je cijeli luk izvediv.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
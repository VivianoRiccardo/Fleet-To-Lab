import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Odometry
from threading import Thread, Lock
import numpy as np
import time
import sys
import termios
import tty
import math
import csv
from collections import deque

class WindowedDataCollector(Thread):
    G = 9.80665  

    def __init__(self, env, log_file="training_data.csv"):
        super().__init__()
        self.env           = env
        self.log_file_path = log_file
        self.running       = True
        self.daemon        = True

        # --- CONFIGURATION  ---
        #self.WHEEL_RADIUS = 0.095   # cadre rover
        self.WHEEL_RADIUS = 0.168   # husky
        #self.WHEEL_RADIUS    = 0.065  # leo rover

        self.IMU_INCLUDES_GRAVITY = False

        self.WINDOW_DURATION = 2.0    
        self.WINDOW_STRIDE   = 2.0   
        self.MIN_SAMPLES     = 5    
        self.POLL_RATE       = 50    

        self.data_buffer     = deque()
        self.prev_sim_time   = None
        self.prev_linear_vel = np.zeros(3)
        self.last_emit_time  = None
        self.use_wall_clock  = False

        self.header = [
            "feat_w_vel_mean", "feat_w_vel_std",     
            "feat_curr_mean",  "feat_curr_std",      
            "feat_acc_x_mean", "feat_acc_x_std",     
            "feat_acc_y_mean", "feat_acc_y_std",     
            "feat_acc_z_mean", "feat_acc_z_std",     
            "feat_pitch_mean", "feat_roll_mean",     
            "total_dist_actual", "total_dist_cmd",
            "slip_percentage",   "slip_class",
            "n_samples", "win_duration_s",
        ]
        with open(self.log_file_path, mode='w', newline='') as f:
            csv.writer(f).writerow(self.header)

        print(f"[DataCollector] Logging to '{self.log_file_path}' | "
              f"wheel_radius={self.WHEEL_RADIUS}m | window={self.WINDOW_DURATION}s sim | "
              f"stride={self.WINDOW_STRIDE}s | 12 features")

    @staticmethod
    def euler_from_quaternion(q):
        x, y, z, w = q[0], q[1], q[2], q[3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = max(min(t2, 1.0), -1.0)
        pitch = math.asin(t2)
        return roll, pitch

    @staticmethod
    def quat_to_rot(q):
        x, y, z, w = q[0], q[1], q[2], q[3]
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
            [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
        ])

    def _linear_accel_body(self, imu_accel, quat):
        if not self.IMU_INCLUDES_GRAVITY:
            return imu_accel
        R = self.quat_to_rot(quat)
        g_reaction_body = R.T @ np.array([0.0, 0.0, self.G])
        return imu_accel - g_reaction_body

    def _detect_clock(self):
        for _ in range(30):
            with self.env.lock:
                stamp = self.env.odom_stamp
            if stamp > 0.0:
                self.use_wall_clock = False
                print(f"[DataCollector] Using SIM clock (odom header, t0={stamp:.3f}s).")
                return
            time.sleep(0.1)
        self.use_wall_clock = True
        print("[DataCollector] *** WARNING: odom header.stamp is 0 -- no sim clock. "
              "Falling back to WALL CLOCK. If SRB lags, slip labels will be distorted. "
              "Relaunch the node with use_sim_time:=true to fix this. ***")

    def _report_imu_baseline(self):
        with self.env.lock:
            ok    = self.env.imu_available
            accel = self.env.imu_accel.copy()
        if not ok:
            print("[DataCollector] IMU not detected -- acceleration will be derived (dv/dt). "
                  "Check the topic name /srb/<env>/imu_robot if you expected an IMU.")
            return
        mag = float(np.linalg.norm(accel))
        print(f"[DataCollector] IMU detected. Resting |accel| = {mag:.2f} m/s^2 "
              f"(axes = [{accel[0]:.2f}, {accel[1]:.2f}, {accel[2]:.2f}]).")
        if mag > 5.0:
            print("                ~9.8 -> gravity IS included. Compensation ON "
                  "(IMU_INCLUDES_GRAVITY=True). Looks correct.")
        else:
            print("                ~0 -> gravity already removed. Set "
                  "IMU_INCLUDES_GRAVITY=False to avoid double subtraction.")

    def _pace(self, loop_start):
        elapsed = time.time() - loop_start
        time.sleep(max(0.0, (1.0 / self.POLL_RATE) - elapsed))

    def run(self):
        time.sleep(1.0)            
        self._detect_clock()
        self._report_imu_baseline()

        while self.running:
            loop_start = time.time()

            with self.env.lock:
                sim_time  = self._now()
                gt_pos    = self.env.gt_position.copy()
                quat      = self.env.orientation.copy()
                curr_lin  = self.env.linear.copy()
                imu_accel = self.env.imu_accel.copy()
                imu_ok    = self.env.imu_available
                wh_vels   = self.env.wheel_velocities.copy()
                wh_effs   = self.env.wheel_efforts.copy()

            if len(wh_vels) < 4: wh_vels = np.zeros(4)
            if len(wh_effs) < 4: wh_effs = np.zeros(4)

            if self.prev_sim_time is None:
                if sim_time > 0.0 or self.use_wall_clock:
                    self.prev_sim_time   = sim_time
                    self.last_emit_time  = sim_time
                    self.prev_linear_vel = curr_lin
                self._pace(loop_start)
                continue

            if sim_time <= self.prev_sim_time:
                self._pace(loop_start)
                continue

            dt = sim_time - self.prev_sim_time

            w_vel_abs   = float(np.mean(np.abs(wh_vels)))
            current_abs = float(np.mean(np.abs(wh_effs)))
            roll, pitch = self.euler_from_quaternion(quat)

            if imu_ok:
                accel = self._linear_accel_body(imu_accel, quat)
            else:
                accel = (curr_lin - self.prev_linear_vel) / dt

            mid     = len(wh_vels) // 2
            v_left  = np.mean(wh_vels[:mid])
            v_right = np.mean(wh_vels[mid:])
            cmd_lin_vel = (v_left + v_right) / 2.0 * self.WHEEL_RADIUS

            self.data_buffer.append({
                "time":        sim_time,
                "dt":          dt,
                "gt_pos":      gt_pos,
                "cmd_lin_vel": cmd_lin_vel,
                "w_vel_abs":   w_vel_abs,
                "current_abs": current_abs,
                "accel":       accel,
                "pitch":       pitch,
                "roll":        roll,
            })

            while (len(self.data_buffer) > 1 and
                   self.data_buffer[-1]["time"] - self.data_buffer[0]["time"] > self.WINDOW_DURATION):
                self.data_buffer.popleft()

            span = self.data_buffer[-1]["time"] - self.data_buffer[0]["time"]
            if (span >= self.WINDOW_DURATION - 0.2 and
                    len(self.data_buffer) >= self.MIN_SAMPLES and
                    sim_time - self.last_emit_time >= self.WINDOW_STRIDE):
                self._process_and_save_window()
                self.last_emit_time = sim_time

            self.prev_sim_time   = sim_time
            self.prev_linear_vel = curr_lin
            self._pace(loop_start)

    def _now(self):
        return time.time() if self.use_wall_clock else self.env.odom_stamp

    def _process_and_save_window(self):
        buf = list(self.data_buffer)

        w_vels = [s["w_vel_abs"]   for s in buf]
        currs  = [s["current_abs"] for s in buf]
        acc_x  = [s["accel"][0]    for s in buf]
        acc_y  = [s["accel"][1]    for s in buf]
        acc_z  = [s["accel"][2]    for s in buf]
        pitch  = [s["pitch"]       for s in buf]
        roll   = [s["roll"]        for s in buf]

        features = [
            np.mean(w_vels), np.std(w_vels),
            np.mean(currs),  np.std(currs),
            np.mean(acc_x),  np.std(acc_x),
            np.mean(acc_y),  np.std(acc_y),
            np.mean(acc_z),  np.std(acc_z),
            np.mean(pitch),  np.mean(roll),
        ]

        d_C = abs(sum(s["cmd_lin_vel"] * s["dt"] for s in buf))
        d_A = sum(np.linalg.norm(buf[i]["gt_pos"] - buf[i - 1]["gt_pos"]) for i in range(1, len(buf)))

        slip_pct   = max(min((1.0 - d_A / d_C) * 100.0, 100.0), -100.0) if d_C > 0.05 else 0.0
        abs_slip   = abs(slip_pct)
        slip_class = 0 if abs_slip < 30.0 else (1 if abs_slip < 60.0 else 2)

        n_samples = len(buf)
        win_dur   = buf[-1]["time"] - buf[0]["time"]

        row = features + [d_A, d_C, slip_pct, slip_class, n_samples, round(win_dur, 4)]
        try:
            with open(self.log_file_path, mode='a', newline='') as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            print(f"[DataCollector] CSV write error: {e}")

    def stop(self):
        self.running = False
        self.join()


def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class Logger(Thread):
    def __init__(self, env, log_file="observation_log.txt"):
        super().__init__()
        self.env      = env
        self.log_file = open(log_file, "a+")
        self.running  = True
        self.daemon   = True

    def run(self):
        while self.running:
            with self.env.lock:
                obs    = self.env.get_obs()
                action = self.env.current_action
                if obs['joint_states'] is None or len(obs['joint_states']) < 4:
                    time.sleep(0.1)
                    continue
                row = [
                    obs['gt_position'][0],      obs['gt_position'][1],      obs['gt_position'][2],
                    obs['orientation'][0],      obs['orientation'][1],
                    obs['orientation'][2],      obs['orientation'][3],
                    obs['linear_velocity'][0],  obs['linear_velocity'][1],  obs['linear_velocity'][2],
                    obs['angular_velocity'][0], obs['angular_velocity'][1], obs['angular_velocity'][2],
                    obs['joint_states'][0],     obs['joint_states'][1],
                    obs['joint_states'][2],     obs['joint_states'][3],
                    action[0], action[1],
                ]
                self.log_file.write(f"{row}\n")
                self.log_file.flush()
            time.sleep(0.3)

    def stop(self):
        self.running = False
        self.join(timeout=0.1)
        self.log_file.close()


class SRBGymEnv(Node):
    def __init__(self, env_id="env0"):
        rclpy.init()
        super().__init__(f"srb_gym_env_{env_id}")
        self.env_id         = env_id
        self.current_action = np.zeros(6, dtype=float)
        self.number_of_observations_per_actions = 4
        self.lock = Lock()

        self.cmd_pub = self.create_publisher(Twist, f"/srb/{env_id}/action/cmd_vel", 10)

        self.create_subscription(Bool,    f"/srb/{env_id}/terminated", self._done_cb,      10)
        self.create_subscription(Bool,    f"/srb/{env_id}/truncated",  self._truncated_cb, 10)
        self.create_subscription(Float32, f"/srb/{env_id}/reward",     self._reward_cb,    10)

        self.create_subscription(JointState, f"/srb/{env_id}/robot/joint_states",          self._joint_state_cb, 10)
        self.create_subscription(Odometry,   f"/srb/{env_id}/robot/robot/custom_odometry",  self._odom_cb,        10)
        self.create_subscription(Imu,        f"/srb/{env_id}/imu_robot",                    self._imu_cb,         10)

        self.create_subscription(PoseArray,  f"/srb/{env_id}/dynamic_transforms",           self._sim_gt_cb,      10)

        self.reward    = 0.0
        self.done      = False
        self.truncated = False

        self.gt_position      = np.zeros(3)
        self.sim_gt_available = False

        self.odom_position = np.zeros(3)
        self.odom_stamp    = 0.0                     
        self.orientation   = np.array([0., 0., 0., 1.])
        self.linear        = np.zeros(3)
        self.angular       = np.zeros(3)

        self.imu_accel     = np.zeros(3)
        self.imu_available = False

        self.joint_states     = None
        self.wheel_velocities = np.zeros(4)
        self.wheel_efforts    = np.zeros(4)

        self.spin_running = True
        self.spin_thread  = Thread(target=self._spin_thread, daemon=True)
        self.spin_thread.start()

        self.data_collector = WindowedDataCollector(self)
        self.data_collector.start()
        self.logger = Logger(self)
        self.logger.start()

        self.get_logger().info(
            f"SRBGymEnv ready (12-feature sim-time slip collector)\n"
            f"  GT pos : /srb/{env_id}/dynamic_transforms (PoseArray poses[0])\n"
            f"  Clock  : /srb/{env_id}/robot/robot/custom_odometry header.stamp\n"
            f"  IMU    : /srb/{env_id}/imu_robot (acceleration)"
        )

    def _sim_gt_cb(self, msg: PoseArray):
        if not msg.poses:
            return
        p = msg.poses[0].position
        with self.lock:
            self.sim_gt_available = True
            self.gt_position = np.array([p.x, p.y, p.z])

    def _odom_cb(self, msg: Odometry):
        with self.lock:
            self.odom_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

            self.odom_position = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ])
            self.orientation = np.array([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ])
            self.linear = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ])
            self.angular = np.array([
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ])

            if not self.sim_gt_available:
                self.gt_position = self.odom_position.copy()

    def _imu_cb(self, msg: Imu):
        with self.lock:
            self.imu_accel = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ])
            self.imu_available = True

    def _joint_state_cb(self, msg: JointState):
        if not hasattr(self, '_joints_printed'):
            print("\n[JointState] Joint order:")
            for i, name in enumerate(msg.name):
                print(f"  [{i}] {name}")
            print()
            self._joints_printed = True

        if len(msg.velocity) != len(msg.name):
            return

        l_vels, r_vels, l_effs, r_effs = [], [], [], []
        for i, name in enumerate(msg.name):
            n = name.lower()
            if "wheel" not in n:
                continue
            is_left  = any(k in n for k in ("_l", "left",  "_fl", "_rl"))
            is_right = any(k in n for k in ("_r", "right", "_fr", "_rr"))
            if is_left:
                l_vels.append(msg.velocity[i])
                if i < len(msg.effort): l_effs.append(msg.effort[i])
            elif is_right:
                r_vels.append(msg.velocity[i])
                if i < len(msg.effort): r_effs.append(msg.effort[i])

        if not l_vels and not r_vels:
            return

        with self.lock:
            self.wheel_velocities = np.array(l_vels + r_vels)
            self.wheel_efforts    = np.array(l_effs + r_effs)
            self.joint_states     = np.array(msg.position)

    def _reward_cb(self, msg: Float32): self.reward    = float(msg.data)
    def _done_cb(self,      msg):       self.done      = msg.data
    def _truncated_cb(self, msg):       self.truncated = msg.data

    def _spin_thread(self):
        while self.spin_running:
            rclpy.spin_once(self, timeout_sec=0.01)

    def gt_source(self):
        return "sim/dynamic_transforms" if self.sim_gt_available else "odom fallback"

    def reset(self):
        self.done = False
        self.truncated = False
        time.sleep(0.1)
        return self.get_obs()

    def step(self, action):
        safe_action = action.copy()
        if abs(self.gt_position[0] + safe_action[0] * 0.1) > 14.5 or \
           abs(self.gt_position[1] + safe_action[1] * 0.1) > 14.5:
            safe_action[0] = 0.0

        with self.lock:
            self.current_action = safe_action

        msg = Twist()
        msg.linear.x,  msg.linear.y,  msg.linear.z  = float(safe_action[0]), float(safe_action[1]), float(safe_action[2])
        msg.angular.x, msg.angular.y, msg.angular.z = float(safe_action[3]), float(safe_action[4]), float(safe_action[5])
        self.cmd_pub.publish(msg)

        obs = []
        for _ in range(self.number_of_observations_per_actions):
            time.sleep(0.1 / self.number_of_observations_per_actions)
            with self.lock:
                obs.append(self.get_obs())

        return obs, self.reward, self.done, self.truncated, {}

    def get_obs(self):
        return {
            "gt_position":      self.gt_position.copy(),
            "gt_source":        self.gt_source(),
            "odom_position":    self.odom_position.copy(),
            "orientation":      self.orientation.copy(),
            "linear_velocity":  self.linear.copy(),
            "angular_velocity": self.angular.copy(),
            "imu_accel":        self.imu_accel.copy(),
            "imu_available":    self.imu_available,
            "joint_states":     self.joint_states.copy() if self.joint_states is not None else np.zeros(4),
            "wheel_velocities": self.wheel_velocities.copy(),
            "wheel_efforts":    self.wheel_efforts.copy(),
        }

    def close(self):
        self.spin_running = False
        self.data_collector.stop()
        self.logger.stop()
        time.sleep(0.05)
        rclpy.shutdown()
        self.get_logger().info("Environment closed.")


if __name__ == "__main__":
    env = SRBGymEnv(env_id="env0")

    time.sleep(1.0) 

    try:
        print("\n--- ROVER SLIPPAGE DATA COLLECTOR (12-feature, sim-time, IMU accel) ---")
        print(f"Ground truth source : {env.gt_source()}")
        print(f"IMU available       : {env.imu_available}")
        print("Drive the rover. Data -> 'training_data.csv'")
        print("Controls: W=Forward  S=Back  A=Left  D=Right  SPACE=Stop  X=Quit\n")

        env.reset()
        action  = np.zeros(6, dtype=float)
        key_map = {'w': 1, 's': 2, 'a': 3, 'd': 4, ' ': 0}

        while True:
            key = getch()
            if key == 'x':
                break
            aid = key_map.get(key, -1)
            if   aid == 1: action[0] = min(action[0] + 0.1,  1.0)
            elif aid == 2: action[0] = max(action[0] - 0.1, -1.0)
            elif aid == 3: action[1] = min(action[1] + 0.1,  1.0)
            elif aid == 4: action[1] = max(action[1] - 0.1, -1.0)
            elif aid == 0: action.fill(0.0)

            obs, r, d, t, _ = env.step(action)
            gt = env.gt_position
            print(
                f"\rCmd:[{action[0]:.1f},{action[1]:.1f}] | "
                f"GT({env.gt_source()}):({gt[0]:.2f},{gt[1]:.2f}) | "
                f"IMU:{'yes' if env.imu_available else 'derived'}",
                end=""
            )

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        env.close()

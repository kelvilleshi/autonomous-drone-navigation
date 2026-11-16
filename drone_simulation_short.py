import heapq
import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Parametrat e simulimit
DT, MAX_STEPS = 0.1, 1500
ROOM_MIN = np.array([-7.0, -5.5, 0.0])
ROOM_MAX = np.array([7.0, 5.5, 3.0])
START = np.array([-6.0, -4.5, 0.35])
DESTINATION = np.array([6.0, 4.5, 0.35])
LAND_Z, CRUISE_Z = 0.35, 1.45
GRID, WALL_CLEARANCE = 0.25, 0.60
SAFE_RADIUS, PLAN_MARGIN = 0.58, 0.85
SENSOR_RANGE, MAX_SPEED, MAX_ACCEL = 4.0, 1.10, 1.65
LAND_HOLD_STEPS = 50


@dataclass
class Box:
    center: np.ndarray
    size: np.ndarray
    label: str
    color: str

    @property
    def low(self):
        return self.center - self.size / 2

    @property
    def high(self):
        return self.center + self.size / 2

    def bounds_xy(self, margin=0.0):
        return (self.low[0] - margin, self.high[0] + margin,
                self.low[1] - margin, self.high[1] + margin)


def obstacles():
    return [
        Box(np.array([-3.0, -0.7, 0.75]), np.array([1.4, 2.8, 1.5]), "Pengesa A", "tomato"),
        Box(np.array([0.1, 1.5, 0.75]), np.array([1.5, 2.6, 1.5]), "Pengesa B", "orange"),
        Box(np.array([3.0, -1.2, 0.75]), np.array([1.4, 2.8, 1.5]), "Pengesa C", "mediumseagreen"),
    ]


def norm_limit(vector, maximum):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    return vector * (maximum / length) if length > maximum else vector


def unit(vector):
    vector = np.asarray(vector, dtype=float)
    length = np.linalg.norm(vector)
    return vector / length if length > 1e-9 else np.zeros_like(vector)


class LowPass:
    """Filtër i vetëm për vlera numerike dhe vektorë."""
    def __init__(self, alpha, initial):
        self.alpha = alpha
        self.value = np.asarray(initial, dtype=float) if np.ndim(initial) else float(initial)

    def update(self, measurement):
        self.value = self.alpha * np.asarray(measurement) + (1 - self.alpha) * self.value
        return self.value.copy() if isinstance(self.value, np.ndarray) else float(self.value)


class Kalman3D:
    def __init__(self, position):
        self.x = np.zeros((6, 1))
        self.x[:3, 0] = position
        self.P = np.eye(6) * 0.60
        self.A = np.eye(6)
        self.A[:3, 3:] = np.eye(3) * DT
        self.B = np.vstack((np.eye(3) * (0.5 * DT**2), np.eye(3) * DT))
        self.H = np.hstack((np.eye(3), np.zeros((3, 3))))
        self.Q = np.eye(6) * 0.012
        self.R = np.diag([0.08, 0.08, 0.04]) ** 2

    def predict(self, acceleration):
        self.x = self.A @ self.x + self.B @ np.asarray(acceleration).reshape(3, 1)
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, measured_position):
        innovation = np.asarray(measured_position).reshape(3, 1) - self.H @ self.x
        covariance = self.H @ self.P @ self.H.T + self.R
        gain = self.P @ self.H.T @ np.linalg.inv(covariance)
        self.x += gain @ innovation
        self.P = (np.eye(6) - gain @ self.H) @ self.P

    def set_state(self, position, velocity=None):
        self.x[:3, 0] = position
        self.x[3:, 0] = 0 if velocity is None else velocity

    @property
    def position(self):
        return self.x[:3, 0].copy()

    @property
    def velocity(self):
        return self.x[3:, 0].copy()


class PID:
    def __init__(self):
        self.kp, self.ki, self.kd = 1.65, 0.0, 0.22
        self.integral = np.zeros(3)
        self.previous = np.zeros(3)

    def reset(self):
        self.integral[:] = 0
        self.previous[:] = 0

    def update(self, error):
        self.integral += error * DT
        derivative = (error - self.previous) / DT
        self.previous = error.copy()
        return norm_limit(self.kp * error + self.ki * self.integral + self.kd * derivative, MAX_ACCEL)


# Planifikimi A* në planin XY
def free_xy(x, y, boxes, margin=PLAN_MARGIN):
    if not (ROOM_MIN[0] + WALL_CLEARANCE <= x <= ROOM_MAX[0] - WALL_CLEARANCE and
            ROOM_MIN[1] + WALL_CLEARANCE <= y <= ROOM_MAX[1] - WALL_CLEARANCE):
        return False
    return not any(x0 <= x <= x1 and y0 <= y <= y1
                   for x0, x1, y0, y1 in (box.bounds_xy(margin) for box in boxes))


def to_grid(point):
    return tuple(int(round((point[i] - ROOM_MIN[i]) / GRID)) for i in range(2))


def from_grid(cell):
    return np.array([ROOM_MIN[i] + cell[i] * GRID for i in range(2)])


def astar(start_xy, goal_xy, boxes):
    start, goal = to_grid(start_xy), to_grid(goal_xy)
    max_cell = tuple(int(round((ROOM_MAX[i] - ROOM_MIN[i]) / GRID)) for i in range(2))

    def valid(cell):
        return (0 <= cell[0] <= max_cell[0] and 0 <= cell[1] <= max_cell[1]
                and free_xy(*from_grid(cell), boxes))

    if not valid(start) or not valid(goal):
        raise RuntimeError("Pika e nisjes ose destinacioni është shumë afër murit/pengesës.")

    moves = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
    queue, came_from, cost = [(0.0, start)], {}, {start: 0.0}
    while queue:
        _, current = heapq.heappop(queue)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return [from_grid(cell) for cell in reversed(path)]
        for dx, dy in moves:
            nxt = current[0] + dx, current[1] + dy
            if not valid(nxt):
                continue
            new_cost = cost[current] + math.hypot(dx, dy)
            if new_cost < cost.get(nxt, math.inf):
                cost[nxt] = new_cost
                came_from[nxt] = current
                priority = new_cost + math.dist(nxt, goal)
                heapq.heappush(queue, (priority, nxt))
    raise RuntimeError("A* nuk gjeti rrugë. Ndrysho pengesat ose PLAN_MARGIN.")


def line_free(a, b, boxes):
    distance = np.linalg.norm(b - a)
    samples = max(2, int(distance / (GRID / 2)))
    return all(free_xy(*(a + (b - a) * t / samples), boxes)
               for t in range(samples + 1))


def build_waypoints(boxes):
    path = astar(START[:2], DESTINATION[:2], boxes)
    simple, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_free(path[i], path[j], boxes):
            j -= 1
        simple.append(path[j])
        i = j
    return [np.array([x, y, CRUISE_Z]) for x, y in simple]


# Kufizimet e sigurisë dhe lëvizja
def distance_to_box(point, box):
    nearest = np.maximum(box.low, np.minimum(point, box.high))
    return np.linalg.norm(point - nearest)


def safe_position(point, boxes):
    if not (ROOM_MIN[0] + WALL_CLEARANCE <= point[0] <= ROOM_MAX[0] - WALL_CLEARANCE and
            ROOM_MIN[1] + WALL_CLEARANCE <= point[1] <= ROOM_MAX[1] - WALL_CLEARANCE and
            LAND_Z <= point[2] <= ROOM_MAX[2] - 0.25):
        return False
    return all(distance_to_box(point, box) >= SAFE_RADIUS for box in boxes)


def inside_room(point):
    return np.maximum(ROOM_MIN + [WALL_CLEARANCE, WALL_CLEARANCE, LAND_Z],
                      np.minimum(point, ROOM_MAX - [WALL_CLEARANCE, WALL_CLEARANCE, 0.25]))


def safe_move(current, proposed, velocity, boxes):
    proposed = inside_room(proposed)
    if safe_position(proposed, boxes):
        return proposed, velocity
    low, high, best = 0.0, 1.0, current.copy()
    for _ in range(25):
        middle = (low + high) / 2
        candidate = inside_room(current + middle * (proposed - current))
        if safe_position(candidate, boxes):
            best, low = candidate, middle
        else:
            high = middle
    return best, norm_limit((best - current) / DT, MAX_SPEED * 0.30)


# Sensorët e distancës
def ray_box(origin, direction, box, max_range):
    near, far = -math.inf, math.inf
    for i in range(3):
        if abs(direction[i]) < 1e-9:
            if not box.low[i] <= origin[i] <= box.high[i]:
                return max_range
            continue
        t1, t2 = (box.low[i] - origin[i]) / direction[i], (box.high[i] - origin[i]) / direction[i]
        near, far = max(near, min(t1, t2)), min(far, max(t1, t2))
    hit = max(near, 0.0)
    return hit if far >= hit and hit <= max_range else max_range


def sensor_reading(origin, direction, boxes):
    room_distance = min(
        ((ROOM_MAX[i] if direction[i] > 0 else ROOM_MIN[i]) - origin[i]) / direction[i]
        for i in range(3) if abs(direction[i]) > 1e-9
    )
    return min(SENSOR_RANGE, room_distance,
               *(ray_box(origin, direction, box, SENSOR_RANGE) for box in boxes))


def sensor_directions(yaw):
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    left = np.array([math.cos(yaw + math.pi / 2), math.sin(yaw + math.pi / 2), 0.0])
    right = np.array([math.cos(yaw - math.pi / 2), math.sin(yaw - math.pi / 2), 0.0])
    return (forward, unit(forward + left), unit(forward + right),
            left, right, np.array([0.0, 0.0, -1.0]))


def measure_position(position):
    noise = np.random.normal(0, [0.08, 0.08, 0.04])
    return inside_room(position + noise)


def navigation_command(position, goal, directions, distances, phase, boxes):
    command = 0.95 * (goal - position)
    if phase == "LANDING":
        command[:2] *= 0.65
        command[2] *= 0.30

    for direction, distance in zip(directions[:5], distances[:5]):
        if distance < 1.10:
            command -= 1.25 * (1.10 - distance) / 1.10 * direction

    for box in boxes:
        away = position - np.maximum(box.low, np.minimum(position, box.high))
        distance = np.linalg.norm(away)
        if 1e-6 < distance < 1.20:
            command += unit(away) * 1.20 * (1.20 - distance) / 1.20

    for axis in range(2):
        low_gap, high_gap = position[axis] - ROOM_MIN[axis], ROOM_MAX[axis] - position[axis]
        influence = WALL_CLEARANCE + 0.45
        if low_gap < influence:
            command[axis] += 1.70 * (influence - low_gap)
        if high_gap < influence:
            command[axis] -= 1.70 * (influence - high_gap)

    if phase in ("TAKEOFF", "CRUISE"):
        command[2] += 0.90 * (CRUISE_Z - position[2])
    elif phase == "LANDING" and distances[-1] < 0.55:
        command[2] = max(command[2], -0.14)
    return norm_limit(command, MAX_SPEED)


# Vizualizimi
def draw_box(ax, box, color=None, alpha=0.5):
    x0, y0, z0 = box.low
    x1, y1, z1 = box.high
    v = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                  [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    faces = [[v[i] for i in face] for face in
             [(0,1,2,3),(4,5,6,7),(0,1,5,4),(2,3,7,6),(1,2,6,5),(0,3,7,4)]]
    poly = Poly3DCollection(faces, facecolor=color or box.color, edgecolor="black", alpha=alpha)
    ax.add_collection3d(poly)


def draw_room(ax):
    x, y = [ROOM_MIN[0], ROOM_MAX[0]], [ROOM_MIN[1], ROOM_MAX[1]]
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, np.zeros_like(X), color="lightgray", alpha=0.25)
    yy, zz = np.meshgrid(y, [ROOM_MIN[2], ROOM_MAX[2]])
    xx, zz2 = np.meshgrid(x, [ROOM_MIN[2], ROOM_MAX[2]])
    for wall_x in x:
        ax.plot_surface(np.full_like(yy, wall_x), yy, zz, color="skyblue", alpha=0.07)
    for wall_y in y:
        ax.plot_surface(xx, np.full_like(xx, wall_y), zz2, color="skyblue", alpha=0.07)


def draw_safety_area(ax, box):
    x0, x1, y0, y1 = box.bounds_xy(PLAN_MARGIN)
    corners = [[x0,y0,0.04],[x1,y0,0.04],[x1,y1,0.04],[x0,y1,0.04]]
    ax.add_collection3d(Poly3DCollection([corners], facecolor="yellow", edgecolor="red", alpha=0.18))
    corners.append(corners[0])
    ax.plot(*np.array(corners).T, color="red", linestyle="--", linewidth=1.2)


def rotate_xy(point, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * point[0] - s * point[1], s * point[0] + c * point[1], point[2]])


def draw_drone(ax, center, yaw):
    # Trupi, krahet, katër rotorët dhe hunda tregojnë orientimin.
    body = Box(center + [0, 0, 0.10], np.array([0.40, 0.24, 0.11]), "", "royalblue")
    draw_box(ax, body, alpha=0.95)
    arm_points = [np.array([x, y, 0.16]) for x in (-0.48, 0.48) for y in (-0.34, 0.34)]
    rotor_radius = 0.14
    theta = np.linspace(0, 2 * math.pi, 28)
    for local in arm_points:
        rotor = center + rotate_xy(local, yaw)
        ax.plot([center[0], rotor[0]], [center[1], rotor[1]],
                [center[2] + 0.10, rotor[2]], color="black", linewidth=2)
        ax.plot(rotor[0] + rotor_radius * np.cos(theta),
                rotor[1] + rotor_radius * np.sin(theta),
                np.full_like(theta, rotor[2]), color="darkred", linewidth=1.5)
    nose = center + rotate_xy(np.array([0.33, 0, 0.17]), yaw)
    ax.scatter(*nose, color="orange", marker="^", s=45)


def setup_axis(ax):
    ax.set(xlim=(ROOM_MIN[0], ROOM_MAX[0]), ylim=(ROOM_MIN[1], ROOM_MAX[1]),
           zlim=(ROOM_MIN[2], ROOM_MAX[2]), xlabel="X", ylabel="Y", zlabel="Z")
    ax.set_title("Droni 3D – shmangia e pengesave", pad=20)
    ax.view_init(elev=32, azim=-55)


def main():
    np.random.seed(10)
    boxes = obstacles()
    waypoints = build_waypoints(boxes)
    position, velocity = START.copy(), np.zeros(3)
    kf, pid = Kalman3D(position), PID()
    sensor_filters = [LowPass(0.35, SENSOR_RANGE) for _ in range(5)] + [LowPass(0.45, LAND_Z)]
    command_filter = LowPass(0.28, np.zeros(3))
    phase, waypoint_index, landed_counter = "TAKEOFF", 0, 0
    yaw, last_acceleration = 0.0, np.zeros(3)
    true_path, estimated_path = [], []
    previous_position = position.copy()
    total_distance, min_clearance = 0.0, math.inf

    plt.ion()
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.03, right=0.72, top=0.88, bottom=0.06)

    for step in range(MAX_STEPS):
        if phase == "LANDED":
            position, velocity = DESTINATION.copy(), np.zeros(3)
            last_acceleration = np.zeros(3)
            kf.set_state(position, velocity)
            estimated_position, estimated_velocity = position.copy(), velocity.copy()
        else:
            kf.predict(last_acceleration)
            kf.update(measure_position(position))
            estimated_position, estimated_velocity = kf.position, kf.velocity

            if phase == "TAKEOFF" and estimated_position[2] >= CRUISE_Z - 0.08:
                phase, waypoint_index = "CRUISE", min(1, len(waypoints) - 1)
            elif phase == "CRUISE" and np.linalg.norm(estimated_position - waypoints[waypoint_index]) < 0.42:
                waypoint_index += 1
                if waypoint_index >= len(waypoints):
                    phase, waypoint_index = "LANDING", len(waypoints) - 1
            elif phase == "LANDING":
                landed_xy = np.linalg.norm(estimated_position[:2] - DESTINATION[:2]) < 0.28
                landed_z = estimated_position[2] <= LAND_Z + 0.06
                if landed_xy and landed_z and np.linalg.norm(velocity) < 0.22:
                    phase = "LANDED"
                    position, velocity = DESTINATION.copy(), np.zeros(3)
                    last_acceleration = np.zeros(3)
                    estimated_position, estimated_velocity = position.copy(), velocity.copy()
                    pid.reset()
                    command_filter = LowPass(0.28, np.zeros(3))
                    kf.set_state(position, velocity)
                    print("Droni u ul me sukses dhe nuk preku asnjë objekt.")

        if phase == "TAKEOFF":
            goal = waypoints[0]
        elif phase == "CRUISE":
            goal = waypoints[waypoint_index]
        else:
            goal = DESTINATION.copy()

        direction_to_goal = goal - estimated_position
        if np.linalg.norm(direction_to_goal[:2]) > 1e-6:
            yaw = math.atan2(direction_to_goal[1], direction_to_goal[0])
        directions = sensor_directions(yaw)

        if phase == "LANDED":
            distances, acceleration = [SENSOR_RANGE] * 5 + [LAND_Z], np.zeros(3)
        else:
            raw = [np.clip(sensor_reading(position, direction, boxes) +
                           np.random.normal(0, 0.03), 0, SENSOR_RANGE) for direction in directions]
            distances = [f.update(value) for f, value in zip(sensor_filters, raw)]
            desired_velocity = navigation_command(
                estimated_position, goal, directions, distances, phase, boxes
            )
            desired_velocity = command_filter.update(desired_velocity)
            acceleration = pid.update(desired_velocity - estimated_velocity)
            if phase == "LANDING":
                acceleration[2] = max(acceleration[2], -0.60)
                velocity[2] = max(velocity[2], -0.28)
            velocity = norm_limit(velocity + acceleration * DT, MAX_SPEED)
            position, velocity = safe_move(position, position + velocity * DT, velocity, boxes)
            if position[2] <= LAND_Z + 0.005:
                position[2] = LAND_Z
                velocity[2] = max(velocity[2], 0.0)

        last_acceleration = acceleration.copy()
        min_clearance = min(min_clearance, *(distance_to_box(position, box) for box in boxes))
        total_distance += np.linalg.norm(position - previous_position)
        previous_position = position.copy()
        true_path.append(position.copy())
        estimated_path.append(estimated_position.copy())

        ax.clear()
        setup_axis(ax)
        draw_room(ax)
        for box in boxes:
            draw_safety_area(ax, box)
            draw_box(ax, box, alpha=0.82)
            ax.text(*[box.center[0], box.center[1], box.high[2] + 0.12], box.label,
                    fontsize=9, ha="center")
        draw_drone(ax, position, yaw)
        ax.scatter(*START, color="lime", s=65, label="START")
        ax.scatter(*DESTINATION, color="red", marker="*", s=140, label="DESTINACION")
        ax.scatter(*goal, color="black", marker="^", s=65, label="Qëllimi aktual")
        ax.scatter(*estimated_position, color="purple", marker="x", s=50, label="Pozicioni Kalman")
        wp = np.asarray(waypoints)
        ax.plot(wp[:, 0], wp[:, 1], wp[:, 2], ":", color="green", linewidth=2.5, label="Rruga A*")
        if len(true_path) > 1:
            path = np.asarray(true_path)
            ax.plot(path[:, 0], path[:, 1], path[:, 2], color="blue", linewidth=2, label="Rruga reale")
            estimate = np.asarray(estimated_path)
            ax.plot(estimate[:, 0], estimate[:, 1], estimate[:, 2], "--",
                    color="purple", linewidth=1, label="Rruga Kalman")

        for name, direction, distance in zip(
                ("Përpara", "Para-majtas", "Para-djathtas", "Majtas", "Djathtas", "Poshtë"),
                directions, distances):
            end = position + direction * distance
            ax.plot([position[0], end[0]], [position[1], end[1]], [position[2], end[2]],
                    color="black", alpha=0.45, linewidth=1)
            if name in ("Përpara", "Poshtë"):
                ax.text(*end, f"{name}: {distance:.2f} m", fontsize=7)

        info = (f"Faza: {phase}\nHapi: {step}\nWaypoint: {waypoint_index + 1}/{len(waypoints)}"
                f"\nDistanca: {total_distance:.2f} m\nClearance minimal: {min_clearance:.2f} m"
                f"\nPozicioni: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
                f"\nShpejtësia: {np.linalg.norm(velocity):.2f} m/s")
        if phase == "LANDED":
            info += f"\nQëndrimi pas uljes: {landed_counter * DT:.1f} s"
        ax.text2D(1.04, 0.95, info, transform=ax.transAxes, va="top", fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9))
        ax.text2D(1.04, 0.28, "Zona e verdhë = zonë sigurie\nDroni shmang objektet pa kontakt",
                  transform=ax.transAxes, va="top", fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9))
        ax.legend(loc="upper left", bbox_to_anchor=(1.04, 0.62), fontsize=8)
        plt.pause(DT)

        if phase == "LANDED":
            landed_counter += 1
            if landed_counter >= LAND_HOLD_STEPS:
                print("Misioni përfundoi: droni shmangu pengesat, u ul dhe qëndroi në destinacion.")
                break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()

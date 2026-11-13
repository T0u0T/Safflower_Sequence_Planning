# -*- coding: utf-8 -*-


import math
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ============================================================
# Matplotlib 后端设置
# ============================================================
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon
from concurrent.futures import ProcessPoolExecutor, as_completed

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# 实验参数
# ============================================================
TEST_CONFIGS = [
    {"name": "Small", "points": 10, "obstacles": 15},
    {"name": "Medium", "points": 20, "obstacles": 20},
    {"name": "Large", "points": 30, "obstacles": 15},
    {"name": "XL", "points": 40, "obstacles": 20},
]

# 每种规模20个实例，每个实例使用5个随机种子
N_INSTANCES = 20
N_SEEDS = 5

UNIFIED_MAX_EVALS = 4000
PRINT_FREQ_INSTANCE = 2

# ============================================================
# 多进程并行设置
# ============================================================
# True：完整多算法实验按 instance-seed 任务并行；False：保持原串行逻辑。
ENABLE_PARALLEL = True

# 并行工作进程数，可根据计算机配置调整
N_WORKERS = 6

# GA/ACO参数
N_GENERATIONS = 60
POP_SIZE = 40



@dataclass
class WorkspaceConfig:
    x_min: float = 0
    x_max: float = 300
    y_min: float = 0
    y_max: float = 680
    z_min: float = 0
    z_max: float = 550


@dataclass
class PickingConfig:
    picking_height_offset: float = 100
    safe_dist: float = 70
    target_radius_min: float = 15
    target_radius_max: float = 20
    obstacle_radius_min: float = 12
    obstacle_radius_max: float = 22


@dataclass
class CollisionConfig:
    safety_margin: float = 10


@dataclass
class PenaltyConfig:
    # 长度权重固定为1；碰撞和平滑为自适应罚权重
    collision_weight_init: float = 10.0
    smoothness_weight_init: float = 0.5
    collision_weight_max: float = 1000.0
    smoothness_weight_max: float = 50.0
    weight_increase_rate: float = 1.05
    weight_aggressive_rate: float = 1.2
    feasible_collision_threshold: float = 0.05
    feasible_smoothness_threshold: float = 0.20
    length_norm: float = 1200.0
    z_norm: float = 10000.0
    z_weight: float = 0.05


@dataclass
class AlgorithmConfig:
    pop_size: int = POP_SIZE
    generations: int = N_GENERATIONS
    crossover_rate: float = 0.85
    elite_ratio: float = 0.1
    tournament_size: int = 3
    base_mutation_rate: float = 0.15
    max_mutation_rate: float = 0.5

    # ACO
    aco_ants: int = 45
    aco_alpha: float = 1.0
    aco_beta: float = 3.0
    aco_rho: float = 0.45
    aco_q: float = 100.0
    aco_elite_ratio: float = 0.20
    pheromone_min: float = 0.01
    pheromone_max: float = 20.0

    # 约束感知构造项
    edge_collision_weight: float = 0.0
    edge_z_weight: float = 0.05

    # 2-OPT
    local_search_top_k: int = 12
    local_search_trials_light: int = 60
    local_search_trials_deep: int = 300
    local_search_rounds: int = 2

    # 启发式初始化分配：保留给ACO候选探索的比例
    reserve_for_aco_ratio: float = 0.35


WORKSPACE = WorkspaceConfig()
PICKING = PickingConfig()
COLLISION = CollisionConfig()
PENALTY = PenaltyConfig()
ALGO = AlgorithmConfig()


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


# ============================================================
# 几何与实例生成
# ============================================================
def get_approach_point(point):
    safe_z = point[2] + PICKING.picking_height_offset
    safe_z = min(safe_z, WORKSPACE.z_max - 10)
    safe_z = max(safe_z, WORKSPACE.z_min + 10)
    return np.array([point[0], point[1], safe_z], dtype=float)


def get_robot_start():
    """机械臂真实安全起点：X=0 mm，Y=90 mm，Z=0 mm。"""
    return np.array([0.0, 90.0, 0.0], dtype=float)


def segment_length(a, b):
    """两点间欧氏距离。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(b - a))


def make_transfer_nodes(a, b):
    """
    从当前安全位置 a 转移到下一个安全接近点 b 的真实运动节点。

    真实执行顺序：
    1. Z 轴先升/降到下一个目标安全接近点高度；
    2. X、Y 两轴在该安全高度下同步移动到目标正上方。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    z_first = np.array([a[0], a[1], b[2]], dtype=float)
    nodes = [a]

    if segment_length(nodes[-1], z_first) > 1e-9:
        nodes.append(z_first)

    if segment_length(nodes[-1], b) > 1e-9:
        nodes.append(b)

    return nodes


def segmented_path_length(nodes):
    """按分段运动节点累计路径长度。"""
    if nodes is None or len(nodes) < 2:
        return 0.0
    return float(sum(segment_length(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)))


def make_execution_segments(path, points, robot_start, radii=None, return_start=True):
    """
    根据真实机械臂执行顺序生成分段轨迹。

    对每个目标点执行：
    当前点 -> Z 到目标上方 100 mm 安全高度 -> XY 同步到目标正上方
    -> Z 下降到目标点 -> Z 回到目标上方安全点。

    返回：[(a, b, target_radius), ...]
    其中 target_radius 用于下降/回升段的碰撞包络。
    """
    if not path:
        return []

    segments = []
    current = np.asarray(robot_start, dtype=float)

    def _append(a, b, radius=0.0):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if segment_length(a, b) > 1e-9:
            segments.append((a, b, float(radius)))

    for node in path:
        target = np.asarray(points[node], dtype=float)
        approach = get_approach_point(target)
        radius = float(radii[node]) if radii is not None else 0.0

        # 1. Z 轴先到目标上方 100 mm 安全高度，XY 不变
        z_first = np.array([current[0], current[1], approach[2]], dtype=float)
        _append(current, z_first, 0.0)

        # 2. XY 同步运动到目标正上方，Z 保持安全高度
        _append(z_first, approach, 0.0)

        # 3. Z 轴下降到目标点
        _append(approach, target, radius)

        # 4. Z 轴回到目标上方安全点
        _append(target, approach, radius)

        current = approach.copy()

    if return_start:
        start = np.asarray(robot_start, dtype=float)

        # 返回起点：先在当前安全高度平面内回到起点 XY，再回到起点 Z。
        return_xy = np.array([start[0], start[1], current[2]], dtype=float)
        _append(current, return_xy, 0.0)
        _append(return_xy, start, 0.0)

    return segments


def make_execution_nodes(path, points, robot_start, radii=None, return_start=True):
    """将执行分段轨迹转换为节点序列，便于计算长度、Z 轴累计运动和绘图。"""
    start = np.asarray(robot_start, dtype=float)
    nodes = [start]
    for _, b, _ in make_execution_segments(path, points, robot_start, radii=radii, return_start=return_start):
        if segment_length(nodes[-1], b) > 1e-9:
            nodes.append(np.asarray(b, dtype=float))
    return nodes


def point_to_segment_distance(p, a, b):
    p, a, b = np.array(p, dtype=float), np.array(a, dtype=float), np.array(b, dtype=float)
    ab = b - a
    ap = p - a
    denom = np.dot(ab, ab)
    if denom < 1e-10:
        return float(np.linalg.norm(p - a))
    t = np.dot(ap, ab) / denom
    t = np.clip(t, 0, 1)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def compute_turning_angle(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_angle = np.dot(v1, v2) / (n1 * n2)
    cos_angle = np.clip(cos_angle, -1, 1)
    return float(np.arccos(cos_angle))


def generate_random_instance(n_points, n_obstacles, instance_id=None):
    """
    随机生成目标点和障碍物。
    """
    rng = np.random.RandomState(instance_id) if instance_id is not None else np.random
    points, radii, obstacles = [], [], []
    margin = 80
    max_attempts = 5000
    fallback_candidates = 2000

    def _sample_point(is_target=True):
        x_low = WORKSPACE.x_min + margin
        x_high = WORKSPACE.x_max - margin
        y_low = WORKSPACE.y_min + margin
        y_high = WORKSPACE.y_max - margin

        if is_target:
            z_low = WORKSPACE.z_min + margin
            z_high = WORKSPACE.z_max - PICKING.picking_height_offset - margin
        else:
            z_low = WORKSPACE.z_min + margin
            z_high = WORKSPACE.z_max - margin - 100

        # 若工作空间较小导致 margin 后范围无效，则自动收缩边界余量，但不越界。
        if x_high <= x_low:
            x_low, x_high = WORKSPACE.x_min + 10, WORKSPACE.x_max - 10
        if y_high <= y_low:
            y_low, y_high = WORKSPACE.y_min + 10, WORKSPACE.y_max - 10
        if z_high <= z_low:
            z_low = WORKSPACE.z_min + 10
            z_high = WORKSPACE.z_max - PICKING.picking_height_offset - 10 if is_target else WORKSPACE.z_max - 10

        if x_high <= x_low or y_high <= y_low or z_high <= z_low:
            raise RuntimeError("工作空间过小，无法生成有效随机点。请检查 WORKSPACE 与安全高度设置。")

        return np.array([
            rng.uniform(x_low, x_high),
            rng.uniform(y_low, y_high),
            rng.uniform(z_low, z_high),
        ], dtype=float)

    for i in range(n_points):
        placed = False
        for _ in range(max_attempts):
            p = _sample_point(is_target=True)
            r = rng.uniform(PICKING.target_radius_min, PICKING.target_radius_max)
            if all(np.linalg.norm(p - q) >= r + qr + 30 for q, qr in zip(points, radii)):
                points.append(p)
                radii.append(float(r))
                placed = True
                break

        if not placed:
            # 不再插入工作空间外兜底点；改为在工作空间内选距离已有目标最远的候选点。
            best_p, best_r, best_score = None, None, -float("inf")
            for _ in range(fallback_candidates):
                p = _sample_point(is_target=True)
                r = rng.uniform(PICKING.target_radius_min, PICKING.target_radius_max)
                if points:
                    score = min(np.linalg.norm(p - q) - r - qr for q, qr in zip(points, radii))
                else:
                    score = float("inf")
                if score > best_score:
                    best_p, best_r, best_score = p, r, score
            if best_p is None:
                raise RuntimeError(f"第 {i} 个目标点无法在工作空间内生成。")
            points.append(best_p)
            radii.append(float(best_r))

    for i in range(n_obstacles):
        placed = False
        for _ in range(max_attempts):
            c = _sample_point(is_target=False)
            r = rng.uniform(PICKING.obstacle_radius_min, PICKING.obstacle_radius_max)
            if all(np.linalg.norm(c - p) >= r + pr + 40 for p, pr in zip(points, radii)):
                obstacles.append([float(c[0]), float(c[1]), float(c[2]), float(r)])
                placed = True
                break

        if not placed:
            # 不再插入工作空间外障碍物；改为在工作空间内选距离目标最远的障碍物候选点。
            best_c, best_r, best_score = None, None, -float("inf")
            for _ in range(fallback_candidates):
                c = _sample_point(is_target=False)
                r = rng.uniform(PICKING.obstacle_radius_min, PICKING.obstacle_radius_max)
                if points:
                    score = min(np.linalg.norm(c - p) - r - pr for p, pr in zip(points, radii))
                else:
                    score = float("inf")
                if score > best_score:
                    best_c, best_r, best_score = c, r, score
            if best_c is None:
                raise RuntimeError(f"第 {i} 个障碍物无法在工作空间内生成。")
            obstacles.append([float(best_c[0]), float(best_c[1]), float(best_c[2]), float(best_r)])

    return points, radii, obstacles


# ============================================================
# 约束、指标、适应度
# ============================================================
class ConstraintViolation:
    def __init__(self, obstacles, points, radii):
        self.obstacles = obstacles
        self.points = points
        self.radii = radii
        self.safety_margin = COLLISION.safety_margin
        # 方向连续性阈值：直角坐标机械臂点到点采摘不宜使用45°过严限制，这里用120°作为大角度换向惩罚阈值。
        self.max_angle_allowed = 2 * np.pi / 3

    def edge_collision_violation(self, a, b, target_radius=0.0):
        if not self.obstacles:
            return 0.0
        max_v = 0.0
        for obs in self.obstacles:
            center = np.array(obs[:3], dtype=float)
            r = obs[3] + self.safety_margin + target_radius
            d = point_to_segment_distance(center, a, b)
            max_v = max(max_v, max(0.0, 1.0 - d / (r + 1e-8)))
        return min(1.0, max_v)

    def compute_collision_violation(self, path, robot_start):
        """
        按真实执行轨迹计算碰撞违反度：
        Z 到安全高度 -> XY 同步转移 -> Z 下降采摘 -> Z 回升。
        """
        if len(self.obstacles) == 0 or not path:
            return 0.0

        max_v = 0.0
        segments = make_execution_segments(
            path=path,
            points=self.points,
            robot_start=robot_start,
            radii=self.radii,
            return_start=True,
        )

        for a, b, target_radius in segments:
            max_v = max(max_v, self.edge_collision_violation(a, b, target_radius))

        return min(1.0, max_v)

    def compute_smoothness_violation(self, path, robot_start):
        if len(path) < 2:
            return 0.0

        # 平滑性基于安全接近点层面的采摘顺序评价，避免将直角坐标机械臂必然产生的Z-XY-Z 分段运动误判为路径排序不平滑。
        traj = [robot_start] + [get_approach_point(self.points[i]) for i in path] + [robot_start]
        violations = []

        for i in range(1, len(traj) - 1):
            angle = compute_turning_angle(traj[i - 1], traj[i], traj[i + 1])
            if angle > self.max_angle_allowed:
                violations.append((angle - self.max_angle_allowed) / (np.pi - self.max_angle_allowed + 1e-8))

        if not violations:
            return 0.0

        # 使用均值+最大值混合，避免所有路径都饱和为1，增强指标区分度。
        return float(np.clip(0.5 * np.mean(violations) + 0.5 * np.max(violations), 0.0, 1.0))


class PathMetricsCalculator:
    def __init__(self, points, robot_start, constraint):
        self.points = points
        self.robot_start = robot_start
        self.constraint = constraint
        self.n = len(points)
        self.approach_points = np.array([get_approach_point(p) for p in points]) if points else np.empty((0, 3))
        self._dist_matrix = self._compute_dist_matrix()
        self._edge_collision_matrix = self._compute_edge_collision_matrix()
        self._edge_cost_matrix = self._compute_edge_cost_matrix()

    def _compute_dist_matrix(self):
        """相邻两个安全接近点之间的真实分段运动距离。"""
        dist = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    nodes = make_transfer_nodes(self.approach_points[i], self.approach_points[j])
                    dist[i, j] = segmented_path_length(nodes)
        return dist

    def _compute_edge_collision_matrix(self):
        """相邻两个安全接近点之间按 Z-XY 分段轨迹计算碰撞违反度。"""
        mat = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    nodes = make_transfer_nodes(self.approach_points[i], self.approach_points[j])
                    max_v = 0.0
                    for a, b in zip(nodes[:-1], nodes[1:]):
                        max_v = max(max_v, self.constraint.edge_collision_violation(a, b, 0.0))
                    mat[i, j] = max_v
        return mat

    def _compute_edge_cost_matrix(self):
        if self.n == 0:
            return np.empty((0, 0))

        positive_dist = self._dist_matrix[self._dist_matrix > 0]
        dist_norm_base = np.mean(positive_dist) if len(positive_dist) > 0 else 1.0
        norm_d = self._dist_matrix / (dist_norm_base + 1e-8)

        z_delta = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    z_delta[i, j] = abs(self.approach_points[i, 2] - self.approach_points[j, 2]) / PENALTY.z_norm

        cost = norm_d + ALGO.edge_collision_weight * self._edge_collision_matrix + ALGO.edge_z_weight * z_delta
        np.fill_diagonal(cost, np.inf)
        return cost

    def compute_length(self, path):
        """按真实执行轨迹累计路径长度。"""
        if not path:
            return 0.0
        nodes = make_execution_nodes(path, self.points, self.robot_start, radii=self.constraint.radii, return_start=True)
        return segmented_path_length(nodes)

    def compute_z_motion(self, path):
        """按真实执行轨迹累计 Z 轴运动量。"""
        if not path:
            return 0.0
        nodes = make_execution_nodes(path, self.points, self.robot_start, radii=self.constraint.radii, return_start=True)
        return float(sum(abs(nodes[i + 1][2] - nodes[i][2]) for i in range(len(nodes) - 1)))

    def compute_all_metrics(self, path):
        if not path:
            return 1.0, 1.0, 1.0, 1.0
        norm_len = self.compute_length(path) / PENALTY.length_norm
        coll = self.constraint.compute_collision_violation(path, self.robot_start)
        smooth = self.constraint.compute_smoothness_violation(path, self.robot_start)
        norm_z = self.compute_z_motion(path) / PENALTY.z_norm
        return norm_len, coll, smooth, norm_z


class UnifiedPenaltyFitness:
    def __init__(self):
        self.collision_weight = PENALTY.collision_weight_init
        self.smoothness_weight = PENALTY.smoothness_weight_init
        self.history = []

    def is_feasible(self, coll, smooth):
        return coll <= PENALTY.feasible_collision_threshold and smooth <= PENALTY.feasible_smoothness_threshold

    def compute_fitness(self, path, metrics):
        norm_len, coll, smooth, norm_z = metrics.compute_all_metrics(path)
        fit = norm_len + self.collision_weight * coll + self.smoothness_weight * smooth + PENALTY.z_weight * norm_z
        return float(fit), norm_len, coll, smooth, norm_z

    def update_weights(self, population, metrics):
        if not population:
            return
        cols, sms, feas = [], [], 0
        for ind in population:
            _, c, s, _ = metrics.compute_all_metrics(ind)
            cols.append(c)
            sms.append(s)
            if self.is_feasible(c, s):
                feas += 1
        fr = feas / len(population)
        self.history.append((fr, float(np.mean(cols)), float(np.mean(sms))))
        if fr < 0.2:
            rate = PENALTY.weight_aggressive_rate
        elif fr < 0.5:
            rate = PENALTY.weight_increase_rate
        elif fr > 0.8 and len(self.history) > 20:
            rate = 0.97
        else:
            rate = 1.0
        self.collision_weight = min(PENALTY.collision_weight_max, max(PENALTY.collision_weight_init, self.collision_weight * rate))
        self.smoothness_weight = min(PENALTY.smoothness_weight_max, max(PENALTY.smoothness_weight_init, self.smoothness_weight * rate))

    def get_weights(self):
        return self.collision_weight, self.smoothness_weight


class EvaluationBudgetExceeded(Exception):
    pass


class FitnessEvaluator:
    def __init__(self, fitness_func, metrics, max_evals=UNIFIED_MAX_EVALS):
        self.fitness_func = fitness_func
        self.metrics = metrics
        self.max_evals = max_evals
        self.evals = 0

    def has_budget(self, n=1):
        return self.evals + n <= self.max_evals

    def evaluate(self, path):
        if not self.has_budget(1):
            raise EvaluationBudgetExceeded("统一适应度评估预算已耗尽")
        self.evals += 1
        fit, _, coll, smooth, _ = self.fitness_func.compute_fitness(path, self.metrics)
        length = self.metrics.compute_length(path)
        return fit, length, coll, smooth


# ============================================================
# 2-OPT
# ============================================================
class ConstraintAwareTwoOpt:
    def __init__(self, evaluator: FitnessEvaluator, metrics: PathMetricsCalculator):
        self.evaluator = evaluator
        self.metrics = metrics
        self.used_evals = 0

    @staticmethod
    def swap(path, i, j):
        out = path.copy()
        out[i:j + 1] = reversed(out[i:j + 1])
        return out

    def improve(self, path, mode="constraint", max_trials=100, max_rounds=1, shuffle=True):
        if len(path) < 4 or not self.evaluator.has_budget(1):
            return path.copy(), None
        before = self.evaluator.evals
        try:
            best_fit, best_len, best_coll, best_smooth = self.evaluator.evaluate(path)
        except EvaluationBudgetExceeded:
            return path.copy(), None
        best_path = path.copy()
        n = len(path)
        trials = 0
        for _ in range(max_rounds):
            improved = False
            pairs = [(i, j) for i in range(n - 1) for j in range(i + 2, n)]
            if shuffle:
                random.shuffle(pairs)
            for i, j in pairs:
                if trials >= max_trials or not self.evaluator.has_budget(1):
                    break
                cand = self.swap(best_path, i, j)
                cand_fit, cand_len, cand_coll, cand_smooth = self.evaluator.evaluate(cand)
                trials += 1
                if mode == "length":
                    accept = cand_len < best_len - 1e-6 and cand_coll <= best_coll + 1e-6
                else:
                    accept = cand_fit < best_fit - 1e-8
                if accept:
                    best_path = cand
                    best_fit, best_len, best_coll, best_smooth = cand_fit, cand_len, cand_coll, cand_smooth
                    improved = True
                    break
            if not improved or trials >= max_trials:
                break
        self.used_evals += max(0, self.evaluator.evals - before)
        return best_path, (best_fit, best_len, best_coll, best_smooth)


# ============================================================
# 搜索算子与基线算法
# ============================================================
class SearchOperators:
    @staticmethod
    def swap_mutation(ind):
        out = ind.copy()
        if len(out) >= 2:
            i, j = random.sample(range(len(out)), 2)
            out[i], out[j] = out[j], out[i]
        return out

    @staticmethod
    def inversion_mutation(ind):
        out = ind.copy()
        if len(out) >= 4:
            i, j = sorted(random.sample(range(len(out)), 2))
            if j - i > 1:
                out[i:j + 1] = reversed(out[i:j + 1])
        return out

    @staticmethod
    def scramble_mutation(ind):
        out = ind.copy()
        if len(out) >= 4:
            i, j = sorted(random.sample(range(len(out)), 2))
            if j - i > 1:
                sub = out[i:j + 1]
                random.shuffle(sub)
                out[i:j + 1] = sub
        return out

    @staticmethod
    def insert_mutation(ind):
        out = ind.copy()
        if len(out) >= 2:
            i, j = random.sample(range(len(out)), 2)
            node = out.pop(i)
            out.insert(j, node)
        return out

    @staticmethod
    def ordered_crossover(p1, p2):
        n = len(p1)
        if n < 2:
            return p1.copy()
        a, b = sorted(random.sample(range(n), 2))
        child = [-1] * n
        child[a:b] = p1[a:b]
        pos = b
        for gene in p2:
            if gene not in child:
                if pos >= n:
                    pos = 0
                child[pos] = gene
                pos += 1
        return child


class BaseAlgorithm:
    def _final_stats(self, path, fit, length, t0):
        self.planning_time = time.perf_counter() - t0
        return path, fit, length, 0


class NearestNeighbor(BaseAlgorithm):
    def __init__(self, points, robot_start, constraint):
        self.points = points
        self.robot_start = robot_start
        self.metrics = PathMetricsCalculator(points, robot_start, constraint)
        self.fitness_func = UnifiedPenaltyFitness()
        self.evaluator = FitnessEvaluator(self.fitness_func, self.metrics, UNIFIED_MAX_EVALS)
        self.planning_time = 0

    def _construct_path(self):
        unvisited = set(range(len(self.points)))
        path = []
        current = self.robot_start
        aps = self.metrics.approach_points
        while unvisited:
            node = min(unvisited, key=lambda k: segmented_path_length(make_transfer_nodes(current, aps[k])))
            path.append(node)
            unvisited.remove(node)
            current = aps[node]
        return path

    def solve(self):
        t0 = time.perf_counter()
        path = self._construct_path()
        fit, length, _, _ = self.evaluator.evaluate(path)
        return self._final_stats(path, fit, length, t0)


class GreedyWithCollision(BaseAlgorithm):
    def __init__(self, points, robot_start, constraint):
        self.points = points
        self.robot_start = robot_start
        self.constraint = constraint
        self.metrics = PathMetricsCalculator(points, robot_start, constraint)
        self.fitness_func = UnifiedPenaltyFitness()
        self.evaluator = FitnessEvaluator(self.fitness_func, self.metrics, UNIFIED_MAX_EVALS)
        self.planning_time = 0

    def solve(self):
        t0 = time.perf_counter()
        unvisited = set(range(len(self.points)))
        path = []
        current = self.robot_start
        aps = self.metrics.approach_points
        while unvisited:
            best_node, best_score = None, float("inf")
            for node in unvisited:
                trial = path + [node]
                coll = self.constraint.compute_collision_violation(trial, self.robot_start)
                dist = segmented_path_length(make_transfer_nodes(current, aps[node]))
                z = abs(current[2] - aps[node][2])
                score = dist + 1000 * coll + 0.05 * z
                if score < best_score:
                    best_node, best_score = node, score
            path.append(best_node)
            unvisited.remove(best_node)
            current = aps[best_node]
        fit, length, _, _ = self.evaluator.evaluate(path)
        return self._final_stats(path, fit, length, t0)


class StandardGA(BaseAlgorithm):
    def __init__(self, points, robot_start, constraint):
        self.points = points
        self.robot_start = robot_start
        self.metrics = PathMetricsCalculator(points, robot_start, constraint)
        self.fitness_func = UnifiedPenaltyFitness()
        self.evaluator = FitnessEvaluator(self.fitness_func, self.metrics, UNIFIED_MAX_EVALS)
        self.n = len(points)
        self.ops = SearchOperators()
        self.planning_time = 0
        self.convergence = []

    def _create(self):
        ind = list(range(self.n))
        random.shuffle(ind)
        return ind

    def _tour(self, pop, fits):
        ids = random.sample(range(len(pop)), min(ALGO.tournament_size, len(pop)))
        return min(ids, key=lambda i: fits[i])

    def _mutate(self, ind, rate):
        if random.random() > rate:
            return ind
        return random.choice([self.ops.swap_mutation, self.ops.inversion_mutation, self.ops.scramble_mutation, self.ops.insert_mutation])(ind)

    def solve(self):
        t0 = time.perf_counter()
        if self.n <= 1:
            path = list(range(self.n)); fit, length, _, _ = self.evaluator.evaluate(path); return self._final_stats(path, fit, length, t0)
        pop = [self._create() for _ in range(ALGO.pop_size)]
        fits = [self.evaluator.evaluate(ind)[0] for ind in pop]
        bidx = int(np.argmin(fits)); best = pop[bidx].copy(); best_fit = fits[bidx]; best_len = self.metrics.compute_length(best)
        self.convergence = [best_len]
        for _ in range(ALGO.generations):
            if not self.evaluator.has_budget(ALGO.pop_size):
                break
            self.fitness_func.update_weights(pop, self.metrics)
            lengths = np.array([self.metrics.compute_length(ind) for ind in pop])
            div = min(1.0, np.std(lengths) / (np.mean(lengths) + 1e-8))
            mut = min(ALGO.max_mutation_rate, ALGO.base_mutation_rate * (1 + 2 * (1 - div)))
            elite_n = max(1, int(ALGO.pop_size * ALGO.elite_ratio))
            elites = [pop[i].copy() for i in np.argsort(fits)[:elite_n]]
            new_pop = elites
            while len(new_pop) < ALGO.pop_size:
                p1 = pop[self._tour(pop, fits)]; p2 = pop[self._tour(pop, fits)]
                child = self.ops.ordered_crossover(p1, p2) if random.random() < ALGO.crossover_rate else p1.copy()
                new_pop.append(self._mutate(child, mut))
            pop = new_pop
            fits = [self.evaluator.evaluate(ind)[0] for ind in pop]
            ci = int(np.argmin(fits))
            if fits[ci] < best_fit:
                best = pop[ci].copy(); best_fit = fits[ci]; best_len = self.metrics.compute_length(best)
            self.convergence.append(best_len)
        return self._final_stats(best, best_fit, best_len, t0)


class SimulatedAnnealing(BaseAlgorithm):
    def __init__(self, points, robot_start, constraint):
        self.points = points; self.robot_start = robot_start
        self.metrics = PathMetricsCalculator(points, robot_start, constraint)
        self.fitness_func = UnifiedPenaltyFitness(); self.evaluator = FitnessEvaluator(self.fitness_func, self.metrics, UNIFIED_MAX_EVALS)
        self.n = len(points); self.planning_time = 0

    def solve(self):
        t0 = time.perf_counter()
        cur = list(range(self.n)); random.shuffle(cur)
        cur_fit, cur_len, _, _ = self.evaluator.evaluate(cur); best = cur.copy(); best_fit = cur_fit; best_len = cur_len
        temp = 1000.0
        while temp > 1e-3 and self.evaluator.has_budget(1):
            nb = cur.copy()
            if len(nb) >= 2:
                i, j = random.sample(range(len(nb)), 2); nb[i], nb[j] = nb[j], nb[i]
            nf, nl, _, _ = self.evaluator.evaluate(nb); delta = nf - cur_fit
            if delta < 0 or random.random() < math.exp(-delta / temp):
                cur, cur_fit, cur_len = nb, nf, nl
                if nf < best_fit:
                    best, best_fit, best_len = nb.copy(), nf, nl
            temp *= 0.995
        return self._final_stats(best, best_fit, best_len, t0)


class ParticleSwarmOptimization(BaseAlgorithm):
    def __init__(self, points, robot_start, constraint):
        self.points = points; self.robot_start = robot_start
        self.metrics = PathMetricsCalculator(points, robot_start, constraint)
        self.fitness_func = UnifiedPenaltyFitness(); self.evaluator = FitnessEvaluator(self.fitness_func, self.metrics, UNIFIED_MAX_EVALS)
        self.n = len(points); self.n_particles = max(2, int(np.sqrt(UNIFIED_MAX_EVALS))); self.iterations = max(1, UNIFIED_MAX_EVALS // self.n_particles); self.planning_time = 0

    @staticmethod
    def decode(pos):
        return np.argsort(np.nan_to_num(pos)).tolist()

    def solve(self):
        t0 = time.perf_counter()
        pos = [np.random.randn(self.n) for _ in range(self.n_particles)]
        vel = [np.random.randn(self.n) for _ in range(self.n_particles)]
        pbest = [p.copy() for p in pos]
        pfit = [float("inf")] * self.n_particles
        gbest = pos[0].copy(); gfit = float("inf"); glen = 0
        for _ in range(self.iterations):
            for i in range(self.n_particles):
                if not self.evaluator.has_budget(1):
                    break
                perm = self.decode(pos[i]); fit, length, _, _ = self.evaluator.evaluate(perm)
                if fit < pfit[i]:
                    pfit[i] = fit; pbest[i] = pos[i].copy()
                    if fit < gfit:
                        gfit = fit; gbest = pos[i].copy(); glen = length
            for i in range(self.n_particles):
                r1, r2 = random.random(), random.random()
                vel[i] = 0.7 * vel[i] + 1.5 * r1 * (pbest[i] - pos[i]) + 1.5 * r2 * (gbest - pos[i])
                pos[i] = pos[i] + vel[i]
        return self._final_stats(self.decode(gbest), gfit, glen, t0)




class ReinforcementLearningPlanner(BaseAlgorithm):
    """
    强化学习基线：基于 Q-learning 的采摘顺序构造方法。

    说明：
    1. 将当前目标点视为状态，将下一个待访问目标视为动作；
    2. 在统一评估预算内进行多回合路径采样与 Q 值更新；
    3. 最终按学习得到的 Q 表贪婪构造采摘顺序；
    4. 该方法作为强化学习对比基线，不改变主算法逻辑。
    """
    def __init__(self, points, robot_start, constraint):
        self.points = points
        self.robot_start = robot_start
        self.metrics = PathMetricsCalculator(points, robot_start, constraint)
        self.fitness_func = UnifiedPenaltyFitness()
        self.evaluator = FitnessEvaluator(self.fitness_func, self.metrics, UNIFIED_MAX_EVALS)
        self.n = len(points)
        self.planning_time = 0
        self.q_table = np.zeros((self.n, self.n), dtype=float)
        self.learning_rate = 0.25
        self.discount = 0.90
        self.epsilon_start = 0.35
        self.epsilon_end = 0.05
        self.convergence = []

    def _nearest_start(self):
        if self.n == 0:
            return 0
        d0 = [segmented_path_length(make_transfer_nodes(self.robot_start, self.metrics.approach_points[i])) for i in range(self.n)]
        return int(np.argmin(d0))

    def _select_next(self, current, unvisited, epsilon):
        nodes = list(unvisited)
        if random.random() < epsilon:
            return random.choice(nodes)
        q_vals = np.array([self.q_table[current, j] for j in nodes], dtype=float)
        if np.allclose(q_vals, 0.0):
            # Q表尚未形成有效偏好时，使用距离启发作为兜底。
            return min(nodes, key=lambda j: self.metrics._dist_matrix[current, j] if current != j else 0.0)
        return int(nodes[int(np.argmax(q_vals))])

    def _construct_episode_path(self, epsilon):
        if self.n <= 0:
            return []
        unvisited = set(range(self.n))
        if random.random() < 0.7:
            current = self._nearest_start()
        else:
            current = random.randrange(self.n)
        path = [current]
        unvisited.remove(current)
        while unvisited:
            nxt = self._select_next(current, unvisited, epsilon)
            path.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return path

    def _greedy_policy_path(self):
        if self.n <= 0:
            return []
        unvisited = set(range(self.n))
        current = self._nearest_start()
        path = [current]
        unvisited.remove(current)
        while unvisited:
            nodes = list(unvisited)
            q_vals = np.array([self.q_table[current, j] for j in nodes], dtype=float)
            if np.allclose(q_vals, 0.0):
                nxt = min(nodes, key=lambda j: self.metrics._dist_matrix[current, j])
            else:
                nxt = int(nodes[int(np.argmax(q_vals))])
            path.append(nxt)
            unvisited.remove(nxt)
            current = nxt
        return path

    def _update_q_table(self, path, fit):
        if len(path) < 2:
            return
        # 适应度越小越好，奖励取负值；同时加入边距离惩罚，引导短边转移。
        terminal_reward = -float(fit)
        for idx in range(len(path) - 2, -1, -1):
            a, b = path[idx], path[idx + 1]
            edge_cost = self.metrics._edge_cost_matrix[a, b]
            step_reward = -0.05 * float(edge_cost) + terminal_reward / max(1, len(path) - 1)
            if idx < len(path) - 2:
                future = np.max(self.q_table[b, path[idx + 2:]]) if len(path[idx + 2:]) > 0 else 0.0
            else:
                future = 0.0
            target = step_reward + self.discount * future
            self.q_table[a, b] += self.learning_rate * (target - self.q_table[a, b])

    def solve(self):
        t0 = time.perf_counter()
        if self.n <= 1:
            path = list(range(self.n))
            fit, length, _, _ = self.evaluator.evaluate(path)
            return self._final_stats(path, fit, length, t0)

        best_path, best_fit, best_len = None, float("inf"), 0.0
        max_episodes = max(1, UNIFIED_MAX_EVALS - 2)
        episode = 0
        while episode < max_episodes and self.evaluator.has_budget(1):
            ratio = episode / max(1, max_episodes - 1)
            epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * ratio
            path = self._construct_episode_path(epsilon)
            try:
                fit, length, _, _ = self.evaluator.evaluate(path)
            except EvaluationBudgetExceeded:
                break
            self._update_q_table(path, fit)
            if fit < best_fit:
                best_path, best_fit, best_len = path.copy(), fit, length
            self.convergence.append(best_len)
            episode += 1

        # 用学习到的Q表再构造一次贪婪路径，与训练过程中的最优路径竞争。
        policy_path = self._greedy_policy_path()
        if self.evaluator.has_budget(1):
            fit, length, _, _ = self.evaluator.evaluate(policy_path)
            if fit < best_fit:
                best_path, best_fit, best_len = policy_path.copy(), fit, length

        if best_path is None:
            best_path = policy_path
            best_fit, _, _, _, _ = self.fitness_func.compute_fitness(best_path, self.metrics)
            best_len = self.metrics.compute_length(best_path)
        return self._final_stats(best_path, best_fit, best_len, t0)

# ============================================================
# ACO 与主方法
# ============================================================
class PlainACO(BaseAlgorithm):
    def __init__(self, points, robot_start, constraint, constraint_aware=False):
        self.points = points
        self.robot_start = robot_start
        self.constraint = constraint
        self.metrics = PathMetricsCalculator(points, robot_start, constraint)
        self.fitness_func = UnifiedPenaltyFitness()
        self.evaluator = FitnessEvaluator(self.fitness_func, self.metrics, UNIFIED_MAX_EVALS)
        self.n = len(points)
        self.constraint_aware = constraint_aware
        self.planning_time = 0
        self.convergence = []
        self.pheromone = np.ones((self.n, self.n), dtype=float)

    def _transition_probs(self, current, unvisited):
        nodes = list(unvisited)
        probs = []
        for j in nodes:
            tau = self.pheromone[current, j] ** ALGO.aco_alpha
            if self.constraint_aware:
                c = self.metrics._edge_cost_matrix[current, j]
            else:
                c = self.metrics._dist_matrix[current, j]
            eta = (1.0 / (c + 1e-8)) ** ALGO.aco_beta
            probs.append(tau * eta)
        probs = np.array(probs, dtype=float)
        s = probs.sum()
        if (not np.isfinite(s)) or s <= 0:
            probs = np.ones(len(nodes), dtype=float) / len(nodes)
        else:
            probs = probs / s
            probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            s = probs.sum()
            probs = probs / s if s > 0 else np.ones(len(nodes), dtype=float) / len(nodes)
        return nodes, probs

    def _construct_ant_path(self):
        start = random.randrange(self.n)
        path = [start]
        unvisited = set(range(self.n))
        unvisited.remove(start)
        current = start
        while unvisited:
            nodes, probs = self._transition_probs(current, unvisited)
            next_node = int(np.random.choice(nodes, p=probs))
            path.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return path

    def _update_pheromone(self, paths_fits):
        self.pheromone *= (1 - ALGO.aco_rho)
        if not paths_fits:
            return
        paths_fits = sorted(paths_fits, key=lambda x: x[1])
        elite_count = max(1, int(len(paths_fits) * ALGO.aco_elite_ratio))
        for path, fit in paths_fits[:elite_count]:
            delta = ALGO.aco_q / (fit + 1e-8)
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                self.pheromone[a, b] += delta
                self.pheromone[b, a] += delta * 0.2
        self.pheromone = np.clip(self.pheromone, ALGO.pheromone_min, ALGO.pheromone_max)

    def solve(self):
        t0 = time.perf_counter()
        if self.n <= 1:
            path = list(range(self.n)); fit, length, _, _ = self.evaluator.evaluate(path); return self._final_stats(path, fit, length, t0)
        best_path, best_fit, best_len = None, float("inf"), 0.0
        ants = min(ALGO.aco_ants, max(2, UNIFIED_MAX_EVALS // 10))
        while self.evaluator.has_budget(1):
            batch = []
            for _ in range(ants):
                if not self.evaluator.has_budget(1):
                    break
                path = self._construct_ant_path()
                fit, length, _, _ = self.evaluator.evaluate(path)
                batch.append((path, fit))
                if fit < best_fit:
                    best_path, best_fit, best_len = path.copy(), fit, length
            self.fitness_func.update_weights([p for p, _ in batch], self.metrics)
            self._update_pheromone(batch)
            if best_path is not None:
                self.convergence.append(best_len)
        return self._final_stats(best_path, best_fit, best_len, t0)


class TwoOptLocalSearch(BaseAlgorithm):
    def __init__(self, points, robot_start, constraint, constraint_aware=True):
        self.points = points
        self.robot_start = robot_start
        self.metrics = PathMetricsCalculator(points, robot_start, constraint)
        self.fitness_func = UnifiedPenaltyFitness()
        self.evaluator = FitnessEvaluator(self.fitness_func, self.metrics, UNIFIED_MAX_EVALS)
        self.local = ConstraintAwareTwoOpt(self.evaluator, self.metrics)
        self.constraint_aware = constraint_aware
        self.planning_time = 0
        self.convergence = []

    def _nearest_initial(self):
        return NearestNeighbor(self.points, self.robot_start, self.metrics.constraint)._construct_path()

    def solve(self):
        t0 = time.perf_counter()
        current = self._nearest_initial()
        fit, length, _, _ = self.evaluator.evaluate(current)
        self.convergence = [length]
        best, metrics_tuple = self.local.improve(
            current,
            mode="constraint" if self.constraint_aware else "length",
            max_trials=UNIFIED_MAX_EVALS - 2,
            max_rounds=999,
            shuffle=False,
        )
        if metrics_tuple:
            fit, length, _, _ = metrics_tuple
        self.convergence.append(length)
        return self._final_stats(best, fit, length, t0)


class DualSourceCAACO2OPT(BaseAlgorithm):
    def __init__(self, points, robot_start, constraint,
                 use_constraint_aware=True,
                 use_two_opt=True,
                 use_dual_source=True,
                 use_pheromone_boost=True,
                 deep_local=True):
        self.points = points
        self.robot_start = robot_start
        self.constraint = constraint
        self.metrics = PathMetricsCalculator(points, robot_start, constraint)
        self.fitness_func = UnifiedPenaltyFitness()
        self.evaluator = FitnessEvaluator(self.fitness_func, self.metrics, UNIFIED_MAX_EVALS)
        self.local = ConstraintAwareTwoOpt(self.evaluator, self.metrics)
        self.n = len(points)
        self.use_constraint_aware = use_constraint_aware
        self.use_two_opt = use_two_opt
        self.use_dual_source = use_dual_source
        self.use_pheromone_boost = use_pheromone_boost
        self.deep_local = deep_local
        self.planning_time = 0
        self.convergence = []
        self.local_search_evals = 0
        self.pheromone = np.ones((self.n, self.n), dtype=float)

    def _safe_probs(self, probs, n):
        probs = np.array(probs, dtype=float)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, np.inf)
        s = probs.sum()
        if (not np.isfinite(s)) or s <= 0:
            return np.ones(n, dtype=float) / n
        probs = probs / s
        # 最后一次严格归一化，避免 np.random.choice 对概率和的严格检查报错
        if n > 0:
            probs[-1] = 1.0 - np.sum(probs[:-1])
            probs = np.clip(probs, 0.0, 1.0)
            s = probs.sum()
            if (not np.isfinite(s)) or s <= 0:
                return np.ones(n, dtype=float) / n
            probs = probs / s
        return probs

    def _transition_probs(self, current, unvisited):
        nodes = list(unvisited)
        probs = []
        for j in nodes:
            tau = self.pheromone[current, j] ** ALGO.aco_alpha
            if self.use_constraint_aware:
                base = self.metrics._edge_cost_matrix[current, j]
            else:
                base = self.metrics._dist_matrix[current, j]
            eta = (1.0 / (base + 1e-8)) ** ALGO.aco_beta
            probs.append(tau * eta)
        return nodes, self._safe_probs(probs, len(nodes))

    def _construct_ant_path(self):
        if self.use_dual_source and random.random() < 0.70:
            d0 = [segmented_path_length(make_transfer_nodes(self.robot_start, self.metrics.approach_points[i])) for i in range(self.n)]
            k = max(2, min(5, self.n))
            start = random.choice(np.argsort(d0)[:k].tolist())
        else:
            start = random.randrange(self.n)
        path = [start]
        unvisited = set(range(self.n))
        unvisited.remove(start)
        current = start
        while unvisited:
            nodes, probs = self._transition_probs(current, unvisited)
            next_node = int(np.random.choice(nodes, p=probs))
            path.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return path

    def _nearest_initial(self):
        # 内部生成最近邻路径，不调用 NearestNeighbor.solve()，避免额外独立评估预算。
        unvisited = set(range(self.n))
        path = []
        current = self.robot_start
        aps = self.metrics.approach_points
        while unvisited:
            node = min(unvisited, key=lambda k: segmented_path_length(make_transfer_nodes(current, aps[k])))
            path.append(node)
            unvisited.remove(node)
            current = aps[node]
        return path

    def _boost_pheromone(self, path, fit, strength=5.0):
        if path is None or len(path) < 2:
            return
        delta = strength * ALGO.aco_q / (fit + 1e-8)
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            self.pheromone[a, b] += delta
            self.pheromone[b, a] += 0.2 * delta
        self.pheromone = np.clip(self.pheromone, ALGO.pheromone_min, ALGO.pheromone_max)

    def _update_pheromone(self, paths_fits):
        self.pheromone *= (1 - ALGO.aco_rho)
        if not paths_fits:
            return
        paths_fits = sorted(paths_fits, key=lambda x: x[1])
        elite_count = max(1, int(len(paths_fits) * ALGO.aco_elite_ratio))
        for path, fit in paths_fits[:elite_count]:
            delta = ALGO.aco_q / (fit + 1e-8)
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                self.pheromone[a, b] += delta
                self.pheromone[b, a] += 0.2 * delta
        self.pheromone = np.clip(self.pheromone, ALGO.pheromone_min, ALGO.pheromone_max)

    def _local_improve_one(self, path, max_trials, max_rounds, shuffle=True):
        if (not self.use_two_opt) or (not self.evaluator.has_budget(2)):
            try:
                fit, length, _, _ = self.evaluator.evaluate(path)
                return path, fit, length
            except EvaluationBudgetExceeded:
                return path, float('inf'), self.metrics.compute_length(path)
        before = self.evaluator.evals
        mode = "constraint" if self.use_constraint_aware else "length"
        new_path, mt = self.local.improve(path, mode=mode, max_trials=max_trials, max_rounds=max_rounds, shuffle=shuffle)
        self.local_search_evals += max(0, self.evaluator.evals - before)
        if mt is not None:
            new_fit, new_len, _, _ = mt
            return new_path, new_fit, new_len
        try:
            fit, length, _, _ = self.evaluator.evaluate(path)
            return path, fit, length
        except EvaluationBudgetExceeded:
            return path, float('inf'), self.metrics.compute_length(path)

    def _local_improve_batch(self, batch):
        if (not self.use_two_opt) or not batch:
            return batch
        improved_batch = []
        sorted_batch = sorted(batch, key=lambda x: x[1])
        top_k = min(ALGO.local_search_top_k, len(sorted_batch))
        for idx, (path, fit, length) in enumerate(sorted_batch):
            if idx < top_k and self.evaluator.has_budget(2):
                # 第一名深度搜索，其余轻量搜索，兼顾质量和预算。
                trials = ALGO.local_search_trials_deep if self.deep_local and idx == 0 else ALGO.local_search_trials_light
                rounds = ALGO.local_search_rounds if self.deep_local and idx == 0 else 1
                new_path, new_fit, new_len = self._local_improve_one(path, trials, rounds, shuffle=True)
                improved_batch.append((new_path, new_fit, new_len))
            else:
                improved_batch.append((path, fit, length))
        return improved_batch

    def solve(self):
        t0 = time.perf_counter()
        if self.n <= 1:
            path = list(range(self.n))
            fit, length, _, _ = self.evaluator.evaluate(path)
            return self._final_stats(path, fit, length, t0)

        best_path, best_fit, best_len = None, float("inf"), 0.0

        # 1) NN + 2OPT，作为强保底精英。
        if self.use_dual_source and self.evaluator.has_budget(2):
            nn_path = self._nearest_initial()
            # 给 NN+2OPT 分配一部分预算，但不耗尽全部预算，保留 ACO 探索空间。
            reserve_for_aco = max(300, int(ALGO.reserve_for_aco_ratio * UNIFIED_MAX_EVALS))
            nn_budget = max(80, self.evaluator.max_evals - reserve_for_aco)
            nn_trials = min(nn_budget, max(80, UNIFIED_MAX_EVALS // 2))
            nn_path, nn_fit, nn_len = self._local_improve_one(
                nn_path,
                max_trials=nn_trials,
                max_rounds=999,
                shuffle=False,
            )
            best_path, best_fit, best_len = nn_path.copy(), nn_fit, nn_len
            self.convergence.append(best_len)
            if self.use_pheromone_boost:
                self._boost_pheromone(best_path, best_fit, strength=6.0)

        # 未启用启发式随机初始化机制时，采用ACO随机生成初始候选路径。
        if best_path is None and self.evaluator.has_budget(1):
            init_path = self._construct_ant_path()
            best_fit, best_len, _, _ = self.evaluator.evaluate(init_path)
            best_path = init_path.copy()
            self.convergence.append(best_len)

        # 2)ACO 生成多样化候选，对优质候选执行 2OPT，再与保底精英竞争。
        ants = min(ALGO.aco_ants, max(2, UNIFIED_MAX_EVALS // 10))
        while self.evaluator.has_budget(1):
            batch = []
            for _ in range(ants):
                if not self.evaluator.has_budget(1):
                    break
                path = self._construct_ant_path()
                fit, length, _, _ = self.evaluator.evaluate(path)
                batch.append((path, fit, length))

            if not batch:
                break

            batch = self._local_improve_batch(batch)

            # 只有启用精英信息素增强时，才让保底精英参与信息素更新。
            # A3引入启发式随机初始化，A4在此基础上进一步引入精英信息素增强。
            if best_path is not None and self.use_pheromone_boost:
                batch_for_update = batch + [(best_path, best_fit, best_len)]
            else:
                batch_for_update = batch

            self.fitness_func.update_weights([p for p, _, _ in batch_for_update], self.metrics)
            self._update_pheromone([(p, f) for p, f, _ in batch_for_update])

            for path, fit, length in batch:
                if fit < best_fit:
                    best_path, best_fit, best_len = path.copy(), fit, length
                    if self.use_pheromone_boost:
                        self._boost_pheromone(best_path, best_fit, strength=2.0)

            self.convergence.append(best_len)

        return self._final_stats(best_path, best_fit, best_len, t0)


# ============================================================
# 实验与统计
# ============================================================
def build_instance(config, instance_id):
    """
    构建测试实例。

    Generate a fixed instance with all sampled targets; collision costs are
    evaluated by the objective rather than used to remove target points.
    """
    pts, radii, obs = generate_random_instance(config["points"], config["obstacles"], instance_id + 1000)
    robot_start = get_robot_start()
    constraint = ConstraintViolation(obs, pts, radii)
    return pts, radii, obs, robot_start, constraint


def run_one_instance_seed_task(args):
    """
    多进程并行的单个完整实验任务。

    每个任务对应一个 config + instance + seed，内部运行该场景下的所有算法。
    """
    config, inst, seed, run_ablation = args

    pts, radii, obs, robot_start, cons = build_instance(config, inst)
    if len(pts) < 2:
        return None

    set_seed(seed + inst * 100)

    algos = {
        "Greedy": GreedyWithCollision(pts, robot_start, cons),
        "SA": SimulatedAnnealing(pts, robot_start, cons),
        "PSO": ParticleSwarmOptimization(pts, robot_start, cons),
        "GA": StandardGA(pts, robot_start, cons),
        "RL": ReinforcementLearningPlanner(pts, robot_start, cons),
        "DS-CA-ACO-2OPT": DualSourceCAACO2OPT(
            pts, robot_start, cons,
            use_constraint_aware=True,
            use_two_opt=True,
            use_dual_source=True,
            use_pheromone_boost=True,
        ),
    }

    if run_ablation:
        # 逐步叠加式消融实验：
        # A1 Plain ACO：普通 ACO；
        # A2 ACO-2OPT：普通 ACO + 2-OPT；
        # A3 DS-ACO-2OPT：A2 + 启发式随机初始化；
        # A4 DS-PB-ACO-2OPT：A3 + 精英信息素增强；
        # A5 DS-CA-ACO-2OPT：A4 + 约束感知。
        algos["A1 Plain ACO"] = PlainACO(
            pts, robot_start, cons, constraint_aware=False
        )
        algos["A2 ACO-2OPT"] = DualSourceCAACO2OPT(
            pts, robot_start, cons,
            use_constraint_aware=False,
            use_two_opt=True,
            use_dual_source=False,
            use_pheromone_boost=False,
        )
        algos["A3 DS-ACO-2OPT"] = DualSourceCAACO2OPT(
            pts, robot_start, cons,
            use_constraint_aware=False,
            use_two_opt=True,
            use_dual_source=True,
            use_pheromone_boost=False,
        )
        algos["A4 DS-PB-ACO-2OPT"] = DualSourceCAACO2OPT(
            pts, robot_start, cons,
            use_constraint_aware=False,
            use_two_opt=True,
            use_dual_source=True,
            use_pheromone_boost=True,
        )
        algos["A5 DS-CA-ACO-2OPT"] = DualSourceCAACO2OPT(
            pts, robot_start, cons,
            use_constraint_aware=True,
            use_two_opt=True,
            use_dual_source=True,
            use_pheromone_boost=True,
        )

    task_result = {}
    task_conv = {}

    for name, algo in algos.items():
        path, fit, length, _ = algo.solve()

        metrics = PathMetricsCalculator(pts, robot_start, cons)
        ffunc = UnifiedPenaltyFitness()
        _, c, sm, _ = metrics.compute_all_metrics(path)

        task_result[name] = {
            "length": length,
            "time": getattr(algo, "planning_time", 0.0),
            "feasible": 1 if ffunc.is_feasible(c, sm) else 0,
            "collision": c,
            "smoothness": sm,
            "z_motion": metrics.compute_z_motion(path),
            "evals": getattr(getattr(algo, "evaluator", None), "evals", np.nan),
            "local_evals": float(
                getattr(
                    algo,
                    "local_search_evals",
                    getattr(getattr(algo, "local", None), "used_evals", 0.0),
                )
            ),
        }

        if name in {"GA", "RL", "DS-CA-ACO-2OPT", "A1 Plain ACO", "A2 ACO-2OPT", "A3 DS-ACO-2OPT", "A4 DS-PB-ACO-2OPT", "A5 DS-CA-ACO-2OPT"} and hasattr(algo, "convergence") and algo.convergence:
            task_conv[name] = algo.convergence

    return {
        "config_name": config["name"],
        "inst": inst,
        "seed": seed,
        "results": task_result,
        "convergence": task_conv,
    }


class CompleteExperiment:
    def __init__(self):
        self.convergence_data = {}

    def run_single_config(self, config, run_ablation=False):
        print(f"\n{'='*60}")
        print(f"测试: {config['name']} (点数: {config['points']}, 障碍物: {config['obstacles']})")
        print(f"实例数: {N_INSTANCES}, 每实例种子数: {N_SEEDS}")
        print(f"统一计算预算: {UNIFIED_MAX_EVALS}次评估")
        print(f"{'='*60}")

        names = ["Greedy", "SA", "PSO", "GA", "RL", "DS-CA-ACO-2OPT"]
        results = {k: [] for k in names}
        times = {k: [] for k in names}
        feas = {k: [] for k in names}
        collisions = {k: [] for k in names}
        smoothness = {k: [] for k in names}
        z_motions = {k: [] for k in names}
        eval_counts = {k: [] for k in names}
        local_evals = {k: [] for k in names}
        conv = {"GA": [], "RL": [], "DS-CA-ACO-2OPT": []}

        if run_ablation:
            for k in ["A1 Plain ACO", "A2 ACO-2OPT", "A3 DS-ACO-2OPT", "A4 DS-PB-ACO-2OPT", "A5 DS-CA-ACO-2OPT"]:
                results[k] = []; times[k] = []; feas[k] = []
                collisions[k] = []; smoothness[k] = []; z_motions[k] = []; eval_counts[k] = []; local_evals[k] = []

        tasks = [
            (config, inst, seed, run_ablation)
            for inst in range(N_INSTANCES)
            for seed in range(N_SEEDS)
        ]

        def merge_payload(payload):
            if payload is None:
                return
            for name, d in payload["results"].items():
                results[name].append(d["length"])
                times[name].append(d["time"])
                feas[name].append(d["feasible"])
                collisions[name].append(d["collision"])
                smoothness[name].append(d["smoothness"])
                z_motions[name].append(d["z_motion"])
                eval_counts[name].append(d["evals"])
                local_evals[name].append(d["local_evals"])

            for name, curve in payload["convergence"].items():
                if name in conv and curve:
                    conv[name].append(curve)

        completed = 0

        if ENABLE_PARALLEL:
            print(f"并行模式: N_WORKERS={N_WORKERS}, 任务数={len(tasks)}")
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = [executor.submit(run_one_instance_seed_task, task) for task in tasks]
                for future in as_completed(futures):
                    payload = future.result()
                    merge_payload(payload)
                    completed += 1
                    if completed % max(1, PRINT_FREQ_INSTANCE * N_SEEDS) == 0 or completed == len(tasks):
                        print(f"  已完成 {completed}/{len(tasks)} 个 instance-seed 任务")
        else:
            print("串行模式运行")
            for task in tasks:
                payload = run_one_instance_seed_task(task)
                merge_payload(payload)
                completed += 1
                if completed % max(1, PRINT_FREQ_INSTANCE * N_SEEDS) == 0 or completed == len(tasks):
                    print(f"  已完成 {completed}/{len(tasks)} 个 instance-seed 任务")

        summary = {}
        for name, vals in results.items():
            if vals:
                arr = np.array(vals, dtype=float)
                summary[name] = {
                    "mean": float(np.mean(arr)), "std": float(np.std(arr)), "values": arr,
                    "time_mean": float(np.mean(times[name])),
                    "feasible_rate": float(np.mean(feas[name])),
                    "collision_mean": float(np.mean(collisions[name])) if collisions[name] else np.nan,
                    "smoothness_mean": float(np.mean(smoothness[name])) if smoothness[name] else np.nan,
                    "z_motion_mean": float(np.mean(z_motions[name])) if z_motions[name] else np.nan,
                    "evals_mean": float(np.nanmean(eval_counts[name])) if eval_counts[name] else np.nan,
                    "local_evals_mean": float(np.mean(local_evals[name])) if local_evals[name] else 0.0,
                }
        self.convergence_data[config["name"]] = conv
        self._print_summary(config["name"], summary)
        return summary

    @staticmethod
    def _cliff_delta(a, b):
        n1, n2 = len(a), len(b)
        if n1 * n2 == 0:
            return 0.0
        greater = sum(1 for x in a for y in b if x > y)
        less = sum(1 for x in a for y in b if x < y)
        return (greater - less) / (n1 * n2)

    def _safe_wilcoxon(self, a, b):
        try:
            if len(a) != len(b):
                m = min(len(a), len(b)); a = a[:m]; b = b[:m]
            if np.allclose(a, b):
                return 1.0
            _, p = wilcoxon(a, b)
            return float(p)
        except Exception:
            return 1.0

    def _print_summary(self, cname, summary):
        print(f"\n结果汇总 ({cname}):")
        print("  评价指标: 平均路径长度±标准差(mm), 平滑违反度, Z轴累计运动量(mm), 平均规划时间(s)")
        for name, d in summary.items():
            print(f"  {name}: 长度={d['mean']:.0f}±{d['std']:.0f}mm, "
                  f"平滑={d['smoothness_mean']:.3f}, "
                  f"Z轴={d['z_motion_mean']:.0f}mm, "
                  f"规划时间={d['time_mean']:.3f}s")

        if "GA" in summary and "DS-CA-ACO-2OPT" in summary:
            ga = summary["GA"]["values"]; main = summary["DS-CA-ACO-2OPT"]["values"]
            p = self._safe_wilcoxon(main, ga)
            imp = (np.mean(ga) - np.mean(main)) / (np.mean(ga) + 1e-8) * 100
            print("\n统计检验:")
            print(f"  DS-CA-ACO-2OPT vs GA: p={p:.6f}, Cliff's δ={self._cliff_delta(main, ga):.3f}, 改善={imp:.1f}%")
        if "2-OPT" in summary and "DS-CA-ACO-2OPT" in summary:
            opt = summary["2-OPT"]["values"]; main = summary["DS-CA-ACO-2OPT"]["values"]
            p = self._safe_wilcoxon(main, opt)
            imp = (np.mean(opt) - np.mean(main)) / (np.mean(opt) + 1e-8) * 100
            print(f"  DS-CA-ACO-2OPT vs 2-OPT: p={p:.6f}, 相对2-OPT改善={imp:.1f}%")
        if "ACO" in summary and "DS-CA-ACO-2OPT" in summary:
            aco = summary["ACO"]["values"]; main = summary["DS-CA-ACO-2OPT"]["values"]
            p = self._safe_wilcoxon(main, aco)
            imp = (np.mean(aco) - np.mean(main)) / (np.mean(aco) + 1e-8) * 100
            print(f"  DS-CA-ACO-2OPT vs ACO: p={p:.6f}, 相对ACO改善={imp:.1f}%")

    def run_all(self):
        all_results = {}
        for config in TEST_CONFIGS:
            all_results[config["name"]] = self.run_single_config(config, run_ablation=(config["name"] == "Medium"))
        return all_results

    def plot_results(self, all_results):
        if not all_results:
            return None
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        configs = list(all_results.keys())
        algos = ["Greedy", "SA", "PSO", "GA", "RL", "DS-CA-ACO-2OPT"]
        label_map = {"RL": "强化学习", "DS-CA-ACO-2OPT": "本文算法"}
        x = np.arange(len(configs)); width = 0.12

        ax = axes[0, 0]
        for i, algo in enumerate(algos):
            means = [all_results[c].get(algo, {}).get("mean", 0) for c in configs]
            stds = [all_results[c].get(algo, {}).get("std", 0) for c in configs]
            if any(m > 0 for m in means):
                ax.bar(x + (i - 2.5) * width, means, width, yerr=stds, capsize=2, label=label_map.get(algo, algo), alpha=0.85)
        ax.set_title("(a) 路径长度对比")
        ax.set_xlabel("问题规模"); ax.set_ylabel("路径长度(mm)")
        ax.set_xticks(x); ax.set_xticklabels(configs)
        ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3, axis="y")

        ax = axes[0, 1]
        compare_algos = ["Greedy", "SA", "PSO", "RL", "DS-CA-ACO-2OPT"]
        width2 = 0.14
        for i, algo in enumerate(compare_algos):
            imps = []
            for c in configs:
                ga = all_results[c].get("GA", {}).get("mean", np.nan)
                val = all_results[c].get(algo, {}).get("mean", np.nan)
                imps.append((ga - val) / ga * 100 if ga and not np.isnan(val) else 0)
            ax.bar(x + (i - 2) * width2, imps, width2, label=f"{label_map.get(algo, algo)} vs GA")
        ax.axhline(0, color="gray", linestyle="--")
        ax.set_title("(b) 相对GA改善率"); ax.set_ylabel("改善率(%)")
        ax.set_xticks(x); ax.set_xticklabels(configs); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1, 0]
        if "Medium" in self.convergence_data:
            for algo in ["GA", "RL", "DS-CA-ACO-2OPT"]:
                curves = self.convergence_data["Medium"].get(algo, [])
                if curves:
                    ml = min(len(c) for c in curves)
                    arr = np.array([c[:ml] for c in curves], dtype=float)
                    ax.plot(range(ml), arr.mean(axis=0), label=label_map.get(algo, algo), linewidth=2)
                    ax.fill_between(range(ml), arr.mean(axis=0)-arr.std(axis=0), arr.mean(axis=0)+arr.std(axis=0), alpha=0.2)
            ax.set_title("(c) 收敛曲线(Medium)")
            ax.set_xlabel("迭代/更新次数"); ax.set_ylabel("最优路径长度(mm)")
            ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        data, labels = [], []
        # 多算法对比为 Greedy、SA、PSO、GA、RL 和本文算法，
        # 这里以 Medium 规模为代表绘制六种算法的结果分布。
        medium_res = all_results.get("Medium", {})
        for algo in algos:
            if algo in medium_res:
                data.append(medium_res[algo]["values"])
                labels.append(label_map.get(algo, algo))
        if data:
            ax.boxplot(data, labels=labels, showmeans=True, meanline=True)
            ax.set_title("(d) Medium规模结果分布")
            ax.set_ylabel("路径长度(mm)")
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        return fig


def save_results(all_results, filename="experiment_results_ablation_metrics.csv"):
    rows = []
    for cfg, res in all_results.items():
        for algo, d in res.items():
            rows.append({
                "配置": cfg,
                "算法": algo,
                "平均路径长度(mm)": d["mean"],
                "路径长度标准差(mm)": d["std"],
                "平均平滑违反度": d.get("smoothness_mean", np.nan),
                "平均Z轴累计运动量(mm)": d.get("z_motion_mean", np.nan),
                "平均规划时间(s)": d.get("time_mean", np.nan),
            })
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"\n结果已保存到 {filename}")



def _path_to_approach_traj(path, points, robot_start):
    """用于论文绘图：绘制与指标一致的真实执行轨迹节点。"""
    if not path:
        return np.array([robot_start], dtype=float)
    return np.array(make_execution_nodes(path, points, robot_start, radii=None, return_start=True), dtype=float)


def _build_paper_plot_algorithms(pts, robot_start, cons):
    """论文路径图使用的六种对比算法。"""
    return {
        "Greedy": GreedyWithCollision(pts, robot_start, cons),
        "SA": SimulatedAnnealing(pts, robot_start, cons),
        "PSO": ParticleSwarmOptimization(pts, robot_start, cons),
        "GA": StandardGA(pts, robot_start, cons),
        "RL": ReinforcementLearningPlanner(pts, robot_start, cons),
        "DS-CA-ACO-2OPT": DualSourceCAACO2OPT(
            pts, robot_start, cons,
            use_constraint_aware=True,
            use_two_opt=True,
            use_dual_source=True,
            use_pheromone_boost=True,
        ),
    }


def generate_medium_path_plots(instance_id=0, seed=0, out_prefix="medium_path_planning"):
    """
    输出 Medium 场景下 Greedy、SA、PSO、GA、强化学习和本文算法的二维/三维采摘路径规划图。

    生成文件：
    1. medium_path_planning_2d.png / .svg
    2. medium_path_planning_3d.png / .svg
    3. medium_path_planning_metrics.csv
    """
    print("\n" + "=" * 80)
    print("生成 Medium 场景下不同算法二维/三维采摘路径规划图")
    print("=" * 80)

    config = next(c for c in TEST_CONFIGS if c["name"] == "Medium")
    pts, radii, obs, robot_start, cons = build_instance(config, instance_id)
    if len(pts) < 2:
        print("Medium绘图实例可采目标数量不足，跳过路径图输出。")
        return

    set_seed(seed + instance_id * 100)
    algos = _build_paper_plot_algorithms(pts, robot_start, cons)
    label_map = {
        "Greedy": "Greedy",
        "SA": "SA",
        "PSO": "PSO",
        "GA": "GA",
        "RL": "强化学习",
        "DS-CA-ACO-2OPT": "本文算法",
    }

    solved = {}
    rows = []
    for name, algo in algos.items():
        path, fit, length, _ = algo.solve()
        metrics = PathMetricsCalculator(pts, robot_start, cons)
        _, coll, smooth, _ = metrics.compute_all_metrics(path)
        z_motion = metrics.compute_z_motion(path)
        solved[name] = {
            "path": path,
            "traj": _path_to_approach_traj(path, pts, robot_start),
            "length": length,
            "smoothness": smooth,
            "collision": coll,
            "z_motion": z_motion,
            "time": getattr(algo, "planning_time", 0.0),
        }
        rows.append({
            "算法": label_map.get(name, name),
            "路径顺序": "-".join(map(str, path)),
            "路径长度(mm)": length,
            "碰撞违反度": coll,
            "平滑违反度": smooth,
            "Z轴累计运动量(mm)": z_motion,
            "规划时间(s)": getattr(algo, "planning_time", 0.0),
        })
        print(f"  {label_map.get(name, name)}: 长度={length:.0f}mm, 平滑={smooth:.3f}, Z轴={z_motion:.0f}mm, 时间={getattr(algo, 'planning_time', 0.0):.3f}s")

    pd.DataFrame(rows).to_csv(f"{out_prefix}_metrics.csv", index=False, encoding="utf-8-sig")

    pts_arr = np.array(pts, dtype=float)
    obs_arr = np.array(obs, dtype=float) if len(obs) else np.empty((0, 4))

    # 二维XY路径图
    fig2d, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    for ax, (name, d) in zip(axes, solved.items()):
        traj = d["traj"]
        ax.scatter(pts_arr[:, 0], pts_arr[:, 1], s=28, marker="o", label="采摘目标")
        if len(obs_arr):
            ax.scatter(obs_arr[:, 0], obs_arr[:, 1], s=18, marker="x", label="障碍物")
        ax.plot(traj[:, 0], traj[:, 1], marker=".", linewidth=1.8, label="规划路径")
        ax.scatter([robot_start[0]], [robot_start[1]], s=60, marker="*", label="起点")
        for order, node in enumerate(d["path"], 1):
            ap = get_approach_point(pts[node])
            ax.text(ap[0], ap[1], str(order), fontsize=8)
        ax.set_title(f"{label_map.get(name, name)}\nL={d['length']:.0f} mm, Z={d['z_motion']:.0f} mm")
        ax.set_xlabel("X/mm")
        ax.set_ylabel("Y/mm")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")
    handles, labels = axes[0].get_legend_handles_labels()
    fig2d.legend(handles, labels, loc="upper center", ncol=4)
    fig2d.suptitle("Medium规模下不同算法二维采摘路径规划结果", y=0.98)
    fig2d.tight_layout(rect=[0, 0, 1, 0.94])
    fig2d.savefig(f"{out_prefix}_2d.png", dpi=300, bbox_inches="tight")
    fig2d.savefig(f"{out_prefix}_2d.svg", bbox_inches="tight")
    plt.close(fig2d)

    # 三维路径图
    fig3d = plt.figure(figsize=(18, 10))
    for idx, (name, d) in enumerate(solved.items(), 1):
        ax = fig3d.add_subplot(2, 3, idx, projection="3d")
        traj = d["traj"]
        ax.scatter(pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2], s=24, marker="o", label="采摘目标")
        if len(obs_arr):
            ax.scatter(obs_arr[:, 0], obs_arr[:, 1], obs_arr[:, 2], s=18, marker="x", label="障碍物")
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], marker=".", linewidth=1.8, label="规划路径")
        ax.scatter([robot_start[0]], [robot_start[1]], [robot_start[2]], s=60, marker="*", label="起点")
        for order, node in enumerate(d["path"], 1):
            ap = get_approach_point(pts[node])
            ax.text(ap[0], ap[1], ap[2], str(order), fontsize=7)
        ax.set_title(f"{label_map.get(name, name)}\nL={d['length']:.0f} mm, Z={d['z_motion']:.0f} mm")
        ax.set_xlabel("X/mm")
        ax.set_ylabel("Y/mm")
        ax.set_zlabel("Z/mm")
        ax.view_init(elev=24, azim=-55)
    handles, labels = fig3d.axes[0].get_legend_handles_labels()
    fig3d.legend(handles, labels, loc="upper center", ncol=4)
    fig3d.suptitle("Medium规模下不同算法三维采摘路径规划结果", y=0.98)
    fig3d.tight_layout(rect=[0, 0, 1, 0.94])
    fig3d.savefig(f"{out_prefix}_3d.png", dpi=300, bbox_inches="tight")
    fig3d.savefig(f"{out_prefix}_3d.svg", bbox_inches="tight")
    plt.close(fig3d)

    print(f"路径规划图已保存: {out_prefix}_2d.png/.svg, {out_prefix}_3d.png/.svg")
    print(f"路径图指标已保存: {out_prefix}_metrics.csv")


# ============================================================
# 正式仿真实验
# ============================================================
def main():
    print("DS-CA-ACO-2OPT | Formal simulation")
    print(f"Problem scales: {[c['name'] for c in TEST_CONFIGS]}")
    print(f"Instances: {N_INSTANCES}, seeds: {N_SEEDS}")
    print(f"Max fitness evaluations: {UNIFIED_MAX_EVALS}")

    experiment = CompleteExperiment()
    results = experiment.run_all()

    save_results(results)

    try:
        figure = experiment.plot_results(results)
        if figure is not None:
            figure.savefig(
                "performance_comparison.png",
                dpi=300,
                bbox_inches="tight"
            )
            figure.savefig(
                "performance_comparison.svg",
                bbox_inches="tight"
            )
            plt.close(figure)

        generate_medium_path_plots(
            instance_id=0,
            seed=0,
            out_prefix="medium_path_planning"
        )

    except Exception as exc:
        print(f"Warning: plot generation failed: {exc}")

    print("Formal simulation complete.")


if __name__ == "__main__":
    main()


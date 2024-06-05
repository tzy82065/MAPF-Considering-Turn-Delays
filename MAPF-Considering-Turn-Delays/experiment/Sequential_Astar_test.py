import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import random
import threading
import time
import pandas as pd

show_animation = False


class TimeoutException(Exception):
    pass


def run_with_timeout(func, args=(), kwargs={}, timeout_duration=10):
    # 创建一个列表来保存函数的执行结果
    result_container = [None]

    def wrapper():
        try:
            # 执行函数并将结果存储
            result_container[0] = func(*args, **kwargs)
        except Exception as e:
            result_container[0] = e

    # 创建线程来运行函数
    thread = threading.Thread(target=wrapper)
    # 启动线程
    thread.start()
    # 等待线程完成或超时
    thread.join(timeout_duration)

    # 如果线程仍在运行，我们认定为超时
    if thread.is_alive():
        # 可以在这里处理线程的停止，但在Python中没有安全的停止线程的方法
        # 所以我们只标记为超时并返回None
        return None
    else:
        # 如果线程没有超时，返回函数执行的结果
        return result_container[0]


def show_map(ox, oy, sx, sy, gx, gy, plan_x, plan_y):
    # 显示地图
    fig, ax = plt.subplots()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([x + 0.5 for x in range(int(min(ox) - 1), int(max(ox) + 1))])
    ax.set_yticks([y + 0.5 for y in range(int(min(oy) - 1), int(max(oy) + 1))])
    ax.grid(which='both')
    plt.axis('equal')
    # 显示地图特征、起点、终点、规划的路线
    plt.plot(ox, oy, 'sk', markersize=10)
    plt.plot(sx, sy, 'sr')
    plt.plot(gx, gy, 'sb')
    for plan_index in range(len(plan_x)):
        plt.plot(plan_x[plan_index], plan_y[plan_index])
    plt.show()


# __________路径规划算法类__________
class AStar:
    # 初始化
    def __init__(self, ox, oy, resolution, agent_radius):
        # 属性分配
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None
        self.x_width = None
        self.y_width = None
        self.obstacle_map = None
        self.resolution = resolution  # 网格大小
        self.agent_radius = agent_radius  # 智能体半径
        self.calc_obstacle_map(ox, oy)  # 绘制栅格地图
        self.motion = self.get_motion_model()  # 智能体运动方式

    # 构建节点，每个网格代表一个节点
    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # 网格索引
            self.y = y
            self.cost = cost  # 路径代价
            self.parent_index = parent_index  # 该网格的父节点

        def __str__(self):
            return str(self.x) + ',' + str(self.y) + ',' + str(self.cost) + ',' + str(self.parent_index)

    # 寻找最优路径，网格起始坐标(sx,sy)，终点坐标（gx,gy）
    def planning(self, sx, sy, gx, gy):
        # 节点初始化
        # 将已知的起点和终点坐标形式转化为节点类型，0代表路径权重，-1代表无父节点
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        # 终点
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)
        # 保存入库节点和待计算节点
        open_set, closed_set = dict(), dict()
        # 先将起点入库，获取每个网格对应的key
        open_set[self.calc_index(start_node)] = start_node

        # 循环
        while True:
            # 使用启发函数f(n)=g(n)+h(n)搜索新节点
            c_id = min(open_set,
                       key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))

            current = open_set[c_id]  # 从字典中取出该节点

            # 搜索动画可视化
            if show_animation:
                # 网格索引转换为真实坐标
                plt.plot(self.calc_position(current.x, self.min_x),
                         self.calc_position(current.y, self.min_y), '*y')
                plt.pause(0.0001)

            # 判断是否是终点，如果选出来的损失最小的点是终点
            if current.x == goal_node.x and current.y == goal_node.y:
                # 更新终点的父节点
                goal_node.cost = current.cost
                # 更新终点的损失
                goal_node.parent_index = current.parent_index
                break

            # 在open_set中删除该最小代价点，把它入库
            del open_set[c_id]
            closed_set[c_id] = current

            # 遍历邻接节点
            for move_x, move_y, move_cost in self.motion:
                # 获取每个邻接节点的节点坐标和到起点的距离
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)
                # 获取该邻居节点的key
                n_id = self.calc_index(node)

                # 如果该节点入库了，检查下一个
                if n_id in closed_set:
                    continue

                # 判断邻居节点是否超出范围了，是否在障碍物上
                if not self.verify_node(node):
                    continue

                # 如果该节点不在open_set中，就作为一个新节点加入open_set
                if n_id not in open_set:
                    open_set[n_id] = node
                # 节点在open_set中时
                else:
                    # 如果该点到起点的距离小于open_set当前点到该点的距离，就更新open_set中的该点信息，更改路径
                    if node.cost <= open_set[n_id].cost:
                        open_set[n_id] = node

        # 直到找到终点，返回终点和closed_set中的路径坐标
        rx, ry = self.calc_final_path(goal_node, closed_set)
        rx_rev = rx[::-1]
        ry_rev = ry[::-1]
        return rx_rev, ry_rev

    # A* 的启发函数
    @staticmethod
    def calc_heuristic(n1, n2):  # n1为终点，n2为当前节点
        w = 1  # 启发函数的权重
        manhattan_d = w * (abs(n1.x - n2.x) + abs(n1.y - n2.y))  # 当前网格和终点的曼哈顿距离
        return manhattan_d

    # 智能体运动模式，经典MAPF允许4向移动
    @staticmethod
    def get_motion_model():
        # [dx, dy, cost]
        motion = [[1, 0, 1],  # 右
                  [0, 1, 1],  # 上
                  [-1, 0, 1],  # 左
                  [0, -1, 1]]  # 下
        return motion

    # 绘制栅格地图
    def calc_obstacle_map(self, ox, oy):
        # 地图边界坐标
        self.min_x = round(min(ox))  # 四舍五入取整
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        # 地图的x和y方向的栅格个数，长度/每个网格的长度=网格个数
        self.x_width = round((self.max_x - self.min_x) / self.resolution)  # x方向网格个数
        self.y_width = round((self.max_y - self.min_y) / self.resolution)  # y方向网格个数
        # 初始化地图，二维列表，每个网格的值为False
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        # 设置障碍物
        for ix in range(self.x_width):  # 遍历x方向的网格 [0:x_width]
            x = self.calc_position(ix, self.min_x)  # 根据网格索引计算x坐标位置
            for iy in range(self.y_width):  # 遍历y方向的网格 [0:y_width]
                y = self.calc_position(iy, self.min_y)  # 根据网格索引计算y坐标位置
                # 遍历障碍物坐标(实际坐标)
                for iox, ioy in zip(ox, oy):
                    # 计算障碍物和网格点之间的距离
                    d = math.hypot(iox - x, ioy - y)
                    # 膨胀障碍物，如果障碍物和网格之间的距离小，机器人无法通行，对障碍物膨胀
                    if d <= self.agent_radius:
                        # 将障碍物所在网格设置为True
                        self.obstacle_map[ix][iy] = True
                        break  # 每个障碍物膨胀一次就足够了

    # 根据网格编号计算实际坐标
    def calc_position(self, index, minp):
        # minp代表起点坐标，左下x或左下y
        pos = minp + index * self.resolution  # 网格点左下左下坐标
        return pos

    # 位置坐标转为网格坐标
    def calc_xy_index(self, position, minp):
        # (目标位置坐标-起点坐标)/一个网格的长度==>目标位置的网格索引
        return round((position - minp) / self.resolution)

    # 给每个网格编号，得到每个网格的key
    def calc_index(self, node):
        # 从左到右增大，从下到上增大
        return node.y * self.x_width + node.x

    # 邻居节点是否超出范围
    def verify_node(self, node):
        # 根据网格坐标计算实际坐标
        px = self.calc_position(node.x, self.min_x)
        py = self.calc_position(node.y, self.min_y)
        # 判断是否超出边界
        if px < self.min_x:
            return False
        if py < self.min_y:
            return False
        if px >= self.max_x:
            return False
        if py >= self.max_y:
            return False
        # 节点是否在障碍物上，障碍物标记为True
        if self.obstacle_map[node.x][node.y]:
            return False
        # 没超过就返回True
        return True

    # 计算路径, parent属性记录每个节点的父节点
    def calc_final_path(self, goal_node, closed_set):
        # 先存放终点坐标（真实坐标）
        rx = [self.calc_position(goal_node.x, self.min_x)]
        ry = [self.calc_position(goal_node.y, self.min_y)]
        # 获取终点的父节点索引
        parent_index = goal_node.parent_index
        # 起点的父节点==-1
        while parent_index != -1:
            n = closed_set[parent_index]  # 在入库中选择父节点
            rx.append(self.calc_position(n.x, self.min_x))  # 节点的x坐标
            ry.append(self.calc_position(n.y, self.min_y))  # 节点的y坐标
            parent_index = n.parent_index  # 节点的父节点索引

        return rx, ry


def add_rotate_motion(rx, ry):
    # 一条路径经过三个或更多顶点才查找转向动作
    for j in range(len(rx)):
        if len(rx[j]) < 3 or len(ry[j]) < 3:
            continue

        i = 0
        while i < len(rx[j]) - 2:
            # 发生90°转向，在转弯处添加一个timestep的等待动作
            if (rx[j][i] == rx[j][i + 1] and abs(ry[j][i + 1] - ry[j][i]) == 1 and ry[j][i + 1] == ry[j][i + 2] and abs(
                    rx[j][i + 2] - rx[j][i + 1]) == 1) \
                    or (ry[j][i] == ry[j][i + 1] and abs(rx[j][i + 1] - rx[j][i]) == 1 and
                        rx[j][i + 1] == rx[j][i + 2] and abs(ry[j][i + 2] - ry[j][i + 1]) == 1):
                rx[j].insert(i + 1, rx[j][i + 1])
                ry[j].insert(i + 1, ry[j][i + 1])
                i += 2
            # 发生180°转向，在转弯处添加两个timestep的等待动作
            elif rx[j][i] == rx[j][i + 2] and ry[j][i] == ry[j][i + 2]:
                rx[j].insert(i + 1, rx[j][i + 1])
                rx[j].insert(i + 1, rx[j][i + 1])
                ry[j].insert(i + 1, ry[j][i + 1])
                ry[j].insert(i + 1, ry[j][i + 1])
                i += 3
            else:
                i += 1

    return rx, ry


# 设置地图大小及障碍物，设置缓冲区
def create_void_map(size):
    ox = []
    oy = []
    # 设置下边界
    for i in range(0, size + 2):
        ox.append(i)
        oy.append(0)
    # 设置右边界
    for i in range(0, size + 1):
        ox.append(size + 1)
        oy.append(i)
    # 设置左边界
    for i in range(0, size + 1):
        ox.append(0)
        oy.append(i)
    # 设置上边界
    for i in range(0, size + 2):
        ox.append(i)
        oy.append(size + 1)

    return ox, oy


def create_warehouse_map():
    ox = []
    oy = []
    # 设置下边界
    for i in range(0, 19):
        ox.append(i)
        oy.append(0)
    # 设置右边界
    for i in range(0, 14):
        ox.append(18)
        oy.append(i)
    # 设置左边界
    for i in range(0, 14):
        ox.append(0)
        oy.append(i)
    # 设置上边界
    for i in range(0, 19):
        ox.append(i)
        oy.append(14)
    # 设置额外障碍（执行区域）
    for i in range(4, 9):
        for j in range(2, 5):
            ox.append(i)
            oy.append(j)
    for i in range(10, 15):
        for j in range(2, 5):
            ox.append(i)
            oy.append(j)
    for i in range(4, 9):
        for j in range(6, 9):
            ox.append(i)
            oy.append(j)
    for i in range(10, 15):
        for j in range(6, 9):
            ox.append(i)
            oy.append(j)
    for i in range(4, 9):
        for j in range(10, 13):
            ox.append(i)
            oy.append(j)
    for i in range(10, 15):
        for j in range(10, 13):
            ox.append(i)
            oy.append(j)

    return ox, oy


def create_large_warehouse_map():
    ox = []
    oy = []
    # 设置下边界
    for i in range(0, 162):
        ox.append(i)
        oy.append(0)
    # 设置右边界
    for i in range(0, 62):
        ox.append(161)
        oy.append(i)
    # 设置左边界
    for i in range(0, 62):
        ox.append(0)
        oy.append(i)
    # 设置上边界
    for i in range(0, 162):
        ox.append(i)
        oy.append(62)
    # 设置额外障碍（执行区域）
    long, wide = 10, 2
    for ix in range(0, 10):
        x = 11 * ix + 26
        for iy in range(0, 20):
            h = iy * 3 + 2
            for i in range(x, x + long):
                for j in range(h, h + wide):
                    ox.append(i)
                    oy.append(j)

    return ox, oy


class Node:
    def __init__(self, node_id, ox_k, oy_k, plan_x, plan_y):
        self.node_id = node_id
        self.ox_k = ox_k
        self.oy_k = oy_k
        self.plan_x = plan_x
        self.plan_y = plan_y
        self.children = []

    def __repr__(self):
        return (f"Node({self.node_id},"
                f"plan_x: {self.plan_x}, plan_y: {self.plan_y})")

    def add_child(self, child):
        self.children.append(child)

    def print_tree(self, level=0):
        indent = " "
        print(
            f"{indent}Node({self.node_id},"
            f"plan_x: {self.plan_x}, plan_y: {self.plan_y})")
        for child in self.children:
            child.print_tree(level + 1)  # 递归调用打印子节点


def simulator(ox, oy, sx, sy, gx, gy, plan_x, plan_y, update_time, if_show_map):
    # 显示地图
    if if_show_map:
        fig, ax = plt.subplots()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([x + 0.5 for x in range(int(min(ox) - 1), int(max(ox) + 1))])
        ax.set_yticks([y + 0.5 for y in range(int(min(oy) - 1), int(max(oy) + 1))])
        ax.grid(which='both')
        plt.axis('equal')
        # 显示地图特征
        plt.plot(ox, oy, 'sk', markersize=10)

        colors = [np.random.rand(3, ) for _ in range(len(plan_x))]
        # 绘制起点、终点和规划的路线
        for i in range(len(sx)):
            # 绘制起点
            plt.plot(sx[i], sy[i], 's', color=colors[i])
            # 绘制终点
            plt.plot(gx[i], gy[i], '*', color=colors[i])
            # 绘制路径
            plt.plot(plan_x[i], plan_y[i], color=colors[i])
        # 智能体沿规划路径运动仿真
        max_path_length = max(len(path) for path in plan_x)
        plan_points = []
        available_plan = True
        for i in range(max_path_length):
            if i == 0:
                detect_swapping = False
            else:
                detect_swapping = True
            if not available_plan:
                break
            # 清除上一时间步绘制的点
            for plan_point in plan_points:
                plan_point.remove()
            plan_points = []
            points_current_timestep = []
            for j in range(len(plan_x)):
                if j == len(plan_x):
                    detect_swapping = False
                if i < len(plan_x[j]):
                    plan_point, = ax.plot(plan_x[j][i], plan_y[j][i], 'o', color=colors[j], markersize=10)
                    plan_points.append(plan_point)
                    points_current_timestep.append((plan_x[j][i], plan_y[j][i]))
                # 错误检测
                # 检查智能体初始时刻是否从给定的起点出发
                if plan_x[j][0] != sx[j] or plan_y[j][0] != sy[j]:
                    available_plan = False
                    print('Not an available plan (Unexpected start)')
                # 如果与障碍/边界重合，则路径规划不可行
                if any(plan_x[j][i] == ox[k] and plan_y[j][i] == oy[k] for k in range(len(ox))):
                    available_plan = False
                    print('Not an available plan (reached obstacle)')
                    break
                # 如果与超出边界，则路径规划不可行
                if plan_x[j][i] > max(ox) or plan_y[j][i] > max(oy) or plan_x[j][i] < min(ox) or plan_y[j][i] < min(oy):
                    available_plan = False
                    print('Not an available plan (reached obstacle)')
                    break
                # 如果在仿真过程中发现智能体在一个timestep内到达非相邻点，则路径规划不可行
                if i > 0 and not ((plan_x[j][i] == plan_x[j][i - 1] and abs(plan_y[j][i] - plan_y[j][i - 1]) == 1) or
                                  (plan_y[j][i] == plan_y[j][i - 1] and abs(plan_x[j][i] - plan_x[j][i - 1]) == 1) or
                                  (plan_x[j][i] == plan_x[j][i - 1] and plan_y[j][i] == plan_y[j][i - 1])):
                    available_plan = False
                    print('Not an available plan (Unreasonable movement)')
                    break
                # 检查智能体最后是否到达给定的目标点
                if i == len(plan_x[0]) and (plan_x[j][-1] != gx[j] or plan_y[j][-1] != gy[j]):
                    available_plan = False
                    print('Not an available plan (Not arriving at the destination)')
                    break
                # 检查各智能体在当前timestep下是否存在交换冲突
                if detect_swapping is True:
                    for k in range(j + 1, len(plan_x)):
                        if (plan_x[j][i - 1] == plan_x[k][i] and plan_x[j][i] == plan_x[k][i - 1]
                                and plan_y[j][i - 1] == plan_y[k][i] and plan_y[j][i] == plan_y[k][i - 1]):
                            available_plan = False
                            print('Swapping conflict detected')
                            break
                # 检查各智能体在当前timestep下是否存在顶点冲突
                for k in range(j + 1, len(plan_x)):
                    if plan_x[j][i] == plan_x[k][i] and plan_y[j][i] == plan_y[k][i]:
                        available_plan = False
                        print("Vertex conflict detected")
                        break

            plt.draw()
            plt.pause(update_time)
        plt.show()


def read_map_file(file_path):
    ox, oy = [], []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for y, line in enumerate(lines):
            for x, char in enumerate(line.strip()):
                if char == '@':
                    ox.append(x)
                    oy.append(y)
        # 添加边界
        for i in range(32):
            ox.append(32)
            oy.append(i + 4)
        for i in range(33):
            ox.append(i)
            oy.append(36)
        # 若为random或room地图，额外添加如下边界
        for i in range(-1, 33):
            ox.append(i)
            oy.append(3)
        for i in range(33):
            ox.append(-1)
            oy.append(i + 4)
    return ox, oy


# 检查是否存在顶点冲突
def find_vertex_conflict(node):
    # 考虑到达终点后等待，将路径补齐至相同长度
    max_length = max(len(plan) for plan in node.plan_x)
    for rxi in node.plan_x:
        if len(rxi) < max_length:
            rxi.extend([rxi[-1]] * (max_length - len(rxi)))
    for ryi in node.plan_y:
        if len(ryi) < max_length:
            ryi.extend([ryi[-1]] * (max_length - len(ryi)))
    # 如果在同一时间步下坐标相同则返回冲突的时间和位置
    for i in range(len(node.plan_x)):
        # 遍历当前子列表之后的子列表以避免重复比较
        for j in range(i + 1, len(node.plan_x)):
            # 遍历子列表rx[i]中的每个元素
            # 检查是否存在顶点冲突
            for t in range(len(node.plan_x[i])):
                # 确保不会因为rx[j]长度不足而出错
                if t < len(node.plan_x[j]) and t < len(node.plan_y[i]) and t < len(node.plan_y[j]):
                    # 检查rx的两个子列表中的元素是否相等
                    if node.plan_x[i][t] == node.plan_x[j][t]:
                        # 检查对应的ry中的元素是否也相等
                        if node.plan_y[i][t] == node.plan_y[j][t]:
                            # 如果两个条件都满足，返回元素和索引作为顶点冲突信息
                            return i, j, node.plan_x[i][t], node.plan_y[i][t], t

    return None


# 检查是否存在交换/合并冲突
def find_swapping_conflict(node):
    overlap_subseq = None
    max_length = 0

    list_length = len(node.plan_x)

    # 检查每对子列表 (rx[i], rx[j]) 和 (ry[i], ry[j])
    for i in range(list_length):
        for j in range(i + 1, list_length):
            # 获取两对子列表的较短长度
            min_sub_length = min(len(node.plan_x[i]), len(node.plan_x[j]), len(node.plan_y[i]), len(node.plan_y[j]))

            # 从最长可能长度开始向下检查每个可能的切片长度
            for length in range(min_sub_length, 1, -1):
                for t in range(min_sub_length - length + 1):
                    sub_rx_i = node.plan_x[i][t:t + length]
                    sub_rx_j = node.plan_x[j][t:t + length]
                    sub_ry_i = node.plan_y[i][t:t + length]
                    sub_ry_j = node.plan_y[j][t:t + length]

                    # 检查是否是倒序切片
                    if sub_rx_i == sub_rx_j[::-1] and sub_ry_i == sub_ry_j[::-1] and length > max_length:
                        max_length = length
                        overlap_subseq = {
                            'i': i, 'j': j,
                            'time': t, 'length': length,
                            'sub_rx_i': sub_rx_i, 'sub_rx_j': sub_rx_j,
                            'sub_ry_i': sub_ry_i, 'sub_ry_j': sub_ry_j
                        }
                        # 找到最长的切片后，不需要检查更短的切片
                        break
                # 已找到当前子列表对的最长切片，继续检查下一对子列表
                if max_length == length:
                    break
    if overlap_subseq is not None:
        return (overlap_subseq['i'], overlap_subseq['j'], overlap_subseq['sub_rx_i'], overlap_subseq['sub_ry_i'],
                overlap_subseq['length'], overlap_subseq['time'])
    else:
        return None


# 检查是否存在冲突，如有返回冲突信息元组，如无返回None
def check_goal(v_conflict, s_conflict):
    if s_conflict is not None:
        conflict = s_conflict
    elif v_conflict is not None:
        conflict = v_conflict
    else:
        conflict = None
    return conflict


# 检查是否存在Deadlock（一条路径完全包含于另一条路径）
def check_subsequence(list_1, list_2):
    for start_idx in range(len(list_2) - len(list_1) + 1):
        if list_1 == list_2[start_idx + len(list_1)]:
            return True, start_idx, start_idx + len(list_1)
    list_1_reversed = list_1[::-1]
    for start_idx in range(len(list_2) - len(list_1) + 1):
        if list_1_reversed == list_2[start_idx:start_idx + len(list_1)]:
            return False, start_idx, start_idx + len(list_1)
    return None, None, None


def check_deadlock(rx, ry):
    n = len(rx)
    result = []
    for i in range(n):
        for j in range(n):
            if i != j and len(rx[i]) <= len(rx[j]) and len(ry[i]) <= len(ry[j]):
                is_slice, start_idx, end_idx = check_subsequence(rx[i], rx[j])
                if is_slice is not None and (end_idx - start_idx) >= 2:
                    if is_slice and ry[i] == ry[j][start_idx:end_idx]:
                        result.append((i, j, start_idx, end_idx))
                    elif not is_slice and ry[i] == ry[j][start_idx:end_idx][::-1]:
                        result.append((i, j, start_idx, end_idx))
    return result


def update_plan(j, ox, oy, grid_size, agent_radius, sx, sy, gx, gy):
    plan = AStar(ox[j], oy[j], grid_size, agent_radius)
    rx_j, ry_j = plan.planning(sx[j], sy[j], gx[j], gy[j])
    return rx_j, ry_j


def generate_nodes(root, grid_size, agent_radius, sx, sy, gx, gy):
    queue = [root]
    while True:
        current_node = queue.pop(0)
        # 检查当前节点是否满足条件
        conflict = check_goal(find_vertex_conflict(current_node), find_swapping_conflict(current_node))
        # 满足条件后结束生成
        if conflict is None:
            break
        j = conflict[1]
        conflict_x = conflict[2]
        conflict_y = conflict[3]

        # 为发生冲突的低优先级智能体添加障碍
        new_ox = copy.deepcopy(current_node.ox_k)
        new_oy = copy.deepcopy(current_node.oy_k)
        if len(conflict) == 5:
            new_ox[j].append(conflict_x)
            new_oy[j].append(conflict_y)
        else:
            new_ox[j].extend(conflict_x)
            new_oy[j].extend(conflict_y)

        # 根据新添加的障碍，更新低优先级智能体的路径及节点信息
        plan_x = copy.deepcopy(current_node.plan_x)
        plan_y = copy.deepcopy(current_node.plan_y)
        plan_x[j], plan_y[j] = update_plan(j, new_ox, new_oy, grid_size, agent_radius, sx, sy, gx, gy)
        plan_x, plan_y = add_rotate_motion(plan_x, plan_y)
        child = Node(current_node.node_id + 1, new_ox, new_oy, plan_x, plan_y)
        current_node.add_child(child)
        queue.append(child)

    return current_node.node_id, current_node.plan_x, current_node.plan_y


def simplify_ending(plan_x, plan_y):
    true_rx, true_ry = [], []
    for k in range(len(plan_x)):
        index = len(plan_x[k]) - 1
        while index > 0 and plan_x[k][index] == plan_x[k][index - 1] and plan_y[k][index] == plan_y[k][index - 1]:
            index -= 1
        # 保留从开始到第一个不重复元素的所有元素，同时保留最后一个重复的元素
        true_rx.append(plan_x[k][:index + 1])
        true_ry.append(plan_y[k][:index + 1])
    return true_rx, true_ry


def calculate_cost_makespan(plan_x, plan_y):
    makespan = len(plan_x[0]) - 1
    cost = 0
    rx, ry = simplify_ending(plan_x, plan_y)
    for rxi in rx:
        cost += len(rxi) - 1
    return cost, makespan


def generate_points(size_x, size_y, ox, oy, num_points, min_distance):
    # 生成所有可能的点
    all_points = [(x, y) for x in range(size_x) for y in range(size_y)]
    # 从中排除障碍物所在的点
    obstacle_points = set(zip(ox, oy))
    available_points = [p for p in all_points if p not in obstacle_points]

    # 检查是否有足够的点来放置起点和终点
    if len(available_points) < 2 * num_points:
        raise ValueError("Not enough available points to place the requested number of start and end points")

    # 随机选择起点
    start_points = random.sample(available_points, num_points)
    for sp in start_points:
        available_points.remove(sp)  # 移除已经选为起点的位置

    # 选择终点，确保每个终点与所有起点的距离至少为 min_distance
    end_points = []
    for _ in range(num_points):
        gx, gy = None, None
        while not gx or not gy or any(
                [abs(gx - sx) + abs(gy - sy) < min_distance for sx, sy in start_points + end_points]):
            gx, gy = random.choice(available_points)
        end_points.append((gx, gy))
        available_points.remove((gx, gy))  # 移除已经选为终点的位置

    # 解包列表获取单独的坐标列表
    sx, sy = zip(*start_points)
    gx, gy = zip(*end_points)

    return list(sx), list(sy), list(gx), list(gy)


def generate_points_for_special_map(size_x_min, size_x_max, size_y_min, size_y_max, ox, oy, num_points, min_distance):
    # 生成所有可能的点
    all_points = [(x, y) for x in range(size_x_min, size_x_max) for y in range(size_y_min, size_y_max)]
    # 从中排除障碍物所在的点
    obstacle_points = set(zip(ox, oy))
    available_points = [p for p in all_points if p not in obstacle_points]

    # 检查是否有足够的点来放置起点和终点
    if len(available_points) < 2 * num_points:
        raise ValueError("Not enough available points to place the requested number of start and end points")

    # 随机选择起点
    start_points = random.sample(available_points, num_points)
    for sp in start_points:
        available_points.remove(sp)  # 移除已经选为起点的位置

    # 选择终点，确保每个终点与所有起点的距离至少为 min_distance
    end_points = []
    for _ in range(num_points):
        gx, gy = None, None
        while not gx or not gy or any(
                [abs(gx - sx) + abs(gy - sy) < min_distance for sx, sy in start_points + end_points]):
            gx, gy = random.choice(available_points)
        end_points.append((gx, gy))
        available_points.remove((gx, gy))  # 移除已经选为终点的位置

    # 解包列表获取单独的坐标列表
    sx, sy = zip(*start_points)
    gx, gy = zip(*end_points)

    return list(sx), list(sy), list(gx), list(gy)


def main():
    '''
    # 设置地图及障碍物
    # 常规地图
    ox, oy = create_void_map(size=8)
    # ox, oy = create_warehouse_map()
    # ox, oy = read_map_file('C:/Users/10721/PycharmProjects/pythonProject/Bachelorarbeit/map/maze-32-32-2.map')
    # 非常规地图
    # ox, oy = read_map_file('C:/Users/10721/PycharmProjects/pythonProject/Bachelorarbeit/map/random-32-32-10.map')
    # ox, oy = read_map_file('C:/Users/10721/PycharmProjects/pythonProject/Bachelorarbeit/map/random-32-32-20.map')
    # ox, oy = read_map_file('C:/Users/10721/PycharmProjects/pythonProject/Bachelorarbeit/map/room-32-32-4.map')
    # 设置起点和终点
    # 常规地图
    sx, sy, gx, gy = generate_points(size_x=32, size_y=32, ox=ox, oy=oy, num_points=5, min_distance=1)
    # 非常规地图
    sx, sy, gx, gy = generate_points_for_special_map(size_x_min=-1, size_x_max=31, size_y_min=4, size_y_max=34, ox=ox,
                                                     oy=oy, num_points=10, min_distance=1)

    sx = [4, 4, 8, 27, 12, 7, 1, 13, 2, 0]
    sy = [28, 5, 4, 11, 4, 12, 11, 14, 4, 13]
    gx = [25, 29, 7, 7, 2, 23, 30, 29, 12, 17]
    gy = [21, 26, 11, 28, 18, 14, 6, 21, 20, 21]

    sx = [3, 4]
    sy = [3, 7]
    gx = [6, 4]
    gy = [4, 3]

    print(sx)
    print(sy)
    print(gx)
    print(gy)

    # 网格大小
    grid_size = 1.0
    # 智能体半径
    agent_radius = 0.5

    # 程序开始时间
    start = time.perf_counter()

    # 为k个智能体规划初始路径
    plan = AStar(ox, oy, grid_size, agent_radius)
    rx, ry = [], []
    for sx_i, sy_i, gx_i, gy_i in zip(sx, sy, gx, gy):
        rx_i, ry_i = plan.planning(sx_i, sy_i, gx_i, gy_i)
        rx.append(rx_i)
        ry.append(ry_i)
    # 在初始计划中添加转向动作
    rx, ry = add_rotate_motion(rx, ry)
    print(rx)
    print(ry)
    # k个智能体的初始可行区域
    ox_k, oy_k = [], []
    for i in range(len(sx)):
        ox_k.append(ox)
        oy_k.append(oy)
    # 创建初始节点
    nodes = []
    root_node = Node(1, ox_k, oy_k, rx, ry)
    nodes.append(root_node)

    # 查找路径，直到无冲突
    result = run_with_timeout(generate_nodes, args=(root_node, grid_size, agent_radius, sx, sy, gx, gy,),
                              timeout_duration=180)
    print('----------结果检查----------')
    print(result)
    if result is None:
        print("Function execution exceeded 180 seconds and was interrupted, returning None.")
        show_map(ox, oy, sx, sy, gx, gy, rx, ry)
    else:
        goal_node, plan_x, plan_y = result

    # 程序结束时间
    end = time.perf_counter()

    cost, makespan = calculate_cost_makespan(plan_x, plan_y)
    # 打印CBS搜索树
    print('----------节点生成----------')
    root_node.print_tree()
    print('----------最终路线----------')
    print(goal_node)
    print(plan_x)
    print(plan_y)
    print(f"cost={cost}")
    print(f"makespan={makespan}")
    print('----------算例输入----------')
    print(f"sx={sx}")
    print(f"sy={sy}")
    print(f"gx={gx}")
    print(f"gy={gy}")
    print('----------运行耗时----------')
    print(f"运行耗时：{int((end - start) * 1000)}ms")
    # 仿真
    simulator(ox, oy, sx, sy, gx, gy, plan_x, plan_y, 1, True)
    '''

    # 随机生成起点终点并进行实验
    success_result = 0
    success_rate = []
    error_indices = []
    df = pd.DataFrame(
        columns=['Test Number', 'sx', 'sy', 'gx', 'gy', 'cost', 'makespan', 'run_time', 'deadlock'])
    for test_number in range(1, 101):
        print(f'轮次：{test_number}/100')
        try:
            ox, oy = create_void_map(size=16)
            # ox, oy = create_large_warehouse_map()
            # ox, oy = read_map_file('C:/Users/10721/PycharmProjects/pythonProject/Bachelorarbeit/map/random-32-32-10.map')
            # 网格大小
            grid_size = 1.0
            # 智能体半径
            agent_radius = 0.5

            start = time.perf_counter()

            # 为k个智能体规划初始路径
            plan = AStar(ox, oy, grid_size, agent_radius)
            rx, ry = [], []
            '''
            sx, sy, gx, gy = generate_points_for_special_map(size_x_min=-1, size_x_max=31, size_y_min=4, size_y_max=34,
                                                             ox=ox,
                                                             oy=oy, num_points=2, min_distance=1)
            '''
            sx, sy, gx, gy = generate_points(size_x=161, size_y=61, ox=ox, oy=oy, num_points=10, min_distance=1)
            for sx_i, sy_i, gx_i, gy_i in zip(sx, sy, gx, gy):
                rx_i, ry_i = plan.planning(sx_i, sy_i, gx_i, gy_i)
                rx.append(rx_i)
                ry.append(ry_i)
            # 在初始计划中添加转向动作
            rx, ry = add_rotate_motion(rx, ry)

            '''
            deadlock = check_deadlock(rx, ry)
            if len(deadlock) != 0:
                deadlock_detected = 1
                print('Deadlock_detected')
            else:
                deadlock_detected = 0
            '''
            nodes = []
            cost = sum(len(rxi) - 1 for rxi in rx)
            root_node = Node(1, cost, [], rx, ry)
            nodes.append(root_node)

            result = run_with_timeout(generate_nodes, args=(root_node, grid_size, agent_radius, sx, sy, gx, gy,),
                                      timeout_duration=60)
            if result is None:
                goal_node, plan_x, plan_y = -1, -1, -1
                print("Function execution exceeded 180 seconds and was interrupted, returning None.")
                cost, makespan = -1, -1
            else:
                goal_node, plan_x, plan_y = result
                cost, makespan = calculate_cost_makespan(plan_x, plan_y)
                success_result += 1

            end = time.perf_counter()
            run_time = int((end - start) * 1000)
            # 将实验结果保存到表格中
            df = df._append({
                'Test Number': test_number,
                'sx': sx,
                'sy': sy,
                'gx': gx,
                'gy': gy,
                'cost': cost,
                'makespan': makespan,
                'run_time': run_time,
                'deadlock': -1
            }, ignore_index=True)

            print(f"运行耗时：{run_time}ms")

        except Exception as e:
            df = df._append({
                'Test Number': test_number,
                'sx': sx,
                'sy': sy,
                'gx': gx,
                'gy': gy,
                'cost': -1,
                'makespan': -1,
                'run_time': -1,
                'deadlock': -1
            }, ignore_index=True)
            error_indices.append(test_number)
            print(f"Error at iteration {test_number}: {e}")
            continue

    df.to_excel(f'{10}_2.xlsx', index=False)
    print(f"----------k={10}结果写入完成----------")
    print(f"----------成功率为{success_result}%----------")


if __name__ == '__main__':
    main()

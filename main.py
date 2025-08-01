import numpy as np
import cv2
import math
import threading
from typing import List, Tuple, Optional


def calculate_line_params(point1: Tuple[int, int], point2: Tuple[int, int]) -> Tuple[float, float, float]:
    """
    计算直线的参数 ax + by + c = 0
    :param point1: 直线上的第一个点
    :param point2: 直线上的第二个点
    :return: (a, b, c) 直线参数
    """
    x1, y1 = point1
    x2, y2 = point2

    # 如果两点重合，返回无效参数
    if x1 == x2 and y1 == y2:
        return 0, 0, 1

    # 计算 ax + by + c = 0 的参数
    a = y2 - y1
    b = x1 - x2
    c = x2 * y1 - x1 * y2

    # 归一化
    norm = np.sqrt(a * a + b * b)
    if norm > 0:
        a, b, c = a / norm, b / norm, c / norm

    return a, b, c


def calculate_square_center(square: List[Tuple[int, int]]) -> Tuple[float, float]:
    """计算正方形的中心点"""
    center_x = sum(point[0] for point in square) / len(square)
    center_y = sum(point[1] for point in square) / len(square)
    return center_x, center_y


def calculate_square_area(square: List[Tuple[int, int]]) -> float:
    """计算正方形的面积"""
    if len(square) != 4:
        return 0.0

    # 使用鞋带公式计算多边形面积
    area = 0.0
    n = len(square)
    for i in range(n):
        j = (i + 1) % n
        area += square[i][0] * square[j][1]
        area -= square[j][0] * square[i][1]
    return abs(area) / 2.0


def calculate_square_rotation_angle(square: List[Tuple[int, int]]) -> float:
    """
    计算正方形的旋转角度，考虑90度对称性
    返回值范围: [0, 90)
    """
    if len(square) != 4:
        return 0.0

    # 计算第一条边的角度
    p1, p2 = square[0], square[1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # 计算角度（弧度转度）
    angle = math.degrees(math.atan2(dy, dx))
    
    # 考虑正方形的90度对称性，将角度映射到 [0, 90) 范围
    angle = angle % 90
    
    return angle


def order_square_vertices(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    按顺序排列正方形的顶点（顺时针）
    """
    # 找到中心点
    center_x = sum(p[0] for p in points) / 4
    center_y = sum(p[1] for p in points) / 4

    # 计算每个点相对于中心的角度
    angles = []
    for p in points:
        angle = np.arctan2(p[1] - center_y, p[0] - center_x)
        angles.append((angle, p))

    # 按角度排序
    angles.sort()

    return [p for _, p in angles]


def print_corner_info(corners: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]):
    """打印外直角信息"""
    print(f"检测到 {len(corners)} 个外直角")


def calculate_line_intersection(line1_params: Tuple[float, float, float], line2_params: Tuple[float, float, float]) -> Optional[Tuple[float, float]]:
    """
    计算两条直线的交点
    :param line1_params: 第一条直线参数 (a1, b1, c1)
    :param line2_params: 第二条直线参数 (a2, b2, c2)
    :return: 交点坐标 (x, y) 或 None（如果平行）
    """
    a1, b1, c1 = line1_params
    a2, b2, c2 = line2_params

    # 计算行列式
    det = a1 * b2 - a2 * b1

    # 如果行列式为0，直线平行
    if abs(det) < 1e-10:
        return None

    # 计算交点
    x = (b1 * c2 - b2 * c1) / det
    y = (a2 * c1 - a1 * c2) / det

    return x, y


def are_lines_parallel(line1_params: Tuple[float, float, float], line2_params: Tuple[float, float, float], tolerance: float = 0.3) -> bool:
    """
    检查两条直线是否平行
    """
    a1, b1, _ = line1_params
    a2, b2, _ = line2_params

    # 检查方向向量是否平行
    cross_product = abs(a1 * b2 - a2 * b1)
    return cross_product < tolerance


def calculate_square_overlap(square1: List[Tuple[int, int]], square2: List[Tuple[int, int]]) -> float:
    """
    计算两个正方形的重叠面积比例
    返回重叠面积占较小正方形面积的比例
    """
    # 计算每个正方形的面积
    area1 = calculate_square_area(square1)
    area2 = calculate_square_area(square2)

    if area1 == 0 or area2 == 0:
        return 0.0

    # 使用简化的重叠检测：检查中心点距离和尺寸相似度
    center1 = calculate_square_center(square1)
    center2 = calculate_square_center(square2)

    # 计算中心点距离
    center_distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    # 计算正方形的边长
    side1 = math.sqrt(area1)
    side2 = math.sqrt(area2)

    # 如果中心距离很小且尺寸相近，认为是重叠的
    max_side = max(side1, side2)
    min_side = min(side1, side2)

    # 尺寸相似度阈值
    size_similarity = min_side / max_side if max_side > 0 else 0

    # 如果中心距离小于较小边长的一半且尺寸相似度高，认为高度重叠
    if center_distance < min_side * 0.5 and size_similarity > 0.8:
        return size_similarity  # 返回尺寸相似度作为重叠程度

    return 0.0


class CornerDetector:
    # ========== 配置参数 ==========
    # 角点检测参数
    CORNER_SEARCH_RADIUS = 10         # 角点搜索半径
    MIN_BLACK_RATIO = 0.4             # 最小黑色像素比例
    
    # 几何验证参数
    GEOMETRIC_TOLERANCE = 5.0         # 几何容差（像素）
    PARALLEL_ANGLE_TOLERANCE = 10.0    # 平行线角度容差（度）
    SQUARE_VALIDATION_BLACK_RATIO = 0.95  # 正方形验证黑色像素比例
    
    # 重复检测参数
    OVERLAP_THRESHOLD = 0.95          # 重叠度阈值
    ROTATION_ANGLE_THRESHOLD = 10.0   # 旋转角度差异阈值（度）
    
    # 去重参数
    DUPLICATE_DISTANCE_THRESHOLD = 10.0  # 重复角点距离阈值
    # ========== 配置参数结束 ==========
    
    def __init__(self, image_size: Tuple[int, int] = (1000, 1000)):
        """
        初始化角点检测器
        :param image_size: 图像尺寸 (width, height)
        """
        self.prefix_sum = None
        self.image_size = image_size
        self.image = np.ones((image_size[1], image_size[0]), dtype=np.uint8) * 255  # 白色背景
    
    def load_image_from_file(self, image_path: str, binarize: bool = True, threshold: int = 127):
        """
        从文件加载位图图像并进行二值化处理
        :param image_path: 图像文件路径
        :param binarize: 是否进行二值化处理
        :param threshold: 二值化阈值 (0-255)
        """
        try:
            # 使用OpenCV加载图像
            loaded_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if loaded_image is None:
                raise ValueError(f"无法加载图像文件: {image_path}")
            
            print(f"成功加载图像: {image_path}")
            print(f"原始图像尺寸: {loaded_image.shape[1]} x {loaded_image.shape[0]}")
            
            if binarize:
                # 进行二值化处理
                # 使用OTSU自适应阈值或固定阈值
                if threshold == -1:
                    # 使用OTSU自适应阈值
                    threshold_value, binary_image = cv2.threshold(loaded_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    print(f"使用OTSU自适应阈值: {threshold_value:.1f}")
                else:
                    # 使用固定阈值
                    _, binary_image = cv2.threshold(loaded_image, threshold, 255, cv2.THRESH_BINARY)
                    print(f"使用固定阈值: {threshold}")
                
                self.image = binary_image
                print("图像已进行二值化处理")
                
                # 统计黑白像素比例
                total_pixels = binary_image.size
                white_pixels = np.sum(binary_image == 255)
                black_pixels = total_pixels - white_pixels
                white_ratio = white_pixels / total_pixels
                black_ratio = black_pixels / total_pixels
                print(f"白色像素比例: {white_ratio:.1%}, 黑色像素比例: {black_ratio:.1%}")
            else:
                self.image = loaded_image
                print("保持原始灰度图像")
            
            # 更新图像尺寸
            self.image_size = (self.image.shape[1], self.image.shape[0])  # (width, height)
            
        except Exception as e:
            print(f"加载图像失败: {e}")
            raise
    
    def save_current_image(self, filename: str = "loaded_image.png"):
        """
        保存当前图像到文件
        :param filename: 保存的文件名
        """
        cv2.imwrite(filename, self.image)
        print(f"当前图像已保存为: {filename}")
        
    def is_black_pixel(self, x: int, y: int) -> bool:
        """检查像素是否为黑色"""
        if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
            return self.image[y, x] == 0
        return False
    
    def build_prefix_sum_matrix(self):
        """
        构建2D前缀和矩阵，用于快速计算任意矩形区域的黑色像素数量
        这是DP算法的核心，time complexity: O(W*H), space complexity: O(W*H)
        """
        height, width = self.image.shape
        # 创建前缀和矩阵，多一行一列便于边界处理
        self.prefix_sum = np.zeros((height + 1, width + 1), dtype=np.int32)
        
        print("构建前缀和矩阵...")
        for y in range(1, height + 1):
            for x in range(1, width + 1):
                # 当前像素是否为黑色 (0表示黑色)
                is_black = 1 if self.image[y-1, x-1] == 0 else 0
                # DP状态转移方程
                self.prefix_sum[y, x] = (is_black + 
                                       self.prefix_sum[y-1, x] + 
                                       self.prefix_sum[y, x-1] - 
                                       self.prefix_sum[y-1, x-1])
        print(f"前缀和矩阵构建完成，尺寸: {self.prefix_sum.shape}")
    
    def get_black_pixel_count_in_rect(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """
        使用前缀和快速计算矩形区域内的黑色像素数量
        时间复杂度: O(1)
        :param x1, y1: 矩形左上角坐标
        :param x2, y2: 矩形右下角坐标 (不包含)
        :return: 黑色像素数量
        """
        # 边界检查
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.image_size[0], x2)
        y2 = min(self.image_size[1], y2)
        
        if x1 >= x2 or y1 >= y2:
            return 0
        
        # 前缀和查询，注意坐标偏移
        return (self.prefix_sum[y2, x2] - 
                self.prefix_sum[y1, x2] - 
                self.prefix_sum[y2, x1] + 
                self.prefix_sum[y1, x1])
    
    def get_black_pixel_ratio_in_rect(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        快速计算矩形区域内的黑色像素比例
        """
        black_count = self.get_black_pixel_count_in_rect(x1, y1, x2, y2)
        total_pixels = max(1, (x2 - x1) * (y2 - y1))  # 避免除零
        return black_count / total_pixels
    
    def is_edge_pixel(self, x: int, y: int) -> bool:
        """
        检查像素是否为边缘像素（黑色像素且周围8个像素不全为黑）
        :param x: 像素x坐标
        :param y: 像素y坐标
        :return: 是否为边缘像素
        """
        # 首先检查当前像素是否为黑色
        if not self.is_black_pixel(x, y):
            return False
        
        # 检查周围8个像素
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        black_neighbors = 0
        valid_neighbors = 0
        
        for dx, dy in neighbors:
            check_x = x + dx
            check_y = y + dy
            
            # 确保邻居像素在图像范围内
            if (0 <= check_x < self.image_size[0] and 
                0 <= check_y < self.image_size[1]):
                valid_neighbors += 1
                if self.is_black_pixel(check_x, check_y):
                    black_neighbors += 1
        
        # 如果周围所有有效邻居都是黑色，则不是边缘像素
        if 0 < valid_neighbors == black_neighbors:
            return False
        
        # 如果周围至少有一个白色像素，则是边缘像素
        return valid_neighbors > black_neighbors
    
    def calculate_black_pixel_ratio(self, center_x: int, center_y: int, radius: int = CORNER_SEARCH_RADIUS) -> float:
        """
        计算指定点周围指定半径内黑色像素的比例
        使用矩形区域而不是圆形区域，与DP算法保持一致
        :param center_x: 中心点x坐标  
        :param center_y: 中心点y坐标
        :param radius: 检测半径
        :return: 黑色像素比例 (0.0-1.0)
        """
        # 使用矩形区域，与DP算法保持一致
        x1 = center_x - radius
        y1 = center_y - radius
        x2 = center_x + radius + 1
        y2 = center_y + radius + 1
        
        return self.get_black_pixel_ratio_in_rect(x1, y1, x2, y2)
    
    def find_two_edge_assist_points(self, corner_x: int, corner_y: int, radius: int = 10) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        在圆圈上寻找边缘像素（周围8个像素不全为黑的黑色像素），
        从中选择两个使得与角点形成的夹角最接近90度
        :param corner_x: 角点x坐标
        :param corner_y: 角点y坐标
        :param radius: 搜索半径
        :return: 两个辅助点坐标
        """
        edge_pixels = []
        
        # 在圆圈上寻找边缘像素（黑色像素且周围8个像素不全为黑）
        for angle_deg in range(0, 360, 3):  # 每3度检查一次
            angle_rad = np.radians(angle_deg)
            check_x = int(corner_x + radius * np.cos(angle_rad))
            check_y = int(corner_y + radius * np.sin(angle_rad))
            
            # 确保在图像范围内且是黑色像素，并且不是角点本身
            if (0 <= check_x < self.image_size[0] and 
                0 <= check_y < self.image_size[1] and 
                self.is_black_pixel(check_x, check_y) and
                (check_x != corner_x or check_y != corner_y)):
                
                # 检查周围8个像素是否不全为黑（即这是一个边缘像素）
                is_edge = self.is_edge_pixel(check_x, check_y)
                
                if is_edge:
                    edge_pixels.append({
                        'point': (check_x, check_y),
                        'angle': angle_rad
                    })
        
        # 如果找到的边缘像素少于2个，扩大搜索范围
        if len(edge_pixels) < 2:
            for radius_ext in [radius - 2, radius + 2, radius - 4, radius + 4]:
                if radius_ext <= 0:
                    continue
                    
                for angle_deg in range(0, 360, 3):
                    angle_rad = np.radians(angle_deg)
                    check_x = int(corner_x + radius_ext * np.cos(angle_rad))
                    check_y = int(corner_y + radius_ext * np.sin(angle_rad))
                    
                    if (0 <= check_x < self.image_size[0] and 
                        0 <= check_y < self.image_size[1] and 
                        self.is_black_pixel(check_x, check_y) and
                        (check_x != corner_x or check_y != corner_y)):
                        
                        is_edge = self.is_edge_pixel(check_x, check_y)
                        
                        if is_edge:
                            # 检查是否已经存在相近的点
                            is_duplicate = False
                            for existing in edge_pixels:
                                existing_point = existing['point']
                                distance = np.sqrt((check_x - existing_point[0])**2 + 
                                                 (check_y - existing_point[1])**2)
                                if distance < 3:
                                    is_duplicate = True
                                    break
                            
                            if not is_duplicate:
                                edge_pixels.append({
                                    'point': (check_x, check_y),
                                    'angle': angle_rad
                                })
                
                if len(edge_pixels) >= 2:
                    break
        
        # 从边缘像素中选择两个，使得它们与角点形成的夹角最接近90度
        best_pair = None
        best_angle_diff = float('inf')
        
        for i in range(len(edge_pixels)):
            for j in range(i + 1, len(edge_pixels)):
                point1 = edge_pixels[i]['point']
                point2 = edge_pixels[j]['point']
                
                # 计算两个点相对于角点的角度
                angle1 = np.arctan2(point1[1] - corner_y, point1[0] - corner_x)
                angle2 = np.arctan2(point2[1] - corner_y, point2[0] - corner_x)
                
                # 计算两个角度之间的夹角
                angle_diff = abs(angle2 - angle1)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                
                # 计算与90度的差异
                target_angle = np.pi / 2  # 90度
                diff_from_90 = abs(angle_diff - target_angle)
                
                # 如果这个组合更接近90度，更新最佳组合
                if diff_from_90 < best_angle_diff:
                    best_angle_diff = diff_from_90
                    best_pair = (point1, point2)
        
        # 如果找到了合适的边缘像素对
        if best_pair is not None:
            return best_pair[0], best_pair[1]
        
        # 如果没有找到合适的边缘像素，使用原来的方向偏移方法作为后备
        assist_point1 = (int(corner_x + 8),
                       int(corner_y + 8))
        assist_point2 = (int(corner_x + 8),
                       int(corner_y + 8))
        
        # 确保在图像范围内
        assist_point1 = (max(0, min(self.image_size[0] - 1, assist_point1[0])),
                       max(0, min(self.image_size[1] - 1, assist_point1[1])))
        assist_point2 = (max(0, min(self.image_size[0] - 1, assist_point2[0])),
                       max(0, min(self.image_size[1] - 1, assist_point2[1])))
        
        return assist_point1, assist_point2

    def are_corners_collinear_by_connection_angle(self, corner1: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                                                  corner2: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                                                  edge1_idx: int, edge2_idx: int, angle_tolerance: float = None) -> Tuple[bool, float]:
        """
        通过连接线角度检查两个角点的边是否共线
        检查角点1的某条边向量与角点2的某条边向量，相对于两角点连接线的夹角是否足够小
        :param corner1: 第一个角点 (角点坐标, 辅助点1, 辅助点2)
        :param corner2: 第二个角点 (角点坐标, 辅助点1, 辅助点2)  
        :param edge1_idx: 角点1的边索引 (1或2)
        :param edge2_idx: 角点2的边索引 (1或2)
        :param angle_tolerance: 角度容差（度）
        :return: (是否共线, 最大角度差)
        """
        if angle_tolerance is None:
            angle_tolerance = self.PARALLEL_ANGLE_TOLERANCE
            
        corner1_point, assist1_1, assist1_2 = corner1
        corner2_point, assist2_1, assist2_2 = corner2
        
        # 获取边向量
        if edge1_idx == 1:
            edge1_vector = np.array(assist1_1) - np.array(corner1_point)
        else:
            edge1_vector = np.array(assist1_2) - np.array(corner1_point)
            
        if edge2_idx == 1:
            edge2_vector = np.array(assist2_1) - np.array(corner2_point)  
        else:
            edge2_vector = np.array(assist2_2) - np.array(corner2_point)
        
        # 计算连接线向量
        connection_vector = np.array(corner2_point) - np.array(corner1_point)
        
        # 归一化所有向量
        edge1_norm = np.linalg.norm(edge1_vector)
        edge2_norm = np.linalg.norm(edge2_vector)
        conn_norm = np.linalg.norm(connection_vector)
        
        if edge1_norm == 0 or edge2_norm == 0 or conn_norm == 0:
            return False, 90.0
            
        edge1_unit = edge1_vector / edge1_norm
        edge2_unit = edge2_vector / edge2_norm
        conn_unit = connection_vector / conn_norm
        
        # 计算边向量与连接线向量的夹角
        cos_angle1 = abs(np.dot(edge1_unit, conn_unit))
        cos_angle2 = abs(np.dot(edge2_unit, -conn_unit))  # edge2相对于反向连接线
        
        cos_angle1 = np.clip(cos_angle1, 0, 1)
        cos_angle2 = np.clip(cos_angle2, 0, 1)
        
        angle1 = np.degrees(np.arccos(cos_angle1))
        angle2 = np.degrees(np.arccos(cos_angle2))
        
        # 两个角度都应该接近0度（共线）
        max_angle = max(angle1, angle2)
        is_collinear = max_angle < angle_tolerance
        
        return is_collinear, max_angle

    def calculate_square_black_ratio(self, corners: List[Tuple[int, int]]) -> float:
        """
        计算正方形区域内黑色像素的比例（双线程并行处理）
        :param corners: 正方形的四个顶点
        :return: 黑色像素比例
        """
        if len(corners) != 4:
            return 0.0
        
        # 创建掩码
        mask = np.zeros((self.image_size[1], self.image_size[0]), dtype=np.uint8)
        pts = np.array(corners, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        
        # 将图像区域分成上下两半进行并行处理
        height = self.image_size[1]
        mid_y = height // 2
        
        # 线程共享变量
        total_pixels_thread1 = 0
        black_pixels_thread1 = 0
        total_pixels_thread2 = 0
        black_pixels_thread2 = 0
        lock = threading.Lock()
        
        def scan_upper_half():
            """线程1：扫描上半部分"""
            nonlocal total_pixels_thread1, black_pixels_thread1
            local_total = 0
            local_black = 0
            
            for y in range(0, mid_y):
                for x in range(self.image_size[0]):
                    if mask[y, x] > 0:  # 在正方形区域内
                        local_total += 1
                        if self.is_black_pixel(x, y):
                            local_black += 1
            
            with lock:
                total_pixels_thread1 = local_total
                black_pixels_thread1 = local_black
        
        def scan_lower_half():
            """线程2：扫描下半部分"""
            nonlocal total_pixels_thread2, black_pixels_thread2
            local_total = 0
            local_black = 0
            
            for y in range(mid_y, height):
                for x in range(self.image_size[0]):
                    if mask[y, x] > 0:  # 在正方形区域内
                        local_total += 1
                        if self.is_black_pixel(x, y):
                            local_black += 1
            
            with lock:
                total_pixels_thread2 = local_total
                black_pixels_thread2 = local_black
        
        # 创建并启动两个线程
        thread1 = threading.Thread(target=scan_upper_half)
        thread2 = threading.Thread(target=scan_lower_half)
        
        thread1.start()
        thread2.start()
        
        # 等待两个线程完成
        thread1.join()
        thread2.join()
        
        # 合并结果
        total_pixels = total_pixels_thread1 + total_pixels_thread2
        black_pixels = black_pixels_thread1 + black_pixels_thread2
        
        return black_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def detect_all_outer_corners(self) -> List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        """
        使用DP算法的角点检测方法（双线程并行遍历）
        不再使用粗分割，直接对每个像素进行基于前缀和的精确分析
        """
        print("角点检测中...")
        
        # 首先构建前缀和矩阵
        self.build_prefix_sum_matrix()
        
        all_corners = []
        search_radius = self.CORNER_SEARCH_RADIUS
        edge_threshold = 5  # 边界安全距离
        
        # 计算扫描区域
        start_y = edge_threshold
        end_y = self.image_size[1] - edge_threshold
        mid_y = (start_y + end_y) // 2
        
        # 线程共享变量
        corners_thread1 = []
        corners_thread2 = []
        lock = threading.Lock()
        
        def scan_upper_half():
            """线程1：扫描上半部分"""
            local_corners = []
            for y in range(start_y, mid_y):
                for x in range(edge_threshold, self.image_size[0] - edge_threshold):
                    # 快速过滤：只检查黑色像素
                    if not self.is_black_pixel(x, y):
                        continue
                    
                    # 使用原来的边缘像素检测方法
                    if not self.is_edge_pixel(x, y):
                        continue
                    
                    # 直接在这里计算黑色比例过滤角点
                    corner_black_ratio = self.calculate_black_pixel_ratio(x, y, radius=self.CORNER_SEARCH_RADIUS)
                    
                    # 基于黑色比例过滤
                    if corner_black_ratio < 0.1 or corner_black_ratio > self.MIN_BLACK_RATIO:
                        continue  # 跳过黑色比例不合适的点
                    
                    # 直接调用现有函数计算辅助点
                    try:
                        assist_points = self.find_two_edge_assist_points(x, y, search_radius)
                        if assist_points and len(assist_points) == 2:
                            assist1, assist2 = assist_points
                            if assist1 and assist2:
                                local_corners.append(((x, y), assist1, assist2))
                    except:
                        # 如果辅助点计算失败，跳过
                        continue
            
            with lock:
                corners_thread1.extend(local_corners)
        
        def scan_lower_half():
            """线程2：扫描下半部分"""
            local_corners = []
            for y in range(mid_y, end_y):
                for x in range(edge_threshold, self.image_size[0] - edge_threshold):
                    # 快速过滤：只检查黑色像素
                    if not self.is_black_pixel(x, y):
                        continue
                    
                    # 使用原来的边缘像素检测方法
                    if not self.is_edge_pixel(x, y):
                        continue
                    
                    # 直接在这里计算黑色比例过滤角点
                    corner_black_ratio = self.calculate_black_pixel_ratio(x, y, radius=self.CORNER_SEARCH_RADIUS)
                    
                    # 基于黑色比例过滤
                    if corner_black_ratio < 0.1 or corner_black_ratio > self.MIN_BLACK_RATIO:
                        continue  # 跳过黑色比例不合适的点
                    
                    # 直接调用现有函数计算辅助点
                    try:
                        assist_points = self.find_two_edge_assist_points(x, y, search_radius)
                        if assist_points and len(assist_points) == 2:
                            assist1, assist2 = assist_points
                            if assist1 and assist2:
                                local_corners.append(((x, y), assist1, assist2))
                    except:
                        # 如果辅助点计算失败，跳过
                        continue
            
            with lock:
                corners_thread2.extend(local_corners)
        
        # 创建并启动两个线程
        thread1 = threading.Thread(target=scan_upper_half)
        thread2 = threading.Thread(target=scan_lower_half)
        
        thread1.start()
        thread2.start()
        
        # 等待两个线程完成
        thread1.join()
        thread2.join()
        
        # 合并结果
        all_corners.extend(corners_thread1)
        all_corners.extend(corners_thread2)
        
        print(f"检测完成，找到 {len(all_corners)} 个角点候选")
        
        # 去重
        filtered_corners = self.remove_duplicate_corners(all_corners)
        print(f"去重后剩余 {len(filtered_corners)} 个角点")
        
        return filtered_corners
    
    def remove_duplicate_corners(self, corners: List) -> List:
        """
        角点去重，通过缓存距离计算结果避免重复计算
        """
        if not corners:
            return []
        
        distance_cache = {}  # 缓存距离计算结果
        
        def get_distance_cached(p1, p2):
            """缓存版本的距离计算"""
            key = (min(p1, p2), max(p1, p2))  # 确保键的唯一性
            if key not in distance_cache:
                distance_cache[key] = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            return distance_cache[key]
        
        filtered_corners = []
        threshold = self.DUPLICATE_DISTANCE_THRESHOLD
        
        for corner in corners:
            corner_point = corner[0]
            duplicate_index = -1
            
            # 检查是否与已有角点重复
            for i, existing_corner in enumerate(filtered_corners):
                existing_point = existing_corner[0]
                distance = get_distance_cached(corner_point, existing_point)
                
                if distance < threshold:
                    duplicate_index = i
                    break
            
            if duplicate_index == -1:
                # 没有重复，直接添加
                filtered_corners.append(corner)
            else:
                # 发现重复，比较黑色比例，保留较小的
                current_ratio = self.calculate_black_pixel_ratio(corner_point[0], corner_point[1], radius=3)
                existing_point = filtered_corners[duplicate_index][0]
                existing_ratio = self.calculate_black_pixel_ratio(existing_point[0], existing_point[1], radius=3)
                
                if current_ratio < existing_ratio:
                    # 当前角点黑色比例更小，替换已有的
                    filtered_corners[duplicate_index] = corner
        
        # 缓存统计（简化）
        return filtered_corners

    def are_squares_different_rotation(self, square1: List[Tuple[int, int]], square2: List[Tuple[int, int]], angle_threshold: float = None) -> bool:
        """
        检查两个正方形是否有显著不同的旋转角度
        考虑90度对称性，角度差异计算基于 [0, 90) 范围
        """
        if angle_threshold is None:
            angle_threshold = self.ROTATION_ANGLE_THRESHOLD
            
        angle1 = calculate_square_rotation_angle(square1)
        angle2 = calculate_square_rotation_angle(square2)
        
        # 由于角度已经映射到 [0, 90) 范围，直接计算差异
        angle_diff = abs(angle1 - angle2)
        
        # 考虑90度边界的情况：如果一个角度接近0，另一个接近90
        # 那么它们实际上可能是相同的（因为90度等价于0度）
        boundary_diff = min(angle_diff, 90 - angle_diff)
        
        return boundary_diff > angle_threshold
    
    def remove_duplicate_squares(self, squares: List[List[Tuple[int, int]]], overlap_threshold: float = None) -> List[List[Tuple[int, int]]]:
        """
        移除重叠度过高的重复正方形
        保留黑色像素比例更高的正方形
        """
        if overlap_threshold is None:
            overlap_threshold = self.OVERLAP_THRESHOLD
            
        if len(squares) <= 1:
            return squares
        
        # 计算每个正方形的黑色像素比例
        square_ratios = []
        for square in squares:
            ratio = self.calculate_square_black_ratio(square)
            square_ratios.append(ratio)
        
        # 标记要保留的正方形
        keep_flags = [True] * len(squares)
        
        for i in range(len(squares)):
            if not keep_flags[i]:
                continue
                
            for j in range(i + 1, len(squares)):
                if not keep_flags[j]:
                    continue
                
                # 计算重叠度
                overlap = calculate_square_overlap(squares[i], squares[j])
                
                if overlap > overlap_threshold:
                    # 检查是否是不同旋转角度的正方形
                    if self.are_squares_different_rotation(squares[i], squares[j]):
                        print(f"正方形 {i+1} 和 {j+1} 重叠度高但旋转角度不同，保留两者")
                        continue
                        
                    # 重叠度过高且旋转角度相似，保留黑色像素比例更高的
                    if square_ratios[i] >= square_ratios[j]:
                        keep_flags[j] = False
                        print(f"移除重复正方形 {j+1}，与正方形 {i+1} 重叠度: {overlap:.3f}")
                    else:
                        keep_flags[i] = False
                        print(f"移除重复正方形 {i+1}，与正方形 {j+1} 重叠度: {overlap:.3f}")
                        break
        
        # 返回保留的正方形
        filtered_squares = [squares[i] for i in range(len(squares)) if keep_flags[i]]
        return filtered_squares

    def reconstruct_squares(self, corners: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]) -> List[List[Tuple[int, int]]]:
        """
        从检测到的角点重建正方形
        :param corners: 检测到的角点列表 [(corner_point, assist_point1, assist_point2), ...]
        :return: 重建的正方形列表，每个正方形包含四个顶点
        """
        squares = []
        used_corners = set()  # 记录已使用的角点索引
        
        print(f"正方形重建: {len(corners)}个角点 -> ", end="")
        
        # 直接调用各个方法，按优先级排序（4角点 > 3角点 > 2角点）
        squares_4_corners = self.find_squares_with_4_corners(corners, used_corners)
        squares_3_corners = self.find_squares_with_3_corners(corners, used_corners)
        squares_2_corners = self.find_squares_with_2_corners(corners, used_corners)
        
        # 合并结果
        squares.extend(squares_4_corners)
        squares.extend(squares_3_corners)
        squares.extend(squares_2_corners)
        
        print(f"{len(squares)}个正方形")
        
        # 移除重复的正方形
        if len(squares) > 1:
            squares = self.remove_duplicate_squares(squares, overlap_threshold=0.8)
            print(f" (去重后{len(squares)}个)")
        
        return squares
    
    def find_squares_with_4_corners(self, corners: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]], used_corners: set) -> List[List[Tuple[int, int]]]:
        """
        使用4个角点重建正方形
        """
        squares = []
        
        # 尝试所有4个角点的组合
        for i in range(len(corners)):
            if i in used_corners:
                continue
            for j in range(i + 1, len(corners)):
                if j in used_corners:
                    continue
                for k in range(j + 1, len(corners)):
                    if k in used_corners:
                        continue
                    for l in range(k + 1, len(corners)):
                        if l in used_corners:
                            continue
                        
                        four_corners = [corners[i], corners[j], corners[k], corners[l]]
                        square = self.build_square_from_4_corners(four_corners)
                        
                        if square and self.validate_square(square):
                            squares.append(square)
                            used_corners.update([i, j, k, l])
                            # 成功找到正方形（简化输出）
        
        return squares
    
    def find_squares_with_3_corners(self, corners: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]], used_corners: set) -> List[List[Tuple[int, int]]]:
        """
        使用3个角点重建正方形
        """
        squares = []
        
        # 尝试所有3个角点的组合
        for i in range(len(corners)):
            if i in used_corners:
                continue
            for j in range(i + 1, len(corners)):
                if j in used_corners:
                    continue
                for k in range(j + 1, len(corners)):
                    if k in used_corners:
                        continue
                    
                    three_corners = [corners[i], corners[j], corners[k]]
                    square = self.build_square_from_3_corners(three_corners)
                    
                    if square and self.validate_square(square):
                        squares.append(square)
                        used_corners.update([i, j, k])
                        # 简化输出
        
        return squares
    
    def find_squares_with_2_corners(self, corners: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]], used_corners: set) -> List[List[Tuple[int, int]]]:
        """
        使用2个角点重建正方形（原有逻辑）
        """
        squares = []
        
        for i in range(len(corners)):
            if i in used_corners:
                continue
                
            corner1 = corners[i]
            corner1_point, assist1_1, assist1_2 = corner1
            
            # 计算第一个角的两条边的直线参数
            line1_1 = calculate_line_params(corner1_point, assist1_1)
            line1_2 = calculate_line_params(corner1_point, assist1_2)
            
            for j in range(i + 1, len(corners)):
                if j in used_corners:
                    continue
                    
                corner2 = corners[j]
                corner2_point, assist2_1, assist2_2 = corner2
                
                # 计算第二个角的两条边的直线参数
                line2_1 = calculate_line_params(corner2_point, assist2_1)
                line2_2 = calculate_line_params(corner2_point, assist2_2)
                
                # 检查相邻角的情况（一条边共线，另一条边平行）
                square_found = False
                
                # 情况1: line1_1 与 line2_1 共线, line1_2 与 line2_2 平行
                is_col_11, angle_11 = self.are_corners_collinear_by_connection_angle(corner1, corner2, 1, 1)
                if is_col_11 and are_lines_parallel(line1_2, line2_2):
                    square = self.build_square_from_adjacent_corners(corner1, corner2)
                    if square and self.validate_square(square):
                        squares.append(square)
                        used_corners.add(i)
                        used_corners.add(j)
                        square_found = True
                        # 简化输出
                
                # 情况2: line1_1 与 line2_2 共线, line1_2 与 line2_1 平行
                is_col_12, angle_12 = self.are_corners_collinear_by_connection_angle(corner1, corner2, 1, 2)
                if not square_found and (is_col_12 and are_lines_parallel(line1_2, line2_1)):
                    square = self.build_square_from_adjacent_corners(corner1, corner2)
                    if square and self.validate_square(square):
                        squares.append(square)
                        used_corners.add(i)
                        used_corners.add(j)
                        square_found = True
                        # 简化输出
                
                # 情况3: line1_2 与 line2_1 共线, line1_1 与 line2_2 平行
                is_col_21, angle_21 = self.are_corners_collinear_by_connection_angle(corner1, corner2, 2, 1)
                if not square_found and (is_col_21 and are_lines_parallel(line1_1, line2_2)):
                    square = self.build_square_from_adjacent_corners(corner1, corner2)
                    if square and self.validate_square(square):
                        squares.append(square)
                        used_corners.add(i)
                        used_corners.add(j)
                        square_found = True
                        # 简化输出
                
                # 情况4: line1_2 与 line2_2 共线, line1_1 与 line2_1 平行
                is_col_22, angle_22 = self.are_corners_collinear_by_connection_angle(corner1, corner2, 2, 2)
                if not square_found and (is_col_22 and are_lines_parallel(line1_1, line2_1)):
                    square = self.build_square_from_adjacent_corners(corner1, corner2)
                    if square and self.validate_square(square):
                        squares.append(square)
                        used_corners.add(i)
                        used_corners.add(j)
                        square_found = True
                        # 简化输出
                
                # 检查对角关系
                if not square_found:
                    diagonal_square = self.build_square_from_diagonal_corners(corner1, corner2)
                    if diagonal_square and self.validate_square(diagonal_square):
                        squares.append(diagonal_square)
                        used_corners.add(i)
                        used_corners.add(j)
                        square_found = True
                        # 简化输出
                
                if square_found:
                    break
        
        return squares
    
    def build_square_from_4_corners(self, four_corners: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]) -> Optional[List[Tuple[int, int]]]:
        """
        从4个角点构建正方形
        """
        corner_points = [corner[0] for corner in four_corners]
        
        # 检查4个点是否能构成正方形
        if not self.are_4_points_square(corner_points):
            return None
        
        # 按顺序排列角点（顺时针或逆时针）
        ordered_points = order_square_vertices(corner_points)
        
        return ordered_points
    
    def build_square_from_3_corners(self, three_corners: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]) -> Optional[List[Tuple[int, int]]]:
        """
        从3个角点推断第4个角点并构建正方形
        """
        corner_points = [corner[0] for corner in three_corners]
        
        # 推断第4个角点
        fourth_point = self.calculate_fourth_corner(corner_points)
        if fourth_point is None:
            return None
        
        all_points = corner_points + [fourth_point]
        
        # 检查是否构成正方形
        if not self.are_4_points_square(all_points):
            return None
        
        # 按顺序排列角点
        ordered_points = order_square_vertices(all_points)
        
        return ordered_points
    
    def are_4_points_square(self, points: List[Tuple[int, int]], tolerance: float = None) -> bool:
        """
        检查4个点是否构成正方形
        """
        if tolerance is None:
            tolerance = self.GEOMETRIC_TOLERANCE
            
        if len(points) != 4:
            return False
        
        # 计算所有边长
        distances = []
        for i in range(4):
            for j in range(i + 1, 4):
                dist = np.sqrt((points[i][0] - points[j][0])**2 + (points[i][1] - points[j][1])**2)
                distances.append(dist)
        
        distances.sort()
        
        # 正方形应该有4条相等的边和2条相等的对角线
        # 前4个距离应该是边长，后2个应该是对角线长度
        side_length = distances[0]
        diagonal_length = distances[4]
        
        # 检查4条边是否相等
        for i in range(4):
            if abs(distances[i] - side_length) > tolerance:
                return False
        
        # 检查2条对角线是否相等
        for i in range(4, 6):
            if abs(distances[i] - diagonal_length) > tolerance:
                return False
        
        # 检查对角线长度是否等于边长的√2倍
        expected_diagonal = side_length * np.sqrt(2)
        if abs(diagonal_length - expected_diagonal) > tolerance:
            return False
        
        return True
    
    def calculate_fourth_corner(self, three_points: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        根据3个角点计算第4个角点位置
        """
        # 尝试每种可能的组合，看哪种能形成正方形
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                
                p1, p2, p3 = three_points[i], three_points[j], three_points[(set(range(3)) - {i, j}).pop()]
                
                # 假设p1和p2相邻，p3是对角
                # 计算第4个点
                p4 = (p1[0] + p3[0] - p2[0], p1[1] + p3[1] - p2[1])
                
                # 检查是否构成正方形
                if self.are_4_points_square([p1, p2, p3, p4]):
                    return p4
        
        return None

    def build_square_from_adjacent_corners(self, corner1: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
                                           corner2: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """
        从相邻的两个角点构建正方形
        按照平行辅助线的方向构建，并选择黑色占比最大的正方形
        """
        corner1_point, assist1_1, assist1_2 = corner1
        corner2_point, assist2_1, assist2_2 = corner2
        
        # 计算边长（使用共线边的长度）
        edge_length = np.sqrt((corner2_point[0] - corner1_point[0])**2 + 
                             (corner2_point[1] - corner1_point[1])**2)
        
        # 计算边的方向向量
        edge_vector = ((corner2_point[0] - corner1_point[0]) / edge_length,
                      (corner2_point[1] - corner1_point[1]) / edge_length)
        
        # 计算垂直向量（两个方向）
        perp_vector1 = (-edge_vector[1], edge_vector[0])
        perp_vector2 = (edge_vector[1], -edge_vector[0])
        
        best_square = None
        best_ratio = 0
        
        # 尝试两个方向构建正方形，选择黑色占比最大的
        for perp_vector in [perp_vector1, perp_vector2]:
            vertex1 = corner1_point
            vertex2 = corner2_point
            vertex3 = (int(corner2_point[0] + edge_length * perp_vector[0]),
                      int(corner2_point[1] + edge_length * perp_vector[1]))
            vertex4 = (int(corner1_point[0] + edge_length * perp_vector[0]),
                      int(corner1_point[1] + edge_length * perp_vector[1]))
            
            candidate_square = [vertex1, vertex2, vertex3, vertex4]
            
            # 检查几何形状是否为正方形
            if self.are_4_points_square(candidate_square):
                # 计算黑色像素占比
                black_ratio = self.calculate_square_black_ratio(candidate_square)
                if black_ratio > best_ratio:
                    best_ratio = black_ratio
                    best_square = candidate_square
        
        # 返回黑色占比最大的正方形（如果占比足够高）
        if best_square and best_ratio >= 0.95:
            return best_square
        
        return None
    
    def build_square_from_diagonal_corners(self, corner1: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]], 
                                         corner2: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
        """
        从对角的两个外直角构建正方形
        通过延长每个角点的辅助边，找到它们的交点来构建正方形
        """
        corner1_point, assist1_1, assist1_2 = corner1
        corner2_point, assist2_1, assist2_2 = corner2
        
        # 计算corner1的两条边的直线参数
        line1_1 = calculate_line_params(corner1_point, assist1_1)
        line1_2 = calculate_line_params(corner1_point, assist1_2)
        
        # 计算corner2的两条边的直线参数
        line2_1 = calculate_line_params(corner2_point, assist2_1)
        line2_2 = calculate_line_params(corner2_point, assist2_2)
        
        # 计算所有可能的交点
        intersections = [
            calculate_line_intersection(line1_1, line2_1),
            calculate_line_intersection(line1_1, line2_2),
            calculate_line_intersection(line1_2, line2_1),
            calculate_line_intersection(line1_2, line2_2)
        ]
        
        # 过滤掉无效的交点
        valid_intersections = []
        for intersection in intersections:
            if intersection is not None:
                x, y = intersection
                # 确保交点在合理范围内
                if 0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]:
                    valid_intersections.append((int(x), int(y)))
        
        # 如果有2个有效交点，尝试构建正方形
        if len(valid_intersections) >= 2:
            for i in range(len(valid_intersections)):
                for j in range(i + 1, len(valid_intersections)):
                    vertex3 = valid_intersections[i]
                    vertex4 = valid_intersections[j]
                    
                    # 构建候选正方形
                    candidate_square = [corner1_point, vertex3, corner2_point, vertex4]
                    
                    # 验证是否为有效正方形
                    if self.are_4_points_square(candidate_square):
                        return order_square_vertices(candidate_square)
                    
                    # 尝试另一种顺序
                    candidate_square2 = [corner1_point, vertex4, corner2_point, vertex3]
                    if self.are_4_points_square(candidate_square2):
                        return order_square_vertices(candidate_square2)
        
        return None

    def validate_square(self, square: List[Tuple[int, int]], min_black_ratio: float = None) -> bool:
        """
        验证正方形是否有效
        """
        if min_black_ratio is None:
            min_black_ratio = self.SQUARE_VALIDATION_BLACK_RATIO
            
        if len(square) != 4:
            return False
        
        # 检查所有顶点是否在图像范围内
        for x, y in square:
            if not (0 <= x < self.image_size[0] and 0 <= y < self.image_size[1]):
                return False
        
        # 计算正方形区域内的黑色像素比例
        black_ratio = self.calculate_square_black_ratio(square)
        
        return black_ratio >= min_black_ratio
    
    def visualize_corners(self, corners: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]):
        """
        在图像上可视化检测到的外直角
        """
        # 创建彩色图像用于显示
        vis_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        
        for i, (corner_point, assist_point1, assist_point2) in enumerate(corners):
            cx, cy = corner_point
            ax1, ay1 = assist_point1
            ax2, ay2 = assist_point2
            
            # 绘制直角点（红色圆圈）
            cv2.circle(vis_image, (cx, cy), 3, (0, 0, 255), -1)
            
            # 绘制辅助点（绿色圆圈）
            cv2.circle(vis_image, (ax1, ay1), 2, (0, 255, 0), -1)
            cv2.circle(vis_image, (ax2, ay2), 2, (0, 255, 0), -1)
            
            # 绘制直角边（蓝色线条）
            cv2.line(vis_image, (cx, cy), (ax1, ay1), (255, 0, 0), 1)
            cv2.line(vis_image, (cx, cy), (ax2, ay2), (255, 0, 0), 1)
            
            # 添加标号
            cv2.putText(vis_image, str(i), (cx + 5, cy - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return vis_image
    
    def visualize_squares(self, squares: List[List[Tuple[int, int]]]):
        """
        在图像上可视化重建的正方形
        """
        # 创建彩色图像用于显示
        vis_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        
        # 定义不同的颜色用于不同的正方形
        colors = [
            (0, 255, 255),    # 黄色
            (255, 0, 255),    # 品红色
            (255, 255, 0),    # 青色
            (128, 0, 255),    # 紫色
            (255, 128, 0),    # 橙色
            (0, 255, 128),    # 春绿色
            (128, 255, 0),    # 黄绿色
            (0, 128, 255),    # 天蓝色
        ]
        
        for i, square in enumerate(squares):
            color = colors[i % len(colors)]
            
            # 绘制正方形边框
            pts = np.array(square, dtype=np.int32)
            cv2.polylines(vis_image, [pts], True, color, 2)
            
            # 绘制顶点
            for j, (x, y) in enumerate(square):
                cv2.circle(vis_image, (x, y), 3, color, -1)
                cv2.putText(vis_image, str(j), (x + 5, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 在正方形中心添加编号
            center_x = int(sum(p[0] for p in square) / 4)
            center_y = int(sum(p[1] for p in square) / 4)
            cv2.putText(vis_image, f"S{i}", (center_x - 10, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return vis_image
    
    def visualize_corners_and_squares(self, corners: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]], 
                                    squares: List[List[Tuple[int, int]]]):
        """
        同时可视化角点和重建的正方形
        """
        # 先绘制角点
        vis_image = self.visualize_corners(corners)
        
        # 定义正方形的颜色
        square_colors = [
            (0, 255, 255),    # 黄色
            (255, 0, 255),    # 品红色
            (255, 255, 0),    # 青色
            (128, 0, 255),    # 紫色
            (255, 128, 0),    # 橙色
            (0, 255, 128),    # 春绿色
            (128, 255, 0),    # 黄绿色
            (0, 128, 255),    # 天蓝色
        ]
        
        # 绘制正方形
        for i, square in enumerate(squares):
            color = square_colors[i % len(square_colors)]
            
            # 绘制正方形边框（较粗的线条）
            pts = np.array(square, dtype=np.int32)
            cv2.polylines(vis_image, [pts], True, color, 3)
            
            # 在正方形中心添加编号
            center_x = int(sum(p[0] for p in square) / 4)
            center_y = int(sum(p[1] for p in square) / 4)
            cv2.putText(vis_image, f"SQ{i}", (center_x - 15, center_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return vis_image
    
    def save_image(self, filename: str, image=None):
        """保存图像"""
        if image is None:
            image = self.image
        cv2.imwrite(filename, image)


def test_external_image(image_path: str):
    """测试外部图像文件"""
    print(f"=== 测试外部图像: {image_path} ===")
    
    try:
        # 询问二值化设置
        print("\n二值化设置:")
        print("1. 使用OTSU自适应阈值 (推荐)")
        print("2. 使用固定阈值 (默认127)")
        print("3. 自定义阈值")
        print("4. 不进行二值化")
        
        binarize_choice = input("请选择二值化方式 (1-4, 默认为1): ").strip()
        
        binarize = True
        threshold = -1  # OTSU
        
        if binarize_choice == "2":
            threshold = 127
        elif binarize_choice == "3":
            try:
                threshold = int(input("请输入阈值 (0-255): "))
                threshold = max(0, min(255, threshold))
            except ValueError:
                print("输入无效，使用默认阈值127")
                threshold = 127
        elif binarize_choice == "4":
            binarize = False
        else:
            # 默认使用OTSU
            threshold = -1
        
        # 创建检测器并加载外部图像
        detector = CornerDetector()
        detector.load_image_from_file(image_path, binarize=binarize, threshold=threshold)
        
        print(f"图像处理完成，已保存 external_processed.png")
        
        print("检测外直角...")
        corners = detector.detect_all_outer_corners()
        
        # 打印简化的结果
        squares = detector.reconstruct_squares(corners)
        print(f"检测完成: {len(corners)}个角点, {len(squares)}个正方形")
        
        # 可视化结果
        corners_vis_image = detector.visualize_corners(corners)
        squares_vis_image = detector.visualize_squares(squares)
        combined_vis_image = detector.visualize_corners_and_squares(corners, squares)
        
        # 保存图像
        detector.save_image("external_corners.png", corners_vis_image)
        detector.save_image("external_squares.png", squares_vis_image)
        detector.save_image("external_combined.png", combined_vis_image)
        
        print("结果图像已保存: external_processed.png, external_corners.png, external_squares.png, external_combined.png")
        
        return True
        
    except Exception as e:
        print(f"处理外部图像失败: {e}")
        return False

def main():
    """主函数 - 简化的外部图像处理"""
    try:
        # 获取图像文件路径
        image_path = input("请输入图像文件路径 (直接回车退出): ").strip()
        
        if not image_path:
            print("程序退出")
            return
            
        # 简化的二值化选择
        print("\n二值化方式: 1=OTSU自适应 2=固定阈值127 3=不进行二值化")
        binarize_choice = input("选择 (1-3, 默认1): ").strip()
        
        # 创建检测器并处理
        detector = CornerDetector()
        
        # 加载图像
        if binarize_choice == "3":
            detector.load_image_from_file(image_path, binarize=False)
        elif binarize_choice == "2":
            detector.load_image_from_file(image_path, binarize=True, threshold=127)
        else:
            detector.load_image_from_file(image_path, binarize=True, threshold=-1)  # OTSU
        
        # 检测和重建
        corners = detector.detect_all_outer_corners()
        squares = detector.reconstruct_squares(corners)
        
        # 保存结果
        detector.save_image("result_processed.png")
        combined_image = detector.visualize_corners_and_squares(corners, squares)
        detector.save_image("result_combined.png", combined_image)
        
        print("结果已保存: result_processed.png, result_combined.png")
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()
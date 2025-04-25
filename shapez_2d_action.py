from itertools import islice
from PIL import Image
import random
import requests
import json
import base64
from PIL import Image, ImageDraw
import math
import random
import string

# 定义图形的旋转和染色操作函数

def action_to_natural_language(action_list):
    natural_language_action = []
    for action in action_list:
        if action == 'cut':
            natural_language_action.append('cut the right side of the shape')
        elif action == 'rotate_clockwise':
            natural_language_action.append('rotate the shape clockwise 90 degrees')
        elif action == 'rotate_counterclockwise':
            natural_language_action.append('rotate the shape counterclockwise 90 degrees')
        elif action == 'mirror':
            natural_language_action.append('perform a horizontal mirror operation on the entire shape.')
        else:
            if action[0] == 'fill':
                natural_language_action.append(f'fill all blank quadrants within the shape using \'{action[1]}\'.')
            elif action[0] == 'colorize':
                color_dict = {'r': 'red', 'g': 'green', 'b': 'blue', 'y': 'yellow', 'p': 'purple', 'c': 'cyan', 'w': 'white'}
                natural_language_action.append(f'colorize the top layer of the original shape with the color {color_dict[action[1]]}.')
    return natural_language_action

def randomize_shape(shape, steps):
    raw_action_list = generate_actions(num_act = steps)
    new_shape = execute_actions(raw_action_list, shape)
    natural_language_action = action_to_natural_language(raw_action_list)
    return new_shape, natural_language_action, raw_action_list

def randomize_shape_not_so_random(shape, steps, raw_action_list):
    new_action_list = raw_action_list.copy()
    while new_action_list == raw_action_list:
        mu = 3*(steps/10)
        sigma = max(1, mu/3)
        modified_action_number = max(1, min(steps, round(random.gauss(mu, sigma))))
        replace_action_list = generate_actions(num_act = modified_action_number)
        replace_id = random.sample(range(len(raw_action_list)), modified_action_number)
        new_action_list = raw_action_list.copy()
        for i in range(modified_action_number):
            new_action_list[replace_id[i]] = replace_action_list[i]

    new_shape = execute_actions(new_action_list, shape)
    natural_language_action = action_to_natural_language(new_action_list)
    return new_shape, natural_language_action, new_action_list

def extract_unique_shapes_and_colors(structure):
    # 拆分层，用冒号分隔
    layers = structure.split(":")
    
    # 使用集合去重
    shapes = set()
    colors = set()
    
    # 遍历每层
    for layer in layers:
        # 将每层按两个字符分隔，如 "Cr" 表示红色的圆形
        quadrants = [layer[i:i+2] for i in range(0, len(layer), 2)]
        
        # 解析每个象限的形状和颜色
        for quadrant in quadrants:
            if quadrant != "--":  # 跳过空位
                shape = quadrant[0]  # 形状代码
                color = quadrant[1]  # 颜色代码
                shapes.add(shape)
                colors.add(color)
    
    # 转换集合为列表并返回
    return list(shapes), list(colors)

def generate_raw_minerals(num_minerals):
    # 定义形状和颜色的代码
    shapes = ['C', 'R', 'W', 'S']  # C: Circle, R: Rectangle, W: Windmill, S: Star
    colors = ['r', 'g', 'b', 'y', 'p', 'c', 'u', 'w']  # r: Red, g: Green, etc.

    def generate_mineral_layer():
        # 初始化四个象限为空位
        layer = ['--'] * 4
        
        # 随机选择填充2到4个象限
        num_filled = random.randint(2, 4)

        filled_positions = random.sample(range(4), num_filled)

        for pos in filled_positions:
            # 在选定象限位置填入随机形状和颜色
            shape = random.choice(shapes)
            color = random.choice(colors)
            layer[pos] = f"{shape}{color}"

        return "".join(layer)  # 连接成一个字符串表示单层矿物结构

    # 生成指定数量的原始矿物结构
    raw_minerals = [generate_mineral_layer() for _ in range(num_minerals)]
    return raw_minerals

def generate_actions(num_act = 3, seed = None):
    if seed is not None:
        random.seed(seed)

    # actions_list = ['fill', 'cut']        
    actions_list = ['cut', 'rotate_clockwise', 'rotate_counterclockwise', 'fill', 'colorize', 'mirror', 'fill']        
    actions = []
    for _ in range(num_act):
        action = random.choice(actions_list)
        if action == 'fill':
            new_shape = generate_shape_structures(num_structures = 1, num_layers = 1, color = 'yes', num_shapes = 1, num_colors = 1, all_the_same = True)[0]
            fill_shape = new_shape[:2]
            actions.append([action, fill_shape])
        elif action == 'colorize':
            color = random.choice(['r', 'g', 'b', 'y', 'p', 'c', 'w'])
            actions.append([action, color])
        else:
            actions.append(action)
    return actions

def draw_shape(shape_code):
    generator = ShapeGenerator()
    image = generator.generate_image(shape_code)
    image.show()

class ShapeGenerator:
    def __init__(self):
        self.colors = {
            'r': (255, 102, 106),  # 红色
            'g': (120, 255, 102),  # 绿色
            'b': (102, 167, 255),  # 蓝色
            'y': (252, 245, 42),   # 黄色
            'p': (221, 102, 255),  # 紫色
            'c': (0, 252, 255),    # 青色
            'u': (170, 170, 170),  # 无色
            'w': (255, 255, 255)   # 白色
        }
        
        self.size = 512  # 画布大小
        self.center = self.size // 2
        self.radius = self.size // 4
        
    def create_canvas(self):
        """创建一个新的透明画布"""
        return Image.new('RGBA', (self.size, self.size), (0, 0, 0, 0))
    
    def draw_circle(self, center, draw, position, color, border_width = 10):
        """绘制1/4圆形"""
        x, y = position
        quadrant_position = self._get_quadrant_from_position(position)
        start_angle, end_angle = self._get_quadrant_angles(quadrant_position)
        
        # 画内部填充
        draw.pieslice((self.center - center, self.center - center,
                       self.center + center, self.center + center),
                      start=start_angle, end=end_angle,
                      fill=color)
        
        # 画圆弧边框
        draw.arc((self.center - center, self.center - center,
                  self.center + center, self.center + center),
                 start=start_angle, end=end_angle,
                 fill=(85, 85, 85), width=border_width)
        
        # 画射线边框
        if start_angle == 270:  # 右上象限
            draw.line([(self.center, self.center), (self.size - (self.center - center), self.center)], 
                     fill=(85, 85, 85), width=border_width)  # 水平线
            draw.line([(self.center, self.center), (self.center, self.center - center)], 
                     fill=(85, 85, 85), width=border_width)  # 垂直线
        elif start_angle == 0:  # 右下象限
            draw.line([(self.center, self.center), (self.size - (self.center - center), self.center)], 
                     fill=(85, 85, 85), width=border_width)
            draw.line([(self.center, self.center), (self.center, self.size - (self.center - center))], 
                     fill=(85, 85, 85), width=border_width)
        elif start_angle == 90:  # 左下象限
            draw.line([(self.center, self.center), (self.center - center, self.center)], 
                     fill=(85, 85, 85), width=border_width)
            draw.line([(self.center, self.center), (self.center, self.size - (self.center - center))], 
                     fill=(85, 85, 85), width=border_width)
        else:  # 左上象限
            draw.line([(self.center, self.center), (self.center - center, self.center)], 
                     fill=(85, 85, 85), width=border_width)
            draw.line([(self.center, self.center), (self.center, self.center - center)], 
                     fill=(85, 85, 85), width=border_width)

    def _get_quadrant_from_position(self, position):
        """根据位置确定象限"""
        x, y = position
        if x > self.center:  # 右侧
            return 0 if y < self.center else 1  # 右上为0，右下为1
        else:  # 左侧
            return 2 if y > self.center else 3  # 左下为2，左上为3

    def _get_quadrant_angles(self, quadrant):
        """获取象限的起始和结束角度"""
        angles = {
            0: (270, 360),  # 右上象限
            1: (0, 90),     # 右下象限
            2: (90, 180),   # 左下象限
            3: (180, 270)   # 左上象限
        }
        return angles[quadrant]
    
    def draw_rectangle(self, center, draw, position, color, border_width = 10):
        """绘制1/4矩形"""
        x, y = position
        quadrant_position = self._get_quadrant_from_position(position)
        
        # 计算矩形的大小（使用象限的大小）

        rect_size = center
        bias = border_width/2.45

        if quadrant_position == 0:  # 右上象限
                # 矩形的四个点保持不变
                rect_points = [
                    (self.center + border_width - bias, self.center - rect_size +border_width),  # 左上
                    (self.center + rect_size - border_width, self.center - rect_size + border_width),  # 右上
                    (self.center + rect_size - border_width, self.center - border_width + bias),  # 右下
                    (self.center + border_width - bias, self.center - border_width + bias)  # 左下
                ]

                line_points = [
                    (self.center - bias, self.center - rect_size),  # 左上
                    (self.center + rect_size, self.center - rect_size),  # 右上
                    (self.center + rect_size, self.center + bias),  # 右下
                    (self.center - bias, self.center + bias)  # 左下
                ]
                
        elif quadrant_position == 1:  # 右下象限
            rect_points = [
                (self.center + border_width - bias, self.center + border_width - bias),  # 左上
                (self.center + rect_size - border_width, self.center + border_width - bias),  # 右上
                (self.center + rect_size - border_width, self.center + rect_size - border_width),  # 右下
                (self.center + border_width - bias, self.center + rect_size - border_width)  # 左下
            ]                

            
            line_points = [
                (self.center - bias, self.center - bias),  # 左上
                (self.center + rect_size, self.center - bias),  # 右上
                (self.center + rect_size, self.center + rect_size),  # 右下
                (self.center - bias, self.center + rect_size)  # 左下
            
            ]
            
        elif quadrant_position == 2:  # 左下象限
            rect_points = [
                (self.center - rect_size + border_width, self.center + border_width - bias),  # 左上
                (self.center - border_width + bias, self.center + border_width - bias),  # 右上
                (self.center - border_width + bias, self.center + rect_size - border_width),  # 右下
                (self.center - rect_size + border_width, self.center + rect_size - border_width)  # 左下
            ]
            
            line_points = [
                (self.center - rect_size, self.center - bias),  # 左上
                (self.center + bias, self.center - bias),  # 右上
                (self.center + bias, self.center + rect_size),  # 右下
                (self.center - rect_size, self.center + rect_size)  # 左下
            ]
            
        else:  # 左上象限
            rect_points = [
                (self.center - rect_size + border_width, self.center - rect_size + border_width),  # 左上
                (self.center - border_width + bias, self.center - rect_size + border_width),  # 右上
                (self.center - border_width + bias, self.center - border_width + bias),  # 右下
                (self.center - rect_size + border_width, self.center - border_width + bias)  # 左下
            ]
            
            line_points = [
                (self.center - rect_size, self.center - rect_size),  # 左上
                (self.center + bias, self.center - rect_size),  # 右上
                (self.center + bias, self.center + bias),  # 右下
                (self.center - rect_size, self.center + bias)  # 左下
            ]
            
        # 绘制填充
        draw.polygon(line_points, fill=(85, 85, 85))  # 画扇形
        draw.polygon(rect_points, fill=color)  # 画扇形

    def draw_star(self, center, draw, position, color, border_width = 10):
        """绘制1/4星形"""
        x, y = position
        quadrant_position = self._get_quadrant_from_position(position)

        alpha = 2

        beta_1 = alpha/(1+math.sqrt(1+alpha*alpha))
        beta_2 = 1/alpha + math.sqrt(1/(alpha*alpha) + 1) 
           
        rect_size = center
        bias = border_width/2.45

        if quadrant_position == 0:  # 右上象限
                # 矩形的四个点保持不变
                rect_points = [
                    (self.center + border_width - bias, self.center - rect_size/2 +border_width*beta_1),  # 左上
                    (self.center + rect_size - border_width*beta_2, self.center - rect_size + border_width*beta_2),  # 右上
                    (self.center + rect_size/2 - border_width*beta_1, self.center - border_width + bias),  # 右下
                    (self.center + border_width - bias, self.center - border_width + bias)  # 左下
                ]

                line_points = [
                    (self.center - bias, self.center - rect_size/2),  # 左上
                    (self.center + rect_size, self.center - rect_size),  # 右上
                    (self.center + rect_size/2, self.center + bias),  # 右下
                    (self.center - bias, self.center + bias)  # 左下
                ]
                
        elif quadrant_position == 1:  # 右下象限
            rect_points = [
                (self.center + border_width - bias, self.center + border_width - bias),  # 左上
                (self.center + rect_size/2 - border_width*beta_1, self.center + border_width - bias),  # 右上
                (self.center + rect_size - border_width*beta_2, self.center + rect_size - border_width*beta_2),  # 右下
                (self.center + border_width - bias, self.center + rect_size/2 - border_width*beta_1)  # 左下
            ]                

            
            line_points = [
                (self.center - bias, self.center - bias),  # 左上
                (self.center + rect_size/2, self.center - bias),  # 右上
                (self.center + rect_size, self.center + rect_size),  # 右下
                (self.center - bias, self.center + rect_size/2)  # 左下
            
            ]
            
        elif quadrant_position == 2:  # 左下象限
            rect_points = [
                (self.center - rect_size/2 + border_width*beta_1, self.center + border_width - bias),  # 左上
                (self.center - border_width + bias, self.center + border_width - bias),  # 右上
                (self.center - border_width + bias, self.center + rect_size/2 - border_width*beta_1),  # 右下
                (self.center - rect_size + border_width*beta_2, self.center + rect_size - border_width*beta_2)  # 左下
            ]
            
            line_points = [
                (self.center - rect_size/2, self.center - bias),  # 左上
                (self.center + bias, self.center - bias),  # 右上
                (self.center + bias, self.center + rect_size/2),  # 右下
                (self.center - rect_size, self.center + rect_size)  # 左下
            ]
            
        else:  # 左上象限
            rect_points = [
                (self.center - rect_size + border_width*beta_2, self.center - rect_size + border_width*beta_2),  # 左上
                (self.center - border_width + bias, self.center - rect_size/2 + border_width*beta_1),  # 右上
                (self.center - border_width + bias, self.center - border_width + bias),  # 右下
                (self.center - rect_size/2 + border_width*beta_1, self.center - border_width + bias)  # 左下
            ]
            
            line_points = [
                (self.center - rect_size, self.center - rect_size),  # 左上
                (self.center + bias, self.center - rect_size/2),  # 右上
                (self.center + bias, self.center + bias),  # 右下
                (self.center - rect_size/2, self.center + bias)  # 左下
            ]
            
        # 绘制填充
        draw.polygon(line_points, fill=(85, 85, 85))  # 画扇形
        draw.polygon(rect_points, fill=color)  # 画扇形

    def draw_windmill(self, center, draw, position, color, border_width = 10):
        x, y = position
        quadrant_position = self._get_quadrant_from_position(position)
        
        # 计算矩形的大小（使用象限的大小）

        rect_size = center
        short_size = rect_size / 2
        alpha = 2
        beta_1 = alpha/(1+math.sqrt(1+alpha*alpha))
        beta_2 = 1/alpha + math.sqrt(1/(alpha*alpha) + 1)
                         
        bias_1 = border_width/2.45
        bias_2 = border_width/(2.45*alpha)

        if quadrant_position == 0:  # 右上象限
            # 矩形的四个点保持不变
            rect_points = [
                (self.center + border_width - bias_1, self.center - rect_size + short_size +border_width*beta_1 + bias_2),  # 左上
                (self.center + rect_size - border_width, self.center - rect_size + border_width*beta_2),  # 右上
                (self.center + rect_size - border_width, self.center - border_width + bias_1),  # 右下
                (self.center + border_width - bias_1, self.center - border_width + bias_1)  # 左下
            ]

            line_points = [
                (self.center - bias_1, self.center - rect_size + short_size + bias_2),  # 左上
                (self.center + rect_size, self.center - rect_size),  # 右上
                (self.center + rect_size, self.center + bias_1),  # 右下
                (self.center - bias_1, self.center + bias_1)  # 左下
            ]
            
        elif quadrant_position == 1:  # 右下象限
            rect_points = [
                (self.center + border_width - bias_1, self.center + border_width - bias_1),  # 左上
                (self.center + rect_size - short_size - border_width*beta_1 - bias_2, self.center + border_width - bias_1),  # 右上
                (self.center + rect_size - border_width*beta_2, self.center + rect_size - border_width),  # 右下
                (self.center + border_width - bias_1, self.center + rect_size - border_width)  # 左下
            ]                

            
            line_points = [
                (self.center - bias_1, self.center - bias_1),  # 左上
                (self.center + rect_size - short_size - bias_2, self.center - bias_1),  # 右上
                (self.center + rect_size, self.center + rect_size),  # 右下
                (self.center - bias_1, self.center + rect_size)  # 左下
            
            ]
            
        elif quadrant_position == 2:  # 左下象限
            rect_points = [
                (self.center - rect_size + border_width, self.center + border_width - bias_1),  # 左上
                (self.center - border_width + bias_1, self.center + border_width - bias_1),  # 右上
                (self.center - border_width + bias_1, self.center + rect_size - short_size - border_width*beta_1 - bias_2),  # 右下
                (self.center - rect_size + border_width, self.center + rect_size - border_width*beta_2)  # 左下
            ]
            
            line_points = [
                (self.center - rect_size, self.center - bias_1),  # 左上
                (self.center + bias_1, self.center - bias_1),  # 右上
                (self.center + bias_1, self.center + rect_size - short_size - bias_2),  # 右下
                (self.center - rect_size, self.center + rect_size)  # 左下
            ]
            
        else:  # 左上象限
            rect_points = [
                (self.center - rect_size + border_width*beta_2, self.center - rect_size + border_width),  # 左上
                (self.center - border_width + bias_1, self.center - rect_size + border_width),  # 右上
                (self.center - border_width + bias_1, self.center - border_width + bias_1),  # 右下
                (self.center - rect_size + short_size + border_width*beta_1 + bias_2, self.center - border_width + bias_1)  # 左下
            ]
            
            line_points = [
                (self.center - rect_size, self.center - rect_size),  # 左上
                (self.center + bias_1, self.center - rect_size),  # 右上
                (self.center + bias_1, self.center + bias_1),  # 右下
                (self.center - rect_size + short_size + bias_2, self.center + bias_1)  # 左下
            ]
        
        # 绘制填充
        draw.polygon(line_points, fill=(85, 85, 85))  # 画扇形
        draw.polygon(rect_points, fill=color)  # 画扇形

    def generate_image(self, shape_code): 
        """根据形状代码生成图像"""
        layers = shape_code.split(':')
        image = self.create_canvas()
        
        border_width = 10
        for layer_index, layer in enumerate(layers):
            layer_image = self.create_canvas()
            draw = ImageDraw.Draw(layer_image)
            
            scale_ratio = 1 - (layer_index * 0.25)  # 每层缩小20%

            if len(layer) != 8 and layer != '':
                raise ValueError("每层必须是8个字符(4个象限)")
                
            # 修改象限顺序为时针从右上开始
            quadrants = [(layer[i:i+2]) for i in range(0, len(layer), 2)]
            positions = [
                (self.center + self.radius, self.center - self.radius),  # 右上
                (self.center + self.radius, self.center + self.radius),  # 右下
                (self.center - self.radius, self.center + self.radius),  # 左下
                (self.center - self.radius, self.center - self.radius)   # 左上
            ]
            
            for quad, pos in zip(quadrants, positions):
                if quad == '--':
                    continue
                    
                shape, color = quad
                if color not in self.colors:
                    raise ValueError(f"未知的颜色代码: {color}")
                    
                color_rgb = self.colors[color]
                
                if shape == 'C':
                    self.draw_circle(center=self.center*scale_ratio, draw=draw, position=pos, color=color_rgb, border_width=border_width)
                elif shape == 'R':
                    self.draw_rectangle(center=self.center*scale_ratio, draw=draw, position=pos, color=color_rgb, border_width=border_width)
                elif shape == 'S':
                    self.draw_star(center=self.center*scale_ratio, draw=draw, position=pos, color=color_rgb, border_width=border_width)
                elif shape == 'W':
                    self.draw_windmill(center=self.center*scale_ratio, draw=draw, position=pos, color=color_rgb, border_width=border_width)
                else:
                    raise ValueError(f"未知的形状代码: {shape}")
            
            # 合并图层
            image = Image.alpha_composite(image, layer_image)
            
        return image 

    def save_shape(self, shape_code, save_path = None):
        """
        生成并保存形状图像
        
        参数:
        - shape_code: 形状代码字符串
        - save_path: 保存图像的完整路径（包括文件名和扩展名）
        """
        if save_path is None:
            random_path = ''.join(random.choices(string.ascii_letters + string.digits, k=15))
            save_path = f"images/{random_path}.png"

        try:
            # 生成图像
            image = self.generate_image(shape_code)
            
            # 保存图像
            image.save(save_path)
            # print(f"图像已成功保存到: {save_path}")
            
            return save_path
        except Exception as e:
            print(f"保存图像失败: {str(e)}")
            return False

def generate_shape_structures(num_structures, num_layers = 'random', color = 'yes', num_shapes = 4, num_colors = 8, seed=None, all_the_same = False):

    if seed is not None:
        random.seed(seed) 

    # 定义形状和颜色的代码
    all_shapes = ['C', 'R', 'W', 'S']  # C: Circle, R: Rectangle, W: Windmill, S: Star
    all_colors = ['r', 'g', 'b', 'y', 'p', 'c', 'u', 'w']  # r: Red, g: Green, b: Blue, y: Yellow, p: Purple, c: Cyan, u: Uncolored, w: White
    shapes = random.sample(all_shapes, k=num_shapes)
    colors = random.sample(all_colors, k=num_colors)
    def generate_shape(color):
        if color == 'yes':
            return f"{random.choice(shapes)}{random.choice(colors)}" if random.choice([True, False]) else "--"
        # 随机生成一个形状（形状在前，颜色在后）或空位
        else:
            return f"{random.choice(shapes)}u" if random.choice([True, False]) else "--"
        
    def generate_non_empty_layer(color):
        # 生成至少一个象限有形状的一层
        while True:
            layer = [generate_shape(color) for _ in range(4)]
            if any(shape != "--" for shape in layer):  # 确保至少一个象限有形状
                return layer

    def check_layer_validity(current_layer, previous_layer):
        # 统计当前层中有形状的象限数量及其索引
        filled_quadrants = [i for i, shape in enumerate(current_layer) if shape != "--"]
        num_filled = len(filled_quadrants)
        
        # 规则 1：当前层只有一个象限有形状
        if num_filled == 1:
            if previous_layer[filled_quadrants[0]] == "--":
                return False  # 上一层相同象限必须有形状

        # 规则 2：当前层有两个不相邻的象限有形状
        elif num_filled == 2:
            q1, q2 = filled_quadrants
            if abs(q1 - q2) % 2 == 0:  # 检查是否相邻
                # 如果不相邻，上一层的对应象限必须有形状
                if previous_layer[q1] == "--" or previous_layer[q2] == "--":
                    return False
            else:
                if previous_layer[q1] == "--" and previous_layer[q2] == "--":
                    return False
                
        # 规则 3：当前层有三和四个象限有形状
        elif num_filled == 3 or num_filled == 4:
            # 有形状的三个象限中至少一个在上一层中有形状
            if all(previous_layer[q] == "--" for q in filled_quadrants):
                return False
        
        return True


    def generate_all_quadrant_same_shape(colors, shapes):
        color = random.choice(colors)
        shape = random.choice(shapes)

        layer = [f"{shape}{color}"] * 4

        return layer

    def generate_shape_structure(num_layers, color):
        if num_layers == 'random':
            num_layers = random.randint(1, 4)
        # 随机确定层数
        if all_the_same:
            layers = [generate_all_quadrant_same_shape(colors, shapes)]  # 初始化第1层，确保有形状
        else:
            layers = [generate_non_empty_layer(color)]  # 初始化第1层，确保有形状

        # 从第2层开始逐层生成
        for _ in range(1, num_layers):
            while True:
                new_layer = generate_non_empty_layer(color)
                if check_layer_validity(new_layer, layers[-1]):  # 检查生成的层是否符合规则
                    layers.append(new_layer)  # 如果符合规则，加入该层
                    break  # 跳出循环，生成下一层
        
        # 将每层连接成字符串，并返回结构
        return ":".join("".join(layer) for layer in layers)

    # 主循环，生成指定数量的形状结构
    shape_structures = [generate_shape_structure(num_layers, color) for _ in range(num_structures)]
    return shape_structures

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_url(path):
    headers = {'Authorization': 'tLivBvDacjF2fCqMEykEaUFzKmBOn1kU'}
    files = {'smfile': open(path, 'rb')}
    url = 'https://sm.ms/api/v2/upload'
    res = requests.post(url, files=files, headers=headers).json()
    
    if res['success']==True:
        return res['data']['url']
    else:
        if res['code'] == 'image_repeated':
            return res['images']
        else:
            return res['message']

def concatenate_images_horizontally(image_paths, savefolder = 'images'):

    image_number = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz', k=20))
    save_path = f"{savefolder}/{image_number}.png"

    # 打开所有图片
    images = [Image.open(image_path) for image_path in image_paths]
    
    # 确保所有图片的高度一致，若不一致则调整为最小高度
    min_height = min(img.height for img in images)
    resized_images = [img.resize((int(img.width * min_height / img.height), min_height)) for img in images]
    
    # 计算总宽度
    total_width = sum(img.width for img in resized_images)
    
    # 创建新图像画布
    new_image = Image.new('RGB', (total_width, min_height))
    
    # 逐个拼接图片
    current_x = 0
    for img in resized_images:
        new_image.paste(img, (current_x, 0))
        current_x += img.width
    
    # 保存拼接后的图像
    new_image.save(save_path)
    return save_path

# def execute(action, shape_code, color = None, up_shape_code = None, down_shape_code = None):
#     if action == 'rotate_clockwise':
#         new_shape_code = rotate_clockwise(shape_code)
#     elif action == 'rotate_counterclockwise':
#         new_shape_code = rotate_counterclockwise(shape_code)
#     elif action == 'cut':
#         new_shape_code = cut(shape_code)
#     elif action == 'stack_shapes_on_top':
#         if up_shape_code is None:
#             up_shape_code = generate_shape_structures(1)[0]
#         new_shape_code = stack(up_shape_code, shape_code)
#     elif action == 'stack_shapes_below':
#         if down_shape_code is None:
#             down_shape_code = generate_shape_structures(1)[0]
#         new_shape_code = stack(shape_code, down_shape_code)
#     elif action == 'colorize':
#         new_shape_code = colorize(shape_code, color)
#     return new_shape_code

def execute_actions(actions, shape_code):
    def execute(action, shape_code):
        if action == 'rotate_clockwise':
            new_shape_code = rotate_clockwise(shape_code)
        elif action == 'rotate_counterclockwise':
            new_shape_code = rotate_counterclockwise(shape_code)
        elif action == 'cut':
            new_shape_code = cut(shape_code)
        elif action == 'mirror':    
            new_shape_code = mirror(shape_code)
        else:
            if action[0] == 'fill':
                new_shape_code = fill(shape_code, action[1])
            elif action[0] == 'colorize':
                new_shape_code = colorize(shape_code, action[1])
        return new_shape_code
    
    for action in actions:
        shape_code = execute(action, shape_code)
    return shape_code

# 切割（去除输入图形的每一层的一二象限的形状）
def cut(shape, out_dict = False):
    if type(shape) != dict:
        shape = decode_shape_code(shape)
    new_shape = shape.copy()
    for layer in shape:
        # 保留第三和第四象限，清空第一和第二象限
        new_shape[layer] = f"----{shape[layer][4:]}"

    new_shape = drop(new_shape)
    if out_dict:
        return new_shape
    else:
        return make_shape_code(new_shape)

# 顺时针旋转90°（把所有的图形顺时针旋转90°）
def rotate_clockwise(shape, out_dict = False):
    if type(shape) != dict:
        shape = decode_shape_code(shape)
    new_shape = shape.copy()
    for layer in shape:
        # 顺时针旋转：第一象限变成第二象限，依次类推
        new_shape[layer] = shape[layer][6:8] + shape[layer][0:6]
    if out_dict:
        return new_shape
    else:
        return make_shape_code(new_shape)    

# 逆时针旋转90°（把所有的图形逆时针旋转90°）
def rotate_counterclockwise(shape, out_dict = False):
    if type(shape) != dict:
        shape = decode_shape_code(shape)
    new_shape = shape.copy()
    for layer in shape:
        # 逆时针旋转：第一象限变成第四象限，依次类推
        new_shape[layer] = shape[layer][2:] + shape[layer][0:2]
    if out_dict:
        return new_shape
    else:
        return make_shape_code(new_shape)

# 堆叠（把一层堆叠到另一层）
def stack(upper_shape, lower_shape, out_dict = False):
    if type(upper_shape) != dict:
        upper_shape = decode_shape_code(upper_shape)
    if type(lower_shape) != dict:
        lower_shape = decode_shape_code(lower_shape) 
    # 合并 upper_shape 和 lower_shape 到新图形中，upper_shape 在上，lower_shape 在下
    new_shape = {}
    
    lower_shape = [add_empty_layer(lower_shape) for _ in range(1, 5)][-1]

    # 将 lower_shape 层放入新图形中
    layer_count = 1
    for layer in lower_shape:
        new_shape[f"layer{layer_count}"] = lower_shape[layer]
        layer_count += 1
    
    # 将 upper_shape 层放入新图形的上层
    for layer in upper_shape:
        new_shape[f"layer{layer_count}"] = upper_shape[layer]
        layer_count += 1

    new_shape = drop(new_shape)
    new_shape = remove_empty_layers(new_shape)
    new_shape = dict(islice(new_shape.items(), 4))

    if out_dict:
        return new_shape
    else:
        return make_shape_code(new_shape)

def mirror(shape, out_dict = False):
    def mirror_layer(layer_shape_code):
        q1 = layer_shape_code[0:2]
        q2 = layer_shape_code[2:4]
        q3 = layer_shape_code[4:6]
        q4 = layer_shape_code[6:8]

        new_layer_shape_code = q4 + q3 + q2 + q1
        return new_layer_shape_code
    if type(shape) != dict:
        shape = decode_shape_code(shape)
    new_shape = shape.copy()
    for layer in shape:
        new_shape[layer] = mirror_layer(shape[layer])

    if out_dict:
        return new_shape
    else:
        return make_shape_code(new_shape)

def fill(shape, fill_shape, out_dict = False):
    def fill_layer(layer_shape_code, fill_shape):
        q1 = layer_shape_code[0:2]
        q2 = layer_shape_code[2:4]
        q3 = layer_shape_code[4:6]
        q4 = layer_shape_code[6:8]

        if q1 == '--':
            q1 = fill_shape
        if q2 == '--':
            q2 = fill_shape
        if q3 == '--':
            q3 = fill_shape
        if q4 == '--':
            q4 = fill_shape

        new_layer_shape_code = q1 + q2 + q3 + q4
        return new_layer_shape_code

    if type(shape) != dict:
        shape = decode_shape_code(shape)
    new_shape = shape.copy()
    for layer in shape:
        new_shape[layer] = fill_layer(shape[layer], fill_shape)

    if out_dict:
        return new_shape
    else:
        return make_shape_code(new_shape)

def add_empty_layer(shape):
    # 获取当前层数
    new_shape = shape.copy()
    current_layers = len(shape)
    
    # 新增一层空白层，并将其编号为第 current_layers + 1 层
    new_shape[f"layer{current_layers + 1}"] = "--------"
    
    return new_shape

def remove_empty_layers(shape):
    new_shape = {}
    
    # 过滤掉所有全为 "--------" 的层
    non_empty_layers = [layer for layer in shape if shape[layer] != "--------"]
    
    # 重新分配层数，让非空层向下移动
    for i, layer in enumerate(non_empty_layers):
        new_shape[f"layer{i + 1}"] = shape[layer]
    
    return new_shape

def count_shapes_in_layer(layer_shape):
    """
    统计给定层中有多少个形状。
    
    参数：
    - layer_shape: str，表示层的形状代码，例如 'Su--Ry--'
    
    返回值：
    - int，当前层的形状数量
    """
    shape_count = 0
    for i in range(0, len(layer_shape), 2):
        if layer_shape[i:i+2] != "--":
            shape_count += 1
    return shape_count

def get_non_empty_positions(layer_shape):
    """
    获取给定层中所有非空位置的坐标。

    参数：
    - layer_shape: str，表示层的形状代码，例如 'Su--Ry--'

    返回值：
    - list，包含所有非空位置的坐标（索引）。
    """
    positions = []
    for i in range(0, len(layer_shape), 2):
        if layer_shape[i:i+2] != "--":
            positions.append(i // 2)  # 使用 i // 2 表示位置
    return positions

def get_shapes_at_positions(layer_shape, positions):
    """
    根据指定位置的数列获取对应的形状字符串。

    参数：
    - layer_shape: str，表示层的形状代码，例如 'Su--Ry--'
    - positions: list，包含非空位置的坐标，例如 [0, 2]

    返回值：
    - list，包含数列中每个位置对应的形状字符串。
    """
    shapes = ''.join(layer_shape[pos * 2 : pos * 2 + 2] for pos in positions)

    return shapes

def swap_shapes_at_positions(str1, str2, positions):
    """
    在两个字符串中调换指定位置上的所有字符。

    参数：
    - str1: str，第一个字符串，例如 'Su--Ry--'
    - str2: str，第二个字符串，例如 'Ab--Cd--'
    - positions: list，包含需要交换的坐标位置，例如 [0, 2]

    返回值：
    - tuple，包含交换后的两个新字符串
    """
    # 将字符串转换为可修改的列表
    list1 = list(str1)
    list2 = list(str2)
    
    # 交换指定位置的字符
    for pos in positions:
        # 计算字符的起始位置（每个形状占两个字符）
        start_index = pos * 2
        # 交换两个字符串中对应位置的两个字符
        list1[start_index:start_index+2], list2[start_index:start_index+2] = (
            list2[start_index:start_index+2],
            list1[start_index:start_index+2],
        )
    
    # 将列表重新转换为字符串
    swapped_str1 = ''.join(list1)
    swapped_str2 = ''.join(list2)
    
    return swapped_str1, swapped_str2

def drop(shape):

        # 掉落过程，从第二层开始，直到最上层，不检查第一层
    new_shape = shape.copy()
    
    layers = list(new_shape.keys())
    for i in range(1, len(layers)):  # 从第二层到最上层
        current_layer = list(new_shape[layers[i]])
        current_upper_layer = new_shape[layers[i - 1]]

        curren_layer_shape_count = count_shapes_in_layer(new_shape[layers[i]]) 

        if curren_layer_shape_count == 0:
            continue

        elif curren_layer_shape_count == 2:

            positions = get_non_empty_positions(new_shape[layers[i]])
            a, b = positions[0], positions[1]
            if b - a == 2:
                for j in [a, b]:
                    part_positions = [j]
                    not_empty_part_of_upper_layer = get_shapes_at_positions(current_upper_layer, part_positions) 
                    if count_shapes_in_layer(not_empty_part_of_upper_layer) == 0:
                        k = i - 1
                        not_empty_part_of_target_layer = get_shapes_at_positions(new_shape[layers[k]], part_positions)
                        while k >= 0 and count_shapes_in_layer(not_empty_part_of_target_layer) == 0:
                            k -= 1
                            not_empty_part_of_target_layer = get_shapes_at_positions(new_shape[layers[k]], part_positions)
                        target_layer = max(0, k + 1)
                        new_shape[layers[target_layer]], current_layer = swap_shapes_at_positions(new_shape[layers[target_layer]], "".join(current_layer), part_positions)   

            else:
                not_empty_part_of_upper_layer = get_shapes_at_positions(current_upper_layer, positions) 
                if count_shapes_in_layer(not_empty_part_of_upper_layer) == 0:
                    k = i - 1
                    not_empty_part_of_target_layer = get_shapes_at_positions(new_shape[layers[k]], positions)
                    while k >= 0 and count_shapes_in_layer(not_empty_part_of_target_layer) == 0:
                        k -= 1
                        not_empty_part_of_target_layer = get_shapes_at_positions(new_shape[layers[k]], positions)
                    target_layer = max(0, k + 1)
                    new_shape[layers[target_layer]], current_layer = swap_shapes_at_positions(new_shape[layers[target_layer]], "".join(current_layer), positions)
            new_shape[layers[i]] = "".join(current_layer)
                        
        elif curren_layer_shape_count == 1 or curren_layer_shape_count == 3 or curren_layer_shape_count == 4:
            positions = get_non_empty_positions(new_shape[layers[i]])
            not_empty_part_of_upper_layer = get_shapes_at_positions(current_upper_layer, positions) 
            if count_shapes_in_layer(not_empty_part_of_upper_layer) == 0:
                k = i - 1
                not_empty_part_of_target_layer = get_shapes_at_positions(new_shape[layers[k]], positions)
                while k >= 0 and count_shapes_in_layer(not_empty_part_of_target_layer) == 0:
                    k -= 1
                    not_empty_part_of_target_layer = get_shapes_at_positions(new_shape[layers[k]], positions)
                target_layer = max(0, k + 1)
                new_shape[layers[target_layer]], current_layer = swap_shapes_at_positions(new_shape[layers[target_layer]], "".join(current_layer), positions)            


        # 更新当前层
            new_shape[layers[i]] = "".join(current_layer)

    return new_shape

def mix_colors(color1, color2):
    color_map = {
        'r': 'r', 'g': 'g', 'b': 'b',
        'y': ('r', 'g'), 'p': ('r', 'b'),
        'c': ('g', 'b'), 'w': ('r', 'g', 'b')
    }
    
    all_colors = set(color_map.get(color1, ())) | set(color_map.get(color2, ()))

    if len(all_colors) == 1:
        return all_colors.pop()
    elif len(all_colors) == 2:
        return ('y' if 'r' in all_colors and 'g' in all_colors else
                'p' if 'r' in all_colors and 'b' in all_colors else
                'c')
    elif len(all_colors) == 3:
        return 'w'

def make_shape_code(shape):
    """
    用 ":" 分割字符串，得到每层的编码
    从dict->str
    """
    return ":".join("".join(shape[layer]) for layer in shape)

def decode_shape_code(shape_code):
    """
    用 ":" 分割字符串，得到每层的编码
    从str->dict
    """ 
    layers = shape_code.split(":")
    
    # 构造字典，每层用 layer1, layer2 等表示
    decoded_shape = {
        f"layer{i+1}": layers[i]
        for i in range(len(layers))
    }
    
    return decoded_shape

# 染色（将最上层的所有象限的颜色染成指定颜色）
def colorize(shape, new_color, out_dict = False):
    if type(shape) != dict:
        shape = decode_shape_code(shape)
    last_layer_key = list(shape.keys())[-1]  # 获取最后一层的键
    new_layer = ""
    for i in range(0, 8, 2):
        if shape[last_layer_key][i:i+2] != "--":
            # 仅改变颜色部分，保留形状类型
            new_layer += shape[last_layer_key][i] + new_color
        else:
            new_layer += "--"
    shape[last_layer_key] = new_layer
    if out_dict:
        return shape
    else:
        return make_shape_code(shape)



shape = {
    '第一层形状': 'Su--Ry--',
    '第二层形状': '--Cg--Su',
}

# print(make_shape_code(shape))

# cut_shape = cut(shape)

# print(make_shape_code(cut_shape))

# rotate_clockwise_shape = rotate_clockwise(shape)

# print(make_shape_code(rotate_clockwise_shape))

# rotate_counterclockwise_shape = rotate_counterclockwise(shape)

# print(make_shape_code(rotate_counterclockwise_shape))

# 示例调用

# lower_shape = {
#     '第一层形状': '--Sg----',
#     '第二层形状': '--RgRu--',
# }

# upper_shape = {
#     '第一层形状': 'Wr--Wu--',
#     '第二层形状': '----RyRy',
# }

# # lower_shape = {
# #     '第一层形状': 'SgSgSgSg',
# #     '第二层形状': '----RgRu',
# # }

# # upper_shape = {
# #     '第一层形状': 'WuWu----',
# #     '第二层形状': 'RyRyRyRy',
# #     # '第二层形状': 'Ry-------',
# # }



# stack_shape = stack(upper_shape, lower_shape)
# # stack_shape = stack(upper_shape, lower_shape)
# print(stack_shape)
# # print(make_shape_code(stack_shape))

# color1 = 'g'
# color2 = 'p'

# colorized_shape = colorize(stack_shape, mix_colors(color1, color2))
# print(colorized_shape)



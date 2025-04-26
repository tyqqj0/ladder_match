import json
from tqdm import tqdm
from conversation_generator import Client
from prompt.shapez_2d_prompt import promt_make_option, prompt_1_llm, prompt_2_llm, prompt_1_mllm, prompt_2_mllm
import concurrent.futures
import pandas as pd
from tabulate import tabulate
import os
import base64
import multiprocessing
import argparse
import random
import sys
import string
import logging
from datetime import datetime

def write_log(jsonl_path, log_to_be_write):
    with open(jsonl_path, 'a') as file:
        json.dump(log_to_be_write, file, ensure_ascii=False)
        file.write('\n')

def modify_last_line(jsonl_path, new_last_line):
    """
    修改 JSONL 文件的最后一行
    
    :param jsonl_path: JSONL 文件路径
    :param new_last_line: 用于替换最后一行的新内容（应该是一个字典对象）
    """
    # 读取文件并获取所有行
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # 确保文件不为空
    if lines:
        # 将最后一行替换为新的内容
        lines[-1] = json.dumps(new_last_line, ensure_ascii=False) + '\n'
    else:
        # 如果文件为空，直接写入新的内容
        lines.append(json.dumps(new_last_line, ensure_ascii=False) + '\n')

    # 写回修改后的内容到文件
    with open(jsonl_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

def categorize_data_by_steps(data):
    # 将数据按 steps_number 分类
    steps_data = {}
    for item in data:
        steps_number = item['steps_number']
        if steps_number not in steps_data:
            steps_data[steps_number] = []
        steps_data[steps_number].append(item)
    return steps_data

def filter_data_by_steps_number(steps_data, number, min_steps, max_steps, random_sample = True):
    # 按照 steps_number 筛选符合条件的数据
    filtered_data = []
    for steps_number in range(min_steps, max_steps + 1):
        if steps_number in steps_data:

            if len(steps_data[steps_number]) >= number:
                if random_sample:
                    filtered_data.extend(random.sample(steps_data[steps_number], number))
                else:
                    filtered_data.extend(steps_data[steps_number][:number])
            else:
                filtered_data.extend(steps_data[steps_number])
    return filtered_data

def filter_data_by_steps_min_max(data, number, min_steps, max_steps, random_sample = True):
    # 使用上述两个子函数来完成任务
    steps_data = categorize_data_by_steps(data)
    filtered_data = filter_data_by_steps_number(steps_data, number, min_steps, max_steps, random_sample)
    return filtered_data

def save_data(datas, output_path, transfer = False):
    # 保存数据到文件

    if transfer:
        data_list= []
        for _, step_datas in datas.items():
            for data in step_datas:
                data_list.append(data)
        with open(output_path, 'w') as file:
            json.dump(data_list, file, ensure_ascii=False, indent=4)

    else:   
        with open(output_path, 'w') as file:
            json.dump(datas, file, ensure_ascii=False, indent=4) 

def get_most_possible_tokens(top_tokens_logprobs):
    most_possible_tokens = max(top_tokens_logprobs, key=top_tokens_logprobs.get)
    most_possible_tokens_logprobs = top_tokens_logprobs[most_possible_tokens]
    return most_possible_tokens, most_possible_tokens_logprobs

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def load_data(file_path):
    # 打开 JSONL 文件或 JSON 文件
    if file_path.endswith('.jsonl'):
        with open(file_path, 'r', encoding='utf-8') as infile:
            data = [json.loads(line.strip()) for line in infile]
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
    return data
    
def generate_response_process(service_name, model_name, system_prompt, user_prompt, config_path, queue, max_waiting_time):
    client = Client(service_name, model_name, config_path)
    response = client.generate_response(system_prompt, user_prompt)
    queue.put(response)  # 将响应放入队列中，返回给主进程

def generate_response(service_name, model_name, system_prompt, user_prompt, config_path=None, max_waiting_time=60):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=generate_response_process, args=(service_name, model_name, system_prompt, user_prompt, config_path, queue, max_waiting_time))
    
    response = None
    process.start()
    
    try:
        # 等待 max_waiting_time 秒
        process.join(timeout=max_waiting_time)
        
        if process.is_alive():  # 如果进程还在运行，说明超时
            print("超时，强制终止进程")
            process.terminate()  # 强制终止进程
            process.join(timeout=1)  # 再等待1秒确保进程终止
            
            # 如果仍然存活，使用kill确保进程被终止
            if process.is_alive():
                import signal
                os.kill(process.pid, signal.SIGKILL)
                
            return None, None
        
        # 获取进程中的返回值
        if not queue.empty():
            response = queue.get(block=False)
    except Exception as e:
        print(f"生成响应时出错: {str(e)}")
        return None, None
    finally:
        # 确保进程已终止
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
        
        # 清空队列，防止内存泄漏
        while not queue.empty():
            try:
                queue.get(block=False)
            except:
                pass
    
    # 防止返回值异常
    if response is None:
        return None, None
    
    return response[0], response[1]

def to_ABCD(response, service_name = 'qianduoduo', model_name = 'gpt-4o-mini'):
    max_attempts = 3  # 设置最多尝试次数
    attempts = 0  # 初始化尝试计数器
    response_to_be_processed = response
    while response_to_be_processed not in ['A', 'B', 'C', 'D'] and attempts < max_attempts:
        ABCD_system_prompt, ABCD_user_prompt = promt_make_option(response)
        response_to_be_processed, _ = generate_response(service_name, model_name, ABCD_system_prompt, ABCD_user_prompt)
        attempts += 1  # 增加尝试次数
        
    if max_attempts == attempts:
        return response
    else:
        response = response_to_be_processed
        return response

def get_single_response(service_name, model_name, prompt_type, data, config_path = None, max_waiting_time = 500):
    steps = data['steps_number']
    target_action_list = data['target_action_list']
    target_action_str = ''
    for i in range(steps):
        target_action_str += f"the {i+1}th action is {target_action_list[i]}\n"   
    original_shape_coordinates = data['original_shape']
    shape_items = data['shape_items']
    target_shape_coordinates = data['target_shape']
    image_path = data['image_path']
    original_image_path = image_path['original']
    correct_option = data['ground_truth']
    correct_option_image_path = image_path[f'option_{correct_option}']

    if prompt_type == "1_llm":
        system_prompt, user_prompt = prompt_1_llm(original_shape_coordinates = original_shape_coordinates, steps = steps, target_action_str = target_action_str, shape_items = shape_items)
    elif prompt_type == "2_llm":
        system_prompt, user_prompt = prompt_2_llm(original_shape_coordinates = original_shape_coordinates, steps = steps, target_shape_coordinates = target_shape_coordinates, shape_items = shape_items)
    elif prompt_type == "1_mllm":
        system_prompt, user_prompt = prompt_1_mllm(image_path, steps = steps, target_action_str = target_action_str)
    elif prompt_type == "2_mllm":
        system_prompt, user_prompt = prompt_2_mllm(original_image_path, correct_option_image_path, steps = steps, shape_items = shape_items)
    else:
        raise ValueError(f"Invalid test type: {prompt_type}")

    response_clean = 'useless_response'
    max_attempts = 3  # 在外部定义
    attempts = 0  # 在外部初始化 attempts

    while response_clean not in ['A', 'B', 'C', 'D'] and attempts < max_attempts:
        response, _ = generate_response(service_name, model_name, system_prompt, user_prompt, config_path, max_waiting_time)

        if response is None:
            response = 'useless_response'
            response_clean = 'useless_response'
            break

        response_clean = to_ABCD(response)
        attempts += 1
        
        if attempts == max_attempts:
            response = 'useless_response'
            response_clean = 'useless_response'

    return response, response_clean

def parallel_get_response(datas, service_name, model_name, prompt_type, num_workers = 8, save_interval = 50, interval_save_path = None, config_path = None, max_waiting_time = 500):
    def execute_single_task(data, service_name, model_name, prompt_type, config_path, max_waiting_time):
        return get_single_response(service_name, model_name, prompt_type, data, config_path, max_waiting_time)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 提交每个数据项的处理任务
        future_to_data = {executor.submit(execute_single_task, data, service_name, model_name, prompt_type, config_path, max_waiting_time): data for data in datas if 'response' not in data or data['response'] == 'useless_response'}
        # 使用tqdm进行进度条显示
        processed_count = 0
        for future in tqdm(concurrent.futures.as_completed(future_to_data), total=len(future_to_data), desc="Processing data", unit="data"):
            data = future_to_data[future]
            try:
                response = future.result()  # 获取任务结果
                data['response'] = response[0]
                data['response_clean'] = response[1]
                processed_count += 1
                
                # 每处理50次保存一次结果
                if processed_count % save_interval == 0:
                    save_data(datas, interval_save_path)

            except Exception as exc:
                print(f"Error processing data: {data['id']} - {exc}")
        
        # 完成所有任务后保存最终数据
        save_data(datas, interval_save_path)

def filter_json(input_file_path, output_file_path):
    # 读取输入文件
    with open(input_file_path, 'r') as file:
        data = json.load(file)
    
    filtered_data = []
    for item in data:
        filtered_item = {key: item[key] for key in ['id','steps_number', 'ground_truth', 'problem_A', 'problem_B', 'problem_C', 'problem_D'] if key in item}
        filtered_data.append(filtered_item)

    # 将筛选后的数据写入输出文件
    with open(output_file_path, 'w') as file:
        json.dump(filtered_data, file, ensure_ascii=False, indent=4)

def calculate_accuracy(datas):
    correct_count = 0
    total_count = len(datas)

    for data in datas:
        if data['response_clean'] == data['ground_truth']:
            correct_count += 1

    return correct_count, total_count

def calculate_accuracy_by_steps(input_file_path, threshold = 4):

    # 读取JSON文件
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    # 将数据按steps_number分类
    steps_data = {}
    for item in data:
        steps_number = item['steps_number']
        if steps_number not in steps_data:
            steps_data[steps_number] = []
        steps_data[steps_number].append(item)
    
    # 准备一个列表来存储每个steps的正确率和数据量
    results = []
    # 遍历每个steps_number，计算正确率
    for steps_number in sorted(steps_data.keys()):
        step_items = steps_data[steps_number]
        accuracy, item_count, correct_id = calculate_accuracy(step_items, threshold)
        results.append({
            'steps_number': steps_number, 
            'data_count': item_count,  # 添加数据量字段
            'accuracy': accuracy,
        })
    
    # 将结果转换为DataFrame
    df = pd.DataFrame(results)
    
    # 设置表格显示的样式
    df['data_count'] = df['data_count'].apply(lambda x: f'{x:,}')  # 添加千位分隔符
    df['accuracy'] = df['accuracy'].apply(lambda x: f'{x:.2%}')  # 将accuracy格式化为百分比
        # 使用tabulate库生成带有线框的表格
    table = tabulate(df, headers='keys', tablefmt='grid', showindex=False)
    
    # 输出带有线框的表格
    print(table)

def single_layer_race(all_datas, rank, output_path, service_name, model_name, prompt_type, num_workers, save_interval, config_path, max_waiting_time, rank_failure_times):

    all_rank_datas = all_datas[rank]
    rank_datas = random.sample(all_rank_datas, 5)

    if os.path.exists(output_path):
        with open(output_path, 'r') as file:
            lines = file.readlines()  
    else:
        lines = []

    rank_interval_save_path = output_path.replace('.jsonl', f'_intermediate_{len(lines) - (0 if len(lines) == 0 or "rank_interval_save_path" in lines[-1] else 1)}.json')


    log_to_be_write = {'rank': rank, 'rank_failure_times': rank_failure_times}
    write_log(output_path, log_to_be_write)
    # log_to_be_write = {'rank': rank, 'rank_interval_save_path': rank_interval_save_path, 'rank_failure_times': rank_failure_times}

    if os.path.exists(rank_interval_save_path):
        rank_datas_to_be_processed = load_data(rank_interval_save_path)
    else:
        rank_datas_to_be_processed = []
        for rank_data in rank_datas:
            problems = rank_data['problems']
            problem_id, problem_data = random.choice(list(problems.items()))

            data_to_be_processed = {
                'id': rank_data['id'],
                'original_shape': rank_data['original_shape'],
                'steps_number': rank_data['steps_number'],
                'target_action_list': rank_data['target_action_list'],
                'target_shape': rank_data['target_shape'],
                'ground_truth': problem_data['ground_truth'],
                'shape_items': problem_data['shape_items'],
                'image_path': problem_data['image_path']
            }

            rank_datas_to_be_processed.append(data_to_be_processed)

    # Get the response from the service
    parallel_get_response(
        rank_datas_to_be_processed,
        service_name,
        model_name,
        prompt_type,
        num_workers,
        save_interval,
        rank_interval_save_path,
        config_path,
        max_waiting_time
    )

    # Load the processed data
    processed_rank_datas = load_data(rank_interval_save_path)

    correct_count = None
    total_count = None

    if any('response' not in data or data['response'] == 'useless_response' for data in processed_rank_datas):
        modify_last_line(output_path, {'rank': rank, 'rank_failure_times': rank_failure_times})
        print(f'游戏失败，请重新开始')
        # 不直接退出程序，而是抛出异常，让上层函数处理
        raise RuntimeError("模型响应无效，游戏失败")
    else:
        correct_count, total_count = calculate_accuracy(processed_rank_datas)

    if correct_count >= 3:
        state = 'success'
    else:
        state = 'failure'

    log_to_be_write = {'rank': rank, 'rank_interval_save_path': rank_interval_save_path, 'rank_failure_times': rank_failure_times, 'state': state}
    modify_last_line(output_path, log_to_be_write)    

    return state

def ladder_match(source_data_path, output_dir_path, service_name, model_name, prompt_type, num_workers = 8, save_interval = 50, config_path = None, min_steps = None, max_steps = None, each_number = 100, max_waiting_time = 500, threshold = 3, filter = True, initial_rank = 1, try_times = 1):
    output_path = os.path.join(output_dir_path, f'{model_name}_{prompt_type}_initial_rank_{initial_rank}_try_times_{try_times}.jsonl')
    # 定义结果文件路径
    result_file = os.path.join(output_dir_path, f'result_{model_name}_{prompt_type}_try{try_times}.json')
    
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    datas = load_data(source_data_path)
    if min_steps is not None and max_steps is not None:
        datas = filter_data_by_steps_min_max(datas, each_number, min_steps, max_steps, random_sample = True)

    all_datas = categorize_data_by_steps(datas)

    final_rank = -1  # 默认为-1，表示出错
    error_info = None

    if os.path.exists(output_path):
        with open(output_path, 'r') as file:
            lines = file.readlines()
            last_line = lines[-1]
            last_line = json.loads(last_line)
            rank = last_line['rank']
            rank_failure_times = last_line['rank_failure_times']
            if any(value >= 2 for value in rank_failure_times.values()):
                print(f'游戏已经结束了，最终结果为{rank - 1}')
                final_rank = rank - 1
                # 将结果写入JSON文件
                save_result_to_file(result_file, model_name, prompt_type, try_times, final_rank)
                return final_rank
            else:
                print(f'继续之前运行失败的游戏，当前挑战第{rank}级')
    else:
        rank_failure_times = {str(1): 0}
        rank = initial_rank

    if rank == 1:
        print(f'游戏开始，现在挑战第{rank}级')

    try:
        while rank <= max_steps:
            try:
                state = single_layer_race(all_datas, rank, output_path, service_name, model_name, prompt_type, num_workers, save_interval, config_path, max_waiting_time, rank_failure_times)
            except RuntimeError as e:
                # 捕获single_layer_race中抛出的异常
                logging.error(f"单层级游戏失败: {str(e)}")
                final_rank = -1
                error_info = str(e)
                break
                
            if state == 'success':
                print(f'第{rank}级成功，开始挑战第{rank + 1}级')
                rank = rank + 1
                rank_failure_times[str(rank)] = 0

            else:
                rank_failure_times[str(rank)] += 1

                if rank_failure_times[str(rank)] == 1:
                    if rank == 1:
                        print(f'第{rank_failure_times[str(rank)]}次挑战第{rank}级失败，重新开始挑战，来获得继续挑战的机会')
                    else:
                        print(f'第{rank_failure_times[str(rank)]}次挑战第{rank}级失败，重新完成第{rank - 1}级的挑战，来获得继续挑战的机会')
                else:
                    print(f'第{rank_failure_times[str(rank)]}次挑战第{rank}级失败，游戏结束，最终结果为{rank - 1}')
                    log_to_be_write = {'rank': rank, 'rank_failure_times': rank_failure_times, 'state': 'failure'}
                    modify_last_line(output_path, log_to_be_write)
                    final_rank = rank - 1
                    break
                
                state = single_layer_race(all_datas, max(rank-1, 1), output_path, service_name, model_name, prompt_type, num_workers, save_interval, config_path, max_waiting_time, rank_failure_times)

                if state == 'success':
                    print(f'第{rank - 1}级成功，重新获得资格，开始挑战第{rank}级')
                    continue
                else:
                    if rank == 1:
                        rank_failure_times[str(rank)] += 1
                        print(f'第1级重新挑战也失败，挑战结束，最终结果为0')
                        log_to_be_write = {'rank': rank, 'rank_failure_times': rank_failure_times, 'state': 'failure'}
                        modify_last_line(output_path, log_to_be_write)
                        final_rank = 0
                    else:
                        rank_failure_times[str(rank)] += 1
                        print(f'第{rank - 1}级也失败，挑战结束，最终结果为{rank - 1}')
                        log_to_be_write = {'rank': rank, 'rank_failure_times': rank_failure_times, 'state': 'failure'}
                        modify_last_line(output_path, log_to_be_write)
                        final_rank = rank - 1
                    break
            
        # 如果成功通过所有关卡
        if rank > max_steps:
            print(f'成功通过所有{max_steps}关卡！')
            final_rank = max_steps
            
    except Exception as e:
        print(f"测试过程出错: {str(e)}")
        # 发生错误时尝试保存当前状态
        try:
            log_to_be_write = {'rank': rank, 'rank_failure_times': rank_failure_times, 'state': 'error', 'error': str(e)}
            modify_last_line(output_path, log_to_be_write)
        except:
            pass
        final_rank = -1
        error_info = str(e)
    
    # 保存结果到JSON文件
    save_result_to_file(result_file, model_name, prompt_type, try_times, final_rank, error_info)
    
    return final_rank

def save_result_to_file(result_file, model_name, prompt_type, try_times, final_rank, error_info=None):
    """将结果保存到JSON文件中"""
    result_info = {
        "model_name": model_name,
        "prompt_type": prompt_type,
        "try_times": try_times,
        "final_rank": final_rank,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "status": "success" if final_rank >= 0 else "error",
        "error_info": error_info
    }
    
    with open(result_file, 'w') as f:
        json.dump(result_info, f, indent=2)
    
    print(f"结果已保存到: {result_file}")

def begin_test(args):
    if args.test_type == 'ladder_match':
        final_rank = ladder_match(
            args.source_data_path, 
            args.output_dir_path, 
            args.service_name, 
            args.model_name, 
            args.prompt_type, 
            num_workers = args.num_workers, 
            save_interval = args.save_interval, 
            config_path = args.config_path, 
            min_steps = args.min_steps, 
            max_steps = args.max_steps, 
            each_number = args.each_number, 
            max_waiting_time = args.max_waiting_time, 
            threshold = args.threshold, 
            initial_rank = args.initial_rank, 
            try_times = args.try_times
        )
        # 返回最终结果
        return final_rank

def parse_args():
    parser = argparse.ArgumentParser(description="命令行参数示例")

    parser.add_argument('--test_type', type=str, default='all_steps', required=True, help="测试类型")
    parser.add_argument('--source_data_path', type=str, default='dataset/source_data_min_1_max_20_each_100/source_data.jsonl', required=True, help="源数据路径")
    parser.add_argument('--output_dir_path', type=str, default='result/test', required=True, help="输出目录路径")
    parser.add_argument('--service_name', type=str, default='tju_api', required=True, help="服务名称")
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini', required=True, help="模型名称")
    parser.add_argument('--response_name', type=str, default='response_test_1_llm', required=True, help="响应名称")
    parser.add_argument('--prompt_type', type=str, default='1_llm', required=True, help="提示类型")
    parser.add_argument('--min_steps', type=int, default=1, help="最小步数")
    parser.add_argument('--max_steps', type=int, default=3, help="最大步数")
    parser.add_argument('--each_number', type=int, default=20, help="每次数量")
    parser.add_argument('--stop_if_accuracy_below_threshold', type=bool, default=True, help="如果准确率低于阈值，是否停止")
    parser.add_argument('--accuracy_threshold', type=float, default=0.33, help="准确率阈值")
    parser.add_argument('--max_waiting_time', type=int, default=500, help="最大等待时间")
    parser.add_argument('--num_workers', type=int, default=20, help="工作线程数")
    parser.add_argument('--save_interval', type=int, default=1, help="保存间隔")
    parser.add_argument('--config_path', type=str, default='original', help="配置文件路径")
    parser.add_argument('--threshold', type=int, default=4, help="阈值")
    parser.add_argument('--try_times', type=int, default=1, help="尝试次数")
    parser.add_argument('--initial_rank', type=int, default=1, help="初始等级")

    return parser.parse_args()

def main():
    args = parse_args()
    final_rank = begin_test(args)
    
    # 不再使用返回码携带层数信息
    # 始终返回0表示程序正常结束
    print(f"程序正常结束，结果已保存到结果文件中，最终层数: {final_rank}")
    sys.exit(0)

if __name__ == "__main__":
    main()


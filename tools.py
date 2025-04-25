import json

def modify_image_path(jsonl_file_path, old_content, new_content):
    modified_data = []
    
    # 打开jsonl文件
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        # 逐行读取
        for line in file:
            # 加载每一行的数据为字典
            data = json.loads(line.strip())
            new_problems = {}
            problems = data['problems']
            for problem_id, problem in problems.items():
                new_image_path = {}
                image_path = problem['image_path']
                for option_id, option_image_path in image_path.items():
                    new_image_path[option_id] = option_image_path.replace(old_content, new_content)
                problem['image_path'] = new_image_path
                new_problems[problem_id] = problem
            data['problems'] = new_problems

            modified_data.append(data)
    
    with open(jsonl_file_path, 'w', encoding='utf-8') as outfile:
        for entry in modified_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"文件 '{jsonl_file_path}' 已被成功覆盖！")


jsonl_file_path = 'source_dataset/shapez_source_data_min_1_max_100_each_100/source_data.jsonl'  # 替换为你的jsonl文件路径
old_content = 'dataset/new_source_data_min_1_max_100_each_100/source_data_images/'  # 你希望替换掉的内容
new_content = 'source_dataset/shapez_source_data_min_1_max_100_each_100/source_data_images/'  # 新内容
modify_image_path(jsonl_file_path, old_content, new_content)

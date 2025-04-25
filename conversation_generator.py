from openai import OpenAI
import time
from openai import RateLimitError
import yaml 
import numpy as np
  
# 加载 YAML 配置文件
def load_config():
    # 加载 api_key.yaml
    with open('config/api_key.yaml', 'r') as file:
        api_config = yaml.safe_load(file)
    
    # 加载 openai_params.yaml
    with open('config/config.yaml', 'r') as file:
        openai_params = yaml.safe_load(file)
    
    # 合并两个配置
    api_config['api_params'] = openai_params.get('api_params')
    
    return api_config

class Client:
    def __init__(self, service_name, model_name, config_path = None):
        config = load_config()['api_config'].get(service_name)
        if not config:
            raise ValueError(f"Service {service_name} not found in config.")
        
        self.api_key = config['api_key']
        self.base_url = config['base_url']
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        self.model_name = model_name
        if not config_path:
            config_path = 'original'
            self.api_config = load_config()['api_params'][config_path]  # 修改这里加载另一个YAML文件
        else:
            self.api_config = load_config()['api_params'][config_path] 

    def chat_with_backoff(self, **kwargs):
        """
        调用OpenAI API，带有指数退避机制。

        :param kwargs: 调用API的参数。
        :return: API的响应。
        """
        retries = 5  # 最大重试次数
        backoff_time = 1  # 初始等待时间（秒）

        for attempt in range(retries):
            try:
                # 尝试调用API
                response = self.client.chat.completions.create(**kwargs)
                return response
            except RateLimitError as e:
                if attempt < retries - 1:
                    # print(f"Rate limit exceeded, retrying in {backoff_time} seconds...")
                    time.sleep(backoff_time)
                    backoff_time *= 2  # 指数退避，等待时间加倍
                else:
                    # print("Maximum retry attempts reached. Rate limit not cleared.")
                    raise e
            except Exception as e:
                print(f"An error occurred: {e}")
                raise e

    def generate_response(self, system_prompt, user_prompt):
        prompt = self.prompt_generate(system_prompt, user_prompt)
        model  = self.model_name
        # 从配置文件中获取参数
        while True:
            # response = self.chat_with_backoff(
            #     model=model,
            #     messages=prompt,
            #     temperature=self.api_config.get("temperature", 0.95),
            #     max_tokens=self.api_config.get("max_tokens", 2048),
            #     top_p=self.api_config.get("top_p", 0.95),
            #     n=self.api_config.get("n", 1),
            #     frequency_penalty=self.api_config.get("frequency_penalty", 0),
            #     presence_penalty=self.api_config.get("presence_penalty", 0)
            # )

            allowed_params = [
                "temperature",
                "max_tokens",
                "top_p",
                "n",
                "frequency_penalty",
                "presence_penalty",
                "logprobs",
                "top_logprobs"
            ]

            # 从 api_config 中挑出实际设置了的参数
            chat_kwargs = {
                "model": model,
                "messages": prompt,
                
            }
            for key in allowed_params:
                if key in self.api_config:
                    chat_kwargs[key] = self.api_config[key]

            if 'top_logprobs' in self.api_config:

            # 调用时只展开 chat_kwargs
                response = self.chat_with_backoff(**chat_kwargs)
                if response and response.choices:
                    reply = response.choices[0].message.content
                    if response.choices[0].logprobs is not None:
                        top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
                        top_tokens_logprobs = {}
                        for logprob in top_logprobs:
                            top_tokens_logprobs[logprob.token] = np.round(np.exp(logprob.logprob)*100,2)         
                    else:
                        top_tokens_logprobs = {'useless_response': 100}
                    return reply, top_tokens_logprobs
                else:
                    continue

            else:
                response = self.chat_with_backoff(**chat_kwargs)
                if response and response.choices:
                    reply = response.choices[0].message.content
                    return reply, {}
                else:
                    continue

    @staticmethod
    def prompt_generate(system_prompt, user_prompt):
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return prompt

def create_generator(service_name, model_name):
    return Client(service_name, model_name)


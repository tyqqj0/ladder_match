#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型配置文件
定义了批量测试中使用的所有模型和它们支持的提示类型
"""

# 模型配置，键是模型名称，值是该模型支持的提示类型列表
MODEL_CONFIG = {
    # OpenAI 模型
    'gpt-4o-mini': ['1_llm', '2_llm', '1_mllm', '2_mllm'],  # 支持所有提示类型
    'gpt-4o': ['1_llm', '2_llm', '1_mllm', '2_mllm'],       # 支持所有提示类型
    'gpt-4.1': ['1_llm', '2_llm', '1_mllm', '2_mllm'],      # 支持所有提示类型
    'gpt-4.1-mini': ['1_llm', '2_llm', '1_mllm', '2_mllm'], # 支持所有提示类型
    'gpt-4.1-nano': ['1_llm', '2_llm', '1_mllm', '2_mllm'], # 支持所有提示类型
    
    # OpenAI 其他模型
    'o3-mini': ['1_llm', '2_llm', '1_mllm', '2_mllm'],      # 支持所有提示类型
    'o4-mini': ['1_llm', '2_llm', '1_mllm', '2_mllm'],      # 支持所有提示类型
    
    # Anthropic Claude 模型
    'claude-3-7-sonnet-20250219': ['1_llm', '2_llm', '1_mllm', '2_mllm'],      # 支持所有提示类型
    'claude-3-7-sonnet-latest-thinking': ['1_llm', '2_llm', '1_mllm', '2_mllm'],  # 支持所有提示类型
    
    # Google Gemini 模型
    'gemini-2.0-flash': ['1_llm', '2_llm', '1_mllm', '2_mllm'],               # 支持所有提示类型
    'gemini-2.0-flash-thinking-exp': ['1_llm', '2_llm', '1_mllm', '2_mllm'],  # 支持所有提示类型
    'gemini-2.5-pro-exp-03-25': ['1_llm', '2_llm', '1_mllm', '2_mllm'],       # 支持所有提示类型
    
    # Grok 模型
    'grok-3': ['1_llm', '2_llm', '1_mllm', '2_mllm'],      # 支持所有提示类型
    'grok-2': ['1_llm', '2_llm', '1_mllm', '2_mllm'],      # 支持所有提示类型
    'grok-3-reasoner': ['1_llm', '2_llm', '1_mllm', '2_mllm'],  # 支持所有提示类型
}

# 尝试次数范围
TRY_TIMES_RANGE = range(1, 4)  # 1, 2, 3

# 默认的最大重试次数
DEFAULT_MAX_RETRIES = 10

# 默认的并行工作线程数
DEFAULT_MAX_WORKERS = 40 
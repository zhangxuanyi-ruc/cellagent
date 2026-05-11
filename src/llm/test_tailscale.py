import socket
import httpx
from openai import OpenAI
import json

HOST = 'localagent.tail053d0c.ts.net'
TAILSCALE_IP = '100.105.94.49'

orig_getaddrinfo = socket.getaddrinfo

def patched_getaddrinfo(host, port, *args, **kwargs):
    if host == HOST:
        return orig_getaddrinfo(TAILSCALE_IP, port, *args, **kwargs)
    return orig_getaddrinfo(host, port, *args, **kwargs)

socket.getaddrinfo = patched_getaddrinfo

client = OpenAI(
    base_url=f'https://{HOST}/v1',
    api_key='unused',
    http_client=httpx.Client(trust_env=False, timeout=120),
)

print("=== 测试1: 获取可用模型 ===")
try:
    models = client.models.list()
    print("可用模型:")
    for m in models:
        print(f"  - {m.id}")
    print()
except Exception as e:
    print(f"获取模型列表失败: {e}")
    exit(1)

model_name = models.data[0].id if models.data else "default"

print("=== 测试2: 基础对话能力 ===")
try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个有用的AI助手。请直接回答问题，不要展示思考过程。"},
            {"role": "user", "content": "你好，请简单介绍一下你自己。"}
        ],
        temperature=0.7,
        max_tokens=200,
        extra_body={
            "enable_thinking": False
        }
    )
    
    print(f"模型: {response.model}")
    print(f"回答: {response.choices[0].message.content}")
    print(f"使用token数: {response.usage.total_tokens if response.usage else 'N/A'}")
    print()
except Exception as e:
    print(f"对话测试失败: {e}")
    print("尝试不使用extra_body参数...")
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个有用的AI助手。请直接回答问题，不要展示思考过程。"},
            {"role": "user", "content": "你好，请简单介绍一下你自己。"}
        ],
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"模型: {response.model}")
    print(f"回答: {response.choices[0].message.content}")
    print(f"使用token数: {response.usage.total_tokens if response.usage else 'N/A'}")
    print()

print("=== 测试3: 多轮对话能力 ===")
try:
    messages = [
        {"role": "system", "content": "你是一个专业的编程助手。请直接给出答案，不要展示思考过程。"},
        {"role": "user", "content": "Python中如何定义一个函数？"},
    ]
    
    response1 = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=150,
        extra_body={
            "enable_thinking": False
        }
    )
    
    assistant_reply = response1.choices[0].message.content
    print(f"第一轮回答:\n{assistant_reply}\n")
    
    messages.append({"role": "assistant", "content": assistant_reply})
    messages.append({"role": "user", "content": "能给我一个具体的例子吗？"})
    
    response2 = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=200,
        extra_body={
            "enable_thinking": False
        }
    )
    
    print(f"第二轮回答:\n{response2.choices[0].message.content}")
    print()
except Exception as e:
    print(f"多轮对话测试失败: {e}")
    print("尝试不使用extra_body参数...")
    
    messages = [
        {"role": "system", "content": "你是一个专业的编程助手。请直接给出答案，不要展示思考过程。"},
        {"role": "user", "content": "Python中如何定义一个函数？"},
    ]
    
    response1 = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )
    
    assistant_reply = response1.choices[0].message.content
    print(f"第一轮回答:\n{assistant_reply}\n")
    
    messages.append({"role": "assistant", "content": assistant_reply})
    messages.append({"role": "user", "content": "能给我一个具体的例子吗？"})
    
    response2 = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=200
    )
    
    print(f"第二轮回答:\n{response2.choices[0].message.content}")
    print()

print("=== 测试4: 代码生成能力 ===")
try:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "你是一个专业的程序员。请直接给出代码，不要展示思考过程。"},
            {"role": "user", "content": "用Python写一个简单的快速排序算法"}
        ],
        temperature=0.3,
        max_tokens=300,
        extra_body={
            "enable_thinking": False
        }
    )
    
    print(f"代码生成结果:\n{response.choices[0].message.content}")
    print()
except Exception as e:
    print(f"代码生成测试失败: {e}")

print("=== 所有测试完成 ===")
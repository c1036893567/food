from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import uvicorn
import json
import re
from typing import Optional

app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应更严格
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key="sk-5fa4d9fc58424f07952bf193595ff2fb",  # 替换为你的API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class FoodRequest(BaseModel):
    budget: int
    mood: str
    weather: str
    lastClass: str
    nextClass: str
    hunger: int
    energy: int
    vegetarian: bool
    spicy: bool
    quick: bool


def translate_to_chinese(data: dict) -> dict:
    """将英文参数转换为中文描述"""
    mood_map = {
        "sad": "难过",
        "depressed": "抑郁",
        "normal": "一般",
        "happy": "开心",
        "excited": "兴奋"
    }

    weather_map = {
        "sunny": "阳光明媚",
        "cloudy": "多云转晴",
        "rainy": "阴雨绵绵",
        "hot": "热到融化",
        "cold": "冷到发抖",
        "windy": "狂风大作"
    }

    class_map = {
        "none": "没有课",
        "math": "高数/线代",
        "physics": "大学物理",
        "programming": "编程课",
        "english": "英语课",
        "philosophy": "马原/毛概",
        "experiment": "实验课",
        "pe": "体育课"
    }

    return {
        "budget": f"{data['budget']}元",
        "mood": mood_map.get(data['mood'], data['mood']),
        "weather": weather_map.get(data['weather'], data['weather']),
        "lastClass": class_map.get(data['lastClass'], data['lastClass']),
        "nextClass": class_map.get(data['nextClass'], data['nextClass']),
        "hunger": f"{data['hunger']}/10",
        "energy": f"{data['energy']}/10",
        "vegetarian": "是" if data['vegetarian'] else "否",
        "spicy": "是" if data['spicy'] else "否",
        "quick": "是" if data['quick'] else "否"
    }


def extract_json_from_response(text: str) -> Optional[dict]:
    """从AI响应中提取JSON内容"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                return None
    return None


@app.post("/api/recommend-food")
async def recommend_food(request: FoodRequest):
    try:
        # 使用model_dump()替代已弃用的dict()
        chinese_data = translate_to_chinese(request.model_dump())

        # 构建系统提示词
        system_prompt = """你是一个专业的大学生饮食推荐助手，能够根据各种条件推荐最适合的食物。
        请始终返回有效的JSON格式数据，包含以下字段：
        - food: 推荐的食物名称
        - reason: 详细的推荐理由
        - tip: 实用的小贴士"""

        # 构建用户提示词
        user_prompt = f"""请根据以下条件推荐最适合的食物：

        预算: {chinese_data['budget']}
        心情: {chinese_data['mood']}
        天气: {chinese_data['weather']}
        上一节课: {chinese_data['lastClass']}
        下一节课: {chinese_data['nextClass']}
        饥饿程度: {chinese_data['hunger']}
        能量需求: {chinese_data['energy']}
        素食要求: {chinese_data['vegetarian']}
        想吃辣的: {chinese_data['spicy']}
        需要快速解决: {chinese_data['quick']}"""

        # 调用AI模型
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            response_format={"type": "json_object"}  # 要求返回JSON格式
        )

        # 处理AI响应
        ai_response = response.choices[0].message.content
        result = extract_json_from_response(ai_response)

        if not result:
            result = {
                "food": "黄焖鸡米饭",
                "reason": "AI响应解析失败，提供默认推荐",
                "tip": "建议检查AI返回的数据格式"
            }

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"推荐服务暂时不可用: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # "文件名:FastAPI应用变量名"
        host="0.0.0.0",
        port=8000,
        reload=False  # 开发时启用热重载
    )

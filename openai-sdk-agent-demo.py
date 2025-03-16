# # 安装配置
# python -m venv env
# source env/bin/activate

# pip install openai-agents

# # Mac/Linux设置api key
# export OPENAI_API_KEY=sk-...

# # Windows设置api key
# set OPENAI_API_KEY=sk-...

# ------------------------------代码------------------------------
# 示例
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)

# ------------------------------代码------------------------------
# 旅游智能体
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, handoff
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
import asyncio
import json
from pydantic import BaseModel, ValidationError

# 设置OpenAI客户端
openai_client = AsyncOpenAI()

# 最终旅行计划的输出类型
class TravelPlan(BaseModel):
    destination: str  # 目的地
    duration: str  # 持续时间
    itinerary: str  # 行程
    local_recommendations: str  # 当地推荐
    language_tips: str  # 语言提示
    summary: str  # 摘要

# 定义具有特定角色的旅行规划代理
planner_agent = Agent(
    name="旅行规划师",
    handoff_description="创建初始旅行计划和行程的主要代理",
    instructions=prompt_with_handoff_instructions("""
    你是一个专业的旅行规划师，能够根据用户请求提供旅行计划。
    
    你的职责是：
    1. 理解用户的旅行需求和目的地
    2. 创建一个初步的有日常活动的结构化行程
    3. 考虑目的地的主要景点、交通和适合的活动安排
    
    重要：提供初步行程后，你必须交接给"当地专家"获取更地道的体验建议。不要等待用户回应，直接使用consult_local_expert工具交接。
    """),
    model=OpenAIChatCompletionsModel(
        model="gpt-4o-mini",
        openai_client=openai_client
    ),
    model_settings=ModelSettings(temperature=0.7)
)

local_agent = Agent(
    name="当地专家",
    handoff_description="专门推荐地道当地活动和旅游景点的专家",
    instructions=prompt_with_handoff_instructions("""
    你是目的地的当地专家，能够推荐地道有趣的当地活动或旅游景点。

    请专注于：
    1. 推荐鲜为人知的景点，避开纯粹的旅游陷阱
    2. 提供当地美食和餐厅建议
    3. 分享特色文化体验和活动
    4. 提供只有当地人才知道的实用提示

    如果用户可能需要目的地的语言帮助，可以交接给"语言指南"。
    当你完成当地推荐后，请交接给"旅行计划编译器"以整合所有建议。
    """),
    model=OpenAIChatCompletionsModel(
        model="gpt-4o-mini",
        openai_client=openai_client
    )
)

language_agent = Agent(
    name="语言指南",
    handoff_description="目的地语言和沟通技巧的专家",
    instructions=prompt_with_handoff_instructions("""
    你是语言和文化沟通专家，提供针对旅行目的地的语言支持。

    请专注于：
    1. 提供目的地常用语言的关键短语和表达
    2. 解释可能的沟通挑战和解决方案
    3. 介绍与语言相关的当地文化礼仪
    4. 提供实用的语言学习资源或建议

    完成语言指南后，请交接给"旅行计划编译器"以整合所有建议成为完整的旅行计划。
    """),
    model=OpenAIChatCompletionsModel(
        model="gpt-4o-mini",
        openai_client=openai_client
    )
)

# 定义汇总所有建议的总结代理
summary_agent = Agent(
    name="旅行计划编译器",
    handoff_description="将所有建议汇编成完整旅行计划的最终代理",
    instructions="""
    你是一个专业的旅行计划编译器，负责整合各专家的建议成为完整的旅行计划。

    你的任务是：
    1. 审查旅行规划师、当地专家和语言指南提供的所有信息
    2. 创建一个整合所有观点的详细旅行计划
    3. 确保计划完整、实用且结构良好
    4. 包括每天的具体细节、当地推荐和语言提示

    请以以下JSON格式提供您的回复：
    {
        "destination": "目的地名称",
        "duration": "旅行时长",
        "itinerary": "详细的日程安排",
        "local_recommendations": "当地专家推荐",
        "language_tips": "目的地语言提示",
        "summary": "整体计划简要总结"
    }

    确保所有字段都有完整的内容，输出格式必须是有效的JSON。
    """,
    output_type=TravelPlan,
    model=OpenAIChatCompletionsModel(
        model="gpt-4o-mini",
        openai_client=openai_client
    )
)

# 设置代理网络和适当的交接 - 使用handoff函数创建更明确的交接
# 规划师的交接选项
to_local_expert = handoff(
    agent=local_agent,
    tool_name_override="consult_local_expert",
    tool_description_override="当需要地道的当地体验、隐藏景点或特色美食建议时使用此工具。"
)

to_language_guide = handoff(
    agent=language_agent,
    tool_name_override="consult_language_guide",
    tool_description_override="当需要目的地语言指南、关键短语或文化沟通建议时使用此工具。"
)

to_travel_compiler = handoff(
    agent=summary_agent,
    tool_name_override="compile_travel_plan",
    tool_description_override="当所有必要信息都已收集完毕，需要编译最终旅行计划时使用此工具。"
)

# 设置交接关系
planner_agent.handoffs = [to_local_expert, to_language_guide]
local_agent.handoffs = [to_language_guide, to_travel_compiler]
language_agent.handoffs = [to_travel_compiler]

# 入口点函数
async def plan_trip(destination_prompt):
    # 添加错误处理
    try:
        # 从规划师代理开始
        result = await Runner.run(planner_agent, destination_prompt)

        # 打印输出
        print("\n=== 旅行计划 ===\n")

        # 尝试解析输出为TravelPlan对象
        try:
            if isinstance(result.final_output, TravelPlan):
                # 如果直接得到TravelPlan对象
                travel_plan = result.final_output
                print(f"目的地: {travel_plan.destination}")
                print(f"时长: {travel_plan.duration}")
                print("\n行程安排:")
                print(travel_plan.itinerary)
                print("\n当地推荐:")
                print(travel_plan.local_recommendations)
                print("\n语言提示:")
                print(travel_plan.language_tips)
                print("\n摘要:")
                print(travel_plan.summary)
            elif isinstance(result.final_output, str):
                # 在输出中查找JSON
                try:
                    # 尝试提取嵌入在文本中的JSON
                    start_idx = result.final_output.find('{')
                    end_idx = result.final_output.rfind('}') + 1

                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = result.final_output[start_idx:end_idx]
                        travel_data = json.loads(json_str)
                        travel_plan = TravelPlan(**travel_data)

                        print(f"目的地: {travel_plan.destination}")
                        print(f"时长: {travel_plan.duration}")
                        print("\n行程安排:")
                        print(travel_plan.itinerary)
                        print("\n当地推荐:")
                        print(travel_plan.local_recommendations)
                        print("\n语言提示:")
                        print(travel_plan.language_tips)
                        print("\n摘要:")
                        print(travel_plan.summary)
                    else:
                        # 如果没有找到JSON，直接打印输出
                        print("未找到结构化旅行计划，原始输出:")
                        print(result.final_output)
                except (json.JSONDecodeError, ValidationError) as e:
                    print(f"解析JSON失败: {e}")
                    print("原始输出:")
                    print(result.final_output)
            else:
                # 未知输出类型
                print(f"未知输出类型: {type(result.final_output)}")
                print(result.final_output)
        except Exception as e:
            print(f"处理结果时出错: {e}")
            print("原始输出:")
            print(result.final_output)

        # 打印交接路径信息，便于调试
        if hasattr(result, 'new_items') and result.new_items:
            handoffs_occurred = [item for item in result.new_items if item.type == "handoff_output_item"]
            if handoffs_occurred:
                print("\n===== 交接路径 =====")
                for idx, handoff_item in enumerate(handoffs_occurred):
                    print(f"{idx + 1}. {handoff_item.source_agent.name} → {handoff_item.target_agent.name}")

        return result
    except Exception as e:
        print(f"执行过程中出错: {e}")
        return None

# 运行旅行规划器
if __name__ == "__main__":
    try:
        asyncio.run(plan_trip("规划一个3天的尼泊尔旅行。"))
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")

# ------------------------------代码------------------------------
# 模拟电商系统
from agents import Agent, Runner, handoff, AsyncOpenAI, function_tool
from agents.extensions.handoff_prompt import prompt_with_handoff_instructions
import asyncio

# 模拟数据库
order_database = {
    "ORD12345": {
        "status": "已发货",
        "date": "2025-03-05",
        "items": ["手机壳", "耳机"],
        "total": 299.99,
        "tracking": "SF1234567890",
        "customer_email": "customer@example.com"
    },
    "ORD67890": {
        "status": "待付款",
        "date": "2025-03-10",
        "items": ["平板电脑", "保护膜"],
        "total": 2499.99,
        "customer_email": "another@example.com"
    }
}

# 定义工具函数
@function_tool
def check_order_status(order_id: str) -> str:
    """查询订单状态"""
    if order_id in order_database:
        order = order_database[order_id]
        return f"订单 {order_id} 当前状态: {order['status']}，下单日期: {order['date']}，金额: ¥{order['total']}"
    return f"未找到订单 {order_id}"

@function_tool
def get_tracking_info(order_id: str) -> str:
    """获取物流信息"""
    if order_id in order_database and order_database[order_id].get("tracking"):
        return f"订单 {order_id} 的物流单号是: {order_database[order_id]['tracking']}"
    return f"订单 {order_id} 暂无物流信息或订单不存在"

# 定义专业代理

# 订单查询代理
order_agent = Agent(
    name="订单查询专员",
    instructions="""
    你是电子商务平台的订单查询专员。你可以帮助客户查询订单状态和物流信息。

    你需要获取订单号才能提供帮助。如果客户没有提供订单号，请礼貌地询问。

    请记住，你的职责只是查询和提供订单信息。如果客户提出其他需求（如退款或投诉），请向客户说明你只负责订单查询，并建议他们联系相关部门。
    """,
    tools=[check_order_status, get_tracking_info]
)

# 退款处理代理
refund_agent = Agent(
    name="退款处理专员",
    instructions="""
    你是电子商务平台的退款处理专员。

    处理退款请求时，请遵循以下步骤：
    1. 确认订单信息和退款原因
    2. 检查退款资格（例如退货时间、产品状态等）
    3. 解释退款流程和预计到账时间

    对于不符合退款条件的情况，请清楚解释原因并提供替代解决方案。
    """,
    tools=[check_order_status]
)

# 投诉处理代理
complaint_agent = Agent(
    name="客户投诉专员",
    instructions="""
    你是电子商务平台的客户投诉专员。你的目标是理解客户的不满并寻找解决方案。

    请记住：
    1. 首先表示同理心和理解
    2. 获取所有相关信息
    3. 提供明确的解决方案或后续步骤
    4. 在适当的情况下提供补偿（如优惠券、积分等）

    对于特别严重或复杂的投诉，可以承诺由主管跟进处理。
    """,
    tools=[check_order_status]
)

# 设置交接 - 修改名称使其更具描述性和区分度
transfer_to_order_specialist = handoff(
    agent=order_agent,
    tool_name_override="transfer_to_order_specialist",
    tool_description_override="当客户需要查询订单状态或物流信息时使用此工具。例如：'我想查询订单状态'、'我的包裹到哪了'等情况。"
)

transfer_to_refund_specialist = handoff(
    agent=refund_agent,
    tool_name_override="transfer_to_refund_specialist",
    tool_description_override="当客户明确要求退款或退货时使用此工具。例如：'我想申请退款'、'如何退货'等情况。"
)

transfer_to_complaint_specialist = handoff(
    agent=complaint_agent,
    tool_name_override="transfer_to_complaint_specialist",
    tool_description_override="仅当客户明确表示不满、抱怨或投诉时使用此工具。例如：'我对服务很不满'、'我要投诉'等情况。"
)

# 前台接待代理 - 使用更明确的指令和例子
main_agent = Agent(
    name="客服前台",
    instructions=prompt_with_handoff_instructions("""
    你是电子商务平台的客服前台。你的工作是了解客户需求并将他们引导至合适的专业客服。请根据以下明确的指引决定如何处理客户查询：

    1. 订单查询类问题：
       - 示例："我想查询订单状态"、"我的包裹什么时候到"、"能告诉我订单号XXX的情况吗"
       - 操作：使用transfer_to_order_specialist工具

    2. 退款类问题：
       - 示例："我想申请退款"、"这个产品有问题，我要退货"、"如何办理退款"
       - 操作：使用transfer_to_refund_specialist工具

    3. 投诉类问题：
       - 示例："我对你们的服务很不满"、"我要投诉"、"这个体验太糟糕了"
       - 操作：使用transfer_to_complaint_specialist工具

    4. 一般问题：
       - 示例："你们的营业时间是什么时候"、"如何修改收货地址"等
       - 操作：直接回答客户

    重要规则：
    - 请严格按照上述分类选择适当的交接工具
    - 不要过度解读客户意图，根据客户明确表达的需求选择工具
    - 如果不确定，先询问更多信息，而不是急于交接
    - 首次交流时，除非客户明确表达了投诉或退款需求，否则应该先用order_specialist处理
    """)
)

# 设置主代理的交接，按常见度排序
main_agent.handoffs = [
    transfer_to_order_specialist,  # 最常见的请求类型，放在最前面
    transfer_to_refund_specialist,  # 第二常见
    transfer_to_complaint_specialist  # 最不常见
]

# 主函数
async def handle_customer_query(query):
    print(f"\n===== 新的客户查询 =====")
    print(f"客户: {query}")

    try:
        result = await Runner.run(main_agent, query)
        print(f"\n客服回复: {result.final_output}")

        # 打印交接路径信息
        if hasattr(result, 'new_items') and result.new_items:
            handoffs_occurred = [item for item in result.new_items if item.type == "handoff_output_item"]
            if handoffs_occurred:
                print("\n===== 交接路径 =====")
                for idx, handoff_item in enumerate(handoffs_occurred):
                    print(f"{idx + 1}. {handoff_item.source_agent.name} → {handoff_item.target_agent.name}")
                    # 打印使用的工具名称，帮助调试
                    if hasattr(handoff_item, 'tool_name') and handoff_item.tool_name:
                        print(f"   使用工具: {handoff_item.tool_name}")
            else:
                # 如果没有交接发生，也打印出来便于调试
                print("\n没有交接发生，主代理直接处理了请求")

        return result
    except Exception as e:
        print(f"处理查询时出错: {e}")
        return None

# 示例查询
async def run_demo():
    queries = [
        "你好，我想查询一下我的订单状态",
        "我的订单号是ORD12345",
        "我想申请退款，订单ORD12345中的耳机质量有问题",
        "我对你们的配送速度非常不满，已经等了一周还没收到货！",
    ]

    for query in queries:
        try:
            await handle_customer_query(query)
        except Exception as e:
            print(f"处理查询'{query}'时出错: {e}")
        print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    asyncio.run(run_demo())


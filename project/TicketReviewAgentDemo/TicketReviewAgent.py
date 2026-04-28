# ================== 导入核心依赖 ==================
import argparse
import json
import os
import random
import re
import textwrap
from typing import Dict, List, NotRequired, TypedDict

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# ================== 初始化大模型 ==================
load_dotenv()


def create_llm():
    api_key = os.getenv("API_KEY")
    if not api_key:
        return None

    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("BASE_URL") or "https://api.deepseek.com",
        model=os.getenv("MODEL") or "deepseek-chat",
        temperature=0.3,
        max_tokens=800,
    )


llm = create_llm()
parser = StrOutputParser()


# ================== 1. 定义工单状态 ==================
class TicketState(TypedDict):
    """
    工单状态字典，记录从接收工单到人工审核、发送回复的全流程数据。
    MemorySaver 会按 thread_id 保存这些状态，后续可从中断点继续执行。
    """

    ticket_id: str
    customer_name: str
    issue: str
    priority: str
    category: NotRequired[str]
    risk_level: NotRequired[str]
    summary: NotRequired[str]
    draft_reply: NotRequired[str]
    internal_note: NotRequired[str]
    human_review: NotRequired[str]
    final_reply: NotRequired[str]
    risk_note: NotRequired[str]
    send_status: NotRequired[str]
    audit_log: List[str]


def init_ticket_state() -> TicketState:
    return {
        "ticket_id": "TK-20260428-001",
        "customer_name": "林同学",
        "issue": (
            "我购买的 LangGraph 训练营已经付款成功，但课程后台仍显示未开通。"
            "我明天就要跟着直播学习，请尽快帮我处理。"
        ),
        "priority": "high",
        "audit_log": [],
    }


# ================== 2. 工具函数 ==================
def append_log(state: TicketState, message: str) -> List[str]:
    logs = list(state.get("audit_log", []))
    logs.append(message)
    return logs


def extract_json(text: str) -> Dict[str, str]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("未找到 JSON 内容")
    return json.loads(match.group(0))


def run_chain(prompt: ChatPromptTemplate, inputs: Dict[str, str], fallback: Dict[str, str]) -> Dict[str, str]:
    if llm is None:
        return fallback

    chain = prompt | llm | parser
    try:
        result = chain.invoke(inputs)
        return extract_json(result.strip())
    except Exception as exc:
        print(f"\n⚠️ 模型输出解析失败，使用兜底结果：{exc}")
        return fallback


# ================== 3. 节点函数 ==================
def classify_ticket(state: TicketState) -> Dict[str, object]:
    """
    节点1：工单分类 Agent。
    根据用户问题判断工单类型、风险等级，并生成一句话摘要。
    """
    fallback = {
        "category": "课程开通",
        "risk_level": "高",
        "summary": "用户已付款但课程未开通，要求尽快处理以便参加直播学习。",
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是客服工单分类专家，请根据用户问题输出 JSON。
要求：
1. category：从 课程开通、退款咨询、技术故障、账号问题、其他 中选择
2. risk_level：从 低、中、高 中选择
3. summary：用一句话概括用户诉求
只返回 JSON，格式示例：
{{"category": "课程开通", "risk_level": "高", "summary": "用户付款后课程未开通，需要尽快处理"}}""",
            ),
            ("user", "客户：{customer_name}\n优先级：{priority}\n问题：{issue}"),
        ]
    )

    data = run_chain(
        prompt,
        {
            "customer_name": state["customer_name"],
            "priority": state["priority"],
            "issue": state["issue"],
        },
        fallback,
    )

    print("\n📥 工单分类完成")
    print(f"  工单号：{state['ticket_id']}")
    print(f"  分类：{data['category']}")
    print(f"  风险：{data['risk_level']}")
    print(f"  摘要：{data['summary']}")

    return {
        "category": data["category"],
        "risk_level": data["risk_level"],
        "summary": data["summary"],
        "audit_log": append_log(state, "工单已完成分类"),
    }


def draft_solution(state: TicketState) -> Dict[str, object]:
    """
    节点2：回复生成 Agent。
    该节点执行后会触发 interrupt_after，方便人工先检查模型生成的草稿。
    """
    fallback = {
        "draft_reply": (
            f"{state['customer_name']}您好，已收到您反馈的课程未开通问题。"
            "我们会优先核对付款记录和课程权限，处理完成后第一时间通知您。"
            "如需补充信息，请提供付款时间或订单截图，感谢您的理解。"
        ),
        "internal_note": "高优先级课程开通问题，需要人工核对订单与课程权限。",
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是客服回复 Agent，请生成一份可发送给客户的回复草稿。
要求：
1. 语气礼貌、明确、可执行
2. 不承诺无法确认的结果
3. 如果是高风险或高优先级，需要说明会优先处理
4. 同时生成 internal_note，供内部客服查看
只返回 JSON，格式示例：
{{"draft_reply": "客户可见回复", "internal_note": "内部处理备注"}}""",
            ),
            (
                "user",
                """客户：{customer_name}
问题：{issue}
分类：{category}
风险：{risk_level}
摘要：{summary}""",
            ),
        ]
    )

    data = run_chain(
        prompt,
        {
            "customer_name": state["customer_name"],
            "issue": state["issue"],
            "category": state["category"],
            "risk_level": state["risk_level"],
            "summary": state["summary"],
        },
        fallback,
    )

    print("\n✍️ 已生成客服回复草稿")
    print(textwrap.indent(data["draft_reply"], "  "))
    print("\n🗒 内部备注")
    print(textwrap.indent(data["internal_note"], "  "))

    return {
        "draft_reply": data["draft_reply"],
        "internal_note": data["internal_note"],
        "audit_log": append_log(state, "已生成回复草稿，等待人工复核"),
    }


def human_review_node(state: TicketState) -> Dict[str, object]:
    """
    节点3：人工复核节点。
    interrupt_after 会让流程停在 draft_solution 后，人工把审核意见写回状态。
    """
    review = state.get("human_review", "approve")

    if review == "reject":
        print("\n❌ 人工复核：驳回草稿，流程结束，不发送回复")
        return {
            "send_status": "cancelled",
            "audit_log": append_log(state, "人工驳回草稿，流程终止"),
        }

    if review == "revise":
        final_reply = state.get("final_reply") or state["draft_reply"]
        print("\n🛠 人工复核：已采用人工修改后的最终回复")
        return {
            "final_reply": final_reply,
            "audit_log": append_log(state, "人工修改并通过草稿"),
        }

    print("\n✅ 人工复核：草稿通过")
    return {
        "final_reply": state["draft_reply"],
        "audit_log": append_log(state, "人工审核通过草稿"),
    }


def risk_check(state: TicketState) -> Dict[str, object]:
    """
    节点4：发送前风险检查。
    该节点之后会进入 send_reply；由于配置了 interrupt_before=["send_reply"]，
    系统会在真正发送前再次暂停，等待最终授权。
    """
    if state.get("risk_level") == "高" or state.get("priority") == "high":
        risk_note = "高优先级/高风险工单，发送前必须人工最终确认。"
    else:
        risk_note = "普通工单，发送前做常规确认。"

    print("\n🔎 发送前风险检查")
    print(f"  结论：{risk_note}")

    return {
        "risk_note": risk_note,
        "audit_log": append_log(state, "已完成发送前风险检查"),
    }


def send_reply(state: TicketState) -> Dict[str, object]:
    """
    节点5：模拟发送客服回复。
    真实业务里这里可以替换成邮件、短信、客服系统 API。
    """
    ticket_no = state["ticket_id"]
    print("\n📤 【客服回复已发送】")
    print(f"工单号：{ticket_no}")
    print(f"客户：{state['customer_name']}")
    print("发送内容：")
    print(textwrap.indent(state["final_reply"], "  "))

    return {
        "send_status": "success",
        "audit_log": append_log(state, "客服回复已发送"),
    }


def show_final_result(state: TicketState) -> Dict[str, object]:
    print("\n" + "=" * 56)
    print("📜 工单处理结束 · 审计记录")
    print(f"工单号：{state['ticket_id']}")
    print(f"处理状态：{state.get('send_status', 'pending')}")
    for idx, item in enumerate(state.get("audit_log", []), 1):
        print(f"{idx}. {item}")
    print("=" * 56)
    return {}


# ================== 4. 构建 LangGraph ==================
def build_ticket_graph():
    graph = StateGraph(TicketState)

    graph.add_node("classify_ticket", classify_ticket)
    graph.add_node("draft_solution", draft_solution)
    graph.add_node("human_review", human_review_node)
    graph.add_node("risk_check", risk_check)
    graph.add_node("send_reply", send_reply)
    graph.add_node("show_final_result", show_final_result)

    graph.add_edge(START, "classify_ticket")
    graph.add_edge("classify_ticket", "draft_solution")
    graph.add_edge("draft_solution", "human_review")

    def route_after_review(state: TicketState) -> str:
        if state.get("send_status") == "cancelled":
            return "show_final_result"
        return "risk_check"

    graph.add_conditional_edges("human_review", route_after_review)
    graph.add_edge("risk_check", "send_reply")
    graph.add_edge("send_reply", "show_final_result")
    graph.add_edge("show_final_result", END)

    return graph


# ================== 5. 交互入口 ==================
def ask_review(auto: bool, draft_reply: str) -> Dict[str, str]:
    if auto:
        return {"human_review": "approve"}

    print("\n请选择人工复核结果：")
    print("1. approve  直接通过")
    print("2. revise   修改后通过")
    print("3. reject   驳回，不发送")
    choice = input("请输入 approve / revise / reject：").strip().lower()

    if choice == "reject":
        return {"human_review": "reject"}

    if choice == "revise":
        print("\n请输入修改后的最终回复，直接回车则使用系统草稿：")
        edited = input("最终回复：").strip()
        return {
            "human_review": "revise",
            "final_reply": edited or draft_reply,
        }

    return {"human_review": "approve"}


def run_demo(auto: bool = False) -> None:
    memory = MemorySaver()
    app = build_ticket_graph().compile(
        checkpointer=memory,
        interrupt_after=["draft_solution"],
        interrupt_before=["send_reply"],
    )

    thread_id = f"ticket_review_{random.randint(1000, 9999)}"
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 56)
    print("🎫 客服工单智能处理系统 · LangGraph 中断与记忆 Demo")
    print("=" * 56)
    if llm is None:
        print("⚠️ 未检测到 API_KEY，本次将使用内置兜底回复演示流程。")

    print("\n=== 第一阶段：接收工单、分类、生成回复草稿 ===")
    first_state = init_ticket_state()
    draft_reply = ""

    for step in app.stream(first_state, config=config):
        if "draft_solution" in step:
            draft_reply = step["draft_solution"]["draft_reply"]
            print("\n🛑 interrupt_after 生效：系统已在【生成草稿后】暂停")
            print("   此时 MemorySaver 已保存 thread_id 对应的流程状态")

    review_update = ask_review(auto, draft_reply)
    app.update_state(config, review_update)

    print("\n=== 第二阶段：从草稿后恢复，执行人工复核与风险检查 ===")
    interrupted_before_send = False
    for step in app.stream(None, config=config):
        if "risk_check" in step:
            print("\n🛑 interrupt_before 生效：系统即将在【发送回复】前暂停")
            print("   需要最终确认后，才能继续执行 send_reply 节点")
            interrupted_before_send = True

    if not interrupted_before_send:
        return

    if auto:
        confirm = "确认发送"
    else:
        confirm = input("\n请输入最终授权指令（确认发送 / 取消）：").strip()

    if confirm == "确认发送":
        print("\n=== 第三阶段：最终授权通过，从发送前恢复 ===")
        app.invoke(None, config=config)
    else:
        print("\n❌ 未获得最终授权，流程停留在发送前，不会发送回复。")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="客服工单智能处理系统 Demo")
    arg_parser.add_argument("--auto", action="store_true", help="自动通过人工审核和最终发送确认")
    args = arg_parser.parse_args()
    run_demo(auto=args.auto)

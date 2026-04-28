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
        max_tokens=1000,
    )


llm = create_llm()
parser = StrOutputParser()


# ================== 1. 定义面试状态 ==================
class InterviewState(TypedDict):
    """
    面试状态字典，记录候选人从简历筛选、面试问答、AI评估、
    HR复核到最终通知发送的全流程信息。
    """

    candidate_id: str
    candidate_name: str
    position: str
    resume: str
    resume_summary: NotRequired[str]
    risk_flags: NotRequired[List[str]]
    questions: NotRequired[List[str]]
    answers: NotRequired[List[str]]
    interview_report: NotRequired[str]
    score: NotRequired[int]
    ai_decision: NotRequired[str]
    hr_review: NotRequired[str]
    hr_note: NotRequired[str]
    final_decision: NotRequired[str]
    notification: NotRequired[str]
    send_status: NotRequired[str]
    audit_log: List[str]


def init_interview_state() -> InterviewState:
    return {
        "candidate_id": "CAND-20260428-001",
        "candidate_name": "张三",
        "position": "初级 Python / LangGraph 开发工程师",
        "resume": (
            "候选人张三，计算机相关专业，2年 Python 开发经验。"
            "熟悉 FastAPI、SQL、基础前端，做过一个 RAG 问答系统 Demo，"
            "了解 LangChain 的 PromptTemplate、OutputParser 和简单工具调用。"
            "项目经验偏学习型，生产环境经验较少。"
        ),
        "audit_log": [],
    }


# ================== 2. 工具函数 ==================
def append_log(state: InterviewState, message: str) -> List[str]:
    logs = list(state.get("audit_log", []))
    logs.append(message)
    return logs


def extract_json(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("未找到 JSON 内容")
    return json.loads(match.group(0))


def run_chain(prompt: ChatPromptTemplate, inputs: Dict[str, object], fallback: Dict[str, object]) -> Dict[str, object]:
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
def screen_resume(state: InterviewState) -> Dict[str, object]:
    """
    节点1：简历筛选 Agent。
    提取简历摘要和风险点，帮助后续面试问题更聚焦。
    """
    fallback = {
        "resume_summary": "候选人有 Python 基础和 LangChain 学习项目经验，适合初级岗位继续考察。",
        "risk_flags": ["生产环境经验较少", "LangGraph 深度经验需要面试验证"],
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是招聘简历筛选专家，请分析候选人简历并输出 JSON。
要求：
1. resume_summary：一句话总结候选人与岗位匹配度
2. risk_flags：列出 1-3 个需要面试验证的风险点
只返回 JSON，格式示例：
{{"resume_summary": "候选人有 Python 基础，适合初级岗位", "risk_flags": ["项目经验偏学习型"]}}""",
            ),
            ("user", "岗位：{position}\n简历：{resume}"),
        ]
    )

    data = run_chain(
        prompt,
        {"position": state["position"], "resume": state["resume"]},
        fallback,
    )

    print("\n📄 简历筛选完成")
    print(f"  候选人：{state['candidate_name']}")
    print(f"  岗位：{state['position']}")
    print(f"  摘要：{data['resume_summary']}")
    print(f"  风险点：{'；'.join(data['risk_flags'])}")

    return {
        "resume_summary": data["resume_summary"],
        "risk_flags": data["risk_flags"],
        "audit_log": append_log(state, "已完成简历筛选"),
    }


def generate_questions(state: InterviewState) -> Dict[str, object]:
    """
    节点2：面试题生成 Agent。
    基于简历和岗位生成结构化问题。
    """
    fallback = {
        "questions": [
            "请介绍你做过的 RAG 问答系统 Demo，重点说明数据加载、切分、检索和生成链路。",
            "如果 LangGraph 工作流执行到一半失败，你会如何保存状态并恢复执行？",
            "请说明 FastAPI 接口中如何处理异常、日志和参数校验。",
        ]
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是技术面试官，请为候选人生成 3 个面试问题。
要求：
1. 问题要结合岗位和简历
2. 至少 1 个问题考察 LangGraph 或智能体工作流
3. 问题具体、可回答，不要太泛
只返回 JSON，格式示例：
{{"questions": ["问题1", "问题2", "问题3"]}}""",
            ),
            (
                "user",
                "岗位：{position}\n简历摘要：{resume_summary}\n风险点：{risk_flags}",
            ),
        ]
    )

    data = run_chain(
        prompt,
        {
            "position": state["position"],
            "resume_summary": state["resume_summary"],
            "risk_flags": "；".join(state["risk_flags"]),
        },
        fallback,
    )

    print("\n🎯 面试问题生成完成")
    for idx, question in enumerate(data["questions"], 1):
        print(f"  Q{idx}: {question}")

    return {
        "questions": data["questions"],
        "audit_log": append_log(state, "已生成面试问题"),
    }


def simulate_interview(state: InterviewState) -> Dict[str, object]:
    """
    节点3：候选人回答模拟。
    为了让 Demo 可独立运行，这里用 Agent 模拟候选人回答。
    真实系统中可替换成用户输入或语音转写结果。
    """
    fallback = {
        "answers": [
            "我的 RAG Demo 使用文档加载器读取资料，再用文本切分器切成小段，向量化后检索相关片段，最后把检索结果和问题一起交给大模型回答。",
            "我会使用 LangGraph 的 checkpointer，比如 MemorySaver 保存 thread_id 对应的状态。如果流程中断，可以用同一个 thread_id 从 checkpoint 继续执行。",
            "FastAPI 中我会用 Pydantic 做参数校验，用统一异常处理返回清晰错误信息，并在关键请求链路记录日志，方便排查问题。",
        ]
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你正在模拟一名真实候选人的面试回答。
要求：
1. 回答要符合简历水平，不要过度完美
2. 每个问题回答 50-120 字
3. 只返回 JSON，格式示例：
{{"answers": ["回答1", "回答2", "回答3"]}}""",
            ),
            (
                "user",
                "候选人简历：{resume}\n面试问题：{questions}",
            ),
        ]
    )

    data = run_chain(
        prompt,
        {
            "resume": state["resume"],
            "questions": "\n".join(state["questions"]),
        },
        fallback,
    )

    print("\n🗣 面试问答记录")
    for idx, (question, answer) in enumerate(zip(state["questions"], data["answers"]), 1):
        print(f"\nQ{idx}: {question}")
        print(f"A{idx}: {answer}")

    return {
        "answers": data["answers"],
        "audit_log": append_log(state, "已完成面试问答记录"),
    }


def evaluate_candidate(state: InterviewState) -> Dict[str, object]:
    """
    节点4：面试评估 Agent。
    该节点执行后会触发 interrupt_after，让 HR 先复核 AI 评估报告。
    """
    fallback = {
        "interview_report": (
            "候选人 Python 和 Web 基础较扎实，能描述 RAG 基本链路，"
            "也理解 LangGraph checkpoint 的恢复思路。但回答偏概念化，"
            "生产排障和复杂项目经验仍需入职后培养。"
        ),
        "score": 78,
        "ai_decision": "hire",
    }

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是资深技术面试官，请根据简历、问题和回答评估候选人。
要求：
1. interview_report：给出客观评价，包含优势、风险和培养建议
2. score：0-100 的整数分
3. ai_decision：hire / reject / hold 三选一
4. 不要夸大，不要做歧视性判断
只返回 JSON，格式示例：
{{"interview_report": "评价内容", "score": 80, "ai_decision": "hire"}}""",
            ),
            (
                "user",
                """岗位：{position}
简历摘要：{resume_summary}
风险点：{risk_flags}
问题：{questions}
回答：{answers}""",
            ),
        ]
    )

    data = run_chain(
        prompt,
        {
            "position": state["position"],
            "resume_summary": state["resume_summary"],
            "risk_flags": "；".join(state["risk_flags"]),
            "questions": "\n".join(state["questions"]),
            "answers": "\n".join(state["answers"]),
        },
        fallback,
    )

    print("\n📊 AI 面试评估完成")
    print(f"  分数：{data['score']}")
    print(f"  建议：{data['ai_decision']}")
    print("  评估：")
    print(textwrap.indent(str(data["interview_report"]), "    "))

    return {
        "interview_report": data["interview_report"],
        "score": int(data["score"]),
        "ai_decision": data["ai_decision"],
        "audit_log": append_log(state, "AI 已生成面试评估，等待 HR 复核"),
    }


def hr_review_node(state: InterviewState) -> Dict[str, object]:
    """
    节点5：HR 复核节点。
    人工复核意见通过 update_state 写入 checkpoint，再从中断点恢复执行。
    """
    review = state.get("hr_review", "approve")
    final_decision = state.get("final_decision") or state.get("ai_decision", "hold")
    hr_note = state.get("hr_note", "HR 同意 AI 面试建议。")

    if review == "reject":
        final_decision = "reject"
        hr_note = state.get("hr_note", "HR 驳回录用建议，转为不通过。")
        print("\n❌ HR 复核：驳回 AI 建议")
    elif review == "revise":
        print("\n🛠 HR 复核：已人工调整最终决策")
    else:
        print("\n✅ HR 复核：通过 AI 评估建议")

    print(f"  最终决策：{final_decision}")
    print(f"  HR 备注：{hr_note}")

    return {
        "final_decision": final_decision,
        "hr_note": hr_note,
        "audit_log": append_log(state, "HR 已完成面试评估复核"),
    }


def prepare_notification(state: InterviewState) -> Dict[str, object]:
    """
    节点6：通知生成 Agent。
    根据最终决策生成候选人通知文案。
    """
    decision = state["final_decision"]
    if decision == "hire":
        fallback_text = (
            f"{state['candidate_name']}您好，感谢您参加{state['position']}岗位面试。"
            "经过综合评估，我们认为您与岗位要求较为匹配，拟进入后续录用沟通环节。"
            "HR 将与您确认薪资、入职时间和材料准备事项。"
        )
    elif decision == "hold":
        fallback_text = (
            f"{state['candidate_name']}您好，感谢您参加{state['position']}岗位面试。"
            "目前我们希望再补充一次沟通，以进一步确认项目经验和岗位匹配度。"
            "HR 将联系您安排后续面谈。"
        )
    else:
        fallback_text = (
            f"{state['candidate_name']}您好，感谢您参加{state['position']}岗位面试。"
            "经过综合评估，本次暂未进入后续流程。感谢您的时间，也祝您求职顺利。"
        )

    fallback = {"notification": fallback_text}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是招聘 HR，请根据最终决策生成候选人通知。
要求：
1. 语气礼貌、克制、专业
2. 不泄露内部评分和详细面试评价
3. hire 表示进入录用沟通，reject 表示暂不通过，hold 表示需要补充面谈
只返回 JSON，格式示例：
{{"notification": "通知正文"}}""",
            ),
            (
                "user",
                "候选人：{candidate_name}\n岗位：{position}\n最终决策：{final_decision}\nHR备注：{hr_note}",
            ),
        ]
    )

    data = run_chain(
        prompt,
        {
            "candidate_name": state["candidate_name"],
            "position": state["position"],
            "final_decision": state["final_decision"],
            "hr_note": state["hr_note"],
        },
        fallback,
    )

    print("\n✉️ 已生成候选人通知")
    print(textwrap.indent(str(data["notification"]), "  "))

    return {
        "notification": data["notification"],
        "audit_log": append_log(state, "已生成候选人通知，等待最终发送确认"),
    }


def send_notification(state: InterviewState) -> Dict[str, object]:
    """
    节点7：模拟发送通知。
    由于 compile 时配置了 interrupt_before=["send_notification"]，
    真正发送前会先暂停，等待人工最终授权。
    """
    print("\n📤 【候选人通知已发送】")
    print(f"候选人：{state['candidate_name']}")
    print(f"岗位：{state['position']}")
    print("通知内容：")
    print(textwrap.indent(state["notification"], "  "))

    return {
        "send_status": "success",
        "audit_log": append_log(state, "候选人通知已发送"),
    }


def show_final_result(state: InterviewState) -> Dict[str, object]:
    print("\n" + "=" * 60)
    print("📜 招聘面试流程结束 · 审计记录")
    print(f"候选人：{state['candidate_name']}（{state['candidate_id']}）")
    print(f"岗位：{state['position']}")
    print(f"AI建议：{state.get('ai_decision')} | 最终决策：{state.get('final_decision')}")
    print(f"发送状态：{state.get('send_status', 'pending')}")
    for idx, item in enumerate(state.get("audit_log", []), 1):
        print(f"{idx}. {item}")
    print("=" * 60)
    return {}


# ================== 4. 构建 LangGraph ==================
def build_interview_graph():
    graph = StateGraph(InterviewState)

    graph.add_node("screen_resume", screen_resume)
    graph.add_node("generate_questions", generate_questions)
    graph.add_node("simulate_interview", simulate_interview)
    graph.add_node("evaluate_candidate", evaluate_candidate)
    graph.add_node("hr_review", hr_review_node)
    graph.add_node("prepare_notification", prepare_notification)
    graph.add_node("send_notification", send_notification)
    graph.add_node("show_final_result", show_final_result)

    graph.add_edge(START, "screen_resume")
    graph.add_edge("screen_resume", "generate_questions")
    graph.add_edge("generate_questions", "simulate_interview")
    graph.add_edge("simulate_interview", "evaluate_candidate")
    graph.add_edge("evaluate_candidate", "hr_review")
    graph.add_edge("hr_review", "prepare_notification")
    graph.add_edge("prepare_notification", "send_notification")
    graph.add_edge("send_notification", "show_final_result")
    graph.add_edge("show_final_result", END)

    return graph


# ================== 5. 交互入口 ==================
def ask_hr_review(auto: bool, ai_decision: str) -> Dict[str, str]:
    if auto:
        return {
            "hr_review": "approve",
            "final_decision": ai_decision,
            "hr_note": "自动演示模式：HR 同意 AI 面试建议。",
        }

    print("\n请选择 HR 复核结果：")
    print("1. approve  同意 AI 建议")
    print("2. revise   调整最终决策")
    print("3. reject   驳回录用建议，转为不通过")
    choice = input("请输入 approve / revise / reject：").strip().lower()

    if choice == "reject":
        note = input("请输入 HR 备注：").strip() or "HR 驳回 AI 建议，候选人暂不通过。"
        return {
            "hr_review": "reject",
            "final_decision": "reject",
            "hr_note": note,
        }

    if choice == "revise":
        decision = input("请输入最终决策 hire / hold / reject：").strip().lower()
        if decision not in {"hire", "hold", "reject"}:
            decision = ai_decision
        note = input("请输入 HR 备注：").strip() or "HR 根据业务情况调整了最终决策。"
        return {
            "hr_review": "revise",
            "final_decision": decision,
            "hr_note": note,
        }

    return {
        "hr_review": "approve",
        "final_decision": ai_decision,
        "hr_note": "HR 同意 AI 面试建议。",
    }


def run_demo(auto: bool = False) -> None:
    memory = MemorySaver()
    app = build_interview_graph().compile(
        checkpointer=memory,
        interrupt_after=["evaluate_candidate"],
        interrupt_before=["send_notification"],
    )

    thread_id = f"interview_{random.randint(1000, 9999)}"
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 60)
    print("🧑‍💼 AI 招聘面试官系统 · LangGraph 中断与记忆 Demo")
    print("=" * 60)
    if llm is None:
        print("⚠️ 未检测到 API_KEY，本次将使用内置兜底内容演示流程。")

    print("\n=== 第一阶段：简历筛选、出题、面试、AI评估 ===")
    ai_decision = "hold"
    for step in app.stream(init_interview_state(), config=config):
        if "evaluate_candidate" in step:
            ai_decision = step["evaluate_candidate"]["ai_decision"]
            print("\n🛑 interrupt_after 生效：系统已在【AI 生成面试评估后】暂停")
            print("   HR 可以先查看评分、建议和风险，再写入人工复核意见。")

    review_update = ask_hr_review(auto, ai_decision)
    app.update_state(config, review_update)

    print("\n=== 第二阶段：从 AI 评估后恢复，执行 HR 复核并生成通知 ===")
    interrupted_before_send = False
    for step in app.stream(None, config=config):
        if "prepare_notification" in step:
            print("\n🛑 interrupt_before 即将生效：下一步是正式发送候选人通知")
        if "__interrupt__" in step:
            print("   系统已在【发送通知前】暂停，等待最终授权。")
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
        print("\n❌ 未获得最终授权，流程停留在发送前，不会发送通知。")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="AI 招聘面试官系统 Demo")
    arg_parser.add_argument("--auto", action="store_true", help="自动通过 HR 复核和最终发送确认")
    args = arg_parser.parse_args()
    run_demo(auto=args.auto)

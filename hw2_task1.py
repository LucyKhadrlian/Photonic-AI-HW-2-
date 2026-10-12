import operator
import re
import subprocess
import time
from typing import Annotated, Literal, TypedDict, List, Optional
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

model_name = "gemini-2.5-flash"
data_path = r"C:\Users\kevor\OneDrive\Desktop\Lucy_AI Training\hw_1\task 5\SilverboxFiles\SNLS80mV.mat"
work_dir = Path("./pinn_work")
work_dir.mkdir(exist_ok=True)

call_pause = 13
retries = 6
max_decisions = 8


def get_llm(max_tokens=4096):
    return ChatGoogleGenerativeAI(model=model_name, max_output_tokens=max_tokens, temperature=0.3)


def with_retry(fn):
    for attempt in range(retries):
        try:
            out = fn()
            time.sleep(call_pause)
            return out
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"[model call failed: {str(e)[:150]}] retrying in {wait}s ({attempt + 1}/{retries})")
            time.sleep(wait)
    raise RuntimeError("Model call failed after several retries")


task5 = f"""
Background (Homework 1, Task 5): PINN-style model of the Silverbox oscillator.

Data: SNLS80mV.mat at {data_path}. V1 = input u(t), V2 = output y(t), fs = 610.35 Hz.
Excerpt: slice [42650:46746], centered. First 3072 samples = training, last 1024 = test.
Physics: y'' + a y' + b y + c y^3 = g u(t). Acceleration given by an MLP, state (y, v), RK4 integration.
Metric: one free run over the whole excerpt; TRAIN_RMSE and TEST_RMSE are printed by the script,
and a plot test_free_run.png is saved. Predicting zero scores about 0.06 V.
Scripts are limited to 5 minutes on a CPU.
"""


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    decision_count: int
    instruction: str
    train_rmse: float | None
    test_rmse: float | None
    pending: Optional[dict]   # what the human node is asking about


def dialogue_to_str(messages: List[BaseMessage]) -> str:
    lines = []
    for m in messages:
        role = getattr(m, "name", None) or getattr(m, "type", m.__class__.__name__)
        lines.append(f"{role}: {m.content}")
    return "\n\n".join(lines)


def text_of(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in msg.content)


def ask(system: str, human: str) -> str:
    out = with_retry(lambda: get_llm().invoke([SystemMessage(content=system), HumanMessage(content=human)]))
    return text_of(out)


class Decision(BaseModel):
    next_role: Literal["scientist", "general", "executor", "human", "finish"]
    instruction: str
    reason: str
    # only used when next_role == "human"
    question: str = ""
    kind: Literal["ambiguous", "hard", "none"] = "none"
    then_role: Literal["scientist", "general", "executor", "finish", "none"] = "none"
    then_instruction: str = ""


orch_sys = """You are the Orchestrator of a 4-agent system for the Silverbox task (Homework 1, Task 5). You write no theory and no code and have no tools.
Roles:
- scientist : writes the differential equation, network, training plan, free-run test and numbered coding steps (NO code)
- general   : answers high-level side questions (what a record is, what a term means, what an error measures)
- executor  : the only role with a coding tool; writes and runs a script for the scientist's latest plan, returns the printed RMSE or traceback
- human     : pauses the graph and asks the person a question; their reply is added to the dialogue as [human]
- finish    : stop (allowed only after an executor turn contains a test RMSE printed by a script that ran)

Choose `human` in exactly two situations:
1. AMBIGUOUS QUERY: the latest request has more than one reading, and the readings would send the work to different roles
   or change the number that gets reported. Set kind="ambiguous", put the question (listing the readings) in `question`,
   and leave then_role="none".
2. HARD DECISION: the next step would run the coding tool (executor), overwrite a result already on disk
   (e.g. test_free_run.png or a pinn_run_N.py), or end the dialogue and report a measurement (finish).
   Set kind="hard", state the step plainly in `question` and ask whether to proceed, and set then_role and then_instruction
   to the step that should run if the person accepts.

Rules:
- If the latest [human] message already approved a step, do not ask again about that same step.
- If the latest [System] message says the person refused a step, do not repeat that step at once: choose scientist,
  general, or ask the person what they want instead.
- A question about the plan goes to the scientist. A traceback or failed run goes to the executor to repair.
- You have at most 8 decisions in total (asking the human counts as one).
Return next_role, an instruction for that role, a one-sentence reason, and the human fields when relevant."""


def decide(system: str, human: str) -> Decision:
    llm = get_llm().with_structured_output(Decision)
    return with_retry(lambda: llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]))


def orchestrator(state: AgentState) -> Command[Literal["scientist", "general", "executor", "human", "__end__"]]:
    n = state["decision_count"]
    if n >= max_decisions:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=f"[Orchestrator] Max {max_decisions} decisions reached -> stopping")]}
        )

    facts = f"Decisions used: {n} of {max_decisions}. Verified test RMSE from a script that ran: {state['test_rmse']}. Scripts run so far: {len(runs)}."
    d = decide(orch_sys, f"Dialogue so far:\n{dialogue_to_str(state['messages'])}\n\n---\n{facts}\nMake your decision.")
    next_role, instruction, reason = d.next_role, d.instruction, d.reason

    if next_role == "finish" and state["test_rmse"] is None:
        has_plan = any(m.content.startswith("[Scientist]") for m in state["messages"])
        next_role = "executor" if has_plan else "scientist"
        instruction = ("Implement the latest plan, run it and report the printed RMSE." if has_plan
                       else "Write the plan for Task 5.")
        reason += " [overridden: cannot finish before a script has printed a test RMSE]"

    update = {"decision_count": n + 1, "instruction": instruction}

    if next_role == "human":
        kind = d.kind if d.kind != "none" else "ambiguous"
        question = d.question or instruction
        pending = {
            "kind": kind,
            "question": question,
            "then_role": (d.then_role if d.then_role != "none" else "executor") if kind == "hard" else "none",
            "then_instruction": d.then_instruction or instruction,
        }
        update["pending"] = pending
        update["messages"] = [AIMessage(content=f"[Orchestrator] decision {n + 1}: next=human ({kind}) | question={question} | reason={reason}")]
        return Command(goto="human", update=update)

    update["messages"] = [AIMessage(content=f"[Orchestrator] decision {n + 1}: next={next_role} | instruction={instruction} | reason={reason}")]
    if next_role == "finish":
        return Command(goto=END, update=update)
    return Command(goto=next_role, update=update)


# only parses the person's yes/no reply; the routing choice itself is still a model call
yes_pattern = re.compile(r"^\s*(y|yes|yep|ok|okay|proceed|go|go ahead|approve|approved|accept|sure|do it)\b", re.I)


def human(state: AgentState) -> Command[Literal["orchestrator", "scientist", "general", "executor", "__end__"]]:
    p = state["pending"]
    # Pauses here. The graph stays paused until resumed on the same thread with Command(resume=reply).
    reply = interrupt({"kind": p["kind"], "question": p["question"]})

    msgs = [HumanMessage(content=f"[human] {reply}", name="human")]

    if p["kind"] == "hard":
        if yes_pattern.match(str(reply)):
            msgs.append(AIMessage(content=f"[System] human approved. Running step: {p['then_role']}."))
            goto = END if p["then_role"] == "finish" else p["then_role"]
            return Command(goto=goto,
                           update={"messages": msgs, "instruction": p["then_instruction"], "pending": None})
        msgs.append(AIMessage(content=(
            f"[System] human refused the step '{p['then_role']}'. It did not run "
            f"(scripts run so far: {len(runs)}). Returning to the orchestrator.")))
        return Command(goto="orchestrator", update={"messages": msgs, "pending": None})

    # ambiguous: the next orchestrator decision reads the reply
    return Command(goto="orchestrator", update={"messages": msgs, "pending": None})


sci_sys = """You are the Scientist. You own the theory and the plan. Write: the differential equation, the network you propose
(inputs, scaling, depth, widths, activation, output scaling), the state equation to integrate and the integrator (RK4), how to train
(optimizer, learning rate, steps, rollout length; think about how to avoid a degenerate fit), the free-run test, and a numbered list
of steps specific enough to code. You have NO coding tool and you may NOT claim a number was measured unless an executor turn already
printed it. If the executor reported a failure or a poor RMSE, revise the training details but keep the same differential equation and
the same free-run test. Follow any [human] clarification in the dialogue. Keep it under about 700 words."""


def scientist(state: AgentState) -> Command[Literal["orchestrator"]]:
    human_msg = (f"Dialogue so far:\n{dialogue_to_str(state['messages'])}\n\n---\n"
                 f"Instruction from the orchestrator: {state['instruction']}")
    text = ask(sci_sys, human_msg)
    return Command(goto="orchestrator", update={"messages": [AIMessage(content=f"[Scientist]\n{text}")]})


gen_sys = """You are the General Agent. You answer high-level questions beside the fit: what a record in the file is, what a term in the
problem means, what a reported error measures. You have NO coding tool and you do NOT write the fitting plan. Answer briefly and do not
invent measured numbers."""


def general(state: AgentState) -> Command[Literal["orchestrator"]]:
    human_msg = (f"Dialogue so far:\n{dialogue_to_str(state['messages'])}\n\n---\n"
                 f"Instruction from the orchestrator: {state['instruction']}")
    text = ask(gen_sys, human_msg)
    return Command(goto="orchestrator", update={"messages": [AIMessage(content=f"[General]\n{text}")]})


runs = []


@tool
def write_and_run_python(code: str) -> str:
    """Write the Python source to a NEW file in the working directory, run it (5 minute limit) and return the exit code, stdout and stderr."""
    name = f"pinn_run_{len(runs) + 1}.py"
    (work_dir / name).write_text(code, encoding="utf-8")
    try:
        r = subprocess.run(["python", name], cwd=work_dir, capture_output=True, text=True, timeout=300)
        exit_code, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        exit_code, out, err = -1, "", "TIMEOUT after 300 s"
    runs.append({"file": name, "exit": exit_code, "stdout": out, "stderr": err})
    return f"file: {name}\nexit code: {exit_code}\n--- stdout ---\n{out[-3000:]}\n--- stderr ---\n{err[-3000:]}"


exec_sys = f"""You are the Executor, the only role with a coding tool (write_and_run_python). Implement the Scientist's LATEST plan in a new
script and run it with the tool. Do not change the scientific plan: same differential equation, same free-run test. The script must load
the data from {data_path}, follow the Task 5 excerpt, centering and split exactly, use PyTorch, print exactly the lines
`TRAIN_RMSE: <float>` and `TEST_RMSE: <float>` from the free run, save test_free_run.png, and finish inside 4 minutes on a CPU.
If a run fails, read the traceback, repair the script (a new file) and run it again. Report only numbers the script printed. In your
final message give the printed RMSEs or the traceback, briefly."""


def exec_agent(state: AgentState) -> str:
    llm_tool = get_llm(8192).bind_tools([write_and_run_python])
    human_msg = (f"Dialogue so far:\n{dialogue_to_str(state['messages'])}\n\n---\n"
                 f"Instruction from the orchestrator: {state['instruction']}")
    msgs = [SystemMessage(content=exec_sys), HumanMessage(content=human_msg)]
    for _ in range(3):
        ai = with_retry(lambda: llm_tool.invoke(msgs))
        msgs.append(ai)
        if not ai.tool_calls:
            return text_of(ai)
        for tc in ai.tool_calls:
            msgs.append(ToolMessage(content=write_and_run_python.invoke(tc["args"]),
                                    tool_call_id=tc["id"], name=tc["name"]))
    final = with_retry(lambda: get_llm().invoke(msgs))
    return text_of(final)


def executor(state: AgentState) -> Command[Literal["orchestrator"]]:
    start = len(runs)
    final = exec_agent(state)
    new = runs[start:]
    log = "\n".join(f"- {r['file']}: exit {r['exit']}\n  stdout (tail): {r['stdout'][-600:].strip()}\n  stderr (tail): {r['stderr'][-600:].strip()}"
                    for r in new) or "- no script was run this turn"
    update = {"messages": [AIMessage(content=f"[Executor]\n{final}\n\n[tool runs this turn]\n{log}")]}

    for r in reversed(new):
        if r["exit"] == 0:
            te = re.search(r"TEST_RMSE\s*[:=]\s*([0-9eE+.\-]+)", r["stdout"])
            tr = re.search(r"TRAIN_RMSE\s*[:=]\s*([0-9eE+.\-]+)", r["stdout"])
            if te:
                update["test_rmse"] = float(te.group(1))
                update["train_rmse"] = float(tr.group(1)) if tr else None
                break
    return Command(goto="orchestrator", update=update)


builder = StateGraph(AgentState)
builder.add_node("orchestrator", orchestrator)
builder.add_node("scientist", scientist)
builder.add_node("general", general)
builder.add_node("executor", executor)
builder.add_node("human", human)
builder.add_edge(START, "orchestrator")
graph = builder.compile(checkpointer=MemorySaver())


def run_session(request: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 80}
    inp = {
        "messages": [HumanMessage(content=task5.strip(), name="task"),
                     HumanMessage(content=f"[User request] {request}", name="user")],
        "decision_count": 0,
        "instruction": "",
        "train_rmse": None,
        "test_rmse": None,
        "pending": None,
    }
    printed = 0
    while True:
        for event in graph.stream(inp, config, stream_mode="values"):
            msgs = event["messages"]
            for m in msgs[printed:]:
                if printed > 0 or m is not msgs[0]:
                    print(m.content)
                    print("-" * 60)
            printed = len(msgs)

        snap = graph.get_state(config)
        if not snap.next:          # graph finished
            break
        payload = snap.tasks[0].interrupts[0].value
        print(f"\n>>> GRAPH PAUSED ({payload['kind']}): {payload['question']}")
        reply = input("Your reply > ")
        inp = Command(resume=reply)   # resume the SAME thread

    final_state = graph.get_state(config).values
    print("\n=== Finished ===")
    print("decisions used     :", final_state["decision_count"])
    print("free-run RMSE train:", final_state["train_rmse"])
    print("free-run RMSE test :", final_state["test_rmse"])
    Path(f"dialogue_{thread_id}.txt").write_text(dialogue_to_str(final_state["messages"]), encoding="utf-8")
import operator
import re
import subprocess
import time
from typing import Annotated, Literal, TypedDict, List
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from pydantic import BaseModel


# Free Gemini model. If it is rejected, open Google AI Studio and use a model name shown there.
model_name = "gemini-2.5-flash"
DATA_PATH = r"C:\Users\kevor\OneDrive\Desktop\Lucy_AI Training\hw_1\task 5\SilverboxFiles\SNLS80mV.mat"
WORK_DIR = Path("./pinn_work")
WORK_DIR.mkdir(exist_ok=True)

# The free tier allows only a few requests per minute, so pause between calls and retry on errors.
CALL_PAUSE = 13      # seconds to wait after every model call
RETRIES = 6


def get_llm(max_tokens=4096):
    return ChatGoogleGenerativeAI(model=model_name, max_output_tokens=max_tokens, temperature=0.3)


def with_retry(fn):
    """Run fn(); on a rate-limit or temporary error wait and try again."""
    for attempt in range(RETRIES):
        try:
            out = fn()
            time.sleep(CALL_PAUSE)
            return out
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"[model call failed: {str(e)[:150]}] retrying in {wait}s ({attempt + 1}/{RETRIES})")
            time.sleep(wait)
    raise RuntimeError("Model call failed after several retries")


TASK5 = f"""
Task 5: (PINN-style) model of the Silverbox oscillator.

Data: file SNLS80mV.mat at {DATA_PATH} (use scipy.io.loadmat, write the path as a raw string r"..."). Channel V1 is the
input voltage u(t), channel V2 is the output voltage y(t). Sampling fs = 610.35 Hz, dt = 1/fs.
Excerpt: the 4096 samples starting at index 42650, i.e. the slice [42650:46746]. Subtract the mean of u and the
mean of y on this excerpt and use the centered signals afterwards.
Split: first 3072 samples = training record, last 1024 samples = test record. On the test record u is known
and y is what the model must produce.

Physics: y'' + a y' + b y + c y^3 = g u(t)  (a: damping, a few tens of 1/s; b: linear stiffness, the output is
dominated by frequencies near 70 Hz; c: cubic stiffness; g: input gain). These are circuit properties, not
network weights.

Model: the acceleration is produced by a multilayer fully connected PyTorch network. With velocity v = y',
the state is (y, v) and  y' = v,  v' = alpha_theta(y, v, u). Integrate the state at step dt, driven by the
measured input, with classical Runge-Kutta 4. Depth, widths, activation and the scaling of inputs and output
are free. Try more than one architecture; tune optimizer, learning rate, number of steps and the length of the
rollout used for training.

Metric: after the weights are fixed, do ONE free run of the whole excerpt: start at the first sample with the
measured position and a finite-difference velocity, integrate through the training samples and on into the test
samples, keeping the same state when the test segment begins. On the test segment the rollout may read u but
may not read y. Performance = RMSE of this free run on the 1024 test samples. Also compute the same free-run
RMSE on the 3072 training samples. Predicting zero scores about 0.06 V (the baseline).

Practical limits: every script run is limited to 5 minutes on a CPU, so size the training to finish inside about
4 minutes. Scripts must print exactly two lines
TRAIN_RMSE: <float>
TEST_RMSE: <float>
and save a plot of measured vs predicted output on the test segment as test_free_run.png.
"""


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    decision_count: int
    instruction: str
    train_rmse: float | None
    test_rmse: float | None


def dialogue_to_str(messages: List[BaseMessage]) -> str:
    lines = []
    for m in messages:
        role = getattr(m, "type", m.__class__.__name__)
        lines.append(f"{role.upper()}: {m.content}")
    return "\n\n".join(lines)


def text_of(msg) -> str:
    if isinstance(msg.content, str):
        return msg.content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in msg.content)


def ask(system: str, human: str) -> str:
    """One model call. The task goes in a HumanMessage."""
    out = with_retry(lambda: get_llm().invoke([SystemMessage(content=system), HumanMessage(content=human)]))
    return text_of(out)


class Decision(BaseModel):
    next_role: Literal["scientist", "general", "executor", "finish"]
    instruction: str
    reason: str


ORCH_SYS = """You are the Orchestrator of a 4-agent system that must solve the Silverbox task. You write no theory and no code and have no tools.
Roles:
- scientist : writes the differential equation, network, state equation, training plan, free-run test and numbered coding steps (NO code)
- general   : answers high-level side questions (what a record is, what a term means, what an error measures); does not write the fitting plan
- executor  : the only role with a coding tool; writes and runs a script for the scientist's latest plan, returns the printed RMSE or the traceback
- finish    : stop (allowed only after an executor turn contains a test RMSE printed by a script that ran)
A question about the plan goes to the scientist. A traceback or failed run goes to the executor to repair. You have at most 8 decisions in total.
Return next_role, an instruction for that role, and a one-sentence reason."""


def decide(system: str, human: str) -> Decision:
    llm = get_llm().with_structured_output(Decision)
    return with_retry(lambda: llm.invoke([SystemMessage(content=system), HumanMessage(content=human)]))


def orchestrator(state: AgentState) -> Command[Literal["scientist", "general", "executor", "__end__"]]:
    n = state["decision_count"]
    if n >= 8:
        return Command(
            goto=END,
            update={"messages": [AIMessage(content="[Orchestrator] Max 8 decisions reached -> stopping")]}
        )

    facts = f"Decisions used: {n} of 8. Verified test RMSE from a script that ran: {state['test_rmse']}."
    d = decide(ORCH_SYS, f"Dialogue so far:\n{dialogue_to_str(state['messages'])}\n\n---\n{facts}\nMake your decision.")
    next_role, instruction, reason = d.next_role, d.instruction, d.reason

    if next_role == "finish" and state["test_rmse"] is None:
        has_plan = any(m.content.startswith("[Scientist]") for m in state["messages"])
        next_role = "executor" if has_plan else "scientist"
        instruction = ("Implement the latest plan, run it and report the printed RMSE." if has_plan
                       else "Write the plan for Task 5.")
        reason += " [overridden: cannot finish before a script has printed a test RMSE]"

    msg = AIMessage(content=f"[Orchestrator] decision {n + 1}: next={next_role} | instruction={instruction} | reason={reason}")
    update = {"messages": [msg], "decision_count": n + 1, "instruction": instruction}
    if next_role == "finish":
        return Command(goto=END, update=update)
    return Command(goto=next_role, update=update)


# ──────────────────────────────────────────────
# SCIENTIST  (no coding tool)
# ──────────────────────────────────────────────
SCI_SYS = """You are the Scientist. You own the theory and the plan. Write: the differential equation, the network you propose
(inputs, scaling, depth, widths, activation, output scaling), the state equation to integrate and the integrator (RK4), how to train
(optimizer, learning rate, steps, rollout length; think about how to avoid a degenerate fit), the free-run test, and a numbered list
of steps specific enough to code. You have NO coding tool and you may NOT claim a number was measured unless an executor turn already
printed it. If the executor reported a failure or a poor RMSE, revise the training details but keep the same differential equation and
the same free-run test. Keep it under about 700 words."""


def scientist(state: AgentState) -> Command[Literal["orchestrator"]]:
    human = (f"Dialogue so far:\n{dialogue_to_str(state['messages'])}\n\n---\n"
             f"Instruction from the orchestrator: {state['instruction']}")
    text = ask(SCI_SYS, human)
    return Command(goto="orchestrator", update={"messages": [AIMessage(content=f"[Scientist]\n{text}")]})



GEN_SYS = """You are the General Agent. You answer high-level questions beside the fit: what a record in the file is, what a term in the
problem means, what a reported error measures. You have NO coding tool and you do NOT write the fitting plan. Answer briefly and do not
invent measured numbers."""


def general(state: AgentState) -> Command[Literal["orchestrator"]]:
    human = (f"Dialogue so far:\n{dialogue_to_str(state['messages'])}\n\n---\n"
             f"Instruction from the orchestrator: {state['instruction']}")
    text = ask(GEN_SYS, human)
    return Command(goto="orchestrator", update={"messages": [AIMessage(content=f"[General]\n{text}")]})


RUNS = []   


@tool
def write_and_run_python(code: str) -> str:
    """Write the Python source to a NEW file in the working directory, run it (5 minute limit) and return the exit code, stdout and stderr."""
    name = f"pinn_run_{len(RUNS) + 1}.py"
    (WORK_DIR / name).write_text(code, encoding="utf-8")
    try:
        r = subprocess.run(["python", name], cwd=WORK_DIR, capture_output=True, text=True, timeout=300)
        exit_code, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        exit_code, out, err = -1, "", "TIMEOUT after 300 s"
    RUNS.append({"file": name, "exit": exit_code, "stdout": out, "stderr": err})
    return f"file: {name}\nexit code: {exit_code}\n--- stdout ---\n{out[-3000:]}\n--- stderr ---\n{err[-3000:]}"


EXEC_SYS = f"""You are the Executor, the only role with a coding tool (write_and_run_python). Implement the Scientist's LATEST plan in a new
script and run it with the tool. Do not change the scientific plan: same differential equation, same free-run test. The script must load
the data from {DATA_PATH}, follow the Task 5 excerpt, centering and split exactly, use PyTorch, print exactly the lines
`TRAIN_RMSE: <float>` and `TEST_RMSE: <float>` from the free run, save test_free_run.png, and finish inside 4 minutes on a CPU.
If a run fails, read the traceback, repair the script (a new file) and run it again. Report only numbers the script printed. In your
final message give the printed RMSEs or the traceback, briefly."""


def exec_agent(state: AgentState) -> str:
    """The executor's model loop: it calls the coding tool itself, up to 3 write+run rounds per turn."""
    llm_tool = get_llm(8192).bind_tools([write_and_run_python])
    human = (f"Dialogue so far:\n{dialogue_to_str(state['messages'])}\n\n---\n"
             f"Instruction from the orchestrator: {state['instruction']}")
    msgs = [SystemMessage(content=EXEC_SYS), HumanMessage(content=human)]
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
    start = len(RUNS)
    final = exec_agent(state)
    new = RUNS[start:]
    log = "\n".join(f"- {r['file']}: exit {r['exit']}\n  stdout (tail): {r['stdout'][-600:].strip()}\n  stderr (tail): {r['stderr'][-600:].strip()}"
                    for r in new) or "- no script was run this turn"
    update = {"messages": [AIMessage(content=f"[Executor]\n{final}\n\n[tool runs this turn]\n{log}")]}

    # The RMSE counts only if a script that actually ran (exit code 0) printed it.
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
builder.add_edge(START, "orchestrator")
graph = builder.compile()


if __name__ == "__main__":
    initial = {
        "messages": [HumanMessage(content=TASK5.strip())],
        "decision_count": 0,
        "instruction": "",
        "train_rmse": None,
        "test_rmse": None,
    }

    print("=== Starting Agentic System (Gemini) ===\n")
    final_state = None
    for event in graph.stream(initial, stream_mode="values", config={"recursion_limit": 60}):
        print(event["messages"][-1].content)
        print("-" * 60)
        final_state = event

    print("\n=== Finished ===")
    print("decisions used     :", final_state["decision_count"])
    print("free-run RMSE train:", final_state["train_rmse"])
    print("free-run RMSE test :", final_state["test_rmse"])
    Path("dialogue.txt").write_text(dialogue_to_str(final_state["messages"]), encoding="utf-8")
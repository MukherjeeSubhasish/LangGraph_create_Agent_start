# Simple ReAct Agent from Scratch — Anthropic (Claude) version
# Based on https://til.simonwillison.net/llms/python-react-pattern
#
# Setup:
#   uv pip install anthropic python-dotenv
#   .env file in this folder:  ANTHROPIC_API_KEY=sk-ant-...

import re
from dotenv import load_dotenv
from anthropic import Anthropic

_ = load_dotenv()
client = Anthropic()  # reads ANTHROPIC_API_KEY from env

MODEL = "claude-sonnet-5-5"  # check console.anthropic.com for current model names

resp = client.messages.create(
    model=MODEL,
    max_tokens=256,                      # required by Anthropic
    messages=[{"role": "user", "content": "Hello world"}],
)
print(resp.content[0].text)


# Agent class
class Agent:
    def __init__(self, system=""):
        self.system = system             # Anthropic: system prompt is a separate param,
        self.messages = []               # not a "system" message in the list

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    # def execute(self):
    #     resp = client.messages.create(
    #         model=MODEL,
    #         max_tokens=1024,
    #         # temperature=0,
    #         system=self.system,
    #         messages=self.messages,
    #         stop_sequences=["PAUSE"],    # stop so Claude doesn't invent its own Observation
    #     )
    #     return resp.content[0].text

    # def execute(self):
    #     resp = client.messages.create(
    #         model=MODEL,
    #         max_tokens=1024,
    #         system=self.system,
    #         messages=self.messages,
    #         stop_sequences=["PAUSE"],
    #     )
    #     text = "".join(b.text for b in resp.content if b.type == "text").strip()
    #     if resp.stop_reason == "stop_sequence":
    #         text += "\nPAUSE"          # keep history matching the prompt's format
    #     return text or "(empty response)"

    def execute(self, retries=1):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=self.system,
            messages=self.messages,
            stop_sequences=["PAUSE"],
        )
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not text:
            print(f"[debug] empty reply, stop_reason={resp.stop_reason}")
            if retries:
                return self.execute(retries - 1)
        if resp.stop_reason == "stop_sequence":
            text += "\nPAUSE"
        return text or "(empty response)"


# ReAct prompt
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()


# Tools
def calculate(what):
    return eval(what)  # fine for learning; never eval untrusted input in real code


def average_dog_weight(name):
    if name in "Scottish Terrier":
        return "Scottish Terriers average 20 lbs"
    elif name in "Border Collie":
        return "a Border Collies average weight is 37 lbs"
    elif name in "Toy Poodle":
        return "a toy poodles average weight is 7 lbs"
    else:
        return "An average dog weights 50 lbs"


known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight,
}

# Manual loop, single tool call
abot = Agent(prompt)
print(abot("How much does a toy poodle weigh?"))

result = average_dog_weight("Toy Poodle")
print(abot(f"Observation: {result}"))
print(abot.messages)

# Manual loop, multiple tool calls
abot = Agent(prompt)
question = "I have 2 dogs, a border collie and a scottish terrier. What is their combined weight"
print(abot(question))
print(abot(f"Observation: {average_dog_weight('Border Collie')}"))
print(abot(f"Observation: {average_dog_weight('Scottish Terrier')}"))
print(abot(f"Observation: {eval('37 + 20')}"))

# Automated loop
action_re = re.compile(r"^Action: (\w+): (.*)$")  # raw string avoids a warning in Python 3.12


def query(question, max_turns=5):
    bot = Agent(prompt)
    next_prompt = question
    for _ in range(max_turns):
        result = bot(next_prompt)
        print(result)
        actions = [action_re.match(a) for a in result.split("\n") if action_re.match(a)]
        if not actions:
            return
        action, action_input = actions[0].groups()
        if action not in known_actions:
            raise Exception(f"Unknown action: {action}: {action_input}")
        print(f" -- running {action} {action_input}")
        observation = known_actions[action](action_input)
        print("Observation:", observation)
        next_prompt = f"Observation: {observation}"


query("I have 2 dogs, a border collie and a scottish terrier. What is their combined weight")
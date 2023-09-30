import subprocess

# Prompt formatting for Mistral
B_INST, E_INST = "[INST]", "[/INST]"


class BufferWindowMemory:
    """
    The conversation memory (buffer) of the LLM, which is windowed and limited to the number of steps given.

    Default steps = 2.
    """
    def __init__(self, steps=2):
        self.memory = []
        self.steps = steps

    def add_to_history(self, user_input: str, ai_output: str):
        user = "<User>: " + user_input.strip()
        ai = "<AI>: " + ai_output.strip()

        memory = (user, ai)
        self.memory.append(memory)
    
    def get_history_windowed(self):
        history: str = ""
        memory = None
        if len(self.memory) >= self.steps:
            memory = self.memory[-self.steps: ]
        else:
            memory = self.memory

        for tup in memory:
            history += f"{tup[0]}\n{tup[1]}\n"
        
        return history

    
prompt_template = """\
You are given a Chat History between you and a user, along with a followup question.
Your task is to answer the given question considering it as a standalone question, using the Chat History.

Note: Chat History can be empty. In that case, answer the given question considering it as a standalone question.
Make sure it seems like a conversation.

Chat History:
{history}
Followup question:
{question}

Always start your response with '<AI>:'
"""


def make_prompt(user_prompt: str):
    """
    This function creates a prompt that is suitable for Mistral AI.

    Can change according to the LLM
    """
    global PROMPT_END_INDEX
    prompt = "<s>" + B_INST + user_prompt + E_INST

    PROMPT_END_INDEX = len(prompt)
    return prompt


# TODO: can change steps according to the system
history = BufferWindowMemory(steps=2)

# TODO: GGUF mistral model path/HF id
mistral_path = "<MODEL_PATH_OR_HUGGING_FACE_MODEL_ID>"

while True:
    user_input = str(input("\nPrompt (q to quit): "))
    user_input = user_input.strip()

    if user_input == 'q' or user_input == 'exit' or user_input == 'Q' or user_input == 'quit':
        break

    hist = history.get_history_windowed()
    prompt = prompt_template.format(history=hist, question=user_input)

    prompt = make_prompt(prompt)

    print("\nThinking...\n")

    # Make sure llama.cpp in same path and compiled beforehand
    command = f'cd llama.cpp; ./main -m {mistral_path} -n 512 -ngl 1 --multiline-input --color -p "{prompt}" 2>/dev/null'

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True
    )

    # Clean output
    response = result.stdout.decode("utf-8")
    index = response.find(E_INST)
    response = response[index + len(E_INST): ].strip()
    response = response.replace("<AI>:", "").strip()

    print("🤖: " + response)

    history.add_to_history(user_input, response)

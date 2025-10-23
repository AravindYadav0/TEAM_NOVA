# AI-Powered Veterinary Appointment Scheduler

## 1. Problem Statement

In a veterinary clinic, when a client cancels an appointment, the valuable time slot often goes unused while other pet owners are still waiting for earlier availability. This module aims to **automatically fill canceled slots** by selecting the best replacement client from the waitlist using AI. The selection considers:  
- Pet urgency or type of service  
- Client flexibility  
- Fairness (who has been waiting the longest)  

The AI provides both the **selected client** and a brief **reasoning** behind the choice, improving clinic efficiency and customer satisfaction.

---

## 2. Tools / Technologies Used

- **Python 3.10+** – programming language  
- **LangChain** – for prompt templates and chaining LLMs  
- **Hugging Face Hub** – to access the `google/flan-t5-base` model   
- **Optional**: Google Colab for cloud-based execution  

---

## 3. Code with Team Name

**Team Name:** *[NOVA]*  

```python
# Import Libraries
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFaceHub
import os

# Set Hugging Face API Token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_token_here"

# Initialize Hugging Face LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base", 
    model_kwargs={"temperature": 0.6, "max_length": 200}
)

# Define prompt template
template = """
You are an AI scheduling assistant for a veterinary clinic.
A client canceled their appointment for {service_type} at {time_slot}.
Here is the waitlist:

{waitlist}

Select the best replacement client for this open slot, considering:
- Pet urgency or type of service
- Client flexibility
- Fairness (who has been waiting longest)
Explain your reasoning briefly and output the chosen client name.

Output format:
Client: <name>
Reason: <short explanation>
"""

prompt = PromptTemplate(
    input_variables=["service_type", "time_slot", "waitlist"],
    template=template
)

# Create the AI chain
chain = LLMChain(llm=llm, prompt=prompt)

# Example Data
service_type = "vaccination"
time_slot = "Monday 10:00 AM"
waitlist = """
1. Sarah - vaccination for puppy, flexible any time
2. Tom - surgery follow-up, needs early slot
3. Alice - routine check-up, prefers afternoons
"""

# Run the AI
response = chain.run(service_type=service_type, time_slot=time_slot, waitlist=waitlist)
print(response)

from ragas import evaluate
from dotenv import load_dotenv
from src.rag.pipeline import RAGAgent
from ragas import EvaluationDataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.metrics import LLMContextRecall, Faithfulness, ContextUtilization
from configs.prompts.configs import configs
import yaml
import json

load_dotenv()

with open("data/test_questions.yaml", "r") as file:
    data = yaml.safe_load(file)

questions = data.get("questions", [])

questions = [ q["question"] for q in questions]

expected_responses = [
  "Utilities are using AMI data to proactively communicate unusual or high usage to customers through mid-cycle alerts rather than waiting for the bill. Common methods include email, text messages, automated phone calls, mobile apps, and customer web portals. For example, Central Maine Power (CMP) implemented a Bill Alert program that provided weekly updates on usage and costs via email, text, or phone, helping customers identify spikes early. These approaches emphasize advance customer awareness and behavior change, contrasting with tactics like unannounced field visits triggered by usage anomalies. ",

  "Successful AMI deployments have used customer web portals (similar to 'My Account') combined with proactive messaging. Reliant’s e-Sense portal provided near-real-time usage data, supported by weekly email summaries with clear visuals and minimal navigation, which customers reported helped avoid bill shock. CMP similarly paired its customer portal with Bill Alerts, enabling customers to track consumption and costs online. Both examples highlight simple design, frequent summaries, and integration of AMI data into self-serve customer accounts. ",

  "The provided research discusses market saturation indicators for lighting but does not define specific thresholds for transitioning to market transformation or sunsetting programs. Findings show CFL socket saturation plateauing around 35%, LED sales shares varying roughly 10–43% by state, and about 29% of U.S. households reporting at least one LED installed. While these metrics are commonly used to assess market maturity, the references do not establish clear cutoff levels nor analyze the pros and cons of ending incentives by distributor versus regionally. ",

  "There is documented evidence that providing customers with AMI-enabled digital access and feedback can result in validated energy savings. CMP’s Bill Alert program, supported by a customer web portal that allowed users to monitor energy use and costs, achieved approximately a 1.8% annual reduction in electricity consumption among participants, with about 70% reporting that the alerts prompted them to take action. This demonstrates measurable kWh savings tied to AMI-driven customer insights. ",

  "High bill or usage alerts are being used in a DSM-like capacity by some utilities. CMP’s Bill Alert pilot is a clear example, functioning as a behavioral intervention that produced measurable energy savings. However, while the program’s impacts are documented, the reference materials do not provide specific contact names at utilities currently operating these high-bill-alert initiatives. :contentReference[oaicite:4]{index=4}"
]


dataset = []

for query,reference in zip(questions,expected_responses):
    
    conversation_history = [{"role": "assistant", "content": "Hi there! I'm your E Source Assistant. How can I help you today?"}, {"role": "user", "content": query}]
    
    response, relevant_docs = RAGAgent(user_query=query, conversation_history=conversation_history, configs=configs)
    
    print(f"relevant_docs: {relevant_docs}")
    
    dataset.append(
        {
            "user_input":query,
            "retrieved_contexts":[docs["page_content"] for docs in relevant_docs],
            "response":response.answer,
            "reference":reference
        }
    )
    
evaluation_dataset = EvaluationDataset.from_list(dataset)

llm = ChatOpenAI(model="gpt-4.1-mini")
evaluator_llm = LangchainLLMWrapper(llm)

result = evaluate(dataset=evaluation_dataset,metrics=[LLMContextRecall(), Faithfulness(), ContextUtilization()],llm=evaluator_llm)
print(f"Evaluation Results: {result}")

row_scores_path = "evaluation_results/ragas_row_scores-v3.jsonl"
with open(row_scores_path, "w", encoding="utf-8") as f:
    for r in result.to_pandas().to_dict(orient="records"):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def classify_request(user_request: str) -> str:
    prompt = f"""
You are a data request classifier. Classify the request below into exactly ONE category.

Categories:
- DATA_PULL: User wants data, numbers, a list, a chart, a trend, or any visual/summary from the data.
  Examples: "show monthly sales trend", "top 10 products", "sales by region chart",
            "give me revenue in 2017", "show me orders from California", "total units sold",
            "sales breakdown by category", "visualize profit by segment"
- WHY_QUESTION: User wants to understand WHY a metric is high/low, what is causing a problem,
  or what is driving performance. Must contain "why", "underperform", "reason", "cause",
  "performing well/poorly", "what's driving", or similar analytical intent.
  Examples: "why is central region underperforming", "which region is performing well and why",
            "why did sales drop", "what's causing low profit in furniture"
- FILE_ANALYSIS: User explicitly uploaded a file and wants general insights about that file
  with no specific question. Only use this when there is NO specific question asked.

Request: "{user_request}"

Rules:
- When in doubt between DATA_PULL and FILE_ANALYSIS, always choose DATA_PULL
- DASHBOARD is not a valid category — classify visual/chart requests as DATA_PULL
- Reply with only the category name. Nothing else.
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip().upper()
    # Validate against known categories — default to DATA_PULL if unknown
    valid = {"DATA_PULL", "WHY_QUESTION", "FILE_ANALYSIS"}
    for category in valid:
        if category in raw:
            return category
    return "DATA_PULL"
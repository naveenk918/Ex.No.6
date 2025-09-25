# Ex.No.6 Development of Python Code Compatible with Multiple AI Tools

**Date:** 10-09-2025  
**Register No:** 212223060184  
**Name:** NAVEEN K  

---

## Aim
To develop and implement Python code that works with multiple AI tools for automating tasks like interacting with APIs, comparing their outputs, and generating actionable insights.

---

## AI Tools Used
- **OpenAI GPT (ChatGPT API)**
- **Hugging Face Transformers**
- **LangChain Framework** (optional orchestration)
- **Google Generative AI API** (optional extension)

---

## Explanation
In this experiment, the **Persona Prompting Pattern** is applied by acting as a **Business Data Analyst**.  
The focus is **customer review analysis** through:
- Sentiment classification  
- Keyword extraction  
- Summarization  

---

## Steps Followed
1. **Define Persona** → Role as a Business Data Analyst to study customer feedback.  
2. **Hugging Face Transformers** → Perform quick sentiment detection and summarization.  
3. **OpenAI GPT** → Provide deeper insights such as actionable suggestions and improvement points.  
4. **LangChain** → Create structured workflows for chaining prompts and enhancing automation.  
5. **Google Generative AI (Gemini)** → Optionally extend with another LLM for cross-verification of insights.  

---

## Conclusion
The integration demonstrated that different AI tools complement one another:

- **Hugging Face** → quick & lightweight analysis  
- **OpenAI GPT** → more contextual and actionable insights  
- **LangChain** → structured orchestration with prompt templating  
- **Google Generative AI** → optional extra layer of validation  

Thus, combining these tools helps achieve **faster, richer, and more reliable analysis of customer feedback**.

## Import Required Libraries
```python
from transformers import pipeline
import openai
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Google Generative AI (optional)
try:
    import google.generativeai as genai
except ImportError:
    genai = None
```
## Input Text (Customer Review)
```python
review_text = """ The product quality is amazing, especially the battery backup and display. 
However, the delivery was delayed by a week, and the mobile app has frequent crashes. 
Customer support was polite and helpful. Overall, satisfied but improvements are needed. """
```
## 1. Hugging Face Transformers
```python
print("=== Hugging Face Summarization & Sentiment ===")

summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")

hf_summary = summarizer(review_text, max_length=50, min_length=15, do_sample=False)
hf_sentiment = sentiment_analyzer(review_text)

print("HF Summary:", hf_summary[0]['summary_text'])
print("HF Sentiment:", hf_sentiment[0])
```
## 2. OpenAI GPT (ChatGPT API)
```python
print("\n=== OpenAI GPT Analysis ===")

openai.api_key = "YOUR_OPENAI_API_KEY"

prompt = f"""
Analyze the following customer review:

Review: {review_text}

1. Provide a short summary (max 40 words).
2. Identify the sentiment (positive/negative/neutral).
3. Suggest one improvement for the service/product.
"""

gpt_response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=150,
    temperature=0.5
)

gpt_output = gpt_response["choices"][0]["text"].strip()
print(gpt_output)
```
## 3. LangChain Framework (optional orchestration
```python
print("\n=== LangChain Workflow (using OpenAI GPT) ===")

template = """ You are a business data analyst. 
Analyze this customer review: {review}

Return:
- Summary (20–30 words)
- Sentiment
- Top 3 keywords
"""

prompt_template = PromptTemplate(input_variables=["review"], template=template)
llm = LangChainOpenAI(openai_api_key="YOUR_OPENAI_API_KEY", model_name="text-davinci-003")
chain = LLMChain(llm=llm, prompt=prompt_template)

langchain_result = chain.run(review=review_text)
print(langchain_result)
```
## 4. Google Generative AI API (Optional)
```python
if genai:
    print("\n=== Google Generative AI (Gemini) ===")
    genai.configure(api_key="YOUR_GOOGLE_API_KEY")

    model = genai.GenerativeModel("gemini-pro")
    gemini_prompt = f"Summarize and analyze the following review:\n\n{review_text}"

    response = model.generate_content(gemini_prompt)
    print(response.text)
else:
    print("\n[Google Generative AI not installed. Skipping this step.]")
```
## Result
The Python code executed successfully. It showed that sentiment analysis and keyword extraction become much more powerful when multiple AI tools are combined, as each tool contributes unique strengths.

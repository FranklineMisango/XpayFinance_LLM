# XpayFinance_LLM
A Large Language ML Model that summarizes the Financial health of a company using OpenAI APIs and Streamlit Rendering. This project demonstrates the utilization of a ChromaDB vectorstore within a Langchain pipeline to develop a GPT Investment Summarizer. The system allows for loading PDF documents and employing them in conjunction with a pre-trained language model, such as GPT, without requiring additional fine-tuning.

## The LangChain Architecture
![description](images/Prompt.jpg)
<p>The main LG Agent used:<a href="https://python.langchain.com/en/latest/modules/agents/toolkits/examples/vectorstore.html">Langchain VectorStore Agents
</a></p>
# Web interface 
![description](images/web.png)
# Installation 🚀
1. Create a virtual environment `python -m venv langchainenv`
2. Clone this repo 
3. Go into the directory `cd xpayFinanceLLM` 
4. create a file name config.py and type `Openai_api_key = YOUR_OWN_KEY`
4. Install the required dependencies `pip install -r requirements.txt`
7. Start the app `streamlit run app.py`  
8. Upload Any Financial document and prompt any question you may have
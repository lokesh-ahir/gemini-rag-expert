# How I built gemini-rag-expert

I wanted to build a chatbot that actually "thinks" before it talks, rather than just doing a basic search and guess. Here is how the whole system actually works under the hood.

## 1. How the documents get stored
First, when you upload a PDF or any other file, the system reads all the text page by page. I used a text splitter to break the long documents into smaller 1000-character chunks so the AI doesn't get overwhelmed. 

Each of these chunks is turned into a vector (basically a list of numbers) using the Gemini Embedding 2 model. These are then saved into a private collection in Qdrant for each user. I added a "smart batching" logic here because sometimes uploading 20+ chunks at once can cause errors, so it adds them in groups of 5 to make sure nothing gets lost.

## 2. How the AI "thinks" (The Deep Agent)
This is the coolest part. Instead of just doing a single search, I used something called a "Deep Agent." 

When you ask a question, the AI (Gemini 3.1 Pro) doesn't just reply immediately. It writes a quick plan in its head. It uses a tool I built called `search_my_documents` to go into Qdrant and find the right info. If it finds something but realizes it needs more context (like if it sees "Part 1" but not "Part 2"), it will actually go back and do another search automatically. 

It keeps doing this loop until it has the full answer, then it finally puts it all together and sends it to you.

### 3. Tech I used
- **FastAPI**: For the backend because it's super fast and handles files well.
- **Qdrant**: To store all the document vectors.
- **Gemini 2.0**: For both the logic and the embeddings.
- **DeepAgents**: This is the engine that handles the "thinking" loops.

Basically, it's a multi-user system that treats each person's data separately and uses a research agent to make sure the answers are actually accurate.

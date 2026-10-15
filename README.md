# Deal Hunter — Autonomous AI Deal Finding Agent

An autonomous multi-agent AI system that continuously scans the internet for deals, estimates their true value using an ensemble of AI models, and sends push notifications to your phone when a great bargain is found.

## What It Does

- 🔍 **Scans** deal RSS feeds from DealNews every 5 minutes for Electronics, Computers, and Smart Home products
- 🤖 **Evaluates** each deal using an ensemble of 3 AI models working together
- 📊 **Compares** the estimated true value against the deal price to calculate the discount
- 📱 **Notifies** you via Pushover push notification when a deal with >$50 discount is found
- 🖥️ **Displays** a live Gradio dashboard with all discovered deals and a 3D vector store visualization

## Agent Architecture

```
DealAgentFramework
└── PlanningAgent
    ├── ScannerAgent          — Scrapes RSS feeds, uses GPT-5-mini to select best 5 deals
    ├── EnsembleAgent         — Combines 3 models for price estimation
    │   ├── SpecialistAgent   — Fine-tuned Llama 3.2-3B deployed on Modal (10% weight)
    │   ├── FrontierAgent     — RAG + GPT-5.1 with ChromaDB context (80% weight)
    │   └── NeuralNetworkAgent — Deep Neural Network with 289M parameters (10% weight)
    └── MessagingAgent        — Crafts messages with Claude, sends via Pushover
```

## Tech Stack

- Python
- OpenAI API (GPT-5.1, GPT-5-mini)
- Modal.com (serverless GPU hosting for fine-tuned Llama 3.2-3B)
- ChromaDB (vector store for RAG pipeline)
- SentenceTransformers (all-MiniLM-L6-v2 embeddings)
- Ollama + Llama3.2 (local preprocessing)
- LiteLLM
- Gradio
- Pushover (push notifications)
- PyTorch (Deep Neural Network inference)

## Setup

1. Clone the repository:
```
git clone https://github.com/Iqan14/deal-hunter
cd deal-hunter
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Install Ollama and pull Llama3.2:
```
# Download from https://ollama.com
ollama pull llama3.2
```

4. Set up your environment variables:
```
cp .env.example .env
```
Then open `.env` and fill in all your API keys.

5. Set up Modal:
```
modal token set --token-id <your-token-id> --token-secret <your-token-secret>
```
Add your HuggingFace token as a Modal secret named `huggingface-secret`.

6. Deploy the pricer service to Modal:
```
modal deploy pricer_service2.py
```

7. Build the ChromaDB vector store:
```
python build_vectorstore.py
```

8. Download the Deep Neural Network weights from:
```
https://drive.google.com/drive/folders/1uq5C9edPIZ1973dArZiEO-VE13F7m8MK
```
Place `deep_neural_network.pth` in the root of the project folder.

9. Set up Pushover:
   - Sign up at https://pushover.net
   - Create an application and get your User Key and API Token
   - Add both to your `.env` file

10. Run the app:
```
python price_is_right.py
```

## How It Works

1. Every 5 minutes, the ScannerAgent scrapes deal RSS feeds and uses GPT-5-mini to select the 5 most promising deals
2. For each deal, the EnsembleAgent estimates its true market value by combining outputs from 3 specialized models
3. The deal with the highest discount is selected
4. If the discount exceeds $50, a push notification is crafted by Claude and sent to your phone via Pushover
5. All discovered deals are displayed in the Gradio dashboard in real time

## Note

This project requires:
- A Modal account with payment method (for GPU hosting)
- A Pushover account and app installed on your phone
- OpenAI API credits
- HuggingFace account
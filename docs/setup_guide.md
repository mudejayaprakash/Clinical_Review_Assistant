# Setup Guide - Clinical Review Assistant

## Prerequisites

Before you begin, ensure you have:
- **Python 3.9+** installed
- **Git** installed
- **OpenAI API key** ([Get one here](https://platform.openai.com/api-keys))
- **Pinecone account and API key** ([Sign up here](https://www.pinecone.io/))

---

## Step 1: Clone the Repository
```bash
git clone https://github.com/mudejayaprakash/Clinical_Review_Assistant

cd clinical-review-assistant
```

---

## Step 2: Create Virtual Environment (Recommended)

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

---

## Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Expected installation time:** 2-3 minutes

---

## Step 4: Configure Environment Variables

1. **Copy the example file:**
```bash
cp .env.example .env
```

2. **Edit `.env` file** and add your API keys:
```bash
# Required API Keys
OPENAI_API_KEY=sk-your-openai-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX_NAME=medical-policies
PINECONE_NAMESPACE=policies

# Application Settings (Optional - defaults provided)
MODEL_SUMMARY=gpt-4o
MODEL_EVALUATION=gpt-4o
EMBEDDING_MODEL=cambridgeltl/SapBERT-from-PubMedBERT-fulltext
```

**Important:** Never commit the `.env` file to Git (already in `.gitignore`)

---

## Step 5: Set Up Pinecone Index

### Option A: Create Index via Pinecone Dashboard

1. Go to [Pinecone Console](https://app.pinecone.io/)
2. Click "Create Index"
3. Configure:
   - **Name:** `medical-policies`
   - **Dimensions:** `768` (for SapBERT embeddings)
   - **Metric:** `cosine`
   - **Region:** Choose closest to you
4. Click "Create Index"

### Option B: Create Index via Python
```bash
python3 << 'EOF'
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Create index
pc.create_index(
    name='medical-policies',
    dimension=768,
    metric='cosine',
    spec={'serverless': {'cloud': 'aws', 'region': 'us-east-1'}}
)
print("✅ Pinecone index created successfully!")
EOF
```

---

## Step 6: Load Policy Documents (Optional)

**If you have insurance policy PDFs to load:**

1. **Place policy PDFs** in the `data/raw_policy_pdf/` folder:
```bash
mkdir -p data/raw_policy_pdf
# Copy your policy PDFs into this folder
```

2. **Run the data ingestion script:**
```bash
python tools/data_ingestion.py
```

This will:
- Extract text from PDFs
- Create chunks with section-aware splitting
- Generate SapBERT embeddings
- Upload to Pinecone index

**Expected time:** 2-5 minutes for 10 policies

**Note:** You can skip this step and test with an empty policy database, but Node 2 won't retrieve any policies.

---

## Step 7: Run the Application
```bash
streamlit run app.py
```

The application will open in your browser at: `http://localhost:8501`

---

## Step 8: Create Your First Account

1. On the login page, click **"Register"** tab
2. Enter a username and password
3. Click **"Create Account"**
4. Login with your new credentials

---

## Testing the Application

### Quick Test Workflow:

1. **Upload a test medical record** (PDF format)
2. Click **"Summarize and Analyze Records"**
3. Review the generated summary and chief complaints
4. View retrieved policies (if you loaded policy documents)
5. Enter test criteria:
```
   • Patient must be 18 years or older
   • Conservative medical management has failed
   • CT scan or endoscopy confirms septal deviation
```
6. Click **"Evaluate Criteria"**
7. Review results with evidence, page numbers and confidence scores

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution:** Ensure you're in the virtual environment and run:
```bash
pip install -r requirements.txt
```

### Issue: "OpenAI API key not found"
**Solution:** Check that your `.env` file exists and contains valid API keys:
```bash
cat .env | grep OPENAI_API_KEY
```

### Issue: "Pinecone index not found"
**Solution:** Verify index name matches in `.env` and Pinecone dashboard:
```bash
python3 -c "from pinecone import Pinecone; import os; from dotenv import load_dotenv; load_dotenv(); pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY')); print(pc.list_indexes())"
```

### Issue: "PDF processing fails"
**Solution:** Ensure PDF is:
- Under 50MB
- Not password-protected
- Contains extractable text (not just scanned images)

### Issue: "Port 8501 already in use"
**Solution:** Stop other Streamlit instances or use a different port:
```bash
streamlit run app.py --server.port 8502
```

---

## Project Structure
```
clinical-review-assistant/
├── app.py                          # Main Streamlit application
├── agents/
│   ├── __init__.py
│   ├── config.py                   # Configuration settings
│   ├── agent.py                    # LangGraph agent orchestrator
│   ├── nodes.py                    # Node 1, 2, 3 implementations
│   ├── security.py                 # Security & audit logging
│   └── auth.py                     # Authentication system
├── tools/
│   ├── __init__.py
│   ├── rag.py                      # RAG utilities
│   ├── rag_pinecone.py             # Pinecone integration
│   └── data_ingestion.py           # Policy ingestion pipeline
├── data/
│   ├── raw_policy_pdf/             # Policy PDFs (you create)
│   └── policy_txt/                 # Extracted text (auto-generated)
├── docs/
│   ├── agent_workflow.png
│   ├── architecture_diagram.png    
│   ├── screnshots/                 # To display in README
│   └── setup_guide.md              # This file                
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment template
├── .env                            # Your API keys (create from .env.example)
├── .gitignore                      # Git ignore file
└── README.md                       # Project documentation
```

---

## Next Steps

- **Customize policies:** Add new insurance policies to `data/raw_policy_pdf/`
- **Test with real data:** Upload actual medical records (ensure PHI compliance)
- **Adjust configuration:** Modify `agents/config.py` for custom settings
- **Review logs:** Check `security.log` for audit trail
- **Scale deployment:** Deploy to Streamlit Cloud or AWS for production use

---

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section above
- Review README for detailed documentation


---

## Development Mode

To run in development mode with auto-reload:
```bash
streamlit run app.py --server.runOnSave true
```

To view detailed logs:
```bash
tail -f security.log
```

---

**Setup complete!** You're ready to start using the Clinical Review Assistant. 🎉
